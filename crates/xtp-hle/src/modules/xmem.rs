//! `xmem.dll`: console memory, served from host memory. The GPU side works
//! from CPU addresses (unified memory), so memory types and bandwidth
//! classes change nothing here; what constrains addresses (alignment,
//! 2 MiB pages, address-bit limits) is honoured.
#![allow(non_snake_case)]
// Title code calls these through its import table, passing pointers the real
// API would trust.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::alloc::{Layout, alloc, dealloc};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ffi::{OsStr, c_void};
use std::ops::Range;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};

use windows_sys::Win32::Foundation::{
    CloseHandle, ERROR_COMMITMENT_LIMIT, GetLastError, HANDLE, INVALID_HANDLE_VALUE, SetLastError,
};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_DECOMMIT, MEM_EXTENDED_PARAMETER,
    MEM_FREE, MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER, MEM_RESERVE,
    MEM_RESERVE_PLACEHOLDER, MEMORY_BASIC_INFORMATION, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile3,
    MemExtendedParameterAddressRequirements, PAGE_NOACCESS, PAGE_READWRITE, SEC_COMMIT,
    SEC_RESERVE, UnmapViewOfFile2, VirtualAlloc, VirtualAlloc2, VirtualFree, VirtualQuery,
    VirtualQueryEx,
};
use windows_sys::Win32::System::ProcessStatus::{
    K32GetPerformanceInfo, K32GetProcessMemoryInfo, PERFORMANCE_INFORMATION,
    PROCESS_MEMORY_COUNTERS,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;
use windows_sys::core::BOOL;
use xtp_gpu::uma;

use crate::{HleExport, HleModule, hle_export};

const MIB: usize = 1 << 20;

/// Whether we run under Wine (its ntdll exports `wine_get_version`).
pub(crate) fn on_wine() -> bool {
    static WINE: LazyLock<bool> = LazyLock::new(|| {
        use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
        // SAFETY: plain lookups in ntdll, always loaded.
        unsafe {
            let ntdll = GetModuleHandleA(c"ntdll.dll".as_ptr().cast());
            !ntdll.is_null() && GetProcAddress(ntdll, c"wine_get_version".as_ptr().cast()).is_some()
        }
    });
    *WINE
}

/// Standard memory the title has handed to the system.
static TRANSFERRED: AtomicUsize = AtomicUsize::new(0);

// Console additions to the host's flags.
/// 2 MiB pages; the host reads this bit as large pages, which need a
/// privilege.
const PAGES_2M: u32 = 0x2000_0000;
/// With `PAGES_2M`: 64 KiB pages; the host reads it as AWE memory.
const PAGES_64K_BIT: u32 = 0x40_0000;
const PAGES_64K: u32 = PAGES_2M | PAGES_64K_BIT;
/// The host's protection constants and `PAGE_GUARD`. Above them the console
/// adds GPU access bits (0x800-0x40000). Write-combining and uncached are
/// dropped too: our GPU side reads this memory through the CPU.
const HOST_PROTECTION: u32 = 0x1ff;
/// XMem flags of memory both processors reach: all of ours.
const GRAPHICS: u64 = 3;
/// XMem flag bit of 32-bit graphics memory.
const ADDRESS_32BIT: u64 = 4;
/// The host's write watching (not in windows-sys' memory module).
const MEM_WRITE_WATCH: u32 = 0x20_0000;

/// The addresses XMem `flags` allow, first and last: the low 4 GiB for
/// 32-bit memory; else from 4 GiB (kept for 32-bit requests, which the
/// title's big pools would otherwise crowd out) up to 46 down to 40 bits
/// from the field at bit 12, or 40 bits without one: the console keeps
/// title memory there, and its ray-tracing structures must be (BTP 6
///'s BVH builder refuses one above; the title asks for its GPU heap
/// with no field). Decision 0001.
fn address_range(flags: u64) -> (usize, usize) {
    let bits = match (flags & ADDRESS_32BIT, flags >> 12 & 7) {
        (0, 0) => 40,
        (0, field) => 47 - field,
        _ => return (0, u32::MAX as usize),
    };
    (1 << 32, (1 << bits) - 1)
}

const PLACEHOLDER_RESERVE: u32 = MEM_RESERVE | MEM_RESERVE_PLACEHOLDER;
static LOW_VA_RESERVE: Mutex<Option<LowVaReserve>> = Mutex::new(None);

struct LowVaReserve {
    base: usize,
    len: usize,
    exact_size: bool,
}

impl LowVaReserve {
    fn new(len: usize, exact_size: bool) -> std::io::Result<Self> {
        let (_, highest) = address_range(ADDRESS_32BIT);
        let mut range = MEM_ADDRESS_REQUIREMENTS {
            HighestEndingAddress: highest as *mut c_void,
            ..Default::default()
        };
        let base = allocate_virtual(
            std::ptr::null_mut(),
            len,
            PLACEHOLDER_RESERVE,
            PAGE_NOACCESS,
            Some(&mut range),
        );
        if base.is_null() {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Self {
            base: base as usize,
            len,
            exact_size,
        })
    }

    /// Hands out a placeholder that the caller can split, map and release.
    fn take(&mut self, len: usize) -> Option<*mut c_void> {
        if len == 0
            || !len.is_multiple_of(PAGE)
            || len > self.len
            || (self.exact_size && len != self.len)
        {
            return None;
        }
        let base = self.base as *mut c_void;
        if len < self.len {
            // SAFETY: the reserve still owns this placeholder. Splitting its
            // prefix keeps both parts reserved throughout the handoff.
            if unsafe { VirtualFree(base, len, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) } == 0 {
                return None;
            }
        }
        self.base += len;
        self.len -= len;
        Some(base)
    }
}

/// Holds a placeholder before the host graphics device and title start.
/// The default covers one measured startup request from a known build.
/// It takes that much low address space until the matching request uses it,
/// without committing pages. Reserving exactly that size leaves the rest of
/// the low range available and prevents smaller requests from using it up.
/// An explicit MiB count instead supplies pieces on allocation failure.
/// Zero disables the reserve.
pub fn reserve_low_va_from_env(default_mib: usize) {
    const VARIABLE: &str = "XTP_LOW_VA_RESERVE_MIB";
    let value = std::env::var_os(VARIABLE);
    let override_mib = value.as_deref().and_then(|value| {
        let mib = low_va_override_mib(value);
        if mib.is_none() {
            tracing::warn!(?value, "{VARIABLE} must be a MiB count from 0 to 4095");
        }
        mib
    });
    let mib = override_mib.unwrap_or(default_mib);
    if mib == 0 {
        return;
    }
    let mut reserve = LOW_VA_RESERVE.lock().unwrap_or_else(|e| e.into_inner());
    if reserve.is_some() {
        return;
    }
    let automatic = override_mib.is_none();
    match LowVaReserve::new(mib * MIB, automatic) {
        Ok(saved) => {
            tracing::info!(
                mib,
                automatic,
                base = format_args!("{:#x}", saved.base),
                "held a low VA placeholder"
            );
            *reserve = Some(saved);
        }
        Err(error) => tracing::warn!(mib, %error, "could not hold a low VA placeholder"),
    }
}

fn low_va_override_mib(value: &OsStr) -> Option<usize> {
    value.to_str()?.parse().ok().filter(|&mib| mib < 4096)
}

fn take_low_va_reserve(size: usize, exact_size: bool) -> Option<*mut c_void> {
    let at = LOW_VA_RESERVE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_mut()
        .filter(|reserve| reserve.exact_size == exact_size)?
        .take(size)?;
    tracing::info!(
        size,
        at = format_args!("{at:p}"),
        "used the low VA placeholder reserve"
    );
    Some(at)
}

fn allocate_virtual(
    base: *mut c_void,
    size: usize,
    kind: u32,
    protect: u32,
    range: Option<&mut MEM_ADDRESS_REQUIREMENTS>,
) -> *mut c_void {
    let mut parameter = MEM_EXTENDED_PARAMETER::default();
    let count = u32::from(range.is_some());
    if let Some(range) = range {
        parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
        parameter.Anonymous2.Pointer = std::ptr::from_mut(range).cast();
    }
    // SAFETY: the caller's address and size, with host flags. When passed,
    // the address requirements and their parameter live through the call.
    unsafe {
        VirtualAlloc2(
            GetCurrentProcess(),
            base,
            size,
            kind,
            protect,
            if count == 0 {
                std::ptr::null_mut()
            } else {
                &mut parameter
            },
            count,
        )
    }
}

pub extern "system" fn XMemVirtualAlloc(
    base: *mut c_void,
    size: usize,
    kind: u32,
    flags: u64,
    protect: u32,
) -> *mut c_void {
    let (lowest, highest) = address_range(flags);
    let mut requirements = MEM_ADDRESS_REQUIREMENTS {
        LowestStartingAddress: lowest as *mut c_void,
        HighestEndingAddress: highest as *mut c_void,
        Alignment: if kind & (PAGES_2M | PAGES_64K_BIT) == PAGES_2M {
            2 * MIB
        } else {
            0
        },
    };
    let mut protect = match protect & HOST_PROTECTION {
        0 => PAGE_NOACCESS,
        p => p,
    };
    // A reservation of 64 KiB pages takes physical pages later
    // (`XMemMapPhysicalPages`); on the host only a placeholder can.
    // shortcut: committing part of such a reservation later would fail
    // (BTP 6's 23 calls never do); replace that part of the
    // placeholder then.
    let placeholder = kind & PAGES_64K == PAGES_64K && kind & MEM_COMMIT == 0;
    if placeholder {
        if kind & MEM_REPLACE_PLACEHOLDER != 0 {
            // Part of a placeholder the title split: it stays one.
            return base;
        }
        protect = PAGE_NOACCESS;
    }
    let mut kind = kind & !(PAGES_2M | PAGES_64K_BIT);
    if placeholder {
        kind |= PLACEHOLDER_RESERVE;
    }
    // Address requirements apply to reservations at an address of our choice.
    let constrained = base.is_null();
    if constrained {
        kind |= MEM_RESERVE;
        // The GPU side copies only the pages the title wrote since its last
        // copy (`xtp_gpu`, decision 0007); placeholders cannot be
        // watched (their physical-page views are copied whole).
        if kind & MEM_RESERVE_PLACEHOLDER == 0 {
            kind |= MEM_WRITE_WATCH;
        }
    }
    let low_placeholder = constrained
        && flags & ADDRESS_32BIT != 0
        && kind == PLACEHOLDER_RESERVE
        && requirements.Alignment == 0
        && protect == PAGE_NOACCESS;
    let saved = if low_placeholder {
        take_low_va_reserve(size, true)
    } else {
        None
    };
    let mut at = saved.unwrap_or_else(|| {
        allocate_virtual(
            base,
            size,
            kind,
            protect,
            constrained.then_some(&mut requirements),
        )
    });
    // SAFETY: capture this thread's allocation error before any reporting.
    let error = at.is_null().then(|| unsafe { GetLastError() });
    if let Some(error) = error
        && low_placeholder
    {
        if let Some(saved) = take_low_va_reserve(size, false) {
            at = saved;
        } else {
            tracing::warn!(size, error, "32-bit placeholder allocation failed");
        }
    }
    note_virtual(kind, size, !at.is_null());
    if !at.is_null() {
        // Heaps over title memory (`xtp_gpu::uma`, with `XTP_UMA=1`) follow
        // what the title reserves and commits. Placeholders are left out:
        // the title maps physical pages there, and heaps over those sit on
        // our own view of them (`Section`).
        if constrained && kind & MEM_RESERVE_PLACEHOLDER == 0 {
            uma::reserved(at as usize, size);
        } else if kind & MEM_COMMIT != 0 {
            uma::committed(at as usize, size);
        }
        // The GPU side's copies are told of it all: among it, commits into
        // a reservation the title placed itself, which they take for a
        // placeholder until then.
        remapped(at as usize, size);
    }
    if base.is_null() {
        tracing::debug!(
            size,
            kind = format_args!("{kind:#x}"),
            flags = format_args!("{flags:#x}"),
            at = format_args!("{at:p}"),
            "XMemVirtualAlloc"
        );
    }
    if at.is_null()
        && let Some(error) = error
    {
        // SAFETY: the caller can read the original Windows error.
        unsafe { SetLastError(error) };
    }
    at
}

/// `MEMORY_BASIC_INFORMATION` (the console's first 0x30 bytes are the
/// same), then the XMem flags and tag.
#[repr(C)]
pub struct BasicInformation {
    host: MEMORY_BASIC_INFORMATION,
    flags: u64,
    tag_code: u32,
    tag: [u8; 4],
}

// shortcut: every committed page reports graphics memory and no tag, since
// all of it is GPU-reachable here (BTP 6's BVH builder, the
// only caller, wants graphics memory); remember the flags of
// `XMemVirtualAlloc` regions if a title tells CPU-only memory apart.
pub extern "system" fn XMemVirtualQuery(
    process: HANDLE,
    address: *const c_void,
    out: *mut BasicInformation,
) -> BOOL {
    let size = size_of::<MEMORY_BASIC_INFORMATION>();
    // SAFETY: the title passes room for the whole structure; the host
    // fails on a null one.
    unsafe {
        if VirtualQueryEx(process, address, out.cast(), size) == 0 {
            return 0;
        }
        let info = &mut *out;
        info.flags = if info.host.State == MEM_COMMIT {
            GRAPHICS
        } else {
            0
        };
        info.tag_code = 0;
        info.tag = [0; 4];
    }
    1
}

// `XMemAlloc` attributes: alignment as log2 (5 bits at 24, 4 bytes and up
// from 2), then the memory type, then bit 33: a 32-bit address.
const ALIGNMENT_SHIFT: u64 = 24;
const ALLOC_32BIT: u64 = 1 << 33;
/// Our block header: the block's size and alignment, just below the
/// pointer handed out.
const HEADER: usize = 2 * size_of::<usize>();

fn layout(size: usize, attributes: u64) -> Option<Layout> {
    // An alignment of at least 16 leaves room for the header.
    let align = 1usize << (attributes >> ALIGNMENT_SHIFT & 0x1f).max(4);
    Layout::from_size_align(size.checked_add(align)?, align).ok()
}

type AllocRoutine = extern "system" fn(usize, u64) -> *mut c_void;
type FreeRoutine = extern "system" fn(*mut c_void, u64);

/// The allocation and free routines the title's exe exports as
/// `__XMemAllocationHooks__` (BTP 6 does): `XMemAlloc` and
/// `XMemFree` call them, for the title's own calls and its middleware's;
/// the `…Default` exports do not.
fn hooks() -> [usize; 2] {
    static AT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let at = match AT.get() {
        Some(&at) => at,
        // Looked up once the title is mapped, which is before it runs.
        None => match crate::runtime::main_image() {
            Some(base) => *AT.get_or_init(|| {
                use windows_sys::Win32::System::LibraryLoader::GetProcAddress;
                // SAFETY: a mapped image and a NUL-terminated name.
                let at = unsafe {
                    GetProcAddress(base as _, c"__XMemAllocationHooks__".as_ptr().cast())
                }
                .map_or(0, |f| f as usize);
                tracing::info!(found = at != 0, "xmem allocation hooks");
                at
            }),
            None => 0,
        },
    };
    if at == 0 {
        return [0; 2];
    }
    // SAFETY: the export is two routine pointers.
    unsafe { (at as *const [usize; 2]).read() }
}

/// The title's general allocator, for middleware too: its hook if it has
/// one, else [`XMemAllocDefault`].
pub extern "system" fn XMemAlloc(size: usize, attributes: u64) -> *mut c_void {
    let block = match hooks()[0] {
        0 => XMemAllocDefault(size, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let alloc: AllocRoutine = unsafe { std::mem::transmute(hook) };
            alloc(size, attributes)
        }
    };
    let kind = ledger::Kind::of_attributes(attributes, false);
    ledger::note(|l| l.handed_out(kind, size as u64, block as usize));
    block
}

/// See [`XMemAlloc`].
pub extern "system" fn XMemFree(address: *mut c_void, attributes: u64) {
    if !address.is_null() {
        ledger::note(|l| l.given_back(false, address as usize));
    }
    match hooks()[1] {
        0 => XMemFreeDefault(address, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let free: FreeRoutine = unsafe { std::mem::transmute(hook) };
            free(address, attributes)
        }
    }
}

/// The system allocator behind [`XMemAlloc`].
pub extern "system" fn XMemAllocDefault(size: usize, attributes: u64) -> *mut c_void {
    let block = alloc_default(size, attributes);
    let kind = ledger::Kind::of_attributes(attributes, true);
    ledger::note(|l| l.handed_out(kind, size as u64, block as usize));
    block
}

fn alloc_default(size: usize, attributes: u64) -> *mut c_void {
    if attributes & ALLOC_32BIT != 0 {
        return XMemVirtualAlloc(
            std::ptr::null_mut(),
            size,
            MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | ADDRESS_32BIT,
            PAGE_READWRITE,
        );
    }
    let Some(layout) = layout(size, attributes) else {
        return std::ptr::null_mut();
    };
    // shortcut: the host heap's addresses lie above 2^40, unlike the
    // console's; nothing seen builds ray-tracing structures in this memory
    // (BTP 6 uses its GPU heap). Sub-allocate from a
    // reservation below 2^40 if one does.
    // SAFETY: the layout has a non-zero size.
    let block = unsafe { alloc(layout) };
    if block.is_null() {
        return block.cast();
    }
    // SAFETY: the block holds the alignment (at least the header) and then
    // `size` bytes.
    unsafe {
        let user = block.add(layout.align());
        user.sub(HEADER)
            .cast::<[usize; 2]>()
            .write([layout.size(), layout.align()]);
        user.cast()
    }
}

/// Frees what [`XMemAllocDefault`] returned, given the same attributes.
pub extern "system" fn XMemFreeDefault(address: *mut c_void, attributes: u64) {
    if address.is_null() {
        return;
    }
    ledger::note(|l| l.given_back(true, address as usize));
    if attributes & ALLOC_32BIT != 0 {
        free(address, 0, MEM_RELEASE);
        return;
    }
    // SAFETY: the address came from `XMemAlloc` with these attributes: a
    // header-prefixed block.
    unsafe {
        let user = address.cast::<u8>();
        let [size, align] = user.sub(HEADER).cast::<[usize; 2]>().read();
        dealloc(
            user.sub(align),
            Layout::from_size_align_unchecked(size, align),
        );
    }
}

/// A retail console has no tools memory.
pub extern "system" fn XMemIsToolsMemAvailable() -> BOOL {
    0
}

// The memory API set's `VirtualAlloc` and `VirtualFree`, which the title's
// exe imports: the host's, with the GPU side's heaps over title memory
// (`xtp_gpu::uma`) kept in step, and its copies told (`remapped`). Nothing
// else changes; with `XTP_UMA` off, no heaps. The set's other functions
// still go to the host.

/// Tells the GPU side's copies that title memory `[at, at + len)` changed
/// (`xtp_gpu::title_memory_remapped`): what they noted of it (views,
/// placeholders, guarded blocks) is out of date. The thread's last error is
/// kept for the title: the report gives guarded blocks write access back,
/// which fails harmlessly where they are gone. Our own views of physical
/// pages (`Section`) are not reported: no copy takes them.
fn remapped(at: usize, len: usize) {
    // SAFETY: plain reads and writes of this thread's last error.
    let error = unsafe { GetLastError() };
    xtp_gpu::title_memory_remapped(at, len);
    // SAFETY: as above.
    unsafe { SetLastError(error) };
}

/// The title's `VirtualAlloc`: what it commits of title memory gets heaps,
/// and what it reserves or commits is reported to the GPU side's copies.
pub extern "system" fn title_virtual_alloc(
    address: *const c_void,
    size: usize,
    kind: u32,
    protect: u32,
) -> *mut c_void {
    // SAFETY: the title's own arguments, forwarded.
    let at = unsafe { VirtualAlloc(address, size, kind, protect) };
    note_virtual(kind, size, !at.is_null());
    if !at.is_null() {
        let from = if address.is_null() {
            at.cast_const()
        } else {
            address
        };
        if kind & MEM_COMMIT != 0 {
            uma::committed(from as usize, size);
        }
        remapped(from as usize, size);
    }
    at
}

/// The title's `VirtualFree`.
pub extern "system" fn title_virtual_free(address: *mut c_void, size: usize, kind: u32) -> BOOL {
    free(address, size, kind)
}

/// For the ledger: `size` bytes reserved or committed (host `kind`).
fn note_virtual(kind: u32, size: usize, ok: bool) {
    ledger::note(|l| {
        if kind & MEM_RESERVE != 0 {
            l.flowed(ledger::Kind::Reserve, size as u64, ok);
        }
        if kind & MEM_COMMIT != 0 {
            l.flowed(ledger::Kind::Commit, size as u64, ok);
        }
    });
}

/// The host's `MEM_COALESCE_PLACEHOLDERS` (not in windows-sys).
const MEM_COALESCE_PLACEHOLDERS: u32 = 1;

/// `VirtualFree`, the heaps over the memory released first: the host
/// refuses to decommit or release memory under an open heap (error 87).
/// What is still committed there gets heaps again after. The GPU side's
/// copies are told once it is done.
fn free(address: *mut c_void, size: usize, kind: u32) -> BOOL {
    let (at, len) = match size {
        // The whole allocation, as the host reads a size of 0.
        0 => allocation(address as usize),
        size => (address as usize, size),
    };
    // Splitting and merging placeholders changes no committed memory.
    let placeholders = kind & (MEM_PRESERVE_PLACEHOLDER | MEM_COALESCE_PLACEHOLDERS) != 0;
    let done = if !uma::enabled() || placeholders {
        // SAFETY: the caller's own arguments, forwarded.
        unsafe { VirtualFree(address, size, kind) != 0 }
    } else {
        let change = uma::changing(at, len);
        // SAFETY: as above.
        let done = change.call(|| unsafe { VirtualFree(address, size, kind) } != 0);
        // SAFETY: plain calls; the caller reads the host's error after.
        let error = unsafe { GetLastError() };
        change.done(done && kind & MEM_RELEASE != 0);
        // SAFETY: as above.
        unsafe { SetLastError(error) };
        done
    };
    if done {
        remapped(at, len);
    }
    if !placeholders {
        ledger::note(|l| {
            if kind & MEM_DECOMMIT != 0 {
                l.flowed(ledger::Kind::Decommit, len as u64, done);
            }
            if kind & MEM_RELEASE != 0 {
                l.flowed(ledger::Kind::Release, len as u64, done);
            }
        });
    }
    done.into()
}

/// The allocation holding `at`: its base and its size, all its regions.
fn allocation(at: usize) -> (usize, usize) {
    let query = |at: usize| {
        // SAFETY: plain data; zero is a valid value.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a query of our own address space into a sized structure.
        let found = unsafe { VirtualQuery(at as *const c_void, &mut info, size_of_val(&info)) };
        (found != 0).then_some(info)
    };
    let Some(info) = query(at) else {
        return (at, 0);
    };
    let base = info.AllocationBase as usize;
    let mut end = base;
    while let Some(region) =
        query(end).filter(|r| r.AllocationBase as usize == base && r.State != MEM_FREE)
    {
        end = region.BaseAddress as usize + region.RegionSize;
    }
    (base, end - base)
}

// Physical pages: on the console, 64 KiB frames a title maps into its own
// reservations, at more than one address if it likes. Here each allocation
// is a pagefile-backed section, its pages are 64 KiB slices of it, and a
// mapping is a view of consecutive slices in place of a host placeholder.
//
// The runtime writes pages too, mapped or not (DirectStorage destinations,
// through `page_window` and `write_physical`). For that each allocation
// keeps one view of all of it, mapped once (docs/analysis/formats.md, F3),
// so such a write is a plain copy. Mapping, committing and unmapping change
// the address space, which the host serialises per process, page faults
// included, and an unmap flushes the TLB of every core: doing them for each
// request kept DirectStorage's workers waiting on each other and on the
// title.

const PAGE: usize = 64 << 10;

/// The address-space changes physical pages have cost since the process
/// started: views mapped (ours and the title's) and unmapped, and commit
/// calls. DirectStorage's debug summary logs how they grow.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Activity {
    pub(crate) maps: u64,
    pub(crate) unmaps: u64,
    pub(crate) commits: u64,
}

static MAPS: AtomicU64 = AtomicU64::new(0);
static UNMAPS: AtomicU64 = AtomicU64::new(0);
static COMMITS: AtomicU64 = AtomicU64::new(0);

pub(crate) fn activity() -> Activity {
    Activity {
        maps: MAPS.load(Ordering::Relaxed),
        unmaps: UNMAPS.load(Ordering::Relaxed),
        commits: COMMITS.load(Ordering::Relaxed),
    }
}

/// One allocation: its section, and our own view of all of it.
struct Section {
    handle: HANDLE,
    /// Where our view starts. It shows every page read-write (once
    /// committed) and stays mapped as long as the section lives: until its
    /// last page is freed and the last title view of it unmapped.
    base: usize,
    pages: usize,
    /// One bit per page: committed. Commitment belongs to the section (a
    /// page committed through one view is committed in every view, those
    /// mapped later too), so the title's mappings set these bits as well.
    committed: Box<[AtomicU64]>,
    /// Views of it mapped, ours included, and commit calls for it.
    maps: AtomicU64,
    commits: AtomicU64,
}

// SAFETY: a section handle and a view may be used from any thread.
unsafe impl Send for Section {}
// SAFETY: as above; the rest is atomics.
unsafe impl Sync for Section {}

impl Section {
    /// A section of `pages` pages, its pages committed when first mapped or
    /// written, and our view of it.
    fn new(pages: usize) -> std::io::Result<Section> {
        let size = pages * PAGE;
        // SAFETY: plain section creation.
        let handle = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                // Under Wine a committed section is a sparse file, free until
                // touched, and committing a reserved one page by page costs a
                // wineserver request each that scans every range committed so
                // far: commit it whole there.
                PAGE_READWRITE | if on_wine() { SEC_COMMIT } else { SEC_RESERVE },
                (size as u64 >> 32) as u32,
                size as u32,
                std::ptr::null(),
            )
        };
        if handle.is_null() {
            return Err(std::io::Error::last_os_error());
        }
        let base = view_of_all(handle, size);
        if base == 0 {
            let error = std::io::Error::last_os_error();
            // SAFETY: the handle just created, used by nothing else.
            unsafe { CloseHandle(handle) };
            return Err(error);
        }
        MAPS.fetch_add(1, Ordering::Relaxed);
        // The GPU side's heaps over physical pages sit on this view, which
        // nothing unmaps while the section lives (`xtp_gpu::uma`).
        uma::mapped(base, size);
        Ok(Section {
            handle,
            base,
            pages,
            committed: (0..pages.div_ceil(64)).map(|_| AtomicU64::new(0)).collect(),
            maps: AtomicU64::new(1),
            commits: AtomicU64::new(0),
        })
    }

    /// Our view's bytes from page `index` on.
    fn at(&self, index: usize) -> *mut u8 {
        (self.base + index * PAGE) as *mut u8
    }

    fn is_committed(&self, page: usize) -> bool {
        self.committed[page / 64].load(Ordering::Acquire) >> (page % 64) & 1 != 0
    }

    /// Commits pages `[first, first + n)`: one host call per run of pages
    /// not committed yet, none once they all are (the steady state). Two
    /// threads may commit a page at once; committing twice is harmless.
    fn commit(&self, first: usize, n: usize) -> bool {
        let end = first + n;
        let mut page = first;
        while page < end {
            if self.is_committed(page) {
                page += 1;
                continue;
            }
            // The run starts at this page, seen uncommitted just now, even
            // if another thread commits it before the next look: asked
            // again, it could end the run before it starts, and committing
            // no bytes fails.
            let from = page;
            page += 1;
            while page < end && !self.is_committed(page) {
                page += 1;
            }
            self.commits.fetch_add(1, Ordering::Relaxed);
            if !commit(self.at(from) as usize, (page - from) * PAGE) {
                return false;
            }
            for p in from..page {
                self.committed[p / 64].fetch_or(1 << (p % 64), Ordering::Release);
            }
            uma::committed(self.at(from) as usize, (page - from) * PAGE);
        }
        true
    }
}

impl Drop for Section {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::UnmapViewOfFile;
        tracing::debug!(
            pages = self.pages,
            views = self.maps.load(Ordering::Relaxed),
            commits = self.commits.load(Ordering::Relaxed),
            "physical pages released"
        );
        // Heaps on our view go first: the host keeps a view mapped under one.
        let change = uma::changing(self.base, self.pages * PAGE);
        // SAFETY: our view and handle; nothing refers to the section any more
        // (title views hold it through their slices).
        change.call(|| unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: self.base as *mut c_void,
            }) != 0
        });
        // SAFETY: as above.
        unsafe { CloseHandle(self.handle) };
        change.done(true);
        UNMAPS.fetch_add(1, Ordering::Relaxed);
    }
}

/// Maps a view of all of a section, read-write, at 2^40 or above if it can:
/// out of the way of the title's own memory, whose constrained ranges lie
/// below (`address_range`). Returns its address, or 0.
fn view_of_all(handle: HANDLE, size: usize) -> usize {
    let mut requirements = MEM_ADDRESS_REQUIREMENTS {
        LowestStartingAddress: (1usize << 40) as *mut c_void,
        HighestEndingAddress: std::ptr::null_mut(),
        Alignment: 0,
    };
    let mut parameter = MEM_EXTENDED_PARAMETER::default();
    parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
    parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
    let map = |parameter: *mut MEM_EXTENDED_PARAMETER, count: u32| {
        // SAFETY: a section we own, mapped where the host likes; the
        // parameter, if any, points at `requirements`, which outlives the
        // call.
        unsafe {
            MapViewOfFile3(
                handle,
                GetCurrentProcess(),
                std::ptr::null(),
                0,
                size,
                0,
                PAGE_READWRITE,
                parameter,
                count,
            )
        }
        .Value as usize
    };
    match map(&mut parameter, 1) {
        0 => map(std::ptr::null_mut(), 0),
        at => at,
    }
}

/// Where a page lives, or what a view starts with: a section and a page
/// index in it.
#[derive(Clone)]
struct Slice {
    section: Arc<Section>,
    index: usize,
}

/// Page numbers and where they live. Every DirectStorage worker and the
/// title's mapping calls look pages up here, all at once (read locks);
/// only allocating and freeing pages write.
struct Pages {
    next: usize,
    table: HashMap<usize, Slice>,
}

/// The first page number given out; 0 stays unused.
const FIRST_PAGE: usize = 0x10_0000;
/// Numbers from here on do not survive the title's 21-bit `number >> 4`
/// ([`FRAMES`]).
const PAGE_LIMIT: usize = 1 << 25;

impl Pages {
    /// The first of `n` numbers, [`FRAMES`] apart, for new pages: after the
    /// last given while they fit the title's 21 bits (as before), then the
    /// lowest run of `n` freed ones. `None` if there is no such run.
    fn take(&mut self, n: usize) -> Option<usize> {
        let len = n.checked_mul(FRAMES)?;
        let end = self.next.checked_add(len);
        if let Some(end) = end.filter(|&end| end <= PAGE_LIMIT) {
            let first = self.next;
            self.next = end;
            return Some(first);
        }
        static REUSING: std::sync::Once = std::sync::Once::new();
        REUSING.call_once(|| {
            tracing::info!("page numbers reach the title's 21 bits: freed ones given again");
        });
        let mut used: Vec<usize> = self.table.keys().copied().collect();
        used.sort_unstable();
        free_run(used, len)
    }
}

/// The lowest `start` from [`FIRST_PAGE`] where `len` (a multiple of
/// [`FRAMES`]) numbers up to [`PAGE_LIMIT`] are all outside `used`
/// (sorted).
fn free_run(used: impl IntoIterator<Item = usize>, len: usize) -> Option<usize> {
    let mut start = FIRST_PAGE;
    for at in used.into_iter().chain([PAGE_LIMIT]) {
        if at.checked_sub(start).is_some_and(|gap| gap >= len) {
            return Some(start);
        }
        start = start.max(at.checked_add(FRAMES)?);
    }
    None
}

static PAGES: LazyLock<RwLock<Pages>> = LazyLock::new(|| {
    RwLock::new(Pages {
        // Page numbers are opaque to titles.
        next: FIRST_PAGE,
        table: HashMap::new(),
    })
});

fn page_table() -> RwLockReadGuard<'static, Pages> {
    PAGES.read().unwrap_or_else(|e| e.into_inner())
}

fn page_table_mut() -> RwLockWriteGuard<'static, Pages> {
    PAGES.write().unwrap_or_else(|e| e.into_inner())
}

/// The title's views of physical pages by address: their length and what
/// they show. Locked apart from the page table: the title's mapping calls
/// change the address space while they hold this, and lookups need not
/// wait for that. (Taken before the page table when both are.)
type Views = BTreeMap<usize, (usize, Slice)>;

static VIEWS: Mutex<Views> = Mutex::new(BTreeMap::new());

fn views() -> MutexGuard<'static, Views> {
    VIEWS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Every address of the title's views that shows each physical page, for
/// the GPU side's write guards (`xtp_gpu::set_alias_lookup`): BTP 6 writes
/// a page through one view (a pool) while the GPU reads it through another
/// (its tiled heap, `map_alias`), and a read-only guard on one view does
/// not see writes through the other. Our own view of each allocation
/// (`Section`) is not in it: no copy takes it, and our writes through it
/// open the guards of the title's views from here (`opened`). (Ported from
/// Marvisak/VIctory#1122 (head 9ed6093) by tightfall.)
///
/// Updated with `VIEWS` held: a view is noted before it is reported mapped,
/// so that the report opens the guards of the page's other views, and
/// forgotten once it is unmapped, before the report after (see `map` and
/// `unmap`). Its lock is taken last, after `VIEWS` and the GPU side's
/// guards (whose fault handler asks it), and nothing is called while it is
/// held.
#[derive(Default)]
struct Aliases {
    /// (section, page index) to the addresses showing it.
    by_page: HashMap<(usize, usize), Vec<usize>>,
    /// Address to (section, page index), for every page of the title's
    /// views.
    by_address: HashMap<usize, (usize, usize)>,
    /// The addresses whose page shows at another address too, in order:
    /// the GPU side asks about ranges, mostly of pages shown once.
    shared: BTreeSet<usize>,
}

static ALIASES: LazyLock<Mutex<Aliases>> = LazyLock::new(Mutex::default);

fn aliases() -> MutexGuard<'static, Aliases> {
    ALIASES.lock().unwrap_or_else(|e| e.into_inner())
}

impl Aliases {
    /// `at` shows page `key` no more.
    fn forget(&mut self, at: usize, key: (usize, usize)) {
        self.shared.remove(&at);
        let Some(list) = self.by_page.get_mut(&key) else {
            return;
        };
        list.retain(|&x| x != at);
        if let [last] = list[..] {
            self.shared.remove(&last);
        }
        if list.is_empty() {
            self.by_page.remove(&key);
        }
    }
}

/// Notes `[base, base + len)` as showing `slice` and the slices after it
/// (`shown`), or as no longer showing them.
fn note_views(base: usize, len: usize, slice: &Slice, shown: bool) {
    static LOOKUP: std::sync::Once = std::sync::Once::new();
    LOOKUP.call_once(|| {
        xtp_gpu::set_alias_lookup(other_views);
        // SAFETY: map, unmap, write_physical and PageWindow pin the permanent
        // range throughout a change. Each live title view retains its Section.
        unsafe { xtp_gpu::set_copy_back_lookup(copy_back_page) };
    });
    let section = Arc::as_ptr(&slice.section) as usize;
    let mut a = aliases();
    let a = &mut *a;
    for k in 0..len / PAGE {
        let (at, key) = (base + k * PAGE, (section, slice.index + k));
        if !shown {
            if a.by_address.get(&at) == Some(&key) {
                a.by_address.remove(&at);
                a.forget(at, key);
            }
            continue;
        }
        match a.by_address.insert(at, key) {
            Some(old) if old == key => continue,
            // Views do not overlap: not expected, but kept exact.
            Some(old) => a.forget(at, old),
            None => {}
        }
        let list = a.by_page.entry(key).or_default();
        list.push(at);
        if list.len() > 1 {
            a.shared.extend(list.iter().copied());
        }
    }
}

fn copy_back_page(at: usize, views: &mut Vec<usize>) -> Option<usize> {
    let a = aliases();
    let key = a.by_address.get(&at)?;
    views.extend_from_slice(a.by_page.get(key)?);
    // SAFETY: every indexed view holds its Section through a Slice until
    // note_views removes the entry under this same lock.
    let section = unsafe { &*(key.0 as *const Section) };
    Some(section.at(key.1) as usize)
}

/// For each page of `range` (title addresses) that the title shows at other
/// addresses too, `(its address, the other's)`, once per other address: the
/// GPU side's write guards ask it, with their lock held.
fn other_views(range: Range<usize>, out: &mut Vec<(usize, usize)>) {
    let a = aliases();
    for &at in a.shared.range(range) {
        if let Some(key) = a.by_address.get(&at)
            && let Some(list) = a.by_page.get(key)
        {
            out.extend(list.iter().filter(|&&x| x != at).map(|&x| (at, x)));
        }
    }
}

/// After our write into pages `[index, index + n)` of `section` through
/// our view of it: the GPU side's guards on the title's views of them are
/// opened, as by a kernel write. A copy of a title view (a buffer the GPU
/// reads) leaves its blocks read-only, but our view is not one of them and
/// does not fault: without this the copy kept the page's old bytes, a
/// stale mesh once DirectStorage streams a model into a page the title
/// reused (the #1122 review). After the write is enough: the title reads
/// the bytes once the request has completed, and its status is written
/// after this. Nothing is looked up while nothing is guarded.
fn opened(section: &Section, index: usize, n: usize) {
    // Our writes before the look at the guards' count: a copy that guards
    // after it takes what they wrote.
    std::sync::atomic::fence(Ordering::SeqCst);
    if !xtp_gpu::guarding() {
        return;
    }
    let key = std::ptr::from_ref(section) as usize;
    let mut at: Vec<usize> = {
        let a = aliases();
        (index..index + n)
            .filter_map(|i| a.by_page.get(&(key, i)))
            .flatten()
            .copied()
            .collect()
    };
    // Out of the index's lock, which the guards' handler takes after theirs
    // (see `Aliases`). An address shows one page: no repeats.
    at.sort_unstable();
    for run in at.chunk_by(|a, b| a + PAGE == *b) {
        xtp_gpu::kernel_wrote(run[0], run.len() * PAGE);
    }
}

/// For DirectStorage's statistics (`XTP_MESH_STATS`), read only: of the
/// pages behind bytes `[offset, offset + len)` of `pages`, how many there
/// are, how many the title shows (at least one view), how many at more than
/// one address, and at how many addresses those are shown in all: each is
/// one `opened` opens after a write, while the GPU side guards anything.
/// Zeros if a page is unknown.
pub(crate) fn title_views(
    pages: &[usize],
    offset: usize,
    len: usize,
) -> (usize, usize, usize, usize) {
    let Some(runs) = runs(pages, offset, len) else {
        return (0, 0, 0, 0);
    };
    let a = aliases();
    let mut out = (0, 0, 0, 0);
    for (slice, n) in &runs {
        let key = Arc::as_ptr(&slice.section) as usize;
        for index in slice.index..slice.index + n {
            let shown = a.by_page.get(&(key, index)).map_or(0, Vec::len);
            out.0 += 1;
            out.1 += usize::from(shown > 0);
            if shown > 1 {
                out.2 += 1;
                out.3 += shown;
            }
        }
    }
    out
}

/// `count` is in and out (pages wanted, pages given); all or nothing here.
pub extern "system" fn XMemAllocatePhysicalPages(
    flags: u64,
    count: *mut usize,
    pages: *mut usize,
) -> BOOL {
    if count.is_null() || pages.is_null() {
        return 0;
    }
    // SAFETY: checked non-null.
    let n = unsafe { *count };
    if n.checked_mul(PAGE).is_none_or(|s| s == 0) {
        return 0;
    }
    // BTP 6 asks for 6.1 GiB at once, which a busy host often
    // cannot commit (error 1450): pages are committed when first mapped or
    // written (`Section::commit`).
    let section = Section::new(n);
    let bytes = (n * PAGE) as u64;
    let section = match section {
        Ok(section) => Arc::new(section),
        Err(error) => {
            ledger::note(|l| l.handed_out(ledger::Kind::PhysicalPages, bytes, 0));
            tracing::warn!(n, %error, "XMemAllocatePhysicalPages failed");
            // SAFETY: checked non-null.
            unsafe { count.write(0) };
            return 0;
        }
    };
    tracing::debug!(
        n,
        flags = format_args!("{flags:#x}"),
        "XMemAllocatePhysicalPages"
    );
    let first = {
        let mut table = page_table_mut();
        let Some(first) = table.take(n) else {
            drop(table);
            ledger::note(|l| l.handed_out(ledger::Kind::PhysicalPages, bytes, 0));
            // The title keeps 21 bits of a number (`FRAMES`).
            tracing::warn!(n, "XMemAllocatePhysicalPages: out of page numbers");
            // SAFETY: checked non-null. The section goes with `section`.
            unsafe { count.write(0) };
            return 0;
        };
        for index in 0..n {
            let slice = Slice {
                section: section.clone(),
                index,
            };
            table.table.insert(first + index * FRAMES, slice);
        }
        first
    };
    ledger::note(|l| l.handed_out(ledger::Kind::PhysicalPages, bytes, 1));
    for index in 0..n {
        // SAFETY: the title passes room for `n` page numbers.
        unsafe { pages.add(index).write(first + index * FRAMES) };
    }
    1
}

/// Page numbers count 4 KiB frames, so a 64 KiB page's is a multiple of
/// 16: BTP 6 keeps them as 21-bit `number >> 4` and rebuilds
/// them with `<< 4` (128 GiB of pages at most).
const FRAMES: usize = PAGE >> 12;

/// Pages still mapped stay readable through their views until unmapped.
pub extern "system" fn XMemFreePhysicalPages(n: usize, pages: *const usize) -> BOOL {
    if pages.is_null() {
        return 0;
    }
    // SAFETY: the title passes `n` page numbers.
    let numbers = unsafe { std::slice::from_raw_parts(pages, n) };
    let freed: Vec<Slice> = {
        let mut table = page_table_mut();
        numbers
            .iter()
            .filter_map(|p| table.table.remove(p))
            .collect()
    };
    let bytes = (freed.len() * PAGE) as u64;
    ledger::note(|l| l.freed(ledger::Kind::PhysicalPages, bytes));
    // Out of the lock: the last page of an allocation unmaps our view.
    drop(freed);
    1
}

/// `pages` as runs of consecutive slices of one section (each run's first
/// slice and its length in pages), up to the first unknown page, which
/// comes back too. One look at the page table.
fn group(pages: &[usize]) -> (Vec<(Slice, usize)>, Option<usize>) {
    let table = page_table();
    let mut runs: Vec<(Slice, usize)> = Vec::new();
    for &page in pages {
        let Some(slice) = table.table.get(&page) else {
            return (runs, Some(page));
        };
        match runs.last_mut() {
            Some((first, n))
                if Arc::ptr_eq(&first.section, &slice.section)
                    && slice.index == first.index + *n =>
            {
                *n += 1;
            }
            _ => runs.push((slice.clone(), 1)),
        }
    }
    (runs, None)
}

/// The pages behind bytes `[offset, offset + len)` of `pages` (64 KiB
/// each), as runs (see `group`). `None` if a page is unknown or `pages`
/// ends early.
fn runs(pages: &[usize], offset: usize, len: usize) -> Option<Vec<(Slice, usize)>> {
    let end = offset.checked_add(len)?;
    let pages = pages.get(offset / PAGE..end.div_ceil(PAGE))?;
    match group(pages) {
        (runs, None) => Some(runs),
        (_, Some(_)) => None,
    }
}

/// Copies `data` into physical pages, from `offset` bytes into the first,
/// whether or not they are mapped anywhere (DirectStorage destinations):
/// a copy per run of consecutive pages into our view of their allocation,
/// committing pages written for the first time, and the guards of the
/// title's views of each run opened after it (`opened`). False if a page
/// is unknown or the host fails.
pub(crate) fn write_physical(pages: &[usize], offset: usize, data: &[u8]) -> bool {
    let Some(runs) = runs(pages, offset, data.len()) else {
        return false;
    };
    let (mut within, mut done) = (offset % PAGE, 0);
    for (slice, n) in &runs {
        if !slice.section.commit(slice.index, *n) {
            return false;
        }
        let _pin = xtp_gpu::pin_copy_back(slice.section.at(slice.index) as usize, *n * PAGE);
        let k = (n * PAGE - within).min(data.len() - done);
        // SAFETY: `within + k` bytes fit in the run's pages, which our view
        // shows committed read-write; other views of the pages only alias
        // them, as the console's do.
        unsafe {
            let to = slice.section.at(slice.index).add(within);
            std::ptr::copy_nonoverlapping(data[done..].as_ptr(), to, k);
        }
        // The bytes reach every page of the run (`runs`).
        opened(&slice.section, slice.index, *n);
        (within, done) = (0, done + k);
    }
    true
}

/// Bytes `[offset, offset + len)` of physical pages, writable in place
/// until dropped: DirectStorage decodes into it instead of a buffer it
/// would then copy. Dropping it opens the guards of the title's views of
/// its pages (`opened`).
pub(crate) struct PageWindow {
    _pin: xtp_gpu::CopyBackPin,
    /// Keeps our view mapped, should the title free the pages meanwhile.
    section: Arc<Section>,
    at: *mut u8,
    len: usize,
}

impl PageWindow {
    pub(crate) fn bytes(&mut self) -> &mut [u8] {
        // SAFETY: `len` committed read-write bytes of our view, ours alone
        // while `self` lives (other views of the pages only alias them, as
        // the console's do).
        unsafe { std::slice::from_raw_parts_mut(self.at, self.len) }
    }
}

impl Drop for PageWindow {
    fn drop(&mut self) {
        let from = self.at as usize - self.section.base;
        let (first, end) = (from / PAGE, (from + self.len).div_ceil(PAGE));
        opened(&self.section, first, end - first);
    }
}

/// A window on bytes `[offset, offset + len)` of `pages` when they are
/// consecutive pages of one allocation (how titles usually hand them out):
/// address arithmetic in our view of it, and a commit the first time.
/// `None` otherwise (pages of several allocations, out of order, or
/// unknown, or a host failure), which `write_physical` then handles or
/// reports.
pub(crate) fn page_window(pages: &[usize], offset: usize, len: usize) -> Option<PageWindow> {
    let mut runs = runs(pages, offset, len)?;
    let (Some((slice, n)), true) = (runs.pop(), runs.is_empty()) else {
        return None;
    };
    if !slice.section.commit(slice.index, n) {
        return None;
    }
    // SAFETY: the run's pages hold `offset % PAGE + len` bytes of our view.
    let at = unsafe { slice.section.at(slice.index).add(offset % PAGE) };
    Some(PageWindow {
        _pin: xtp_gpu::pin_copy_back(at as usize, len),
        section: slice.section,
        at,
        len,
    })
}

/// Maps `n` pages at `address`, in place of what is mapped there; null
/// `pages` unmaps. Returns `address`, or null if a page or the host fails.
pub extern "system" fn XMemMapPhysicalPages(
    address: *mut c_void,
    n: usize,
    pages: *const usize,
) -> *mut c_void {
    let base = address as usize;
    let mut views = views();
    unmap(&mut views, base, n * PAGE);
    if pages.is_null() {
        return address;
    }
    // SAFETY: the title passes `n` page numbers.
    let pages = unsafe { std::slice::from_raw_parts(pages, n) };
    // One view per run of consecutive slices of one section.
    let (runs, unknown) = group(pages);
    let mut at = base;
    for (first, run) in runs {
        if !map(&mut views, at, run * PAGE, first) {
            return std::ptr::null_mut();
        }
        at += run * PAGE;
    }
    if let Some(page) = unknown {
        tracing::warn!(page, "XMemMapPhysicalPages: unknown page");
        return std::ptr::null_mut();
    }
    address
}

/// One page at each address; null `pages`, or a page number 0, unmaps
/// (BTP 6 passes 0 for addresses to clear).
pub extern "system" fn XMemMapPhysicalPagesScatter(
    addresses: *const *mut c_void,
    n: usize,
    pages: *const usize,
) -> BOOL {
    if addresses.is_null() {
        return 0;
    }
    for i in 0..n {
        // SAFETY: the title passes `n` addresses and, unless null, `n` pages.
        let (address, page) = unsafe {
            (
                *addresses.add(i),
                if pages.is_null() || *pages.add(i) == 0 {
                    std::ptr::null()
                } else {
                    pages.add(i)
                },
            )
        };
        if XMemMapPhysicalPages(address, 1, page).is_null() {
            return 0;
        }
    }
    1
}

/// Maps the physical pages shown at `[from, from + len)` at `to` as well,
/// in place of what `to` shows; `from` 0 unmaps. The GPU's page-mapping
/// copies (`xtp_gpu`, queue `CopyPageMappingsBatchX`): BTP 6
/// maps pages into a pool view, then has the GPU map them into its tiled
/// heap. Page by page; false if a source page is not mapped physical memory.
pub fn map_alias(to: u64, from: u64, len: u64) -> bool {
    let (to, from, len) = (to as usize, from as usize, len as usize);
    let mut views = views();
    unmap(&mut views, to, len);
    for k in (0..len).step_by(PAGE) {
        if from == 0 {
            break;
        }
        let at = from + k;
        let Some(slice) = views
            .range(..=at)
            .next_back()
            .filter(|(base, (size, _))| at < *base + size)
            .map(|(base, (_, s))| Slice {
                index: s.index + (at - base) / PAGE,
                section: s.section.clone(),
            })
        else {
            return false;
        };
        if !map(&mut views, to + k, PAGE, slice) {
            return false;
        }
    }
    true
}

/// Maps `slice` and the slices after it over `[base, base + len)`, which
/// must lie in one placeholder.
fn map(views: &mut Views, base: usize, len: usize, slice: Slice) -> bool {
    let _pin = xtp_gpu::pin_copy_back(slice.section.at(slice.index) as usize, len);
    // SAFETY: splits the placeholder around the range (failing harmlessly
    // when the range is the whole placeholder), then maps the section
    // there; both act only on placeholders.
    let view = unsafe {
        VirtualFree(
            base as *mut c_void,
            len,
            MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
        );
        MapViewOfFile3(
            slice.section.handle,
            GetCurrentProcess(),
            base as *const c_void,
            (slice.index * PAGE) as u64,
            len,
            MEM_REPLACE_PLACEHOLDER,
            PAGE_READWRITE,
            std::ptr::null_mut(),
            0,
        )
    };
    if view.Value as usize != base {
        tracing::warn!(
            address = format_args!("{base:#x}"),
            len,
            error = %std::io::Error::last_os_error(),
            "mapping physical pages failed (not a placeholder?)"
        );
        return false;
    }
    MAPS.fetch_add(1, Ordering::Relaxed);
    slice.section.maps.fetch_add(1, Ordering::Relaxed);
    // Pages committed before, through any view, need no host call.
    if !slice.section.commit(slice.index, len / PAGE) {
        // SAFETY: the view just mapped, back to a placeholder.
        unsafe { UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER) };
        UNMAPS.fetch_add(1, Ordering::Relaxed);
        return false;
    }
    // Noted first: the report then opens the guards of the pages' other
    // views, which do not see writes through this one yet.
    note_views(base, len, &slice, true);
    // New contents, read-write: the GPU side's copies no longer guard it.
    remapped(base, len);
    views.insert(base, (len, slice));
    true
}

/// Commits `[base, base + len)` of a view of a reserved section, which
/// commits those pages of the section for every view.
fn commit(base: usize, len: usize) -> bool {
    if on_wine() {
        return true; // sections are committed whole there
    }
    COMMITS.fetch_add(1, Ordering::Relaxed);
    // SAFETY: `[base, base + len)` is a view of a reserved section.
    if !unsafe { VirtualAlloc(base as *const c_void, len, MEM_COMMIT, PAGE_READWRITE) }.is_null() {
        return true;
    }
    let error = std::io::Error::last_os_error();
    if error.raw_os_error() == Some(ERROR_COMMITMENT_LIMIT as i32) {
        let mut info = PERFORMANCE_INFORMATION {
            cb: size_of::<PERFORMANCE_INFORMATION>() as u32,
            ..Default::default()
        };
        // SAFETY: the buffer and size describe a writable PERFORMANCE_INFORMATION.
        let queried = unsafe { K32GetPerformanceInfo(&mut info, info.cb) } != 0;
        xtp_loader::log_crash(&commit_limit_message(len, &error, queried.then_some(&info)));
    } else {
        tracing::warn!(len, %error, "committing physical pages failed");
    }
    false
}

fn commit_limit_message(
    len: usize,
    error: &std::io::Error,
    info: Option<&PERFORMANCE_INFORMATION>,
) -> String {
    let mut message = format!(
        "XTP: Windows commit limit reached while committing {len} bytes of physical pages: {error}.\n"
    );
    if let Some(info) = info {
        use std::fmt::Write;

        let committed = info.CommitTotal * info.PageSize / MIB;
        let limit = info.CommitLimit * info.PageSize / MIB;
        let _ = writeln!(
            message,
            "System commit: {committed} MiB of {limit} MiB in use."
        );
    }
    message.push_str(
        "Close other applications or increase the Windows paging file size, then retry.\n",
    );
    message
}

/// Returns `[base, base + len)` to placeholders; views reaching past it
/// are mapped again outside it. No GPU heap is ever open over these views
/// (`xtp_gpu::uma` opens physical pages on our own view of them), so none
/// holds them mapped. The GPU side's copies are told before each view goes
/// and after.
fn unmap(views: &mut Views, base: usize, len: usize) {
    // No bytes: no view to return. The search below would still find the
    // view holding `base` and cut it in two there, and its range of views
    // starting inside would run backwards (which `BTreeMap` panics on).
    if len == 0 {
        return;
    }
    let end = base + len;
    // Views do not overlap: the one holding `base`, then those starting
    // inside (the GPU's page mappings make one view per page).
    let first = views
        .range(..=base)
        .next_back()
        .filter(|(at, (size, _))| *at + size > base)
        .map(|(at, _)| *at);
    let hit: Vec<usize> = first
        .into_iter()
        .chain(views.range(base + 1..end).map(|(at, _)| *at))
        .collect();
    for at in hit {
        let Some((size, slice)) = views.remove(&at) else {
            continue;
        };
        // Before: guarded blocks get write access back, for the title's
        // writes until the view goes, and so do those of the pages' other
        // views, which do not see these writes.
        let _pin = xtp_gpu::pin_copy_back(slice.section.at(slice.index) as usize, size);
        remapped(at, size);
        // SAFETY: a view we mapped; it becomes a placeholder again.
        unsafe {
            UnmapViewOfFile2(
                GetCurrentProcess(),
                MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: at as *mut c_void,
                },
                MEM_PRESERVE_PLACEHOLDER,
            )
        };
        // Forgotten once gone: until then a copy of another view of its
        // pages guards this one with it.
        note_views(at, size, &slice, false);
        // After: a copy may have guarded the view, and noted it one, since
        // the report before.
        remapped(at, size);
        UNMAPS.fetch_add(1, Ordering::Relaxed);
        if at < base {
            map(views, at, base - at, slice.clone());
        }
        if at + size > end {
            let index = slice.index + (end - at) / PAGE;
            map(views, end, at + size - end, Slice { index, ..slice });
        }
    }
}

/// Views mapped of the allocation holding `page`, ours included, and commit
/// calls for it, for tests that check they stay put; `None` for an unknown
/// page.
#[cfg(test)]
pub(crate) fn section_activity(page: usize) -> Option<(u64, u64)> {
    let table = page_table();
    let section = &table.table.get(&page)?.section;
    let load = |n: &AtomicU64| n.load(Ordering::Relaxed);
    Some((load(&section.maps), load(&section.commits)))
}

/// The oldest layout we know; newer titles pass other flag values (Minecraft
/// 0x20000001) and read the same leading fields.
#[repr(C)]
#[derive(Debug, Default)]
pub struct WorkingSet {
    pub title_limit: usize,
    pub title_used: usize,
    pub tools_limit: usize,
    pub tools_used: usize,
    pub stack_used: u32,
    pub physical_pages: u32,
    pub gpu_optimal_limit: usize,
    pub gpu_optimal_used: usize,
    pub tools_gpu_optimal_limit: usize,
    pub tools_gpu_optimal_used: usize,
    pub kernel_pool_used: u32,
    pub kernel_pool_limit: u32,
}

/// The title's budget and our process's commit as what it uses. No tools
/// memory: a retail console.
// shortcut: all fields are filled whatever the flags ask for, and use is
// not split by bandwidth class (GPU-optimal use reads 0); track
// allocations per class if a title budgets from that field.
pub extern "system" fn XMemGetWorkingSetStatistics(_which: u32, out: *mut WorkingSet) -> BOOL {
    if out.is_null() {
        return 0;
    }
    let mut counters = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    let (gpu_optimal, standard) = xtp_gpu::console::current().title_memory();
    let stats = WorkingSet {
        title_limit: gpu_optimal + standard - TRANSFERRED.load(Ordering::Relaxed),
        title_used: counters.PagefileUsage,
        gpu_optimal_limit: gpu_optimal,
        ..Default::default()
    };
    ledger::note(|l| {
        l.budget = (
            l.budget.0 + 1,
            stats.title_limit as u64,
            stats.title_used as u64,
        )
    });
    // SAFETY: checked non-null; the title passes room for the structure.
    unsafe { out.write(stats) };
    1
}

/// Hands title memory (standard, in 2 MiB pages) to the system. The other
/// transfers involve tools memory, which a retail console has none of.
pub extern "system" fn XMemTransferMemory(kind: u32, size: usize) -> BOOL {
    const TITLE_TO_SYSTEM: u32 = 0;
    tracing::info!(kind, size, "XMemTransferMemory");
    if size == 0 {
        return 1;
    }
    let Some(size) = size.checked_next_multiple_of(2 * MIB) else {
        return 0;
    };
    let standard = xtp_gpu::console::current().title_memory().1;
    let moved = kind == TITLE_TO_SYSTEM
        && TRANSFERRED
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |t| {
                t.checked_add(size).filter(|&t| t <= standard)
            })
            .is_ok();
    moved.into()
}

/// The XMem ledger, a debugging aid off by default: with
/// `XTP_XMEM_LEDGER=1`, what the title asks of XMem is counted by category
/// (`XMemAlloc` by allocator id and memory type, the share our
/// `XMemAllocDefault` serves, physical pages, virtual memory, the budget we
/// report), and a line per category that changed is logged every 10 s: its
/// calls, failures, frees, live and peak bytes, and bytes asked in all.
/// `XMemAlloc` includes the committed resources of the title's device (its
/// allocator id 128), whose memory comes from the title's own allocator
/// hooks. Without the variable nothing is counted.
pub(crate) mod ledger {
    use std::collections::{BTreeMap, HashMap};
    use std::sync::{LazyLock, Mutex};

    /// Set (e.g. `XTP_XMEM_LEDGER=1`) to keep the ledger.
    pub const ENV: &str = "XTP_XMEM_LEDGER";

    /// What a line counts.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub enum Kind {
        /// `XMemAlloc`, whatever serves it (the title's hooks, else ours).
        Alloc {
            allocator: u8,
            memory: u8,
        },
        /// What reaches our `XMemAllocDefault`, by the same keys.
        Default {
            allocator: u8,
            memory: u8,
        },
        /// `XMemAllocatePhysicalPages`, in bytes of 64 KiB pages.
        PhysicalPages,
        /// `XMemVirtualAlloc` and the title's `VirtualAlloc`, reserved and
        /// committed; `VirtualFree` decommits and releases.
        Reserve,
        Commit,
        Decommit,
        Release,
    }

    impl Kind {
        /// `XMemAlloc` attributes: allocator id at bit 16 (8 bits), memory
        /// type at bit 29 (4 bits).
        pub fn of_attributes(attributes: u64, default: bool) -> Kind {
            let allocator = (attributes >> 16) as u8;
            let memory = (attributes >> 29 & 0xf) as u8;
            if default {
                Kind::Default { allocator, memory }
            } else {
                Kind::Alloc { allocator, memory }
            }
        }

        fn name(self) -> String {
            match self {
                Kind::Alloc { allocator, memory } => {
                    format!("XMemAlloc allocator {allocator} memory {memory}")
                }
                Kind::Default { allocator, memory } => {
                    format!("XMemAllocDefault allocator {allocator} memory {memory}")
                }
                Kind::PhysicalPages => "physical pages".into(),
                Kind::Reserve => "virtual reserve".into(),
                Kind::Commit => "virtual commit".into(),
                Kind::Decommit => "virtual decommit".into(),
                Kind::Release => "virtual release".into(),
            }
        }
    }

    /// Running totals of one kind.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Totals {
        pub calls: u64,
        pub failed: u64,
        pub frees: u64,
        /// Bytes handed out and not yet freed, and their highest.
        pub live: u64,
        pub peak: u64,
        /// Bytes asked for in all.
        pub asked: u64,
    }

    /// Every kind's totals, and the live blocks that a free gives back, by
    /// route (ours: `XMemAllocDefault`) and address: the title's hooks can
    /// hand on a block of ours, which is then live in both.
    #[derive(Debug, Default)]
    pub struct Ledger {
        totals: BTreeMap<Kind, Totals>,
        blocks: HashMap<(bool, usize), (Kind, u64)>,
        /// Frees of blocks the ledger did not see handed out.
        pub unknown_frees: u64,
        /// The budget last reported (`XMemGetWorkingSetStatistics`):
        /// calls, limit and use.
        pub budget: (u64, u64, u64),
    }

    impl Ledger {
        /// `size` bytes asked of `kind`; `at` is the block, 0 if refused.
        /// Physical pages pass any other value: they are freed by count.
        pub fn handed_out(&mut self, kind: Kind, size: u64, at: usize) {
            let t = self.totals.entry(kind).or_default();
            t.calls += 1;
            t.asked += size;
            if at == 0 {
                t.failed += 1;
                return;
            }
            t.live += size;
            t.peak = t.peak.max(t.live);
            if kind != Kind::PhysicalPages {
                let ours = matches!(kind, Kind::Default { .. });
                self.blocks.insert((ours, at), (kind, size));
            }
        }

        /// The block at `at` is given back, to our `XMemFreeDefault` if
        /// `ours`, else to `XMemFree`.
        pub fn given_back(&mut self, ours: bool, at: usize) {
            match self.blocks.remove(&(ours, at)) {
                Some((kind, size)) => self.freed(kind, size),
                None => self.unknown_frees += 1,
            }
        }

        /// `size` bytes of `kind` are given back.
        pub fn freed(&mut self, kind: Kind, size: u64) {
            let t = self.totals.entry(kind).or_default();
            t.frees += 1;
            t.live = t.live.saturating_sub(size);
        }

        /// A flow without a live amount (decommits, releases).
        pub fn flowed(&mut self, kind: Kind, size: u64, ok: bool) {
            let t = self.totals.entry(kind).or_default();
            t.calls += 1;
            t.asked += size;
            if !ok {
                t.failed += 1;
            }
        }

        #[cfg(test)]
        pub fn totals(&self, kind: Kind) -> Totals {
            self.totals.get(&kind).copied().unwrap_or_default()
        }

        /// Every kind's totals now, to compare later ones with.
        pub fn snapshot(&self) -> BTreeMap<Kind, Totals> {
            self.totals.clone()
        }

        /// One line per kind whose totals differ from `before`'s.
        pub fn changes(&self, before: &BTreeMap<Kind, Totals>) -> Vec<String> {
            const MIB: f64 = (1 << 20) as f64;
            self.totals
                .iter()
                .filter(|(k, t)| before.get(k) != Some(t))
                .map(|(k, t)| {
                    format!(
                        "{}: {} calls ({} failed), {} frees, live {:.1} MiB (peak {:.1}), {:.1} MiB asked in all",
                        k.name(),
                        t.calls,
                        t.failed,
                        t.frees,
                        t.live as f64 / MIB,
                        t.peak as f64 / MIB,
                        t.asked as f64 / MIB
                    )
                })
                .collect()
        }
    }

    static ON: LazyLock<bool> = LazyLock::new(|| std::env::var_os(ENV).is_some());
    static LEDGER: LazyLock<Mutex<Ledger>> = LazyLock::new(|| {
        start_logging();
        Mutex::default()
    });

    /// Runs `f` on the ledger when it is kept.
    pub fn note(f: impl FnOnce(&mut Ledger)) {
        if *ON {
            f(&mut LEDGER.lock().unwrap_or_else(|e| e.into_inner()));
        }
    }

    /// Every 10 s, the kinds that changed.
    fn start_logging() {
        let spawned = std::thread::Builder::new()
            .name("xtp-xmem-ledger".into())
            .spawn(|| {
                let mut before = BTreeMap::new();
                let mut budget = (0, 0, 0);
                loop {
                    std::thread::sleep(std::time::Duration::from_secs(10));
                    let ledger = LEDGER.lock().unwrap_or_else(|e| e.into_inner());
                    let lines = ledger.changes(&before);
                    let (calls, limit, used) = ledger.budget;
                    let new_budget = ledger.budget != budget;
                    let unknown = ledger.unknown_frees;
                    before = ledger.snapshot();
                    budget = ledger.budget;
                    drop(ledger);
                    for line in lines {
                        tracing::info!("xmem ledger: {line}");
                    }
                    if new_budget {
                        tracing::info!(
                            calls,
                            limit_mib = limit >> 20,
                            used_mib = used >> 20,
                            unknown_frees = unknown,
                            "xmem ledger: budget reported (XMemGetWorkingSetStatistics)"
                        );
                    }
                }
            });
        if let Err(e) = spawned {
            tracing::warn!("{ENV}: the ledger will not be logged: {e}");
        }
    }
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "xmem",
        // The other functions of this set are still provided by Windows.
        dlls: &["xmem.dll", "api-ms-win-core-memory-*"],
        exports: vec![
            HleExport {
                name: "VirtualProtect",
                address: xtp_gpu::title_virtual_protect as *const () as usize,
            },
            HleExport {
                name: "VirtualAlloc",
                address: title_virtual_alloc as *const () as usize,
            },
            HleExport {
                name: "VirtualFree",
                address: title_virtual_free as *const () as usize,
            },
            hle_export!(XMemAlloc),
            hle_export!(XMemFree),
            hle_export!(XMemAllocDefault),
            hle_export!(XMemFreeDefault),
            hle_export!(XMemIsToolsMemAvailable),
            hle_export!(XMemAllocatePhysicalPages),
            hle_export!(XMemFreePhysicalPages),
            hle_export!(XMemMapPhysicalPages),
            hle_export!(XMemMapPhysicalPagesScatter),
            hle_export!(XMemVirtualAlloc),
            hle_export!(XMemVirtualQuery),
            hle_export!(XMemGetWorkingSetStatistics),
            hle_export!(XMemTransferMemory),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query(address: *const c_void) -> BasicInformation {
        // SAFETY: plain data; zero is a valid value.
        let mut info: BasicInformation = unsafe { std::mem::zeroed() };
        // SAFETY: a pseudo handle needs no closing.
        let process = unsafe { GetCurrentProcess() };
        assert_eq!(XMemVirtualQuery(process, address, &mut info), 1);
        info
    }

    #[test]
    fn alloc_honours_alignment_and_frees() {
        for (field, align) in [(0, 16), (4, 16), (0xc, 4096), (0x10, 64 * 1024)] {
            let attributes = field << ALIGNMENT_SHIFT | 1 << 29; // a graphics type
            let p = XMemAlloc(100, attributes);
            assert!(!p.is_null());
            assert_eq!(p as usize % align, 0, "field {field:#x}");
            // SAFETY: 100 bytes were allocated.
            unsafe { p.cast::<u8>().write_bytes(0xab, 100) };
            XMemFree(p, attributes);
        }
        let p = XMemAlloc(4096, ALLOC_32BIT);
        assert!(!p.is_null() && (p as usize) < 1 << 32);
        XMemFree(p, ALLOC_32BIT);
    }

    #[test]
    fn virtual_alloc_meets_console_constraints() {
        const GPU_READ_WRITE: u32 = 0x2000;
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            4 * MIB,
            PAGES_2M | MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | 7 << 12, // 40-bit addresses
            PAGE_READWRITE | GPU_READ_WRITE,
        );
        assert!(!p.is_null());
        assert_eq!(p as usize % (2 * MIB), 0);
        assert!((p as usize) < 1 << 40);
        let info = query(p);
        assert_eq!(info.host.State, MEM_COMMIT);
        assert_eq!(info.host.Protect, PAGE_READWRITE);
        assert_eq!(info.flags & GRAPHICS, GRAPHICS);
        // Write-watched, for the GPU side's copies.
        let (mut found, mut count, mut granularity) = ([0usize; 4], 4, 0);
        // SAFETY: room for `count` addresses.
        let watched = unsafe {
            windows_sys::Win32::System::Memory::GetWriteWatch(
                0,
                p,
                4 * MIB,
                found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        };
        assert_eq!(watched, 0);
        // SAFETY: our own reservation.
        assert_ne!(unsafe { VirtualFree(p, 0, MEM_RELEASE) }, 0);
        assert_eq!(query(p).flags, 0);
        // No address field: above the 32-bit range and below 2^40, where
        // the console keeps it.
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            16 << 30,
            MEM_RESERVE,
            GRAPHICS,
            PAGE_NOACCESS,
        );
        assert!(!p.is_null() && (p as usize) >= 1 << 32, "{p:p}");
        assert!((p as usize) + (16 << 30) <= 1 << 40, "{p:p}");
        // SAFETY: our own reservation.
        assert_ne!(unsafe { VirtualFree(p, 0, MEM_RELEASE) }, 0);
    }

    #[test]
    fn the_low_va_reserve_hands_out_independent_mappable_placeholders() {
        let mut saved = LowVaReserve::new(3 * PAGE, false).unwrap();
        let base = saved.base;
        assert!(base + 3 * PAGE <= 1 << 32);
        let mut range = MEM_ADDRESS_REQUIREMENTS {
            LowestStartingAddress: base as *mut c_void,
            HighestEndingAddress: (base + 3 * PAGE - 1) as *mut c_void,
            Alignment: PAGE,
        };
        assert!(
            allocate_virtual(
                std::ptr::null_mut(),
                PAGE,
                PLACEHOLDER_RESERVE,
                PAGE_NOACCESS,
                Some(&mut range),
            )
            .is_null()
        );
        assert!(saved.take(0).is_none());
        assert!(saved.take(PAGE + 1).is_none());
        assert!(saved.take(4 * PAGE).is_none());

        let first = saved.take(2 * PAGE).unwrap();
        assert_eq!(first as usize, base);
        assert_eq!(query(first).host.RegionSize, 2 * PAGE);
        let rest = saved.take(PAGE).unwrap();
        assert_eq!(rest as usize, base + 2 * PAGE);
        assert_eq!(query(rest).host.RegionSize, PAGE);
        assert!(saved.take(PAGE).is_none());

        let pages = allocate(2);
        assert_eq!(XMemMapPhysicalPages(first, 2, pages.as_ptr()), first);
        assert_eq!(query(first).host.State, MEM_COMMIT);
        assert_eq!(query(rest).host.State, MEM_RESERVE);
        assert_eq!(XMemMapPhysicalPages(first, 2, std::ptr::null()), first);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
        assert_eq!(title_virtual_free(first, 0, MEM_RELEASE), 1);
        assert_eq!(title_virtual_free(rest, 0, MEM_RELEASE), 1);
    }

    #[test]
    fn automatic_low_va_reserve_waits_for_an_exact_size() {
        let mut saved = LowVaReserve::new(3 * PAGE, true).unwrap();
        assert!(saved.take(PAGE).is_none());
        assert!(saved.take(4 * PAGE).is_none());
        let at = saved.take(3 * PAGE).unwrap();
        let info = query(at).host;
        assert_eq!(info.State, MEM_RESERVE);
        assert_eq!(info.RegionSize, 3 * PAGE);
        assert!(saved.take(3 * PAGE).is_none());
        assert_eq!(title_virtual_free(at, 0, MEM_RELEASE), 1);
    }

    #[test]
    fn low_va_override_accepts_zero_and_rejects_out_of_range_values() {
        for (value, expected) in [
            ("0", Some(0)),
            ("185", Some(185)),
            ("4095", Some(4095)),
            ("4096", None),
            ("-1", None),
            ("", None),
        ] {
            assert_eq!(low_va_override_mib(OsStr::new(value)), expected);
        }
    }

    #[test]
    fn transfers_shrink_the_title_budget() {
        let mut before = WorkingSet::default();
        assert_eq!(XMemGetWorkingSetStatistics(0x2000_0001, &mut before), 1);
        assert!(before.title_used > 0);
        assert_eq!(XMemTransferMemory(0, 128 * MIB - 1), 1);
        assert_eq!(XMemTransferMemory(2, MIB), 0, "no tools memory");
        let standard = xtp_gpu::console::current().title_memory().1;
        assert_eq!(XMemTransferMemory(0, standard), 0, "over budget");
        let mut after = WorkingSet::default();
        XMemGetWorkingSetStatistics(0x2000_0001, &mut after);
        assert_eq!(before.title_limit - after.title_limit, 128 * MIB);
    }

    #[test]
    fn the_ledger_counts_live_blocks_by_route() {
        use ledger::{Kind, Ledger};
        let mut l = Ledger::default();
        // A committed resource's attributes (xtp_gpu::resource): allocator
        // 128, write-combined, 4 KiB aligned.
        let gpu = Kind::of_attributes(128 << 16 | 12 << ALIGNMENT_SHIFT | 1 << 29, false);
        assert_eq!(
            gpu,
            Kind::Alloc {
                allocator: 128,
                memory: 1
            }
        );
        l.handed_out(gpu, 4096, 0x1000);
        l.handed_out(gpu, 8192, 0x3000);
        l.handed_out(gpu, MIB as u64, 0);
        // The title's hook handing on a block of ours: live on both routes.
        let ours = Kind::of_attributes(128 << 16 | 1 << 29, true);
        l.handed_out(ours, 4096, 0x1000);
        l.given_back(false, 0x1000);
        let t = l.totals(gpu);
        assert_eq!(
            (t.calls, t.failed, t.frees, t.live, t.peak),
            (3, 1, 1, 8192, 12288)
        );
        assert_eq!(t.asked, 4096 + 8192 + MIB as u64);
        assert_eq!(l.totals(ours).live, 4096);
        l.given_back(true, 0x1000);
        assert_eq!(l.totals(ours).live, 0);
        l.given_back(false, 0x9000);
        assert_eq!(l.unknown_frees, 1);
        // Physical pages are freed by count; flows have no live amount.
        l.handed_out(Kind::PhysicalPages, 3 * PAGE as u64, 1);
        l.freed(Kind::PhysicalPages, PAGE as u64);
        assert_eq!(l.totals(Kind::PhysicalPages).live, 2 * PAGE as u64);
        l.flowed(Kind::Commit, MIB as u64, true);
        l.flowed(Kind::Commit, MIB as u64, false);
        let c = l.totals(Kind::Commit);
        assert_eq!(
            (c.calls, c.failed, c.live, c.asked),
            (2, 1, 0, 2 * MIB as u64)
        );
    }

    #[test]
    fn the_ledger_logs_what_changed() {
        use ledger::{Kind, Ledger};
        let mut l = Ledger::default();
        l.handed_out(Kind::PhysicalPages, MIB as u64, 1);
        l.flowed(Kind::Reserve, 2 * MIB as u64, true);
        let all = l.changes(&Default::default());
        assert_eq!(
            all,
            [
                "physical pages: 1 calls (0 failed), 0 frees, live 1.0 MiB (peak 1.0), 1.0 MiB asked in all",
                "virtual reserve: 1 calls (0 failed), 0 frees, live 0.0 MiB (peak 0.0), 2.0 MiB asked in all",
            ]
        );
        let before = l.snapshot();
        assert!(l.changes(&before).is_empty());
        l.freed(Kind::PhysicalPages, MIB as u64);
        let one = l.changes(&before);
        assert_eq!(one.len(), 1);
        assert!(
            one[0].starts_with(
                "physical pages: 1 calls (0 failed), 1 frees, live 0.0 MiB (peak 1.0)"
            )
        );
    }

    #[test]
    fn statistics_have_the_console_layout() {
        assert_eq!(size_of::<WorkingSet>(), 0x50);
        assert_eq!(size_of::<BasicInformation>(), 0x40);
        assert_eq!(std::mem::offset_of!(WorkingSet, gpu_optimal_limit), 0x28);
    }

    #[test]
    fn commit_limit_report_includes_request_and_system_commit() {
        let error = std::io::Error::from_raw_os_error(ERROR_COMMITMENT_LIMIT as i32);
        let info = PERFORMANCE_INFORMATION {
            CommitTotal: 512,
            CommitLimit: 768,
            PageSize: 4096,
            ..Default::default()
        };
        let report = commit_limit_message(4 * MIB, &error, Some(&info));
        assert!(report.contains("4194304 bytes"));
        assert!(report.contains("2 MiB of 3 MiB in use"));
        let fallback = commit_limit_message(4 * MIB, &error, None);
        for report in [&report, &fallback] {
            assert!(report.contains("os error 1455"));
            assert!(report.contains("Windows paging file size"));
        }
        assert!(!fallback.contains("System commit:"));
    }

    #[test]
    fn page_numbers_freed_are_given_again_past_the_title_s_21_bits() {
        let mut table = Pages {
            next: PAGE_LIMIT - 2 * FRAMES,
            table: HashMap::new(),
        };
        // The last numbers that fit, as before.
        assert_eq!(table.take(2), Some(PAGE_LIMIT - 2 * FRAMES));
        assert_eq!(table.next, PAGE_LIMIT);
        // Then the lowest free run, still packed into 21 bits.
        assert_eq!(table.take(3), Some(FIRST_PAGE));
        assert_eq!(table.take(usize::MAX), None);
        let f = |k: usize| FIRST_PAGE + k * FRAMES;
        assert_eq!(free_run([f(0), f(1), f(3)], FRAMES), Some(f(2)));
        assert_eq!(free_run([f(0), f(1), f(3)], 2 * FRAMES), Some(f(4)));
        assert_eq!(free_run([], PAGE_LIMIT - FIRST_PAGE), Some(FIRST_PAGE));
        assert_eq!(free_run([f(0)], PAGE_LIMIT - FIRST_PAGE), None);
        assert_eq!(free_run([], PAGE_LIMIT - FIRST_PAGE + FRAMES), None);
        let p = free_run([PAGE_LIMIT - FRAMES], FRAMES).unwrap();
        assert_eq!((p >> 4 & 0x1f_ffff) << 4, p);
    }

    #[test]
    fn physical_pages_map_alias_and_unmap() {
        let mut pages = [0usize; 3];
        let mut n = 3;
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        assert_eq!(n, 3);
        // Packed into 21 bits as `number >> 4`, as the title keeps them.
        for p in pages {
            assert_eq!((p >> 4 & 0x1f_ffff) << 4, p);
        }
        // Reservations of 64 KiB pages, as titles make them for mapping.
        let reserve = || {
            XMemVirtualAlloc(
                std::ptr::null_mut(),
                3 * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            )
        };
        let (a, b) = (reserve(), reserve());
        assert!(!a.is_null() && !b.is_null());
        let at = |p: *mut c_void, page: usize| p.wrapping_byte_add(page * PAGE);
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);

        // The same page at a second address shows the same bytes.
        assert_eq!(
            XMemMapPhysicalPages(at(b, 1), 1, pages[2..].as_ptr()),
            at(b, 1)
        );
        // SAFETY: both pages are mapped read-write.
        unsafe {
            at(a, 2).cast::<u8>().write(7);
            assert_eq!(at(b, 1).cast::<u8>().read(), 7);
            a.cast::<u8>().write(1);
        }
        // Writes into pages (DirectStorage's) show through their views.
        assert!(write_physical(&pages[2..], 1, &[9]));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { at(b, 1).cast::<u8>().add(1).read() }, 9);

        // Unmapping the middle page keeps its neighbours.
        assert_eq!(
            XMemMapPhysicalPages(at(a, 1), 1, std::ptr::null()),
            at(a, 1)
        );
        assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(query(at(a, 2)).host.State, MEM_COMMIT);
        // SAFETY: still mapped.
        assert_eq!(unsafe { at(a, 2).cast::<u8>().read() }, 7);

        // A scattered page lands in the hole.
        let addresses = [at(a, 1)];
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, pages.as_ptr()),
            1
        );
        // SAFETY: page 0 is now mapped at both `a` and `a + PAGE`.
        assert_eq!(unsafe { at(a, 1).cast::<u8>().read() }, 1);
        // Page number 0 unmaps.
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, [0usize].as_ptr()),
            1
        );
        assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
        // The GPU's page-mapping copy: what `a + 2 pages` shows, at `b`.
        assert!(map_alias(b as u64, at(a, 2) as u64, PAGE as u64));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { b.cast::<u8>().read() }, 7);
        assert!(map_alias(b as u64, 0, PAGE as u64));
        assert_eq!(query(b).host.State, MEM_RESERVE);
        assert!(!map_alias(b as u64, at(a, 1) as u64, PAGE as u64));
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
        assert_eq!(XMemIsToolsMemAvailable(), 0);
    }

    /// The index of the title's views (VIctory #1122): a page in a pool view
    /// that the GPU maps into its heap too is known at both addresses, and
    /// the index stays exact as one view is mapped again, unmapped in part
    /// or unmapped while the other stays.
    #[test]
    fn the_views_of_each_page_are_known() {
        let pages = allocate(2);
        let (pool, heap) = (reserve(2) as usize, reserve(2) as usize);
        let others = |at: usize| {
            let mut out = Vec::new();
            other_views(at..at + 2 * PAGE, &mut out);
            out.sort_unstable();
            out
        };
        let pool_at = pool as *mut c_void;
        assert_eq!(XMemMapPhysicalPages(pool_at, 2, pages.as_ptr()), pool_at);
        assert!(others(pool).is_empty(), "each page shown once");
        // The GPU maps the pool's second page into its heap.
        assert!(map_alias(heap as u64, (pool + PAGE) as u64, PAGE as u64));
        assert_eq!(others(pool), [(pool + PAGE, heap)]);
        assert_eq!(others(heap), [(heap, pool + PAGE)]);
        // There the first page instead, and the second after it.
        assert!(map_alias(heap as u64, pool as u64, PAGE as u64));
        assert_eq!(others(pool), [(pool, heap)]);
        let second = (pool + PAGE) as u64;
        assert!(map_alias((heap + PAGE) as u64, second, PAGE as u64));
        assert_eq!(others(heap), [(heap, pool), (heap + PAGE, pool + PAGE)]);
        // The pool's first page unmapped: its view goes, and its second
        // page is mapped again on its own.
        assert_eq!(XMemMapPhysicalPages(pool_at, 1, std::ptr::null()), pool_at);
        assert_eq!(others(pool), [(pool + PAGE, heap + PAGE)]);
        assert_eq!(others(heap), [(heap + PAGE, pool + PAGE)]);
        // The heap unmapped, the pool's second page staying.
        assert!(map_alias(heap as u64, 0, 2 * PAGE as u64));
        assert!(others(pool).is_empty() && others(heap).is_empty());
        let rest = (pool + PAGE) as *mut c_void;
        assert_eq!(XMemMapPhysicalPages(rest, 1, std::ptr::null()), rest);
        {
            let a = aliases();
            let known = |at: usize| (0..2).any(|k| a.by_address.contains_key(&(at + k * PAGE)));
            assert!(!known(pool) && !known(heap), "nothing kept of them");
        }
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    /// What a host copy of `blocks` pages of title memory at `at` asks
    /// before it copies (the GPU side's guards): the pages written since its
    /// last asking, by address.
    fn copy_asks(at: usize, blocks: usize) -> Vec<usize> {
        let pages = xtp_gpu::copy_asks(at, at + blocks * PAGE).expect("view memory");
        pages.into_iter().map(|p| p * PAGE).collect()
    }

    /// Whether a copy's guard leaves `at` read-only.
    fn guarded(at: usize) -> bool {
        use windows_sys::Win32::System::Memory::PAGE_READONLY;
        query(at as *const c_void).host.Protect == PAGE_READONLY
    }

    /// DirectStorage's writes into physical pages (the #1122 review): they
    /// go through our view of them, which no copy guards and which does not
    /// fault, so each write opens the guards of every title view of the
    /// pages it wrote: as a copy, after each run, and in place, when the
    /// window drops. The pages it did not write stay guarded.
    #[test]
    fn runtime_writes_open_the_guards_of_every_view() {
        let pages = allocate(2);
        let (pool, heap) = (reserve(2) as usize, reserve(1) as usize);
        let pool_at = pool as *mut c_void;
        assert_eq!(XMemMapPhysicalPages(pool_at, 2, pages.as_ptr()), pool_at);
        // The second page in the GPU's heap too.
        assert!(map_alias(heap as u64, (pool + PAGE) as u64, PAGE as u64));
        // A copy of the pool takes both pages and guards them, and the heap
        // with the second; then a copy of the heap takes it.
        assert_eq!(copy_asks(pool, 2), [pool, pool + PAGE]);
        assert_eq!(copy_asks(heap, 1), [heap]);
        assert!(copy_asks(pool, 2).is_empty() && copy_asks(heap, 1).is_empty());
        assert!(guarded(pool) && guarded(pool + PAGE) && guarded(heap));
        assert!(xtp_gpu::guarding());

        // A copy into the second page: opened in both views.
        assert!(write_physical(&pages[1..], 8, &[7]));
        assert!(!guarded(pool + PAGE) && !guarded(heap));
        assert!(guarded(pool), "the first page was not written");
        assert_eq!(copy_asks(pool, 2), [pool + PAGE]);
        assert_eq!(copy_asks(heap, 1), [heap]);
        // SAFETY: mapped, readable.
        assert_eq!(unsafe { ((heap + 8) as *const u8).read() }, 7);

        // In place, over both pages: opened when the window drops.
        let mut window = page_window(&pages, PAGE - 4, 8).expect("one run");
        window.bytes().fill(9);
        assert!(guarded(pool) && guarded(pool + PAGE) && guarded(heap));
        drop(window);
        assert!(!guarded(pool) && !guarded(pool + PAGE) && !guarded(heap));
        assert_eq!(copy_asks(pool, 2), [pool, pool + PAGE]);
        assert_eq!(copy_asks(heap, 1), [heap]);
        // SAFETY: mapped, readable.
        assert_eq!(unsafe { ((heap + 3) as *const u8).read() }, 9);

        assert!(map_alias(heap as u64, 0, PAGE as u64));
        assert_eq!(XMemMapPhysicalPages(pool_at, 2, std::ptr::null()), pool_at);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    /// The late view aycow found on VIctory #1122, with xmem's own ordering
    /// (the #1122 review): a copy guards a pool view, then the GPU maps the
    /// page into its heap. xmem notes the heap's view before it reports it,
    /// so the report opens the pool's guard, and the title's write through
    /// the heap is seen by the pool's copy. From then on the heap is guarded
    /// with the pool, and a write through it faults and opens both.
    #[test]
    fn a_late_view_opens_the_guard_of_the_first() {
        let pages = allocate(1);
        let (pool, heap) = (reserve(1) as usize, reserve(1) as usize);
        let pool_at = pool as *mut c_void;
        assert_eq!(XMemMapPhysicalPages(pool_at, 1, pages.as_ptr()), pool_at);
        assert_eq!(copy_asks(pool, 1), [pool]);
        assert!(copy_asks(pool, 1).is_empty(), "clean");
        assert!(guarded(pool));
        assert!(map_alias(heap as u64, pool as u64, PAGE as u64));
        assert!(!guarded(pool), "the late view's report opened the guard");
        // SAFETY: mapped read-write.
        unsafe { ((heap + 8) as *mut u8).write_volatile(1) };
        assert_eq!(copy_asks(pool, 1), [pool], "the write through the heap");
        assert!(guarded(pool) && guarded(heap), "guarded together now");
        // SAFETY: mapped; the guards' handler opens the write.
        unsafe { ((heap + 8) as *mut u8).write_volatile(2) };
        assert!(!guarded(pool) && !guarded(heap));
        assert_eq!(copy_asks(pool, 1), [pool]);
        // SAFETY: mapped, readable.
        assert_eq!(unsafe { ((pool + 8) as *const u8).read_volatile() }, 2);
        assert!(map_alias(heap as u64, 0, PAGE as u64));
        assert_eq!(XMemMapPhysicalPages(pool_at, 1, std::ptr::null()), pool_at);
        assert_eq!(XMemFreePhysicalPages(1, pages.as_ptr()), 1);
    }

    #[test]
    fn pending_copy_back_through_title_imports() {
        use std::io::{Read, Seek};
        use std::os::windows::io::{AsRawHandle, AsRawSocket};

        const CHILD: &str = "XTP_TEST_PENDING_IMPORTS";
        if std::env::var_os(CHILD).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "modules::xmem::tests::pending_copy_back_through_title_imports",
                    "--exact",
                ])
                .env(CHILD, "1")
                .env("XTP_READMAP", "1")
                .status()
                .unwrap();
            assert!(status.success());
            return;
        }
        xtp_gpu::configure_copy_back(None);
        let pages = allocate(1);
        let at = reserve(1);
        let alias = reserve(1);
        assert_eq!(XMemMapPhysicalPages(at, 1, pages.as_ptr()), at);
        assert!(map_alias(alias as u64, at as u64, PAGE as u64));
        let publish = |writer, value| {
            assert_eq!(
                xtp_gpu::publish_copy_back_for_test(at as usize, &mut vec![value; PAGE], writer),
                Some(0)
            );
            assert_eq!(
                query(alias).host.Protect,
                windows_sys::Win32::System::Memory::PAGE_NOACCESS
            );
        };

        publish(1, 0x5a);
        let mut file = tempfile::tempfile().unwrap();
        let mut n = 0;
        assert_ne!(
            crate::modules::files::WriteFile(
                file.as_raw_handle(),
                alias.cast(),
                4,
                &mut n,
                std::ptr::null_mut()
            ),
            0
        );
        assert_eq!(n, 4);
        file.rewind().unwrap();
        let mut bytes = [0; 4];
        file.read_exact(&mut bytes).unwrap();
        assert_eq!(bytes, [0x5a; 4]);

        publish(6, 0x64);
        crate::modules::files::check_fwrite_from(alias.cast(), PAGE, 0x64);

        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let client = std::net::TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (mut server, _) = listener.accept().unwrap();
        publish(7, 0x83);
        assert_eq!(
            crate::modules::net::send(client.as_raw_socket() as usize, alias.cast(), 4, 0),
            4
        );
        server.read_exact(&mut bytes).unwrap();
        assert_eq!(bytes, [0x83; 4]);

        publish(2, 0xc1);
        // SAFETY: both views map the same page of the title.
        unsafe { alias.cast::<u8>().write_volatile(0x75) };
        assert_eq!(&page_bytes(pages[0])[..4], &[0x75, 0xc1, 0xc1, 0xc1]);

        publish(4, 0x33);
        let mut old = 0;
        // SAFETY: two views of the test and a valid output for the previous protection.
        unsafe {
            let ro = windows_sys::Win32::System::Memory::PAGE_READONLY;
            assert_ne!(xtp_gpu::title_virtual_protect(alias, 4096, ro, &mut old), 0);
            assert_eq!(old, PAGE_READWRITE);
            assert_eq!(alias.cast::<u8>().read_volatile(), 0x33);
            copy_asks(at as usize, 1);
            assert_eq!(query(alias).host.Protect, ro);
            assert_eq!(
                xtp_gpu::publish_copy_back_for_test(at as usize, &mut [0x45], 5),
                None
            );
            assert_ne!(
                xtp_gpu::title_virtual_protect(alias, PAGE, PAGE_READWRITE, &mut old),
                0
            );
            assert_eq!(old, ro);
            alias.cast::<u8>().write_volatile(0x58);
        }
        assert_eq!(page_bytes(pages[0])[0], 0x58);
        publish(3, 0x92);
        assert!(map_alias(alias as u64, 0, PAGE as u64));
        assert_eq!(query(alias).host.State, MEM_RESERVE);
        assert_eq!(&page_bytes(pages[0])[..4], &[0x92; 4]);
        assert_eq!(XMemMapPhysicalPages(at, 1, std::ptr::null()), at);
        assert_eq!(XMemFreePhysicalPages(1, pages.as_ptr()), 1);
    }

    #[test]
    fn mapping_no_pages_changes_nothing() {
        let pages = allocate(2);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            2 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 2, pages.as_ptr()), a);
        // SAFETY: both pages are mapped read-write.
        unsafe { a.cast::<u8>().write(5) };
        let second = a.wrapping_byte_add(PAGE);

        // A count of 0 at the start of a view, inside it and past it: with
        // pages, without (an unmap), and as the GPU's page-mapping copy.
        for at in [a, second, a.wrapping_byte_add(2 * PAGE)] {
            assert_eq!(XMemMapPhysicalPages(at, 0, pages.as_ptr()), at);
            assert_eq!(XMemMapPhysicalPages(at, 0, std::ptr::null()), at);
            assert!(map_alias(at as u64, a as u64, 0));
            assert!(map_alias(at as u64, 0, 0));
        }

        // The view is still the one view it was.
        let view = views().get(&(a as usize)).map(|v| v.0);
        assert_eq!(view, Some(2 * PAGE));
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(query(second).host.State, MEM_COMMIT);
        // SAFETY: still mapped.
        assert_eq!(unsafe { a.cast::<u8>().read() }, 5);

        // A count of 1 still unmaps that page alone.
        assert_eq!(XMemMapPhysicalPages(second, 1, std::ptr::null()), second);
        assert_eq!(query(second).host.State, MEM_RESERVE);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(XMemMapPhysicalPages(a, 1, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    /// `n` pages of one new allocation.
    fn allocate(n: usize) -> Vec<usize> {
        let (mut pages, mut given) = (vec![0; n], n);
        assert_eq!(
            XMemAllocatePhysicalPages(0, &mut given, pages.as_mut_ptr()),
            1
        );
        assert_eq!(given, n);
        pages
    }

    fn slice(page: usize) -> Slice {
        page_table().table.get(&page).cloned().expect("known page")
    }

    /// The bytes of one page, through our view of its allocation.
    fn page_bytes(page: usize) -> Vec<u8> {
        let slice = slice(page);
        assert!(slice.section.commit(slice.index, 1));
        // SAFETY: a committed page of our view, which `slice` keeps mapped.
        unsafe { std::slice::from_raw_parts(slice.section.at(slice.index), PAGE) }.to_vec()
    }

    /// A reservation of 64 KiB pages, as titles make them for mapping.
    fn reserve(n: usize) -> *mut c_void {
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            n * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!p.is_null());
        p
    }

    #[test]
    fn runs_group_consecutive_pages_of_one_allocation() {
        let (a, b) = (allocate(3), allocate(1));
        let lengths = |pages: &[usize], offset, len| {
            runs(pages, offset, len).map(|r| r.iter().map(|&(_, n)| n).collect::<Vec<_>>())
        };
        assert_eq!(lengths(&a, 0, 3 * PAGE), Some(vec![3]));
        // Only the pages the bytes reach.
        assert_eq!(lengths(&a, PAGE + 5, 10), Some(vec![1]));
        assert_eq!(lengths(&a, PAGE - 1, 2), Some(vec![2]));
        assert_eq!(lengths(&a, 0, 0), Some(vec![]));
        // Out of order, or across allocations: one run each.
        assert_eq!(
            lengths(&[a[0], a[2], a[1]], 0, 3 * PAGE),
            Some(vec![1, 1, 1])
        );
        assert_eq!(lengths(&[a[1], a[2], b[0]], 0, 3 * PAGE), Some(vec![2, 1]));
        // A slice that follows on in another section starts a run.
        assert_eq!(lengths(&[b[0], a[1]], 0, 2 * PAGE), Some(vec![1, 1]));
        // Too few pages for the bytes, or an unknown one.
        assert_eq!(lengths(&a, 0, 3 * PAGE + 1), None);
        assert_eq!(lengths(&[a[0], 0x10], 0, 2 * PAGE), None);
        assert_eq!(lengths(&a, usize::MAX, 2), None);
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    #[test]
    fn writes_cross_runs_and_allocations() {
        let (a, b) = (allocate(2), allocate(1));
        let pages = [a[0], a[1], b[0]];
        let data: Vec<u8> = (0..2 * PAGE + 64).map(|i| (i * 7 % 251) as u8).collect();
        let offset = PAGE - 32;
        // Past the third page: the second run is too short.
        assert!(!write_physical(&pages, offset, &data));
        assert!(write_physical(&pages, offset, &data[..2 * PAGE]));
        let seen: Vec<u8> = pages.iter().flat_map(|&p| page_bytes(p)).collect();
        assert!(seen[..offset].iter().all(|&b| b == 0), "before the offset");
        assert_eq!(seen[offset..offset + 2 * PAGE], data[..2 * PAGE]);
        assert!(seen[offset + 2 * PAGE..].iter().all(|&b| b == 0), "after");
        assert!(
            !write_physical(&[a[0], 0x10], 0, &data[..PAGE + 1]),
            "unknown page"
        );
        assert_eq!(XMemFreePhysicalPages(2, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    #[test]
    fn page_windows_cover_one_run_in_place() {
        let (a, b) = (allocate(3), allocate(1));
        let mut window = page_window(&a, PAGE + 3, PAGE).expect("one run");
        assert_eq!(window.bytes().len(), PAGE);
        window.bytes().fill(0x5a);
        drop(window);
        let (second, third) = (page_bytes(a[1]), page_bytes(a[2]));
        assert!(second[..3].iter().all(|&b| b == 0));
        assert!(second[3..].iter().all(|&b| b == 0x5a));
        assert!(third[..3].iter().all(|&b| b == 0x5a));
        assert!(third[3..].iter().all(|&b| b == 0));
        // Several runs, or an unknown page: no window.
        assert!(page_window(&[a[0], b[0]], 0, PAGE + 1).is_none());
        assert!(page_window(&[a[1], a[0]], 0, PAGE + 1).is_none());
        assert!(page_window(&[0x10], 0, 1).is_none());
        // One page of several is still one run.
        assert!(page_window(&[a[0], b[0]], 0, PAGE).is_some());
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    /// Our view of an allocation is mapped once; writes then map nothing,
    /// and commit each page once.
    #[test]
    fn writes_map_nothing_and_commit_once() {
        let a = allocate(4);
        assert_eq!(
            section_activity(a[0]).unwrap(),
            (1, 0),
            "our view, nothing committed"
        );
        let data = pattern(PAGE);
        for round in 0..1000 {
            let p = round % 4;
            if round % 2 == 0 {
                assert!(write_physical(&a[p..=p], 0, &data));
            } else {
                page_window(&a, p * PAGE, PAGE)
                    .expect("one run")
                    .bytes()
                    .fill(1);
            }
        }
        assert_eq!(section_activity(a[0]).unwrap(), (1, 4), "a commit per page");
        // Pages already committed: none, however the run is cut.
        let data = pattern(4 * PAGE - 10);
        assert!(write_physical(&a, 10, &data));
        assert_eq!(section_activity(a[0]).unwrap(), (1, 4));
        let seen: Vec<u8> = a.iter().flat_map(|&p| page_bytes(p)).collect();
        assert_eq!(seen[10..], data[..]);
        assert_eq!(XMemFreePhysicalPages(4, a.as_ptr()), 1);
    }

    fn pattern(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i * 13 + i / 4093) as u8).collect()
    }

    fn state(at: usize) -> u32 {
        query(at as *const c_void).host.State
    }

    /// Commitment belongs to the section: pages written through our view
    /// show committed in title views mapped later, with no further commit;
    /// pages a title view committed need none for our writes.
    #[test]
    fn title_views_and_ours_share_commitment_and_bytes() {
        let a = allocate(2);
        assert!(write_physical(&a[..1], 0, b"from the runtime"));
        let view = reserve(2);
        assert_eq!(XMemMapPhysicalPages(view, 2, a.as_ptr()), view);
        // The second page was committed by the mapping, the first before.
        assert_eq!(section_activity(a[0]).unwrap(), (2, 2));
        assert_eq!(state(view as usize), MEM_COMMIT);
        assert_eq!(query(view).flags & GRAPHICS, GRAPHICS);
        // SAFETY: both pages are mapped read-write.
        unsafe {
            assert_eq!(
                std::slice::from_raw_parts(view.cast::<u8>(), 16),
                b"from the runtime"
            );
            view.cast::<u8>().add(PAGE + 7).write(42);
        }
        assert_eq!(page_bytes(a[1])[7], 42, "the title's write, in our view");
        assert!(write_physical(&a[1..], 8, &[43]));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { view.cast::<u8>().add(PAGE + 8).read() }, 43);
        assert_eq!(
            section_activity(a[0]).unwrap(),
            (2, 2),
            "no commit for either"
        );
        assert_eq!(XMemMapPhysicalPages(view, 2, std::ptr::null()), view);
        assert_eq!(XMemFreePhysicalPages(2, a.as_ptr()), 1);
    }

    /// Freeing an allocation's pages unmaps our view of it, once no title
    /// view shows them any more. Run alone in a child process: our views go
    /// at the lowest free address from 2^40 on, so a test running alongside
    /// maps its own allocation where ours was as soon as it is gone (one
    /// run in five read `MEM_RESERVE` there). The address cannot tell the
    /// two apart: same base, same kind of view.
    #[test]
    fn freeing_an_allocation_unmaps_our_view() {
        use windows_sys::Win32::System::Memory::MEM_FREE;
        if !alone("freeing_an_allocation_unmaps_our_view", false) {
            return;
        }
        let a = allocate(2);
        let ours = slice(a[0]).section.base;
        assert!(write_physical(&a, 0, &pattern(2 * PAGE)));
        assert_eq!(state(ours), MEM_COMMIT);
        assert_eq!(XMemFreePhysicalPages(2, a.as_ptr()), 1);
        assert_eq!(state(ours), MEM_FREE);

        // Still mapped by the title: readable there, our view kept.
        let b = allocate(1);
        let ours = slice(b[0]).section.base;
        assert!(write_physical(&b, 0, &[7]));
        let view = reserve(1);
        assert_eq!(XMemMapPhysicalPages(view, 1, b.as_ptr()), view);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
        assert!(!write_physical(&b, 0, &[8]), "a freed page is unknown");
        // SAFETY: still mapped.
        assert_eq!(unsafe { view.cast::<u8>().read() }, 7);
        assert_eq!(state(ours), MEM_COMMIT);
        assert_eq!(XMemMapPhysicalPages(view, 1, std::ptr::null()), view);
        assert_eq!(state(ours), MEM_FREE);
    }

    /// Many writers into one allocation, some through windows and some as
    /// copies, some across two allocations, while the title maps and unmaps
    /// views of the same pages (one of them read-only, as the GPU side's
    /// copies guard views) and allocates and frees other pages, some of
    /// them while being written.
    #[test]
    fn writes_race_mappings_and_frees() {
        use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
        const WRITERS: usize = 16;
        const ROUNDS: usize = 200;
        let (a, b) = (allocate(WRITERS * 2), allocate(WRITERS));
        // The first half of the pages shows through a read-only title view.
        let guarded = reserve(WRITERS);
        assert_eq!(XMemMapPhysicalPages(guarded, WRITERS, a.as_ptr()), guarded);
        let mut old = 0;
        // SAFETY: our title view; only its protection changes.
        let protected = unsafe { VirtualProtect(guarded, WRITERS * PAGE, PAGE_READONLY, &mut old) };
        assert_ne!(protected, 0);
        let stop = std::sync::atomic::AtomicBool::new(false);
        let value = |w: usize, round: usize| (w * 31 + round) as u8;
        std::thread::scope(|s| {
            let writers: Vec<_> = (0..WRITERS)
                .map(|w| {
                    let (a, b) = (&a, &b);
                    s.spawn(move || {
                        for round in 0..ROUNDS {
                            let v = value(w, round);
                            // In place, in the guarded half.
                            let mut window = page_window(&a[w..=w], 0, PAGE).expect("one run");
                            window.bytes().fill(v);
                            drop(window);
                            // A copy into the unguarded half, half a page in
                            // each of two allocations.
                            let pages = [a[WRITERS + w], b[w]];
                            assert!(write_physical(&pages, PAGE / 2, &vec![v; PAGE]));
                        }
                    })
                })
                .collect();
            // The title maps and unmaps the unguarded pages meanwhile.
            s.spawn(|| {
                let view = reserve(WRITERS);
                while !stop.load(Ordering::Relaxed) {
                    let at = XMemMapPhysicalPages(view, WRITERS, a[WRITERS..].as_ptr());
                    assert_eq!(at, view);
                    assert_eq!(XMemMapPhysicalPages(view, WRITERS, std::ptr::null()), view);
                }
            });
            // And allocates and frees pages, some of them while a writer
            // writes them: a write lands or finds the page gone.
            s.spawn(|| {
                while !stop.load(Ordering::Relaxed) {
                    let c = allocate(2);
                    std::thread::scope(|s| {
                        s.spawn(|| {
                            if let Some(mut w) = page_window(&c, 0, 2 * PAGE) {
                                w.bytes().fill(9);
                            }
                        });
                        assert_eq!(XMemFreePhysicalPages(2, c.as_ptr()), 1);
                    });
                }
            });
            // The others stop once the writers are done.
            let written = writers.into_iter().all(|w| w.join().is_ok());
            stop.store(true, Ordering::Relaxed);
            assert!(written, "a writer failed");
        });
        for w in 0..WRITERS {
            let v = value(w, ROUNDS - 1);
            // SAFETY: the guarded view is still mapped, read-only.
            let seen =
                unsafe { std::slice::from_raw_parts(guarded.cast::<u8>().add(w * PAGE), PAGE) };
            assert!(seen.iter().all(|&x| x == v), "writer {w}, guarded page");
            let unguarded = page_bytes(a[WRITERS + w]);
            assert!(unguarded[PAGE / 2..].iter().all(|&x| x == v), "writer {w}");
            assert!(
                page_bytes(b[w])[..PAGE / 2].iter().all(|&x| x == v),
                "writer {w}"
            );
        }
        // The guard is still there.
        assert_eq!(query(guarded).host.Protect, PAGE_READONLY);
        assert_eq!(
            XMemMapPhysicalPages(guarded, WRITERS, std::ptr::null()),
            guarded
        );
        assert_eq!(XMemFreePhysicalPages(a.len(), a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(b.len(), b.as_ptr()), 1);
    }

    /// With heaps open over title memory (`XTP_UMA=1`), the title's calls
    /// that change it still succeed: xmem has the heaps released first, then
    /// opened again over what remains. Physical pages get theirs on our own
    /// view, so the title's views still unmap. Run alone in a child process:
    /// with UMA on, other tests' plain host frees would fail under heaps.
    #[test]
    fn title_calls_succeed_under_open_heaps() {
        use windows_sys::Win32::System::Memory::{MEM_DECOMMIT, MEM_FREE};
        use xtp_gpu::uma::{lookup, settle};
        const TEST: &str = "title_calls_succeed_under_open_heaps";
        if !alone(TEST, true) {
            return;
        }
        if let Err(reason) = xtp_gpu::uma::ready() {
            let strict =
                std::env::var_os("XTP_REQUIRE_GPU").is_some_and(|v| !v.is_empty() && v != "0");
            assert!(!strict, "{TEST} needs a host GPU: {reason}");
            eprintln!("SKIPPED {TEST}: {reason}");
            return;
        }
        const LEN: usize = 16 * PAGE;
        let open = |at: *mut c_void, len: usize| lookup(at as usize, len).is_some();
        let page = |at: *mut c_void, n: usize| at.wrapping_byte_add(n * PAGE);
        let host_decommit = |at: *mut c_void, len: usize| {
            // SAFETY: title memory of this test; refused under a heap.
            unsafe { VirtualFree(at, len, MEM_DECOMMIT) != 0 }
        };

        // Reserved and committed at once, as the title's pools are.
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            LEN,
            MEM_RESERVE | MEM_COMMIT,
            GRAPHICS,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        settle();
        assert!(open(a, LEN), "a heap over it");
        assert!(!host_decommit(page(a, 1), PAGE), "the heap pins it");
        // SAFETY: committed read-write.
        unsafe { page(a, 8).cast::<u8>().write(0x5a) };
        // The title decommits a page: done, and what remains opens again.
        assert_eq!(title_virtual_free(page(a, 1), PAGE, MEM_DECOMMIT), 1);
        assert_eq!(state(page(a, 1) as usize), MEM_RESERVE);
        settle();
        assert!(open(a, PAGE) && open(page(a, 2), LEN - 2 * PAGE));
        assert!(!open(page(a, 1), 1));
        // SAFETY: still committed.
        assert_eq!(unsafe { page(a, 8).cast::<u8>().read() }, 0x5a);
        // Commits it again (the memory API set's `VirtualAlloc`), then
        // releases it all.
        let again = title_virtual_alloc(page(a, 1), PAGE, MEM_COMMIT, PAGE_READWRITE);
        assert_eq!(again, page(a, 1));
        settle();
        assert!(open(page(a, 1), PAGE));
        assert_eq!(title_virtual_free(a, 0, MEM_RELEASE), 1);
        assert_eq!(state(a as usize), MEM_FREE);
        assert!(!open(a, 1));

        // Reserved, then committed and decommitted by the title.
        let b = XMemVirtualAlloc(
            std::ptr::null_mut(),
            LEN,
            MEM_RESERVE,
            GRAPHICS,
            PAGE_READWRITE,
        );
        assert!(!b.is_null());
        settle();
        assert!(!open(b, 1));
        assert_eq!(title_virtual_alloc(b, LEN, MEM_COMMIT, PAGE_READWRITE), b);
        settle();
        assert!(open(b, LEN));
        assert_eq!(title_virtual_free(b, LEN, MEM_DECOMMIT), 1);
        settle();
        assert!(!open(b, 1));
        assert_eq!(title_virtual_free(b, 0, MEM_RELEASE), 1);

        // 32-bit memory, freed by `XMemFreeDefault`.
        let c = XMemAllocDefault(4 * PAGE, ALLOC_32BIT);
        assert!(!c.is_null());
        settle();
        assert!(open(c, 4 * PAGE));
        XMemFreeDefault(c, ALLOC_32BIT);
        assert_eq!(state(c as usize), MEM_FREE);

        // Physical pages, all written (so committed): a heap on our view.
        let pages = allocate(16);
        let ours = slice(pages[0]).section.base;
        assert!(write_physical(&pages, 0, &pattern(16 * PAGE)));
        settle();
        assert!(lookup(ours, 16 * PAGE).is_some(), "a heap on our view");
        // The title maps them and unmaps them again.
        let view = reserve(16);
        assert_eq!(XMemMapPhysicalPages(view, 16, pages.as_ptr()), view);
        assert_eq!(XMemMapPhysicalPages(view, 16, std::ptr::null()), view);
        assert_eq!(
            state(view as usize),
            MEM_RESERVE,
            "the title's view unmapped"
        );
        // Freeing them unmaps our view, its heap released first.
        assert_eq!(XMemFreePhysicalPages(16, pages.as_ptr()), 1);
        assert_eq!(state(ours), MEM_FREE);
        assert!(lookup(ours, 1).is_none());
    }

    /// Runs test `name` of this module alone, in a child process, with
    /// `XTP_UMA=1` if `uma` (else without it), and checks that it passed
    /// there. True in that child, where the test then goes on.
    fn alone(name: &str, uma: bool) -> bool {
        const CHILD: &str = "XTP_ALONE_TEST_CHILD";
        if std::env::var_os(CHILD).is_some() {
            assert_eq!(xtp_gpu::uma::enabled(), uma, "XTP_UMA as asked");
            return true;
        }
        let module = module_path!();
        let path = module.split_once("::").map_or(module, |(_, rest)| rest);
        let test = format!("{path}::{name}");
        let exe = std::env::current_exe().expect("the test binary");
        let mut child = std::process::Command::new(exe);
        child
            .args([test.as_str(), "--exact", "--nocapture", "--test-threads=1"])
            .env(CHILD, "1");
        if uma {
            child.env("XTP_UMA", "1");
        } else {
            child.env_remove("XTP_UMA");
        }
        let out = child.output().expect("the child runs");
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        println!("{text}");
        assert!(out.status.success(), "{test} failed in its child process");
        assert!(text.contains("1 passed"), "{test} did not run in its child");
        false
    }

    /// Writes into physical pages as DirectStorage's workers make them:
    /// 64 KiB each, from 1 and from 16 threads, in place through a window on
    /// one allocation, as a copy, and as a copy across two allocations (half
    /// a page in each). Prints MB/s; run it with
    /// `cargo test -p xtp-hle --release --locked -- --ignored --nocapture physical_page_write_throughput`.
    #[test]
    #[ignore = "benchmark"]
    fn physical_page_write_throughput() {
        const PAGES_PER_THREAD: usize = 64;
        const PASSES: usize = 20;
        let (a, b) = (
            allocate(16 * PAGES_PER_THREAD),
            allocate(16 * PAGES_PER_THREAD),
        );
        let data = vec![0x3c_u8; PAGE];
        type Write<'a> = &'a (dyn Fn(usize) + Sync);
        let run = |name: &str, threads: usize, passes: usize, write: Write| {
            let started = std::time::Instant::now();
            std::thread::scope(|s| {
                for t in 0..threads {
                    s.spawn(move || {
                        for _ in 0..passes {
                            for i in 0..PAGES_PER_THREAD {
                                write(t * PAGES_PER_THREAD + i);
                            }
                        }
                    });
                }
            });
            let bytes = (threads * passes * PAGES_PER_THREAD * PAGE) as f64;
            let mb_per_s = bytes / started.elapsed().as_secs_f64() / 1e6;
            println!("{name:<34} {threads:>2} threads: {mb_per_s:>8.0} MB/s");
        };
        let window = |p: usize| {
            let mut w = page_window(&a[p..=p], 0, PAGE).expect("one run");
            w.bytes().fill(p as u8);
        };
        let copy = |p: usize| assert!(write_physical(&a[p..=p], 0, &data));
        let across = |p: usize| assert!(write_physical(&[a[p], b[p]], PAGE / 2, &data));
        run("window, first touch", 16, 1, &window);
        for threads in [1, 16] {
            run("window", threads, PASSES, &window);
            run("copy", threads, PASSES, &copy);
            run("copy across two allocations", threads, PASSES, &across);
        }
        assert_eq!(XMemFreePhysicalPages(a.len(), a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(b.len(), b.as_ptr()), 1);
    }
}
