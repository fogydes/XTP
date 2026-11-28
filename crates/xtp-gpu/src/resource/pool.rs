//! Host textures placed in large heaps of our own (task 450ec9ba,
//! `docs/analysis/vram.md`).
//!
//! A committed resource is a heap of its own, made resident on its own.
//! Past about 9.5 GB resident on the PC's RTX 4080, VidMm admits about one
//! new resident allocation a second whatever its size, so a world loading
//! thousands of textures stalled for good (task 6bb65399). Here the
//! textures the title fills itself are placed (`CreatePlacedResource`) in
//! heaps of 256 MiB (by default) that a thread of ours makes ahead of need:
//! - two classes by placement alignment: 4 KiB for textures D3D12's small
//!   resource rule allows ([`may_be_small`]), 64 KiB for the others. MSAA
//!   textures (4 MiB) are always targets, which stay committed;
//! - heaps are made not resident and not zeroed (the title's contents are
//!   uploaded over every texel before a shader reads them), then asked to
//!   become resident within the budget, one request a heap
//!   (`residency`): no request for any texture;
//! - the thread keeps a reserve of free bytes in each class (heaps still
//!   becoming resident count), so the title's thread only places. A
//!   texture placed in a heap not resident yet is handed to the GPU once
//!   it is (`HostTexture::ready`); only in the first [`HEAP_WAIT`] of a
//!   heap does a placement wait for it, [`PLACE_WAIT`] at most, so a heap
//!   costs the title's thread that much at most, never a texture. Only
//!   when the reserve is gone does the title's thread make a heap itself
//!   (not resident, not waited for beyond that);
//! - past the video memory budget (QueryVideoMemoryInfo, and its change
//!   notification) no heap is made: the texture is made as a committed one
//!   (asked to become resident within the budget, retried) and the refusal
//!   is logged;
//! - a texture's range goes back when its host copy is dropped (lists hold
//!   it until their submission has run); an empty heap is released unless
//!   the reserve needs it, and all empty ones go when usage passes the
//!   budget.
//!
//! Textures of the 64 KiB class of two tiles or more are reserved
//! resources (`CreateReservedResource`) whose 64 KiB tiles are mapped onto
//! the heaps a unit at a time, a unit being a standard mip of an array
//! slice or a slice's packed mips ([`Tiles`]), as `texture` asks: the
//! units whose title memory is committed get memory when the texture's
//! contents are detiled, and a texture the title's streaming left behind
//! gives its tiles back while the texture itself stays (its views stay
//! valid: from tiled resources tier 2 on, unmapped tiles read zero and
//! writes into them are dropped), mapped again by its next upload. Before,
//! every host texture held memory for its whole mip chain for its whole
//! life: in the Series S world 24,800 textures held 11 GiB, 85 % of the
//! video memory budget, and the textures shaders pick by index waited for
//! room (task cba81604). Mappings are queued on the host queue, so they
//! fall between submissions in their order; ranges a texture gives back
//! return to the heaps once the host fence passes the submission after the
//! unmapping (a heap is never released under an in-flight read). Only
//! resident heaps get tiles. On by default with the pool on a host with
//! less than 32 GiB of RAM ([`low_memory_host`]), off elsewhere:
//! `XTP_TEXTURE_TILES=1` turns it on, `0` off (these textures are then
//! placed as before). The idea of reserved textures mapped by level from their
//! title memory comes from xcl #258 (branch
//! `d3d12x/streamed-texture-level-tiles`, GPL-2.0-only: its behaviour
//! only, this is written without its code).
//!
//! The pool is on by default on a host with less than 32 GiB of RAM and off
//! elsewhere ([`enabled`]): `XTP_TEXTURE_POOL=1` turns it on, `0` off
//! (every host texture a committed resource, as before; measured in game in
//! docs/analysis/vram.md §3.2 and §3.5). `XTP_TEXTURE_HEAP_MIB` sets the
//! heap size (the small class gets an eighth), `XTP_TEXTURE_RESERVE_MIB` the
//! reserve (one heap).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, Weak};
use std::time::{Duration, Instant};

use windows::Win32::Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIAdapter3, IDXGIFactory4,
};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::Interface;

use super::ranges::Ranges;
use super::residency::Residency;

const KIB_4: u64 = 4 << 10;
const KIB_64: u64 = 64 << 10;
const MIB: u64 = 1 << 20;

/// A tile of a reserved resource.
pub(crate) const TILE: u64 = D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES as u64;

/// The fewest tiles a texture is reserved with: one of a single tile gains
/// nothing from it.
const TILED_FROM: u64 = 2;

/// The most pieces a unit's tiles are taken in, within one heap (a heap
/// whose free bytes are that scattered is passed over).
const PIECES: usize = 16;

/// How long after its heap was made a placement may wait for the heap's
/// residency, and at most how long one placement waits: below VidMm's
/// ceiling a fresh heap is resident well within it; past the ceiling the
/// title's thread waits that long at most a heap, not a texture.
const HEAP_WAIT: Duration = Duration::from_millis(100);
const PLACE_WAIT: Duration = Duration::from_millis(10);

/// Whether host textures go to the pool: with `XTP_TEXTURE_POOL` set (and
/// not `0`), or, when it is not set, on a host with less than
/// [`LOW_MEMORY_BYTES`] of RAM. On such a host every committed DEFAULT
/// texture costs its whole size in Windows commit, resident or not
/// (docs/analysis/vram.md §3.4), and the world pages the whole machine;
/// pooled reserved textures cost only the tiles title memory fills. Off
/// elsewhere: in game (Series X, docs/analysis/vram.md §3.2) it reached the
/// world without loading faster than committed textures, with more video
/// memory resident (its reserve).
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let value = std::env::var("XTP_TEXTURE_POOL").ok();
        let on = asks_pool(value.as_deref(), low_memory_host());
        if on {
            let why = if value.is_some() {
                "XTP_TEXTURE_POOL"
            } else {
                "less than 32 GiB of RAM"
            };
            tracing::info!(why, "host textures the title fills go to pooled heaps");
        }
        on
    })
}

/// Below this much RAM, the pool and its reserved textures are on unless
/// switched off.
pub(crate) const LOW_MEMORY_BYTES: u64 = 32 << 30;

/// Whether the machine has less than [`LOW_MEMORY_BYTES`] of RAM. Unit
/// tests keep the pool's own defaults whatever the machine.
pub(crate) fn low_memory_host() -> bool {
    static LOW: OnceLock<bool> = OnceLock::new();
    *LOW.get_or_init(|| !cfg!(test) && total_ram().is_some_and(|b| b < LOW_MEMORY_BYTES))
}

fn total_ram() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut status = MEMORYSTATUSEX {
        dwLength: size_of::<MEMORYSTATUSEX>() as u32,
        ..Default::default()
    };
    // SAFETY: a sized structure of ours.
    (unsafe { GlobalMemoryStatusEx(&mut status) } != 0).then_some(status.ullTotalPhys)
}

/// `XTP_TEXTURE_POOL`'s value: set and not empty or `0` turns it on, `0`
/// or empty off; unset follows `low` (a low-memory host).
fn asks_pool(value: Option<&str>, low: bool) -> bool {
    match value {
        None => low,
        Some(v) => !v.is_empty() && v != "0",
    }
}

static POOL: OnceLock<Option<Pool>> = OnceLock::new();

/// The run's pool if it was made (the frame log never makes it).
pub(crate) fn global_if_made() -> Option<&'static Pool> {
    POOL.get()?.as_ref()
}

/// The run's pool, made with the host device on first use; `None` when
/// switched off or without a host GPU. In tests, the textures a thread
/// makes inside [`tests::Pooled`] go to a pool of the tests' own.
pub(crate) fn global() -> Option<&'static Pool> {
    #[cfg(test)]
    if tests::POOLED.get() {
        return tests::pool_for_tests();
    }
    POOL.get_or_init(|| {
        if !enabled() {
            return None;
        }
        let host = crate::host::host()?;
        let config = Config::from_env();
        tracing::info!(
            heap_mib = config.heap[Class::Normal.index()] / MIB,
            small_heap_mib = config.heap[Class::Small.index()] / MIB,
            reserve_mib = config.reserve[Class::Normal.index()] / MIB,
            largest_mib = config.largest / MIB,
            tiles = config.tiles,
            "host textures placed in pooled heaps"
        );
        Some(Pool::new(
            host.device.clone(),
            config,
            Box::new(crate::dxgi::video_memory),
        ))
    })
    .as_ref()
}

/// Placement classes, by alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Class {
    /// 4 KiB: D3D12's small resources.
    Small,
    /// 64 KiB: every other texture that is no target.
    Normal,
}

impl Class {
    const ALL: [Class; 2] = [Class::Small, Class::Normal];

    fn index(self) -> usize {
        match self {
            Class::Small => 0,
            Class::Normal => 1,
        }
    }

    pub(crate) fn alignment(self) -> u64 {
        match self {
            Class::Small => KIB_4,
            Class::Normal => KIB_64,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Class::Small => "4 KiB",
            Class::Normal => "64 KiB",
        }
    }
}

/// Sizes, by class (index [`Class::index`]).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Config {
    /// Bytes of one heap.
    pub(crate) heap: [u64; 2],
    /// Free bytes kept ahead.
    pub(crate) reserve: [u64; 2],
    /// The largest texture placed; larger ones are committed.
    pub(crate) largest: u64,
    /// Whether textures of the 64 KiB class are reserved and their tiles
    /// mapped as they are filled ([`Tiles`]), where the device can.
    pub(crate) tiles: bool,
}

impl Config {
    /// From `XTP_TEXTURE_HEAP_MIB` (256), `XTP_TEXTURE_RESERVE_MIB` (one
    /// heap: making a heap not resident takes about a millisecond) and
    /// `XTP_TEXTURE_TILES` (on when set to something else than `0`, `off`
    /// or `false`; unset, on a host with less than 32 GiB of RAM only).
    pub(crate) fn from_env() -> Config {
        let mib = |name: &str| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.trim().parse::<u64>().ok())
                .filter(|&v| v > 0)
        };
        let heap = mib("XTP_TEXTURE_HEAP_MIB").unwrap_or(256).clamp(8, 4096) * MIB;
        let reserve = mib("XTP_TEXTURE_RESERVE_MIB").map_or(heap, |v| v * MIB);
        Config {
            tiles: asks_tiles(
                std::env::var("XTP_TEXTURE_TILES").ok().as_deref(),
                low_memory_host(),
            ),
            ..Config::sized(heap, reserve)
        }
    }

    /// Heaps of `heap` bytes and a reserve of `reserve` for the 64 KiB
    /// class; an eighth of each for the small class. Textures placed, not
    /// reserved.
    pub(crate) fn sized(heap: u64, reserve: u64) -> Config {
        let small = (heap / 8).next_multiple_of(KIB_64).max(KIB_64);
        Config {
            heap: [small, heap],
            reserve: [(reserve / 8).max(small.min(reserve)), reserve],
            largest: heap / 4,
            tiles: false,
        }
    }
}

/// Whether `XTP_TEXTURE_TILES`'s value asks for reserved textures: set,
/// and not empty, `0`, `off` or `false`.
fn asks_tiles(value: Option<&str>, low: bool) -> bool {
    let Some(v) = value.map(str::trim) else {
        return low;
    };
    !(v.is_empty() || v == "0" || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false"))
}

/// Whether D3D12's small resource rule may give the host texture `d` a
/// 4 KiB placement: no target, one sample, one array slice, and a most
/// detailed mip of at most 16 tiles of 4 KiB in the standard tile shapes.
/// The device has the last word (`GetResourceAllocationInfo` grants it or
/// not).
pub(crate) fn may_be_small(d: &D3D12_RESOURCE_DESC) -> bool {
    let targets = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    let texture = matches!(
        d.Dimension,
        D3D12_RESOURCE_DIMENSION_TEXTURE1D
            | D3D12_RESOURCE_DIMENSION_TEXTURE2D
            | D3D12_RESOURCE_DIMENSION_TEXTURE3D
    );
    if !texture || d.Flags.0 & targets.0 != 0 || d.SampleDesc.Count > 1 || d.DepthOrArraySize != 1 {
        return false;
    }
    let Some((bytes, block)) = super::element(d.Format.0 as u32) else {
        return false;
    };
    // Texels (or blocks) of a 4 KiB tile, by element size.
    let (tw, th) = match bytes {
        1 => (64, 64),
        2 => (64, 32),
        4 => (32, 32),
        8 => (32, 16),
        16 => (16, 16),
        _ => return false,
    };
    let (w, h) = (d.Width.div_ceil(block), u64::from(d.Height).div_ceil(block));
    w.div_ceil(tw) * h.div_ceil(th) <= 16
}

/// Whether 4 KiB small-texture placement is active on `device`. Enabled by
/// default except on AMD adapters (vendor `0x1002`), where 4 KiB placed
/// textures caused device losses; `XTP_TEXTURE_SMALL=1` or `0`/`off`/`false`
/// overrides on any adapter.
// Ported from TaynanGT/Vice-Tay commit 7d62e536 (GPL-3.0-or-later).
fn small_placement(device: &ID3D12Device) -> bool {
    let env = std::env::var("XTP_TEXTURE_SMALL").ok();
    let on = asks_small(env.as_deref(), is_amd(device));
    if !on && env.is_none() {
        tracing::info!(
            "texture pool: 4 KiB small-texture placement disabled on AMD adapter \
             (set XTP_TEXTURE_SMALL=1 to force)"
        );
    }
    on
}

fn asks_small(value: Option<&str>, amd: bool) -> bool {
    match value.map(str::trim) {
        None => !amd,
        Some(v) => {
            !(v.is_empty()
                || v == "0"
                || v.eq_ignore_ascii_case("off")
                || v.eq_ignore_ascii_case("false"))
        }
    }
}

fn is_amd(device: &ID3D12Device) -> bool {
    const AMD_VENDOR: u32 = 0x1002;
    // SAFETY: queries DXGI adapter vendor ID matching `device`'s LUID.
    unsafe {
        let luid = device.GetAdapterLuid();
        CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            .and_then(|f| f.EnumAdapterByLuid::<IDXGIAdapter1>(luid))
            .and_then(|a| a.GetDesc1())
            .is_ok_and(|d| d.VendorId == AMD_VENDOR)
    }
}

/// The pool: heaps by class, and the thread that keeps their reserve.
pub(crate) struct Pool {
    inner: Arc<Inner>,
    thread: Option<std::thread::JoinHandle<()>>,
}

/// The budget and usage of local video memory for a device.
type Budget = Box<dyn Fn(&ID3D12Device) -> Option<(u64, u64)> + Send + Sync>;

struct Inner {
    device: ID3D12Device,
    config: Config,
    budget: Budget,
    state: Mutex<State>,
    /// Wakes the thread.
    wake: Condvar,
    /// DXGI's budget change notification, if it gave one.
    changes: Option<BudgetEvent>,
    /// Heaps refuse `CREATE_NOT_ZEROED` (an older runtime): made without.
    zeroed: AtomicBool,
    /// Whether eligible textures take 4 KiB (`Class::Small`) placement.
    small: bool,
    /// The device's tiled resources tier (`D3D12_OPTIONS`).
    tier: i32,
    /// Whether each host format may be reserved (`FORMAT_SUPPORT2_TILED`),
    /// asked once.
    tileable: Mutex<HashMap<i32, bool>>,
    /// Reserved textures alive, and the bytes of their tiles mapped.
    reserved: AtomicU64,
    mapped: AtomicU64,
}

// SAFETY: D3D12 devices and heaps are free-threaded; the state is behind
// its lock.
unsafe impl Send for Inner {}
// SAFETY: as above.
unsafe impl Sync for Inner {}

/// How long to hold off new `CreateHeap` calls after a driver refusal
/// (`E_OUTOFMEMORY` or similar), while still serving placements from free
/// ranges of already-created heaps.
const HEAP_FAIL_BACKOFF: Duration = Duration::from_millis(500);

fn backoff_active(until: Option<Instant>, now: Instant) -> bool {
    until.is_some_and(|u| now < u)
}

fn backoff_wait(until: Option<Instant>, now: Instant, idle: Duration) -> Duration {
    until
        .and_then(|u| u.checked_duration_since(now))
        .filter(|d| !d.is_zero())
        .map_or(idle, |d| d.min(idle))
}

struct State {
    classes: [Vec<Heap>; 2],
    /// Heaps to release (on the thread).
    releasing: Vec<ID3D12Heap>,
    /// Ranges of the 64 KiB class a reserved texture gave back (heap id,
    /// offset, length), and the host fence value after which they return
    /// to their heap: the unmapping was queued before the submission that
    /// signals it.
    retired: std::collections::VecDeque<(u64, u64, u64, u64)>,
    stop: bool,
    woken: bool,
    next_id: u64,
    /// Whether heaps are held back by the budget now (logged once each
    /// time it starts).
    over_budget: bool,
    /// Hold off new `CreateHeap` calls until this instant after a refusal.
    backoff_until: Option<Instant>,
    /// Driver `CreateHeap` refusals since the pool was made (rate-limits logs).
    refusals: u64,
    stats: Stats,
}

/// One heap of the pool.
struct Heap {
    id: u64,
    heap: ID3D12Heap,
    size: u64,
    ranges: Ranges,
    /// Ranges taken: a placed texture's, a piece of a reserved texture's
    /// tiles (mapped or retired).
    textures: u32,
    residency: Arc<Residency>,
    /// When it was made.
    made: Instant,
}

/// What the pool did since it was made.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Stats {
    /// Heaps made, and of them by the title's thread (the reserve was
    /// empty).
    pub(crate) heaps: u64,
    pub(crate) emergency: u64,
    pub(crate) released: u64,
    /// Residency requests made (one a heap).
    pub(crate) asked: u64,
    /// Textures placed, and those the pool did not place (committed
    /// instead).
    pub(crate) placed: u64,
    pub(crate) not_placed: u64,
    /// Textures handed out before their heap was resident (views made of
    /// them meanwhile are null).
    pub(crate) early: u64,
    /// Heaps not made because of the budget.
    pub(crate) over_budget: u64,
    /// Textures reserved ([`Tiles`]); units of them mapped, unmapped, and
    /// left unmapped for want of room in a resident heap; tile mapping
    /// calls queued.
    pub(crate) reserved: u64,
    pub(crate) units_mapped: u64,
    pub(crate) units_unmapped: u64,
    pub(crate) units_short: u64,
    pub(crate) tile_calls: u64,
}

/// What the pool holds now.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Holding {
    pub(crate) heaps: u64,
    /// Heaps not resident yet.
    pub(crate) pending: u64,
    pub(crate) bytes: u64,
    pub(crate) used: u64,
    /// Ranges taken: placed textures, pieces of reserved textures' tiles.
    pub(crate) textures: u64,
    /// Reserved textures alive, and the bytes of their tiles mapped (part
    /// of `used`); bytes given back waiting for the fence (also part of it).
    pub(crate) reserved: u64,
    pub(crate) mapped: u64,
    pub(crate) retired: u64,
}

/// A texture's memory in the pool: its range (a placed texture), given
/// back when dropped, or its tiles (a reserved texture).
pub(crate) struct Slot {
    inner: Arc<Inner>,
    kind: Kind,
}

enum Kind {
    Range {
        class: Class,
        heap: u64,
        offset: u64,
        len: u64,
    },
    /// Its tiles go back when the last holder of them goes (the texture's
    /// host copy, and a mapping under way).
    Tiles(Arc<Tiles>),
}

impl Slot {
    /// Its class, heap offset and length (a placed texture's).
    #[cfg(test)]
    pub(crate) fn place(&self) -> (Class, u64, u64) {
        match self.kind {
            Kind::Range {
                class, offset, len, ..
            } => (class, offset, len),
            Kind::Tiles(_) => panic!("a reserved texture has no range"),
        }
    }

    /// Its tiles, if it is reserved.
    #[cfg(test)]
    pub(crate) fn tiles(&self) -> Option<&Arc<Tiles>> {
        match &self.kind {
            Kind::Tiles(tiles) => Some(tiles),
            Kind::Range { .. } => None,
        }
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        if let Kind::Range {
            class,
            heap,
            offset,
            len,
        } = self.kind
        {
            self.inner.give_range(class, heap, offset, len);
        }
    }
}

impl std::fmt::Debug for Slot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            Kind::Range {
                class,
                heap,
                offset,
                len,
            } => f
                .debug_struct("Slot")
                .field("class", class)
                .field("heap", heap)
                .field("offset", offset)
                .field("len", len)
                .finish(),
            Kind::Tiles(tiles) => f
                .debug_struct("Slot")
                .field("units", &tiles.units.len())
                .field("mapped", &tiles.mapped())
                .finish(),
        }
    }
}

/// A unit of a reserved texture's tiles, mapped or not as a whole: a
/// standard mip of an array slice, or a slice's packed mips.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Unit {
    /// Its first subresource (the slice's first packed mip for its packed
    /// mips; any of them names the same tiles).
    subresource: u32,
    /// The array slice (0 for a 3D texture), and its mips: the first and
    /// how many.
    pub(crate) slice: u32,
    pub(crate) first_mip: u32,
    pub(crate) mips: u32,
    /// Its tiles.
    pub(crate) tiles: u32,
}

/// The heap ranges a mapped unit's tiles are in: its heap and the pieces
/// (offset, length, in bytes), in the order of the unit's tiles.
#[derive(Debug)]
struct Pieces {
    heap: u64,
    ranges: Vec<(u64, u64)>,
}

/// Ranges taken for a unit's tiles ([`Inner::take_tiles`]): the heap, its
/// id and the pieces.
type Taken = (ID3D12Heap, u64, Vec<(u64, u64)>);

/// One mapping call of a [`Tiles::map`]: the heap, the units mapped into
/// it and their ranges, in order.
struct Call {
    heap: ID3D12Heap,
    units: Vec<usize>,
    ranges: Vec<(u64, u64)>,
}

/// A reserved texture's tiles: its units and the heap ranges of those
/// mapped. Registered by texture ([`tiles_of`]) while it lives.
pub(crate) struct Tiles {
    inner: Arc<Inner>,
    texture: ID3D12Resource,
    units: Vec<Unit>,
    /// The mapped units' ranges, by unit, under the lock every mapping of
    /// the texture is queued under (their order on the queue is the
    /// lock's).
    pieces: Mutex<Vec<Option<Pieces>>>,
}

// SAFETY: D3D12 resources are free-threaded; the mappings are behind their
// lock.
unsafe impl Send for Tiles {}
// SAFETY: as above.
unsafe impl Sync for Tiles {}

/// What a [`Tiles::map`] did: units mapped and unmapped, and those wanted
/// left unmapped (no room in a resident heap now).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Mapped {
    pub(crate) added: u32,
    pub(crate) removed: u32,
    pub(crate) short: u32,
}

/// Reserved textures' tiles by texture (its interface pointer: the tiles
/// hold a reference, so it is not reused while registered).
static TILES: Mutex<Option<HashMap<usize, Weak<Tiles>>>> = Mutex::new(None);

/// The tiles of host texture `texture`, if it is a reserved one of the
/// pool.
pub(crate) fn tiles_of(texture: &ID3D12Resource) -> Option<Arc<Tiles>> {
    TILES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()?
        .get(&(texture.as_raw() as usize))?
        .upgrade()
}

impl Tiles {
    /// Its units, in the order [`Tiles::map`] takes what it wants.
    pub(crate) fn units(&self) -> &[Unit] {
        &self.units
    }

    /// The bytes of its tiles mapped now.
    pub(crate) fn mapped(&self) -> u64 {
        let pieces = self.pieces.lock().unwrap_or_else(|e| e.into_inner());
        self.mapped_of(&pieces)
    }

    fn mapped_of(&self, pieces: &[Option<Pieces>]) -> u64 {
        self.units
            .iter()
            .zip(pieces)
            .filter(|(_, p)| p.is_some())
            .map(|(u, _)| u64::from(u.tiles) * TILE)
            .sum()
    }

    /// Whether every unit `wanted` says is mapped (tests).
    #[cfg(test)]
    pub(crate) fn holds(&self, wanted: &[bool]) -> bool {
        let pieces = self.pieces.lock().unwrap_or_else(|e| e.into_inner());
        pieces
            .iter()
            .zip(wanted)
            .all(|(p, &want)| !want || p.is_some())
    }

    /// Whether any unit is mapped.
    pub(crate) fn any(&self) -> bool {
        let pieces = self.pieces.lock().unwrap_or_else(|e| e.into_inner());
        pieces.iter().any(Option::is_some)
    }

    /// Whether the resident heaps have free bytes for the units `wanted`
    /// says that are not mapped (in one heap each, or in pieces).
    pub(crate) fn room_for(&self, wanted: &[bool]) -> bool {
        let missing: Vec<u64> = {
            let pieces = self.pieces.lock().unwrap_or_else(|e| e.into_inner());
            self.units
                .iter()
                .zip(pieces.iter())
                .zip(wanted)
                .filter(|((_, p), want)| **want && p.is_none())
                .map(|((u, _), _)| u64::from(u.tiles) * TILE)
                .collect()
        };
        if missing.is_empty() {
            return true;
        }
        let s = self.inner.lock();
        let free: Vec<u64> = s.classes[Class::Normal.index()]
            .iter()
            .filter(|h| h.residency.is_ready())
            .map(|h| h.ranges.free())
            .collect();
        missing.iter().all(|&len| free.iter().any(|&f| f >= len))
            && missing.iter().sum::<u64>() <= free.iter().sum::<u64>()
    }

    /// Maps the units `wanted` says (one flag a unit) and unmaps the
    /// others, queued on `host`'s queue: work submitted after this reads
    /// the new mapping, work submitted before the old one. A unit mapped
    /// anew holds nothing yet: its texels come with the next copy into it.
    /// A wanted unit no resident heap has room for stays unmapped (`short`;
    /// the pool's thread is woken to make room).
    pub(crate) fn map(&self, host: &crate::host::Host, wanted: &[bool]) -> Mapped {
        let inner = &self.inner;
        inner.reclaim(host.completed());
        let mut pieces = self.pieces.lock().unwrap_or_else(|e| e.into_inner());
        let mut done = Mapped::default();
        let before = self.mapped_of(&pieces);
        // Unmapped first: their ranges may serve the units mapped below,
        // once the fence passes.
        let gone: Vec<usize> = (0..self.units.len())
            .filter(|&i| pieces[i].is_some() && !wanted.get(i).copied().unwrap_or(false))
            .collect();
        if !gone.is_empty() {
            self.unmap(host, &gone, &mut pieces);
            done.removed = gone.len() as u32;
        }
        // Mapped by heap: one call for the units that land in the same one.
        let mut calls: Vec<Call> = Vec::new();
        for i in 0..self.units.len() {
            if pieces[i].is_some() || !wanted.get(i).copied().unwrap_or(false) {
                continue;
            }
            let len = u64::from(self.units[i].tiles) * TILE;
            let Some((heap, id, ranges)) = inner.take_tiles(len) else {
                done.short += 1;
                continue;
            };
            match calls.iter_mut().find(|c| c.heap.as_raw() == heap.as_raw()) {
                Some(call) => {
                    call.units.push(i);
                    call.ranges.extend_from_slice(&ranges);
                }
                None => calls.push(Call {
                    heap,
                    units: vec![i],
                    ranges: ranges.clone(),
                }),
            }
            pieces[i] = Some(Pieces { heap: id, ranges });
            done.added += 1;
        }
        for Call {
            heap,
            units,
            ranges,
        } in &calls
        {
            let (coordinates, sizes) = self.regions(units);
            let starts: Vec<u32> = ranges.iter().map(|&(at, _)| (at / TILE) as u32).collect();
            let counts: Vec<u32> = ranges.iter().map(|&(_, len)| (len / TILE) as u32).collect();
            let flags = vec![D3D12_TILE_RANGE_FLAG_NONE; ranges.len()];
            // SAFETY: a reserved texture of ours and a heap of the pool
            // the ranges were taken from (alive while a range is taken);
            // as many tiles in the regions as in the ranges.
            unsafe {
                host.queue().UpdateTileMappings(
                    &self.texture,
                    units.len() as u32,
                    Some(coordinates.as_ptr()),
                    Some(sizes.as_ptr()),
                    heap,
                    ranges.len() as u32,
                    Some(flags.as_ptr()),
                    Some(starts.as_ptr()),
                    Some(counts.as_ptr()),
                    D3D12_TILE_MAPPING_FLAG_NONE,
                )
            };
        }
        let after = self.mapped_of(&pieces);
        drop(pieces);
        inner.mapped.fetch_add(after, Ordering::Relaxed);
        inner.mapped.fetch_sub(before, Ordering::Relaxed);
        let mut s = inner.lock();
        s.stats.units_mapped += u64::from(done.added);
        s.stats.units_unmapped += u64::from(done.removed);
        s.stats.units_short += u64::from(done.short);
        s.stats.tile_calls += calls.len() as u64 + u64::from(done.removed > 0);
        done
    }

    /// Unmaps every unit: what is read of the texture after work submitted
    /// before this is zero, until it is mapped again. The units unmapped.
    pub(crate) fn evict(&self, host: &crate::host::Host) -> u32 {
        self.map(host, &vec![false; self.units.len()]).removed
    }

    /// Unmaps units `gone` (mapped), queued on `host`'s queue; their ranges
    /// go back once the host fence passes the next submission.
    fn unmap(&self, host: &crate::host::Host, gone: &[usize], pieces: &mut [Option<Pieces>]) {
        let (coordinates, sizes) = self.regions(gone);
        let tiles: u32 = gone.iter().map(|&i| self.units[i].tiles).sum();
        let flags = [D3D12_TILE_RANGE_FLAG_NULL];
        // SAFETY: a reserved texture of ours; one null range for all the
        // regions' tiles.
        unsafe {
            host.queue().UpdateTileMappings(
                &self.texture,
                gone.len() as u32,
                Some(coordinates.as_ptr()),
                Some(sizes.as_ptr()),
                None::<&ID3D12Heap>,
                1,
                Some(flags.as_ptr()),
                None,
                Some(&raw const tiles),
                D3D12_TILE_MAPPING_FLAG_NONE,
            )
        };
        // Read after the call: the next value the queue signals is queued
        // after the unmapping, and every read of the old tiles before it.
        let after = host.executing() + 1;
        let mut retired = Vec::new();
        for &i in gone {
            if let Some(p) = pieces[i].take() {
                retired.extend(
                    p.ranges
                        .into_iter()
                        .map(|(at, len)| (after, p.heap, at, len)),
                );
            }
        }
        self.inner.lock().retired.extend(retired);
    }

    /// The regions of units `of`: each a whole unit, its tiles in linear
    /// order from its first subresource.
    fn regions(
        &self,
        of: &[usize],
    ) -> (
        Vec<D3D12_TILED_RESOURCE_COORDINATE>,
        Vec<D3D12_TILE_REGION_SIZE>,
    ) {
        of.iter()
            .map(|&i| {
                let u = &self.units[i];
                (
                    D3D12_TILED_RESOURCE_COORDINATE {
                        X: 0,
                        Y: 0,
                        Z: 0,
                        Subresource: u.subresource,
                    },
                    D3D12_TILE_REGION_SIZE {
                        NumTiles: u.tiles,
                        UseBox: false.into(),
                        Width: 0,
                        Height: 0,
                        Depth: 0,
                    },
                )
            })
            .unzip()
    }
}

impl Drop for Tiles {
    fn drop(&mut self) {
        // The texture goes: the lists that named it were reset (they held
        // its host copy), so nothing reads its tiles; its ranges go back now,
        // as a placed texture's do.
        let pieces = std::mem::take(self.pieces.get_mut().unwrap_or_else(|e| e.into_inner()));
        let mapped = self.mapped_of(&pieces);
        for p in pieces.into_iter().flatten() {
            for (at, len) in p.ranges {
                self.inner.give_range(Class::Normal, p.heap, at, len);
            }
        }
        self.inner.mapped.fetch_sub(mapped, Ordering::Relaxed);
        self.inner.reserved.fetch_sub(1, Ordering::Relaxed);
        let key = self.texture.as_raw() as usize;
        let mut map = TILES.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(map) = map.as_mut()
            && map
                .get(&key)
                .is_some_and(|w| std::ptr::eq(w.as_ptr(), std::ptr::from_ref(self)))
        {
            map.remove(&key);
        }
    }
}

/// A texture placed by [`Pool::place`].
pub(crate) struct Placed {
    pub(crate) texture: ID3D12Resource,
    pub(crate) slot: Slot,
    /// Its heap's residency.
    pub(crate) residency: Arc<Residency>,
    /// The bytes it takes.
    pub(crate) bytes: u64,
}

/// DXGI's budget change event and the adapter it is registered with.
struct BudgetEvent {
    adapter: IDXGIAdapter3,
    event: HANDLE,
    cookie: u32,
}

impl BudgetEvent {
    fn register(device: &ID3D12Device) -> Option<BudgetEvent> {
        // SAFETY: plain DXGI calls; the event is closed if it is not kept.
        unsafe {
            let luid = device.GetAdapterLuid();
            let adapter: IDXGIAdapter3 =
                CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
                    .and_then(|f| f.EnumAdapterByLuid(luid))
                    .ok()?;
            let event = CreateEventW(None, false, false, None).ok()?;
            match adapter.RegisterVideoMemoryBudgetChangeNotificationEvent(event) {
                Ok(cookie) => Some(BudgetEvent {
                    adapter,
                    event,
                    cookie,
                }),
                Err(e) => {
                    tracing::info!(error = %e, "no video memory budget notification");
                    let _ = CloseHandle(event);
                    None
                }
            }
        }
    }

    /// Whether the budget changed since the last look.
    fn fired(&self) -> bool {
        // SAFETY: our own event.
        unsafe { WaitForSingleObject(self.event, 0) == WAIT_OBJECT_0 }
    }
}

impl Drop for BudgetEvent {
    fn drop(&mut self) {
        // SAFETY: registered and made by `register`.
        unsafe {
            self.adapter
                .UnregisterVideoMemoryBudgetChangeNotification(self.cookie);
            let _ = CloseHandle(self.event);
        }
    }
}

impl Pool {
    /// A pool on `device`; its thread starts filling the reserve.
    pub(crate) fn new(device: ID3D12Device, config: Config, budget: Budget) -> Pool {
        let small = cfg!(test) || small_placement(&device);
        Pool::with_small(device, config, budget, small)
    }

    fn with_small(device: ID3D12Device, mut config: Config, budget: Budget, small: bool) -> Pool {
        if !small {
            config.heap[Class::Small.index()] = 0;
            config.reserve[Class::Small.index()] = 0;
        }
        let changes = BudgetEvent::register(&device);
        let tier = tiled_tier(&device);
        if config.tiles && tier < D3D12_TILED_RESOURCES_TIER_2.0 {
            tracing::info!(
                tier,
                "texture pool: no reserved textures (tiled resources tier 2 needed): placed"
            );
        }
        let inner = Arc::new(Inner {
            device,
            config,
            budget,
            state: Mutex::new(State {
                classes: [Vec::new(), Vec::new()],
                releasing: Vec::new(),
                retired: std::collections::VecDeque::new(),
                stop: false,
                woken: true,
                next_id: 1,
                over_budget: false,
                backoff_until: None,
                refusals: 0,
                stats: Stats::default(),
            }),
            wake: Condvar::new(),
            changes,
            zeroed: AtomicBool::new(false),
            small,
            tier,
            tileable: Mutex::new(HashMap::new()),
            reserved: AtomicU64::new(0),
            mapped: AtomicU64::new(0),
        });
        let thread = {
            let inner = inner.clone();
            std::thread::Builder::new()
                .name("xtp-texture-heaps".into())
                .spawn(move || inner.run())
        };
        let thread = match thread {
            Ok(thread) => Some(thread),
            Err(e) => {
                tracing::warn!(error = %e, "no texture heap thread: heaps made when textures need them");
                None
            }
        };
        Pool { inner, thread }
    }

    /// `d` (a host texture description, no target) placed in the pool in
    /// `state`; `Err` says why not (then it is committed).
    pub(crate) fn place(
        &self,
        d: &D3D12_RESOURCE_DESC,
        state: D3D12_RESOURCE_STATES,
    ) -> Result<Placed, &'static str> {
        let inner = &self.inner;
        let refused = |why| {
            inner.lock().stats.not_placed += 1;
            Err(why)
        };
        let Some((class, desc, len)) = inner.class_of(d) else {
            return refused("no 4 or 64 KiB placement");
        };
        if len > inner.config.largest {
            return refused("too large for the pool");
        }
        if class == Class::Normal && len >= TILED_FROM * TILE && inner.reservable(d) {
            match inner.reserve(d, state) {
                Ok(reserved) => return Ok(reserved),
                Err(why) => {
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    ONCE.call_once(|| {
                        tracing::warn!(why, ?d, "a reserved host texture refused: placed instead")
                    });
                }
            }
        }
        let taken = match inner.take(class, len) {
            Some(taken) => taken,
            None => {
                // The reserve is gone: a heap now, not waited for.
                let size = inner.config.heap[class.index()].max(len.next_multiple_of(KIB_64));
                if !inner.add_heap(class, size, true) {
                    return refused("no heap (budget or driver)");
                }
                match inner.take(class, len) {
                    Some(taken) => taken,
                    None => return refused("no room in a new heap"),
                }
            }
        };
        let (heap, id, offset, residency, made) = taken;
        let slot = Slot {
            inner: inner.clone(),
            kind: Kind::Range {
                class,
                heap: id,
                offset,
                len,
            },
        };
        let mut texture: Option<ID3D12Resource> = None;
        // SAFETY: a live heap, a range of it no other live resource uses,
        // aligned for the description's placement; the out pointer is ours.
        let placed = unsafe {
            inner
                .device
                .CreatePlacedResource(&heap, offset, &desc, state, None, &mut texture)
        };
        match (placed, texture) {
            (Ok(()), Some(texture)) => {
                // A heap just made is usually resident within a few
                // milliseconds: its first textures wait that long, so the
                // title can use them at once. Past VidMm's ceiling the rest
                // are handed out not resident, without waiting.
                if !residency.is_ready() {
                    let left = (made + HEAP_WAIT).saturating_duration_since(Instant::now());
                    if left.is_zero() || !residency.wait(left.min(PLACE_WAIT)) {
                        inner.lock().stats.early += 1;
                    }
                }
                Ok(Placed {
                    texture,
                    slot,
                    residency,
                    bytes: len,
                })
            }
            (result, _) => {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::warn!(
                        ?result,
                        ?desc,
                        "a placed host texture refused: committed instead"
                    )
                });
                drop(slot);
                refused("placement refused")
            }
        }
    }

    /// What it did since it was made.
    pub(crate) fn stats(&self) -> Stats {
        self.inner.lock().stats
    }

    /// What it holds now.
    pub(crate) fn holding(&self) -> Holding {
        self.inner.holding()
    }

    /// The frame log's part: what the pool holds and did.
    pub(crate) fn report(&self) -> String {
        let (h, s) = (self.holding(), self.stats());
        format!(
            "texture pool: {} heaps ({} not resident yet), {} MiB, {} MiB placed in {} ranges \
             ({} reserved textures, {} MiB of their tiles mapped, {} MiB given back waiting for \
             the fence); so far {} heaps made ({} by the title's thread), {} released, {} textures \
             not placed, {} handed out before their heap was resident, {} heaps held back by the \
             budget, {} textures reserved, tile units {} mapped, {} unmapped, {} left for want of \
             room, {} tile mapping calls",
            h.heaps,
            h.pending,
            h.bytes / MIB,
            h.used / MIB,
            h.textures,
            h.reserved,
            h.mapped / MIB,
            h.retired / MIB,
            s.heaps,
            s.emergency,
            s.released,
            s.not_placed,
            s.early,
            s.over_budget,
            s.reserved,
            s.units_mapped,
            s.units_unmapped,
            s.units_short,
            s.tile_calls
        )
    }

    /// Wakes the thread and waits until it has nothing left to make (at
    /// most `timeout`): tests.
    #[cfg(test)]
    pub(crate) fn settle(&self, timeout: Duration) -> bool {
        let until = Instant::now() + timeout;
        while Instant::now() < until {
            let short = {
                let s = self.inner.lock();
                Class::ALL.iter().any(|&c| self.inner.short(&s, c))
            };
            if !short {
                return true;
            }
            self.inner.notify();
            std::thread::sleep(Duration::from_millis(5));
        }
        false
    }

    /// Waits until every heap is resident (at most `timeout`): tests.
    #[cfg(test)]
    pub(crate) fn resident(&self, timeout: Duration) -> bool {
        let until = Instant::now() + timeout;
        while Instant::now() < until {
            if self.holding().pending == 0 {
                return true;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        false
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.inner.lock().stop = true;
        self.inner.wake.notify_all();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Inner {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[cfg(test)]
    fn notify(&self) {
        self.lock().woken = true;
        self.wake.notify_all();
    }

    /// The class of `d`, its description as placed and the bytes it takes
    /// (a multiple of the class's alignment); `None` for an alignment the
    /// pool has no class for.
    fn class_of(&self, d: &D3D12_RESOURCE_DESC) -> Option<(Class, D3D12_RESOURCE_DESC, u64)> {
        let size = |desc: &D3D12_RESOURCE_DESC| {
            // SAFETY: a host description.
            let info = unsafe {
                self.device
                    .GetResourceAllocationInfo(0, std::slice::from_ref(desc))
            };
            (info.SizeInBytes != u64::MAX && info.SizeInBytes > 0).then_some(info)
        };
        let mut desc = *d;
        let small = Class::Small.alignment();
        if self.small && may_be_small(d) {
            desc.Alignment = small;
            if let Some(info) = size(&desc)
                && info.Alignment == small
            {
                return Some((Class::Small, desc, info.SizeInBytes.next_multiple_of(small)));
            }
        }
        desc.Alignment = 0;
        let info = size(&desc)?;
        let normal = Class::Normal.alignment();
        (info.Alignment <= normal).then(|| {
            (
                Class::Normal,
                desc,
                info.SizeInBytes.next_multiple_of(normal),
            )
        })
    }

    /// Free bytes of `class`.
    fn free(s: &State, class: Class) -> u64 {
        s.classes[class.index()]
            .iter()
            .map(|h| h.ranges.free())
            .sum()
    }

    /// Whether `class` has less than its reserve free.
    fn short(&self, s: &State, class: Class) -> bool {
        (self.small || class != Class::Small)
            && Self::free(s, class) < self.config.reserve[class.index()]
    }

    /// `len` bytes of a heap of `class`: the tightest fit in a resident
    /// heap, else in one still becoming resident. The thread is woken if
    /// the reserve runs short.
    fn take(
        &self,
        class: Class,
        len: u64,
    ) -> Option<(ID3D12Heap, u64, u64, Arc<Residency>, Instant)> {
        let mut s = self.lock();
        let heaps = &mut s.classes[class.index()];
        let (_, _, at) = heaps
            .iter()
            .enumerate()
            .filter_map(|(i, h)| {
                h.ranges
                    .fit(len)
                    .map(|fit| (!h.residency.is_ready(), fit, i))
            })
            .min()?;
        let h = &mut heaps[at];
        let offset = h.ranges.take(len)?;
        h.textures += 1;
        let taken = (h.heap.clone(), h.id, offset, h.residency.clone(), h.made);
        s.stats.placed += 1;
        if self.short(&s, class) && !backoff_active(s.backoff_until, Instant::now()) {
            s.woken = true;
            self.wake.notify_all();
        }
        Some(taken)
    }

    /// Gives a range back; its heap goes if it is empty and the reserve
    /// does not need it.
    fn give_range(&self, class: Class, heap: u64, offset: u64, len: u64) {
        let mut s = self.lock();
        self.give_locked(&mut s, class, heap, offset, len);
    }

    /// [`Inner::give_range`] under the state's lock.
    fn give_locked(&self, s: &mut State, class: Class, heap: u64, offset: u64, len: u64) {
        let reserve = self.config.reserve[class.index()];
        let heaps = &mut s.classes[class.index()];
        let Some(at) = heaps.iter().position(|h| h.id == heap) else {
            return;
        };
        heaps[at].ranges.give(offset, len);
        heaps[at].textures -= 1;
        if heaps[at].textures > 0 {
            return;
        }
        let others: u64 = heaps
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != at)
            .map(|(_, h)| h.ranges.free())
            .sum();
        if others >= reserve {
            let heap = heaps.swap_remove(at);
            s.releasing.push(heap.heap);
            s.stats.released += 1;
            s.woken = true;
            self.wake.notify_all();
        }
    }

    /// Gives back the ranges reserved textures retired before the host
    /// fence reached `completed`, oldest first (one retired a little out of
    /// order waits for those before it).
    fn reclaim(&self, completed: u64) {
        let mut s = self.lock();
        while let Some(&(after, heap, at, len)) = s.retired.front()
            && after <= completed
        {
            s.retired.pop_front();
            self.give_locked(&mut s, Class::Normal, heap, at, len);
        }
    }

    /// Whether host texture `d` may be reserved: the pool reserves, the
    /// device has tier 2 (3 for 3D textures) and tiles its format.
    fn reservable(&self, d: &D3D12_RESOURCE_DESC) -> bool {
        let tier = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
            D3D12_TILED_RESOURCES_TIER_3.0
        } else {
            D3D12_TILED_RESOURCES_TIER_2.0
        };
        if !self.config.tiles || self.tier < tier || d.SampleDesc.Count > 1 {
            return false;
        }
        let mut known = self.tileable.lock().unwrap_or_else(|e| e.into_inner());
        *known.entry(d.Format.0).or_insert_with(|| {
            let mut data = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
                Format: d.Format,
                ..Default::default()
            };
            // SAFETY: the query writes the structure of the given size.
            let asked = unsafe {
                self.device.CheckFeatureSupport(
                    D3D12_FEATURE_FORMAT_SUPPORT,
                    (&raw mut data).cast(),
                    size_of_val(&data) as u32,
                )
            };
            asked.is_ok() && data.Support2.contains(D3D12_FORMAT_SUPPORT2_TILED)
        })
    }

    /// `d` made a reserved texture in `state`, nothing of it mapped.
    fn reserve(
        self: &Arc<Self>,
        d: &D3D12_RESOURCE_DESC,
        state: D3D12_RESOURCE_STATES,
    ) -> Result<Placed, &'static str> {
        let desc = D3D12_RESOURCE_DESC {
            Alignment: 0,
            Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE,
            ..*d
        };
        let mut texture: Option<ID3D12Resource> = None;
        // SAFETY: a texture description in the reserved layout; the out
        // pointer is ours.
        unsafe {
            self.device
                .CreateReservedResource(&desc, state, None, &mut texture)
        }
        .map_err(|_| "the driver refused it")?;
        let texture = texture.ok_or("no texture")?;
        let units = units_of(&self.device, &texture, &desc).ok_or("no tiling")?;
        let tiles: u64 = units.iter().map(|u| u64::from(u.tiles)).sum();
        let reserved = Arc::new(Tiles {
            inner: self.clone(),
            texture: texture.clone(),
            pieces: Mutex::new(units.iter().map(|_| None).collect()),
            units,
        });
        TILES
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_or_insert_with(HashMap::new)
            .insert(texture.as_raw() as usize, Arc::downgrade(&reserved));
        self.reserved.fetch_add(1, Ordering::Relaxed);
        self.lock().stats.reserved += 1;
        Ok(Placed {
            texture,
            slot: Slot {
                inner: self.clone(),
                kind: Kind::Tiles(reserved),
            },
            // Nothing of its own to make resident: its tiles are mapped
            // into resident heaps only.
            residency: Arc::new(Residency::resident()),
            bytes: tiles * TILE,
        })
    }

    /// `len` bytes (whole tiles) of a resident heap of the 64 KiB class for
    /// a unit of a reserved texture: a range of the tightest fit, else
    /// pieces of the heap with the most free bytes (in the order taken);
    /// with the heap and its id. The thread is woken if the reserve runs
    /// short, or if none has room.
    fn take_tiles(&self, len: u64) -> Option<Taken> {
        let mut s = self.lock();
        let class = Class::Normal;
        let heaps = &mut s.classes[class.index()];
        let resident = |h: &Heap| h.residency.is_ready();
        let whole = heaps
            .iter()
            .enumerate()
            .filter(|(_, h)| resident(h))
            .filter_map(|(i, h)| h.ranges.fit(len).map(|fit| (fit, i)))
            .min();
        let taken = match whole {
            Some((_, i)) => {
                let h = &mut heaps[i];
                h.ranges.take(len).map(|at| (i, vec![(at, len)]))
            }
            None => heaps
                .iter()
                .enumerate()
                .filter(|(_, h)| resident(h) && h.ranges.free() >= len)
                .max_by_key(|(_, h)| h.ranges.free())
                .map(|(i, _)| i)
                .and_then(|i| scattered(&mut heaps[i].ranges, len).map(|r| (i, r))),
        };
        let wake = (taken.is_none() || self.short(&s, class))
            && !backoff_active(s.backoff_until, Instant::now());
        let found = taken.map(|(i, ranges)| {
            let h = &mut s.classes[class.index()][i];
            h.textures += ranges.len() as u32;
            (h.heap.clone(), h.id, ranges)
        });
        if wake {
            s.woken = true;
            self.wake.notify_all();
        }
        found
    }

    /// Makes a heap of `size` bytes for `class`, not resident, and asks for
    /// it to become resident (not waited for). False, logged, past the
    /// budget or if the driver refuses (backing off for [`HEAP_FAIL_BACKOFF`]
    /// on driver refusal while existing heaps keep serving placements).
    fn add_heap(&self, class: Class, size: u64, emergency: bool) -> bool {
        let pending: u64 = {
            let s = self.lock();
            if backoff_active(s.backoff_until, Instant::now()) {
                return false;
            }
            s.classes
                .iter()
                .flatten()
                .filter(|h| !h.residency.is_ready())
                .map(|h| h.size)
                .sum()
        };
        if let Some((budget, usage)) = (self.budget)(&self.device)
            && usage.saturating_add(pending).saturating_add(size) > budget
        {
            let mut s = self.lock();
            s.stats.over_budget += 1;
            if !std::mem::replace(&mut s.over_budget, true) {
                tracing::warn!(
                    class = class.name(),
                    mib = size / MIB,
                    budget_mib = budget / MIB,
                    usage_mib = usage / MIB,
                    pending_mib = pending / MIB,
                    "no texture heap: past the video memory budget (textures made committed meanwhile, within the budget)"
                );
            }
            return false;
        }
        let started = Instant::now();
        let heap = match self.make_heap(size) {
            Ok(heap) => heap,
            Err(e) => {
                let mut s = self.lock();
                s.refusals += 1;
                s.backoff_until = Some(Instant::now() + HEAP_FAIL_BACKOFF);
                let n = s.refusals;
                drop(s);
                if n <= 4 || n.is_power_of_two() {
                    tracing::warn!(
                        n,
                        error = %e,
                        mib = size / MIB,
                        backoff_ms = HEAP_FAIL_BACKOFF.as_millis() as u64,
                        "no texture heap: the driver refused it; backing off heap creation"
                    );
                }
                return false;
            }
        };
        let pageable: Option<ID3D12Pageable> = heap.cast().ok();
        let residency = Arc::new(Residency::ask(&self.device, vec![pageable]));
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        let mut s = self.lock();
        s.backoff_until = None;
        if std::mem::replace(&mut s.over_budget, false) {
            tracing::info!("texture heaps within the video memory budget again");
        }
        let id = s.next_id;
        s.next_id += 1;
        s.stats.heaps += 1;
        s.stats.asked += 1;
        if emergency {
            s.stats.emergency += 1;
        }
        let n = s.stats.heaps;
        s.classes[class.index()].push(Heap {
            id,
            heap,
            size,
            ranges: Ranges::new(size),
            textures: 0,
            residency,
            made: Instant::now(),
        });
        drop(s);
        if emergency || ms > 100.0 || n <= 4 || n.is_power_of_two() {
            tracing::info!(
                n,
                class = class.name(),
                mib = size / MIB,
                ms,
                emergency,
                "texture heap made (not resident yet)"
            );
        }
        true
    }

    /// A default heap of `size` bytes for textures that are no targets,
    /// not resident and (if the runtime can) not zeroed.
    fn make_heap(&self, size: u64) -> windows::core::Result<ID3D12Heap> {
        let flags = |zeroed: bool| {
            let base =
                D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES | D3D12_HEAP_FLAG_CREATE_NOT_RESIDENT;
            if zeroed {
                base
            } else {
                base | D3D12_HEAP_FLAG_CREATE_NOT_ZEROED
            }
        };
        let desc = |zeroed| D3D12_HEAP_DESC {
            SizeInBytes: size,
            Properties: D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            },
            Alignment: 0,
            Flags: flags(zeroed),
        };
        let make = |zeroed| {
            let mut heap: Option<ID3D12Heap> = None;
            // SAFETY: a plain heap description; the out pointer is ours.
            unsafe { self.device.CreateHeap(&desc(zeroed), &mut heap) }.map(|()| heap)
        };
        let zeroed = self.zeroed.load(Ordering::Relaxed);
        let made = match make(zeroed) {
            Err(e) if !zeroed => {
                tracing::info!(error = %e, "texture heaps refuse CREATE_NOT_ZEROED: made zeroed");
                self.zeroed.store(true, Ordering::Relaxed);
                make(true)
            }
            made => made,
        };
        match made {
            Ok(Some(heap)) => Ok(heap),
            Ok(None) => Err(windows::core::Error::empty()),
            Err(e) => Err(e),
        }
    }

    fn holding(&self) -> Holding {
        let s = self.lock();
        let mut h = Holding::default();
        for heap in s.classes.iter().flatten() {
            h.heaps += 1;
            h.pending += u64::from(!heap.residency.is_ready());
            h.bytes += heap.size;
            h.used += heap.size - heap.ranges.free();
            h.textures += u64::from(heap.textures);
        }
        h.retired = s.retired.iter().map(|&(.., len)| len).sum();
        h.reserved = self.reserved.load(Ordering::Relaxed);
        h.mapped = self.mapped.load(Ordering::Relaxed);
        h
    }

    /// The thread: heaps released, retired ranges given back, the budget
    /// followed, refused heaps asked again, the reserve kept; then sleep
    /// until woken (or a second).
    fn run(&self) {
        loop {
            let releasing = {
                let mut s = self.lock();
                if s.stop {
                    return;
                }
                s.woken = false;
                std::mem::take(&mut s.releasing)
            };
            drop(releasing);
            if let Some(host) = crate::host::host() {
                self.reclaim(host.completed());
            }
            if self.changes.as_ref().is_some_and(BudgetEvent::fired) {
                self.budget_changed();
            }
            // Looking at a refused heap asks for it again when due.
            let _ = self.holding();
            for class in Class::ALL {
                while self.short(&self.lock(), class) {
                    if !self.add_heap(class, self.config.heap[class.index()], false) {
                        break;
                    }
                }
            }
            let s = self.lock();
            if s.stop {
                return;
            }
            let wait = backoff_wait(s.backoff_until, Instant::now(), Duration::from_secs(1));
            let _ = self
                .wake
                .wait_timeout_while(s, wait, |s| !s.stop && !s.woken);
        }
    }

    /// The budget changed: logged, and past it every empty heap goes.
    fn budget_changed(&self) {
        let Some((budget, usage)) = (self.budget)(&self.device) else {
            return;
        };
        tracing::info!(
            budget_mib = budget / MIB,
            usage_mib = usage / MIB,
            "video memory budget changed"
        );
        if usage <= budget {
            return;
        }
        let mut s = self.lock();
        let mut gone = Vec::new();
        for heaps in &mut s.classes {
            let (empty, kept): (Vec<Heap>, Vec<Heap>) = std::mem::take(heaps)
                .into_iter()
                .partition(|h| h.textures == 0);
            *heaps = kept;
            gone.extend(empty.into_iter().map(|h| h.heap));
        }
        s.stats.released += gone.len() as u64;
        drop(s);
        if !gone.is_empty() {
            tracing::info!(
                heaps = gone.len(),
                "empty texture heaps released: past the budget"
            );
        }
    }
}

/// The device's tiled resources tier (0 if it cannot tell).
fn tiled_tier(device: &ID3D12Device) -> i32 {
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    // SAFETY: the query writes the structure of the given size.
    let asked = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS,
            (&raw mut options).cast(),
            size_of_val(&options) as u32,
        )
    };
    if asked.is_ok() {
        options.TiledResourcesTier.0
    } else {
        0
    }
}

/// The units of reserved texture `texture` (made from `desc`): each
/// standard mip of each array slice (3D: of the texture), then each
/// slice's packed mips, in subresource order.
fn units_of(
    device: &ID3D12Device,
    texture: &ID3D12Resource,
    desc: &D3D12_RESOURCE_DESC,
) -> Option<Vec<Unit>> {
    let mips = u32::from(desc.MipLevels.max(1));
    let slices = if desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        1
    } else {
        u32::from(desc.DepthOrArraySize.max(1))
    };
    let mut count = mips * slices;
    let mut tilings = vec![D3D12_SUBRESOURCE_TILING::default(); count as usize];
    let mut packed = D3D12_PACKED_MIP_INFO::default();
    let mut total = 0u32;
    // SAFETY: room for `count` subresource tilings; plain outputs.
    unsafe {
        device.GetResourceTiling(
            texture,
            Some(&raw mut total),
            Some(&raw mut packed),
            None,
            Some(&raw mut count),
            0,
            tilings.as_mut_ptr(),
        )
    };
    let standard = u32::from(packed.NumStandardMips).min(mips);
    let packed_mips = u32::from(packed.NumPackedMips);
    let mut units = Vec::with_capacity((slices * (standard + 1)) as usize);
    for slice in 0..slices {
        for mip in 0..standard {
            let subresource = slice * mips + mip;
            let t = tilings.get(subresource as usize)?;
            if t.StartTileIndexInOverallResource == D3D12_PACKED_TILE {
                return None;
            }
            let tiles = t.WidthInTiles * u32::from(t.HeightInTiles) * u32::from(t.DepthInTiles);
            units.push(Unit {
                subresource,
                slice,
                first_mip: mip,
                mips: 1,
                tiles,
            });
        }
        if packed_mips > 0 && packed.NumTilesForPackedMips > 0 {
            units.push(Unit {
                subresource: slice * mips + standard,
                slice,
                first_mip: standard,
                mips: packed_mips,
                tiles: packed.NumTilesForPackedMips,
            });
        }
    }
    let sum: u32 = units.iter().map(|u| u.tiles).sum();
    (sum == total && sum > 0).then_some(units)
}

/// `len` bytes (whole tiles) of `ranges` in pieces: the largest free range
/// up to what is left each time, [`PIECES`] at most; `None` (nothing taken)
/// if they do not hold them so.
fn scattered(ranges: &mut Ranges, len: u64) -> Option<Vec<(u64, u64)>> {
    let mut taken = Vec::new();
    let mut left = len;
    while left > 0 && taken.len() < PIECES {
        // The longest piece up to `left` a free range holds: by halves
        // over whole tiles.
        let (mut low, mut high) = (0, left / TILE);
        while low < high {
            let mid = (low + high).div_ceil(2);
            if ranges.fit(mid * TILE).is_some() {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        let Some(at) = (low > 0).then(|| ranges.take(low * TILE)).flatten() else {
            break;
        };
        taken.push((at, low * TILE));
        left -= low * TILE;
    }
    if left > 0 {
        for (at, len) in taken {
            ranges.give(at, len);
        }
        return None;
    }
    Some(taken)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use windows::Win32::Graphics::Dxgi::Common::*;

    thread_local! {
        /// Host textures this thread makes go to the tests' pool
        /// ([`Pooled`]).
        pub(crate) static POOLED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    /// While it lives, the host textures this thread makes that the pool
    /// takes go to the tests' own pool ([`pool_for_tests`]), reserved where
    /// the device can: other tests' textures stay committed.
    pub(crate) struct Pooled;

    impl Pooled {
        pub(crate) fn new() -> Pooled {
            POOLED.set(true);
            Pooled
        }
    }

    impl Drop for Pooled {
        fn drop(&mut self) {
            POOLED.set(false);
        }
    }

    /// The tests' pool: 4 MiB heaps, reserved textures, no budget.
    pub(crate) fn pool_for_tests() -> Option<&'static Pool> {
        static TEST_POOL: OnceLock<Option<Pool>> = OnceLock::new();
        TEST_POOL
            .get_or_init(|| {
                let host = crate::host::host()?;
                let config = Config {
                    tiles: true,
                    ..Config::sized(4 * MIB, 4 * MIB)
                };
                Some(Pool::new(host.device.clone(), config, unlimited()))
            })
            .as_ref()
    }

    /// Whether the test device reserves textures (tiled resources tier 2).
    pub(crate) fn reserves(pool: &Pool) -> bool {
        pool.inner.tier >= D3D12_TILED_RESOURCES_TIER_2.0
    }

    fn texture(width: u64, height: u32, format: DXGI_FORMAT, mips: u16) -> D3D12_RESOURCE_DESC {
        D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: width,
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: mips,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..Default::default()
        }
    }

    #[test]
    fn the_small_rule_counts_standard_tiles() {
        // 64x64 RGBA8 is 4 tiles of 32x32; 128x128 is 16; 256x128 is 32.
        assert!(may_be_small(&texture(
            64,
            64,
            DXGI_FORMAT_R8G8B8A8_UNORM,
            1
        )));
        assert!(may_be_small(&texture(
            128,
            128,
            DXGI_FORMAT_R8G8B8A8_UNORM,
            1
        )));
        assert!(!may_be_small(&texture(
            256,
            128,
            DXGI_FORMAT_R8G8B8A8_UNORM,
            1
        )));
        // BC7: 16-byte blocks, 16x16 blocks a tile: 256x256 texels is one.
        assert!(may_be_small(&texture(256, 256, DXGI_FORMAT_BC7_UNORM, 9)));
        assert!(!may_be_small(&texture(
            1024,
            1024,
            DXGI_FORMAT_BC7_UNORM,
            9
        )));
        // Targets, samples and arrays never.
        let mut rt = texture(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, 1);
        rt.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        assert!(!may_be_small(&rt));
        let mut array = texture(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, 1);
        array.DepthOrArraySize = 6;
        assert!(!may_be_small(&array));
        // Formats without a known element size: no.
        assert!(!may_be_small(&texture(16, 16, DXGI_FORMAT_NV12, 1)));
    }

    #[test]
    fn the_pool_is_asked_for_by_its_variable() {
        for low in [false, true] {
            assert!(asks_pool(Some("1"), low));
            assert!(!asks_pool(Some("0"), low));
            assert!(!asks_pool(Some(""), low));
        }
        assert!(!asks_pool(None, false), "off by default");
        assert!(asks_pool(None, true), "on by default below 32 GiB of RAM");
    }

    #[test]
    fn sizes_follow_the_heap_size() {
        let c = Config::sized(256 * MIB, 512 * MIB);
        assert_eq!(c.heap, [32 * MIB, 256 * MIB]);
        assert_eq!(c.reserve, [64 * MIB, 512 * MIB]);
        assert_eq!(c.largest, 64 * MIB);
    }

    /// A pool on the test device with small heaps (1 MiB a 64 KiB heap).
    fn pool(test: &str, budget: Budget) -> Option<(crate::testgpu::Gpu, Pool)> {
        let gpu = crate::testgpu::device(test)?;
        let pool = Pool::new(gpu.device.clone(), Config::sized(MIB, MIB), budget);
        Some((gpu, pool))
    }

    fn unlimited() -> Budget {
        Box::new(|_| Some((u64::MAX / 4, 0)))
    }

    /// Textures of both classes are placed at their alignment, reuse what
    /// is given back, and cost no residency request of their own: one a
    /// heap. Empty heaps beyond the reserve go.
    #[test]
    fn textures_are_placed_by_class_and_ranges_reused() {
        let test = "resource::pool::tests::textures_are_placed_by_class_and_ranges_reused";
        let Some((_gpu, pool)) = pool(test, unlimited()) else {
            return;
        };
        assert!(
            pool.settle(Duration::from_secs(5)),
            "the thread fills the reserve"
        );
        let before = pool.stats();
        assert_eq!(before.emergency, 0);
        let state = D3D12_RESOURCE_STATE_COPY_DEST;
        let small = texture(64, 64, DXGI_FORMAT_R8G8B8A8_UNORM, 1);
        let normal = texture(256, 256, DXGI_FORMAT_R8G8B8A8_UNORM, 1);
        let mut held = Vec::new();
        for _ in 0..24 {
            for d in [&small, &normal] {
                let p = pool.place(d, state).expect("placed");
                let (class, offset, len) = p.slot.place();
                assert_eq!(offset % class.alignment(), 0);
                assert_eq!(len % class.alignment(), 0);
                assert_eq!(
                    class,
                    if std::ptr::eq(d, &small) {
                        Class::Small
                    } else {
                        Class::Normal
                    }
                );
                held.push(p);
            }
        }
        let made = pool.stats();
        assert_eq!(made.placed - before.placed, 48);
        assert_eq!(
            made.asked, made.heaps,
            "one residency request a heap, none a texture"
        );
        assert!(made.heaps < 48, "{made:?}");
        // A range given back is taken again.
        let gone = held.remove(1);
        let (class, offset, _) = gone.slot.place();
        drop(gone);
        let again = pool.place(&normal, state).unwrap();
        assert_eq!(again.slot.place().0, class);
        assert_eq!(again.slot.place().1, offset, "the same place");
        held.push(again);
        // Everything back: the heaps the reserve does not need go.
        let holding = pool.holding();
        assert_eq!(holding.textures, 48);
        drop(held);
        let after = pool.holding();
        assert_eq!((after.textures, after.used), (0, 0));
        assert!(after.heaps < holding.heaps, "{after:?} after {holding:?}");
        assert!(pool.stats().released > 0);
        // The reserve is kept (1 MiB for 64 KiB textures, 128 KiB small).
        assert!(after.bytes >= MIB + MIB / 8, "{after:?}");
    }

    /// The heap of a placed texture becomes resident; the texture then
    /// holds what a copy writes into it.
    #[test]
    fn a_placed_texture_holds_what_is_copied_into_it() {
        let test = "resource::pool::tests::a_placed_texture_holds_what_is_copied_into_it";
        let Some((gpu, pool)) = pool(test, unlimited()) else {
            return;
        };
        let host = gpu.host();
        let d = texture(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, 1);
        let placed = pool.place(&d, D3D12_RESOURCE_STATE_COPY_DEST).unwrap();
        assert!(placed.residency.wait(Duration::from_secs(5)));
        let rows = 16 * 256u64;
        let (upload, data) = host.upload_buffer(rows).unwrap();
        for i in 0..rows as usize {
            // SAFETY: inside the mapped buffer.
            unsafe { data.add(i).write((i % 251) as u8) };
        }
        let list = host.new_list().unwrap();
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(&placed.texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&upload) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                        Width: 16,
                        Height: 16,
                        Depth: 1,
                        RowPitch: 256,
                    },
                },
            },
        };
        // SAFETY: an open list; both locations name live resources.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
        list.close();
        host.execute(&[&list]);
        host.drain();
        const COPY_DEST: u32 = 0x400;
        let read = super::super::tests_support::read_back(host, &placed.texture, COPY_DEST);
        for row in 0..16usize {
            let at = row * 256;
            let want: Vec<u8> = (at..at + 64).map(|i| (i % 251) as u8).collect();
            assert_eq!(read[at..at + 64], want[..], "row {row}");
        }
    }

    /// Past the budget no heap is made and the texture is not placed (it is
    /// made committed, within the budget); nothing is asked of VidMm.
    #[test]
    fn past_the_budget_no_heap_is_made() {
        let test = "resource::pool::tests::past_the_budget_no_heap_is_made";
        let Some((_gpu, pool)) = pool(test, Box::new(|_| Some((MIB, MIB)))) else {
            return;
        };
        let d = texture(256, 256, DXGI_FORMAT_R8G8B8A8_UNORM, 1);
        assert!(pool.place(&d, D3D12_RESOURCE_STATE_COPY_DEST).is_err());
        let s = pool.stats();
        assert_eq!((s.heaps, s.asked, s.placed), (0, 0, 0));
        assert!(s.over_budget > 0 && s.not_placed == 1, "{s:?}");
        assert!(pool.report().contains("held back by the budget"));
    }

    /// The process's private bytes (WARP's video memory is the process's
    /// own).
    fn private_bytes() -> u64 {
        use windows_sys::Win32::System::ProcessStatus::{
            K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: plain data.
        let mut counters: PROCESS_MEMORY_COUNTERS_EX = unsafe { std::mem::zeroed() };
        counters.cb = size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32;
        // SAFETY: the structure and its size are ours.
        let ok = unsafe {
            K32GetProcessMemoryInfo(GetCurrentProcess(), (&raw mut counters).cast(), counters.cb)
        };
        if ok == 0 {
            0
        } else {
            counters.PrivateUsage as u64
        }
    }

    /// One path of the benchmark: `n` textures made by `make` (which says
    /// how many residency requests it made), each creation timed, then
    /// `ready` waited for; video memory and private bytes before and after.
    fn bench_path<T>(
        name: &str,
        host: &crate::host::Host,
        n: usize,
        mut make: impl FnMut() -> T,
        ready: impl Fn(&[T]) -> bool,
        asked: impl Fn(&[T]) -> u64,
    ) -> String {
        let memory = || crate::dxgi::video_memory(&host.device).map_or(0, |(_, usage)| usage);
        let (vram0, private0) = (memory(), private_bytes());
        let started = Instant::now();
        let mut times = Vec::with_capacity(n);
        let mut made = Vec::with_capacity(n);
        for _ in 0..n {
            let t = Instant::now();
            made.push(make());
            times.push(t.elapsed());
        }
        let creating = started.elapsed();
        while !ready(&made) {
            assert!(
                started.elapsed() < Duration::from_secs(600),
                "{name}: never resident"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        let all = started.elapsed();
        let (vram1, private1) = (memory(), private_bytes());
        let requests = asked(&made);
        drop(made);
        times.sort();
        let us = |d: Duration| d.as_secs_f64() * 1e6;
        let mean = us(creating) / n as f64;
        let at = |q: f64| us(times[((n as f64 * q) as usize).min(n - 1)]);
        let per = |a: u64, b: u64| b.saturating_sub(a) as f64 / n as f64 / 1024.0;
        format!(
            "| {name} | {n} | {mean:.1} | {:.1} | {:.1} | {:.1} | {:.0} | {:.0} | {:.0} | {requests} |",
            at(0.5),
            at(0.99),
            us(*times.last().unwrap()),
            all.as_secs_f64() * 1000.0,
            per(vram0, vram1),
            per(private0, private1),
        )
    }

    /// The micro-benchmark of `docs/analysis/vram.md`: `XTP_POOL_BENCH`
    /// (5,000) BC7 1024² textures of 9 mips, made as committed resources
    /// resident at creation (before 2026-11-24), committed not resident and
    /// made resident one by one (dark-world's path, each waited for), and
    /// placed in the pool (256 MiB heaps). Run on its own:
    /// `cargo test -p xtp-gpu --release -- --ignored --nocapture pool::tests::bench`.
    #[test]
    #[ignore = "a benchmark: several GB of video memory"]
    fn bench_5000_bc7_textures() {
        let test = "resource::pool::tests::bench_5000_bc7_textures";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let n: usize = std::env::var("XTP_POOL_BENCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5000);
        let d = texture(1024, 1024, DXGI_FORMAT_BC7_UNORM, 9);
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let committed = |flags: D3D12_HEAP_FLAGS| {
            let mut t: Option<ID3D12Resource> = None;
            // SAFETY: plain creation.
            unsafe {
                host.device
                    .CreateCommittedResource(
                        &heap,
                        flags,
                        &d,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        None,
                        &mut t,
                    )
                    .unwrap();
            }
            t.unwrap()
        };
        // SAFETY: a host description.
        let info = unsafe { host.device.GetResourceAllocationInfo(0, &[d]) };
        let mut rows = vec![
            format!(
                "WARP: {} ({} KiB a texture as D3D12 sizes it, {} alignment)",
                n,
                info.SizeInBytes / 1024,
                info.Alignment
            ),
            "| path | textures | µs/texture (mean) | p50 µs | p99 µs | max µs | ms until all resident | VRAM KiB/texture | private KiB/texture | residency requests |".into(),
            "|---|---|---|---|---|---|---|---|---|---|".into(),
        ];
        rows.push(bench_path(
            "committed, resident at creation",
            host,
            n,
            || committed(D3D12_HEAP_FLAG_NONE),
            |_| true,
            |made| made.len() as u64,
        ));
        rows.push(bench_path(
            "committed, not resident + EnqueueMakeResident each",
            host,
            n,
            || {
                let t = committed(D3D12_HEAP_FLAG_CREATE_NOT_RESIDENT);
                let r = Residency::ask(&host.device, vec![t.cast().ok()]);
                assert!(r.wait(Duration::from_secs(5)));
                (t, r)
            },
            |made| made.iter().all(|(_, r)| r.is_ready()),
            |made| made.len() as u64,
        ));
        let pool = Pool::new(
            host.device.clone(),
            Config::sized(256 * MIB, 512 * MIB),
            Box::new(crate::dxgi::video_memory),
        );
        assert!(pool.settle(Duration::from_secs(30)));
        let before = pool.stats();
        let at_start = pool.holding().bytes;
        rows.push(bench_path(
            "pooled (256 MiB heaps, 512 MiB reserve)",
            host,
            n,
            || pool.place(&d, D3D12_RESOURCE_STATE_COPY_DEST).unwrap(),
            |made| made.iter().all(|p| p.residency.is_ready()),
            |_| pool.stats().asked - before.asked,
        ));
        let s = pool.stats();
        rows.push(format!(
            "pool: {} heaps made ({} by the title's thread), {} MiB held before the run; {:?}",
            s.heaps - before.heaps,
            s.emergency - before.emergency,
            at_start / MIB,
            pool.holding()
        ));
        drop(pool);
        eprintln!("{}", rows.join("\n"));
    }

    /// The bytes of subresource `sub` of 2D colour texture `texture` (in
    /// the copy destination state), rows as their copyable footprint lays
    /// them out: row pitch and rows.
    pub(crate) fn read_sub(
        host: &crate::host::Host,
        texture: &ID3D12Resource,
        sub: u32,
    ) -> (Vec<u8>, usize, usize) {
        // SAFETY: a live resource.
        let d = unsafe { texture.GetDesc() };
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let (mut rows, mut total) = (0u32, 0u64);
        // SAFETY: room for the one footprint requested.
        unsafe {
            host.device.GetCopyableFootprints(
                &d,
                sub,
                1,
                0,
                Some(&mut footprint),
                Some(&mut rows),
                None,
                Some(&mut total),
            );
        }
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: total,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut readback: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a readback buffer.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .unwrap();
        }
        let readback = readback.unwrap();
        let list = host.new_list().unwrap();
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(&readback) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: footprint,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: sub,
            },
        };
        let t = crate::host::transition;
        let (dest, source) = (
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );
        // SAFETY: an open list; both locations name live resources.
        unsafe {
            list.list.ResourceBarrier(&[t(texture, dest, source)]);
            list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
            list.list.ResourceBarrier(&[t(texture, source, dest)]);
        }
        list.close();
        host.execute(&[&list]);
        host.drain();
        let mut data = std::ptr::null_mut();
        // SAFETY: a readback buffer the GPU is done with.
        unsafe { readback.Map(0, None, Some(&mut data)).unwrap() };
        // SAFETY: `total` bytes mapped.
        let bytes =
            unsafe { std::slice::from_raw_parts(data.cast::<u8>(), total as usize) }.to_vec();
        let f = footprint.Footprint;
        (bytes, f.RowPitch as usize, rows as usize)
    }

    /// Fills subresource `sub` of `texture` (RGBA8, in the copy destination
    /// state) with `byte`, through an upload buffer, and runs the copy.
    fn fill_sub(host: &crate::host::Host, texture: &ID3D12Resource, sub: u32, byte: u8) {
        // SAFETY: a live resource.
        let d = unsafe { texture.GetDesc() };
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut total = 0u64;
        // SAFETY: room for the one footprint requested.
        unsafe {
            host.device.GetCopyableFootprints(
                &d,
                sub,
                1,
                0,
                Some(&mut footprint),
                None,
                None,
                Some(&mut total),
            );
        }
        footprint.Offset = 0;
        let (upload, data) = host.upload_buffer(total).unwrap();
        // SAFETY: `total` bytes of the mapped buffer.
        unsafe { data.write_bytes(byte, total as usize) };
        let list = host.new_list().unwrap();
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: sub,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&upload) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: footprint,
            },
        };
        // SAFETY: an open list; both locations name live resources.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
        list.close();
        host.execute(&[&list]);
        host.drain();
    }

    /// Whether every texel row of `read` (pitch, rows) is `byte` over
    /// `width` bytes.
    pub(crate) fn rows_are(read: &(Vec<u8>, usize, usize), width: usize, byte: u8) -> bool {
        let (bytes, pitch, rows) = read;
        (0..*rows).all(|r| {
            bytes[r * pitch..r * pitch + width]
                .iter()
                .all(|&b| b == byte)
        })
    }

    /// A texture of two tiles or more is reserved, nothing of it mapped:
    /// its units (each standard mip, the packed mips) are mapped as asked,
    /// a copy reaches the mapped ones and is dropped in the others, which
    /// read zero; unmapped, the texture reads zero again, and the ranges
    /// it gave back return to their heap only once the host fence passes
    /// the submission after the unmapping; mapped again, it takes copies
    /// again; dropped, its ranges go back at once and it leaves the
    /// registry.
    #[test]
    fn reserved_textures_map_and_give_back_their_tiles() {
        let _serial = crate::host::submissions::serial();
        let test = "resource::pool::tests::reserved_textures_map_and_give_back_their_tiles";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let config = Config {
            tiles: true,
            ..Config::sized(4 * MIB, 4 * MIB)
        };
        let pool = Pool::new(gpu.device.clone(), config, unlimited());
        if !reserves(&pool) {
            eprintln!("{test}: skipped, the device has no tiled resources tier 2");
            return;
        }
        assert!(pool.settle(Duration::from_secs(5)));
        assert!(pool.resident(Duration::from_secs(5)), "the heaps resident");
        let state = D3D12_RESOURCE_STATE_COPY_DEST;
        let d = texture(256, 256, DXGI_FORMAT_R8G8B8A8_UNORM, 9);
        let placed = pool.place(&d, state).expect("reserved");
        let tiles = placed.slot.tiles().expect("a reserved texture").clone();
        let texture = placed.texture.clone();
        assert!(Arc::ptr_eq(&tiles_of(&texture).unwrap(), &tiles));
        // Units: standard mips from the largest, then the packed ones,
        // covering the chain once.
        let units = tiles.units().to_vec();
        assert!(units.len() >= 2, "{units:?}");
        let covered: Vec<u32> = units
            .iter()
            .flat_map(|u| u.first_mip..u.first_mip + u.mips)
            .collect();
        assert_eq!(covered, (0..9).collect::<Vec<_>>(), "{units:?}");
        let all: u64 = units.iter().map(|u| u64::from(u.tiles) * TILE).sum();
        assert_eq!(placed.bytes, all);
        assert_eq!(tiles.mapped(), 0, "nothing mapped yet");
        assert_eq!(pool.stats().reserved, 1);

        // The first unit (mip 0) and the last; not the second (mip 1).
        let mut wanted = vec![false; units.len()];
        wanted[0] = true;
        *wanted.last_mut().unwrap() = true;
        let done = tiles.map(host, &wanted);
        assert_eq!((done.added, done.removed, done.short), (2, 0, 0));
        assert!(tiles.holds(&wanted));
        let both = u64::from(units[0].tiles + units.last().unwrap().tiles) * TILE;
        assert_eq!(tiles.mapped(), both);
        assert_eq!(pool.holding().mapped, both);
        fill_sub(host, &texture, 0, 0x5a);
        fill_sub(host, &texture, 1, 0x6b);
        assert!(rows_are(&read_sub(host, &texture, 0), 256 * 4, 0x5a));
        assert!(
            rows_are(&read_sub(host, &texture, 1), 128 * 4, 0),
            "an unmapped unit reads zero"
        );

        // Unmapped: zero again, the ranges retired until the fence passes
        // a submission queued after the unmapping (the gate holds the queue
        // meanwhile, other tests' submissions too).
        let used = pool.holding().used;
        let mut gate = crate::host::Gate::close(host);
        assert_eq!(tiles.evict(host), 2);
        assert_eq!(tiles.mapped(), 0);
        assert!(!tiles.any());
        pool.inner.reclaim(host.completed());
        let held = pool.holding();
        assert_eq!(held.retired, both, "not back before the fence");
        assert_eq!(held.used, used);
        gate.open();
        assert!(rows_are(&read_sub(host, &texture, 0), 256 * 4, 0));
        pool.inner.reclaim(host.completed());
        assert_eq!(pool.holding().retired, 0, "back once one ran");
        assert_eq!(pool.holding().used, used - both);

        // Mapped again: copies reach it again.
        let done = tiles.map(host, &vec![true; units.len()]);
        assert_eq!((done.added, done.short), (units.len() as u32, 0));
        fill_sub(host, &texture, 1, 0x7c);
        assert!(rows_are(&read_sub(host, &texture, 1), 128 * 4, 0x7c));

        // Dropped: its ranges go back at once, and it leaves the registry.
        drop(placed);
        drop(tiles);
        assert!(tiles_of(&texture).is_none());
        let after = pool.holding();
        assert_eq!((after.mapped, after.reserved), (0, 0));
        assert_eq!(after.used, used - both, "{after:?}");
    }

    /// A unit's tiles are taken in pieces of a fragmented heap, the longest
    /// first, or not at all.
    #[test]
    fn units_are_taken_in_pieces_or_not_at_all() {
        let mut r = Ranges::new(8 * TILE);
        let [_a, b, _c, d, _e] = [2, 1, 2, 1, 2].map(|n| r.take(n * TILE).unwrap());
        r.give(b, TILE);
        r.give(d, TILE);
        assert_eq!(scattered(&mut r, 3 * TILE), None);
        assert_eq!(r.free(), 2 * TILE, "nothing taken");
        assert_eq!(
            scattered(&mut r, 2 * TILE),
            Some(vec![(b, TILE), (d, TILE)])
        );
        assert_eq!(r.free(), 0);
    }

    #[test]
    fn tiles_are_off_unless_asked_or_low_on_memory() {
        assert!(!asks_tiles(None, false), "off by default");
        assert!(asks_tiles(None, true), "on by default below 32 GiB of RAM");
        for low in [false, true] {
            assert!(asks_tiles(Some("1"), low));
            for off in ["0", "off", "FALSE", " 0 ", ""] {
                assert!(!asks_tiles(Some(off), low), "{off}");
            }
        }
    }

    /// Larger textures than a quarter heap and alignments the pool has no
    /// class for are not placed.
    #[test]
    fn what_the_pool_does_not_take() {
        let test = "resource::pool::tests::what_the_pool_does_not_take";
        let Some((_gpu, pool)) = pool(test, unlimited()) else {
            return;
        };
        let big = texture(1024, 1024, DXGI_FORMAT_R8G8B8A8_UNORM, 1);
        assert_eq!(
            pool.place(&big, D3D12_RESOURCE_STATE_COPY_DEST).err(),
            Some("too large for the pool")
        );
    }

    #[test]
    fn small_placement_follows_vendor_default_and_override() {
        assert!(!asks_small(None, true), "off by default on AMD");
        assert!(asks_small(None, false), "on by default on non-AMD");
        for on in ["1", "true", "on"] {
            assert!(asks_small(Some(on), true), "{on} forces on for AMD");
            assert!(asks_small(Some(on), false), "{on}");
        }
        for off in ["0", "off", "FALSE", " 0 ", ""] {
            assert!(!asks_small(Some(off), false), "{off} disables small");
            assert!(!asks_small(Some(off), true), "{off}");
        }
    }

    #[test]
    fn disabling_small_placement_routes_small_textures_to_64_kib_class() {
        let test = "resource::pool::tests::disabling_small_placement_routes_small_textures_to_64_kib_class";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let pool = Pool::with_small(
            gpu.device.clone(),
            Config::sized(MIB, MIB),
            unlimited(),
            false,
        );
        assert!(pool.settle(Duration::from_secs(5)));
        assert!(pool.resident(Duration::from_secs(5)));
        let settled = pool.holding();
        assert_eq!(settled.heaps, 1, "only Class::Normal heap reserved");
        assert_eq!(settled.bytes, MIB, "no Class::Small heap reserved");
        assert!(
            pool.inner.lock().classes[Class::Small.index()].is_empty(),
            "no Class::Small heap created when disabled"
        );
        let small = texture(64, 64, DXGI_FORMAT_R8G8B8A8_UNORM, 1);
        let placed = pool
            .place(&small, D3D12_RESOURCE_STATE_COPY_DEST)
            .expect("placed");
        let (class, offset, len) = placed.slot.place();
        assert_eq!(class, Class::Normal);
        assert_eq!(offset % KIB_64, 0);
        assert_eq!(len % KIB_64, 0);
    }

    #[test]
    fn heap_failure_backoff_window_and_wait_duration() {
        let now = Instant::now();
        let idle = Duration::from_secs(1);
        assert!(!backoff_active(None, now));
        assert_eq!(backoff_wait(None, now, idle), idle);

        let until = now + HEAP_FAIL_BACKOFF;
        assert!(backoff_active(Some(until), now));
        assert_eq!(backoff_wait(Some(until), now, idle), HEAP_FAIL_BACKOFF);
        assert_eq!(
            backoff_wait(Some(now + Duration::from_secs(5)), now, idle),
            idle,
            "wait is capped to the normal idle poll interval"
        );
        // At or past the deadline, backoff is inactive and normal idle wait resumes.
        assert!(!backoff_active(Some(until), until));
        assert_eq!(backoff_wait(Some(until), until, idle), idle);
    }

    #[test]
    fn existing_heap_free_ranges_serve_placements_during_heap_backoff() {
        let test =
            "resource::pool::tests::existing_heap_free_ranges_serve_placements_during_heap_backoff";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let pool = Pool::with_small(
            gpu.device.clone(),
            Config::sized(MIB, MIB),
            unlimited(),
            false,
        );
        assert!(pool.settle(Duration::from_secs(5)));
        assert!(pool.resident(Duration::from_secs(5)));
        assert_eq!(pool.stats().heaps, 1);

        // Enter the post-refusal backoff window: existing heap free ranges must
        // keep serving placements while new CreateHeap attempts are suppressed.
        pool.inner.lock().backoff_until = Some(Instant::now() + Duration::from_secs(10));
        let desc = texture(256, 256, DXGI_FORMAT_R8G8B8A8_UNORM, 1);
        let state = D3D12_RESOURCE_STATE_COPY_DEST;
        let mut kept: Vec<Placed> = (0..4)
            .map(|_| {
                pool.place(&desc, state)
                    .expect("fits in existing 1 MiB heap")
            })
            .collect();
        assert_eq!(pool.stats().heaps, 1, "no new heap made during backoff");

        // Existing heap is now full; new heap creation is suppressed by backoff.
        assert_eq!(
            pool.place(&desc, state).err(),
            Some("no heap (budget or driver)")
        );
        assert_eq!(pool.stats().heaps, 1, "still no new heap during backoff");

        // Freeing one slot lets another placement succeed from the existing heap.
        kept.pop();
        let reused = pool
            .place(&desc, state)
            .expect("freed range serves placement during backoff");
        kept.push(reused);
        assert_eq!(pool.stats().heaps, 1);

        // Once the backoff window expires, new heap growth resumes.
        pool.inner.lock().backoff_until = Some(Instant::now() - Duration::from_millis(1));
        let overflow = pool
            .place(&desc, state)
            .expect("heap growth resumes after backoff expires");
        assert!(pool.stats().heaps >= 2);
        drop(overflow);
        drop(kept);
    }
}
