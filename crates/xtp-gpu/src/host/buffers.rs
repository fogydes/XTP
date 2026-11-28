//! Host buffers placed in heaps reserved ahead (frame-prof's session A,
//! task 7112c207). Past the machine's VidMm threshold every new allocation
//! waits for one admission, about 0.4 s, and a committed resource per host
//! copy of title memory, per upload, budget word or scratch chunk paid one
//! on a title thread: 25 % of the loading screen's time and 33 % of the
//! world's, with `shadow::find`'s lock held across the wait by the thread
//! that paid it. Here each buffer is a placed resource in a heap of
//! [`HEAP`] bytes that a thread of ours made beforehand (one admission per
//! heap): placing one allocates nothing in VidMm.
//!
//! Each buffer keeps its own resource (and so its own resource state:
//! copies, indirect arguments and #334's barriers move whole resources),
//! aligned as D3D12 places buffers, to 64 KiB. Its range goes back to its
//! heap when the resource is destroyed, which XTP does only once no
//! submission uses it (lists keep what they name until they are retired):
//! the resource holds a [`Lease`] as private data, which D3D12 releases
//! with it.
//!
//! The reserve thread keeps at least [`RESERVE`] bytes free in each pool,
//! makes a heap when a thread waits for room, and releases empty heaps
//! past one heap's worth of spare room. A buffer larger than a heap gets a
//! heap of its own, made on the asking thread. `XTP_BUFFER_POOL=0` brings
//! back the committed buffers.
//!
//! The card's own memory has a pool too ([`Kind::Card`]), for the copies of
//! big title buffers that live there ([`crate::shadow::cards`]). It is
//! started only when those are on, so a run without them reserves no
//! memory on the card.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::core::{GUID, HRESULT, IUnknown, IUnknown_Vtbl, Interface};

/// What a buffer is for, and so the heaps it comes from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Kind {
    /// System memory the CPU reads and writes cached and shaders read and
    /// write: host copies of title memory, budget words (a custom heap,
    /// write-back, L0). Starts in the common state, zeroed.
    System,
    /// Upload memory (write-combined, read by the GPU): staging, scratch,
    /// list uploads. In the generic read state; its bytes are the caller's
    /// to write.
    Upload,
    /// The card's own memory (a default heap), which shaders read and write
    /// without going through PCIe and the CPU cannot map: the card side of
    /// big title buffers' copies. Starts in the common state; its bytes are
    /// undefined until the caller's GPU work writes them.
    Card,
}

/// The placement unit of buffers.
const ALIGN: u64 = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT as u64;
/// One heap's bytes (smaller in tests, to see heaps fill up).
#[cfg(not(test))]
const HEAP: u64 = 256 << 20;
#[cfg(test)]
const HEAP: u64 = 4 << 20;
/// Free bytes the reserve thread keeps in each pool.
const RESERVE: u64 = HEAP / 2;

/// Free ranges of a heap: best fit, merged when given back.
#[derive(Default)]
pub(crate) struct Ranges {
    /// Offset → length.
    by_offset: BTreeMap<u64, u64>,
    /// (length, offset).
    by_size: BTreeSet<(u64, u64)>,
    free: u64,
}

impl Ranges {
    pub(crate) fn new(size: u64) -> Ranges {
        let mut r = Ranges::default();
        r.insert(0, size);
        r
    }

    fn insert(&mut self, offset: u64, len: u64) {
        self.by_offset.insert(offset, len);
        self.by_size.insert((len, offset));
        self.free += len;
    }

    fn remove(&mut self, offset: u64, len: u64) {
        self.by_offset.remove(&offset);
        self.by_size.remove(&(len, offset));
        self.free -= len;
    }

    /// The offset of `len` bytes taken from the smallest free range that
    /// holds them.
    pub(crate) fn take(&mut self, len: u64) -> Option<u64> {
        let &(size, offset) = self.by_size.range((len, 0)..).next()?;
        self.remove(offset, size);
        if size > len {
            self.insert(offset + len, size - len);
        }
        Some(offset)
    }

    /// Gives `[offset, offset + len)` back, merged with its free neighbours.
    pub(crate) fn give(&mut self, mut offset: u64, mut len: u64) {
        if let Some((&before, &n)) = self.by_offset.range(..offset).next_back()
            && before + n == offset
        {
            self.remove(before, n);
            (offset, len) = (before, n + len);
        }
        if let Some(&n) = self.by_offset.get(&(offset + len)) {
            self.remove(offset + len, n);
            len += n;
        }
        self.insert(offset, len);
    }

    fn largest(&self) -> u64 {
        self.by_size.last().map_or(0, |&(len, _)| len)
    }
}

struct Heap {
    id: u64,
    heap: ID3D12Heap,
    size: u64,
    free: Ranges,
    /// Per [`ALIGN`] unit: whether a buffer had it (so it may hold old
    /// bytes; a new heap is zeroed).
    used: Vec<bool>,
    rt_reservation: Option<RtReservation>,
}

#[derive(Default)]
struct State {
    heaps: Vec<Heap>,
    next_id: u64,
    /// The reserve thread is making a heap.
    growing: bool,
    /// Threads waiting for room.
    waiting: usize,
    /// The last heap the reserve thread tried could not be made.
    failed: bool,
}

impl State {
    fn free(&self) -> u64 {
        self.heaps.iter().map(|h| h.free.free).sum()
    }

    fn reserved(&self) -> u64 {
        self.heaps.iter().map(|h| h.size).sum()
    }

    /// Takes `len` bytes from the heap whose best fit is smallest: the
    /// heap, its id, the offset, and the units that may hold old bytes.
    fn take(&mut self, len: u64) -> Option<Room> {
        let heap = self
            .heaps
            .iter_mut()
            .filter(|h| h.free.largest() >= len)
            .min_by_key(|h| h.free.largest())?;
        let offset = heap.free.take(len)?;
        let mut dirty: Vec<(u64, u64)> = Vec::new();
        for unit in offset / ALIGN..(offset + len) / ALIGN {
            if std::mem::replace(&mut heap.used[unit as usize], true) {
                let at = unit * ALIGN - offset;
                match dirty.last_mut() {
                    Some((from, n)) if *from + *n == at => *n += ALIGN,
                    _ => dirty.push((at, ALIGN)),
                }
            }
        }
        Some(Room {
            heap: heap.heap.clone(),
            id: heap.id,
            offset,
            dirty,
        })
    }

    fn add(&mut self, heap: ID3D12Heap, size: u64) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.heaps.push(Heap {
            rt_reservation: None,
            id,
            heap,
            size,
            free: Ranges::new(size),
            used: vec![false; (size / ALIGN) as usize],
        });
        id
    }
}

/// Room taken for a buffer: its heap, the heap's id, the offset, and the
/// pieces (offset in the buffer, bytes) that may hold old bytes.
struct Room {
    heap: ID3D12Heap,
    id: u64,
    offset: u64,
    dirty: Vec<(u64, u64)>,
}

/// The heaps of one kind of buffer.
pub(crate) struct Pool {
    kind: Kind,
    device: ID3D12Device,
    state: Mutex<State>,
    /// Signalled when room is given back or wanted, and when a heap is made.
    changed: Condvar,
    /// Microseconds title threads waited for room, and how many times.
    waited: [AtomicU64; 2],
}

// SAFETY: D3D12 devices and heaps are free-threaded.
unsafe impl Send for Pool {}
// SAFETY: as above.
unsafe impl Sync for Pool {}

/// The pools, System then Upload.
static POOLS: OnceLock<[Pool; 2]> = OnceLock::new();

/// The pool of the card's memory, started with the others when copies of
/// big title buffers go to the card ([`crate::shadow::cards`]).
static CARDS: OnceLock<Pool> = OnceLock::new();

/// Whether buffers come from the pools: unless `XTP_BUFFER_POOL=0`.
fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("XTP_BUFFER_POOL")
            .ok()
            .is_none_or(|v| v.trim() != "0")
    })
}

/// Starts the pools of the host device and their reserve threads, which
/// make the first heaps at once.
pub(crate) fn start(device: &ID3D12Device) {
    if !enabled() {
        return;
    }
    let mut made = false;
    let pools = POOLS.get_or_init(|| {
        made = true;
        [Kind::System, Kind::Upload].map(|kind| Pool::new(kind, device))
    });
    if !made {
        return;
    }
    for pool in pools {
        pool.start_reserve();
    }
    if crate::shadow::cards() {
        let _ = card_pool(device);
    }
}

/// Start the card reserve on its first DXR use when shadow copies use system memory.
fn card_pool(device: &ID3D12Device) -> Option<&'static Pool> {
    static STARTED: OnceLock<bool> = OnceLock::new();
    if !enabled() {
        return None;
    }
    let pool = CARDS.get_or_init(|| Pool::new(Kind::Card, device));
    STARTED.get_or_init(|| pool.start_reserve()).then_some(pool)
}

/// Actual local memory backing RT pages and heaps, including allocations
/// still held by recordings. Concurrent heap creation reserves before D3D12.
static RT_RESERVED: AtomicU64 = AtomicU64::new(0);
/// Reserved bytes whose resource D3D12 is still making: not yet in the
/// usage DXGI reports, so the headroom check adds them.
static RT_PENDING: AtomicU64 = AtomicU64::new(0);
pub(crate) struct RtReservation(u64);

fn fits_rt_budget(reserved: u64, size: u64, budget: u64, used: u64) -> bool {
    let cap = (budget / 8).min(512 << 20);
    let headroom = (budget / 16).min(256 << 20);
    size <= cap.saturating_sub(reserved)
        && size <= budget.saturating_sub(used).saturating_sub(headroom)
}

/// The RT memory cap before DXGI has ever reported a budget: an eighth of a
/// 2 GiB card, as [`fits_rt_budget`] gives it, without the usage check.
const UNREAD_RT_CAP: u64 = 256 << 20;

/// Whether `size` more bytes fit the RT budget with `reserved` taken: by
/// the DXGI reading `(budget, used)`, or by [`UNREAD_RT_CAP`] without one.
fn fits_rt(reserved: u64, size: u64, reading: Option<(u64, u64)>) -> bool {
    match reading {
        Some((budget, used)) => fits_rt_budget(reserved, size, budget, used),
        None => size <= UNREAD_RT_CAP.saturating_sub(reserved),
    }
}

/// The local budget and usage to check against: this reading, kept as the
/// last good one, or the last good one when DXGI cannot report (a failed
/// query is no budget of 0). `None` before any reading.
fn budget_reading(
    reading: Option<(u64, u64)>,
    last: &mut Option<(u64, u64)>,
) -> Option<(u64, u64)> {
    match reading {
        Some(now) => {
            *last = Some(now);
            Some(now)
        }
        None => *last,
    }
}

impl RtReservation {
    /// Reserves `size` bytes of the RT budget and makes their resource with
    /// `make`, outside the lock that orders reservations (D3D12 can take
    /// milliseconds there). The reservation is given back if `make` fails.
    pub(crate) fn allocate<T>(
        device: &ID3D12Device,
        size: u64,
        make: impl FnOnce() -> Option<T>,
    ) -> Option<(T, Self)> {
        static GROW: Mutex<Option<(u64, u64)>> = Mutex::new(None);
        let reservation = {
            let mut last = GROW.lock().unwrap_or_else(|e| e.into_inner());
            let reading = crate::dxgi::video_memory(device);
            if reading.is_none() {
                static FAILED: AtomicU64 = AtomicU64::new(0);
                let failed = FAILED.fetch_add(1, Ordering::Relaxed) + 1;
                if failed.is_power_of_two() {
                    tracing::warn!(
                        failed,
                        last = ?*last,
                        cap = UNREAD_RT_CAP,
                        "DXGI reports no local video memory budget: host DXR allocations use \
                         the last reading, or a fixed cap before any"
                    );
                }
            }
            // Resources reserved but still being made are not in `used` yet.
            let pending = RT_PENDING.load(Ordering::Acquire);
            let reading = budget_reading(reading, &mut last)
                .map(|(budget, used)| (budget, used.saturating_add(pending)));
            let granted =
                RT_RESERVED.fetch_update(Ordering::AcqRel, Ordering::Acquire, |reserved| {
                    fits_rt(reserved, size, reading).then(|| reserved + size)
                });
            if let Err(reserved) = granted {
                static REFUSED: AtomicU64 = AtomicU64::new(0);
                let refused = REFUSED.fetch_add(1, Ordering::Relaxed) + 1;
                if refused.is_power_of_two() {
                    tracing::warn!(
                        size,
                        reserved,
                        budget_and_used = ?reading,
                        refused,
                        "host DXR memory budget exhausted: the allocation is refused"
                    );
                }
                return None;
            }
            RT_PENDING.fetch_add(size, Ordering::AcqRel);
            Self(size)
        };
        let made = make();
        RT_PENDING.fetch_sub(size, Ordering::AcqRel);
        Some((made?, reservation))
    }
}

impl Drop for RtReservation {
    fn drop(&mut self) {
        RT_RESERVED.fetch_sub(self.0, Ordering::AcqRel);
    }
}

/// Scratch and instance buffers use on-demand heaps separate from title
/// shadows. Releasing the last placed resource returns the heap's budget.
#[track_caller]
pub(crate) fn rt_buffer(
    device: &ID3D12Device,
    size: u64,
    state: D3D12_RESOURCE_STATES,
) -> Option<ID3D12Resource> {
    static POOL: OnceLock<Pool> = OnceLock::new();
    let pool = POOL.get_or_init(|| Pool::new(Kind::Card, device));
    let len = size.max(1).checked_next_multiple_of(ALIGN)?;
    let room = if let Some(room) = {
        let mut s = pool.lock();
        s.take(len)
    } {
        room
    } else {
        let bytes = if enabled() { len.max(4 << 20) } else { len };
        let (heap, charge) = RtReservation::allocate(device, bytes, || pool.make_heap(bytes))?;
        let mut s = pool.lock();
        // A simultaneous release can supply a better fit while CreateHeap runs.
        if let Some(room) = s.take(len) {
            room
        } else {
            s.add(heap, bytes);
            s.heaps.last_mut()?.rt_reservation = Some(charge);
            s.take(len)?
        }
    };
    pool.place_room(room, len, size, state)
        .map(|(resource, _)| resource)
}

/// A buffer of `size` bytes of `kind`, mapped (but a [`Kind::Card`] one,
/// which the CPU cannot map: its pointer is null), placed in a pool's
/// heap; `None` without pools (the caller makes a committed one).
#[track_caller]
pub(crate) fn buffer(kind: Kind, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
    let pool = match kind {
        Kind::Card => card_pool(&CARDS.get()?.device)?,
        kind => &POOLS.get()?[kind as usize],
    };
    pool.place(size)
}

/// A buffer of `size` bytes in the card's memory ([`Kind::Card`]): placed
/// in the card's pool when it runs, else committed (made on this thread).
/// `None` if it cannot be made.
#[track_caller]
pub(crate) fn card_buffer(device: &ID3D12Device, size: u64) -> Option<ID3D12Resource> {
    if let Some((buffer, _)) = buffer(Kind::Card, size) {
        return Some(buffer);
    }
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = buffer_desc(size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation with a valid description.
    let made = unsafe {
        device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut buffer,
        )
    };
    match made {
        Ok(()) => buffer,
        Err(e) => {
            tracing::warn!(error = %e, size, "no buffer in the card's memory");
            None
        }
    }
}

/// A buffer's description: `size` bytes (at least 1), `flags`.
fn buffer_desc(size: u64, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size.max(1),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    }
}

/// For the frame-rate line: per pool, heaps, MiB reserved and used, and
/// the title's waits for room (count, ms) since the last call.
pub(crate) fn summary() -> Option<String> {
    let pools = POOLS.get()?;
    let mut parts: Vec<String> = pools
        .iter()
        .chain(CARDS.get())
        .map(|p| {
            let s = p.lock();
            let (reserved, free) = (s.reserved(), s.free());
            let waits = p.waited[1].swap(0, Ordering::Relaxed);
            let us = p.waited[0].swap(0, Ordering::Relaxed);
            let heaps = s.heaps.len();
            drop(s);
            let callers = by_caller(p.kind, 6);
            let callers = if callers.is_empty() {
                String::new()
            } else {
                format!(" (by caller: {callers})")
            };
            format!(
                "{:?} {heaps} heaps {} MiB, {} MiB used, {waits} waits {:.1} ms{callers}",
                p.kind,
                reserved >> 20,
                (reserved - free) >> 20,
                us as f64 / 1e3,
            )
        })
        .collect();
    parts.extend(crate::shadow::card_summary());
    Some(parts.join("; "))
}

impl Pool {
    fn new(kind: Kind, device: &ID3D12Device) -> Self {
        Self {
            kind,
            device: device.clone(),
            state: Mutex::default(),
            changed: Condvar::new(),
            waited: Default::default(),
        }
    }

    fn start_reserve(&'static self) -> bool {
        let spawned = std::thread::Builder::new()
            .name(format!("xtp-gpu-{:?}-buffers", self.kind).to_lowercase())
            .spawn(move || self.reserve());
        if let Err(e) = spawned {
            tracing::warn!(error = %e, kind = ?self.kind, "no buffer reserve thread");
            return false;
        }
        true
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn heap_properties(&self) -> D3D12_HEAP_PROPERTIES {
        match self.kind {
            Kind::System => D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                ..Default::default()
            },
            Kind::Upload => D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_UPLOAD,
                ..Default::default()
            },
            Kind::Card => D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            },
        }
    }

    /// A new heap of `size` bytes (resident once made: the VidMm wait).
    fn make_heap(&self, size: u64) -> Option<ID3D12Heap> {
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: size,
            Properties: self.heap_properties(),
            Alignment: ALIGN,
            Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS,
        };
        let mut heap: Option<ID3D12Heap> = None;
        let started = std::time::Instant::now();
        // SAFETY: plain creation with a valid description.
        match unsafe { self.device.CreateHeap(&desc, &mut heap) } {
            Ok(()) => {
                tracing::debug!(
                    kind = ?self.kind,
                    mib = size >> 20,
                    ms = started.elapsed().as_millis(),
                    "buffer heap made"
                );
                heap
            }
            Err(e) => {
                tracing::warn!(error = %e, kind = ?self.kind, size, "no buffer heap");
                None
            }
        }
    }

    /// The reserve thread: makes heaps while the pool has less than
    /// [`RESERVE`] free or a thread waits for room, and releases empty
    /// heaps past one heap's worth of spare room.
    fn reserve(&self) {
        let name = format!("xtp-gpu-{:?}-buffers", self.kind).to_lowercase();
        crate::cores::keep_helper_on_fast_cores(&name);
        loop {
            let surplus = {
                let mut s = self.lock();
                loop {
                    let spare = s.free();
                    if (spare < RESERVE || s.waiting > 0) && !s.failed {
                        break None;
                    }
                    // An empty heap the rest of the reserve can spare.
                    if let Some(i) = s.heaps.iter().position(|h| h.free.free == h.size)
                        && spare - s.heaps[i].size >= RESERVE + HEAP
                    {
                        let heap = s.heaps.swap_remove(i);
                        break Some(heap.heap);
                    }
                    // After a failure, again only once room is wanted anew.
                    s = self.changed.wait(s).unwrap_or_else(|e| e.into_inner());
                    if s.waiting > 0 && s.failed {
                        s.failed = false;
                    }
                }
            };
            if let Some(heap) = surplus {
                drop(heap);
                continue;
            }
            self.lock().growing = true;
            let heap = self.make_heap(HEAP);
            let mut s = self.lock();
            s.growing = false;
            match heap {
                Some(heap) => {
                    s.add(heap, HEAP);
                }
                None => s.failed = true,
            }
            self.changed.notify_all();
        }
    }

    /// Room for `len` bytes (a multiple of [`ALIGN`]): taken from a heap,
    /// or from one the reserve thread makes meanwhile (waiting for it, no
    /// lock of ours held but the pool's, which the wait releases), or from
    /// a heap of its own for a buffer larger than a heap. `None` if no heap
    /// can be made.
    fn room(&self, len: u64) -> Option<Room> {
        let mut s = self.lock();
        if let Some(found) = s.take(len) {
            if s.free() < RESERVE {
                self.changed.notify_all();
            }
            return Some(found);
        }
        if len > HEAP {
            drop(s);
            let heap = self.make_heap(len)?;
            let mut s = self.lock();
            s.add(heap, len);
            return s.take(len);
        }
        let started = std::time::Instant::now();
        s.waiting += 1;
        self.changed.notify_all();
        let found = loop {
            if let Some(found) = s.take(len) {
                break Some(found);
            }
            if s.failed && !s.growing {
                break None;
            }
            s = self.changed.wait(s).unwrap_or_else(|e| e.into_inner());
        };
        s.waiting -= 1;
        self.waited[0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        self.waited[1].fetch_add(1, Ordering::Relaxed);
        found
    }

    /// Gives a buffer's range back (its resource is gone).
    fn give(&self, heap: u64, offset: u64, len: u64) {
        let mut s = self.lock();
        if let Some(i) = s.heaps.iter().position(|h| h.id == heap) {
            let h = &mut s.heaps[i];
            h.free.give(offset, len);
            if h.rt_reservation.is_some() && h.free.free == h.size {
                s.heaps.swap_remove(i);
            }
        }
        // Room for a waiting thread, or an empty heap to release.
        self.changed.notify_all();
    }

    #[track_caller]
    fn place(&'static self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let state = if self.kind == Kind::Upload {
            D3D12_RESOURCE_STATE_GENERIC_READ
        } else {
            D3D12_RESOURCE_STATE_COMMON
        };
        self.place_in_state(size, state)
    }

    #[track_caller]
    fn place_in_state(
        &'static self,
        size: u64,
        state: D3D12_RESOURCE_STATES,
    ) -> Option<(ID3D12Resource, *mut u8)> {
        #[cfg(test)]
        before_placing(size);
        let len = size.max(1).next_multiple_of(ALIGN);
        let room = self.room(len)?;
        self.place_room(room, len, size, state)
    }

    #[track_caller]
    fn place_room(
        &'static self,
        room: Room,
        len: u64,
        size: u64,
        state: D3D12_RESOURCE_STATES,
    ) -> Option<(ID3D12Resource, *mut u8)> {
        let site = std::panic::Location::caller();
        let Room {
            heap,
            id,
            offset,
            dirty,
        } = room;
        let flags = if self.kind == Kind::Upload {
            D3D12_RESOURCE_FLAG_NONE
        } else {
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: flags,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: a buffer placed in our heap, in a range of its own;
        // mapped as committed ones are (but in the card's memory, which
        // cannot be).
        let mut lease_created = false;
        // SAFETY: the room is exclusively reserved through the resource lease.
        let made = unsafe {
            self.device
                .CreatePlacedResource(&heap, offset, &desc, state, None, &mut buffer)
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    if self.kind != Kind::Card {
                        b.Map(0, None, Some(&mut data))?;
                    }
                    let lease = Lease::interface(self, id, offset, len, site);
                    lease_created = true;
                    b.SetPrivateDataInterface(&LEASE, &lease)
                })
        };
        let buffer = match (made, buffer) {
            (Ok(()), Some(buffer)) => buffer,
            (made, buffer) => {
                // Not leased (or the lease went with the resource, giving
                // the range back): give it back if it did not.
                drop(buffer);
                if !lease_created {
                    self.give(id, offset, len);
                }
                if let Err(e) = made {
                    tracing::warn!(error = %e, kind = ?self.kind, size, "no placed buffer");
                }
                return None;
            }
        };
        let data: *mut u8 = data.cast();
        if self.kind == Kind::System {
            // A committed buffer starts zeroed; a range used before is not.
            for (at, n) in dirty {
                let n = n.min(size.saturating_sub(at));
                // SAFETY: inside the buffer's mapping.
                unsafe { data.add(at as usize).write_bytes(0, n as usize) };
            }
        }
        Some((buffer, data))
    }
}

/// The private data key of a buffer's [`Lease`].
const LEASE: GUID = GUID::from_u128(0x6a1f_02c4_3b7d_4e1a_9c55_8d0e_b2f3_7a61);

/// Whether a resource holds a lease (and so gives its range back itself).
#[cfg(test)]
fn has_lease(resource: &ID3D12Resource) -> bool {
    let mut size = 0u32;
    // SAFETY: asks the size of our private data only.
    unsafe { resource.GetPrivateData(&LEASE, &mut size, None) }.is_ok()
}

/// Placed bytes alive, by pool and by the line of XTP that asked for them:
/// buffers alive and their bytes (placement units). Counted as leases are
/// made and released; it changes nothing of either.
type Held = std::collections::HashMap<(usize, &'static str, u32), (u64, u64)>;
static HELD: Mutex<Option<Held>> = Mutex::new(None);

/// Counts `len` bytes of pool `kind` asked for at `site` as made (`made`)
/// or given back. Takes its own lock only, never a pool's.
fn held(kind: Kind, site: &'static std::panic::Location<'static>, len: u64, made: bool) {
    let mut held = HELD.lock().unwrap_or_else(|e| e.into_inner());
    let entry = held
        .get_or_insert_with(Default::default)
        .entry((kind as usize, site.file(), site.line()))
        .or_default();
    if made {
        entry.0 += 1;
        entry.1 += len;
    } else {
        entry.0 = entry.0.saturating_sub(1);
        entry.1 = entry.1.saturating_sub(len);
    }
}

/// The lines holding the most placed bytes of pool `kind`, at most
/// `top` of them with 16 MiB or more: `file:line MiB in buffers`.
pub(crate) fn by_caller(kind: Kind, top: usize) -> String {
    let held = HELD.lock().unwrap_or_else(|e| e.into_inner());
    let mut rows: Vec<(&'static str, u32, u64, u64)> = held
        .iter()
        .flatten()
        .filter(|((k, _, _), (_, bytes))| *k == kind as usize && *bytes >= 16 << 20)
        .map(|((_, file, line), (n, bytes))| (*file, *line, *n, *bytes))
        .collect();
    drop(held);
    rows.sort_by_key(|row| std::cmp::Reverse(row.3));
    rows.iter()
        .take(top)
        .map(|(file, line, n, bytes)| {
            let name = file.rsplit(['/', '\\']).next().unwrap_or(file);
            format!("{name}:{line} {} MiB in {n}", bytes >> 20)
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// A placed buffer's range in its heap: a COM object (`IUnknown` alone)
/// that the resource holds as private data, which D3D12 releases with the
/// resource. Its last release gives the range back.
#[repr(C)]
struct Lease {
    /// First, as in every COM object.
    vtable: &'static IUnknown_Vtbl,
    refs: std::sync::atomic::AtomicU32,
    pool: &'static Pool,
    heap: u64,
    offset: u64,
    len: u64,
    /// Where the buffer was asked for ([`by_caller`]).
    site: &'static std::panic::Location<'static>,
}

static LEASE_VTABLE: IUnknown_Vtbl = IUnknown_Vtbl {
    QueryInterface: lease_query,
    AddRef: lease_add_ref,
    Release: lease_release,
};

impl Lease {
    /// A new lease, as the interface that holds its one reference.
    fn interface(
        pool: &'static Pool,
        heap: u64,
        offset: u64,
        len: u64,
        site: &'static std::panic::Location<'static>,
    ) -> IUnknown {
        held(pool.kind, site, len, true);
        let lease = Box::into_raw(Box::new(Lease {
            vtable: &LEASE_VTABLE,
            refs: std::sync::atomic::AtomicU32::new(1),
            pool,
            heap,
            offset,
            len,
            site,
        }));
        // SAFETY: a COM object of ours with one reference, which the
        // interface takes over.
        unsafe { IUnknown::from_raw(lease.cast()) }
    }
}

unsafe extern "system" fn lease_query(
    this: *mut std::ffi::c_void,
    iid: *const GUID,
    out: *mut *mut std::ffi::c_void,
) -> HRESULT {
    const E_NOINTERFACE: HRESULT = HRESULT(0x8000_4002_u32 as i32);
    const E_POINTER: HRESULT = HRESULT(0x8000_4003_u32 as i32);
    if out.is_null() || iid.is_null() {
        return E_POINTER;
    }
    // SAFETY: the caller's pointers, checked above; `this` is a lease.
    unsafe {
        if *iid == IUnknown::IID {
            lease_add_ref(this);
            *out = this;
            HRESULT(0)
        } else {
            *out = std::ptr::null_mut();
            E_NOINTERFACE
        }
    }
}

unsafe extern "system" fn lease_add_ref(this: *mut std::ffi::c_void) -> u32 {
    // SAFETY: a live lease (the caller holds a reference).
    let lease = unsafe { &*this.cast::<Lease>() };
    lease.refs.fetch_add(1, Ordering::Relaxed) + 1
}

unsafe extern "system" fn lease_release(this: *mut std::ffi::c_void) -> u32 {
    let lease = this.cast::<Lease>();
    // SAFETY: a live lease (the caller gives up its reference).
    let left = unsafe { &*lease }.refs.fetch_sub(1, Ordering::Release) - 1;
    if left == 0 {
        std::sync::atomic::fence(Ordering::Acquire);
        // SAFETY: the last reference: the box made in `Lease::new`.
        let lease = unsafe { Box::from_raw(lease) };
        held(lease.pool.kind, lease.site, lease.len, false);
        lease.pool.give(lease.heap, lease.offset, lease.len);
    }
    left
}

/// For tests: a buffer size whose placing waits until the receiver gets a
/// message (or closes), as a creation that takes long.
#[cfg(test)]
pub(crate) static HOLD: Mutex<Option<(u64, std::sync::mpsc::Receiver<()>)>> = Mutex::new(None);

#[cfg(test)]
fn before_placing(size: u64) {
    let held = {
        let mut hold = HOLD.lock().unwrap_or_else(|e| e.into_inner());
        match &*hold {
            Some((held, _)) if *held == size => hold.take(),
            _ => None,
        }
    };
    if let Some((_, release)) = held {
        let _ = release.recv_timeout(std::time::Duration::from_secs(20));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bytes a line holds are counted up as leases are made and down as
    /// they go, and only lines with 16 MiB or more are named.
    #[test]
    fn placed_bytes_are_counted_by_the_line_that_asked() {
        let site = std::panic::Location::caller();
        let line = format!("buffers.rs:{} ", site.line());
        held(Kind::Card, site, 40 << 20, true);
        held(Kind::Card, site, 24 << 20, true);
        assert!(by_caller(Kind::Card, 6).contains(&format!("{line}64 MiB in 2")));
        assert!(!by_caller(Kind::Upload, 6).contains(&line));
        held(Kind::Card, site, 40 << 20, false);
        assert!(by_caller(Kind::Card, 6).contains(&format!("{line}24 MiB in 1")));
        held(Kind::Card, site, 24 << 20, false);
        assert!(!by_caller(Kind::Card, 6).contains(&line));
    }

    #[test]
    fn ranges_take_the_best_fit_and_merge_what_comes_back() {
        let mut r = Ranges::new(10 * ALIGN);
        let a = r.take(3 * ALIGN).unwrap();
        let b = r.take(2 * ALIGN).unwrap();
        let c = r.take(ALIGN).unwrap();
        assert_eq!((a, b, c), (0, 3 * ALIGN, 5 * ALIGN));
        r.give(b, 2 * ALIGN);
        // The 2-unit hole fits a 2-unit buffer better than the 4-unit tail.
        assert_eq!(r.take(2 * ALIGN), Some(b));
        r.give(a, 3 * ALIGN);
        r.give(b, 2 * ALIGN);
        assert_eq!(r.largest(), 5 * ALIGN, "merged with each other");
        r.give(c, ALIGN);
        assert_eq!(r.largest(), 10 * ALIGN, "and with the tail");
        assert_eq!((r.free, r.by_offset.len()), (10 * ALIGN, 1));
        assert_eq!(r.take(11 * ALIGN), None);
    }

    /// Against a model: random takes and gives keep the free ranges
    /// disjoint, merged and accounted for.
    #[test]
    fn ranges_match_a_model() {
        let units = 64u64;
        let mut r = Ranges::new(units * ALIGN);
        let mut held: Vec<(u64, u64)> = Vec::new();
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _ in 0..5000 {
            if next() % 2 == 0 || held.is_empty() {
                let len = (next() % 6 + 1) * ALIGN;
                if let Some(at) = r.take(len) {
                    assert!(held.iter().all(|&(o, n)| at + len <= o || o + n <= at));
                    held.push((at, len));
                }
            } else {
                let (at, len) = held.swap_remove((next() % held.len() as u64) as usize);
                r.give(at, len);
            }
            let used: u64 = held.iter().map(|&(_, n)| n).sum();
            assert_eq!(r.free + used, units * ALIGN);
            let mut end = None;
            for (&o, &n) in &r.by_offset {
                assert!(end.is_none_or(|e| e < o), "free ranges merged");
                end = Some(o + n);
            }
        }
    }

    /// Buffers of a pool on the host device: placed, mapped, distinct; a
    /// range used before is zeroed for a system buffer; the pool grows by
    /// whole heaps, a buffer larger than a heap gets one of its own, and
    /// ranges come back when the resources go.
    #[test]
    fn pooled_buffers_are_placed_zeroed_and_given_back() {
        let test = "host::buffers::tests::pooled_buffers_are_placed_zeroed_and_given_back";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        if !enabled() {
            return;
        }
        // A pool of the test's own, without a reserve thread: every heap is
        // made when a buffer waits for room (none here: `room` would wait
        // for ever), so it is given heaps by hand.
        let pool: &'static Pool = Box::leak(Box::new(Pool {
            kind: Kind::System,
            device: host.device.clone(),
            state: Mutex::default(),
            changed: Condvar::new(),
            waited: Default::default(),
        }));
        let add = || {
            let heap = pool.make_heap(HEAP).expect("a heap");
            pool.lock().add(heap, HEAP);
        };
        add();
        let units = HEAP / ALIGN;
        let mut buffers: Vec<(ID3D12Resource, *mut u8)> = (0..units)
            .map(|i| {
                pool.place(if i % 2 == 0 { 256 } else { ALIGN })
                    .expect("a buffer")
            })
            .collect();
        // SAFETY: plain queries of live buffers.
        let mut vas: Vec<u64> = buffers
            .iter()
            .map(|(b, _)| unsafe { b.GetGPUVirtualAddress() })
            .collect();
        vas.sort_unstable();
        vas.dedup();
        assert_eq!(vas.len() as u64, units, "each its own range");
        assert!(vas.windows(2).all(|w| w[1] - w[0] >= ALIGN));
        assert_eq!(pool.lock().free(), 0);
        // Written, released, placed again: zeroed.
        let (b, data) = buffers.pop().unwrap();
        // SAFETY: the buffer's mapping, ALIGN bytes.
        unsafe { data.write_bytes(0xab, ALIGN as usize) };
        drop(b);
        assert_eq!(pool.lock().free(), ALIGN, "given back with its resource");
        let (again, data) = pool.place(ALIGN).expect("the range again");
        // SAFETY: the new buffer's mapping.
        let bytes = unsafe { std::slice::from_raw_parts(data, ALIGN as usize) };
        assert!(bytes.iter().all(|&b| b == 0), "zeroed");
        buffers.push((again, data));
        // A second heap; a buffer past a heap's size gets its own.
        add();
        let (big, _) = pool.place(HEAP + ALIGN).expect("a heap of its own");
        assert_eq!(pool.lock().heaps.len(), 3);
        drop(big);
        buffers.clear();
        let s = pool.lock();
        assert_eq!(s.free(), s.reserved(), "everything given back");
    }

    /// A thread waiting for room gets a heap from the reserve thread, and
    /// spare empty heaps are released once the buffers go (a pool and
    /// reserve thread of the test's own).
    #[test]
    fn the_reserve_thread_makes_room_and_releases_spare_heaps() {
        let test = "host::buffers::tests::the_reserve_thread_makes_room_and_releases_spare_heaps";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let pool: &'static Pool = Box::leak(Box::new(Pool {
            kind: Kind::Upload,
            device: host.device.clone(),
            state: Mutex::default(),
            changed: Condvar::new(),
            waited: Default::default(),
        }));
        std::thread::spawn(move || pool.reserve());
        // More than the reserve holds: most of them wait for heaps.
        let buffers: Vec<(ID3D12Resource, *mut u8)> = (0..3 * HEAP / ALIGN)
            .map(|_| pool.place(ALIGN).expect("an upload buffer"))
            .collect();
        assert!(pool.lock().heaps.len() >= 3);
        assert!(pool.waited[1].load(Ordering::Relaxed) > 0, "it waited");
        drop(buffers);
        let started = std::time::Instant::now();
        while pool.lock().reserved() > RESERVE + 2 * HEAP
            && started.elapsed() < std::time::Duration::from_secs(10)
        {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let s = pool.lock();
        assert!(s.reserved() <= RESERVE + 2 * HEAP, "spare heaps released");
        assert_eq!(s.free(), s.reserved());
    }

    /// The host's buffers are placed (they hold a lease), and
    /// `XTP_BUFFER_POOL=0` brings back committed ones: checked in a child
    /// process, where the switch is read afresh.
    #[test]
    fn committed_buffers_come_back_with_xtp_buffer_pool_0() {
        const CHILD: &str = "XTP_TEST_BUFFER_POOL_OFF";
        let test = "host::buffers::tests::committed_buffers_come_back_with_xtp_buffer_pool_0";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let (shared, data) = host.shared_buffer(256).expect("a shared buffer");
        let (upload, _) = host.upload_buffer(256).expect("an upload buffer");
        if std::env::var_os(CHILD).is_some() {
            assert!(POOLS.get().is_none(), "no pools");
            assert!(!has_lease(&shared) && !has_lease(&upload), "committed");
            // SAFETY: the buffer's mapping, 256 bytes.
            unsafe { data.write_bytes(7, 256) };
            return;
        }
        assert_eq!(
            (has_lease(&shared), has_lease(&upload)),
            (enabled(), enabled()),
            "placed"
        );
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args([test, "--exact", "--nocapture", "--test-threads=1"])
            .env(CHILD, "1")
            .env("XTP_BUFFER_POOL", "0")
            .output()
            .unwrap();
        let (stdout, stderr) = (
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
        assert!(out.status.success(), "the child: {stdout}{stderr}");
        assert!(stdout.contains("1 passed"), "{stdout}");
    }

    /// Runs `body` on the host device in a process of its own (the test
    /// binary again, with this test alone), where title buffers of 1 MiB
    /// or more get their copy in the card's memory, 3 MiB of them at most
    /// (`shadow::cards`; the limits are read once per process).
    fn with_cards(test: &str, body: impl FnOnce(&'static crate::host::Host)) {
        const CHILD: &str = "XTP_TEST_CARD_BUFFERS";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        if std::env::var_os(CHILD).is_some() {
            body(host);
            return;
        }
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args([test, "--exact", "--nocapture", "--test-threads=1"])
            .env(CHILD, "1")
            .env("XTP_CARD_BUFFERS", "1")
            .env("XTP_CARD_BUFFERS_MIN_MB", "1")
            .env("XTP_CARD_BUFFERS_MB", "3")
            .output()
            .unwrap();
        let (stdout, stderr) = (
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
        assert!(
            out.status.success() && stdout.contains("1 passed"),
            "{test} in a process of its own: {stdout}\n{stderr}"
        );
    }

    /// Byte offsets in a 2 MiB title buffer: what a test's passes read,
    /// what the first writes, where the second copies it (pages of their
    /// own, the last 256 bytes before a page's end).
    const SCRATCH: u64 = 0;
    const OUT: u64 = 1 << 16;
    const COPY: u64 = (3 << 16) - crate::testgpu::KERNEL_BYTES;

    /// The 64 words of title memory at byte offset `at` of `buffer`.
    fn words(buffer: &crate::testgpu::TitleBuffer, at: u64) -> Vec<u32> {
        buffer.words()[at as usize / 4..][..64].to_vec()
    }

    /// What shaders write to a copy in the card's memory goes back to title
    /// memory after the GPU ran, and only that: no snapshot is taken, the
    /// title's bytes around stay. A later pass reads the first's output on
    /// the card, and a GPU copy's destination goes back too (a copy within
    /// the buffer, staged in the card's memory, which the debug layer
    /// passes when on).
    #[test]
    fn a_shader_write_to_a_copy_on_the_card_goes_back() {
        use crate::testgpu::{Kernel, Pass, TitleBuffer, pattern};
        let test = "host::buffers::tests::a_shader_write_to_a_copy_on_the_card_goes_back";
        with_cards(test, |host| {
            let kernel = Kernel::new(host);
            let big = TitleBuffer::new(2 << 20);
            big.set_words(|i| 0x7e00_0000 + i as u32);
            let shadow = big.shadow();
            assert!(shadow.carded(), "a copy in the card's memory");
            let at = big.address();
            let (first, second) = (host.new_list().unwrap(), host.new_list().unwrap());
            kernel.dispatch(&first, Pass::Fill, at + SCRATCH, at + OUT);
            first.close();
            kernel.dispatch(&second, Pass::Copy, at + OUT, at + COPY);
            // A GPU copy of 256 bytes of the first's output, page 5: within
            // the buffer, so staged, in the card's memory too.
            let staged = || crate::shadow::CARD_STAGED.load(Ordering::Relaxed);
            let before = staged();
            assert!(crate::shadow::copy(&second, at + (5 << 16), at + OUT, 256));
            assert_eq!(staged(), before + 1, "staged on the card");
            second.close();
            host.execute(&[&first]);
            assert_eq!(shadow.snapshots(), 0, "no snapshot of a copy on the card");
            host.execute(&[&second]);
            host.drain();
            let want: Vec<u32> = (0..64).map(pattern).collect();
            assert_eq!(words(&big, OUT), want, "the shaders' write went back");
            assert_eq!(words(&big, COPY), want, "the second read it on the card");
            assert_eq!(words(&big, 5 << 16), want, "and so did the GPU copy");
            let title = big.words();
            for word in [OUT / 4 - 1, OUT / 4 + 64, COPY / 4 - 1, (5 << 14) + 64] {
                assert_eq!(
                    title[word as usize],
                    0x7e00_0000 + word as u32,
                    "word {word}"
                );
            }
        });
    }

    /// A title write into what shaders write, made while the GPU runs them,
    /// is kept (the three-way merge, with the copy's CPU side as its
    /// "before"), and the next pass reads it on the card; the shaders'
    /// other bytes go back.
    #[test]
    fn a_title_write_while_the_gpu_runs_is_kept_on_a_copy_on_the_card() {
        use crate::testgpu::{Kernel, Pass, TitleBuffer, pattern};
        let test =
            "host::buffers::tests::a_title_write_while_the_gpu_runs_is_kept_on_a_copy_on_the_card";
        with_cards(test, |host| {
            let kernel = Kernel::new(host);
            let big = TitleBuffer::new(2 << 20);
            assert!(big.shadow().carded());
            let at = big.address();
            let first = host.new_list().unwrap();
            kernel.dispatch(&first, Pass::Fill, at + SCRATCH, at + OUT);
            first.close();
            let mut gate = crate::host::submissions::Gate::close(host);
            host.execute(&[&first]);
            // The GPU holds the submission: the title writes word 5 of what
            // the shaders write, and the word past it.
            let word = |i: u64, value: u32| {
                // SAFETY: the buffer's own title memory.
                unsafe {
                    ((at + OUT) as *mut u32)
                        .add(i as usize)
                        .write_volatile(value)
                };
            };
            word(5, 0x5555_5555);
            word(64, 0x6666_6666);
            gate.open();
            host.drain();
            let mut want: Vec<u32> = (0..64).map(pattern).collect();
            want[5] = 0x5555_5555;
            assert_eq!(
                words(&big, OUT),
                want,
                "the title's word kept, the GPU's back"
            );
            assert_eq!(big.words()[(OUT / 4 + 64) as usize], 0x6666_6666);
            let second = host.new_list().unwrap();
            kernel.dispatch(&second, Pass::Copy, at + OUT, at + COPY);
            second.close();
            host.execute(&[&second]);
            host.drain();
            assert_eq!(words(&big, COPY), want, "the title's word reached the card");
        });
    }

    #[test]
    fn overlapping_card_copies_publish_each_submissions_own_words() {
        use crate::testgpu::TitleBuffer;
        let test =
            "host::buffers::tests::overlapping_card_copies_publish_each_submissions_own_words";
        with_cards(test, |host| {
            let big = TitleBuffer::new(2 << 20);
            let input = TitleBuffer::new(2 * crate::watch::PAGE);
            let at = big.address();
            let page = crate::watch::PAGE as u64;
            input.set_words(|i| 0x1000 + i as u32);
            assert!(big.shadow().carded());
            let [first, second] = std::array::from_fn(|_| host.new_list().unwrap());
            for (list, source) in [(&first, input.address()), (&second, input.address() + page)] {
                assert!(crate::shadow::copy(list, at + 2 * page, source, 256));
                list.close();
            }
            let mut gate = crate::host::submissions::Gate::close(host);
            host.execute(&[&first]);
            let (send, observed) = std::sync::mpsc::channel();
            host.after(move || {
                // SAFETY: the test owns the live title buffer through drain.
                send.send(unsafe { ((at + 2 * page) as *const u32).read_volatile() })
                    .unwrap();
            });
            std::thread::scope(|s| {
                s.spawn(move || {
                    std::thread::sleep(std::time::Duration::from_millis(80));
                    gate.open();
                });
                host.execute(&[&second]);
                assert_eq!(
                    observed.try_recv(),
                    Ok(0x1000),
                    "A was consumed before B could reuse its return bytes"
                );
            });
            host.drain();
            assert_eq!(words(&big, 2 * page), words(&input, page));
        });
    }

    /// One word the GPU changes in a piece of 64 KiB is found, wherever it
    /// is in the piece (each of the 256 threads of the card's compare takes
    /// other words of it: the shaders are declared to write the whole
    /// piece), and a pass that writes what the bytes held already leaves a
    /// title write made meanwhile.
    #[test]
    fn one_word_the_gpu_changes_is_found_anywhere_in_a_piece() {
        use crate::testgpu::{KERNEL_BYTES, Kernel, Pass, TitleBuffer, pattern};
        let test = "host::buffers::tests::one_word_the_gpu_changes_is_found_anywhere_in_a_piece";
        with_cards(test, |host| {
            let kernel = Kernel::new(host);
            let big = TitleBuffer::new(2 << 20);
            let shadow = big.shadow();
            assert!(shadow.carded());
            let at = big.address();
            let piece = at + (1 << 20);
            // Words at the piece's start, in its middle (across the 256th
            // step), at its end: each 256 bytes hold the fill's words but
            // one, which the fill changes.
            let places = [0, 0x10, 0x7ff0, 0x8100, (1 << 16) - KERNEL_BYTES];
            for (n, &place) in places.iter().enumerate() {
                let dst = piece + place;
                let odd = (n * 13) % 64;
                // SAFETY: the buffer's own title memory.
                unsafe {
                    for i in 0..64 {
                        let value = if i == odd { 0 } else { pattern(i) };
                        (dst as *mut u32).add(i).write_volatile(value);
                    }
                }
                let list = host.new_list().unwrap();
                kernel.dispatch(&list, Pass::Fill, at + SCRATCH, dst);
                shadow.gpu_writes(&list, piece, 1 << 16);
                list.close();
                let before = changed();
                host.execute(&[&list]);
                host.drain();
                let want: Vec<u32> = (0..64).map(pattern).collect();
                assert_eq!(words(&big, dst - at), want, "word {odd} at {place:#x}");
                assert_eq!(changed(), before + (1 << 16), "the piece found changed");
            }
            // The fill's words already at 16 KiB in the piece: the GPU
            // writes what they hold, and the title writes word 3 meanwhile.
            let dst = piece + 0x4000;
            // SAFETY: the buffer's own title memory.
            unsafe {
                for i in 0..64 {
                    (dst as *mut u32).add(i).write_volatile(pattern(i));
                }
            }
            let list = host.new_list().unwrap();
            kernel.dispatch(&list, Pass::Fill, at + SCRATCH, dst);
            list.close();
            let before = changed();
            let mut gate = crate::host::submissions::Gate::close(host);
            host.execute(&[&list]);
            // SAFETY: the buffer's own title memory.
            unsafe { (dst as *mut u32).add(3).write_volatile(7) };
            gate.open();
            host.drain();
            let mut want: Vec<u32> = (0..64).map(pattern).collect();
            want[3] = 7;
            assert_eq!(words(&big, dst - at), want);
            assert_eq!(changed(), before, "nothing the GPU changed");
        });
    }

    /// Bytes of pieces the card found changed so far in this process.
    fn changed() -> usize {
        crate::shadow::CARD_CHANGED.load(Ordering::Relaxed)
    }

    /// A title buffer past the card's budget for copies keeps its copy in
    /// system memory (snapshot and compare on the CPU, as before), next to
    /// one in the card's memory; both bring back what their shaders write.
    #[test]
    fn a_buffer_past_the_cards_budget_keeps_its_copy_in_system_memory() {
        use crate::testgpu::{Kernel, Pass, TitleBuffer, pattern};
        let test =
            "host::buffers::tests::a_buffer_past_the_cards_budget_keeps_its_copy_in_system_memory";
        with_cards(test, |host| {
            let kernel = Kernel::new(host);
            let (a, b) = (TitleBuffer::new(2 << 20), TitleBuffer::new(2 << 20));
            let (on_card, in_memory) = (a.shadow(), b.shadow());
            assert!(on_card.carded(), "the first fits in the budget of 3 MiB");
            assert!(!in_memory.carded(), "the second does not");
            let list = host.new_list().unwrap();
            for buffer in [&a, &b] {
                let at = buffer.address();
                kernel.dispatch(&list, Pass::Fill, at + SCRATCH, at + OUT);
            }
            list.close();
            host.execute(&[&list]);
            host.drain();
            assert_eq!(on_card.snapshots(), 0);
            assert_eq!(in_memory.snapshots(), 1, "the old way: a snapshot mirror");
            let want: Vec<u32> = (0..64).map(pattern).collect();
            assert_eq!(words(&a, OUT), want);
            assert_eq!(words(&b, OUT), want);
            // The budget is given back with the copy.
            drop((on_card, in_memory, list));
            drop((a, b));
            host.drain();
            let c = TitleBuffer::new(2 << 20);
            assert!(
                c.shadow().carded(),
                "room again once the first copy is gone"
            );
        });
    }
}

#[cfg(test)]
mod rt_budget_tests {
    use super::{
        RT_PENDING, RtReservation, UNREAD_RT_CAP, budget_reading, fits_rt, fits_rt_budget,
    };
    use std::sync::atomic::Ordering;

    #[test]
    fn rt_budget_limits_physical_growth_and_leaves_local_headroom() {
        let mib = 1 << 20;
        assert!(fits_rt_budget(496 * mib, 16 * mib, 8192 * mib, 0));
        assert!(!fits_rt_budget(497 * mib, 16 * mib, 8192 * mib, 0));
        assert!(!fits_rt_budget(0, 16 * mib, 1024 * mib, 945 * mib));
        assert!(!fits_rt_budget(0, 1, 0, 0));
    }

    #[test]
    fn rt_budget_keeps_the_last_reading_when_dxgi_fails() {
        let mut last = None;
        assert_eq!(budget_reading(None, &mut last), None, "no reading yet");
        assert_eq!(budget_reading(Some((8, 1)), &mut last), Some((8, 1)));
        assert_eq!(budget_reading(None, &mut last), Some((8, 1)), "not 0");
        assert_eq!(budget_reading(Some((9, 2)), &mut last), Some((9, 2)));
        assert_eq!(last, Some((9, 2)));
        // Before any reading, a fixed cap: not a budget of 0.
        let mib = 1 << 20;
        assert!(fits_rt(0, 16 * mib, None));
        assert!(fits_rt(UNREAD_RT_CAP - mib, mib, None));
        assert!(!fits_rt(UNREAD_RT_CAP - mib, 2 * mib, None));
        assert!(!fits_rt(0, 16 * mib, Some((0, 0))), "a reading of 0 is 0");
    }

    /// The resource is made outside the lock: an allocation made from
    /// inside another's `make` does not wait for it, and sees its bytes as
    /// pending.
    #[test]
    fn rt_budget_makes_resources_outside_its_lock() {
        let test = "host::buffers::rt_budget_tests::rt_budget_makes_resources_outside_its_lock";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let device = &gpu.host().device;
        let (inner, outer) = RtReservation::allocate(device, 4096, || {
            assert!(RT_PENDING.load(Ordering::Acquire) >= 4096);
            RtReservation::allocate(device, 4096, || Some(7))
        })
        .expect("room for 8 KiB");
        assert_eq!(inner.0, 7);
        drop((inner, outer));
        // A failed make gives its reservation back.
        assert!(RtReservation::allocate(device, 4096, || None::<()>).is_none());
    }
}
