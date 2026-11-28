//! Descriptor heaps: zeroed memory, with handles that are plain addresses
//! (CPU and GPU alike, as on the console).
//!
//! Descriptor sizes: views 32 bytes and samplers 16, the size of RDNA 2's
//! image and sampler descriptors, so a title that copies descriptors itself
//! moves whole ones. Render target and depth views are ours alone; 128 bytes
//! leaves room for whatever stride a title assumes.
//!
//! Optional heap sync reads copied records from dirty pages into snapshots.
//! Its background worker can scan ahead. Only submission rounds mirror the
//! snapshots to D3D12, with heap and view lifetimes held through the update.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::object::{self, DESCRIPTOR_HEAP, E_INVALIDARG, HRESULT, Iid, Object, Table, This, guid};
use crate::resource::{Memory, PAGEABLE_IIDS};

#[cfg(windows)]
pub(crate) mod host;

#[cfg(windows)]
use std::ops::Range;

const HEAP_IIDS: [Iid; 4] = [
    guid(
        0x8efb471d,
        0x616c,
        0x4f49,
        [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct HeapDesc {
    kind: u32,
    count: u32,
    flags: u32,
    nodes: u32,
}

/// Newer titles (BTP 6) pass the description followed by
/// memory of their own for the descriptors: an address and its size in
/// bytes, or a null address to have the heap allocate (inferred from the
/// title's call sites; older titles pass the plain description).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Placement {
    address: u64,
    size: u64,
}

#[repr(C)]
pub(crate) struct Heap {
    object: Object,
    desc: HeapDesc,
    /// The first descriptor: in `memory`, or in the title's own memory.
    pub(crate) start: u64,
    memory: Option<Memory>,
    /// Shader-visible view and sampler heaps get a host heap with the same
    /// descriptor order when a draw first uses them (decision 0007, step
    /// 3); slots written before wait in `pending`.
    #[cfg(windows)]
    host: std::sync::OnceLock<Option<host::HostHeap>>,
    #[cfg(windows)]
    pending: std::sync::Mutex<Vec<u32>>,
    /// The heap's life, for work that runs after the recordings that use
    /// it ([`HeapHold`]).
    #[cfg(windows)]
    life: std::sync::Arc<Life>,
    /// Shader-visible heaps placed in the title's memory, in a run that
    /// looks for the records the title copies into them as bytes: what that
    /// look keeps ([`sync_heaps`]).
    #[cfg(windows)]
    watched: Option<Watch>,
}

/// Whether a title heap is still there, for work that outlives the
/// recordings that use it: the sweeps a list runs as it closes and at its
/// submissions (`draw.rs` `picked_buffers`).
#[cfg(windows)]
struct Life {
    /// The heap's address while the title holds it; 0 once its release has
    /// begun. Work on the heap reads it locked, and the heap's drop locks it
    /// for writing first: a release waits for that work to end, and work
    /// after it finds 0.
    heap: std::sync::RwLock<usize>,
}

/// A hold on one title heap, this one and no other: a heap made later at
/// the same address has a life of its own (Codex review of a090f22, P1:
/// the work a list runs as it closes kept the heap's address, checked it
/// against the heaps registered, then used it without a lock, so a
/// concurrent last release could free it between the two, and a heap made
/// at the same address since passed the check with the old one's start).
#[cfg(windows)]
#[derive(Clone)]
pub(crate) struct HeapHold(std::sync::Arc<Life>);

/// Holds on the same heap are equal (whatever its address holds now).
#[cfg(windows)]
impl PartialEq for HeapHold {
    fn eq(&self, other: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.0, &other.0)
    }
}

#[cfg(windows)]
impl Eq for HeapHold {}

#[cfg(windows)]
impl std::fmt::Debug for HeapHold {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HeapHold({:p})", std::sync::Arc::as_ptr(&self.0))
    }
}

#[cfg(windows)]
impl HeapHold {
    /// Runs `f` with the heap if the title has not released it, which keeps
    /// it until `f` returns: its release waits. `f` must not release a
    /// title heap.
    pub(crate) fn with<R>(&self, f: impl FnOnce(&Heap) -> R) -> Option<R> {
        let at = self.0.heap.read().unwrap_or_else(|e| e.into_inner());
        if *at == 0 {
            return None;
        }
        // SAFETY: the heap's drop sets 0 under the write lock before any of
        // it goes, and the read lock held until `f` returns keeps it from
        // doing so meanwhile.
        Some(f(unsafe { &*(*at as *const Heap) }))
    }
}

pub(crate) fn heap<'a>(this: This) -> &'a Heap {
    // SAFETY: called from descriptor heap methods only.
    unsafe { &*this.cast::<Heap>() }
}

impl Heap {
    /// Whether this is a sampler heap.
    pub(crate) fn samplers(&self) -> bool {
        self.desc.kind == SAMPLERS
    }

    /// The host heap, made on first use from what the title wrote so far.
    /// Made late because a host view heap the size of BTP 6's
    /// costs over 100 MB of commit, which its boot needs for its page pool.
    #[cfg(windows)]
    pub(crate) fn host(&self) -> Option<&host::HostHeap> {
        if let Some(h) = self.host.get() {
            return h.as_ref();
        }
        if self.desc.flags & SHADER_VISIBLE == 0 || !matches!(self.desc.kind, VIEWS_KIND | SAMPLERS)
        {
            return None;
        }
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        // A recorder can take the fast path as soon as the OnceLock is
        // initialized. Publish only after replaying the queued writes.
        self.host
            .get_or_init(|| {
                let step = u64::from(increment(self.desc.kind)?);
                let h = host::HostHeap::new(self.samplers(), self.desc.count)?;
                for i in pending.drain(..) {
                    h.mirror(i, self.start + u64::from(i) * step, false);
                }
                Some(h)
            })
            .as_ref()
    }

    /// A hold on this heap ([`HeapHold`]).
    #[cfg(windows)]
    pub(crate) fn hold(&self) -> HeapHold {
        HeapHold(self.life.clone())
    }

    /// The descriptor index of `handle` in this heap, if it lies in it.
    pub(crate) fn index(&self, handle: u64) -> Option<u32> {
        let step = u64::from(increment(self.desc.kind)?);
        let at = handle.checked_sub(self.start)?;
        let index = at / step;
        (index < u64::from(self.desc.count)).then_some(index as u32)
    }
}

const VIEWS_KIND: u32 = 0;
const SAMPLERS: u32 = 1;
const SHADER_VISIBLE: u32 = 1;

/// Shader-visible heaps by address range, for mirroring descriptor writes:
/// start, end, heap, and a hold on it. Work that finds a heap here and may
/// run while the title releases it (giving a list's slots back, the frame
/// line's count, the look for records copied in as bytes) uses it through
/// the hold, taken under this lock and used after it: never by its address
/// alone, whose last release may be taking its box back meanwhile
/// (stride4c, Codex review of stride4b, P1: the entry goes only in the
/// heap's drop, once the box is rebuilt).
#[cfg(windows)]
static SHADER_HEAPS: std::sync::Mutex<Vec<(u64, u64, usize, HeapHold)>> =
    std::sync::Mutex::new(Vec::new());

/// Holds on the registered heaps `keep` takes by address, taken under the
/// registry's lock ([`SHADER_HEAPS`]) and used after it.
#[cfg(windows)]
fn registered(keep: impl Fn(usize) -> bool) -> Vec<(usize, HeapHold)> {
    let heaps = SHADER_HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    heaps
        .iter()
        .filter(|e| keep(e.2))
        .map(|e| (e.2, e.3.clone()))
        .collect()
}

/// The shader-visible heap holding `handle`, and the descriptor's index.
#[cfg(windows)]
fn shader_slot<'a>(handle: u64) -> Option<(&'a Heap, u32)> {
    let heaps = SHADER_HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    let &(_, _, at, _) = heaps
        .iter()
        .find(|&&(start, end, ..)| (start..end).contains(&handle))?;
    // SAFETY: registered heaps are live (removed when dropped).
    let heap = unsafe { &*(at as *const Heap) };
    Some((heap, heap.index(handle)?))
}

/// Makes the host descriptor for what the title wrote at `handle` through
/// a call, if it is in a shader-visible heap (`slot`: [`shader_slot`]).
#[cfg(windows)]
fn mirror(slot: Option<(&Heap, u32)>, handle: u64) {
    let Some((heap, index)) = slot else {
        return;
    };
    let mut pending = heap.pending.lock().unwrap_or_else(|e| e.into_inner());
    match heap.host.get() {
        None => pending.push(index),
        Some(host) => {
            drop(pending);
            if let Some(host) = host {
                host.mirror(index, handle, false);
            }
        }
    }
}

/// The key of a record header (tag, view index, generation): the view index
/// and generation of a record of ours (never 0: generations start at 1), 0
/// for anything else. Two slots hold the same view when their keys match.
#[cfg(windows)]
pub(crate) fn record_key([tag, i, generation]: [u32; 3]) -> u64 {
    if tag == VIEW_TAG {
        u64::from(generation) << 32 | u64::from(i)
    } else {
        0
    }
}

/// Notes `key` as the record the title's call wrote at `slot` (a slot of a
/// shader-visible heap, [`shader_slot`]): called within the write of
/// [`RECORDS`] that wrote the record ([`Watch::noted`]).
#[cfg(windows)]
fn note_call(slot: Option<(&Heap, u32)>, key: u64) {
    if let Some((heap, index)) = slot
        && let Some(note) = heap
            .watched
            .as_ref()
            .and_then(|w| w.noted.get(index as usize))
    {
        note.store(key, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Whether [`sync_heaps`] looks for records copied in as bytes and mirrors
/// them: `XTP_HEAP_SYNC=1` does, `ab` in turns within the run
/// ([`xtp_core::ab`]: phase A not, B does). Off by default: counting alone
/// took 6.7 to 9.9 ms a frame in the world (session faces-1, Series S, 24
/// to 34 submissions a frame), and the mirror 12 to 35 ms a frame for 15,000
/// to 48,000 copied slots, with frames of 1.7 to 7.1 s at bursts of 68,000
/// (session faces2-on). The textures it queues for the pass over textures
/// shaders pick by index waited there for video memory (28,000 to 36,000
/// due, none uploaded), so no image was seen to gain yet.
#[cfg(windows)]
fn heap_sync() -> &'static xtp_core::ab::Switch<bool> {
    static SWITCH: std::sync::OnceLock<xtp_core::ab::Switch<bool>> = std::sync::OnceLock::new();
    SWITCH.get_or_init(|| xtp_core::ab::switch("XTP_HEAP_SYNC", false))
}

/// Whether the run looks for records copied in as bytes at all (in either
/// arm of an A/B; tests look themselves): placed heaps then keep what the
/// look needs ([`Watch`]), from their creation, so that the calls of a
/// phase that does not look are noted too. Nothing is kept otherwise.
#[cfg(windows)]
fn heap_sync_used() -> bool {
    use xtp_core::ab::Arm;
    heap_sync().at(Arm::A) || heap_sync().at(Arm::B)
}

/// A placed shader-visible heap's part in the look for the records the
/// title copies into it as bytes ([`sync_heaps`]).
#[cfg(windows)]
struct Watch {
    /// The record ([`record_key`]) each slot was last noted with: by the
    /// title's call that wrote it (view creation, descriptor copies), in
    /// the same write of [`RECORDS`] as the record itself, or by the look
    /// that found it, which reads slots and notes under that lock. A slot
    /// whose record is not its note was copied in as bytes since.
    /// Codex review of 7c64b95: the call noted its record under the host
    /// heap's lock after writing it, and the look noted what it had read
    /// once its read was over, so either could come between the other's
    /// two steps (a call's record counted as copied, or a look's older
    /// record noted over it and both counted); and a call before the host
    /// heap was made queued the slot's index alone, so the queue's replay
    /// noted whatever the slot held by then, copied or not.
    noted: Box<[std::sync::atomic::AtomicU64]>,
    /// What the look keeps from one look to the next, one look at a time.
    look: std::sync::Mutex<Watched>,
}

#[cfg(windows)]
impl Watch {
    fn new(count: u32) -> Watch {
        Watch {
            noted: (0..count).map(|_| Default::default()).collect(),
            look: Default::default(),
        }
    }
}

/// What the look for records copied in as bytes ([`sync_heaps`]) keeps of a
/// placed heap from one look to the next.
#[cfg(windows)]
#[derive(Default)]
struct Watched {
    /// The lowest write-watch asking of the last look (0: none yet): a page
    /// stamped after it may have changed since
    /// ([`crate::watch::last_write`]).
    since: u64,
    /// The heap's memory as runs of blocks ([`crate::watch::PAGE`]) the
    /// write watch answers for, each asked as a whole: the whole heap at
    /// first. The watch decides for a range by its first block (`watch.rs`
    /// `with_writes` and `guard` look at it alone), so over memory it
    /// watches only in part it answers nothing at all; such a run is split
    /// in two until each part gets an answer or is one block.
    runs: Vec<(usize, usize)>,
    /// Blocks no answer comes for: neither write-watched nor a view the
    /// guards take. Counted, not looked at; asked again every
    /// [`UNWATCHED_AGAIN`] looks (the title may map memory there since).
    unwatched: std::collections::BTreeSet<usize>,
    /// Blocks holding the first byte of slots a look could not read (none
    /// of their memory mapped then): looked at again at each look until
    /// read, whatever the watch says. Before, a changed block whose slots
    /// one read failed was passed over until written again (Codex review
    /// of 7c64b95), with every slot of that read, and the read reached into
    /// the next block for the slot across their boundary.
    unread: std::collections::BTreeSet<usize>,
    /// Changes observed between submissions, coalesced by slot.
    pending: std::collections::BTreeMap<u32, u64>,
    /// Looks so far.
    looks: u64,
}

/// Looks after which the blocks of a placed heap that no write watch
/// answered for are asked about again ([`Watched::unwatched`]).
#[cfg(windows)]
const UNWATCHED_AGAIN: u64 = 256;

/// What looks for records copied in as bytes found ([`sync_heaps`]), summed.
#[cfg(windows)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Copies {
    /// Changed pages read, and the slots on them looked at.
    pub(crate) pages: u64,
    pub(crate) slots: u64,
    /// Slots holding a record the title copied in as bytes, with no view
    /// call nor descriptor copy, since the last look (each mirrored).
    /// Changes between looks: two copies that bring a slot back to the
    /// record it had are none (the host slot shows that record already).
    pub(crate) copied: u64,
    /// Of them, views of textures the title fills, and of those the ones a
    /// draw read through a table lately (draws make them: the others are
    /// read only by shaders that pick textures by index, if any does: not
    /// read through a table lately does not say that a shader reads them).
    pub(crate) textures: u64,
    pub(crate) tabled: u64,
    /// Buffer views, and of those the ones in heaps whose shaders pick
    /// buffers at run time (as the heap's last sweep says).
    pub(crate) buffers: u64,
    pub(crate) picked_buffers: u64,
    /// Samplers, other views of ours, and no view of ours.
    pub(crate) samplers: u64,
    pub(crate) other: u64,
    pub(crate) none: u64,
    /// Of the copied, those whose host slot was not made from their record
    /// when the look found them (null, or another view), which the mirror
    /// made (a buffer view null until a draw or a sweep makes it).
    pub(crate) unmade: u64,
    /// Slots on changed pages a look could not read (none of their memory
    /// mapped): looked at again at the next ([`Watched::unread`]).
    pub(crate) unread: u64,
    /// Blocks of placed heaps no write watch answers for (the most at one
    /// look, not summed).
    pub(crate) unwatched: u64,
}

#[cfg(windows)]
impl Copies {
    /// One slot the title copied a record into as bytes: what it holds,
    /// and whether its host slot was made from that record.
    fn add_slot(&mut self, kind: host::CopiedKind, made: bool) {
        use host::CopiedKind as K;
        self.copied += 1;
        self.unmade += u64::from(!made);
        match kind {
            K::Texture { tabled } => {
                self.textures += 1;
                self.tabled += u64::from(tabled);
            }
            K::Buffer { picked } => {
                self.buffers += 1;
                self.picked_buffers += u64::from(picked);
            }
            K::Sampler => self.samplers += 1,
            K::Other => self.other += 1,
            K::None => self.none += 1,
        }
    }

    /// The slots shaders may pick by index, an estimate: views of textures
    /// the title fills that no table read lately, and buffer views in heaps
    /// whose shaders pick buffers at run time.
    pub(crate) fn picked(&self) -> u64 {
        self.textures - self.tabled + self.picked_buffers
    }

    /// `other`'s counts added (blocks not watched: the most).
    fn add(&mut self, other: &Copies) {
        self.pages += other.pages;
        self.slots += other.slots;
        self.copied += other.copied;
        self.textures += other.textures;
        self.tabled += other.tabled;
        self.buffers += other.buffers;
        self.picked_buffers += other.picked_buffers;
        self.samplers += other.samplers;
        self.other += other.other;
        self.none += other.none;
        self.unmade += other.unmade;
        self.unread += other.unread;
        self.unwatched = self.unwatched.max(other.unwatched);
    }
}

/// Since the last frame line's count ([`take_synced`]): what
/// [`sync_heaps`] found, the microseconds it took, and the microseconds
/// submissions waited for the background thread ([`SyncedHeaps`]).
#[cfg(windows)]
static SYNCED: std::sync::Mutex<(Copies, u64, u64)> = std::sync::Mutex::new((
    Copies {
        pages: 0,
        slots: 0,
        copied: 0,
        textures: 0,
        tabled: 0,
        buffers: 0,
        picked_buffers: 0,
        samplers: 0,
        other: 0,
        none: 0,
        unmade: 0,
        unread: 0,
        unwatched: 0,
    },
    0,
    0,
));

/// Whether a submission's [`sync_heaps`] runs on a thread of its own beside
/// the rest of submission preparation (`XTP_HEAP_SYNC_ASIDE=0`: inline on
/// the submitting thread, as before). On by default whenever `XTP_HEAP_SYNC`
/// is active.
#[cfg(windows)]
fn sync_aside() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("XTP_HEAP_SYNC_ASIDE").is_none_or(|v| v != "0"))
}

/// How long the `xtp-heap-sync` thread waits between submissions before
/// scanning dirty pages (`XTP_HEAP_SYNC_EARLY=<ms>`, 4 by default, 0 off).
/// Only a requested round writes host descriptors after the submitter
/// drains in-flight work. The worker stops polling after 500 ms idle.
#[cfg(windows)]
fn sync_early() -> Option<std::time::Duration> {
    static EVERY: std::sync::OnceLock<Option<std::time::Duration>> = std::sync::OnceLock::new();
    *EVERY.get_or_init(|| {
        let ms = std::env::var("XTP_HEAP_SYNC_EARLY")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(4u64);
        (ms > 0).then(|| std::time::Duration::from_millis(ms))
    })
}

/// Worker threads used by parallel [`sync_heaps`] (`XTP_HEAP_SYNC_THREADS`,
/// 6 by default including the calling thread, clamped to `1..=16`; `1` runs
/// single-threaded).
#[cfg(windows)]
fn sync_threads() -> usize {
    static THREADS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *THREADS.get_or_init(|| {
        std::env::var("XTP_HEAP_SYNC_THREADS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(6usize)
            .clamp(1, 16)
    })
}

/// Reusable scoped worker pool for parallel descriptor heap sync.
#[cfg(windows)]
struct SyncPool {
    state: std::sync::Mutex<SyncPoolState>,
    more: std::sync::Condvar,
    idle: std::sync::Condvar,
    /// One batch at a time.
    turn: std::sync::Mutex<()>,
    threads: usize,
}

#[cfg(windows)]
#[derive(Default)]
struct SyncPoolState {
    work: Option<SyncWork>,
    next: usize,
    count: usize,
    running: usize,
    panicked: bool,
}

#[cfg(windows)]
#[derive(Clone, Copy)]
struct SyncWork(*const (dyn Fn(usize) + Sync));

// SAFETY: the closure is `Sync` so it may be called concurrently from any
// worker thread, and `SyncPool::each` keeps it alive until `state.work` is
// cleared and `state.running == 0` (enforced by a `Drop` guard on unwind).
#[cfg(windows)]
unsafe impl Send for SyncWork {}

#[cfg(windows)]
impl SyncPool {
    fn state(&self) -> std::sync::MutexGuard<'_, SyncPoolState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn take(state: &mut SyncPoolState) -> Option<(SyncWork, usize)> {
        let work = state.work?;
        if state.next >= state.count {
            return None;
        }
        let i = state.next;
        state.next += 1;
        Some((work, i))
    }

    fn empty(threads: usize) -> Self {
        SyncPool {
            state: std::sync::Mutex::default(),
            more: std::sync::Condvar::new(),
            idle: std::sync::Condvar::new(),
            turn: std::sync::Mutex::new(()),
            threads,
        }
    }

    fn new(threads: usize) -> &'static SyncPool {
        let pool: &'static SyncPool = Box::leak(Box::new(Self::empty(threads)));
        for i in 1..threads {
            let _ = std::thread::Builder::new()
                .name(format!("xtp-heap-sync-{i}"))
                .spawn(move || pool.help());
        }
        pool
    }

    fn help(&self) {
        let mut state = self.state();
        loop {
            let Some((work, i)) = Self::take(&mut state) else {
                state = self.more.wait(state).unwrap_or_else(|e| e.into_inner());
                continue;
            };
            state.running += 1;
            drop(state);
            // SAFETY: this thread is counted in `state.running`, so
            // `SyncPool::each` has not returned and `work.0` is still valid.
            let ran =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe { (*work.0)(i) }));
            state = self.state();
            state.running -= 1;
            state.panicked |= ran.is_err();
            if state.running == 0 {
                self.idle.notify_all();
            }
        }
    }

    /// Calls `f` with `0..count` across the pool's threads and this thread,
    /// returning once every call has finished.
    fn each(&self, count: usize, f: &(dyn Fn(usize) + Sync)) {
        if count < 2 || self.threads < 2 {
            (0..count).for_each(f);
            return;
        }
        let _turn = self.turn.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: the lifetime is erased only for the duration of this call;
        // `Finish`'s drop guard clears `state.work` and waits until
        // `state.running == 0` before `each` can return or unwind.
        let work = SyncWork(unsafe {
            std::mem::transmute::<
                *const (dyn Fn(usize) + Sync + '_),
                *const (dyn Fn(usize) + Sync + 'static),
            >(f)
        });
        struct Finish<'a>(&'a SyncPool);
        impl Drop for Finish<'_> {
            fn drop(&mut self) {
                let mut state = self.0.state();
                state.work = None;
                while state.running > 0 {
                    state = self.0.idle.wait(state).unwrap_or_else(|e| e.into_inner());
                }
            }
        }
        {
            let mut state = self.state();
            (state.work, state.next, state.count, state.panicked) = (Some(work), 0, count, false);
        }
        let finish = Finish(self);
        if count > self.threads {
            self.more.notify_all();
        } else {
            for _ in 1..count {
                self.more.notify_one();
            }
        }
        let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            loop {
                let next = Self::take(&mut self.state());
                let Some((_, i)) = next else { break };
                f(i);
            }
        }));
        drop(finish);
        let panicked = self.state().panicked;
        if let Err(panic) = ran {
            std::panic::resume_unwind(panic);
        }
        assert!(!panicked, "a heap sync worker panicked");
    }
}

/// Runs `f` over `0..count` on the descriptor heap sync worker pool.
#[cfg(windows)]
fn each_sync(count: usize, f: &(dyn Fn(usize) + Sync)) {
    if count < 2 {
        (0..count).for_each(f);
        return;
    }
    static POOL: std::sync::OnceLock<&'static SyncPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| SyncPool::new(sync_threads()))
        .each(count, f);
}

/// The `xtp-heap-sync` thread's state: rounds requested, rounds completed,
/// and whether a scan or mirror panicked. A partial pass can consume notes
/// and dirty bits, so a failed worker cannot approve later submissions.
#[cfg(windows)]
static SYNC_ROUNDS: (std::sync::Mutex<(u64, u64, bool)>, std::sync::Condvar) = (
    std::sync::Mutex::new((0, 0, false)),
    std::sync::Condvar::new(),
);

/// A [`sync_heaps`] round in flight on `xtp-heap-sync`; joined when dropped
/// (before `ExecuteCommandLists`). Holds `None` when sync is off or ran
/// inline.
#[cfg(windows)]
pub(crate) struct SyncedHeaps(Option<u64>);

/// Starts a [`sync_heaps`] round for a submission: off-thread on
/// `xtp-heap-sync` when `XTP_HEAP_SYNC_ASIDE` is on (the default), inline
/// when `XTP_HEAP_SYNC_ASIDE=0`, or a no-op when `XTP_HEAP_SYNC` is off.
#[cfg(windows)]
pub(crate) fn begin_sync_heaps(before_mirroring: impl FnOnce()) -> SyncedHeaps {
    if !heap_sync().get() {
        return SyncedHeaps(None);
    }
    before_mirroring();
    begin_sync_heaps_inner()
}

#[cfg(windows)]
fn begin_sync_heaps_inner() -> SyncedHeaps {
    if !sync_aside() {
        sync_heaps(false);
        return SyncedHeaps(None);
    }
    static STARTED: std::sync::Once = std::sync::Once::new();
    static THREAD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    STARTED.call_once(|| {
        let made = std::thread::Builder::new()
            .name("xtp-heap-sync".into())
            .spawn(|| {
                let rounds = || SYNC_ROUNDS.0.lock().unwrap_or_else(|e| e.into_inner());

                let mut last = std::time::Instant::now();
                loop {
                    let mut r = rounds();
                    let mut asked = r.0 != r.1;
                    while !asked {
                        let remaining =
                            std::time::Duration::from_millis(500).saturating_sub(last.elapsed());
                        let Some(every) = sync_early()
                            .filter(|_| !r.2 && !remaining.is_zero() && heap_sync().get())
                            .map(|every| every.min(remaining))
                        else {
                            r = SYNC_ROUNDS.1.wait(r).unwrap_or_else(|e| e.into_inner());
                            asked = r.0 != r.1;
                            continue;
                        };
                        let (back, over) = SYNC_ROUNDS
                            .1
                            .wait_timeout(r, every)
                            .unwrap_or_else(|e| e.into_inner());
                        r = back;
                        asked = r.0 != r.1;
                        if over.timed_out() {
                            break;
                        }
                    }
                    let already_failed = r.2;
                    drop(r);
                    if asked {
                        last = std::time::Instant::now();
                        let failed = already_failed
                            || std::panic::catch_unwind(|| sync_heaps(false)).is_err();
                        let mut r = rounds();
                        r.1 += 1;
                        r.2 = failed;
                        SYNC_ROUNDS.1.notify_all();
                    } else if !already_failed
                        && heap_sync().get()
                        && std::panic::catch_unwind(|| sync_heaps(true)).is_err()
                    {
                        rounds().2 = true;
                        tracing::error!("early descriptor heap scan failed");
                    }
                }
            });
        THREAD.store(made.is_ok(), std::sync::atomic::Ordering::Release);
    });
    if !THREAD.load(std::sync::atomic::Ordering::Acquire) {
        tracing::warn!("could not start the descriptor heap sync thread, running inline");
        sync_heaps(false);
        return SyncedHeaps(None);
    }
    let mut rounds = SYNC_ROUNDS.0.lock().unwrap_or_else(|e| e.into_inner());
    rounds.0 += 1;
    let mine = rounds.0;
    drop(rounds);
    SYNC_ROUNDS.1.notify_all();
    SyncedHeaps(Some(mine))
}

#[cfg(windows)]
impl SyncedHeaps {
    /// Waits for a successful mirror before the submission can use it.
    pub(crate) fn finish(mut self) {
        assert!(self.wait(), "descriptor heap sync failed");
    }

    fn wait(&mut self) -> bool {
        let Some(mine) = self.0.take() else {
            return true;
        };
        let began = std::time::Instant::now();
        let mut rounds = SYNC_ROUNDS.0.lock().unwrap_or_else(|e| e.into_inner());
        while rounds.1 < mine {
            rounds = SYNC_ROUNDS
                .1
                .wait(rounds)
                .unwrap_or_else(|e| e.into_inner());
        }
        let succeeded = !rounds.2;
        drop(rounds);
        SYNCED.lock().unwrap_or_else(|e| e.into_inner()).2 += began.elapsed().as_micros() as u64;
        succeeded
    }
}

#[cfg(windows)]
impl Drop for SyncedHeaps {
    fn drop(&mut self) {
        // Join during unwinding as well, without starting a second panic.
        self.wait();
    }
}

/// Heaps the tests look at themselves ([`Heap::look_for_copies`]), which
/// [`sync_heaps`] on other tests' threads leaves alone.
#[cfg(all(test, windows))]
static LOOKED_AT_BY_TESTS: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());

#[cfg(all(test, windows))]
struct BeforeLook {
    heap: usize,
    pause: Box<dyn FnOnce() + Send>,
}

#[cfg(all(test, windows))]
static BEFORE_LOOK: std::sync::Mutex<Option<BeforeLook>> = std::sync::Mutex::new(None);

/// Mirrors the descriptor records the title copied into its placed
/// shader-visible heaps as bytes (stores or memcpy, which no `Create*View`
/// nor `CopyDescriptors` shows), before each title submission: the pages the
/// write watch (the guards, for view memory) reports changed since the last
/// look are read through the kernel (a page the title unmapped meanwhile
/// fails the read instead of faulting), and each slot whose record is not
/// the one last noted there ([`Watch::noted`]: by the title's call that
/// wrote it, or by the look that found it) is made on the host as a call's
/// would be ([`host::HostHeap::mirror`]: a texture view at once and queued
/// for the pass over textures shaders pick by index, a buffer view null
/// until a draw or a sweep makes it), and counted once, with what it holds
/// ([`Copies`], on the frame line: [`take_synced`]). Without it, shaders
/// that pick such a slot by index read null, or the view it showed before.
///
/// Ported from Marvisak/VIctory (86a5ed6) by Marvisak: the mirror at each
/// submission of the slots on pages written since the last. Ported from
/// fxceless/victory-fork (a26859f) by Flyander (its PR #305) and fxceless:
/// the pages read through the kernel first, only the records that changed
/// mirrored, and the count on the frame line. What differs:
/// - a slot's record is compared by its key (the header XTP writes,
///   [`record_key`]) with the one last noted there, which the title's calls
///   note in the same write as the record: not by a fingerprint of its
///   bytes against the last look's, so a slot the title rewrites through a
///   call is not mirrored a second time, and a slot that is not mirrored
///   again holds the record its host slot was made from (VIctory mirrored
///   every slot of a written page, each a buffer view made null again);
/// - a slot draws read through tables lately is mirrored all the same
///   (9376fd6 left those to their draws whatever they held, and a shader
///   picking one by index read the view it showed before): one a draw made
///   from the same record is left as it is;
/// - a record across two pages is looked at from either page (a26859f left
///   it to the range holding its first byte, which fork #802 reports);
/// - the watch is asked per run of blocks it answers for (the fork asked
///   once for the heap, which over memory the watch answers for only in
///   part gave nothing at all: it decides by the first block);
/// - slots a read could not reach are read in smaller pieces, and those
///   still unread are looked at again ([`Watched::unread`]).
///
/// The mirror consumes kernel-read keys without rereading title memory
/// directly. API writes supersede captured keys under RECORDS. A raw
/// store racing the scan may be observed on the next write-watch poll.
/// With `XTP_HEAP_SYNC=1`, or `ab` in turns ([`heap_sync`]); off, nothing is
/// looked at.
#[cfg(windows)]
fn sync_heaps(scan_only: bool) {
    let started = std::time::Instant::now();
    // Held, not the registry's lock: a heap's release waits for its look,
    // and a heap made since at the same address is another.
    #[cfg(test)]
    let tests = LOOKED_AT_BY_TESTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    #[cfg(test)]
    let holds = registered(|at| !tests.contains(&at));
    #[cfg(not(test))]
    let holds = registered(|_| true);
    let mut found = Copies::default();
    for (_, hold) in holds {
        hold.with(|heap| {
            #[cfg(test)]
            {
                let pause = {
                    let mut hook = BEFORE_LOOK.lock().unwrap_or_else(|e| e.into_inner());
                    let address = std::ptr::from_ref(heap) as usize;
                    if hook.as_ref().is_some_and(|hook| hook.heap == address) {
                        hook.take()
                    } else {
                        None
                    }
                };
                if let Some(hook) = pause {
                    (hook.pause)();
                }
            }
            heap.look_for_copies(&mut found, scan_only);
        });
    }
    let mut synced = SYNCED.lock().unwrap_or_else(|e| e.into_inner());
    synced.0.add(&found);
    synced.1 += started.elapsed().as_micros() as u64;
}

/// The slots of a heap of `count` slots of `step` bytes from `start` that
/// have a byte in page `page` ([`crate::watch::PAGE`]): a record across the
/// page's start or end is looked at from either page.
#[cfg(windows)]
fn page_slots(page: usize, start: usize, step: usize, count: usize) -> std::ops::Range<usize> {
    use crate::watch::PAGE;
    let end = start + count * step;
    let Some(page_start) = page.checked_mul(PAGE) else {
        return 0..0;
    };
    let (from, to) = (
        page_start.max(start),
        page_start.saturating_add(PAGE).min(end),
    );
    if from >= to {
        return 0..0;
    }
    (from - start) / step..(to - start).div_ceil(step)
}

/// The pages of `[start, end)` that changed since `w`'s last look, asking
/// the write watch per run of blocks it answers for ([`Watched::runs`]).
#[cfg(windows)]
fn changed_pages(w: &mut Watched, start: usize, end: usize) -> Vec<usize> {
    use crate::watch::{PAGE, last_write, writes};
    w.looks += 1;
    if w.runs.is_empty() && w.unwatched.is_empty() {
        w.runs.push((start / PAGE, end.div_ceil(PAGE)));
    } else if w.looks.is_multiple_of(UNWATCHED_AGAIN) {
        let again = std::mem::take(&mut w.unwatched);
        w.runs
            .extend(again.into_iter().map(|block| (block, block + 1)));
    }
    let mut todo = std::mem::take(&mut w.runs);
    let mut lowest: Option<u64> = None;
    while let Some((first, last)) = todo.pop() {
        match writes((first * PAGE).max(start), (last * PAGE).min(end), true) {
            Some(asked) => {
                lowest = Some(lowest.map_or(asked, |l| l.min(asked)));
                w.runs.push((first, last));
            }
            None if last - first > 1 => {
                let middle = first + (last - first) / 2;
                todo.push((middle, last));
                todo.push((first, middle));
            }
            None => {
                w.unwatched.insert(first);
            }
        }
    }
    w.runs.sort_unstable();
    let Some(lowest) = lowest else {
        return Vec::new();
    };
    // The lowest asking: a page another asking stamped between this look's
    // own (which cleared its kernel bit) is not taken for seen.
    let since = std::mem::replace(&mut w.since, lowest);
    w.runs
        .iter()
        .flat_map(|&(first, last)| first..last)
        .filter(|&page| last_write(page) > since)
        .collect()
}

/// `slots` (of `step` bytes each from `start`) in pieces that a kernel
/// page (4 KiB) not mapped leaves whole: runs of slots within one kernel
/// page, and each slot across two on its own.
#[cfg(windows)]
fn kernel_pieces(start: usize, step: usize, slots: Range<usize>) -> Vec<Range<usize>> {
    const KERNEL_PAGE: usize = 4096;
    let page = |byte: usize| byte / KERNEL_PAGE;
    // The kernel page of a run's bytes, if they are in one.
    let within = |r: &Range<usize>| {
        let first = page(start + r.start * step);
        (first == page(start + r.end * step - 1)).then_some(first)
    };
    let mut pieces: Vec<Range<usize>> = Vec::new();
    for slot in slots {
        let one = slot..slot + 1;
        match (pieces.last_mut(), within(&one)) {
            (Some(last), Some(page)) if within(last) == Some(page) => last.end = slot + 1,
            _ => pieces.push(one),
        }
    }
    pieces
}

/// Pages [`Heap::look_for_copies`] gives a worker thread at a time.
#[cfg(windows)]
const SYNC_PAGES: usize = 8;

/// Records [`Heap::look_for_copies`] gives a worker thread at a time.
#[cfg(windows)]
const SYNC_AT_A_TIME: usize = 256;

/// What one worker chunk found on its pages of a heap during
/// [`Heap::look_for_copies`].
#[cfg(windows)]
#[derive(Default)]
struct SyncPart {
    found: Copies,
    unread: Vec<usize>,
    changed: Vec<(u32, u64)>,
}

/// Of slots `slots` of a placed heap from `start` (`step` bytes each),
/// appends those whose record is not the one noted there ([`Watch::noted`])
/// into `part.changed`, noting them now using [`host::changed_records`]'s
/// 16-record fast skip. Slots and notes are read in one shared read of
/// [`RECORDS`], so no call writes a record and its note meanwhile. A read
/// that fails (some of the memory not mapped) is retried in
/// [`kernel_pieces`], and unreadable slots are left to the next look
/// ([`Watched::unread`]).
#[cfg(windows)]
fn changed_slots_into(
    watch: &Watch,
    bytes: &mut Vec<u8>,
    (start, step): (usize, usize),
    slots: Range<usize>,
    part: &mut SyncPart,
) {
    bytes.resize(slots.len() * step, 0);
    let at = |slot: usize| start + slot * step;
    let into = |bytes: &mut Vec<u8>, piece: &Range<usize>| {
        let room = &mut bytes[(piece.start - slots.start) * step..(piece.end - slots.start) * step];
        crate::host::read_into(at(piece.start) as *const u8, room)
    };
    let _records = records_read();
    let whole = into(bytes, &slots);
    #[cfg(test)]
    if let Some(between) = LOOK_BETWEEN.with(|b| b.borrow_mut().take()) {
        between();
    }
    let mut read = |bytes: &[u8], piece: Range<usize>| {
        part.found.slots += piece.len() as u64;
        let offset = (piece.start - slots.start) * step;
        let len = piece.len() * step;
        host::changed_records(
            &watch.noted,
            &bytes[offset..offset + len],
            step,
            piece.start as u32,
            &mut part.changed,
        );
    };
    if whole {
        read(bytes, slots.clone());
        part.found.pages += 1;
    } else {
        let mut missed = 0;
        let mut any = false;
        for piece in kernel_pieces(start, step, slots.clone()) {
            if into(bytes, &piece) {
                read(bytes, piece);
                any = true;
            } else {
                missed += piece.len() as u64;
                part.unread
                    .extend(piece.map(|slot| at(slot) / crate::watch::PAGE));
            }
        }
        part.found.pages += u64::from(any);
        part.found.unread += missed;
    }
}

#[cfg(all(test, windows))]
thread_local! {
    /// Run once by the next look on this thread, between its read of a
    /// range of slots and its comparison with their notes, while it holds
    /// [`RECORDS`] (tests: a call publishing a record then).
    static LOOK_BETWEEN: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(windows)]
impl Heap {
    /// [`sync_heaps`]' look at this heap, if it is placed in the title's
    /// memory and has its host heap: the records copied in as bytes since
    /// the last look mirrored in parallel across the sync worker pool, and
    /// what it found added to `found`.
    fn look_for_copies(&self, found: &mut Copies, scan_only: bool) {
        let (Some(watch), Some(Some(host)), Some(step)) =
            (&self.watched, self.host.get(), increment(self.desc.kind))
        else {
            return;
        };
        let step = step as usize;
        let start = self.start as usize;
        let mut w = watch.look.lock().unwrap_or_else(|e| e.into_inner());
        let w = &mut *w;
        let mut pages = changed_pages(w, start, start + self.desc.count as usize * step);
        // Blocks with slots the last looks could not read: again.
        pages.extend(std::mem::take(&mut w.unread));
        pages.sort_unstable();
        pages.dedup();
        found.unwatched += w.unwatched.len() as u64;
        // Slots past the host heap (a capped one) have no host slot.
        let count = self.desc.count.min(host.count()) as usize;
        // Compute strictly disjoint slot ranges across changed pages before
        // fanning out: a record straddling two changed pages is checked once,
        // and no two worker threads ever touch the same slot or note.
        let mut ranges: Vec<Range<usize>> = Vec::with_capacity(pages.len());
        let mut next = 0;
        for page in pages {
            let slots = page_slots(page, start, step, count);
            let slots = slots.start.max(next)..slots.end;
            if slots.is_empty() {
                continue;
            }
            next = slots.end;
            ranges.push(slots);
        }
        let read_parts: Vec<std::sync::Mutex<SyncPart>> = ranges
            .chunks(SYNC_PAGES)
            .map(|_| Default::default())
            .collect();
        each_sync(read_parts.len(), &|n| {
            let mut out = SyncPart::default();
            let mut bytes = Vec::new();
            for slots in ranges.chunks(SYNC_PAGES).nth(n).unwrap_or_default() {
                changed_slots_into(watch, &mut bytes, (start, step), slots.clone(), &mut out);
            }
            *read_parts[n].lock().unwrap_or_else(|e| e.into_inner()) = out;
        });
        let mut changed: Vec<(u32, u64)> = Vec::new();
        for part in read_parts {
            let mut part = part.into_inner().unwrap_or_else(|e| e.into_inner());
            found.add(&part.found);
            w.unread.extend(part.unread);
            changed.append(&mut part.changed);
        }
        if scan_only {
            w.pending.extend(changed);
            return;
        }
        if !w.pending.is_empty() {
            let mut pending = std::mem::take(&mut w.pending);
            pending.extend(changed);
            changed = pending.into_iter().collect();
        }
        let mirror_parts: Vec<std::sync::Mutex<Copies>> = changed
            .chunks(SYNC_AT_A_TIME)
            .map(|_| Default::default())
            .collect();
        each_sync(mirror_parts.len(), &|n| {
            let mut local = Copies::default();
            for &(index, key) in changed.chunks(SYNC_AT_A_TIME).nth(n).unwrap_or_default() {
                let made = host.made_key(index);
                let Some(kind) = host.mirror_copied(index, key, &watch.noted[index as usize])
                else {
                    // Superseded by a newer API write before mirroring ran.
                    continue;
                };
                // In the frames listed for capture: each record mirrored,
                // what its host slot was made from before, and its view.
                #[cfg(feature = "diagnostics")]
                if crate::capture::listed(crate::frame::current()) {
                    tracing::debug!(
                        target: "xtp_gpu::heap_sync",
                        heap = format_args!("{start:#x}"),
                        slot = index,
                        record = format_args!("{key:#x}"),
                        before = format_args!("{made:#x}"),
                        ?kind,
                        "processed a record the title copied in as bytes"
                    );
                }
                local.add_slot(kind, key != 0 && made == key);
            }
            *mirror_parts[n].lock().unwrap_or_else(|e| e.into_inner()) = local;
        });
        for part in mirror_parts {
            found.add(&part.into_inner().unwrap_or_else(|e| e.into_inner()));
        }
    }
}

/// What [`sync_heaps`] found since the last call, for the frame line over
/// `frames` frames ([`HeapSyncLine::log`]). Nothing is looked at in a run
/// that does not look for records copied in as bytes.
#[cfg(windows)]
pub(crate) fn take_synced(frames: f64) -> HeapSyncLine {
    let (copies, micros, waited_micros) =
        std::mem::take(&mut *SYNCED.lock().unwrap_or_else(|e| e.into_inner()));
    let mut line = HeapSyncLine {
        copies,
        micros,
        waited_micros,
        unplaced: 0,
        past: 0,
        frames: frames.max(1.0),
    };
    if !heap_sync_used() {
        return line;
    }
    for (_, hold) in registered(|_| true) {
        hold.with(|heap| {
            let Some(Some(host)) = heap.host.get() else {
                return;
            };
            if heap.watched.is_none() {
                // In XTP's own memory: the title may copy records into it
                // as well, and no write watch shows it.
                line.unplaced += u64::from(heap.desc.count);
            } else {
                // Past the host heap (a capped one): no host slot to make.
                line.past += u64::from(heap.desc.count.saturating_sub(host.count()));
            }
        });
    }
    line
}

/// [`sync_heaps`]' part of the frame line ([`take_synced`]).
#[cfg(windows)]
pub(crate) struct HeapSyncLine {
    copies: Copies,
    micros: u64,
    waited_micros: u64,
    /// Slots of shader-visible heaps in XTP's memory, whose host heap is
    /// made, and slots of placed heaps past their host heap: not looked at.
    unplaced: u64,
    past: u64,
    frames: f64,
}

#[cfg(windows)]
impl HeapSyncLine {
    /// Logs it, per frame, unless the look is off for the whole run.
    pub(crate) fn log(&self) {
        use xtp_core::ab::Arm;
        if !heap_sync().at(Arm::A) && !heap_sync().at(Arm::B) {
            return;
        }
        let c = &self.copies;
        let per = |n: u64| n as f64 / self.frames;
        tracing::info!(
            "descriptor records the title copied into its placed heaps as bytes, with no view call \
             nor descriptor copy (mirrored, with XTP_HEAP_SYNC=1 or ab), per frame: \
             {:.1} changed heap pages, {:.0} slots looked at, {:.1} copied slots; of them, slots \
             shaders may pick by index {:.1} ({:.1} views of textures the title fills that no table \
             read lately, {:.1} buffer views in heaps whose shaders pick buffers at run time); \
             besides {:.1} textures read through tables, {:.1} buffer views, {:.1} samplers, {:.1} \
             other views, {:.1} no view of ours; {:.1} not what their host slot was made from; \
             {:.1} slots not mapped (again at the next look); {:.3} ms ({:.3} ms waited for); \
             {} blocks no write watch answers for, {} slots of heaps in XTP's memory and {} past \
             host heaps (not looked at)",
            per(c.pages),
            per(c.slots),
            per(c.copied),
            per(c.picked()),
            per(c.textures - c.tabled),
            per(c.picked_buffers),
            per(c.tabled),
            per(c.buffers - c.picked_buffers),
            per(c.samplers),
            per(c.other),
            per(c.none),
            per(c.unmade),
            per(c.unread),
            self.micros as f64 / 1000.0 / self.frames,
            self.waited_micros as f64 / 1000.0 / self.frames,
            c.unwatched,
            self.unplaced,
            self.past,
        );
    }
}

/// Gives back dynamic slots of the host heap behind the title heap at
/// `heap`, if that heap still exists: what a recording leases with
/// [`crate::host::HostList::keep_dynamic`], once it has gone. A
/// [`Recording`]'s lease (`heap` [`BATCH`]) gives back all it holds.
#[cfg(windows)]
pub(crate) fn release_dynamic(heap: usize, first: u32, n: u32) {
    if heap == BATCH {
        return batch_gone(u64::from(first) | u64::from(n) << 32);
    }
    // shortcut: a lease by address alone (`keep_dynamic` with a title
    // heap's, tests only now) is given back to the heap at that address,
    // whichever it is; recordings keep their heaps' own ([`Held`]).
    give_back(&mut Held {
        ranges: vec![(heap, first, n)],
        ..Held::default()
    });
}

/// The lease key of [`Recording`]s (no title heap is at this address).
#[cfg(windows)]
const BATCH: usize = usize::MAX - 1;

/// What one recording of a list holds: dynamic slots, holds on shared
/// views, and memory of the list's own its commands use (dropped with it).
/// It all goes back in one batch, per heap one lock, once both the list's
/// state is done with the recording (reset or released) and the recording
/// has gone: at the host list's reset, or, for a list reset while it runs,
/// once that submission has finished. A recording leases one entry of
/// [`crate::host::HostList::keep_dynamic`] for it all.
#[cfg(windows)]
#[derive(Default)]
pub(crate) struct Recording {
    /// Its batch, from the first thing it holds on.
    id: Option<u64>,
    held: Held,
}

/// A recording's batch: what its state handed over, and whether the state
/// is done with it and whether the recording has gone.
#[cfg(windows)]
#[derive(Default)]
struct Batch {
    held: Held,
    memory: Vec<Box<dyn Send>>,
    finished: bool,
    gone: bool,
}

#[cfg(windows)]
static BATCHES: std::sync::Mutex<Option<std::collections::HashMap<u64, Batch>>> =
    std::sync::Mutex::new(None);

#[cfg(windows)]
fn batches() -> std::sync::MutexGuard<'static, Option<std::collections::HashMap<u64, Batch>>> {
    BATCHES.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(windows)]
impl Recording {
    /// Leases the recording's batch in `list`, once.
    pub(crate) fn lease(&mut self, list: &crate::host::HostList) {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        if self.id.is_some() {
            return;
        }
        let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        batches()
            .get_or_insert_with(Default::default)
            .insert(id, Batch::default());
        list.keep_dynamic(BATCH, id as u32, (id >> 32) as u32);
        self.id = Some(id);
    }

    /// `n` dynamic slots from `first` of `heap`'s host heap, held until the
    /// recording has gone.
    pub(crate) fn range(&mut self, list: &crate::host::HostList, heap: usize, first: u32, n: u32) {
        self.lease(list);
        self.held.add_range(heap, first, n);
    }

    /// A hold on shared view `slot` of `heap`'s host heap (see
    /// [`host::HostHeap::shared_raw`]), given back once the recording has
    /// gone.
    pub(crate) fn shared(&mut self, list: &crate::host::HostList, heap: usize, slot: u32) {
        self.lease(list);
        self.held.add_shared(heap, slot);
    }

    /// Descriptor leases already held before attempting a bind.
    pub(crate) fn checkpoint(&self) -> (usize, usize) {
        (self.held.ranges.len(), self.held.shared.len())
    }

    /// Only descriptors acquired by a bind that issued no draw or dispatch.
    /// Copies and other commands already recorded retain their resources.
    pub(crate) fn reject(&mut self, (ranges, shared): (usize, usize)) {
        let held = &mut self.held;
        give_back(&mut Held {
            ranges: held.ranges.split_off(ranges),
            shared: held.shared.split_off(shared),
            range_heaps: held
                .range_heaps
                .split_off(ranges.min(held.range_heaps.len())),
            shared_heaps: held
                .shared_heaps
                .split_off(shared.min(held.shared_heaps.len())),
        });
    }

    /// The state is done with the recording: what it holds goes, now or
    /// once the recording has gone, with `memory`.
    pub(crate) fn finish(&mut self, memory: Box<dyn Send>) {
        let Some(id) = self.id.take() else {
            // Nothing held, nothing the recording's commands use.
            return;
        };
        let mut all = batches();
        let Some(batch) = all.as_mut().and_then(|b| b.get_mut(&id)) else {
            return;
        };
        batch.held.append(&mut self.held);
        batch.memory.push(memory);
        batch.finished = true;
        if batch.gone {
            let batch = all.as_mut().and_then(|b| b.remove(&id));
            drop(all);
            if let Some(mut batch) = batch {
                give_back(&mut batch.held);
            }
        }
    }

    /// What the recording holds (tests).
    #[cfg(test)]
    pub(crate) fn held(&self) -> &Held {
        &self.held
    }
}

/// Recording `id` has gone: its batch goes back once its state is done
/// with it.
#[cfg(windows)]
fn batch_gone(id: u64) {
    let mut all = batches();
    let Some(batch) = all.as_mut().and_then(|b| b.get_mut(&id)) else {
        return;
    };
    batch.gone = true;
    if batch.finished {
        let batch = all.as_mut().and_then(|b| b.remove(&id));
        drop(all);
        if let Some(mut batch) = batch {
            give_back(&mut batch.held);
        }
    }
}

/// What a list holds in host view heaps, by heap: the title heap's
/// address, or [`OWN_VIEW_HEAP`].
#[cfg(windows)]
#[derive(Default, Debug, PartialEq, Eq)]
pub(crate) struct Held {
    /// Dynamic slot ranges of its own: heap, first, count.
    pub(crate) ranges: Vec<(usize, u32, u32)>,
    /// Shared views it holds (see [`host::HostHeap::shared_raw`]): heap,
    /// slot.
    pub(crate) shared: Vec<(usize, u32)>,
    /// The title heap each of `ranges` and of `shared` was taken from (none
    /// for XTP's own view heap, or a lease by address alone): the address
    /// says which heap only until the title releases it, and a heap it
    /// makes at the same address since has slots of its own. Codex review
    /// of the port #1331 (task 69adac42, point 1): a list A held slot S of
    /// heap H1, the title made H2 at H1's address, a list B took S of H2,
    /// and A's give-back found H2 by the address and freed S, which a third
    /// list could then take while B read it.
    range_heaps: Vec<Option<HeapHold>>,
    shared_heaps: Vec<Option<HeapHold>>,
}

#[cfg(windows)]
impl Held {
    /// `n` dynamic slots from `first` of the host heap behind `heap`, which
    /// a recording uses (so live).
    fn add_range(&mut self, heap: usize, first: u32, n: u32) {
        self.ranges.push((heap, first, n));
        self.range_heaps.push(hold_of(heap));
    }

    /// Shared view `slot` of the host heap behind `heap`, live as above.
    fn add_shared(&mut self, heap: usize, slot: u32) {
        self.shared.push((heap, slot));
        self.shared_heaps.push(hold_of(heap));
    }

    /// All `other` holds, moved here.
    fn append(&mut self, other: &mut Held) {
        self.ranges.append(&mut other.ranges);
        self.shared.append(&mut other.shared);
        self.range_heaps.append(&mut other.range_heaps);
        self.shared_heaps.append(&mut other.shared_heaps);
    }
}

/// A hold on the title heap at `key`, live (a recording uses it); none for
/// XTP's own view heap and the keys that name no title heap.
#[cfg(windows)]
fn hold_of(key: usize) -> Option<HeapHold> {
    // SAFETY: the caller's recording uses the heap, which the title keeps
    // while its lists record with it.
    (key != 0 && key != OWN_VIEW_HEAP && key != BATCH).then(|| heap(key as This).hold())
}

/// The key of XTP's own view heap in [`Held`] (no title heap is at this
/// address).
#[cfg(windows)]
pub(crate) const OWN_VIEW_HEAP: usize = usize::MAX;

/// Dynamic slots in XTP's own view heap: root SRV tables of lists that
/// bind no view heap.
#[cfg(windows)]
const OWN_DYNAMIC: u32 = 1 << 14;

/// XTP's own view heap (made on first use): bound in the view slot of a
/// list that binds no hosted view heap of the title's, since a root
/// signature that indexes the heaps directly needs both slots bound, and
/// root SRVs are read through tables in it.
#[cfg(windows)]
pub(crate) fn own_view_heap() -> Option<&'static host::HostHeap> {
    static OWN: std::sync::OnceLock<Option<host::HostHeap>> = std::sync::OnceLock::new();
    OWN.get_or_init(|| host::HostHeap::own(OWN_DYNAMIC))
        .as_ref()
}

/// The key of XTP's own sampler heap (no title heap is at this address).
#[cfg(windows)]
pub(crate) const OWN_SAMPLER_HEAP: usize = usize::MAX - 1;

/// XTP's own sampler heap (made on first use), of null samplers: bound for
/// lists that bind no sampler heap of the title's, since a root signature
/// that indexes the sampler heap directly may only be set once one is
/// bound (D3D12 debug layer message 1419).
#[cfg(windows)]
pub(crate) fn own_sampler_heap() -> Option<&'static host::HostHeap> {
    static OWN: std::sync::OnceLock<Option<host::HostHeap>> = std::sync::OnceLock::new();
    OWN.get_or_init(host::HostHeap::own_samplers).as_ref()
}

/// In tests: the heaps to set before a root signature that indexes both
/// heaps directly (message 1419): `views` (XTP's own view heap if none),
/// with XTP's own sampler heap.
#[cfg(all(windows, test))]
pub(crate) fn heaps_with_own_samplers(
    views: Option<&windows::Win32::Graphics::Direct3D12::ID3D12DescriptorHeap>,
) -> [Option<windows::Win32::Graphics::Direct3D12::ID3D12DescriptorHeap>; 2] {
    let views = views
        .cloned()
        .or_else(|| own_view_heap().map(|h| h.heap.clone()));
    [views, own_sampler_heap().map(|h| h.heap.clone())]
}

/// Gives back what a list held, in one batch per heap: each host heap is
/// locked once, not once per range. Heaps the title released since are
/// passed over (their slots went with them), and so is a heap made since at
/// a released one's address: what was taken from the first goes back to
/// it alone ([`Held::range_heaps`]).
#[cfg(windows)]
pub(crate) fn give_back(held: &mut Held) {
    if held.ranges.is_empty() && held.shared.is_empty() {
        return;
    }
    // Usually one heap or two.
    let mut keys: Vec<usize> = held
        .ranges
        .iter()
        .map(|r| r.0)
        .chain(held.shared.iter().map(|s| s.0))
        .collect();
    keys.sort_unstable();
    keys.dedup();
    // Each heap held while its slots go back: its release waits, and finds
    // them given back; one released before is passed over.
    let holds = registered(|at| keys.contains(&at));
    for key in keys {
        let hold = holds.iter().find(|(at, _)| *at == key).map(|(_, h)| h);
        if key == OWN_VIEW_HEAP {
            if let Some(host) = own_view_heap() {
                give_back_to(held, key, host, None);
            }
        } else if let Some(hold) = hold {
            hold.with(|heap| {
                if let Some(Some(host)) = heap.host.get() {
                    give_back_to(held, key, host, Some(hold));
                }
            });
        }
    }
    held.ranges.clear();
    held.shared.clear();
    held.range_heaps.clear();
    held.shared_heaps.clear();
}

/// [`give_back`] of what `held` took from the host heap `host` behind
/// `key`, held by `life` (none for XTP's own view heap).
#[cfg(windows)]
fn give_back_to(held: &Held, key: usize, host: &host::HostHeap, life: Option<&HeapHold>) {
    // Taken from the heap at the address now, or of no heap known.
    let ours = |taken: Option<&Option<HeapHold>>| match (taken.and_then(Option::as_ref), life) {
        (Some(taken), Some(life)) => taken == life,
        _ => true,
    };
    let ranges: Vec<(u32, u32)> = held
        .ranges
        .iter()
        .enumerate()
        .filter(|&(i, r)| r.0 == key && ours(held.range_heaps.get(i)))
        .map(|(_, &(_, first, n))| (first, n))
        .collect();
    host.release_all(&ranges);
    let shared: Vec<u32> = held
        .shared
        .iter()
        .enumerate()
        .filter(|&(i, s)| s.0 == key && ours(held.shared_heaps.get(i)))
        .map(|(_, s)| s.1)
        .collect();
    host.give_back(&shared);
}

#[cfg(windows)]
impl Drop for Heap {
    fn drop(&mut self) {
        let at = std::ptr::from_ref::<Heap>(self) as usize;
        // First, before any of the heap goes: work holding it ends, and
        // work after finds it gone ([`HeapHold`]). Done already when the
        // title's last release drops it ([`release`]).
        if RELEASING.get() != at {
            *self.life.heap.write().unwrap_or_else(|e| e.into_inner()) = 0;
        }
        let mut heaps = SHADER_HEAPS.lock().unwrap_or_else(|e| e.into_inner());
        heaps.retain(|&(_, _, h, _)| h != at);
    }
}

/// Bytes per descriptor, by heap type (views, samplers, render targets,
/// depth).
fn increment(kind: u32) -> Option<u32> {
    [32, 16, 128, 128].get(kind as usize).copied()
}

/// Device method.
pub(crate) extern "system" fn increment_size(_: This, kind: u32) -> u32 {
    increment(kind).unwrap_or(0)
}

/// What our render target views hold: the resource and the view
/// description, turned into a host view when a command uses them.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RtvRecord {
    tag: u32,
    described: u32,
    pub(crate) resource: This,
    desc: [u32; 6],
}
const RTV_TAG: u32 = u32::from_le_bytes(*b"vRTV");
const _: () = assert!(size_of::<RtvRecord>() <= 112);

impl RtvRecord {
    /// The view description in the host's type (the same layout).
    #[cfg(windows)]
    pub(crate) fn desc(
        &self,
    ) -> Option<windows::Win32::Graphics::Direct3D12::D3D12_RENDER_TARGET_VIEW_DESC> {
        const TEXTURE_2D: u32 = 4;
        self.host_desc(TEXTURE_2D)
    }

    /// The depth view description in the host's type (the same layout).
    #[cfg(windows)]
    pub(crate) fn depth_desc(
        &self,
    ) -> Option<windows::Win32::Graphics::Direct3D12::D3D12_DEPTH_STENCIL_VIEW_DESC> {
        const TEXTURE_2D: u32 = 3;
        self.host_desc(TEXTURE_2D)
    }

    /// The description words as a host type, the format as the host has it.
    /// Without one, a typeless host copy gets a 2D view (`dimension`) in
    /// the title's format, which the host needs spelled out.
    #[cfg(windows)]
    fn host_desc<T>(&self, dimension: u32) -> Option<T> {
        assert!(size_of::<T>() <= size_of::<[u32; 6]>());
        let mut words = if self.described != 0 {
            self.desc
        } else if self.resource.is_null() {
            return None;
        } else {
            [
                crate::resource::view_format(self.resource)?,
                dimension,
                0,
                0,
                0,
                0,
            ]
        };
        words[0] = crate::resource::pc_format(words[0]);
        // SAFETY: both host view descriptions are 24 bytes in this layout.
        Some(unsafe { (&raw const words).cast::<T>().read_unaligned() })
    }
}

/// Device method: records the view at `handle` (descriptor memory).
pub(crate) extern "system" fn create_rtv(
    _: This,
    resource: This,
    desc: *const [u32; 6],
    handle: u64,
) {
    target_record(RTV_TAG, resource, desc, handle);
}

/// Device method: a depth view, recorded as render target views are.
pub(crate) extern "system" fn create_dsv(
    _: This,
    resource: This,
    desc: *const [u32; 6],
    handle: u64,
) {
    target_record(DSV_TAG, resource, desc, handle);
}

/// Records a render target or depth view (`tag`) at `handle`, in one write
/// under [`RECORDS`] (a depth view's tag was written over a render target
/// record's after it, so a reader could find the latter).
fn target_record(tag: u32, resource: This, desc: *const [u32; 6], handle: u64) {
    if handle == 0 {
        return;
    }
    // SAFETY: the title passes a description or null.
    let described = unsafe { desc.as_ref() };
    let record = RtvRecord {
        tag,
        described: u32::from(described.is_some()),
        resource,
        desc: described.copied().unwrap_or_default(),
    };
    let _records = records_written();
    // SAFETY: the handle is a descriptor slot, at least 112 bytes.
    unsafe { (handle as *mut RtvRecord).write_unaligned(record) };
}

/// The one lock of every access XTP makes to the records it keeps in
/// descriptor memory (Codex review of 244cf78, finding 7): view creation
/// writes a record's header (tag, view index, generation) under it, the
/// device's descriptor copies copy under it, and every read of a header
/// ([`header`], [`view`], render target and depth records) takes it, so no
/// read finds half of one record and half of another (two threads at the
/// same slot were a data race: `record` wrote under [`VIEWS`] only, and
/// copies took no lock at all). Readers share it; it is held for the
/// access alone, never while taking another lock.
/// shortcut: a title that copies descriptors with its own stores, not the
/// device's methods, takes no lock; what it leaves is read as before (a
/// record of another view, or none).
static RECORDS: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// [`RECORDS`] for a read.
fn records_read() -> std::sync::RwLockReadGuard<'static, ()> {
    RECORDS.read().unwrap_or_else(|e| e.into_inner())
}

/// [`RECORDS`] for a write.
fn records_written() -> std::sync::RwLockWriteGuard<'static, ()> {
    RECORDS.write().unwrap_or_else(|e| e.into_inner())
}

/// The record header at `handle` (tag, view index, generation), read under
/// [`RECORDS`]: one the title's view creation or descriptor copies wrote
/// whole, never part of one.
pub(crate) fn header(handle: u64) -> [u32; 3] {
    let _records = records_read();
    // SAFETY: the handle is a descriptor slot in title memory, at least 16
    // bytes.
    unsafe { (handle as *const [u32; 3]).read_unaligned() }
}
const DSV_TAG: u32 = u32::from_le_bytes(*b"vDSV");

/// A shader view or sampler as the title described it (descriptions in the
/// host's layout, kept as words), for the host to build its own from.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum View {
    /// Constant buffer: GPU address and size.
    Constant(u64, u32),
    /// Shader resource: resource, description (none: the resource's own).
    Shader(usize, Option<[u32; 10]>),
    /// Unordered access: resource, counter resource, description.
    Unordered(usize, usize, Option<[u32; 10]>),
    Sampler([u32; 13]),
    /// The console's own sampler description: flags, filters per axis,
    /// addressing, LOD and anisotropy settings, border color, then
    /// console-only tuning words.
    ConsoleSampler([u32; 27]),
    /// A view of memory without a resource object: whether it is unordered
    /// access, the resource description, then the view's (the plain one
    /// followed by the memory address and console fields).
    Placed(bool, [u32; 14], [u32; 16]),
}

impl View {
    /// Whether the view reads buffer memory (constant buffers, buffer views
    /// of resources or of memory) or memory the title placed a texture in:
    /// such views get host slots only for draws.
    #[cfg(windows)]
    fn is_buffer(&self) -> bool {
        match self {
            View::Constant(..) | View::Placed(..) => true,
            View::Shader(r, _) | View::Unordered(r, _, _) => {
                crate::resource::buffer_bytes(*r as This).is_some()
            }
            View::Sampler(_) | View::ConsoleSampler(_) => false,
        }
    }
}

/// Every distinct view in use; a descriptor holds a tag, an index into it
/// and the entry's generation, which survive titles copying descriptors as
/// bytes. A view of a resource lives as long as the resource: when the
/// title releases it, its entries are freed for reuse under a new
/// generation, so descriptors (and host slots made from them) naming the
/// old one go stale instead of naming whatever is made at the same address
/// next.
/// shortcut: views without a resource object (constant buffers, memory
/// views, samplers) are kept for the run; they are bounded by the title's
/// memory and sampler states.
struct Views {
    /// Each entry's view and generation.
    list: Vec<(View, u32)>,
    index: std::collections::BTreeMap<View, u32>,
    /// Entries naming each resource.
    by_resource: std::collections::BTreeMap<usize, Vec<u32>>,
    free: Vec<u32>,
}

#[cfg(windows)]
impl Views {
    fn live(&self, key: u64) -> Option<&View> {
        let index = key as u32;
        let (view, generation) = self.list.get(index as usize)?;
        (key != 0 && *generation == (key >> 32) as u32 && self.index.get(view) == Some(&index))
            .then_some(view)
    }
}

/// Read far more often than written: every slot a draw reads through a table
/// or `sync_heaps` checks is looked up here from several threads at once.
static VIEWS: std::sync::RwLock<Views> = std::sync::RwLock::new(Views {
    list: Vec::new(),
    index: std::collections::BTreeMap::new(),
    by_resource: std::collections::BTreeMap::new(),
    free: Vec::new(),
});
const VIEW_TAG: u32 = u32::from_le_bytes(*b"vVIE");

impl View {
    /// The resource objects the view names.
    fn resources(&self) -> [usize; 2] {
        match *self {
            View::Shader(r, _) => [r, 0],
            View::Unordered(r, counter, _) => [r, counter],
            _ => [0, 0],
        }
    }
}

fn record(handle: u64, view: View) {
    if handle == 0 {
        return;
    }
    // Found first: the slot's note is written with its record.
    #[cfg(windows)]
    let slot = shader_slot(handle);
    // Most views have already been seen; check under the shared read lock
    // first so concurrent recording and sync threads do not stall.
    let existing = VIEWS.read().unwrap_or_else(|e| e.into_inner());
    if let Some(&i) = existing.index.get(&view) {
        let generation = existing.list[i as usize].1;
        let _records = records_written();
        let header = [VIEW_TAG, i, generation];
        // SAFETY: the handle is a descriptor slot, at least 16 bytes.
        unsafe { (handle as *mut [u32; 3]).write_unaligned(header) };
        #[cfg(windows)]
        note_call(slot, record_key(header));
        drop(existing);
    } else {
        drop(existing);
        let mut views = VIEWS.write().unwrap_or_else(|e| e.into_inner());
        let i = match views.index.get(&view) {
            Some(&i) => i,
            None => {
                let i = match views.free.pop() {
                    Some(i) => {
                        views.list[i as usize].0 = view.clone();
                        i
                    }
                    None => {
                        views.list.push((view.clone(), 1));
                        views.list.len() as u32 - 1
                    }
                };
                for r in view.resources().into_iter().filter(|&r| r != 0) {
                    views.by_resource.entry(r).or_default().push(i);
                }
                views.index.insert(view, i);
                i
            }
        };
        let generation = views.list[i as usize].1;
        {
            let _records = records_written();
            let header = [VIEW_TAG, i, generation];
            // SAFETY: the handle is a descriptor slot, at least 16 bytes.
            unsafe { (handle as *mut [u32; 3]).write_unaligned(header) };
            #[cfg(windows)]
            note_call(slot, record_key(header));
        }
        drop(views);
    }
    #[cfg(windows)]
    mirror(slot, handle);
}

/// The title released `resource`: the views naming it go stale.
pub(crate) fn forget(resource: usize) {
    let mut views = VIEWS.write().unwrap_or_else(|e| e.into_inner());
    let Some(entries) = views.by_resource.remove(&resource) else {
        return;
    };
    for i in entries {
        let (view, generation) = views.list[i as usize].clone();
        // Entries freed through a view's other resource may hold another
        // view by now.
        if views.index.get(&view) != Some(&i) || !view.resources().contains(&resource) {
            continue;
        }
        views.index.remove(&view);
        views.list[i as usize].1 = generation.wrapping_add(1).max(1);
        // The view's other resource (a counter) keeps a stale entry, which
        // `forget` of that resource skips.
        views.free.push(i);
    }
}

/// The view of entry `i` if `generation` is still its own.
#[cfg(windows)]
fn view_at(i: u32, generation: u32) -> Option<View> {
    let views = VIEWS.read().unwrap_or_else(|e| e.into_inner());
    let (view, current) = views.list.get(i as usize)?;
    (*current == generation).then(|| view.clone())
}

/// The title resources backing a copied view, retained through host
/// descriptor creation. The final release invalidates VIEWS before freeing
/// the allocation, so a read of VIEWS can safely try to retain it.
#[cfg(windows)]
struct ViewPins([usize; 2]);

#[cfg(windows)]
impl Drop for ViewPins {
    fn drop(&mut self) {
        for resource in self.0.into_iter().filter(|&r| r != 0) {
            crate::resource::resource_release(resource as This);
        }
    }
}

#[cfg(windows)]
fn pin_view(key: u64) -> Option<(View, ViewPins)> {
    let mut pins = ViewPins([0; 2]);
    let views = VIEWS.read().unwrap_or_else(|e| e.into_inner());
    let view = views.live(key)?;
    for (pin, resource) in pins.0.iter_mut().zip(view.resources()) {
        if resource != 0 {
            // SAFETY: a live VIEWS entry is invalidated before final release
            // frees its resource, and the read lock prevents invalidation.
            if !object::add_ref_alive(resource as This) {
                // The read lock drops before releasing any successful pins.
                return None;
            }
            *pin = resource;
        }
    }
    Some((view.clone(), pins))
}

/// The view recorded at `handle`, if one is (and its resource is alive).
pub(crate) fn view(handle: u64) -> Option<View> {
    if handle == 0 {
        return None;
    }
    let [tag, i, generation] = header(handle);
    let views = VIEWS.read().unwrap_or_else(|e| e.into_inner());
    let (view, current) = views.list.get(i as usize)?;
    (tag == VIEW_TAG && *current == generation).then(|| view.clone())
}

/// A description of `N` words, if the title passes one.
fn words<const N: usize>(desc: *const [u32; N]) -> Option<[u32; N]> {
    // SAFETY: the title passes a description or null.
    (!desc.is_null()).then(|| unsafe { desc.read_unaligned() })
}

/// Device method.
/// Without a description it makes a null constant view, as D3D12's does:
/// `View::Constant(0, 0)`, which the host makes a null CBV (reads zeros),
/// not the view the slot held before.
pub(crate) extern "system" fn create_cbv(_: This, desc: *const [u32; 4], handle: u64) {
    let [lo, hi, size, _] = words(desc).unwrap_or_default();
    let address = u64::from(lo) | u64::from(hi) << 32;
    record(handle, View::Constant(address, size));
}

/// Device method.
pub(crate) extern "system" fn create_srv(
    _: This,
    resource: This,
    desc: *const [u32; 10],
    handle: u64,
) {
    record(handle, View::Shader(resource as usize, words(desc)));
}

/// Device method.
pub(crate) extern "system" fn create_uav(
    _: This,
    resource: This,
    counter: This,
    desc: *const [u32; 10],
    handle: u64,
) {
    let view = View::Unordered(resource as usize, counter as usize, words(desc));
    record(handle, view);
}

/// The words of a placed view's description with unused padding and union-tail
/// words set to zero. The first ten words are the host view description: for a
/// shader resource view, format, dimension, shader4 component mapping, one
/// padding word (the union starts on 8 bytes), and six union words; for an
/// unordered access view, format, dimension, and eight union words. Each
/// dimension uses only the head of the union; words 10..16 (address and
/// console fields) are preserved, and an unknown dimension keeps all words.
// Ported from TaynanGT/Vice-Tay commit ce914907 (GPL-3.0-or-later).
fn placed_view_words(desc: *const [u32; 16], unordered: bool) -> Option<[u32; 16]> {
    let mut d = words(desc)?;
    let (start, used) = if unordered {
        // D3D12_UAV_DIMENSION: buffer, 1D, 1D array, 2D, 2D array, 2D MS,
        // 2D MS array, 3D.
        const USED: [usize; 9] = [0, 7, 1, 3, 2, 4, 0, 2, 3];
        (2, USED.get(d[1] as usize).copied())
    } else {
        // D3D12_SRV_DIMENSION: buffer, 1D, 1D array, 2D, 2D array, 2D MS,
        // 2D MS array, 3D, cube, cube array, acceleration structure.
        const USED: [usize; 12] = [0, 5, 3, 5, 4, 6, 0, 2, 3, 3, 5, 2];
        (4, USED.get(d[1] as usize).copied())
    };
    if let Some(used) = used.filter(|_| d[1] != 0) {
        if !unordered {
            d[3] = 0;
        }
        d[start + used..10].fill(0);
    }
    Some(d)
}

fn record_placed(handle: u64, unordered: bool, resource: *const [u32; 14], desc: *const [u32; 16]) {
    if let (Some(resource), Some(desc)) = (words(resource), placed_view_words(desc, unordered)) {
        record(handle, View::Placed(unordered, resource, desc));
    }
}

/// Device methods 0x230 and 0x238: shader resource and unordered access
/// views of memory the title placed itself, `(resource desc, view desc,
/// handle)`.
pub(crate) extern "system" fn create_placed_srv(
    _: This,
    resource: *const [u32; 14],
    desc: *const [u32; 16],
    handle: u64,
) {
    record_placed(handle, false, resource, desc);
}

pub(crate) extern "system" fn create_placed_uav(
    _: This,
    resource: *const [u32; 14],
    desc: *const [u32; 16],
    handle: u64,
) {
    record_placed(handle, true, resource, desc);
}

/// Device method.
pub(crate) extern "system" fn create_sampler(_: This, desc: *const [u32; 13], handle: u64) {
    if let Some(desc) = words(desc) {
        record(handle, View::Sampler(desc));
    }
}

/// Device method 0x1e8.
pub(crate) extern "system" fn create_sampler_x(_: This, desc: *const [u32; 27], handle: u64) {
    if let Some(desc) = words(desc) {
        record(handle, View::ConsoleSampler(desc));
    }
}

/// Device method: `count` descriptors of `kind` from `src` to `dst`.
pub(crate) extern "system" fn copy_simple(_: This, count: u32, dst: u64, src: u64, kind: u32) {
    let step = u64::from(increment(kind).unwrap_or(0));
    let len = u64::from(count) * step;
    if dst != 0 && src != 0 {
        // Found first: the slots' notes are written with their records.
        #[cfg(windows)]
        let slots: Vec<_> = (0..u64::from(count))
            .map(|i| shader_slot(dst + i * step))
            .collect();
        {
            let _records = records_written();
            // SAFETY: both ranges are descriptor slots the title owns.
            unsafe { std::ptr::copy(src as *const u8, dst as *mut u8, len as usize) };
            #[cfg(windows)]
            for (i, &slot) in (0..).zip(&slots) {
                if slot.is_some() {
                    // SAFETY: a descriptor slot just written, at least 16
                    // bytes.
                    let header = unsafe { ((dst + i * step) as *const [u32; 3]).read_unaligned() };
                    note_call(slot, record_key(header));
                }
            }
        }
        #[cfg(windows)]
        for (i, slot) in (0..).zip(slots) {
            mirror(slot, dst + i * step);
        }
    }
}

/// Device method: ranges of descriptors (start and count each) gathered
/// from the sources and scattered to the destinations, in order.
#[allow(clippy::too_many_arguments)] // the method's own arguments
pub(crate) extern "system" fn copy(
    _: This,
    dst_ranges: u32,
    dst_starts: *const u64,
    dst_sizes: *const u32,
    src_ranges: u32,
    src_starts: *const u64,
    src_sizes: *const u32,
    kind: u32,
) {
    let Some(step) = increment(kind) else {
        return;
    };
    // Each range as (start, count); a null size array means one descriptor
    // per range.
    let ranges = |n: u32, starts: *const u64, sizes: *const u32| -> Vec<(u64, u32)> {
        (0..n as usize)
            .map(|i| {
                // SAFETY: the title passes `n` starts and sizes (or null sizes).
                unsafe {
                    (
                        *starts.add(i),
                        if sizes.is_null() { 1 } else { *sizes.add(i) },
                    )
                }
            })
            .collect()
    };
    if dst_starts.is_null() || src_starts.is_null() {
        return;
    }
    let src = ranges(src_ranges, src_starts, src_sizes);
    let mut dst = ranges(dst_ranges, dst_starts, dst_sizes).into_iter();
    let (mut to, mut room) = dst.next().unwrap_or((0, 0));
    for (mut from, mut left) in src {
        while left > 0 {
            while room == 0 {
                let Some(next) = dst.next() else { return };
                (to, room) = next;
            }
            let n = left.min(room);
            copy_simple(std::ptr::null(), n, to, from, kind);
            let bytes = u64::from(n) * u64::from(step);
            (to, from, room, left) = (to + bytes, from + bytes, room - n, left - n);
        }
    }
}

/// The view recorded at `handle`, if one is.
pub(crate) fn rtv_record(handle: u64) -> Option<RtvRecord> {
    tagged_record(handle, RTV_TAG)
}

/// The depth view recorded at `handle`, if one is.
pub(crate) fn dsv_record(handle: u64) -> Option<RtvRecord> {
    tagged_record(handle, DSV_TAG)
}

fn tagged_record(handle: u64, tag: u32) -> Option<RtvRecord> {
    if handle == 0 {
        return None;
    }
    let _records = records_read();
    // SAFETY: the handle is a descriptor slot the title passes back.
    let record = unsafe { (handle as *const RtvRecord).read_unaligned() };
    (record.tag == tag).then_some(record)
}

/// Device method.
pub(crate) extern "system" fn create_heap(
    _: This,
    desc_ptr: *const HeapDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc_ptr.as_ref() }) else {
        return E_INVALIDARG;
    };
    let Some(size) = increment(desc.kind).map(|i| u64::from(i) * u64::from(desc.count.max(1)))
    else {
        return E_INVALIDARG;
    };
    // SAFETY: read only as a hint: a title that passes the plain
    // description has its own stack or data after it.
    let placement = unsafe { desc_ptr.add(1).cast::<Placement>().read_unaligned() };
    // shortcut: a placement is recognised by its size matching the count,
    // since the plain and extended descriptions share the slot and the
    // interface version; find the flag that tells them apart if a title
    // trips over this.
    let placed = placement.address != 0 && placement.size == size;
    tracing::debug!(?desc, placed, "descriptor heap");
    let (start, memory) = if placed {
        (placement.address, None)
    } else {
        match Memory::zeroed(size, 64) {
            Ok(memory) => (memory.address(), Some(memory)),
            Err(hr) => return hr,
        }
    };
    let Some(end) = start
        .checked_add(size)
        .filter(|&end| end <= isize::MAX as u64)
    else {
        return E_INVALIDARG;
    };
    let visible = desc.flags & SHADER_VISIBLE != 0 && matches!(desc.kind, VIEWS_KIND | SAMPLERS);
    let heap = Box::new(Heap {
        object: Object::new(&FUNCTIONS),
        desc,
        start,
        memory,
        #[cfg(windows)]
        host: std::sync::OnceLock::new(),
        #[cfg(windows)]
        pending: Default::default(),
        #[cfg(windows)]
        life: std::sync::Arc::new(Life {
            heap: std::sync::RwLock::new(0),
        }),
        #[cfg(windows)]
        watched: (placed && visible && (cfg!(test) || heap_sync_used()))
            .then(|| Watch::new(desc.count)),
    });
    // The box does not move: its address is the heap's for its life.
    #[cfg(windows)]
    {
        *heap.life.heap.write().unwrap_or_else(|e| e.into_inner()) =
            std::ptr::from_ref::<Heap>(&heap) as usize;
    }
    #[cfg(windows)]
    if visible && !out.is_null() && object::is(iid, &HEAP_IIDS) {
        let at = std::ptr::from_ref::<Heap>(&heap) as usize;
        let mut heaps = SHADER_HEAPS.lock().unwrap_or_else(|e| e.into_inner());
        heaps.push((start, end, at, heap.hold()));
    }
    #[cfg(not(windows))]
    let _ = (visible, end);
    object::hand_out(DESCRIPTOR_HEAP, heap, iid, &HEAP_IIDS, out)
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let found = object::is(iid, &HEAP_IIDS).then_some(this);
    object::answer(DESCRIPTOR_HEAP, iid, found, out)
}
/// The title's release of a heap. Work holding the heap ([`HeapHold`])
/// ends, and the heap is marked gone for work after, before the last
/// release rebuilds its box: before, the heap's drop did that, once the box
/// was rebuilt, so the box and that work's `&Heap` were alive together
/// (stride4c, Codex review of stride4b, P1). A release that is not the last
/// leaves the heap as it was; work waiting meanwhile finds it there.
extern "system" fn release(this: This) -> u32 {
    #[cfg(windows)]
    {
        let life = heap(this).life.clone();
        let mut at = life.heap.write().unwrap_or_else(|e| e.into_inner());
        let address = std::mem::replace(&mut *at, 0);
        #[cfg(test)]
        if let Some(between) = RELEASE_BETWEEN.with(|b| b.borrow_mut().take()) {
            between();
        }
        let before = RELEASING.replace(this as usize);
        let left = object::release_boxed::<Heap>(this);
        RELEASING.set(before);
        if left != 0 {
            *at = address;
        }
        left
    }
    #[cfg(not(windows))]
    object::release_boxed::<Heap>(this)
}

#[cfg(windows)]
thread_local! {
    /// The heap whose release this thread runs ([`release`]), which holds
    /// the lock of its life for writing: the heap's drop leaves it alone.
    static RELEASING: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(all(test, windows))]
thread_local! {
    /// Run once by the next [`release`] on this thread, once the heap is
    /// marked gone and before the box may be rebuilt (tests: work asking
    /// for the heap then).
    static RELEASE_BETWEEN: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
}
extern "system" fn desc(this: This, out: *mut HeapDesc) {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a description.
        unsafe { out.write(heap(this).desc) };
    }
}
/// The same address for the CPU and the GPU start.
extern "system" fn start(this: This) -> u64 {
    heap(this).start
}

/// IUnknown 3, object 4, device child 1, heap 3.
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<DESCRIPTOR_HEAP, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = desc as *const (); // 0x40
    f[9] = start as *const (); // 0x48 CPU
    f[10] = start as *const (); // 0x50 GPU
    Table(f)
};

/// A shader-visible title view heap for tests, released when dropped.
#[cfg(all(test, windows))]
pub(crate) struct TestHeap(pub(crate) This);

#[cfg(all(test, windows))]
impl TestHeap {
    /// A heap of `count` view slots, made as the title makes one.
    pub(crate) fn views(count: u32) -> TestHeap {
        Self::made(VIEWS_KIND, count)
    }

    /// A heap of `count` sampler slots, made as the title makes one.
    pub(crate) fn samplers(count: u32) -> TestHeap {
        Self::made(SAMPLERS, count)
    }

    fn made(kind: u32, count: u32) -> TestHeap {
        /// What newer titles pass, so that reading the placement stays
        /// inside the value.
        #[repr(C)]
        struct Extended(HeapDesc, Placement);
        let extended = Extended(
            HeapDesc {
                kind,
                count,
                flags: SHADER_VISIBLE,
                nodes: 0,
            },
            Placement {
                address: 0,
                size: 0,
            },
        );
        let mut out = std::ptr::null();
        let made = create_heap(std::ptr::null(), &extended.0, &HEAP_IIDS[0], &mut out);
        assert_eq!(made, crate::object::S_OK, "a title heap");
        TestHeap(out)
    }
}

#[cfg(all(test, windows))]
impl Drop for TestHeap {
    fn drop(&mut self) {
        release(self.0);
    }
}

/// A shader-visible title view heap for tests placed in title memory the
/// runtime write-watches, as BTP 6 places its own, which only its test
/// looks at for records copied in as bytes ([`PlacedTestHeap::look`]);
/// released, and its memory freed, when dropped.
#[cfg(all(test, windows))]
pub(crate) struct PlacedTestHeap {
    pub(crate) this: This,
    memory: usize,
}

#[cfg(all(test, windows))]
impl PlacedTestHeap {
    /// A heap of `count` view slots (at most 64 KiB of them).
    pub(crate) fn views(count: u32) -> PlacedTestHeap {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        let size = count as usize * 32;
        assert!(size <= crate::watch::PAGE);
        // SAFETY: a fresh allocation of our own, freed when dropped.
        let memory = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                crate::watch::PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(memory, 0, "write-watched memory");
        #[repr(C)]
        struct Extended(HeapDesc, Placement);
        let extended = Extended(
            HeapDesc {
                kind: VIEWS_KIND,
                count,
                flags: SHADER_VISIBLE,
                nodes: 1,
            },
            Placement {
                address: memory as u64,
                size: size as u64,
            },
        );
        let mut this = std::ptr::null();
        let made = create_heap(std::ptr::null(), &extended.0, &HEAP_IIDS[0], &mut this);
        assert_eq!(made, crate::object::S_OK, "a placed title view heap");
        LOOKED_AT_BY_TESTS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(this as usize);
        PlacedTestHeap { this, memory }
    }

    /// One look for the records copied into it as bytes, as before a
    /// submission ([`sync_heaps`]): they are mirrored.
    pub(crate) fn look(&self) -> Copies {
        let mut found = Copies::default();
        heap(self.this).look_for_copies(&mut found, false);
        found
    }
}

#[cfg(all(test, windows))]
impl Drop for PlacedTestHeap {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
        LOOKED_AT_BY_TESTS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|&at| at != self.this as usize);
        release(self.this);
        // SAFETY: our own allocation, which the released heap no longer
        // uses.
        unsafe { VirtualFree(self.memory as *mut _, 0, MEM_RELEASE) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{S_OK, slot};

    /// What newer titles pass; tests use it throughout, so that reading the
    /// placement stays inside the value.
    #[repr(C)]
    struct Extended(HeapDesc, Placement);

    fn create(desc: HeapDesc, placement: Placement, out: &mut This) -> HRESULT {
        let extended = Extended(desc, placement);
        create_heap(std::ptr::null(), &extended.0, &HEAP_IIDS[0], out)
    }

    #[test]
    fn heaps_hold_count_descriptors_from_their_start() {
        let null = std::ptr::null();
        let desc = HeapDesc {
            kind: 0,
            count: 1000,
            flags: 1,
            nodes: 0,
        };
        let none = Placement {
            address: 0,
            size: 0,
        };
        let mut h = null;
        assert_eq!(create(desc, none, &mut h), S_OK);
        let cpu = slot::<extern "system" fn(This) -> u64>(h, 0x48)(h);
        assert_eq!(cpu % 64, 0);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x50)(h), cpu);
        assert_eq!(increment_size(null, 0), 32);
        // SAFETY: the heap holds 1000 descriptors of 32 bytes.
        unsafe { (cpu as *mut u8).add(999 * 32 + 31).write(1) };
        assert_eq!(increment_size(null, 9), 0);
        let bad = HeapDesc { kind: 9, ..desc };
        assert_eq!(create(bad, none, &mut h), E_INVALIDARG);
    }

    /// A constant view made without a description is a null one, as on
    /// D3D12: it replaces the view the slot held, which the host would
    /// otherwise go on reading.
    #[test]
    fn a_constant_view_without_a_description_is_a_null_one() {
        let null = std::ptr::null();
        let heap = [0u64; 4]; // one 32-byte slot
        let s0 = heap.as_ptr() as u64;
        create_cbv(null, &[0x2000, 1, 512, 0], s0);
        assert_eq!(view(s0), Some(View::Constant(0x1_0000_2000, 512)));
        create_cbv(null, std::ptr::null(), s0);
        assert_eq!(view(s0), Some(View::Constant(0, 0)));
    }

    #[test]
    fn views_survive_copies() {
        let null = std::ptr::null();
        let mut heap = vec![0u64; 4 * 8]; // four 32-byte slots
        let at = |i: usize| heap.as_ptr() as u64 + 32 * i as u64;
        let (s0, s1, s2, s3) = (at(0), at(1), at(2), at(3));
        create_cbv(null, &[0x1000, 1, 256, 0], s0);
        assert_eq!(view(s0), Some(View::Constant(0x1_0000_1000, 256)));
        create_srv(null, null, std::ptr::null(), s1);
        assert_eq!(view(s1), Some(View::Shader(0, None)));
        // The same view twice is one entry.
        create_srv(null, null, std::ptr::null(), s2);
        // SAFETY: both slots are in `heap`.
        assert_eq!(unsafe { *(s1 as *const [u32; 3]) }, unsafe {
            *(s2 as *const [u32; 3])
        });
        // Two ranges of one into one range of two.
        let (dst, dst_n, src, src_n) = ([s2], [2u32], [s0, s1], [1u32, 1]);
        copy(
            null,
            1,
            dst.as_ptr(),
            dst_n.as_ptr(),
            2,
            src.as_ptr(),
            src_n.as_ptr(),
            0,
        );
        assert_eq!(view(s2), view(s0));
        assert_eq!(view(s3), view(s1));
        assert_eq!(view(at(0) + 8), None);
        heap[0] = 0;
    }

    /// A released resource's views go stale; the same view of a new
    /// resource at the same address is a new record.
    #[test]
    fn views_die_with_their_resource() {
        let null = std::ptr::null();
        let slots = [0u64; 2 * 4];
        let (a, b) = (slots.as_ptr() as u64, slots.as_ptr() as u64 + 32);
        let resource = 0x7777_0000usize as This;
        create_srv(null, resource, std::ptr::null(), a);
        assert_eq!(view(a), Some(View::Shader(0x7777_0000, None)));
        forget(0x7777_0000);
        assert_eq!(view(a), None, "stale after release");
        create_srv(null, resource, std::ptr::null(), b);
        assert_eq!(view(b), Some(View::Shader(0x7777_0000, None)));
        // SAFETY: both slots are in `slots`.
        let (old, new) = unsafe { (*(a as *const [u32; 3]), *(b as *const [u32; 3])) };
        assert_ne!(old, new, "a new generation");
    }

    /// Codex review of a090f22, P1: the work a list runs as it closes kept
    /// the title heap's address, and took a heap made at that address after
    /// the title released the first for the first. A hold is of one heap:
    /// gone with it, whatever is made at its address after.
    #[cfg(windows)]
    #[test]
    fn a_hold_ends_with_its_heap_not_with_its_address() {
        let first = TestHeap::views(8);
        let at = first.0 as usize;
        let hold = heap(first.0).hold();
        assert_eq!(hold.with(|h| h.start), Some(heap(first.0).start));
        drop(first);
        assert_eq!(hold.with(|h| h.start), None, "released");
        // Heaps made until one takes the address (the allocator usually
        // gives it back at once), and a few more if none does.
        let mut later = Vec::new();
        for _ in 0..64 {
            let h = TestHeap::views(8);
            let same = h.0 as usize == at;
            later.push(h);
            if same {
                break;
            }
        }
        assert_eq!(hold.with(|h| h.start), None, "another heap");
        let last = later.last().expect("a heap");
        assert!(heap(last.0).hold().with(|_| ()).is_some());
    }

    /// Codex review of the port #1331 (task 69adac42, point 1): a slot a
    /// list took from a heap goes back to that heap alone. Given back after
    /// the title released the heap and made another at its address, it
    /// leaves the slot the new heap gave another list since taken.
    #[cfg(windows)]
    #[test]
    fn a_slot_goes_back_to_its_own_heap_not_to_one_made_at_its_address() {
        let test =
            "descriptor::tests::a_slot_goes_back_to_its_own_heap_not_to_one_made_at_its_address";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let first = TestHeap::views(8);
        let at = first.0 as usize;
        let host = heap(first.0).host().expect("a host heap");
        let slot = host.dynamic(1).expect("a dynamic slot");
        // List A's lease.
        let mut a = Held::default();
        a.add_range(at, slot, 1);
        drop(first);
        // Heaps made until one takes the address (the allocator usually
        // gives it back at once).
        let mut later = Vec::new();
        for _ in 0..64 {
            let h = TestHeap::views(8);
            let same = h.0 as usize == at;
            later.push(h);
            if same {
                break;
            }
        }
        let second = later.last().expect("a heap");
        if second.0 as usize != at {
            // No heap at the address: nothing to tell apart.
            eprintln!("{test}: no heap made at the released one's address; not exercised");
            return;
        }
        let host = heap(second.0).host().expect("a host heap");
        // List B's slot: the same one, of the new heap.
        assert_eq!(host.dynamic(1), Some(slot));
        give_back(&mut a);
        assert_eq!(a, Held::default());
        assert_ne!(host.dynamic(1), Some(slot), "B's slot stays taken");
        // B's own lease goes back to its heap.
        let mut b = Held::default();
        b.add_range(at, slot, 1);
        give_back(&mut b);
        assert_eq!(host.dynamic(1), Some(slot), "given back");
    }

    /// Codex review of a090f22, P1: the close-time work checked the heap
    /// under the registry's lock, let it go, then used the heap, which a
    /// concurrent last release could free between the two. A release now
    /// waits for the work that holds the heap.
    #[cfg(windows)]
    #[test]
    fn a_release_waits_for_work_that_holds_the_heap() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let made = TestHeap::views(8);
        let this = made.0 as usize;
        // The thread below releases it.
        std::mem::forget(made);
        let hold = heap(this as This).hold();
        let released = std::sync::Arc::new(AtomicBool::new(false));
        let releaser = {
            let released = released.clone();
            move || {
                release(this as This);
                released.store(true, Ordering::SeqCst);
            }
        };
        let mut thread = None;
        let start = hold.with(|h| {
            thread = Some(std::thread::spawn(releaser));
            std::thread::sleep(std::time::Duration::from_millis(200));
            assert!(!released.load(Ordering::SeqCst), "the release waits");
            h.start
        });
        assert!(start.is_some_and(|s| s != 0));
        thread
            .expect("a releasing thread")
            .join()
            .expect("released");
        assert!(released.load(Ordering::SeqCst));
        assert!(hold.with(|_| ()).is_none());
    }

    /// stride4c, Codex review of stride4b, P1: the last release rebuilt the
    /// heap's box before its drop marked it gone, while work holding the
    /// heap still had it. Work asking for the heap once a release has begun
    /// waits until the release knows whether it was the last: then it finds
    /// the heap gone (the box is rebuilt only after), or, for a release
    /// that was not the last, there as before.
    #[cfg(windows)]
    #[test]
    fn work_asking_during_a_release_finds_the_heap_gone_only_after_the_last() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let made = TestHeap::views(8);
        let this = made.0;
        // The last release below is the test's own.
        std::mem::forget(made);
        let start = heap(this).start;
        let hold = heap(this).hold();
        // Releases `this` while another thread asks for the heap: what it
        // found, and the release's count.
        let release_asked = |hold: HeapHold| {
            let asker = std::sync::Arc::new(std::sync::Mutex::new(None));
            let done = std::sync::Arc::new(AtomicBool::new(false));
            let (spawned, finished) = (asker.clone(), done.clone());
            RELEASE_BETWEEN.with(|b| {
                *b.borrow_mut() = Some(Box::new(move || {
                    let waited = finished.clone();
                    let thread = std::thread::spawn(move || {
                        let found = hold.with(|h| h.start);
                        waited.store(true, Ordering::SeqCst);
                        found
                    });
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    assert!(!finished.load(Ordering::SeqCst), "the work waits");
                    *spawned.lock().unwrap() = Some(thread);
                }));
            });
            let left = release(this);
            let thread = asker.lock().unwrap().take().expect("asked meanwhile");
            (thread.join().expect("the asking thread"), left)
        };
        object::add_ref(this);
        assert_eq!(
            release_asked(hold.clone()),
            (Some(start), 1),
            "not the last"
        );
        assert_eq!(release_asked(hold.clone()), (None, 0), "the last");
        assert!(hold.with(|_| ()).is_none());
    }

    /// stride4c, Codex review of stride4b, P1, the rest of it: work that
    /// finds a heap in the registry (giving a list's slots back, the frame
    /// line's count, the look for records copied in as bytes) used it by its
    /// address under the registry's lock, which the heap's drop takes only
    /// once its last release has rebuilt the box. It now holds the heap
    /// through the registry: slots given back while the last release runs
    /// wait for it, then find the heap gone and leave it alone.
    #[cfg(windows)]
    #[test]
    fn slots_given_back_during_the_last_release_wait_and_find_the_heap_gone() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let test = "descriptor::tests::slots_given_back_during_the_last_release_wait_and_find_the_heap_gone";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let made = TestHeap::views(8);
        let this = made.0;
        // The last release below is the test's own.
        std::mem::forget(made);
        let slot = heap(this).host().expect("a host heap").dynamic(1);
        let slot = slot.expect("a dynamic slot");
        let mut lease = Held::default();
        lease.add_range(this as usize, slot, 1);
        let giver = std::sync::Arc::new(std::sync::Mutex::new(None));
        let done = std::sync::Arc::new(AtomicBool::new(false));
        let (spawned, finished) = (giver.clone(), done.clone());
        RELEASE_BETWEEN.with(|b| {
            *b.borrow_mut() = Some(Box::new(move || {
                let waited = finished.clone();
                let thread = std::thread::spawn(move || {
                    give_back(&mut lease);
                    waited.store(true, Ordering::SeqCst);
                    lease
                });
                std::thread::sleep(std::time::Duration::from_millis(100));
                assert!(!finished.load(Ordering::SeqCst), "the give-back waits");
                *spawned.lock().unwrap() = Some(thread);
            }));
        });
        assert_eq!(release(this), 0, "the last release");
        let thread = giver.lock().unwrap().take().expect("given back meanwhile");
        let lease = thread.join().expect("the giving thread");
        assert!(done.load(Ordering::SeqCst));
        assert_eq!(lease, Held::default(), "emptied, the heap passed over");
    }

    /// A record across a page's start or end is looked at from either page;
    /// pages before and past the heap hold none of its slots.
    #[cfg(windows)]
    #[test]
    fn slots_of_a_changed_page_reach_across_its_ends() {
        use crate::watch::PAGE;
        // 16 bytes into a page: slot 2047 lies across the next page's start.
        let start = 5 * PAGE + 16;
        let count = 3 * PAGE / 32;
        assert_eq!(page_slots(5, start, 32, count), 0..2048);
        assert_eq!(page_slots(6, start, 32, count), 2047..4096);
        // The heap's last 16 bytes are on page 8.
        assert_eq!(page_slots(8, start, 32, count), 6143..6144);
        assert_eq!(page_slots(4, start, 32, count), 0..0);
        assert_eq!(page_slots(9, start, 32, count), 0..0);
        // Samplers, 16 bytes each, from a page's start.
        assert_eq!(page_slots(2, 2 * PAGE, 16, 8192), 0..4096);
        assert_eq!(page_slots(3, 2 * PAGE, 16, 8192), 4096..8192);
    }

    /// Title memory of `watched` write-watched blocks (as the runtime
    /// reserves the title's GPU memory), then `plain` blocks no write watch
    /// answers for at the addresses after them; freed when dropped.
    #[cfg(windows)]
    struct TitlePages {
        at: usize,
        parts: Vec<usize>,
    }

    #[cfg(windows)]
    impl TitlePages {
        fn new(watched: usize, plain: usize) -> TitlePages {
            use windows_sys::Win32::System::Memory::{
                MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
                VirtualFree,
            };
            const MEM_WRITE_WATCH: u32 = 0x20_0000;
            let page = crate::watch::PAGE;
            for _ in 0..64 {
                // SAFETY: allocations of our own, where the system finds
                // room, then at the addresses it gave (which another thread
                // may take meanwhile: then again).
                unsafe {
                    let room = VirtualAlloc(
                        std::ptr::null(),
                        (watched + plain) * page,
                        MEM_RESERVE,
                        PAGE_NOACCESS,
                    );
                    assert!(!room.is_null());
                    VirtualFree(room, 0, MEM_RELEASE);
                    let at = room as usize;
                    let first = VirtualAlloc(
                        room,
                        watched * page,
                        MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                        PAGE_READWRITE,
                    );
                    if first.is_null() {
                        continue;
                    }
                    let mut parts = vec![first as usize];
                    if plain > 0 {
                        let second = VirtualAlloc(
                            (at + watched * page) as *const _,
                            plain * page,
                            MEM_RESERVE | MEM_COMMIT,
                            PAGE_READWRITE,
                        );
                        if second.is_null() {
                            VirtualFree(first, 0, MEM_RELEASE);
                            continue;
                        }
                        parts.push(second as usize);
                    }
                    return TitlePages { at, parts };
                }
            }
            panic!("no room for the title's pages");
        }
    }

    #[cfg(windows)]
    impl Drop for TitlePages {
        fn drop(&mut self) {
            use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
            for &part in &self.parts {
                // SAFETY: our own allocations, no longer used.
                unsafe { VirtualFree(part as *mut _, 0, MEM_RELEASE) };
            }
        }
    }

    /// A shader-visible view heap of `count` slots the title placed at
    /// `address`, which [`sync_heaps`] on other tests' threads leaves to
    /// this test; [`release_placed`] releases it.
    #[cfg(windows)]
    fn placed_views(address: u64, count: u32) -> This {
        let desc = HeapDesc {
            kind: VIEWS_KIND,
            count,
            flags: SHADER_VISIBLE,
            nodes: 1,
        };
        let placement = Placement {
            address,
            size: u64::from(count) * 32,
        };
        let mut h = std::ptr::null();
        assert_eq!(create(desc, placement, &mut h), S_OK);
        assert!(heap(h).watched.is_some(), "placed: looked at");
        LOOKED_AT_BY_TESTS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(h as usize);
        h
    }

    #[cfg(windows)]
    fn release_placed(h: This) {
        LOOKED_AT_BY_TESTS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|&at| at != h as usize);
        release(h);
    }

    /// One look at `title` for records copied in as bytes.
    #[cfg(windows)]
    fn look(title: &Heap) -> Copies {
        let mut found = Copies::default();
        title.look_for_copies(&mut found, false);
        found
    }

    /// Copies the 32 bytes of title slot `from` into slot `to` with plain
    /// stores, as the title copies records: no call sees it.
    #[cfg(windows)]
    fn copy_bytes(title: &Heap, from: u32, to: u32) {
        let at = |i: u32| (title.start + u64::from(i) * 32) as usize;
        // SAFETY: both are slots of the heap, in memory the test holds.
        unsafe { std::ptr::copy_nonoverlapping(at(from) as *const u8, at(to) as *mut u8, 32) };
    }

    /// The counter of records the title copies into its placed heap as
    /// bytes (the first step of the port of VIctory 86a5ed6 and the fork's
    /// a26859f): a record copied in with plain stores is counted once, in
    /// whichever block it lands, with what it holds, and so is one whose
    /// bytes changed only in the second of the two blocks it lies across
    /// (fork #802: such records were left out); records written through the
    /// device's view calls and descriptor copies are not.
    #[cfg(windows)]
    #[test]
    fn records_copied_as_bytes_are_counted_once_and_calls_are_not() {
        use crate::resource::tests_support::{committed, sampled};
        use crate::watch::PAGE;
        let test = "descriptor::tests::records_copied_as_bytes_are_counted_once_and_calls_are_not";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let null = std::ptr::null();
        let memory = TitlePages::new(3, 0);
        // 24 bytes into the first block: the header of slot 2047 lies across
        // the first two blocks, its generation in the second.
        let h = placed_views(memory.at as u64 + 24, ((3 * PAGE - 24) / 32) as u32);
        let title = heap(h);
        let host = title.host().expect("a host heap");
        let at = |i: u32| title.start + u64::from(i) * 32;
        // Through the device: a constant buffer view, a view of a texture
        // the title fills, a descriptor copy of it, and the same view in the
        // slot across the blocks.
        let texture = committed(&sampled(16, 16, 28));
        create_cbv(null, &[0x1000, 1, 256, 0], at(1));
        let texture_2d = [28, 4, 0x1688, 0, 0, 1, 0, 0, 0, 0];
        create_srv(null, texture, &texture_2d, at(2));
        copy_simple(null, 1, at(3), at(2), VIEWS_KIND);
        create_srv(null, texture, &texture_2d, at(2047));
        let first = look(title);
        assert!(first.pages >= 2, "{first:?}");
        assert_eq!(first.copied, 0, "the title's calls: {first:?}");
        // The same records with plain stores: the constant buffer in the
        // first block, the texture in the second, and in the third into a
        // slot a draw read through a table.
        copy_bytes(title, 1, 10);
        copy_bytes(title, 2, 4000);
        host.texture_slot_tabled(5000);
        copy_bytes(title, 2, 5000);
        let second = look(title);
        let kinds = (second.copied, second.textures, second.tabled, second.other);
        assert_eq!(kinds, (3, 2, 1, 1), "{second:?}");
        assert_eq!(second.picked(), 1, "the texture no table read: {second:?}");
        assert_eq!(second.unmade, 3, "their host slots were as they were");
        assert_eq!(second.pages, 3, "a page in each block: {second:?}");
        // Mirrored: the views of the texture made from their records, in
        // the slot a table read lately too (9376fd6 left it as it was), and
        // the constant buffer view null until a draw makes it.
        let texture_key = record_key(header(at(2)));
        assert_eq!(
            host.made_keys(&[4000, 5000, 10]),
            [texture_key, texture_key, 0]
        );
        // Nothing since, then the same record copied again: nothing new.
        assert_eq!(look(title).copied, 0);
        copy_bytes(title, 2, 4000);
        assert_eq!(look(title).copied, 0, "the record it had");
        // The generation of the record across the blocks rewritten with a
        // plain store: only the second block changed, and the record is
        // found from it (a stale generation: no view of ours).
        let [_, _, generation] = header(at(2047));
        assert_eq!(
            (at(2047) + 8) % PAGE as u64,
            0,
            "the second block's first word"
        );
        // SAFETY: the slot's third word, in memory the test holds.
        unsafe { ((at(2047) + 8) as *mut u32).write_unaligned(generation ^ 0xffff) };
        let across = look(title);
        let found = (across.pages, across.copied, across.none);
        assert_eq!(found, (1, 1, 1), "{across:?}");
        // Bytes that hold no record of ours over a copied slot: counted.
        // SAFETY: a slot of the heap.
        unsafe { (at(10) as *mut u8).write_bytes(0, 32) };
        let cleared = look(title);
        assert_eq!((cleared.copied, cleared.none), (1, 1), "{cleared:?}");
        // A view call over a copied slot: the title's own write.
        create_cbv(null, &[0x2000, 1, 256, 0], at(4000));
        assert_eq!(look(title).copied, 0);
        release_placed(h);
        type Release = extern "system" fn(This) -> u32;
        slot::<Release>(texture, 0x10)(texture);
        drop(memory);
    }

    /// The write watch decides for a range by its first block (`watch.rs`
    /// `with_writes` and `guard`): asked for a heap over a write-watched
    /// block and blocks it does not watch, it answers nothing at all, and a
    /// look asking once for the heap (as the fork's `Tracked::take` did)
    /// would never see the watched block change. The look asks per run of
    /// blocks it answers for: a record copied into the watched block is
    /// found, and the other blocks are counted.
    #[cfg(windows)]
    #[test]
    fn a_heap_the_watch_answers_for_in_part_is_looked_at_where_it_does() {
        use crate::watch::PAGE;
        let test =
            "descriptor::tests::a_heap_the_watch_answers_for_in_part_is_looked_at_where_it_does";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let null = std::ptr::null();
        let memory = TitlePages::new(1, 2);
        assert_eq!(
            crate::watch::writes(memory.at, memory.at + 3 * PAGE, true),
            None,
            "no answer for the whole range"
        );
        let h = placed_views(memory.at as u64, (3 * PAGE / 32) as u32);
        let title = heap(h);
        title.host().expect("a host heap");
        create_cbv(null, &[0x3000, 1, 256, 0], title.start + 32);
        let first = look(title);
        assert_eq!((first.copied, first.unwatched), (0, 2), "{first:?}");
        copy_bytes(title, 1, 20);
        let second = look(title);
        assert_eq!(second.copied, 1, "found in the watched block: {second:?}");
        release_placed(h);
        drop(memory);
    }

    /// A constant buffer view's record written at `at` with no call that
    /// sees a heap (the title's records it copies from), `size` bytes.
    #[cfg(windows)]
    fn constant_record(size: u32) -> [u64; 4] {
        let source = [0u64; 4];
        create_cbv(
            std::ptr::null(),
            &[0x4000, 1, size, 0],
            source.as_ptr() as u64,
        );
        source
    }

    /// Copies `record` into title slot `to` with plain stores.
    #[cfg(windows)]
    fn store_record(title: &Heap, record: &[u64; 4], to: u32) {
        let at = (title.start + u64::from(to) * 32) as *mut u8;
        // SAFETY: a slot of the heap, in memory the test holds.
        unsafe { std::ptr::copy_nonoverlapping(record.as_ptr().cast::<u8>(), at, 32) };
    }

    /// Codex review of 7c64b95: a call wrote A into a slot while the host
    /// heap was not made yet, which queued the slot's index alone; the
    /// title copied B over it as bytes; the queue's replay found B and
    /// noted it as the call's, so no look counted it. The call notes its
    /// own record, in the write that wrote it.
    #[cfg(windows)]
    #[test]
    fn a_record_copied_over_a_call_before_the_host_heap_is_made_is_found() {
        let test =
            "descriptor::tests::a_record_copied_over_a_call_before_the_host_heap_is_made_is_found";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let memory = TitlePages::new(1, 0);
        let h = placed_views(memory.at as u64, 64);
        let title = heap(h);
        let at4 = title.start + 4 * 32;
        create_cbv(std::ptr::null(), &[0x5000, 1, 256, 0], at4);
        assert!(title.host.get().is_none(), "the call is queued");
        let copied = constant_record(512);
        store_record(title, &copied, 4);
        let host = title.host().expect("a host heap");
        let found = look(title);
        assert_eq!((found.copied, found.other), (1, 1), "{found:?}");
        assert_eq!(look(title).copied, 0, "counted once");
        assert_eq!(
            host.made_keys(&[4]),
            [0],
            "a constant view: null until a draw"
        );
        release_placed(h);
        drop(memory);
    }

    /// Codex review of 7c64b95: the look read all the slots of a changed
    /// page at once, the slot across the next block's start included, and
    /// passed them all over when any of that memory was not mapped; the
    /// page's other slots were then looked at again only once the title
    /// wrote there again. Now the slots of a failed read are read in pieces
    /// a kernel page not mapped leaves whole, and those no piece reaches
    /// are read at the next looks, until they are.
    #[cfg(windows)]
    #[test]
    fn slots_next_to_memory_not_mapped_are_found_and_the_rest_later() {
        use crate::watch::PAGE;
        use windows_sys::Win32::System::Memory::{PAGE_NOACCESS, PAGE_READWRITE, VirtualProtect};
        let test =
            "descriptor::tests::slots_next_to_memory_not_mapped_are_found_and_the_rest_later";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let memory = TitlePages::new(2, 0);
        // 16 bytes into the first block: slot 2047 lies across the blocks.
        let count = ((2 * PAGE - 16) / 32) as u32;
        let h = placed_views(memory.at as u64 + 16, count);
        let title = heap(h);
        title.host().expect("a host heap");
        let record = constant_record(256);
        for slot in [5, 2047, 3000] {
            store_record(title, &record, slot);
        }
        // The second block's first kernel page goes away: the end of slot
        // 2047, and the slots after it there.
        let gone = memory.at + PAGE;
        let protect = |access| {
            let mut old = 0;
            // SAFETY: memory the test holds.
            unsafe { VirtualProtect(gone as *const _, 4096, access, &mut old) }
        };
        assert_ne!(protect(PAGE_NOACCESS), 0);
        let reached = |slot: u32| {
            let first = title.start as usize + slot as usize * 32;
            first + 32 <= gone || first >= gone + 4096
        };
        let unreachable = (0..count).filter(|&s| !reached(s)).count() as u64;
        let first = look(title);
        assert_eq!(
            (first.copied, first.unread),
            (2, unreachable),
            "slots 5 and 3000, not 2047: {first:?}"
        );
        // Left for the next looks: the blocks of slot 2047's first byte and
        // of the slots past it, whatever the watch says of them.
        let unread = |title: &Heap| {
            let watch = title.watched.as_ref().expect("looked at");
            let w = watch.look.lock().unwrap();
            w.unread.iter().copied().collect::<Vec<_>>()
        };
        let block = memory.at / PAGE;
        assert_eq!(unread(title), [block, block + 1]);
        assert_ne!(protect(PAGE_READWRITE), 0);
        // Nothing written since: the slots not read are read again.
        let second = look(title);
        assert_eq!(
            (second.copied, second.unread),
            (1, 0),
            "slot 2047: {second:?}"
        );
        assert_eq!(unread(title), [] as [usize; 0]);
        // The kernel may report the block whose protection changed as
        // written again; nothing new is found there.
        let third = look(title);
        assert_eq!((third.copied, third.unread), (0, 0), "{third:?}");
        release_placed(h);
        drop(memory);
    }

    /// Codex review of 7c64b95: a look read a slot, a call published
    /// another record there and noted it, then the look compared what it
    /// had read with that note and noted its own over it: the call's
    /// record was counted as copied at the next look, and so was the
    /// first, which no copy had written. Now a look reads slots and notes
    /// in one read of the records' lock, which a call's write waits for.
    #[cfg(windows)]
    #[test]
    fn a_call_while_a_look_reads_is_neither_counted_nor_lost() {
        use crate::resource::tests_support::{committed, sampled};
        use std::sync::atomic::{AtomicBool, Ordering};
        let test = "descriptor::tests::a_call_while_a_look_reads_is_neither_counted_nor_lost";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let memory = TitlePages::new(1, 0);
        let h = placed_views(memory.at as u64, 64);
        let title = heap(h);
        let host = title.host().expect("a host heap");
        let handle = title.start + 7 * 32;
        let texture = committed(&sampled(16, 16, 28)) as usize;
        // Views of the texture, each a record of its own: format, 2D, a
        // mapping, then (from word 4) the first mip and the count.
        let view = |mapping: u32| [28, 4, mapping, 0, 0, 1, 0, 0, 0, 0];
        let call = move |mapping: u32| {
            create_srv(std::ptr::null(), texture as This, &view(mapping), handle);
        };
        // Runs a look during which another thread's call publishes the
        // record of `mapping` at the slot: what the look found.
        let look_while_called = |mapping: u32| {
            let caller = std::sync::Arc::new(std::sync::Mutex::new(None));
            let (spawned, done) = (caller.clone(), std::sync::Arc::new(AtomicBool::new(false)));
            LOOK_BETWEEN.with(|b| {
                *b.borrow_mut() = Some(Box::new(move || {
                    let finished = done.clone();
                    let thread = std::thread::spawn(move || {
                        call(mapping);
                        finished.store(true, Ordering::SeqCst);
                    });
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    assert!(!done.load(Ordering::SeqCst), "the call waits for the look");
                    *spawned.lock().unwrap() = Some(thread);
                }));
            });
            let found = look(title);
            let thread = caller.lock().unwrap().take().expect("called meanwhile");
            thread.join().expect("the calling thread");
            found
        };
        call(0x1688);
        assert_eq!(look(title).copied, 0);
        // No copy at all: a call, then another while the look reads.
        call(0x1689);
        let during = look_while_called(0x168a);
        let after = look(title);
        assert_eq!(
            (during.copied, after.copied),
            (0, 0),
            "{during:?} {after:?}"
        );
        // A copy, then a call while the look reads: the copy is mirrored at
        // most once (skipped if the call already superseded it before
        // mirroring), never misclassified as Other, and the call is never
        // counted at the next look or lost.
        let source = [0u64; 4];
        create_srv(
            std::ptr::null(),
            texture as This,
            &view(0x168b),
            source.as_ptr() as u64,
        );
        store_record(title, &source, 7);
        let during = look_while_called(0x1688);
        let after = look(title);
        assert!(
            during.copied <= 1 && during.other == 0 && after.copied == 0,
            "{during:?} {after:?}"
        );
        assert_eq!(
            host.made_keys(&[7]),
            [record_key(header(handle))],
            "the call's"
        );
        release_placed(h);
        type Release = extern "system" fn(This) -> u32;
        slot::<Release>(texture as This, 0x10)(texture as This);
        drop(memory);
    }

    /// BTP 6 places its shader-visible heap in its own memory
    /// and checks that the heap starts there.
    #[test]
    fn placed_heaps_start_at_the_title_memory() {
        let mut memory = vec![0u8; 64 * 32];
        let address = memory.as_mut_ptr() as u64;
        let desc = HeapDesc {
            kind: 0,
            count: 64,
            flags: 1,
            nodes: 1,
        };
        let mut h = std::ptr::null();
        assert_eq!(
            create(
                desc,
                Placement {
                    address,
                    size: 64 * 32
                },
                &mut h
            ),
            S_OK
        );
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x48)(h), address);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x50)(h), address);
        // A size that does not match the count is not a placement.
        assert_eq!(create(desc, Placement { address, size: 7 }, &mut h), S_OK);
        assert_ne!(slot::<extern "system" fn(This) -> u64>(h, 0x48)(h), address);
    }

    /// Two placed views alike apart from padding or unused union-tail words
    /// share one descriptor record; changing a used word makes a new record.
    #[test]
    fn placed_views_ignore_unused_view_words() {
        let mut slot = [0u32; 8];
        let handle = slot.as_mut_ptr() as u64;
        let null = std::ptr::null();
        let resource = [1, 0, 0, 0, 0x4000, 0, 1, 0x10001, 0, 1, 0, 1, 0, 0];
        let srv = [
            0, 1, 0x1688, 0, 16, 0, 64, 16, 0, 0, 0x1000, 0x6, 0, 0, 0, 0,
        ];
        let uav = [0, 1, 16, 0, 64, 16, 0, 0, 0, 0, 0x1000, 0x6, 0, 0, 0, 0];
        for (unordered, create, desc, unused, used) in [
            (
                false,
                create_placed_srv as extern "system" fn(_, _, _, _),
                srv,
                [3usize, 9],
                6usize,
            ),
            (true, create_placed_uav, uav, [9, 9], 4),
        ] {
            create(null, &resource, &desc, handle);
            let first = slot;
            assert_eq!(view(handle), Some(View::Placed(unordered, resource, desc)));
            let mut noisy = desc;
            for w in unused {
                noisy[w] = 0xdead_beef;
            }
            create(null, &resource, &noisy, handle);
            assert_eq!(slot, first, "unused words left out");
            let mut other = desc;
            other[used] += 1;
            create(null, &resource, &other, handle);
            assert_ne!(slot, first, "a used word counts");
        }
        // A 2D texture view uses four words of the union (4..8): words 3, 8, and 9 are zeroed.
        let mut tex = [
            28, 4, 0x1688, 0x77, 0, 1, 0, 0, 0x55, 0x66, 0x2000, 0x6, 0, 0, 0, 0,
        ];
        create_placed_srv(null, &resource, &tex, handle);
        tex[3] = 0;
        (tex[8], tex[9]) = (0, 0);
        assert_eq!(view(handle), Some(View::Placed(false, resource, tex)));
    }

    /// With `XTP_HEAP_SYNC` unset (the default), `begin_sync_heaps` does
    /// nothing and spawns no work.
    #[cfg(windows)]
    #[test]
    fn begin_sync_heaps_is_noop_when_unset() {
        if heap_sync().get() {
            return;
        }
        let synced = begin_sync_heaps(|| panic!("disabled sync must not drain"));
        assert!(synced.0.is_none());
    }

    #[cfg(windows)]
    #[test]
    fn sync_pool_caller_runs_jobs_outside_the_state_lock() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        // No helpers, so the submitting thread must take both jobs itself.
        let pool = SyncPool::empty(2);
        let visited = AtomicUsize::new(0);
        pool.each(2, &|job| {
            assert!(
                pool.state.try_lock().is_ok(),
                "job runs under the state lock"
            );
            visited.fetch_or(1 << job, Ordering::Relaxed);
        });
        assert_eq!(visited.load(Ordering::Relaxed), 0b11);
    }

    #[test]
    fn placed_heap_rejects_address_wrap() {
        let mut h = std::ptr::null();
        let desc = HeapDesc {
            kind: VIEWS_KIND,
            count: 1,
            flags: SHADER_VISIBLE,
            nodes: 1,
        };
        let placement = Placement {
            address: u64::MAX - 15,
            size: 32,
        };
        assert_eq!(create(desc, placement, &mut h), E_INVALIDARG);
        assert!(h.is_null());
    }

    #[cfg(windows)]
    #[test]
    fn captured_record_does_not_reread_a_page_that_becomes_unreadable() {
        use windows_sys::Win32::System::Memory::{PAGE_NOACCESS, VirtualProtect};
        let test =
            "descriptor::tests::captured_record_does_not_reread_a_page_that_becomes_unreadable";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let memory = TitlePages::new(1, 0);
        let h = placed_views(memory.at as u64, 16);
        let title = heap(h);
        let host = title.host().expect("a host heap");
        let mut record = [0u64; 4];
        let desc = [0, 0, 0, 0];
        create_cbv(std::ptr::null(), &desc, record.as_mut_ptr() as u64);
        store_record(title, &record, 7);
        let address = memory.at;
        LOOK_BETWEEN.with(|hook| {
            *hook.borrow_mut() = Some(Box::new(move || {
                let mut old = 0;
                // SAFETY: one page of TitlePages owned by this test.
                let protected =
                    unsafe { VirtualProtect(address as *const _, 4096, PAGE_NOACCESS, &mut old) };
                assert_ne!(protected, 0);
            }));
        });
        assert_eq!(look(title).copied, 1);
        assert_eq!(host.made_key(7), 0, "copied CBV is null until a draw");
        release_placed(h);
    }

    #[cfg(windows)]
    #[test]
    fn early_snapshots_wait_for_submission_and_lose_to_newer_api_writes() {
        use crate::resource::tests_support::{committed, sampled};
        let test =
            "descriptor::tests::early_snapshots_wait_for_submission_and_lose_to_newer_api_writes";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let memory = TitlePages::new(1, 0);
        let h = placed_views(memory.at as u64, 16);
        let title = heap(h);
        let host = title.host().expect("a host heap");
        let tex = committed(&sampled(16, 16, 28));
        let desc = [28, 4, 0x1688, 0, 0, 1, 0, 0, 0, 0];
        let mut record = [0u64; 4];
        create_srv(std::ptr::null(), tex, &desc, record.as_mut_ptr() as u64);
        let key = record_key(header(record.as_ptr() as u64));
        store_record(title, &record, 3);
        store_record(title, &record, 4);
        title.look_for_copies(&mut Copies::default(), true);
        assert_eq!(host.made_keys(&[3, 4]), [0, 0]);
        let at = title.start + 4 * 32;
        let rewrite = [28, 4, 0x1689, 0, 0, 1, 0, 0, 0, 0];
        create_srv(std::ptr::null(), tex, &rewrite, at);
        let newer = record_key(header(at));
        assert_ne!(newer, key);
        let found = look(title);
        assert_eq!(found.copied, 1);
        assert_eq!(found.other, 0);
        assert_eq!(found.unmade, 1);
        assert_eq!(host.made_keys(&[3, 4]), [key, newer]);

        // A prior table read does not let a copied CBV keep the old texture.
        host.texture_slot_tabled(3);
        create_cbv(std::ptr::null(), &[0; 4], record.as_mut_ptr() as u64);
        store_record(title, &record, 3);
        assert_eq!(look(title).copied, 1);
        assert_eq!(host.made_key(3), 0);

        // A sampler copied into a view heap must never reach CreateSampler
        // with a CPU handle from the other heap type.
        create_sampler(std::ptr::null(), &[0; 13], record.as_mut_ptr() as u64);
        store_record(title, &record, 4);
        assert_eq!(look(title).none, 1);
        assert_eq!(host.made_key(4), 0);
        release_placed(h);
        crate::resource::resource_release(tex);
    }

    #[cfg(windows)]
    #[test]
    fn copied_view_keeps_its_resource_alive_and_rejects_a_vacant_entry() {
        use crate::resource::tests_support::{committed, sampled};
        let test =
            "descriptor::tests::copied_view_keeps_its_resource_alive_and_rejects_a_vacant_entry";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let tex = committed(&sampled(16, 16, 28));
        let mut record = [0u64; 4];
        create_srv(
            std::ptr::null(),
            tex,
            std::ptr::null(),
            record.as_mut_ptr() as u64,
        );
        let key = record_key(header(record.as_ptr() as u64));
        let pins = pin_view(key).expect("a retained view");
        let address = tex as usize;
        std::thread::scope(|scope| {
            scope.spawn(move || assert_eq!(crate::resource::resource_release(address as This), 1));
        });
        assert!(pin_view(key).is_some());
        drop(pins);
        assert!(pin_view(key).is_none());
        // Another test can reuse the just-freed global entry immediately.
        // Check a vacant generation against a registry private to this test.
        let vacant = Views {
            list: vec![(View::Shader(address, None), 2)],
            index: Default::default(),
            by_resource: Default::default(),
            free: vec![0],
        };
        assert!(vacant.live(2 << 32).is_none());
    }

    /// 8 threads mirror disjoint and overlapping slots while the background
    /// `xtp-heap-sync` thread scans a multi-page write-watched heap with
    /// cross-page records (more than `SYNC_PAGES` pages and `SYNC_AT_A_TIME`
    /// copied slots so `SyncPool` fans out), plus concurrent heap releases
    /// guarded by `HeapHold`: no missed mirror, no use after free, and the
    /// lock-free `quick` path hits when records are unchanged.
    #[cfg(windows)]
    #[test]
    fn concurrent_mirror_and_background_sync_with_cross_page_and_release() {
        use crate::resource::tests_support::{committed, sampled};
        use crate::watch::PAGE;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::{Arc, Barrier};

        let test =
            "descriptor::tests::concurrent_mirror_and_background_sync_with_cross_page_and_release";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let null = std::ptr::null();
        // 10 write-watched pages (> SYNC_PAGES = 8) placed 24 bytes into page 0
        // so slot 2047, 4095, 6143, ... straddle every 64 KiB page boundary.
        const PAGES: usize = 10;
        let memory = TitlePages::new(PAGES, 0);
        let count = ((PAGES * PAGE - 24) / 32) as u32;
        let desc = HeapDesc {
            kind: VIEWS_KIND,
            count,
            flags: SHADER_VISIBLE,
            nodes: 1,
        };
        let placement = Placement {
            address: memory.at as u64 + 24,
            size: u64::from(count) * 32,
        };
        // Not added to `LOOKED_AT_BY_TESTS` so `begin_sync_heaps_inner()`
        // scans this heap on the real `xtp-heap-sync` background thread.
        let mut h = std::ptr::null();
        assert_eq!(create(desc, placement, &mut h), S_OK);
        let title = heap(h);
        let host = title.host().expect("a host heap");
        let at = |i: u32| title.start + u64::from(i) * 32;

        // Create two resident texture objects so their slots hit `quick`.
        let wait_resident = |r: This| {
            let started = std::time::Instant::now();
            while crate::resource::host_resource(r).is_none() {
                assert!(started.elapsed().as_secs() < 5, "never resident");
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        };
        let tex_a = committed(&sampled(16, 16, 28));
        let tex_b = committed(&sampled(16, 16, 28));
        wait_resident(tex_a);
        wait_resident(tex_b);
        let srv_desc = [28, 4, 0x1688, 0, 0, 1, 0, 0, 0, 0];
        let mut rec_a = [0u64; 4];
        let mut rec_b = [0u64; 4];
        create_srv(null, tex_a, &srv_desc, rec_a.as_mut_ptr() as u64);
        create_srv(null, tex_b, &srv_desc, rec_b.as_mut_ptr() as u64);
        let key_a = record_key(header(rec_a.as_ptr() as u64));
        let key_b = record_key(header(rec_b.as_ptr() as u64));
        assert_ne!(key_a, 0);
        assert_ne!(key_b, 0);
        assert_ne!(key_a, key_b);

        // Pre-populate:
        // - Cross-page boundary slots across all 9 page boundaries (copied as bytes).
        // - Over 320 copied slots spread across all 10 pages (> SYNC_AT_A_TIME).
        // - Disjoint ranges for 8 mirroring threads (pre-written via API calls).
        // - Overlapping slots hammered by all 8 threads simultaneously.
        let mut copied_expected: Vec<(u32, u64)> = Vec::new();
        for boundary in 1..PAGES {
            let boundary_addr = memory.at + boundary * PAGE;
            let slot = ((boundary_addr - title.start as usize) / 32) as u32;
            let slot_addr = title.start as usize + slot as usize * 32;
            assert!(
                slot_addr < boundary_addr && slot_addr + 32 > boundary_addr,
                "slot {slot} must straddle page boundary {boundary}"
            );
            let (rec, key) = if boundary % 2 == 0 {
                (&rec_a, key_a)
            } else {
                (&rec_b, key_b)
            };
            store_record(title, rec, slot);
            copied_expected.push((slot, key));
        }
        for page in 0..PAGES {
            let base_slot = (page * (PAGE / 32)) as u32 + 100;
            for offset in 0..36u32 {
                let slot = base_slot + offset;
                if slot < count {
                    let (rec, key) = if (page + offset as usize).is_multiple_of(2) {
                        (&rec_a, key_a)
                    } else {
                        (&rec_b, key_b)
                    };
                    store_record(title, rec, slot);
                    copied_expected.push((slot, key));
                }
            }
        }
        assert!(copied_expected.len() > SYNC_AT_A_TIME);

        // Disjoint per-thread slots (64 each) + 16 shared overlapping slots.
        const THREADS: usize = 8;
        const DISJOINT_BASE: u32 = 10;
        const DISJOINT_PER_THREAD: u32 = 8;
        const OVERLAP_BASE: u32 = 80;
        const OVERLAP_COUNT: u32 = 16;
        for t in 0..THREADS {
            for k in 0..DISJOINT_PER_THREAD {
                let slot = DISJOINT_BASE + (t as u32) * DISJOINT_PER_THREAD + k;
                create_srv(null, tex_a, &srv_desc, at(slot));
            }
        }
        for k in 0..OVERLAP_COUNT {
            create_srv(
                null,
                if k % 2 == 0 { tex_a } else { tex_b },
                &srv_desc,
                at(OVERLAP_BASE + k),
            );
        }

        // Make the background mirror and all recording threads contend for
        // this same slot. It was originally made from tex_a by the API.
        store_record(title, &rec_b, OVERLAP_BASE);
        copied_expected.push((OVERLAP_BASE, key_b));

        // Start 8 mirroring threads while the main thread runs background sync
        // rounds on `xtp-heap-sync`.
        let barrier = Arc::new(Barrier::new(THREADS + 1));
        let stop = Arc::new(AtomicBool::new(false));
        let hold = title.hold();
        let quick_before = host.quick_hits();
        let mut workers = Vec::with_capacity(THREADS);
        for t in 0..THREADS {
            let barrier = barrier.clone();
            let stop = stop.clone();
            let hold = hold.clone();
            workers.push(std::thread::spawn(move || {
                barrier.wait();
                let my_base = DISJOINT_BASE + (t as u32) * DISJOINT_PER_THREAD;
                let mut rounds = 0u32;
                while !stop.load(Ordering::Acquire) || rounds < 20 {
                    rounds += 1;
                    hold.with(|heap| {
                        let Some(Some(host)) = heap.host.get() else {
                            return;
                        };
                        for k in 0..DISJOINT_PER_THREAD {
                            let slot = my_base + k;
                            let handle = heap.start + u64::from(slot) * 32;
                            assert!(host.mirror(slot, handle, true).is_some());
                        }
                        for k in 0..OVERLAP_COUNT {
                            let slot = OVERLAP_BASE + k;
                            let handle = heap.start + u64::from(slot) * 32;
                            assert!(host.mirror(slot, handle, true).is_some());
                        }
                    });
                }
            }));
        }

        barrier.wait();
        for round in 0..8 {
            let synced = begin_sync_heaps_inner();
            // Copy one more batch of records mid-flight on odd rounds.
            if round % 2 == 1 {
                let slot = 1500 + round as u32;
                store_record(title, &rec_b, slot);
                copied_expected.push((slot, key_b));
            }
            if round == 0 {
                // Two requests can be outstanding. The later ticket cannot
                // finish until both rounds have completed.
                begin_sync_heaps_inner().finish();
            }
            synced.finish();
        }
        stop.store(true, Ordering::Release);
        for w in workers {
            w.join().expect("mirror worker finished cleanly");
        }
        // One final background sync round ensures late byte copies are mirrored.
        begin_sync_heaps_inner().finish();

        let quick_delta = host.quick_hits() - quick_before;
        assert!(
            quick_delta > 0,
            "unchanged mirrors under 8 threads must hit the lock-free quick path"
        );
        for (slot, expected_key) in &copied_expected {
            assert_eq!(
                host.made_key(*slot),
                *expected_key,
                "copied slot {slot} was missed by background sync"
            );
        }
        for t in 0..THREADS {
            for k in 0..DISJOINT_PER_THREAD {
                let slot = DISJOINT_BASE + (t as u32) * DISJOINT_PER_THREAD + k;
                assert_eq!(host.made_key(slot), key_a);
            }
        }
        for k in 0..OVERLAP_COUNT {
            let slot = OVERLAP_BASE + k;
            let expected = if k != 0 && k % 2 == 0 { key_a } else { key_b };
            assert_eq!(host.made_key(slot), expected);
        }

        // Concurrent heap release while background sync is running: `HeapHold`
        // prevents any use after free and release completes cleanly.
        for _ in 0..6 {
            let pages = TitlePages::new(2, 0);
            let c2 = ((2 * PAGE - 16) / 32) as u32;
            let d2 = HeapDesc {
                kind: VIEWS_KIND,
                count: c2,
                flags: SHADER_VISIBLE,
                nodes: 1,
            };
            let p2 = Placement {
                address: pages.at as u64 + 16,
                size: u64::from(c2) * 32,
            };
            let mut h2 = std::ptr::null();
            assert_eq!(create(d2, p2, &mut h2), S_OK);
            let t2 = heap(h2);
            t2.host().expect("a host heap");
            let hold2 = t2.hold();
            for s in (0..64u32).chain([2047]) {
                store_record(t2, &rec_a, s);
            }
            let raw_h2 = h2 as usize;
            let (held, scanning) = std::sync::mpsc::channel();
            let (resume, paused) = std::sync::mpsc::channel();
            *BEFORE_LOOK.lock().unwrap() = Some(BeforeLook {
                heap: raw_h2,
                pause: Box::new(move || {
                    held.send(()).expect("test waiting for the hold");
                    paused.recv().expect("test resumes the scan");
                }),
            });
            std::thread::scope(|scope| {
                let sync = scope.spawn(|| begin_sync_heaps_inner().finish());
                if scanning
                    .recv_timeout(std::time::Duration::from_secs(10))
                    .is_err()
                {
                    drop(resume);
                    panic!("background scan did not acquire the heap");
                }
                let (started, releasing) = std::sync::mpsc::channel();
                let (done, released) = std::sync::mpsc::channel();
                let releaser = scope.spawn(move || {
                    started.send(()).unwrap();
                    release(raw_h2 as This);
                    done.send(()).unwrap();
                });
                releasing.recv().unwrap();
                let blocked = released
                    .recv_timeout(std::time::Duration::from_millis(20))
                    .is_err();
                resume.send(()).unwrap();
                releaser.join().expect("heap release did not panic");
                sync.join().expect("scan completed");
                assert!(blocked, "release waits for the live scan");
            });
            assert!(hold2.with(|_| ()).is_none(), "released heap is gone");
            drop(pages);
        }

        release(h);
        type Release = extern "system" fn(This) -> u32;
        slot::<Release>(tex_a, 0x10)(tex_a);
        slot::<Release>(tex_b, 0x10)(tex_b);
        drop(memory);
    }
}
