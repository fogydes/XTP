//! The console's DirectStorage (`XCRDAPI.dll`): a factory that opens files
//! and creates request queues and status arrays the title polls. Its
//! interfaces have grown since 2020 (new interface ids, new slots); the
//! slots here are the ones BTP 6 calls, read from its call
//! sites. Slots nobody has needed yet log their offset.
//!
//! A queue hands what the title enqueued to the workers on `Submit`, or on
//! its own once half of it is enqueued, unless the title disabled that
//! (BTP 6 does, on every queue: it writes page lists after enqueueing their
//! requests).
//!
//! Submitted requests of every queue run on one pool of workers, as the
//! console's hardware runs them: in parallel, started in queue order,
//! queues served by weighted round robin over their priorities (about 10x
//! per level, counted in bytes), with at most the staging buffer's worth of
//! compressed source in flight. Each request reads its source (a file or
//! memory), inflates zlib data and/or unpacks BCPack data, and writes the
//! destination (an address, or physical pages). Completion is what stays
//! in order: a status entry is written once every request enqueued before
//! it on its queue has finished, with the first failure (in queue order)
//! since the previous status entry, or success.
//!
//! Every 5 s a debug line sums up what the workers did: requests, bytes,
//! and their time split by stage (read, inflate, BCPack, destination,
//! completion), with the address-space changes physical pages cost meanwhile
//! (xmem's views mapped and unmapped, commits); an info line follows when
//! something went wrong or could have (`Faults`). Page destinations are copied
//! into xmem's permanent views of their allocations, and each worker keeps
//! its buffers across requests (docs/analysis/formats.md, F2 to F4).
#![allow(non_snake_case)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::ffi::{CStr, c_char, c_void};
use std::os::windows::fs::FileExt;
use std::os::windows::io::FromRawHandle;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use xtp_formats::bcpack::{self, Tables};

use super::xasync::{E_INVALIDARG, E_NOTIMPL, E_PENDING};
use super::xgameruntime::{Guid, HRESULT, Methods, Object, S_OK, add_ref, missing, put};
use super::xmem;
use crate::{HleModule, hle_export};

/// The factory interface BTP 6 asks for.
const FACTORY: Guid = Guid::new(
    0x6295_9d11,
    0xd45f,
    0x4884,
    [0xa0, 0xa3, 0xf7, 0x86, 0x2e, 0xa2, 0x43, 0xae],
);
const E_NOINTERFACE: HRESULT = 0x8000_4002_u32 as i32;
const E_OUTOFMEMORY: HRESULT = 0x8007_000E_u32 as i32;
/// `HRESULT_FROM_WIN32(ERROR_HANDLE_EOF)`: the source ends early.
const END_OF_FILE: HRESULT = 0x8007_0026_u32 as i32;
/// `HRESULT_FROM_WIN32(ERROR_INVALID_DATA)`: a bad compressed stream.
const INVALID_DATA: HRESULT = 0x8007_000D_u32 as i32;

type Out = *mut *mut c_void;

static FACTORY_OBJECT: Object = Object::new("dstorage factory", &FACTORY_VTABLE);

pub extern "system" fn DStorageGetFactory(iid: *const Guid, out: Out) -> HRESULT {
    // SAFETY: the title passes a GUID (null tolerated).
    match unsafe { iid.as_ref() } {
        Some(&iid) if iid == FACTORY => {
            put(out, std::ptr::from_ref(&FACTORY_OBJECT).cast_mut().cast())
        }
        iid => {
            tracing::warn!(iid = ?iid.map(Guid::to_string), "DStorageGetFactory: unknown interface");
            E_NOINTERFACE
        }
    }
}

/// One of our objects with state: the runtime's object header (method
/// table first), a reference count, then `T`.
#[repr(C)]
struct Instance<T> {
    base: Object,
    refs: AtomicU32,
    state: T,
}

impl<T> Instance<T> {
    fn create(base: Object, state: T) -> *mut c_void {
        Box::into_raw(Box::new(Instance {
            base,
            refs: AtomicU32::new(1),
            state,
        }))
        .cast()
    }
}

extern "system" fn add_ref_instance<T>(this: *const Instance<T>) -> u32 {
    // SAFETY: an instance we created and the title still holds.
    unsafe { &*this }.refs.fetch_add(1, Ordering::Relaxed) + 1
}

extern "system" fn release<T>(this: *mut Instance<T>) -> u32 {
    // SAFETY: as above.
    let left = unsafe { &*this }.refs.fetch_sub(1, Ordering::AcqRel) - 1;
    if left == 0 {
        // SAFETY: the last reference; created by `Instance::create`.
        drop(unsafe { Box::from_raw(this) });
    }
    left
}

fn iid_text(iid: *const Guid) -> String {
    // SAFETY: the title passes a GUID (null tolerated).
    unsafe { iid.as_ref() }.map_or_else(|| "null".into(), Guid::to_string)
}

/// `HRESULT_FROM_WIN32` of a host error.
fn from_win32(error: &std::io::Error) -> HRESULT {
    error
        .raw_os_error()
        .map_or(E_INVALIDARG, |e| (0x8007_0000 | (e as u32 & 0xffff)) as i32)
}

// The factory.

type Factory = *const Object;

/// Opens `path` for reading; the file object owns the handle.
extern "system" fn open_file(_: Factory, path: *const u16, iid: *const Guid, out: Out) -> HRESULT {
    const GENERIC_READ: u32 = 0x8000_0000;
    const FILE_SHARE_READ: u32 = 1;
    const OPEN_EXISTING: u32 = 3;
    if path.is_null() {
        return E_INVALIDARG;
    }
    // Goes through `files`, so paths on our virtual drives work too.
    let handle = super::files::CreateFileW(
        path,
        GENERIC_READ,
        FILE_SHARE_READ,
        std::ptr::null(),
        OPEN_EXISTING,
        0,
        std::ptr::null_mut(),
    );
    // SAFETY: the title passes a NUL-terminated path.
    let name = unsafe {
        let len = (0..).take_while(|&i| *path.add(i) != 0).count();
        String::from_utf16_lossy(std::slice::from_raw_parts(path, len))
    };
    if handle as isize == -1 {
        let error = std::io::Error::last_os_error();
        tracing::warn!(path = name, %error, "dstorage: opening a file failed");
        return from_win32(&error);
    }
    tracing::info!(path = name, iid = iid_text(iid), "dstorage: file opened");
    // SAFETY: a handle we just opened and nobody else owns.
    let file = unsafe { std::fs::File::from_raw_handle(handle) };
    let file = Instance::create(Object::new("dstorage file", &FILE_VTABLE), File { file });
    trace_opened(file, &name);
    put(out, file)
}

/// `(capacity, name, iid, out)`.
extern "system" fn create_status_array(
    _: Factory,
    capacity: u32,
    name: *const c_char,
    iid: *const Guid,
    out: Out,
) -> HRESULT {
    tracing::info!(
        capacity,
        name = text(name),
        iid = iid_text(iid),
        "dstorage: status array"
    );
    let entries = (0..capacity).map(|_| AtomicI32::new(S_OK)).collect();
    let array = Instance::create(
        Object::new("dstorage status array", &STATUS_VTABLE),
        StatusArray { entries },
    );
    put(out, array)
}

/// The queue's description, `DSTORAGE_QUEUE_DESC1` (the GDK's
/// `CreateQueue1`): as in 2020 (source type, capacity, priority, name),
/// plus the queue's options. BTP 6 sets `AUTO_SUBMIT_DISABLED` on all 17 of
/// its queues (4,159 entries each), and submits batches of up to 3,677
/// commands while the world loads, past the 2,079 at which they would
/// otherwise submit themselves.
#[repr(C)]
struct QueueDesc {
    source_type: u64,
    capacity: u16,
    priority: i8,
    name: *const c_char,
    options: u64,
}

/// Queue option bit 0, `AutoSubmitDisabled`: the queue never submits on its
/// own, only on `Submit`. The other bits are reserved (zero).
const AUTO_SUBMIT_DISABLED: u64 = 1;

extern "system" fn create_queue(
    _: Factory,
    desc: *const QueueDesc,
    iid: *const Guid,
    out: Out,
) -> HRESULT {
    // SAFETY: the title passes its description (null tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    let name = text(desc.name);
    let auto_submit = desc.options & AUTO_SUBMIT_DISABLED == 0;
    tracing::info!(
        source = desc.source_type,
        capacity = desc.capacity,
        priority = desc.priority,
        name,
        options = format_args!("{:#x}", desc.options),
        auto_submit,
        iid = iid_text(iid),
        "dstorage: queue"
    );
    if desc.options & !AUTO_SUBMIT_DISABLED != 0 {
        tracing::warn!(
            options = format_args!("{:#x}", desc.options),
            "dstorage: reserved queue options set; ignored"
        );
    }
    let Some(scheduler) = scheduler() else {
        return E_OUTOFMEMORY;
    };
    let mut lanes = scheduler.lock();
    let id = lanes.next_id;
    lanes.next_id += 1;
    let lane = Lane::new(desc.priority, desc.capacity, auto_submit);
    lanes.lanes.insert(id, lane);
    drop(lanes);
    put(
        out,
        Instance::create(Object::new("dstorage queue", &QUEUE_VTABLE), Queue { id }),
    )
}

/// BTP 6 asks for 32 MiB and stops if this fails.
extern "system" fn set_staging_buffer_size(_: Factory, size: u32) -> HRESULT {
    tracing::info!(size, "dstorage: staging buffer size");
    if let Some(scheduler) = scheduler() {
        scheduler.lock().staging = u64::from(size);
    }
    S_OK
}

fn text(name: *const c_char) -> String {
    if name.is_null() {
        return String::new();
    }
    // SAFETY: a NUL-terminated name from the title.
    unsafe { CStr::from_ptr(name) }
        .to_string_lossy()
        .into_owned()
}

static FACTORY_VTABLE: Methods<16> = Methods([
    missing::<0> as *const (),
    add_ref as *const (),
    add_ref as *const (),
    missing::<3> as *const (),            // 0x18 (2020: queue creation)
    open_file as *const (),               // 0x20
    create_status_array as *const (),     // 0x28
    missing::<6> as *const (),            // 0x30 (2020: CPU affinity)
    missing::<7> as *const (),            // 0x38 (2020: debug flags)
    set_staging_buffer_size as *const (), // 0x40
    missing::<9> as *const (),
    create_queue as *const (), // 0x50
    missing::<11> as *const (),
    missing::<12> as *const (),
    missing::<13> as *const (),
    missing::<14> as *const (),
    missing::<15> as *const (),
]);

// Files.

struct File {
    file: std::fs::File,
}

static FILE_VTABLE: Methods<8> = Methods([
    missing::<0> as *const (),
    add_ref_instance::<File> as *const (),
    release::<File> as *const (),
    missing::<3> as *const (),
    missing::<4> as *const (),
    missing::<5> as *const (),
    missing::<6> as *const (),
    missing::<7> as *const (),
]);

// Status arrays.

struct StatusArray {
    entries: Box<[AtomicI32]>,
}

type Array = *mut Instance<StatusArray>;

/// The result of the requests before entry `index`: pending, the first
/// failure, or success.
extern "system" fn status_result(this: Array, index: u32) -> HRESULT {
    // SAFETY: an array we created and the title still holds.
    let entries = &unsafe { &*this }.state.entries;
    entries
        .get(index as usize)
        .map_or(E_INVALIDARG, |e| e.load(Ordering::Acquire))
}

/// BTP 6 polls this (entry 0, a `Sleep(1)` loop) and never
/// reads the result itself.
extern "system" fn is_complete(this: Array, index: u32) -> bool {
    status_result(this, index) != E_PENDING
}

static STATUS_VTABLE: Methods<5> = Methods([
    missing::<0> as *const (),
    add_ref_instance::<StatusArray> as *const (),
    release::<StatusArray> as *const (),
    is_complete as *const (),   // 0x18
    status_result as *const (), // 0x20
]);

// Queues.

/// A queue object: its lane in the scheduler.
struct Queue {
    id: u64,
}

impl Drop for Queue {
    fn drop(&mut self) {
        close_lane(self.id);
    }
}

type QueueRef = *const Instance<Queue>;

/// Every queue's requests, run by a pool of workers.
struct Scheduler {
    lanes: Mutex<Lanes>,
    /// Requests to start, or staging space freed.
    work: Condvar,
    /// A request finished (for full queues and `close`).
    finished: Condvar,
}

/// Priority weights per class (low, normal, high and real-time), as the
/// console's: about 10x per level, counted against request sizes.
const COST: [u64; 3] = [100, 10, 1];
/// The console's default staging buffer.
const STAGING: u64 = 32 << 20;

struct Lanes {
    lanes: HashMap<u64, Lane>,
    next_id: u64,
    /// Submission order over all queues.
    next_seq: u64,
    /// Bytes of staging held by running requests, and the limit.
    staged: u64,
    staging: u64,
    /// A start waited for staging space since the last finish.
    starved: bool,
    /// Weighted round robin: each class's virtual time (bytes × cost), and
    /// the time of the last start; within a class, the lane clock of the
    /// last start.
    clock: [u64; 3],
    now: u64,
    lane_now: [u64; 3],
    /// Since the last debug summary: requests, bytes written, time spent
    /// running them, staging waits and BCPack bytes written.
    done: (u64, u64, Duration),
    refused: u64,
    bcpack: u64,
    since: Option<Instant>,
    /// Time workers spent per stage since the process started; that, and
    /// xmem's activity, at the last debug summary.
    spent: Spent,
    logged: (Spent, xmem::Activity),
}

/// What a request's time goes to, as the debug summary splits it.
#[derive(Clone, Copy)]
enum Stage {
    /// Reading the source from its file, or a raw read's copy from memory.
    Read,
    Inflate,
    Bcpack,
    /// Reaching title memory, besides decoding into it: opening guarded
    /// blocks, looking pages up and committing them, copying the bytes for
    /// scattered pages.
    Destination,
    /// After it: taking the scheduler's lock back and writing status
    /// entries.
    Completion,
}

const STAGES: usize = 5;

/// Time per stage.
type Spent = [Duration; STAGES];

/// What went wrong, or could have, since the last 5 s line: page
/// destinations' reads that a queue submitted on its own (before the
/// title's `Submit`, so maybe before it wrote their page lists), page lists
/// the title rewrote after the read took them (seen when the read finishes,
/// before it completes: a lower bound), page lists naming a page xmem does
/// not know (or that the host failed to commit), and failed requests.
struct Faults {
    auto_submitted: AtomicU64,
    stale: AtomicU64,
    invalid: AtomicU64,
    failed: AtomicU64,
}

static FAULTS: Faults = Faults {
    auto_submitted: AtomicU64::new(0),
    stale: AtomicU64::new(0),
    invalid: AtomicU64::new(0),
    failed: AtomicU64::new(0),
};

impl Faults {
    /// The counts, reset.
    fn take(&self) -> [u64; 4] {
        [
            &self.auto_submitted,
            &self.stale,
            &self.invalid,
            &self.failed,
        ]
        .map(|n| n.swap(0, Ordering::Relaxed))
    }
}

/// A queue's requests in the scheduler.
struct Lane {
    class: usize,
    capacity: usize,
    /// Submits on its own once half of it is enqueued (unless the title
    /// set `AUTO_SUBMIT_DISABLED`).
    auto_submit: bool,
    /// Bytes started, for round robin among the lanes of a class.
    clock: u64,
    /// Enqueued, not yet submitted.
    unsubmitted: Vec<Command>,
    /// Submitted, by sequence number: reads not started, status entries
    /// not written, reads running, and failures not yet reported.
    reads: VecDeque<(u64, Read)>,
    statuses: VecDeque<(u64, Status)>,
    running: BTreeSet<u64>,
    failures: BTreeMap<u64, HRESULT>,
    closed: bool,
}

// SAFETY: requests point into title memory and our own objects, which stay
// valid until the title has seen them complete; status arrays count their
// references atomically.
unsafe impl Send for Lane {}

/// A read a worker runs.
struct Job {
    lane: u64,
    seq: u64,
    read: Read,
    staging: u64,
}

// SAFETY: as for `Lane`.
unsafe impl Send for Job {}

/// How many DirectStorage workers run at once. The pool reads, inflates and
/// unpacks in parallel, so the useful count is physical cores: logical
/// processors count SMT siblings that add little while contending with the
/// submitting thread and host copy-backs. `XTP_DSTORAGE_WORKERS` (`1..=64`)
/// overrides the count.
// Ported from edsuuu/VI-BR-PORT commit 4ec5c417 (upstream 0bba7624, author Key,
// GPL-3.0-or-later).
fn worker_count() -> usize {
    static COUNT: OnceLock<usize> = OnceLock::new();
    *COUNT.get_or_init(|| {
        let override_val = std::env::var("XTP_DSTORAGE_WORKERS").ok();
        let logical = std::thread::available_parallelism().map_or(4, |n| n.get());
        worker_count_for(override_val.as_deref(), physical_cores(), logical)
    })
}

fn worker_count_for(override_val: Option<&str>, physical: Option<usize>, logical: usize) -> usize {
    if let Some(n) = override_val
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| (1..=64).contains(n))
    {
        return n;
    }
    physical
        .unwrap_or_else(|| (logical / 2).max(2))
        .clamp(1, 32)
}

/// The PC's physical core count (`RelationProcessorCore` returns one record per
/// core regardless of SMT siblings).
#[cfg(windows)]
fn physical_cores() -> Option<usize> {
    use windows_sys::Win32::System::SystemInformation::{
        GetLogicalProcessorInformationEx, RelationProcessorCore,
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
    };
    let mut len: u32 = 0;
    // SAFETY: a null buffer with `len = 0` queries the required byte length.
    unsafe {
        GetLogicalProcessorInformationEx(RelationProcessorCore, std::ptr::null_mut(), &mut len)
    };
    if len == 0 {
        return None;
    }
    let mut buf = vec![0u8; len as usize];
    // SAFETY: `buf` is `len` bytes long and writable for the call.
    let ok = unsafe {
        GetLogicalProcessorInformationEx(
            RelationProcessorCore,
            buf.as_mut_ptr()
                .cast::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(),
            &mut len,
        )
    };
    if ok == 0 {
        return None;
    }
    let valid_len = (len as usize).min(buf.len());
    const SIZE_OFFSET: usize = std::mem::offset_of!(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Size);
    let header_end = SIZE_OFFSET + size_of::<u32>();
    let mut at = 0usize;
    let mut cores = 0usize;
    while at + header_end <= valid_len {
        // SAFETY: `at + header_end <= valid_len` keeps the `u32` `Size` field
        // strictly within the buffer written by `GetLogicalProcessorInformationEx`.
        let size =
            unsafe { std::ptr::read_unaligned(buf.as_ptr().add(at + SIZE_OFFSET).cast::<u32>()) }
                as usize;
        if size < header_end || at + size > valid_len {
            break;
        }
        cores += 1;
        at += size;
    }
    (cores > 0).then_some(cores)
}

#[cfg(not(windows))]
fn physical_cores() -> Option<usize> {
    None
}

/// The scheduler, with its workers started on first use; `None` if not
/// even one worker could start.
fn scheduler() -> Option<&'static Scheduler> {
    static SCHEDULER: OnceLock<Option<Scheduler>> = OnceLock::new();
    static STARTED: std::sync::Once = std::sync::Once::new();
    let scheduler = SCHEDULER
        .get_or_init(|| {
            Some(Scheduler {
                lanes: Mutex::new(Lanes {
                    lanes: HashMap::new(),
                    next_id: 0,
                    next_seq: 0,
                    staged: 0,
                    staging: STAGING,
                    starved: false,
                    clock: [0; 3],
                    now: 0,
                    lane_now: [0; 3],
                    done: Default::default(),
                    refused: 0,
                    bcpack: 0,
                    since: None,
                    spent: Spent::default(),
                    logged: Default::default(),
                }),
                work: Condvar::new(),
                finished: Condvar::new(),
            })
        })
        .as_ref()?;
    STARTED.call_once(|| {
        let workers = worker_count();
        let started = (0..workers)
            .filter(|i| {
                std::thread::Builder::new()
                    .name(format!("dstorage {i}"))
                    .spawn(|| work(scheduler))
                    .is_ok()
            })
            .count();
        tracing::info!(workers = started, "dstorage: workers");
    });
    Some(scheduler)
}

impl Scheduler {
    fn lock(&self) -> MutexGuard<'_, Lanes> {
        self.lanes.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// A worker: runs the next read of the most deserving queue.
fn work(scheduler: &'static Scheduler) {
    let mut worker = Worker::default();
    let mut lanes = scheduler.lock();
    loop {
        let Some(mut job) = lanes.start() else {
            lanes = scheduler
                .work
                .wait(lanes)
                .unwrap_or_else(|e| e.into_inner());
            continue;
        };
        drop(lanes);
        let started = Instant::now();
        let result = execute(&mut job.read, &mut worker);
        trace_done(&job.read, &result);
        worker.trim();
        let finishing = Instant::now();
        lanes = scheduler.lock();
        lanes.finish(&job, result);
        worker.spent[Stage::Completion as usize] += finishing.elapsed();
        lanes.count(&job, started, std::mem::take(&mut worker.spent));
        if std::mem::take(&mut lanes.starved) {
            scheduler.work.notify_all();
        }
        scheduler.finished.notify_all();
    }
}

impl Lanes {
    /// Counts a finished request and the time its stages took; every 5 s a
    /// debug line sums them up. Time is given in workers: seconds spent per
    /// second elapsed.
    fn count(&mut self, job: &Job, started: Instant, spent: Spent) {
        let request = &job.read.request;
        let (requests, bytes, busy) = &mut self.done;
        *requests += 1;
        *bytes += u64::from(request.destination_size);
        *busy += started.elapsed();
        if request.options & BCPACK != 0 {
            self.bcpack += u64::from(request.destination_size);
        }
        for (total, stage) in self.spent.iter_mut().zip(spent) {
            *total += stage;
        }
        let us = |stage: Stage| spent[stage as usize].as_micros();
        tracing::trace!(
            options = format_args!("{:#x}", request.options),
            offset = request.offset,
            size = request.destination_size,
            read_us = us(Stage::Read),
            inflate_us = us(Stage::Inflate),
            bcpack_us = us(Stage::Bcpack),
            destination_us = us(Stage::Destination),
            completion_us = us(Stage::Completion),
            "dstorage: request done"
        );
        let since = *self.since.get_or_insert(started);
        let elapsed = since.elapsed();
        if elapsed.as_secs() >= 5 {
            let seconds = elapsed.as_secs_f64();
            let waiting: usize = self.lanes.values().map(|l| l.reads.len()).sum();
            let (before, was) = self.logged;
            let workers = |stage: Stage| {
                let i = stage as usize;
                (self.spent[i] - before[i]).as_secs_f64() / seconds
            };
            let now = xmem::activity();
            tracing::debug!(
                requests = *requests,
                mb_per_s = format_args!("{:.1}", *bytes as f64 / seconds / 1e6),
                busy_workers = format_args!("{:.2}", busy.as_secs_f64() / seconds),
                read_workers = format_args!("{:.2}", workers(Stage::Read)),
                inflate_workers = format_args!("{:.2}", workers(Stage::Inflate)),
                bcpack_workers = format_args!("{:.2}", workers(Stage::Bcpack)),
                destination_workers = format_args!("{:.2}", workers(Stage::Destination)),
                completion_workers = format_args!("{:.2}", workers(Stage::Completion)),
                views_mapped = now.maps - was.maps,
                views_unmapped = now.unmaps - was.unmaps,
                commits = now.commits - was.commits,
                staging_waits = std::mem::take(&mut self.refused),
                bcpack_mb = self.bcpack / 1_000_000,
                waiting,
                "dstorage: last {seconds:.1} s"
            );
            let [auto_submitted, stale, invalid, failed] = FAULTS.take();
            if auto_submitted + stale + invalid + failed > 0 {
                tracing::info!(
                    auto_submitted_page_reads = auto_submitted,
                    stale_page_lists = stale,
                    invalid_page_lists = invalid,
                    failed_requests = failed,
                    "dstorage: faults in the last {seconds:.1} s"
                );
            }
            if mesh_stats() {
                let mut d = destinations();
                tracing::info!("{}", d.line(seconds));
                *d = Destinations::default();
            }
            self.done = Default::default();
            self.bcpack = 0;
            self.since = None;
            self.logged = (self.spent, now);
        }
    }

    /// Starts the next read of the class with the least virtual time, from
    /// its lane that has started the fewest bytes, if the staging buffer
    /// has room for it (or nothing else holds any).
    fn start(&mut self) -> Option<Job> {
        let busy = || self.lanes.iter().filter(|(_, l)| !l.reads.is_empty());
        let (_, class) = busy().map(|(_, l)| (self.clock[l.class], l.class)).min()?;
        let (_, _, id) = busy()
            .filter(|(_, l)| l.class == class)
            .filter_map(|(&id, l)| Some((l.clock, l.reads.front()?.0, id)))
            .min()?;
        let lane = self.lanes.get_mut(&id)?;
        let staging = staging_bytes(&lane.reads.front()?.1.request);
        if self.staged > 0 && self.staged + staging > self.staging {
            self.starved = true;
            self.refused += 1;
            return None;
        }
        let (seq, read) = lane.reads.pop_front()?;
        lane.running.insert(seq);
        self.staged += staging;
        let request = &read.request;
        let size = u64::from(request.destination_size.max(request.source_size)).max(1);
        self.now = self.clock[class];
        self.clock[class] += size * COST[class];
        self.lane_now[class] = lane.clock;
        lane.clock += size;
        Some(Job {
            lane: id,
            seq,
            read,
            staging,
        })
    }

    fn finish(&mut self, job: &Job, result: Result<(), HRESULT>) {
        self.staged -= job.staging;
        if job.read.stale() {
            FAULTS.stale.fetch_add(1, Ordering::Relaxed);
        }
        if result.is_err() {
            FAULTS.failed.fetch_add(1, Ordering::Relaxed);
        }
        if let Some(lane) = self.lanes.get_mut(&job.lane) {
            lane.running.remove(&job.seq);
            if let Err(hr) = result {
                lane.failures.insert(job.seq, hr);
            }
            lane.flush();
        }
    }

    /// Moves a queue's enqueued commands to the workers.
    fn submit(&mut self, id: u64) {
        let Some(class) = self.lanes.get(&id).map(|l| l.class) else {
            return;
        };
        // A class that had nothing waiting starts from now, not from the
        // credit it left behind.
        if !self
            .lanes
            .values()
            .any(|l| l.class == class && !l.reads.is_empty())
        {
            self.clock[class] = self.clock[class].max(self.now);
        }
        let Some(lane) = self.lanes.get_mut(&id) else {
            return;
        };
        if lane.reads.is_empty() {
            lane.clock = lane.clock.max(self.lane_now[class]);
        }
        for command in std::mem::take(&mut lane.unsubmitted) {
            let seq = self.next_seq;
            self.next_seq += 1;
            match command {
                Command::Read(request) => lane.reads.push_back((seq, request)),
                Command::Status(status) => lane.statuses.push_back((seq, status)),
            }
        }
        lane.flush();
    }
}

impl Lane {
    fn new(priority: i8, capacity: u16, auto_submit: bool) -> Lane {
        Lane {
            class: priority.clamp(-1, 1).wrapping_add(1) as usize,
            capacity: usize::from(capacity),
            auto_submit,
            clock: 0,
            unsubmitted: Vec::new(),
            reads: VecDeque::new(),
            statuses: VecDeque::new(),
            running: BTreeSet::new(),
            failures: BTreeMap::new(),
            closed: false,
        }
    }

    /// Requests and status entries held, submitted or not.
    fn entries(&self) -> usize {
        self.unsubmitted.len() + self.reads.len() + self.statuses.len() + self.running.len()
    }

    /// Writes the status entries whose earlier reads have all finished.
    fn flush(&mut self) {
        while let Some(&(seq, _)) = self.statuses.front() {
            let waiting = self.reads.front().map(|r| r.0);
            if waiting
                .into_iter()
                .chain(self.running.first().copied())
                .any(|s| s < seq)
            {
                return;
            }
            let Some((_, status)) = self.statuses.pop_front() else {
                return;
            };
            let later = self.failures.split_off(&seq);
            let first = std::mem::replace(&mut self.failures, later)
                .into_values()
                .next();
            status
                .entry()
                .store(first.unwrap_or(S_OK), Ordering::Release);
        }
    }
}

/// Staging bytes a request holds while it runs: what it buffers between
/// the file and the destination. Raw reads go straight to their
/// destination, memory sources need none (as on the console).
fn staging_bytes(request: &Request) -> u64 {
    let buffered = request.options & (ZLIB | BCPACK | PAGE_DESTINATION) != 0;
    if buffered && request.options & MEMORY_SOURCE == 0 {
        u64::from(request.source_size)
    } else {
        0
    }
}

/// Ends a queue: what was not started is dropped, and this waits for the
/// reads already running (they write title memory).
fn close_lane(id: u64) {
    let Some(scheduler) = scheduler() else {
        return;
    };
    let mut lanes = scheduler.lock();
    let Some(lane) = lanes.lanes.get_mut(&id) else {
        return;
    };
    lane.closed = true;
    lane.unsubmitted.clear();
    lane.reads.clear();
    lane.statuses.clear(); // releases their arrays
    while lanes.lanes.get(&id).is_some_and(|l| !l.running.is_empty()) {
        lanes = scheduler
            .finished
            .wait(lanes)
            .unwrap_or_else(|e| e.into_inner());
    }
    lanes.lanes.remove(&id);
    scheduler.finished.notify_all();
}

/// A read request as the title builds it, 0x48 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
struct Request {
    options: u64,
    /// An address, or with `PAGE_DESTINATION` an array of 64 KiB page
    /// numbers.
    destination: *mut c_void,
    /// With `PAGE_DESTINATION`, the low 16 bits are where in the first page
    /// the destination starts.
    page_offset: u64,
    destination_size: u32,
    /// Our file object, or with `MEMORY_SOURCE` an address.
    source: *const c_void,
    offset: u64,
    source_size: u32,
    /// Compressed size between two decompression stages; unused here.
    intermediate_size: u32,
    /// Matched by cancellation; BTP 6 leaves it 0.
    tag: u64,
    name: *const c_char,
}

// Request options.
/// Inflate zlib (RFC 1950) data.
const ZLIB: u64 = 1;
/// Bits 1-4: decode BCPack data of this BC format number (0: none).
const BCPACK: u64 = 0xf << 1;
const PAGE_DESTINATION: u64 = 1 << 21;
const MEMORY_SOURCE: u64 = 1 << 22;
// Bits 5-20 select a further transform, bit 23 a source in physical pages;
// BTP 6's packages need neither, so requests with them fail.

const PAGE: usize = 64 * 1024;

/// A read as queued: the title's request and, for a page destination, its
/// page numbers once it runs.
struct Read {
    request: Request,
    /// Read from the title's array when the request runs, as the console's
    /// DMA reads it: the title writes its page numbers there after it
    /// enqueues the request and before it submits it, over what the
    /// array's previous use left (0 or other pages). A copy taken at
    /// enqueue sent BTP 6's textures into pages it had since given to
    /// models (docs/analysis/series-s.md).
    pages: Box<[usize]>,
}

impl Read {
    fn new(request: Request) -> Read {
        Read {
            request,
            pages: Box::default(),
        }
    }

    /// The destination's page numbers (empty if the destination is an
    /// address).
    fn pages(&mut self) -> &[usize] {
        let request = &self.request;
        if request.options & PAGE_DESTINATION != 0 && !request.destination.is_null() {
            // SAFETY: the title passes a page number per 64 KiB of
            // destination, its array until the request completes.
            self.pages = unsafe {
                std::slice::from_raw_parts(request.destination.cast::<usize>(), page_count(request))
            }
            .into();
        }
        &self.pages
    }

    /// Whether the title's array no longer holds the page numbers the
    /// request took when it ran: the title wrote its list after the read
    /// had begun. Asked before the request completes.
    fn stale(&self) -> bool {
        if self.pages.is_empty() {
            return false;
        }
        // SAFETY: as in `pages`; the request has not completed yet.
        let now = unsafe {
            std::slice::from_raw_parts(self.request.destination.cast::<usize>(), self.pages.len())
        };
        *now != *self.pages
    }
}

/// The page numbers a page destination's bytes reach, from where in the
/// first page it starts.
fn page_count(request: &Request) -> usize {
    let offset = (request.page_offset & 0xffff) as usize;
    (offset + request.destination_size as usize).div_ceil(PAGE)
}

enum Command {
    Read(Read),
    Status(Status),
}

/// A status entry to write, holding a reference on its array.
struct Status {
    array: Array,
    index: u32,
}

impl Drop for Status {
    fn drop(&mut self) {
        release(self.array);
    }
}

impl Status {
    fn entry(&self) -> &AtomicI32 {
        // SAFETY: we hold a reference; the index was checked when enqueued.
        &unsafe { &*self.array }.state.entries[self.index as usize]
    }
}

extern "system" fn enqueue_request(this: QueueRef, request: *const Request) {
    // SAFETY: the title passes its request (null tolerated).
    let Some(&request) = (unsafe { request.as_ref() }) else {
        return;
    };
    queue(this, Command::Read(Read::new(request)));
}

extern "system" fn enqueue_status(this: QueueRef, array: Array, index: u32) {
    // SAFETY: the title passes an array we created (null tolerated).
    let Some(entries) = (unsafe { array.as_ref() }).map(|a| &a.state.entries) else {
        return;
    };
    let Some(entry) = entries.get(index as usize) else {
        tracing::warn!(index, "dstorage: status entry out of range");
        return;
    };
    entry.store(E_PENDING, Ordering::Release);
    add_ref_instance(array);
    queue(this, Command::Status(Status { array, index }));
}

/// Adds a command to the queue; blocks while the queue is full, and
/// submits on its own once half of it is enqueued, as the console does,
/// unless the title disabled that. A full queue that does not submit itself
/// waits for `Submit` from another thread, as the console's does (the GDK's
/// `DSTORAGE_QUEUE_OPTIONS`).
fn queue(this: QueueRef, command: Command) {
    // SAFETY: a queue we created and the title still holds.
    let id = unsafe { &*this }.state.id;
    let Some(scheduler) = scheduler() else {
        return;
    };
    let mut lanes = scheduler.lock();
    let mut stalled = false;
    loop {
        let Some(lane) = lanes.lanes.get(&id).filter(|l| !l.closed) else {
            return; // closed: dropped (a status releases its array)
        };
        // One slot stays free, as the console's count of empty slots says.
        if lane.entries() + 1 < lane.capacity.max(2) {
            break;
        }
        if !lane.auto_submit && !stalled && lane.unsubmitted.len() == lane.entries() {
            stalled = true;
            tracing::warn!(
                capacity = lane.capacity,
                "dstorage: enqueue on a full queue that does not submit itself: waiting for Submit"
            );
        }
        lanes = scheduler
            .finished
            .wait(lanes)
            .unwrap_or_else(|e| e.into_inner());
    }
    let Some(lane) = lanes.lanes.get_mut(&id) else {
        return;
    };
    lane.unsubmitted.push(command);
    if lane.auto_submit && lane.unsubmitted.len() >= (lane.capacity / 2).max(1) {
        let pages = lane
            .unsubmitted
            .iter()
            .filter(|c| matches!(c, Command::Read(r) if r.request.options & PAGE_DESTINATION != 0))
            .count();
        FAULTS
            .auto_submitted
            .fetch_add(pages as u64, Ordering::Relaxed);
        lanes.submit(id);
        scheduler.work.notify_all();
    }
}

extern "system" fn submit(this: QueueRef) {
    // SAFETY: as above.
    let id = unsafe { &*this }.state.id;
    let Some(scheduler) = scheduler() else {
        return;
    };
    let mut lanes = scheduler.lock();
    let pending = lanes.lanes.get(&id).map_or(0, |l| l.unsubmitted.len());
    tracing::debug!(commands = pending, "dstorage: submit");
    lanes.submit(id);
    scheduler.work.notify_all();
}

/// Ends the queue whatever its reference count: what was not started never
/// completes, and this returns once the reads already running have
/// finished. BTP 6 calls this, not `Release`.
extern "system" fn close(this: QueueRef) {
    // SAFETY: as above.
    close_lane(unsafe { &*this }.state.id);
}

/// What a worker keeps from one request to the next: fresh heap buffers
/// for each request cost a demand-zero fault per 4 KiB touched, and big
/// ones a map and an unmap, which flushes every core's TLB.
#[derive(Default)]
struct Worker {
    /// A compressed source read from its file.
    source: Vec<u8>,
    /// zlib's output on its way to BCPack.
    intermediate: Vec<u8>,
    /// What goes to pages that are not consecutive pages of one allocation,
    /// before it is copied run by run.
    scatter: Vec<u8>,
    /// The zlib decoder, reset for each stream.
    inflater: Inflater,
    /// The current request's time per stage.
    spent: Spent,
}

/// A buffer is given back after a request that grew it past this, so 32
/// workers do not each keep their largest request's worth (commit charge).
/// Above it the request's own work dwarfs what a fresh buffer costs.
const KEEP: usize = 8 << 20;

impl Worker {
    fn trim(&mut self) {
        for buffer in [&mut self.source, &mut self.intermediate, &mut self.scatter] {
            if buffer.len() > KEEP {
                *buffer = Vec::new();
            }
        }
    }
}

/// The first `len` bytes of `buffer`, which keeps what it held: its users
/// read only what they wrote. Grown to exactly `len`, zeroed, if shorter.
fn room(buffer: &mut Vec<u8>, len: usize) -> &mut [u8] {
    if buffer.len() < len {
        *buffer = vec![0; len];
    }
    &mut buffer[..len]
}

/// Reads, inflates and/or unpacks, and writes one request, with the
/// worker's buffers, timing each stage.
fn execute(read: &mut Read, worker: &mut Worker) -> Result<(), HRESULT> {
    let request = read.request;
    let request = &request;
    let options = request.options;
    let fail = |hr: HRESULT, why: &str| {
        tracing::warn!(
            options = format_args!("{options:#x}"),
            offset = request.offset,
            source_size = request.source_size,
            destination_size = request.destination_size,
            intermediate_size = request.intermediate_size,
            tag = request.tag,
            name = text(request.name),
            hr = format_args!("{:#010x}", hr as u32),
            "dstorage: request failed: {why}"
        );
        hr
    };
    if options & !(ZLIB | BCPACK | PAGE_DESTINATION | MEMORY_SOURCE) != 0 {
        return Err(fail(E_NOTIMPL, "unsupported options"));
    }
    if request.source.is_null() || request.destination.is_null() {
        return Err(fail(E_INVALIDARG, "no source or destination"));
    }
    let bc = ((options & BCPACK) >> 1) as u8;
    let size = request.destination_size as usize;
    let pages = read.pages();
    let Worker {
        source,
        intermediate,
        scatter,
        inflater,
        spent,
    } = worker;
    let mut to = Destination {
        request,
        pages,
        scatter,
        spent,
    };
    if options & (ZLIB | BCPACK) == 0 {
        if request.source_size > request.destination_size {
            return Err(fail(E_INVALIDARG, "source larger than destination"));
        }
        let size = request.source_size as usize;
        return to
            .fill(size, Stage::Read, |out| {
                read_into(request, out).map(|()| size)
            })
            .map_err(|hr| fail(hr, "copying"));
    }
    let tables = if bc != 0 {
        Some(bcpack_tables()?)
    } else {
        None
    };
    let started = Instant::now();
    let source = read_source(request, source);
    to.spent[Stage::Read as usize] += started.elapsed();
    let source = source.map_err(|hr| fail(hr, "reading the source"))?;
    // The console's order: zlib first, then BCPack.
    let source = if bc != 0 && options & ZLIB != 0 {
        let started = Instant::now();
        let packed = room(intermediate, request.intermediate_size as usize);
        let n = inflater.inflate(source, packed);
        to.spent[Stage::Inflate as usize] += started.elapsed();
        &packed[..n.map_err(|hr| fail(hr, "zlib"))?]
    } else {
        source
    };
    match tables {
        Some(tables) => to.fill(size, Stage::Bcpack, |out| {
            bcpack::decode(bc, source, out, tables)
                .map(|()| out.len())
                .map_err(|e| fail(INVALID_DATA, &e.to_string()))
        }),
        None => to.fill(size, Stage::Inflate, |out| {
            inflater.inflate(source, out).map_err(|hr| fail(hr, "zlib"))
        }),
    }
}

/// zlib-rs's decoder. Each worker keeps one: a new one allocates its state.
struct Inflater(zlib_rs::Inflate);

impl Default for Inflater {
    fn default() -> Self {
        // A zlib header, and a window of up to 32 KiB.
        Self(zlib_rs::Inflate::new(true, 15))
    }
}

impl Inflater {
    /// Inflates a zlib stream into `out`, returning the bytes written. A
    /// stream that ends short of `out` passes: the title's chunks fill it
    /// exactly, and nothing says that is required. One that needs more room
    /// than `out`, is cut short or fails its checksum does not. zlib-rs
    /// copies in chunks of up to 32 bytes, so up to 31 bytes right after a
    /// short stream may change; the rest of `out` is left as it was.
    fn inflate(&mut self, source: &[u8], out: &mut [u8]) -> Result<usize, HRESULT> {
        use zlib_rs::{InflateFlush, Status};
        let state = &mut self.0;
        state.reset(true);
        // With no room at all zlib-rs cannot tell an empty stream from one
        // that does not fit, so an empty `out` gets one byte of scratch, and
        // a stream that writes it fails below.
        let len = out.len();
        let mut scratch = [0u8];
        let out = if len == 0 { &mut scratch[..] } else { out };
        let status = loop {
            let at = (state.total_in() as usize, state.total_out() as usize);
            let status = state.decompress(&source[at.0..], &mut out[at.1..], InflateFlush::NoFlush);
            let moved = (state.total_in() as usize, state.total_out() as usize) != at;
            match status {
                Ok(Status::Ok) if moved => continue,
                status => break status,
            }
        };
        let written = state.total_out() as usize;
        match status {
            Ok(Status::StreamEnd) if written <= len => Ok(written),
            status => {
                tracing::debug!(?status, written, "dstorage: zlib stream");
                Err(INVALID_DATA)
            }
        }
    }
}

/// The BCPack format's constant tables, parsed once for the process. If
/// they do not parse every BCPack request fails: one warning, then debug.
fn bcpack_tables() -> Result<&'static Tables, HRESULT> {
    static TABLES: OnceLock<Result<Tables, String>> = OnceLock::new();
    static WARNED: AtomicBool = AtomicBool::new(false);
    match TABLES.get_or_init(load_bcpack_tables) {
        Ok(tables) => Ok(tables),
        Err(why) if !WARNED.swap(true, Ordering::Relaxed) => {
            tracing::warn!("dstorage: BCPack requests fail: {why} (more at debug)");
            Err(E_NOTIMPL)
        }
        Err(_) => {
            tracing::debug!("dstorage: BCPack request failed: no tables");
            Err(E_NOTIMPL)
        }
    }
}

/// The tables XTP ships with.
#[cfg(not(test))]
fn load_bcpack_tables() -> Result<Tables, String> {
    Tables::builtin().map_err(|e| format!("the built-in BCPack tables: {e}"))
}

/// Tests decode streams made against tables of their own, from this folder.
#[cfg(test)]
static BCPACK_DIR: OnceLock<std::path::PathBuf> = OnceLock::new();

#[cfg(test)]
fn use_bcpack_tables(dir: std::path::PathBuf) {
    let _ = BCPACK_DIR.set(dir);
}

#[cfg(test)]
fn load_bcpack_tables() -> Result<Tables, String> {
    let dir = BCPACK_DIR
        .get()
        .ok_or("no folder set for the test tables")?;
    let bytes = std::fs::read(dir.join("tables.bin")).map_err(|e| e.to_string())?;
    Tables::parse(&bytes).map_err(|e| e.to_string())
}

/// A compressed request's source: in memory, or read from the file into
/// `buffer`.
fn read_source<'a>(request: &Request, buffer: &'a mut Vec<u8>) -> Result<&'a [u8], HRESULT> {
    let size = request.source_size as usize;
    if request.options & MEMORY_SOURCE != 0 {
        // SAFETY: the title's source holds `size` bytes until completion.
        return Ok(unsafe { std::slice::from_raw_parts(request.source.cast(), size) });
    }
    let bytes = room(buffer, size);
    read_into(request, bytes)?;
    Ok(bytes)
}

/// Fills `out` (the source size) from the request's file or memory.
fn read_into(request: &Request, out: &mut [u8]) -> Result<(), HRESULT> {
    if request.options & MEMORY_SOURCE != 0 {
        // SAFETY: the title's source holds `out.len()` bytes until completion.
        out.copy_from_slice(unsafe {
            std::slice::from_raw_parts(request.source.cast(), out.len())
        });
        return Ok(());
    }
    // SAFETY: a file object we created; the title keeps it open.
    let file = &unsafe { &*request.source.cast::<Instance<File>>() }
        .state
        .file;
    let mut done = 0;
    let mut again = true;
    loop {
        if done == out.len() {
            return Ok(());
        }
        match file.seek_read(&mut out[done..], request.offset + done as u64) {
            Ok(0) => return Err(END_OF_FILE),
            Ok(n) => done += n,
            // The destination turned read-only during the read. The write
            // guards are held open for it (`Destination::write`), so this
            // is not expected: once more, and said.
            Err(error) if again && error.raw_os_error() == Some(INVALID_USER_BUFFER) => {
                again = false;
                tracing::warn!(
                    offset = request.offset + done as u64,
                    len = out.len() - done,
                    "dstorage: a read was cut (ERROR_INVALID_USER_BUFFER); reading it again"
                );
            }
            Err(error) => return Err(from_win32(&error)),
        }
    }
}

/// `ERROR_INVALID_USER_BUFFER`: the kernel could not write the destination
/// while it read.
const INVALID_USER_BUFFER: i32 = 1784;

/// A request's destination, with what writing it needs: the page numbers
/// of a page destination, the worker's buffer for scattered pages, and
/// where its time goes.
struct Destination<'a> {
    request: &'a Request,
    pages: &'a [usize],
    scatter: &'a mut Vec<u8>,
    spent: &'a mut Spent,
}

impl Destination<'_> {
    /// Runs `fill` on the destination's first `len` bytes; `fill` returns
    /// how many it produced. Its time counts for `stage`, the rest for the
    /// destination. GPU-visible destinations are host memory like the rest
    /// (GPU address = CPU address), so an address is written in place;
    /// physical pages are too when they are consecutive pages of one
    /// allocation, and otherwise go through the buffer.
    fn fill(
        &mut self,
        len: usize,
        stage: Stage,
        fill: impl FnOnce(&mut [u8]) -> Result<usize, HRESULT>,
    ) -> Result<(), HRESULT> {
        let started = Instant::now();
        let mut filling = Duration::ZERO;
        let result = self.write(len, |out: &mut [u8]| {
            let started = Instant::now();
            let n = fill(out);
            filling = started.elapsed();
            n
        });
        self.spent[stage as usize] += filling;
        self.spent[Stage::Destination as usize] += started.elapsed().saturating_sub(filling);
        result
    }

    fn write(
        &mut self,
        len: usize,
        fill: impl FnOnce(&mut [u8]) -> Result<usize, HRESULT>,
    ) -> Result<(), HRESULT> {
        let request = self.request;
        if request.options & PAGE_DESTINATION == 0 {
            // Raw reads land through the kernel, which does not fault into
            // the GPU side's guard on title buffers: open them, and hold
            // them open until the read is done (a copy guarding them again
            // meanwhile cut reads: docs/analysis/archive-null.md).
            let _writing = xtp_gpu::kernel_will_write(request.destination as usize, len);
            // SAFETY: the title's destination holds `destination_size`
            // bytes, at least `len`, and nothing else writes it until
            // completion.
            let out =
                unsafe { std::slice::from_raw_parts_mut(request.destination.cast::<u8>(), len) };
            let n = fill(out)?;
            self.written(Form::Address, n);
            return Ok(());
        }
        let offset = (request.page_offset & 0xffff) as usize;
        // xmem's view of the pages does not fault into the guards on the
        // title's views of them either: xmem opens those after each write
        // (the window's when it drops), before the request's status.
        // Consecutive pages of one allocation (the usual case): written in
        // place in xmem's view of it.
        if let Some(mut window) = xmem::page_window(self.pages, offset, len) {
            let n = fill(window.bytes())?;
            drop(window);
            self.written(Form::InPlace, n);
            return Ok(());
        }
        let data = room(self.scatter, len);
        let n = fill(data)?;
        if xmem::write_physical(self.pages, offset, &data[..n]) {
            self.written(Form::Scattered, n);
            Ok(())
        } else {
            FAULTS.invalid.fetch_add(1, Ordering::Relaxed);
            Err(E_INVALIDARG)
        }
    }

    /// After `n` bytes were written into the destination: where (with this
    /// module at debug), to follow a destination to the host copies that
    /// read it (`XTP_FULL_REFRESH`), and the destinations' statistics
    /// (`XTP_MESH_STATS=1`). Logged once written, with what was produced:
    /// a short zlib stream leaves the end of the destination as it was, but
    /// for up to 31 bytes right after it (`Inflater::inflate`).
    fn written(&self, form: Form, n: usize) {
        let request = self.request;
        let offset = (request.page_offset & 0xffff) as usize;
        match form {
            Form::Address => tracing::debug!(
                destination = format_args!("{:#x}", request.destination as usize),
                len = n,
                "dstorage: destination written"
            ),
            Form::InPlace | Form::Scattered => tracing::debug!(
                form = form.name(),
                offset,
                len = n,
                pages = ?&self.pages[..self.pages.len().min(4)],
                "dstorage: destination written"
            ),
        }
        if !mesh_stats() {
            return;
        }
        // Looked up before the statistics' lock: xmem's index has its own.
        match form {
            Form::Address => {
                #[cfg(feature = "diagnostics")]
                let (pages, views) =
                    xtp_gpu::meshstat::other_views(request.destination as usize, n);
                #[cfg(not(feature = "diagnostics"))]
                let (pages, views) = (0, 0);
                let mut d = destinations();
                d.note(form, n);
                d.address_aliased += pages as u64;
                d.address_views += views as u64;
            }
            Form::InPlace | Form::Scattered => {
                let views = xmem::title_views(self.pages, offset, n);
                let guarding = xtp_gpu::guarding();
                let mut d = destinations();
                d.note(form, n);
                d.pages(views, guarding);
            }
        }
    }
}

/// How a request's destination is written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Form {
    /// At a title address (a raw read through the kernel, or in place).
    Address,
    /// Into physical pages, consecutive pages of one allocation, in place
    /// in xmem's view of it.
    InPlace,
    /// Into physical pages through the worker's buffer.
    Scattered,
}

impl Form {
    fn name(self) -> &'static str {
        match self {
            Form::Address => "address",
            Form::InPlace => "pages in place",
            Form::Scattered => "scattered pages",
        }
    }
}

/// Whether DirectStorage counts its destinations (`XTP_MESH_STATS=1`, off by
/// default), for the line [`Destinations::line`] makes every 5 s: whether
/// the title shows the physical pages it writes at more than one address,
/// and at how many xmem opened the GPU side's guards (fxceless's VIctory fork
/// #571 found a write signalled at one view of a page only).
fn mesh_stats() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XTP_MESH_STATS").is_ok_and(|v| v.trim() == "1"))
}

/// What DirectStorage's destinations were since the last line.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Destinations {
    /// Requests and bytes written, by [`Form`].
    requests: [u64; 3],
    bytes: [u64; 3],
    /// Physical pages written; of them those the title shows, and those it
    /// shows at more than one address, in how many requests.
    pages: u64,
    shown: u64,
    aliased: u64,
    aliased_requests: u64,
    /// The title addresses showing those pages shown more than once, and
    /// of them those whose guards xmem opened after the write: all while
    /// the GPU side guards anything, none otherwise (then every view
    /// counts as written at the next asking anyway).
    aliased_views: u64,
    signalled: u64,
    /// Destinations at title addresses: their 64 KiB pages that the title
    /// shows at other addresses too, and those other addresses
    /// (diagnostics builds).
    address_aliased: u64,
    address_views: u64,
}

fn destinations() -> MutexGuard<'static, Destinations> {
    static DESTINATIONS: Mutex<Destinations> = Mutex::new(Destinations {
        requests: [0; 3],
        bytes: [0; 3],
        pages: 0,
        shown: 0,
        aliased: 0,
        aliased_requests: 0,
        aliased_views: 0,
        signalled: 0,
        address_aliased: 0,
        address_views: 0,
    });
    DESTINATIONS.lock().unwrap_or_else(|e| e.into_inner())
}

impl Destinations {
    fn note(&mut self, form: Form, n: usize) {
        self.requests[form as usize] += 1;
        self.bytes[form as usize] += n as u64;
    }

    /// A page destination's pages, as `xmem::title_views` counts them, and
    /// whether the GPU side guarded anything when xmem opened their views.
    fn pages(
        &mut self,
        (pages, shown, aliased, views): (usize, usize, usize, usize),
        guarding: bool,
    ) {
        self.pages += pages as u64;
        self.shown += shown as u64;
        self.aliased += aliased as u64;
        self.aliased_requests += u64::from(aliased > 0);
        self.aliased_views += views as u64;
        if guarding {
            self.signalled += views as u64;
        }
    }

    /// The line, over `seconds`.
    fn line(&self, seconds: f64) -> String {
        let mb = |form: Form| self.bytes[form as usize] as f64 / 1e6;
        let n = |form: Form| self.requests[form as usize];
        format!(
            "dstorage: destinations in the last {seconds:.1} s: {} at title addresses ({:.1} MB; \
             {} of their pages shown at other addresses too, at {} other addresses), {} into \
             pages in place ({:.1} MB), {} into scattered pages ({:.1} MB); pages written {}, \
             shown by the title {}, at more than one address {} (in {} requests, at {} \
             addresses), of those addresses opened after the writes {}",
            n(Form::Address),
            mb(Form::Address),
            self.address_aliased,
            self.address_views,
            n(Form::InPlace),
            mb(Form::InPlace),
            n(Form::Scattered),
            mb(Form::Scattered),
            self.pages,
            self.shown,
            self.aliased,
            self.aliased_requests,
            self.aliased_views,
            self.signalled,
        )
    }
}

/// Slots by the 2020 order, which the argument shapes at BTP 6
///'s call sites match (0x18, 0x20, 0x30 and 0x40 are called).
static QUEUE_VTABLE: Methods<12> = Methods([
    missing::<0> as *const (),
    add_ref_instance::<Queue> as *const (),
    release::<Queue> as *const (),
    enqueue_request as *const (), // 0x18
    enqueue_status as *const (),  // 0x20
    missing::<5> as *const (),    // 0x28 (2020: fence signal)
    submit as *const (),          // 0x30
    missing::<7> as *const (),    // 0x38 (2020: cancel by tag)
    close as *const (),           // 0x40
    missing::<9> as *const (),    // 0x48 (2020: error event)
    missing::<10> as *const (),   // 0x50 (2020: error record)
    missing::<11> as *const (),   // 0x58 (2020: queue info)
]);

// The read trace (diagnostics): one line per finished request.

/// `XTP_DSTORAGE_TRACE`: `1` logs every finished request; any other value
/// logs those whose source contains it, ASCII case ignored (a file's path,
/// say `common.rpf`, or `memory` for requests that read memory). Unset or
/// empty: off. For finding which archive reads the title makes, and where
/// their data lands (the lighting's time-of-day data, docs/analysis/
/// lighting-sources.md).
fn trace_filter() -> Option<&'static str> {
    static FILTER: OnceLock<Option<String>> = OnceLock::new();
    FILTER
        .get_or_init(|| {
            std::env::var("XTP_DSTORAGE_TRACE")
                .ok()
                .map(|v| v.trim().to_ascii_lowercase())
                .filter(|v| !v.is_empty())
        })
        .as_deref()
}

/// The paths of the files the title opened, by file object, for the trace.
fn traced_files() -> MutexGuard<'static, HashMap<usize, String>> {
    static FILES: OnceLock<Mutex<HashMap<usize, String>>> = OnceLock::new();
    FILES
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// Remembers the path of a file object, while the trace is on.
fn trace_opened(file: *mut c_void, path: &str) {
    if trace_filter().is_some() {
        traced_files().insert(file as usize, path.to_owned());
    }
}

/// Whether the trace's `filter` keeps a request whose source is `source`.
fn trace_keeps(filter: &str, source: &str) -> bool {
    filter == "1" || source.to_ascii_lowercase().contains(filter)
}

/// Logs a request that has run, before its completion is signalled: its
/// source and offset, its sizes and options, the title's name for it, its
/// destination, how it ended and, for a destination at an address, the
/// first bytes written there.
fn trace_done(read: &Read, result: &Result<(), HRESULT>) {
    let Some(filter) = trace_filter() else {
        return;
    };
    let request = &read.request;
    let source = if request.options & MEMORY_SOURCE != 0 {
        format!("memory {:#x}", request.source as usize)
    } else {
        traced_files()
            .get(&(request.source as usize))
            .cloned()
            .unwrap_or_else(|| format!("file object {:#x}", request.source as usize))
    };
    if !trace_keeps(filter, &source) {
        return;
    }
    let pages = request.options & PAGE_DESTINATION != 0;
    let destination = if pages {
        format!(
            "pages from {:#x} (+{:#x})",
            read.pages.first().copied().unwrap_or(0) * PAGE,
            request.page_offset & 0xffff
        )
    } else {
        format!("{:#x}", request.destination as usize)
    };
    let head = match result {
        Ok(()) if !pages && !request.destination.is_null() => {
            let n = (request.destination_size as usize).min(16);
            // SAFETY: the request has just written its destination, which
            // the title keeps until the completion this precedes.
            let bytes = unsafe { std::slice::from_raw_parts(request.destination.cast::<u8>(), n) };
            bytes
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<Vec<_>>()
                .join(" ")
        }
        _ => String::new(),
    };
    let result = match result {
        Ok(()) => "ok".to_owned(),
        Err(hr) => format!("{:#010x}", *hr as u32),
    };
    tracing::info!(
        source,
        offset = format_args!("{:#x}", request.offset),
        source_size = request.source_size,
        destination_size = request.destination_size,
        options = format_args!("{:#x}", request.options),
        name = text(request.name),
        destination,
        result,
        head,
        "dstorage: read"
    );
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "dstorage",
        dlls: &["xcrdapi.dll"],
        exports: vec![hle_export!(DStorageGetFactory)],
    }
}

#[cfg(test)]
mod tests {
    use std::os::windows::ffi::OsStrExt;
    use std::time::{Duration, Instant};

    use super::*;
    use crate::modules::xgameruntime::method;

    fn call<F: Copy>(object: *mut c_void, slot: usize) -> F {
        // SAFETY: every object here starts with the runtime header.
        method(unsafe { &*object.cast::<Object>() }, slot)
    }

    fn factory() -> *mut c_void {
        let mut factory = std::ptr::null_mut();
        assert_eq!(DStorageGetFactory(&FACTORY, &mut factory), S_OK);
        factory
    }

    fn open(path: &std::path::Path) -> *mut c_void {
        let wide: Vec<u16> = path.as_os_str().encode_wide().chain([0]).collect();
        let open: extern "system" fn(*mut c_void, *const u16, *const Guid, Out) -> HRESULT =
            call(factory(), 0x20);
        let mut file = std::ptr::null_mut();
        assert_eq!(
            open(factory(), wide.as_ptr(), std::ptr::null(), &mut file),
            S_OK
        );
        file
    }

    fn status_array(capacity: u32) -> *mut c_void {
        let create: extern "system" fn(
            *mut c_void,
            u32,
            *const c_char,
            *const Guid,
            Out,
        ) -> HRESULT = call(factory(), 0x28);
        let mut array = std::ptr::null_mut();
        let name = c"test".as_ptr();
        assert_eq!(
            create(factory(), capacity, name, std::ptr::null(), &mut array),
            S_OK
        );
        array
    }

    /// A queue that submits itself at half capacity: some tests enqueue
    /// more than a queue holds before they submit.
    fn new_queue(source_type: u64) -> *mut c_void {
        queue_with(source_type, 0x80, 0, 0)
    }

    fn queue_with(source_type: u64, capacity: u16, priority: i8, options: u64) -> *mut c_void {
        let create: extern "system" fn(*mut c_void, *const QueueDesc, *const Guid, Out) -> HRESULT =
            call(factory(), 0x50);
        let desc = QueueDesc {
            source_type,
            capacity,
            priority,
            name: c"test".as_ptr(),
            options,
        };
        let mut queue = std::ptr::null_mut();
        assert_eq!(create(factory(), &desc, std::ptr::null(), &mut queue), S_OK);
        queue
    }

    fn request(
        options: u64,
        destination: *mut c_void,
        destination_size: usize,
        source: *const c_void,
        offset: u64,
        source_size: usize,
    ) -> Request {
        Request {
            options,
            destination,
            page_offset: 0,
            destination_size: destination_size as u32,
            source,
            offset,
            source_size: source_size as u32,
            intermediate_size: 0,
            tag: 0,
            name: std::ptr::null(),
        }
    }

    fn enqueue(queue: *mut c_void, request: &Request) {
        let f: extern "system" fn(*mut c_void, *const Request) = call(queue, 0x18);
        f(queue, request);
    }

    fn enqueue_status(queue: *mut c_void, array: *mut c_void, index: u32) {
        let f: extern "system" fn(*mut c_void, *mut c_void, u32) = call(queue, 0x20);
        f(queue, array, index);
    }

    fn void(object: *mut c_void, slot: usize) {
        let f: extern "system" fn(*mut c_void) = call(object, slot);
        f(object);
    }

    fn result(array: *mut c_void, index: u32) -> HRESULT {
        let f: extern "system" fn(*mut c_void, u32) -> HRESULT = call(array, 0x20);
        f(array, index)
    }

    /// Polls as the title does, then returns the entry's result.
    fn wait(array: *mut c_void, index: u32) -> HRESULT {
        let done: extern "system" fn(*mut c_void, u32) -> bool = call(array, 0x18);
        let start = Instant::now();
        while !done(array, index) {
            assert!(start.elapsed() < Duration::from_secs(10), "never completed");
            std::thread::sleep(Duration::from_millis(1));
        }
        result(array, index)
    }

    fn file_with(bytes: &[u8]) -> (tempfile::TempDir, *mut c_void) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.bin");
        std::fs::write(&path, bytes).unwrap();
        let file = open(&path);
        (dir, file)
    }

    fn pattern(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i * 7 + i / 251) as u8).collect()
    }

    #[test]
    fn request_layout_is_the_titles() {
        assert_eq!(size_of::<Request>(), 0x48);
        assert_eq!(std::mem::offset_of!(Request, destination_size), 0x18);
        assert_eq!(std::mem::offset_of!(Request, source), 0x20);
        assert_eq!(std::mem::offset_of!(Request, source_size), 0x30);
        assert_eq!(std::mem::offset_of!(Request, name), 0x40);
        assert_eq!(std::mem::offset_of!(QueueDesc, options), 0x18);
    }

    #[test]
    fn raw_and_zlib_reads_from_a_file() {
        let plain = pattern(300_000);
        let packed = miniz_oxide::deflate::compress_to_vec_zlib(&plain, 6);
        let mut contents = plain.clone();
        contents.extend_from_slice(&packed);
        let (_dir, file) = file_with(&contents);
        let (queue, array) = (new_queue(0), status_array(1));

        let mut raw = vec![0u8; 1000];
        let mut inflated = vec![0u8; plain.len()];
        let at = |b: &mut Vec<u8>| b.as_mut_ptr().cast();
        enqueue(queue, &request(0, at(&mut raw), 1000, file, 5000, 1000));
        let offset = plain.len() as u64;
        let zlib = request(
            ZLIB,
            at(&mut inflated),
            plain.len(),
            file,
            offset,
            packed.len(),
        );
        enqueue(queue, &zlib);
        enqueue_status(queue, array, 0);
        assert_eq!(result(array, 0), E_PENDING, "pending until done");
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        assert_eq!(raw, plain[5000..6000]);
        assert_eq!(inflated, plain);
        void(queue, 0x40);
    }

    #[test]
    fn memory_sources_inflate_too() {
        let plain = pattern(70_000);
        let packed = miniz_oxide::deflate::compress_to_vec_zlib(&plain, 6);
        let (queue, array) = (new_queue(1), status_array(1));
        let mut out = vec![0u8; plain.len()];
        let options = ZLIB | MEMORY_SOURCE;
        let r = request(
            options,
            out.as_mut_ptr().cast(),
            out.len(),
            packed.as_ptr().cast(),
            0,
            packed.len(),
        );
        enqueue(queue, &r);
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        assert_eq!(out, plain);
    }

    /// How the decoder before zlib-rs answered, for comparison.
    fn miniz(source: &[u8], out: &mut [u8]) -> Result<usize, HRESULT> {
        miniz_oxide::inflate::decompress_slice_iter_to_slice(
            out,
            std::iter::once(source),
            true,
            false,
        )
        .map_err(|_| INVALID_DATA)
    }

    /// 4 KiB runs of text, of a table of counters and of pseudo-random bytes.
    fn mixed(n: usize) -> Vec<u8> {
        let mut x = 0x9e37_79b9_7f4a_7c15_u64;
        let mut v = Vec::with_capacity(n + 32);
        while v.len() < n {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let at = v.len();
            match at / 4096 % 3 {
                0 => v.extend_from_slice(format!("sector {} entity ", x % 97).as_bytes()),
                1 => v.extend_from_slice(&(at as u32).to_le_bytes()),
                _ => v.extend_from_slice(&x.to_le_bytes()),
            }
        }
        v.truncate(n);
        v
    }

    /// Streams of each size and level inflate into room of their size, or
    /// more, of which no more than 31 bytes past the stream change, and
    /// padding after them is left alone. One decoder serves them all.
    #[test]
    fn inflate_round_trips() {
        let mut inflater = Inflater::default();
        for n in [0, 1, 4096, 70_000, 1 << 20] {
            let plain = mixed(n);
            for level in [0, 1, 6, 9] {
                let packed = miniz_oxide::deflate::compress_to_vec_zlib(&plain, level);
                let mut out = vec![0xa5; n + 100];
                assert_eq!(
                    inflater.inflate(&packed, &mut out[..n]),
                    Ok(n),
                    "{n} at {level}"
                );
                assert_eq!(out[..n], plain);
                out.fill(0xa5);
                assert_eq!(
                    inflater.inflate(&packed, &mut out),
                    Ok(n),
                    "{n} at {level}, more"
                );
                assert_eq!(out[..n], plain);
                assert!(
                    out[n + 32..].iter().all(|&b| b == 0xa5),
                    "{n} at {level}: past it"
                );
                let mut padded = packed.clone();
                padded.extend_from_slice(&[0xff; 16]);
                assert_eq!(
                    inflater.inflate(&padded, &mut out[..n]),
                    Ok(n),
                    "{n}, padded"
                );
                assert_eq!(miniz(&padded, &mut out[..n]), Ok(n), "{n}, padded, before");
            }
        }
    }

    /// A stream that is cut short, damaged, not zlib or bigger than its room
    /// fails, as it did before zlib-rs, and the decoder still serves the
    /// next one.
    #[test]
    fn inflate_rejects_what_is_not_a_whole_stream_that_fits() {
        let mut inflater = Inflater::default();
        let plain = mixed(70_000);
        let n = plain.len();
        let packed = miniz_oxide::deflate::compress_to_vec_zlib(&plain, 6);
        let mut bad = Vec::new();
        let end = packed.len();
        for cut in [0, 1, 2, 100, end / 2, end - 4, end - 1] {
            bad.push((packed[..cut].to_vec(), n));
        }
        let damaged = |at: usize| {
            let mut v = packed.clone();
            v[at] ^= 0x55;
            v
        };
        bad.push((damaged(1), n)); // the header's check bits
        bad.push((damaged(end / 2), n));
        bad.push((damaged(end - 1), n)); // the checksum
        bad.push((miniz_oxide::deflate::compress_to_vec(&plain, 6), n)); // no header
        bad.push((packed.clone(), n - 1));
        bad.push((packed.clone(), 0));
        for (i, (source, len)) in bad.iter().enumerate() {
            let mut out = vec![0u8; *len];
            assert_eq!(
                inflater.inflate(source, &mut out),
                Err(INVALID_DATA),
                "case {i}"
            );
            assert_eq!(
                miniz(source, &mut out),
                Err(INVALID_DATA),
                "case {i}, before"
            );
            let mut out = vec![0u8; n];
            assert_eq!(inflater.inflate(&packed, &mut out), Ok(n), "after case {i}");
            assert_eq!(out, plain);
        }
        // An empty stream needs no room.
        let empty = miniz_oxide::deflate::compress_to_vec_zlib(&[], 6);
        assert_eq!(inflater.inflate(&empty, &mut []), Ok(0));
        assert_eq!(miniz(&empty, &mut []), Ok(0));
    }

    #[test]
    fn statuses_complete_in_order_with_the_first_failure() {
        let contents = pattern(4096);
        let (_dir, file) = file_with(&contents);
        let (queue, array) = (new_queue(0), status_array(3));
        // Requests in flight together run in any order, so none of them
        // shares a destination (failed ones leave theirs undefined).
        let mut buffers = [[0u8; 16]; 5];
        let [failed, failed_too, out, other, bcpack] =
            buffers.each_mut().map(|b| b.as_mut_ptr().cast());

        // Segment 0: a read past the end, then a bad stream; the first wins.
        enqueue(queue, &request(0, failed, 16, file, 4090, 16));
        enqueue(queue, &request(ZLIB, failed_too, 16, file, 0, 16));
        enqueue_status(queue, array, 0);
        // Segment 1: two reads.
        enqueue(queue, &request(0, other, 16, file, 0, 16));
        enqueue(queue, &request(0, out, 16, file, 100, 16));
        enqueue_status(queue, array, 1);
        void(queue, 0x30);
        // Segment 2: a BCPack request (BC7) whose stream is empty fails.
        with_test_tables();
        enqueue(queue, &request(7 << 1, bcpack, 16, file, 0, 0));
        enqueue_status(queue, array, 2);
        void(queue, 0x30);

        assert_eq!(wait(array, 2), INVALID_DATA);
        assert_eq!(result(array, 0), END_OF_FILE);
        assert_eq!(result(array, 1), S_OK);
        assert_eq!(buffers[2], contents[100..116]);
        assert_eq!(buffers[3], contents[..16]);
    }

    #[test]
    fn half_a_queue_submits_itself() {
        let (_dir, file) = file_with(&pattern(64));
        let queue = queue_with(0, 4, 1, 0);
        let array = status_array(1);
        let mut out = [0u8; 16];
        enqueue(queue, &request(0, out.as_mut_ptr().cast(), 16, file, 8, 16));
        enqueue_status(queue, array, 0); // the second of four: no submit needed
        assert_eq!(wait(array, 0), S_OK);
        assert_eq!(out, pattern(64)[8..24]);
    }

    /// The commands a queue holds that it has not submitted.
    fn unsubmitted(queue: *mut c_void) -> usize {
        // SAFETY: a queue we created and still hold.
        let id = unsafe { &*queue.cast::<Instance<Queue>>() }.state.id;
        scheduler().unwrap().lock().lanes[&id].unsubmitted.len()
    }

    /// A queue created with `AUTO_SUBMIT_DISABLED` (as BTP 6 creates all
    /// of its queues) runs nothing before `Submit`, however full it gets.
    #[test]
    fn a_queue_with_auto_submit_disabled_never_submits_itself() {
        let contents = pattern(64);
        let (_dir, file) = file_with(&contents);
        let queue = queue_with(0, 4, 1, AUTO_SUBMIT_DISABLED);
        let array = status_array(1);
        let mut outs = [[0u8; 16]; 2];
        let [first, second] = outs.each_mut().map(|o| o.as_mut_ptr().cast());
        enqueue(queue, &request(0, first, 16, file, 8, 16));
        enqueue_status(queue, array, 0); // half: an automatic queue submits here
        enqueue(queue, &request(0, second, 16, file, 24, 16)); // full
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(unsubmitted(queue), 3);
        assert_eq!(result(array, 0), E_PENDING);
        assert_eq!(outs, [[0; 16]; 2], "nothing ran");
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        assert_eq!(outs[0], contents[8..24]);
        void(queue, 0x40); // the second read: completes or is dropped
    }

    /// An enqueue on a full queue that does not submit itself waits until
    /// another thread submits, as the GDK documents for the option.
    #[test]
    fn a_full_queue_with_auto_submit_disabled_waits_for_submit() {
        let contents = pattern(64);
        let (_dir, file) = file_with(&contents);
        let queue = queue_with(0, 4, 0, AUTO_SUBMIT_DISABLED);
        let array = status_array(1);
        let mut outs = [[0u8; 16]; 4];
        let at = outs.each_mut().map(|o| o.as_mut_ptr() as usize);
        let file = file as usize;
        let read = move |k: usize| {
            let (to, from) = (at[k] as *mut c_void, file as *const c_void);
            request(0, to, 16, from, 16 * k as u64, 16)
        };
        for k in 0..3 {
            enqueue(queue, &read(k));
        }
        let queue_at = queue as usize;
        let enqueued = AtomicBool::new(false);
        std::thread::scope(|s| {
            s.spawn(|| {
                enqueue(queue_at as *mut c_void, &read(3));
                enqueued.store(true, Ordering::Release);
            });
            std::thread::sleep(Duration::from_millis(50));
            assert!(!enqueued.load(Ordering::Acquire), "the queue was full");
            void(queue, 0x30);
        });
        assert_eq!(unsubmitted(queue), 1, "the fourth, enqueued after Submit");
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        let expected: Vec<&[u8]> = contents.chunks(16).collect();
        assert!(outs.iter().zip(expected).all(|(o, e)| o == e));
        void(queue, 0x40);
    }

    /// BTP 6 enqueues a request into pages, then writes the request's page
    /// list, then submits. A queue that disabled automatic submission must
    /// not run it in between, even past half its capacity: the request
    /// lands in the pages the title wrote, not in those its array held.
    #[test]
    fn a_page_list_written_before_submit_is_the_one_read() {
        let (old, new) = (allocate(1), allocate(1));
        let plain = pattern(100);
        let queue = queue_with(1, 4, 0, AUTO_SUBMIT_DISABLED);
        let array = status_array(1);
        let mut list = [old[0]]; // what the array's last use left
        let entries = list.as_mut_ptr();
        let options = PAGE_DESTINATION | MEMORY_SOURCE;
        let r = request(options, entries.cast(), 100, plain.as_ptr().cast(), 0, 100);
        enqueue(queue, &r);
        enqueue_status(queue, array, 0); // half: an automatic queue submits here
        // An automatic queue's read would have taken the old page by now.
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(result(array, 0), E_PENDING);
        // SAFETY: our array, which the request reads only once submitted.
        unsafe { entries.write(new[0]) };
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        assert_eq!(page_bytes(&new)[..100], plain);
        assert!(page_bytes(&old).iter().all(|&b| b == 0), "the old page");
        void(queue, 0x40);
        free(&old);
        free(&new);
    }

    /// A page list rewritten after its read took it counts as stale.
    #[test]
    fn a_page_list_rewritten_after_the_read_is_stale() {
        let mut list = [5usize, 6];
        let entries = list.as_mut_ptr();
        let source = std::ptr::dangling::<c_void>(); // never read here
        let options = PAGE_DESTINATION | MEMORY_SOURCE;
        let mut r = request(options, entries.cast(), PAGE, source, 0, 8);
        r.page_offset = 10; // two pages
        let mut read = Read::new(r);
        assert!(!read.stale(), "not run yet");
        assert_eq!(read.pages(), [5, 6]);
        assert!(!read.stale());
        // SAFETY: our array.
        unsafe { entries.add(1).write(7) };
        assert!(read.stale());
        // Destinations at an address have no list.
        let mut plain = Read::new(request(0, entries.cast(), 8, source, 0, 8));
        assert!(plain.pages().is_empty());
        assert!(!plain.stale());
    }

    /// Lanes without workers: what `start` picks, and when statuses land.
    fn lanes(staging: u64) -> Lanes {
        Lanes {
            lanes: HashMap::new(),
            next_id: 0,
            next_seq: 0,
            staged: 0,
            staging,
            starved: false,
            clock: [0; 3],
            now: 0,
            lane_now: [0; 3],
            done: Default::default(),
            refused: 0,
            bcpack: 0,
            since: None,
            spent: Spent::default(),
            logged: Default::default(),
        }
    }

    fn read(lanes: &mut Lanes, id: u64, options: u64, size: u32) {
        let mut r = request(options, std::ptr::null_mut(), 0, std::ptr::null(), 0, 0);
        (r.destination_size, r.source_size) = (size, size);
        let lane = lanes.lanes.get_mut(&id).unwrap();
        lane.unsubmitted.push(Command::Read(Read::new(r)));
    }

    #[test]
    fn priorities_share_by_bytes_about_tenfold() {
        let mut l = lanes(STAGING);
        l.lanes.insert(0, Lane::new(0, 4096, true));
        l.lanes.insert(1, Lane::new(1, 4096, true));
        l.lanes.insert(2, Lane::new(2, 4096, true)); // real-time: as high
        for _ in 0..400 {
            for id in 0..3 {
                read(&mut l, id, 0, 4096);
            }
        }
        (0..3).for_each(|id| l.submit(id));
        let mut started = [0u32; 3];
        for _ in 0..330 {
            let job = l.start().unwrap();
            started[job.lane as usize] += 1;
            l.finish(&job, Ok(()));
        }
        // Normal gets about a tenth of what high and real-time get together.
        assert_eq!(started[1] + started[2], 300, "{started:?}");
        assert_eq!(started[0], 30, "{started:?}");
        assert!(started[1].abs_diff(started[2]) <= 1, "{started:?}");
    }

    #[test]
    fn staging_bounds_compressed_bytes_in_flight() {
        let mut l = lanes(100);
        l.lanes.insert(0, Lane::new(0, 64, true));
        read(&mut l, 0, ZLIB, 60);
        read(&mut l, 0, ZLIB, 60);
        read(&mut l, 0, 0, 1000); // raw: straight to its destination
        l.submit(0);
        let first = l.start().unwrap();
        assert!(l.start().is_none(), "60 + 60 > 100");
        l.finish(&first, Ok(()));
        let second = l.start().unwrap();
        let raw = l.start().unwrap();
        assert_eq!((second.staging, raw.staging), (60, 0));
        // Alone, a request larger than the buffer still runs.
        read(&mut l, 0, ZLIB, 500);
        l.submit(0);
        assert!(l.start().is_none());
        l.finish(&second, Ok(()));
        assert!(l.start().is_some());
    }

    #[test]
    fn a_status_waits_for_every_earlier_read() {
        let mut l = lanes(STAGING);
        l.lanes.insert(0, Lane::new(0, 64, true));
        let array = status_array(2).cast::<Instance<StatusArray>>();
        let status = |index| {
            add_ref_instance(array);
            Command::Status(Status { array, index })
        };
        read(&mut l, 0, 0, 16);
        read(&mut l, 0, 0, 16);
        l.lanes.get_mut(&0).unwrap().unsubmitted.push(status(0));
        read(&mut l, 0, 0, 16);
        l.lanes.get_mut(&0).unwrap().unsubmitted.push(status(1));
        // SAFETY: an array we created.
        for entry in unsafe { &(*array).state.entries }.iter() {
            entry.store(E_PENDING, Ordering::Release);
        }
        l.submit(0);
        let jobs: Vec<Job> = (0..3).map(|_| l.start().unwrap()).collect();
        // The last read and the first finish; the second then fails.
        l.finish(&jobs[2], Ok(()));
        l.finish(&jobs[0], Ok(()));
        let results = || (result(array.cast(), 0), result(array.cast(), 1));
        assert_eq!(results(), (E_PENDING, E_PENDING));
        l.finish(&jobs[1], Err(INVALID_DATA));
        assert_eq!(results(), (INVALID_DATA, S_OK));
    }

    #[test]
    fn zlib_then_bcpack() {
        with_test_tables();
        let packed = miniz_oxide::deflate::compress_to_vec_zlib(&BC4_STREAM, 6);
        let (_dir, file) = file_with(&packed);
        let (queue, array) = (new_queue(0), status_array(1));
        let mut out = [0u8; 16];
        let options = ZLIB | 4 << 1;
        let mut r = request(options, out.as_mut_ptr().cast(), 16, file, 0, packed.len());
        r.intermediate_size = BC4_STREAM.len() as u32;
        enqueue(queue, &r);
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        assert_eq!(out, bc4_blocks());
    }

    /// BCPack tables we made (uniform models; BC4 index pairs: low 3 bits,
    /// next 3 bits), in a folder kept for the test process.
    fn with_test_tables() {
        static DIR: OnceLock<()> = OnceLock::new();
        DIR.get_or_init(|| {
            let mut t = Vec::new();
            for m in 0..128 {
                let g = m / 16;
                for b in 0..9 {
                    let f: u16 = if b < g + 2 { 1 << (15 - g) } else { 0 };
                    t.extend_from_slice(&f.to_le_bytes());
                }
            }
            t.extend([0; 256 + 64 + 256]);
            t.extend((0..64u8).map(|v| (v & 7) | (v >> 3) << 4));
            t.resize(bcpack::TABLES_SIZE, 0);
            let dir = tempfile::tempdir().unwrap().keep();
            std::fs::write(dir.join("tables.bin"), t).unwrap();
            use_bcpack_tables(dir);
        });
    }

    /// Two BC4 blocks from the format's test encoder: endpoints 10, -3 with
    /// indices p -> p & 7, then the same block plus 1 everywhere.
    const BC4_STREAM: [u8; 88] = [
        0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 4, 4, 4, 4, 1, 1, 1, 1, 0x10, 0x10, 0x10, 0x10, 2, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0x14, 5, 0x20, 0x68, 0xb0, 0xf8, 3, 3, 2, 2, 0x24, 0x24, 0x24,
        0x24, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 0x10, 0x10, 0x10, 0x10, 0, 0, 0, 0,
        0x20, 0x68, 0xb0, 0xf8, 0x24, 0x24, 0x24, 0x24, 0, 0, 0, 0, 0, 0, 0, 0,
    ];

    /// What `BC4_STREAM` unpacks to.
    fn bc4_blocks() -> [u8; 16] {
        let code = [0u64, 2, 3, 4, 5, 6, 7, 1];
        let block = |e0: u64, e1: u64, add: usize| {
            let bits = (0..16).fold(0, |a, p| a | code[(p + add) & 7] << (16 + 3 * p));
            (e0 | e1 << 8 | bits).to_le_bytes()
        };
        let mut out = [0; 16];
        out[..8].copy_from_slice(&block(138, 125, 0));
        out[8..].copy_from_slice(&block(139, 126, 1));
        out
    }

    #[test]
    fn bcpack_requests_unpack_into_the_destination() {
        with_test_tables();
        let mut contents = pattern(100);
        contents.extend_from_slice(&BC4_STREAM);
        let (_dir, file) = file_with(&contents);
        let (queue, array) = (new_queue(0), status_array(1));
        let mut out = [0u8; 16];
        enqueue(
            queue,
            &request(4 << 1, out.as_mut_ptr().cast(), 16, file, 100, 88),
        );
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        assert_eq!(out, bc4_blocks());
    }

    #[test]
    fn a_closed_queue_completes_nothing() {
        let (_dir, file) = file_with(&pattern(64));
        let (queue, array) = (new_queue(0), status_array(1));
        let mut out = [0u8; 16];
        enqueue(queue, &request(0, out.as_mut_ptr().cast(), 16, file, 0, 16));
        enqueue_status(queue, array, 0);
        void(queue, 0x40);
        void(queue, 0x30);
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(result(array, 0), E_PENDING);
        assert_eq!(out, [0; 16]);
    }

    /// DirectStorage requests into physical pages, end to end: four queues
    /// of 256 KiB raw reads into consecutive pages of one allocation (how
    /// the title streams), from memory and from a file: a generated one, or
    /// the one `XTP_DSTORAGE_BENCH_FILE` names (only read; at least 256
    /// MiB). Prints MB/s; run it with
    /// `cargo test -p xtp-hle --release --locked -- --ignored --nocapture page_request_throughput`.
    #[test]
    #[ignore = "benchmark"]
    fn page_request_throughput() {
        use crate::modules::xmem;
        const PAGES: usize = 4096;
        const SIZE: usize = 256 << 10;
        const QUEUES: usize = 4;
        let (mut pages, mut n) = (vec![0usize; PAGES], PAGES);
        assert_eq!(
            xmem::XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()),
            1
        );
        let memory = pattern(PAGES * PAGE);
        let dir = tempfile::tempdir().unwrap();
        let path = match std::env::var_os("XTP_DSTORAGE_BENCH_FILE") {
            Some(path) => std::path::PathBuf::from(path),
            None => {
                let path = dir.path().join("data.bin");
                std::fs::write(&path, &memory).unwrap();
                path
            }
        };
        let file = open(&path) as usize;
        let run = |name: &str, passes: usize, in_memory: bool| {
            let started = Instant::now();
            std::thread::scope(|s| {
                for q in 0..QUEUES {
                    let (pages, memory) = (&pages, &memory);
                    s.spawn(move || {
                        let (queue, array) = (new_queue(in_memory.into()), status_array(1));
                        let per = PAGES * PAGE / SIZE / QUEUES;
                        for _ in 0..passes {
                            for k in 0..per {
                                let at = (q * per + k) * SIZE;
                                let to = pages[at / PAGE..].as_ptr().cast_mut().cast();
                                let r = if in_memory {
                                    let from = memory[at..].as_ptr().cast();
                                    let options = PAGE_DESTINATION | MEMORY_SOURCE;
                                    request(options, to, SIZE, from, 0, SIZE)
                                } else {
                                    let from = file as *const c_void;
                                    request(PAGE_DESTINATION, to, SIZE, from, at as u64, SIZE)
                                };
                                enqueue(queue, &r);
                            }
                        }
                        enqueue_status(queue, array, 0);
                        void(queue, 0x30);
                        assert_eq!(wait(array, 0), S_OK);
                        void(queue, 0x40);
                    });
                }
            });
            let bytes = (passes * PAGES * PAGE) as f64;
            let mb_per_s = bytes / started.elapsed().as_secs_f64() / 1e6;
            println!("{name:<28} {mb_per_s:>8.0} MB/s");
        };
        run("from memory, first touch", 1, true);
        run("from memory", 4, true);
        run("from the file", 4, false);
        assert_eq!(xmem::XMemFreePhysicalPages(PAGES, pages.as_ptr()), 1);
    }

    #[test]
    fn physical_page_destinations() {
        use crate::modules::xmem;
        let mut pages = [0usize; 2];
        let mut n = 2;
        assert_eq!(
            xmem::XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()),
            1
        );
        // Where the title would see them: a reservation of 64 KiB pages.
        let mapped = xmem::XMemVirtualAlloc(
            std::ptr::null_mut(),
            2 * PAGE,
            0x2000 | 0x2040_0000, // reserve, 64 KiB pages
            0,
            0x04,
        );
        assert!(!mapped.is_null());
        assert_eq!(
            xmem::XMemMapPhysicalPages(mapped, 2, pages.as_ptr()),
            mapped
        );

        let plain = pattern(100);
        let (queue, array) = (new_queue(1), status_array(1));
        let mut r = request(
            PAGE_DESTINATION | MEMORY_SOURCE,
            pages.as_mut_ptr().cast(),
            plain.len(),
            plain.as_ptr().cast(),
            0,
            plain.len(),
        );
        r.page_offset = (PAGE - 40) as u64; // across the page boundary
        enqueue(queue, &r);
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        // SAFETY: both pages are mapped read-write.
        let seen = unsafe { std::slice::from_raw_parts(mapped.cast::<u8>().add(PAGE - 40), 100) };
        assert_eq!(seen, plain);
    }
    #[test]
    fn zlib_into_pages_in_place_and_across_allocations() {
        use crate::modules::xmem;
        let allocate = |n: usize| {
            let (mut pages, mut given) = (vec![0usize; n], n);
            assert_eq!(
                xmem::XMemAllocatePhysicalPages(0, &mut given, pages.as_mut_ptr()),
                1
            );
            pages
        };
        let (a, b) = (allocate(3), allocate(1));
        let plain = pattern(2 * PAGE + 100);
        let packed = miniz_oxide::deflate::compress_to_vec_zlib(&plain, 6);
        // One allocation's pages in order (one window), then pages of two
        // (through a buffer), both from 50 bytes into the first.
        let mut destinations = [a.clone(), vec![a[0], a[1], b[0]]];
        let (queue, array) = (new_queue(1), status_array(1));
        for pages in &mut destinations {
            let mut r = request(
                ZLIB | PAGE_DESTINATION | MEMORY_SOURCE,
                pages.as_mut_ptr().cast(),
                plain.len(),
                packed.as_ptr().cast(),
                0,
                packed.len(),
            );
            r.page_offset = 50;
            enqueue(queue, &r);
        }
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        for pages in &destinations {
            let mapped = xmem::XMemVirtualAlloc(
                std::ptr::null_mut(),
                3 * PAGE,
                0x2000 | 0x2040_0000, // reserve, 64 KiB pages
                0,
                0x04,
            );
            assert!(!mapped.is_null());
            assert_eq!(
                xmem::XMemMapPhysicalPages(mapped, 3, pages.as_ptr()),
                mapped
            );
            // SAFETY: all three pages are mapped read-write.
            let seen = unsafe { std::slice::from_raw_parts(mapped.cast::<u8>(), 3 * PAGE) };
            assert!(seen[..50].iter().all(|&b| b == 0));
            assert_eq!(seen[50..50 + plain.len()], plain);
            assert_eq!(
                xmem::XMemMapPhysicalPages(mapped, 3, std::ptr::null()),
                mapped
            );
        }
        assert_eq!(xmem::XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(xmem::XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    /// `n` pages of one new allocation.
    fn allocate(n: usize) -> Vec<usize> {
        let (mut pages, mut given) = (vec![0usize; n], n);
        assert_eq!(
            xmem::XMemAllocatePhysicalPages(0, &mut given, pages.as_mut_ptr()),
            1
        );
        pages
    }

    /// A reservation of 64 KiB pages, as titles make them for mapping.
    fn reserve(n: usize) -> *mut c_void {
        let p = xmem::XMemVirtualAlloc(
            std::ptr::null_mut(),
            n * PAGE,
            0x2000 | 0x2040_0000, // reserve, 64 KiB pages
            0,
            0x04,
        );
        assert!(!p.is_null());
        p
    }

    /// The bytes of `pages`, through a title view mapped for the purpose.
    fn page_bytes(pages: &[usize]) -> Vec<u8> {
        let view = reserve(pages.len());
        let len = pages.len() * PAGE;
        assert_eq!(
            xmem::XMemMapPhysicalPages(view, pages.len(), pages.as_ptr()),
            view
        );
        // SAFETY: the pages are mapped read-write there.
        let bytes = unsafe { std::slice::from_raw_parts(view.cast::<u8>(), len) }.to_vec();
        assert_eq!(
            xmem::XMemMapPhysicalPages(view, pages.len(), std::ptr::null()),
            view
        );
        bytes
    }

    fn free(pages: &[usize]) {
        assert_eq!(xmem::XMemFreePhysicalPages(pages.len(), pages.as_ptr()), 1);
    }

    /// The statistics of `XTP_MESH_STATS` see a page the title shows at two
    /// addresses (the aliases of the fork's #571) as such, from wherever in
    /// its pages a destination starts, and pages it shows nowhere yet.
    #[test]
    fn destinations_count_the_title_views_of_their_pages() {
        let pages = allocate(3);
        let (a, b) = (reserve(2), reserve(1));
        // Pages 0 and 1 at `a`, page 0 at `b` too; page 2 nowhere.
        assert_eq!(xmem::XMemMapPhysicalPages(a, 2, pages.as_ptr()), a);
        assert_eq!(xmem::XMemMapPhysicalPages(b, 1, pages.as_ptr()), b);
        assert_eq!(xmem::title_views(&pages, 0, 3 * PAGE), (3, 2, 1, 2));
        assert_eq!(xmem::title_views(&pages, PAGE - 10, 20), (2, 2, 1, 2));
        assert_eq!(xmem::title_views(&pages[1..], 0, 100), (1, 1, 0, 0));
        assert_eq!(xmem::title_views(&pages[2..], 0, 100), (1, 0, 0, 0));
        // An unknown page: nothing counted.
        assert_eq!(xmem::title_views(&[7], 0, 100), (0, 0, 0, 0));

        let mut d = Destinations::default();
        d.note(Form::InPlace, 3 * PAGE);
        d.pages(xmem::title_views(&pages, 0, 3 * PAGE), true);
        d.note(Form::Scattered, 100);
        d.pages(xmem::title_views(&pages, 0, 100), false);
        assert_eq!(d.requests, [0, 1, 1]);
        assert_eq!(
            (d.pages, d.shown, d.aliased, d.aliased_requests),
            (4, 3, 2, 2)
        );
        // Opened after the writes only while something was guarded.
        assert_eq!((d.aliased_views, d.signalled), (4, 2));
        let line = d.line(5.0);
        assert!(
            line.contains("at more than one address 2 (in 2 requests, at 4 addresses)"),
            "{line}"
        );

        // Unmapped, the page shows once more.
        assert_eq!(xmem::XMemMapPhysicalPages(b, 1, std::ptr::null()), b);
        assert_eq!(xmem::title_views(&pages, 0, PAGE), (1, 1, 0, 0));
        assert_eq!(xmem::XMemMapPhysicalPages(a, 2, std::ptr::null()), a);
        free(&pages);
    }

    fn spent() -> Spent {
        scheduler().unwrap().lock().spent
    }

    /// Each stage's time shows in the totals behind the debug summary: zlib
    /// reads from a file into memory, BCPack, and raw copies into physical
    /// pages (several of each, so that none rounds down to nothing).
    #[test]
    fn each_stage_is_timed() {
        with_test_tables();
        let plain = pattern(100_000);
        let packed = miniz_oxide::deflate::compress_to_vec_zlib(&plain, 6);
        let mut contents = packed.clone();
        contents.extend_from_slice(&BC4_STREAM);
        let (_dir, file) = file_with(&contents);
        let pages = allocate(1);
        let before = spent();
        let (queue, array) = (new_queue(0), status_array(1));
        let mut outs = vec![vec![0u8; plain.len()]; 10];
        let mut blocks = [[0u8; 16]; 10];
        let mut arrays = vec![pages.clone(); 10];
        for i in 0..10 {
            let at = outs[i].as_mut_ptr().cast();
            enqueue(
                queue,
                &request(ZLIB, at, plain.len(), file, 0, packed.len()),
            );
            let at = blocks[i].as_mut_ptr().cast();
            let offset = packed.len() as u64;
            enqueue(
                queue,
                &request(4 << 1, at, 16, file, offset, BC4_STREAM.len()),
            );
            let (options, at) = (PAGE_DESTINATION | MEMORY_SOURCE, arrays[i].as_mut_ptr());
            let mut r = request(options, at.cast(), 100, plain.as_ptr().cast(), 0, 100);
            r.page_offset = (i * 100) as u64;
            enqueue(queue, &r);
        }
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        assert!(outs.iter().all(|out| *out == plain));
        assert!(blocks.iter().all(|b| *b == bc4_blocks()));
        assert_eq!(page_bytes(&pages)[..1000], plain[..100].repeat(10));
        let after = spent();
        for stage in [
            Stage::Read,
            Stage::Inflate,
            Stage::Bcpack,
            Stage::Destination,
            Stage::Completion,
        ] {
            let i = stage as usize;
            assert!(after[i] > before[i], "stage {i}");
        }
        void(queue, 0x40);
        free(&pages);
    }

    /// Ten thousand requests into pages map no view and commit nothing once
    /// each page has been written once: they copy into xmem's view of the
    /// allocation.
    #[test]
    fn page_requests_map_nothing_once_pages_are_committed() {
        const PAGES: usize = 16;
        const REQUESTS: usize = 10_000;
        let pages = allocate(PAGES);
        let plain = pattern(PAGES * PAGE);
        let (queue, array) = (new_queue(1), status_array(2));
        let options = PAGE_DESTINATION | MEMORY_SOURCE;
        // A first request per page commits it.
        let mut first = pages.clone();
        for p in 0..PAGES {
            let (at, from) = (first[p..].as_mut_ptr(), plain[p * PAGE..].as_ptr());
            enqueue(queue, &request(options, at.cast(), 8, from.cast(), 0, 8));
        }
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        let committed = xmem::section_activity(pages[0]).unwrap();
        assert_eq!(committed, (1, PAGES as u64), "our view, a commit per page");
        // Then every request writes bytes of its own, some across two pages.
        let chunk = PAGES * PAGE / REQUESTS;
        let mut arrays = pages.clone();
        for k in 0..REQUESTS {
            let at = k * chunk;
            let to = arrays[at / PAGE..].as_mut_ptr().cast();
            let mut r = request(options, to, chunk, plain[at..].as_ptr().cast(), 0, chunk);
            r.page_offset = (at % PAGE) as u64;
            enqueue(queue, &r);
        }
        enqueue_status(queue, array, 1);
        void(queue, 0x30);
        assert_eq!(wait(array, 1), S_OK);
        assert_eq!(xmem::section_activity(pages[0]), Some(committed));
        let seen = page_bytes(&pages);
        assert_eq!(seen[..REQUESTS * chunk], plain[..REQUESTS * chunk]);
        void(queue, 0x40);
        free(&pages);
    }

    /// One worker's buffers across requests of growing, then shrinking,
    /// sizes: they only grow, and serve scattered pages over two
    /// allocations too. Past `KEEP` they are given back.
    #[test]
    fn a_worker_keeps_its_buffers() {
        with_test_tables();
        let mut worker = Worker::default();
        let capacities = |w: &Worker| {
            [
                w.source.capacity(),
                w.intermediate.capacity(),
                w.scatter.capacity(),
            ]
        };
        let mut last = capacities(&worker);
        let mut check = |w: &Worker| {
            let now = capacities(w);
            assert!(
                now.iter().zip(last).all(|(&n, l)| n >= l),
                "{now:?} after {last:?}"
            );
            last = now;
        };
        let mut streams = Vec::new();
        let mut contents = Vec::new();
        for size in [1_000, 30_000, 200_000, 50_000, 2_000] {
            let plain = pattern(size);
            let packed = miniz_oxide::deflate::compress_to_vec_zlib(&plain, 6);
            streams.push((plain, contents.len() as u64, packed.len()));
            contents.extend_from_slice(&packed);
        }
        let (_dir, file) = file_with(&contents);
        for (plain, offset, packed) in &streams {
            let mut out = vec![0u8; plain.len()];
            let to = out.as_mut_ptr().cast();
            let r = request(ZLIB, to, out.len(), file, *offset, *packed);
            assert_eq!(execute(&mut Read::new(r), &mut worker), Ok(()));
            assert_eq!(out, *plain);
            check(&worker);
        }
        assert_eq!(worker.source.capacity(), streams[2].2, "the largest source");
        // zlib, then BCPack: through the intermediate buffer.
        let packed = miniz_oxide::deflate::compress_to_vec_zlib(&BC4_STREAM, 6);
        let mut blocks = [0u8; 16];
        let options = ZLIB | 4 << 1 | MEMORY_SOURCE;
        let (to, from) = (blocks.as_mut_ptr().cast(), packed.as_ptr().cast());
        let mut r = request(options, to, 16, from, 0, packed.len());
        r.intermediate_size = BC4_STREAM.len() as u32;
        assert_eq!(execute(&mut Read::new(r), &mut worker), Ok(()));
        assert_eq!(blocks, bc4_blocks());
        check(&worker);
        // Scattered pages, more then fewer bytes, over two allocations.
        let (a, b) = (allocate(2), allocate(2));
        for (size, offset) in [(PAGE + 100, 70), (300, PAGE - 150)] {
            let plain = pattern(size);
            let packed = miniz_oxide::deflate::compress_to_vec_zlib(&plain, 6);
            let mut pages = [a[1], b[0], b[1]];
            let options = ZLIB | PAGE_DESTINATION | MEMORY_SOURCE;
            let (to, from) = (pages.as_mut_ptr().cast(), packed.as_ptr().cast());
            let mut r = request(options, to, size, from, 0, packed.len());
            r.page_offset = offset as u64;
            assert_eq!(execute(&mut Read::new(r), &mut worker), Ok(()));
            assert_eq!(page_bytes(&pages)[offset..offset + size], plain);
            check(&worker);
        }
        assert_eq!(worker.scatter.capacity(), PAGE + 100);
        // What grew past the limit is given back after its request.
        worker.source = vec![0; KEEP + 1];
        worker.scatter = vec![0; KEEP];
        worker.trim();
        assert_eq!(worker.source.capacity(), 0);
        assert_eq!(worker.scatter.capacity(), KEEP);
        free(&a);
        free(&b);
    }

    /// The title enqueues a request with its page array still holding the
    /// pages of the array's last use, then writes this request's pages
    /// there before it submits: the request lands in those.
    #[test]
    fn page_arrays_are_read_when_the_request_runs() {
        let (a, b) = (allocate(2), allocate(2));
        let plain = pattern(100);
        let (queue, array) = (new_queue(1), status_array(1));
        let mut frame = a.clone();
        let options = PAGE_DESTINATION | MEMORY_SOURCE;
        let (to, from) = (frame.as_mut_ptr().cast(), plain.as_ptr().cast());
        let mut r = request(options, to, 100, from, 0, 100);
        r.page_offset = (PAGE - 40) as u64; // across the page boundary
        enqueue(queue, &r);
        // Nothing runs before the submission (half the queue is 64).
        frame.copy_from_slice(&b);
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        assert_eq!(page_bytes(&b)[PAGE - 40..PAGE + 60], plain);
        assert!(page_bytes(&a).iter().all(|&x| x == 0), "the old pages");
        void(queue, 0x40);
        free(&a);
        free(&b);
    }

    /// An entry still 0 when the request is enqueued is read when it runs:
    /// the title fills it in before it submits.
    #[test]
    fn a_page_filled_in_after_enqueue_is_read_again() {
        let (a, b) = (allocate(1), allocate(1));
        let plain = pattern(100);
        let (queue, array) = (new_queue(1), status_array(1));
        let mut pages = [a[0], 0];
        let options = PAGE_DESTINATION | MEMORY_SOURCE;
        let entries = pages.as_mut_ptr();
        let mut r = request(options, entries.cast(), 100, plain.as_ptr().cast(), 0, 100);
        r.page_offset = (PAGE - 40) as u64;
        enqueue(queue, &r);
        // SAFETY: our array, which the request reads only once submitted.
        unsafe { entries.add(1).write(b[0]) };
        enqueue_status(queue, array, 0);
        void(queue, 0x30);
        assert_eq!(wait(array, 0), S_OK);
        assert_eq!(page_bytes(&a)[PAGE - 40..], plain[..40]);
        assert_eq!(page_bytes(&b)[..60], plain[40..]);
        void(queue, 0x40);
        free(&a);
        free(&b);
    }

    /// Bytes unique to one destination.
    fn chunk(seed: usize, n: usize) -> Vec<u8> {
        (0..n)
            .map(|i| (i * 7 + seed * 13 + i / 251) as u8)
            .collect()
    }

    /// Every worker at once, from four queues on four threads: requests
    /// into pages a title view shows read-only (as the GPU side's copies
    /// guard views), pages a title view shows read-write, pages no view
    /// shows, across two allocations, and into title memory at an address.
    /// Each lands, and the guard stays.
    #[test]
    fn many_workers_into_guarded_and_unguarded_pages() {
        use windows_sys::Win32::System::Memory::{
            MEMORY_BASIC_INFORMATION, PAGE_READONLY, VirtualProtect, VirtualQuery,
        };
        const PAGES: usize = 256;
        const HALF: usize = PAGES / 2;
        const THREADS: usize = 4;
        const BLOCK: usize = 4096;
        let (a, b) = (allocate(PAGES), allocate(PAGES));
        // The first half of each allocation is mapped; `a`'s guarded.
        let (guarded, unguarded) = (reserve(HALF), reserve(HALF));
        assert_eq!(
            xmem::XMemMapPhysicalPages(guarded, HALF, a.as_ptr()),
            guarded
        );
        let mapped = xmem::XMemMapPhysicalPages(unguarded, HALF, b.as_ptr());
        assert_eq!(mapped, unguarded);
        let mut old = 0;
        // SAFETY: our title view; only its protection changes.
        let protected = unsafe { VirtualProtect(guarded, HALF * PAGE, PAGE_READONLY, &mut old) };
        assert_ne!(protected, 0);
        let contents = pattern(PAGES * BLOCK);
        let (_dir, file) = file_with(&contents);
        let file = file as usize;
        let mut memory = vec![0u8; PAGES * BLOCK];
        let memory_at = memory.as_mut_ptr() as usize;
        std::thread::scope(|s| {
            for t in 0..THREADS {
                let (a, b) = (&a, &b);
                s.spawn(move || {
                    let (queue, array) = (new_queue(t as u64 % 2), status_array(1));
                    // Sources and page arrays live until completion.
                    let (mut sources, mut arrays) = (Vec::new(), Vec::new());
                    for p in (t..PAGES).step_by(THREADS) {
                        // Half of `a[p]`, in place: zlib or a raw copy.
                        let plain = chunk(2 * p, PAGE / 2);
                        let (options, source) = if p % 2 == 0 {
                            let packed = miniz_oxide::deflate::compress_to_vec_zlib(&plain, 1);
                            (ZLIB | PAGE_DESTINATION | MEMORY_SOURCE, packed)
                        } else {
                            (PAGE_DESTINATION | MEMORY_SOURCE, plain)
                        };
                        arrays.push(Box::new([a[p], b[p]]));
                        let to = arrays.last_mut().unwrap().as_mut_ptr().cast();
                        let from = source.as_ptr().cast();
                        enqueue(
                            queue,
                            &request(options, to, PAGE / 2, from, 0, source.len()),
                        );
                        sources.push(source);
                        // The other half of `a[p]` and half of `b[p]`.
                        let plain = chunk(2 * p + 1, PAGE);
                        let options = PAGE_DESTINATION | MEMORY_SOURCE;
                        arrays.push(Box::new([a[p], b[p]]));
                        let to = arrays.last_mut().unwrap().as_mut_ptr().cast();
                        let mut r = request(options, to, PAGE, plain.as_ptr().cast(), 0, PAGE);
                        r.page_offset = (PAGE / 2) as u64;
                        enqueue(queue, &r);
                        sources.push(plain);
                        // A block of title memory, from the file.
                        let to = (memory_at + p * BLOCK) as *mut c_void;
                        let (from, offset) = (file as *const c_void, (p * BLOCK) as u64);
                        enqueue(queue, &request(0, to, BLOCK, from, offset, BLOCK));
                    }
                    enqueue_status(queue, array, 0);
                    void(queue, 0x30);
                    assert_eq!(wait(array, 0), S_OK);
                    void(queue, 0x40);
                });
            }
        });
        let (seen_a, seen_b) = (page_bytes(&a), page_bytes(&b));
        for p in 0..PAGES {
            let page = &seen_a[p * PAGE..(p + 1) * PAGE];
            assert_eq!(page[..PAGE / 2], chunk(2 * p, PAGE / 2), "page {p}");
            let across = chunk(2 * p + 1, PAGE);
            assert_eq!(page[PAGE / 2..], across[..PAGE / 2], "page {p}");
            let next = &seen_b[p * PAGE..][..PAGE / 2];
            assert_eq!(next, &across[PAGE / 2..], "page {p}");
        }
        assert_eq!(memory, contents);
        // Through the title's views too; the guarded one still read-only.
        // SAFETY: both views are mapped, readable.
        unsafe {
            let view = std::slice::from_raw_parts(guarded.cast::<u8>(), HALF * PAGE);
            assert_eq!(view, &seen_a[..HALF * PAGE]);
            let view = std::slice::from_raw_parts(unguarded.cast::<u8>(), HALF * PAGE);
            assert_eq!(view, &seen_b[..HALF * PAGE]);
        }
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        unsafe { VirtualQuery(guarded, &mut info, size_of_val(&info)) };
        assert_eq!(info.Protect, PAGE_READONLY);
        for view in [guarded, unguarded] {
            let unmapped = xmem::XMemMapPhysicalPages(view, HALF, std::ptr::null());
            assert_eq!(unmapped, view);
        }
        free(&a);
        free(&b);
    }

    /// The read trace keeps every request for `1`, else those whose source
    /// contains the filter (the filter is lower case, the path need not be).
    #[test]
    fn the_read_trace_keeps_the_sources_its_filter_names() {
        assert!(trace_keeps("1", r"G:\data.rpf"));
        assert!(trace_keeps("common.rpf", r"G:\COMMON.RPF"));
        assert!(trace_keeps("memory", "memory 0x140400000"));
        assert!(!trace_keeps("common.rpf", r"G:\levels.rpf"));
        assert!(!trace_keeps("common.rpf", "memory 0x140400000"));
    }

    #[test]
    fn worker_count_prefers_physical_cores_and_respects_override() {
        assert_eq!(worker_count_for(None, Some(8), 16), 8);
        assert_eq!(worker_count_for(None, Some(64), 128), 32);
        assert_eq!(worker_count_for(None, None, 16), 8);
        assert_eq!(worker_count_for(None, None, 2), 2);
        assert_eq!(worker_count_for(Some("12"), Some(8), 16), 12);
        assert_eq!(worker_count_for(Some("64"), Some(8), 16), 64);
        assert_eq!(worker_count_for(Some("0"), Some(8), 16), 8);
        assert_eq!(worker_count_for(Some("65"), Some(8), 16), 8);
        assert_eq!(worker_count_for(Some("invalid"), Some(8), 16), 8);
        #[cfg(windows)]
        {
            use windows_sys::Win32::System::Threading::{
                ALL_PROCESSOR_GROUPS, GetActiveProcessorCount,
            };
            let cores = physical_cores().expect("physical core count on Windows");
            // SAFETY: query-only count of active logical processors across every group.
            let logical = unsafe { GetActiveProcessorCount(ALL_PROCESSOR_GROUPS) } as usize;
            assert!((1..=logical).contains(&cores));
        }
    }
}
