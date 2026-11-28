//! GPU timestamps per list and per command, behind `XTP_GPU_TIMESTAMPS=1`
//! (off by default): where a frame's GPU time goes, by kind of command and
//! by program, and where each submission sits on the CPU's clock.
//!
//! A host list's recording gets a timestamp when it opens ([`open`]: the
//! list made or reset), one before each command a caller marks
//! ([`command`]: draws, dispatches, mesh dispatches, indirect executions,
//! XTP's own passes, texture uploads) and one after it, and one when it
//! closes ([`close`]), which resolves them into a read-back buffer at the
//! list's end. The time from one timestamp to the next goes to what the
//! first one starts: the marked command, or "between commands" for what
//! the list records outside them (copies, barriers, clears, and what comes
//! before the first mark). Once a submission ([`submitted`]) has finished
//! (`Host::after`, after its copy-back), its timestamps are read and summed:
//! per kind of command, per program (named as host pipelines are, by their
//! translated shaders' hashes) and per submission. A submission's first and
//! last timestamps, put on the CPU's clock (`GetClockCalibration`), say how
//! long it waited in the queue, how long the GPU sat idle before it, when
//! it ended after it was submitted and when it was read after it ended.
//! Every few seconds (`XTP_FRAME_LINE`, as the frame line) the sums are
//! logged per frame; with `xtp_gpu::timestamps=debug`, a line per
//! submission too.
//!
//! Timestamps come from one query heap and one read-back buffer, in chunks
//! of [`CHUNK`] that a recording takes as it goes. They go back once the
//! list has been reset or dropped and every submission of the old recording
//! has been read: a list reset while its submission runs records into other
//! chunks. A recording that finds no chunk left and a list submitted without
//! a closed recording are not timed (both counted on the line).
//!
//! shortcut: a timestamp is taken when the GPU has finished the commands
//! recorded before it, so commands that overlap on the GPU share out their
//! time in recorded order: a list's and a submission's sums are exact, the
//! split between commands is approximate, as for any profiler that times
//! between commands. The timestamps may also cost some of that overlap:
//! `docs/analysis/gpu-frame.md` compares frames with and without them.
//!
//! shortcut: a list submitted again without a reset is timed again; if both
//! runs finished before the first was read, both read the second's
//! timestamps.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::core::Interface;

use crate::host::HostList;
use crate::pipeline::host::HostPipeline;

/// Timestamps per chunk.
pub(crate) const CHUNK: u32 = 256;
/// Chunks in the heap: 512 Ki timestamps, 4 MiB of query heap and as much
/// read-back memory. A world frame takes about two per marked command.
const CHUNKS: u32 = 2048;

/// Whether `value` (of `XTP_GPU_TIMESTAMPS`) turns timestamps on: set,
/// and neither empty nor `0`.
fn switch(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_some_and(|v| !v.is_empty() && v != "0")
}

/// Whether timestamps are taken (`XTP_GPU_TIMESTAMPS=1`).
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    let on = *ON.get_or_init(|| {
        let on = switch(std::env::var_os("XTP_GPU_TIMESTAMPS").as_deref());
        if on {
            tracing::info!("GPU timestamps on (XTP_GPU_TIMESTAMPS)");
        }
        on
    });
    #[cfg(test)]
    let on = on || FORCED.with(std::cell::Cell::get);
    on
}

#[cfg(test)]
thread_local! {
    /// Tests turn timestamps on for the lists their own thread records.
    static FORCED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// What the time from a timestamp to the next one is spent on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum What {
    /// What a list records outside marked commands: copies, barriers,
    /// clears, state.
    Between,
    Dispatch,
    /// A mesh dispatch.
    Mesh,
    Draw,
    /// An indexed draw.
    Indexed,
    /// An indirect execution of the title's, by argument kind (0 draw, 1
    /// indexed draw, 2 dispatch, 10 mesh dispatch).
    Indirect(u32),
    /// XTP's pass that bounds an indirect execution's arguments.
    Clamp,
    /// XTP's passes that expand indirect rectangle lists (`rectangles`).
    RectangleExpansion,
    RaytracingBuild,
    /// The indirect draw of expanded rectangles.
    Rectangles,
    /// A texture's staging copy into its host copy (`texture::upload`).
    TextureUpload,
}

impl std::fmt::Display for What {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            What::Between => f.write_str("between commands"),
            What::Dispatch => f.write_str("dispatch"),
            What::Mesh => f.write_str("mesh dispatch"),
            What::Draw => f.write_str("draw"),
            What::Indexed => f.write_str("indexed draw"),
            What::Indirect(0) => f.write_str("indirect draw"),
            What::Indirect(1) => f.write_str("indirect indexed draw"),
            What::Indirect(2) => f.write_str("indirect dispatch"),
            What::Indirect(10) => f.write_str("indirect mesh dispatch"),
            What::Indirect(kind) => write!(f, "indirect kind {kind}"),
            What::Clamp => f.write_str("indirect argument clamp"),
            What::RaytracingBuild => f.write_str("host DXR build"),
            What::RectangleExpansion => f.write_str("rectangle expansion"),
            What::Rectangles => f.write_str("indirect rectangles"),
            What::TextureUpload => f.write_str("texture upload"),
        }
    }
}

/// What a timestamp starts: a kind and the program it runs (0: none).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Label {
    what: What,
    program: u32,
}

const BETWEEN: Label = Label {
    what: What::Between,
    program: 0,
};

/// Free chunks of the heap.
struct Pool {
    free: Mutex<Vec<u32>>,
}

impl Pool {
    fn new(chunks: u32) -> Pool {
        // Taken from the end: the first chunks first.
        Pool {
            free: Mutex::new((0..chunks).rev().collect()),
        }
    }

    fn take(&self) -> Option<u32> {
        self.free.lock().unwrap_or_else(|e| e.into_inner()).pop()
    }

    fn give_back(&self, chunks: &[u32]) {
        self.free
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .extend_from_slice(chunks);
    }

    #[cfg(test)]
    fn left(&self) -> usize {
        self.free.lock().unwrap_or_else(|e| e.into_inner()).len()
    }
}

/// The query heap, its read-back buffer (mapped for good) and its chunks.
struct Heap {
    queries: ID3D12QueryHeap,
    back: ID3D12Resource,
    data: *const u64,
    pool: Pool,
}

// SAFETY: D3D12 objects are free-threaded; `data` is a persistent mapping
// of a read-back buffer, read only where finished submissions resolved.
unsafe impl Send for Heap {}
// SAFETY: as above.
unsafe impl Sync for Heap {}

static HEAP: OnceLock<Option<Heap>> = OnceLock::new();

/// The heap, made on first use (`None` without a host device or if it
/// could not be made, logged once).
fn heap() -> Option<&'static Heap> {
    HEAP.get_or_init(|| {
        let device = &crate::host::made()?.device;
        match make_heap(device) {
            Ok(heap) => Some(heap),
            Err(e) => {
                tracing::warn!(error = %e, "no GPU timestamps: their query heap was not made");
                None
            }
        }
    })
    .as_ref()
}

fn make_heap(device: &ID3D12Device) -> windows::core::Result<Heap> {
    let count = CHUNKS * CHUNK;
    let mut queries: Option<ID3D12QueryHeap> = None;
    let mut back: Option<ID3D12Resource> = None;
    let readback = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_READBACK,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: u64::from(count) * 8,
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
    let mut data = std::ptr::null_mut();
    // SAFETY: plain object creation; the mapping is of the whole buffer.
    unsafe {
        device.CreateQueryHeap(
            &D3D12_QUERY_HEAP_DESC {
                Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                Count: count,
                NodeMask: 0,
            },
            &mut queries,
        )?;
        device.CreateCommittedResource(
            &readback,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut back,
        )?;
        let back = back.as_ref().ok_or(windows::core::Error::empty())?;
        back.Map(0, None, Some(&mut data))?;
    }
    let (Some(queries), Some(back)) = (queries, back) else {
        return Err(windows::core::Error::empty());
    };
    crate::host::name(&back, || "GPU timestamps read-back".into());
    Ok(Heap {
        queries,
        back,
        data: data.cast::<u64>().cast_const(),
        pool: Pool::new(CHUNKS),
    })
}

/// One recording of a list: the chunks its timestamps are in, and what
/// each timestamp starts (the last one, the close, starts nothing).
struct Recording {
    pool: &'static Pool,
    chunks: Vec<u32>,
    labels: Vec<Label>,
    /// No chunk was left for a timestamp: not timed.
    lost: bool,
}

impl Recording {
    fn new(pool: &'static Pool) -> Recording {
        Recording {
            pool,
            chunks: Vec::new(),
            labels: Vec::new(),
            lost: false,
        }
    }

    /// The query index of the next timestamp, labelled `label`; `None` once
    /// the pool is out of chunks (the recording is then lost).
    fn next(&mut self, label: Label) -> Option<u32> {
        if self.lost {
            return None;
        }
        let k = self.labels.len() as u32;
        if k.is_multiple_of(CHUNK) {
            match self.pool.take() {
                Some(chunk) => self.chunks.push(chunk),
                None => {
                    self.lost = true;
                    LOST.fetch_add(1, Ordering::Relaxed);
                    return None;
                }
            }
        }
        self.labels.push(label);
        Some(self.index(k))
    }

    /// The query index of timestamp `k`.
    fn index(&self, k: u32) -> u32 {
        self.chunks[(k / CHUNK) as usize] * CHUNK + k % CHUNK
    }

    /// Each chunk's first query index and how many of its timestamps the
    /// recording took.
    fn ranges(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        let n = self.labels.len() as u32;
        self.chunks
            .iter()
            .enumerate()
            .map(move |(i, &c)| (c * CHUNK, (n - i as u32 * CHUNK).min(CHUNK)))
    }
}

impl Drop for Recording {
    fn drop(&mut self) {
        self.pool.give_back(&self.chunks);
    }
}

/// A list's recording: being recorded, or closed (shared with the
/// submissions that run it).
enum Entry {
    Open(Recording),
    Closed(Arc<Recording>),
}

/// Recordings by host list (its D3D12 list, which stays where it is when
/// the `HostList` moves).
static LISTS: Mutex<Option<HashMap<usize, Entry>>> = Mutex::new(None);

fn key(list: &HostList) -> usize {
    list.list.as_raw() as usize
}

/// Recordings that found no chunk left, and lists submitted without a
/// closed recording, since the last line.
static LOST: AtomicU64 = AtomicU64::new(0);
static UNTIMED: AtomicU64 = AtomicU64::new(0);

/// Adds a timestamp labelled `label` to `list`'s recording, which starts
/// here if it has none open (a list made before the first [`open`]).
fn stamp(list: &HostList, label: Label) {
    let Some(heap) = heap() else { return };
    let mut lists = LISTS.lock().unwrap_or_else(|e| e.into_inner());
    let entry = lists
        .get_or_insert_with(HashMap::new)
        .entry(key(list))
        .or_insert_with(|| Entry::Open(Recording::new(&heap.pool)));
    if let Entry::Closed(_) = entry {
        *entry = Entry::Open(Recording::new(&heap.pool));
    }
    let Entry::Open(recording) = entry else {
        return;
    };
    if let Some(index) = recording.next(label) {
        // SAFETY: an open list and our heap; the index is in a chunk the
        // recording holds.
        unsafe {
            list.list
                .EndQuery(&heap.queries, D3D12_QUERY_TYPE_TIMESTAMP, index)
        };
    }
}

/// A list opens for recording (made, or reset): its recording starts. The
/// last one goes once its submissions have been read.
pub(crate) fn open(list: &HostList) {
    if !on() {
        return;
    }
    let Some(heap) = heap() else { return };
    let old = LISTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(HashMap::new)
        .insert(key(list), Entry::Open(Recording::new(&heap.pool)));
    // Its chunks go back outside the lock (unless a submission holds them).
    drop(old);
    stamp(list, BETWEEN);
    crate::pipestats::open(list);
}

/// A list is being dropped (after its submissions settled): forgets its
/// recording so its query chunks return to the pool instead of leaking
/// until another list reuses the COM pointer address.
pub(crate) fn forget(list: &HostList) {
    let old = {
        let mut lists = LISTS.lock().unwrap_or_else(|e| e.into_inner());
        lists.as_mut().and_then(|l| l.remove(&key(list)))
    };
    // Its chunks go back outside the lock.
    drop(old);
    crate::pipestats::forget(list);
}

/// Just before a list is closed: its last timestamp, and the resolve of
/// all of them into the read-back buffer.
pub(crate) fn close(list: &HostList) {
    if !on() {
        return;
    }
    let Some(heap) = heap() else { return };
    stamp(list, BETWEEN);
    crate::pipestats::close(list);
    let mut lists = LISTS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(entry) = lists.as_mut().and_then(|l| l.get_mut(&key(list))) else {
        return;
    };
    let Entry::Open(recording) = entry else {
        return;
    };
    let recording = std::mem::replace(recording, Recording::new(&heap.pool));
    if !recording.lost {
        for (first, count) in recording.ranges() {
            // SAFETY: an open list, our heap and read-back buffer; the
            // queries were ended in this list, the destination is theirs.
            unsafe {
                list.list.ResolveQueryData(
                    &heap.queries,
                    D3D12_QUERY_TYPE_TIMESTAMP,
                    first,
                    count,
                    &heap.back,
                    u64::from(first) * 8,
                )
            };
        }
    }
    *entry = Entry::Closed(Arc::new(recording));
}

/// A marked command: a timestamp before it ([`command`]) and one after it
/// when this goes.
#[must_use = "the command's time ends when this goes"]
pub(crate) struct Command<'a>(Option<&'a HostList>);

impl Drop for Command<'_> {
    fn drop(&mut self) {
        if let Some(list) = self.0 {
            crate::pipestats::end(list);
            stamp(list, BETWEEN);
        }
    }
}

/// Marks the command recorded next into `list` as `what`, running
/// `pipeline`'s programs if it has one: its time goes to them until the
/// returned [`Command`] goes.
pub(crate) fn command<'a>(
    list: &'a HostList,
    what: What,
    pipeline: Option<&HostPipeline>,
) -> Command<'a> {
    if !on() {
        return Command(None);
    }
    let program = pipeline.map_or(0, |p| program(&p.name, || wave64(p)));
    stamp(list, Label { what, program });
    crate::pipestats::begin(list, what, pipeline);
    Command(Some(list))
}

/// Whether one of a pipeline's console programs runs in 64-lane waves
/// (on a 32-lane host, as two 32-lane waves).
fn wave64(p: &HostPipeline) -> bool {
    p.stages
        .iter()
        .any(|s| s.shader.translated().is_some_and(|t| t.wave64))
}

/// Programs by name (host pipelines' names: their shaders' hashes) and
/// whether they are wave64; 0 is none.
#[derive(Default)]
struct Programs {
    ids: HashMap<String, u32>,
    names: Vec<(String, bool)>,
}

static PROGRAMS: Mutex<Option<Programs>> = Mutex::new(None);

/// The number of the program named `name` (`wave64` asked the first time).
fn program(name: &str, wave64: impl FnOnce() -> bool) -> u32 {
    let mut programs = PROGRAMS.lock().unwrap_or_else(|e| e.into_inner());
    let programs = programs.get_or_insert_with(|| Programs {
        ids: HashMap::new(),
        names: vec![(String::new(), false)],
    });
    if let Some(&id) = programs.ids.get(name) {
        return id;
    }
    let id = programs.names.len() as u32;
    programs.names.push((name.to_owned(), wave64()));
    programs.ids.insert(name.to_owned(), id);
    id
}

/// The CPU's and the GPU's clocks at one moment, and their rates.
#[derive(Debug, Clone, Copy)]
struct Clock {
    gpu_hz: u64,
    qpc_hz: u64,
    gpu: u64,
    qpc: u64,
}

impl Clock {
    /// GPU timestamp `t` on the CPU's clock (`QueryPerformanceCounter`).
    fn qpc(&self, t: u64) -> f64 {
        self.qpc as f64 + (t as f64 - self.gpu as f64) * self.qpc_hz as f64 / self.gpu_hz as f64
    }

    /// From `a` to `b` on the CPU's clock, in microseconds.
    fn us(&self, a: f64, b: f64) -> f64 {
        (b - a) * 1e6 / self.qpc_hz as f64
    }

    /// GPU ticks in milliseconds.
    fn ms(&self, ticks: u64) -> f64 {
        ticks as f64 * 1e3 / self.gpu_hz as f64
    }
}

/// The last calibration and when it was made.
static CLOCK: Mutex<Option<(Clock, Instant)>> = Mutex::new(None);

/// The clocks, calibrated again every second on `queue`.
fn clock(queue: &ID3D12CommandQueue) -> Option<Clock> {
    let mut clock = CLOCK.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((c, at)) = *clock
        && at.elapsed() < Duration::from_secs(1)
    {
        return Some(c);
    }
    let (mut gpu, mut qpc, mut qpc_hz) = (0u64, 0u64, 0i64);
    // SAFETY: plain queries into locals.
    let gpu_hz = unsafe {
        let gpu_hz = queue.GetTimestampFrequency().ok()?;
        queue.GetClockCalibration(&mut gpu, &mut qpc).ok()?;
        windows_sys::Win32::System::Performance::QueryPerformanceFrequency(&mut qpc_hz);
        gpu_hz
    };
    let c = Clock {
        gpu_hz: gpu_hz.max(1),
        qpc_hz: (qpc_hz as u64).max(1),
        gpu,
        qpc,
    };
    *clock = Some((c, Instant::now()));
    Some(c)
}

fn qpc_now() -> u64 {
    let mut now = 0i64;
    // SAFETY: a plain query into a local.
    unsafe { windows_sys::Win32::System::Performance::QueryPerformanceCounter(&mut now) };
    now as u64
}

/// A submission whose timestamps are read once it has finished.
struct Submission {
    value: u64,
    /// The title's (else ours: the present's copy, planes, captures).
    title: bool,
    /// When it was submitted, on the CPU's clock.
    at: u64,
    /// The title's frame then (`frame::current`).
    frame: u64,
    clock: Clock,
    /// Its lists' recordings, in order (`None`: not timed).
    lists: Vec<Option<(usize, Arc<Recording>)>>,
    /// The thread that submitted it (tests: whose lists were timed).
    #[cfg(test)]
    thread: std::thread::ThreadId,
}

/// Host submissions made so far, timed or not.
static SUBMISSIONS: AtomicU64 = AtomicU64::new(0);

/// How many host submissions were made so far: while one is prepared, the
/// number of those before it (`texture` counts its checks by it).
pub(crate) fn submissions() -> u64 {
    SUBMISSIONS.load(Ordering::Acquire)
}

/// Just after a submission of `runs` that signals `value` on `queue`: its
/// timestamps are read once it has finished.
pub(crate) fn submitted(queue: &ID3D12CommandQueue, runs: &[&HostList], value: u64, title: bool) {
    // Counted whether timestamps are on or not, so before the check below:
    // `texture`'s diagnostics count its checks per submission by this count,
    // and timestamps are off by default
    // (`submissions_are_counted_with_timestamps_off`).
    SUBMISSIONS.fetch_add(1, Ordering::Release);
    if !on() {
        return;
    }
    let (Some(host), Some(clock)) = (crate::host::made(), clock(queue)) else {
        return;
    };
    let lists: Vec<Option<(usize, Arc<Recording>)>> = {
        let lists = LISTS.lock().unwrap_or_else(|e| e.into_inner());
        runs.iter()
            .map(|l| match lists.as_ref().and_then(|m| m.get(&key(l))) {
                Some(Entry::Closed(r)) => Some((key(l), r.clone())),
                _ => None,
            })
            .collect()
    };
    let untimed = lists.iter().filter(|l| l.is_none()).count();
    UNTIMED.fetch_add(untimed as u64, Ordering::Relaxed);
    let submission = Submission {
        value,
        title,
        at: qpc_now(),
        frame: crate::frame::current(),
        clock,
        lists,
        #[cfg(test)]
        thread: std::thread::current().id(),
    };
    host.after(move || read(submission));
    log_due();
    crate::pipestats::submitted(runs);
}

/// One list's timestamps, read: what each span went to and how long it
/// took (GPU ticks), and the list's first and last timestamps.
#[derive(Debug, Clone, PartialEq)]
struct Spans {
    spans: Vec<(Label, u64)>,
    first: u64,
    last: u64,
}

/// The spans between consecutive `stamps`, each labelled as the timestamp
/// that starts it; `None` without two timestamps or if one was never
/// written (0) or goes back in time.
fn spans(stamps: &[u64], labels: &[Label]) -> Option<Spans> {
    let (&first, &last) = (stamps.first()?, stamps.last()?);
    if stamps.len() < 2 || stamps.len() != labels.len() {
        return None;
    }
    if stamps.contains(&0) || stamps.windows(2).any(|w| w[1] < w[0]) {
        return None;
    }
    let spans = stamps
        .windows(2)
        .zip(labels)
        .map(|(w, &label)| (label, w[1] - w[0]))
        .collect();
    Some(Spans { spans, first, last })
}

/// Reads a finished submission's timestamps into the sums (on the thread
/// that finished it, under the timeline's lock: no waits here).
fn read(submission: Submission) {
    let Some(heap) = heap() else { return };
    let read_at = qpc_now();
    let mut lists = Vec::with_capacity(submission.lists.len());
    // Lost recordings, and timestamps that did not read right (those
    // without a recording were counted when submitted).
    let mut bad = 0;
    for (key, recording) in submission.lists.iter().flatten() {
        if recording.lost {
            bad += 1;
            continue;
        }
        let stamps: Vec<u64> = (0..recording.labels.len() as u32)
            .map(|k| {
                // SAFETY: the query index is in the read-back buffer, which
                // the finished submission resolved into.
                unsafe { heap.data.add(recording.index(k) as usize).read_volatile() }
            })
            .collect();
        match spans(&stamps, &recording.labels) {
            Some(s) => lists.push((*key, s)),
            None => bad += 1,
        }
    }
    let c = submission.clock;
    let busy: u64 = lists.iter().map(|(_, s)| s.last - s.first).sum();
    let first = lists.iter().map(|(_, s)| s.first).min();
    let last = lists.iter().map(|(_, s)| s.last).max();
    let mut sums = SUMS.lock().unwrap_or_else(|e| e.into_inner());
    let sums = sums.get_or_insert_with(Sums::new);
    sums.untimed += bad;
    let (Some(first), Some(last)) = (first, last) else {
        return;
    };
    let class = &mut sums.classes[usize::from(!submission.title)];
    let at = submission.at as f64;
    let (queued, ended, late) = (
        c.us(at, c.qpc(first)),
        c.us(at, c.qpc(last)),
        c.us(c.qpc(last), read_at as f64),
    );
    class.add(lists.len(), busy, last - first, (queued, ended, late));
    if let Some(end) = sums.last_end
        && first > end
    {
        sums.idle += first - end;
    }
    sums.last_end = Some(sums.last_end.map_or(last, |e| e.max(last)));
    for (_, s) in &lists {
        sums.add(&s.spans);
    }
    tracing::debug!(
        value = submission.value,
        title = submission.title,
        frame = submission.frame,
        lists = submission.lists.len(),
        timed = lists.len(),
        gpu_ms = c.ms(busy),
        span_ms = c.ms(last - first),
        queued_ms = queued / 1e3,
        ended_ms = ended / 1e3,
        read_ms = late / 1e3,
        "GPU timestamps of a submission"
    );
    #[cfg(test)]
    READ.lock()
        .unwrap_or_else(|e| e.into_inner())
        .extend(lists.into_iter().map(|(key, spans)| ReadList {
            thread: submission.thread,
            key,
            spans,
            chunks: recording_chunks(&submission, key),
        }));
}

#[cfg(test)]
fn recording_chunks(submission: &Submission, key: usize) -> Vec<u32> {
    submission
        .lists
        .iter()
        .flatten()
        .find(|(k, _)| *k == key)
        .map(|(_, r)| r.chunks.clone())
        .unwrap_or_default()
}

/// A list read in tests: the thread that submitted it, its key, spans and
/// chunks.
#[cfg(test)]
struct ReadList {
    thread: std::thread::ThreadId,
    key: usize,
    spans: Spans,
    chunks: Vec<u32>,
}

/// Lists read in tests.
#[cfg(test)]
static READ: Mutex<Vec<ReadList>> = Mutex::new(Vec::new());

/// A sum of GPU ticks and of what they were counted over.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
struct Sum {
    ticks: u64,
    n: u64,
}

impl Sum {
    fn add(&mut self, ticks: u64) {
        self.ticks += ticks;
        self.n += 1;
    }
}

/// Submissions of one class (the title's, ours).
#[derive(Debug, Default, Clone, Copy)]
struct Class {
    submissions: u64,
    lists: u64,
    /// Sum of the lists' times, and of the submissions' (first to last
    /// timestamp: with the preludes and gaps between lists).
    busy: u64,
    span: u64,
    /// Sums, in microseconds, from submission to the GPU's first timestamp
    /// and to its last, and from the last to its reading; the longest from
    /// submission to the last.
    queued: f64,
    ended: f64,
    ended_max: f64,
    read: f64,
}

impl Class {
    fn add(&mut self, lists: usize, busy: u64, span: u64, (queued, ended, read): (f64, f64, f64)) {
        self.submissions += 1;
        self.lists += lists as u64;
        self.busy += busy;
        self.span += span;
        self.queued += queued;
        self.ended += ended;
        self.ended_max = self.ended_max.max(ended);
        self.read += read;
    }
}

/// What was read since the last line.
struct Sums {
    since: Instant,
    frame: u64,
    classes: [Class; 2],
    kinds: BTreeMap<What, Sum>,
    programs: HashMap<(u32, What), Sum>,
    /// GPU ticks between a submission's last timestamp and the next one's
    /// first (the GPU idle, or running what is not timed).
    idle: u64,
    last_end: Option<u64>,
    /// Lists not timed or whose timestamps did not read right.
    untimed: u64,
    /// The GPU clock's rate, for the line.
    gpu_hz: u64,
}

impl Sums {
    fn new() -> Sums {
        Sums {
            since: Instant::now(),
            frame: crate::frame::current(),
            classes: [Class::default(); 2],
            kinds: BTreeMap::new(),
            programs: HashMap::new(),
            idle: 0,
            last_end: None,
            untimed: 0,
            gpu_hz: 0,
        }
    }

    fn add(&mut self, spans: &[(Label, u64)]) {
        for &(label, ticks) in spans {
            self.kinds.entry(label.what).or_default().add(ticks);
            if label.program != 0 {
                self.programs
                    .entry((label.program, label.what))
                    .or_default()
                    .add(ticks);
            }
        }
    }
}

static SUMS: Mutex<Option<Sums>> = Mutex::new(None);

/// Seconds between lines: `XTP_FRAME_LINE`, as the frame line (1 to 60,
/// 5 by default).
fn line_every() -> Duration {
    static EVERY: OnceLock<Duration> = OnceLock::new();
    *EVERY.get_or_init(|| {
        let seconds = std::env::var("XTP_FRAME_LINE")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map_or(5, |s| s.clamp(1, 60));
        Duration::from_secs(seconds)
    })
}

/// Logs the sums once they cover [`line_every`], and starts new ones.
fn log_due() {
    let (sums, frames) = {
        let mut sums = SUMS.lock().unwrap_or_else(|e| e.into_inner());
        if sums
            .as_ref()
            .is_none_or(|s| s.since.elapsed() < line_every())
        {
            return;
        }
        let Some(mut done) = sums.replace(Sums::new()) else {
            return;
        };
        // Idle time is counted from the last submission read on.
        if let Some(next) = sums.as_mut() {
            next.last_end = done.last_end;
        }
        done.gpu_hz = CLOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .map_or(0, |(c, _)| c.gpu_hz);
        let frames = crate::frame::current().saturating_sub(done.frame);
        (done, frames)
    };
    if sums.gpu_hz == 0 {
        return;
    }
    let programs = PROGRAMS.lock().unwrap_or_else(|e| e.into_inner());
    let names = programs.as_ref().map_or(&[][..], |p| &p.names[..]);
    for line in lines(
        &sums,
        frames,
        names,
        LOST.swap(0, Ordering::Relaxed),
        UNTIMED.swap(0, Ordering::Relaxed),
    ) {
        tracing::info!("{line}");
    }
}

/// Programs on the line.
const TOP: usize = 25;

/// The lines for `sums` over `frames` frames (at least one): the totals,
/// the kinds, the programs that took most.
fn lines(
    sums: &Sums,
    frames: u64,
    names: &[(String, bool)],
    lost: u64,
    untimed: u64,
) -> Vec<String> {
    let f = frames.max(1) as f64;
    let ms = |ticks: u64| ticks as f64 * 1e3 / sums.gpu_hz.max(1) as f64 / f;
    let [title, ours] = sums.classes;
    let per = |c: &Class, v: f64| v / c.submissions.max(1) as f64 / 1e3;
    let seconds = sums.since.elapsed().as_secs_f64();
    let mut lines = vec![format!(
        "GPU timestamps over {frames} frames ({seconds:.1} s), per frame: {:.1} ms on the GPU \
         in {:.1} timed lists; title submissions {:.1}, {:.1} ms on the GPU, from first to last \
         timestamp {:.1} ms, each queued {:.2} ms and ended {:.2} ms after it was submitted \
         (at most {:.1}), read {:.2} ms after it ended; ours (present, planes) {:.1}, {:.1} ms \
         on the GPU, ended {:.2} ms after submission (at most {:.1}); the GPU idle or untimed \
         {:.1} ms between submissions; not timed {:.1} lists, {lost} recordings out of \
         timestamps",
        ms(title.busy + ours.busy),
        (title.lists + ours.lists) as f64 / f,
        title.submissions as f64 / f,
        ms(title.busy),
        ms(title.span),
        per(&title, title.queued),
        per(&title, title.ended),
        title.ended_max / 1e3,
        per(&title, title.read),
        ours.submissions as f64 / f,
        ms(ours.busy),
        per(&ours, ours.ended),
        ours.ended_max / 1e3,
        ms(sums.idle),
        (sums.untimed + untimed) as f64 / f,
    )];
    let mut kinds: Vec<(&What, &Sum)> = sums.kinds.iter().collect();
    kinds.sort_by(|a, b| b.1.ticks.cmp(&a.1.ticks).then(a.0.cmp(b.0)));
    let kinds: Vec<String> = kinds
        .iter()
        .map(|(what, s)| format!("{what} {:.2} ({:.0})", ms(s.ticks), s.n as f64 / f))
        .collect();
    let wave64: Sum = sums
        .programs
        .iter()
        .filter(|((p, _), _)| names.get(*p as usize).is_some_and(|n| n.1))
        .fold(Sum::default(), |a, (_, s)| Sum {
            ticks: a.ticks + s.ticks,
            n: a.n + s.n,
        });
    lines.push(format!(
        "GPU timestamps by kind, ms per frame (commands): {}; of the programs, wave64 ones \
         (run as two 32-lane waves) {:.2} ({:.0}) in {} programs",
        kinds.join(", "),
        ms(wave64.ticks),
        wave64.n as f64 / f,
        sums.programs
            .keys()
            .filter(|(p, _)| names.get(*p as usize).is_some_and(|n| n.1))
            .count()
    ));
    let mut programs: Vec<(&(u32, What), &Sum)> = sums.programs.iter().collect();
    programs.sort_by(|a, b| b.1.ticks.cmp(&a.1.ticks).then(a.0.cmp(b.0)));
    for ((p, what), s) in programs.into_iter().take(TOP) {
        let (name, wave64) = names
            .get(*p as usize)
            .map_or(("?", false), |(n, w)| (n.as_str(), *w));
        lines.push(format!(
            "GPU timestamps, program: {:.2} ms per frame, {:.1} {what} per frame of {:.0} us, \
             {name}{}",
            ms(s.ticks),
            s.n as f64 / f,
            s.ticks as f64 * 1e6 / sums.gpu_hz.max(1) as f64 / s.n.max(1) as f64,
            if wave64 { " (wave64)" } else { "" }
        ));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testgpu::{Kernel, Pass, TitleBuffer, TitleList, device};

    fn label(what: What, program: u32) -> Label {
        Label { what, program }
    }

    #[test]
    fn the_switch_is_off_unless_set_to_something_but_0() {
        use std::ffi::OsStr;
        assert!(!switch(None));
        assert!(!switch(Some(OsStr::new(""))));
        assert!(!switch(Some(OsStr::new("0"))));
        assert!(switch(Some(OsStr::new("1"))));
        assert!(switch(Some(OsStr::new("yes"))));
    }

    #[test]
    fn spans_go_to_the_timestamp_that_starts_them() {
        let labels = [
            BETWEEN,
            label(What::Dispatch, 3),
            BETWEEN,
            label(What::Draw, 4),
            BETWEEN,
        ];
        let s = spans(&[100, 110, 160, 161, 200], &labels).unwrap();
        assert_eq!(
            s.spans,
            vec![
                (BETWEEN, 10),
                (label(What::Dispatch, 3), 50),
                (BETWEEN, 1),
                (label(What::Draw, 4), 39),
            ]
        );
        assert_eq!((s.first, s.last), (100, 200));
        assert_eq!(
            s.spans.iter().map(|(_, t)| t).sum::<u64>(),
            s.last - s.first
        );
        // Not written, back in time, too short, not matching their labels.
        assert_eq!(spans(&[100, 0, 160, 161, 200], &labels), None);
        assert_eq!(spans(&[100, 90, 160, 161, 200], &labels), None);
        assert_eq!(spans(&[100], &labels[..1]), None);
        assert_eq!(spans(&[100, 110], &labels), None);
    }

    #[test]
    fn a_recording_takes_chunks_as_it_goes_and_gives_them_back() {
        let pool: &'static Pool = Box::leak(Box::new(Pool::new(3)));
        let mut a = Recording::new(pool);
        let first: Vec<u32> = (0..CHUNK + 1).map(|_| a.next(BETWEEN).unwrap()).collect();
        assert_eq!(a.chunks.len(), 2);
        assert_eq!(pool.left(), 1);
        // Chunk 0 then chunk 1, each one's indices in order.
        assert_eq!(first[0], 0);
        assert_eq!(first[CHUNK as usize - 1], CHUNK - 1);
        assert_eq!(first[CHUNK as usize], CHUNK);
        assert_eq!(a.ranges().collect::<Vec<_>>(), vec![(0, CHUNK), (CHUNK, 1)]);
        let mut b = Recording::new(pool);
        assert_eq!(b.next(BETWEEN), Some(2 * CHUNK));
        // Out of chunks: the next recording is lost, and counted.
        let mut c = Recording::new(pool);
        let lost = LOST.load(Ordering::Relaxed);
        assert_eq!(c.next(BETWEEN), None);
        assert!(c.lost);
        assert!(LOST.load(Ordering::Relaxed) > lost);
        assert_eq!(c.next(BETWEEN), None);
        drop((a, c));
        assert_eq!(pool.left(), 2);
        let mut d = Recording::new(pool);
        assert!(d.next(BETWEEN).is_some());
        drop((b, d));
        assert_eq!(pool.left(), 3);
    }

    /// A closed recording's chunks go back only once the list's entry and
    /// every submission that holds it have let it go.
    #[test]
    fn a_submission_keeps_its_recordings_chunks() {
        let pool: &'static Pool = Box::leak(Box::new(Pool::new(2)));
        let mut r = Recording::new(pool);
        assert!(r.next(BETWEEN).is_some() && r.next(BETWEEN).is_some());
        let closed = Arc::new(r);
        let submission = closed.clone();
        assert_eq!(pool.left(), 1);
        // The list is reset: its entry takes a new recording.
        drop(closed);
        assert_eq!(pool.left(), 1, "given back while a submission holds it");
        drop(submission);
        assert_eq!(pool.left(), 2);
    }

    #[test]
    fn the_line_ranks_kinds_and_programs_by_their_time() {
        let mut sums = Sums::new();
        sums.gpu_hz = 1_000_000;
        sums.classes[0].add(2, 9_000, 10_000, (100.0, 10_500.0, 50.0));
        sums.add(&[
            (BETWEEN, 1_000),
            (label(What::Dispatch, 1), 6_000),
            (label(What::Draw, 2), 2_000),
            (label(What::Dispatch, 1), 0),
        ]);
        let names = vec![
            (String::new(), false),
            ("[compute 13dfdd3b1136e407]".to_owned(), true),
            (
                "[vertex 167501e87c356551, pixel 9727fc3c51b2f11c]".to_owned(),
                false,
            ),
        ];
        let lines = lines(&sums, 1, &names, 0, 0);
        assert!(lines[0].contains("9.0 ms on the GPU"), "{}", lines[0]);
        assert!(lines[0].contains("in 2.0 timed lists"), "{}", lines[0]);
        assert!(
            lines[1].starts_with(
                "GPU timestamps by kind, ms per frame (commands): dispatch 6.00 (2), draw 2.00 \
                 (1), between commands 1.00 (1)"
            ),
            "{}",
            lines[1]
        );
        assert!(
            lines[1].contains("wave64 ones (run as two 32-lane waves) 6.00 (2) in 1 programs"),
            "{}",
            lines[1]
        );
        assert!(
            lines[2].contains("6.00 ms per frame, 2.0 dispatch per frame of 3000 us"),
            "{}",
            lines[2]
        );
        assert!(
            lines[2].ends_with("[compute 13dfdd3b1136e407] (wave64)"),
            "{}",
            lines[2]
        );
        assert!(
            lines[3].ends_with("pixel 9727fc3c51b2f11c]"),
            "{}",
            lines[3]
        );
        assert_eq!(lines.len(), 4);
    }

    /// The lists `key` was read as, in order, of those this thread
    /// submitted. A key is a D3D12 list's address, which a list made after
    /// another was released may take: another test's lists read under the
    /// same key (CI, PR #31: `a_title_list_is_timed_per_command` found the
    /// two reads of `a_list_reset_after_its_submission_is_timed_again`'s
    /// list with its own). Each test runs on a thread of its own, and a
    /// thread's id is never reused.
    fn read_as(key: usize) -> Vec<(Spans, Vec<u32>)> {
        let thread = std::thread::current().id();
        READ.lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|r| r.thread == thread && r.key == key)
            .map(|r| (r.spans.clone(), r.chunks.clone()))
            .collect()
    }

    /// Submissions are counted with timestamps off (the default) too:
    /// `texture`'s diagnostics count its checks per submission by the count.
    #[test]
    fn submissions_are_counted_with_timestamps_off() {
        if std::env::var_os("XTP_GPU_TIMESTAMPS").is_some() {
            return;
        }
        let Some(gpu) = device("timestamps::tests::submissions_are_counted_with_timestamps_off")
        else {
            return;
        };
        let host = gpu.host();
        assert!(!on());
        let list = host.new_list().unwrap();
        list.close();
        for _ in 0..2 {
            // Others' submissions may come in between: at least ours.
            let before = submissions();
            host.execute(&[&list]);
            assert!(submissions() > before);
            host.drain();
        }
    }

    /// A title list with two marked dispatches is timed end to end: its
    /// recording starts at its first mark (a list made, not reset yet),
    /// is closed and submitted as the title does, and read after its fence,
    /// every span labelled and the spans adding up to the list.
    #[test]
    fn a_title_list_is_timed_per_command() {
        let Some(gpu) = device("timestamps::tests::a_title_list_is_timed_per_command") else {
            return;
        };
        let host = gpu.host();
        FORCED.with(|f| f.set(true));
        let kernel = Kernel::new(host);
        let (a, b) = (TitleBuffer::new(256), TitleBuffer::new(256));
        let list = TitleList::new();
        {
            let _timed = command(list.host(), What::Dispatch, None);
            kernel.dispatch(list.host(), Pass::Fill, a.address(), b.address());
        }
        {
            let _timed = command(list.host(), What::Indirect(2), None);
            kernel.dispatch(list.host(), Pass::Copy, b.address(), a.address());
        }
        list.run();
        FORCED.with(|f| f.set(false));
        let read = read_as(key(list.host()));
        assert_eq!(read.len(), 1, "{read:?}");
        let (s, chunks) = &read[0];
        let whats: Vec<What> = s.spans.iter().map(|(l, _)| l.what).collect();
        assert_eq!(
            whats,
            [
                What::Dispatch,
                What::Between,
                What::Indirect(2),
                What::Between
            ]
        );
        assert!(s.first > 0 && s.last >= s.first, "{s:?}");
        assert_eq!(
            s.spans.iter().map(|(_, t)| t).sum::<u64>(),
            s.last - s.first
        );
        assert_eq!(chunks.len(), 1);
        // The fill ran: the kernel really was in the timed list.
        let want: Vec<u32> = (0..64).map(crate::testgpu::pattern).collect();
        assert_eq!(a.words(), want);
    }

    /// A list reset while its submission may still run starts a new
    /// recording at the reset, and both recordings are read with their own
    /// commands.
    #[test]
    fn a_list_reset_after_its_submission_is_timed_again() {
        let Some(gpu) =
            device("timestamps::tests::a_list_reset_after_its_submission_is_timed_again")
        else {
            return;
        };
        let host = gpu.host();
        FORCED.with(|f| f.set(true));
        let kernel = Kernel::new(host);
        let (a, b) = (TitleBuffer::new(256), TitleBuffer::new(256));
        let list = host.new_list().unwrap();
        {
            let _timed = command(&list, What::Dispatch, None);
            kernel.dispatch(&list, Pass::Fill, a.address(), b.address());
        }
        list.close();
        host.execute(&[&list]);
        list.reset();
        for _ in 0..3 {
            let _timed = command(&list, What::Mesh, None);
            kernel.dispatch(&list, Pass::Copy, b.address(), a.address());
        }
        list.close();
        host.execute(&[&list]);
        host.drain();
        FORCED.with(|f| f.set(false));
        let read = read_as(key(&list));
        assert_eq!(read.len(), 2, "{read:?}");
        let counts: Vec<usize> = read.iter().map(|(s, _)| s.spans.len()).collect();
        // Dispatch, after it; from the reset, three meshes and after each.
        assert_eq!(counts, [2, 7]);
        assert!(
            read[1]
                .0
                .spans
                .iter()
                .filter(|(l, _)| l.what == What::Mesh)
                .count()
                == 3
        );
        assert_eq!(read[1].0.spans[0].0.what, What::Between, "from the reset");
        for (s, _) in &read {
            assert_eq!(
                s.spans.iter().map(|(_, t)| t).sum::<u64>(),
                s.last - s.first
            );
        }
    }

    /// Off (the default), nothing is recorded: no timestamp, no entry.
    #[test]
    fn nothing_is_recorded_when_off() {
        if std::env::var_os("XTP_GPU_TIMESTAMPS").is_some() {
            return;
        }
        let Some(gpu) = device("timestamps::tests::nothing_is_recorded_when_off") else {
            return;
        };
        let host = gpu.host();
        let list = host.new_list().unwrap();
        // What is noted of the list's address (a list of another test that
        // was there may have left its last recording).
        let entry = || {
            let lists = LISTS.lock().unwrap_or_else(|e| e.into_inner());
            match lists.as_ref().and_then(|l| l.get(&key(&list))) {
                None => None,
                Some(Entry::Open(r)) => Some((true, r.labels.len(), 0)),
                Some(Entry::Closed(r)) => Some((false, r.labels.len(), Arc::as_ptr(r) as usize)),
            }
        };
        let (before, read_before) = (entry(), read_as(key(&list)).len());
        for _ in 0..2 {
            drop(command(&list, What::Draw, None));
            list.close();
            host.execute(&[&list]);
            host.drain();
            list.reset();
        }
        list.close();
        assert_eq!(entry(), before);
        assert_eq!(read_as(key(&list)).len(), read_before);
    }

    /// Dropping a host list without resetting it removes its entry and returns
    /// its timestamp recording's chunks to the pool instead of leaking them.
    #[test]
    fn dropping_a_list_returns_its_chunks_without_a_reset() {
        let Some(gpu) =
            device("timestamps::tests::dropping_a_list_returns_its_chunks_without_a_reset")
        else {
            return;
        };
        let host = gpu.host();
        FORCED.with(|f| f.set(true));
        let list = host.new_list().unwrap();
        let k = key(&list);
        drop(command(&list, What::Draw, None));
        list.close();
        host.execute(&[&list]);
        host.drain();
        FORCED.with(|f| f.set(false));

        let weak = {
            let lists = LISTS.lock().unwrap_or_else(|e| e.into_inner());
            match lists.as_ref().and_then(|l| l.get(&k)) {
                Some(Entry::Closed(r)) => Arc::downgrade(r),
                _ => panic!("expected a closed timestamp recording for the submitted list"),
            }
        };
        assert_eq!(weak.strong_count(), 1, "only LISTS holds the recording");
        drop(list);
        assert_eq!(
            weak.strong_count(),
            0,
            "dropping the list frees the recording and returns its chunks"
        );
    }
}
