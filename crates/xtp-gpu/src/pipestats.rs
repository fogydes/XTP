//! Pipeline statistics per marked command, behind `XTP_GPU_PIPESTATS=1`
//! with `XTP_GPU_TIMESTAMPS=1` (off by default; a measurement, like the
//! timestamps whose marks it follows): how many vertex, mesh, pixel and
//! compute invocations, and how many primitives, each program runs a frame.
//! Milliseconds say how long a command took; these counts say what it did.
//! A draw of a few hundred primitives that runs the pixel program fifty
//! million times (wrong positions filling the screen many times over), or a
//! dispatch of millions of threads where a few thousand were meant, shows
//! here and not in the times (VIctory #1723 used such a query to find
//! one; this is XTP's own, written for `timestamps`).
//!
//! Each command [`crate::timestamps::command`] marks gets one query:
//! [`begin`] before it, [`end`] after it. A list's queries are resolved
//! into a read-back buffer when it closes ([`close`]), and read once its
//! submission has finished ([`submitted`], after the copy-back). Every few
//! seconds (`XTP_FRAME_LINE`, as the frame line) the sums are logged per
//! frame: the totals, per kind of command, and per program for the
//! programs that run the most invocations or primitives (named as host
//! pipelines are, by their translated shaders' hashes).
//!
//! The query is `PIPELINE_STATISTICS1` where the device counts mesh and
//! amplification work (`MeshShaderPipelineStatsSupported`), else
//! `PIPELINE_STATISTICS` without those counters: the line says which.
//!
//! shortcut: queries of one type cannot nest, so a command marked while
//! another one's query is open in the same list is not queried on its own:
//! the outer query runs until the outer command ends, and counts both
//! (counted on the line as nested).
//!
//! shortcut: a recording that finds no query left is not counted (on the
//! line).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

use crate::host::HostList;
use crate::pipeline::host::HostPipeline;
use crate::timestamps::What;

/// Queries per chunk.
const CHUNK: u32 = 256;
/// Chunks in the heap: 256 Ki queries, one per marked command (a world
/// frame marks about 8,000), 28 MiB of query heap and as much read-back.
const CHUNKS: u32 = 1024;

/// Whether `value` (of `XTP_GPU_PIPESTATS`) turns the queries on: set, and
/// neither empty nor `0`.
fn switch(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_some_and(|v| !v.is_empty() && v != "0")
}

/// Whether commands are queried (`XTP_GPU_PIPESTATS=1`; only marked
/// commands are, and commands are marked with `XTP_GPU_TIMESTAMPS=1`).
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    let on = *ON.get_or_init(|| {
        let on = switch(std::env::var_os("XTP_GPU_PIPESTATS").as_deref());
        if on {
            tracing::info!(
                "GPU pipeline statistics on (XTP_GPU_PIPESTATS, with XTP_GPU_TIMESTAMPS)"
            );
        }
        on
    });
    #[cfg(test)]
    let on = on || FORCED.with(std::cell::Cell::get);
    on
}

#[cfg(test)]
thread_local! {
    /// Tests turn the queries on for the lists their own thread records.
    static FORCED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The counters kept, in the order the line prints them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Counter {
    Vs,
    Ms,
    MsPrimitives,
    ClipperIn,
    ClipperOut,
    Ps,
    Cs,
    As,
}

const COUNTERS: [Counter; 8] = [
    Counter::Vs,
    Counter::Ms,
    Counter::MsPrimitives,
    Counter::ClipperIn,
    Counter::ClipperOut,
    Counter::Ps,
    Counter::Cs,
    Counter::As,
];

impl Counter {
    /// Its word in `D3D12_QUERY_DATA_PIPELINE_STATISTICS1` (the first 11
    /// are `D3D12_QUERY_DATA_PIPELINE_STATISTICS`'s).
    fn word(self) -> usize {
        match self {
            Counter::Vs => 2,
            Counter::ClipperIn => 5,
            Counter::ClipperOut => 6,
            Counter::Ps => 7,
            Counter::Cs => 10,
            Counter::As => 11,
            Counter::Ms => 12,
            Counter::MsPrimitives => 13,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Counter::Vs => "VS",
            Counter::Ms => "MS",
            Counter::MsPrimitives => "MS primitives",
            Counter::ClipperIn => "primitives into the clipper",
            Counter::ClipperOut => "out of it",
            Counter::Ps => "PS",
            Counter::Cs => "CS",
            Counter::As => "AS",
        }
    }
}

/// One command's counts, or a sum of them (`n` commands).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct Counts {
    n: u64,
    c: [u64; COUNTERS.len()],
}

impl Counts {
    /// A query's result, `words` long (14 with mesh counters, else 11).
    fn of(words: &[u64]) -> Counts {
        let mut c = [0; COUNTERS.len()];
        for (i, counter) in COUNTERS.iter().enumerate() {
            c[i] = words.get(counter.word()).copied().unwrap_or(0);
        }
        Counts { n: 1, c }
    }

    fn add(&mut self, other: &Counts) {
        self.n += other.n;
        for (a, b) in self.c.iter_mut().zip(other.c) {
            *a += b;
        }
    }

    fn get(&self, counter: Counter) -> u64 {
        self.c[COUNTERS.iter().position(|&c| c == counter).unwrap_or(0)]
    }

    /// All invocations of every stage: how the programs are ordered.
    fn invocations(&self) -> u64 {
        [
            Counter::Vs,
            Counter::Ms,
            Counter::Ps,
            Counter::Cs,
            Counter::As,
        ]
        .iter()
        .map(|&c| self.get(c))
        .sum()
    }
}

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
}

/// The query heap, its read-back buffer (mapped for good) and its chunks.
struct Heap {
    queries: ID3D12QueryHeap,
    back: ID3D12Resource,
    data: *const u64,
    pool: Pool,
    /// `PIPELINE_STATISTICS1`, with the mesh counters.
    mesh: bool,
}

// SAFETY: D3D12 objects are free-threaded; `data` is a persistent mapping
// of a read-back buffer, read only where finished submissions resolved.
unsafe impl Send for Heap {}
// SAFETY: as above.
unsafe impl Sync for Heap {}

impl Heap {
    fn kind(&self) -> D3D12_QUERY_TYPE {
        if self.mesh {
            D3D12_QUERY_TYPE_PIPELINE_STATISTICS1
        } else {
            D3D12_QUERY_TYPE_PIPELINE_STATISTICS
        }
    }

    /// Words per query result.
    fn words(&self) -> usize {
        if self.mesh {
            size_of::<D3D12_QUERY_DATA_PIPELINE_STATISTICS1>() / 8
        } else {
            size_of::<D3D12_QUERY_DATA_PIPELINE_STATISTICS>() / 8
        }
    }
}

static HEAP: OnceLock<Option<Heap>> = OnceLock::new();

/// The heap, made on first use (`None` without a host device or if it
/// could not be made, logged once).
fn heap() -> Option<&'static Heap> {
    HEAP.get_or_init(|| {
        let device = &crate::host::made()?.device;
        match make_heap(device) {
            Ok(heap) => Some(heap),
            Err(e) => {
                tracing::warn!(error = %e, "no GPU pipeline statistics: their query heap was not made");
                None
            }
        }
    })
    .as_ref()
}

/// Whether `device` counts mesh and amplification work in pipeline
/// statistics.
fn mesh_counters(device: &ID3D12Device) -> bool {
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS9::default();
    // SAFETY: a plain query into a local of the feature's own type.
    let asked = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS9,
            (&raw mut options).cast(),
            size_of_val(&options) as u32,
        )
    };
    asked.is_ok() && options.MeshShaderPipelineStatsSupported.as_bool()
}

fn make_heap(device: &ID3D12Device) -> windows::core::Result<Heap> {
    let mesh = mesh_counters(device);
    let count = CHUNKS * CHUNK;
    let size = if mesh {
        size_of::<D3D12_QUERY_DATA_PIPELINE_STATISTICS1>()
    } else {
        size_of::<D3D12_QUERY_DATA_PIPELINE_STATISTICS>()
    } as u64;
    let mut queries: Option<ID3D12QueryHeap> = None;
    let mut back: Option<ID3D12Resource> = None;
    let readback = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_READBACK,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: u64::from(count) * size,
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
                Type: if mesh {
                    D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS1
                } else {
                    D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS
                },
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
    crate::host::name(&back, || "GPU pipeline statistics read-back".into());
    Ok(Heap {
        queries,
        back,
        data: data.cast::<u64>().cast_const(),
        pool: Pool::new(CHUNKS),
        mesh,
    })
}

/// What a query counted: a kind of command and the program it ran (0:
/// none).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Label {
    what: What,
    program: u32,
}

/// One recording of a list: the chunks its queries are in, and what each
/// query counted.
struct Recording {
    pool: &'static Pool,
    chunks: Vec<u32>,
    labels: Vec<Label>,
    /// Marked commands begun and not ended: the last query is open while
    /// this is not 0.
    depth: u32,
    /// No chunk was left for a query: not counted.
    lost: bool,
}

impl Recording {
    fn new(pool: &'static Pool) -> Recording {
        Recording {
            pool,
            chunks: Vec::new(),
            labels: Vec::new(),
            depth: 0,
            lost: false,
        }
    }

    /// The query index of the next query, counting `label`; `None` once
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

    /// The query index of query `k`.
    fn index(&self, k: u32) -> u32 {
        self.chunks[(k / CHUNK) as usize] * CHUNK + k % CHUNK
    }

    /// Each chunk's first query index and how many of its queries the
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

/// Recordings by host list (its D3D12 list).
static LISTS: Mutex<Option<HashMap<usize, Entry>>> = Mutex::new(None);

fn key(list: &HostList) -> usize {
    use windows::core::Interface;
    list.list.as_raw() as usize
}

/// Recordings that found no query left, and commands marked while another
/// one's query was open, since the last line.
static LOST: AtomicU64 = AtomicU64::new(0);
static NESTED: AtomicU64 = AtomicU64::new(0);

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
}

/// A list is being dropped (after its submissions settled): forgets its
/// recording so its query chunks return to the pool instead of leaking.
pub(crate) fn forget(list: &HostList) {
    let old = {
        let mut lists = LISTS.lock().unwrap_or_else(|e| e.into_inner());
        lists.as_mut().and_then(|l| l.remove(&key(list)))
    };
    // Its chunks go back outside the lock.
    drop(old);
}

/// Before a marked command of kind `what` running `pipeline`'s programs:
/// its query begins (none if one is open in the list already).
pub(crate) fn begin(list: &HostList, what: What, pipeline: Option<&HostPipeline>) {
    if !on() {
        return;
    }
    let Some(heap) = heap() else { return };
    let program = pipeline.map_or(0, |p| program(&p.name));
    let mut lists = LISTS.lock().unwrap_or_else(|e| e.into_inner());
    let entry = lists
        .get_or_insert_with(HashMap::new)
        .entry(key(list))
        .or_insert_with(|| Entry::Open(Recording::new(&heap.pool)));
    if let Entry::Closed(_) = entry {
        // Another list made at the address of one closed before.
        *entry = Entry::Open(Recording::new(&heap.pool));
    }
    let Entry::Open(recording) = entry else {
        return;
    };
    if recording.depth > 0 {
        recording.depth += 1;
        NESTED.fetch_add(1, Ordering::Relaxed);
        return;
    }
    if let Some(index) = recording.next(Label { what, program }) {
        recording.depth = 1;
        // SAFETY: an open list and our heap; the index is in a chunk the
        // recording holds, and no query of this type is open in the list.
        unsafe { list.list.BeginQuery(&heap.queries, heap.kind(), index) };
    }
}

/// After a marked command: the query [`begin`] began ends, once the
/// commands marked inside it have ended too (nothing if none is open).
pub(crate) fn end(list: &HostList) {
    if !on() {
        return;
    }
    let Some(heap) = heap() else { return };
    let mut lists = LISTS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(Entry::Open(recording)) = lists.as_mut().and_then(|l| l.get_mut(&key(list)))
        && recording.depth > 0
    {
        recording.depth -= 1;
        if recording.depth == 0 {
            end_query(list, heap, recording);
        }
    }
}

/// Ends `recording`'s last query in `list`: the one open.
fn end_query(list: &HostList, heap: &Heap, recording: &Recording) {
    let index = recording.index(recording.labels.len() as u32 - 1);
    // SAFETY: an open list and our heap; the query was begun in this list.
    unsafe { list.list.EndQuery(&heap.queries, heap.kind(), index) };
}

/// Just before a list is closed: the resolve of its queries into the
/// read-back buffer.
pub(crate) fn close(list: &HostList) {
    if !on() {
        return;
    }
    let Some(heap) = heap() else { return };
    let mut lists = LISTS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(entry) = lists.as_mut().and_then(|l| l.get_mut(&key(list))) else {
        return;
    };
    let Entry::Open(recording) = entry else {
        return;
    };
    // A command still marked at the close: its query ends with the list.
    if std::mem::take(&mut recording.depth) > 0 {
        end_query(list, heap, recording);
    }
    let recording = std::mem::replace(recording, Recording::new(&heap.pool));
    if !recording.lost {
        let size = (heap.words() * 8) as u64;
        for (first, count) in recording.ranges() {
            // SAFETY: an open list, our heap and read-back buffer; the
            // queries were begun and ended in this list, the destination is
            // theirs.
            unsafe {
                list.list.ResolveQueryData(
                    &heap.queries,
                    heap.kind(),
                    first,
                    count,
                    &heap.back,
                    u64::from(first) * size,
                )
            };
        }
    }
    *entry = Entry::Closed(Arc::new(recording));
}

/// Just after a submission of `runs`: their queries are read once it has
/// finished, and the sums logged when due.
pub(crate) fn submitted(runs: &[&HostList]) {
    if !on() {
        return;
    }
    let Some(host) = crate::host::made() else {
        return;
    };
    let recordings: Vec<Arc<Recording>> = {
        let lists = LISTS.lock().unwrap_or_else(|e| e.into_inner());
        runs.iter()
            .filter_map(|l| match lists.as_ref().and_then(|m| m.get(&key(l))) {
                Some(Entry::Closed(r)) => Some(r.clone()),
                _ => None,
            })
            .collect()
    };
    if !recordings.is_empty() {
        host.after(move || read(&recordings));
    }
    log_due();
}

/// Reads finished recordings' queries into the sums (on the thread that
/// finished their submission: no waits here).
fn read(recordings: &[Arc<Recording>]) {
    let Some(heap) = heap() else { return };
    let words = heap.words();
    let mut read = Vec::new();
    for recording in recordings.iter().filter(|r| !r.lost) {
        for (k, label) in recording.labels.iter().enumerate() {
            let at = recording.index(k as u32) as usize * words;
            let result: Vec<u64> = (0..words)
                // SAFETY: the query's result is in the read-back buffer,
                // which the finished submission resolved into.
                .map(|w| unsafe { heap.data.add(at + w).read_volatile() })
                .collect();
            read.push((*label, Counts::of(&result)));
        }
    }
    let mut sums = SUMS.lock().unwrap_or_else(|e| e.into_inner());
    let sums = sums.get_or_insert_with(Sums::new);
    for (label, counts) in &read {
        sums.add(*label, counts);
    }
    #[cfg(test)]
    READ.lock().unwrap_or_else(|e| e.into_inner()).extend(read);
}

/// Queries read in tests, in order.
#[cfg(test)]
static READ: Mutex<Vec<(Label, Counts)>> = Mutex::new(Vec::new());

/// Programs by name (host pipelines' names: their shaders' hashes); 0 is
/// none.
#[derive(Default)]
struct Programs {
    ids: HashMap<String, u32>,
    names: Vec<String>,
}

static PROGRAMS: Mutex<Option<Programs>> = Mutex::new(None);

/// The number of the program named `name`.
fn program(name: &str) -> u32 {
    let mut programs = PROGRAMS.lock().unwrap_or_else(|e| e.into_inner());
    let programs = programs.get_or_insert_with(|| Programs {
        ids: HashMap::new(),
        names: vec![String::new()],
    });
    if let Some(&id) = programs.ids.get(name) {
        return id;
    }
    let id = programs.names.len() as u32;
    programs.names.push(name.to_owned());
    programs.ids.insert(name.to_owned(), id);
    id
}

/// What was read since the last line.
struct Sums {
    since: Instant,
    frame: u64,
    total: Counts,
    kinds: BTreeMap<What, Counts>,
    programs: HashMap<Label, Counts>,
}

impl Sums {
    fn new() -> Sums {
        Sums {
            since: Instant::now(),
            frame: crate::frame::current(),
            total: Counts::default(),
            kinds: BTreeMap::new(),
            programs: HashMap::new(),
        }
    }

    fn add(&mut self, label: Label, counts: &Counts) {
        self.total.add(counts);
        self.kinds.entry(label.what).or_default().add(counts);
        if label.program != 0 {
            self.programs.entry(label).or_default().add(counts);
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
        let Some(done) = sums.replace(Sums::new()) else {
            return;
        };
        let frames = crate::frame::current().saturating_sub(done.frame);
        (done, frames)
    };
    let mesh = heap().is_some_and(|h| h.mesh);
    let programs = PROGRAMS.lock().unwrap_or_else(|e| e.into_inner());
    let names = programs.as_ref().map_or(&[][..], |p| &p.names[..]);
    for line in lines(
        &sums,
        frames,
        names,
        mesh,
        (
            NESTED.swap(0, Ordering::Relaxed),
            LOST.swap(0, Ordering::Relaxed),
        ),
    ) {
        tracing::info!("{line}");
    }
}

/// Programs on the line for each counter ranked: the union is printed.
const TOP: usize = 15;

/// `counts`' counters per frame, `f` frames.
fn per_frame(counts: &Counts, f: f64, mesh: bool) -> String {
    COUNTERS
        .iter()
        .filter(|&&c| mesh || !matches!(c, Counter::Ms | Counter::MsPrimitives | Counter::As))
        .map(|&c| format!("{} {:.0}", c.name(), counts.get(c) as f64 / f))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The lines for `sums` over `frames` frames (at least one): the totals,
/// the kinds, and the programs that ran the most invocations of each stage
/// or the most primitives, by most invocations in all.
fn lines(
    sums: &Sums,
    frames: u64,
    names: &[String],
    mesh: bool,
    (nested, lost): (u64, u64),
) -> Vec<String> {
    let f = frames.max(1) as f64;
    let seconds = sums.since.elapsed().as_secs_f64();
    let mut lines = vec![format!(
        "GPU pipeline statistics over {frames} frames ({seconds:.1} s), per frame: {:.1} commands \
         queried ({}): {}; {nested} commands nested in another one's query and {lost} \
         recordings out of queries",
        sums.total.n as f64 / f,
        if mesh {
            "with mesh counters"
        } else {
            "no mesh counters on this device"
        },
        per_frame(&sums.total, f, mesh),
    )];
    let kinds: Vec<String> = sums
        .kinds
        .iter()
        .filter(|(_, c)| c.invocations() > 0 || c.get(Counter::ClipperIn) > 0)
        .map(|(what, c)| format!("{what} ({:.0}): {}", c.n as f64 / f, per_frame(c, f, mesh)))
        .collect();
    lines.push(format!(
        "GPU pipeline statistics by kind, per frame: {}",
        kinds.join("; ")
    ));
    let mut shown: HashSet<Label> = HashSet::new();
    for counter in [
        Counter::Ps,
        Counter::ClipperOut,
        Counter::Cs,
        Counter::Vs,
        Counter::Ms,
    ] {
        let mut ranked: Vec<(&Label, &Counts)> = sums
            .programs
            .iter()
            .filter(|(_, c)| c.get(counter) > 0)
            .collect();
        ranked.sort_by(|a, b| {
            b.1.get(counter)
                .cmp(&a.1.get(counter))
                .then(a.0.program.cmp(&b.0.program))
        });
        shown.extend(ranked.iter().take(TOP).map(|(l, _)| **l));
    }
    let mut shown: Vec<(&Label, &Counts)> = sums
        .programs
        .iter()
        .filter(|(l, _)| shown.contains(l))
        .collect();
    shown.sort_by(|a, b| {
        b.1.invocations()
            .cmp(&a.1.invocations())
            .then(a.0.program.cmp(&b.0.program))
            .then(a.0.what.cmp(&b.0.what))
    });
    for (label, c) in shown {
        let name = names
            .get(label.program as usize)
            .map_or("?", String::as_str);
        let out = c.get(Counter::ClipperOut);
        let per_primitive = if out > 0 {
            format!("{:.1}", c.get(Counter::Ps) as f64 / out as f64)
        } else {
            "-".to_owned()
        };
        lines.push(format!(
            "GPU pipeline statistics, program: {:.1} {} per frame: {}; PS per primitive out of \
             the clipper {per_primitive}; {name}",
            c.n as f64 / f,
            label.what,
            per_frame(c, f, mesh),
        ));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testgpu::{Kernel, Pass, TitleBuffer, device};

    #[test]
    fn the_switch_is_off_unless_set_to_something_but_0() {
        use std::ffi::OsStr;
        assert!(!switch(None));
        assert!(!switch(Some(OsStr::new(""))));
        assert!(!switch(Some(OsStr::new("0"))));
        assert!(switch(Some(OsStr::new("1"))));
    }

    #[test]
    fn a_result_is_read_by_counter_with_or_without_mesh_counters() {
        let words: Vec<u64> = (0..14).map(|w| 100 + w).collect();
        let c = Counts::of(&words);
        assert_eq!(c.get(Counter::Vs), 102);
        assert_eq!(c.get(Counter::ClipperIn), 105);
        assert_eq!(c.get(Counter::ClipperOut), 106);
        assert_eq!(c.get(Counter::Ps), 107);
        assert_eq!(c.get(Counter::Cs), 110);
        assert_eq!(c.get(Counter::As), 111);
        assert_eq!(c.get(Counter::Ms), 112);
        assert_eq!(c.get(Counter::MsPrimitives), 113);
        // Without mesh counters, those read as 0.
        let c = Counts::of(&words[..11]);
        assert_eq!((c.get(Counter::Cs), c.get(Counter::Ms)), (110, 0));
    }

    #[test]
    fn the_line_shows_the_programs_that_run_the_most_of_each_counter() {
        let mut sums = Sums::new();
        let counts = |vs, cin, cout, ps, cs| {
            let mut words = [0u64; 14];
            (words[2], words[5], words[6], words[7], words[10]) = (vs, cin, cout, ps, cs);
            Counts::of(&words)
        };
        let label = |what, program| Label { what, program };
        // A draw of 184 primitives that runs the pixel program 51 million
        // times, a big dispatch, and an ordinary draw.
        sums.add(label(What::Mesh, 1), &counts(0, 184, 184, 51_119_928, 0));
        sums.add(label(What::Dispatch, 2), &counts(0, 0, 0, 0, 3_000_000));
        sums.add(label(What::Draw, 3), &counts(300, 100, 90, 9_000, 0));
        sums.add(label(What::Clamp, 0), &counts(0, 0, 0, 0, 64));
        let names = vec![
            String::new(),
            "[mesh 51ff151f75cd5759, pixel 7fe75645c406cac6]".to_owned(),
            "[compute 13dfdd3b1136e407]".to_owned(),
            "[vertex 167501e87c356551, pixel 9727fc3c51b2f11c]".to_owned(),
        ];
        let lines = lines(&sums, 1, &names, true, (2, 0));
        assert!(
            lines[0].contains("4.0 commands queried (with mesh counters)"),
            "{}",
            lines[0]
        );
        assert!(lines[0].contains("PS 51128928"), "{}", lines[0]);
        assert!(lines[0].contains("2 commands nested"), "{}", lines[0]);
        assert!(
            lines[1].contains("indirect argument clamp (1): VS 0"),
            "{}",
            lines[1]
        );
        assert!(lines[1].contains("mesh dispatch (1)"), "{}", lines[1]);
        assert_eq!(lines.len(), 5, "{lines:#?}");
        assert!(
            lines[2].contains("1.0 mesh dispatch per frame")
                && lines[2].contains("PS 51119928")
                && lines[2].contains("PS per primitive out of the clipper 277825.7")
                && lines[2].ends_with("[mesh 51ff151f75cd5759, pixel 7fe75645c406cac6]"),
            "{}",
            lines[2]
        );
        assert!(
            lines[3].ends_with("[compute 13dfdd3b1136e407]"),
            "{}",
            lines[3]
        );
        assert!(lines[3].contains("CS 3000000"), "{}", lines[3]);
        assert!(
            lines[3].contains("PS per primitive out of the clipper -"),
            "{}",
            lines[3]
        );
        assert!(
            lines[4].contains("PS per primitive out of the clipper 100.0"),
            "{}",
            lines[4]
        );
        // Without mesh counters the line does not print them.
        let plain = super::lines(&sums, 1, &names, false, (0, 0));
        assert!(!plain[0].contains("MS"), "{}", plain[0]);
    }

    /// Queries made, ended and resolved in a list, read after its
    /// submission: a marked dispatch of the test kernel (one group of 64
    /// threads) counts 64 compute invocations; a command marked inside
    /// another one's query is counted with it, and the outer query runs to
    /// the outer command's end (three dispatches, 192).
    #[test]
    fn each_marked_command_counts_what_it_ran() {
        let Some(gpu) = device("pipestats::tests::each_marked_command_counts_what_it_ran") else {
            return;
        };
        let host = gpu.host();
        if std::env::var_os("XTP_GPU_TIMESTAMPS").is_some() {
            // The timestamps' hooks would query these commands as well.
            return;
        }
        FORCED.with(|f| f.set(true));
        let kernel = Kernel::new(host);
        let (a, b) = (TitleBuffer::new(256), TitleBuffer::new(256));
        let list = host.new_list().unwrap();
        open(&list);
        begin(&list, What::Dispatch, None);
        kernel.dispatch(&list, Pass::Fill, a.address(), b.address());
        end(&list);
        let nested = NESTED.load(Ordering::Relaxed);
        begin(&list, What::Indirect(2), None);
        kernel.dispatch(&list, Pass::Copy, b.address(), a.address());
        begin(&list, What::Clamp, None);
        kernel.dispatch(&list, Pass::Copy, a.address(), b.address());
        end(&list);
        // Ended with the outer command, not the inner one.
        kernel.dispatch(&list, Pass::Copy, b.address(), a.address());
        end(&list);
        assert!(NESTED.load(Ordering::Relaxed) > nested);
        close(&list);
        list.close();
        let before = READ.lock().unwrap_or_else(|e| e.into_inner()).len();
        host.execute(&[&list]);
        submitted(&[&list]);
        host.drain();
        FORCED.with(|f| f.set(false));
        let read: Vec<(Label, Counts)> =
            READ.lock().unwrap_or_else(|e| e.into_inner())[before..].to_vec();
        let ours: Vec<(What, u64)> = read
            .iter()
            .map(|(l, c)| (l.what, c.get(Counter::Cs)))
            .collect();
        assert_eq!(
            ours,
            [(What::Dispatch, 64), (What::Indirect(2), 192)],
            "{read:?}"
        );
        assert!(
            read.iter()
                .all(|(_, c)| c.n == 1 && c.get(Counter::Ps) == 0)
        );
        // The kernel really ran in the queried list.
        let want: Vec<u32> = (0..64).map(crate::testgpu::pattern).collect();
        assert_eq!((a.words(), b.words()), (want.clone(), want));
    }

    /// Off (the default), nothing is asked: no entry for the list.
    #[test]
    fn nothing_is_asked_when_off() {
        if std::env::var_os("XTP_GPU_PIPESTATS").is_some() {
            return;
        }
        let Some(gpu) = device("pipestats::tests::nothing_is_asked_when_off") else {
            return;
        };
        let host = gpu.host();
        let list = host.new_list().unwrap();
        let has_entry = || {
            LISTS
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_ref()
                .is_some_and(|l| l.contains_key(&key(&list)))
        };
        let before = has_entry();
        open(&list);
        begin(&list, What::Draw, None);
        end(&list);
        close(&list);
        list.close();
        assert_eq!(has_entry(), before);
    }

    /// Dropping a host list without resetting it removes its entry and returns
    /// its pipeline-statistics query chunks to the pool instead of leaking them.
    #[test]
    fn dropping_a_list_returns_its_query_chunks_without_a_reset() {
        let Some(gpu) =
            device("pipestats::tests::dropping_a_list_returns_its_query_chunks_without_a_reset")
        else {
            return;
        };
        let host = gpu.host();
        FORCED.with(|f| f.set(true));
        let list = host.new_list().unwrap();
        let k = key(&list);
        open(&list);
        begin(&list, What::Draw, None);
        end(&list);
        close(&list);
        list.close();
        host.execute(&[&list]);
        submitted(&[&list]);
        host.drain();
        FORCED.with(|f| f.set(false));

        let weak = {
            let lists = LISTS.lock().unwrap_or_else(|e| e.into_inner());
            match lists.as_ref().and_then(|l| l.get(&k)) {
                Some(Entry::Closed(r)) => Arc::downgrade(r),
                _ => panic!("expected a closed pipestats recording for the submitted list"),
            }
        };
        assert_eq!(weak.strong_count(), 1, "only LISTS holds the recording");
        drop(list);
        assert_eq!(
            weak.strong_count(),
            0,
            "dropping the list frees the recording and returns its query chunks"
        );
    }
}
