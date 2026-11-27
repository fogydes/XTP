//! Host pipelines are made by workers, not by the title thread that first
//! records a draw with them. Translating a shader and the host driver's
//! compile take from milliseconds to seconds each (structurisation of the
//! biggest compute programs ~0.4 s, some driver compiles 3.4 s), and BTP 6's
//! first world frame wants over 1,200 new pipelines: made
//! on its recording threads, that frame took more than 30 s and the
//! title's GPU watchdog ended the run (decision 0007, "Pipelines made off
//! the title's threads").
//!
//! A first use queues the pipeline ahead of those started early and waits
//! for it within a budget per title frame ([`Budget`]): a few
//! milliseconds for graphics pipelines, more for compute ones, whose
//! results later passes read (docs/analysis/gpu.md, G12). The cheap ones
//! (most, and every one the pipeline library holds) are there at once;
//! past the budget, a draw or dispatch whose pipeline is not ready is
//! dropped, as an untranslated one is, until a worker has made it: a few
//! frames, instead of up to half a second of waiting per frame (the budget
//! before).
//!
//! Heavy pipelines ([`heavy`]: a large console program, whose translation
//! or driver compile takes seconds, up to 38 s) have a pool of workers of
//! their own, so a cheap pipeline never waits behind them: with one pool,
//! the Series S world dropped 566 to 1,122 draws a frame "being made" while
//! slow programs held most of the workers (docs/analysis/drops.md, §5.3).
//! The idea is OpenScarlett's (a lane for heavy pipelines), written again
//! here. Each pool makes its pipelines first use first, then those started
//! early, and no title thread waits for a heavy one.
//!
//! shortcut: a pass the title runs only once and whose pipeline takes
//! longer than the budget is lost; `XTP_PIPELINE_WAIT_MS` raises it.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::Shader;
use super::host::{GraphicsState, HostPipeline};

/// What a host pipeline is made from, and the host pipeline once made.
pub(crate) struct Pending {
    shaders: Vec<Arc<Shader>>,
    pub(crate) graphics: Option<Box<GraphicsState>>,
    /// [`IDLE`], [`EARLY`], [`URGENT`] or [`MAKING`]: the queue it was
    /// handed to last, or taken out of it. Changed under the queues' lock.
    queued: AtomicU8,
    /// [`CHEAP`] or [`HEAVY`]: the pool it was handed to last, which makes
    /// it. Changed under the queues' lock.
    pool: AtomicU8,
    made: OnceLock<Option<HostPipeline>>,
    /// Set by the call that made it, once what that call counts is in
    /// place ([`Pending::make`]).
    finished: AtomicBool,
    /// What identifies it across runs (shader containers and state:
    /// [`key`]), and whether a draw has used it yet.
    pub(crate) key: String,
    used: AtomicBool,
    /// The graphics state as the title's stream gave it, kept for the
    /// precook once a worker has made the pipeline ([`super::record`]).
    pub(crate) stream: Option<super::record::Stream>,
    /// Why the host refused it, once it has (diagnostics).
    #[cfg(feature = "diagnostics")]
    refusal: OnceLock<String>,
}

// SAFETY: D3D12 objects are free-threaded; the graphics state (whose input
// elements point into its own names) is only read after creation.
unsafe impl Send for Pending {}
// SAFETY: as above.
unsafe impl Sync for Pending {}

/// Not queued yet; queued as started early; queued for a first use; taken
/// by the worker that makes it (VIctory #546: a pipeline is never handed to
/// a second worker, which would wait on the first's compile).
const IDLE: u8 = 0;
const EARLY: u8 = 1;
const URGENT: u8 = 2;
const MAKING: u8 = 3;

/// The pools: cheap pipelines, heavy ones ([`heavy`]).
const CHEAP: usize = 0;
const HEAVY: usize = 1;
const POOL_NAMES: [&str; 2] = ["cheap", "heavy"];

/// What identifies a title pipeline of `shaders` and `graphics` across
/// runs: its early key (its description is kept under a key of its own,
/// `super::record`'s).
pub(crate) fn key(shaders: &[Arc<Shader>], graphics: Option<&GraphicsState>) -> String {
    let mut key = blake3::Hasher::new();
    for s in shaders {
        key.update(s.stage.as_bytes());
        key.update(s.hash.as_bytes());
    }
    key.update(
        graphics
            .map(|g| g.describe())
            .unwrap_or_default()
            .as_bytes(),
    );
    key.finalize().to_hex()[..32].to_owned()
}

impl Pending {
    /// A title pipeline of `shaders`, with the graphics state `graphics`
    /// read from the title's `stream` (none for a compute pipeline or a
    /// plain description).
    pub(crate) fn new(
        shaders: Vec<Arc<Shader>>,
        graphics: Option<Box<GraphicsState>>,
        stream: Option<super::record::Stream>,
    ) -> Pending {
        Pending {
            key: key(&shaders, graphics.as_deref()),
            shaders,
            graphics,
            queued: AtomicU8::new(IDLE),
            pool: AtomicU8::new(CHEAP as u8),
            made: OnceLock::new(),
            finished: AtomicBool::new(false),
            used: AtomicBool::new(false),
            stream,
            #[cfg(feature = "diagnostics")]
            refusal: OnceLock::new(),
        }
    }

    /// Makes the host pipeline (once); true for the call that made it,
    /// which runs `counted` before signalling it finished: whoever sees
    /// `finished` sees what `counted` counted.
    pub(super) fn make(&self, counted: impl FnOnce()) -> bool {
        let mut first = false;
        self.made.get_or_init(|| {
            first = true;
            #[cfg(feature = "diagnostics")]
            let _ = super::host::take_refusal();
            let started = Instant::now();
            let made = HostPipeline::new(&self.shaders, self.graphics.as_deref());
            let ms = started.elapsed().as_millis() as u64;
            if ms >= SLOW_MS {
                tracing::info!(
                    ms,
                    shaders = ?self.shaders,
                    program_bytes = self.shaders.iter().map(|s| s.container.len()).max(),
                    pool = self.pool_name(),
                    made = made.is_some(),
                    "host pipeline slow to make (its translations and the driver's compile)"
                );
            }
            #[cfg(feature = "diagnostics")]
            if made.is_none()
                && let Some(why) = super::host::take_refusal()
            {
                let _ = self.refusal.set(why);
            }
            made
        });
        if first {
            counted();
            self.finished.store(true, Ordering::Release);
            let (lock, done) = &DONE;
            let held = lock.lock().unwrap_or_else(|e| e.into_inner());
            done.notify_all();
            drop(held);
            // Once the title can draw with it: kept for the precook, made
            // or not (a translator that refuses its shaders today may take
            // them tomorrow). Compute pipelines are the pack's.
            if let (Some(stream), false) = (&self.stream, self.compute()) {
                super::record::made(&self.shaders, stream);
            }
        }
        first
    }

    fn compute(&self) -> bool {
        matches!(&*self.shaders, [cs] if cs.stage == "compute")
    }

    /// The pool it was handed to last.
    fn pool(&self) -> usize {
        usize::from(self.pool.load(Ordering::Relaxed))
    }

    fn pool_name(&self) -> &'static str {
        POOL_NAMES[self.pool()]
    }
}

/// A console program of this many bytes or more (its shader container) is
/// heavy to make. In three Series S sessions of 2026-11-24 (runs
/// `20261124-180617`, `-210555` and `-212205`, cold shader caches), the
/// 171 to 204 containers of 48 KiB or more took 93 to 97 % of all the
/// translation time: 635 to 794 s, up to 38 s each (most of it on programs
/// the translator refuses in the end), and the largest driver compiles
/// (13 s). The 1,116 to 2,016 smaller ones took 24 to 47 s together, the
/// slowest 0.7 to 3.8 s; from 32 to 48 KiB the median was under 0.1 s.
const HEAVY_BYTES: usize = 48 << 10;

/// Whether making `p` may hold a worker for seconds: one of its shaders is
/// a large console program ([`HEAVY_BYTES`]) not known to be refused (a
/// refused translation fails its pipelines at once). It is known before
/// anything is made; the translated size is not: it comes with the
/// translation, the slow part, and a refused program has none.
fn heavy(p: &Pending) -> bool {
    p.shaders
        .iter()
        .any(|s| s.container.len() >= HEAVY_BYTES && !matches!(s.translated.get(), Some(None)))
}

/// Why a pipeline has no host pipeline, and which shaders that concerns
/// (diagnostics): "being made" and the pool making it ("being made
/// (heavy)"); "does not translate", with its shaders that do not;
/// "refused" and the host's reason, with all its shaders. `None` if it has
/// one.
#[cfg(feature = "diagnostics")]
pub(crate) fn missing(p: &Pending) -> Option<(String, Option<String>)> {
    match p.made.get() {
        Some(Some(_)) => None,
        None => Some((format!("being made ({})", p.pool_name()), None)),
        Some(None) => {
            let failed: Vec<String> = p
                .shaders
                .iter()
                .filter(|s| matches!(s.translated.get(), Some(None)))
                .map(|s| format!("{} {}", s.stage, s.hash))
                .collect();
            Some(if failed.is_empty() {
                let why = p
                    .refusal
                    .get()
                    .map_or("refused".into(), |r| format!("refused, {r}"));
                (why, Some(format!("{:?}", p.shaders)))
            } else {
                ("does not translate".into(), Some(failed.join(" + ")))
            })
        }
    }
}

/// Signalled whenever a pipeline has been made.
static DONE: (Mutex<()>, Condvar) = (Mutex::new(()), Condvar::new());

/// A pipeline that takes a worker this long or more is logged.
const SLOW_MS: u64 = 100;

/// Waiting allowed per title frame, over all threads: for graphics
/// pipelines, then for compute ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Budget([Duration; 2]);

impl Budget {
    const DEFAULT: Budget = Budget([Duration::from_millis(2), Duration::from_millis(8)]);

    /// `XTP_PIPELINE_WAIT_MS`: `<graphics>,<compute>` milliseconds, or one
    /// number for both.
    fn parse(text: &str) -> Option<Budget> {
        let ms: Vec<u64> = text
            .split(',')
            .map(|n| n.trim().parse().ok())
            .collect::<Option<_>>()?;
        let [graphics, compute] = match ms[..] {
            [both] => [both, both],
            [graphics, compute] => [graphics, compute],
            _ => return None,
        };
        Some(Budget([graphics, compute].map(Duration::from_millis)))
    }

    /// The run's, read once.
    fn get() -> Budget {
        static BUDGET: OnceLock<Budget> = OnceLock::new();
        *BUDGET.get_or_init(|| {
            let Ok(text) = std::env::var("XTP_PIPELINE_WAIT_MS") else {
                return Budget::DEFAULT;
            };
            match Budget::parse(&text) {
                Some(budget) => {
                    tracing::info!(?budget, "XTP_PIPELINE_WAIT_MS: pipeline waits per frame");
                    budget
                }
                None => {
                    tracing::warn!(
                        text,
                        "XTP_PIPELINE_WAIT_MS expects <graphics>,<compute> milliseconds"
                    );
                    Budget::DEFAULT
                }
            }
        })
    }
}

/// The waiting done in one title frame, per kind (graphics, compute).
struct Frame {
    frame: AtomicU64,
    waited_us: [AtomicU64; 2],
}

impl Frame {
    const fn new() -> Frame {
        Frame {
            frame: AtomicU64::new(u64::MAX),
            waited_us: [AtomicU64::new(0), AtomicU64::new(0)],
        }
    }

    /// What is left of `budget` for `kind` in title frame `frame`.
    fn left(&self, frame: u64, kind: usize, budget: Budget) -> Duration {
        if self.frame.swap(frame, Ordering::Relaxed) != frame {
            for w in &self.waited_us {
                w.store(0, Ordering::Relaxed);
            }
        }
        let waited = Duration::from_micros(self.waited_us[kind].load(Ordering::Relaxed));
        budget.0[kind].saturating_sub(waited)
    }

    fn add(&self, kind: usize, waited: Duration) {
        let us = waited.as_micros() as u64;
        self.waited_us[kind].fetch_add(us, Ordering::Relaxed);
        WAITED_US[kind].fetch_add(us, Ordering::Relaxed);
    }
}

/// Waiting since the last [`take_wait_us`], in microseconds (graphics,
/// compute).
static WAITED_US: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Microseconds title threads waited for pipelines (graphics, compute)
/// since the last call (for the frame line).
pub(crate) fn take_wait_us() -> [u64; 2] {
    [0, 1].map(|k| WAITED_US[k].swap(0, Ordering::Relaxed))
}

/// Pipelines queued and not made yet, in both pools.
static WAITING: AtomicU64 = AtomicU64::new(0);

/// Pipelines queued and not made yet (for the frame line).
pub(crate) fn waiting() -> u64 {
    WAITING.load(Ordering::Relaxed)
}

/// The host pipeline if it is made (or, for a cheap one, is made within
/// the frame's budget); queues it ahead of the early ones on first use.
pub(crate) fn ready(p: &Arc<Pending>) -> Option<&HostPipeline> {
    static FRAME: Frame = Frame::new();
    let frame = crate::frame::current();
    report(frame);
    POOLS.ready(p, &FRAME, frame, Budget::get())
}

/// At the title's creation of a pipeline: starts it at once if an earlier
/// run drew with it (during loading, so the world's first frames find it).
pub(crate) fn early(p: &Arc<Pending>) {
    if !p.shaders.is_empty() && super::library::wanted_early(&p.key) {
        super::library::EARLY.fetch_add(1, Ordering::Relaxed);
        POOLS.start(p, EARLY);
    }
}

pub(super) fn start_builder(p: &Arc<Pending>) {
    POOLS.start(p, EARLY);
}

/// Un seul verdict pour toute la session. Un refus suffit, même si d'autres
/// compilations sont en cours. Le succès exige le dernier pipeline hôte.
fn builder_verdict(states: impl Iterator<Item = Option<bool>>) -> Option<bool> {
    let mut ready = Some(true);
    for state in states {
        match state {
            Some(false) => return Some(false),
            None => ready = None,
            Some(true) => {}
        }
    }
    ready
}

pub(super) fn admit_builder<'a>(
    members: impl Iterator<Item = &'a Arc<Pending>> + Clone,
) -> Option<bool> {
    let state = |p: &Arc<Pending>| p.made.get().map(Option::is_some);
    let admitted = builder_verdict(members.clone().map(state))?;
    let total = members.clone().count();
    if admitted {
        tracing::info!(members = total, "BVH builder admitted");
    } else if let Some(p) = members.clone().find(|p| state(p) == Some(false)) {
        tracing::warn!(members = total, refused = %p.shaders[0].hash, "BVH builder held");
    }
    Some(admitted)
}

#[test]
fn builder_requires_every_host_pipeline() {
    let verdict = |states: &[Option<bool>]| builder_verdict(states.iter().copied());
    assert_eq!(verdict(&[Some(true), None]), None);
    assert_eq!(verdict(&[Some(true), None, Some(false)]), Some(false));
    assert_eq!(verdict(&[Some(false), Some(true)]), Some(false));
    assert_eq!(verdict(&[Some(true), Some(true)]), Some(true));
}

/// Waits for `p` while title frame `frame` has budget left for its kind.
fn wait(p: &Pending, frames: &Frame, frame: u64, budget: Budget) {
    let kind = usize::from(p.compute());
    let left = frames.left(frame, kind, budget);
    if left.is_zero() {
        return;
    }
    let started = Instant::now();
    let (lock, done) = &DONE;
    let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
    while p.made.get().is_none() {
        let rest = left.saturating_sub(started.elapsed());
        if rest.is_zero() {
            break;
        }
        held = done
            .wait_timeout(held, rest)
            .unwrap_or_else(|e| e.into_inner())
            .0;
    }
    drop(held);
    frames.add(kind, started.elapsed());
}

/// One pool's pipelines: first uses first, then those started early. A
/// pipeline started early and then used is in two queues (of this pool or
/// of the other); the early entry is passed over once the pipeline is no
/// longer [`EARLY`].
struct Queues {
    urgent: VecDeque<Arc<Pending>>,
    early: VecDeque<Arc<Pending>>,
    /// Being made by the pool's workers.
    making: Vec<Arc<Pending>>,
    /// Whether a pipeline was passed over since a worker last finished,
    /// for one of `making` ([`Queues::pop`]).
    passed: bool,
}

impl Queues {
    const fn new() -> Queues {
        Queues {
            urgent: VecDeque::new(),
            early: VecDeque::new(),
            making: Vec::new(),
            passed: false,
        }
    }

    /// Queues `p` as `how` ([`EARLY`] or [`URGENT`]) unless it is queued
    /// that way already, or taken: whether it was not queued before, or
    /// `None` if nothing changed.
    fn push(&mut self, p: &Arc<Pending>, how: u8) -> Option<bool> {
        let was = p.queued.load(Ordering::Relaxed);
        if was >= how {
            return None;
        }
        p.queued.store(how, Ordering::Relaxed);
        if how == URGENT {
            self.urgent.push_back(p.clone());
        } else {
            self.early.push_back(p.clone());
        }
        Some(was == IDLE)
    }

    /// The next pipeline to make, taken ([`MAKING`]) and listed as being
    /// made: the first, first uses then early ones, that would not wait
    /// for another worker of the pool. Two pipelines of one program would:
    /// the second worker would wait for the first one's translation, which
    /// takes up to 38 s for a heavy program ([`waits_for`]). Those passed
    /// over keep their place.
    fn pop(&mut self) -> Option<Arc<Pending>> {
        let Queues {
            urgent,
            early,
            making,
            passed,
        } = self;
        let p = take(urgent, making, None, passed)
            .or_else(|| take(early, making, Some(EARLY), passed))?;
        p.queued.store(MAKING, Ordering::Relaxed);
        making.push(p.clone());
        Some(p)
    }

    /// `p` is made: no longer listed as being made. Whether a pipeline was
    /// passed over for one being made since the last call.
    fn made(&mut self, p: &Arc<Pending>) -> bool {
        self.making.retain(|m| !Arc::ptr_eq(m, p));
        std::mem::take(&mut self.passed)
    }
}

/// The first entry of `queue` that does not wait for one of `making`,
/// removed; an entry no longer queued as `live` (an early entry since
/// used, or taken) is dropped on the way. `passed` is set if one was
/// passed over.
fn take(
    queue: &mut VecDeque<Arc<Pending>>,
    making: &[Arc<Pending>],
    live: Option<u8>,
    passed: &mut bool,
) -> Option<Arc<Pending>> {
    let mut i = 0;
    while i < queue.len() {
        let p = &queue[i];
        if live.is_some_and(|how| p.queued.load(Ordering::Relaxed) != how) {
            queue.remove(i);
        } else if waits_for(p, making) {
            *passed = true;
            i += 1;
        } else {
            return queue.remove(i);
        }
    }
    None
}

/// Whether making `p` now would wait for a translation another worker is
/// doing: one of its heavy programs ([`HEAVY_BYTES`]), not translated yet,
/// is a shader of a pipeline in `making`. A small program's translation
/// takes milliseconds: waiting for it is cheaper than passing over.
fn waits_for(p: &Pending, making: &[Arc<Pending>]) -> bool {
    p.shaders.iter().any(|s| {
        s.container.len() >= HEAVY_BYTES
            && s.translated.get().is_none()
            && making
                .iter()
                .any(|m| m.shaders.iter().any(|t| Arc::ptr_eq(s, t)))
    })
}

/// What a pool did since the pools' last line ([`report`]).
struct Counts {
    /// Queued and not made yet (now, not since the last line).
    waiting: AtomicU64,
    /// Pipelines made, the microseconds that took their workers, and the
    /// longest.
    made: AtomicU64,
    make_us: AtomicU64,
    longest_us: AtomicU64,
    /// Asks for a host pipeline (draws and dispatches recorded, in
    /// diagnostics builds also their trace) that found it being made: what
    /// a frame drops "being made".
    unready: AtomicU64,
}

impl Counts {
    const fn new() -> Counts {
        Counts {
            waiting: AtomicU64::new(0),
            made: AtomicU64::new(0),
            make_us: AtomicU64::new(0),
            longest_us: AtomicU64::new(0),
            unready: AtomicU64::new(0),
        }
    }

    /// What it counted since the last call.
    fn take(&self) -> Taken {
        Taken {
            waiting: self.waiting.load(Ordering::Relaxed),
            made: self.made.swap(0, Ordering::Relaxed),
            make_us: self.make_us.swap(0, Ordering::Relaxed),
            longest_us: self.longest_us.swap(0, Ordering::Relaxed),
            unready: self.unready.swap(0, Ordering::Relaxed),
        }
    }
}

/// [`Counts`] taken.
#[derive(Debug, Clone, Copy, Default)]
struct Taken {
    waiting: u64,
    made: u64,
    make_us: u64,
    longest_us: u64,
    unready: u64,
}

/// The workers, their pools' queues, and what they did.
struct Pools {
    /// Both pools' queues ([`CHEAP`], [`HEAVY`]), under one lock: a
    /// pipeline started early in one pool can be used, and queued again,
    /// in the other.
    queues: Mutex<[Queues; 2]>,
    /// Signalled when a pool has a pipeline to make.
    ready: [Condvar; 2],
    /// Workers per pool, started on first use.
    workers: OnceLock<[usize; 2]>,
    counts: [Counts; 2],
}

/// The run's.
static POOLS: Pools = Pools::new();

impl Pools {
    const fn new() -> Pools {
        Pools {
            queues: Mutex::new([Queues::new(), Queues::new()]),
            ready: [Condvar::new(), Condvar::new()],
            workers: OnceLock::new(),
            counts: [Counts::new(), Counts::new()],
        }
    }

    /// The workers per pool, started now if they are not yet.
    fn workers(&'static self) -> [usize; 2] {
        *self.workers.get_or_init(|| {
            let processors = std::thread::available_parallelism().map_or(2, |n| n.get());
            self.spawn(wanted(processors))
        })
    }

    /// Starts `wanted` workers per pool: how many started.
    fn spawn(&'static self, wanted: [usize; 2]) -> [usize; 2] {
        let mut started = [0; 2];
        for pool in [CHEAP, HEAVY] {
            started[pool] = (0..wanted[pool])
                .filter(|&i| {
                    let name = match pool {
                        CHEAP => format!("xtp-pipelines-{i}"),
                        _ => format!("xtp-pipelines-heavy-{i}"),
                    };
                    let spawned = worker(name).spawn(move || self.work(pool));
                    if let Err(e) = &spawned {
                        tracing::warn!(error = %e, pool = POOL_NAMES[pool], "no pipeline worker");
                    }
                    spawned.is_ok()
                })
                .count();
        }
        tracing::info!(
            cheap = started[CHEAP],
            heavy = started[HEAVY],
            heavy_from_bytes = HEAVY_BYTES,
            "pipeline workers"
        );
        started
    }

    /// The host pipeline if it is made, or (for a cheap one) is made within
    /// the frame's budget; queues it ahead of the early ones on first use.
    /// A heavy one is not waited for: it takes seconds, and the frame's
    /// budget is the cheap ones'.
    fn ready<'a>(
        &'static self,
        p: &'a Arc<Pending>,
        frames: &Frame,
        frame: u64,
        budget: Budget,
    ) -> Option<&'a HostPipeline> {
        if p.made.get().is_none() {
            self.start(p, URGENT);
            if !heavy(p) {
                wait(p, frames, frame, budget);
            }
            if p.made.get().is_none() {
                self.counts[p.pool()]
                    .unready
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
        let made = p.made.get()?.as_ref();
        if made.is_some() && !p.used.swap(true, Ordering::Relaxed) {
            super::library::note_used(&p.key);
        }
        made
    }

    /// Hands `p` to its pool's workers ([`pool_for`]): once, and once more
    /// ahead of the early ones on a first use, unless a worker has taken
    /// it already.
    fn start(&'static self, p: &Arc<Pending>, how: u8) {
        if p.queued.load(Ordering::Relaxed) >= how {
            return;
        }
        let workers = self.workers();
        if workers == [0, 0] {
            // No worker: made here instead.
            let was = {
                let _held = self.queues.lock().unwrap_or_else(|e| e.into_inner());
                p.queued.swap(MAKING, Ordering::Relaxed)
            };
            if was == IDLE {
                WAITING.fetch_add(1, Ordering::Relaxed);
                self.counts[p.pool()]
                    .waiting
                    .fetch_add(1, Ordering::Relaxed);
                self.finish(p);
            }
            return;
        }
        let pool = pool_for(p, workers);
        let mut queues = self.queues.lock().unwrap_or_else(|e| e.into_inner());
        let was = p.pool();
        match queues[pool].push(p, how) {
            Some(true) => {
                PEAK.fetch_max(
                    WAITING.fetch_add(1, Ordering::Relaxed) + 1,
                    Ordering::Relaxed,
                );
                self.counts[pool].waiting.fetch_add(1, Ordering::Relaxed);
            }
            Some(false) if was != pool => {
                self.counts[was].waiting.fetch_sub(1, Ordering::Relaxed);
                self.counts[pool].waiting.fetch_add(1, Ordering::Relaxed);
            }
            Some(false) => {}
            None => return,
        }
        p.pool.store(pool as u8, Ordering::Relaxed);
        self.ready[pool].notify_one();
    }

    /// A worker of `pool`: makes its pipelines, first uses first, at the
    /// title's own priority. Below it, the world's busy threads starved the
    /// workers (VIctory #1505: 236 pipelines a minute at the start screen,
    /// 7 a minute in the world), and XTP's Series S world dropped 571 to
    /// 747 draws a frame waiting for a pipeline "being made".
    fn work(&self, pool: usize) {
        let mut done: Option<Arc<Pending>> = None;
        loop {
            let next = {
                let mut queues = self.queues.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(done) = done.take()
                    && queues[pool].made(&done)
                {
                    // What it passed over may be free now.
                    self.ready[pool].notify_all();
                }
                loop {
                    if let Some(p) = queues[pool].pop() {
                        break p;
                    }
                    queues = self.ready[pool]
                        .wait(queues)
                        .unwrap_or_else(|e| e.into_inner());
                }
            };
            self.finish(&next);
            done = Some(next);
        }
    }

    /// Makes `p` and keeps the counts, in place before `p` is signalled
    /// finished.
    fn finish(&self, p: &Pending) {
        let started = Instant::now();
        let made = p.make(|| {
            let us = started.elapsed().as_micros() as u64;
            let counts = &self.counts[p.pool()];
            counts.made.fetch_add(1, Ordering::Relaxed);
            counts.make_us.fetch_add(us, Ordering::Relaxed);
            counts.longest_us.fetch_max(us, Ordering::Relaxed);
            counts.waiting.fetch_sub(1, Ordering::Relaxed);
        });
        if !made {
            return;
        }
        if WAITING.fetch_sub(1, Ordering::Relaxed) == 1 {
            drained();
        } else {
            super::library::save(false);
        }
    }
}

/// The pool that makes `p`: the heavy one for a heavy pipeline
/// ([`heavy`]), else the cheap one; the other one if that has no worker.
fn pool_for(p: &Pending, workers: [usize; 2]) -> usize {
    let pool = if heavy(p) { HEAVY } else { CHEAP };
    if workers[pool] > 0 { pool } else { 1 - pool }
}

/// Stack of each pipeline worker: room for the driver's shader compiler,
/// which runs on it (reserved address space; committed only as used).
/// Every title pipeline is compiled by [`Pending::make`], from
/// [`Pools::finish`]: on a worker, or, when none could start, on the
/// title's thread with the title's stack.
const PIPELINE_STACK: usize = 64 << 20;

/// A pipeline worker's thread, before it starts: its name and its stack.
fn worker(name: String) -> std::thread::Builder {
    std::thread::Builder::new()
        .name(name)
        // The driver compiles pipelines on this thread, recursively:
        // NVIDIA's compiler (nvgpucomp64) overflowed the default 2 MB
        // on a world compute program and refused it as out of memory.
        .stack_size(PIPELINE_STACK)
}

/// Pipeline workers at most, both pools.
///
/// One translation can hold a worker for seconds: in the Series S world
/// 16 compute programs took 20 to 28 s each before the translator refused
/// them (lane masks read as data; docs/analysis/drops.md), and with eight
/// workers a few of them held most of the pool while cheap pipelines
/// waited, with 8 to 10 of the process's threads running on 32 logical
/// processors. The workers run at the title's own priority.
const MAX_WORKERS: usize = 16;

/// Workers of the heavy pool, of 8 workers or more (one below): a few
/// heavy programs at a time, while all the others make cheap pipelines.
const HEAVY_WORKERS: usize = 2;

/// Workers per pool ([`CHEAP`], [`HEAVY`]) for `processors` logical
/// processors: half of them, 2 to [`MAX_WORKERS`], of which
/// [`HEAVY_WORKERS`] (one, of fewer than 8) are the heavy pool's.
fn wanted(processors: usize) -> [usize; 2] {
    let all = (processors / 2).clamp(2, MAX_WORKERS);
    let heavy = if all >= 8 { HEAVY_WORKERS } else { 1 };
    [all - heavy, heavy]
}

/// The most pipelines waiting at once since the workers last ran out.
static PEAK: AtomicU64 = AtomicU64::new(0);

/// The workers ran out of work: logged after a batch of 10 or more, and
/// the cache saved (at most every 20 s).
fn drained() {
    use super::library::{EARLY, LOADED, MADE};
    if PEAK.swap(0, Ordering::Relaxed) >= 10 {
        tracing::info!(
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            "host pipelines all made"
        );
    }
    super::library::save(false);
}

/// The pools' line comes at most this often.
const REPORT_EVERY: Duration = Duration::from_secs(10);

/// The pools' line ([`describe`]), at the first ask for a pipeline of a
/// title frame, [`REPORT_EVERY`] at most; none while the pools have
/// nothing to say.
fn report(frame: u64) {
    static SEEN: AtomicU64 = AtomicU64::new(u64::MAX);
    static LAST: Mutex<Option<(Instant, u64)>> = Mutex::new(None);
    if SEEN.load(Ordering::Relaxed) == frame || SEEN.swap(frame, Ordering::Relaxed) == frame {
        return;
    }
    let Ok(mut last) = LAST.try_lock() else {
        return;
    };
    let now = Instant::now();
    let &mut (at, since) = last.get_or_insert((now, frame));
    if now.duration_since(at) < REPORT_EVERY {
        return;
    }
    *last = Some((now, frame));
    let taken = [CHEAP, HEAVY].map(|pool| POOLS.counts[pool].take());
    if let Some(line) = describe(taken, frame.saturating_sub(since)) {
        tracing::info!("{line}");
    }
}

/// The pools' line for what they did over `frames` title frames: what a
/// frame found being made, what was made and how long that took, and what
/// waits. `None` if nothing was made, found being made, or waits.
fn describe(taken: [Taken; 2], frames: u64) -> Option<String> {
    if taken
        .iter()
        .all(|t| t.made == 0 && t.unready == 0 && t.waiting == 0)
    {
        return None;
    }
    let per_frame = |n: u64| n as f64 / frames.max(1) as f64;
    let time = |us: u64| {
        if us >= 1_000_000 {
            format!("{:.1} s", us as f64 / 1e6)
        } else {
            format!("{} ms", us / 1000)
        }
    };
    let made = |t: &Taken, pool: usize| {
        let name = POOL_NAMES[pool];
        match t.made {
            0 => format!("0 {name}"),
            n => format!(
                "{n} {name} (mean {}, longest {})",
                time(t.make_us / n),
                time(t.longest_us)
            ),
        }
    };
    let [cheap, heavy] = &taken;
    Some(format!(
        "pipeline pools over {frames} title frames: being made when drawn {:.1}/frame heavy, \
         {:.1} cheap; made {}, {}; waiting {} heavy, {} cheap",
        per_frame(heavy.unready),
        per_frame(cheap.unready),
        made(heavy, HEAVY),
        made(cheap, CHEAP),
        heavy.waiting,
        cheap.waiting,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    fn until_finished(p: &Pending) {
        let started = Instant::now();
        while !p.finished.load(Ordering::Acquire) && started.elapsed() < Duration::from_secs(60) {
            std::thread::yield_now();
        }
    }

    /// A shader whose console program is `bytes` long, not translated.
    fn program(stage: &'static str, bytes: usize) -> Arc<Shader> {
        Arc::new(Shader {
            stage,
            hash: format!("{bytes:016x}"),
            ambient: None,
            container: vec![0; bytes],
            native: OnceLock::new(),
            translated: OnceLock::new(),
        })
    }

    /// Pools of their own with `workers` per pool, for a test.
    fn pools(workers: [usize; 2]) -> &'static Pools {
        let pools: &'static Pools = Box::leak(Box::new(Pools::new()));
        let started = pools.spawn(workers);
        assert_eq!(started, workers, "workers started");
        pools.workers.set(started).unwrap();
        pools
    }

    /// A first use queues the pipeline and, within the budget, gets the
    /// worker's answer (here: no host pipeline for no shaders).
    #[test]
    fn first_use_is_made_by_a_worker() {
        let p = Arc::new(Pending::new(Vec::new(), None, None));
        assert!(ready(&p).is_none());
        // Queued for a first use, or taken by a worker already.
        assert!(p.queued.load(Ordering::Relaxed) >= URGENT);
        until_finished(&p);
        assert!(matches!(p.made.get(), Some(None)));
    }

    /// A pipeline that is not made in time costs a frame its budget once:
    /// later uses in that frame drop at once, the next frame waits again.
    #[test]
    fn waits_stop_at_the_frames_budget() {
        let never = Pending::new(Vec::new(), None, None);
        let frames = Frame::new();
        let budget = Budget([Duration::from_millis(30), Duration::from_millis(60)]);
        let timed = |frame| {
            let started = Instant::now();
            wait(&never, &frames, frame, budget);
            started.elapsed()
        };
        let first = timed(7);
        assert!(first >= Duration::from_millis(30), "{first:?}");
        assert!(timed(7) < Duration::from_millis(10), "the budget is spent");
        assert!(
            timed(8) >= Duration::from_millis(30),
            "a new frame waits again"
        );
        // Compute pipelines have their own.
        assert_eq!(frames.left(8, 1, budget), Duration::from_millis(60));
        assert!(take_wait_us()[0] >= 60_000);
    }

    /// A worker's stack, as the thread itself sees it, is the 64 MiB kept
    /// for the driver's compiler, not Rust's default 2 MiB (VIctory #1129),
    /// and only reserved: what is committed at its start is less than the
    /// whole default stack.
    #[test]
    fn workers_have_room_for_the_drivers_compiler() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery,
        };
        let (reserved, committed) = worker("xtp-pipelines-test".into())
            .spawn(|| {
                let (mut low, mut high) = (0, 0);
                // SAFETY: two out-pointers to this thread's own locals.
                unsafe {
                    windows::Win32::System::Threading::GetCurrentThreadStackLimits(
                        &mut low, &mut high,
                    );
                }
                let (mut at, mut committed) = (low, 0);
                while at < high {
                    // SAFETY: a query of our own stack's pages into a local.
                    let info = unsafe {
                        let mut info = std::mem::zeroed::<MEMORY_BASIC_INFORMATION>();
                        let n = VirtualQuery(at as *const _, &mut info, size_of_val(&info));
                        assert_ne!(n, 0, "VirtualQuery");
                        info
                    };
                    let end = (info.BaseAddress as usize + info.RegionSize).min(high);
                    if info.State == MEM_COMMIT {
                        committed += end - at;
                    }
                    at = end;
                }
                (high - low, committed)
            })
            .unwrap()
            .join()
            .unwrap();
        assert!(reserved >= PIPELINE_STACK, "{reserved} bytes reserved");
        assert!(
            committed < 2 << 20,
            "{committed} bytes committed at the start"
        );
    }

    #[test]
    fn budgets_read_one_or_two_numbers() {
        let ms = |g, c| Some(Budget([Duration::from_millis(g), Duration::from_millis(c)]));
        assert_eq!(Budget::parse("5"), ms(5, 5));
        assert_eq!(Budget::parse(" 1, 20"), ms(1, 20));
        assert_eq!(Budget::parse("0,0"), ms(0, 0));
        assert_eq!(Budget::parse("1,2,3"), None);
        assert_eq!(Budget::parse("fast"), None);
        assert_eq!(Budget::parse(""), None);
    }

    /// A pipeline started early and then used is queued again, ahead, and
    /// made (and counted) once.
    #[test]
    fn early_then_used_is_made_once() {
        let p = Arc::new(Pending::new(Vec::new(), None, None));
        POOLS.start(&p, EARLY);
        POOLS.start(&p, URGENT);
        POOLS.start(&p, EARLY);
        assert!(p.queued.load(Ordering::Relaxed) >= URGENT);
        until_finished(&p);
        assert_eq!(p.queued.load(Ordering::Relaxed), MAKING);
        assert!(matches!(p.made.get(), Some(None)));
        assert!(!p.make(|| {}), "already finished");
    }

    /// First uses come out first; a pipeline started early and then used
    /// comes out once, at its first use's place, and is not handed out
    /// again once taken (VIctory #546's test).
    #[test]
    fn first_uses_come_first_and_each_pipeline_once() {
        let pending = || Arc::new(Pending::new(Vec::new(), None, None));
        let (earlier, promoted, used) = (pending(), pending(), pending());
        let mut queues = Queues::new();
        assert_eq!(queues.push(&earlier, EARLY), Some(true));
        assert_eq!(queues.push(&promoted, EARLY), Some(true));
        assert_eq!(queues.push(&used, URGENT), Some(true));
        assert_eq!(queues.push(&promoted, URGENT), Some(false));
        assert_eq!(queues.push(&promoted, URGENT), None);
        assert_eq!(queues.push(&promoted, EARLY), None);

        assert!(Arc::ptr_eq(&queues.pop().unwrap(), &used));
        assert!(Arc::ptr_eq(&queues.pop().unwrap(), &promoted));
        assert_eq!(promoted.queued.load(Ordering::Relaxed), MAKING);
        assert_eq!(queues.push(&promoted, URGENT), None, "taken already");
        assert!(Arc::ptr_eq(&queues.pop().unwrap(), &earlier));
        assert!(queues.pop().is_none(), "promoted's early entry passed over");
        assert_eq!(queues.making.len(), 3, "all three being made");
        assert!(!queues.made(&used), "nothing passed over");
        assert_eq!(queues.making.len(), 2);
    }

    /// A pipeline is heavy when one of its console programs is 48 KiB or
    /// more, unless that program is known to be refused.
    #[test]
    fn large_programs_make_heavy_pipelines() {
        let pipeline = |shaders: Vec<Arc<Shader>>| Pending::new(shaders, None, None);
        let small = program("vertex", HEAVY_BYTES - 1);
        let large = program("pixel", HEAVY_BYTES);
        assert!(!heavy(&pipeline(vec![small.clone()])));
        assert!(heavy(&pipeline(vec![large.clone()])));
        assert!(heavy(&pipeline(vec![small.clone(), large.clone()])));
        assert!(!heavy(&pipeline(Vec::new())));
        assert!(large.translated.set(None).is_ok());
        assert!(
            !heavy(&pipeline(vec![small, large])),
            "a refused translation fails at once"
        );
        assert_eq!(
            pool_for(&pipeline(vec![program("compute", 1 << 20)]), [3, 0]),
            CHEAP,
            "no heavy worker"
        );
        assert_eq!(
            pool_for(&pipeline(Vec::new()), [0, 1]),
            HEAVY,
            "no cheap worker"
        );
    }

    /// Half the logical processors, 2 to 16, of which 2 (1 of fewer than
    /// 8 workers) for the heavy pipelines.
    #[test]
    fn workers_split_between_the_pools() {
        assert_eq!(wanted(32), [14, 2]);
        assert_eq!(wanted(64), [14, 2]);
        assert_eq!(wanted(16), [6, 2]);
        assert_eq!(wanted(12), [5, 1]);
        assert_eq!(wanted(4), [1, 1]);
        assert_eq!(wanted(1), [1, 1]);
    }

    /// A pipeline whose heavy program another worker is translating is
    /// passed over, keeps its place, and is taken once that one is made.
    /// A small program shared with one being made does not hold it back.
    #[test]
    fn a_pipeline_waiting_for_a_translation_is_passed_over() {
        let large = program("compute", HEAVY_BYTES);
        let small = program("vertex", 64);
        let pending = |shaders: Vec<Arc<Shader>>| Arc::new(Pending::new(shaders, None, None));
        let (first, twin, other) = (
            pending(vec![large.clone()]),
            pending(vec![large.clone()]),
            pending(vec![small.clone(), program("pixel", HEAVY_BYTES + 4)]),
        );
        let mut queues = Queues::new();
        for p in [&first, &twin, &other] {
            assert_eq!(queues.push(p, URGENT), Some(true));
        }
        assert!(Arc::ptr_eq(&queues.pop().unwrap(), &first));
        assert!(
            Arc::ptr_eq(&queues.pop().unwrap(), &other),
            "the twin waits for the first's translation"
        );
        let sharing = pending(vec![small.clone()]);
        assert_eq!(queues.push(&sharing, URGENT), Some(true));
        assert!(
            Arc::ptr_eq(&queues.pop().unwrap(), &sharing),
            "a small program is not waited for"
        );
        assert!(queues.pop().is_none(), "the twin still waits");
        // The first is made (its program refused).
        assert!(large.translated.set(None).is_ok());
        assert!(queues.made(&first), "the twin was passed over");
        assert!(Arc::ptr_eq(&queues.pop().unwrap(), &twin));
        assert!(!queues.made(&twin));
    }

    #[test]
    fn the_pools_line_says_what_each_pool_did() {
        let quiet = [Taken::default(); 2];
        assert_eq!(describe(quiet, 10), None);
        let cheap = Taken {
            waiting: 3,
            made: 120,
            make_us: 120 * 35_000,
            longest_us: 900_000,
            unready: 3_272,
        };
        let heavy = Taken {
            waiting: 10,
            made: 4,
            make_us: 4 * 12_100_000,
            longest_us: 27_000_000,
            unready: 851,
        };
        assert_eq!(
            describe([cheap, heavy], 10).unwrap(),
            "pipeline pools over 10 title frames: being made when drawn 85.1/frame heavy, \
             327.2 cheap; made 4 heavy (mean 12.1 s, longest 27.0 s), 120 cheap (mean 35 ms, \
             longest 900 ms); waiting 10 heavy, 3 cheap"
        );
        let waiting = Taken {
            waiting: 1,
            ..Taken::default()
        };
        assert!(
            describe([Taken::default(), waiting], 0)
                .unwrap()
                .contains("made 0 heavy, 0 cheap")
        );
    }

    /// Words a big console compute program inserts before its store, to be
    /// heavy ([`HEAVY_BYTES`]).
    const HEAVY_WORDS: usize = 16 << 10;

    /// A compute shader of a console program,
    /// `crate::shader_cache::wave64_program(false)` with `extra` ALU
    /// instructions on the stored value before its store (`salt` makes
    /// another program of the same size); its container is the program,
    /// which tells a heavy one. Its translation, by the translator for
    /// this host, is made at once, or, `held`, on a thread of its own that
    /// holds it until the sender returned sends or is dropped: a slow
    /// translation, as long as the test needs, which a worker making the
    /// pipeline waits for as for another worker's.
    fn compute(extra: usize, salt: u8, held: bool) -> (Arc<Shader>, Option<mpsc::Sender<()>>) {
        let (c, mut native) = crate::shader_cache::wave64_program(false);
        let mut words = vec![
            // v_add_nc_u32 v2, <salt>, v2
            0x4A04_0400 | (0x80 + u32::from(salt % 64)),
        ];
        words.extend((1..extra).map(|i| {
            if i % 2 == 0 {
                0x4A04_0408 // v_add_nc_u32 v2, s8, v2
            } else {
                0x3404_0481 // v_lshlrev_b32 v2, 1, v2
            }
        }));
        // After the prologue (7 words) and the value's read (1 word).
        let at = 8 * 4;
        native
            .code
            .splice(at..at, words.iter().flat_map(|w| w.to_le_bytes()));
        let shader = Arc::new(Shader {
            stage: "compute",
            hash: format!("{:08x}{salt:08x}", native.code.len()),
            ambient: None,
            container: native.code.clone(),
            native: OnceLock::new(),
            translated: OnceLock::new(),
        });
        let translate = move || {
            let host =
                xtp_shader::emit::translate_program_for(&c, &native, crate::device::host_target())
                    .unwrap_or_else(|(step, e)| panic!("{}: {e}", step.name()));
            Some(super::super::Translated {
                host,
                used: Vec::new(),
                wave64: native.wave64,
            })
        };
        if !held {
            assert!(shader.translated.set(translate()).is_ok());
            return (shader, None);
        }
        let (entered, translating) = mpsc::channel();
        let (release, released) = mpsc::channel::<()>();
        let s = shader.clone();
        std::thread::spawn(move || {
            s.translated.get_or_init(|| {
                entered.send(()).unwrap();
                let _ = released.recv();
                translate()
            });
        });
        translating.recv().unwrap();
        (shader, Some(release))
    }

    /// A heavy and a cheap pipeline asked for together, the heavy one
    /// first, made on the host (WARP in the check): the cheap one is ready
    /// first, within the frame's budget, while the heavy one, not waited
    /// for, is still being made (its translation held); each pool counts
    /// what it made and what was found being made. With one queue for
    /// both, the cheap one waits until the heavy one is made.
    #[test]
    fn a_cheap_pipeline_does_not_wait_behind_a_heavy_one() {
        let test = "pipeline::make::tests::a_cheap_pipeline_does_not_wait_behind_a_heavy_one";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let pipeline = |extra, salt, held| {
            let (shader, release) = compute(extra, salt, held);
            (Arc::new(Pending::new(vec![shader], None, None)), release)
        };
        let frames = Frame::new();

        let split = pools([1, 1]);
        let ((heavy, release), (cheap, _)) =
            (pipeline(HEAVY_WORDS, 1, true), pipeline(0, 2, false));
        assert!(super::heavy(&heavy) && !super::heavy(&cheap));
        let budget = Budget([Duration::from_secs(30); 2]);
        assert!(
            split.ready(&heavy, &frames, 1, budget).is_none(),
            "a heavy pipeline is not waited for"
        );
        assert!(
            split.ready(&cheap, &frames, 1, budget).is_some(),
            "the cheap one is made within the budget"
        );
        assert!(
            !heavy.finished.load(Ordering::Relaxed),
            "the cheap one is ready first"
        );
        drop(release);
        until_finished(&heavy);
        assert!(
            matches!(heavy.made.get(), Some(Some(_))),
            "the heavy one is made too"
        );
        // Each pool's counts are in place once its pipeline is finished
        // (`ready` returns the cheap one as soon as it is made).
        until_finished(&cheap);
        let [cheap_did, heavy_did] = split.counts.each_ref().map(Counts::take);
        assert_eq!(
            (heavy_did.made, heavy_did.unready, heavy_did.waiting),
            (1, 1, 0)
        );
        assert_eq!(
            (cheap_did.made, cheap_did.unready, cheap_did.waiting),
            (1, 0, 0)
        );
        // No comparison of the two pools' longest times: on a loaded check
        // machine the cheap pipeline's creation can outlast the heavy one's
        // (its hold ends as soon as `release` drops), and the order is
        // already pinned by "the cheap one is ready first" above.

        // One worker, one queue: the cheap one waits behind.
        let one = pools([1, 0]);
        let ((heavy, release), (cheap, _)) =
            (pipeline(HEAVY_WORDS, 3, true), pipeline(0, 4, false));
        let short = Budget([Duration::from_millis(300); 2]);
        assert!(one.ready(&heavy, &frames, 2, short).is_none());
        assert!(
            one.ready(&cheap, &frames, 2, short).is_none(),
            "with one queue, the cheap one is not made while the heavy one is"
        );
        assert!(!cheap.finished.load(Ordering::Relaxed));
        drop(release);
        until_finished(&cheap);
        assert!(
            heavy.finished.load(Ordering::Relaxed),
            "with one queue, the cheap one is made after the heavy one"
        );
        assert!(matches!(cheap.made.get(), Some(Some(_))));
    }
}
