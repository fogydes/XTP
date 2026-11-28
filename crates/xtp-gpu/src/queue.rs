//! Command queues. Submitted lists run on the host GPU in submission order
//! (all queues on one host queue); a fence signal takes effect once the
//! submissions before it have finished.
//!
//! A queue told to wait for a fence holds what it is given after the wait
//! ([`Held`]) until the wait is over: the fence is there, or a queue signal
//! made before gets it there (what is submitted after that signal runs
//! after the work before it, on the one host queue). That is how the
//! console orders work between queues: the title's compute queue waits for
//! what its graphics queue draws.
//!
//! Ported from fxceless's VIctory fork (commit 578d459): its `Held`,
//! `hold`, `release_held`, `wait` and `signal`, reworked for submissions
//! that run while the title goes on. There a wait was over only once the
//! fence was set, which here would wait for the GPU before every held
//! submission; it held whole submissions for waits for memory values too,
//! and forced every queue at each present.
//!
//! A title list may also wait for a memory value ([`wait32`], [`wait64`]:
//! `Wait32BitValueX`, `Wait64BitValueX`) that a list on another queue
//! stores at its end, or the CPU writes. BTP 6's compute queues start their
//! lists so, and its graphics lists wait for words the compute lists write:
//! run in the order the title submits them, the characters' depth pass ran
//! before the G-buffer it reads. The queue holds such a list, and what it
//! is given after it, until every value the list waits for is met
//! ([`met`]): title memory has it, or a bottom-of-pipe write in a
//! submission made so far stores it (what is submitted now runs after that
//! submission, on the one host queue). A submission is cut before each
//! list that waits, so that the lists before it run at once.
//!
//! Ported from Marvisak/VIctory (9cc37c5f5b) by Marvis: its list waits and
//! their comparisons, held per queue in order and released once met. There
//! a list is also cut where it waits after commands, and the part a queue
//! holds keeps its own host list when the title resets the list; here the
//! whole list waits, and a reset runs it first, counted
//! (docs/analysis/cross-queue-waits.md).
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
#[cfg(windows)]
use std::sync::{Condvar, Mutex, MutexGuard};

use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, QUEUE, S_OK, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

const QUEUE_IIDS: [Iid; 4] = [
    guid(
        0x0ec870a6,
        0x5d7e,
        0x4c22,
        [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct QueueDesc {
    kind: u32,
    priority: i32,
    flags: u32,
    nodes: u32,
}

/// The title reads the list type right after the header.
#[repr(C)]
struct Queue {
    object: Object,
    kind: u32,
    desc: QueueDesc,
    /// What the queue holds back. Locked while what it runs is submitted,
    /// so that the queue's order holds whichever thread submits.
    #[cfg(windows)]
    line: Mutex<Line>,
}
const _: () = assert!(std::mem::offset_of!(Queue, kind) == 0x10);

/// What a queue holds back behind a wait for a fence, in the order it was
/// given.
#[cfg(windows)]
#[derive(Default)]
struct Line {
    held: std::collections::VecDeque<Held>,
    /// When the queue last moved, while it holds something.
    since: Option<std::time::Instant>,
    /// Exercise fence wake-ups without the stuck-queue fallback.
    #[cfg(test)]
    wait_indefinitely: bool,
    /// How long the queue may stand still before it runs on ([`STUCK`]
    /// otherwise).
    #[cfg(test)]
    patience: Option<std::time::Duration>,
    /// Whether it is in [`WAITING`], with a reference to the queue (so a
    /// queue the title releases runs what it holds first).
    listed: bool,
}

/// One thing a queue holds back, with a reference kept to each object it
/// names.
#[cfg(windows)]
enum Held {
    /// A wait for the fence (its address) to reach the value.
    Wait(usize, u64),
    /// The memory values the next list waits for, since when it waits, and
    /// whether it was found not met (for the frame line).
    Values {
        waits: Vec<ValueWait>,
        since: std::time::Instant,
        waited: bool,
    },
    /// A submission: the title's lists, whose host sides run when the
    /// queue gets to it. The title records into a list again only after a
    /// reset, and a reset runs a held list first ([`list_reset`]).
    Run(Vec<usize>),
    /// A signal of the queue's, made once what it holds before it has run.
    Signal(usize, u64),
}

/// The queues that hold something (their addresses).
#[cfg(windows)]
static WAITING: Mutex<Vec<usize>> = Mutex::new(Vec::new());

#[cfg(windows)]
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// One of our objects' methods by its slot, as the title calls it.
#[cfg(windows)]
fn method<F: Copy>(object: This, slot: usize) -> F {
    // SAFETY: one of our objects, whose first field is its table, which has
    // the slot; the caller names the method's signature.
    unsafe {
        let table = *object.cast::<*const *const ()>();
        std::mem::transmute_copy(&*table.add(slot))
    }
}

/// A fence's value (`GetCompletedValue`, slot 8).
#[cfg(windows)]
fn completed(fence: usize) -> u64 {
    method::<extern "system" fn(This) -> u64>(fence as This, 8)(fence as This)
}

/// Drops a reference kept to one of our objects (`Release`, slot 2: the
/// last one frees it).
#[cfg(windows)]
fn release_object(object: usize) {
    method::<extern "system" fn(This) -> u32>(object as This, 2)(object as This);
}

/// Queue signals made and not done yet: the fence (its address) and the
/// value it is set to once the submissions before the signal have finished.
#[cfg(windows)]
static ISSUED: Mutex<Vec<(usize, u64)>> = Mutex::new(Vec::new());

/// Whether a wait for `fence` to reach `value` is over for what is
/// submitted from now on: the fence is there, or a queue signal made so
/// far gets it there (what is submitted now runs after the work before
/// that signal: one host queue runs every queue's submissions in order).
#[cfg(windows)]
fn over(fence: usize, value: u64) -> bool {
    completed(fence) >= value || lock(&ISSUED).iter().any(|&(f, v)| f == fence && v >= value)
}

/// Whether a queue holds what follows a wait that is not over (on by
/// default). `XTP_QUEUE_WAIT=0`: it runs it at once, as before, and such
/// waits are only counted (to compare frame times).
#[cfg(windows)]
fn holds() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("XTP_QUEUE_WAIT").is_none_or(|v| v != "0"))
}

/// A wait for a memory value a title list records ([`wait32`],
/// [`wait64`]): on the console the queue stops there until the value at
/// `address` (`bytes` wide, masked) compares true with `reference`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ValueWait {
    pub(crate) address: u64,
    pub(crate) bytes: u8,
    /// A D3D12 comparison function: 1 never, 2 less, 3 equal, 4 less or
    /// equal, 5 greater, 6 not equal, 7 greater or equal, 8 always.
    pub(crate) comparison: u32,
    pub(crate) reference: u64,
    pub(crate) mask: u64,
    /// Whether the list recorded work before it: the console waits there,
    /// after that work; here the whole list waits.
    pub(crate) work_before: bool,
    /// The list's own bottom-of-pipe writes recorded before it. Another
    /// queue may wait for them; here they land only once the whole list has
    /// run, after this wait.
    pub(crate) writes_before: u32,
}

impl ValueWait {
    fn new(address: u64, bytes: u8, comparison: u32, reference: u64, mask: u64) -> ValueWait {
        ValueWait {
            address,
            bytes,
            comparison,
            reference,
            mask,
            work_before: false,
            writes_before: 0,
        }
    }

    /// Whether memory holding `value` meets the wait: `None` for a
    /// comparison we do not know.
    pub(crate) fn meets(&self, value: u64) -> Option<bool> {
        let (v, r) = (value & width(self.bytes) & self.mask, self.reference);
        Some(match self.comparison {
            1 => false,
            2 => v < r,
            3 => v == r,
            4 => v <= r,
            5 => v > r,
            6 => v != r,
            7 => v >= r,
            8 => true,
            _ => return None,
        })
    }
}

/// The bits a value of `bytes` (4 or 8) has.
fn width(bytes: u8) -> u64 {
    if bytes >= 8 {
        u64::MAX
    } else {
        (1 << (8 * u32::from(bytes))) - 1
    }
}

/// What a write of the low `bytes` of `value` at `address` stores at `at`
/// (`n` bytes), if it covers all of it.
fn stored(address: u64, value: u64, bytes: u8, at: u64, n: u8) -> Option<u64> {
    let offset = at.checked_sub(address)?;
    let inside = offset + u64::from(n) <= u64::from(bytes);
    inside.then(|| ((value & width(bytes)) >> (8 * offset)) & width(n))
}

/// `Wait32BitValueX(address, comparison, reference, mask, flags)`: the
/// list waits there until the masked 32-bit value compares true. Noted in
/// its host list; its queue submits the list once the value is met.
pub(crate) extern "system" fn wait32(
    this: This,
    address: u64,
    comparison: u32,
    reference: u32,
    mask: u32,
    _flags: u32,
) {
    let wait = ValueWait::new(address, 4, comparison, reference.into(), mask.into());
    record_wait(this, wait);
}

/// `Wait64BitValueX(address, comparison, reference, flags)`, as [`wait32`].
pub(crate) extern "system" fn wait64(
    this: This,
    address: u64,
    comparison: u32,
    reference: u64,
    _flags: u32,
) {
    record_wait(
        this,
        ValueWait::new(address, 8, comparison, reference, u64::MAX),
    );
}

fn record_wait(list: This, wait: ValueWait) {
    #[cfg(windows)]
    if let Some(host) = crate::list::host_list(list) {
        host.defer_value_wait(wait);
        return;
    }
    // Without a host GPU nothing runs, so nothing waits.
    let _ = (list, wait);
}

/// How a memory value a list waits for is met, if it is.
#[cfg(windows)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Met {
    /// Title memory has it.
    Memory,
    /// A list before it in the same submission stores it.
    Earlier,
    /// A bottom-of-pipe write in a title submission made so far stores it:
    /// the queue (its address) that made that submission, and its kind.
    Written(usize, u32),
    /// A comparison we do not know: taken as met (counted).
    Unknown,
    /// Not met: what memory holds, if it could be read.
    No(Option<u64>),
}

/// A bottom-of-pipe value write in a title submission not finished yet.
#[cfg(windows)]
#[derive(Clone, Copy)]
struct Write {
    address: u64,
    value: u64,
    bytes: u8,
    /// The queue (its address, 0 for none) that made the submission, and
    /// its kind.
    queue: usize,
    kind: u32,
    /// Which submission: its writes go once it has finished.
    run: u64,
}

/// The value writes of title submissions made and not finished yet: a
/// list submitted now runs after them (one host queue).
#[cfg(windows)]
static WRITES: Mutex<Vec<Write>> = Mutex::new(Vec::new());

/// Whether `w` is met for a list submitted now, after the lists whose
/// value writes are `before` (in order) on its queue.
#[cfg(windows)]
fn met(w: &ValueWait, before: &[(u64, u64, u8)]) -> Met {
    if w.meets(0).is_none() {
        return Met::Unknown;
    }
    let now = read_value(w.address, w.bytes);
    if now.and_then(|v| w.meets(v)) == Some(true) {
        return Met::Memory;
    }
    // The last of them is what the list finds.
    let last = before
        .iter()
        .rev()
        .find_map(|&(address, value, bytes)| stored(address, value, bytes, w.address, w.bytes));
    if last.and_then(|v| w.meets(v)) == Some(true) {
        return Met::Earlier;
    }
    let meets = |p: &&Write| {
        let value = stored(p.address, p.value, p.bytes, w.address, w.bytes);
        value.and_then(|v| w.meets(v)) == Some(true)
    };
    match lock(&WRITES).iter().find(meets) {
        Some(p) => Met::Written(p.queue, p.kind),
        None => Met::No(now),
    }
}

/// The value a wait tests (`bytes` of title memory), if it can be read.
#[cfg(windows)]
fn read_value(address: u64, bytes: u8) -> Option<u64> {
    let mut value = [0u8; 8];
    let into = &mut value[..usize::from(bytes.min(8))];
    crate::host::read_into(address as *const u8, into).then(|| u64::from_le_bytes(value))
}

/// Whether a queue holds a list until the memory values it waits for are
/// met (on by default). `XTP_VALUE_WAIT=0`: it runs at once, as before,
/// and the waits are only measured.
#[cfg(windows)]
fn holds_values() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("XTP_VALUE_WAIT").is_none_or(|v| v != "0"))
}

/// Why a queue runs something before its waits are over.
#[cfg(windows)]
#[derive(Clone, Copy, Debug)]
enum Why {
    /// The title resets a list the queue holds, to record into it again.
    Reset,
    /// The queue presents.
    Present,
    /// The queue maps pages.
    Mapping,
    /// The queue stood still for [`STUCK`].
    Stuck,
}

#[cfg(windows)]
impl Why {
    fn text(self) -> &'static str {
        match self {
            Why::Reset => "the title resets a list it holds",
            Why::Present => "it presents",
            Why::Mapping => "it maps pages",
            Why::Stuck => "it stood still too long",
        }
    }
}

/// For the frame line, title lists' waits for memory values, at
/// submission: all, and how they were met then (title memory; a write
/// submitted before on the same queue or another; a list before them in
/// the submission), or not; with a comparison we do not know (taken as
/// met); not met and unreadable; recorded after work, and after the list's
/// own bottom-of-pipe writes. Then lists held for them and released once
/// met by title memory (the CPU, or a write that landed), a write of the
/// same queue or one of another; lists run before they were met, by
/// [`Why`]; and the microseconds held lists waited.
static VALUES: [AtomicU64; 18] = [const { AtomicU64::new(0) }; 18];
const V_WAITS: usize = 0;
const V_MEMORY: usize = 1;
const V_HERE: usize = 2;
const V_ELSEWHERE: usize = 3;
const V_EARLIER: usize = 4;
const V_UNMET: usize = 5;
const V_UNKNOWN: usize = 6;
const V_UNREADABLE: usize = 7;
const V_AFTER_WORK: usize = 8;
const V_AFTER_WRITE: usize = 9;
const V_BY_MEMORY: usize = 10;
const V_BY_HERE: usize = 11;
const V_BY_ELSEWHERE: usize = 12;
/// Then one per [`Why`], in its order.
const V_FORCED: usize = 13;
const V_HELD_US: usize = 17;

/// Whether one more line of a kind (`lines` so far) is logged:
/// `XTP_VALUE_WAIT_LOG` of each (64 by default), and at least `least`.
#[cfg(windows)]
fn log_line(lines: &AtomicU64, least: u64) -> bool {
    static LIMIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let limit = *LIMIT.get_or_init(|| {
        let set = std::env::var("XTP_VALUE_WAIT_LOG").ok();
        set.and_then(|v| v.parse().ok()).unwrap_or(64)
    });
    lines.fetch_add(1, Ordering::Relaxed) < limit.max(least)
}

/// Counts and logs a wait of a list submitted on queue `queue` (`kind`),
/// as it stands then.
#[cfg(windows)]
fn note_submitted(queue: usize, kind: u32, w: &ValueWait, met: Met) {
    static LINES: AtomicU64 = AtomicU64::new(0);
    let count = |i: usize| VALUES[i].fetch_add(1, Ordering::Relaxed);
    count(V_WAITS);
    count(match met {
        Met::Memory => V_MEMORY,
        Met::Earlier => V_EARLIER,
        Met::Written(q, _) if q == queue => V_HERE,
        Met::Written(..) => V_ELSEWHERE,
        Met::Unknown => V_UNKNOWN,
        Met::No(_) => V_UNMET,
    });
    if met == Met::No(None) {
        count(V_UNREADABLE);
    }
    if w.work_before {
        count(V_AFTER_WORK);
    }
    if w.writes_before > 0 {
        count(V_AFTER_WRITE);
    }
    if log_line(&LINES, 0) {
        tracing::info!(
            frame = presented(),
            queue = format_args!("{queue:#x}"),
            kind,
            address = format_args!("{:#x}", w.address),
            bytes = w.bytes,
            comparison = w.comparison,
            reference = format_args!("{:#x}", w.reference),
            mask = format_args!("{:#x}", w.mask),
            work_before = w.work_before,
            writes_before = w.writes_before,
            ?met,
            "a submitted list waits for a memory value"
        );
    }
}

/// Counts and logs a held list whose memory values are met now: by what
/// (title memory, a write of its queue `queue`, or of another queue's),
/// after `held`.
#[cfg(windows)]
fn note_released(
    queue: usize,
    kind: u32,
    waits: &[ValueWait],
    before: &[(u64, u64, u8)],
    held: std::time::Duration,
) {
    static LINES: AtomicU64 = AtomicU64::new(0);
    let mets: Vec<Met> = waits.iter().map(|w| met(w, before)).collect();
    let writer = mets.iter().find_map(|m| match *m {
        Met::Written(q, k) if q != queue => Some(k),
        _ => None,
    });
    let here = mets
        .iter()
        .any(|m| matches!(*m, Met::Earlier) || matches!(*m, Met::Written(q, _) if q == queue));
    let by = match (writer, here) {
        (Some(_), _) => V_BY_ELSEWHERE,
        (None, true) => V_BY_HERE,
        (None, false) => V_BY_MEMORY,
    };
    VALUES[by].fetch_add(1, Ordering::Relaxed);
    VALUES[V_HELD_US].fetch_add(held.as_micros() as u64, Ordering::Relaxed);
    if log_line(&LINES, 0) {
        let addresses: Vec<String> = waits.iter().map(|w| format!("{:#x}", w.address)).collect();
        tracing::info!(
            frame = presented(),
            queue = format_args!("{queue:#x}"),
            kind,
            ?addresses,
            ?mets,
            writer_kind = ?writer,
            held_us = held.as_micros() as u64,
            "a held list's memory values are met: it runs"
        );
    }
}

/// Counts and logs a held list run before the memory values it waits for
/// are met (`unmet`, with what memory holds), after `held`.
#[cfg(windows)]
fn note_forced(
    queue: usize,
    kind: u32,
    unmet: &[(ValueWait, Option<u64>)],
    held: std::time::Duration,
    why: Why,
) {
    static LINES: AtomicU64 = AtomicU64::new(0);
    VALUES[V_FORCED + why as usize].fetch_add(1, Ordering::Relaxed);
    VALUES[V_HELD_US].fetch_add(held.as_micros() as u64, Ordering::Relaxed);
    if log_line(&LINES, 16) {
        for (w, now) in unmet {
            tracing::warn!(
                frame = presented(),
                queue = format_args!("{queue:#x}"),
                kind,
                address = format_args!("{:#x}", w.address),
                comparison = w.comparison,
                reference = format_args!("{:#x}", w.reference),
                now = ?now.map(|v| format!("{v:#x}")),
                held_ms = held.as_millis() as u64,
                why = why.text(),
                "a list runs before a memory value it waits for is met (the console would wait for it)"
            );
        }
    }
}

/// For the frame line: waits for fences; those not over when made (with
/// [`holds`], the queue held what came next); title lists held; and waits
/// passed before they were over ([`Force`]).
static COUNTS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
const WAITS: usize = 0;
const HOLDS: usize = 1;
const HELD_LISTS: usize = 2;
const FORCED: usize = 3;

/// A queue that has not moved for this long while its wait is not over
/// runs on past it.
///
/// shortcut: on the console it would wait for ever. A signal XTP does not
/// see as one (a title write we miss) would then stop the queue for good;
/// past this the queue runs on, logged, as it did before waits held.
#[cfg(windows)]
const STUCK: std::time::Duration = std::time::Duration::from_secs(4);

#[cfg(windows)]
impl Line {
    fn stuck(&self) -> bool {
        #[cfg(test)]
        if self.wait_indefinitely {
            return false;
        }
        #[cfg(test)]
        let patience = self.patience.unwrap_or(STUCK);
        #[cfg(not(test))]
        let patience = STUCK;
        self.since.is_some_and(|t| t.elapsed() >= patience)
    }
}

/// What a queue runs before its waits are over.
#[cfg(windows)]
#[derive(Clone, Copy)]
enum Force {
    /// All the queue (its address) holds, and why.
    Queue(usize, Why),
    /// What a queue holds up to the submission naming the list (its
    /// address), which the title resets to record into it again.
    List(usize),
}

/// Puts `what` behind what queue `this` holds (`line`, locked), listing
/// the queue in [`WAITING`] if it held nothing.
#[cfg(windows)]
fn push(this: This, line: &mut Line, what: Held) {
    if !line.listed {
        line.listed = true;
        line.since = Some(std::time::Instant::now());
        object::add_ref(this);
        lock(&WAITING).push(this as usize);
        watch();
    }
    line.held.push_back(what);
}

/// Sets `fence` to `value` once every submission made so far has finished,
/// and drops the reference kept to it.
#[cfg(windows)]
fn issue(host: &crate::host::Host, fence: usize, value: u64) {
    // Not locked across `after`, which runs it at once when nothing is in
    // flight.
    lock(&ISSUED).push((fence, value));
    host.after(move || {
        crate::fence::signal(fence as This, value);
        let mut issued = lock(&ISSUED);
        if let Some(i) = issued.iter().position(|&s| s == (fence, value)) {
            issued.swap_remove(i);
        }
        drop(issued);
        crate::fence::release_kept(fence as This);
    });
}

/// Runs title lists (their host sides) as one title submission, made on
/// `queue` (its address, 0 for none) of `kind`. The values their
/// bottom-of-pipe writes store count as written for the waits tested after
/// it ([`met`]) until it has finished: whether there are any.
#[cfg(windows)]
fn run(
    host: &crate::host::Host,
    queue: usize,
    kind: u32,
    lists: impl Iterator<Item = This>,
) -> bool {
    let host_lists: Vec<_> = lists.filter_map(crate::list::host_list).collect();
    let values: Vec<_> = host_lists.iter().flat_map(|l| l.value_writes()).collect();
    host.submit(&host_lists);
    if values.is_empty() {
        return false;
    }
    // Noted once submitted: a list tested meanwhile waits for them (and the
    // queues' thread looks again), and is never submitted before them.
    static RUNS: AtomicU64 = AtomicU64::new(0);
    let run = RUNS.fetch_add(1, Ordering::Relaxed);
    let writes = values.into_iter().map(|(address, value, bytes)| Write {
        address,
        value,
        bytes,
        queue,
        kind,
        run,
    });
    lock(&WRITES).extend(writes);
    // Once it has finished: its writes are in title memory then.
    host.after(move || lock(&WRITES).retain(|w| w.run != run));
    true
}

/// Runs what waiting queues hold as far as their waits are over, each
/// queue's in its own order, until nothing moves; with `force`, also what
/// it names. Called where a wait may have ended: after a queue's signal,
/// wait and submission, and from [`watch`]'s thread for what moves fences
/// without a queue (the title's CPU signals, bottom-of-pipe writes into a
/// fence once their submission has finished).
#[cfg(windows)]
fn release_held(force: Option<Force>) {
    /// One at a time: each queue's things are taken in order, and a listed
    /// queue stays alive meanwhile (only this takes queues off the list).
    static RELEASING: Mutex<()> = Mutex::new(());
    if lock(&WAITING).is_empty() {
        return;
    }
    let Some(host) = crate::host::made() else {
        return;
    };
    let _one = lock(&RELEASING);
    loop {
        let queues = lock(&WAITING).clone();
        let mut moved = false;
        for q in queues {
            moved |= run_line(host, q, force);
        }
        if !moved {
            break;
        }
    }
}

/// Runs what queue `q` holds as far as its waits are over, and what
/// `force` names: whether anything ran.
#[cfg(windows)]
fn run_line(host: &crate::host::Host, q: usize, force: Option<Force>) -> bool {
    // SAFETY: a listed queue, kept alive by the list's reference.
    let queue = unsafe { &*(q as *const Queue) };
    let mut line = lock(&queue.line);
    // How many of its things go whatever their waits.
    let (mut forced, why) = match force {
        Some(Force::Queue(at, why)) if at == q => (usize::MAX, why),
        Some(Force::List(list)) => {
            let names = |h: &Held| matches!(h, Held::Run(lists) if lists.contains(&list));
            let n = line.held.iter().position(names).map_or(0, |i| i + 1);
            (n, Why::Reset)
        }
        _ => (0, Why::Stuck),
    };
    let mut moved = false;
    // Lists that may run, gone as one submission before anything else
    // happens on the queue, and the values they write, in order.
    let mut batch: Vec<usize> = Vec::new();
    let mut batch_writes: Vec<(u64, u64, u8)> = Vec::new();
    let flush = |batch: &mut Vec<usize>, writes: &mut Vec<(u64, u64, u8)>| {
        if !batch.is_empty() {
            run(host, q, queue.kind, batch.iter().map(|&l| l as This));
            batch.drain(..).for_each(release_object);
        }
        writes.clear();
    };
    while let Some(mut next) = line.held.pop_front() {
        let forcing = forced > 0;
        forced = forced.saturating_sub(1);
        match &mut next {
            Held::Wait(fence, value) if !over(*fence, *value) => {
                if !forcing && !line.stuck() {
                    line.held.push_front(next);
                    break;
                }
                let why = if forcing { why } else { Why::Stuck };
                let n = COUNTS[FORCED].fetch_add(1, Ordering::Relaxed);
                if n < 16 {
                    tracing::warn!(
                        kind = queue.kind,
                        value = *value,
                        now = completed(*fence),
                        why = why.text(),
                        "a queue runs on past a wait for a fence that is not there yet"
                    );
                }
            }
            Held::Values {
                waits,
                since,
                waited,
            } => {
                let unmet: Vec<(ValueWait, Option<u64>)> = waits
                    .iter()
                    .filter_map(|w| match met(w, &batch_writes) {
                        Met::No(now) => Some((*w, now)),
                        _ => None,
                    })
                    .collect();
                if unmet.is_empty() {
                    if *waited {
                        note_released(q, queue.kind, waits, &batch_writes, since.elapsed());
                    }
                } else {
                    if !forcing && !line.stuck() {
                        *waited = true;
                        line.held.push_front(next);
                        break;
                    }
                    let why = if forcing { why } else { Why::Stuck };
                    note_forced(q, queue.kind, &unmet, since.elapsed(), why);
                }
            }
            _ => {}
        }
        moved = true;
        match next {
            Held::Wait(fence, _) => crate::fence::release_kept(fence as This),
            Held::Values { .. } => {}
            Held::Run(lists) => {
                for &l in &lists {
                    if let Some(host_list) = crate::list::host_list(l as This) {
                        batch_writes.extend(host_list.value_writes());
                    }
                }
                batch.extend(lists);
            }
            Held::Signal(fence, value) => {
                // After the lists before it.
                flush(&mut batch, &mut batch_writes);
                issue(host, fence, value);
            }
        }
    }
    flush(&mut batch, &mut batch_writes);
    if moved {
        line.since = Some(std::time::Instant::now());
    }
    let unlisted = line.held.is_empty() && std::mem::take(&mut line.listed);
    if unlisted {
        line.since = None;
        lock(&WAITING).retain(|&w| w != q);
    }
    drop(line);
    if unlisted {
        // The list's reference: the last one frees the queue.
        release(q as This);
    }
    moved
}

/// Whether the queue holds anything.
#[cfg(windows)]
fn holding(this: This) -> bool {
    !lock(&queue(this).line).held.is_empty()
}

/// Before the title resets `list` to record into it again: a queue that
/// holds it runs what it holds up to it first, waits over or not (its
/// commands are about to be replaced).
#[cfg(windows)]
pub(crate) fn list_reset(list: This) {
    if !lock(&WAITING).is_empty() {
        release_held(Some(Force::List(list as usize)));
    }
}

/// Wakes the thread [`watch`] starts.
#[cfg(windows)]
static WAKE: (Mutex<bool>, Condvar) = (Mutex::new(false), Condvar::new());

/// Starts, once, the thread that runs what queues hold when fences move
/// without a queue: it looks every millisecond while a queue holds
/// something, and sleeps otherwise. Wakes it now.
///
/// shortcut: it looks instead of being told. A hook in the fence's own
/// signal would release at once, for the CPU's signals.
#[cfg(windows)]
fn watch() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let spawned = std::thread::Builder::new()
            .name("xtp-gpu-queue-waits".into())
            .spawn(|| {
                crate::cores::keep_on_fast_cores("xtp-gpu-queue-waits");
                loop {
                    let holding = !lock(&WAITING).is_empty();
                    let mut woken = lock(&WAKE.0);
                    if !*woken {
                        woken = if holding {
                            let look = std::time::Duration::from_millis(1);
                            let waited = WAKE.1.wait_timeout(woken, look);
                            waited.unwrap_or_else(|e| e.into_inner()).0
                        } else {
                            WAKE.1.wait(woken).unwrap_or_else(|e| e.into_inner())
                        };
                    }
                    *woken = false;
                    drop(woken);
                    release_held(None);
                }
            });
        if let Err(e) = spawned {
            tracing::warn!(error = %e, "no thread for queue waits: held lists run at the queues' next calls");
        }
    });
    *lock(&WAKE.0) = true;
    WAKE.1.notify_one();
}

fn queue<'a>(this: This) -> &'a Queue {
    // SAFETY: called from queue methods only.
    unsafe { &*this.cast::<Queue>() }
}

pub(crate) extern "system" fn create(
    _: This,
    desc: *const QueueDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(?desc, "command queue");
    let queue = Box::new(Queue {
        object: if crate::device::newer() {
            Object::new(&NEWER_FUNCTIONS)
        } else {
            Object::new(&FUNCTIONS)
        },
        kind: desc.kind,
        desc,
        #[cfg(windows)]
        line: Mutex::default(),
    });
    object::hand_out(QUEUE, queue, iid, &QUEUE_IIDS, out)
}

/// The console's queue description: an engine or pipe and a queue on it
/// in place of PC's priority and node mask.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct QueueDescX {
    kind: u32,
    flags: u32,
    engine_or_pipe: u32,
    queue: u32,
}

/// Device method `CreateCommandQueueX`. The host picks engines itself, so
/// the queue is created as on PC.
pub(crate) extern "system" fn create_x(
    this: This,
    desc: *const QueueDescX,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&x) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(desc = ?x, "console command queue");
    let desc = QueueDesc {
        kind: x.kind,
        priority: 0,
        flags: x.flags,
        nodes: 1,
    };
    create(this, &desc, iid, out)
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        QUEUE,
        iid,
        object::is(iid, &QUEUE_IIDS).then_some(this),
        out,
    )
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Queue>(this)
}

/// Runs the lists' host lists (decision 0007), counted as one title
/// submission for the frame-rate line; behind what the queue holds, if it
/// holds anything ([`Held`]), and cut before each list that waits for
/// memory values ([`Held::Values`]) if one of them is not met yet.
extern "system" fn execute(this: This, count: u32, lists: *const This) {
    if lists.is_null() {
        return;
    }
    #[cfg(windows)]
    if let Some(host) = crate::host::host() {
        // SAFETY: the title passes `count` lists.
        let lists = unsafe { std::slice::from_raw_parts(lists, count as usize) };
        if this.is_null() {
            run(host, 0, u32::MAX, lists.iter().copied());
            return;
        }
        let (at, kind) = (this as usize, queue(this).kind);
        // Each list's memory values, as they stand now, after the lists
        // before it.
        let mut before: Vec<(u64, u64, u8)> = Vec::new();
        let mut waits: Vec<Vec<ValueWait>> = Vec::with_capacity(lists.len());
        let mut unmet = false;
        for &l in lists {
            let host_list = crate::list::host_list(l);
            let list_waits = host_list.map(|h| h.value_waits()).unwrap_or_default();
            for w in &list_waits {
                let met = met(w, &before);
                note_submitted(at, kind, w, met);
                unmet |= matches!(met, Met::No(_));
            }
            waits.push(list_waits);
            before.extend(host_list.into_iter().flat_map(|h| h.value_writes()));
        }
        let holding_values = holds_values();
        let mut line = lock(&queue(this).line);
        if line.held.is_empty() && !(unmet && holding_values) {
            // Under the lock: after what the queue ran before.
            let wrote = run(host, at, kind, lists.iter().copied());
            drop(line);
            // Another queue may wait for what it writes.
            if wrote {
                release_held(None);
            }
            return;
        }
        let mut part = Vec::new();
        for (&l, waits) in lists.iter().zip(waits) {
            if l.is_null() {
                continue;
            }
            if holding_values && !waits.is_empty() {
                if !part.is_empty() {
                    push(this, &mut line, Held::Run(std::mem::take(&mut part)));
                }
                let since = std::time::Instant::now();
                let waited = false;
                push(
                    this,
                    &mut line,
                    Held::Values {
                        waits,
                        since,
                        waited,
                    },
                );
            }
            object::add_ref(l);
            part.push(l as usize);
            COUNTS[HELD_LISTS].fetch_add(1, Ordering::Relaxed);
        }
        if !part.is_empty() {
            push(this, &mut line, Held::Run(part));
        }
        drop(line);
        // The waits may be over already.
        release_held(None);
        return;
    }
    let _ = this;
    static ONCE: AtomicBool = AtomicBool::new(false);
    if !ONCE.swap(true, Ordering::Relaxed) {
        tracing::warn!(count, "command lists submitted; no host GPU runs them");
    }
}

/// The start of a present plane: frame token, then the plane's resources.
#[repr(C)]
pub(crate) struct Plane {
    token: u64,
    count: u32,
    resources: *const This,
}
/// Submission tagged with the frame it belongs to `(count, lists, frame
/// token, flags)`; frames are paced elsewhere, so it runs as a plain one.
extern "system" fn execute_x(this: This, count: u32, lists: *const This, _token: u64, _flags: u32) {
    execute(this, count, lists);
}
/// Signal with sync settings for the top and bottom of the pipe: as a plain
/// signal, once earlier submissions have finished.
extern "system" fn signal2_x(
    this: This,
    fence: This,
    value: u64,
    _bottom: *const u8,
    _top: *const u8,
) -> HRESULT {
    signal(this, fence, value)
}
/// Markers and events only matter to profilers.
extern "system" fn marker(_: This) {}
/// Sets the fence once every submission so far has finished (written back,
/// bottom-of-pipe writes done): at once if none is in flight. A queue that
/// holds something signals after it.
pub(crate) extern "system" fn signal(this: This, fence: This, value: u64) -> HRESULT {
    if fence.is_null() {
        return E_INVALIDARG;
    }
    #[cfg(windows)]
    if let Some(host) = crate::host::made() {
        // Kept until it is set.
        object::add_ref(fence);
        let at = fence as usize;
        if this.is_null() {
            issue(host, at, value);
        } else {
            let mut line = lock(&queue(this).line);
            if line.held.is_empty() {
                // Under the lock: after what the queue ran before.
                issue(host, at, value);
            } else {
                push(this, &mut line, Held::Signal(at, value));
            }
        }
        // What waits for it may run now.
        release_held(None);
        return S_OK;
    }
    let _ = this;
    crate::fence::signal(fence, value)
}
/// From here on the queue runs nothing it is given until `fence` reaches
/// `value`, or a queue signal made before gets it there ([`over`]); what
/// it is given meanwhile is held ([`Held`]).
extern "system" fn wait(this: This, fence: This, value: u64) -> HRESULT {
    if fence.is_null() {
        return E_INVALIDARG;
    }
    COUNTS[WAITS].fetch_add(1, Ordering::Relaxed);
    #[cfg(windows)]
    if !this.is_null() && crate::host::made().is_some() && !over(fence as usize, value) {
        let n = COUNTS[HOLDS].fetch_add(1, Ordering::Relaxed);
        if n < 8 {
            tracing::info!(
                kind = queue(this).kind,
                now = completed(fence as usize),
                value,
                held = holds(),
                "a queue waits for a fence no signal made so far gets to: what it is given next waits"
            );
        }
        if holds() {
            object::add_ref(fence);
            let mut line = lock(&queue(this).line);
            push(this, &mut line, Held::Wait(fence as usize, value));
            drop(line);
            // The fence may have got there meanwhile.
            release_held(None);
        }
    }
    let _ = (this, value);
    S_OK
}
/// Ticks per second of GPU timestamps (see [`Written::GpuTimestamp`]).
extern "system" fn timestamp_frequency(_: This, out: *mut u64) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null.
    unsafe { out.write(TIMESTAMP_FREQUENCY) };
    S_OK
}
const TIMESTAMP_FREQUENCY: u64 = 100_000_000;

/// What a list's bottom-of-pipe write stores when the list has run.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Written {
    Value(u64),
    /// The GPU clock: [`TIMESTAMP_FREQUENCY`] ticks per second since the
    /// process started (our GPU work is done on the CPU's timeline).
    GpuTimestamp,
    /// The CPU's performance counter.
    SystemTimestamp,
}

impl Written {
    /// Stores the low `bytes` (4 or 8) of the value at `at`.
    pub(crate) fn write(self, at: *mut u8, bytes: usize) {
        let value = match self {
            Written::Value(v) => v,
            Written::GpuTimestamp => {
                static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
                let elapsed = START.get_or_init(std::time::Instant::now).elapsed();
                (elapsed.as_nanos() / (1_000_000_000 / u128::from(TIMESTAMP_FREQUENCY))) as u64
            }
            Written::SystemTimestamp => system_timestamp(),
        };
        // SAFETY: the title names writable GPU memory (unified with ours).
        unsafe {
            if bytes == 4 {
                at.cast::<u32>().write_unaligned(value as u32);
            } else {
                at.cast::<u64>().write_unaligned(value);
            }
        }
    }
}

fn system_timestamp() -> u64 {
    #[cfg(windows)]
    {
        let mut count = 0;
        // SAFETY: a valid out pointer.
        unsafe { windows_sys::Win32::System::Performance::QueryPerformanceCounter(&mut count) };
        count as u64
    }
    #[cfg(not(windows))]
    0
}

/// Frames presented so far.
static FRAMES: AtomicU64 = AtomicU64::new(0);

/// Frames the title presented so far.
pub fn presented() -> u64 {
    FRAMES.load(Ordering::Relaxed)
}

/// Shows the first plane's first resource in our window.
///
/// shortcut: one plane, no scaling or placement, the source expected in
/// the common state.
extern "system" fn present(
    this: This,
    planes: u32,
    plane_parameters: *const Plane,
    _parameters: *const u8,
) -> HRESULT {
    // The frame shows what the queue was given before the present: what it
    // still holds runs first, as far as its waits are over, then the rest.
    #[cfg(windows)]
    if !this.is_null() && holding(this) {
        release_held(None);
        release_held(Some(Force::Queue(this as usize, Why::Present)));
    }
    let _ = this;
    let frame = FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
    if frame.is_power_of_two() {
        let (draws, dispatches) = crate::list::take_counts();
        #[cfg(windows)]
        let (drawn, dispatched) = {
            static DRAWN: AtomicU64 = AtomicU64::new(0);
            static DISPATCHED: AtomicU64 = AtomicU64::new(0);
            let since = |last: &AtomicU64, now: u64| now - last.swap(now, Ordering::Relaxed);
            (
                since(&DRAWN, crate::draw::drawn()),
                since(&DISPATCHED, crate::draw::dispatched()),
            )
        };
        #[cfg(not(windows))]
        let (drawn, dispatched) = (0, 0);
        tracing::info!(
            frame,
            draws,
            drawn,
            dispatches,
            dispatched,
            "frame presented (since the last line: draws and dispatches, and those run on the host)"
        );
        let [waits, held, lists, forced] = COUNTS.each_ref().map(|n| n.load(Ordering::Relaxed));
        tracing::info!(
            waits,
            held,
            lists,
            forced,
            "queue waits for fences so far (held: no signal made yet got there, so what came next waited; lists: title lists held; forced: waits run past before they were over)"
        );
        let v = VALUES.each_ref().map(|n| n.load(Ordering::Relaxed));
        if v[V_WAITS] > 0 {
            tracing::info!(
                waits = v[V_WAITS],
                in_memory = v[V_MEMORY],
                written_here = v[V_HERE],
                written_elsewhere = v[V_ELSEWHERE],
                earlier_list = v[V_EARLIER],
                unmet = v[V_UNMET],
                unknown = v[V_UNKNOWN],
                unreadable = v[V_UNREADABLE],
                after_work = v[V_AFTER_WORK],
                after_write = v[V_AFTER_WRITE],
                released_by_memory = v[V_BY_MEMORY],
                released_by_here = v[V_BY_HERE],
                released_by_elsewhere = v[V_BY_ELSEWHERE],
                forced_reset = v[V_FORCED],
                forced_present = v[V_FORCED + 1],
                forced_mapping = v[V_FORCED + 2],
                forced_stuck = v[V_FORCED + 3],
                held_ms = v[V_HELD_US] / 1000,
                "list waits for memory values so far (at submission: met in title memory, by a write submitted before on the same queue or another, by an earlier list of the submission, not met, unknown comparison, unreadable; recorded after work, after the list's own bottom-of-pipe writes; held lists released by title memory, a write of their queue, another queue's; run before met by a reset of the list, a present, a page mapping, 4 s standing still; time held)"
            );
        }
        let [mappings, waited, past_writers] = REMAPS.each_ref().map(|n| n.load(Ordering::Relaxed));
        tracing::info!(
            mappings,
            waited,
            past_writers,
            "page-mapping copies so far (waited: for submissions in flight touching the pages; past writers: further than the newest one writing them, for one reading them)"
        );
        #[cfg(windows)]
        if frame >= 256 {
            let (hits, misses) = crate::shader_cache::counts();
            tracing::info!(hits, misses, "shaders from the cache and translated so far");
        }
    }
    #[cfg(all(windows, feature = "diagnostics"))]
    if crate::capture::listed(frame) {
        crate::capture::dump_targets(frame);
    }
    // XTP_PRESENT_TARGET: a target of the frame shown instead (`passdump`).
    #[cfg(all(windows, feature = "diagnostics"))]
    if crate::passdump::present(frame) {
        return S_OK;
    }
    // Several planes: composed into one image (`dxgi::planes`).
    #[cfg(windows)]
    let composed = crate::dxgi::planes::present(planes, plane_parameters.cast(), frame);
    #[cfg(all(windows, feature = "diagnostics"))]
    if let Some((resource, shown)) = composed {
        blinks(frame, resource, 0x4, shown);
    }
    #[cfg(windows)]
    if composed.is_none()
        && planes > 0
        // SAFETY: the title passes `planes` plane descriptions.
        && let Some(plane) = unsafe { plane_parameters.as_ref() }
        && plane.count > 0
        && !plane.resources.is_null()
        // SAFETY: the plane lists `count` resources.
        && let resource = unsafe { *plane.resources }
        && let Some(texture) = crate::resource::host_resource(resource)
        && let Some(host) = crate::host::host()
    {
        const COPY_SOURCE: u32 = 0x800;
        let state = crate::resource::host_transition(resource, COPY_SOURCE)
            .map_or(COPY_SOURCE, |(_, before)| before);
        let shown = host.present(texture, state, frame);
        crate::resource::host_transition(resource, state);
        #[cfg(feature = "diagnostics")]
        blinks(frame, resource, state, shown);
        #[cfg(not(feature = "diagnostics"))]
        let _ = shown;
    }
    #[cfg(not(windows))]
    let _ = (planes, plane_parameters);
    S_OK
}
/// Diagnostics: logs a presented frame that is black after one that was
/// not, with what was presented and how many draws ran since.
#[cfg(all(windows, feature = "diagnostics"))]
fn blinks(frame: u64, resource: This, state: u32, shown: Option<bool>) {
    static LAST: std::sync::Mutex<(bool, u64, u64)> = std::sync::Mutex::new((false, 0, 0));
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let drawn = crate::draw::drawn();
    let black = shown.unwrap_or(false);
    let draws = crate::draw::take_frame();
    if black && !last.0 {
        last.2 += 1;
        tracing::warn!(
            frame,
            resource = ?resource,
            state = format_args!("{state:#x}"),
            draws = drawn - last.1,
            copied = shown.is_some(),
            black_frames = last.2,
            levels = ?crate::capture::recent_levels(),
            "a black frame after a frame that was not"
        );
        for d in draws {
            tracing::warn!("black frame {frame}: {d}");
        }
        AFTER.store(frame + 6, Ordering::Relaxed);
    }
    // Six frames on: the levels around it (a fade or a single drop).
    static AFTER: AtomicU64 = AtomicU64::new(0);
    if AFTER.load(Ordering::Relaxed) == frame {
        tracing::warn!(frame, levels = ?crate::capture::recent_levels(), "levels after the black frame");
    }
    *last = (black, drawn, last.2);
}

/// A run of destination pages: how they are mapped (0 pool pages from
/// `start` on, 1 pool page `start` each, 2 unmapped, 3 left as they are)
/// and how many.
#[repr(C)]
pub(crate) struct PageRange {
    kind: u32,
    start: u32,
    count: u32,
}

/// The ranges mapped from a destination address on.
#[repr(C)]
pub(crate) struct PageBatch {
    destination: u64,
    count: u32,
    ranges: *const PageRange,
}

/// Bytes per mapped page.
const PAGE: u64 = 64 << 10;

/// Page-mapping copies so far: all, those that waited for submissions in
/// flight (that touch the pages), and of those, the ones that waited past
/// the newest submission writing the pages (for one that only reads them).
static REMAPS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
const REMAPPED: usize = 0;
const REMAP_WAITS: usize = 1;
const REMAP_READERS: usize = 2;

/// `CopyPageMappingsBatchX(count, batches, pool, pool pages, flags)`: maps
/// pages of a pool (memory the title mapped physical pages into) at the
/// batches' addresses. BTP 6 streams textures into pool pages
/// and maps them into its tiled heap this way (title+0xf2c470, ranges of
/// one constant page or of unmapped pages). Done once the earlier
/// submissions whose GPU work touches the pages it maps have finished,
/// what the queue holds included (`Host::settle_remap`): what they write
/// goes to the pages mapped when they were submitted, and the host copies
/// they read keep what those pages held. Submissions elsewhere run on.
///
/// shortcut: what the queue holds runs first, waits over or not, instead
/// of the mapping being held behind it.
extern "system" fn copy_page_mappings(
    this: This,
    count: u32,
    batches: *const PageBatch,
    pool: u64,
    pool_pages: u32,
    _flags: u32,
) -> HRESULT {
    let Some(map) = crate::resource::page_mapper() else {
        return E_INVALIDARG;
    };
    if batches.is_null() && count > 0 {
        return E_INVALIDARG;
    }
    let batches = (count > 0).then(|| {
        // SAFETY: the title passes `count` batches.
        unsafe { std::slice::from_raw_parts(batches, count as usize) }
    });
    // The mappings to make: `times` runs of `bytes` from `at` on, one after
    // the other, each of the pages at `source` (0 unmaps).
    let mut maps: Vec<(u64, u64, u64, u32)> = Vec::new();
    let mut ok = true;
    for b in batches.into_iter().flatten() {
        if b.ranges.is_null() {
            continue;
        }
        // SAFETY: each batch holds `count` ranges.
        let ranges = unsafe { std::slice::from_raw_parts(b.ranges, b.count as usize) };
        let mut at = b.destination;
        for r in ranges {
            let len = u64::from(r.count) * PAGE;
            let inside = |pages: u32| r.start.checked_add(pages).is_some_and(|e| e <= pool_pages);
            let source = pool + u64::from(r.start) * PAGE;
            match r.kind {
                0 if inside(r.count) => maps.push((at, source, len, 1)),
                1 if inside(1) => maps.push((at, source, PAGE, r.count)),
                2 => maps.push((at, 0, len, 1)),
                3 => {}
                _ => ok = false,
            }
            at += len;
        }
    }
    #[cfg(windows)]
    if !this.is_null() && holding(this) {
        release_held(None);
        release_held(Some(Force::Queue(this as usize, Why::Mapping)));
    }
    let _ = this;
    REMAPS[REMAPPED].fetch_add(1, Ordering::Relaxed);
    #[cfg(windows)]
    if let Some(host) = crate::host::made() {
        let pages = maps
            .iter()
            .map(|&(at, _, bytes, times)| (at as usize, (at + bytes * u64::from(times)) as usize));
        let (waited, readers) = host.settle_remap(pages.collect());
        if waited {
            REMAPS[REMAP_WAITS].fetch_add(1, Ordering::Relaxed);
        }
        if readers {
            REMAPS[REMAP_READERS].fetch_add(1, Ordering::Relaxed);
        }
    }
    for (at, source, bytes, times) in maps {
        ok &= (0..u64::from(times)).all(|k| map(at + k * bytes, source, bytes));
    }
    if !ok {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::warn!(pool = format_args!("{pool:#x}"), "a page mapping failed")
        });
    }
    S_OK
}

extern "system" fn desc(this: This, out: *mut QueueDesc) {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a description.
        unsafe { out.write(queue(this).desc) };
    }
}

/// The queue methods by their slot in older titles' queues (the 2020-era
/// layout): IUnknown 3, object 4, device child 1, queue 26 (12 as on PC,
/// then console ones).
const METHODS: [(usize, *const ()); 15] = [
    (0, query as *const ()),
    (2, release as *const ()),
    (7, object::parent_device as *const ()), // 0x38
    (10, execute as *const ()),              // 0x50
    (11, marker as *const ()),               // 0x58 set marker
    (12, marker as *const ()),               // 0x60 begin event
    (13, marker as *const ()),               // 0x68 end event
    (14, signal as *const ()),               // 0x70
    (15, wait as *const ()),                 // 0x78
    (16, timestamp_frequency as *const ()),  // 0x80
    (26, desc as *const ()),                 // 0xd0
    (29, copy_page_mappings as *const ()),   // 0xe8 page mapping batch
    (30, present as *const ()),              // 0xf0
    (33, execute_x as *const ()),            // 0x108
    (32, signal2_x as *const ()),            // 0x100
];

/// Newer titles' queues have two more methods before Signal, which moved
/// from 0x70 to 0x80: every slot from 0x70 on is two later (taken here as
/// starting exactly at 0x70, the methods before it kept in place).
/// Inferred from BTP 6's calls: Signal 0x80 `(fence,
/// value)`, timestamp frequency 0x90 (title+0x24841a6), suspend and resume
/// 0xb8/0xc0, PresentX 0x100, the frame-token submission 0x118 `(count,
/// lists, token, flags)` (title+0x24821dd).
const fn newer_slot(slot: usize) -> usize {
    if slot >= 14 { slot + 2 } else { slot }
}

const fn queue_table(newer: bool) -> [*const (); 36] {
    let mut f = object::d3d12_object(object::missing_table::<QUEUE, 36>());
    let mut i = 0;
    while i < METHODS.len() {
        let (slot, method) = METHODS[i];
        f[if newer { newer_slot(slot) } else { slot }] = method;
        i += 1;
    }
    f
}

static FUNCTIONS: Table<36> = Table(queue_table(false));
static NEWER_FUNCTIONS: Table<36> = Table(queue_table(true));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;

    /// Where a queue's signal is: newer layouts have it two slots later.
    /// Another test may switch layouts while this one runs, so the queue's
    /// own table decides.
    fn signal_slot(q: This) -> usize {
        // SAFETY: one of our queues; its first field is the table.
        let table = unsafe { *q.cast::<*const *const ()>() };
        if table == NEWER_FUNCTIONS.0.as_ptr() {
            0x80
        } else {
            0x70
        }
    }

    #[test]
    fn queue_signals_complete_at_once() {
        let null = std::ptr::null();
        let desc = QueueDesc {
            kind: 0,
            priority: 0,
            flags: 0,
            nodes: 0,
        };
        let mut q = null;
        assert_eq!(create(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        // SAFETY: the title reads the type at +0x10.
        assert_eq!(unsafe { q.cast::<u8>().add(0x10).cast::<u32>().read() }, 0);
        let mut fence = null;
        let fence_iid = &crate::fence::FENCE_IIDS[0];
        assert_eq!(
            crate::fence::create(null, 0, 0, fence_iid, &mut fence),
            S_OK
        );
        let signal = slot::<extern "system" fn(This, This, u64) -> HRESULT>(q, signal_slot(q));
        assert_eq!(signal(q, fence, 7), S_OK);
        // Other tests' submissions may be in flight: the signal is behind
        // them.
        #[cfg(windows)]
        if let Some(host) = crate::host::made() {
            host.drain();
        }
        assert_eq!(
            slot::<extern "system" fn(This) -> u64>(fence, 0x40)(fence),
            7
        );
        assert_eq!(slot::<extern "system" fn(This) -> u32>(q, 0x10)(q), 0);
    }

    /// BTP 6's compute queue: `{COMPUTE, 0, pipe 1, queue 0}`.
    #[test]
    fn console_queues_are_created_as_on_pc() {
        let null = std::ptr::null();
        let desc = QueueDescX {
            kind: 2,
            flags: 0,
            engine_or_pipe: 1,
            queue: 0,
        };
        let mut q = null;
        assert_eq!(create_x(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        assert_eq!(queue(q).desc.kind, 2);
        let none = std::ptr::null();
        assert_eq!(create_x(null, none, &QUEUE_IIDS[0], &mut q), E_INVALIDARG);
    }

    #[test]
    fn newer_queues_have_their_methods_two_slots_later() {
        let (old, new) = (queue_table(false), queue_table(true));
        assert_eq!(old[10], new[10]); // submission stays at 0x50
        assert_eq!(old[14], new[16]); // signal 0x70 → 0x80
        assert_eq!(old[30], new[32]); // present 0xf0 → 0x100
        assert_eq!(new[35], execute_x as *const ()); // 0x118
    }

    /// A 32-bit wait sees the low half of the word, masked; unknown
    /// comparisons say so.
    #[test]
    fn value_waits_compare_the_masked_value() {
        let w = |comparison, reference| ValueWait::new(8, 4, comparison, reference, 0xffff_ffff);
        let value = 0x1_0000_0005;
        let expect = [
            (1, false),
            (2, false),
            (3, true),
            (4, true),
            (5, false),
            (6, false),
            (7, true),
            (8, true),
        ];
        for (comparison, met) in expect {
            assert_eq!(
                w(comparison, 5).meets(value),
                Some(met),
                "comparison {comparison}"
            );
        }
        assert_eq!(w(2, 6).meets(value), Some(true));
        assert_eq!(w(5, 4).meets(value), Some(true));
        assert_eq!(w(0, 5).meets(value), None, "unknown comparison");
        assert_eq!(w(9, 5).meets(value), None);
        let masked = ValueWait::new(8, 4, 3, 0x4, 0xc);
        assert_eq!(masked.meets(0xf5), Some(true));
        let wide = ValueWait::new(8, 8, 7, 0x1_0000_0000, u64::MAX);
        assert_eq!(wide.meets(0x1_0000_0000), Some(true));
        assert_eq!(wide.meets(0xffff_ffff), Some(false));
    }

    /// What a bottom-of-pipe write stores where a wait reads, if it covers
    /// it.
    #[test]
    fn a_write_stores_what_a_wait_inside_it_reads() {
        let v = 0x1122_3344_5566_7788;
        assert_eq!(stored(0x100, v, 8, 0x100, 8), Some(v));
        assert_eq!(stored(0x100, v, 8, 0x100, 4), Some(0x5566_7788));
        assert_eq!(stored(0x100, v, 8, 0x104, 4), Some(0x1122_3344));
        assert_eq!(stored(0x100, v, 4, 0x100, 4), Some(0x5566_7788));
        assert_eq!(stored(0x100, v, 4, 0x100, 8), None, "wider than the write");
        assert_eq!(stored(0x100, v, 8, 0xfc, 4), None, "before it");
        assert_eq!(stored(0x100, v, 8, 0x106, 4), None, "past its end");
    }

    /// Queues that wait for fences, on the host GPU (`XTP_TEST_ADAPTER=warp`
    /// for the software adapter).
    #[cfg(windows)]
    mod waits {
        use super::*;
        use crate::testgpu::{Kernel, Pass, TitleBuffer, TitleList, device, pattern};
        use windows_sys::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
        use windows_sys::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

        /// Other tests submit to the same host and can release held lists or
        /// finish pending write-backs. Give this test its own host and queues.
        fn isolated_device(test: &str) -> Option<crate::testgpu::Gpu> {
            const CHILD: &str = "XTP_TEST_ISOLATED_QUEUE";
            if std::env::var(CHILD).as_deref() == Ok(test) {
                return device(test);
            }
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args([test, "--exact", "--nocapture"])
                .env(CHILD, test)
                .env_remove("XTP_SYNC_SUBMIT")
                .stderr(std::process::Stdio::inherit())
                .output()
                .unwrap();
            let stdout = String::from_utf8_lossy(&out.stdout);
            assert!(
                out.status.success() && stdout.contains("1 passed"),
                "{test}: {stdout}"
            );
            None
        }

        /// A title queue: 0 direct (graphics), 2 compute.
        fn new_queue(kind: u32) -> This {
            let null = std::ptr::null();
            let desc = QueueDesc {
                kind,
                priority: 0,
                flags: 0,
                nodes: 0,
            };
            let mut q = null;
            assert_eq!(create(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
            q
        }

        fn new_fence() -> This {
            let null = std::ptr::null();
            let mut fence = null;
            let iid = &crate::fence::FENCE_IIDS[0];
            assert_eq!(crate::fence::create(null, 0, 0, iid, &mut fence), S_OK);
            fence
        }

        /// Closes the list, as the title does before submitting it.
        fn close(list: &TitleList) -> This {
            let this = list.this();
            assert_eq!(
                slot::<extern "system" fn(This) -> HRESULT>(this, 0x48)(this),
                S_OK
            );
            this
        }

        /// Closes the list and submits it on `queue`, as the title does.
        fn submit(queue: This, list: &TitleList) {
            execute(queue, 1, &close(list));
        }

        /// A title list's method by its slot in the older layout: the
        /// list's table says which layout it has (where `wait64` is).
        fn list_method<F: Copy>(list: This, slot: usize) -> F {
            // SAFETY: one of our lists, whose first field is its table of
            // 160 slots.
            let newer = unsafe {
                let table = *list.cast::<*const *const ()>();
                *table.add(74) == wait64 as *const ()
            };
            method(list, if newer && slot >= 59 { slot + 2 } else { slot })
        }

        const GREATER_EQUAL: u32 = 7;

        /// Records `Wait64BitValueX(address, GREATER_EQUAL, value)` into
        /// the list, as BTP 6's queues do at the start of their lists.
        fn wait_for(list: &TitleList, address: u64, value: u64) {
            let this = list.this();
            let wait = list_method::<extern "system" fn(This, u64, u32, u64, u32)>(this, 72);
            wait(this, address, GREATER_EQUAL, value, 0);
        }

        /// Records `Write64BitValueBottomOfPipeX(address, value)` into the
        /// list: stored once the list has run.
        fn write_value(list: &TitleList, address: u64, value: u64) {
            let this = list.this();
            let write = list_method::<extern "system" fn(This, u64, u64, u32, u32)>(this, 70);
            write(this, address, value, 0, 0);
        }

        /// Waits up to 30 s for the queue to hold nothing.
        fn until_released(queue: This) {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
            while holding(queue) {
                assert!(
                    std::time::Instant::now() < deadline,
                    "still held after 30 s"
                );
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        }

        fn release_all(objects: &[This]) {
            for &o in objects {
                slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
            }
        }

        /// What a fill writes.
        fn filled() -> Vec<u32> {
            (0..64).map(pattern).collect()
        }

        const ALLOCATOR_IID: Iid = guid(
            0x6102dee4,
            0xaf59,
            0x4b09,
            [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
        );

        /// Resets the list with a new allocator, as the title does to
        /// record into it again: the allocator.
        fn reset(list: &TitleList) -> This {
            let null = std::ptr::null();
            let mut allocator = null;
            let made = crate::list::create_allocator(null, 0, &ALLOCATOR_IID, &mut allocator);
            assert_eq!(made, S_OK);
            let reset = slot::<extern "system" fn(This, This, This) -> HRESULT>(list.this(), 0x50);
            assert_eq!(reset(list.this(), allocator, null), S_OK);
            allocator
        }

        /// The title's compute queue waits for the depth its graphics queue
        /// draws: its list, submitted first, runs behind the graphics list
        /// the signal follows, and reads what that list wrote.
        #[test]
        fn a_compute_list_behind_a_fence_sees_what_the_graphics_list_signalling_it_wrote() {
            let test = "queue::tests::waits::a_compute_list_behind_a_fence_sees_what_the_graphics_list_signalling_it_wrote";
            let Some(gpu) = device(test) else { return };
            let host = gpu.host();
            let kernel = Kernel::new(host);
            let (scratch, depth, read) = (
                TitleBuffer::new(256),
                TitleBuffer::new(256),
                TitleBuffer::new(256),
            );
            let (graphics, compute, fence) = (new_queue(0), new_queue(2), new_fence());
            assert_eq!(wait(compute, fence, 1), S_OK);
            let reader = TitleList::new();
            kernel.dispatch(reader.host(), Pass::Copy, depth.address(), read.address());
            submit(compute, &reader);
            assert!(holding(compute), "held until the fence gets there");
            let drawer = TitleList::new();
            kernel.dispatch(
                drawer.host(),
                Pass::Fill,
                scratch.address(),
                depth.address(),
            );
            submit(graphics, &drawer);
            assert_eq!(signal(graphics, fence, 1), S_OK);
            assert!(!holding(compute), "run once the signal is made");
            host.drain();
            assert_eq!(read.words(), filled());
            release_all(&[graphics, compute, fence]);
        }

        /// A wait for a value the fence has, or that a queue signal made
        /// before gets it to, holds nothing.
        #[test]
        fn a_wait_for_a_value_already_reached_runs_the_next_list_at_once() {
            let test = "queue::tests::waits::a_wait_for_a_value_already_reached_runs_the_next_list_at_once";
            let Some(gpu) = device(test) else { return };
            let host = gpu.host();
            let kernel = Kernel::new(host);
            let (scratch, out) = (TitleBuffer::new(256), TitleBuffer::new(256));
            let (graphics, compute, fence) = (new_queue(0), new_queue(2), new_fence());
            assert_eq!(crate::fence::signal(fence, 3), S_OK);
            assert_eq!(wait(compute, fence, 2), S_OK);
            assert!(!holding(compute), "the fence is there");
            assert_eq!(signal(graphics, fence, 5), S_OK);
            assert_eq!(wait(compute, fence, 5), S_OK);
            assert!(!holding(compute), "a signal made before gets it there");
            let list = TitleList::new();
            kernel.dispatch(list.host(), Pass::Fill, scratch.address(), out.address());
            submit(compute, &list);
            assert!(!holding(compute), "run at once");
            host.drain();
            assert_eq!(out.words(), filled());
            release_all(&[graphics, compute, fence]);
        }

        /// A queue waiting for a fence the CPU signals runs on once it is
        /// signalled, with no queue call after it (the title then only waits
        /// for the queue's own signal), and reads what the CPU wrote before.
        #[test]
        fn a_queue_waiting_for_a_cpu_signal_runs_on_without_another_call() {
            let test = "queue::tests::waits::a_queue_waiting_for_a_cpu_signal_runs_on_without_another_call";
            let Some(gpu) = isolated_device(test) else {
                return;
            };
            let host = gpu.host();
            let kernel = Kernel::new(host);
            let (data, read) = (TitleBuffer::new(256), TitleBuffer::new(256));
            // A host copy, as if a draw had read the buffer before.
            let _ = data.shadow();
            let (compute, fence, done) = (new_queue(2), new_fence(), new_fence());
            let reader = TitleList::new();
            kernel.dispatch(reader.host(), Pass::Copy, data.address(), read.address());
            lock(&queue(compute).line).wait_indefinitely = true;
            assert_eq!(wait(compute, fence, 1), S_OK);
            submit(compute, &reader);
            assert_eq!(signal(compute, done, 1), S_OK);
            assert!(holding(compute));
            // SAFETY: plain auto-reset event.
            let event = unsafe { CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) };
            assert!(!event.is_null());
            let on_completion =
                slot::<extern "system" fn(This, u64, *mut std::ffi::c_void) -> HRESULT>(done, 0x48);
            assert_eq!(on_completion(done, 1, event), S_OK);
            data.set_words(pattern);
            assert_eq!(crate::fence::signal(fence, 1), S_OK);
            // SAFETY: a valid event.
            let set = unsafe { WaitForSingleObject(event, 30_000) };
            assert_eq!(set, WAIT_OBJECT_0, "the queue's signal is made");
            assert_eq!(read.words(), filled(), "what the CPU wrote first");
            // SAFETY: ours.
            unsafe { CloseHandle(event) };
            release_all(&[compute, fence, done]);
        }

        /// The title resets a list a queue holds to record into it again:
        /// the list runs first, as it was recorded.
        #[test]
        fn a_held_list_the_title_resets_runs_first() {
            let test = "queue::tests::waits::a_held_list_the_title_resets_runs_first";
            let Some(gpu) = device(test) else { return };
            let host = gpu.host();
            let kernel = Kernel::new(host);
            let (scratch, out) = (TitleBuffer::new(256), TitleBuffer::new(256));
            let (compute, fence) = (new_queue(2), new_fence());
            // Never signalled.
            assert_eq!(wait(compute, fence, 1), S_OK);
            let list = TitleList::new();
            kernel.dispatch(list.host(), Pass::Fill, scratch.address(), out.address());
            submit(compute, &list);
            assert!(holding(compute));
            let allocator = reset(&list);
            assert!(!holding(compute), "run before the reset");
            host.drain();
            assert_eq!(out.words(), filled());
            release_all(&[compute, fence, allocator]);
        }

        /// Slots 71 and 72 of a list (0x238 and 0x240, two later in newer
        /// lists) are the waits for memory values.
        #[test]
        fn lists_wait_for_memory_values_at_their_slots() {
            let list = TitleList::new();
            // SAFETY: one of our lists, whose first field is its table of
            // 160 slots.
            let table =
                unsafe { std::slice::from_raw_parts(*list.this().cast::<*const *const ()>(), 160) };
            let at = |slot: usize| (table[slot], table[slot + 1]);
            let waits = (wait32 as *const (), wait64 as *const ());
            assert!(
                (at(71) == waits) != (at(73) == waits),
                "at 0x238 in older lists, 0x248 in newer ones"
            );
        }

        /// The title's compute queue waits for a word its graphics queue
        /// writes at the end of the list that draws what it reads: its
        /// list, submitted first, runs after that list, and reads what it
        /// drew.
        #[test]
        fn a_list_waiting_for_a_value_another_queue_writes_runs_after_the_writer() {
            let test = "queue::tests::waits::a_list_waiting_for_a_value_another_queue_writes_runs_after_the_writer";
            let Some(gpu) = device(test) else { return };
            let host = gpu.host();
            let kernel = Kernel::new(host);
            let [scratch, depth, read, word] = [(); 4].map(|()| TitleBuffer::new(256));
            let (graphics, compute) = (new_queue(0), new_queue(2));
            let reader = TitleList::new();
            wait_for(&reader, word.address(), 1);
            kernel.dispatch(reader.host(), Pass::Copy, depth.address(), read.address());
            submit(compute, &reader);
            assert!(holding(compute), "held until the value is written");
            let drawer = TitleList::new();
            kernel.dispatch(
                drawer.host(),
                Pass::Fill,
                scratch.address(),
                depth.address(),
            );
            write_value(&drawer, word.address(), 1);
            submit(graphics, &drawer);
            assert!(!holding(compute), "run once the writer is submitted");
            host.drain();
            assert_eq!(read.words(), filled(), "what the writer drew");
            assert_eq!(word.words()[..2], [1, 0]);
            release_all(&[graphics, compute]);
        }

        /// A value title memory has, or that a list before it in the same
        /// submission writes, holds nothing.
        #[test]
        fn a_wait_for_a_value_already_there_holds_nothing() {
            let test = "queue::tests::waits::a_wait_for_a_value_already_there_holds_nothing";
            let Some(gpu) = device(test) else { return };
            let host = gpu.host();
            let kernel = Kernel::new(host);
            let [scratch, out, word] = [(); 3].map(|()| TitleBuffer::new(256));
            word.set_words(|i| if i == 0 { 5 } else { 0 });
            let compute = new_queue(2);
            let list = TitleList::new();
            wait_for(&list, word.address(), 5);
            kernel.dispatch(list.host(), Pass::Fill, scratch.address(), out.address());
            submit(compute, &list);
            assert!(!holding(compute), "title memory has it");
            let (first, second) = (TitleList::new(), TitleList::new());
            write_value(&first, word.address(), 7);
            wait_for(&second, word.address(), 7);
            execute(compute, 2, [close(&first), close(&second)].as_ptr());
            assert!(!holding(compute), "a list before it writes it");
            host.drain();
            assert_eq!(out.words(), filled());
            assert_eq!(word.words()[0], 7);
            release_all(&[compute]);
        }

        /// A value a submission still in flight writes holds nothing: the
        /// list goes after that submission on the host queue.
        #[test]
        fn a_wait_for_a_value_a_submission_in_flight_writes_holds_nothing() {
            let test = "queue::tests::waits::a_wait_for_a_value_a_submission_in_flight_writes_holds_nothing";
            let Some(gpu) = isolated_device(test) else {
                return;
            };
            let host = gpu.host();
            let kernel = Kernel::new(host);
            // The two lists share no buffer: the second one's submission
            // has nothing to wait for while the gate holds the first.
            let [scratch, depth, other, out, word] = [(); 5].map(|()| TitleBuffer::new(256));
            let (graphics, compute) = (new_queue(0), new_queue(2));
            let mut gate = crate::host::Gate::close(host);
            let drawer = TitleList::new();
            kernel.dispatch(
                drawer.host(),
                Pass::Fill,
                scratch.address(),
                depth.address(),
            );
            write_value(&drawer, word.address(), 1);
            submit(graphics, &drawer);
            let list = TitleList::new();
            wait_for(&list, word.address(), 1);
            kernel.dispatch(list.host(), Pass::Fill, other.address(), out.address());
            submit(compute, &list);
            assert!(!holding(compute), "the write in flight gets it there");
            assert_eq!(word.words()[0], 0, "not in title memory yet");
            gate.open();
            host.drain();
            assert_eq!(out.words(), filled());
            assert_eq!(word.words()[0], 1);
            release_all(&[graphics, compute]);
        }

        /// A list waiting for a value the CPU writes runs once it is
        /// written, with no queue call after it.
        #[test]
        fn a_list_waiting_for_a_value_the_cpu_writes_runs_once_it_is_written() {
            let test = "queue::tests::waits::a_list_waiting_for_a_value_the_cpu_writes_runs_once_it_is_written";
            let Some(gpu) = device(test) else { return };
            let host = gpu.host();
            let kernel = Kernel::new(host);
            let [scratch, out, word] = [(); 3].map(|()| TitleBuffer::new(256));
            let compute = new_queue(2);
            lock(&queue(compute).line).wait_indefinitely = true;
            let list = TitleList::new();
            wait_for(&list, word.address(), 3);
            kernel.dispatch(list.host(), Pass::Fill, scratch.address(), out.address());
            submit(compute, &list);
            assert!(holding(compute));
            word.set_words(|i| if i == 0 { 3 } else { 0 });
            until_released(compute);
            host.drain();
            assert_eq!(out.words(), filled());
            release_all(&[compute]);
        }

        /// Graphics: [G0 writes A], [G1 waits for B, then reads]; compute:
        /// [waits for A, draws, writes B]. Held whole, the graphics
        /// submission and the compute list would wait for each other: cut
        /// before G1, G0 runs at once, then the compute list, then G1.
        #[test]
        fn a_submission_is_cut_before_a_list_that_waits() {
            let test = "queue::tests::waits::a_submission_is_cut_before_a_list_that_waits";
            let Some(gpu) = device(test) else { return };
            let host = gpu.host();
            let kernel = Kernel::new(host);
            let [scratch, data, read, a, b] = [(); 5].map(|()| TitleBuffer::new(256));
            let (graphics, compute) = (new_queue(0), new_queue(2));
            // Exercise queue ordering without the stuck-queue fallback.
            lock(&queue(graphics).line).wait_indefinitely = true;
            lock(&queue(compute).line).wait_indefinitely = true;
            let (g0, g1) = (TitleList::new(), TitleList::new());
            write_value(&g0, a.address(), 1);
            wait_for(&g1, b.address(), 1);
            kernel.dispatch(g1.host(), Pass::Copy, data.address(), read.address());
            execute(graphics, 2, [close(&g0), close(&g1)].as_ptr());
            assert!(holding(graphics), "G1 waits");
            let c = TitleList::new();
            wait_for(&c, a.address(), 1);
            kernel.dispatch(c.host(), Pass::Fill, scratch.address(), data.address());
            write_value(&c, b.address(), 1);
            submit(compute, &c);
            assert!(!holding(compute), "G0 ran: its write gets it there");
            assert!(!holding(graphics), "the compute list's write gets G1 there");
            host.drain();
            assert_eq!(read.words(), filled(), "what the compute list drew");
            assert_eq!([a.words()[0], b.words()[0]], [1, 1]);
            release_all(&[graphics, compute]);
        }

        /// A value nothing writes: the console would wait for ever. The
        /// queue runs on once it has stood still too long, and says so.
        #[test]
        fn a_value_nothing_writes_is_reported_and_does_not_hold_the_queue_for_ever() {
            let test = "queue::tests::waits::a_value_nothing_writes_is_reported_and_does_not_hold_the_queue_for_ever";
            let Some(gpu) = isolated_device(test) else {
                return;
            };
            let host = gpu.host();
            let kernel = Kernel::new(host);
            let [scratch, out, word] = [(); 3].map(|()| TitleBuffer::new(256));
            let compute = new_queue(2);
            lock(&queue(compute).line).patience = Some(std::time::Duration::from_millis(50));
            let list = TitleList::new();
            wait_for(&list, word.address(), 1);
            kernel.dispatch(list.host(), Pass::Fill, scratch.address(), out.address());
            submit(compute, &list);
            assert!(holding(compute));
            until_released(compute);
            let stuck = &VALUES[V_FORCED + Why::Stuck as usize];
            assert_eq!(
                stuck.load(Ordering::Relaxed),
                1,
                "counted for the frame line"
            );
            host.drain();
            assert_eq!(out.words(), filled(), "it ran");
            assert_eq!(word.words()[0], 0);
            release_all(&[compute]);
        }

        /// The title resets a list held for a memory value: it runs first
        /// (its commands are about to be replaced), counted.
        #[test]
        fn a_list_held_for_a_value_the_title_resets_runs_first_and_is_counted() {
            let test = "queue::tests::waits::a_list_held_for_a_value_the_title_resets_runs_first_and_is_counted";
            let Some(gpu) = isolated_device(test) else {
                return;
            };
            let host = gpu.host();
            let kernel = Kernel::new(host);
            let [scratch, out, word] = [(); 3].map(|()| TitleBuffer::new(256));
            let compute = new_queue(2);
            lock(&queue(compute).line).wait_indefinitely = true;
            let list = TitleList::new();
            wait_for(&list, word.address(), 1);
            kernel.dispatch(list.host(), Pass::Fill, scratch.address(), out.address());
            submit(compute, &list);
            assert!(holding(compute));
            let allocator = reset(&list);
            assert!(!holding(compute), "run before the reset");
            let resets = &VALUES[V_FORCED + Why::Reset as usize];
            assert_eq!(resets.load(Ordering::Relaxed), 1);
            host.drain();
            assert_eq!(out.words(), filled());
            release_all(&[compute, allocator]);
        }

        /// A present shows what its queue was given before it: what the
        /// queue holds runs first.
        #[test]
        fn a_present_runs_what_its_queue_holds() {
            let test = "queue::tests::waits::a_present_runs_what_its_queue_holds";
            let Some(gpu) = device(test) else { return };
            let host = gpu.host();
            let kernel = Kernel::new(host);
            let (scratch, out) = (TitleBuffer::new(256), TitleBuffer::new(256));
            let (graphics, fence) = (new_queue(0), new_fence());
            // Only the present should release this unsignalled wait.
            lock(&queue(graphics).line).wait_indefinitely = true;
            assert_eq!(wait(graphics, fence, 1), S_OK);
            let list = TitleList::new();
            kernel.dispatch(list.host(), Pass::Fill, scratch.address(), out.address());
            submit(graphics, &list);
            assert!(holding(graphics));
            assert_eq!(
                present(graphics, 0, std::ptr::null(), std::ptr::null()),
                S_OK
            );
            assert!(!holding(graphics), "run before the present");
            host.drain();
            assert_eq!(out.words(), filled());
            release_all(&[graphics, fence]);
        }

        /// Where the test's page mapper mapped, and the first 64 words there
        /// just before (what the old pages held).
        static MAPPED: Mutex<Vec<(u64, Vec<u32>)>> = Mutex::new(Vec::new());

        /// The runtime's page mapper as the tests see it: notes the mapping,
        /// then shows the new pages at `to` (zeros for an unmapping, else
        /// the source's bytes), so that a write-back made after it would
        /// land on them.
        fn note_mapping(to: u64, from: u64, len: u64) -> bool {
            // SAFETY: the tests' own memory, `len` bytes at each address.
            unsafe {
                let old = (0..64).map(|i| (to as *const u32).add(i).read_volatile());
                lock(&MAPPED).push((to, old.collect()));
                if from == 0 {
                    std::ptr::write_bytes(to as *mut u8, 0, len as usize);
                } else {
                    std::ptr::copy(from as *const u8, to as *mut u8, len as usize);
                }
            }
            true
        }

        /// What the old pages at `at` held when a mapping replaced them.
        fn mapped(at: u64) -> Option<Vec<u32>> {
            let mapped = lock(&MAPPED);
            mapped.iter().find(|m| m.0 == at).map(|m| m.1.clone())
        }

        /// A page-mapping copy on `queue` that unmaps the page at `at`.
        fn unmap_page(queue: This, at: u64) -> HRESULT {
            let range = PageRange {
                kind: 2,
                start: 0,
                count: 1,
            };
            let batch = PageBatch {
                destination: at,
                count: 1,
                ranges: &range,
            };
            copy_page_mappings(queue, 1, &batch, 0, 0, 0)
        }

        /// A page mapping waits for the submission in flight whose shaders
        /// write the old pages, which get what they wrote (the new pages
        /// nothing); a mapping elsewhere does not wait for it.
        #[test]
        fn a_page_mapping_waits_only_for_the_submissions_touching_its_pages() {
            let test = "queue::tests::waits::a_page_mapping_waits_only_for_the_submissions_touching_its_pages";
            let Some(gpu) = isolated_device(test) else {
                return;
            };
            let host = gpu.host();
            host.finish_on_caller();
            crate::resource::use_page_mapper(note_mapping);
            let kernel = Kernel::new(host);
            // A page each: buffers are 64 KiB aligned.
            let [scratch, old, elsewhere] = [(); 3].map(|()| TitleBuffer::new(PAGE as usize));
            let queue = new_queue(0);
            let writer = TitleList::new();
            kernel.dispatch(writer.host(), Pass::Fill, scratch.address(), old.address());
            submit(queue, &writer);
            assert_eq!(unmap_page(queue, elsewhere.address()), S_OK);
            assert!(mapped(elsewhere.address()).is_some(), "mapped at once");
            assert!(writer.host().running(), "the writer not waited for");
            assert!(old.words().iter().all(|&w| w == 0), "nothing written back");
            assert_eq!(unmap_page(queue, old.address()), S_OK);
            assert_eq!(
                mapped(old.address()),
                Some(filled()),
                "the old pages got what it wrote"
            );
            assert!(old.words().iter().all(|&w| w == 0), "the new pages nothing");
            assert!(!writer.host().running());
            release_all(&[queue]);
        }
    }

    #[test]
    fn timestamps_and_values_are_written() {
        let mut at = [0u64; 2];
        Written::Value(0x1_0000_0002).write(at.as_mut_ptr().cast(), 4);
        assert_eq!(at[0], 2);
        Written::GpuTimestamp.write(at[1..].as_mut_ptr().cast(), 8);
        let first = at[1];
        std::thread::sleep(std::time::Duration::from_millis(2));
        Written::GpuTimestamp.write(at[1..].as_mut_ptr().cast(), 8);
        assert!(at[1] > first);
    }
}
