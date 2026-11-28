//! Which title memory the title wrote since a host copy took it (decision
//! 0007, "Copies of title memory"). The runtime reserves title GPU memory
//! with write watching (`xtp-hle` xmem), so the kernel keeps a dirty bit
//! per 4 KiB page; asking for a range's written pages also clears them. Host
//! copies of the same memory can overlap (nested buffers, textures in
//! buffers), so what one asks is kept here for the others: a journal of
//! each page's last write, numbered by the asking.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. The runtime tells us when title memory changes
//! ([`title_memory_remapped`]): after it maps a view, before and after it
//! unmaps one, and after the title reserves, commits or frees memory
//! (`XMemVirtualAlloc`, the memory API set's `VirtualAlloc` and
//! `VirtualFree`). That resets the blocks' protection and all we noted of
//! them. Blocks found unmapped are not asked about again until a report
//! there: view memory comes from placeholders, which only the runtime's
//! mapping fills. Other blocks that cannot be made read-only (free address
//! space, a reservation not committed yet) are reported written at each
//! asking, and asked about again only after a report there too: anyone may
//! allocate free address space, so memory that appears there unreported is
//! copied each time, never missed.
//!
//! The title shows some physical pages in more than one view: BTP 6 writes
//! them through a pool and the GPU reads them through its tiled heap
//! (xmem's `map_alias`). A write through any view of a page is a write to
//! all of them, so the runtime tells us where each page shows
//! ([`set_alias_lookup`]). Guarding a block guards the other views of its
//! page too, a write to any of them opens them all, and a report of a
//! change to one opens the others: a view mapped after a copy guarded
//! another was not guarded with it. A block is one console page (64 KiB),
//! so a block's other views are those of one page. Before, a guard on one
//! view did not see writes through another, and its copy kept what it first
//! took (VIctory #1122: stale mesh data, then page faults in the world).
//!
//! shortcut: the title's memory calls that bypass xmem are not reported:
//! its own views of sections (`MapViewOfFile` and its kin) and the memory
//! API set's other calls (`VirtualAllocEx`, `VirtualAlloc2`), which go to
//! the host. Where a range reaches into such memory from a view, a
//! reservation the title placed itself (not write-watched, so it looks like
//! a placeholder) that such a call commits is not seen, and what we noted
//! of memory they unmap or free stays until a report there. Report them
//! from xmem if a title makes them on GPU memory.
//!
//! The runtime writes physical pages through its own view of them too
//! (xmem's `write_physical` and page windows: DirectStorage's page
//! destinations, which carry texture and buffer data, docs/analysis/
//! formats.md §6.1). That view is not in its index and no copy guards it,
//! so such a write does not fault: after it the runtime looks up the
//! title's views of the pages written and opens their guards ([`writing`]),
//! before the request's status says it completed, which is when the title
//! may read them. While nothing is guarded ([`guarding`]) it does not look.
//!
//! The kernel's writes into title memory for the runtime (file reads,
//! DirectStorage's raw reads) do not fault into our handler either: a
//! system call that meets a read-only block fails as a whole
//! (`ERROR_NOACCESS`, or `ERROR_INVALID_USER_BUFFER` when the block turns
//! read-only while it copies). The runtime announces them
//! ([`kernel_writing`]), which opens their blocks and holds them open until
//! the write is done: a copy meanwhile reports them written without guarding
//! them again, nor guards another view of their pages. Before, the opening
//! was all, and a copy guarding the blocks again cut reads in the middle
//! (session mesh64-a's failed DirectStorage reads, then the title's
//! assertion; docs/analysis/archive-null.md).
//!
//! Costs (the review of the VIctory #464 and #490 ports; gpu-measure.md
//! §4, run1's world frames): the 4.6 to 9.9 GB a world frame checks is not
//! reduced yet. `guard` walks every block of an asking with one to four set
//! lookups each (one for a clean block), and `writes` and the copies'
//! `copy_stale` walk every page. The [`writing`] fast path never fires in
//! the world, where some view is always guarded. Cutting that volume is
//! G16 to G18's. Other views (the VIctory #1122 port): `protect` asks the
//! runtime's index once per run it guards, and a write only for the blocks
//! a guard linked to other views (`Guards::aliased`), so a write where no
//! page shows twice costs what it did. Each other view guarded adds a
//! protection change to the guard and to the write that opens it. A runtime
//! write into physical pages while something is guarded costs a look in
//! the index per page written and a [`writing`] per run of the addresses
//! showing them; a protection change only where one is guarded.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::sync::{Condvar, Mutex, OnceLock};

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved.
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;

// Windows x64 user addresses have 31 page-number bits. A 4 GiB region
// allocates a directory on its first write, then 2 KiB per 16 MiB touched.
// Readers never allocate. Only the holder of JOURNAL publishes a stamp.
type Stamps = [std::sync::atomic::AtomicU64; 256];
type StampRegion = [OnceLock<Box<Stamps>>; 256];
static STAMPS: [OnceLock<Box<StampRegion>>; 1 << 15] = [const { OnceLock::new() }; 1 << 15];

fn stamp(page: usize, now: u64) {
    let Some(region) = STAMPS.get(page >> 16) else {
        return;
    };
    let region = region.get_or_init(|| Box::new(std::array::from_fn(|_| OnceLock::new())));
    let stamps =
        region[(page >> 8) & 255].get_or_init(|| Box::new(std::array::from_fn(|_| 0.into())));
    stamps[page & 255].store(now, std::sync::atomic::Ordering::Release);
}

/// A concurrent poll or kernel write may already have advanced the stamp.
/// Copies record only their own poll number, so the later write stays due.
pub(crate) fn last_write(page: usize) -> u64 {
    let Some(region) = STAMPS.get(page >> 16) else {
        return u64::MAX;
    };
    region
        .get()
        .and_then(|region| region[(page >> 8) & 255].get())
        .map_or(0, |stamps| {
            stamps[page & 255].load(std::sync::atomic::Ordering::Acquire)
        })
}

pub(crate) fn changed_since(start: usize, end: usize, since: u64) -> bool {
    (start / PAGE..end.div_ceil(PAGE)).any(|page| last_write(page) > since)
}

struct Journal {
    /// Numbers the askings; 0 is "never".
    asked: u64,
    /// Room for the kernel's answer.
    found: Vec<usize>,
}

static JOURNAL: Mutex<Journal> = Mutex::new(Journal {
    asked: 0,
    found: Vec::new(),
});

impl Journal {
    /// Interroge Windows sur des pages entières et garde sa réponse dans `found`.
    fn poll(&mut self, pages: Range<usize>) -> Option<usize> {
        use windows_sys::Win32::System::Memory::GetWriteWatch;
        const WRITE_WATCH_FLAG_RESET: u32 = 1;
        let size = pages.len() * PAGE;
        let mut count = size / KERNEL_PAGE;
        self.found.resize(count.max(self.found.len()), 0);
        let mut granularity = 0;
        #[cfg(test)]
        tests::WATCH_QUERIES.set(tests::WATCH_QUERIES.get() + 1);
        // SAFETY: the answer buffer holds `count` entries.
        let status = unsafe {
            GetWriteWatch(
                WRITE_WATCH_FLAG_RESET,
                (pages.start * PAGE) as *const _,
                size,
                self.found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        };
        (status == 0).then_some(count)
    }
}

/// Poll the kernel or guards for this range, publish changed page stamps,
/// and return the poll number. Readers can then consult individual pages.
pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<u64> {
    with_writes(start, end, guarded, |now| now)
}

/// Polls write-watching and section guards across `ranges` (each a half-open
/// `PAGE`-indexed range `(first_page, last_page)`) under a single `JOURNAL`
/// lock after sorting and coalescing overlapping or adjacent ranges.
///
/// Returns `(now, unwatched)` where every dirty page across the polled spans
/// has been stamped with `now` (`last_write(page) == now`) and `unwatched`
/// holds sorted, coalesced page ranges that neither `GetWriteWatch` nor
/// `guard` could answer for (caller should fall back to `Shadow::update`).
pub(crate) fn poll_merged(ranges: &[(usize, usize)]) -> (u64, Vec<(usize, usize)>) {
    let mut todo: Vec<(usize, usize)> = ranges.iter().copied().filter(|&(a, b)| a < b).collect();
    if todo.is_empty() {
        return (0, Vec::new());
    }
    let _watch = crate::host::preparation::Phase::Watch.enter();
    todo.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(todo.len());
    for (a, b) in todo {
        if let Some(last) = merged.last_mut()
            && a <= last.1
        {
            last.1 = last.1.max(b);
        } else {
            merged.push((a, b));
        }
    }

    let (now, unwatched) = {
        let mut guard_lock = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
        let j = &mut *guard_lock;
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let ask_guard = |first: usize, last: usize| -> Option<Vec<usize>> {
            #[cfg(feature = "diagnostics")]
            let guarding = std::time::Instant::now();
            let written = guard(first * PAGE, last * PAGE);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(
                crate::copystat::C::GuardUs,
                guarding.elapsed().as_micros() as u64,
            );
            written
        };

        let mut stamps = Vec::new();
        let mut unwatched: Vec<(usize, usize)> = Vec::new();
        let mut stack: Vec<(usize, usize)> = merged.into_iter().rev().collect();
        while let Some((a, b)) = stack.pop() {
            if KNOWN_VIEWS.contains(a) {
                let mut k = a + 1;
                while k < b && KNOWN_VIEWS.contains(k) {
                    k += 1;
                }
                if k < b {
                    stack.push((k, b));
                }
                if let Some(dirty) = ask_guard(a, k) {
                    stamps.extend(dirty);
                } else {
                    unwatched.push((a, k));
                }
                continue;
            }
            if let Some(count) = j.poll(a..b) {
                stamps.extend(j.found[..count].iter().map(|&at| at / PAGE));
            } else if b - a > 1 {
                let mid = a + (b - a) / 2;
                stack.push((mid, b));
                stack.push((a, mid));
            } else if let Some(dirty) = ask_guard(a, b) {
                stamps.extend(dirty);
            } else {
                unwatched.push((a, b));
            }
        }

        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::Watches, 1);
            add(C::WatchUs, started.elapsed().as_micros() as u64);
        }
        j.asked += 1;
        let now = j.asked;
        for page in stamps {
            stamp(page, now);
        }
        (now, unwatched)
    };
    let mut unwatched = unwatched;

    unwatched.sort_unstable();
    let mut merged_unwatched: Vec<(usize, usize)> = Vec::with_capacity(unwatched.len());
    for (a, b) in unwatched {
        if let Some(last) = merged_unwatched.last_mut()
            && a <= last.1
        {
            last.1 = last.1.max(b);
        } else {
            merged_unwatched.push((a, b));
        }
    }
    (now, merged_unwatched)
}

fn with_writes<T>(
    start: usize,
    end: usize,
    guarded: bool,
    result: impl FnOnce(u64) -> T,
) -> Option<T> {
    let _watch = crate::host::preparation::Phase::Watch.enter();
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    // Une vue de section ne peut pas avoir MEM_WRITE_WATCH.
    let known_view = KNOWN_VIEWS.contains(first);
    if known_view && !guarded {
        return None;
    }
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    let ask_guard = || {
        #[cfg(feature = "diagnostics")]
        let guarding = std::time::Instant::now();
        let written = guard(first * PAGE, last * PAGE);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(
            crate::copystat::C::GuardUs,
            guarding.elapsed().as_micros() as u64,
        );
        written
    };
    // Si un remappage efface le bit avant que guard prenne GUARDS, guard
    // vérifie la nouvelle allocation et on revient à Windows si elle est privée.
    let (count, dirty) = if let Some(dirty) = known_view.then(ask_guard).flatten() {
        (0, dirty)
    } else if let Some(count) = j.poll(first..last) {
        (count, Vec::new())
    } else if guarded {
        (0, ask_guard()?)
    } else {
        return None;
    };
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
    }
    j.asked += 1;
    let now = j.asked;
    let pages = j.found[..count].iter().map(|&at| at / PAGE).chain(dirty);
    for page in pages {
        stamp(page, now);
    }
    Some(result(now))
}

/// Guarded blocks of view memory: one page each, so a block shows one
/// physical page (xmem's pages are 64 KiB too).
const BLOCK: usize = PAGE;

struct Guards {
    owners: crate::readmap::Owners,
    /// Blocks we made read-only.
    protected: BTreeSet<usize>,
    /// Of those, the ones not written since a copy took them.
    clean: BTreeSet<usize>,
    /// Blocks removed from `protected` while Windows restores write access.
    opening: BTreeSet<usize>,
    /// Blocks found unmapped ([`placeholder`]) when a copy took them: the
    /// copy holds their zeros until the runtime maps them
    /// ([`title_memory_remapped`]). Without this, every asking found them
    /// "written" again (they cannot be made read-only) and the copy read
    /// them page by page through the kernel's failing reads, hundreds of
    /// MB per world frame.
    absent: BTreeSet<usize>,
    /// Blocks that could not be made read-only and are no placeholders
    /// (free address space, a reservation or a view not committed yet) when
    /// a copy took them, until the runtime reports a change there: reported
    /// written at every asking, without asking the system again (a failed
    /// protection change, a `VirtualQuery` and a `GetWriteWatch` each, since
    /// 4a873fb). Memory that appears there unreported stays reported
    /// written: copied each time, never missed.
    unguardable: BTreeSet<usize>,
    /// Blocks whose page shows at other blocks too (other views of it) and
    /// that a guard made read-only together, each end: a write to one opens
    /// the others ([`open`]). Only these are looked up in the runtime's
    /// index on a write, which a kernel write over a big range would
    /// otherwise pay for every block. Opening a page's views drops them all
    /// from here: none of them is clean any more, so none needs the others.
    aliased: BTreeSet<usize>,
}

static GUARDS: Mutex<Guards> = Mutex::new(Guards {
    owners: crate::readmap::Owners::new(crate::readmap::MAX_PAGES),
    protected: BTreeSet::new(),
    opening: BTreeSet::new(),
    clean: BTreeSet::new(),
    absent: BTreeSet::new(),
    unguardable: BTreeSet::new(),
    aliased: BTreeSet::new(),
});

static REOPENED: Condvar = Condvar::new();

/// How many blocks are protected, being guarded or being reopened, set under the guards' lock,
/// so that [`writing`] (after the runtime's own writes into physical pages,
/// and before the GPU side's) sees without the lock that nothing is
/// guarded. Raised before blocks are made read-only and lowered after
/// write access is back: when it reads 0, no block is read-only. A kernel
/// write under way is another matter: a guard may be about to make its
/// blocks read-only, so [`kernel_writing`] always takes the lock.
static GUARDED: AtomicUsize = AtomicUsize::new(0);

// Le gestionnaire peut interrompre le thread qui modifie les gardes.
thread_local! { static HOLDING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }

struct GuardLock(std::sync::MutexGuard<'static, Guards>);

impl GuardLock {
    fn new(lock: std::sync::MutexGuard<'static, Guards>) -> Self {
        HOLDING.set(true);
        Self(lock)
    }
}
impl std::ops::Deref for GuardLock {
    type Target = Guards;
    fn deref(&self) -> &Guards {
        &self.0
    }
}
impl std::ops::DerefMut for GuardLock {
    fn deref_mut(&mut self) -> &mut Guards {
        &mut self.0
    }
}
impl Drop for GuardLock {
    fn drop(&mut self) {
        HOLDING.set(false);
    }
}

#[cfg(feature = "diagnostics")]
static GUARD_WAIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cfg(feature = "diagnostics")]
pub(crate) fn take_guard_wait() -> std::time::Duration {
    std::time::Duration::from_nanos(GUARD_WAIT_NS.swap(0, std::sync::atomic::Ordering::Relaxed))
}

#[cfg(feature = "diagnostics")]
fn record_guard_wait(started: std::time::Instant) {
    if !crate::host::preparation::active() {
        GUARD_WAIT_NS.fetch_add(
            started.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
}

fn lock_guards() -> std::sync::MutexGuard<'static, Guards> {
    #[cfg(feature = "diagnostics")]
    let started = match GUARDS.try_lock() {
        Ok(g) => return g,
        Err(std::sync::TryLockError::Poisoned(e)) => return e.into_inner(),
        Err(std::sync::TryLockError::WouldBlock) => std::time::Instant::now(),
    };
    let g = GUARDS.lock().unwrap_or_else(|e| e.into_inner());
    #[cfg(feature = "diagnostics")]
    record_guard_wait(started);
    g
}

fn guards() -> GuardLock {
    GuardLock::new(lock_guards())
}

/// Un appel noyau ou un remappage doit trouver la protection effective.
fn settled(range: Range<usize>) -> GuardLock {
    let lock = lock_guards();
    GuardLock::new(
        REOPENED
            .wait_while(lock, |g| g.opening.range(range.clone()).next().is_some())
            .unwrap_or_else(|e| e.into_inner()),
    )
}

// Sur Windows x64, les adresses utilisateur tiennent sur 47 bits. Une région
// de 4 Gio ne coûte 8 Kio que si une vue y a effectivement été protégée.
// GUARDS sérialise les changements. Le gestionnaire consulte les bits en cas
// de contention pour distinguer un défaut attendu d’une panne indépendante.
type RegionBits = OnceLock<Box<[std::sync::atomic::AtomicU64; 1024]>>;
struct FaultPages([RegionBits; 1 << 15]);

impl FaultPages {
    const fn new() -> Self {
        Self([const { OnceLock::new() }; 1 << 15])
    }

    fn contains(&self, page: usize) -> bool {
        self.0
            .get(page >> 16)
            .and_then(OnceLock::get)
            .is_some_and(|bits| bits[(page >> 6) & 1023].load(SeqCst) & (1 << (page % 64)) != 0)
    }

    fn set(&self, mut pages: Range<usize>, on: bool) {
        while pages.start < pages.end {
            let page = pages.start;
            let Some(region) = self.0.get(page >> 16) else {
                break;
            };
            if !on && region.get().is_none() {
                pages.start = ((page >> 16) + 1) << 16;
                continue;
            }
            let count = pages.len().min(64 - page % 64);
            let mask = (u64::MAX >> (64 - count)) << (page % 64);
            let word = (page >> 6) & 1023;
            if on {
                let bits = region.get_or_init(|| Box::new(std::array::from_fn(|_| 0.into())));
                bits[word].fetch_or(mask, SeqCst);
            } else if let Some(bits) = region.get() {
                bits[word].fetch_and(!mask, SeqCst);
            }
            pages.start += count;
        }
    }
}

static WRITE_FAULTS: FaultPages = FaultPages::new();
static READ_FAULTS: FaultPages = FaultPages::new();
static KNOWN_READS: FaultPages = FaultPages::new();
/// Starting blocks identified as section views by `guard`, until a remap.
/// Calling VirtualQuery at every asking cost 100–230 ms per world frame.
/// The cache was ported in 6b8be7e from Marvisak/VIctory#490 by wbkepler.
static KNOWN_VIEWS: FaultPages = FaultPages::new();

/// The runtime's answer to where else the title shows the physical pages of
/// a range of title memory ([`set_alias_lookup`]).
pub type AliasLookup = fn(Range<usize>, &mut Vec<(usize, usize)>);

static ALIAS_LOOKUP: OnceLock<AliasLookup> = OnceLock::new();

/// The runtime tells where else the physical pages of view memory show
/// (xmem's index of the title's views): `lookup` adds, for each 64 KiB page
/// of a range of title memory that the title shows at other addresses too,
/// `(its address, the other's)`, once per other address. A write through
/// any view of a page is a write to all of them: guarding a block guards
/// the others, and a write to one opens them all. `lookup` runs with the
/// guards' lock held, in the handler of guarded writes too: it must not
/// call anything, nor wait for a lock the runtime holds when it reports a
/// change to us ([`title_memory_remapped`]). (Ported from
/// Marvisak/VIctory#1122 (head 9ed6093) by tightfall.)
pub fn set_alias_lookup(lookup: AliasLookup) {
    let _ = ALIAS_LOOKUP.set(lookup);
}

/// The blocks of `blocks` whose page shows at other blocks too, each with
/// each of those: `(block, other)`.
fn aliases(blocks: Range<usize>, out: &mut Vec<(usize, usize)>) {
    out.clear();
    let Some(lookup) = ALIAS_LOOKUP.get() else {
        return;
    };
    // Up to the end of the address space at most (`blocks`' end may lie
    // past it, see `blocks`).
    lookup(
        blocks.start.saturating_mul(BLOCK)..blocks.end.saturating_mul(BLOCK),
        out,
    );
    for pair in out.iter_mut() {
        *pair = (pair.0 / BLOCK, pair.1 / BLOCK);
    }
    out.retain(|&(block, other)| block != other);
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (through any view of them), guarding them again
/// before the caller copies them. `None` if the memory is not a view.
fn guard(start: usize, end: usize) -> Option<Vec<usize>> {
    install();
    let mut g = guards();
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    // Asked under the lock, which `title_memory_remapped` takes. The runtime
    // tells us before and after it unmaps a view, so a view noted between
    // the two may be gone, its blocks guarded again: the report after
    // forgets it all. An asking in between that finds the view gone notes
    // its blocks absent (the report before opened them, 5553e29), which the
    // report after forgets too.
    if !KNOWN_VIEWS.contains(first) {
        if !view(start) {
            return None;
        }
        KNOWN_VIEWS.set(first..first + 1, true);
    }
    let mut written = Vec::new();
    let mut run: Option<usize> = None;
    // Blocks the kernel is writing for the runtime: left open, reported
    // written, not clean (see `kernel_writing`).
    let flying = flying(first..last);
    for block in first..=last {
        #[cfg(feature = "diagnostics")]
        if block < last && g.absent.contains(&block) {
            crate::copystat::add(crate::copystat::C::Absent, 1);
        }
        let landing = block < last && (is_flying(&flying, block) || g.opening.contains(&block));
        let open =
            block < last && (landing || !g.clean.contains(&block)) && !g.absent.contains(&block);
        if open {
            written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
            if landing {
                // Its write opens it if a guard got there first.
            } else if g.protected.contains(&block) {
                // Written through the runtime, still read-only.
                g.clean.insert(block);
            } else if !g.unguardable.contains(&block) {
                run.get_or_insert(block);
                continue;
            }
        }
        if let Some(from) = run.take() {
            protect(&mut g, from, block);
        }
    }
    Some(written)
}

/// What [`alike`] found for the blocks `[first, last)` starting at `first`.
enum Alike {
    /// `span` contiguous blocks (`>= 1`) starting at `first` lie in one
    /// committed `PAGE_READONLY | PAGE_READWRITE` region.
    Plain(usize),
    /// Block `first` starts committed, so it either has another protection
    /// (such as `PAGE_GUARD`) or is split across protections/states.
    Mixed,
    /// Block `first` does not start in committed memory.
    Uncommitted,
}

/// How many of `[first, last)`, starting from `first`, lie in one committed
/// region of `PAGE_READONLY` or `PAGE_READWRITE`. A `VirtualProtect` call
/// resets the protection of every page it covers, so a block containing a
/// page the title set `PAGE_GUARD` on must be left alone.
// Ported from PastiiJay/klbf-lab commit 409d69cc (GPL-3.0-or-later).
fn alike(first: usize, last: usize) -> Alike {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READONLY, PAGE_READWRITE, VirtualQuery,
    };
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::ViewQueries, 1);
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery((first * BLOCK) as *const _, &mut info, size_of_val(&info)) };
    if n == 0 || info.State != MEM_COMMIT {
        return Alike::Uncommitted;
    }
    if !matches!(info.Protect, PAGE_READONLY | PAGE_READWRITE) {
        return Alike::Mixed;
    }
    let end = (info.BaseAddress as usize).saturating_add(info.RegionSize) / BLOCK;
    let span = end.min(last).saturating_sub(first);
    if span == 0 {
        Alike::Mixed
    } else {
        Alike::Plain(span)
    }
}

/// Makes contiguous `range` (already verified [`Alike::Plain`]) read-only,
/// falling back block by block if a whole-range call still fails.
fn protect_span(
    g: &mut Guards,
    range: Range<usize>,
    mut on_protected: impl FnMut(&mut Guards, Range<usize>),
    mut on_failed: impl FnMut(&mut Guards, usize),
) {
    use windows_sys::Win32::System::Memory::PAGE_READONLY;
    WRITE_FAULTS.set(range.clone(), true);
    if set_protection(range.clone(), PAGE_READONLY) {
        on_protected(g, range);
        return;
    }
    let many = range.len() > 1;
    for block in range {
        if many
            && matches!(alike(block, block + 1), Alike::Plain(_))
            && set_protection(block..block + 1, PAGE_READONLY)
        {
            on_protected(g, block..block + 1);
        } else {
            WRITE_FAULTS.set(block..block + 1, false);
            on_failed(g, block);
        }
    }
}

/// Makes blocks `[from, to)` read-only and notes them guarded and clean;
/// spanning each uniform committed region and leaving out non-plain blocks
/// (such as a block with a title `PAGE_GUARD` page), noting placeholders
/// absent and other uncommitted blocks unguardable. The other views of the
/// pages guarded are made read-only too, and linked to them.
fn protect(g: &mut Guards, from: usize, to: usize) {
    // Counted before they are read-only (see `GUARDED`); none of them is
    // guarded yet.
    GUARDED.store(g.protected.len() + g.opening.len() + (to - from), SeqCst);
    let note = |g: &mut Guards, range: Range<usize>| {
        g.protected.extend(range.clone());
        g.clean.extend(range);
    };
    let unmapped = |g: &mut Guards, block: usize| {
        if placeholder(block * BLOCK) {
            g.absent.insert(block);
        } else {
            g.unguardable.insert(block);
        }
    };
    let mut at = from;
    while at < to {
        match alike(at, to) {
            Alike::Plain(span) => {
                protect_span(g, at..at + span, note, unmapped);
                at += span;
            }
            Alike::Mixed => {
                // Leave committed non-plain blocks (e.g. a one-shot title
                // `PAGE_GUARD`) untouched and uncached so their writes keep
                // being reported until they clear.
                at += 1;
            }
            Alike::Uncommitted => {
                unmapped(g, at);
                at += 1;
            }
        }
    }
    protect_other_views(g, from, to);
    GUARDED.store(g.protected.len() + g.opening.len(), SeqCst);
}

/// The other views of the pages `protect` just guarded in `[from, to)`
/// made read-only too, so that a write through them faults here; not clean,
/// though: no copy of theirs took them. Each pair is linked both ways
/// (`Guards::aliased`). Where a view of a page cannot be made read-only
/// (gone since the runtime's index was asked, or carrying a title guard
/// page), the page's other views are opened again, the block guarded
/// included: reported at the next asking, which tries again, rather than
/// clean and missing writes through it.
fn protect_other_views(g: &mut Guards, from: usize, to: usize) {
    let mut pairs = Vec::new();
    aliases(from..to, &mut pairs);
    // Those left open are reported at every asking anyway.
    pairs.retain(|(block, _)| g.protected.contains(block));
    if pairs.is_empty() {
        return;
    }
    let mut others: Vec<usize> = pairs
        .iter()
        .map(|&(_, other)| other)
        .filter(|other| !g.protected.contains(other))
        .collect();
    others.sort_unstable();
    others.dedup();
    // A view the kernel is writing for the runtime is not made read-only
    // (`kernel_writing`): its page's views are opened again, as when one
    // cannot be.
    let mut failed = Vec::new();
    others.retain(|other| {
        if g.unguardable.contains(other) || g.opening.contains(other) {
            failed.push(*other);
            return false;
        }
        true
    });
    if let (Some(&low), Some(&high)) = (others.first(), others.last()) {
        let flying = flying(low..high + 1);
        others.retain(|&other| {
            let landing = is_flying(&flying, other);
            if landing {
                failed.push(other);
            }
            !landing
        });
    }
    // Counted before they are read-only (see `GUARDED`).
    GUARDED.store(g.protected.len() + g.opening.len() + others.len(), SeqCst);
    for run in others.chunk_by(|a, b| a + 1 == *b) {
        let (mut at, end) = (run[0], run[run.len() - 1] + 1);
        while at < end {
            match alike(at, end) {
                Alike::Plain(span) => {
                    protect_span(
                        g,
                        at..at + span,
                        |g, range| g.protected.extend(range),
                        |_, block| failed.push(block),
                    );
                    at += span;
                }
                Alike::Mixed | Alike::Uncommitted => {
                    failed.push(at);
                    at += 1;
                }
            }
        }
    }
    failed.sort_unstable();
    let mut left_open: Vec<usize> = pairs
        .iter()
        .filter(|(_, other)| failed.binary_search(other).is_ok())
        .map(|&(block, _)| block)
        .collect();
    left_open.sort_unstable();
    let mut reopen = Vec::new();
    for (block, other) in pairs {
        if left_open.binary_search(&block).is_ok() {
            // None of the page's views is clean then: none needs the others.
            g.aliased.remove(&block);
            g.aliased.remove(&other);
            reopen.extend([block, other]);
        } else {
            g.aliased.insert(block);
            g.aliased.insert(other);
        }
    }
    let blocks = take_guarded(g, reopen);
    make_writable(&blocks);
    finish_opening(g, &blocks);
}

/// Sets the protection of `blocks` (title memory); false if it failed: not
/// all mapped, or more than one view (whose protections change apart).
pub(super) fn set_protection(blocks: Range<usize>, protection: u32) -> bool {
    use windows_sys::Win32::System::Memory::VirtualProtect;
    #[cfg(test)]
    tests::PROTECTED.with(|n| n.set(n.get() + 1));
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::Protections, 1);
    let mut old = 0;
    // SAFETY: title view memory; only its protection changes.
    unsafe {
        VirtualProtect(
            (blocks.start * BLOCK) as *const _,
            blocks.len() * BLOCK,
            protection,
            &mut old,
        ) != 0
    }
}

/// Whether `at` is in a view (of a section).
fn view(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery};
    #[cfg(test)]
    tests::QUERIED.with(|n| n.set(n.get() + 1));
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::ViewQueries, 1);
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.Type == MEM_MAPPED
}

/// Whether the block at `at` holds nothing until the runtime maps a view
/// there: reserved private memory, not committed and not write-watched, as
/// a placeholder between views is. The system tells a placeholder from a
/// reservation the title placed itself only by the write watch. Free
/// address space (anyone may allocate it), a write-watched reservation (the
/// title commits it without a mapping) and a view not committed yet are
/// not placeholders.
fn placeholder(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        GetWriteWatch, MEM_PRIVATE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    #[cfg(test)]
    tests::QUERIED.with(|n| n.set(n.get() + 1));
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::ViewQueries, 1);
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    if n == 0 || info.State != MEM_RESERVE || info.Type != MEM_PRIVATE {
        return false;
    }
    let (mut found, mut count, mut granularity) = (0usize, 1, 0);
    // SAFETY: room for the one address asked for; no reset. It fails
    // outside write-watched memory.
    let watched = unsafe {
        GetWriteWatch(
            0,
            at as *const _,
            BLOCK,
            (&raw mut found).cast(),
            &mut count,
            &mut granularity,
        )
    } == 0;
    !watched
}

/// Note l'écriture sur les blocs gardés et leurs alias. L'appelant rend
/// ensuite l'accès en écriture à toutes les vues renvoyées.
fn open(g: &mut Guards, blocks: Range<usize>, learn: bool) -> Vec<usize> {
    let mut opening: Vec<usize> = g.protected.range(blocks.clone()).copied().collect();
    for &block in &opening {
        g.owners.touch(block, learn);
    }
    // Only linked blocks are looked up: a kernel write range can be huge.
    let linked: Vec<usize> = g.aliased.range(blocks).copied().collect();
    let mut pairs = Vec::new();
    for run in linked.chunk_by(|a, b| a + 1 == *b) {
        aliases(run[0]..run[run.len() - 1] + 1, &mut pairs);
        for &block in run {
            g.aliased.remove(&block);
        }
        for &(_, other) in &pairs {
            g.aliased.remove(&other);
            opening.push(other);
        }
    }
    take_guarded(g, opening)
}

/// Restitue les octets GPU avant qu'une seule vue puisse devenir accessible en écriture.
fn take_guarded(g: &mut Guards, mut blocks: Vec<usize>) -> Vec<usize> {
    blocks.sort_unstable();
    blocks.dedup();
    for &block in &blocks {
        g.owners.touch(block, false);
    }
    blocks.retain(|block| g.protected.remove(block));
    for block in &blocks {
        g.clean.remove(block);
    }
    blocks
}

fn make_writable(blocks: &[usize]) {
    use windows_sys::Win32::System::Memory::PAGE_READWRITE;
    for run in blocks.chunk_by(|a, b| a + 1 == *b) {
        let (first, last) = (run[0], run[run.len() - 1]);
        if !set_protection(first..last + 1, PAGE_READWRITE) && run.len() > 1 {
            for &block in run {
                set_protection(block..block + 1, PAGE_READWRITE);
            }
        }
    }
}

fn finish_opening(g: &Guards, blocks: &[usize]) {
    for run in blocks.chunk_by(|a, b| a + 1 == *b) {
        WRITE_FAULTS.set(run[0]..run[run.len() - 1] + 1, false);
    }
    // Lowered once write access is back (see `GUARDED`).
    GUARDED.store(g.protected.len() + g.opening.len(), SeqCst);
}

/// Complete the opening even if the test hook unwinds. Restore access before
/// clearing the marks so that a waiting kernel write can use every alias.
struct Opening(Vec<usize>);

impl Drop for Opening {
    fn drop(&mut self) {
        make_writable(&self.0);
        let mut g = guards();
        for block in &self.0 {
            g.opening.remove(block);
        }
        finish_opening(&g, &self.0);
        drop(g);
        REOPENED.notify_all();
    }
}

fn reopen(mut g: GuardLock, blocks: Vec<usize>) {
    if blocks.is_empty() {
        return;
    }
    g.opening.extend(&blocks);
    drop(g);
    let opening = Opening(blocks);
    #[cfg(test)]
    if let Some(hook) = tests::REOPENING.take() {
        hook(&opening.0);
    }
    drop(opening);
}

/// The blocks `[at, at + len)` touches; the length is the title's
/// (`fread`'s size times count, saturated), so the end is cut to the
/// address space.
fn blocks(at: usize, len: usize) -> Range<usize> {
    at / BLOCK..at.saturating_add(len).div_ceil(BLOCK)
}

/// Opens the guarded blocks of title memory `[at, at + len)` and notes them
/// written: before a write that does not fault into our handler, or after
/// one. The runtime calls it (as `kernel_wrote`) after its own writes into
/// physical pages, for each title address showing them (xmem); writes the
/// kernel makes for it take [`kernel_writing`], which calls this.
pub fn writing(at: usize, len: usize) {
    // Without the lock when nothing is guarded (no copy of view memory
    // taken yet, or all of it remapped since), which DirectStorage's
    // workers would contend for; with blocks guarded, the lock as before.
    if !guarding() {
        return;
    }
    let range = blocks(at, len);
    let mut g = settled(range.clone());
    let opening = open(&mut g, range, false);
    reopen(g, opening);
}

/// Kernel writes into title memory under way ([`kernel_writing`]): each
/// one's blocks, by number.
struct InFlight {
    next: u64,
    writes: BTreeMap<u64, Range<usize>>,
}

static IN_FLIGHT: Mutex<InFlight> = Mutex::new(InFlight {
    next: 1,
    writes: BTreeMap::new(),
});

/// How many writes `IN_FLIGHT` holds, set under its lock, so that `guard`
/// takes that lock only when some are out. A write is counted before it
/// opens its blocks (under the guards' lock): a guard that saw none out
/// finished before that opening, which then undoes what it guarded.
static FLYING: AtomicUsize = AtomicUsize::new(0);

fn in_flight() -> std::sync::MutexGuard<'static, InFlight> {
    IN_FLIGHT.lock().unwrap_or_else(|e| e.into_inner())
}

/// The blocks of kernel writes under way that meet `blocks`, merged, in
/// order; empty, without the lock, when none is out. Taken with the guards'
/// lock held (that lock first, this one after; the writes take only this).
fn flying(blocks: Range<usize>) -> Vec<Range<usize>> {
    if FLYING.load(SeqCst) == 0 {
        return Vec::new();
    }
    let mut out: Vec<Range<usize>> = in_flight()
        .writes
        .values()
        .filter(|r| r.start < blocks.end && blocks.start < r.end)
        .cloned()
        .collect();
    out.sort_unstable_by_key(|r| r.start);
    let mut merged: Vec<Range<usize>> = Vec::with_capacity(out.len());
    for r in out {
        match merged.last_mut() {
            Some(last) if r.start <= last.end => last.end = last.end.max(r.end),
            _ => merged.push(r),
        }
    }
    merged
}

/// Whether `block` lies in `flying` (merged and in order, as [`flying`]
/// gives them).
fn is_flying(flying: &[Range<usize>], block: usize) -> bool {
    let i = flying.partition_point(|r| r.end <= block);
    flying.get(i).is_some_and(|r| r.start <= block)
}

/// A kernel write into title memory under way, from [`kernel_writing`] until
/// this drops: its blocks stay open. When it drops, the write is done, and
/// its pages are noted written for copies of write-watched memory too
/// ([`kernel_write_done`]).
#[must_use = "dropping it at once leaves the blocks to the next copy before the kernel writes them"]
#[derive(Debug)]
pub struct KernelWrite {
    id: u64,
    pages: Range<usize>,
}

impl Drop for KernelWrite {
    fn drop(&mut self) {
        if self.id != 0 {
            // Noted before the write stops counting as under way: a copy
            // in between finds the pages both ways.
            kernel_write_done(self.pages.clone());
            let mut f = in_flight();
            f.writes.remove(&self.id);
            FLYING.store(f.writes.len(), SeqCst);
        }
    }
}

/// The largest kernel write whose pages [`kernel_write_done`] notes one by
/// one (16 GiB): a title buffer is far smaller.
const NOTED_PAGES: usize = (16 << 30) / PAGE;

/// A kernel write into pages `pages` is done: where they are write-watched,
/// they are noted written now, after any asking made while it was under
/// way, so that the next copy takes them again. A device that writes into
/// pages locked when the read began (unbuffered or overlapped reads) does
/// so without a fault and need not leave a trace a copy has not already
/// cleared. View memory needs nothing: its blocks were left open meanwhile
/// and are reported written until a copy guards them again.
fn kernel_write_done(pages: Range<usize>) {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    if pages.is_empty() || pages.len() > NOTED_PAGES {
        return;
    }
    let (mut found, mut count, mut granularity) = (0usize, 1, 0);
    // SAFETY: room for the one address asked for; no reset. It fails
    // outside write-watched memory.
    let watched = unsafe {
        GetWriteWatch(
            0,
            (pages.start * PAGE) as *const _,
            pages.len() * PAGE,
            (&raw mut found).cast(),
            &mut count,
            &mut granularity,
        )
    } == 0;
    if !watched {
        return;
    }
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    j.asked += 1;
    let now = j.asked;
    for page in pages {
        stamp(page, now);
    }
}

/// Before the kernel writes into title memory `[at, at + len)` for us (a
/// file read, DirectStorage's raw reads), until the returned value drops:
/// [`writing`], and the blocks are held open meanwhile. A copy that asks
/// about them then neither guards them again nor counts them clean: it
/// reports them written, so the copy after it takes them again. Without
/// this, a copy could make a block read-only in the middle of the read,
/// and the kernel failed the whole call (`ERROR_INVALID_USER_BUFFER`,
/// `ERROR_NOACCESS` if before it; docs/analysis/archive-null.md). Drop it
/// once the write is done: for an overlapped read, once it completed.
pub fn kernel_writing(at: usize, len: usize) -> KernelWrite {
    kernel_writing_range(at, len, true)
}

pub(super) fn kernel_writing_range(at: usize, len: usize, learn: bool) -> KernelWrite {
    if len == 0 {
        return KernelWrite { id: 0, pages: 0..0 };
    }
    let pages = blocks(at, len);
    let id = {
        let mut f = in_flight();
        let id = f.next;
        f.next += 1;
        f.writes.insert(id, pages.clone());
        FLYING.store(f.writes.len(), SeqCst);
        id
    };
    // Under the guards' lock even when nothing is guarded (not `writing`'s
    // shortcut): a copy may be guarding these blocks right now, having
    // looked for writes under way before this one was counted. The lock
    // waits for it, and the opening undoes what it guarded; a copy after it
    // sees this write. (`writing`'s shortcut never fires in the world, where
    // some view is always guarded.)
    let mut g = settled(pages.clone());
    let opening = open(&mut g, pages.clone(), learn);
    reopen(g, opening);
    KernelWrite { id, pages }
}

/// The permanent address of an xmem page and all title views that show it.
type PageLookup = fn(usize, &mut Vec<usize>) -> Option<usize>;
static PAGE_LOOKUP: OnceLock<PageLookup> = OnceLock::new();

/// # Safety
/// The returned addresses must designate the same committed page. The runtime
/// must pin the permanent range before changing any mapping or writing it.
pub unsafe fn set_copy_back_lookup(lookup: PageLookup) {
    let _ = PAGE_LOOKUP.set(lookup);
}

#[must_use]
pub struct CopyBackPin(Option<Range<usize>>);

impl Drop for CopyBackPin {
    fn drop(&mut self) {
        if let Some(r) = self.0.take() {
            guards().owners.unpin(r);
        }
    }
}

/// Holds a range accessible across a kernel call or a physical mapping change.
pub fn pin_copy_back(at: usize, len: usize) -> CopyBackPin {
    CopyBackPin(pin_range(at, len, false))
}

/// Garde toutes les vues de la source lisibles jusqu’à la fin de la lecture du titre.
pub fn pin_title_read(at: usize, len: usize) -> Option<CopyBackPin> {
    pin_range(at, len, true).map(|range| CopyBackPin(Some(range)))
}

fn pin_range(at: usize, len: usize, learn: bool) -> Option<Range<usize>> {
    if !crate::readmap::enabled() || len == 0 {
        return None;
    }
    let range = blocks(at, len);
    guards().owners.pin(range.clone(), learn);
    Some(range)
}

pub(super) fn pin_read(at: usize, len: usize) -> Option<CopyBackPin> {
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    let r = blocks(at, len);
    let mut g = guards();
    if !g.owners.may_have_faulted(r.clone()) {
        return None;
    }
    let _released = g.owners.pin(r.clone(), false);
    #[cfg(feature = "diagnostics")]
    if _released > 0 {
        use crate::copystat::{C, add};
        add(C::KernelReadPages, _released as u64);
        add(C::KernelReadNs, started.elapsed().as_nanos() as u64);
    }
    Some(CopyBackPin(Some(r)))
}

pub(super) fn publish(
    at: usize,
    key: crate::readmap::Key,
    gpu: &mut [u8],
    old: Option<&[u8]>,
) -> Option<(usize, u64)> {
    let lookup = PAGE_LOOKUP.get()?;
    let block = at / PAGE;
    let mut journal = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let mut g = guards();
    let mut views = Vec::new();
    let physical = lookup(block * PAGE, &mut views)? / PAGE;
    views.iter_mut().for_each(|v| *v /= PAGE);
    if !views.contains(&block)
        || views
            .iter()
            .any(|v| g.unguardable.contains(v) || g.opening.contains(v))
        || g.owners.pinned(physical, &views)
    {
        return None;
    }
    if !g.protected.contains(&block) {
        // A write-only GPU destination may never have been refreshed. Start
        // watching it now. A pending kernel write keeps every alias open.
        if views
            .iter()
            .any(|&v| !g.protected.contains(&v) && !writable_range(v * PAGE, PAGE))
            || FLYING.load(SeqCst) != 0 && views.iter().any(|&v| !flying(v..v + 1).is_empty())
        {
            return None;
        }
        install();
        protect(&mut g, block, block + 1);
        // A newly guarded view may already have changed since another
        // shadow used it. The next asking must report it even if publication
        // below falls back. The journal stamp covers successful publication.
        for view in &views {
            g.clean.remove(view);
        }
    }
    if views.iter().any(|v| !g.protected.contains(v)) {
        return None;
    }
    let written = g
        .owners
        .publish(physical, &views, key, at % PAGE, gpu, old)?;
    journal.asked += 1;
    let now = journal.asked;
    for view in views {
        stamp(view, now);
        g.clean.insert(view);
    }
    Some((written, now))
}

/// Two readers may fault together. The second retries the already restored
/// mapping. Only our own remembered pages with read access qualify.
pub(super) fn read_fault(at: usize, learn: bool) -> bool {
    use windows_sys::Win32::Foundation::{GetLastError, SetLastError};
    if !KNOWN_READS.contains(at / PAGE) {
        return false;
    }
    // SAFETY: saves/restores the faulting thread's last error across the query.
    let error = unsafe { GetLastError() };
    let handled = resolve_read_fault(at, learn);
    // SAFETY: restores the caller's last-error value on this thread.
    unsafe { SetLastError(error) };
    handled
}
fn resolve_read_fault(at: usize, learn: bool) -> bool {
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    let block = at / PAGE;
    let mut g = match GUARDS.try_lock() {
        Ok(lock) => GuardLock::new(lock),
        Err(std::sync::TryLockError::Poisoned(_)) => return false,
        Err(std::sync::TryLockError::WouldBlock) => {
            let retry = retry_contended(&READ_FAULTS, block) || !HOLDING.get() && readable_now(at);
            #[cfg(feature = "diagnostics")]
            record_guard_wait(started);
            return retry;
        }
    };
    if g.owners.touch(block, learn) {
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::ReadFaults, 1);
            add(C::ReadFaultNs, started.elapsed().as_nanos() as u64);
        }
        return true;
    }
    g.owners.restored(block) && readable_now(at)
}

fn readable_now(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READONLY, PAGE_READWRITE, VirtualQuery,
    };
    let mut info = MEMORY_BASIC_INFORMATION::default();
    // SAFETY: a sized result and a query-only address.
    unsafe {
        VirtualQuery(at as _, &mut info, size_of_val(&info)) != 0
            && info.State == MEM_COMMIT
            && matches!(info.Protect, PAGE_READONLY | PAGE_READWRITE)
    }
}

/// For diagnostics (`XTP_FULL_REFRESH`): at how many other addresses the
/// title shows the physical page of the 64 KiB block holding `at`, by the
/// runtime's index of its views ([`set_alias_lookup`]); 0 without it. Read
/// only; no guard lock is taken.
pub(crate) fn views_of(at: usize) -> usize {
    let Some(lookup) = ALIAS_LOOKUP.get() else {
        return 0;
    };
    let block = at / BLOCK * BLOCK;
    let mut pairs = Vec::new();
    lookup(block..block + BLOCK, &mut pairs);
    pairs
        .iter()
        .filter(|&&(a, other)| a == block && other != block)
        .count()
}

/// Whether some block of view memory is guarded now, without the guards'
/// lock (see `GUARDED`): if not, a write needs no [`writing`], and the
/// runtime need not look up where the physical pages it wrote show.
pub fn guarding() -> bool {
    GUARDED.load(SeqCst) != 0
}

/// What a host copy of title memory `[start, end)` asks before it copies,
/// for the runtime's tests (xmem's): the pages found written since the last
/// asking, view memory guarded again. `None` if the memory is neither
/// write-watched nor a view.
#[doc(hidden)]
pub fn copy_asks(start: usize, end: usize) -> Option<Vec<usize>> {
    with_writes(start, end, true, |now| {
        (start / PAGE..end.div_ceil(PAGE))
            .filter(|&page| last_write(page) == now)
            .collect()
    })
}

/// Applique la protection demandée par le titre après restitution de la page.
/// Le verrou empêche une publication entre cette restitution et VirtualProtect.
///
/// # Safety
/// Les arguments suivent le contrat de VirtualProtect.
pub unsafe extern "system" fn title_virtual_protect(
    address: *const std::ffi::c_void,
    len: usize,
    protection: u32,
    previous: *mut u32,
) -> i32 {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    // No plain read of title memory runs over the range until the call is
    // made, and reads forget what they learned of its pages: a guard the
    // title puts on now is never met by one (`crate::shadow::title_changing`,
    // first: it waits for reads, which may wait for the guards' lock).
    let _reads = crate::shadow::title_changing(address as usize, len);
    let range = blocks(address as usize, len);
    let mut g = settled(range.clone());
    if len != 0 {
        let opening = open(&mut g, range.clone(), false);
        make_writable(&opening);
        finish_opening(&g, &opening);
    }
    // SAFETY: paramètres de l'appel original, transmis à Windows.
    let ok = unsafe { VirtualProtect(address, len, protection, previous) };
    if ok != 0 && view(address as usize) {
        if protection != PAGE_READWRITE {
            // Une seule sous-page de 4 Kio suffit à interdire une garde de 64 Kio.
            g.unguardable.extend(range);
        } else {
            let first = (address as usize).div_ceil(BLOCK);
            let end = (address as usize).saturating_add(len) / BLOCK;
            for block in first..end {
                g.unguardable.remove(&block);
            }
        }
    }
    ok
}

/// Title memory `[at, at + len)` changed: the runtime mapped or unmapped a
/// view there, or the title reserved, committed or freed memory there. What
/// it shows is new, so it is no longer guarded, nor known to be a view,
/// unmapped or unguardable.
///
/// The runtime tells us after it maps, and both before and after it
/// unmaps, so a copy may have guarded the view in between: blocks still
/// guarded get write access back, else the title's next write there would
/// fault with nothing to take it (the old view may still be written before
/// it goes). The report after an unmapping also forgets what a copy noted
/// of the view since the report before, its start block as a view and its
/// blocks as guarded: the address may be ordinary memory next.
///
/// The other views of the pages shown there are opened too, as by a write,
/// whether or not a guard linked them to this memory, and all of them are
/// unlinked (`Guards::aliased`). A view mapped after a copy of another
/// view of its pages guarded that one is not linked to it, and the title
/// may write through it at once: the runtime notes the new view in its
/// index before it reports it, so the report opens the guarded one, and
/// its copy takes the page again and guards both (the late view aycow
/// found on VIctory #1122). A view the runtime unmaps is opened by the
/// report before, and may still be written until it goes: its other views
/// are opened with it. The runtime forgets it in its index once it is
/// gone, before the report after.
pub fn title_memory_remapped(at: usize, len: usize) {
    crate::texback::remapped(at, len);
    crate::rt::release_buffer(None, at as u64, at.saturating_add(len) as u64);
    // Reads of title memory forget what they learned of its pages, once the
    // reads under way are done (`crate::shadow::title_changing`; before the
    // guards' lock, which reads may wait for).
    drop(crate::shadow::title_changing(at, len));
    crate::resource::memory_remapped(at, len);
    forget_guards(blocks(at, len));
}

fn forget_guards(range: Range<usize>) {
    let mut lock = settled(range.clone());
    let g = &mut *lock;
    g.owners.forget(range.clone());
    KNOWN_VIEWS.set(range.clone(), false);
    for known in [&mut g.absent, &mut g.unguardable, &mut g.aliased] {
        let gone: Vec<usize> = known.range(range.clone()).copied().collect();
        for block in gone {
            known.remove(&block);
        }
    }
    let mut opening: Vec<usize> = g.protected.range(range.clone()).copied().collect();
    let mut pairs = Vec::new();
    aliases(range, &mut pairs);
    for (_, other) in pairs {
        g.aliased.remove(&other);
        opening.push(other);
    }
    let opening = take_guarded(g, opening);
    reopen(lock, opening);
}

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's.
fn install() {
    use windows_sys::Win32::Foundation::{
        EXCEPTION_ACCESS_VIOLATION, GetLastError, NTSTATUS, SetLastError,
    };
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const WRITE: usize = 1;
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] > WRITE
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let at = r.ExceptionInformation[1];
        if r.ExceptionInformation[0] == 0 {
            return if read_fault(
                at,
                !crate::shadow::copy_instruction(r.ExceptionAddress as usize),
            ) {
                EXCEPTION_CONTINUE_EXECUTION
            } else {
                EXCEPTION_CONTINUE_SEARCH
            };
        }
        // SAFETY: conservation et restitution de l’erreur sur le thread du titre.
        let error = unsafe { GetLastError() };
        let retry = write_fault(
            at,
            !crate::shadow::copy_instruction(r.ExceptionAddress as usize),
        );
        // SAFETY: même thread que GetLastError.
        unsafe { SetLastError(error) };
        if retry {
            EXCEPTION_CONTINUE_EXECUTION
        } else {
            EXCEPTION_CONTINUE_SEARCH
        }
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

// Une autre publication peut tenir GUARDS après avoir fermé la vue.
// Seule une garde de XTP mérite alors de réessayer l'instruction.
fn retry_contended(pages: &FaultPages, block: usize) -> bool {
    if !HOLDING.get() && pages.contains(block) {
        std::thread::yield_now();
        return true;
    }
    false
}

pub(super) fn set_pending_view(view: usize, pending: bool) {
    READ_FAULTS.set(view..view + 1, pending);
}

pub(super) fn remember_read_view(view: usize, known: bool) {
    KNOWN_READS.set(view..view + 1, known);
}

pub(super) fn write_fault(at: usize, learn: bool) -> bool {
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    let block = at / BLOCK;
    let mut g = match GUARDS.try_lock() {
        Ok(lock) => GuardLock::new(lock),
        Err(std::sync::TryLockError::Poisoned(_)) => return false,
        Err(std::sync::TryLockError::WouldBlock) => {
            let retry = retry_contended(&WRITE_FAULTS, block) || !HOLDING.get() && writable(at);
            #[cfg(feature = "diagnostics")]
            record_guard_wait(started);
            return retry;
        }
    };
    #[cfg(feature = "diagnostics")]
    let owned = g.owners.owns(block);
    if g.opening.contains(&block) {
        drop(g);
        std::thread::yield_now();
        return true;
    }
    let opening = open(&mut g, block..block + 1, learn);
    if opening.is_empty() {
        return writable(at);
    }
    reopen(g, opening);
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::GuardFaults, 1);
        if owned {
            add(C::WriteFaults, 1);
            add(C::ReadFaultNs, started.elapsed().as_nanos() as u64);
        }
    }
    true
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    writable_range(at, 1)
}

fn writable_range(at: usize, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0
        && info.State == MEM_COMMIT
        && info.Protect == PAGE_READWRITE
        && info.RegionSize - (at - info.BaseAddress as usize) >= len
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    type Reopening = Box<dyn FnOnce(&[usize])>;
    thread_local! {
        pub(super) static REOPENING: std::cell::Cell<Option<Reopening>> = const { std::cell::Cell::new(None) };
        pub(super) static WATCH_QUERIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
        /// The address-space queries `guard` made on this thread (`view`,
        /// `placeholder`; each test runs on its own).
        pub(super) static QUERIED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
        /// The protection changes made on this thread.
        pub(super) static PROTECTED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    fn queried() -> usize {
        QUERIED.with(|n| n.get())
    }

    fn protected() -> usize {
        PROTECTED.with(|n| n.get())
    }

    /// Held by every test that guards blocks: the count of guarded blocks
    /// is the process's, and some tests need it at 0. Each test gives its
    /// guards back before it ends; what one that failed left guarded is
    /// given back here, so that none depends on the others.
    pub(crate) fn serial() -> std::sync::MutexGuard<'static, ()> {
        static SERIAL: Mutex<()> = Mutex::new(());
        let serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        // Reset the guards and the page-read cache these tests share. A
        // title remap would also evict live GPU resources in parallel tests.
        drop(crate::shadow::title_changing(0, usize::MAX));
        forget_guards(blocks(0, usize::MAX));
        same().clear();
        serial
    }

    /// The runtime's index of the title's views, as these tests stand in for
    /// it: pairs of blocks showing the same page ([`same_pages`]).
    fn same() -> std::sync::MutexGuard<'static, Vec<(usize, usize)>> {
        static SAME: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
        SAME.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// What the index answers ([`set_alias_lookup`]), as xmem's does.
    fn other_views(range: Range<usize>, out: &mut Vec<(usize, usize)>) {
        for &(a, b) in same().iter() {
            for (at, other) in [(a, b), (b, a)] {
                if range.contains(&(at * BLOCK)) {
                    out.push((at * BLOCK, other * BLOCK));
                }
            }
        }
    }

    /// `views` (of `blocks` blocks each) show the same pages, as xmem notes
    /// a view before it reports it mapped.
    fn same_pages(views: &[usize], blocks: usize) {
        set_alias_lookup(other_views);
        let mut same = same();
        for (i, &a) in views.iter().enumerate() {
            for &b in &views[i + 1..] {
                same.extend((0..blocks).map(|k| (a / BLOCK + k, b / BLOCK + k)));
            }
        }
    }

    /// `view` (of `blocks` blocks) shows its pages no more, as xmem notes
    /// once it unmapped it, before it reports it again.
    fn not_same(view: usize, blocks: usize) {
        let gone = view / BLOCK..view / BLOCK + blocks;
        same().retain(|(a, b)| !gone.contains(a) && !gone.contains(b));
    }

    /// Whether a guard linked block `at` to other views of its page.
    fn linked(at: usize) -> bool {
        guards().aliased.contains(&(at / BLOCK))
    }

    #[test]
    fn an_opening_keeps_aliases_dirty_until_the_kernel_can_write() {
        use std::sync::mpsc;
        use std::time::{Duration, Instant};

        let _serial = serial();
        let section = section(1);
        let (a, b) = (view_of(section, 1), view_of(section, 1));
        // SAFETY: les deux vues gardent la section en vie.
        unsafe { windows_sys::Win32::Foundation::CloseHandle(section) };
        same_pages(&[a, b], 1);
        assert_eq!(guard(a, a + PAGE), Some(vec![a / PAGE]));

        std::thread::scope(|s| {
            let (ready, paused) = mpsc::channel();
            let (resume, resumed) = mpsc::channel();
            let limit = Duration::from_secs(10);
            s.spawn(move || {
                REOPENING.set(Some(Box::new(move |pages| {
                    assert!(pages.contains(&(a / PAGE)) && pages.contains(&(b / PAGE)));
                    assert!(!HOLDING.get());
                    ready.send(()).unwrap();
                    resumed.recv_timeout(limit).unwrap();
                })));
                writing(a, 1);
            });
            paused.recv_timeout(limit).unwrap();
            assert!(guarding());
            assert!(!writable(a) && !writable(b));
            assert!(write_fault(b, false));
            assert_eq!(guard(a, a + PAGE), Some(vec![a / PAGE]));
            assert_eq!(guard(b, b + PAGE), Some(vec![b / PAGE]));

            let kernel = s.spawn(move || {
                let _token = kernel_writing(b, 1);
                assert!(writable(b));
                assert_eq!(guard(a, a + PAGE), Some(vec![a / PAGE]));
                assert!(writable(b));
                poke(b, 17);
            });
            let until = Instant::now() + limit;
            while flying(b / PAGE..b / PAGE + 1).is_empty() {
                assert!(Instant::now() < until && !kernel.is_finished());
                std::thread::yield_now();
            }
            assert!(!kernel.is_finished());
            resume.send(()).unwrap();
            kernel.join().unwrap();
        });
        assert_eq!(peek(a), 17);
        assert_eq!(guard(a, a + PAGE), Some(vec![a / PAGE]));
        assert_eq!(guard(a, a + PAGE), Some(vec![]));
        unmapped(a, 1);
        unmapped(b, 1);
    }

    #[test]
    fn unwinding_an_opening_restores_write_access() {
        let _serial = serial();
        let view = mapped(1);
        assert_eq!(guard(view, view + PAGE), Some(vec![view / PAGE]));
        REOPENING.set(Some(Box::new(|_| panic!("interrupt the opening"))));
        assert!(std::panic::catch_unwind(|| writing(view, PAGE)).is_err());
        assert!(!guards().opening.contains(&(view / PAGE)));
        assert!(writable(view));
        assert!(!WRITE_FAULTS.contains(view / PAGE));
        let token = kernel_writing(view, PAGE);
        poke(view, 42);
        drop(token);
        assert_eq!(guard(view, view + PAGE), Some(vec![view / PAGE]));
        unmapped(view, 1);
    }

    /// A pagefile section of `blocks` blocks.
    fn section(blocks: usize) -> windows_sys::Win32::Foundation::HANDLE {
        use windows_sys::Win32::System::Memory::CreateFileMappingW;
        // SAFETY: plain section creation.
        let section = unsafe {
            CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (blocks * BLOCK) as u32,
                std::ptr::null(),
            )
        };
        assert!(!section.is_null());
        section
    }

    /// A pagefile section of `blocks` blocks, mapped read-write: its view.
    fn mapped(blocks: usize) -> usize {
        use windows_sys::Win32::System::Memory::{FILE_MAP_WRITE, MapViewOfFile};
        let section = section(blocks);
        // SAFETY: the fresh section, mapped whole; the view keeps it.
        let view = unsafe {
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, blocks * BLOCK).Value;
            windows_sys::Win32::Foundation::CloseHandle(section);
            view as usize
        };
        assert_ne!(view, 0);
        view
    }

    /// Unmaps a view of `blocks` blocks, as the runtime does: reported
    /// before and after, and forgotten in the index in between.
    fn unmapped(view: usize, blocks: usize) {
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
        title_memory_remapped(view, blocks * BLOCK);
        // SAFETY: our view.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view as *mut _,
            })
        };
        not_same(view, blocks);
        title_memory_remapped(view, blocks * BLOCK);
    }

    /// Another view of all of `section` (`blocks` blocks), read-write.
    fn view_of(section: windows_sys::Win32::Foundation::HANDLE, blocks: usize) -> usize {
        use windows_sys::Win32::System::Memory::{FILE_MAP_WRITE, MapViewOfFile};
        // SAFETY: our section, mapped whole.
        let view = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, blocks * BLOCK) }.Value;
        assert!(!view.is_null());
        view as usize
    }

    /// xmem fixture for the isolated shadow test. A private view stays
    /// writable while both addresses available to the title are no-access.
    pub(crate) fn with_physical_views(body: impl FnOnce(usize, usize, usize)) {
        const N: usize = 4;
        static VIEWS: OnceLock<[usize; 3]> = OnceLock::new();
        fn lookup(at: usize, out: &mut Vec<usize>) -> Option<usize> {
            let &[first, second, private] = VIEWS.get()?;
            let offset = [first, second]
                .into_iter()
                .find_map(|v| at.checked_sub(v).filter(|n| *n < N * PAGE))?;
            out.extend([first + offset, second + offset]);
            Some(private + offset)
        }
        let section = section(N);
        let [first, second, private] = std::array::from_fn(|_| view_of(section, N));
        VIEWS.set([first, second, private]).unwrap();
        same_pages(&[first, second], N);
        // SAFETY: all three views live throughout the test, and the private
        // range is pinned over unmapping.
        unsafe {
            set_copy_back_lookup(lookup);
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        body(first, second, private);
        let _pin = pin_copy_back(private, N * PAGE);
        for at in [first, second, private] {
            unmapped(at, N);
        }
    }

    #[test]
    fn pending_capacity_and_faults_while_the_lock_is_held() {
        let _serial = serial();
        let section = section(2);
        let private = view_of(section, 2);
        let view = view_of(section, 2);
        // SAFETY: les deux vues gardent la section en vie.
        unsafe { windows_sys::Win32::Foundation::CloseHandle(section) };
        let mut owners = crate::readmap::Owners::new(1);
        let key = crate::readmap::Key {
            size: PAGE as u64,
            alignment: 16,
            writer: 0,
            offset: 0,
        };
        let publish = |owners: &mut crate::readmap::Owners, i, byte| {
            owners.publish(
                private / PAGE + i,
                &[view / PAGE + i],
                key,
                0,
                &mut [byte],
                None,
            )
        };
        assert_eq!(publish(&mut owners, 0, 1), Some(0));
        assert_eq!(publish(&mut owners, 0, 2), Some(0));
        assert_eq!(publish(&mut owners, 1, 3), Some(1));
        assert_eq!(peek(private), 0);
        assert_eq!(peek(private + PAGE), 3);
        {
            let _lock = guards();
            // Un défaut dans la section critique est transmis au rapporteur.
            assert!(!read_fault(view, true));
            assert!(!write_fault(view, true));
            // Un autre thread peut reprendre une lecture sur notre garde.
            std::thread::scope(|s| {
                assert!(s.spawn(|| read_fault(view, true)).join().unwrap());
            });
        }
        assert!(owners.touch(view / PAGE, false));
        assert_eq!(peek(private), 2);
        assert_eq!(publish(&mut owners, 1, 4), Some(0));
        assert!(owners.touch(view / PAGE + 1, false));
        assert_eq!(peek(private + PAGE), 4);
        owners.forget(view / PAGE..view / PAGE + 2);
        unmapped(view, 2);
        unmapped(private, 2);
    }

    /// Writes byte `byte` at `at` as the title's CPU does (into our handler
    /// where guarded).
    fn poke(at: usize, byte: u8) {
        // SAFETY: a mapped byte of a test's view.
        unsafe { (at as *mut u8).write_volatile(byte) };
    }

    /// The byte at `at`.
    fn peek(at: usize) -> u8 {
        // SAFETY: a mapped byte of a test's view.
        unsafe { (at as *const u8).read_volatile() }
    }

    /// A fresh placeholder of `blocks` blocks: its address.
    fn placeholder_of(blocks: usize) -> usize {
        use windows_sys::Win32::System::Memory::{
            MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: plain reservation of a placeholder.
        let p = unsafe {
            VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                blocks * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            )
        } as usize;
        assert_ne!(p, 0);
        p
    }

    /// Maps `blocks` blocks of `section`, from block `from` on, over
    /// `[at, at + blocks)` of a placeholder, as xmem maps physical pages:
    /// split out of the placeholder first (which fails harmlessly when it
    /// is the whole placeholder), then reported. The view.
    fn map_view(
        section: windows_sys::Win32::Foundation::HANDLE,
        at: usize,
        from: usize,
        blocks: usize,
    ) -> windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
        use windows_sys::Win32::System::Memory::{
            MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER, MapViewOfFile3,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: acts on our placeholder only, and maps our section there.
        let view = unsafe {
            VirtualFree(
                at as *mut _,
                blocks * BLOCK,
                MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
            );
            MapViewOfFile3(
                section,
                GetCurrentProcess(),
                at as *const _,
                (from * BLOCK) as u64,
                blocks * BLOCK,
                MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            )
        };
        assert_eq!(view.Value as usize, at);
        title_memory_remapped(at, blocks * BLOCK);
        view
    }

    /// Releases the allocation of `blocks` blocks at `at`, as the title
    /// frees memory through xmem: reported after.
    fn released(at: usize, blocks: usize) {
        // SAFETY: our own allocation.
        assert_ne!(unsafe { VirtualFree(at as *mut _, 0, MEM_RELEASE) }, 0);
        title_memory_remapped(at, blocks * BLOCK);
    }

    /// Whether the guards note anything of blocks `[at, at + len)`.
    fn noted(at: usize, len: usize) -> bool {
        let g = guards();
        blocks(at, len).any(|block| KNOWN_VIEWS.contains(block))
            || [
                &g.protected,
                &g.clean,
                &g.absent,
                &g.unguardable,
                &g.aliased,
            ]
            .into_iter()
            .any(|known| known.range(blocks(at, len)).next().is_some())
    }

    /// One page in two views (a pagefile section mapped twice), as xmem
    /// maps physical pages in a pool and in the GPU's tiled heap: a copy of
    /// one view guards the other with it, and a write through either,
    /// faulting or announced by the kernel, is seen by a guard on the
    /// other. Before, the write through the other view did not fault, and
    /// the copy kept what it first took. (Ported from
    /// Marvisak/VIctory#1122 (head 9ed6093) by tightfall.)
    #[test]
    fn a_write_through_an_alias_view_is_seen_by_a_guard_on_the_other() {
        let _serial = serial();
        let section = section(1);
        let (a, b) = (view_of(section, 1), view_of(section, 1));
        same_pages(&[a, b], 1);
        // A copy of view `a` takes the block and guards it, and `b`.
        assert_eq!(guard(a, a + BLOCK), Some(vec![a / PAGE]));
        assert_eq!(guard(a, a + BLOCK), Some(vec![]), "clean");
        assert!(!writable(a) && !writable(b), "both views guarded");
        assert_eq!(GUARDED.load(SeqCst), 2);
        assert!(linked(a) && linked(b));
        // The title writes the page through `b`: `a` shows it written.
        poke(b + 8, 7);
        assert_eq!(peek(a + 8), 7);
        assert!(writable(a) && writable(b), "both opened");
        assert!(!linked(a) && !linked(b), "unlinked, none of them clean");
        assert_eq!(
            guard(a, a + BLOCK),
            Some(vec![a / PAGE]),
            "written through the alias"
        );
        // And the other way round: a kernel write into `a` opens `b`.
        assert_eq!(
            guard(b, b + BLOCK),
            Some(vec![b / PAGE]),
            "no copy of `b` took it yet"
        );
        assert_eq!(guard(b, b + BLOCK), Some(vec![]));
        writing(a + 16, 4);
        assert!(writable(b));
        assert_eq!(guard(b, b + BLOCK), Some(vec![b / PAGE]));
        unmapped(a, 1);
        unmapped(b, 1);
        assert_eq!(GUARDED.load(SeqCst), 0);
        assert!(!noted(a, BLOCK) && !noted(b, BLOCK));
        // SAFETY: our section.
        unsafe { windows_sys::Win32::Foundation::CloseHandle(section) };
    }

    /// A view of a page mapped after a copy of another view of it guarded
    /// that one (the late view aycow found on VIctory #1122): xmem notes it,
    /// then reports it, and the report opens the guarded view, so that its
    /// copy takes the page again (the title may have written it through the
    /// new view already) and guards both. Then the new view is unmapped while
    /// the first stays: the report before opens both (the view may still be
    /// written until it goes), and once it is gone the first is guarded on
    /// its own, linked to nothing.
    #[test]
    fn views_mapped_and_unmapped_beside_a_guarded_one() {
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
        let _serial = serial();
        let section = section(1);
        let a = view_of(section, 1);
        assert_eq!(guard(a, a + BLOCK), Some(vec![a / PAGE]));
        assert_eq!(guard(a, a + BLOCK), Some(vec![]), "clean");
        // The second view, as xmem maps it: noted, then reported.
        let b = view_of(section, 1);
        same_pages(&[a, b], 1);
        title_memory_remapped(b, BLOCK);
        assert!(writable(a), "the guarded view opened");
        poke(b + 8, 1);
        assert_eq!(
            guard(a, a + BLOCK),
            Some(vec![a / PAGE]),
            "the write through the late view"
        );
        assert!(!writable(b), "guarded with `a`");
        // From now on, through the handler.
        poke(b + 8, 2);
        assert_eq!(guard(a, a + BLOCK), Some(vec![a / PAGE]));
        assert_eq!(guard(a, a + BLOCK), Some(vec![]));
        // `b` unmapped while `a` stays: reported, unmapped, forgotten in the
        // index, reported again.
        title_memory_remapped(b, BLOCK);
        assert!(
            writable(a) && writable(b),
            "`b` may be written until it goes"
        );
        poke(b + 8, 3);
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: b as *mut _ }) };
        not_same(b, 1);
        title_memory_remapped(b, BLOCK);
        assert!(!noted(b, BLOCK), "nothing kept of `b`");
        assert_eq!(
            guard(a, a + BLOCK),
            Some(vec![a / PAGE]),
            "written through `b`"
        );
        assert_eq!(peek(a + 8), 3);
        assert_eq!(guard(a, a + BLOCK), Some(vec![]));
        assert!(!linked(a), "no other view any more");
        assert_eq!(GUARDED.load(SeqCst), 1, "`a` alone");
        unmapped(a, 1);
        // SAFETY: our section.
        unsafe { windows_sys::Win32::Foundation::CloseHandle(section) };
    }

    /// xmem unmaps one of two views of a page as it does: reported, a copy
    /// of the other guarding both in between, unmapped, and only then
    /// forgotten in the index. In that window the gone view cannot be made
    /// read-only or writable again: a write fault on the other view that
    /// opens both leaves the title's last error as it was, and a copy's
    /// guard leaves both views open (reported again, tried again) rather
    /// than clean. Once the index forgets it, the remaining view is guarded
    /// on its own.
    #[test]
    fn a_view_gone_before_the_index_forgets_it() {
        use windows_sys::Win32::Foundation::{GetLastError, SetLastError};
        use windows_sys::Win32::System::Memory::{MEM_PRESERVE_PLACEHOLDER, UnmapViewOfFile2};
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let _serial = serial();
        let section = section(1);
        let (a, b) = (view_of(section, 1), placeholder_of(1));
        let view = map_view(section, b, 0, 1);
        same_pages(&[a, b], 1);
        assert_eq!(guard(a, a + BLOCK), Some(vec![a / PAGE]));
        // xmem unmaps `b`: reported, `a`'s copy guards both again, unmapped.
        title_memory_remapped(b, BLOCK);
        assert_eq!(guard(a, a + BLOCK), Some(vec![a / PAGE]));
        assert!(!writable(a) && guards().protected.contains(&(b / BLOCK)));
        // Keep the address reserved until the alias index forgets it.
        assert_ne!(
            // SAFETY: our view replaced this placeholder.
            unsafe { UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER) },
            0
        );
        // The title writes `a`: the handler opens it, and fails on `b`.
        // SAFETY: plain writes and reads of this thread's last error.
        unsafe { SetLastError(1234) };
        poke(a + 8, 1);
        // SAFETY: as above.
        assert_eq!(unsafe { GetLastError() }, 1234, "the title's last error");
        assert!(writable(a));
        // A copy of `a` cannot guard `b`: `a` is left open, reported again.
        assert_eq!(guard(a, a + BLOCK), Some(vec![a / PAGE]));
        assert!(writable(a), "not clean while `b` cannot be guarded");
        assert_eq!(guard(a, a + BLOCK), Some(vec![a / PAGE]), "tried again");
        assert_eq!(GUARDED.load(SeqCst), 0);
        // Forgotten in the index, reported again: `a` alone.
        not_same(b, 1);
        title_memory_remapped(b, BLOCK);
        assert!(!noted(b, BLOCK));
        assert_eq!(guard(a, a + BLOCK), Some(vec![a / PAGE]));
        assert_eq!(guard(a, a + BLOCK), Some(vec![]));
        assert!(!writable(a) && !linked(a));
        released(b, 1);
        unmapped(a, 1);
        // SAFETY: our section.
        unsafe { windows_sys::Win32::Foundation::CloseHandle(section) };
    }

    /// A host copy of a title buffer made through one view of a page (as
    /// the GPU side copies a buffer it reads) takes the page again once the
    /// title wrote it through another view; before VIctory #1122 it kept
    /// what it first took (the world's stale mesh data).
    #[test]
    fn a_host_copy_through_one_view_takes_writes_through_another() {
        let test = "watch::tests::a_host_copy_through_one_view_takes_writes_through_another";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let _serial = serial();
        let section = section(1);
        let (a, b) = (view_of(section, 1), view_of(section, 1));
        same_pages(&[a, b], 1);
        poke(a + 8, 1);
        crate::shadow::register(a, a as u64, BLOCK as u64);
        let copy = crate::shadow::find(a as u64, BLOCK as u64).expect("a host copy");
        assert_eq!(copy.update(a, BLOCK), BLOCK, "the first copy takes it all");
        assert_eq!(copy.update(a, BLOCK), 0, "nothing written since");
        poke(b + 8, 7);
        assert_eq!(
            copy.update(a, BLOCK),
            BLOCK,
            "written through `b`: taken again"
        );
        assert_eq!(copy.update(a, BLOCK), 0);
        crate::shadow::unregister(a, a as u64);
        drop(copy);
        unmapped(a, 1);
        unmapped(b, 1);
        // SAFETY: our section.
        unsafe { windows_sys::Win32::Foundation::CloseHandle(section) };
    }

    /// F5: 64 neighbouring blocks of one view are guarded with one
    /// protection change and opened by a kernel write over them with one.
    #[test]
    fn neighbouring_blocks_open_in_one_protection_change() {
        let _serial = serial();
        let view = mapped(64);
        let all: Vec<usize> = (view / PAGE..view / PAGE + 64).collect();
        let before = protected();
        assert_eq!(guard(view, view + 64 * BLOCK), Some(all.clone()));
        assert_eq!(protected() - before, 1, "guarded in one change");
        assert_eq!(GUARDED.load(SeqCst), 64);
        assert!((0..64).all(|b| !writable(view + b * BLOCK)));
        let before = protected();
        writing(view + 100, 64 * BLOCK - 200);
        assert_eq!(protected() - before, 1, "opened in one change");
        assert_eq!(GUARDED.load(SeqCst), 0);
        assert!((0..64).all(|b| writable(view + b * BLOCK)));
        // Opened means written: all reported again, and guarded again.
        assert_eq!(guard(view, view + 64 * BLOCK), Some(all));
        assert_eq!(guard(view, view + 64 * BLOCK), Some(Vec::new()));
        unmapped(view, 64);
        assert_eq!(GUARDED.load(SeqCst), 0);
    }

    /// The runtime reports a mapping after it is made: a copy that guarded
    /// the new view in between leaves it read-only, and the report gives
    /// write access back (the title's next write there used to fault with
    /// no guard to take it, and the process ended).
    #[test]
    fn remapped_memory_is_writable_again() {
        let _serial = serial();
        let view = mapped(2);
        let all: Vec<usize> = (view / PAGE..view / PAGE + 2).collect();
        assert_eq!(guard(view, view + 2 * BLOCK), Some(all.clone()));
        assert!(!writable(view) && !writable(view + BLOCK));
        title_memory_remapped(view, 2 * BLOCK);
        assert!(writable(view) && writable(view + BLOCK));
        assert_eq!(GUARDED.load(SeqCst), 0);
        // SAFETY: our view, read-write again.
        unsafe { ((view + BLOCK + 8) as *mut u8).write_volatile(1) };
        // New contents: reported again.
        assert_eq!(guard(view, view + 2 * BLOCK), Some(all));
        unmapped(view, 2);
    }

    /// F5: a run of guarded blocks over two views (which one protection
    /// change cannot span) opens block by block.
    #[test]
    fn runs_over_two_views_open_block_by_block() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let _serial = serial();
        // SAFETY: a placeholder of four blocks, split in two, each half
        // showing two blocks of one section.
        let (p, views) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                4 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (4 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            VirtualFree(
                p as *mut _,
                2 * BLOCK,
                MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
            );
            let views = [0, 2].map(|block| {
                let view = MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    2 * BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                );
                assert_eq!(view.Value as usize, p + block * BLOCK);
                view
            });
            windows_sys::Win32::Foundation::CloseHandle(section);
            (p, views)
        };
        assert!(guard(p, p + 4 * BLOCK).is_some());
        assert_eq!(GUARDED.load(SeqCst), 4);
        assert!((0..4).all(|b| !writable(p + b * BLOCK)));
        let before = protected();
        writing(p, 4 * BLOCK);
        assert_eq!(protected() - before, 5, "the run, then each block");
        assert!((0..4).all(|b| writable(p + b * BLOCK)));
        assert_eq!(GUARDED.load(SeqCst), 0);
        title_memory_remapped(p, 4 * BLOCK);
        // SAFETY: our views, back to free address space.
        unsafe {
            for view in views {
                UnmapViewOfFile2(GetCurrentProcess(), view, 0);
            }
        }
    }

    /// F5: with nothing guarded, a kernel write does not wait for the
    /// guards' lock (many DirectStorage workers call it); with a block
    /// guarded it does.
    #[test]
    fn writing_takes_the_lock_only_when_something_is_guarded() {
        use std::time::Duration;
        let _serial = serial();
        let plain = vec![0u8; PAGE];
        let at = plain.as_ptr() as usize;
        let returns = |lock_held_for: Duration| {
            let (tx, rx) = std::sync::mpsc::channel();
            let g = guards();
            std::thread::scope(|s| {
                s.spawn(move || {
                    writing(at, PAGE);
                    tx.send(()).unwrap();
                });
                let returned = rx.recv_timeout(lock_held_for).is_ok();
                drop(g);
                returned
            })
        };
        assert_eq!(GUARDED.load(SeqCst), 0, "`serial` gave every guard back");
        assert!(!guarding());
        assert!(returns(Duration::from_secs(10)), "it waited for the lock");
        let view = mapped(1);
        assert!(guard(view, view + BLOCK).is_some());
        assert_eq!(GUARDED.load(SeqCst), 1);
        assert!(guarding(), "the runtime looks up the views it writes");
        assert!(
            !returns(Duration::from_millis(200)),
            "it did not take the lock"
        );
        unmapped(view, 1);
        assert!(!guarding());
    }

    /// A view is asked about once per block an asking starts in, again
    /// only after the runtime remapped it; other memory every time.
    #[test]
    fn views_are_queried_once_until_remapped() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile,
        };
        let _serial = serial();
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        let at = queried();
        assert!(guard(view, view + 2 * BLOCK).is_some());
        assert!(guard(view + 8, view + BLOCK).is_some());
        assert_eq!(queried(), at + 1);
        // Another start block is asked about once.
        assert!(guard(view + BLOCK, view + 2 * BLOCK).is_some());
        assert!(guard(view + BLOCK, view + 2 * BLOCK).is_some());
        assert_eq!(queried(), at + 2);
        // The runtime remapped the first block: asked again.
        title_memory_remapped(view, BLOCK);
        assert!(guard(view, view + 2 * BLOCK).is_some());
        assert_eq!(queried(), at + 3);
        // Private memory is not a view, however often asked.
        let plain = vec![0u8; 2 * BLOCK];
        let p = plain.as_ptr() as usize;
        assert!(guard(p, p + 16).is_none());
        assert!(guard(p, p + 16).is_none());
        assert_eq!(queried(), at + 5);
        unmapped(view, 2);
    }

    #[test]
    fn known_views_skip_write_watch_until_their_block_is_remapped() {
        let _serial = serial();
        let view = mapped(2);
        let next = view + PAGE;
        let end = next + PAGE;
        let queries = WATCH_QUERIES.get();
        writes(view, next, true).expect("first view probe");
        writes(next, end, true).expect("second view probe");
        assert_eq!(WATCH_QUERIES.get(), queries + 2);

        assert!(writes(view, end, false).is_none());
        // SAFETY: writing our mapped view goes through the guard handler.
        unsafe { (view as *mut u8).write_volatile(1) };
        let now = writes(view, end, true).expect("known view");
        assert_eq!(last_write(view / PAGE), now);
        assert_ne!(last_write(next / PAGE), now);
        assert_eq!(WATCH_QUERIES.get(), queries + 2);

        title_memory_remapped(view, PAGE);
        assert!(writes(next, end, false).is_none());
        assert_eq!(WATCH_QUERIES.get(), queries + 2);
        writes(view, next, true).expect("probe after remap");
        assert_eq!(WATCH_QUERIES.get(), queries + 3);
        unmapped(view, 2);
    }

    #[test]
    fn a_stale_view_hint_preserves_writes_at_a_reused_address() {
        let _serial = serial();
        let section = section(2);
        let (at, alias) = (view_of(section, 2), view_of(section, 2));
        // SAFETY: les deux vues gardent la section en vie.
        unsafe { windows_sys::Win32::Foundation::CloseHandle(section) };
        same_pages(&[at, alias], 2);
        let end = at + 2 * PAGE;
        let first = at / PAGE;
        copy_asks(at, end).unwrap();
        poke(alias + PAGE, 11);
        assert_eq!(copy_asks(at, end), Some(vec![first + 1]));
        unmapped(alias, 2);
        unmapped(at, 2);

        // SAFETY: nouvelle allocation privée à l'adresse de notre ancienne vue.
        let allocation = unsafe {
            VirtualAlloc(
                at as _,
                2 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        };
        if allocation.is_null() {
            // A parallel allocator can claim the address after the unmap.
            assert_eq!(
                // SAFETY: read the error from VirtualAlloc on this thread.
                unsafe { windows_sys::Win32::Foundation::GetLastError() },
                windows_sys::Win32::Foundation::ERROR_INVALID_ADDRESS
            );
            return;
        }
        assert_eq!(allocation as usize, at);
        // Reproduit un indice périmé, indépendamment de l'ordonnancement du remappage.
        KNOWN_VIEWS.set(first..first + 1, true);
        poke(at + PAGE, 12);
        assert!(writes(at, end, false).is_none());
        assert_eq!(copy_asks(at, end), Some(vec![first, first + 1]));
        poke(at, 13);
        assert_eq!(copy_asks(at, end), Some(vec![first]));
        title_memory_remapped(at, 2 * PAGE);
        writes(at, end, false).expect("private allocation");
        poke(at + PAGE, 14);
        assert_eq!(copy_asks(at, end), Some(vec![first + 1]));
        released(at, 2);
    }

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 4 * PAGE);
        let first = writes(start, end, false).expect("watched");
        assert!((start / PAGE..end / PAGE).all(|page| last_write(page) < first));
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(1) };
        let now = writes(start, end, false).expect("watched");
        assert_eq!(last_write(p / PAGE + 1), now);
        // Another asker after the kernel's bit was cleared still sees it.
        let later = writes(p + PAGE, p + 2 * PAGE, false).expect("watched");
        assert!(later > now);
        assert_eq!(last_write(p / PAGE + 1), now);
        let mut plain = vec![0u8; PAGE];
        assert!(
            writes(
                plain.as_mut_ptr() as usize,
                plain.as_ptr() as usize + PAGE,
                true
            )
            .is_none()
        );
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile,
        };
        let _serial = serial();
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK).is_some()); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(
            !read(&mut file),
            "the kernel does not fault into our handler"
        );
        writing(view + 100, 13);
        assert!(read(&mut file));
        assert_eq!(
            // SAFETY: the bytes just read.
            unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) },
            b"from the disk"
        );
        assert_eq!(
            guard(view, view + 2 * BLOCK),
            Some((view / PAGE..view / PAGE + 1).collect())
        );
        unmapped(view, 2);
    }

    /// What refreshing a buffer the size of the world's 174 MB heap costs:
    /// asking the kernel for its written pages (a few written, or 200 MB of
    /// them as in a world frame), and copying 200 MB of pages through the
    /// kernel (`host::copy_mapped`, as refreshes do) or plainly. Run with
    /// `cargo test --release -p xtp-gpu -- --ignored --nocapture watch_and_copy_costs`.
    #[test]
    #[ignore = "benchmark"]
    fn watch_and_copy_costs() {
        const LEN: usize = 174 << 20;
        const WRITTEN: usize = 200 << 20;
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                LEN,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        // SAFETY: our committed pages.
        let title = unsafe { std::slice::from_raw_parts_mut(p as *mut u8, LEN) };
        title.iter_mut().enumerate().for_each(|(i, b)| *b = i as u8);
        let mut copy = vec![0u8; LEN];
        let ask = || {
            let started = std::time::Instant::now();
            writes(p, p + LEN, false).expect("watched");
            started.elapsed()
        };
        ask();
        let few: Vec<_> = (0..20)
            .map(|k| {
                // A few scattered kernel pages written by the title.
                for i in 0..50 {
                    title[(k * 7919 + i * 104_729) * 4096 % LEN] ^= 1;
                }
                ask()
            })
            .collect();
        let many: Vec<_> = (0..5)
            .map(|_| {
                for page in (0..LEN).step_by(4096) {
                    title[page] ^= 1;
                }
                ask()
            })
            .collect();
        let per_call =
            |t: &[std::time::Duration]| t.iter().sum::<std::time::Duration>() / t.len() as u32;
        println!(
            "174 MB write-watched: asking the kernel takes {:?} with a few pages written, {:?} \
             with all written",
            per_call(&few),
            per_call(&many)
        );
        // 200 MB in runs of stale 64 KiB pages, through the kernel as
        // refreshes copy, or plainly.
        for run in [PAGE, 64 * PAGE] {
            let timed = |copy_run: &mut dyn FnMut(usize)| {
                let started = std::time::Instant::now();
                for r in 0..WRITTEN / run {
                    copy_run(r * run % (LEN - run + 1));
                }
                WRITTEN as f64 / started.elapsed().as_secs_f64() / 1e9
            };
            let kernel = timed(&mut |at| {
                let (to, from) = (copy[at..].as_mut_ptr(), title[at..].as_ptr());
                crate::host::copy_mapped(to, from, run, false)
            });
            let plain = timed(&mut |at| copy[at..at + run].copy_from_slice(&title[at..at + run]));
            println!(
                "200 MB in {} KiB runs: {kernel:.2} GB/s through the kernel, {plain:.2} GB/s plainly",
                run >> 10
            );
        }
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// What a world frame's small buffer copies into guarded view memory
    /// cost (run1: 30 to 37 submissions and about 6 000 copies of about
    /// 600 bytes per frame, into the 174 MB heap, a view): each copied on
    /// the CPU at submission (`copy_mapped` into title memory, which opens
    /// the guard per copy) with the mapping asked first as before or only
    /// when the copy fails, or written back after the GPU copied them,
    /// range by range or merged per submission; the refresh that guards the
    /// blocks again counted too. Run with
    /// `cargo test --release -p xtp-gpu -- --ignored --nocapture guarded_copy_costs`.
    #[test]
    #[ignore = "benchmark"]
    fn guarded_copy_costs() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile,
        };
        let _serial = serial();
        const LEN: usize = 174 << 20;
        const SUBMISSIONS: usize = 30;
        const COPIES: usize = 200;
        const BYTES: usize = 600;
        // SAFETY: a pagefile section mapped read-write, unmapped at the end.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                LEN as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, LEN).Value as usize
        };
        assert_ne!(view, 0);
        let source = vec![7u8; BYTES];
        // Where the copies of submission `s` go: a ring the title fills in
        // order, or scattered over the heap.
        let ring = |s: usize, c: usize| ((s * COPIES + c) * BYTES) % (LEN - BYTES);
        let scattered =
            |s: usize, c: usize| ((s * COPIES + c) * 7_919 * 4_096 + c * 64) % (LEN - BYTES);
        let frame = |place: &dyn Fn(usize, usize) -> usize, how: &str| {
            let started = std::time::Instant::now();
            for s in 0..SUBMISSIONS {
                // The submission's refresh guards the heap's blocks again.
                writes(view, view + LEN, true).expect("a view");
                let ranges: Vec<(usize, usize)> =
                    (0..COPIES).map(|c| (view + place(s, c), BYTES)).collect();
                match how {
                    "asked first" => {
                        for &(at, n) in &ranges {
                            if crate::texture::committed(at as *const u8, n) {
                                crate::host::copy_mapped(at as *mut u8, source.as_ptr(), n, true);
                            }
                        }
                    }
                    "cpu" => {
                        for &(at, n) in &ranges {
                            crate::host::copy_mapped(at as *mut u8, source.as_ptr(), n, true);
                        }
                    }
                    "back" => {
                        for &(at, n) in &ranges {
                            writing(at, n);
                            crate::host::write_mapped(at as *mut u8, source.as_ptr(), n);
                        }
                    }
                    _ => {
                        let ranges = ranges.into_iter().map(|(at, n)| ((), at, n)).collect();
                        let merged = crate::host::merge_ranges(ranges, |_| 0);
                        let block = vec![7u8; merged.iter().map(|r| r.2).max().unwrap_or(0)];
                        for ((), at, n) in merged {
                            writing(at, n);
                            crate::host::write_mapped(at as *mut u8, block.as_ptr(), n);
                        }
                    }
                }
            }
            started.elapsed()
        };
        for (layout, place) in [
            ("in a ring", &ring as &dyn Fn(usize, usize) -> usize),
            ("scattered", &scattered),
        ] {
            let [first, cpu, back, merged] =
                ["asked first", "cpu", "back", "merged"].map(|how| frame(place, how));
            println!(
                "a frame of {SUBMISSIONS} x {COPIES} copies of {BYTES} bytes {layout}: on the \
                 CPU at submission {first:?} with the mapping asked first, {cpu:?} asked on \
                 failure only; {back:?} written back range by range, {merged:?} merged"
            );
        }
        unmapped(view, LEN / BLOCK);
    }

    /// A write or mapping that runs past the end of the address space (a
    /// title's `fread` with a huge count) covers up to the end; the sum used
    /// to overflow. Up there no test has guarded blocks.
    #[test]
    fn ranges_past_the_address_space_end_do_not_overflow() {
        let top = usize::MAX - 16;
        writing(top, PAGE);
        title_memory_remapped(top, usize::MAX);
    }

    /// Blocks of a placeholder between views are reported once, then not
    /// again (they cannot be guarded) until the runtime maps them.
    #[test]
    fn unmapped_blocks_are_reported_once_until_mapped() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let _serial = serial();
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of two; block 0 shows the section's first block, as the
        // runtime maps physical pages.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let first = map(0);
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(
            guard(p, p + 3 * BLOCK),
            Some(Vec::new()),
            "nothing written since"
        );
        let second = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        // As xmem unmaps and the title frees: reported before and after.
        title_memory_remapped(p, 3 * BLOCK);
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), second, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(p, 3 * BLOCK);
    }

    /// A view of the first block of `section` in a block of its own, then a
    /// reservation of one block right after it, write-watched (the
    /// runtime's) or not (one the title placed itself): their address and
    /// the view. Retried if another thread takes the address in between.
    fn view_then_reservation(
        section: windows_sys::Win32::Foundation::HANDLE,
        watched: bool,
    ) -> (
        usize,
        windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS,
    ) {
        use windows_sys::Win32::System::Memory::UnmapViewOfFile2;
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        (0..8)
            .find_map(|_| {
                // A placeholder of two blocks: block 0 shows the section,
                // block 1 goes back to free address space and is reserved
                // again.
                let p = placeholder_of(2);
                let view = map_view(section, p, 0, 1);
                let watching = if watched { MEM_WRITE_WATCH } else { 0 };
                // SAFETY: the placeholder's second block, ours, then a plain
                // reservation there.
                let at = unsafe {
                    VirtualFree((p + BLOCK) as *mut _, 0, MEM_RELEASE);
                    VirtualAlloc(
                        (p + BLOCK) as *const _,
                        BLOCK,
                        MEM_RESERVE | watching,
                        PAGE_READWRITE,
                    )
                } as usize;
                if at == p + BLOCK {
                    return Some((p, view));
                }
                title_memory_remapped(p, BLOCK);
                // SAFETY: our view, back to free address space.
                unsafe { UnmapViewOfFile2(GetCurrentProcess(), view, 0) };
                None
            })
            .expect("the address after the view stays free")
    }

    /// Takes down what [`view_then_reservation`] made, as xmem unmaps and
    /// the title frees: reported before and after.
    fn view_and_reservation_gone(
        p: usize,
        view: windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS,
        section: windows_sys::Win32::Foundation::HANDLE,
    ) {
        use windows_sys::Win32::System::Memory::UnmapViewOfFile2;
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        title_memory_remapped(p, 2 * BLOCK);
        // SAFETY: our view, reservation and section.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), view, 0);
            VirtualFree((p + BLOCK) as *mut _, 0, MEM_RELEASE);
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(p, 2 * BLOCK);
    }

    /// A range from a view into a write-watched reservation (the title's
    /// own allocation) that is not committed yet: the reservation is no
    /// placeholder, since the title commits it without a mapping. It stays
    /// reported, asked about once (since 4a873fb, every asking asked the
    /// system again), until committed: xmem reports the commit, and from
    /// then on it is guarded and its writes are seen.
    #[test]
    fn watched_reservations_next_to_views_are_not_absent() {
        let _serial = serial();
        let section = section(1);
        let (p, view) = view_then_reservation(section, true);
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let both: Vec<usize> = (0..2).flat_map(pages).collect();
        assert_eq!(guard(p, p + 2 * BLOCK), Some(both));
        let asked = (queried(), protected());
        for _ in 0..3 {
            assert_eq!(
                guard(p, p + 2 * BLOCK),
                Some(pages(1).collect()),
                "not committed, but no placeholder"
            );
        }
        assert_eq!((queried(), protected()), asked, "asked about once");
        // The title commits it and writes there, with no mapping.
        // SAFETY: our reservation; one byte of the committed block.
        unsafe {
            let at = VirtualAlloc((p + BLOCK) as *const _, BLOCK, MEM_COMMIT, PAGE_READWRITE);
            assert_eq!(at as usize, p + BLOCK);
            ((p + BLOCK + 8) as *mut u8).write_volatile(1);
        }
        // A commit xmem did not see is still reported, not guarded.
        assert_eq!(guard(p, p + 2 * BLOCK), Some(pages(1).collect()));
        assert!(writable(p + BLOCK));
        // xmem reports the commit.
        title_memory_remapped(p + BLOCK, BLOCK);
        assert_eq!(guard(p, p + 2 * BLOCK), Some(pages(1).collect()));
        assert_eq!(guard(p, p + 2 * BLOCK), Some(Vec::new()));
        // SAFETY: the committed block, now guarded: into our handler.
        unsafe { ((p + BLOCK + 8) as *mut u8).write_volatile(2) };
        assert_eq!(guard(p, p + 2 * BLOCK), Some(pages(1).collect()));
        view_and_reservation_gone(p, view, section);
    }

    /// A range from a view into a reservation the title placed itself (not
    /// write-watched): it looks like a placeholder, so it is noted absent.
    /// The title commits it through `XMemVirtualAlloc`, which reports it
    /// (before, nothing did, and its writes there were never seen).
    #[test]
    fn reservations_the_title_placed_are_seen_once_committed() {
        let _serial = serial();
        let section = section(1);
        let (p, view) = view_then_reservation(section, false);
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let both: Vec<usize> = (0..2).flat_map(pages).collect();
        assert_eq!(guard(p, p + 2 * BLOCK), Some(both));
        assert_eq!(guard(p, p + 2 * BLOCK), Some(Vec::new()), "absent");
        // SAFETY: our reservation; one byte of the committed block.
        unsafe {
            let at = VirtualAlloc((p + BLOCK) as *const _, BLOCK, MEM_COMMIT, PAGE_READWRITE);
            assert_eq!(at as usize, p + BLOCK);
            ((p + BLOCK + 8) as *mut u8).write_volatile(1);
        }
        title_memory_remapped(p + BLOCK, BLOCK);
        assert_eq!(guard(p, p + 2 * BLOCK), Some(pages(1).collect()));
        assert_eq!(guard(p, p + 2 * BLOCK), Some(Vec::new()));
        // SAFETY: the committed block, now guarded: into our handler.
        unsafe { ((p + BLOCK + 8) as *mut u8).write_volatile(2) };
        assert_eq!(guard(p, p + 2 * BLOCK), Some(pages(1).collect()));
        view_and_reservation_gone(p, view, section);
    }

    /// xmem unmaps a view as it does now: reported, unmapped, reported
    /// again, with a copy guarding the view between the two reports (which
    /// notes its start block a view, and its blocks guarded and clean).
    /// Afterwards nothing of it is noted: the next asking finds no view
    /// there. Then the title frees the placeholder (reported) and the
    /// address is reused as ordinary memory: still no view, and writable.
    /// Without the report after the unmapping, the asking hit the stale
    /// view, reported nothing of the blocks gone (the copy kept the view's
    /// bytes), and once the address was reused made ordinary memory
    /// read-only, where system calls writing fail.
    #[test]
    fn unmapping_forgets_a_view_guarded_meanwhile_before_reuse() {
        use windows_sys::Win32::System::Memory::{MEM_PRESERVE_PLACEHOLDER, UnmapViewOfFile2};
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let _serial = serial();
        let section = section(2);
        let p = (0..8)
            .find_map(|_| {
                let p = placeholder_of(2);
                let all: Vec<usize> = (p / PAGE..p / PAGE + 2).collect();
                let view = map_view(section, p, 0, 2);
                assert_eq!(guard(p, p + 2 * BLOCK), Some(all.clone()));
                // xmem's unmapping, a copy guarding the view in between.
                title_memory_remapped(p, 2 * BLOCK);
                assert_eq!(guard(p, p + 2 * BLOCK), Some(all), "opened: reported");
                assert!(!writable(p) && !writable(p + BLOCK), "guarded again");
                // SAFETY: our view, back to a placeholder.
                unsafe { UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER) };
                title_memory_remapped(p, 2 * BLOCK);
                assert!(!noted(p, 2 * BLOCK), "the view forgotten");
                assert_eq!(guard(p, p + 2 * BLOCK), None, "no view any more");
                // The title frees the placeholder; ordinary memory takes the
                // address, unless another thread took it in between.
                released(p, 2);
                // SAFETY: a plain allocation.
                let at = unsafe {
                    VirtualAlloc(
                        p as *const _,
                        2 * BLOCK,
                        MEM_RESERVE | MEM_COMMIT,
                        PAGE_READWRITE,
                    )
                } as usize;
                (at == p).then_some(p)
            })
            .expect("the freed address stays free");
        assert_eq!(guard(p, p + 2 * BLOCK), None, "ordinary memory");
        assert!(writable(p) && writable(p + BLOCK));
        // SAFETY: plain writes into our ordinary memory.
        unsafe { ((p + BLOCK + 8) as *mut u8).write_volatile(1) };
        released(p, 2);
        // SAFETY: our section.
        unsafe { windows_sys::Win32::Foundation::CloseHandle(section) };
    }

    /// A range from a view into a placeholder notes the placeholder absent.
    /// The title frees the placeholder (xmem reports it) and ordinary
    /// memory takes the address, unreported (anyone may allocate it): the
    /// next asking reports it. Without the report of the free, it stayed
    /// absent and was never reported, whatever was written there.
    #[test]
    fn freed_placeholders_are_not_absent_once_reused() {
        let _serial = serial();
        let section = section(1);
        let (p, view) = (0..8)
            .find_map(|_| {
                let p = placeholder_of(2);
                let view = map_view(section, p, 0, 1);
                let pages: Vec<usize> = (p / PAGE..p / PAGE + 2).collect();
                assert_eq!(guard(p, p + 2 * BLOCK), Some(pages));
                assert_eq!(guard(p, p + 2 * BLOCK), Some(Vec::new()), "absent");
                released(p + BLOCK, 1);
                // SAFETY: a plain allocation, written.
                let at = unsafe {
                    let at = VirtualAlloc(
                        (p + BLOCK) as *const _,
                        BLOCK,
                        MEM_RESERVE | MEM_COMMIT,
                        PAGE_READWRITE,
                    ) as usize;
                    if at == p + BLOCK {
                        ((p + BLOCK + 8) as *mut u8).write_volatile(1);
                    }
                    at
                };
                if at == p + BLOCK {
                    return Some((p, view));
                }
                // Another thread took the address: try again.
                title_memory_remapped(p, BLOCK);
                // SAFETY: our view, back to free address space.
                unsafe {
                    windows_sys::Win32::System::Memory::UnmapViewOfFile2(
                        windows_sys::Win32::System::Threading::GetCurrentProcess(),
                        view,
                        0,
                    )
                };
                None
            })
            .expect("the freed address stays free");
        let reused: Vec<usize> = ((p + BLOCK) / PAGE..(p + 2 * BLOCK) / PAGE).collect();
        assert_eq!(guard(p, p + 2 * BLOCK), Some(reused));
        assert_eq!(guard(p, p + 2 * BLOCK), Some(Vec::new()));
        view_and_reservation_gone(p, view, section);
    }

    /// xmem unmaps the middle block of a guarded three-block view: the view
    /// reported, a copy guarding it again in between, unmapped, reported
    /// again, and its outer blocks mapped back (each reported). The next
    /// asking's run of the three blocks then spans two views and a
    /// placeholder: guarded block by block, the middle one noted absent,
    /// and all three reported (the middle one shows nothing now). Without
    /// the report after the unmapping, the middle block stayed guarded and
    /// clean, and was never reported again. A kernel write over the three
    /// then opens the outer two.
    #[test]
    fn a_run_over_a_block_unmapped_since_it_was_guarded() {
        use windows_sys::Win32::System::Memory::{MEM_PRESERVE_PLACEHOLDER, UnmapViewOfFile2};
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let _serial = serial();
        let section = section(3);
        let p = placeholder_of(3);
        let pages = |blocks: Range<usize>| -> Vec<usize> {
            (p / PAGE + blocks.start * BLOCK / PAGE..p / PAGE + blocks.end * BLOCK / PAGE).collect()
        };
        let whole = map_view(section, p, 0, 3);
        let before = protected();
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(0..3)));
        assert_eq!(protected() - before, 1, "one run, one change");
        // xmem's unmapping of block 1, a copy guarding the view in between.
        title_memory_remapped(p, 3 * BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(0..3)));
        // SAFETY: our view, back to a placeholder.
        unsafe { UnmapViewOfFile2(GetCurrentProcess(), whole, MEM_PRESERVE_PLACEHOLDER) };
        title_memory_remapped(p, 3 * BLOCK);
        let outer = [
            map_view(section, p, 0, 1),
            map_view(section, p + 2 * BLOCK, 2, 1),
        ];
        let before = protected();
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(0..3)));
        assert_eq!(
            protected() - before,
            2,
            "each mapped block, skipping unmapped"
        );
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        assert!(!writable(p) && !writable(p + 2 * BLOCK));
        {
            let g = guards();
            let middle = p / BLOCK + 1;
            assert!(g.absent.contains(&middle) && !g.protected.contains(&middle));
        }
        writing(p, 3 * BLOCK);
        assert!(writable(p) && writable(p + 2 * BLOCK));
        assert_eq!(
            guard(p, p + 3 * BLOCK),
            Some([pages(0..1), pages(2..3)].concat())
        );
        // As xmem unmaps and the title frees: reported before and after.
        title_memory_remapped(p, 3 * BLOCK);
        for view in outer {
            // SAFETY: our views, back to placeholders.
            unsafe { UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER) };
        }
        title_memory_remapped(p, 3 * BLOCK);
        for block in 0..3 {
            released(p + block * BLOCK, 1);
        }
        // SAFETY: our section.
        unsafe { windows_sys::Win32::Foundation::CloseHandle(section) };
    }

    /// Writers fault into the handler on guarded blocks while a copier
    /// guards them again and again and copies what is reported, and the
    /// runtime reports some of them remapped meanwhile. Every fault is
    /// taken (else the process ends) and no write is lost: once all is
    /// quiet, the copy is the view.
    #[test]
    fn the_handler_races_guards_and_remap_reports() {
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
        const BLOCKS: usize = 16;
        const WRITERS: usize = 4;
        const REFRESHES: usize = 200;
        /// A word of the view, atomic: other threads write and read it.
        fn word(at: usize) -> &'static AtomicU64 {
            // SAFETY: an aligned word of the test's view, used only while
            // it is mapped.
            unsafe { AtomicU64::from_ptr(at as *mut u64) }
        }
        let _serial = serial();
        let view = mapped(BLOCKS);
        let len = BLOCKS * BLOCK;
        // What a copy does: asks, then copies the pages reported.
        let refresh = |copy: &mut [u64]| {
            for page in guard(view, view + len).expect("a view") {
                let from = (page * PAGE - view) / 8;
                for (i, w) in copy[from..from + PAGE / 8].iter_mut().enumerate() {
                    *w = word(view + (from + i) * 8).load(Relaxed);
                }
            }
        };
        let mut copy = vec![0u64; len / 8];
        // Guarded from the start: the writers' first writes fault too.
        refresh(&mut copy);
        let refreshed = AtomicUsize::new(0);
        let quiet = AtomicBool::new(false);
        std::thread::scope(|s| {
            for w in 0..WRITERS {
                let refreshed = &refreshed;
                s.spawn(move || {
                    let mut k = 0;
                    // As long as the copier copies, and 1 000 writes at least.
                    while k < 1_000 || refreshed.load(Relaxed) < REFRESHES {
                        // Over every block, each writer its own words.
                        let block = (k * 7 + w) % BLOCKS;
                        let at = view + block * BLOCK + (k % 512 * WRITERS + w) * 8;
                        word(at).store(((w as u64) << 32) + k as u64 + 1, Relaxed);
                        k += 1;
                    }
                });
            }
            s.spawn(|| {
                let mut k = 0;
                while !quiet.load(Relaxed) {
                    // One to four blocks remapped.
                    let from = k % BLOCKS;
                    let n = (k % 4 + 1).min(BLOCKS - from);
                    title_memory_remapped(view + from * BLOCK, n * BLOCK);
                    k += 1;
                    std::thread::yield_now();
                }
            });
            for _ in 0..REFRESHES {
                refresh(&mut copy);
                refreshed.fetch_add(1, Relaxed);
            }
            quiet.store(true, Relaxed);
        });
        refresh(&mut copy);
        let seen: Vec<u64> = (0..len / 8)
            .map(|i| word(view + i * 8).load(Relaxed))
            .collect();
        assert!(seen.iter().any(|&w| w != 0), "written");
        assert!(seen == copy, "a write was lost");
        unmapped(view, BLOCKS);
    }

    /// What guarding view memory costs in a world frame: 30 submissions
    /// asking the guards about 35 times each (gpu-measure.md §7: 750 to
    /// 1 500 askings a frame), over a 174 MB heap of physical-page views,
    /// with what they report copied as refreshes copy it (through the
    /// kernel, unmapped pages by the fallback). The heap is one view, then
    /// one view per block with every 16th block an unmapped placeholder
    /// (11 MB; run1 read 55 to 240 MB a frame through the fallback). Each
    /// submission asks about the whole heap and 34 ranges of 64 KiB to 2 MiB
    /// in it, after the title wrote 8 of its blocks. Then a kernel write
    /// over the whole guarded view (`writing`), and `writing` elsewhere from
    /// 16 threads, with the heap guarded and with nothing guarded. Run with
    /// `cargo test --release -p xtp-gpu -- --ignored --nocapture guard_costs`.
    #[test]
    #[ignore = "benchmark"]
    fn guard_costs() {
        use std::time::{Duration, Instant};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, MapViewOfFile3,
            PAGE_NOACCESS, UnmapViewOfFile, UnmapViewOfFile2, VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let _serial = serial();
        const LEN: usize = 174 << 20;
        const BLOCKS: usize = LEN / BLOCK;
        const SUBMISSIONS: usize = 30;
        const RANGES: usize = 34;
        const FRAMES: usize = 4;
        // SAFETY: a pagefile section of the heap's size.
        let section = unsafe {
            CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                LEN as u32,
                std::ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: the section, mapped whole.
        let whole = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, LEN) };
        assert!(!whole.Value.is_null());
        let gap = |block: usize| block % 16 == 15;
        // SAFETY: a fresh placeholder of the heap's size.
        let paged = unsafe {
            VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                LEN,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            )
        } as usize;
        assert_ne!(paged, 0);
        let views: Vec<MEMORY_MAPPED_VIEW_ADDRESS> = (0..BLOCKS)
            .filter(|&b| !gap(b))
            .map(|block| {
                let at = paged + block * BLOCK;
                // SAFETY: splits the placeholder at the block (failing
                // harmlessly for its last block) and maps the section's
                // block there.
                let view = unsafe {
                    VirtualFree(at as *mut _, BLOCK, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
                    MapViewOfFile3(
                        section,
                        GetCurrentProcess(),
                        at as *const _,
                        (block * BLOCK) as u64,
                        BLOCK,
                        MEM_REPLACE_PLACEHOLDER,
                        PAGE_READWRITE,
                        std::ptr::null_mut(),
                        0,
                    )
                };
                assert_eq!(view.Value as usize, at);
                view
            })
            .collect();
        /// Copies pages `pages` of the heap at `heap` into `copy` through
        /// the kernel, as a refresh does.
        fn copy_run(heap: usize, pages: std::ops::Range<usize>, copy: &mut [u8]) -> usize {
            let (at, n) = ((pages.start - heap / PAGE) * PAGE, pages.len() * PAGE);
            let from = (pages.start * PAGE) as *const u8;
            crate::host::copy_mapped(copy[at..].as_mut_ptr(), from, n, false);
            n
        }
        /// The pages of `pages` written since their copy, copied in runs,
        /// as `Shadow::copy_stale` does.
        fn copy_stale(
            heap: usize,
            pages: std::ops::Range<usize>,
            now: u64,
            copied: &mut [u64],
            copy: &mut [u8],
        ) -> usize {
            let base = heap / PAGE;
            let (mut bytes, mut run) = (0, None);
            for page in pages.clone() {
                let write = last_write(page);
                let at = &mut copied[page - base];
                if *at == 0 || write > *at {
                    *at = now;
                    run.get_or_insert(page);
                } else if let Some(first) = run.take() {
                    bytes += copy_run(heap, first..page, copy);
                }
            }
            if let Some(first) = run {
                bytes += copy_run(heap, first..pages.end, copy);
            }
            bytes
        }
        let mut copy = vec![0u8; LEN];
        // One frame: time asking, time copying, bytes copied.
        let frame = |heap: usize, f: usize, copied: &mut [u64], copy: &mut [u8]| {
            let (mut asking, mut copying, mut bytes) = (Duration::ZERO, Duration::ZERO, 0);
            for s in 0..SUBMISSIONS {
                for k in 0..8 {
                    let block = ((f * SUBMISSIONS + s) * 8 + k) * 7_919 % BLOCKS;
                    let block = if gap(block) { block - 1 } else { block };
                    // SAFETY: a mapped block of the heap; guarded, the write
                    // goes through our handler.
                    unsafe { ((heap + block * BLOCK + 64 * k) as *mut u8).write_volatile(s as u8) };
                }
                let ranges = std::iter::once((0, BLOCKS)).chain((0..RANGES).map(|r| {
                    let n = 1 << (r % 6);
                    let block = (s * RANGES + r) * 104_729 % (BLOCKS - n);
                    (if gap(block) { block - 1 } else { block }, n)
                }));
                for (block, n) in ranges {
                    let (from, to) = (heap + block * BLOCK, heap + (block + n) * BLOCK);
                    let started = Instant::now();
                    let now = writes(from, to, true).expect("a view");
                    asking += started.elapsed();
                    let started = Instant::now();
                    let pages = from / PAGE..to / PAGE;
                    bytes += copy_stale(heap, pages, now, copied, copy);
                    copying += started.elapsed();
                }
            }
            (asking, copying, bytes)
        };
        for (name, heap) in [
            ("one view", whole.Value as usize),
            ("a view per block, every 16th unmapped", paged),
        ] {
            let mut copied = vec![0u64; LEN / PAGE];
            // The first copies, which guard it all.
            frame(heap, 0, &mut copied, &mut copy);
            let times: Vec<_> = (1..=FRAMES)
                .map(|f| frame(heap, f, &mut copied, &mut copy))
                .collect();
            let asking = times.iter().map(|t| t.0).sum::<Duration>() / FRAMES as u32;
            let copying = times.iter().map(|t| t.1).sum::<Duration>() / FRAMES as u32;
            let mb = times.iter().map(|t| t.2).sum::<usize>() as f64 / FRAMES as f64 / 1e6;
            println!(
                "174 MB heap, {name}: a frame of {SUBMISSIONS} x {} askings: {asking:?} asking, \
                 {copying:?} copying {mb:.1} MB",
                RANGES + 1
            );
        }
        let heap = whole.Value as usize;
        let opened: Vec<Duration> = (0..10)
            .map(|_| {
                // Guarded whole again.
                writes(heap, heap + LEN, true).expect("a view");
                let started = Instant::now();
                writing(heap, LEN);
                started.elapsed()
            })
            .collect();
        println!(
            "a kernel write over the guarded 174 MB view: {:?}",
            opened.iter().sum::<Duration>() / 10
        );
        const THREADS: usize = 16;
        const CALLS: usize = 100_000;
        let plain = vec![0u8; PAGE];
        let elsewhere = |what: &str| {
            let at = plain.as_ptr() as usize;
            let started = Instant::now();
            std::thread::scope(|s| {
                for _ in 0..THREADS {
                    s.spawn(move || {
                        for _ in 0..CALLS {
                            writing(std::hint::black_box(at), PAGE);
                        }
                    });
                }
            });
            println!(
                "a kernel write elsewhere, {THREADS} threads, {what}: {:.1} ns per call",
                started.elapsed().as_secs_f64() * 1e9 / (THREADS * CALLS) as f64
            );
        };
        writes(heap, heap + LEN, true).expect("a view");
        elsewhere("the heap guarded");
        title_memory_remapped(heap, LEN);
        title_memory_remapped(paged, LEN);
        elsewhere("nothing guarded");
        // SAFETY: our views, the placeholder's blocks and the section.
        unsafe {
            UnmapViewOfFile(whole);
            for view in views {
                UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER);
            }
            for block in 0..BLOCKS {
                VirtualFree((paged + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }

    /// What other views of the pages cost (the VIctory #1122 port): a copy
    /// guarding 64 blocks of one view, the title then writing each block
    /// once (each write a fault into the handler), and a kernel write over
    /// them all. First with the pages in that view alone, then with each
    /// page in a second view of its own as well (as the GPU's page mappings
    /// make them), the title writing through those. Medians of 15 rounds,
    /// per block. Run with
    /// `cargo test --release -p xtp-gpu -- --ignored --nocapture alias_costs`.
    #[test]
    #[ignore = "benchmark"]
    fn alias_costs() {
        use std::time::{Duration, Instant};
        use windows_sys::Win32::System::Memory::{MEM_PRESERVE_PLACEHOLDER, UnmapViewOfFile2};
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        const BLOCKS: usize = 64;
        let _serial = serial();
        let section = section(BLOCKS);
        let a = view_of(section, BLOCKS);
        let b = placeholder_of(BLOCKS);
        let pages: Vec<_> = (0..BLOCKS)
            .map(|k| map_view(section, b + k * BLOCK, k, 1))
            .collect();
        let median = |mut t: Vec<Duration>| {
            t.sort();
            t[t.len() / 2] / BLOCKS as u32
        };
        for (name, through) in [("one view", a), ("a second view per page", b)] {
            if through == b {
                same_pages(&[a, b], BLOCKS);
            }
            let (mut guarding, mut faults, mut kernel) = (Vec::new(), Vec::new(), Vec::new());
            for round in 0..16 {
                let started = Instant::now();
                guard(a, a + BLOCKS * BLOCK);
                let guarded = started.elapsed();
                let started = Instant::now();
                for k in 0..BLOCKS {
                    poke(through + k * BLOCK + 8, round as u8);
                }
                let written = started.elapsed();
                guard(a, a + BLOCKS * BLOCK);
                let started = Instant::now();
                writing(through, BLOCKS * BLOCK);
                let opened = started.elapsed();
                // The first round guards blocks never guarded: left out.
                if round > 0 {
                    guarding.push(guarded);
                    faults.push(written);
                    kernel.push(opened);
                }
            }
            println!(
                "{BLOCKS} blocks, {name}: guarded again {:?}, a write fault {:?}, a kernel \
                 write {:?} (per block)",
                median(guarding),
                median(faults),
                median(kernel)
            );
        }
        unmapped(a, BLOCKS);
        title_memory_remapped(b, BLOCKS * BLOCK);
        for view in pages {
            // SAFETY: our views, back to placeholders.
            unsafe { UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER) };
        }
        not_same(b, BLOCKS);
        title_memory_remapped(b, BLOCKS * BLOCK);
        for k in 0..BLOCKS {
            released(b + k * BLOCK, 1);
        }
        // SAFETY: our section.
        unsafe { windows_sys::Win32::Foundation::CloseHandle(section) };
    }

    /// A kernel read into view memory while a copy keeps asking about it
    /// (and so guards again what the read's announcement opened): every read
    /// lands, and its data is there. Before the read's blocks were held open
    /// until it ended, a copy could make a block read-only in the middle of
    /// the read, and the kernel failed it (ERROR_INVALID_USER_BUFFER when
    /// the block turned read-only during the read, ERROR_NOACCESS before):
    /// DirectStorage's failed reads in session mesh64-a
    /// (docs/analysis/archive-null.md).
    #[test]
    fn a_kernel_read_is_not_cut_by_a_copy_guarding_its_blocks() {
        use std::io::Write;
        use std::os::windows::fs::FileExt;
        use std::sync::atomic::AtomicBool;
        const BLOCKS: usize = 32;
        const READS: usize = 1000;
        let _serial = serial();
        let view = mapped(BLOCKS);
        let len = BLOCKS * BLOCK;
        let path = std::env::temp_dir().join(format!("xtp-watch-race-{}.bin", std::process::id()));
        let data: Vec<u8> = (0..len).map(|i| (i * 7 + i / 4096) as u8).collect();
        std::fs::File::create(&path)
            .and_then(|mut f| f.write_all(&data))
            .unwrap();
        let file = std::fs::File::open(&path).unwrap();
        // Something guarded, as in the world.
        guard(view, view + len);
        let stop = AtomicBool::new(false);
        let (mut failed, mut errors) = (0, Vec::new());
        std::thread::scope(|s| {
            s.spawn(|| {
                while !stop.load(SeqCst) {
                    guard(view, view + len);
                }
            });
            for _ in 0..READS {
                let writing = kernel_writing(view, len);
                // SAFETY: our view, `len` bytes.
                let out = unsafe { std::slice::from_raw_parts_mut(view as *mut u8, len) };
                let read = file.seek_read(out, 0);
                drop(writing);
                match read {
                    Ok(n) if n == len => {}
                    Ok(n) => errors.push(format!("short read: {n}")),
                    Err(e) => {
                        failed += 1;
                        if errors.len() < 4 {
                            errors.push(e.to_string());
                        }
                    }
                }
            }
            stop.store(true, SeqCst);
        });
        drop(file);
        let _ = std::fs::remove_file(&path);
        assert_eq!(failed, 0, "reads failed: {errors:?}");
        // SAFETY: our view, readable.
        let now = unsafe { std::slice::from_raw_parts(view as *const u8, len) };
        assert!(now == &data[..], "the data landed");
        // The read is seen by the next copy: every page written.
        guard(view, view + len);
        drop(kernel_writing(view, len));
        let written = guard(view, view + len).unwrap();
        assert_eq!(written.len(), len / PAGE);
        unmapped(view, BLOCKS);
    }

    /// A kernel write into write-watched memory is noted written once it is
    /// done, after an asking made while it was under way: a device writing
    /// into pages locked when the read began (unbuffered or overlapped
    /// reads) leaves no trace for a copy that asked in between, and copies
    /// are taken only of pages known written.
    #[test]
    fn a_kernel_write_is_noted_written_when_it_is_done() {
        let _serial = serial();
        // SAFETY: a fresh write-watched allocation, released below.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 4 * PAGE);
        writes(start, end, false).expect("watched");
        let writing = kernel_writing(p + PAGE, 2 * PAGE);
        // A copy asks while the device writes: nothing to see yet.
        let during = writes(start, end, false).unwrap();
        drop(writing);
        let after = writes(start, end, false).unwrap();
        assert!(after > during);
        let noted: Vec<_> = (start / PAGE..end / PAGE)
            .filter(|&page| last_write(page) > during)
            .collect();
        assert_eq!(noted, [p / PAGE + 1, p / PAGE + 2]);
        // An announcement outside write-watched memory notes nothing.
        let view = mapped(1);
        let before = last_write(view / PAGE);
        drop(kernel_writing(view, PAGE));
        assert_eq!(last_write(view / PAGE), before);
        unmapped(view, 1);
        // SAFETY: our allocation.
        assert_ne!(unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) }, 0);
    }

    /// Whether one protection change over a run of blocks costs less than
    /// one per block (F5 merges `open`'s), for a 4 MiB buffer and the
    /// 174 MB heap, its pages present, read-only to read-write as `open`
    /// does: medians of 15 rounds, the two ways alternated. Run with
    /// `cargo test --release -p xtp-gpu -- --ignored --nocapture protection_change_costs`.
    #[test]
    #[ignore = "benchmark"]
    fn protection_change_costs() {
        use std::time::{Duration, Instant};
        use windows_sys::Win32::System::Memory::{PAGE_READONLY, PAGE_READWRITE, VirtualProtect};
        let set = |at: usize, n: usize, protection: u32| {
            let mut old = 0;
            // SAFETY: our view; only its protection changes.
            let set = unsafe { VirtualProtect(at as *const _, n * BLOCK, protection, &mut old) };
            assert_ne!(set, 0);
        };
        for blocks in [64, (174 << 20) / BLOCK] {
            let view = mapped(blocks);
            // SAFETY: our view: every page made present.
            unsafe { std::ptr::write_bytes(view as *mut u8, 1, blocks * BLOCK) };
            let (mut merged, mut each) = (Vec::new(), Vec::new());
            for round in 0..15 {
                for way in [round % 2, 1 - round % 2] {
                    set(view, blocks, PAGE_READONLY);
                    let started = Instant::now();
                    if way == 0 {
                        set(view, blocks, PAGE_READWRITE);
                    } else {
                        for b in 0..blocks {
                            set(view + b * BLOCK, 1, PAGE_READWRITE);
                        }
                    }
                    [&mut merged, &mut each][way].push(started.elapsed());
                }
            }
            let median = |t: &mut Vec<Duration>| {
                t.sort();
                t[t.len() / 2]
            };
            println!(
                "{} MiB read-only to read-write: {:?} in one change, {:?} block by block",
                (blocks * BLOCK) >> 20,
                median(&mut merged),
                median(&mut each)
            );
            unmapped(view, blocks);
        }
    }

    const KERNEL_PAGE: usize = 4096;

    /// The protection of the page at `at`, `PAGE_GUARD` included.
    fn protection(at: usize) -> u32 {
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        assert_ne!(n, 0);
        info.Protect
    }

    /// Sets `PAGE_GUARD` on the page at `at`, as the title does.
    fn title_sets_guard(at: usize) {
        use windows_sys::Win32::System::Memory::{PAGE_GUARD, VirtualProtect};
        let mut old = 0;
        // SAFETY: a page of one of our views.
        let set = unsafe {
            VirtualProtect(
                at as *const _,
                KERNEL_PAGE,
                PAGE_READWRITE | PAGE_GUARD,
                &mut old,
            )
        };
        assert_ne!(set, 0);
    }

    /// A page the title guards itself (`PAGE_GUARD`) inside a range that is
    /// taken keeps its guard: its block is left as it is, and the blocks
    /// around it are made read-only.
    #[test]
    fn a_page_the_title_guards_inside_a_range_keeps_its_guard() {
        use windows_sys::Win32::System::Memory::{PAGE_GUARD, PAGE_READONLY};
        let _serial = serial();
        const BLOCKS: usize = 4;
        const SIZE: usize = BLOCKS * BLOCK;
        let section = section(BLOCKS);
        let one = view_of(section, BLOCKS);
        let guarded = one + BLOCK + KERNEL_PAGE;
        title_sets_guard(guarded);
        assert_eq!(guard(one, one + SIZE).map(|written| written.len()), Some(4));
        assert_eq!(
            protection(guarded),
            PAGE_READWRITE | PAGE_GUARD,
            "the title's guard is gone"
        );
        assert_eq!(
            protection(one + BLOCK),
            PAGE_READWRITE,
            "its block is left as it was"
        );
        for block in [0, 2, 3] {
            assert_eq!(
                protection(one + block * BLOCK),
                PAGE_READONLY,
                "block {block}"
            );
        }
        // Not made read-only, so it is reported as written at every asking.
        assert!(guard(one, one + SIZE).is_some_and(|written| written.contains(&(one / PAGE + 1))));
        writing(one, SIZE);
        assert_eq!(
            protection(guarded),
            PAGE_READWRITE | PAGE_GUARD,
            "nor when opened"
        );
        unmapped(one, BLOCKS);
        // SAFETY: our section handle.
        unsafe { windows_sys::Win32::Foundation::CloseHandle(section) };
    }

    /// A page the title guards itself inside a block of another view of
    /// guarded pages keeps its guard: that block is not one plain region (the
    /// guard ends it), so it is left as it is; the blocks around it are made
    /// read-only and the guarded view's block stays open.
    #[test]
    fn a_page_the_title_guards_in_another_view_keeps_its_guard() {
        use windows_sys::Win32::System::Memory::{PAGE_GUARD, PAGE_READONLY};
        let _serial = serial();
        const BLOCKS: usize = 4;
        const SIZE: usize = BLOCKS * BLOCK;
        let section = section(BLOCKS);
        let (guarded, other) = (view_of(section, BLOCKS), view_of(section, BLOCKS));
        same_pages(&[guarded, other], BLOCKS);
        let page = other + BLOCK + KERNEL_PAGE;
        title_sets_guard(page);
        assert!(guard(guarded, guarded + SIZE).is_some());
        assert_eq!(
            protection(page),
            PAGE_READWRITE | PAGE_GUARD,
            "the title's guard is gone"
        );
        assert_eq!(
            protection(other + BLOCK),
            PAGE_READWRITE,
            "its block is left as it was"
        );
        for block in [0, 2, 3] {
            assert_eq!(
                protection(other + block * BLOCK),
                PAGE_READONLY,
                "block {block}"
            );
        }
        // A write through the other view would not fault while it holds a
        // guard, so the guarded block stays open and is reported at every asking.
        let again = guard(guarded, guarded + SIZE).unwrap();
        assert!(
            again.contains(&(guarded / PAGE + 1)),
            "reported as written at every asking"
        );
        unmapped(guarded, BLOCKS);
        unmapped(other, BLOCKS);
        // SAFETY: our section handle.
        unsafe { windows_sys::Win32::Foundation::CloseHandle(section) };
    }

    #[test]
    fn poll_merged_coalesces_disjoint_and_adjacent_write_watched_pages() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        let _serial = serial();
        let alloc = |pages: usize| -> usize {
            // SAFETY: plain private write-watched reservation/commit for the test.
            let p = unsafe {
                VirtualAlloc(
                    std::ptr::null(),
                    pages * PAGE,
                    MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                    PAGE_READWRITE,
                )
            };
            assert!(!p.is_null());
            p as usize
        };
        let a = alloc(4);
        let b = alloc(3);
        let base_a = a / PAGE;
        let base_b = b / PAGE;
        // Clear initial allocation dirty state, then touch page 0 & 2 of `a` and page 1 of `b`.
        let _ = poll_merged(&[(base_a, base_a + 4), (base_b, base_b + 3)]);
        // SAFETY: committed pages inside `a` and `b`.
        unsafe {
            *(a as *mut u8) = 1;
            *((a + 2 * PAGE) as *mut u8) = 2;
            *((b + PAGE) as *mut u8) = 3;
        }
        let (now, unwatched) = poll_merged(&[
            (base_a, base_a + 2),
            (base_a + 2, base_a + 4),
            (base_b, base_b + 3),
        ]);
        assert!(unwatched.is_empty());
        assert_eq!(last_write(base_a), now);
        assert!(last_write(base_a + 1) < now);
        assert_eq!(last_write(base_a + 2), now);
        assert!(last_write(base_a + 3) < now);
        assert_eq!(last_write(base_b + 1), now);
        // SAFETY: allocations obtained from VirtualAlloc above.
        unsafe {
            assert_ne!(VirtualFree(a as *mut _, 0, MEM_RELEASE), 0);
            assert_ne!(VirtualFree(b as *mut _, 0, MEM_RELEASE), 0);
        }
    }
}
