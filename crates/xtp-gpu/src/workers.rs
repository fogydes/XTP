//! Scoped worker pool and hybrid CPU affinity.
//!
//! Submission preparation and copy-back scans share a small pool of reusable
//! helper threads. Set `XTP_WIDE=1` to enable parallel work and AVX2 comparisons.

use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::thread::JoinHandle;

/// Whether `XTP_WIDE=1` enables parallel submission and copy-back scans.
/// Unset or `0` keeps the original single-threaded paths.
pub(crate) fn wide() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| wide_enabled(std::env::var("XTP_WIDE").ok().as_deref()))
}

fn wide_enabled(setting: Option<&str>) -> bool {
    setting.is_some_and(|v| v.trim() == "1")
}

/// Total threads used by the shared pool (including the calling thread):
/// `available_parallelism / 4`, clamped to `2..=6`.
fn pool_threads() -> usize {
    let cpus = std::thread::available_parallelism().ok().map(|n| n.get());
    threads_for(cpus)
}

fn threads_for(logical_cpus: Option<usize>) -> usize {
    logical_cpus.map_or(2, |n| n / 4).clamp(2, 6)
}

/// A reusable scoped worker pool that dispatches `0..count` across its helper
/// threads and the calling thread with zero per-item heap allocation.
pub(crate) struct Pool {
    inner: Arc<Inner>,
    helpers: Vec<JoinHandle<()>>,
}

struct Inner {
    state: Mutex<State>,
    more: Condvar,
    idle: Condvar,
    /// Serializes batches so the single `State` slot stays scoped to one call.
    turn: Mutex<()>,
    threads: usize,
}

struct State {
    /// Closure currently running for `0..count`, cleared before `each` returns.
    work: Option<Work>,
    next: usize,
    count: usize,
    /// Helper threads currently inside `work`.
    running: usize,
    panic: Option<Box<dyn std::any::Any + Send>>,
    stop: bool,
}

#[derive(Clone, Copy)]
struct Work(*const (dyn Fn(usize) + Sync));

// SAFETY: the referenced closure is `Sync`, and `Pool::each` keeps it alive
// until `work` is cleared and every helper has left the call (`running == 0`).
unsafe impl Send for Work {}

impl Inner {
    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn help(&self) {
        let mut state = self.state();
        loop {
            if let Some((work, i)) = take(&mut state) {
                state.running += 1;
                drop(state);
                // SAFETY: `state.running` was incremented while `work` was
                // live, so `Pool::each` cannot return or drop the closure
                // until `state.running` reaches zero below.
                let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
                    (*work.0)(i)
                }));
                state = self.state();
                state.running -= 1;
                let extra = ran
                    .err()
                    .and_then(|payload| record_panic(&mut state, payload));
                if state.running == 0 {
                    self.idle.notify_all();
                }
                if let Some(extra) = extra {
                    drop(state);
                    discard_panic(extra);
                    state = self.state();
                }
                continue;
            }
            if state.stop {
                break;
            }
            state = self.more.wait(state).unwrap_or_else(|e| e.into_inner());
        }
    }
}

impl Pool {
    /// Spawns `threads.saturating_sub(1)` helper threads named `{name}-{i}`.
    pub(crate) fn new(name: &'static str, threads: usize) -> Pool {
        let threads = threads.max(1);
        let inner = Arc::new(Inner {
            state: Mutex::new(State {
                work: None,
                next: 0,
                count: 0,
                running: 0,
                panic: None,
                stop: false,
            }),
            more: Condvar::new(),
            idle: Condvar::new(),
            turn: Mutex::new(()),
            threads,
        });
        let mut helpers = Vec::with_capacity(threads - 1);
        for i in 1..threads {
            let shared = inner.clone();
            // A helper that fails to spawn is simply skipped; the remaining
            // threads (including the caller) still finish every item.
            if let Ok(handle) = std::thread::Builder::new()
                .name(format!("{name}-{i}"))
                .spawn(move || {
                    crate::cores::keep_many_on_fast_cores(name);
                    shared.help();
                })
            {
                helpers.push(handle);
            }
        }
        Pool { inner, helpers }
    }

    /// Calls `f` once for every index in `0..count`, spread across this pool's
    /// helper threads and the calling thread, and returns only once every call
    /// has completed.
    ///
    /// If any invocation panics, remaining unstarted items are skipped, all
    /// in-flight helpers are waited for before unwinding, and the panic is
    /// propagated to the caller without deadlocking or poisoning the pool.
    pub(crate) fn each(&self, count: usize, f: &(dyn Fn(usize) + Sync)) {
        if count < 2 || self.inner.threads < 2 || self.helpers.is_empty() {
            (0..count).for_each(f);
            return;
        }
        let turn = match self.inner.turn.try_lock() {
            Ok(turn) => turn,
            Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
            // A nested or concurrent caller must not wait while holding
            // locks that an active worker could need.
            Err(std::sync::TryLockError::WouldBlock) => {
                (0..count).for_each(f);
                return;
            }
        };
        // SAFETY: erasing the closure's lifetime is sound because `state.work`
        // is set back to `None` and `state.running` is drained to zero before
        // `each` returns or unwinds.
        let work = Work(unsafe {
            std::mem::transmute::<
                *const (dyn Fn(usize) + Sync + '_),
                *const (dyn Fn(usize) + Sync + 'static),
            >(f)
        });
        {
            let mut state = self.inner.state();
            state.work = Some(work);
            state.next = 0;
            state.count = count;
            debug_assert!(state.panic.is_none());
        }
        if count > self.inner.threads {
            self.inner.more.notify_all();
        } else {
            for _ in 1..count {
                self.inner.more.notify_one();
            }
        }
        let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            loop {
                let item = {
                    let mut state = self.inner.state();
                    take(&mut state)
                };
                let Some((_, i)) = item else { break };
                f(i);
            }
        }));
        let (panic, extra) = {
            let mut state = self.inner.state();
            let extra = ran
                .err()
                .and_then(|payload| record_panic(&mut state, payload));
            // Stop helpers from picking up any further items before draining.
            state.work = None;
            while state.running > 0 {
                state = self
                    .inner
                    .idle
                    .wait(state)
                    .unwrap_or_else(|e| e.into_inner());
            }
            (state.panic.take(), extra)
        };
        drop(turn);
        if let Some(extra) = extra {
            discard_panic(extra);
        }
        if let Some(panic) = panic {
            std::panic::resume_unwind(panic);
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        {
            let mut state = self.inner.state();
            state.stop = true;
            state.work = None;
        }
        self.inner.more.notify_all();
        for handle in self.helpers.drain(..) {
            let _ = handle.join();
        }
    }
}

/// Saves the first payload for the caller and returns any later one to drop
/// outside the pool's locks.
fn record_panic(
    state: &mut State,
    payload: Box<dyn std::any::Any + Send>,
) -> Option<Box<dyn std::any::Any + Send>> {
    state.work = None;
    if state.panic.is_none() {
        state.panic = Some(payload);
        None
    } else {
        Some(payload)
    }
}

fn discard_panic(payload: Box<dyn std::any::Any + Send>) {
    if let Err(during_drop) =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(payload)))
    {
        // Rust allows payloads with panicking destructors. A destructor may
        // produce another such payload; do not recurse or unwind a helper.
        std::mem::forget(during_drop);
    }
}

fn take(state: &mut State) -> Option<(Work, usize)> {
    let work = state.work?;
    if state.next >= state.count {
        return None;
    }
    let i = state.next;
    state.next += 1;
    Some((work, i))
}

/// Runs `f` over `0..count`; the caller has selected the wide path.
pub(crate) fn each_wide(count: usize, f: &(dyn Fn(usize) + Sync)) {
    if count < 2 {
        (0..count).for_each(f);
        return;
    }
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| Pool::new("xtp-gpu-wide", pool_threads()))
        .each(count, f);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn wide_switch_and_thread_count_clamp() {
        assert!(!wide_enabled(None));
        assert!(wide_enabled(Some("1")));
        assert!(!wide_enabled(Some("")));
        assert!(!wide_enabled(Some("0")));
        assert!(!wide_enabled(Some(" 0 ")));

        assert_eq!(threads_for(None), 2);
        assert_eq!(threads_for(Some(1)), 2);
        assert_eq!(threads_for(Some(4)), 2);
        assert_eq!(threads_for(Some(8)), 2);
        assert_eq!(threads_for(Some(12)), 3);
        assert_eq!(threads_for(Some(16)), 4);
        assert_eq!(threads_for(Some(24)), 6);
        assert_eq!(threads_for(Some(64)), 6);
    }

    #[test]
    fn pool_runs_every_item_once_under_contention() {
        for threads in [1usize, 2, 4, 6] {
            let pool = Pool::new("xtp-test-pool", threads);
            std::thread::scope(|s| {
                for _ in 0..3 {
                    s.spawn(|| {
                        for round in 0..200usize {
                            let count = (round * 13) % 128;
                            let calls: Vec<AtomicUsize> =
                                (0..count).map(|_| AtomicUsize::new(0)).collect();
                            pool.each(count, &|i| {
                                calls[i].fetch_add(1, Ordering::Relaxed);
                            });
                            assert!(calls.iter().all(|c| c.load(Ordering::Relaxed) == 1));
                        }
                    });
                }
            });
        }
    }

    #[test]
    fn pool_stress_and_panicking_job_does_not_deadlock() {
        let pool = Pool::new("xtp-test-panic", 4);
        // Many items across repeated batches.
        for round in 0..50usize {
            let count = 512 + round * 17;
            let hits: Vec<AtomicUsize> = (0..count).map(|_| AtomicUsize::new(0)).collect();
            pool.each(count, &|i| {
                hits[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
        }
        // A panicking job at any index (whether picked up by caller or helper)
        // reaches the caller and leaves the pool healthy for subsequent batches.
        for bad in 0..16usize {
            let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                pool.each(32, &|i| {
                    assert_ne!(i, bad, "expected test panic");
                });
            }));
            assert!(ran.is_err(), "panic at {bad} must propagate");

            let sum = AtomicUsize::new(0);
            pool.each(100, &|i| {
                sum.fetch_add(i, Ordering::Relaxed);
            });
            assert_eq!(
                sum.into_inner(),
                4950,
                "pool must stay usable after panic {bad}"
            );
        }
    }

    #[test]
    fn nested_and_concurrent_batches_finish_while_the_first_batch_is_active() {
        let pool = Pool::new("xtp-pool-nested", 2);
        let ready = std::sync::Barrier::new(2);
        let done = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                ready.wait();
                let sum = AtomicUsize::new(0);
                pool.each(32, &|i| {
                    sum.fetch_add(i, Ordering::Relaxed);
                });
                assert_eq!(sum.into_inner(), 496);
                done.wait();
            });
            pool.each(32, &|i| {
                if i == 0 {
                    ready.wait();
                    let sum = AtomicUsize::new(0);
                    pool.each(8, &|i| {
                        sum.fetch_add(i, Ordering::Relaxed);
                    });
                    assert_eq!(sum.into_inner(), 28);
                    done.wait();
                }
            });
        });
    }

    #[test]
    fn a_helper_panic_preserves_the_payload_and_finishes_the_scope() {
        let pool = Pool::new("xtp-pool-panic", 2);
        if pool.helpers.is_empty() {
            return;
        }
        let caller = std::thread::current().id();
        let both = std::sync::Barrier::new(2);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.each(2, &|_| {
                both.wait();
                if std::thread::current().id() != caller {
                    std::panic::panic_any(42u32);
                }
            });
        }));
        assert_eq!(
            result.unwrap_err().downcast::<u32>().as_deref().ok(),
            Some(&42)
        );
        let count = AtomicUsize::new(0);
        pool.each(100, &|_| {
            count.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(count.into_inner(), 100);
    }
}
