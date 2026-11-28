//! Host memory pressure, for the pass over the textures shaders pick by
//! index (`draw.rs` `heap_textures`). Each staging buffer a pass makes is
//! upload memory of its texture's size, committed beside the title's own:
//! with the PC's commit (what RAM and the page file can still back) nearly
//! spent, the next allocation of the title or of ours fails. Read at most
//! every [`SAMPLE_MS`]. Under pressure a pass stages fewer bytes
//! ([`Pressure::pass_budget`]) and looks at more textures for idle staging
//! buffers to release ([`Pressure::release_looks`]); below
//! [`STAGING_RESERVE`] of commit, or while texture heaps back off after a
//! refusal, it makes no new staging buffer ([`admits_staging`]): those
//! textures wait for their next turn.
//!
//! Ported from codec-d/cool_fox (GPL-3.0-or-later): `victory-gpu`
//! `residency.rs` (`sample_commit`, `next_pressure`, `critical_commit`,
//! `prefetch_limit`, `allow_staging_growth`, `refresh`) and `pool.rs`
//! (`room`). Changed here: one reading gives the pressure and the
//! admission (the fork reads physical memory once a second, commit every
//! 32 ms); XTP has no staging pool, each staging buffer is its own upload
//! buffer, so every new one is the fork's "growth" and its reserve is the
//! admission; the fork's spare pool blocks kept under pressure (one, none
//! when critical) become idle staging buffers found sooner; there are no
//! bakes, which the fork lets through on a smaller reserve.

use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

/// How often host memory is read, in milliseconds: a pass asks for each
/// texture, and may run for seconds while the world loads.
const SAMPLE_MS: u64 = 32;

/// Commit available below which pressure is critical, whatever came before.
const CRITICAL_COMMIT: u64 = 1536 * MIB;

/// Commit available below which a pass makes no new staging buffer: room
/// for the title's own allocations and the submissions in flight.
const STAGING_RESERVE: u64 = GIB;

/// How hard the host's memory is pressed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Pressure {
    None,
    /// From 3 GiB of physical memory or 4 GiB of commit available down,
    /// until both are back to 5 and 6 GiB.
    Mild,
    /// Under [`CRITICAL_COMMIT`] available, or an allocation refused lately.
    Critical,
}

impl Pressure {
    /// The bytes a pass may stage of its `budget`
    /// (`descriptor::host::heap_textures_budget`): at most 64 MiB under
    /// mild pressure, 16 MiB under critical. The first texture of a pass
    /// fits whatever its bytes all the same (`staging_room`).
    pub(crate) fn pass_budget(self, budget: u64) -> u64 {
        match self {
            Pressure::None => budget,
            Pressure::Mild => budget.min(64 * MIB),
            Pressure::Critical => budget.min(16 * MIB),
        }
    }

    /// The textures a pass looks at for idle staging buffers to release
    /// (`texture::release_staging`), of its own `looks`: four times as many
    /// under mild pressure, all of them under critical. What counts as idle
    /// stays: a texture sampled at every frame keeps its buffer.
    pub(crate) fn release_looks(self, looks: usize) -> usize {
        match self {
            Pressure::None => looks,
            Pressure::Mild => looks.saturating_mul(4),
            Pressure::Critical => usize::MAX,
        }
    }

    fn from_u8(value: u8) -> Pressure {
        match value {
            0 => Pressure::None,
            1 => Pressure::Mild,
            _ => Pressure::Critical,
        }
    }
}

/// The pressure with `commit` bytes of commit and `physical` bytes of
/// physical memory available, after `previous`; `recent_oom`: an
/// allocation was refused lately. Mild pressure starts at 3 GiB of physical
/// memory or 4 GiB of commit and ends once there are 5 and 6 GiB, so that
/// readings near the edge do not change a pass's budget at every turn.
pub(crate) fn classify(
    commit: u64,
    physical: u64,
    previous: Pressure,
    recent_oom: bool,
) -> Pressure {
    if recent_oom || commit < CRITICAL_COMMIT {
        return Pressure::Critical;
    }
    let pressed = if previous == Pressure::None {
        physical <= 3 * GIB || commit <= 4 * GIB
    } else {
        physical < 5 * GIB || commit < 6 * GIB
    };
    if pressed {
        Pressure::Mild
    } else {
        Pressure::None
    }
}

/// Whether a pass may make new staging buffers with `commit` bytes of
/// commit available: not below [`STAGING_RESERVE`], nor after an allocation
/// was refused lately (`recent_oom`).
fn admits(commit: u64, recent_oom: bool) -> bool {
    !recent_oom && commit >= STAGING_RESERVE
}

/// The host's memory pressure now.
pub(crate) fn pressure() -> Pressure {
    SAMPLER.at(super::now_ms(), read).0
}

/// Whether a pass may make new staging buffers now.
pub(crate) fn admits_staging() -> bool {
    SAMPLER.at(super::now_ms(), read).1
}

/// A reading of the host's memory, in bytes available.
#[derive(Clone, Copy)]
struct Reading {
    commit: u64,
    physical: u64,
    /// Texture heaps back off after the driver refused one
    /// (`resource::pool`, `HEAP_FAIL_BACKOFF`).
    recent_oom: bool,
}

/// Reads the host's memory. `None` in tests: the passes tests elsewhere run
/// stage what they count, whatever memory the machine running them has.
fn read() -> Option<Reading> {
    if cfg!(test) {
        return None;
    }
    let (physical, commit) = super::available_memory()?;
    let recent_oom = crate::resource::pool::global_if_made().is_some_and(|p| p.backing_off());
    Some(Reading {
        commit,
        physical,
        recent_oom,
    })
}

static SAMPLER: Sampler = Sampler::new();

/// What the last reading of the host's memory says.
struct Sampler {
    /// When the next reading is due ([`super::now_ms`]).
    next: AtomicU64,
    /// The pressure, as [`Pressure::from_u8`] reads it.
    pressure: AtomicU8,
    /// Whether new staging buffers are admitted ([`admits`]).
    admits: AtomicBool,
}

impl Sampler {
    const fn new() -> Sampler {
        Sampler {
            next: AtomicU64::new(0),
            pressure: AtomicU8::new(Pressure::None as u8),
            admits: AtomicBool::new(true),
        }
    }

    /// The pressure and the admission at `now` (ms), reading the memory
    /// with `read` once [`SAMPLE_MS`] passed since the last reading: one
    /// thread reads, the others take what the last one said. A failed
    /// reading keeps it.
    fn at(&self, now: u64, read: impl FnOnce() -> Option<Reading>) -> (Pressure, bool) {
        let next = self.next.load(Ordering::Relaxed);
        if now >= next
            && self
                .next
                .compare_exchange(
                    next,
                    now.saturating_add(SAMPLE_MS),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
                .is_ok()
            && let Some(r) = read()
        {
            let was = Pressure::from_u8(self.pressure.load(Ordering::Relaxed));
            let pressure = classify(r.commit, r.physical, was, r.recent_oom);
            self.pressure.store(pressure as u8, Ordering::Relaxed);
            self.admits
                .store(admits(r.commit, r.recent_oom), Ordering::Relaxed);
            if pressure != was {
                note_change(was, pressure, r);
            }
        }
        (
            Pressure::from_u8(self.pressure.load(Ordering::Relaxed)),
            self.admits.load(Ordering::Relaxed),
        )
    }
}

/// Logs a change of pressure: a warning as it starts, a note as it ends;
/// between mild and critical (no hysteresis there) only for debugging.
fn note_change(was: Pressure, to: Pressure, r: Reading) {
    let (physical_mib, commit_mib) = (r.physical >> 20, r.commit >> 20);
    let staged_mib = crate::texture::staged_total() >> 20;
    let oom = r.recent_oom;
    match (was, to) {
        (Pressure::None, _) => {
            tracing::warn!(
                ?to,
                physical_mib,
                commit_mib,
                staged_mib,
                oom,
                "host memory pressed"
            )
        }
        (_, Pressure::None) => {
            tracing::info!(physical_mib, commit_mib, "host memory pressure over")
        }
        _ => tracing::debug!(
            ?was,
            ?to,
            physical_mib,
            commit_mib,
            oom,
            "host memory pressure"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mild_pressure_starts_at_3_gib_physical_or_4_gib_commit() {
        let none = Pressure::None;
        assert_eq!(classify(16 * GIB, 16 * GIB, none, false), Pressure::None);
        assert_eq!(classify(16 * GIB, 3 * GIB, none, false), Pressure::Mild);
        assert_eq!(classify(16 * GIB, 3 * GIB + 1, none, false), Pressure::None);
        assert_eq!(classify(4 * GIB, 16 * GIB, none, false), Pressure::Mild);
        assert_eq!(classify(4 * GIB + 1, 16 * GIB, none, false), Pressure::None);
        // Between the edges: none stays none.
        assert_eq!(classify(5 * GIB, 4 * GIB, none, false), Pressure::None);
    }

    #[test]
    fn mild_pressure_ends_only_at_5_gib_physical_and_6_gib_commit() {
        for was in [Pressure::Mild, Pressure::Critical] {
            assert_eq!(classify(5 * GIB, 4 * GIB, was, false), Pressure::Mild);
            assert_eq!(classify(6 * GIB - 1, 8 * GIB, was, false), Pressure::Mild);
            assert_eq!(classify(8 * GIB, 5 * GIB - 1, was, false), Pressure::Mild);
            assert_eq!(classify(6 * GIB, 5 * GIB, was, false), Pressure::None);
        }
    }

    #[test]
    fn critical_below_1536_mib_of_commit_or_after_a_refusal() {
        for was in [Pressure::None, Pressure::Mild, Pressure::Critical] {
            let edge = 1536 * MIB;
            assert_eq!(classify(edge - 1, 16 * GIB, was, false), Pressure::Critical);
            assert_eq!(classify(edge, 16 * GIB, was, false), Pressure::Mild);
            assert_eq!(classify(0, 0, was, false), Pressure::Critical);
            assert_eq!(classify(16 * GIB, 16 * GIB, was, true), Pressure::Critical);
        }
    }

    #[test]
    fn new_staging_buffers_wait_below_the_reserve_or_after_a_refusal() {
        assert!(admits(STAGING_RESERVE, false));
        assert!(!admits(STAGING_RESERVE - 1, false));
        assert!(!admits(16 * GIB, true));
    }

    #[test]
    fn pressure_lowers_a_pass_budget_and_widens_its_release_looks() {
        assert_eq!(Pressure::None.pass_budget(256 * MIB), 256 * MIB);
        assert_eq!(Pressure::Mild.pass_budget(256 * MIB), 64 * MIB);
        assert_eq!(Pressure::Mild.pass_budget(32 * MIB), 32 * MIB);
        assert_eq!(Pressure::Critical.pass_budget(256 * MIB), 16 * MIB);
        assert_eq!(Pressure::None.release_looks(4096), 4096);
        assert_eq!(Pressure::Mild.release_looks(4096), 16384);
        assert_eq!(Pressure::Critical.release_looks(4096), usize::MAX);
    }

    #[test]
    fn memory_is_read_at_most_every_32_ms() {
        let sampler = Sampler::new();
        let reads = &std::cell::Cell::new(0);
        // A reading of `gib` GiB of commit and physical memory, counted.
        let read = |gib: u64, recent_oom: bool| {
            move || {
                reads.set(reads.get() + 1);
                let (commit, physical) = (gib * GIB, gib * GIB);
                Some(Reading {
                    commit,
                    physical,
                    recent_oom,
                })
            }
        };
        assert_eq!(sampler.at(100, read(1, false)), (Pressure::Critical, true));
        assert_eq!(reads.get(), 1);
        // Within 32 ms: what the last reading said, memory not read.
        assert_eq!(sampler.at(131, read(16, false)), (Pressure::Critical, true));
        assert_eq!(reads.get(), 1);
        assert_eq!(sampler.at(132, read(16, false)), (Pressure::None, true));
        assert_eq!(reads.get(), 2);
        // A failed reading keeps the last.
        assert_eq!(sampler.at(200, || None), (Pressure::None, true));
        assert_eq!(sampler.at(232, read(16, true)), (Pressure::Critical, false));
        assert_eq!(reads.get(), 3);
    }
}
