//! Which processors host threads run on when a CPU has heterogeneous core
//! classes (performance and efficiency cores).
//!
//! Adapted from `codec-d/cool_fox` (`crates/victory-gpu/src/cores.rs`,
//! `avatarmanz/VIctory-7900XTX`) and `fxceless/victory-fork` (`framelog.rs`,
//! commit `0b0074e0`, `GPL-3.0-or-later`).

use windows_sys::Win32::System::SystemInformation::{
    CpuSetInformation, GetSystemCpuSetInformation, SYSTEM_CPU_SET_INFORMATION,
};
use windows_sys::Win32::System::Threading::{
    GetCurrentProcess, GetCurrentThread, SetThreadSelectedCpuSets,
};

/// The system's processor sets in group 0: each set's ID and efficiency class
/// (higher is faster).
fn cpu_sets() -> Vec<(u32, u8)> {
    let mut bytes = 0u32;
    // SAFETY: size query with null output buffer.
    unsafe {
        GetSystemCpuSetInformation(std::ptr::null_mut(), 0, &mut bytes, GetCurrentProcess(), 0);
    }
    if bytes == 0 {
        return Vec::new();
    }
    let capacity = bytes;
    let mut buffer = vec![0u64; (capacity as usize).div_ceil(8)];
    // SAFETY: `buffer` is 8-byte aligned and holds at least `bytes` bytes.
    let ok = unsafe {
        GetSystemCpuSetInformation(
            buffer.as_mut_ptr().cast(),
            capacity,
            &mut bytes,
            GetCurrentProcess(),
            0,
        )
    };
    if ok == 0 || bytes > capacity {
        return Vec::new();
    }
    let mut sets = Vec::new();
    let (mut at, end) = (0usize, bytes as usize);
    let header = 2 * size_of::<u32>();
    while end - at >= header {
        // SAFETY: the remaining bytes contain a complete size/type header.
        let ptr = unsafe { buffer.as_ptr().cast::<u8>().add(at) };
        // SAFETY: both fields fit inside the initialized header.
        let (size, kind) = unsafe {
            (
                ptr.cast::<u32>().read_unaligned() as usize,
                ptr.add(4).cast::<i32>().read_unaligned(),
            )
        };
        if size < header || size > end - at {
            break;
        }
        if kind == CpuSetInformation && size >= size_of::<SYSTEM_CPU_SET_INFORMATION>() {
            // SAFETY: a complete CPU-set record within the initialized buffer.
            let set = unsafe {
                ptr.cast::<SYSTEM_CPU_SET_INFORMATION>()
                    .read_unaligned()
                    .Anonymous
                    .CpuSet
            };
            if set.Group == 0 {
                sets.push((set.Id, set.EfficiencyClass));
            }
        }
        at += size;
    }
    sets
}

/// Returns the IDs of the highest efficiency class, or empty if all cores
/// share the same class.
fn fast_of(sets: &[(u32, u8)]) -> Vec<u32> {
    let best = sets.iter().map(|c| c.1).max().unwrap_or(0);
    let fast: Vec<u32> = sets.iter().filter(|c| c.1 == best).map(|c| c.0).collect();
    if fast.len() == sets.len() {
        Vec::new()
    } else {
        fast
    }
}

fn parse_stage(value: Option<&str>) -> u32 {
    value.and_then(|v| v.trim().parse().ok()).unwrap_or(2)
}

/// `XTP_FAST_CORES`: `0` disables affinity hints; `1` pins completion/wait
/// threads that a frame waits on; `2` (default) adds reserve helper threads;
/// `3` also includes pipeline and parallel submission worker pools.
fn stage() -> u32 {
    static STAGE: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *STAGE.get_or_init(|| parse_stage(std::env::var("XTP_FAST_CORES").ok().as_deref()))
}

/// Prefers performance cores for a calling thread that a frame waits on.
pub(crate) fn keep_on_fast_cores(what: &str) {
    ask(what, 1);
}

/// Prefers performance cores for a background reserve helper thread.
pub(crate) fn keep_helper_on_fast_cores(what: &str) {
    ask(what, 2);
}

/// Prefers performance cores for worker-pool threads when `XTP_FAST_CORES=3`.
pub(crate) fn keep_many_on_fast_cores(what: &str) {
    ask(what, 3);
}

fn ask(what: &str, from: u32) {
    if stage() < from {
        return;
    }
    static FAST: std::sync::OnceLock<(Vec<u32>, usize)> = std::sync::OnceLock::new();
    let (fast, of) = FAST.get_or_init(|| {
        let sets = cpu_sets();
        (fast_of(&sets), sets.len())
    });
    if fast.is_empty() {
        return;
    }
    // SAFETY: `GetCurrentThread()` is valid for this thread and `fast` holds valid OS CPU-set IDs.
    let ok =
        unsafe { SetThreadSelectedCpuSets(GetCurrentThread(), fast.as_ptr(), fast.len() as u32) };
    if ok == 0 {
        tracing::warn!(
            what,
            error = %std::io::Error::last_os_error(),
            "the thread could not be kept on fast cores"
        );
    } else {
        tracing::info!(
            what,
            fast = fast.len(),
            of,
            "the thread prefers fast cores (XTP_FAST_CORES=0: disable)"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fast_cores_are_those_of_the_best_class_if_there_is_a_worse_one() {
        assert_eq!(fast_of(&[(256, 1), (257, 1), (258, 0)]), [256, 257]);
        assert!(fast_of(&[(256, 0), (257, 0)]).is_empty());
        assert!(fast_of(&[]).is_empty());
        assert!(!cpu_sets().is_empty());
        std::thread::spawn(|| keep_on_fast_cores("test"))
            .join()
            .unwrap();
    }

    #[test]
    fn stage_parses_with_default_two() {
        assert_eq!(parse_stage(None), 2);
        assert_eq!(parse_stage(Some("0")), 0);
        assert_eq!(parse_stage(Some(" 3 ")), 3);
        assert_eq!(parse_stage(Some("invalid")), 2);
    }
}
