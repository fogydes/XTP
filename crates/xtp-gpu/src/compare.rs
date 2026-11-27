//! AVX2-accelerated buffer comparison with scalar fallback.
//!
//! Used by copy-back scanning ([`differ`]), dword-run discovery
//! ([`word_runs`]), and three-way merge scanning ([`first_change`],
//! [`first_same`]) on slices that are already safely mapped and readable.
//! AVX2 paths are chosen at runtime via `is_x86_feature_detected!("avx2")`.

/// Returns whether two equal-length slices differ in any byte.
pub(crate) fn differ(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return true;
    }
    #[cfg(target_arch = "x86_64")]
    if a.len() >= 32 && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 support was checked at runtime and both slices have the
        // same length.
        return unsafe { differ_avx2(a, b) };
    }
    differ_scalar(a, b)
}

fn differ_scalar(a: &[u8], b: &[u8]) -> bool {
    a != b
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn differ_avx2(a: &[u8], b: &[u8]) -> bool {
    use std::arch::x86_64::{
        __m256i, _mm256_loadu_si256, _mm256_or_si256, _mm256_setzero_si256, _mm256_testz_si256,
        _mm256_xor_si256,
    };

    let blocks = a.len() / 32;
    let mut at = 0;
    while at < blocks {
        let mut any = _mm256_setzero_si256();
        let end = (at + 4).min(blocks);
        for k in at..end {
            // SAFETY: `k * 32 + 32 <= blocks * 32 <= a.len() == b.len()`.
            let (x, y) = unsafe {
                (
                    _mm256_loadu_si256(a.as_ptr().add(k * 32).cast::<__m256i>()),
                    _mm256_loadu_si256(b.as_ptr().add(k * 32).cast::<__m256i>()),
                )
            };
            any = _mm256_or_si256(any, _mm256_xor_si256(x, y));
        }
        if _mm256_testz_si256(any, any) == 0 {
            return true;
        }
        at = end;
    }
    differ_scalar(&a[blocks * 32..], &b[blocks * 32..])
}

/// Finds all contiguous runs `(offset, len)` of aligned dwords (relative to
/// base address `at`, clamped to `0..a.len()` at unaligned ends) where `a`
/// and `b` differ in at least one byte.
pub(crate) fn word_runs(at: usize, a: &[u8], b: &[u8], runs: &mut Vec<(usize, usize)>) {
    assert_eq!(a.len(), b.len());
    runs.clear();
    #[cfg(target_arch = "x86_64")]
    if a.len() >= 32 && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 support was checked at runtime and both slices have the
        // same length.
        return unsafe { word_runs_avx2_into(at, a, b, runs) };
    }
    word_runs_scalar_into(at, a, b, runs)
}

fn push_run(runs: &mut Vec<(usize, usize)>, from: usize, len: usize) {
    match runs.last_mut() {
        Some((start, n)) if *start + *n == from => *n += len,
        _ => runs.push((from, len)),
    }
}

fn word_runs_scalar_into(at: usize, a: &[u8], b: &[u8], runs: &mut Vec<(usize, usize)>) {
    let n = a.len();
    let head = ((4 - (at & 3)) & 3).min(n);
    if head > 0 && a[..head] != b[..head] {
        push_run(runs, 0, head);
    }
    let tail = head + ((n - head) & !3);
    let mut i = head;
    while i < tail {
        if a[i..i + 4] != b[i..i + 4] {
            push_run(runs, i, 4);
        }
        i += 4;
    }
    if tail < n && a[tail..] != b[tail..] {
        push_run(runs, tail, n - tail);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn word_runs_avx2_into(at: usize, a: &[u8], b: &[u8], runs: &mut Vec<(usize, usize)>) {
    use std::arch::x86_64::{
        __m256i, _mm256_castsi256_ps, _mm256_cmpeq_epi32, _mm256_loadu_si256, _mm256_movemask_ps,
    };

    let n = a.len();
    let head = ((4 - (at & 3)) & 3).min(n);
    if head > 0 && a[..head] != b[..head] {
        push_run(runs, 0, head);
    }
    let tail = head + ((n - head) & !3);
    let mut i = head;
    while i + 256 <= tail {
        let mut mask = 0u64;
        for k in 0..8 {
            // SAFETY: `i + 32 * k + 32 <= i + 256 <= tail <= n == a.len() == b.len()`.
            let same = unsafe {
                _mm256_cmpeq_epi32(
                    _mm256_loadu_si256(a.as_ptr().add(i + 32 * k).cast::<__m256i>()),
                    _mm256_loadu_si256(b.as_ptr().add(i + 32 * k).cast::<__m256i>()),
                )
            };
            let diff_byte = (!_mm256_movemask_ps(_mm256_castsi256_ps(same))) as u8;
            mask |= u64::from(diff_byte) << (8 * k);
        }
        while mask != 0 {
            let first = mask.trailing_zeros();
            let count = (mask >> first).trailing_ones();
            push_run(runs, i + 4 * first as usize, 4 * count as usize);
            if first + count >= 64 {
                break;
            }
            mask &= u64::MAX << (first + count);
        }
        i += 256;
    }
    while i + 32 <= tail {
        // SAFETY: `i + 32 <= tail <= n == a.len() == b.len()`.
        let same = unsafe {
            _mm256_cmpeq_epi32(
                _mm256_loadu_si256(a.as_ptr().add(i).cast::<__m256i>()),
                _mm256_loadu_si256(b.as_ptr().add(i).cast::<__m256i>()),
            )
        };
        let mut mask = (!_mm256_movemask_ps(_mm256_castsi256_ps(same))) as u8;
        while mask != 0 {
            let first = mask.trailing_zeros();
            let count = (mask >> first).trailing_ones();
            push_run(runs, i + 4 * first as usize, 4 * count as usize);
            if first + count >= 8 {
                break;
            }
            mask &= u8::MAX << (first + count);
        }
        i += 32;
    }
    while i < tail {
        if a[i..i + 4] != b[i..i + 4] {
            push_run(runs, i, 4);
        }
        i += 4;
    }
    if tail < n && a[tail..] != b[tail..] {
        push_run(runs, tail, n - tail);
    }
}

/// First offset from `at` where `a` and `b` differ (`a.len()` if equal).
pub(crate) fn first_change(a: &[u8], b: &[u8], at: usize) -> usize {
    assert_eq!(a.len(), b.len());
    let at = at.min(a.len());
    #[cfg(target_arch = "x86_64")]
    if a.len().saturating_sub(at) >= 32 && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 support was checked at runtime and both slices have the
        // same length.
        return unsafe { first_change_avx2(a, b, at) };
    }
    first_change_scalar(a, b, at)
}

fn first_change_scalar(a: &[u8], b: &[u8], mut at: usize) -> usize {
    while at + 8 <= a.len() {
        let x = word(a, at) ^ word(b, at);
        if x != 0 {
            return at + (x.trailing_zeros() / 8) as usize;
        }
        at += 8;
    }
    while at < a.len() && a[at] == b[at] {
        at += 1;
    }
    at
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn first_change_avx2(a: &[u8], b: &[u8], mut at: usize) -> usize {
    use std::arch::x86_64::{__m256i, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8};

    while at + 32 <= a.len() {
        // SAFETY: `at + 32 <= a.len() == b.len()`.
        let same = unsafe {
            _mm256_cmpeq_epi8(
                _mm256_loadu_si256(a.as_ptr().add(at).cast::<__m256i>()),
                _mm256_loadu_si256(b.as_ptr().add(at).cast::<__m256i>()),
            )
        };
        let mask = _mm256_movemask_epi8(same) as u32;
        if mask != u32::MAX {
            return at + (!mask).trailing_zeros() as usize;
        }
        at += 32;
    }
    first_change_scalar(a, b, at)
}

/// First offset from `at` where `a` and `b` are equal (`a.len()` if none).
pub(crate) fn first_same(a: &[u8], b: &[u8], at: usize) -> usize {
    assert_eq!(a.len(), b.len());
    let at = at.min(a.len());
    #[cfg(target_arch = "x86_64")]
    if a.len().saturating_sub(at) >= 32 && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 support was checked at runtime and both slices have the
        // same length.
        return unsafe { first_same_avx2(a, b, at) };
    }
    first_same_scalar(a, b, at)
}

fn first_same_scalar(a: &[u8], b: &[u8], mut at: usize) -> usize {
    const ONES: u64 = u64::from_le_bytes([1; 8]);
    const HIGHS: u64 = u64::from_le_bytes([0x80; 8]);
    while at + 8 <= a.len() {
        let x = word(a, at) ^ word(b, at);
        let zero = x.wrapping_sub(ONES) & !x & HIGHS;
        if zero != 0 {
            return at + (zero.trailing_zeros() / 8) as usize;
        }
        at += 8;
    }
    while at < a.len() && a[at] != b[at] {
        at += 1;
    }
    at
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn first_same_avx2(a: &[u8], b: &[u8], mut at: usize) -> usize {
    use std::arch::x86_64::{__m256i, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8};

    while at + 32 <= a.len() {
        // SAFETY: `at + 32 <= a.len() == b.len()`.
        let same = unsafe {
            _mm256_cmpeq_epi8(
                _mm256_loadu_si256(a.as_ptr().add(at).cast::<__m256i>()),
                _mm256_loadu_si256(b.as_ptr().add(at).cast::<__m256i>()),
            )
        };
        let mask = _mm256_movemask_epi8(same) as u32;
        if mask != 0 {
            return at + mask.trailing_zeros() as usize;
        }
        at += 32;
    }
    first_same_scalar(a, b, at)
}

fn word(bytes: &[u8], at: usize) -> u64 {
    let mut w = [0; 8];
    w.copy_from_slice(&bytes[at..at + 8]);
    u64::from_le_bytes(w)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word_runs_check(at: usize, a: &[u8], b: &[u8]) -> Vec<(usize, usize)> {
        let mut runs = vec![(10, 10)];
        word_runs(at, a, b, &mut runs);
        runs
    }

    fn word_runs_scalar_check(at: usize, a: &[u8], b: &[u8]) -> Vec<(usize, usize)> {
        let mut runs = Vec::new();
        word_runs_scalar_into(at, a, b, &mut runs);
        runs
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn word_runs_avx2_check(at: usize, a: &[u8], b: &[u8]) -> Vec<(usize, usize)> {
        let mut runs = Vec::new();
        // SAFETY: the test checks AVX2 at runtime and passes equal-length slices.
        unsafe { word_runs_avx2_into(at, a, b, &mut runs) };
        runs
    }

    struct Random(u64);

    impl Random {
        fn next(&mut self) -> usize {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 8) as usize
        }
    }

    #[test]
    fn avx2_and_scalar_agree_on_random_unaligned_buffers() {
        #[cfg(target_arch = "x86_64")]
        let has_avx2 = std::arch::is_x86_feature_detected!("avx2");
        #[cfg(not(target_arch = "x86_64"))]
        let has_avx2 = false;

        let mut rng = Random(0x5eed_cafe_1234_5678);
        let backing_len = 8192 + 64;
        let base_a: Vec<u8> = (0..backing_len).map(|_| rng.next() as u8).collect();
        let mut base_b = base_a.clone();

        // Exercise a wide matrix of unaligned starts, unaligned ends, edge sizes,
        // single-byte flips, dense flips, and all-equal / all-different cases.
        let lengths = [
            0usize, 1, 2, 3, 4, 5, 7, 8, 15, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257,
            511, 512, 513, 1000, 4095, 4096, 4097,
        ];
        for &len in &lengths {
            for start_a in [0usize, 1, 2, 3, 5, 17, 31] {
                let start_b = (start_a + (rng.next() % 7)) % 32;
                let base_addr = rng.next() & 0xffff;
                let a = &base_a[start_a..start_a + len];
                base_b[start_b..start_b + len].copy_from_slice(a);

                // Equal case.
                {
                    let b = &base_b[start_b..start_b + len];
                    assert_eq!(differ(a, b), differ_scalar(a, b));
                    assert_eq!(
                        word_runs_check(base_addr, a, b),
                        word_runs_scalar_check(base_addr, a, b)
                    );
                    assert_eq!(first_change(a, b, 0), first_change_scalar(a, b, 0));
                    assert_eq!(first_same(a, b, 0), first_same_scalar(a, b, 0));
                    #[cfg(target_arch = "x86_64")]
                    if has_avx2 {
                        // SAFETY: `has_avx2` checked above.
                        unsafe {
                            assert!(!differ_avx2(a, b));
                            assert_eq!(
                                word_runs_avx2_check(base_addr, a, b),
                                word_runs_scalar_check(base_addr, a, b)
                            );
                            assert_eq!(first_change_avx2(a, b, 0), len);
                            assert_eq!(first_same_avx2(a, b, 0), 0);
                        }
                    }
                }

                if len == 0 {
                    continue;
                }

                // Single-byte flip at start, middle, and end.
                for flip in [0, len / 2, len - 1] {
                    base_b[start_b + flip] ^= 0x81;
                    let b = &base_b[start_b..start_b + len];
                    assert!(differ(a, b));
                    assert_eq!(
                        word_runs_check(base_addr, a, b),
                        word_runs_scalar_check(base_addr, a, b)
                    );
                    assert_eq!(first_change(a, b, 0), flip);
                    assert_eq!(first_change(a, b, 0), first_change_scalar(a, b, 0));
                    assert_eq!(first_same(a, b, flip), first_same_scalar(a, b, flip));
                    #[cfg(target_arch = "x86_64")]
                    if has_avx2 {
                        // SAFETY: `has_avx2` checked above.
                        unsafe {
                            assert!(differ_avx2(a, b));
                            assert_eq!(
                                word_runs_avx2_check(base_addr, a, b),
                                word_runs_scalar_check(base_addr, a, b)
                            );
                            assert_eq!(first_change_avx2(a, b, 0), flip);
                            assert_eq!(first_same_avx2(a, b, flip), first_same_scalar(a, b, flip));
                        }
                    }
                    base_b[start_b + flip] ^= 0x81;
                }
            }
        }

        // Random sparse and dense diffs across 400 rounds.
        for _ in 0..400 {
            let start_a = rng.next() % 37;
            let start_b = rng.next() % 37;
            let len = 1 + rng.next() % 4096;
            let base_addr = rng.next() & 0xffff;
            let a = &base_a[start_a..start_a + len];
            base_b[start_b..start_b + len].copy_from_slice(a);

            let flips = rng.next() % 120;
            for _ in 0..flips {
                let at = rng.next() % len;
                let span = (1 + rng.next() % 24).min(len - at);
                for k in at..at + span {
                    base_b[start_b + k] ^= (1 + (rng.next() % 255)) as u8;
                }
            }
            let b = &base_b[start_b..start_b + len];
            let scalar_diff = differ_scalar(a, b);
            let scalar_runs = word_runs_scalar_check(base_addr, a, b);
            assert_eq!(differ(a, b), scalar_diff);
            assert_eq!(word_runs_check(base_addr, a, b), scalar_runs);
            #[cfg(target_arch = "x86_64")]
            if has_avx2 {
                // SAFETY: `has_avx2` checked above.
                unsafe {
                    assert_eq!(differ_avx2(a, b), scalar_diff);
                    assert_eq!(word_runs_avx2_check(base_addr, a, b), scalar_runs);
                }
            }

            let mut cursor = 0;
            while cursor < len {
                let reference = (cursor..len).find(|&i| a[i] != b[i]).unwrap_or(len);
                let c_scalar = first_change_scalar(a, b, cursor);
                assert_eq!(c_scalar, reference);
                assert_eq!(first_change(a, b, cursor), c_scalar);
                #[cfg(target_arch = "x86_64")]
                if has_avx2 {
                    // SAFETY: `has_avx2` checked above.
                    assert_eq!(unsafe { first_change_avx2(a, b, cursor) }, c_scalar);
                }
                if c_scalar == len {
                    break;
                }
                let reference = (c_scalar..len).find(|&i| a[i] == b[i]).unwrap_or(len);
                let s_scalar = first_same_scalar(a, b, c_scalar);
                assert_eq!(s_scalar, reference);
                assert_eq!(first_same(a, b, c_scalar), s_scalar);
                #[cfg(target_arch = "x86_64")]
                if has_avx2 {
                    // SAFETY: `has_avx2` checked above.
                    assert_eq!(unsafe { first_same_avx2(a, b, c_scalar) }, s_scalar);
                }
                cursor = s_scalar;
            }
        }
    }
}
