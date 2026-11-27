//! Compute programs and descriptor bindings for the optional ambient pass.

use xtp_gpu::ambient::{Layout, Role};

const LAYOUT: Layout = Layout {
    light: (1, 176),
    depth: (0, 464),
    radiance: (1, 16),
    air: (1, 32),
};

/// The known build's programs, by the hash the runtime logs.
const BUILD_1_0_2_1: &[(&str, Role)] = &[
    // The six variants of the screen probes' temporal resolve, the packed
    // diffuse light writer of docs/analysis/light-chain.md section 4 (with
    // and without its caustics and constant-buffer forms).
    ("2fc5d11ece80c884", Role::Resolve),
    ("34fb51b8a0312e82", Role::Resolve),
    ("5053352ac4370ca1", Role::Resolve),
    ("2bed88a03a2f5558", Role::Resolve),
    ("cf76630573822d04", Role::Resolve),
    ("3b4f6eb5edb79848", Role::Resolve),
    // The pass that keeps the light visible surfaces send out, read the
    // next frame.
    ("e6d8226a6870f577", Role::Radiance),
    // The pass that accumulates local lights' in-scattering volume.
    ("b7de1a805fbbac3c", Role::Air),
];

fn programs(executable_hash: &str) -> &'static [(&'static str, Role)] {
    if crate::BUILD_1_0_2_1
        .blake3
        .eq_ignore_ascii_case(executable_hash)
    {
        BUILD_1_0_2_1
    } else {
        &[]
    }
}

/// Supplies the known build's hashes before any compute pipelines are made.
/// Other builds can opt in with explicit hashes for the same view layout.
pub fn configure(executable_hash: &str) {
    xtp_gpu::ambient::configure(LAYOUT, programs(executable_hash));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ambient_programs_apply_to_the_known_build_only() {
        let hash = crate::BUILD_1_0_2_1.blake3;
        assert_eq!(programs(hash), BUILD_1_0_2_1);
        assert_eq!(programs(&hash.to_ascii_uppercase()), programs(hash));
        assert!(programs(&hash[..16]).is_empty());
    }

    #[test]
    fn every_role_has_a_program_and_each_hash_is_listed_once() {
        for role in [Role::Resolve, Role::Radiance, Role::Air] {
            assert!(BUILD_1_0_2_1.iter().any(|&(_, r)| r == role), "{role:?}");
        }
        let mut hashes: Vec<_> = BUILD_1_0_2_1.iter().map(|&(h, _)| h).collect();
        assert!(hashes.iter().all(|h| {
            h.len() == 16
                && h.bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        }));
        hashes.sort_unstable();
        hashes.dedup();
        assert_eq!(hashes.len(), BUILD_1_0_2_1.len());
    }
}
