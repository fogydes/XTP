//! Character and decoration bake shader hashes for the known build.
//!
//! Registers the pixel and geometry shaders whose draws sample bindless
//! textures by index while loading and in world scenes, so waiting heap
//! textures are staged and uploaded before those one-shot bakes sample them.
//!
//! Adapted from `fxceless/victory-fork` commit `c96b4f2` and
//! `codec-d/cool_fox` (`GPL-3.0-or-later`).

/// Sixteen-digit logged shader hashes of the known build's one-shot bake
/// pixel and geometry shaders (vertex compositor and projection mesh).
pub(crate) const BUILD_1_0_2_1: &[&str] = &[
    "c0a7fe77b1862e10",
    "0d1575a396dfe513",
    "b7cfd8dda91ebb6b",
    "280967ad158cee0b",
    "fa455c8a3c32577b",
    "d126bcae5d5d8d26",
    "fcee612943c008e0",
    "089d6271be6650ef",
    "d4111a6e6ea49d91",
];

fn shaders(executable_hash: &str) -> &'static [&'static str] {
    if crate::BUILD_1_0_2_1
        .blake3
        .eq_ignore_ascii_case(executable_hash)
    {
        BUILD_1_0_2_1
    } else {
        &[]
    }
}

/// Registers the known title's bake shader hashes with `xtp-gpu`.
#[cfg(windows)]
pub fn configure(executable_hash: &str) {
    xtp_gpu::configure_bake_shaders(shaders(executable_hash));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bake_shaders_apply_to_the_known_build_only() {
        let hash = crate::BUILD_1_0_2_1.blake3;
        assert_eq!(shaders(hash), BUILD_1_0_2_1);
        assert_eq!(shaders(&hash.to_ascii_uppercase()), BUILD_1_0_2_1);
        assert!(shaders(&hash[..16]).is_empty());
    }

    #[test]
    fn every_bake_hash_is_unique_16_hex_digits() {
        let mut hashes: Vec<_> = BUILD_1_0_2_1.to_vec();
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
