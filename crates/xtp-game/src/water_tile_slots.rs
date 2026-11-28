//! Wave32 per-lane tile-list slot allocation config for the known title build.
//!
//! Registers the build's water-light tile-list compute shader with
//! [`xtp_shader::emit::water_tile_slots::configure`] so 32-lane hosts allocate
//! one slot per lane instead of sharing a wave64 ballot across two host waves.
//!
//! Adapted from `fxceless/victory-fork` commit `cd935520` (GPL-3.0-or-later).

use xtp_shader::emit::water_tile_slots::{AppendRegion, AppendRewrite};

/// Two disjoint append blocks of the 1.0.2.1 tile-list compute program:
/// - Front list (tiles without local lights): atomic add at offset 0 and wait.
/// - Back list (tiles with local lights): atomic add at offset 32, list-size
///   load, and two waits (the slot reversal immediately after stays intact).
const APPENDS_1_0_2_1: &[AppendRegion] = &[
    AppendRegion {
        body: (0x6a8, 0x704),
        kept: &[(0x6dc, 0x6e4), (0x6f0, 0x6f4)],
    },
    AppendRegion {
        body: (0x70c, 0x774),
        kept: &[
            (0x740, 0x748),
            (0x74c, 0x754),
            (0x75c, 0x760),
            (0x770, 0x774),
        ],
    },
];

/// Known title program specification (`1.0.2.1` Xbox Series build).
pub const REWRITES: &[AppendRewrite] = &[AppendRewrite {
    container: [
        0xf5, 0xd1, 0x0c, 0xde, 0x74, 0x33, 0x37, 0x82, 0xf3, 0x0f, 0x9b, 0xb5, 0x56, 0xc0, 0x73,
        0xd4,
    ],
    code_bytes: 2112,
    code_fnv1a: 0x377a_8d97_4fb8_631f,
    move_zero: 0x6c4,
    nop: 0x6d8,
    appends: APPENDS_1_0_2_1,
}];

/// Registers the known title's wave32 tile-list rewrites with `xtp-shader`.
pub fn configure() {
    xtp_shader::emit::water_tile_slots::configure(REWRITES);
}

#[cfg(test)]
mod tests {
    use super::*;
    use xtp_shader::emit::water_tile_slots::spec_bounds_valid;

    #[test]
    fn configured_water_tile_rewrite_bounds_are_valid() {
        assert!(!REWRITES.is_empty());
        for spec in REWRITES {
            assert!(spec_bounds_valid(spec));
        }
    }
}
