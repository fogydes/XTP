//! Priority compute pipeline hashes for the known title build.
//!
//! Registers confirmed first-scene and auto-exposure compute shader hashes
//! with [`xtp_gpu::configure_priority_pipelines`] so they start compiling at
//! creation even on a cold cache and jump ahead of bulk scene draws in the
//! pipeline worker queue—preventing black startup/intro frames while
//! auto-exposure waits behind unrelated draws.
//!
//! Adapted from `crbns/victory-tapin` commit `09c6420b` (GPL-3.0-or-later).

/// First eight BLAKE3 bytes (big-endian, matching logged 16-hex-digit hashes)
/// of confirmed first-scene producers, auto-exposure compute stages, and late
/// startup compute shaders for the known title build.
pub const PRIORITY_SHADERS: &[u64] = &[
    // Confirmed first-scene producers.
    0x970c_4fbb_3b01_7c0e,
    0x068c_90ba_dc86_f1fd,
    0xe4a7_e088_c778_0ce8,
    0xebe4_e53f_031f_5249,
    0x2cc4_c197_ee29_e291,
    0x369c_be90_6ea7_2764,
    // Auto-exposure (luminance downsample and exposure value) plus four
    // compute shaders consistently observed late on cold startup.
    0x47d8_c225_abfc_b246,
    0xfdce_2a0a_9fb1_521a,
    0x5282_78df_1274_cb33,
    0x477b_52e7_3dde_b6f2,
    0x8b43_7ab6_1d80_08fe,
    0x17af_85f1_70e6_6f30,
];

/// Registers the known title's priority shader hashes with `xtp-gpu`.
#[cfg(windows)]
pub fn configure() {
    xtp_gpu::configure_priority_pipelines(PRIORITY_SHADERS);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn priority_shader_hashes_are_distinct_and_nonzero() {
        let unique: HashSet<u64> = PRIORITY_SHADERS.iter().copied().collect();
        assert_eq!(unique.len(), PRIORITY_SHADERS.len());
        assert!(!unique.contains(&0));
    }
}
