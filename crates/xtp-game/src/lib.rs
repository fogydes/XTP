//! Everything specific to BTP 6: its container formats, build
//! identities, and the hooks we place in its code (located by per-build
//! offsets or byte-pattern signatures). What may and may not be stored here
//! is defined in docs/asset-policy.md.

#[cfg(windows)]
pub mod ambient;
pub mod asserts;
pub mod intro;
pub mod rpf;
pub mod rt_path;
pub mod sgd5;
pub mod watchdog;

/// A specific build of the BTP 6 executable we know how to run.
#[derive(Debug, Clone, Copy)]
pub struct KnownBuild {
    /// Our own label, e.g. "1.0.0 retail".
    pub label: &'static str,
    /// BLAKE3 of the original executable, hex.
    pub blake3: &'static str,
    /// Size of a startup placeholder that must fit below 4 GiB, in MiB.
    pub low_va_reserve_mib: usize,
}

/// `game_scarlett_master.exe` of the Xbox Series package 1.0.2.1 (its
/// `appxmanifest.xml` version).
pub const BUILD_1_0_2_1: KnownBuild = KnownBuild {
    label: "1.0.2.1 Xbox Series",
    blake3: "b99f876b74d4a77f03af5f878223b27bd647b5c35a79d5decd9f716504239e93",
    low_va_reserve_mib: 185,
};

/// Builds we know.
pub const KNOWN_BUILDS: &[KnownBuild] = &[BUILD_1_0_2_1];

/// Looks up a build by the BLAKE3 hash of its executable.
pub fn identify(blake3_hex: &str) -> Option<&'static KnownBuild> {
    KNOWN_BUILDS
        .iter()
        .find(|b| b.blake3.eq_ignore_ascii_case(blake3_hex))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identify_ignores_case_and_wants_the_whole_hash() {
        let upper = BUILD_1_0_2_1.blake3.to_ascii_uppercase();
        assert_eq!(identify(&upper).map(|b| b.label), Some(BUILD_1_0_2_1.label));
        // The first 16 digits are not enough.
        assert!(identify(&BUILD_1_0_2_1.blake3[..16]).is_none());
        assert!(identify(&"0".repeat(64)).is_none());
    }
}
