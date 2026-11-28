//! Per-lane slot allocation for wave64 tile-list append shaders on 32-lane
//! hosts.
//!
//! A console wave64 compute program that appends tiles to shared lists often
//! compacts slots wave-wide: the first active lane adds the wave's active
//! count to a list counter, and each lane computes its slot from that base
//! plus its rank in the wave's lane mask. When lowered onto two 32-lane host
//! waves, the upper half's rank depends on the lower half's mask, and an
//! imperfect cross-half exchange lets the second host wave overwrite slots
//! claimed by the first—leaving those tiles unlit (dark 16x16 patches on
//! water surfaces).
//!
//! For registered programs (`configure`), this pre-decode rewrite replaces
//! each wave-wide append region with one per-lane atomic add of 1 (keeping
//! the program's own atomic, buffer-size load, and wait instructions in
//! place, and padding the remainder of each region with `s_nop` so all
//! branch targets and trailing code keep their offsets). Every active lane
//! still claims a unique slot and counters end with the same totals; only
//! the intra-list order changes, which consumers do not depend on.
//!
//! `XTP_WATER_TILE_SLOTS=0` disables the rewrite (the program translates as
//! authored). Title-specific program specs live in `xtp-game` and are
//! registered via [`configure`]; `xtp-shader` itself stays title-agnostic.
//!
//! Adapted from `fxceless/victory-fork` commit `cd935520` (GPL-3.0-or-later,
//! itself adapted from xss3009's wave32 tile allocation fix).

use std::ffi::OsStr;
use std::sync::OnceLock;

use crate::dxbc::Container;
use crate::emit::Target;
use crate::native::{HwStage, NativeProgram};

/// One wave-wide append block to rewrite into a per-lane atomic add.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AppendRegion {
    /// Byte range `(start, end)` of the append region in the console code.
    /// Rewritten instructions are placed at `start` and the remaining bytes
    /// up to `end` are filled with `s_nop` so branch offsets stay intact.
    pub body: (usize, usize),
    /// Byte ranges `(start, end)` copied from the original program in order
    /// immediately after the per-lane `v_mov_b32 vN, 1` instruction (e.g.
    /// atomic add, list-size load, and wait instructions).
    pub kept: &'static [(usize, usize)],
}

/// Specification for rewriting a known wave64 compute program's list-append
/// blocks into per-lane slot allocations on 32-lane hosts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AppendRewrite {
    /// Expected DXBC container digest (bytes `4..20`).
    pub container: [u8; 16],
    /// Expected console code length in bytes.
    pub code_bytes: usize,
    /// Expected 64-bit FNV-1a hash of the original console code bytes.
    pub code_fnv1a: u64,
    /// Byte offset of a 4-byte `v_mov_b32 vN, 0` instruction whose inline
    /// constant 0 source (`0x80`) becomes inline constant 1 (`0x81`).
    pub move_zero: usize,
    /// Byte offset of a 4-byte `s_nop` instruction used to pad each region.
    pub nop: usize,
    /// Append regions to rewrite.
    pub appends: &'static [AppendRegion],
}

static REWRITES: OnceLock<Vec<AppendRewrite>> = OnceLock::new();

/// Registers wave32 per-lane append rewrites before shader translation runs.
/// Subsequent calls keep the first registration.
pub fn configure(rewrites: &[AppendRewrite]) {
    let _ = REWRITES.set(rewrites.to_vec());
}

/// Whether the wave32 per-lane slot rewrite is enabled
/// (`XTP_WATER_TILE_SLOTS=0` disables it).
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse_enabled(std::env::var_os("XTP_WATER_TILE_SLOTS").as_deref()))
}

/// Pure parser for `XTP_WATER_TILE_SLOTS`: enabled unless set to `"0"`.
fn parse_enabled(value: Option<&OsStr>) -> bool {
    value.and_then(OsStr::to_str).map(str::trim) != Some("0")
}

/// 64-bit FNV-1a hash used to verify exact console code bytes before
/// applying a byte-range rewrite.
pub fn fnv1a_64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// Applies `spec` to raw console code if stage, wave size, length, hash, and
/// region bounds all match. Returns whether `code` was modified.
pub fn apply_spec_to_code(
    stage: HwStage,
    wave64: bool,
    code: &mut [u8],
    spec: &AppendRewrite,
) -> bool {
    if stage != HwStage::Cs
        || !wave64
        || code.len() != spec.code_bytes
        || fnv1a_64(code) != spec.code_fnv1a
        || !spec_bounds_valid(spec)
    {
        return false;
    }

    let before = code.to_vec();
    let mut one: [u8; 4] = before[spec.move_zero..spec.move_zero + 4]
        .try_into()
        .expect("four bytes");
    // In VOP1 `v_mov_b32`, the low 9-bit source operand encodes inline
    // constant 0 as `0x80` and inline constant 1 as `0x81`.
    one[0] |= 1;
    let nop: [u8; 4] = before[spec.nop..spec.nop + 4]
        .try_into()
        .expect("four bytes");

    for append in spec.appends {
        let (body_start, body_end) = append.body;
        let mut rewritten = Vec::with_capacity(body_end - body_start);
        rewritten.extend_from_slice(&one);
        for &(start, end) in append.kept {
            rewritten.extend_from_slice(&before[start..end]);
        }
        let body = &mut code[body_start..body_end];
        body[..rewritten.len()].copy_from_slice(&rewritten);
        for word in body[rewritten.len()..].as_chunks_mut::<4>().0 {
            *word = nop;
        }
    }
    true
}

/// Validates that every byte range in `spec` fits within `spec.code_bytes`,
/// is 4-byte aligned where required, and leaves enough room for the rewritten
/// instructions plus NOP padding.
pub fn spec_bounds_valid(spec: &AppendRewrite) -> bool {
    let len = spec.code_bytes;
    if len == 0
        || spec.move_zero.checked_add(4).is_none_or(|end| end > len)
        || spec.nop.checked_add(4).is_none_or(|end| end > len)
        || spec.appends.is_empty()
    {
        return false;
    }
    for append in spec.appends {
        let (body_start, body_end) = append.body;
        if body_start >= body_end || body_end > len || (body_end - body_start) % 4 != 0 {
            return false;
        }
        let mut written = 4usize;
        for &(start, end) in append.kept {
            if start >= end || end > len || (end - start) % 4 != 0 {
                return false;
            }
            let Some(next) = written.checked_add(end - start) else {
                return false;
            };
            written = next;
        }
        if written > body_end - body_start {
            return false;
        }
    }
    true
}

/// Applies the matching rewrite from `rewrites` when `target` is a 32-lane
/// host (`target.lanes_max == 32`) and `is_enabled` is true.
pub(super) fn rewrite_code_with(
    c: &Container,
    stage: HwStage,
    wave64: bool,
    code: &mut [u8],
    target: Target,
    rewrites: &[AppendRewrite],
    is_enabled: bool,
) -> bool {
    if target.lanes_max != 32 || !is_enabled {
        return false;
    }
    let Some(spec) = rewrites.iter().find(|spec| spec.container == c.digest) else {
        return false;
    };
    apply_spec_to_code(stage, wave64, code, spec)
}

/// Applies any configured wave32 per-lane append rewrite to `code`.
pub(super) fn rewrite_for_target(
    c: &Container,
    stage: HwStage,
    wave64: bool,
    code: &mut [u8],
    target: Target,
) -> bool {
    let Some(rewrites) = REWRITES.get() else {
        return false;
    };
    rewrite_code_with(c, stage, wave64, code, target, rewrites, enabled())
}

/// Applies any configured wave32 per-lane append rewrite to `native`.
pub(super) fn rewrite_program(c: &Container, native: &mut NativeProgram, target: Target) -> bool {
    rewrite_for_target(c, native.stage, native.wave64, &mut native.code, target)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SYNTHETIC_DIGEST: [u8; 16] = [
        0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xa0, 0xb0, 0xc0, 0xd0, 0xe0, 0xf0,
        0x01,
    ];
    const MOVE_ZERO: usize = 0x20;
    const NOP: usize = 0x24;
    const SYNTHETIC_APPENDS: &[AppendRegion] = &[
        AppendRegion {
            body: (0x40, 0x70),
            kept: &[(0x54, 0x5c), (0x64, 0x68)],
        },
        AppendRegion {
            body: (0x80, 0xc0),
            kept: &[(0x90, 0x98), (0x9c, 0xa4), (0xa8, 0xac), (0xb4, 0xb8)],
        },
    ];

    fn synthetic_code() -> Vec<u8> {
        let mut code = vec![0xaa; 256];
        code[MOVE_ZERO..MOVE_ZERO + 4].copy_from_slice(&0x7e0a_0280u32.to_le_bytes());
        code[NOP..NOP + 4].copy_from_slice(&0xbf80_0000u32.to_le_bytes());
        for (n, &(start, end)) in SYNTHETIC_APPENDS
            .iter()
            .flat_map(|a| a.kept.iter())
            .enumerate()
        {
            code[start..end].fill(n as u8 + 1);
        }
        code
    }

    fn synthetic_spec(code: &[u8]) -> AppendRewrite {
        AppendRewrite {
            container: SYNTHETIC_DIGEST,
            code_bytes: code.len(),
            code_fnv1a: fnv1a_64(code),
            move_zero: MOVE_ZERO,
            nop: NOP,
            appends: SYNTHETIC_APPENDS,
        }
    }

    fn synthetic_container(digest: [u8; 16]) -> Container<'static> {
        Container {
            digest,
            version: (1, 0),
            parts: Vec::new(),
        }
    }

    #[test]
    fn water_tile_env_switch_defaults_on_and_zero_disables() {
        assert!(parse_enabled(None));
        assert!(parse_enabled(Some(OsStr::new(""))));
        assert!(parse_enabled(Some(OsStr::new("1"))));
        assert!(parse_enabled(Some(OsStr::new("true"))));
        assert!(!parse_enabled(Some(OsStr::new("0"))));
        assert!(!parse_enabled(Some(OsStr::new(" 0 "))));
    }

    #[test]
    fn water_tile_rewrite_replaces_appends_and_preserves_layout() {
        let before = synthetic_code();
        let specs = [synthetic_spec(&before)];
        let container = synthetic_container(SYNTHETIC_DIGEST);
        let wave32 = Target {
            lanes_min: 32,
            lanes_max: 32,
            stride_table: false,
        };
        let mut code = before.clone();

        assert!(rewrite_code_with(
            &container,
            HwStage::Cs,
            true,
            &mut code,
            wave32,
            &specs,
            true
        ));
        assert_eq!(code.len(), before.len());

        let one = 0x7e0a_0281u32.to_le_bytes();
        let nop = 0xbf80_0000u32.to_le_bytes();

        // Region 0: move 1, kept marks 1..=2, NOP tail.
        assert_eq!(&code[0x40..0x44], &one);
        assert_eq!(&code[0x44..0x4c], &[1; 8]);
        assert_eq!(&code[0x4c..0x50], &[2; 4]);
        assert!(
            code[0x50..0x70]
                .as_chunks::<4>()
                .0
                .iter()
                .all(|&w| w == nop)
        );

        // Region 1: move 1, marks 3..=6, NOP tail.
        assert_eq!(&code[0x80..0x84], &one);
        assert_eq!(&code[0x84..0x8c], &[3; 8]);
        assert_eq!(&code[0x8c..0x94], &[4; 8]);
        assert_eq!(&code[0x94..0x98], &[5; 4]);
        assert_eq!(&code[0x98..0x9c], &[6; 4]);
        assert!(
            code[0x9c..0xc0]
                .as_chunks::<4>()
                .0
                .iter()
                .all(|&w| w == nop)
        );

        // Bytes outside the append regions stay untouched.
        assert_eq!(&code[..0x40], &before[..0x40]);
        assert_eq!(&code[0x70..0x80], &before[0x70..0x80]);
        assert_eq!(&code[0xc0..], &before[0xc0..]);

        // Re-running on already rewritten code is a no-op (FNV hash changed).
        assert!(!rewrite_code_with(
            &container,
            HwStage::Cs,
            true,
            &mut code,
            wave32,
            &specs,
            true
        ));
    }

    #[test]
    fn water_tile_rewrite_skips_wave64_hosts_opt_out_and_mismatches() {
        let before = synthetic_code();
        let specs = [synthetic_spec(&before)];
        let ok = synthetic_container(SYNTHETIC_DIGEST);
        let other = synthetic_container([0xff; 16]);
        let w32 = Target {
            lanes_min: 32,
            lanes_max: 32,
            stride_table: false,
        };
        let w64 = Target {
            lanes_min: 32,
            lanes_max: 64,
            stride_table: false,
        };
        let mut code = before.clone();

        // Wave64 host (AMD), opt-out (`XTP_WATER_TILE_SLOTS=0`), wrong container,
        // wrong stage, wave32 source, or altered bytes: left untouched.
        let cases = [
            (&ok, HwStage::Cs, true, w64, true),
            (&ok, HwStage::Cs, true, w32, false),
            (&other, HwStage::Cs, true, w32, true),
            (&ok, HwStage::Ps, true, w32, true),
            (&ok, HwStage::Cs, false, w32, true),
        ];
        for (c, stage, wave64, target, on) in cases {
            assert!(!rewrite_code_with(
                c, stage, wave64, &mut code, target, &specs, on
            ));
        }
        assert_eq!(code, before);
        code[0] ^= 1;
        assert!(!rewrite_code_with(
            &ok,
            HwStage::Cs,
            true,
            &mut code,
            w32,
            &specs,
            true
        ));
    }
}
