//! Vector ALU: the dispatch of VOP1/VOP2/VOP3/VOPC and DPP, float and
//! conversion operations and their helpers. Backlog S4 (non-fused
//! `v_mad_f32`), S5 (f64), M1 (`v_cndmask_b32` on
//! an untracked mask).

use std::collections::{HashMap, HashSet};

use super::mask::{LANE_MASK, MaskData};
use super::uniform::Level;
use super::{Lower, UNTRACKED_SELECT, Val, Why};
use crate::emit::bitcode::{Attr, Bin, Cast, V, pred};
use crate::emit::shader::flags;
use crate::gfx10::{Fields, NULL, Operand, Valu};
use crate::ir::{EXEC, Inst, M0, Program, Reg, Value};
use crate::resource::Site;
use crate::structure::{Atom, Bool, LaneExit, Structured};

const TWO_PI: f32 = std::f32::consts::TAU;

impl<'a> Lower<'a> {
    /// Source double précision, avec l’expansion propre aux littéraux flottants.
    fn double_words(&mut self, inst: &Inst, k: usize) -> Result<(V, V), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.sdwa.is_some() || v.dpp.is_some() {
            return Err("a 64-bit source with SDWA or DPP");
        }
        let op = v.src.get(k).ok_or("a missing operand")?;
        let zero = self.s.m.i32(0);
        let (lo, mut hi) = match *op {
            Operand::Vector { reg, .. } | Operand::Scalar { code: reg, .. }
                if !matches!(
                    op,
                    Operand::Scalar {
                        code: crate::gfx10::NULL,
                        ..
                    }
                ) =>
            {
                let high_reg = if matches!(op, Operand::Vector { .. }) {
                    Reg::V(reg + 1)
                } else {
                    Reg::S(reg + 1)
                };
                (
                    self.operand(inst, op)?,
                    self.i32_of(inst.read(high_reg).ok_or("an unread high word")?)?,
                )
            }
            Operand::Literal(bits) => (zero, self.s.m.i32(bits)),
            Operand::Const(248) => (self.s.m.i32(0x6dc9_c882), self.s.m.i32(0x3fc4_5f30)),
            Operand::Const(n @ 240..=247) => {
                let f = f32::from_bits(crate::resource::inline(n).unwrap());
                (zero, self.s.m.i32((f64::from(f).to_bits() >> 32) as u32))
            }
            Operand::Const(193..=208) => (self.operand(inst, op)?, self.s.m.i32(u32::MAX)),
            _ => (self.operand(inst, op)?, zero),
        };
        if v.abs >> k & 1 != 0 {
            let mask = self.s.m.i32(0x7fff_ffff);
            hi = self.s.m.bin(self.cur, Bin::And, hi, mask);
        }
        if v.neg >> k & 1 != 0 {
            let sign = self.s.m.i32(1 << 31);
            hi = self.s.m.bin(self.cur, Bin::Xor, hi, sign);
        }
        Ok((lo, hi))
    }

    /// La conversion en entiers de 32 bits garde la charge utile des NaN et
    /// applique MODE même sur un hôte dépourvu d’opérations en double précision.
    fn narrow_double(&mut self, lo: V, hi: V, mode: V) -> V {
        let c = self.cur;
        let m = &mut self.s.m;
        let imm = |m: &mut crate::emit::bitcode::Module, op, x, k| {
            let k = m.i32(k);
            m.bin(c, op, x, k)
        };
        let zero = m.i32(0);
        let one = m.i32(1);
        let sign = imm(m, Bin::And, hi, 0x8000_0000);
        let mag = imm(m, Bin::And, hi, 0x7fff_ffff);
        let exp = imm(m, Bin::LShr, mag, 20);
        let fraction = imm(m, Bin::And, hi, 0x000f_ffff);

        // 23 bits de fraction, puis trois bits pour l’arrondi. Le dernier
        // vaut un si un des bits moins significatifs était non nul.
        let top = imm(m, Bin::Shl, fraction, 6);
        let low = imm(m, Bin::LShr, lo, 26);
        let frac = m.bin(c, Bin::Or, top, low);
        let lost = imm(m, Bin::And, lo, (1 << 26) - 1);
        let sticky = m.cmp(c, pred::NE, lost, zero);
        let sticky = m.cast(c, Cast::ZExt, sticky, self.s.t.i32);
        let sig = m.bin(c, Bin::Or, frac, sticky);
        let normal = m.cmp(c, pred::NE, exp, zero);
        let implicit = m.i32(1 << 26);
        let implicit = m.select(c, normal, implicit, zero);
        let sig = m.bin(c, Bin::Or, sig, implicit);
        let dp_denorm = imm(m, Bin::And, mode, 1 << 6);
        let preserve = m.cmp(c, pred::NE, dp_denorm, zero);
        let keep = m.bin(c, Bin::Or, normal, preserve);
        let sig = m.select(c, keep, sig, zero);

        // 897 = 1023 - 126, exposant du plus petit f32 normal en notation
        // f64. Un décalage de 27 suffit pour ne garder que le bit collant.
        let min_exp = m.i32(897);
        let subnormal = m.cmp(c, pred::IULT, exp, min_exp);
        let delta = m.bin(c, Bin::Sub, min_exp, exp);
        let cap = m.i32(27);
        let large = m.cmp(c, pred::IUGT, delta, cap);
        let shift = m.select(c, large, cap, delta);
        let shift = m.select(c, subnormal, shift, zero);
        let shifted = m.bin(c, Bin::LShr, sig, shift);
        let back = m.bin(c, Bin::Shl, shifted, shift);
        let sticky = m.cmp(c, pred::NE, back, sig);
        let sticky = m.cast(c, Cast::ZExt, sticky, self.s.t.i32);
        let aligned = m.bin(c, Bin::Or, shifted, sticky);
        let mant = imm(m, Bin::LShr, aligned, 3);
        let tail = imm(m, Bin::And, aligned, 7);

        let odd = imm(m, Bin::And, mant, 1);
        let tail_odd = m.bin(c, Bin::Or, tail, odd);
        let half = m.i32(4);
        let nearest = m.cmp(c, pred::IUGT, tail_odd, half);
        let round = imm(m, Bin::And, mode, 3);
        let even = m.cmp(c, pred::EQ, round, zero);
        let negative = m.cmp(c, pred::NE, sign, zero);
        let two = m.i32(2);
        let away_mode = m.select(c, negative, two, one);
        let away = m.cmp(c, pred::EQ, round, away_mode);
        let inexact = m.cmp(c, pred::NE, tail, zero);
        let directed = m.bin(c, Bin::And, away, inexact);
        let increment = m.select(c, even, nearest, directed);
        let increment = m.cast(c, Cast::ZExt, increment, self.s.t.i32);
        let rounded = m.bin(c, Bin::Add, mant, increment);
        let exp32 = m.bin(c, Bin::Sub, exp, min_exp);
        let exp32 = m.select(c, subnormal, zero, exp32);
        let bias = imm(m, Bin::Shl, exp32, 23);
        let finite = m.bin(c, Bin::Add, bias, rounded);

        let inf = m.i32(0x7f80_0000);
        let max = m.i32(0x7f7f_ffff);
        let to_inf = m.bin(c, Bin::Or, even, away);
        let overflow = m.select(c, to_inf, inf, max);
        let max_exp = m.i32(1023 + 127);
        let too_large = m.cmp(c, pred::IUGT, exp, max_exp);
        let finite = m.select(c, too_large, overflow, finite);
        let min_normal = m.i32(0x0080_0000);
        let tiny = m.cmp(c, pred::IULT, finite, min_normal);
        let sp_denorm = imm(m, Bin::And, mode, 1 << 5);
        let flush = m.cmp(c, pred::EQ, sp_denorm, zero);
        let flush = m.bin(c, Bin::And, flush, tiny);
        let finite = m.select(c, flush, zero, finite);

        let payload = imm(m, Bin::LShr, frac, 3);
        let nan = imm(m, Bin::Or, payload, 0x7fc0_0000);
        let fraction_bits = m.bin(c, Bin::Or, fraction, lo);
        let is_nan = m.cmp(c, pred::NE, fraction_bits, zero);
        let special = m.select(c, is_nan, nan, inf);
        let exp_all = m.i32(0x7ff);
        let all_ones = m.cmp(c, pred::EQ, exp, exp_all);
        let bits = m.select(c, all_ones, special, finite);
        m.bin(c, Bin::Or, bits, sign)
    }

    pub(super) fn valu(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        if self.x.res.waterfall.compares.contains(&(b, i)) {
            // The first lane's dwords of a waterfall loop are each lane's
            // own (see `resource::Waterfall`).
            self.lane(inst, Some(true));
            return Ok(());
        }
        if let Some(rest) = name.strip_prefix("v_cmp_").or(name.strip_prefix("v_cmpx_")) {
            let r = if v.dpp.is_some() {
                Err("a compare with DPP")
            } else {
                self.compare(inst, rest)
            };
            if let Err(why) = r {
                for &(_, w) in &inst.writes {
                    self.lane_fail.insert(w, why);
                }
            }
            return r;
        }
        let mut guard = self.guard(b, i)?;
        if let Some(d) = v.dpp {
            let (x, enable) = self.dpp(inst, d)?;
            guard = match (guard, enable) {
                (Some(g), Some(e)) => Some(self.s.m.bin(self.cur, Bin::And, g, e)),
                (g, e) => g.or(e),
            };
            self.src0 = Some(x);
        }
        let r = self.valu_op(inst, guard);
        self.src0 = None;
        r
    }

    pub(super) fn valu_op(&mut self, inst: &Inst, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        let c = self.cur;
        let f = |l: &mut Self, k| l.fsrc(inst, k);
        let s = |l: &mut Self, k| l.src(inst, k);
        let float = |l: &mut Self, r: V| l.write_f(inst, r, guard);
        let int = |l: &mut Self, r: V| l.write_i(inst, r, guard);
        if let Some(r) = self.valu16(inst, name, guard) {
            return r;
        }
        if let Some(op) = name.strip_prefix("v_pk_") {
            return self.packed16(inst, op, guard);
        }
        match name {
            "v_cvt_f32_f64" => {
                let (lo, hi) = self.double_words(inst, 0)?;
                let mode = self.i32_of(
                    inst.read(Reg::FloatMode)
                        .ok_or("a conversion's float mode")?,
                )?;
                let bits = self.narrow_double(lo, hi, mode);
                if v.omod == 0 && !v.clamp {
                    int(self, bits)
                } else {
                    let result = self.as_float(bits);
                    float(self, result)
                }
            }
            "v_lshlrev_b64" | "v_lshrrev_b64" | "v_ashrrev_i64" => {
                let n = s(self, 0)?;
                let x = self.src64(inst, 1, name == "v_ashrrev_i64")?;
                let i64t = self.s.m.int(64);
                let k63 = self.s.m.i32(63);
                let n = self.s.m.bin(c, Bin::And, n, k63);
                let n = self.s.m.cast(c, Cast::ZExt, n, i64t);
                let op = match name {
                    "v_lshlrev_b64" => Bin::Shl,
                    "v_lshrrev_b64" => Bin::LShr,
                    _ => Bin::AShr,
                };
                let r = self.s.m.bin(c, op, x, n);
                self.write64(inst, v.dst, r, guard)
            }
            "v_mad_i64_i32" | "v_mad_u64_u32" => {
                if v.clamp || v.omod != 0 {
                    return Err("integer output modifiers");
                }
                let signed = name == "v_mad_i64_i32";
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let acc = self.src64(inst, 2, signed)?;
                let product = self.wide_product(a, b, signed);
                let sum = self.s.m.bin(c, Bin::Add, product, acc);
                self.write64(inst, v.dst, sum, guard)?;
                if let Some(Operand::Scalar { code, .. }) = v.carry
                    && let Some(w) = inst.write(Reg::S(code))
                {
                    let mut carry = self.s.m.cmp(c, pred::IULT, sum, product);
                    if signed {
                        // Bit 64 of the exact sum also includes the two sign bits
                        // (Marvisak/VIctory#171, 6b5e0b4, by n-romanovich).
                        let signs = self.s.m.bin(c, Bin::Xor, product, acc);
                        let i64t = self.s.m.int(64);
                        let zero = self.s.m.int_const(i64t, 0);
                        let sign = self.s.m.cmp(c, pred::SLT, signs, zero);
                        carry = self.s.m.bin(c, Bin::Xor, carry, sign);
                    }
                    self.lane_bits.insert(w, carry);
                }
                Ok(())
            }
            "v_cubeid_f32" | "v_cubesc_f32" | "v_cubetc_f32" | "v_cubema_f32" => {
                // Cube-map face coordinates (AMD's ISA): the major axis is z
                // when |z| >= |x| and |z| >= |y|, else y when |y| >= |x|,
                // else x.
                let (x, y, z) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let (ax, ay, az) = (self.unary(6, x), self.unary(6, y), self.unary(6, z));
                let zx = self.s.m.cmp(c, pred::OGE, az, ax);
                let zy = self.s.m.cmp(c, pred::OGE, az, ay);
                let zmaj = self.s.m.bin(c, Bin::And, zx, zy);
                let ymaj = self.s.m.cmp(c, pred::OGE, ay, ax);
                let zero = self.s.m.f32(0);
                let neg = |l: &mut Self, v: V| l.s.m.cmp(c, pred::OLT, v, zero);
                let minus = |l: &mut Self, v: V| {
                    let z = l.s.m.f32(0x8000_0000);
                    l.s.m.bin(c, Bin::Sub, z, v)
                };
                let pick = |l: &mut Self, zv: V, yv: V, xv: V| {
                    let t = l.s.m.select(c, ymaj, yv, xv);
                    l.s.m.select(c, zmaj, zv, t)
                };
                let r = match name {
                    "v_cubeid_f32" => {
                        let k = |l: &mut Self, v: f32| l.s.m.f32(v.to_bits());
                        let (nz, ny, nx) = (neg(self, z), neg(self, y), neg(self, x));
                        let (f5, f4, f3, f2, f1, f0) = (
                            k(self, 5.0),
                            k(self, 4.0),
                            k(self, 3.0),
                            k(self, 2.0),
                            k(self, 1.0),
                            k(self, 0.0),
                        );
                        let zf = self.s.m.select(c, nz, f5, f4);
                        let yf = self.s.m.select(c, ny, f3, f2);
                        let xf = self.s.m.select(c, nx, f1, f0);
                        pick(self, zf, yf, xf)
                    }
                    "v_cubesc_f32" => {
                        let (nz, nx) = (neg(self, z), neg(self, x));
                        let (mx, mz) = (minus(self, x), minus(self, z));
                        let zf = self.s.m.select(c, nz, mx, x);
                        let xf = self.s.m.select(c, nx, z, mz);
                        pick(self, zf, x, xf)
                    }
                    "v_cubetc_f32" => {
                        let ny = neg(self, y);
                        let (my, mz) = (minus(self, y), minus(self, z));
                        let yf = self.s.m.select(c, ny, mz, z);
                        pick(self, my, yf, my)
                    }
                    _ => {
                        let t = pick(self, z, y, x);
                        let two = self.s.m.f32(2.0f32.to_bits());
                        self.s.m.bin(c, Bin::Mul, t, two)
                    }
                };
                float(self, r)
            }
            "v_movrels_b32" | "v_movreld_b32" | "v_movrelsd_b32" => {
                // M0 indexes the source (movrels), the destination (movreld)
                // or both, from the encoded register up to the program's
                // VGPR count (the registers the instruction reads or writes).
                let m0 = self.i32_of(inst.read(Reg::S(crate::ir::M0)).ok_or("M0")?)?;
                let (
                    Some(&Operand::Vector { reg: sreg, .. }),
                    Some(Operand::Vector { reg: dreg, .. }),
                ) = (v.src.first(), v.dst)
                else {
                    return Err("an M0-relative move");
                };
                let indexed_src = name != "v_movreld_b32";
                let value = if indexed_src {
                    let mut r = self.s.m.i32(0);
                    for k in 0..=255u8.saturating_sub(sreg) {
                        let Some(x) = inst.read(Reg::V(sreg + k)) else {
                            break;
                        };
                        let x = self.i32_of(x)?;
                        let kk = self.s.m.i32(u32::from(k));
                        let is = self.s.m.cmp(c, pred::EQ, m0, kk);
                        r = self.s.m.select(c, is, x, r);
                    }
                    r
                } else {
                    s(self, 0)?
                };
                if name == "v_movrels_b32" {
                    return int(self, value);
                }
                for k in 0..=255u8.saturating_sub(dreg) {
                    let Some(_) = inst.write(Reg::V(dreg + k)) else {
                        break;
                    };
                    let old = match inst.read(Reg::V(dreg + k)) {
                        Some(o) => self.i32_of(o)?,
                        None => self.s.m.i32(0),
                    };
                    let kk = self.s.m.i32(u32::from(k));
                    let is = self.s.m.cmp(c, pred::EQ, m0, kk);
                    let x = self.s.m.select(c, is, value, old);
                    self.write_placed(inst, Reg::V(dreg + k), x, guard);
                }
                Ok(())
            }
            "v_swap_b32" => {
                let (
                    Some(&Operand::Vector { reg: sreg, .. }),
                    Some(Operand::Vector { reg: dreg, .. }),
                ) = (v.src.first(), v.dst)
                else {
                    return Err("v_swap_b32");
                };
                let a = self.i32_of(inst.read(Reg::V(dreg)).ok_or("v_swap_b32")?)?;
                let b = self.i32_of(inst.read(Reg::V(sreg)).ok_or("v_swap_b32")?)?;
                self.write_placed(inst, Reg::V(dreg), b, guard);
                if dreg != sreg {
                    self.write_placed(inst, Reg::V(sreg), a, guard);
                }
                Ok(())
            }
            "v_ffbh_u32" | "v_ffbh_i32" => {
                // Counted from the top bit, -1 when none (DXIL's too).
                let a = s(self, 0)?;
                let op = if name == "v_ffbh_u32" { 33 } else { 34 };
                let r = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, op, &[a]);
                int(self, r)
            }
            "v_mov_b32" => {
                if v.abs != 0 || v.neg != 0 || v.clamp || v.omod != 0 {
                    let a = f(self, 0)?;
                    return float(self, a);
                }
                let a = match s(self, 0) {
                    // shortcut: 0 for a descriptor's address or format
                    // copied into an element of an M0-indexed array that
                    // nothing but the array's moves reads, where every
                    // `v_movrels_b32` that may read it has M0 proven below
                    // it (`unused_element`, `Spans`).
                    Err(why)
                        if why == super::smem::DESCRIPTOR_WORD && self.unused_element(inst) =>
                    {
                        self.s.m.i32(0)
                    }
                    a => a?,
                };
                int(self, a)
            }
            "v_add_f32" | "v_sub_f32" | "v_subrev_f32" | "v_mul_f32" | "v_mul_legacy_f32"
            | "v_max_f32" | "v_min_f32" => {
                let (a, bb) = (f(self, 0)?, f(self, 1)?);
                let r = match name {
                    "v_add_f32" => self.s.m.bin(c, Bin::Add, a, bb),
                    "v_sub_f32" => self.s.m.bin(c, Bin::Sub, a, bb),
                    "v_subrev_f32" => self.s.m.bin(c, Bin::Sub, bb, a),
                    "v_mul_f32" => self.s.m.bin(c, Bin::Mul, a, bb),
                    "v_mul_legacy_f32" => self.legacy_mul(a, bb),
                    "v_max_f32" => self.max_min(true, a, bb),
                    _ => self.max_min(false, a, bb),
                };
                float(self, r)
            }
            "v_fma_f32" | "v_fmaak_f32" | "v_fmamk_f32" => {
                // shortcut: DXIL has no fused f32 multiply-add; FMad lets the driver fuse.
                let (a, bb, cc) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let r = self.fmad(a, bb, cc);
                float(self, r)
            }
            "v_fmac_f32" | "v_mac_f32" | "v_mac_legacy_f32" => {
                let (a, bb) = (f(self, 0)?, f(self, 1)?);
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("an accumulator");
                };
                let acc = inst.read(Reg::V(reg)).ok_or("an accumulator")?;
                let mut acc = match self.floats.get(&acc) {
                    Some(&x) => x,
                    None => {
                        let x = self.i32_of(acc)?;
                        self.as_float(x)
                    }
                };
                // VOP3 encodes modifiers for the implicit accumulator in slot 2 (`abs[2]` /
                // `neg[2]`). Ported from TaynanGT/Vice-Tay commit 8fe3ccdd (upstream cf8f543c,
                // author Bartosz, GPL-3.0-or-later).
                if v.abs >> 2 & 1 != 0 {
                    acc = self.unary(6, acc);
                }
                if v.neg >> 2 & 1 != 0 {
                    let z = self.s.m.f32(0x8000_0000);
                    acc = self.s.m.bin(c, Bin::Sub, z, acc);
                }
                let r = match name {
                    "v_fmac_f32" => self.fmad(a, bb, acc),
                    "v_mac_f32" => {
                        let m = self.s.m.bin(c, Bin::Mul, a, bb);
                        self.s.m.bin(c, Bin::Add, m, acc)
                    }
                    _ => {
                        let m = self.legacy_mul(a, bb);
                        self.s.m.bin(c, Bin::Add, m, acc)
                    }
                };
                float(self, r)
            }
            "v_mad_f32" | "v_madak_f32" | "v_madmk_f32" | "v_mad_legacy_f32" => {
                let (a, bb, cc) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let m = if name == "v_mad_legacy_f32" {
                    self.legacy_mul(a, bb)
                } else {
                    self.s.m.bin(c, Bin::Mul, a, bb)
                };
                let r = self.s.m.bin(c, Bin::Add, m, cc);
                float(self, r)
            }
            "v_max3_f32" | "v_min3_f32" | "v_med3_f32" => {
                let (a, bb, cc) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let r = match name {
                    "v_max3_f32" => {
                        let t = self.max_min(true, a, bb);
                        self.max_min(true, t, cc)
                    }
                    "v_min3_f32" => {
                        let t = self.max_min(false, a, bb);
                        self.max_min(false, t, cc)
                    }
                    _ => self.med3(a, bb, cc),
                };
                float(self, r)
            }
            "v_rcp_f32" | "v_rcp_iflag_f32" => {
                // shortcut: the hardware reciprocal is approximate (1 ULP); this divides.
                let a = f(self, 0)?;
                let one = self.s.m.f32(1.0f32.to_bits());
                let r = self.s.m.bin(c, Bin::SDiv, one, a);
                float(self, r)
            }
            "v_rsq_f32" | "v_sqrt_f32" | "v_log_f32" | "v_exp_f32" | "v_fract_f32"
            | "v_trunc_f32" | "v_floor_f32" | "v_ceil_f32" | "v_rndne_f32" | "v_sin_f32"
            | "v_cos_f32" => {
                let a = f(self, 0)?;
                let op = match name {
                    "v_rsq_f32" => 25,
                    "v_sqrt_f32" => 24,
                    "v_log_f32" => 23,
                    "v_exp_f32" => 21,
                    "v_fract_f32" => 22,
                    "v_trunc_f32" => 29,
                    "v_floor_f32" => 27,
                    "v_ceil_f32" => 28,
                    "v_rndne_f32" => 26,
                    "v_sin_f32" => 13,
                    _ => 12,
                };
                let r = match op {
                    12 | 13 => self.trig_turns(a, op == 12),
                    22 => self.fract(a, 1.0, f32::from_bits(0x3F7F_FFFF)),
                    _ => self.unary(op, a),
                };
                float(self, r)
            }
            "v_cvt_i32_f32" | "v_cvt_u32_f32" | "v_cvt_flr_i32_f32" | "v_cvt_rpi_i32_f32" => {
                let mut a = f(self, 0)?;
                if name == "v_cvt_rpi_i32_f32" {
                    let h = self.s.m.f32(0.5f32.to_bits());
                    a = self.s.m.bin(c, Bin::Add, a, h);
                }
                if name != "v_cvt_i32_f32" && name != "v_cvt_u32_f32" {
                    a = self.unary(27, a);
                }
                // D3D's conversions saturate and give 0 for NaN, as the
                // hardware does (checked on the RX 6700 XT and WARP).
                let op = if name == "v_cvt_u32_f32" {
                    Cast::FpToUi
                } else {
                    Cast::FpToSi
                };
                let r = self.s.m.cast(c, op, a, self.s.t.i32);
                int(self, r)
            }
            "v_cvt_f32_i32" | "v_cvt_f32_u32" | "v_cvt_f32_ubyte0" | "v_cvt_f32_ubyte1"
            | "v_cvt_f32_ubyte2" | "v_cvt_f32_ubyte3" => {
                let mut a = s(self, 0)?;
                if let Some(n) = name.strip_prefix("v_cvt_f32_ubyte") {
                    let sh = self.s.m.i32(8 * u32::from(n.as_bytes()[0] - b'0'));
                    let ff = self.s.m.i32(0xff);
                    a = self.s.m.bin(c, Bin::LShr, a, sh);
                    a = self.s.m.bin(c, Bin::And, a, ff);
                }
                let op = if name == "v_cvt_f32_i32" {
                    Cast::SiToFp
                } else {
                    Cast::UiToFp
                };
                let r = self.s.m.cast(c, op, a, self.s.t.f32);
                float(self, r)
            }
            "v_fma_mix_f32" | "v_fma_mixlo_f16" | "v_fma_mixhi_f16" => {
                // Each source is f32, or with op_sel_hi the low (op_sel: high)
                // f16 half; neg_hi is the absolute value here.
                let mut xs = Vec::new();
                for i in 0..3 {
                    let op = v.src.get(i).ok_or("a missing operand")?;
                    let x = self.operand(inst, op)?;
                    let mut x = if v.opsel_hi >> i & 1 != 0 {
                        let h = if v.opsel >> i & 1 != 0 {
                            let k16 = self.s.m.i32(16);
                            self.s.m.bin(c, Bin::LShr, x, k16)
                        } else {
                            x
                        };
                        self.s
                            .op(c, "legacyF16ToF32", Attr::ReadNone, self.s.t.f32, 131, &[h])
                    } else {
                        self.as_float(x)
                    };
                    if v.neg_hi >> i & 1 != 0 {
                        x = self.unary(6, x);
                    }
                    if v.neg >> i & 1 != 0 {
                        let z = self.s.m.f32(0x8000_0000);
                        x = self.s.m.bin(c, Bin::Sub, z, x);
                    }
                    xs.push(x);
                }
                let mut r = self.fmad(xs[0], xs[1], xs[2]);
                if name == "v_fma_mix_f32" {
                    return float(self, r);
                }
                if v.clamp {
                    r = self.unary(7, r);
                }
                // The f16 result goes to one half; the other half stays
                // (GFX10). It is merged as the other 16-bit writes merge it
                // (`place_bits`: SDWA word 1 or 0, the other word kept): an
                // old value the host cannot supply reads as 0 there. A
                // vertex shader's v2 holds the primitive id on entry, and
                // the console compiler may write a half over it; a read of
                // that id as a whole is still refused.
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a mixed-precision result");
                };
                let h = self
                    .s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, self.s.t.i32, 130, &[r]);
                let word = if name == "v_fma_mixhi_f16" { 5 } else { 4 };
                let x = self.place_bits(inst, Reg::V(reg), h, word, 2);
                self.write_placed(inst, Reg::V(reg), x, guard);
                Ok(())
            }
            "v_cvt_pkrtz_f16_f32" => {
                let (a, bb) = (f(self, 0)?, f(self, 1)?);
                let lo = self.f16_toward_zero(a);
                let hi = self.f16_toward_zero(bb);
                let sh = self.s.m.i32(16);
                let hi = self.s.m.bin(c, Bin::Shl, hi, sh);
                let r = self.s.m.bin(c, Bin::Or, lo, hi);
                int(self, r)
            }
            "v_cvt_f16_f32" if v.opsel == 0 => {
                // Rounded to nearest even (the MODE default) into the low half.
                let a = f(self, 0)?;
                let r = self
                    .s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, self.s.t.i32, 130, &[a]);
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a 16-bit result outside a VGPR");
                };
                let r = self.low_half(inst, reg, r);
                int(self, r)
            }
            "v_pack_b32_f16" => {
                // Two f16 sources (op_sel: their high halves; abs and neg on
                // the sign bit) into the low and high half.
                let mut halves = Vec::new();
                for k in 0..2 {
                    let op = v.src.get(k).ok_or("a missing operand")?;
                    let mut x = self.operand(inst, op)?;
                    if v.opsel >> k & 1 != 0 {
                        let k16 = self.s.m.i32(16);
                        x = self.s.m.bin(c, Bin::LShr, x, k16);
                    }
                    let mask = if v.abs >> k & 1 != 0 { 0x7FFF } else { 0xFFFF };
                    let mask = self.s.m.i32(mask);
                    x = self.s.m.bin(c, Bin::And, x, mask);
                    if v.neg >> k & 1 != 0 {
                        let sign = self.s.m.i32(0x8000);
                        x = self.s.m.bin(c, Bin::Xor, x, sign);
                    }
                    halves.push(x);
                }
                let k16 = self.s.m.i32(16);
                let hi = self.s.m.bin(c, Bin::Shl, halves[1], k16);
                let r = self.s.m.bin(c, Bin::Or, halves[0], hi);
                int(self, r)
            }
            "v_cvt_f32_f16" if v.opsel == 0 => {
                let a = s(self, 0)?;
                let r = self
                    .s
                    .op(c, "legacyF16ToF32", Attr::ReadNone, self.s.t.f32, 131, &[a]);
                float(self, r)
            }
            "v_cndmask_b32" => {
                let mask = match v.src.get(2) {
                    Some(&Operand::Scalar { code, .. }) => inst.read(Reg::S(code)),
                    _ => None,
                };
                // A selector that derives from neither EXEC nor a compare
                // (a constant, a number built from scalars) is a mask held as
                // a number. The in-half form cannot test the high word, so
                // defer the select there and retry only if its result is used.
                let numeric_mask = self.console_lanes && (!self.split || self.console_cf);
                let m = mask
                    .and_then(|m| {
                        let others = numeric_mask.then(|| self.x.st.others.get(&m));
                        self.x.st.masks.get(&m).or(others.flatten()).copied()
                    })
                    .ok_or(UNTRACKED_SELECT)?;
                let cond = self.boolean(m, c)?;
                if v.abs != 0 || v.neg != 0 {
                    let (a, bb) = (f(self, 0)?, f(self, 1)?);
                    let r = self.s.m.select(c, cond, bb, a);
                    return float(self, r);
                }
                let (a, bb) = (s(self, 0)?, s(self, 1)?);
                let r = self.s.m.select(c, cond, bb, a);
                int(self, r)
            }
            "v_add_co_u32" | "v_sub_co_u32" | "v_subrev_co_u32" | "v_add_co_ci_u32"
            | "v_sub_co_ci_u32" | "v_subrev_co_ci_u32" => {
                // The carry (borrow) out is a lane mask: this lane's bit.
                let (mut a, mut bb) = (s(self, 0)?, s(self, 1)?);
                if name.contains("subrev") {
                    std::mem::swap(&mut a, &mut bb);
                }
                let cin = if name.ends_with("_ci_u32") {
                    let m = match v.src.get(2) {
                        Some(&Operand::Scalar { code, .. }) => inst.read(Reg::S(code)),
                        _ => None,
                    };
                    let m = m
                        .and_then(|m| self.x.st.masks.get(&m).copied())
                        .ok_or("a carry-in from an untracked mask")?;
                    Some(self.boolean(m, c)?)
                } else {
                    None
                };
                let (r, carry) = self.add_sub_carry(c, name.starts_with("v_add"), a, bb, cin);
                int(self, r)?;
                if let Some(Operand::Scalar { code, .. }) = v.carry
                    && let Some(w) = inst.write(Reg::S(code))
                {
                    self.lane_bits.insert(w, carry);
                }
                for &(_, w) in &inst.writes {
                    self.vals
                        .entry(w)
                        .or_insert(Val::Missing("a lane mask read as data"));
                }
                Ok(())
            }
            "v_permlanex16_b32"
                if self.s.m.paired_lanes() && (guard.is_none() || v.opsel & 1 != 0) =>
            {
                // The 16 selector nibbles index the opposite row of this
                // 32-lane half. Every source is live under full EXEC or FI.
                let (x, sel0, sel1) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let lane = self.lane_index();
                let (seven, eight, fifteen, sixteen, two, zero) = (
                    self.s.m.i32(7),
                    self.s.m.i32(8),
                    self.s.m.i32(15),
                    self.s.m.i32(16),
                    self.s.m.i32(2),
                    self.s.m.i32(0),
                );
                let word = self.s.m.bin(c, Bin::And, lane, eight);
                let low = self.s.m.cmp(c, pred::EQ, word, zero);
                let sel = self.s.m.select(c, low, sel0, sel1);
                let nibble = self.s.m.bin(c, Bin::And, lane, seven);
                let shift = self.s.m.bin(c, Bin::Shl, nibble, two);
                let pick = self.s.m.bin(c, Bin::LShr, sel, shift);
                let pick = self.s.m.bin(c, Bin::And, pick, fifteen);
                let row = self.s.m.bin(c, Bin::And, lane, sixteen);
                let other = self.s.m.bin(c, Bin::Xor, row, sixteen);
                let at = self.s.m.bin(c, Bin::Or, other, pick);
                let r = self.read_lane(x, at);
                int(self, r)
            }
            "v_readlane_b32" => {
                let (a, lane) = (s(self, 0)?, s(self, 1)?);
                // A computed lane in a pixel or vertex program is a host lane
                // (from a mask or `v_mbcnt`); a constant one is the console's.
                if self.mask_data == MaskData::Host && !matches!(lane, V::Const(_)) {
                    self.s.flags |= flags::WAVE_OPS;
                } else {
                    self.cross_lane(false)?;
                }
                let r = if self.console_cf {
                    self.console_read_lane(a, lane)
                } else {
                    self.read_lane(a, lane)
                };
                int(self, r)
            }
            "v_writelane_b32" => {
                // The lane given takes the scalar; the others keep their value.
                let (a, lane) = (s(self, 0)?, s(self, 1)?);
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a lane write");
                };
                let old = match inst.read(Reg::V(reg)) {
                    Some(o) => self.i32_of(o)?,
                    None => self.s.m.i32(0),
                };
                self.cross_lane(false)?;
                let me = if self.console_cf {
                    self.console_lane().0
                } else {
                    self.lane_index()
                };
                let is = self.s.m.cmp(c, pred::EQ, me, lane);
                let r = self.s.m.select(c, is, a, old);
                int(self, r)
            }
            "v_readfirstlane_b32"
                if inst
                    .writes
                    .first()
                    .is_some_and(|w| self.x.res.waterfall.reads.contains(&w.1)) =>
            {
                // A waterfall loop's read of a descriptor dword: the lane's own.
                let a = s(self, 0)?;
                int(self, a)
            }
            "v_readfirstlane_b32"
                if self.s.mesh.is_some()
                    && self.mesh_first_lane_uniform(inst.writes.first().map(|w| w.1)) =>
            {
                // The same in every lane of the console wave (`uniform.rs`):
                // the lane's own value is the first lane's, whatever waves
                // the host forms.
                let a = s(self, 0)?;
                int(self, a)
            }
            "v_readfirstlane_b32"
                if self.mesh_halves()
                    && self
                        .uniform
                        .as_ref()
                        .is_some_and(|u| u.control(self.cur_ir) != Level::Group)
                    && self.uniform_test_first_lane(inst, self.cur_ir) =>
            {
                // The first-lane read of a test for a value the whole wave
                // holds, where lanes of the group reach it differently (no
                // exchange runs there): each lane's own value, exact per lane
                // (`mask.rs`).
                let a = s(self, 0)?;
                int(self, a)
            }
            "v_readfirstlane_b32"
                if self.mesh_halves()
                    && inst
                        .writes
                        .first()
                        .is_some_and(|w| self.x.res.waterfall.indices.contains(&w.1)) =>
            {
                // Every lane uses its own VGPR index in the proven loop.
                let a = s(self, 0)?;
                int(self, a)
            }
            "v_readfirstlane_b32" if self.mesh_halves() => {
                // The console wave's first lane, which the host wave may not
                // hold: through group-shared memory (`mesh.rs`).
                let a = s(self, 0)?;
                let r = self.mesh_first_lane(a, guard)?;
                int(self, r)
            }
            "v_readfirstlane_b32" if self.console_cf => {
                // The console's first lane in EXEC (lane 0 when none).
                let a = s(self, 0)?;
                let word = |l: &mut Self, r: u8| -> Result<V, Why> {
                    match l.get(inst.read(Reg::S(r)).ok_or("EXEC")?)? {
                        Val::I32(x) => Ok(x),
                        _ => Err(LANE_MASK),
                    }
                };
                let lo = word(self, EXEC)?;
                let hi = if self.x.native.wave64 {
                    word(self, EXEC + 1)?
                } else {
                    self.s.m.i32(0)
                };
                let i32t = self.s.t.i32;
                let flo = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[lo]);
                let fhi = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[hi]);
                let (z, k32) = (self.s.m.i32(0), self.s.m.i32(32));
                let fhi = self.s.m.bin(c, Bin::Add, fhi, k32);
                let none_hi = self.s.m.cmp(c, pred::EQ, hi, z);
                let fhi = self.s.m.select(c, none_hi, z, fhi);
                let some_lo = self.s.m.cmp(c, pred::NE, lo, z);
                let first = self.s.m.select(c, some_lo, flo, fhi);
                let r = self.console_read_lane(a, first);
                int(self, r)
            }
            "v_readfirstlane_b32" => {
                let a = s(self, 0)?;
                let r = self.first_lane(inst, a)?;
                int(self, r)
            }
            _ => {
                let r = self.integer(inst, name)?;
                int(self, r)
            }
        }
    }

    pub(super) fn unary(&mut self, op: u32, f: V) -> V {
        self.s.op(
            self.cur,
            "unary.f32",
            Attr::ReadNone,
            self.s.t.f32,
            op,
            &[f],
        )
    }

    pub(super) fn binary(&mut self, op: u32, a: V, b: V) -> V {
        self.s.op(
            self.cur,
            "binary.f32",
            Attr::ReadNone,
            self.s.t.f32,
            op,
            &[a, b],
        )
    }

    /// The hardware's maximum (`max`) or minimum of two floats: as DXIL's
    /// FMax/FMin, but -0 orders below +0 (RDNA2 ISA, `v_max_f32`), where
    /// the compare and select DXIL specifies gives either zero, and a NaN
    /// operand gives the other one (two NaNs give NaN), chosen here: what
    /// hosts' FMax/FMin do with a NaN differs (an Intel UHD driver, issue
    /// #34). A host whose FMax/FMin already skip NaNs folds the selects.
    pub(super) fn max_min(&mut self, max: bool, a: V, b: V) -> V {
        let c = self.cur;
        let r = self.binary(if max { 35 } else { 36 }, a, b);
        let (ia, ib) = (self.as_int(a), self.as_int(b));
        let either = self.s.m.bin(c, Bin::Or, ia, ib);
        let magnitude = self.s.m.i32(0x7FFF_FFFF);
        let either = self.s.m.bin(c, Bin::And, either, magnitude);
        let zero = self.s.m.i32(0);
        let zeros = self.s.m.cmp(c, pred::EQ, either, zero);
        // Of two zeros, +0 unless both are -0 (max), -0 unless neither (min).
        let op = if max { Bin::And } else { Bin::Or };
        let sign = self.s.m.bin(c, op, ia, ib);
        let signed = self.as_float(sign);
        let r = self.s.m.select(c, zeros, signed, r);
        let nan_a = self.s.m.cmp(c, pred::UNO, a, a);
        let nan_b = self.s.m.cmp(c, pred::UNO, b, b);
        let r = self.s.m.select(c, nan_b, a, r);
        self.s.m.select(c, nan_a, b, r)
    }

    pub(super) fn fmad(&mut self, a: V, b: V, c: V) -> V {
        self.s.op(
            self.cur,
            "tertiary.f32",
            Attr::ReadNone,
            self.s.t.f32,
            46,
            &[a, b, c],
        )
    }

    /// `f` as an f16 in the low bits, rounded toward zero: the nearest-even
    /// conversion, one step back toward zero where it rounded away (which
    /// also turns an overflow to infinity into the largest finite half).
    pub(super) fn f16_toward_zero(&mut self, f: V) -> V {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let h = self
            .s
            .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f]);
        let back = self
            .s
            .op(c, "legacyF16ToF32", Attr::ReadNone, self.s.t.f32, 131, &[h]);
        let (fa, ba) = (self.unary(6, f), self.unary(6, back));
        // Ordered: never for NaN.
        let away = self.s.m.cmp(c, pred::OGT, ba, fa);
        let one = self.s.m.i32(1);
        let stepped = self.s.m.bin(c, Bin::Sub, h, one);
        self.s.m.select(c, away, stepped, h)
    }

    /// `(a >> off[4:0]) & ((1 << width) - 1)`, sign-extended from the field's
    /// top bit when `signed`; widths from 32 on take every bit. A signed
    /// field that runs past bit 31 extends from bit 31, as the hardware
    /// does: the shift is arithmetic then.
    pub(super) fn bfe(&mut self, a: V, off: V, width: V, signed: bool) -> V {
        let c = self.cur;
        let (k31, k32, zero, one, ones) = (
            self.s.m.i32(31),
            self.s.m.i32(32),
            self.s.m.i32(0),
            self.s.m.i32(1),
            self.s.m.i32(u32::MAX),
        );
        let off = self.s.m.bin(c, Bin::And, off, k31);
        let shift = if signed { Bin::AShr } else { Bin::LShr };
        let shifted = self.s.m.bin(c, shift, a, off);
        let full = self.s.m.cmp(c, pred::IUGE, width, k32);
        let w = self.s.m.bin(c, Bin::And, width, k31);
        let mask = self.s.m.bin(c, Bin::Shl, one, w);
        let mask = self.s.m.bin(c, Bin::Sub, mask, one);
        let mask = self.s.m.select(c, full, ones, mask);
        let r = self.s.m.bin(c, Bin::And, shifted, mask);
        if !signed {
            return r;
        }
        // Sign-extend a field narrower than 32 bits (none for width 0).
        let left = self.s.m.bin(c, Bin::Sub, k32, w);
        let left = self.s.m.bin(c, Bin::And, left, k31);
        let t = self.s.m.bin(c, Bin::Shl, r, left);
        let t = self.s.m.bin(c, Bin::AShr, t, left);
        let is0 = self.s.m.cmp(c, pred::EQ, w, zero);
        let keep = self.s.m.bin(c, Bin::Or, full, is0);
        self.s.m.select(c, keep, r, t)
    }

    /// The median of three. With a NaN among them the hardware gives the
    /// minimum of the others instead (min3), and NaN when all three are.
    /// Each NaN is replaced by an operand that is not one before that
    /// minimum, so it never meets a NaN: hosts' FMin differ there (an Intel
    /// UHD driver gave 2.0 for (1.0, 2.0, NaN), issue #34).
    pub(super) fn med3(&mut self, a: V, b: V, third: V) -> V {
        let c = self.cur;
        let lo = self.binary(36, a, b);
        let hi = self.binary(35, a, b);
        let t = self.binary(36, hi, third);
        let median = self.binary(35, lo, t);
        let (nan_a, nan_b, nan_c) = (
            self.s.m.cmp(c, pred::UNO, a, a),
            self.s.m.cmp(c, pred::UNO, b, b),
            self.s.m.cmp(c, pred::UNO, third, third),
        );
        // A number among the three if there is one: a, else b, else third.
        let b_or_c = self.s.m.select(c, nan_b, third, b);
        let some = self.s.m.select(c, nan_a, b_or_c, a);
        let (x, y, z) = (
            some,
            self.s.m.select(c, nan_b, some, b),
            self.s.m.select(c, nan_c, some, third),
        );
        let least = self.max_min(false, x, y);
        let least = self.max_min(false, least, z);
        let ab = self.s.m.bin(c, Bin::Or, nan_a, nan_b);
        let nan = self.s.m.bin(c, Bin::Or, ab, nan_c);
        self.s.m.select(c, nan, least, median)
    }

    /// `a - floor(a)`, which on the hardware never reaches 1.0: where it
    /// would round to it (a tiny negative `a`), the result is `below_one`,
    /// the largest value under 1.0 in the operation's width. Hosts differ
    /// here (some give 1.0). Comparing against a `threshold` distinct from
    /// `below_one` prevents drivers (observed on NVIDIA) from folding
    /// `r > k ? k : r` into `minNum` (`FMNMX`) and swallowing NaN.
    pub(super) fn fract(&mut self, a: V, threshold: f32, below_one: f32) -> V {
        let c = self.cur;
        let r = self.unary(22, a);
        let limit = self.s.m.f32(threshold.to_bits());
        let k = self.s.m.f32(below_one.to_bits());
        let over = self.s.m.cmp(c, pred::OGE, r, limit);
        self.s.m.select(c, over, k, r)
    }

    /// Sine, or cosine, of an angle in turns, as the hardware takes it. The
    /// whole turns are taken off first (`x - round_even(x)`, exact, within
    /// half a turn), so a large angle keeps its fraction; scaling to radians
    /// first rounds it away. A zero passes unreduced (the sine of -0 is -0).
    /// The sine of a half turn and the cosine of a quarter turn are exactly
    /// +0 on the hardware, where the host's function of the rounded angle
    /// gives about 1e-7. Infinities and NaN give NaN.
    /// shortcut: below 2^-12 turns the hardware scales by a slightly
    /// different constant than 2π (a last-place difference).
    pub(super) fn trig_turns(&mut self, x: V, cosine: bool) -> V {
        let c = self.cur;
        let zero = self.s.m.f32(0);
        let whole = self.unary(26, x);
        let part = self.s.m.bin(c, Bin::Sub, x, whole);
        let is_zero = self.s.m.cmp(c, pred::OEQ, x, zero);
        let part = self.s.m.select(c, is_zero, x, part);
        let k = self.s.m.f32(TWO_PI.to_bits());
        let angle = self.s.m.bin(c, Bin::Mul, part, k);
        let r = self.unary(if cosine { 12 } else { 13 }, angle);
        let at = if cosine { 0.25f32 } else { 0.5 };
        let at = self.s.m.f32(at.to_bits());
        let size = self.unary(6, part);
        let exact = self.s.m.cmp(c, pred::OEQ, size, at);
        self.s.m.select(c, exact, zero, r)
    }

    /// The complete 32 × 32-bit product.
    fn wide_product(&mut self, a: V, b: V, signed: bool) -> V {
        let c = self.cur;
        let i64t = self.s.m.int(64);
        let ext = if signed { Cast::SExt } else { Cast::ZExt };
        let a = self.s.m.cast(c, ext, a, i64t);
        let b = self.s.m.cast(c, ext, b, i64t);
        self.s.flags |= flags::INT64;
        self.s.m.bin(c, Bin::Mul, a, b)
    }

    /// The high dword of a 32 × 32-bit product.
    pub(super) fn mul_hi(&mut self, a: V, b: V, signed: bool) -> V {
        let c = self.cur;
        let m = self.wide_product(a, b, signed);
        let i64t = self.s.m.int(64);
        let k32 = self.s.m.int_const(i64t, 32);
        let hi = self.s.m.bin(c, Bin::LShr, m, k32);
        self.s.m.cast(c, Cast::Trunc, hi, self.s.t.i32)
    }

    /// `0 × anything = 0`, as in D3D9.
    pub(super) fn legacy_mul(&mut self, a: V, b: V) -> V {
        let c = self.cur;
        let m = self.s.m.bin(c, Bin::Mul, a, b);
        let zero = self.s.m.f32(0);
        let az = self.s.m.cmp(c, pred::OEQ, a, zero);
        let bz = self.s.m.cmp(c, pred::OEQ, b, zero);
        let z = self.s.m.bin(c, Bin::Or, az, bz);
        self.s.m.select(c, z, zero, m)
    }

    /// Whether `inst`, a plain `v_mov_b32` of a descriptor's dword into a
    /// VGPR, fills an element of an M0-indexed array (`v_movrels_b32`,
    /// `v_movreld_b32`) past its first, which nothing else reads. The
    /// console compiler fills such an array from a run of SGPRs; past the
    /// elements the program uses the run holds whatever is there, which can
    /// be a descriptor's address or format, and the moves name every VGPR
    /// from the array's first to the program's last. The light loop's
    /// preparation (11eeb27c2d15980d) fills v3..v18 from s0..s15, the
    /// last four a UAV's first dwords; M0 there is a loop counter that
    /// starts and ends in 0..11 (`v_med3_u32 …, 0, 11` at 0x234 and 0x23c)
    /// and leaves the loop past its end (`v_cmpx_gt_u32` at 0x47c), and
    /// v15..v18 are written again before anything else reads them.
    ///
    /// The copy is followed through joins, plain copies and the elements a
    /// `v_movreld_b32` keeps. Only its real operands are reads of an
    /// instruction: a VGPR write lists the register's old value among its
    /// reads ([`crate::ir`]), which is no use of it. That old value goes on
    /// in the new one where the write keeps it (lanes outside the guard,
    /// half writes: [`super::uniform::keeps_old`]), and refuses where the
    /// instruction computes with it (`v_fmac_f32`, `v_writelane_b32`). A
    /// `v_movrels_b32` reads it from an element past the first only where
    /// M0 is proven below that element in every lane that runs it
    /// ([`Spans`]): the guest would read the descriptor's dword where M0
    /// names it, and the translation the 0 that stands for it. Any other
    /// read (an operand, the array's first element, the value a
    /// `v_movreld_b32` stores, a non-VALU instruction) keeps the dword the
    /// program's: its read stops the program as before.
    // Ported from NuraadGit/VIctory (c1c3fbc) by NuraadGit, with the rule
    // it amends (518c8b9, same author). Here: only on a dword with no host
    // value (`DESCRIPTOR_WORD`, so a size the program reads stays its
    // query); operands are `source_regs`'s; a write keeps the old value
    // where its lowering does (`guard`, `keeps_old`); the copy is checked
    // where it fails, not for every copy of the program; M0 is bounded at
    // each indexed read instead of assumed never to name the element.
    fn unused_element(&self, inst: &Inst) -> bool {
        let p = self.x.p;
        let Fields::Valu(f) = &inst.fields else {
            return false;
        };
        if f.sdwa.is_some() || f.dpp.is_some() {
            return false;
        }
        let (Some(&Operand::Scalar { code, .. }), Some(Operand::Vector { reg, .. })) =
            (f.src.first(), f.dst)
        else {
            return false;
        };
        let (Some(from), Some(copy)) = (inst.read(Reg::S(code)), inst.write(Reg::V(reg))) else {
            return false;
        };
        let sites = || {
            p.blocks.iter().enumerate().flat_map(|(b, block)| {
                block
                    .insts
                    .iter()
                    .enumerate()
                    .map(move |(i, x)| ((b, i), x))
            })
        };
        if !self.descriptor_dword(from) || !sites().any(|(_, x)| x.name().starts_with("v_movrel")) {
            return false;
        }
        let mut readers: HashMap<Value, Vec<Site>> = HashMap::new();
        for (site, x) in sites() {
            for &(_, v) in &x.reads {
                readers.entry(v).or_default().push(site);
            }
        }
        let mut joins: HashMap<Value, Vec<Value>> = HashMap::new();
        for phi in p.blocks.iter().flat_map(|b| &b.phis) {
            for &arg in &phi.args {
                joins.entry(arg).or_default().push(phi.value);
            }
        }
        let mut spans = Spans::new(p, self.x.st, self.console_cf);
        let mut seen = HashSet::new();
        let mut work = vec![copy];
        while let Some(v) = work.pop() {
            if !seen.insert(v) {
                continue;
            }
            work.extend(joins.get(&v).into_iter().flatten());
            for &site in readers.get(&v).into_iter().flatten() {
                if !self.keeps_element(site, v, &mut work, &mut spans) {
                    return false;
                }
            }
        }
        true
    }

    /// Whether the instruction at `site` reads `v`, a value an unused array
    /// element may hold ([`Self::unused_element`]), without using it; the
    /// values that then hold it go to `work`.
    fn keeps_element(
        &self,
        (b, i): Site,
        v: Value,
        work: &mut Vec<Value>,
        spans: &mut Spans<'_>,
    ) -> bool {
        let x: &Inst = &self.x.p.blocks[b].insts[i];
        let Fields::Valu(f) = &x.fields else {
            return false;
        };
        let name = x.name();
        let vector = |o: Option<Operand>| match o {
            Some(Operand::Vector { reg, .. }) => Some(Reg::V(reg)),
            _ => None,
        };
        let (src, dst) = (vector(f.src.first().copied()), vector(f.dst));
        let holds = |r: Option<Reg>| r.is_some_and(|r| x.read(r) == Some(v));
        let plain = f.sdwa.is_none()
            && f.dpp.is_none()
            && f.abs == 0
            && f.neg == 0
            && !f.clamp
            && f.omod == 0;
        let element = match name {
            // The array's first element, or the value stored.
            "v_movreld_b32" if !plain || holds(src) || holds(dst) => return false,
            // Every other element keeps its value where M0 names another,
            // and takes the stored one where M0 names it: no read of the
            // dword either way, whatever M0 holds.
            "v_movreld_b32" => {
                for r in x.reads.iter().filter(|r| r.1 == v) {
                    work.extend(x.write(r.0));
                }
                return true;
            }
            "v_movrels_b32" if !plain || holds(src) => return false,
            // An element past the first: read where M0 names it, so only
            // where M0 is proven below it.
            "v_movrels_b32" if !m0_below(x, v, spans) => return false,
            "v_movrels_b32" => true,
            "v_mov_b32" if plain && holds(src) => {
                work.extend(dst.and_then(|r| x.write(r)));
                return true;
            }
            _ => false,
        };
        let sources = super::uniform::source_regs(x);
        for &(r, w) in &x.reads {
            // An element `v_movrels_b32` names past its first, M0 proven
            // below it (`m0_below`).
            if w != v || element && Some(r) != dst {
                continue;
            }
            // Read as an operand, or neither as one nor as the old value
            // of a register the instruction writes.
            if !element && sources.has(r) || x.write(r).is_none() {
                return false;
            }
        }
        // `v` is the old value of a VGPR `x` writes.
        if computes_with_old(name) {
            return false;
        }
        // Where the lowering keeps it: lanes outside the guard
        // (`Lower::guard`), the bits a half write leaves.
        let guarded = if self.console_cf {
            self.x.st.execs[b]
                .get(i)
                .copied()
                .flatten()
                .is_some_and(|e| e != Bool::TRUE)
        } else {
            self.x.st.guards[b][i].is_some()
        };
        if guarded || super::uniform::keeps_old(x) {
            for r in x.reads.iter().filter(|r| r.1 == v) {
                work.extend(x.write(r.0));
            }
        }
        true
    }
}

/// Whether a VALU instruction named `name` computes its result with the old
/// value of the VGPR it writes, beyond keeping it where it does not write
/// (an accumulator, a lane write, an interpolation's second step).
fn computes_with_old(name: &str) -> bool {
    name.contains("mac")
        || name.starts_with("v_dot")
        || name.contains("qsad")
        || name.starts_with("v_interp")
        || name.contains("permlane")
        || name.contains("writelane")
}

/// Whether M0 is proven below every element of `v_movrels_b32` `x` past its
/// first that holds `v`, in every lane that runs it ([`Spans`]). Such an
/// element is a register from the source up whose value is `v` (the
/// destination too, where it lies in that range); the first is refused
/// before ([`Lower::keeps_element`]).
fn m0_below(x: &Inst, v: Value, spans: &mut Spans<'_>) -> bool {
    let Fields::Valu(f) = &x.fields else {
        return false;
    };
    let (Some(&Operand::Vector { reg: base, .. }), Some(m0)) = (f.src.first(), x.read(Reg::S(M0)))
    else {
        return false;
    };
    let nearest = x
        .reads
        .iter()
        .filter(|r| r.1 == v)
        .filter_map(|r| match r.0 {
            Reg::V(n) if n >= base => Some(n - base),
            _ => None,
        })
        .min();
    match nearest {
        // Only the destination's old value, below the array.
        None => true,
        Some(k) => spans.of(m0).is_some_and(|(_, hi)| hi < u32::from(k)),
    }
}

/// Inclusive unsigned bounds `(lo, hi)` of a 32-bit value.
type Span = (u32, u32);

/// What [`Spans`] knows of a value it cannot follow.
const ANY: Span = (0, u32::MAX);

/// Rounds after which a span still growing jumps to the end it grows
/// towards ([`Spans::of`]).
const WIDEN: usize = 8;

/// Values [`Spans::of`] follows for one bound before it gives up.
const MAX_CONE: usize = 4096;

/// Rounds [`Spans::of`] runs for one bound before it gives up.
const MAX_ROUNDS: usize = 64;

/// The instructions [`Spans`] follows besides `v_movreld_b32` ([`compute`]).
const MODELED: &[&str] = &[
    "v_mov_b32",
    "s_mov_b32",
    "v_readfirstlane_b32",
    "v_and_b32",
    "s_and_b32",
    "v_add_nc_u32",
    "v_add_co_u32",
    "s_add_u32",
    "s_add_i32",
    "v_add3_u32",
    "v_min_u32",
    "s_min_u32",
    "v_min3_u32",
    "v_max_u32",
    "s_max_u32",
    "v_max3_u32",
    "v_med3_u32",
    "v_min_i32",
    "s_min_i32",
    "v_min3_i32",
    "v_max_i32",
    "s_max_i32",
    "v_max3_i32",
    "v_med3_i32",
];

/// How two 32-bit integers compare ([`Fact`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Test {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

impl Test {
    fn parse(s: &str) -> Option<Test> {
        Some(match s {
            "lt" => Test::Lt,
            "le" => Test::Le,
            "gt" => Test::Gt,
            "ge" => Test::Ge,
            "eq" => Test::Eq,
            "ne" | "lg" => Test::Ne,
            _ => return None,
        })
    }

    fn negated(self) -> Test {
        match self {
            Test::Lt => Test::Ge,
            Test::Le => Test::Gt,
            Test::Gt => Test::Le,
            Test::Ge => Test::Lt,
            Test::Eq => Test::Ne,
            Test::Ne => Test::Eq,
        }
    }

    /// The test with its operands exchanged.
    fn swapped(self) -> Test {
        match self {
            Test::Lt => Test::Gt,
            Test::Le => Test::Ge,
            Test::Gt => Test::Lt,
            Test::Ge => Test::Le,
            t => t,
        }
    }
}

/// A 32-bit operand: a value, or a constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Term {
    Value(Value),
    Const(u32),
}

/// `x test y`, as unsigned or (`signed`) signed integers, in every lane
/// that takes an edge ([`Spans::edge_facts`]).
#[derive(Debug, Clone, Copy)]
struct Fact {
    x: Value,
    test: Test,
    y: Term,
    signed: bool,
}

/// Unsigned bounds of 32-bit values in every lane that holds them, for the
/// M0 of an indexed read ([`m0_below`]). Each value's [`Span`] comes from
/// its definition, from nothing up (`None`: no lane holds it yet), joined
/// until no span grows; one still growing after [`WIDEN`] rounds jumps to
/// the end it grows towards, so the search ends. What it does not follow
/// is [`ANY`]: an input, an instruction outside the few it models, a write
/// that keeps part of the old value.
///
/// The argument of a phi is narrowed by the tests the lanes taking its edge
/// passed: what holds there ([`Structured::reach`] of the predecessor and
/// its branch, [`LaneExit`]) about a compare that the predecessor, or the
/// blocks it alone follows, ran on that argument. That is how a loop
/// counter is bounded: in 11eeb27c2d15980d the lanes that go round again
/// passed `!(counter > end)` (`v_cmpx_gt_u32 exec, v28, v19` at 0x47c),
/// with `end` in 0..11. Only in the per-lane form: in the console flow
/// every lane goes round with the wave, so a lane's counter can be past the
/// end; there nothing narrows and such a bound is not proven.
struct Spans<'a> {
    p: &'a Program,
    st: &'a Structured,
    console: bool,
    /// Each value's block, instruction (`None`: a phi) and register.
    defs: HashMap<Value, (usize, Option<usize>, Reg)>,
    /// What the lanes taking each edge know, by block and predecessor index.
    facts: HashMap<(usize, usize), Vec<Fact>>,
    done: HashMap<Value, Option<Span>>,
}

impl<'a> Spans<'a> {
    fn new(p: &'a Program, st: &'a Structured, console: bool) -> Self {
        let mut defs = HashMap::new();
        for (b, block) in p.blocks.iter().enumerate() {
            for phi in &block.phis {
                defs.insert(phi.value, (b, None, phi.reg));
            }
            for (i, inst) in block.insts.iter().enumerate() {
                for &(r, v) in &inst.writes {
                    defs.insert(v, (b, Some(i), r));
                }
            }
        }
        Spans {
            p,
            st,
            console,
            defs,
            facts: HashMap::new(),
            done: HashMap::new(),
        }
    }

    /// The bounds of `v` in every lane that holds it (`None`: no lane can).
    fn of(&mut self, v: Value) -> Option<Span> {
        if let Some(&s) = self.done.get(&v) {
            return s;
        }
        let mut cone = vec![v];
        let mut in_cone: HashSet<Value> = HashSet::from([v]);
        let mut known: HashMap<Value, Span> = HashMap::new();
        let mut round = 0;
        loop {
            let mut grew = false;
            let mut n = 0;
            while n < cone.len() {
                let x = cone[n];
                n += 1;
                let mut needs = Vec::new();
                let new = self.transfer(x, &known, &mut needs);
                for d in needs {
                    if in_cone.insert(d) {
                        cone.push(d);
                        grew = true;
                    }
                }
                if cone.len() > MAX_CONE {
                    return Some(ANY);
                }
                let Some(new) = new else {
                    continue;
                };
                let old = known.get(&x).copied();
                let mut s = old.map_or(new, |o| (o.0.min(new.0), o.1.max(new.1)));
                if let Some(o) = old
                    && round >= WIDEN
                {
                    if s.0 < o.0 {
                        s.0 = 0;
                    }
                    if s.1 > o.1 {
                        s.1 = u32::MAX;
                    }
                }
                if old != Some(s) {
                    known.insert(x, s);
                    grew = true;
                }
            }
            if !grew {
                break;
            }
            round += 1;
            if round > MAX_ROUNDS {
                return Some(ANY);
            }
        }
        let s = known.get(&v).copied();
        self.done.insert(v, s);
        s
    }

    /// What the lanes holding `x` can hold, given `known` (absent: no lane
    /// yet); the values it reads go to `needs`.
    fn transfer(
        &mut self,
        x: Value,
        known: &HashMap<Value, Span>,
        needs: &mut Vec<Value>,
    ) -> Option<Span> {
        let Some(&(b, at, reg)) = self.defs.get(&x) else {
            // An input: what the hardware or a fetch routine set up.
            return Some(ANY);
        };
        let Some(i) = at else {
            return self.joined(b, x, known, needs);
        };
        let p = self.p;
        let mut get = |v: Value| {
            needs.push(v);
            known.get(&v).copied()
        };
        let inst = &p.blocks[b].insts[i];
        let name = inst.name();
        let vector = matches!(inst.fields, Fields::Valu(_));
        // Lanes that reach a vector instruction without running it keep
        // the old value: its guard in the per-lane form, EXEC in the
        // console flow.
        let guarded = vector
            && if self.console {
                self.st.execs[b].get(i).copied().flatten() != Some(Bool::TRUE)
            } else {
                self.st.guards[b].get(i).copied().flatten().is_some()
            };
        // A write that keeps part of the old value mixes the two.
        if vector && matches!(reg, Reg::V(_)) && super::uniform::keeps_old(inst) {
            return Some(ANY);
        }
        let old = match (guarded, reg, inst.read(reg)) {
            (false, ..) => None,
            (true, Reg::V(_), Some(o)) => Some(o),
            // A scalar a vector instruction writes lists no old value.
            (true, ..) => return Some(ANY),
        };
        let new = match &inst.fields {
            Fields::Valu(f) if !plain(f) => return Some(ANY),
            Fields::Valu(f) if name == "v_movreld_b32" => {
                // Element k takes the stored value where M0 is k, and
                // keeps its old one elsewhere.
                let (Some(Operand::Vector { reg: d, .. }), Reg::V(r)) = (f.dst, reg) else {
                    return Some(ANY);
                };
                let (Some(k), Some(prev), Some(m0), Some(stored)) = (
                    r.checked_sub(d),
                    inst.read(reg),
                    inst.read(Reg::S(M0)),
                    f.src.first().and_then(|o| operand(inst, o)),
                ) else {
                    return Some(ANY);
                };
                let prev = get(prev);
                let named = get(m0).is_some_and(|(lo, hi)| (lo..=hi).contains(&u32::from(k)));
                let stored = if named { term(stored, &mut get) } else { None };
                union(prev, stored)
            }
            Fields::Valu(Valu { dst, src, .. })
            | Fields::Sop {
                dst,
                src,
                imm: None,
            } if dst_reg(*dst) == Some(reg) && MODELED.contains(&name) => {
                let mut spans = Vec::with_capacity(src.len());
                for o in src {
                    let Some(t) = operand(inst, o) else {
                        return Some(ANY);
                    };
                    spans.push(term(t, &mut get));
                }
                let all: Option<Vec<Span>> = spans.into_iter().collect();
                match all {
                    // An operand no lane holds yet.
                    None => None,
                    Some(s) => match compute(name, &s) {
                        Some(s) => Some(s),
                        None => return Some(ANY),
                    },
                }
            }
            _ => return Some(ANY),
        };
        match old {
            Some(o) => union(new, get(o)),
            None => new,
        }
    }

    /// A phi's span: each argument's, narrowed by what the lanes taking
    /// its edge know.
    fn joined(
        &mut self,
        h: usize,
        x: Value,
        known: &HashMap<Value, Span>,
        needs: &mut Vec<Value>,
    ) -> Option<Span> {
        let p = self.p;
        let Some(phi) = p.blocks[h].phis.iter().find(|q| q.value == x) else {
            return Some(ANY);
        };
        let mut get = |v: Value| {
            needs.push(v);
            known.get(&v).copied()
        };
        let mut out = None;
        for (n, &arg) in phi.args.iter().enumerate() {
            if !self.facts.contains_key(&(h, n)) {
                let f = self.edge_facts(h, n);
                self.facts.insert((h, n), f);
            }
            let mut s = get(arg);
            for f in self.facts[&(h, n)].iter().filter(|f| f.x == arg) {
                let (Some(a), Some(t)) = (s, term(f.y, &mut get)) else {
                    // No lane takes the edge with these yet.
                    s = None;
                    break;
                };
                s = narrow(a, f, t);
            }
            out = union(out, s);
        }
        out
    }

    /// What every lane taking the edge from the `n`th predecessor of block
    /// `h` knows of the compares it has just run: those of the predecessor
    /// and of the blocks before it that it alone follows (each the only
    /// predecessor of the next), which ran on the values the edge carries.
    /// None in the console flow, where every lane takes the wave's edge.
    fn edge_facts(&self, h: usize, n: usize) -> Vec<Fact> {
        let p = self.p;
        if self.console {
            return Vec::new();
        }
        let Some(&q) = p.blocks[h].preds.get(n) else {
            return Vec::new();
        };
        let (Some(&reach), Some(exit)) = (self.st.reach.get(q), self.st.exits.get(q)) else {
            return Vec::new();
        };
        let mut conds = vec![(reach, true)];
        if let LaneExit::Branch {
            when,
            taken,
            not_taken,
        } = *exit
            && taken != not_taken
        {
            conds.push((when, taken == h));
        }
        let mut chain = vec![q];
        let mut last = q;
        while let &[only] = &p.blocks[last].preds[..]
            && !chain.contains(&only)
        {
            chain.push(only);
            last = only;
        }
        let bools = &self.st.bools;
        let mut atoms: Vec<Atom> = Vec::new();
        for &(c, _) in &conds {
            for a in bools.support(c) {
                if !atoms.contains(&a) {
                    atoms.push(a);
                }
            }
        }
        let mut out = Vec::new();
        for atom in atoms {
            let (Atom::Lane(c) | Atom::Uniform(c)) = atom else {
                continue;
            };
            let Some(&(d, Some(i), _)) = self.defs.get(&c) else {
                continue;
            };
            let Some(var) = bools.var(atom).filter(|_| chain.contains(&d)) else {
                continue;
            };
            let some_lane = |holds: bool| {
                let mut all = conds.clone();
                all.push((var, holds));
                bools.meet(&all)
            };
            let test_holds = match (some_lane(true), some_lane(false)) {
                (true, false) => true,
                (false, true) => false,
                _ => continue,
            };
            let Some((a, test, b, signed)) = compare(&p.blocks[d].insts[i]) else {
                continue;
            };
            let test = if test_holds { test } else { test.negated() };
            if let Term::Value(x) = a {
                out.push(Fact {
                    x,
                    test,
                    y: b,
                    signed,
                });
            }
            if let Term::Value(y) = b {
                out.push(Fact {
                    x: y,
                    test: test.swapped(),
                    y: a,
                    signed,
                });
            }
        }
        out
    }
}

/// No modifier changes the operands or the result: no SDWA, DPP, abs,
/// neg, clamp, output modifier or operand select.
fn plain(f: &Valu) -> bool {
    f.sdwa.is_none()
        && f.dpp.is_none()
        && f.abs == 0
        && f.neg == 0
        && !f.clamp
        && f.omod == 0
        && f.opsel == 0
}

/// The register a VALU or SALU destination names, its first dword.
fn dst_reg(dst: Option<Operand>) -> Option<Reg> {
    match dst? {
        Operand::Vector { reg, .. } => Some(Reg::V(reg)),
        Operand::Scalar { code, .. } if code != NULL => Some(Reg::S(code)),
        _ => None,
    }
}

/// A 32-bit operand of `inst`: the value it reads, or its constant.
fn operand(inst: &Inst, op: &Operand) -> Option<Term> {
    match *op {
        Operand::Scalar { code, .. } if code != NULL => inst.read(Reg::S(code)).map(Term::Value),
        Operand::Vector { reg, .. } => inst.read(Reg::V(reg)).map(Term::Value),
        Operand::Const(c) => crate::resource::inline(c).map(Term::Const),
        Operand::Literal(k) => Some(Term::Const(k)),
        _ => None,
    }
}

fn term(t: Term, get: &mut impl FnMut(Value) -> Option<Span>) -> Option<Span> {
    match t {
        Term::Const(k) => Some((k, k)),
        Term::Value(v) => get(v),
    }
}

fn union(a: Option<Span>, b: Option<Span>) -> Option<Span> {
    match (a, b) {
        (Some(a), Some(b)) => Some((a.0.min(b.0), a.1.max(b.1))),
        (a, None) => a,
        (None, b) => b,
    }
}

/// The span of a 32-bit integer instruction named `name` whose operands
/// have spans `s`; `None` for one it does not model.
fn compute(name: &str, s: &[Span]) -> Option<Span> {
    let lo: Vec<u32> = s.iter().map(|s| s.0).collect();
    let hi: Vec<u32> = s.iter().map(|s| s.1).collect();
    // Unsigned operations that never decrease when an operand grows: their
    // bounds are their results on the bounds.
    let monotone = |f: fn(&[u32]) -> Option<u32>| Some((f(&lo)?, f(&hi)?));
    match name {
        "v_mov_b32" | "s_mov_b32" | "v_readfirstlane_b32" if s.len() == 1 => Some(s[0]),
        "v_and_b32" | "s_and_b32" if s.len() == 2 => Some((0, hi[0].min(hi[1]))),
        "v_add_nc_u32" | "v_add_co_u32" | "s_add_u32" | "s_add_i32" | "v_add3_u32"
            if s.len() >= 2 =>
        {
            // Only where no lane can wrap.
            let sum = |v: &[u32]| v.iter().try_fold(0u32, |a, &b| a.checked_add(b));
            Some((sum(&lo)?, sum(&hi)?))
        }
        "v_min_u32" | "s_min_u32" | "v_min3_u32" => monotone(|v| v.iter().copied().min()),
        "v_max_u32" | "s_max_u32" | "v_max3_u32" => monotone(|v| v.iter().copied().max()),
        "v_med3_u32" if s.len() == 3 => monotone(|v| Some(median(v[0], v[1], v[2]))),
        "v_min_i32" | "s_min_i32" | "v_min3_i32" => signed(s, |v| v.iter().copied().min()),
        "v_max_i32" | "s_max_i32" | "v_max3_i32" => signed(s, |v| v.iter().copied().max()),
        "v_med3_i32" if s.len() == 3 => signed(s, |v| Some(median(v[0], v[1], v[2]))),
        _ => None,
    }
}

fn median<T: Ord + Copy>(a: T, b: T, c: T) -> T {
    a.max(b).min(a.min(b).max(c))
}

/// A signed operation that never decreases when an operand grows, on
/// spans read as signed: one that crosses from `i32::MAX` to `i32::MIN`
/// stands for every signed value.
fn signed(s: &[Span], f: fn(&[i32]) -> Option<i32>) -> Option<Span> {
    let as_signed = |&(lo, hi): &Span| {
        if (lo <= i32::MAX as u32) == (hi <= i32::MAX as u32) {
            (lo as i32, hi as i32)
        } else {
            (i32::MIN, i32::MAX)
        }
    };
    let s: Vec<(i32, i32)> = s.iter().map(as_signed).collect();
    let lo: Vec<i32> = s.iter().map(|s| s.0).collect();
    let hi: Vec<i32> = s.iter().map(|s| s.1).collect();
    let (lo, hi) = (f(&lo)?, f(&hi)?);
    if (lo < 0) == (hi < 0) {
        Some((lo as u32, hi as u32))
    } else {
        Some(ANY)
    }
}

/// `x`'s span `a` in the lanes where `f` holds, `y` in `t`; `None` when no
/// lane can. A signed test narrows only where both lie in 0..=i32::MAX,
/// where it orders them as the unsigned one does.
fn narrow(a: Span, f: &Fact, t: Span) -> Option<Span> {
    if f.signed && (a.1 > i32::MAX as u32 || t.1 > i32::MAX as u32) {
        return Some(a);
    }
    let (lo, hi) = match f.test {
        Test::Lt => (a.0, a.1.min(t.1.checked_sub(1)?)),
        Test::Le => (a.0, a.1.min(t.1)),
        Test::Gt => (a.0.max(t.0.checked_add(1)?), a.1),
        Test::Ge => (a.0.max(t.0), a.1),
        Test::Eq => (a.0.max(t.0), a.1.min(t.1)),
        Test::Ne => a,
    };
    (lo <= hi).then_some((lo, hi))
}

/// A 32-bit integer compare: its operands, test and signedness, for a
/// plain `v_cmp*`/`v_cmpx*`/`s_cmp*`/`s_cmpk*` of `u32` or `i32`.
fn compare(inst: &Inst) -> Option<(Term, Test, Term, bool)> {
    let name = inst.name();
    let rest = ["v_cmpx_", "v_cmp_", "s_cmpk_", "s_cmp_"]
        .into_iter()
        .find_map(|prefix| name.strip_prefix(prefix))?;
    let (test, ty) = rest.split_once('_')?;
    let test = Test::parse(test)?;
    let signed = match ty {
        "u32" => false,
        "i32" => true,
        _ => return None,
    };
    let (a, b) = match &inst.fields {
        Fields::Valu(f) if plain(f) => (f.src.first()?, f.src.get(1)?),
        Fields::Sop {
            src, imm: Some(k), ..
        } => {
            let k = if signed {
                *k as i16 as i32 as u32
            } else {
                u32::from(*k)
            };
            return Some((operand(inst, src.first()?)?, test, Term::Const(k), signed));
        }
        Fields::Sop { src, imm: None, .. } => (src.first()?, src.get(1)?),
        _ => return None,
    };
    Some((operand(inst, a)?, test, operand(inst, b)?, signed))
}
