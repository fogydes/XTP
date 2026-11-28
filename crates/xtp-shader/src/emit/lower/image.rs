//! Images (MIMG): samples, gathers, loads, stores, atomics and sizes.
//! Backlog S1 (A16 addressing, `image_sample_d_cl`, `image_gather4_l`: the
//! address and sample parts of `image`) and I1 (64-bit image atomics: its
//! atomic branch).

use std::collections::HashSet;

use super::{Lower, Why};
use crate::emit::bitcode::{Attr, Bin, Cast, Term, V, pred};
use crate::emit::shader::{Shader, flags};
use crate::gfx10::{Fields, Operand};
use crate::ir::{Inst, Reg, Value};
use crate::native::ResourceType;
use crate::resource::{Site, Source};

/// Résultat MIMG : DXIL impose de joindre séparément les composantes scalaires.
enum Texels {
    Resource(V),
    Channels([V; 4]),
}

impl Texels {
    fn channel(&self, s: &mut Shader, block: usize, ch: u32) -> V {
        match self {
            Self::Resource(r) => s.m.extract(block, *r, ch),
            Self::Channels(xs) => xs[ch as usize],
        }
    }
}

struct SamplerChoice {
    first: V,
    /// Condition vraie sur le chemin qui a fourni le second sampler.
    second: Option<(V, V)>,
}

impl SamplerChoice {
    // Adapté de fxceless/victory-fork#25 (cb80bf0, MartinIndra02, GPL-3.0).
    // Chaque branche utilise un registre statique et ne lit la texture qu'une fois.
    fn dispatch<const N: usize>(
        &self,
        l: &mut Lower<'_>,
        mut call: impl FnMut(&mut Shader, usize, V) -> [V; N],
    ) -> [V; N] {
        let Some((cond, second)) = self.second else {
            return call(&mut l.s, l.cur, self.first);
        };
        let a = l.s.m.new_block();
        let b = l.s.m.new_block();
        let join = l.s.m.new_block();
        l.s.m.terminate(l.cur, Term::CondBr(cond, b, a));
        let left = call(&mut l.s, a, self.first);
        let right = call(&mut l.s, b, second);
        l.s.m.terminate(a, Term::Br(join));
        l.s.m.terminate(b, Term::Br(join));
        l.cur = join;
        std::array::from_fn(|i| {
            let phi = l.s.m.phi(join, l.s.t.f32);
            l.s.m.set_phi(phi, vec![(left[i], a), (right[i], b)]);
            phi
        })
    }

    fn sample(&self, l: &mut Lower<'_>, name: &str, op: u32, args: &mut [V]) -> Texels {
        if self.second.is_none() {
            return Texels::Resource(l.s.op(l.cur, name, Attr::ReadOnly, l.s.t.res_ret, op, args));
        }
        Texels::Channels(self.dispatch(l, |s, block, handle| {
            args[1] = handle;
            let r = s.op(block, name, Attr::ReadOnly, s.t.res_ret, op, args);
            std::array::from_fn(|i| s.m.extract(block, r, i as u32))
        }))
    }
}

/// A 64-bit image atomic, as [`Lower::image`] decoded it.
struct Atomic64<'s> {
    site: Site,
    src: &'s Source,
    /// The DXIL resource kind of the view.
    kind: u32,
    /// The first data VGPR: each value a pair, low dword first.
    reg: u8,
    guard: Option<V>,
    /// `image_atomic_sub`: the add of the negation.
    sub: bool,
    /// The DXIL binary operation, `u32::MAX` a compare-exchange.
    code: u32,
    /// The old value returns to the data VGPRs.
    glc: bool,
}

/// What a sample or gather opcode adds to the plain form, by its name:
/// comparison (`_c`), explicit derivatives (`_d`, coarse `_cd`, 16-bit
/// `_g16`), a level (`_l`) or level 0 (`_lz`), a bias (`_b`), a level
/// clamp (`_cl`), a texel offset (`_o`).
#[derive(Debug, Clone, Copy, Default)]
struct SampleForm {
    gather: bool,
    c: bool,
    d: bool,
    g16: bool,
    l: bool,
    lz: bool,
    b: bool,
    cl: bool,
    o: bool,
}

/// Not a DXIL operation: a gather at a level known only at run time, made
/// of four samples (`Lower::image`).
const GATHER_AT_LEVEL: u32 = 1 << 16;

/// How far [`Lower::layer_of`] follows a face operand back.
const LAYER_DEPTH: u32 = 12;

/// The face of a cube sample: a float from the address (a cube, converted
/// as before), or the face id taken out of a cube array's face-and-slice.
#[derive(Debug, Clone, Copy)]
enum CubeFace {
    Float(V),
    Index(V),
}

/// What a cube sample's face operand is ([`Lower::layer_of`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Layer {
    /// A face id, 0 to 5 (`v_cubeid_f32`): a cube.
    Face,
    /// A face id plus 8 × a slice: a cube array (the console's face
    /// operand of a cube array sample).
    Slice,
    /// Neither, as far as the operand is followed.
    Unknown,
}

impl Layer {
    /// Both, at a join or a select: a cube array if either is one.
    fn join(self, other: Layer) -> Layer {
        match (self, other) {
            (Layer::Unknown, _) | (_, Layer::Unknown) => Layer::Unknown,
            (Layer::Face, Layer::Face) => Layer::Face,
            _ => Layer::Slice,
        }
    }

    /// A float constant: a face id, or a face id plus 8 × a slice.
    fn of_constant(bits: u32) -> Layer {
        let x = f32::from_bits(bits);
        if !((0.0..16_777_216.0).contains(&x) && x.fract() == 0.0) {
            return Layer::Unknown;
        }
        match (x as u32 % 8, x < 8.0) {
            (0..6, true) => Layer::Face,
            (0..6, false) => Layer::Slice,
            _ => Layer::Unknown,
        }
    }
}

/// The bits of 8.0.
const EIGHT: u32 = 0x4100_0000;

/// The dword (and half, with a16) of a cube sample's face operand, its
/// third coordinate: after the offset, bias and comparison dwords and the
/// two face-coordinate derivatives per direction (see [`Lower::image`]).
fn cube_face_at(f: SampleForm, a16: bool) -> (usize, Option<u32>) {
    let whole = usize::from(f.o) + usize::from(f.b) + usize::from(f.c);
    let grads: usize = if f.d { 2 } else { 0 };
    let per_direction = if f.g16 { grads.div_ceil(2) } else { grads };
    let first = whole + 2 * per_direction;
    if a16 {
        (first + 1, Some(0))
    } else {
        (first + 2, None)
    }
}

fn sample_form(name: &str) -> Option<SampleForm> {
    let (gather, rest) = match name.strip_prefix("image_sample") {
        Some(rest) => (false, rest),
        None => (true, name.strip_prefix("image_gather4")?),
    };
    let mut f = SampleForm {
        gather,
        ..SampleForm::default()
    };
    for token in rest.strip_prefix('_').unwrap_or(rest).split_terminator('_') {
        let flag = match token {
            "c" => &mut f.c,
            "d" | "cd" => &mut f.d,
            "g16" => &mut f.g16,
            "l" => &mut f.l,
            "lz" => &mut f.lz,
            "b" => &mut f.b,
            "cl" => &mut f.cl,
            "o" => &mut f.o,
            _ => return None,
        };
        *flag = true;
    }
    Some(f)
}

/// Whether image opcode `name` takes its level from its quad's implicit
/// derivatives: a sample with no level, level 0 or gradients given (a
/// gather with no level is refused, see [`Lower::image`]).
pub(super) fn implicit_derivatives(name: &str) -> bool {
    sample_form(name).is_some_and(|f| !(f.gather || f.d || f.l || f.lz))
}

/// Why a compute program's implicit derivatives cannot be the console's:
/// see [`Lower::compute_quads`].
const NO_CONSOLE_QUADS: Why =
    "an implicit-derivative sample in a compute group of a size not a multiple of 4";

impl<'a> Lower<'a> {
    /// Compute: whether the host's quads are the console's, four
    /// consecutive thread ids. They are in a group declared as one row of a
    /// multiple of 4 threads (SM 6.6's 1D quads), which is how `Lower::new`
    /// declares a group with such samples when its size allows it.
    fn compute_quads(&self) -> Result<(), Why> {
        match self.s.threads {
            Some([x, 1, 1]) if x.is_multiple_of(4) => Ok(()),
            _ => Err(NO_CONSOLE_QUADS),
        }
    }

    /// Les deux entrées constantes d'une jonction scalaire utilisent les arêtes
    /// du CFG émis, y compris les blocs intercalés pour compter les tours de boucle.
    fn image_sampler(
        &mut self,
        site: Site,
        src: &Source,
        comparison: bool,
    ) -> Result<SamplerChoice, Why> {
        let props = (14 | if comparison { 0x8000 } else { 0 }, 0);
        if let Source::Phi { block, args } = src {
            let mut live = self.x.p.blocks[*block]
                .preds
                .iter()
                .copied()
                .zip(args)
                .filter(|(p, _)| self.block[*p].is_some());
            if let (
                Some((a, left @ Source::Constant(_))),
                Some((b, right @ Source::Constant(_))),
                None,
            ) = (live.next(), live.next(), live.next())
            {
                let edge = |p| {
                    self.edges
                        .get(&(p, *block))
                        .copied()
                        .or(self.tail[p])
                        .ok_or("a descriptor chosen by control flow")
                };
                let (a, b) = (edge(a)?, edge(b)?);
                let join = self.block[*block].ok_or("a descriptor chosen by control flow")?;
                let first = self.handle(site, left, true, props)?;
                let second = if left == right {
                    None
                } else {
                    let second = self.handle(site, right, true, props)?;
                    let cond = self.s.m.phi(join, self.s.t.i1);
                    let no = self.s.m.bool(false);
                    let yes = self.s.m.bool(true);
                    self.s.m.set_phi(cond, vec![(no, a), (yes, b)]);
                    Some((cond, second))
                };
                return Ok(SamplerChoice { first, second });
            }
        }
        Ok(SamplerChoice {
            first: self.handle(site, src, true, props)?,
            second: None,
        })
    }

    /// A 64-bit image atomic: DXIL's i64 `atomicBinOp` or
    /// `atomicCompareExchange` on the view as a 64-bit integer texture UAV
    /// (the console's R32G32 view; the host's must be R32G32_UINT, as for
    /// `RWTexture2D<uint64_t>`). Its handle is annotated U32, one channel,
    /// as DXC annotates one.
    fn image_atomic64(&mut self, inst: &Inst, at: &Atomic64, cs: Vec<V>) -> Result<(), Why> {
        let i64t = self.s.m.int(64);
        self.s.flags |= flags::INT64 | flags::ATOMIC_INT64_TYPED;
        // Every source but a root view is a heap handle (see `handle`); a
        // root view is refused there.
        if !matches!(at.src, Source::Built { .. }) {
            self.s.flags |= flags::ATOMIC_INT64_HEAP;
        }
        let pair = |l: &mut Self, r: u8| -> Result<V, Why> {
            let c = l.cur;
            let lo = l.i32_of(inst.read(Reg::V(r)).ok_or("image atomic data")?)?;
            let hi = l.i32_of(inst.read(Reg::V(r + 1)).ok_or("image atomic data")?)?;
            let lo = l.s.m.cast(c, Cast::ZExt, lo, i64t);
            let hi = l.s.m.cast(c, Cast::ZExt, hi, i64t);
            let k32 = l.s.m.int_const(i64t, 32);
            let hi = l.s.m.bin(c, Bin::Shl, hi, k32);
            Ok(l.s.m.bin(c, Bin::Or, lo, hi))
        };
        // Lanes the guard skips keep their registers: the data pair is the
        // value outside the guard. The compare (cmpswap: data is (new,
        // compare), a pair each) is built before the guard too.
        let old = pair(self, at.reg)?;
        let mut value = old;
        if at.sub {
            let z = self.s.m.int_const(i64t, 0);
            value = self.s.m.bin(self.cur, Bin::Sub, z, value);
        }
        let cmp = if at.code == u32::MAX {
            Some(pair(self, at.reg + 2)?)
        } else {
            None
        };
        // A heap handle of its own: `handle` caches by the annotation's first
        // word only, so a float4 handle of the same view (a size query, say)
        // would come back with the wrong element type.
        let props = (at.kind | 0x1000, 0x105);
        let h = match at.src {
            Source::Loaded { load, first } => {
                let index = self.heap_index(*load, *first, false)?;
                // Per lane in a waterfall loop, as `handle` makes it.
                let per_lane = self.x.res.waterfall.loads.contains(load);
                self.s.heap_handle(self.cur, index, false, per_lane, props)
            }
            _ => self.handle(at.site, at.src, false, props)?,
        };
        let guarded = self.begin_guarded(at.guard);
        let mut args = vec![h];
        let r = if let Some(cmp) = cmp {
            args.extend(cs);
            args.extend([cmp, value]);
            self.s.op(
                self.cur,
                "atomicCompareExchange.i64",
                Attr::NoUnwind,
                i64t,
                79,
                &args,
            )
        } else {
            args.push(self.s.m.i32(at.code));
            args.extend(cs);
            args.push(value);
            self.s
                .op(self.cur, "atomicBinOp.i64", Attr::NoUnwind, i64t, 78, &args)
        };
        let r = self.end_guarded(guarded, &[(r, old)])[0];
        if at.glc {
            let (lo, hi) = self.split64(r);
            self.write(inst, Reg::V(at.reg), lo, None);
            self.write(inst, Reg::V(at.reg + 1), hi, None);
        }
        Ok(())
    }

    pub(super) fn image(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Image {
            data,
            addr,
            dmask,
            dim,
            unorm,
            a16,
            d16,
            tfe,
            lwe,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let name = inst.name();
        if name == "image_gather4h" {
            return Err(name);
        }
        if *tfe || *lwe {
            return Err("an image access with tfe or lwe");
        }
        // A load or sample with no channel enabled returns the first.
        let dmask = if *dmask == 0 { &1 } else { dmask };
        let site = (b, i);
        let guard = self.effect_guard(b, i)?;
        let c = self.cur;
        let (mut kind, coords) = match dim {
            0 => (1, 1),
            1 => (2, 2),
            2 => (4, 3),
            // Sampled (face coordinates and face, see `cube_direction`) or
            // sized only; a cube or a cube array (`cube_view`).
            3 if name.starts_with("image_sample")
                || name.starts_with("image_gather")
                || name == "image_get_resinfo" =>
            {
                (5, 3)
            }
            4 => (6, 2),
            5 => (7, 3),
            _ => return Err("an image dimension (cube loads or MSAA)"),
        };
        // Address VGPRs in order.
        let mut regs = Vec::new();
        for a in addr {
            if let Operand::Vector { reg, dwords } = *a {
                regs.extend((0..dwords).map(|k| reg + k));
            }
        }
        // With a16, two 16-bit components per VGPR, low half first.
        let addr_val = |l: &mut Self, k: usize| -> Result<V, Why> {
            let r = *regs
                .get(if *a16 { k / 2 } else { k })
                .ok_or("an image address")?;
            let v = inst.read(Reg::V(r)).ok_or("an image address")?;
            let x = l.i32_of(v)?;
            if !*a16 {
                return Ok(x);
            }
            let c = l.cur;
            let (sh, mask) = (l.s.m.i32(16 * (k as u32 % 2)), l.s.m.i32(0xFFFF));
            let x = l.s.m.bin(c, Bin::LShr, x, sh);
            Ok(l.s.m.bin(c, Bin::And, x, mask))
        };
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked image")?;
        if kind == 5 {
            kind = self.cube_view(inst, src);
        }
        let store = name.starts_with("image_store");
        let uav = store
            || self
                .used_kind(src)
                .is_some_and(|u| u.kind == ResourceType::Uav);
        let props = (kind | if uav { 0x1000 } else { 0 }, 0x409);
        if let Some(op) = name.strip_prefix("image_atomic_") {
            // DXIL's atomic operations on a 32-bit integer texture UAV; the
            // old value returns with glc. Sub is the add of the negation.
            let code = match op {
                "add" | "sub" => 0,
                "and" => 1,
                "or" => 2,
                "xor" => 3,
                "smin" => 4,
                "smax" => 5,
                "umin" => 6,
                "umax" => 7,
                "swap" => 8,
                "cmpswap" => u32::MAX,
                _ => return Err(name),
            };
            let Fields::Image { cache, .. } = &inst.fields else {
                unreachable!()
            };
            if *d16 || *a16 {
                return Err("a 16-bit image atomic");
            }
            // dmask: the data dwords (cmpswap: twice the value's). Two
            // dwords per value are a 64-bit atomic on an R32G32 view.
            let wide = match (*dmask, code == u32::MAX) {
                (1, false) | (3, true) => false,
                (3, false) | (0xF, true) => true,
                _ => return Err("an image atomic's data mask"),
            };
            let iu = self.s.m.undef(self.s.t.i32);
            let mut cs = vec![iu; 3];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = addr_val(self, k)?;
            }
            let Operand::Vector { reg, .. } = *data else {
                return Err("image atomic data");
            };
            if wide {
                let at = Atomic64 {
                    site,
                    src,
                    kind,
                    reg,
                    guard,
                    sub: op == "sub",
                    code,
                    glc: cache.glc,
                };
                return self.image_atomic64(inst, &at, cs);
            }
            let vals = self.vgprs(inst, data, if code == u32::MAX { 2 } else { 1 })?;
            let mut value = vals[0];
            if op == "sub" {
                let z = self.s.m.i32(0);
                value = self.s.m.bin(c, Bin::Sub, z, value);
            }
            let old = match inst.read(Reg::V(reg)) {
                Some(v) if cache.glc => self.i32_of(v)?,
                _ => self.s.m.i32(0),
            };
            let h = self.handle(site, src, false, (kind | 0x1000, 0x105))?;
            let at = self.begin_guarded(guard);
            let i32t = self.s.t.i32;
            let mut args = vec![h];
            let r = if code == u32::MAX {
                // cmpswap: data is (new, compare).
                args.extend(cs);
                args.extend([vals[1], vals[0]]);
                self.s.op(
                    self.cur,
                    "atomicCompareExchange.i32",
                    Attr::NoUnwind,
                    i32t,
                    79,
                    &args,
                )
            } else {
                args.push(self.s.m.i32(code));
                args.extend(cs);
                args.push(value);
                self.s
                    .op(self.cur, "atomicBinOp.i32", Attr::NoUnwind, i32t, 78, &args)
            };
            let r = self.end_guarded(at, &[(r, old)])[0];
            if cache.glc {
                self.write(inst, Reg::V(reg), r, None);
            }
            return Ok(());
        }
        if store {
            if name != "image_store" {
                return Err("an image store to a mip level");
            }
            // shortcut: stores are taken to write all four channels of a
            // float format (the console converts by the view's format). The
            // data VGPRs hold the dmask's channels in order; the others are
            // written as 0.
            let iu = self.s.m.undef(self.s.t.i32);
            let mut cs = vec![iu; 3];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = addr_val(self, k)?;
            }
            let Operand::Vector { reg, .. } = *data else {
                return Err("image store data");
            };
            let mut vals = Vec::new();
            for ch in 0..4u8 {
                if dmask >> ch & 1 == 0 {
                    vals.push(self.s.m.f32(0));
                    continue;
                }
                let k = (dmask & ((1 << ch) - 1)).count_ones() as u8;
                // d16: two f16 channels per VGPR, low half first.
                let r = if *d16 { k / 2 } else { k };
                let v = inst.read(Reg::V(reg + r)).ok_or("image store data")?;
                vals.push(if *d16 {
                    let mut x = self.i32_of(v)?;
                    if k % 2 == 1 {
                        let k16 = self.s.m.i32(16);
                        x = self.s.m.bin(self.cur, Bin::LShr, x, k16);
                    }
                    let f32t = self.s.t.f32;
                    self.s
                        .op(self.cur, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x])
                } else {
                    self.float_of(v)?
                });
            }
            let h = self.handle(site, src, false, props)?;
            let at = self.begin_guarded(guard);
            let mut args = vec![h];
            args.extend(cs);
            args.extend(vals);
            args.push(self.s.m.i8(0xF));
            let void = self.s.t.void;
            self.s.op(
                self.cur,
                "textureStore.f32",
                Attr::NoUnwind,
                void,
                67,
                &args,
            );
            self.end_guarded(at, &[]);
            return Ok(());
        }
        // Samples and gathers: the DXIL operation follows from the form.
        let form = sample_form(name);
        let gather = form.is_some_and(|f| f.gather);
        let (op, sample) = match name {
            "image_get_lod" if self.s.pixel => (81, true),
            "image_load" | "image_load_mip" => (66, false),
            "image_get_resinfo" => (72, false),
            _ if form.is_some() => (0, true),
            _ => return Err(name),
        };
        if op == 72 {
            // Width, height, depth or layers, levels (per dmask) of the level
            // in the first address. Non-cube views return zero for unused
            // extents. A UAV still reports one level.
            let h = self.handle(site, src, false, props)?;
            let mip = if uav {
                self.s.m.undef(self.s.t.i32)
            } else {
                addr_val(self, 0)?
            };
            let i32t = self.s.t.i32;
            let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
            let d = self
                .s
                .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, mip]);
            let one = self.s.m.i32(1);
            // Keep the existing cube depth fallback.
            let missing_extent = self.s.m.i32(u32::from(kind == 5));
            let present = match kind {
                1 => [true, false, false, true],
                2 | 5 | 6 => [true, true, false, true],
                _ => [true, true, true, true],
            };
            let Operand::Vector { reg, .. } = *data else {
                return Err("an image result");
            };
            for (k, ch) in (0..4u32).filter(|ch| dmask >> ch & 1 != 0).enumerate() {
                let x = if ch == 3 && uav {
                    one
                } else if !present[ch as usize] {
                    missing_extent
                } else if ch == 2 && kind == 9 {
                    // The console counts a cube array's layers, six per
                    // cube (its code divides by 6); DXIL counts cubes.
                    let n = self.s.m.extract(c, d, ch);
                    let six = self.s.m.i32(6);
                    self.s.m.bin(c, Bin::Mul, n, six)
                } else {
                    self.s.m.extract(c, d, ch)
                };
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        if uav && op != 66 {
            return Err("sampling a UAV");
        }
        let h = self.handle(site, src, false, props)?;
        let fu = self.s.m.undef(self.s.t.f32);
        let iu = self.s.m.undef(self.s.t.i32);
        let r = if sample {
            let ss = self
                .descriptors
                .get(&(site, true))
                .copied()
                .ok_or("an untracked sampler")?;
            let f = form.unwrap_or_default();
            if f.c && dmask.count_ones() != 1 {
                return Err("a comparison sample of several channels");
            }
            // Comparison: the sampler is a comparison sampler.
            let sampler = self.image_sampler(site, ss, f.c)?;
            // Coordinates the size divides (not the array slice).
            let spatial = coords - usize::from(kind == 6 || kind == 7);
            // Address dwords in order (LLVM ImageDimIntrinsics.td, Mesa
            // ac_build_image_opcode): the texel offset, bias and comparison
            // reference, a dword each (an a16 bias is an f16 in the low
            // half, LLVM SIISelLowering::lowerImage); the derivatives, x
            // then y; the coordinates, then the level or level clamp.
            // Derivatives are 16-bit in the g16 opcodes, coordinates and
            // levels with a16 (GFX10.3 keeps the two apart, as LLVM's
            // lowerImage does where hasG16): two per dword, low half first,
            // each direction starting a dword.
            let whole = usize::from(f.o) + usize::from(f.b) + usize::from(f.c);
            let bias_at = (usize::from(f.o), a16.then_some(0));
            // A cube's derivatives are those of its two face coordinates
            // (the third coordinate is the face).
            let grads = match (f.d, kind) {
                (false, _) => 0,
                (true, 5 | 9) => 2,
                (true, _) => spatial,
            };
            let g16 = f.g16;
            let per_direction = if g16 { grads.div_ceil(2) } else { grads };
            let grad_at = |dir: usize, k: usize| {
                if g16 {
                    (whole + dir * per_direction + k / 2, Some(k as u32 % 2))
                } else {
                    (whole + dir * grads + k, None)
                }
            };
            let first = whole + 2 * per_direction;
            // Coordinate k; k = coords is the level or the level clamp.
            let coord_at = |k: usize| {
                if *a16 {
                    (first + k / 2, Some(k as u32 % 2))
                } else {
                    (first + k, None)
                }
            };
            let raw = |l: &mut Self, at: usize| -> Result<V, Why> {
                let r = *regs.get(at).ok_or("an image address")?;
                let v = inst.read(Reg::V(r)).ok_or("an image address")?;
                l.i32_of(v)
            };
            // A float address: a dword, or an f16 half of one.
            let float = |l: &mut Self, (at, half): (usize, Option<u32>)| -> Result<V, Why> {
                let x = raw(l, at)?;
                let Some(h) = half else {
                    return Ok(l.as_float(x));
                };
                let c = l.cur;
                let (sh, mask) = (l.s.m.i32(16 * h), l.s.m.i32(0xFFFF));
                let x = l.s.m.bin(c, Bin::LShr, x, sh);
                let x = l.s.m.bin(c, Bin::And, x, mask);
                let f32t = l.s.t.f32;
                Ok(l.s.op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x]))
            };
            // A float address known at translation (an f16 only when 0).
            let known =
                |l: &mut Self, (at, half): (usize, Option<u32>)| -> Result<Option<f32>, Why> {
                    let x = raw(l, at)?;
                    Ok(l.s.m.int_value(x).and_then(|bits| match half {
                        None => Some(f32::from_bits(bits as u32)),
                        Some(h) => (bits >> (16 * h) & 0x7FFF == 0).then_some(0.0),
                    }))
                };
            let level = if f.l {
                known(self, coord_at(coords))?
            } else {
                f.lz.then_some(0.0)
            };
            let level_zero = level == Some(0.0);
            // The DXIL operation. SM 6.6 has no comparison at a level, with
            // a bias or with gradients (SampleCmpLevel is SM 6.7,
            // SampleCmpBias and SampleCmpGrad 6.8) and gathers level 0 only;
            // those forms are taken where a level or bias known to be 0 makes
            // them one SM 6.6 has.
            let op = if op == 81 {
                81
            } else if f.gather {
                if f.b {
                    return Err("a gather with a level bias");
                }
                if !f.l && !f.lz {
                    return Err("a gather with implicit level");
                }
                if dmask.count_ones() != 1 {
                    return Err("a gather of several channels");
                }
                if !matches!(kind, 2 | 5 | 7 | 9) {
                    return Err("a gather from a 1D or 3D image");
                }
                if level_zero {
                    if f.c { 74 } else { 73 }
                } else if f.c || matches!(kind, 5 | 9) {
                    // A comparison's function is in the sampler, and a cube's
                    // texel grid is per face.
                    return Err("a gather at a level not known to be 0");
                } else {
                    GATHER_AT_LEVEL
                }
            } else if f.c {
                if f.d {
                    return Err("a comparison sample with explicit derivatives");
                }
                if f.l && !level_zero {
                    return Err("a comparison sample at a level not known to be 0");
                }
                if f.b && known(self, bias_at)?.is_none_or(|b| b != 0.0) {
                    return Err("a comparison sample with a level bias");
                }
                if f.l || f.lz {
                    65
                } else if self.s.pixel {
                    64
                } else if self.s.threads.is_some() && self.s.mesh.is_none() {
                    // Implicit derivatives in compute: the host's, from its
                    // quads, which must be the console's.
                    self.compute_quads()?;
                    64
                } else {
                    return Err(
                        "a comparison sample with implicit derivatives outside pixel and compute shaders",
                    );
                }
            } else if f.d {
                63
            } else if f.l || f.lz {
                62
            } else if f.b {
                61
            } else {
                60
            };
            let mut cs = vec![fu; 4];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = float(self, coord_at(k))?;
            }
            let mut cube_face = None;
            if matches!(kind, 5 | 9) {
                if *unorm || op == 81 {
                    return Err("a cube sample with texel coordinates or a level query");
                }
                if f.o {
                    return Err("a texel offset on a cube image");
                }
                let face = if kind == 9 {
                    // Face id plus 8 × the slice: the face is the low three
                    // bits of the whole number, the slice the rest (the
                    // host's array index, rounded and clamped as the
                    // console clamps it).
                    let i32t = self.s.t.i32;
                    let n = self.s.m.cast(c, Cast::FpToUi, cs[2], i32t);
                    let (k7, k3) = (self.s.m.i32(7), self.s.m.i32(3));
                    let face = self.s.m.bin(c, Bin::And, n, k7);
                    let slice = self.s.m.bin(c, Bin::LShr, n, k3);
                    cs[3] = self.s.m.cast(c, Cast::UiToFp, slice, self.s.t.f32);
                    CubeFace::Index(face)
                } else {
                    CubeFace::Float(cs[2])
                };
                cube_face = Some(face);
                let dir = self.cube_direction(cs[0], cs[1], face);
                cs[..3].copy_from_slice(&dir);
            }
            if *unorm {
                // Texel coordinates: divided by the size of level 0.
                let i32t = self.s.t.i32;
                let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
                let z = self.s.m.i32(0);
                let d = self
                    .s
                    .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, z]);
                for (k, slot) in cs.iter_mut().enumerate().take(spatial) {
                    let n = self.s.m.extract(c, d, k as u32);
                    let n = self.s.m.cast(c, Cast::UiToFp, n, self.s.t.f32);
                    *slot = self.s.m.bin(c, Bin::SDiv, *slot, n);
                }
            }
            if op == 81 {
                // The clamped then the unclamped level of detail.
                let Operand::Vector { reg, .. } = *data else {
                    return Err("an image result");
                };
                let f32t = self.s.t.f32;
                // The coordinates the level depends on: an array's slice is
                // none of them (DXIL wants it undef).
                let mut at = [fu; 3];
                at[..spatial].copy_from_slice(&cs[..spatial]);
                for (k, ch) in (0..2u32).filter(|ch| dmask >> ch & 1 != 0).enumerate() {
                    let clamped = self.s.m.bool(ch == 0);
                    let [lod] = sampler.dispatch(self, |s, block, sh| {
                        [s.op(
                            block,
                            "calculateLOD.f32",
                            Attr::ReadOnly,
                            f32t,
                            81,
                            &[h, sh, at[0], at[1], at[2], clamped],
                        )]
                    });
                    let x = self.as_int(lod);
                    self.write(inst, Reg::V(reg + k as u8), x, guard);
                }
                return Ok(());
            }
            let offsets = if matches!(kind, 5 | 9) {
                0
            } else {
                spatial.min(3)
            };
            let zero = self.s.m.i32(0);
            let mut os = vec![iu; 3];
            for o in os.iter_mut().take(offsets) {
                *o = zero;
            }
            if f.o {
                // Texel offsets: x in bits 0-5, y in 8-13, z in 16-21, signed.
                let w = raw(self, 0)?;
                let field = |w: i64, k: usize| (w as i32) << (26 - 8 * k) >> 26;
                let immediate = self
                    .s
                    .m
                    .int_value(w)
                    .filter(|&w| (0..offsets).all(|k| (-8..=7).contains(&field(w, k))));
                if let (Some(w), false) = (immediate, f.gather) {
                    for (k, o) in os.iter_mut().enumerate().take(offsets) {
                        *o = self.s.m.i32(field(w, k) as u32);
                    }
                } else {
                    // Bits at..at+5 sign-extended: shift up, then down.
                    let k26 = self.s.m.i32(26);
                    for (k, o) in os.iter_mut().enumerate().take(offsets) {
                        let left = self.s.m.i32(26 - 8 * k as u32);
                        let t = self.s.m.bin(c, Bin::Shl, w, left);
                        *o = self.s.m.bin(c, Bin::AShr, t, k26);
                    }
                }
                if !f.gather && immediate.is_none() {
                    // A sample's offsets are immediates in -8..7 (gathers
                    // take any, per lane). At a level known here, the
                    // coordinates move by offset / that level's size instead:
                    // the same texels, the coordinate rounded once more (the
                    // console adds the offset to the texel coordinate). A
                    // sampler whose LOD clamp or bias moves the sampled
                    // level away from the one given, or a level past the
                    // view's last, differs.
                    let Some(n) = level.filter(|n| *n >= 0.0 && n.fract() == 0.0) else {
                        return Err(
                            "a texel offset past -8..7 or per lane at a level not known here",
                        );
                    };
                    let i32t = self.s.t.i32;
                    let f32t = self.s.t.f32;
                    let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
                    let n = self.s.m.i32(n as u32);
                    let d = self
                        .s
                        .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, n]);
                    for (k, o) in os.iter_mut().enumerate().take(offsets) {
                        let size = self.s.m.extract(c, d, k as u32);
                        let size = self.s.m.cast(c, Cast::UiToFp, size, f32t);
                        let texels = self.s.m.cast(c, Cast::SiToFp, *o, f32t);
                        let shift = self.s.m.bin(c, Bin::SDiv, texels, size);
                        cs[k] = self.s.m.bin(c, Bin::Add, cs[k], shift);
                        *o = zero;
                    }
                }
            }
            // The level clamp, after the coordinates.
            let clamp = if f.cl {
                float(self, coord_at(coords))?
            } else {
                fu
            };
            let mut args = vec![h, sampler.first];
            args.extend(&cs);
            match op {
                60 | 61 if !self.s.pixel => {
                    // Outside pixel shaders the console still takes the level
                    // from each quad of lanes (lane 1 minus lane 0 across,
                    // lane 2 minus lane 0 down); DXIL before validator 1.8
                    // has no implicit derivatives there, so the quad's
                    // differences become explicit gradients. A bias scales
                    // them by 2^bias, the same level for isotropic filtering.
                    // A quad's four lanes share a host wave of any size, so
                    // this needs neither the console's wave size nor its
                    // lane order beyond the quad; the host's quad is the
                    // console's in a group declared as one row
                    // (`compute_quads`).
                    if self.s.mesh.is_some() {
                        return Err("a cross-lane operation in a mesh shader");
                    }
                    if self.s.threads.is_none() {
                        return Err("a cross-lane operation outside compute");
                    }
                    self.compute_quads()?;
                    let scale = if op == 61 {
                        let bias = float(self, bias_at)?;
                        Some(self.unary(21, bias))
                    } else {
                        None
                    };
                    let g = self.quad_gradients(&cs[..spatial], scale);
                    args.extend(&os);
                    args.extend(g);
                    args.push(clamp);
                    sampler.sample(self, "sampleGrad.f32", 63, &mut args)
                }
                60 => {
                    args.extend(&os);
                    args.push(clamp);
                    sampler.sample(self, "sample.f32", 60, &mut args)
                }
                61 => {
                    let bias = float(self, bias_at)?;
                    args.extend(&os);
                    args.extend([bias, clamp]);
                    sampler.sample(self, "sampleBias.f32", 61, &mut args)
                }
                63 => {
                    // shortcut: the coarse forms (`_cd`, including `_g16`)
                    // compute one level per quad; SampleGrad selects a level
                    // per lane, which agrees when the gradients match across
                    // the quad.
                    let mut g = vec![fu; 6];
                    if let Some(face) = cube_face {
                        // The face coordinates' derivatives, as the
                        // direction's (`cube_gradient`).
                        for dir in 0..2 {
                            let ds = float(self, grad_at(dir, 0))?;
                            let dt = float(self, grad_at(dir, 1))?;
                            let d = self.cube_gradient(ds, dt, face);
                            g[3 * dir..3 * dir + 3].copy_from_slice(&d);
                        }
                    } else {
                        for k in 0..spatial {
                            g[k] = float(self, grad_at(0, k))?;
                            g[3 + k] = float(self, grad_at(1, k))?;
                        }
                    }
                    args.extend(&os);
                    args.extend(g);
                    args.push(clamp);
                    sampler.sample(self, "sampleGrad.f32", 63, &mut args)
                }
                64 | 65 => {
                    let at = usize::from(f.o) + usize::from(f.b);
                    let reference = float(self, (at, None))?;
                    args.extend(&os);
                    args.push(reference);
                    let name = if op == 64 {
                        args.push(clamp);
                        "sampleCmp.f32"
                    } else {
                        "sampleCmpLevelZero.f32"
                    };
                    sampler.sample(self, name, op, &mut args)
                }
                GATHER_AT_LEVEL => {
                    // A gather at a level the code computes. DXIL's gathers
                    // read level 0 only, so the four texels of the footprint
                    // are sampled one by one at their centres at that level:
                    // at a texel's centre every filter returns that texel,
                    // and the sampler applies its own address mode and level
                    // range, as it does for the console's gather.
                    // shortcut: the footprint is that of the level rounded to
                    // the nearest whole one and clamped to the view's; a
                    // sampler whose LOD bias or range moves the sampled level
                    // off it reads another grid.
                    let lod = float(self, coord_at(coords))?;
                    let (i32t, f32t) = (self.s.t.i32, self.s.t.f32);
                    let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
                    let d0 = self
                        .s
                        .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, zero]);
                    let levels = self.s.m.extract(c, d0, 3);
                    let one = self.s.m.i32(1);
                    let last = self.s.m.bin(c, Bin::Sub, levels, one);
                    let li = self.unary(26, lod);
                    let li = self.s.m.cast(c, Cast::FpToSi, li, i32t);
                    let over = self.s.m.cmp(c, pred::SGT, li, last);
                    let li = self.s.m.select(c, over, last, li);
                    let under = self.s.m.cmp(c, pred::SLT, li, zero);
                    let li = self.s.m.select(c, under, zero, li);
                    let d = self
                        .s
                        .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, li]);
                    let level = self.s.m.cast(c, Cast::SiToFp, li, f32t);
                    let half = self.s.m.f32(0.5f32.to_bits());
                    // The footprint's lower texel, floor(u * size - 0.5),
                    // moved by the texel offsets.
                    let mut size = [fu; 2];
                    let mut low = [fu; 2];
                    for k in 0..2 {
                        let n = self.s.m.extract(c, d, k as u32);
                        let n = self.s.m.cast(c, Cast::UiToFp, n, f32t);
                        let t = self.s.m.bin(c, Bin::Mul, cs[k], n);
                        let t = self.s.m.bin(c, Bin::Sub, t, half);
                        let t = self.unary(27, t);
                        let o = self.s.m.cast(c, Cast::SiToFp, os[k], f32t);
                        low[k] = self.s.m.bin(c, Bin::Add, t, o);
                        size[k] = n;
                    }
                    let mut none = vec![iu; 3];
                    for o in none.iter_mut().take(offsets) {
                        *o = zero;
                    }
                    let channel = dmask.trailing_zeros();
                    let mut texels = [fu; 4];
                    // The console's (and DXIL's) gather order.
                    for (n, by) in [(0.0f32, 1.0f32), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
                        .into_iter()
                        .enumerate()
                    {
                        let c = self.cur;
                        let mut at = cs.clone();
                        for (k, step) in [by.0, by.1].into_iter().enumerate() {
                            let centre = self.s.m.f32((step + 0.5).to_bits());
                            let x = self.s.m.bin(c, Bin::Add, low[k], centre);
                            at[k] = self.s.m.bin(c, Bin::SDiv, x, size[k]);
                        }
                        let mut a = vec![h, sampler.first];
                        a.extend(&at);
                        a.extend(&none);
                        a.push(level);
                        let r = sampler.sample(self, "sampleLevel.f32", 62, &mut a);
                        texels[n] = r.channel(&mut self.s, self.cur, channel);
                    }
                    Texels::Channels(texels)
                }
                62 => {
                    let lod = if f.l {
                        float(self, coord_at(coords))?
                    } else {
                        self.s.m.f32(0)
                    };
                    args.extend(&os);
                    args.push(lod);
                    sampler.sample(self, "sampleLevel.f32", 62, &mut args)
                }
                _ => {
                    args.extend(&os[..2]);
                    args.push(self.s.m.i32(dmask.trailing_zeros()));
                    let name = if op == 74 {
                        let reference = float(self, (usize::from(f.o), None))?;
                        args.push(reference);
                        "textureGatherCmp.f32"
                    } else {
                        "textureGather.f32"
                    };
                    sampler.sample(self, name, op, &mut args)
                }
            }
        } else {
            let mut cs = vec![iu; 3];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = addr_val(self, k)?;
            }
            let mip = if name == "image_load_mip" {
                addr_val(self, coords)?
            } else if uav {
                // Our declarations have four channels.
                self.s.flags |= flags::TYPED_UAV_LOADS;
                iu
            } else {
                self.s.m.i32(0)
            };
            let mut args = vec![h, mip];
            args.extend(&cs);
            args.extend([iu, iu, iu]);
            Texels::Resource(self.s.op(
                c,
                "textureLoad.f32",
                Attr::ReadOnly,
                self.s.t.res_ret,
                66,
                &args,
            ))
        };
        let Operand::Vector { reg, .. } = *data else {
            return Err("an image result");
        };
        let c = self.cur;
        let channels: Vec<u32> = if gather {
            vec![0, 1, 2, 3]
        } else {
            (0..4).filter(|ch| dmask >> ch & 1 != 0).collect()
        };
        let mut xs = Vec::new();
        for ch in channels {
            let x = r.channel(&mut self.s, c, ch);
            xs.push(if *d16 {
                // shortcut: d16 results are taken as floats (rounded to
                // nearest even); integer formats would keep 16 bits.
                let i32t = self.s.t.i32;
                self.s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[x])
            } else {
                self.as_int(x)
            });
        }
        if *d16 {
            // Two channels per VGPR, low half first (an odd last high half 0).
            let k16 = self.s.m.i32(16);
            xs = xs
                .chunks(2)
                .map(|p| match *p {
                    [lo, hi] => {
                        let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                        self.s.m.bin(c, Bin::Or, lo, hi)
                    }
                    _ => p[0],
                })
                .collect();
        }
        for (k, x) in xs.into_iter().enumerate() {
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }

    /// Coarse derivatives of `coords` across this lane's quad (lanes 4n to
    /// 4n + 3: lane 1 minus lane 0, then lane 2 minus lane 0), as DXIL
    /// gradients (x for up to three coordinates, then y), each times
    /// `scale` if given.
    // shortcut: a quad lane the program left inactive gives what the host
    // reads there (the console's derivative reads that lane's stale value).
    fn quad_gradients(&mut self, coords: &[V], scale: Option<V>) -> Vec<V> {
        self.s.flags |= flags::WAVE_OPS;
        let c = self.cur;
        let f32t = self.s.t.f32;
        let fu = self.s.m.undef(f32t);
        let mut g = vec![fu; 6];
        for (k, &x) in coords.iter().enumerate().take(3) {
            // The quad's lane `n` (a constant, as quad reads take it).
            let at = |l: &mut Self, n: u32| {
                let n = l.s.m.i32(n);
                l.s.op(c, "quadReadLaneAt.f32", Attr::NoUnwind, f32t, 122, &[x, n])
            };
            let origin = at(self, 0);
            for d in 0..2 {
                let there = at(self, d as u32 + 1);
                let mut diff = self.s.m.bin(c, Bin::Sub, there, origin);
                if let Some(s) = scale {
                    diff = self.s.m.bin(c, Bin::Mul, diff, s);
                }
                g[3 * d + k] = diff;
            }
        }
        g
    }

    /// The direction a cube sample's face coordinates stand for: the
    /// console takes s and t in [1, 2] (`v_cubesc`/`v_cubetc` over
    /// `|v_cubema|`, plus 1.5) and the face id (`v_cubeid`), which the
    /// host's cube sampling turns back into the same texel. A cube array's
    /// slice is its own coordinate on the host ([`Self::cube_view`]).
    fn cube_direction(&mut self, s: V, t: V, face: CubeFace) -> [V; 3] {
        let c = self.cur;
        let k = |l: &mut Self, v: f32| l.s.m.f32(v.to_bits());
        let (two, three) = (k(self, 2.0), k(self, 3.0));
        let u = self.s.m.bin(c, Bin::Mul, s, two);
        let u = self.s.m.bin(c, Bin::Sub, u, three);
        let v = self.s.m.bin(c, Bin::Mul, t, two);
        let v = self.s.m.bin(c, Bin::Sub, v, three);
        let z = self.s.m.f32(0x8000_0000);
        let (nu, nv) = (
            self.s.m.bin(c, Bin::Sub, z, u),
            self.s.m.bin(c, Bin::Sub, z, v),
        );
        let (one, mone) = (k(self, 1.0), k(self, -1.0));
        // Per face +x, -x, +y, -y, +z, -z.
        self.by_face(
            face,
            [
                [one, mone, u, u, u, nu],
                [nv, nv, one, mone, nv, nv],
                [nu, u, v, nv, one, mone],
            ],
        )
    }

    /// The derivative of a cube sample's direction for derivatives `ds` and
    /// `dt` of its face coordinates: [`Self::cube_direction`] is linear in s
    /// and t on a face (u = 2s - 3, v = 2t - 3), so its derivative is the
    /// same table with 2 ds for u, 2 dt for v and 0 for the constant axis.
    fn cube_gradient(&mut self, ds: V, dt: V, face: CubeFace) -> [V; 3] {
        let c = self.cur;
        let two = self.s.m.f32(2.0f32.to_bits());
        let u = self.s.m.bin(c, Bin::Mul, ds, two);
        let v = self.s.m.bin(c, Bin::Mul, dt, two);
        let z = self.s.m.f32(0x8000_0000);
        let (nu, nv) = (
            self.s.m.bin(c, Bin::Sub, z, u),
            self.s.m.bin(c, Bin::Sub, z, v),
        );
        let o = self.s.m.f32(0);
        self.by_face(
            face,
            [
                [o, o, u, u, u, nu],
                [nv, nv, o, o, nv, nv],
                [nu, u, v, nv, o, o],
            ],
        )
    }

    /// A cube image instruction's view: a cube (DXIL kind 5) or a cube
    /// array (9). The console has one image dimension for both, and the
    /// host view is the one the title made, of the dimension its HLSL
    /// declared, so the translation must declare the same.
    ///
    /// A sample or gather tells by its face operand: the console compiler
    /// gives a cube the face id `v_cubeid_f32` makes, and a cube array that
    /// id plus 8 × the slice (Mesa `ac_prepare_cube_coords` does the same).
    /// A size query tells by asking for the layer count (a cube's HLSL size
    /// has none), or by a sample of the same descriptor that is an array's.
    // shortcut: a face operand traced to neither (one the program loads, or
    // computes otherwise) is taken as a cube's face, as before; a cube
    // array's slice there reads slice 0.
    fn cube_view(&self, inst: &Inst, src: &Source) -> u32 {
        let Fields::Image { dmask, .. } = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        if name == "image_get_resinfo" {
            let p = self.x.p;
            let sampled_as_array = self.descriptors.iter().any(|(&((b, i), sampler), &s)| {
                let other = &p.blocks[b].insts[i];
                !sampler
                    && s == src
                    && other.name() != "image_get_resinfo"
                    && matches!(other.fields, Fields::Image { dim: 3, .. })
                    && self.face_layer(other) == Layer::Slice
            });
            return if dmask >> 2 & 1 != 0 || sampled_as_array {
                9
            } else {
                5
            };
        }
        match self.face_layer(inst) {
            Layer::Face | Layer::Unknown => 5,
            Layer::Slice => 9,
        }
    }

    /// What the face operand of cube sample or gather `inst` is.
    fn face_layer(&self, inst: &Inst) -> Layer {
        let Fields::Image { addr, a16, .. } = &inst.fields else {
            return Layer::Unknown;
        };
        let Some(form) = sample_form(inst.name()) else {
            return Layer::Unknown;
        };
        let regs: Vec<u8> = addr
            .iter()
            .filter_map(|a| match *a {
                Operand::Vector { reg, dwords } => Some((0..dwords).map(move |k| reg + k)),
                _ => None,
            })
            .flatten()
            .collect();
        let (at, half) = cube_face_at(form, *a16);
        let Some(v) = regs.get(at).and_then(|&r| inst.read(Reg::V(r))) else {
            return Layer::Unknown;
        };
        let mut seen = HashSet::new();
        match half {
            None => self.layer_of(v, LAYER_DEPTH, &mut seen),
            Some(_) => self.layer_of_low_half(v, &mut seen),
        }
    }

    /// [`Self::layer_of`] for the low f16 half of `v`: the f32 face the
    /// half was converted from (`v_cvt_pkrtz_f16_f32`, `v_pack_b32_f16` of
    /// a `v_cvt_f16_f32`, or that conversion alone).
    fn layer_of_low_half(&self, v: Value, seen: &mut HashSet<Value>) -> Layer {
        let Some((inst, f)) = self.plain_valu_def(v) else {
            return Layer::Unknown;
        };
        match inst.name() {
            "v_cvt_pkrtz_f16_f32" | "v_cvt_f16_f32" => {
                self.layer_of_operand(inst, f, 0, LAYER_DEPTH, seen)
            }
            "v_pack_b32_f16" => match f.src.first() {
                Some(&Operand::Vector { reg, .. }) => match inst.read(Reg::V(reg)) {
                    Some(x) => self.layer_of_low_half(x, seen),
                    None => Layer::Unknown,
                },
                _ => Layer::Unknown,
            },
            _ => Layer::Unknown,
        }
    }

    /// What value `v`, a cube sample's face operand, is: followed back
    /// through moves, selects and joins to the face id, or to that id plus
    /// 8 × something (a multiply-add by 8.0, or an addition of a product by
    /// 8.0 or of a multiple of 8). `depth` bounds the search.
    fn layer_of(&self, v: Value, depth: u32, seen: &mut HashSet<Value>) -> Layer {
        if depth == 0 {
            return Layer::Unknown;
        }
        if !seen.insert(v) {
            // A loop back to a value on the way: what the other arguments
            // say (`Face` is the join's identity).
            return Layer::Face;
        }
        let p = self.x.p;
        let Some(&b) = self.def_block.get(&v) else {
            return Layer::Unknown;
        };
        if let Some(phi) = p.blocks[b].phis.iter().find(|ph| ph.value == v) {
            let mut l = Layer::Face;
            for &a in &phi.args {
                l = l.join(self.layer_of(a, depth - 1, seen));
            }
            return l;
        }
        let Some((inst, f)) = self.plain_valu_def(v) else {
            return Layer::Unknown;
        };
        let d = depth - 1;
        let op =
            |l: &Self, k: usize, seen: &mut HashSet<Value>| l.layer_of_operand(inst, f, k, d, seen);
        let some_face = |l: Layer| matches!(l, Layer::Face | Layer::Slice);
        match inst.name() {
            "v_cubeid_f32" => Layer::Face,
            "v_mov_b32" => op(self, 0, seen),
            "v_cndmask_b32" => op(self, 0, seen).join(op(self, 1, seen)),
            // D = S0 × S1 + D: a face plus 8 × a slice.
            "v_fmac_f32" | "v_mac_f32" => {
                let Some(Operand::Vector { reg, .. }) = f.dst else {
                    return Layer::Unknown;
                };
                let acc = inst.read(Reg::V(reg));
                let by_eight = (0..2).any(|k| self.constant(inst, f, k) == Some(EIGHT));
                match acc {
                    Some(acc) if by_eight && some_face(self.layer_of(acc, d, seen)) => Layer::Slice,
                    _ => Layer::Unknown,
                }
            }
            // S0 × S1 + S2 (the mk and ak forms in that order).
            "v_fma_f32" | "v_mad_f32" | "v_fmamk_f32" | "v_madmk_f32" => {
                let by_eight = (0..2).any(|k| self.constant(inst, f, k) == Some(EIGHT));
                if by_eight && some_face(op(self, 2, seen)) {
                    Layer::Slice
                } else {
                    Layer::Unknown
                }
            }
            "v_add_f32" => {
                let eight_times = |l: &Self, k: usize| l.eight_times(inst, f, k);
                if eight_times(self, 0) && some_face(op(self, 1, seen))
                    || eight_times(self, 1) && some_face(op(self, 0, seen))
                {
                    Layer::Slice
                } else {
                    Layer::Unknown
                }
            }
            _ => Layer::Unknown,
        }
    }

    /// [`Self::layer_of`] of source `k` of `inst`: a constant's own, a
    /// VGPR's value's.
    fn layer_of_operand(
        &self,
        inst: &Inst,
        f: &crate::gfx10::Valu,
        k: usize,
        depth: u32,
        seen: &mut HashSet<Value>,
    ) -> Layer {
        if let Some(bits) = self.constant(inst, f, k) {
            return Layer::of_constant(bits);
        }
        if f.neg >> k & 1 != 0 || f.abs >> k & 1 != 0 {
            return Layer::Unknown;
        }
        match f.src.get(k) {
            Some(&Operand::Vector { reg, .. }) => match inst.read(Reg::V(reg)) {
                Some(x) => self.layer_of(x, depth, seen),
                None => Layer::Unknown,
            },
            _ => Layer::Unknown,
        }
    }

    /// Whether source `k` of `inst` is a multiple of 8: a constant one, or
    /// a product with 8.0.
    fn eight_times(&self, inst: &Inst, f: &crate::gfx10::Valu, k: usize) -> bool {
        if let Some(bits) = self.constant(inst, f, k) {
            let x = f32::from_bits(bits);
            return (0.0..16_777_216.0).contains(&x)
                && x.fract() == 0.0
                && (x as u32).is_multiple_of(8);
        }
        let Some(&Operand::Vector { reg, .. }) = f.src.get(k) else {
            return false;
        };
        let Some((m, mf)) = inst.read(Reg::V(reg)).and_then(|x| self.plain_valu_def(x)) else {
            return false;
        };
        matches!(m.name(), "v_mul_f32" | "v_mul_legacy_f32")
            && (0..2).any(|j| self.constant(m, mf, j) == Some(EIGHT))
    }

    /// Source `k` of `inst` as a constant: an inline constant or literal,
    /// or a register a move of one wrote (unmodified).
    fn constant(&self, inst: &Inst, f: &crate::gfx10::Valu, k: usize) -> Option<u32> {
        if f.neg >> k & 1 != 0 || f.abs >> k & 1 != 0 {
            return None;
        }
        let reg = match *f.src.get(k)? {
            Operand::Literal(bits) => return Some(bits),
            Operand::Const(code) => return crate::resource::inline(code),
            Operand::Vector { reg, .. } => Reg::V(reg),
            Operand::Scalar { code, .. } => Reg::S(code),
            Operand::Attr { .. } => return None,
        };
        let v = inst.read(reg)?;
        let &b = self.def_block.get(&v)?;
        let def = self.x.p.blocks[b]
            .insts
            .iter()
            .find(|i| i.writes.iter().any(|w| w.1 == v))?;
        if !matches!(def.name(), "v_mov_b32" | "s_mov_b32") {
            return None;
        }
        let src = match &def.fields {
            Fields::Valu(g) if g.dpp.is_none() && g.sdwa.is_none() && g.neg | g.abs == 0 => {
                g.src.first()?
            }
            Fields::Sop { src, .. } => src.first()?,
            _ => return None,
        };
        match *src {
            Operand::Literal(bits) => Some(bits),
            Operand::Const(code) => crate::resource::inline(code),
            _ => None,
        }
    }

    /// The VALU instruction that writes `v` as its result, with no DPP,
    /// SDWA, clamp or output modifier (a move, select, multiply-add...).
    fn plain_valu_def(&self, v: Value) -> Option<(&'a Inst, &'a crate::gfx10::Valu)> {
        let p = self.x.p;
        let &b = self.def_block.get(&v)?;
        let inst = p.blocks[b]
            .insts
            .iter()
            .find(|i| i.writes.iter().any(|w| w.1 == v))?;
        let Fields::Valu(f) = &inst.fields else {
            return None;
        };
        let result = matches!(f.dst, Some(Operand::Vector { reg, .. })
            if inst.write(Reg::V(reg)) == Some(v));
        (result && f.dpp.is_none() && f.sdwa.is_none() && !f.clamp && f.omod == 0)
            .then_some((inst, f))
    }

    /// Each row's value for cube face `face` (0-5).
    fn by_face(&mut self, face: CubeFace, table: [[V; 6]; 3]) -> [V; 3] {
        let c = self.cur;
        let f = match face {
            CubeFace::Float(x) => self.s.m.cast(c, Cast::FpToUi, x, self.s.t.i32),
            CubeFace::Index(n) => n,
        };
        table.map(|row| {
            let mut r = row[0];
            for (n, &x) in row.iter().enumerate().skip(1) {
                let kn = self.s.m.i32(n as u32);
                let is = self.s.m.cmp(c, pred::EQ, f, kn);
                r = self.s.m.select(c, is, x, r);
            }
            r
        })
    }
}
