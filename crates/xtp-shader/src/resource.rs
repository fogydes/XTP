//! Resource tracking (decision 0003): where each descriptor a program uses
//! comes from, in terms of the root signature.
//!
//! Root parameters arrive in user SGPRs ([`RootParam`]): a descriptor table
//! as the 32-bit heap address of its range (the code sets the high half to
//! 0), a root descriptor as a 64-bit buffer address, root constants as
//! themselves. A root signature with dynamic resources (SM 6.6) adds the
//! 32-bit base addresses of both heaps; a descriptor is then loaded from
//! one of them at an index the code computes (from a root constant, say). The code loads table descriptors from heap memory with
//! `s_load_dword*`, and builds a buffer descriptor around a root
//! descriptor's address itself. Each descriptor operand and each scalar
//! memory address is traced through the SSA values back to those
//! parameters, by data flow: copies, phis that agree, and additions.

use crate::ShaderError;
use crate::gfx10::{Fields, NULL, Operand};
use crate::ir::{EXEC, Exit, Inst, Program, Reg, Value};
use crate::native::{HwStage, RootParam, RootType};
use crate::structure::{Atom, Bool, LaneExit, Structured};

#[cfg(test)]
mod guarded_address_tests;
mod spill;
mod uniform;

/// An instruction: block and index in [`crate::ir::Block::insts`].
pub type Site = (usize, usize);

/// A byte address: a root parameter's, plus a constant and at most one
/// value the program computes (an index times a stride, say).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Address {
    /// Index into the layout, or [`CODE`].
    pub param: usize,
    /// Added modulo 2^32 (with carry into the high half of a root
    /// descriptor's address).
    pub offset: u32,
    pub dynamic: Option<Value>,
}

/// [`Address::param`] of the program's own code: a constant table the
/// compiler appends after the code and reads through `s_getpc_b64`; the
/// offset is then the byte offset in the code.
pub const CODE: usize = usize::MAX;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    /// Dwords `first..` of what the scalar load at `load` reads (see
    /// [`Resources::scalar`] for its address).
    Loaded { load: Site, first: u8 },
    /// A buffer descriptor built around an address: `bits` OR-ed into
    /// dword 1 (stride, swizzle), dwords 2 and 3 as given.
    Built {
        address: Address,
        bits: u32,
        words: [u32; 2],
    },
    /// Fixed in the code (an immediate sampler, say).
    Constant(Vec<u32>),
    /// Chosen by control flow: one source per predecessor of `block`, in
    /// [`crate::ir::Block::preds`] order.
    Phi { block: usize, args: Vec<Source> },
    /// Chosen by SCC (`scc`, the value an `s_cselect` or `s_cmov` of each
    /// dword reads): the source when it is set, then when it is clear.
    Select { scc: Value, args: Vec<Source> },
    /// Dwords `first..` of what the vector load at `load` (`global_load_*`)
    /// reads, each lane at its own address (see [`Resources::scalar`]); the
    /// dwords past that load come from loads of the bytes that follow. A
    /// descriptor that differs between lanes.
    Fetched { load: Site, first: u8 },
    /// Picked per lane by `v_cndmask_b32` reading the lane mask whose low
    /// dword is `mask`: the source where the lane's bit is set, then where it
    /// is clear.
    Lanes { mask: Value, args: Vec<Source> },
    /// Per-lane dwords chosen by control flow (VGPRs joined where lanes
    /// that took different paths meet): one source per predecessor of
    /// `block`, in [`crate::ir::Block::preds`] order, `None` on an edge no
    /// lane that accesses the descriptor takes. A descriptor that differs
    /// between lanes.
    LanePhi {
        block: usize,
        args: Vec<Option<Source>>,
    },
}

impl Source {
    /// The sources this one picks from (itself unless a phi).
    pub fn leaves(&self) -> Vec<&Source> {
        match self {
            Source::Phi { args, .. } | Source::Select { args, .. } | Source::Lanes { args, .. } => {
                args.iter().flat_map(Source::leaves).collect()
            }
            Source::LanePhi { args, .. } => {
                args.iter().flatten().flat_map(Source::leaves).collect()
            }
            s => vec![s],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Image, buffer or `s_buffer_load` resource.
    Resource,
    Sampler,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Descriptor {
    pub site: Site,
    pub role: Role,
    pub source: Source,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Resources {
    /// Every scalar memory access by address (`s_load_*`, `s_store_*`), and
    /// every global memory access (`global_*`, the dynamic term per lane).
    pub scalar: Vec<(Site, Address)>,
    /// Lectures à déplacement fixe dont les constantes racine donnent toute l'adresse.
    pub constant_reads: std::collections::BTreeMap<Site, ConstantRead>,
    pub descriptors: Vec<Descriptor>,
    /// Loads replaced with root arguments, sorted by instruction site.
    pub spill_loads: Vec<Site>,
    /// Root arguments from user SGPRs or spill loads, by SSA value, layout
    /// index and dword within that parameter.
    pub params: Vec<(Value, usize, u32)>,
    pub waterfall: Waterfall,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstantRead {
    pub slot: u8,
    pub dword: u32,
    pub packed: bool,
    pub offset: u32,
    /// Déplacement non signé ajouté à la paire de constantes avec retenue.
    pub index: Option<Value>,
}

/// Taille maximale de la copie d'une SRV racine mal alignée sur l'hôte.
pub const MISALIGNED_SRV_BYTES: u32 = 1 << 20;

/// Waterfall loops over descriptors that differ between lanes: each pass
/// reads the first active lane's descriptor with `v_readfirstlane_b32`,
/// compares it with every lane's own (`v_cmp_eq`), and runs the access with
/// the lanes that match. Their accesses take each lane's own descriptor
/// ([`Source::Fetched`], [`Source::Lanes`]), so the first lane's dwords are
/// each lane's own too: the compares hold in every lane and one pass serves
/// all lanes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Waterfall {
    /// The compares, true in every lane.
    pub compares: Vec<Site>,
    /// The `v_readfirstlane_b32` results they compare, each the lane's own
    /// source dword.
    pub reads: Vec<Value>,
    /// Descriptor loads at a table offset among `reads` (the console
    /// compiler's form): each lane loads its own entry.
    pub loads: Vec<Site>,
}

/// Traces every descriptor and scalar address of `p`, which runs as
/// `stage`, to the root parameters in `layout`. Fails on the first one it
/// cannot trace. `lanes`, `p` as one lane runs it, tells which paths the
/// lanes that access a per-lane descriptor took and which writes they ran;
/// without it, per-lane descriptors chosen by control flow are refused and
/// per-lane writes are taken to run in every lane.
pub fn track(
    p: &Program,
    layout: &[RootParam],
    stage: HwStage,
    lanes: Option<&Structured>,
) -> Result<Resources, ShaderError> {
    trace(Tracer::new(p, layout, stage, lanes)?)
}

/// Traces a native program with its dispatch metadata. Compute programs
/// with a spilled root signature get their parameters from table loads.
pub fn track_native(
    p: &Program,
    native: &crate::native::NativeProgram,
    lanes: Option<&Structured>,
) -> Result<Resources, ShaderError> {
    let mut t = Tracer::new(p, &native.layout, native.stage, lanes)?;
    if spill::has_table(native) {
        t.read_spill_table()?;
    }
    trace(t)
}

fn trace(mut t: Tracer<'_>) -> Result<Resources, ShaderError> {
    let p = t.p;
    let mut out = Resources {
        params: p
            .inputs
            .iter()
            .filter_map(|&(reg, v)| {
                let (param, dword) = t.register_param(reg)?;
                Some((v, param, dword))
            })
            .collect(),
        spill_loads: std::mem::take(&mut t.spill.loads),
        ..Resources::default()
    };
    // Group accesses in the same loop so a shared mask can select several
    // descriptors together.
    let mut first_lane_at = std::collections::BTreeMap::<usize, (Vec<usize>, Vec<Value>)>::new();
    // The console compiler's waterfall reads the first lane's table offset
    // and loads that entry: accesses, offset reads and loads by block.
    let mut offset_at =
        std::collections::BTreeMap::<usize, (Vec<usize>, Vec<Value>, Vec<Site>)>::new();
    for (b, block) in p.blocks.iter().enumerate() {
        for (i, inst) in block.insts.iter().enumerate() {
            let site = (b, i);
            if out.spill_loads.binary_search(&site).is_ok() {
                out.params.extend(inst.writes.iter().filter_map(|&(_, v)| {
                    let &(param, dword) = t.spill.bindings.get(&v)?;
                    Some((v, param, dword))
                }));
                continue;
            }
            if t.spill.active
                && let Some(read) = t.constant_read(inst, site)
            {
                out.constant_reads.insert(site, read);
                continue;
            }
            let fail = |what| ShaderError::Unsupported {
                offset: inst.machine.offset,
                what,
            };
            let mut first_lane = Vec::new();
            let mut offset = Vec::new();
            let mut descriptor = |op: &Operand, role| -> Result<(), ShaderError> {
                let source = t.descriptor(inst, op, site).map_err(fail)?;
                first_lane.extend(t.first_lane_words(inst, op).unwrap_or_default());
                if let Source::Loaded { load, .. } = source
                    && let Some((_, a)) = out.scalar.iter().find(|s| s.0 == load)
                    && let Some(Term::Value(v)) = a.dynamic.map(|d| t.origin(d))
                    && t.first_lane_source(v).is_some()
                {
                    offset.push((v, load));
                }
                out.descriptors.push(Descriptor { site, role, source });
                Ok(())
            };
            match &inst.fields {
                Fields::Image { rsrc, sampler, .. } => {
                    descriptor(rsrc, Role::Resource)?;
                    if let Some(s) = sampler {
                        descriptor(s, Role::Sampler)?;
                    }
                }
                Fields::Buffer { rsrc, .. } => descriptor(rsrc, Role::Resource)?,
                Fields::Smem { base, .. } if inst.name().starts_with("s_buffer_") => {
                    descriptor(base, Role::Resource)?;
                }
                Fields::Smem {
                    base,
                    offset,
                    soffset,
                    ..
                } if inst.name().contains("_load_") || inst.name().contains("_store_") => {
                    let address = t.smem_address(site, base, *offset, soffset).map_err(fail)?;
                    out.scalar.push((site, address));
                }
                Fields::Flat {
                    addr,
                    saddr,
                    offset,
                    ..
                } if inst.name().starts_with("global_") => {
                    let address = t
                        .flat_address(site, addr.as_ref(), saddr.as_ref(), *offset)
                        .map_err(fail)?;
                    out.scalar.push((site, address));
                }
                _ => {}
            }
            if !first_lane.is_empty() {
                let (accesses, reads) = first_lane_at.entry(b).or_default();
                accesses.push(i);
                reads.extend(first_lane);
            }
            if !offset.is_empty() {
                let (accesses, reads, loads) = offset_at.entry(b).or_default();
                accesses.push(i);
                for (v, load) in offset {
                    reads.push(v);
                    loads.push(load);
                }
            }
        }
    }
    // The proof needs the per-lane form. Without it nothing is emitted, and
    // its failure is the one to report, so the reads stay unproven here.
    if t.lanes.is_none() {
        return Ok(out);
    }
    for (block, (accesses, mut reads)) in first_lane_at {
        reads.sort_unstable();
        reads.dedup();
        let proof =
            t.waterfall(block, &accesses, &reads)
                .map_err(|what| ShaderError::Unsupported {
                    offset: t.inst((block, accesses[0])).machine.offset,
                    what,
                })?;
        out.waterfall.compares.extend(proof.compares);
        out.waterfall.reads.extend(proof.reads);
    }
    // Unproven, an offset read stays the first lane's (one uniform entry),
    // which is what the hardware reads outside a waterfall loop.
    for (block, (accesses, mut reads, loads)) in offset_at {
        reads.sort_unstable();
        reads.dedup();
        if let Ok(proof) = t.waterfall(block, &accesses, &reads) {
            out.waterfall.compares.extend(proof.compares);
            out.waterfall.reads.extend(proof.reads);
            out.waterfall.loads.extend(loads);
        }
    }
    Ok(out)
}

/// A dword after copy propagation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Term {
    Const(u32),
    Value(Value),
}

/// How many steps deep [`Tracer::code_base`] and [`Tracer::code_high`]
/// follow an address in the code.
const CODE_DEPTH: u32 = 4;

/// What the additions computing a value hold as their base
/// ([`Tracer::code_base`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CodeBase {
    /// None: an index, a constant.
    None,
    /// The low dword of this `s_getpc_b64`, once.
    Code(Value),
    /// A root parameter, two bases, or more steps than followed.
    Other,
}

/// A 32-bit sum: optional root parameter (as its first dword), constant,
/// optional computed term.
struct Sum {
    base: Option<Value>,
    offset: u32,
    dynamic: Option<Value>,
}

struct Tracer<'a> {
    p: &'a Program,
    layout: &'a [RootParam],
    stage: HwStage,
    /// Defining instruction per value (phis and inputs have none).
    defs: Vec<Option<Site>>,
    /// Block and index of each phi value.
    phis: Vec<Option<(usize, usize)>>,
    /// Register per input value.
    inputs: Vec<Option<Reg>>,
    origin: Vec<Term>,
    spill: spill::Spill,
    lanes: Option<&'a Structured>,
    /// Reverse-postorder index per block.
    rpo: Vec<usize>,
    /// Which values every lane holds alike ([`uniform`]).
    uniformity: uniform::AddressUniformity,
}

/// What lanes are known to satisfy: each [`Bool`] where its flag is set,
/// its negation where not.
type Known = Vec<(Bool, bool)>;

/// One way back from a VGPR value in [`Tracer::scalar_copies`]'s walk:
/// what holds in the lanes that come by it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
struct LanePath {
    /// The sides of the branches it came through, and what flags hold
    /// there ([`Flag`]): small conditions.
    sides: Known,
    /// What lanes know entering the blocks it came through.
    reaches: Known,
    /// Whether the conditions of the lanes that access the descriptor hold
    /// on it too: not past a back edge.
    access: bool,
    /// Flags those conditions test whose value on the path is not known yet.
    flags: Vec<Flag>,
}

/// A compare of a VGPR with a constant whose lane bit the conditions of
/// the access test (`v_cmp_eq_u32 vcc, 1, v8` with `v8` a flag the paths
/// set): where the walk reaches a constant value of the VGPR, it knows the
/// bit. The lanes' paths do not follow VGPRs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Flag {
    /// The lane's bit of the compare's result ([`Atom::Lane`]).
    bit: Bool,
    /// The VGPR as the path has it so far (renamed through joins).
    value: Value,
    op: Cmp,
    signed: bool,
    constant: u32,
    /// The VGPR is the compare's first operand.
    flag_first: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Cmp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl Flag {
    /// The compare's result where the VGPR holds `x`.
    fn holds(&self, x: u32) -> bool {
        let (a, b) = if self.flag_first {
            (x, self.constant)
        } else {
            (self.constant, x)
        };
        let order = if self.signed {
            (a as i32).cmp(&(b as i32))
        } else {
            a.cmp(&b)
        };
        match self.op {
            Cmp::Eq => order.is_eq(),
            Cmp::Ne => order.is_ne(),
            Cmp::Lt => order.is_lt(),
            Cmp::Le => order.is_le(),
            Cmp::Gt => order.is_gt(),
            Cmp::Ge => order.is_ge(),
        }
    }
}

const MAX_ROUNDS: usize = 64;
const MAX_JOINS: usize = 8;
/// How many steps (a VGPR value on one path) [`Tracer::scalar_copies`] takes.
const MAX_COPIES: usize = 1024;

impl<'a> Tracer<'a> {
    fn new(
        p: &'a Program,
        layout: &'a [RootParam],
        stage: HwStage,
        lanes: Option<&'a Structured>,
    ) -> Result<Self, ShaderError> {
        let all = || {
            p.inputs
                .iter()
                .map(|i| i.1)
                .chain(p.blocks.iter().flat_map(|b| {
                    b.phis
                        .iter()
                        .map(|phi| phi.value)
                        .chain(b.insts.iter().flat_map(|i| i.writes.iter().map(|w| w.1)))
                }))
        };
        let n = all().map(|v| v.0 as usize + 1).max().unwrap_or(0);
        let mut defs = vec![None; n];
        let mut phis = vec![None; n];
        let mut inputs = vec![None; n];
        for &(reg, v) in &p.inputs {
            inputs[v.0 as usize] = Some(reg);
        }
        for (b, block) in p.blocks.iter().enumerate() {
            for (i, phi) in block.phis.iter().enumerate() {
                phis[phi.value.0 as usize] = Some((b, i));
            }
            for (i, inst) in block.insts.iter().enumerate() {
                for &(_, v) in &inst.writes {
                    defs[v.0 as usize] = Some((b, i));
                }
            }
        }
        let mut rpo = vec![usize::MAX; p.blocks.len()];
        for (n, b) in crate::ir::reverse_postorder(&p.blocks)
            .into_iter()
            .enumerate()
        {
            rpo[b] = n;
        }
        let mut t = Tracer {
            p,
            layout,
            stage,
            defs,
            phis,
            inputs,
            origin: Vec::new(),
            spill: Default::default(),
            lanes,
            rpo,
            uniformity: uniform::AddressUniformity::new(n),
        };
        t.origin = t.propagate_copies(n)?;
        Ok(t)
    }

    /// Optimistic copy propagation: a move's result is its source's
    /// origin, a phi's the one origin its arguments agree on.
    fn propagate_copies(&self, n: usize) -> Result<Vec<Term>, ShaderError> {
        let mut o: Vec<Option<Term>> = vec![None; n];
        for &(_, v) in &self.p.inputs {
            o[v.0 as usize] = Some(Term::Value(v));
        }
        let order = crate::ir::reverse_postorder(&self.p.blocks);
        for _ in 0..MAX_ROUNDS {
            let mut changed = false;
            for &b in &order {
                let block = &self.p.blocks[b];
                for phi in &block.phis {
                    let own = Term::Value(phi.value);
                    let mut args = phi
                        .args
                        .iter()
                        .filter_map(|a| o[a.0 as usize])
                        .filter(|&t| t != own);
                    let new = args
                        .next()
                        .map(|first| if args.all(|t| t == first) { first } else { own });
                    if new.is_some() && new != o[phi.value.0 as usize] {
                        o[phi.value.0 as usize] = new;
                        changed = true;
                    }
                }
                for inst in &block.insts {
                    for (k, &(_, v)) in inst.writes.iter().enumerate() {
                        let new = if let Some(binding) = self.spill.bindings.get(&v) {
                            Some(Term::Value(self.spill.first[binding]))
                        } else {
                            match copy_source(inst, k).or_else(|| self.restored(inst)) {
                                Some(Moved::Const(c)) => Some(Term::Const(c)),
                                Some(Moved::Value(s)) => o[s.0 as usize],
                                None => Some(Term::Value(v)),
                            }
                        };
                        if new.is_some() && new != o[v.0 as usize] {
                            o[v.0 as usize] = new;
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                return Ok(o
                    .into_iter()
                    .enumerate()
                    .map(|(v, t)| t.unwrap_or(Term::Value(Value(v as u32))))
                    .collect());
            }
        }
        Err(ShaderError::Unsupported {
            offset: 0,
            what: "copies that do not settle",
        })
    }

    fn inst(&self, (b, i): Site) -> &'a Inst {
        &self.p.blocks[b].insts[i]
    }

    fn def(&self, v: Value) -> Option<&'a Inst> {
        self.defs[v.0 as usize].map(|s| self.inst(s))
    }

    fn origin(&self, v: Value) -> Term {
        self.origin[v.0 as usize]
    }

    /// The origin of a one-dword operand as `inst` reads it.
    fn term(&self, inst: &Inst, op: &Operand) -> Option<Term> {
        match *op {
            Operand::Scalar { code: NULL, .. } => Some(Term::Const(0)),
            Operand::Scalar { code, .. } => Some(self.origin(inst.read(Reg::S(code))?)),
            Operand::Const(c) => inline(c).map(Term::Const),
            Operand::Literal(x) => Some(Term::Const(x)),
            _ => None,
        }
    }

    /// Layout index and dword of the root parameter `reg` holds on entry.
    fn register_param(&self, reg: Reg) -> Option<(usize, u32)> {
        let Reg::S(code) = reg else { return None };
        if self.spill.active {
            return None;
        }
        self.layout.iter().enumerate().find_map(|(i, p)| {
            let first = self.stage.user_sgpr(p.user_data);
            let dword = u32::from(code).checked_sub(first.into())?;
            (dword < p.sgprs).then_some((i, dword))
        })
    }

    /// The SOP2 operands of `v` if an `s_add` with one of `names` wrote it.
    fn sop2(&self, v: Value, names: &[&str]) -> Option<(&'a Inst, [Term; 2])> {
        let inst = self.def(v)?;
        if !names.contains(&inst.name()) || inst.writes.first()?.1 != v {
            return None;
        }
        let Fields::Sop { src, .. } = &inst.fields else {
            return None;
        };
        Some((
            inst,
            [
                self.term(inst, src.first()?)?,
                self.term(inst, src.get(1)?)?,
            ],
        ))
    }

    fn sum(&self, t: Term) -> Sum {
        let v = match t {
            Term::Const(c) => {
                return Sum {
                    base: None,
                    offset: c,
                    dynamic: None,
                };
            }
            Term::Value(v) => v,
        };
        let dynamic = Sum {
            base: None,
            offset: 0,
            dynamic: Some(v),
        };
        if self.inputs[v.0 as usize].is_some() || self.spill.bindings.contains_key(&v) {
            return match self.param_value(v) {
                Some((_, 0)) => Sum {
                    base: Some(v),
                    offset: 0,
                    dynamic: None,
                },
                _ => dynamic,
            };
        }
        if self.getpc(v).is_some() {
            return Sum {
                base: Some(v),
                offset: 0,
                dynamic: None,
            };
        }
        if let Some(src) = self.uniform_first_lane(v) {
            return self.sum(self.vorigin(src));
        }
        if let Some(s) = self.first_lane_folded(v) {
            return s;
        }
        // A table in the code that paths reach each from an `s_getpc_b64`
        // and join before one access, or whose address is read from the
        // first lane of a value the lanes may hold differently: the join, or
        // the read, is the computed term, and counts the table's base itself
        // ([`Self::counts_code`]).
        if let Some(pc) = self.code_join(v, CODE_DEPTH).or_else(|| self.code_lane(v)) {
            return Sum {
                base: Some(pc),
                offset: 0,
                dynamic: Some(v),
            };
        }
        let Some([a, b]) = self
            .sop2(v, &["s_add_u32", "s_add_i32"])
            .map(|(_, t)| t)
            .or_else(|| self.vadd(v))
        else {
            return dynamic;
        };
        let (x, y) = (self.sum(a), self.sum(b));
        if x.base.is_some() && y.base.is_some() || x.dynamic.is_some() && y.dynamic.is_some() {
            return dynamic;
        }
        Sum {
            base: x.base.or(y.base),
            offset: x.offset.wrapping_add(y.offset),
            dynamic: x.dynamic.or(y.dynamic),
        }
    }

    /// Ported from fxceless's VIctory fork (commit 578d459, `code_join`),
    /// with its paths checked as sums. A join `v` whose every incoming
    /// value is an address in the code (an `s_getpc_b64` low dword plus
    /// terms of no other base, [`Self::code_base`]): the `s_getpc_b64` of
    /// the first. Each path may take a table of its own: BTP 6's fog
    /// reconstruction reads its sample offsets from one of two tables, by
    /// the sample's number ("refused, the lit frame was multiplied by a fog
    /// transmittance nothing had written", in the fork). The emitter gives
    /// each `s_getpc_b64` its offset in the code as its value, so the join
    /// is the byte in the code whichever table a path took. A VGPR join
    /// holds each lane's own path's address on the console too; an SGPR
    /// join, the scalar of the path the lane took, as structurisation
    /// assumes for every scalar.
    fn code_join(&self, v: Value, depth: u32) -> Option<Value> {
        let (b, i) = self.phis[v.0 as usize]?;
        if depth == 0 {
            return None;
        }
        let mut pc = None;
        for &a in &self.p.blocks[b].phis[i].args {
            match self.vorigin(a) {
                Term::Value(x) if x == v => {}
                t => match self.code_base(t, depth - 1) {
                    CodeBase::Code(x) => pc = pc.or(Some(x)),
                    CodeBase::None | CodeBase::Other => return None,
                },
            }
        }
        pc
    }

    /// The `v_readfirstlane_b32` result `v`, read from an address in the
    /// code that the lanes may hold differently ([`Self::code_base`]): the
    /// `s_getpc_b64` it adds to. The fork (same commit) takes each lane's
    /// own address there, which the console reads only for its first lane;
    /// here the read itself is the computed term, as the emitter computes
    /// it, every lane with the first lane's address. Where a guard narrows
    /// the read in the per-lane form, the host's first lane may be one the
    /// console's EXEC leaves out: not taken.
    fn code_lane(&self, v: Value) -> Option<Value> {
        let src = self.first_lane_source(v)?;
        let (b, i) = self.defs[v.0 as usize]?;
        if self.lanes?.guards[b][i].is_some() {
            return None;
        }
        match self.code_base(self.vorigin(src), CODE_DEPTH) {
            CodeBase::Code(pc) => Some(pc),
            CodeBase::None | CodeBase::Other => None,
        }
    }

    /// Whether the computed term `d` counts a table's base itself: a join
    /// or a first-lane read of addresses in the code ([`Self::code_join`],
    /// [`Self::code_lane`]), which [`Self::sum`] gives as the computed term
    /// of a sum with that table's base.
    fn counts_code(&self, d: Value) -> bool {
        self.code_join(d, CODE_DEPTH).is_some() || self.code_lane(d).is_some()
    }

    /// What the additions computing `t` hold as their base, `depth`
    /// additions deep, as [`Self::sum`] counts bases: a root parameter's
    /// first dword, or an `s_getpc_b64` low dword, also through a
    /// first-lane read and through a join of addresses in the code. Two
    /// bases, or one past `depth`, are [`CodeBase::Other`].
    fn code_base(&self, t: Term, depth: u32) -> CodeBase {
        let Term::Value(x) = t else {
            return CodeBase::None;
        };
        if self.getpc(x).is_some() {
            return CodeBase::Code(x);
        }
        if self.inputs[x.0 as usize].is_some() || self.spill.bindings.contains_key(&x) {
            return match self.param_value(x) {
                Some((_, 0)) => CodeBase::Other,
                _ => CodeBase::None,
            };
        }
        if depth == 0 {
            return CodeBase::Other;
        }
        if let Some(src) = self.first_lane_source(x) {
            return self.code_base(self.vorigin(src), depth - 1);
        }
        if self.phis[x.0 as usize].is_some() {
            return match self.code_join(x, depth - 1) {
                Some(pc) => CodeBase::Code(pc),
                None => CodeBase::None,
            };
        }
        let Some([a, b]) = self
            .sop2(x, &["s_add_u32", "s_add_i32"])
            .map(|(_, t)| t)
            .or_else(|| self.vadd(x))
        else {
            return CodeBase::None;
        };
        match (self.code_base(a, depth - 1), self.code_base(b, depth - 1)) {
            (CodeBase::Code(pc), CodeBase::None) | (CodeBase::None, CodeBase::Code(pc)) => {
                CodeBase::Code(pc)
            }
            (CodeBase::None, CodeBase::None) => CodeBase::None,
            _ => CodeBase::Other,
        }
    }

    /// `t` has the shape of a table address's high half in the code: an
    /// `s_getpc_b64` high dword, an addition with carry to one
    /// (`s_addc_u32`, `v_add_co_ci_u32`), a join of such, or a first-lane
    /// read of one; `depth` steps deep. The paths of a join may carry into
    /// different tables' high halves. Only the shape counts: no lowering
    /// uses a code address's high half.
    fn code_high(&self, t: Term, depth: u32) -> bool {
        let Term::Value(x) = t else {
            return false;
        };
        if self
            .def(x)
            .is_some_and(|i| i.name() == "s_getpc_b64" && i.writes.get(1).is_some_and(|w| w.1 == x))
        {
            return true;
        }
        if depth == 0 {
            return false;
        }
        if let Some((b, i)) = self.phis[x.0 as usize] {
            return self.p.blocks[b].phis[i].args.iter().all(|&a| {
                let t = self.vorigin(a);
                t == Term::Value(x) || self.code_high(t, depth - 1)
            });
        }
        if let Some(src) = self.first_lane_source(x) {
            return self.code_high(self.vorigin(src), depth - 1);
        }
        if let Some((_, [a, b])) = self.sop2(x, &["s_addc_u32"]) {
            return self.code_high(a, depth - 1) || self.code_high(b, depth - 1);
        }
        let Some(inst) = self.def(x) else {
            return false;
        };
        let Fields::Valu(f) = &inst.fields else {
            return false;
        };
        inst.name() == "v_add_co_ci_u32"
            && f.src[..2.min(f.src.len())].iter().any(|o| {
                self.vterm(inst, o)
                    .is_some_and(|t| self.code_high(t, depth - 1))
            })
    }

    /// The `s_getpc_b64` writing `lo` as its low dword.
    fn getpc(&self, lo: Value) -> Option<&'a Inst> {
        let inst = self.def(lo)?;
        (inst.name() == "s_getpc_b64" && inst.writes.first().is_some_and(|w| w.1 == lo))
            .then_some(inst)
    }

    /// The operands of a VALU addition (`v_add_co_u32`, `v_add_nc_u32`)
    /// writing `v`, through VGPR copies of scalars.
    fn vadd(&self, v: Value) -> Option<[Term; 2]> {
        let inst = self.def(v)?;
        let Fields::Valu(f) = &inst.fields else {
            return None;
        };
        if !matches!(inst.name(), "v_add_co_u32" | "v_add_nc_u32")
            || f.dst.is_none_or(|d| !matches!(d, Operand::Vector { .. }))
            || inst.write(Reg::V(match f.dst? {
                Operand::Vector { reg, .. } => reg,
                _ => return None,
            })) != Some(v)
            || f.abs != 0
            || f.neg != 0
            || f.sdwa.is_some()
            || f.dpp.is_some()
        {
            return None;
        }
        Some([
            self.vterm(inst, f.src.first()?)?,
            self.vterm(inst, f.src.get(1)?)?,
        ])
    }

    /// The origin of a VALU operand.
    fn vterm(&self, inst: &Inst, op: &Operand) -> Option<Term> {
        match *op {
            Operand::Vector { reg, .. } => Some(self.vorigin(inst.read(Reg::V(reg))?)),
            op => self.term(inst, &op),
        }
    }

    /// A VGPR value's origin through `v_mov_b32` copies.
    fn vorigin(&self, v: Value) -> Term {
        if let Some(inst) = self.def(v)
            && inst.name() == "v_mov_b32"
            && let Fields::Valu(f) = &inst.fields
            && f.abs == 0
            && f.neg == 0
            && f.sdwa.is_none()
            && f.dpp.is_none()
            && let Some(t) = f.src.first().and_then(|s| self.vterm(inst, s))
        {
            return t;
        }
        self.origin(v)
    }

    /// A global memory address: an SGPR pair plus a VGPR offset, or a VGPR
    /// pair.
    fn flat_address(
        &self,
        site: Site,
        addr: Option<&Operand>,
        saddr: Option<&Operand>,
        offset: i16,
    ) -> Result<Address, &'static str> {
        let Some(&Operand::Vector { reg, .. }) = addr else {
            return Err("a global address not in VGPRs");
        };
        let inst = self.inst(site);
        let off = offset as i32 as u32;
        match saddr {
            Some(&Operand::Scalar { code, .. }) if code != NULL => {
                let word = |k| Some(self.origin(inst.read(Reg::S(code + k))?));
                let (Some(lo), Some(hi)) = (word(0), word(1)) else {
                    return Err("an address not in SGPRs");
                };
                let v = inst.read(Reg::V(reg)).ok_or("a global address")?;
                let mut extra = self.sum(self.vorigin(v));
                if extra.base.is_some() {
                    return Err("an offset from a root parameter");
                }
                extra.offset = extra.offset.wrapping_add(off);
                let mut known = self.site_known(site);
                let hi = self.high_on_edge(hi, &mut known);
                self.address(lo, hi, extra)
            }
            _ => {
                let word = |k| inst.read(Reg::V(reg + k)).map(|v| self.vorigin(v));
                let (Some(lo), Some(hi)) = (word(0), word(1)) else {
                    return Err("a global address");
                };
                self.address(
                    lo,
                    hi,
                    Sum {
                        base: None,
                        offset: off,
                        dynamic: None,
                    },
                )
            }
        }
    }

    /// The address `lo` and `hi` hold, plus `extra` (an instruction's
    /// immediate and offset register).
    fn address(&self, lo: Term, hi: Term, extra: Sum) -> Result<Address, &'static str> {
        let (lo, hi) = (self.unlaned(lo), self.unlaned_high(hi));
        let s = self.sum(lo);
        if s.dynamic.is_some() && extra.dynamic.is_some() {
            return Err("an address with two computed terms");
        }
        let base = s.base.ok_or_else(|| self.why_no_base(s.dynamic))?;
        if let Some(pc) = self.getpc(base) {
            // A table in the code: the base is the next instruction's offset,
            // unless the computed term counts it already (a join or a
            // first-lane read of addresses in the code, whose high half may
            // be joined or read from the first lane too).
            let counted = s.dynamic.is_some_and(|d| self.counts_code(d));
            let high = pc.writes.get(1).map(|w| w.1);
            if !self.high_half(hi, high)
                && !high.is_some_and(|h| self.carried(hi, h))
                && !(counted && self.code_high(hi, CODE_DEPTH))
            {
                return Err("an address with an untracked high half");
            }
            let first = if counted {
                0
            } else {
                pc.machine.offset as u32 + 4
            };
            return Ok(Address {
                param: CODE,
                offset: first.wrapping_add(s.offset).wrapping_add(extra.offset),
                dynamic: s.dynamic.or(extra.dynamic),
            });
        }
        let (param, _) = self
            .param_value(base)
            .ok_or("an address not from a root parameter")?;
        let high = match self.layout[param].root {
            RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap => None,
            RootType::Descriptor => Some(
                self.param_high(param)
                    .ok_or("a root descriptor without its high half")?,
            ),
            _ => return Err("an address from a root parameter that is no address"),
        };
        // shortcut: with a computed term, a high half that adds anything to the
        // base's is taken as the carry of that term (offsets below 4 GB).
        let dynamic = s.dynamic.or(extra.dynamic);
        if !self.high_half(hi, high)
            && !(dynamic.is_some() && high.is_some_and(|h| self.carried(hi, h)))
        {
            return Err("an address with an untracked high half");
        }
        Ok(Address {
            param,
            offset: s.offset.wrapping_add(extra.offset),
            dynamic: s.dynamic.or(extra.dynamic),
        })
    }

    /// `hi` is `high` (0 for a table), possibly plus a carry.
    fn high_half(&self, hi: Term, high: Option<Value>) -> bool {
        let want = high.map_or(Term::Const(0), Term::Value);
        hi == want
            || matches!(hi, Term::Value(v) if self.sop2(v, &["s_addc_u32"]).is_some_and(
                |(_, [a, b])| a == want && b == Term::Const(0) || b == want && a == Term::Const(0)
            ))
    }

    /// `hi` adds something (with carry) to `high`: `s_addc_u32` or
    /// `v_add_co_ci_u32` with `high` as an operand.
    fn carried(&self, hi: Term, high: Value) -> bool {
        let Term::Value(v) = self.unlaned_high(hi) else {
            return false;
        };
        if v == high {
            return true;
        }
        if let Some((_, [a, b])) = self.sop2(v, &["s_addc_u32"]) {
            return self.carried(a, high) || self.carried(b, high);
        }
        let Some(inst) = self.def(v) else {
            return false;
        };
        let Fields::Valu(f) = &inst.fields else {
            return false;
        };
        inst.name() == "v_add_co_ci_u32"
            && f.src[..2.min(f.src.len())]
                .iter()
                .any(|o| self.vterm(inst, o).is_some_and(|t| self.carried(t, high)))
    }

    fn why_no_base(&self, dynamic: Option<Value>) -> &'static str {
        let Some(v) = dynamic else {
            return "a constant address";
        };
        if self.inputs[v.0 as usize].is_some() {
            return "an address in an SGPR that is no root parameter";
        }
        if self
            .leaves(v)
            .any(|l| self.def(l).is_some_and(|i| i.name() == "s_getpc_b64"))
        {
            return "an address relative to the program counter";
        }
        match self.def(v).map(|i| i.name()) {
            Some("v_readfirstlane_b32") => {
                "an address read from the first lane of a per-lane value"
            }
            Some(n) if n.starts_with("s_load_") => "an address loaded from memory",
            None if self.spill.bindings.contains_key(&v) => "an address held in spilled root data",
            None => "an address from a phi",
            _ => "an address not from a root parameter",
        }
    }

    /// The terms of the additions that compute `v`.
    fn leaves(&self, v: Value) -> Box<dyn Iterator<Item = Value> + '_> {
        match self.sop2(v, &["s_add_u32", "s_add_i32"]) {
            Some((_, terms)) => Box::new(terms.into_iter().flat_map(move |t| match t {
                Term::Value(x) => self.leaves(x),
                Term::Const(_) => Box::new(std::iter::empty()),
            })),
            None => Box::new(std::iter::once(v)),
        }
    }

    /// Active-lane conditions for an access at `at`: block reachability plus
    /// the instruction's EXEC guard when present (`None` for scalar accesses).
    fn site_known(&self, at: Site) -> Known {
        let mut known = Vec::new();
        if let Some(l) = self.lanes {
            known.push((l.reach[at.0], true));
            if let Some(guard) = l
                .guards
                .get(at.0)
                .and_then(|g| g.get(at.1))
                .copied()
                .flatten()
            {
                known.push((guard, true));
            }
        }
        known
    }

    /// Unwraps SGPR high-word phis whose wave path is forced to a single
    /// incoming edge under `known` (up to [`MAX_JOINS`] nested joins).
    fn high_on_edge(&self, mut hi: Term, known: &mut Known) -> Term {
        let n = known.len();
        for _ in 0..MAX_JOINS {
            let Term::Value(v) = hi else { break };
            let Some((block, index)) = self.phis[v.0 as usize] else {
                break;
            };
            let Some((edge, conditions)) = self.only_edge(block, known) else {
                break;
            };
            known.extend(conditions);
            hi = self.origin(self.p.blocks[block].phis[index].args[edge]);
        }
        known.truncate(n);
        hi
    }

    fn smem_address(
        &self,
        site: Site,
        base: &Operand,
        offset: i32,
        soffset: &Operand,
    ) -> Result<Address, &'static str> {
        let Operand::Scalar { code, .. } = *base else {
            return Err("an address not in SGPRs");
        };
        let inst = self.inst(site);
        let word = |k| Some(self.origin(inst.read(Reg::S(code + k))?));
        let (Some(lo), Some(hi)) = (word(0), word(1)) else {
            return Err("an address not in SGPRs");
        };
        let mut extra = self.sum(
            self.term(inst, soffset)
                .ok_or("an unknown offset operand")?,
        );
        if extra.base.is_some() {
            return Err("an offset from a root parameter");
        }
        extra.offset = extra.offset.wrapping_add(offset as u32);
        // An arm may set the upper SGPR only for waves that reach this load.
        // SMEM instructions are not EXEC-masked (`guards` is `None`), so
        // `site_known` uses block reachability alone.
        let mut known = self.site_known(site);
        let hi = self.high_on_edge(hi, &mut known);
        self.address(lo, hi, extra)
    }

    /// The source of the descriptor operand `op` of `inst`, at `at`.
    fn descriptor(&self, inst: &Inst, op: &Operand, at: Site) -> Result<Source, &'static str> {
        let Operand::Scalar { code, dwords } = *op else {
            return Err("a descriptor not in SGPRs");
        };
        let words = (0..dwords)
            .map(|k| inst.read(Reg::S(code + k)).map(|v| self.origin(v)))
            .collect::<Option<Vec<_>>>()
            .ok_or("a descriptor not in SGPRs")?;
        // Only the lanes that reach and run the access read the descriptor.
        let mut known = self.site_known(at);
        self.classify(&words, &mut Vec::new(), &mut known)
    }

    /// The source of a descriptor's words; `joins` are the phi blocks
    /// being resolved, `known` holds in every lane that reads them.
    fn classify(
        &self,
        words: &[Term],
        joins: &mut Vec<usize>,
        known: &mut Known,
    ) -> Result<Source, &'static str> {
        if let Some(source) = self.phi(words, joins, known)? {
            return Ok(source);
        }
        if let Some(source) = self.select(words, joins, known)? {
            return Ok(source);
        }
        let read = |w: &Term| match *w {
            Term::Value(v) => self.first_lane_source(v),
            Term::Const(_) => None,
        };
        if let Some(sources) = words.iter().map(read).collect::<Option<Vec<_>>>() {
            // Read from the first active lane, as a waterfall loop does: the
            // dwords each lane holds. Where those cannot be traced per path,
            // one descriptor the code keeps per lane ([`Self::lane_copies`]).
            let lanes: Vec<Term> = sources.iter().copied().map(Term::Value).collect();
            return self
                .per_lane(&lanes, joins, known)
                .or_else(|why| self.lane_copies(&sources, known).ok_or(why));
        }
        if words.iter().any(|w| read(w).is_some()) {
            return Err("a descriptor partly read from the first lane");
        }
        if let Some(consts) = words
            .iter()
            .map(|w| match *w {
                Term::Const(c) => Some(c),
                Term::Value(_) => None,
            })
            .collect::<Option<Vec<_>>>()
        {
            return Ok(Source::Constant(consts));
        }
        if let Some(source) = self.loaded(words) {
            return Ok(source);
        }
        if let [lo, hi, Term::Const(w2), Term::Const(w3)] = *words {
            let (mut hi, mut bits) = (hi, 0);
            while let Term::Value(v) = hi
                && let Some((x, more)) = self.set_bits(v)
            {
                (hi, bits) = (x, bits | more);
            }
            let hi = self.high_on_edge(hi, known);
            let address = self.address(
                lo,
                hi,
                Sum {
                    base: None,
                    offset: 0,
                    dynamic: None,
                },
            )?;
            return Ok(Source::Built {
                address,
                bits,
                words: [w2, w3],
            });
        }
        let loaded = |w: &Term| {
            matches!(*w, Term::Value(v)
                if self.def(v).is_some_and(|i| i.name().starts_with("s_load_")))
        };
        Err(if words.iter().any(loaded) {
            "a descriptor changed after loading"
        } else {
            "a descriptor neither loaded nor built"
        })
    }

    /// Words that phis of one block join, traced per predecessor (words
    /// defined before the join hold on every edge).
    fn phi(
        &self,
        words: &[Term],
        joins: &mut Vec<usize>,
        known: &mut Known,
    ) -> Result<Option<Source>, &'static str> {
        let phi_of = |w: &Term| match *w {
            Term::Value(v) => self.phis[v.0 as usize],
            Term::Const(_) => None,
        };
        let Some((block, _)) = words.iter().find_map(phi_of) else {
            return Ok(None);
        };
        if words.iter().filter_map(phi_of).any(|(b, _)| b != block) {
            return Err("a descriptor joined in several blocks");
        }
        if joins.contains(&block) {
            return Err("a descriptor that changes in a loop");
        }
        // Each level multiplies the sources by the predecessor count.
        if joins.len() >= MAX_JOINS {
            return Err("a descriptor joined too often");
        }
        joins.push(block);
        let phis = &self.p.blocks[block].phis;
        let edge = |j: usize| -> Vec<Term> {
            words
                .iter()
                .map(|w| match phi_of(w) {
                    Some((_, i)) => self.origin(phis[i].args[j]),
                    None => *w,
                })
                .collect()
        };
        let args = (0..self.p.blocks[block].preds.len())
            .map(|j| self.classify(&edge(j), joins, known))
            .collect::<Result<_, _>>();
        let source = match args {
            Ok(args) => Ok(Source::Phi { block, args }),
            // An edge holds no descriptor. With one edge left that the lanes
            // reading the descriptor can have come by, it is what the
            // registers hold on that one, if lowering can name it where it
            // is read: a wave that took another edge has no such lane left
            // (an arm under the copy of EXEC the program started with,
            // `s_and_saveexec` of it, where the console compiler set a
            // sampler's constants; what the registers held before the arm
            // is what the skipping edge joins in).
            Err(e) => match self.only_edge(block, known) {
                Some((j, conditions)) => {
                    let n = known.len();
                    known.extend(conditions);
                    let one = self.classify(&edge(j), joins, known);
                    known.truncate(n);
                    match one {
                        Ok(source) if self.anywhere(&source) => Ok(source),
                        _ => Err(e),
                    }
                }
                None => Err(e),
            },
        };
        joins.pop();
        source.map(Some)
    }

    /// The one forward edge into the join `block`, by its index in the
    /// block's predecessors, that lanes satisfying `known` can have taken,
    /// with what holds in the lanes that take it; `None` unless the lanes'
    /// paths ([`Structured`]) rule out every other edge, and at loop
    /// headers.
    ///
    /// Ported from fxceless's VIctory fork (commit 578d459) by fxceless:
    /// `only_edge`, `anywhere` and their use when an edge of a join holds
    /// no descriptor ([`Self::phi`]). Here the lanes that take the edge
    /// are known while its dwords are traced, and a load's base must come
    /// from the entry ([`Self::anywhere`]).
    ///
    /// The descriptor is in SGPRs, which hold what the wave's path left,
    /// not the lane's: a wave runs an arm if any of its lanes does. So the
    /// edge must be one the wave takes whenever such a lane does: a jump,
    /// or a branch the whole wave takes alike. Not one side of a per-lane
    /// branch (an `s_cbranch_execz` over an arm that other lanes of the
    /// wave run, rewriting the registers before these lanes read them), nor
    /// the end of a path that such a branch leaves on the way there
    /// ([`Self::wave_comes_by`]: the same skip to a block that jumps to the
    /// join, while the arm jumps there too).
    fn only_edge(&self, block: usize, known: &mut Known) -> Option<(usize, Known)> {
        let lanes = self.lanes?;
        let preds = &self.p.blocks[block].preds;
        if preds.len() < 2 || preds.iter().any(|&q| self.rpo[q] >= self.rpo[block]) {
            return None;
        }
        let mut only = None;
        for (j, &q) in preds.iter().enumerate() {
            let mut edge = vec![(lanes.reach[q], true)];
            let mut per_lane = false;
            if let LaneExit::Branch {
                when,
                taken,
                not_taken,
            } = lanes.exits[q]
                && taken != not_taken
            {
                edge.push((when, taken == block));
                per_lane = self.per_lane_branch(when);
            }
            let n = known.len();
            known.extend(edge.iter().copied());
            let taken = lanes.bools.meet(known);
            known.truncate(n);
            if taken && only.replace((j, edge, per_lane)).is_some() {
                return None;
            }
        }
        let (j, edge, per_lane) = only?;
        let n = known.len();
        known.extend(edge.iter().copied());
        let wave = !per_lane && self.wave_comes_by(block, preds[j], known);
        known.truncate(n);
        wave.then_some((j, edge))
    }

    /// Whether lanes may take a branch on `when` differently within a wave:
    /// it tests something other than SCC values.
    fn per_lane_branch(&self, when: Bool) -> bool {
        self.lanes.is_some_and(|lanes| {
            lanes
                .bools
                .support(when)
                .iter()
                .any(|a| !matches!(a, Atom::Uniform(_)))
        })
    }

    /// Whether a wave whose lanes satisfying `known` come into `block` from
    /// its predecessor `q` came by that edge itself, as its SGPRs did. Its
    /// path and theirs part only at per-lane branches on the way, from
    /// `block`'s immediate dominator on: there a lane with its bit of the
    /// tested mask set takes the wave along (the side an `s_cbranch_execz`
    /// or `_vccz` does not take, the one an `s_cbranch_execnz` or `_vccnz`
    /// takes), and a lane with its bit clear may see the wave take the
    /// other side. So wherever these lanes may take the side a lane of the
    /// wave can leave them on, every path from the other side must come
    /// into `block` from `q` too. For a branch on SCC either side may be
    /// the wave's. Paths here take back edges too: a wave sent round a loop
    /// by other lanes can come into `block` from elsewhere a pass later.
    fn wave_comes_by(&self, block: usize, q: usize, known: &Known) -> bool {
        let Some(lanes) = self.lanes else {
            return false;
        };
        let blocks = &self.p.blocks;
        let order = crate::ir::reverse_postorder(blocks);
        let idom = crate::ir::idoms(blocks, &order, &self.rpo);
        let Some(&top) = idom.get(block).filter(|&&d| d != usize::MAX) else {
            return false;
        };
        // The blocks that lead to `to`, not through `block`.
        let leading = |to: usize| {
            let mut seen = vec![false; blocks.len()];
            let mut stack = vec![to];
            while let Some(x) = stack.pop() {
                if x == block || std::mem::replace(&mut seen[x], true) {
                    continue;
                }
                stack.extend(&blocks[x].preds);
            }
            seen
        };
        let to_q = leading(q);
        let mut to_block = vec![false; blocks.len()];
        for &p in &blocks[block].preds {
            for (x, on) in leading(p).into_iter().enumerate() {
                to_block[x] |= on;
            }
        }
        // Whether a path from `s` comes into `block` from another block.
        let elsewhere = |s: usize| {
            let mut seen = vec![false; blocks.len()];
            let mut stack = vec![s];
            while let Some(y) = stack.pop() {
                if y == q || !to_block[y] || std::mem::replace(&mut seen[y], true) {
                    continue;
                }
                let next = blocks[y].exit.successors();
                if next.contains(&block) {
                    return true;
                }
                stack.extend(next);
            }
            false
        };
        // The blocks on the paths from the dominator to `block`.
        let mut region = vec![false; blocks.len()];
        let mut stack = vec![top];
        while let Some(x) = stack.pop() {
            if x == block || !to_block[x] || std::mem::replace(&mut region[x], true) {
                continue;
            }
            stack.extend(blocks[x].exit.successors());
        }
        for x in (0..blocks.len()).filter(|&x| region[x]) {
            let LaneExit::Branch {
                when,
                taken,
                not_taken,
            } = lanes.exits[x]
            else {
                continue;
            };
            if taken == not_taken || !self.per_lane_branch(when) {
                continue;
            }
            // Whether these lanes may leave `x` towards `q` over one side.
            let lanes_take = |side: bool| {
                let s = if side { taken } else { not_taken };
                let leads = if s == block { x == q } else { to_q[s] };
                let mut all = known.clone();
                all.extend([(lanes.reach[x], true), (when, side)]);
                leads && lanes.bools.meet(&all)
            };
            // The side a lane with its bit set takes the wave to.
            let along = match blocks[x].insts.last().map(|i| i.name()) {
                Some("s_cbranch_execz" | "s_cbranch_vccz") => Some(false),
                Some("s_cbranch_execnz" | "s_cbranch_vccnz") => Some(true),
                _ => None,
            };
            for wave in [true, false] {
                if along.is_some_and(|a| a != wave) || !lanes_take(!wave) {
                    continue;
                }
                let s = if wave { taken } else { not_taken };
                let apart = if s == block { x != q } else { elsewhere(s) };
                if apart {
                    return false;
                }
            }
        }
        true
    }

    /// Whether lowering can make a handle for `source` in any block: its
    /// heap index comes from root parameters and constants alone. A table
    /// offset the code computes exists only where the load is; so does a
    /// load's base unless it is an input or a constant through moves, which
    /// lowering gives the input's own value (computed in the entry block).
    fn anywhere(&self, source: &Source) -> bool {
        match source {
            Source::Constant(_) => true,
            Source::Built { address, .. } => address.dynamic.is_none(),
            Source::Loaded { load, .. } => {
                let inst = self.inst(*load);
                let Fields::Smem {
                    base,
                    offset,
                    soffset,
                    ..
                } = &inst.fields
                else {
                    return false;
                };
                let entry = |op: &Operand| match *op {
                    Operand::Scalar { code: NULL, .. }
                    | Operand::Const(_)
                    | Operand::Literal(_) => true,
                    Operand::Scalar { code, .. } => {
                        inst.read(Reg::S(code)).is_some_and(|v| self.entry_value(v))
                    }
                    _ => false,
                };
                entry(base)
                    && entry(soffset)
                    && self
                        .smem_address(*load, base, *offset, soffset)
                        .is_ok_and(|a| a.dynamic.is_none())
            }
            _ => false,
        }
    }

    /// Whether `v` is an input or a constant, through scalar moves.
    fn entry_value(&self, mut v: Value) -> bool {
        for _ in 0..MAX_ROUNDS {
            if self.inputs[v.0 as usize].is_some() || self.spill.bindings.contains_key(&v) {
                return true;
            }
            let Some(inst) = self.def(v) else {
                return false;
            };
            if !matches!(inst.name(), "s_mov_b32" | "s_mov_b64" | "s_movk_i32") {
                return false;
            }
            let Some(k) = inst.writes.iter().position(|w| w.1 == v) else {
                return false;
            };
            match copy_source(inst, k) {
                Some(Moved::Value(s)) => v = s,
                Some(Moved::Const(_)) => return true,
                None => return false,
            }
        }
        false
    }

    /// Words that `s_cselect` (or `s_cmov`) picks by one SCC value, traced
    /// per outcome (other words are the same for both).
    fn select(
        &self,
        words: &[Term],
        joins: &mut Vec<usize>,
        known: &mut Known,
    ) -> Result<Option<Source>, &'static str> {
        let picks: Vec<_> = words
            .iter()
            .map(|w| match *w {
                Term::Value(v) => self.picked(v),
                Term::Const(_) => None,
            })
            .collect();
        let Some(scc) = picks.iter().flatten().map(|p| p.0).next() else {
            return Ok(None);
        };
        if picks.iter().flatten().any(|p| p.0 != scc) {
            return Err("a descriptor chosen by several conditions");
        }
        // Each level doubles the sources; selects count like joins (a
        // marker no block index reaches).
        if joins.len() >= MAX_JOINS {
            return Err("a descriptor joined too often");
        }
        joins.push(usize::MAX);
        let side = |set: bool| -> Vec<Term> {
            words
                .iter()
                .zip(&picks)
                .map(|(w, p)| match p {
                    Some((_, t, f)) => {
                        if set {
                            *t
                        } else {
                            *f
                        }
                    }
                    None => *w,
                })
                .collect()
        };
        let args = [true, false]
            .into_iter()
            .map(|set| self.classify(&side(set), joins, known))
            .collect::<Result<_, _>>();
        joins.pop();
        Ok(Some(Source::Select { scc, args: args? }))
    }

    /// For `v` written by `s_cselect_b32`/`_b64` or `s_cmov_b32`/`_b64`:
    /// the SCC it reads and the dword it is when SCC is set, then clear.
    fn picked(&self, v: Value) -> Option<(Value, Term, Term)> {
        let inst = self.def(v)?;
        let Fields::Sop { dst, src, .. } = &inst.fields else {
            return None;
        };
        let k = inst.writes.iter().position(|w| w.1 == v)? as u8;
        let wide = match inst.name() {
            "s_cselect_b32" | "s_cmov_b32" => false,
            "s_cselect_b64" | "s_cmov_b64" => true,
            _ => return None,
        };
        // A dword of a 32- or 64-bit operand: 64-bit inline integers are
        // sign-extended; 64-bit literals and float constants are not
        // followed.
        let dword = |op: &Operand| match *op {
            Operand::Scalar { code: NULL, .. } => Some(Term::Const(0)),
            Operand::Scalar { code, .. } => Some(self.origin(inst.read(Reg::S(code + k))?)),
            Operand::Const(c) if !wide => inline(c).map(Term::Const),
            Operand::Const(c @ 128..=208) => {
                let low = inline(c)?;
                Some(Term::Const(if k == 0 {
                    low
                } else {
                    ((low as i32) >> 31) as u32
                }))
            }
            Operand::Literal(x) if !wide => Some(Term::Const(x)),
            _ => None,
        };
        let set = dword(src.first()?)?;
        let clear = if inst.name().starts_with("s_cmov") {
            dword(dst.as_ref()?)?
        } else {
            dword(src.get(1)?)?
        };
        Some((inst.read(Reg::Scc)?, set, clear))
    }

    /// `v` as some other dword with constant bits set.
    fn set_bits(&self, v: Value) -> Option<(Term, u32)> {
        if let Some((_, [x, Term::Const(bits)] | [Term::Const(bits), x])) =
            self.sop2(v, &["s_or_b32"])
        {
            return Some((x, bits));
        }
        let inst = self.def(v)?;
        let Fields::Sop {
            dst: Some(Operand::Scalar { code, .. }),
            src,
            ..
        } = &inst.fields
        else {
            return None;
        };
        match (inst.name(), self.term(inst, src.first()?)?) {
            ("s_bitset1_b32", Term::Const(bit)) => {
                Some((self.origin(inst.read(Reg::S(*code))?), 1 << (bit & 31)))
            }
            _ => None,
        }
    }

    /// For a `v_readfirstlane_b32` result: the VGPR value it reads.
    fn first_lane_source(&self, v: Value) -> Option<Value> {
        let inst = self.def(v)?;
        let Fields::Valu(f) = &inst.fields else {
            return None;
        };
        if inst.name() != "v_readfirstlane_b32" || f.sdwa.is_some() || f.dpp.is_some() {
            return None;
        }
        let Some(&Operand::Vector { reg, .. }) = f.src.first() else {
            return None;
        };
        inst.read(Reg::V(reg))
    }

    /// The `v_readfirstlane_b32` results the descriptor operand `op` of
    /// `inst` is, when every dword is one.
    fn first_lane_words(&self, inst: &Inst, op: &Operand) -> Option<Vec<Value>> {
        let Operand::Scalar { code, dwords } = *op else {
            return None;
        };
        (0..dwords)
            .map(|k| match self.origin(inst.read(Reg::S(code + k))?) {
                Term::Value(v) => self.first_lane_source(v).map(|_| v),
                Term::Const(_) => None,
            })
            .collect()
    }

    /// `t` through VGPR copies (of VGPRs or of scalars) and copies.
    fn lane_term(&self, mut t: Term) -> Term {
        for _ in 0..MAX_JOINS * 2 {
            let Term::Value(v) = t else { break };
            let next = self.vorigin(v);
            if next == t {
                break;
            }
            t = next;
        }
        t
    }

    /// Whether `t` is a value each lane holds its own of (one in VGPRs).
    fn varies(&self, t: Term) -> bool {
        let Term::Value(v) = t else { return false };
        let reg = match (self.inputs[v.0 as usize], self.phis[v.0 as usize]) {
            (Some(reg), _) => Some(reg),
            (None, Some((b, i))) => Some(self.p.blocks[b].phis[i].reg),
            (None, None) => self
                .def(v)
                .and_then(|i| i.writes.iter().find(|w| w.1 == v))
                .map(|w| w.0),
        };
        matches!(reg, Some(Reg::V(_)))
    }

    /// [`Self::lane_term`], where every lane in `known` runs each VGPR copy
    /// it follows.
    fn lane_value(&self, mut t: Term, known: &Known) -> Result<Term, &'static str> {
        for _ in 0..MAX_JOINS * 2 {
            let Term::Value(v) = t else { break };
            let next = self.vstep(v);
            if next == t {
                break;
            }
            if self.varies(t)
                && let Some(site) = self.defs[v.0 as usize]
            {
                self.runs(site, known)?;
            }
            t = next;
        }
        Ok(t)
    }

    /// One copy of [`Self::vorigin`]: what the `v_mov_b32` writing `v`
    /// reads, or `v`'s origin.
    fn vstep(&self, v: Value) -> Term {
        if let Some(inst) = self.def(v)
            && inst.name() == "v_mov_b32"
            && let Fields::Valu(f) = &inst.fields
            && f.abs == 0
            && f.neg == 0
            && f.sdwa.is_none()
            && f.dpp.is_none()
            && let Some(t) = f.src.first().and_then(|s| self.raw(inst, s))
        {
            return t;
        }
        self.origin(v)
    }

    /// A VALU operand as `inst` reads it: a VGPR's value itself, other
    /// operands' origins.
    fn raw(&self, inst: &Inst, op: &Operand) -> Option<Term> {
        match *op {
            Operand::Vector { reg, .. } => Some(Term::Value(inst.read(Reg::V(reg))?)),
            op => self.term(inst, &op),
        }
    }

    /// Fails unless every lane in `known` that reaches the vector
    /// instruction at `site` runs it: one with its EXEC bit clear there
    /// keeps what the registers held before.
    fn runs(&self, site: Site, known: &Known) -> Result<(), &'static str> {
        let Some(lanes) = self.lanes else {
            return Ok(());
        };
        let Some(&Some(guard)) = lanes.guards.get(site.0).and_then(|g| g.get(site.1)) else {
            return Ok(());
        };
        let mut skips = known.clone();
        skips.extend([(lanes.reach[site.0], true), (guard, false)]);
        if lanes.bools.meet(&skips) {
            return Err("a per-lane descriptor written where some lanes may keep the old one");
        }
        Ok(())
    }

    /// The source of descriptor dwords each lane holds its own of (a
    /// waterfall loop reads them from the first active lane): the same in
    /// every lane, loaded per lane from memory a heap address points to, or
    /// picked per lane between such sources, by a lane mask or by the path
    /// the lane took. `known` holds in every lane that accesses them.
    fn per_lane(
        &self,
        words: &[Term],
        joins: &mut Vec<usize>,
        known: &mut Known,
    ) -> Result<Source, &'static str> {
        let words = words
            .iter()
            .map(|&w| self.lane_value(w, known))
            .collect::<Result<Vec<_>, _>>()?;
        let words = &words[..];
        let Some(&lane) = words.iter().find(|&&w| self.varies(w)) else {
            return self.classify(words, joins, known);
        };
        if let Some(source) = self.fetched(words, known)? {
            return Ok(source);
        }
        if let Some(source) = self.lane_select(words, joins, known)? {
            return Ok(source);
        }
        if let Some(source) = self.lane_phi(words, joins, known)? {
            return Ok(source);
        }
        let Term::Value(v) = lane else { unreachable!() };
        let name = self.def(v).map(|i| i.name());
        Err(match name {
            _ if self.inputs[v.0 as usize].is_some() => {
                "a descriptor read from the first lane of an input VGPR"
            }
            None => "a descriptor read from the first lane of a phi",
            Some(n) if n.starts_with("ds_read") => {
                "a descriptor read from the first lane of an LDS read"
            }
            Some(n) if n.starts_with("buffer_load") => {
                "a descriptor read from the first lane of a buffer load"
            }
            Some(n) if n.starts_with("global_load") || n.starts_with("flat_load") => {
                "a descriptor read from the first lane of a partial or mixed load"
            }
            Some(_) => "a descriptor read from the first lane of computed VGPRs",
        })
    }

    /// One descriptor the code keeps per lane in VGPRs: `lanes`, the VGPR
    /// values a waterfall loop reads the first lane of, hold only copies
    /// (`v_mov_b32` from SGPRs, through joins, VGPR moves and per-lane
    /// selects) of the same dwords of scalar loads from one address. The
    /// console compiler does this to a descriptor it loads on both sides of
    /// a per-lane branch (the sampler of the G-buffer's static and foliage
    /// meshes): the join is per lane, so it lives in VGPRs, and the access
    /// that needs it in SGPRs runs in a loop over the values the lanes hold.
    /// Every lane holds the same dwords, so the descriptor is the first of
    /// those loads, and the loop takes every lane in one pass (its proof is
    /// [`Self::waterfall`]'s).
    ///
    /// Every definition that reaches the VGPRs on a path an accessing lane
    /// can have come by must be such a copy ([`Self::scalar_copies`]): what
    /// they hold on other paths (numbers the code computed for another
    /// variable) no accessing lane reads. Refused: a VGPR that may hold
    /// anything else there (a number, an entry value, a constant, a bit
    /// operation on an SGPR, which can keep another descriptor's bits), a
    /// copy of other dwords or of another address's, a constant copied
    /// from an SGPR (an immediate sampler on one path), a descriptor
    /// fetched per lane ([`Self::fetched`]'s form), or a load whose base is
    /// not a root parameter's SGPR (the handle is made where the
    /// descriptor is used, from the load's base).
    // Ported from fxceless's VIctory fork (commit 578d459) by fxceless
    // (`lane_copies`, `scalar_copies`, `base_from_entry`). Here: a fallback
    // where the per-path trace fails; the old value of a VGPR write is
    // followed where an accessing lane may keep it, whatever the
    // instruction (the fork follows it for moves alone, always); constant
    // copies refuse; the fork's numbers on other paths are taken as no
    // descriptor only where no accessing lane comes by them (the review of
    // meshdata's landing: one copy of A per dword was enough, and lanes
    // that ORed B's dwords with 0 into the VGPRs read A's handle).
    fn lane_copies(&self, lanes: &[Value], known: &Known) -> Option<Source> {
        let mut loads: Vec<Site> = Vec::new();
        let mut first = None;
        let start = self.lane_path(known);
        for (k, &v) in lanes.iter().enumerate() {
            let copies = self.scalar_copies(v, &start, known)?;
            if copies.is_empty() {
                return None;
            }
            for copy in copies {
                let Term::Value(c) = copy else { return None };
                let site = self.defs[c.0 as usize]?;
                let load = self.inst(site);
                if !load.name().starts_with("s_load_dword") {
                    return None;
                }
                // Dword k of the operand is dword `first + k` of each load.
                let dword = load.writes.iter().position(|w| w.1 == c)?;
                let f = dword.checked_sub(k)?;
                if *first.get_or_insert(f) != f {
                    return None;
                }
                if !loads.contains(&site) {
                    loads.push(site);
                }
            }
        }
        let address = |site: Site| {
            let inst = self.inst(site);
            let Fields::Smem {
                base,
                offset,
                soffset,
                ..
            } = &inst.fields
            else {
                return None;
            };
            self.smem_address(site, base, *offset, soffset).ok()
        };
        let a = address(*loads.first()?)?;
        if a.param == CODE || a.dynamic.is_some() || loads.iter().any(|&l| address(l) != Some(a)) {
            return None;
        }
        loads.sort_unstable();
        let load = *loads.iter().find(|&&l| self.base_from_entry(l))?;
        Some(Source::Loaded {
            load,
            first: u8::try_from(first?).ok()?,
        })
    }

    /// The SGPR dwords (as origins) that `v_mov_b32` and `v_cndmask_b32`
    /// copy into VGPR value `v`, through joins and VGPR moves, on each path
    /// that lanes in `known` (those that access the descriptor) can have
    /// come by ([`LanePath`], from `start`: [`Self::lane_path`]); with the
    /// old value of any VGPR write such a lane may reach with its EXEC bit
    /// clear, since it keeps that value. None where any other definition
    /// reaches `v` on such a path (a number the code computed, an entry
    /// value, a constant, a load), or past [`MAX_COPIES`] steps.
    fn scalar_copies(&self, v: Value, start: &LanePath, known: &Known) -> Option<Vec<Term>> {
        let mut seen = std::collections::HashSet::new();
        let (mut work, mut out) = (vec![(v, start.clone())], Vec::new());
        while let Some((v, path)) = work.pop() {
            if seen.len() > MAX_COPIES {
                return None;
            }
            if !seen.insert((v, path.clone())) {
                continue;
            }
            if let Some((b, i)) = self.phis[v.0 as usize] {
                let args = &self.p.blocks[b].phis[i].args;
                for (j, &a) in args.iter().enumerate() {
                    if let Some(next) = self.lane_edge(&path, b, j, known) {
                        work.push((a, next));
                    }
                }
                continue;
            }
            // An entry value is no copy.
            let site = self.defs[v.0 as usize]?;
            let inst = self.inst(site);
            let &(reg, _) = inst.writes.iter().find(|w| w.1 == v)?;
            // Where a lane that accesses the descriptor may have its EXEC
            // bit clear, it keeps the old value.
            if self.may_skip(site, &path, known) {
                work.push((inst.read(reg)?, path.clone()));
            }
            let Fields::Valu(f) = &inst.fields else {
                return None;
            };
            if f.abs != 0
                || f.neg != 0
                || f.clamp
                || f.omod != 0
                || f.sdwa.is_some()
                || f.dpp.is_some()
            {
                return None;
            }
            // The operands copied, each with what holds where it is picked.
            let picked = match (inst.name(), &f.src[..]) {
                ("v_mov_b32", [src, ..]) => vec![(*src, None)],
                ("v_cndmask_b32", [clear, set, rest @ ..]) => {
                    let bit = match rest {
                        [Operand::Scalar { code, .. }] => inst
                            .read(Reg::S(*code))
                            .and_then(|m| self.lanes?.masks.get(&m).copied()),
                        _ => None,
                    };
                    vec![
                        (*clear, bit.map(|b| (b, false))),
                        (*set, bit.map(|b| (b, true))),
                    ]
                }
                // A number the code computed.
                _ => return None,
            };
            for (op, side) in picked {
                let mut path = path.clone();
                if let Some(side) = side {
                    path.sides.push(side);
                    if !self.feasible(&path, known) {
                        continue;
                    }
                }
                match op {
                    Operand::Vector { reg, .. } => work.push((inst.read(Reg::V(reg))?, path)),
                    Operand::Scalar { code, .. } if code != NULL => {
                        out.push(self.origin(inst.read(Reg::S(code))?));
                    }
                    _ => return None,
                }
            }
        }
        Some(out)
    }

    /// The start of [`Self::scalar_copies`]'s walk: the lanes in `known`,
    /// and the VGPR flags their conditions compare with a constant.
    fn lane_path(&self, known: &Known) -> LanePath {
        let mut path = LanePath {
            access: true,
            ..LanePath::default()
        };
        let Some(lanes) = self.lanes else {
            return path;
        };
        let mut seen = std::collections::HashSet::new();
        let atoms = known
            .iter()
            .flat_map(|&(b, _)| lanes.bools.support(b))
            .filter(|&a| seen.insert(a));
        for atom in atoms {
            if let Atom::Lane(c) = atom
                && let Some(bit) = lanes.bools.var(atom)
                && let Some(flag) = self.flag(c, bit)
            {
                match self.flag_constant(flag.value) {
                    Some(x) => path.sides.push((bit, flag.holds(x))),
                    None => path.flags.push(flag),
                }
            }
        }
        path
    }

    /// The compare writing `c` if it tests a VGPR against a constant
    /// (`v_cmp_eq_u32 vcc, 1, v8`): the lane's bit of it is `bit`.
    fn flag(&self, c: Value, bit: Bool) -> Option<Flag> {
        let inst = self.def(c)?;
        if inst.writes.first()?.1 != c {
            return None;
        }
        let Fields::Valu(f) = &inst.fields else {
            return None;
        };
        if f.abs != 0 || f.neg != 0 || f.sdwa.is_some() || f.dpp.is_some() {
            return None;
        }
        let rest = inst.name().strip_prefix("v_cmp_")?;
        let (op, ty) = rest.split_once('_')?;
        let op = match op {
            "eq" => Cmp::Eq,
            "ne" => Cmp::Ne,
            "lt" => Cmp::Lt,
            "le" => Cmp::Le,
            "gt" => Cmp::Gt,
            "ge" => Cmp::Ge,
            _ => return None,
        };
        let signed = match ty {
            "u32" => false,
            "i32" => true,
            _ => return None,
        };
        let constant = |o: &Operand| match *o {
            Operand::Const(k) => inline(k),
            Operand::Literal(x) => Some(x),
            _ => None,
        };
        let vgpr = |o: &Operand| match *o {
            Operand::Vector { reg, .. } => inst.read(Reg::V(reg)),
            _ => None,
        };
        let [a, b] = f.src.get(..2)? else {
            return None;
        };
        let (value, constant, flag_first) = match (constant(a), vgpr(b), vgpr(a), constant(b)) {
            (Some(k), Some(x), _, _) => (x, k, false),
            (_, _, Some(x), Some(k)) => (x, k, true),
            _ => return None,
        };
        Some(Flag {
            bit,
            value,
            op,
            signed,
            constant,
            flag_first,
        })
    }

    /// The constant VGPR value `v` holds in every lane that comes by its
    /// definition: a `v_mov_b32` of a constant that no such lane skips.
    fn flag_constant(&self, v: Value) -> Option<u32> {
        let (b, i) = self.defs[v.0 as usize]?;
        let inst = self.inst((b, i));
        let Fields::Valu(f) = &inst.fields else {
            return None;
        };
        if inst.name() != "v_mov_b32"
            || f.abs != 0
            || f.neg != 0
            || f.sdwa.is_some()
            || f.dpp.is_some()
            || self.lanes?.guards[b][i].is_some()
        {
            return None;
        }
        match self.vterm(inst, f.src.first()?)? {
            Term::Const(x) => Some(x),
            Term::Value(_) => None,
        }
    }

    /// `path` taken back over edge `j` into `block`, a join, if lanes in
    /// `known` can have come by it. Past a back edge the walk is in an
    /// earlier pass, which what lanes knew later says nothing of.
    fn lane_edge(
        &self,
        path: &LanePath,
        block: usize,
        j: usize,
        known: &Known,
    ) -> Option<LanePath> {
        let Some(lanes) = self.lanes else {
            return Some(path.clone());
        };
        let q = self.p.blocks[block].preds[j];
        if self.rpo[q] >= self.rpo[block] {
            return Some(LanePath::default());
        }
        let mut next = path.clone();
        next.reaches.push((lanes.reach[q], true));
        if let LaneExit::Branch {
            when,
            taken,
            not_taken,
        } = lanes.exits[q]
            && taken != not_taken
        {
            next.sides.push((when, taken == block));
        }
        // A flag joined here holds what the edge brings.
        for mut flag in std::mem::take(&mut next.flags) {
            if let Some((b, i)) = self.phis[flag.value.0 as usize]
                && b == block
            {
                flag.value = self.p.blocks[b].phis[i].args[j];
            }
            match self.flag_constant(flag.value) {
                Some(x) => next.sides.push((flag.bit, flag.holds(x))),
                None => next.flags.push(flag),
            }
        }
        self.feasible(&next, known).then_some(next)
    }

    /// Whether some lane in `known` can have come by `path`. Asked of the
    /// path's own branch sides first: what no lane satisfies needs no
    /// more, and `meet` answers yes when its search grows too large, as
    /// with a late block's reach in a long program (the sampler loop at
    /// 0x5780 of 903776ba469c8bac, whose per-path trace fails so).
    fn feasible(&self, path: &LanePath, known: &Known) -> bool {
        let Some(lanes) = self.lanes else {
            return true;
        };
        if !lanes.bools.meet(&path.sides) {
            return false;
        }
        let mut all = path.sides.clone();
        all.extend_from_slice(&path.reaches);
        if path.access {
            all.extend_from_slice(known);
        }
        lanes.bools.meet(&all)
    }

    /// Whether a lane that comes by `path` may skip the vector instruction
    /// at `site` (its EXEC bit clear there), keeping what the registers held
    /// before. Without the lanes' paths, any may.
    fn may_skip(&self, site: Site, path: &LanePath, known: &Known) -> bool {
        let Some(lanes) = self.lanes else {
            return true;
        };
        let Some(&Some(guard)) = lanes.guards.get(site.0).and_then(|g| g.get(site.1)) else {
            return false;
        };
        let mut skips = path.clone();
        skips.sides.push((guard, false));
        skips.reaches.push((lanes.reach[site.0], true));
        self.feasible(&skips, known)
    }

    /// Whether the scalar load at `site` has no register offset and takes
    /// its base from a root parameter's user SGPR through moves alone: a
    /// value that exists wherever the descriptor is used.
    fn base_from_entry(&self, site: Site) -> bool {
        let inst = self.inst(site);
        let Fields::Smem {
            base: Operand::Scalar { code, .. },
            soffset: Operand::Scalar { code: NULL, .. },
            ..
        } = &inst.fields
        else {
            return false;
        };
        let Some(mut v) = inst.read(Reg::S(*code)) else {
            return false;
        };
        for _ in 0..MAX_COPIES {
            if self.param_value(v).is_some() {
                return true;
            }
            let Some(def) = self.def(v) else { return false };
            let Some(k) = def.writes.iter().position(|w| w.1 == v) else {
                return false;
            };
            match copy_source(def, k) {
                Some(Moved::Value(s)) => v = s,
                _ => return false,
            }
        }
        false
    }

    /// Dwords of `global_load_dword*` that line up: `first..` of one load,
    /// then the bytes that follow (an image descriptor takes two
    /// `global_load_dwordx4`).
    fn fetched(&self, words: &[Term], known: &Known) -> Result<Option<Source>, &'static str> {
        let at = |w: &Term| -> Option<(Site, usize)> {
            let Term::Value(v) = *w else { return None };
            let site = self.defs[v.0 as usize]?;
            let inst = self.inst(site);
            if !inst.name().starts_with("global_load_dword") {
                return None;
            }
            let k = inst
                .writes
                .iter()
                .filter(|w| matches!(w.0, Reg::V(_)))
                .position(|w| w.1 == v)?;
            Some((site, k))
        };
        let Some(sites) = words.iter().map(at).collect::<Option<Vec<_>>>() else {
            return Ok(None);
        };
        let address = |site: Site| {
            let inst = self.inst(site);
            let Fields::Flat {
                addr,
                saddr,
                offset,
                ..
            } = &inst.fields
            else {
                return Err("a global load");
            };
            self.flat_address(site, addr.as_ref(), saddr.as_ref(), *offset)
        };
        let (load, first) = sites[0];
        let a0 = address(load)?;
        for (k, &(site, dword)) in sites.iter().enumerate() {
            self.runs(site, known)?;
            let a = address(site)?;
            let want = a0.offset.wrapping_add(4 * (first + k) as u32);
            if (a.param, a.dynamic) != (a0.param, a0.dynamic)
                || a.offset.wrapping_add(4 * dword as u32) != want
            {
                return Err("a descriptor read from the first lane of loads that do not line up");
            }
        }
        Ok(Some(Source::Fetched {
            load,
            first: first as u8,
        }))
    }

    /// Words that `v_cndmask_b32` picks by one lane mask, traced per outcome
    /// (other words are the same for both).
    fn lane_select(
        &self,
        words: &[Term],
        joins: &mut Vec<usize>,
        known: &mut Known,
    ) -> Result<Option<Source>, &'static str> {
        let picks: Vec<_> = words
            .iter()
            .map(|w| match *w {
                Term::Value(v) => self.lane_picked(v),
                Term::Const(_) => None,
            })
            .collect();
        let Some(mask) = picks.iter().flatten().map(|p| p.0).next() else {
            return Ok(None);
        };
        if picks.iter().flatten().any(|p| p.0 != mask) {
            return Err("a descriptor chosen per lane by several masks");
        }
        for (w, p) in words.iter().zip(&picks) {
            if let (Term::Value(v), Some(_)) = (*w, p)
                && let Some(site) = self.defs[v.0 as usize]
            {
                self.runs(site, known)?;
            }
        }
        if joins.len() >= MAX_JOINS {
            return Err("a descriptor joined too often");
        }
        joins.push(usize::MAX);
        // Each side holds where the lane's bit of the mask is set, or clear.
        let bit = self.lanes.and_then(|l| l.masks.get(&mask).copied());
        let side = |set: bool| -> Vec<Term> {
            words
                .iter()
                .zip(&picks)
                .map(|(w, p)| match p {
                    Some((_, t, f)) => {
                        if set {
                            *t
                        } else {
                            *f
                        }
                    }
                    None => *w,
                })
                .collect()
        };
        let args = [true, false]
            .into_iter()
            .map(|set| {
                let n = known.len();
                known.extend(bit.map(|b| (b, set)));
                let arg = self.per_lane(&side(set), joins, known);
                known.truncate(n);
                arg
            })
            .collect::<Result<_, _>>();
        joins.pop();
        Ok(Some(Source::Lanes { mask, args: args? }))
    }

    /// For `v` written by a plain `v_cndmask_b32`: the low dword of the
    /// lane mask it reads, and the dword it is where the lane's bit is set
    /// (src1), then clear (src0).
    fn lane_picked(&self, v: Value) -> Option<(Value, Term, Term)> {
        let inst = self.def(v)?;
        let Fields::Valu(f) = &inst.fields else {
            return None;
        };
        let Some(Operand::Vector { reg, .. }) = f.dst else {
            return None;
        };
        if inst.name() != "v_cndmask_b32"
            || inst.write(Reg::V(reg)) != Some(v)
            || f.abs != 0
            || f.neg != 0
            || f.clamp
            || f.omod != 0
            || f.sdwa.is_some()
            || f.dpp.is_some()
        {
            return None;
        }
        let [clear, set, Operand::Scalar { code, .. }] = f.src[..] else {
            return None;
        };
        let mask = inst.read(Reg::S(code))?;
        Some((mask, self.raw(inst, &set)?, self.raw(inst, &clear)?))
    }

    /// Per-lane words that phis of one block join, as an if/else on a
    /// per-lane condition leaves them: traced on each edge into the block
    /// that lanes in `known` can take (other words hold on every edge).
    /// Lanes take their own paths ([`Structured`]), so each gets the value
    /// of the edge it came by; an edge they cannot take is `None` (a flow
    /// block's phi joins an arm's dwords with what lanes that skipped the
    /// arm held before, which only they read).
    fn lane_phi(
        &self,
        words: &[Term],
        joins: &mut Vec<usize>,
        known: &mut Known,
    ) -> Result<Option<Source>, &'static str> {
        let phi_of = |w: &Term| match *w {
            Term::Value(v) => self.phis[v.0 as usize],
            Term::Const(_) => None,
        };
        let Some((block, _)) = words.iter().find_map(phi_of) else {
            return Ok(None);
        };
        if words.iter().filter_map(phi_of).any(|(b, _)| b != block) {
            return Err("a descriptor joined in several blocks");
        }
        if joins.contains(&block) {
            return Err("a descriptor that changes in a loop");
        }
        let preds = &self.p.blocks[block].preds;
        if preds.iter().any(|&q| self.rpo[q] >= self.rpo[block]) {
            // shortcut: what a lane holds after some passes is not traced.
            return Err("a descriptor read from the first lane of a phi at a loop header");
        }
        let Some(lanes) = self.lanes else {
            return Err("a descriptor read from the first lane of a phi, without the lanes' paths");
        };
        // The other words must be there before the join: defined in a
        // block that dominates it, which (both dominating the access) is
        // one the join does not reach.
        for w in words {
            if let Term::Value(v) = *w
                && phi_of(w).is_none()
                && let Some((d, _)) = self.defs[v.0 as usize]
                && (d == block || self.reaches(block, d))
            {
                return Err("a descriptor partly joined by control flow");
            }
        }
        if joins.len() >= MAX_JOINS {
            return Err("a descriptor joined too often");
        }
        joins.push(block);
        let phis = &self.p.blocks[block].phis;
        let mut args = Vec::new();
        for (j, &q) in preds.iter().enumerate() {
            let n = known.len();
            known.push((lanes.reach[q], true));
            if let LaneExit::Branch {
                when,
                taken,
                not_taken,
            } = lanes.exits[q]
                && taken != not_taken
            {
                known.push((when, taken == block));
            }
            let arg = if lanes.bools.meet(known) {
                let edge: Vec<Term> = words
                    .iter()
                    .map(|w| match phi_of(w) {
                        Some((_, i)) => Term::Value(phis[i].args[j]),
                        None => *w,
                    })
                    .collect();
                self.per_lane(&edge, joins, known).map(Some)
            } else {
                Ok(None)
            };
            known.truncate(n);
            match arg {
                Ok(a) => args.push(a),
                Err(e) => {
                    joins.pop();
                    return Err(e);
                }
            }
        }
        joins.pop();
        Ok(Some(Source::LanePhi { block, args }))
    }

    /// Whether a path leads from block `from` to block `to` (to `from`
    /// itself only through a back edge).
    fn reaches(&self, from: usize, to: usize) -> bool {
        let mut seen = vec![false; self.p.blocks.len()];
        let mut stack = self.p.blocks[from].exit.successors();
        while let Some(x) = stack.pop() {
            if x == to {
                return true;
            }
            if std::mem::replace(&mut seen[x], true) {
                continue;
            }
            stack.extend(self.p.blocks[x].exit.successors());
        }
        false
    }

    /// Prove a single-block waterfall at the descriptor access. In each
    /// iteration the access serves exactly the matching lanes, and the
    /// back edge keeps exactly the other lanes active. Otherwise, why not.
    ///
    // shortcut: only the accesses are proven. Any other per-lane work in the
    // loop is taken to run once for each lane, as the access does; a body
    // whose effect adds up over the passes a lane waits through (a counter
    // before the mask is narrowed) would differ. Compilers put nothing but
    // the reads, the compares and the access in these loops.
    fn waterfall(
        &self,
        b: usize,
        accesses: &[usize],
        reads: &[Value],
    ) -> Result<Waterfall, &'static str> {
        const LOOP: &str = "a descriptor read from the first lane outside a waterfall loop";
        const SPLIT: &str = "a waterfall loop nested in another or split over blocks";
        const SELECT: &str = "a waterfall loop whose compares do not select the access's lanes";
        const RETRY: &str = "a waterfall loop whose retry does not keep the lanes still waiting";
        let access = accesses[0];
        let block = &self.p.blocks[b];
        let lanes = self.lanes.ok_or(LOOP)?;
        let exit = block.insts.last().ok_or(LOOP)?;
        if !matches!(block.exit, Exit::Branch { taken, not_taken } if taken == b && not_taken != b)
            || exit.name() != "s_cbranch_execnz"
        {
            return Err(LOOP);
        }
        let mask = |low: Option<Value>, high: Option<Value>| {
            let high = if self.p.wave64 { Some(high?) } else { None };
            lanes.mask_bits(low?, high)
        };
        let exec = |inst: &Inst| mask(inst.read(Reg::S(EXEC)), inst.read(Reg::S(EXEC + 1)));
        // The reads belong to this loop's own block: a loop around this one
        // (one per operand, as LLVM nests them for a per-lane image and
        // sampler) is not followed.
        if reads
            .iter()
            .any(|v| self.defs[v.0 as usize].is_none_or(|at| at.0 != b))
        {
            return Err(SPLIT);
        }
        let selected = exec(&block.insts[access]).ok_or(SELECT)?;
        if accesses
            .iter()
            .any(|&i| exec(&block.insts[i]) != Some(selected))
        {
            return Err(SELECT);
        }
        let retry = exec(exit).ok_or(RETRY)?;
        let first = self.defs[reads[0].0 as usize].ok_or(LOOP)?;
        let remaining = exec(self.inst(first)).ok_or(SELECT)?;
        if reads.iter().any(|v| {
            self.defs[v.0 as usize]
                .is_none_or(|at| at.1 >= access || exec(self.inst(at)) != Some(remaining))
        }) {
            return Err(SELECT);
        }
        // Whether a lane that reaches the loop can satisfy `conditions`.
        // Asked without the reach first: what no lane satisfies needs no
        // reach, and `meet` answers yes when its search grows too large, as
        // with a late block's reach in a long program (the sampler loops of
        // the G-buffer's static and foliage meshes, kept per lane:
        // `lane_copies`).
        let reach = (lanes.reach[b], true);
        let meet = |conditions: &[(Bool, bool)]| {
            lanes.bools.meet(conditions) && {
                let mut all = vec![reach];
                all.extend_from_slice(conditions);
                lanes.bools.meet(&all)
            }
        };
        let mut out = Waterfall::default();
        let mut required = vec![(remaining, true)];
        for (i, inst) in block.insts[..access].iter().enumerate() {
            let Fields::Valu(f) = &inst.fields else {
                continue;
            };
            let Some(ty) = inst
                .name()
                .strip_prefix("v_cmp_eq_")
                .or(inst.name().strip_prefix("v_cmpx_eq_"))
            else {
                continue;
            };
            let dwords = match ty {
                "u32" | "i32" => 1,
                "u64" | "i64" => 2,
                _ => continue,
            };
            if f.abs != 0 || f.neg != 0 || f.sdwa.is_some() || f.dpp.is_some() {
                continue;
            }
            // The first lane's dwords on one side, the lane's own on the
            // other.
            let pair = |first: &Operand, own: &Operand| -> Option<Vec<Value>> {
                let (&Operand::Scalar { code, .. }, &Operand::Vector { reg, .. }) = (first, own)
                else {
                    return None;
                };
                (0..dwords)
                    .map(|k| {
                        let Term::Value(x) = self.origin(inst.read(Reg::S(code + k))?) else {
                            return None;
                        };
                        let lane = inst.read(Reg::V(reg + k))?;
                        let (_, read) = self.defs[x.0 as usize]?;
                        let same = reads.contains(&x)
                            && read < i
                            && self.lane_term(Term::Value(self.first_lane_source(x)?))
                                == self.lane_term(Term::Value(lane));
                        same.then_some(x)
                    })
                    .collect()
            };
            let (Some(a), Some(c)) = (f.src.first(), f.src.get(1)) else {
                continue;
            };
            if let Some(xs) = pair(a, c).or_else(|| pair(c, a)) {
                let code = if inst.name().starts_with("v_cmpx_") {
                    EXEC
                } else if let Some(Operand::Scalar { code, .. }) = f.dst {
                    code
                } else {
                    continue;
                };
                let Some(matched) = mask(inst.write(Reg::S(code)), inst.write(Reg::S(code + 1)))
                else {
                    continue;
                };
                if meet(&[(selected, true), (matched, false)]) {
                    continue;
                }
                required.push((matched, true));
                out.compares.push((b, i));
                for x in xs {
                    if !out.reads.contains(&x) {
                        out.reads.push(x);
                    }
                }
            }
        }
        if reads.iter().any(|v| !out.reads.contains(v))
            || meet(&[(selected, true), (remaining, false)])
        {
            return Err(SELECT);
        }
        // The access must take every match.
        required.push((selected, false));
        if meet(&required) {
            return Err(SELECT);
        }
        // The retry must keep exactly the lanes that were eligible but did
        // not perform the access.
        if meet(&[(retry, true), (remaining, false)])
            || meet(&[(retry, true), (selected, true)])
            || meet(&[(remaining, true), (selected, false), (retry, false)])
        {
            return Err(RETRY);
        }
        Ok(out)
    }

    /// Consecutive dwords of one scalar load.
    fn loaded(&self, words: &[Term]) -> Option<Source> {
        let Term::Value(v0) = words[0] else {
            return None;
        };
        let load = self.defs[v0.0 as usize]?;
        let inst = self.inst(load);
        if !inst.name().starts_with("s_load_") {
            return None;
        }
        let first = inst.writes.iter().position(|w| w.1 == v0)?;
        let same = words.iter().enumerate().all(|(k, w)| {
            inst.writes
                .get(first + k)
                .is_some_and(|x| *w == Term::Value(x.1))
        });
        same.then_some(Source::Loaded {
            load,
            first: first as u8,
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Moved {
    Const(u32),
    Value(Value),
}

impl Tracer<'_> {
    /// For `v_readlane_b32` of a constant lane: the SGPR spilled to that
    /// lane, found by following the VGPR's `v_writelane_b32` writes.
    fn restored(&self, inst: &Inst) -> Option<Moved> {
        let Fields::Valu(v) = &inst.fields else {
            return None;
        };
        if inst.name() != "v_readlane_b32" {
            return None;
        }
        let [Operand::Vector { reg, .. }, lane] = v.src[..] else {
            return None;
        };
        self.spilled(inst.read(Reg::V(reg))?, lane_of(lane)?, &mut Vec::new())
            .ok()
            .flatten()
    }

    /// What lane `lane` of VGPR value `v` holds: `Ok(None)` for a phi in
    /// `seen` (already being resolved), `Err` if not a spilled SGPR.
    fn spilled(&self, v: Value, lane: u32, seen: &mut Vec<Value>) -> Result<Option<Moved>, ()> {
        if let Some((b, i)) = self.phis[v.0 as usize] {
            if seen.contains(&v) {
                return Ok(None);
            }
            seen.push(v);
            let mut found = None;
            for &a in &self.p.blocks[b].phis[i].args {
                if let Some(m) = self.spilled(a, lane, seen)? {
                    if found.is_some_and(|f| f != m) {
                        return Err(());
                    }
                    found = Some(m);
                }
            }
            return Ok(found);
        }
        let inst = self.def(v).ok_or(())?;
        let Fields::Valu(w) = &inst.fields else {
            return Err(());
        };
        let (Some(Operand::Vector { reg, .. }), [value, at]) = (w.dst, &w.src[..]) else {
            return Err(());
        };
        if inst.name() != "v_writelane_b32" {
            return Err(());
        }
        if lane_of(*at).ok_or(())? != lane {
            return self.spilled(inst.read(Reg::V(reg)).ok_or(())?, lane, seen);
        }
        match *value {
            Operand::Scalar { code, .. } if code != NULL => {
                inst.read(Reg::S(code)).map(Moved::Value)
            }
            Operand::Literal(x) => Some(Moved::Const(x)),
            Operand::Const(c) => inline(c).map(Moved::Const),
            _ => None,
        }
        .map(Some)
        .ok_or(())
    }
}

/// A lane select given as a constant.
fn lane_of(op: Operand) -> Option<u32> {
    match op {
        Operand::Const(c) => inline(c),
        Operand::Literal(x) => Some(x),
        _ => None,
    }
}

/// What dword `k` of a move's result copies.
fn copy_source(inst: &Inst, k: usize) -> Option<Moved> {
    let Fields::Sop { src, imm, .. } = &inst.fields else {
        return None;
    };
    if inst.name() == "s_movk_i32" {
        return Some(Moved::Const(i32::from((*imm)? as i16) as u32));
    }
    let src = src.first()?;
    match (inst.name(), *src) {
        ("s_mov_b32" | "s_mov_b64", Operand::Scalar { code, .. }) if code != NULL => {
            inst.read(Reg::S(code + k as u8)).map(Moved::Value)
        }
        ("s_mov_b32" | "s_mov_b64", Operand::Scalar { .. }) => Some(Moved::Const(0)),
        ("s_mov_b32", Operand::Literal(x)) => Some(Moved::Const(x)),
        ("s_mov_b32", Operand::Const(c)) => inline(c).map(Moved::Const),
        // shortcut: only the one-operand ways the compiler makes a constant
        // that has no inline form; fold more if descriptors need it.
        ("s_brev_b32", Operand::Const(c)) => inline(c).map(|x| Moved::Const(x.reverse_bits())),
        ("s_not_b32", Operand::Const(c)) => inline(c).map(|x| Moved::Const(!x)),
        // 64-bit inline integers are sign-extended.
        ("s_mov_b64", Operand::Const(c @ 128..=208)) => {
            let low = inline(c)?;
            Some(Moved::Const(if k == 0 {
                low
            } else {
                ((low as i32) >> 31) as u32
            }))
        }
        _ => None,
    }
}

/// The 32-bit value of an inline constant operand.
pub(crate) fn inline(code: u8) -> Option<u32> {
    let float = |f: f32| Some(f.to_bits());
    match code {
        128..=192 => Some(u32::from(code) - 128),
        193..=208 => Some((192 - i32::from(code)) as u32),
        240 => float(0.5),
        241 => float(-0.5),
        242 => float(1.0),
        243 => float(-1.0),
        244 => float(2.0),
        245 => float(-2.0),
        246 => float(4.0),
        247 => float(-4.0),
        248 => Some(0x3E22_F983),
        _ => None,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::native::RootType::*;

    const S_ENDPGM: u32 = 0xBF81_0000;

    fn track_words(words: &[u32], layout: &[RootParam], stage: HwStage) -> Resources {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        track(&p, layout, stage, lanes(&p).as_ref()).unwrap()
    }

    /// `p` as one lane runs it, if it structures.
    fn lanes(p: &Program) -> Option<Structured> {
        crate::structure::structure(p).ok()
    }

    fn param(root: RootType, user_data: u8, sgprs: u32) -> RootParam {
        RootParam {
            root,
            table_offset_dwords: 0,
            sgprs,
            base_register: 0,
            user_data,
            slot: 0,
        }
    }

    fn at(param: usize, offset: u32) -> Address {
        Address {
            param,
            offset,
            dynamic: None,
        }
    }

    #[test]
    fn table_descriptors_are_loaded_from_their_range() {
        let r = track_words(
            &[
                0xBE82_0309, // s_mov_b32 s2, s9
                0xBE89_0380, // s_mov_b32 s9, 0
                0xBE83_0380, // s_mov_b32 s3, 0
                0xF408_0204,
                0xFA00_0000, // s_load_dwordx4 s[8:11], s[8:9], null
                0xF40C_0001,
                0xFA00_0000, // s_load_dwordx8 s[0:7], s[2:3], null
                0xF080_0F08,
                0x0040_0002, // image_sample v[0:3], v[2:3], s[0:7], s[8:11]
                S_ENDPGM,
            ],
            &[param(Table, 8, 1), param(Table, 9, 1)],
            HwStage::Ps,
        );
        assert_eq!(r.scalar, [((0, 3), at(0, 0)), ((0, 4), at(1, 0))]);
        let roles: Vec<_> = r.descriptors.iter().map(|d| (d.role, &d.source)).collect();
        assert_eq!(
            roles,
            [
                (
                    Role::Resource,
                    &Source::Loaded {
                        load: (0, 4),
                        first: 0
                    }
                ),
                (
                    Role::Sampler,
                    &Source::Loaded {
                        load: (0, 3),
                        first: 0
                    }
                ),
            ]
        );
    }

    #[test]
    fn root_descriptors_are_built_in_code() {
        let r = track_words(
            &[
                0x8801_FF0B,
                0x0010_0000, // s_or_b32 s1, s11, 0x100000
                0xBE80_030A, // s_mov_b32 s0, s10
                0xBE82_03C1, // s_mov_b32 s2, -1
                0xBE83_03FF,
                0x1101_4FAC, // s_mov_b32 s3, 0x11014fac
                0xF430_0000,
                0xFA00_0000, // s_buffer_load_dwordx16 s[0:15], s[0:3], null
                S_ENDPGM,
            ],
            &[param(Descriptor, 10, 2)],
            HwStage::Vs,
        );
        assert_eq!(
            r.descriptors[0].source,
            Source::Built {
                address: at(0, 0),
                bits: 0x10_0000,
                words: [u32::MAX, 0x1101_4FAC]
            }
        );
        assert_eq!(
            r.params.iter().map(|p| (p.1, p.2)).collect::<Vec<_>>(),
            [(0, 0), (0, 1)]
        );
    }

    #[test]
    fn heap_indices_from_root_constants_in_a_merged_stage() {
        // User data starts at s8: the constant (index 6) is s14, the heap
        // (index 10) s18.
        let r = track_words(
            &[
                0x8F08_850E, // s_lshl_b32 s8, s14, 5
                0xBE93_0380, // s_mov_b32 s19, 0
                0xF408_0209,
                0x1000_0000, // s_load_dwordx4 s[8:11], s[18:19], s8
                0xE000_2000,
                0x8002_0400, // buffer_load_format_x v4, v0, s[8:11], 0 idxen
                S_ENDPGM,
            ],
            &[param(Constants, 6, 1), param(ResourceHeap, 10, 1)],
            HwStage::Ms,
        );
        let index = r.params.iter().find(|p| p.1 == 0).unwrap().0;
        let shift = r.scalar[0].1.dynamic.unwrap();
        assert_eq!(r.scalar[0].1.param, 1);
        // The computed term is the shifted constant, not the constant itself.
        assert_ne!(shift, index);
        assert_eq!(
            r.descriptors[0].source,
            Source::Loaded {
                load: (0, 2),
                first: 0
            }
        );
    }

    #[test]
    fn spilled_sgprs_are_followed_through_vgpr_lanes() {
        let r = track_words(
            &[
                0xD761_0000,
                0x0001_0609, // v_writelane_b32 v0, s9, 3
                0xD760_0002,
                0x0001_0700, // v_readlane_b32 s2, v0, 3
                0xBE83_0380, // s_mov_b32 s3, 0
                0xF40C_0001,
                0xFA00_0020, // s_load_dwordx8 s[0:7], s[2:3], 0x20
                S_ENDPGM,
            ],
            &[param(Table, 9, 1)],
            HwStage::Ps,
        );
        assert_eq!(r.scalar, [((0, 3), at(0, 0x20))]);
    }

    #[test]
    fn descriptors_joined_by_control_flow_are_traced_per_edge() {
        let r = track_words(
            &[
                0xBE82_0309, // s_mov_b32 s2, s9
                0xBE83_0380, // s_mov_b32 s3, 0
                0xBF06_8001, // s_cmp_eq_u32 s1, 0
                0xBF85_0003, // s_cbranch_scc1 -> 0x1c
                0xF408_0001,
                0xFA00_0000, // s_load_dwordx4 s[0:3], s[2:3], null
                0xBF82_0002, // s_branch -> 0x24
                0xF408_0001,
                0xFA00_0010, // 0x1c: s_load_dwordx4 s[0:3], s[2:3], 0x10
                0xF420_0100,
                0xFA00_0000, // 0x24: s_buffer_load_dword s4, s[0:3], null
                S_ENDPGM,
            ],
            &[param(Table, 9, 1)],
            HwStage::Ps,
        );
        let Source::Phi { args, .. } = &r.descriptors[0].source else {
            panic!("{:?}", r.descriptors)
        };
        let offsets: Vec<_> = args
            .iter()
            .map(|a| match a {
                Source::Loaded { load, .. } => {
                    r.scalar.iter().find(|s| s.0 == *load).unwrap().1.offset
                }
                a => panic!("{a:?}"),
            })
            .collect();
        assert_eq!(offsets.len(), 2);
        assert!(
            offsets.contains(&0) && offsets.contains(&0x10),
            "{offsets:?}"
        );
    }

    /// A wave64 compute program that builds a sampler's constants in s[0:3]
    /// in an arm entered by `guard` (an `s_and_saveexec_b64 s[20:21], ...`
    /// skipped by `s_cbranch_execz`) and samples with it after the arm.
    /// Before the arm s[0:3] hold two lane masks: what the edge that skips
    /// the arm joins in.
    fn sampler_built_in_an_arm(guard: u32) -> Result<Resources, ShaderError> {
        sampler_built_in_an_arm_after(guard, &[])
    }

    /// [`sampler_built_in_an_arm`] with `first` at the start of the arm.
    fn sampler_built_in_an_arm_after(guard: u32, first: &[u32]) -> Result<Resources, ShaderError> {
        let mut words = vec![
            0xBEA8_047E, // s_mov_b64 s[40:41], exec
            0xBEFE_0A7E, // s_wqm_b64 exec, exec
            0xBE8C_0309, // s_mov_b32 s12, s9
            0xBE8D_0380, // s_mov_b32 s13, 0
            0xF40C_0106,
            0xFA00_0000, // s_load_dwordx8 s[4:11], s[12:13], null
            0x7E04_0280, // v_mov_b32 v2, 0
            0x7E06_0280, // v_mov_b32 v3, 0
            0x7D88_0085, // v_cmp_gt_u32 vcc, 5, v0: lanes 0 to 4
            0xBE80_046A, // s_mov_b64 s[0:1], vcc
            0xBE82_246A, // s_and_saveexec_b64 s[2:3], vcc
            0xBEFE_0402, // s_mov_b64 exec, s[2:3]
            guard,
            0xBF88_0006 + first.len() as u32, // s_cbranch_execz -> the join
        ];
        words.extend(first);
        words.extend([
            0x7E26_02F2, // v_mov_b32 v19, 1.0
            0xB000_0092, // s_movk_i32 s0, 0x92
            0xBE81_03FF,
            0x00C0_0000, // s_mov_b32 s1, 0xc00000
            0xBE82_0BA8, // s_brev_b32 s2, 40
            0xBE83_0380, // s_mov_b32 s3, 0
            0xBEFE_0414, // the join: s_mov_b64 exec, s[20:21]
            0xF09C_0108,
            0x0001_0102, // image_sample_lz v1, v[2:3], s[4:11], s[0:3]
            S_ENDPGM,
        ]);
        wave64_compute(&words)
    }

    /// The same shape for the image descriptor: s[4:11] hold lane masks
    /// before an arm under the starting EXEC that loads them from the table
    /// at s[12:13] (`arm`: the load, after what else the arm does to the
    /// base), the sampler's constants set before the access.
    fn image_loaded_in_an_arm(arm: &[u32]) -> Result<Resources, ShaderError> {
        let skip = 0xBF88_0000 | arm.len() as u32;
        let mut words = vec![
            0xBEA8_047E, // s_mov_b64 s[40:41], exec
            0xBE8C_0309, // s_mov_b32 s12, s9
            0xBE8D_0380, // s_mov_b32 s13, 0
            0x7E04_0280, // v_mov_b32 v2, 0
            0x7E06_0280, // v_mov_b32 v3, 0
            0x7D88_0085, // v_cmp_gt_u32 vcc, 5, v0
            0xBE84_046A, // s_mov_b64 s[4:5], vcc
            0xBE86_046A, // s_mov_b64 s[6:7], vcc
            0xBE88_046A, // s_mov_b64 s[8:9], vcc
            0xBE8A_046A, // s_mov_b64 s[10:11], vcc
            0xBE94_2428, // s_and_saveexec_b64 s[20:21], s[40:41]
            skip,        // s_cbranch_execz over the arm
        ];
        words.extend(arm);
        words.extend([
            0xBEFE_0414, // s_mov_b64 exec, s[20:21]
            0xB000_0092, // s_movk_i32 s0, 0x92
            0xBE81_03FF,
            0x00C0_0000, // s_mov_b32 s1, 0xc00000
            0xBE82_0BA8, // s_brev_b32 s2, 40
            0xBE83_0380, // s_mov_b32 s3, 0
            0xF09C_0108,
            0x0001_0102, // image_sample_lz v1, v[2:3], s[4:11], s[0:3]
            S_ENDPGM,
        ]);
        wave64_compute(&words)
    }

    /// `words` as a wave64 compute program whose table is at s9, traced.
    fn wave64_compute(words: &[u32]) -> Result<Resources, ShaderError> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, true, 256).unwrap();
        let st = lanes(&p);
        assert!(st.is_some(), "the program structures");
        track(&p, &[param(Table, 9, 1)], HwStage::Cs, st.as_ref())
    }

    const S_LOAD_IMAGE: [u32; 2] = [0xF40C_0106, 0xFA00_0000]; // s_load_dwordx8 s[4:11], s[12:13], null

    /// The arm runs under the copy of EXEC the program started with, which
    /// every lane's bit is set in: the wave skips it only when no lane is
    /// left that would run it, and no lane takes the edge that skips it.
    /// The sampler is its constants (`2da4beb386509b77` at 0x4d0, refused
    /// as "neither loaded nor built" before: docs/analysis/world-black.md
    /// §8.4). Under a per-lane compare instead, lanes 5 and up skip the arm
    /// and sample with the lane masks: refused still.
    #[test]
    fn a_sampler_built_under_the_starting_exec_is_its_constants() {
        // s_and_saveexec_b64 s[20:21], s[40:41]
        let r = sampler_built_in_an_arm(0xBE94_2428).unwrap();
        let samplers: Vec<_> = r
            .descriptors
            .iter()
            .filter(|d| d.role == Role::Sampler)
            .collect();
        assert_eq!(samplers.len(), 1, "{:?}", r.descriptors);
        assert_eq!(
            samplers[0].source,
            Source::Constant(vec![0x92, 0x00C0_0000, 0x1400_0000, 0])
        );
        // s_and_saveexec_b64 s[20:21], s[0:1]: lanes 0 to 4
        match sampler_built_in_an_arm(0xBE94_2400) {
            Err(ShaderError::Unsupported { what, .. }) => {
                assert_eq!(what, "a descriptor neither loaded nor built");
            }
            other => panic!("not refused: {other:?}"),
        }
    }

    /// A per-lane branch inside such an arm, before the constants, parts a
    /// wave's path from its lanes' for a while: the wave runs the inner arm
    /// if lanes 0 to 4 do, and lanes 5 and up skip it. Both sides come
    /// back before the constants, so the wave still comes into the join
    /// from the arm's end, as every lane does: the constants still.
    #[test]
    fn a_per_lane_branch_inside_the_arm_keeps_the_constants() {
        let inner = [
            0xBE96_246A, // s_and_saveexec_b64 s[22:23], vcc: lanes 0 to 4
            0xBF88_0001, // s_cbranch_execz over the next word
            0x7E26_02F4, // v_mov_b32 v19, 2.0
            0xBEFE_0416, // s_mov_b64 exec, s[22:23]
        ];
        // s_and_saveexec_b64 s[20:21], s[40:41]
        let r = sampler_built_in_an_arm_after(0xBE94_2428, &inner).unwrap();
        let sampler = r.descriptors.iter().find(|d| d.role == Role::Sampler);
        assert_eq!(
            sampler.unwrap().source,
            Source::Constant(vec![0x92, 0x00C0_0000, 0x1400_0000, 0])
        );
    }

    /// The program of the relecture of task present's point 5 (below): even
    /// lanes run an arm that rewrites s8 of the sampler loaded from the
    /// table at s1, odd lanes sample with s[8:11] in the else arm.
    fn rewritten_in_an_arm() -> Vec<u32> {
        vec![
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xBE86_0301, // s_mov_b32 s6, s1
            0xBE87_0380, // s_mov_b32 s7, 0
            0xF40C_0602,
            0xFA00_0000, // s_load_dwordx8 s[24:31], s[4:5], 0x0
            0xF408_0203,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
            0xF408_0A02,
            0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
            0x7E04_0D00, // v_cvt_f32_u32 v2, v0
            0x7E06_0302, // v_mov_b32 v3, v2
            0x3420_0082, // v_lshlrev_b32 v16, 2, v0
            0x3602_0081, // v_and_b32 v1, 1, v0
            0x7D84_0280, // v_cmp_eq_u32 vcc_lo, 0, v1: even lanes
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xBE94_3C6A, // s_and_saveexec_b32 s20, vcc_lo
            0x8914_147E, // s_xor_b32 s20, exec_lo, s20
            0xBF88_0001, // s_cbranch_execz -> 0x50
            0xBE88_0309, // s_mov_b32 s8, s9
            0xBE94_3D14, // 0x50: s_or_saveexec_b32 s20, s20
            0x897E_147E, // s_xor_b32 exec_lo, exec_lo, s20: odd lanes
            0xBF88_0005, // s_cbranch_execz -> 0x70
            0xF09C_0F08,
            0x0046_0C02, // image_sample_lz v[12:15], v[2:3], s[24:31], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
            0xE070_1000,
            0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
            0x887E_147E, // 0x70: s_or_b32 exec_lo, exec_lo, s20
            S_ENDPGM,
        ]
    }

    /// The relecture of task present (Codex, point 5): even lanes run an arm
    /// that rewrites the sampler loaded from the table at s1 (dword 0 from
    /// dword 1: no descriptor the trace can name), and the odd lanes sample
    /// with it in the else arm. Those lanes skip the even lanes' arm, but
    /// the sampler is in SGPRs: a wave with an even lane ran that arm, and
    /// the console samples with the rewritten dwords. So the join's edge
    /// that skips the arm, the only one the odd lanes take, is not what the
    /// registers hold there: refused, not the loaded sampler.
    #[test]
    fn a_descriptor_rewritten_in_an_arm_other_lanes_run_is_not_the_skipped_ones() {
        let words = rewritten_in_an_arm();
        assert_eq!(refusal(&words, false), "a descriptor changed after loading");
        // Without the rewrite, the join takes the load on both edges.
        let mut kept = words;
        kept[19] = 0xBF80_0000; // s_nop 0
        let r = waterfall(&kept, false).unwrap();
        let sampler = r.descriptors.iter().find(|d| d.role == Role::Sampler);
        assert!(
            matches!(sampler.unwrap().source, Source::Loaded { first: 0, .. }),
            "{:?}",
            r.descriptors
        );
    }

    /// The same arm with its edges split (the review of meshdata's landing,
    /// P1): the skip goes to a block of its own (an `s_nop`) that falls
    /// into the join, and the arm ends with a jump to the join. Both edges
    /// into the join are jumps, and the odd lanes come by the skip's block
    /// alone; but a wave gets there only when no even lane runs the arm,
    /// and one with an even lane runs it and jumps past that block. So the
    /// odd lanes of a mixed wave read the rewritten s8: refused as with the
    /// direct edge. Without the rewrite, the load still.
    #[test]
    fn a_descriptor_rewritten_in_an_arm_is_not_what_a_split_skip_brings() {
        let mut words = rewritten_in_an_arm();
        assert_eq!(words[18..20], [0xBF88_0001, 0xBE88_0309]);
        words.splice(
            18..20,
            [
                0xBF88_0002, // s_cbranch_execz -> 0x54
                0xBE88_0309, // s_mov_b32 s8, s9
                0xBF82_0001, // s_branch -> 0x58
                0xBF80_0000, // 0x54: s_nop 0
            ],
        );
        // 0x58: s_or_saveexec_b32 s20, s20, the join.
        assert_eq!(words[22], 0xBE94_3D14);
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        let join = block_at(&words, 0x58);
        assert_eq!(p.blocks[join].preds.len(), 2);
        let st = lanes(&p).unwrap();
        assert!(
            p.blocks[join]
                .preds
                .iter()
                .all(|&q| matches!(st.exits[q], LaneExit::Jump(_))),
            "both edges into the join are jumps"
        );
        assert_eq!(refusal(&words, false), "a descriptor changed after loading");
        let mut kept = words;
        kept[19] = 0xBF80_0000; // s_nop 0
        let r = waterfall(&kept, false).unwrap();
        let sampler = r.descriptors.iter().find(|d| d.role == Role::Sampler);
        assert!(
            matches!(sampler.unwrap().source, Source::Loaded { first: 0, .. }),
            "{:?}",
            r.descriptors
        );
    }

    /// The image descriptor of such an arm is its load when the load's
    /// address comes from the entry (the table's own SGPRs). Moved by an
    /// addition in the arm, the base exists only where the arm ran, so the
    /// load cannot be named at the access: refused still.
    #[test]
    fn an_image_loaded_under_the_starting_exec_is_its_load() {
        let r = image_loaded_in_an_arm(&S_LOAD_IMAGE).unwrap();
        let images: Vec<_> = r
            .descriptors
            .iter()
            .filter(|d| d.role == Role::Resource)
            .collect();
        assert_eq!(images.len(), 1, "{:?}", r.descriptors);
        let Source::Loaded { load, first: 0 } = images[0].source else {
            panic!("{:?}", images[0].source)
        };
        let address = r.scalar.iter().find(|s| s.0 == load).unwrap().1;
        assert_eq!(address, at(0, 0));
        let moved = [0x800C_C00C, S_LOAD_IMAGE[0], S_LOAD_IMAGE[1]]; // s_add_u32 s12, s12, 64
        match image_loaded_in_an_arm(&moved) {
            Err(ShaderError::Unsupported { what, .. }) => {
                assert_eq!(what, "a descriptor neither loaded nor built");
            }
            other => panic!("not refused: {other:?}"),
        }
    }

    #[test]
    fn descriptors_selected_by_scc_are_traced_per_outcome() {
        // Two buffer descriptors loaded from the table at s9 and one
        // picked with s_cselect_b64 per half, as the compiler makes a
        // conditional choice of resources without a branch.
        let r = track_words(
            &[
                0xBE8C_0309, // s_mov_b32 s12, s9
                0xBE8D_0380, // s_mov_b32 s13, 0
                0xF408_0006,
                0xFA00_0000, // s_load_dwordx4 s[0:3], s[12:13], null
                0xF408_0106,
                0xFA00_0010, // s_load_dwordx4 s[4:7], s[12:13], 0x10
                0xBF06_800E, // s_cmp_eq_u32 s14, 0
                0x8588_0400, // s_cselect_b64 s[8:9], s[0:1], s[4:5]
                0x858A_0602, // s_cselect_b64 s[10:11], s[2:3], s[6:7]
                0xF420_0404,
                0xFA00_0000, // s_buffer_load_dword s16, s[8:11], null
                S_ENDPGM,
            ],
            &[param(Table, 9, 1)],
            HwStage::Ps,
        );
        let Source::Select { args, .. } = &r.descriptors[0].source else {
            panic!("{:?}", r.descriptors)
        };
        // SCC set picks the first operand.
        assert_eq!(
            args[..],
            [
                Source::Loaded {
                    load: (0, 2),
                    first: 0
                },
                Source::Loaded {
                    load: (0, 3),
                    first: 0
                },
            ]
        );
        assert_eq!(r.scalar, [((0, 2), at(0, 0)), ((0, 3), at(0, 0x10))]);
    }

    #[test]
    fn a_selection_that_mixes_two_descriptors_fails() {
        // The high half picked from the first load whatever SCC is: when
        // clear, the descriptor is half of one load and half of the other.
        let code: Vec<u8> = [
            0xBE8C_0309, // s_mov_b32 s12, s9
            0xBE8D_0380, // s_mov_b32 s13, 0
            0xF408_0006,
            0xFA00_0000, // s_load_dwordx4 s[0:3], s[12:13], null
            0xF408_0106,
            0xFA00_0010, // s_load_dwordx4 s[4:7], s[12:13], 0x10
            0xBF06_800E, // s_cmp_eq_u32 s14, 0
            0x8588_0400, // s_cselect_b64 s[8:9], s[0:1], s[4:5]
            0x858A_0202, // s_cselect_b64 s[10:11], s[2:3], s[2:3]
            0xF420_0404,
            0xFA00_0000, // s_buffer_load_dword s16, s[8:11], null
            S_ENDPGM,
        ]
        .iter()
        .flat_map(|w: &u32| w.to_le_bytes())
        .collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        let err = track(&p, &[param(Table, 9, 1)], HwStage::Ps, lanes(&p).as_ref()).unwrap_err();
        assert_eq!(
            err,
            ShaderError::Unsupported {
                offset: 36,
                what: "a descriptor changed after loading"
            }
        );
    }

    #[test]
    fn untraceable_addresses_fail_with_a_reason() {
        let code: Vec<u8> = [0xF408_0001, 0xFA00_0000, S_ENDPGM] // s_load_dwordx4 s[0:3], s[2:3]
            .iter()
            .flat_map(|w: &u32| w.to_le_bytes())
            .collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        let err = track(&p, &[param(Table, 9, 1)], HwStage::Ps, lanes(&p).as_ref()).unwrap_err();
        assert_eq!(
            err,
            ShaderError::Unsupported {
                offset: 0,
                what: "an address in an SGPR that is no root parameter"
            }
        );
    }

    /// A scalar load of a table in the code, at an address a VGPR sum builds
    /// from `s_getpc_b64` and a uniform index, read back to SGPRs with
    /// `v_readfirstlane_b32` (the console compiler's form). Assembled with
    /// llvm-mc for gfx1030, wave32.
    const CODE_TABLE_FIRST_LANE: &[u32] = &[
        0xBE80_1F00, // s_getpc_b64 s[0:1]
        0x8000_FF00,
        0x0000_0DDC, // s_add_u32 s0, s0, 0xddc
        0x8201_8001, // s_addc_u32 s1, s1, 0
        0x7E00_0283, // v_mov_b32 v0, 3
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0xD70F_6A01,
        0x0000_0101, // v_add_co_u32 v1, vcc_lo, v1, s0
        0xD528_6A02,
        0x01A9_0001, // v_add_co_ci_u32 v2, vcc_lo, s1, 0, vcc_lo
        0x7E04_0501, // v_readfirstlane_b32 s2, v1
        0x7E06_0502, // v_readfirstlane_b32 s3, v2
        0xF400_0181,
        0xFA00_0000, // s_load_dword s6, s[2:3], null
        S_ENDPGM,
    ];

    #[test]
    fn a_first_lane_read_of_a_uniform_code_table_address_is_traced() {
        let r = track_words(CODE_TABLE_FIRST_LANE, &[], HwStage::Cs);
        let (_, a) = r
            .scalar
            .iter()
            .find(|s| s.1.param == CODE)
            .expect("the code table load is traced");
        // s_getpc_b64 is the first instruction: its next one is at 4.
        assert_eq!(a.offset, 4 + 0xDDC);
        assert!(a.dynamic.is_some(), "the index stays a computed term");
    }

    /// Same, but the index is each lane's own input (v0 is never set): the
    /// console reads the first lane's entry for every lane, so the address
    /// is the first lane's read itself, not each lane's index (the fork's
    /// `first_lane` arm takes the lane's index; see `code_lane`).
    #[test]
    fn a_first_lane_read_of_a_per_lane_code_table_address_is_the_first_lanes() {
        let words = [
            0xBE80_1F00, // s_getpc_b64 s[0:1]
            0x8000_FF00,
            0x0000_0DDC, // s_add_u32 s0, s0, 0xddc
            0x8201_8001, // s_addc_u32 s1, s1, 0
            0x3402_0082, // v_lshlrev_b32 v1, 2, v0
            0xD70F_6A01,
            0x0000_0101, // v_add_co_u32 v1, vcc_lo, v1, s0
            0xD528_6A02,
            0x01A9_0001, // v_add_co_ci_u32 v2, vcc_lo, s1, 0, vcc_lo
            0x7E04_0501, // v_readfirstlane_b32 s2, v1
            0x7E06_0502, // v_readfirstlane_b32 s3, v2
            0xF400_0181,
            0xFA00_0000, // s_load_dword s6, s[2:3], null
            S_ENDPGM,
        ];
        assert!(read_from_the_first_lane(&words));
    }

    /// A table in the code read in a loop whose counter (v7) the exit test
    /// compares with `limit`: the console compiler's form for a lookup table
    /// walked at a uniform index. Assembled with llvm-mc for gfx1030, wave32.
    fn code_table_loop(compare: u32) -> Vec<u32> {
        vec![
            0xBE80_1F00, // s_getpc_b64 s[0:1]
            0x8000_FF00,
            0x0000_0DDC, // s_add_u32 s0, s0, 0xddc
            0x8201_8001, // s_addc_u32 s1, s1, 0
            0x7E0E_0280, // v_mov_b32 v7, 0
            0x3402_0E82, // loop: v_lshlrev_b32 v1, 2, v7
            0xD70F_6A01,
            0x0000_0101, // v_add_co_u32 v1, vcc_lo, v1, s0
            0xD528_6A02,
            0x01A9_0001, // v_add_co_ci_u32 v2, vcc_lo, s1, 0, vcc_lo
            0x7E04_0501, // v_readfirstlane_b32 s2, v1
            0x7E06_0502, // v_readfirstlane_b32 s3, v2
            0xF400_0181,
            0xFA00_0000, // s_load_dword s6, s[2:3], null
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0x4A0E_0E81, // v_add_nc_u32 v7, 1, v7
            compare,
            0xBF87_FFF3, // s_cbranch_vccnz loop
            S_ENDPGM,
        ]
    }

    #[test]
    fn a_loop_counter_with_a_uniform_exit_indexes_a_code_table() {
        // v_cmp_gt_i32 vcc_lo, 5, v7
        let r = track_words(&code_table_loop(0x7D08_0E85), &[], HwStage::Cs);
        let (_, a) = r
            .scalar
            .iter()
            .find(|s| s.1.param == CODE)
            .expect("the code table load is traced");
        assert_eq!(a.offset, 4 + 0xDDC);
        assert!(a.dynamic.is_some(), "the counter stays a computed term");
    }

    #[test]
    fn a_loop_counter_with_a_per_lane_exit_is_read_from_the_first_lane() {
        // v_cmp_gt_u32 vcc_lo, v0, v7: lanes leave at their own count, so
        // the counter is per lane and the address the first lane's read.
        assert!(read_from_the_first_lane(&code_table_loop(0x7D88_0F00)));
    }

    const PER_LANE_FIRST_LANE: &str = "an address read from the first lane of a per-lane value";

    /// A table in the code at `s_getpc_b64` + 0xddc, read at index v7 through
    /// first-lane reads of the VGPR address (as [`CODE_TABLE_FIRST_LANE`]),
    /// after `setup`, which sets v7. Wave32, hand-assembled for gfx1030 (the
    /// encodings of the tests above; opcodes from `gfx10/opcodes.rs`). The
    /// `s_load_dword` is at byte `4 * (11 + setup.len())`.
    fn code_table_at_v7(setup: &[u32]) -> Vec<u32> {
        let mut w = vec![
            0xBE80_1F00, // s_getpc_b64 s[0:1]
            0x8000_FF00,
            0x0000_0DDC, // s_add_u32 s0, s0, 0xddc
            0x8201_8001, // s_addc_u32 s1, s1, 0
        ];
        w.extend(setup);
        w.extend([
            0x3402_0E82, // v_lshlrev_b32 v1, 2, v7
            0xD70F_6A01,
            0x0000_0101, // v_add_co_u32 v1, vcc_lo, v1, s0
            0xD528_6A02,
            0x01A9_0001, // v_add_co_ci_u32 v2, vcc_lo, s1, 0, vcc_lo
            0x7E04_0501, // v_readfirstlane_b32 s2, v1
            0x7E06_0502, // v_readfirstlane_b32 s3, v2
            0xF400_0181,
            0xFA00_0000, // s_load_dword s6, s[2:3], null
            S_ENDPGM,
        ]);
        w
    }

    /// The tracing of `words` (wave32 compute, no layout), which must
    /// structure: the rules under test need the per-lane form.
    fn code_table_trace(words: &[u32]) -> Result<Resources, ShaderError> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        let st = lanes(&p).expect("the program structures");
        track(&p, &[], HwStage::Cs, Some(&st))
    }

    fn per_lane_refusal(offset: usize) -> ShaderError {
        ShaderError::Unsupported {
            offset,
            what: PER_LANE_FIRST_LANE,
        }
    }

    /// Whether the code table address of `words` ([`code_table_trace`]) is
    /// read from the first lane: its computed term is the
    /// `v_readfirstlane_b32` result, which counts the table's base (offset
    /// 0), as for an index the lanes may hold differently (`code_lane`).
    /// The tests of the uniformity rules below see each rule through it:
    /// an index proven uniform stays the computed term instead, at the
    /// table's offset, and the lanes compute it from its operands.
    fn read_from_the_first_lane(words: &[u32]) -> bool {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        let st = lanes(&p).expect("the program structures");
        let r = track(&p, &[], HwStage::Cs, Some(&st)).unwrap();
        let (_, a) = r.scalar.iter().find(|s| s.1.param == CODE).unwrap();
        let first_lane = |v: Value| {
            p.blocks.iter().flat_map(|b| &b.insts).any(|i| {
                i.name() == "v_readfirstlane_b32" && i.writes.first().is_some_and(|w| w.1 == v)
            })
        };
        a.offset == 0 && a.dynamic.is_some_and(first_lane)
    }

    /// Where a guard narrows the first-lane read itself (here `v_cmpx`, no
    /// branch), the host's first lane may be one the console's EXEC leaves
    /// out: a per-lane index is refused there, as before `code_lane`.
    #[test]
    fn a_first_lane_read_under_a_guard_of_a_per_lane_address_is_refused() {
        let words = [
            0xBE80_1F00, // s_getpc_b64 s[0:1]
            0x8000_FF00,
            0x0000_0DDC, // s_add_u32 s0, s0, 0xddc
            0x8201_8001, // s_addc_u32 s1, s1, 0
            0x3402_0082, // v_lshlrev_b32 v1, 2, v0
            0xD70F_6A01,
            0x0000_0101, // v_add_co_u32 v1, vcc_lo, v1, s0
            0xD528_6A02,
            0x01A9_0001, // v_add_co_ci_u32 v2, vcc_lo, s1, 0, vcc_lo
            0x7DA8_0085, // v_cmpx_gt_u32 exec_lo, 5, v0
            0x7E04_0501, // v_readfirstlane_b32 s2, v1
            0x7E06_0502, // v_readfirstlane_b32 s3, v2
            0xF400_0181,
            0xFA00_0000, // s_load_dword s6, s[2:3], null
            S_ENDPGM,
        ];
        let text = decoded(&words);
        assert_eq!(text[6], "v_cmpx_gt_u32 exec_lo, 5, v0", "{text:#?}");
        // the s_load_dword, at byte 48
        assert_eq!(code_table_trace(&words).unwrap_err(), per_lane_refusal(48));
    }

    /// The text of the wave32 program of `words`.
    fn decoded(words: &[u32]) -> Vec<String> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        crate::gfx10::decode(&code)
            .unwrap()
            .iter()
            .map(|i| i.text(false))
            .collect()
    }

    /// Ported from fxceless's VIctory fork (commit 578d459,
    /// `two_paths_into_one_code_table_load`): two paths add an index to one
    /// table in the code and meet before one load (BTP 6's hair strands).
    /// The join is the computed term, which counts the table's base
    /// (offset 0): the emitter gives `s_getpc_b64` its offset in the code.
    #[test]
    fn two_paths_into_one_code_table_load() {
        let words = [
            0xBE82_1F00, // 0x00: s_getpc_b64 s[2:3]
            0x8002_FF02,
            0x0000_0040, // s_add_u32 s2, s2, 0x40
            0x8203_8003, // s_addc_u32 s3, s3, 0
            0xBF06_8001, // 0x10: s_cmp_eq_u32 s1, 0
            0xBF85_0005, // 0x14: s_cbranch_scc1 -> 0x2c
            0xD70F_0457,
            0x0000_054A, // 0x18: v_add_co_u32 v87, s4, v74, s2
            0xD528_0458,
            0x0010_0680, // v_add_co_ci_u32 v88, s4, 0, s3, s4
            0xBF82_0004, // 0x28: s_branch -> 0x3c
            0xD70F_0457,
            0x0000_054B, // 0x2c: v_add_co_u32 v87, s4, v75, s2
            0xD528_0458,
            0x0010_0680, // v_add_co_ci_u32 v88, s4, 0, s3, s4
            0xDC34_8000,
            0x577D_0057, // 0x3c: global_load_dwordx2 v[87:88], v[87:88], off
            S_ENDPGM,
        ];
        let text = decoded(&words);
        assert_eq!(text[4], "s_cbranch_scc1 label_002C", "{text:#?}");
        assert_eq!(text[7], "s_branch label_003C", "{text:#?}");
        let r = track_words(&words, &[], HwStage::Cs);
        assert_eq!(r.scalar.len(), 1);
        let a = r.scalar[0].1;
        assert_eq!((a.param, a.offset), (CODE, 0));
        assert!(a.dynamic.is_some(), "the join is the computed term");
    }

    /// The fog's form: each path takes a table of its own (two
    /// `s_getpc_b64`, two offsets) before one load. The join counts each
    /// path's base; its high half is a join of the two tables' high halves.
    /// With one path's address not in the code (an SGPR input that is no
    /// root parameter), the join is no table address and is refused as
    /// before.
    #[test]
    fn two_tables_in_the_code_joined_before_one_load() {
        let words = |other_base: u32| -> Vec<u32> {
            vec![
                0xBF06_8001, // 0x00: s_cmp_eq_u32 s1, 0
                0xBF85_0009, // 0x04: s_cbranch_scc1 -> 0x2c
                0xBE82_1F00, // 0x08: s_getpc_b64 s[2:3]
                0x8002_FF02,
                0x0000_0040, // s_add_u32 s2, s2, 0x40
                0x8203_8003, // s_addc_u32 s3, s3, 0
                0xD70F_0457,
                0x0000_054A, // 0x18: v_add_co_u32 v87, s4, v74, s2
                0xD528_0458,
                0x0010_0680, // v_add_co_ci_u32 v88, s4, 0, s3, s4
                0xBF82_0008, // 0x28: s_branch -> 0x4c
                0xBE82_1F00, // 0x2c: s_getpc_b64 s[2:3]
                0x8002_FF02,
                0x0000_0080, // s_add_u32 s2, s2, 0x80
                0x8203_8003, // s_addc_u32 s3, s3, 0
                0xD70F_0457,
                other_base, // 0x3c: v_add_co_u32 v87, s4, v75, s2 (or s5)
                0xD528_0458,
                0x0010_0680, // v_add_co_ci_u32 v88, s4, 0, s3, s4
                0xDC34_8000,
                0x577D_0057, // 0x4c: global_load_dwordx2 v[87:88], v[87:88], off
                S_ENDPGM,
            ]
        };
        let text = decoded(&words(0x0000_054B));
        assert_eq!(text[1], "s_cbranch_scc1 label_002C", "{text:#?}");
        assert_eq!(text[7], "s_branch label_004C", "{text:#?}");
        assert_eq!(text[11], "v_add_co_u32 v87, s4, v75, s2", "{text:#?}");
        let r = track_words(&words(0x0000_054B), &[], HwStage::Cs);
        assert_eq!(r.scalar.len(), 1);
        let a = r.scalar[0].1;
        assert_eq!((a.param, a.offset), (CODE, 0));
        assert!(a.dynamic.is_some(), "the join is the computed term");
        // v_add_co_u32 v87, s4, v75, s5: s5 is no table.
        let text = decoded(&words(0x0000_0B4B));
        assert_eq!(text[11], "v_add_co_u32 v87, s4, v75, s5", "{text:#?}");
        let code: Vec<u8> = words(0x0000_0B4B)
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        let err = track(&p, &[], HwStage::Cs, lanes(&p).as_ref()).unwrap_err();
        assert_eq!(
            err,
            ShaderError::Unsupported {
                // the global load, at byte 76
                offset: 76,
                what: "an address from a phi"
            }
        );
    }

    /// Ported from fxceless's VIctory fork (commit 578d459,
    /// `a_code_table_address_taken_from_the_first_lane`, BTP 6's
    /// average-luminance downsample, whose output the exposure reads): an
    /// address into a table in the code computed per lane and read from the
    /// first lane. The fork takes each lane's index as the computed term;
    /// the console reads the first lane's entry for every lane, so here the
    /// computed term is the first lane's read itself.
    #[test]
    fn a_code_table_address_taken_from_the_first_lane() {
        let words = [
            0xBE82_1F00, // 0x00: s_getpc_b64 s[2:3]
            0xD70F_0457,
            0x0000_054A, // v_add_co_u32 v87, s4, v74, s2
            0xD528_0458,
            0x0010_0680, // v_add_co_ci_u32 v88, s4, 0, s3, s4
            0x7E0C_0557, // v_readfirstlane_b32 s6, v87
            0x7E0E_0558, // v_readfirstlane_b32 s7, v88
            0xF400_0203,
            0xFA00_0000, // s_load_dword s8, s[6:7], 0x0
            S_ENDPGM,
        ];
        assert!(read_from_the_first_lane(&words));
    }

    #[test]
    fn a_code_table_index_every_lane_sets_alike_is_traced() {
        // v_mov_b32 v7, 3
        let r = code_table_trace(&code_table_at_v7(&[0x7E0E_0283])).unwrap();
        let (_, a) = r.scalar.iter().find(|s| s.1.param == CODE).unwrap();
        assert_eq!(a.offset, 4 + 0xDDC);
        assert!(a.dynamic.is_some());
    }

    /// `s_cbranch_execz` is a branch on each lane's own EXEC bit in the
    /// per-lane form: the lanes the `s_and_saveexec` left out skip the write
    /// and reach the join with the old v7. (VIctory#1231 took the branch as
    /// uniform. The write runs under a narrowed EXEC in the console flow,
    /// which makes it per lane too: the execz rule alone is pinned by
    /// `a_scalar_set_on_one_side_of_an_execz_skip_is_per_lane`.)
    #[test]
    fn a_value_joined_after_an_execz_skip_is_per_lane() {
        let words = code_table_at_v7(&[
            0x7E0E_0280, // v_mov_b32 v7, 0
            0x7D88_0085, // v_cmp_gt_u32 vcc_lo, 5, v0: lanes 0 to 4
            0xBE88_3C6A, // s_and_saveexec_b32 s8, vcc_lo
            0xBF88_0001, // s_cbranch_execz over the next instruction
            0x7E0E_0283, // v_mov_b32 v7, 3
            0x887E_087E, // s_or_b32 exec_lo, exec_lo, s8 (the join)
        ]);
        assert!(read_from_the_first_lane(&words));
    }

    /// The same with a scalar: s9 is 3 in the lanes the `s_and_saveexec`
    /// let in and 0 in the others, which skip the block on their own EXEC
    /// bit. In the console flow the wave skips the block or not as one and
    /// s9 is uniform, so only the per-lane rule for execz makes the address
    /// the first lane's read.
    /// (VIctory#1231 took the branch as uniform and traced it.)
    #[test]
    fn a_scalar_set_on_one_side_of_an_execz_skip_is_per_lane() {
        let words = code_table_at_v7(&[
            0xBE89_0380, // s_mov_b32 s9, 0
            0x7D88_0085, // v_cmp_gt_u32 vcc_lo, 5, v0: lanes 0 to 4
            0xBE88_3C6A, // s_and_saveexec_b32 s8, vcc_lo
            0xBF88_0001, // s_cbranch_execz over the next instruction
            0xBE89_0383, // s_mov_b32 s9, 3
            0x887E_087E, // s_or_b32 exec_lo, exec_lo, s8 (the join)
            0x7E0E_0209, // v_mov_b32 v7, s9
        ]);
        assert!(read_from_the_first_lane(&words));
    }

    /// A write under EXEC narrowed by a per-lane compare, without a branch:
    /// lanes 0 to 4 hold 3, the others 0. (VIctory#1231 took it as uniform
    /// because its old value is. The console flow makes it per lane too; the
    /// per-lane rule alone is pinned by `uniform::tests`.)
    #[test]
    fn a_guarded_write_of_a_uniform_value_is_per_lane() {
        let words = code_table_at_v7(&[
            0x7E0E_0280, // v_mov_b32 v7, 0
            0x7D88_0085, // v_cmp_gt_u32 vcc_lo, 5, v0
            0xBE88_3C6A, // s_and_saveexec_b32 s8, vcc_lo
            0x7E0E_0283, // v_mov_b32 v7, 3
            0x887E_087E, // s_or_b32 exec_lo, exec_lo, s8
        ]);
        assert!(read_from_the_first_lane(&words));
    }

    /// The review's program: s5 decides whether every lane runs the write
    /// of 12 to v7 or none does, so v7 is the same in every lane, but the
    /// address is read from the operands of the instructions that compute
    /// it, which is 12 only where the lanes ran it: for s5 != 0 the console
    /// reads entry 0. b8714dd took the write as uniform (its condition and
    /// its old value are) and traced code + 4 + 0xddc + 12.
    #[test]
    fn a_write_no_lane_runs_is_not_read_from_its_operands() {
        let words = [
            0xBE80_1F00, // s_getpc_b64 s[0:1]
            0x8000_FF00,
            0x0000_0DDC, // s_add_u32 s0, s0, 0xddc
            0x8201_8001, // s_addc_u32 s1, s1, 0
            0x7E0E_0280, // v_mov_b32 v7, 0
            0x7E12_0205, // v_mov_b32 v9, s5
            0x7D84_1280, // v_cmp_eq_u32 vcc_lo, 0, v9
            0xBE88_3C6A, // s_and_saveexec_b32 s8, vcc_lo
            0x7E0E_028C, // v_mov_b32 v7, 12
            0x887E_087E, // s_or_b32 exec_lo, exec_lo, s8
            0xD70F_6A01,
            0x0000_0107, // v_add_co_u32 v1, vcc_lo, v7, s0
            0xD528_6A02,
            0x01A9_0001, // v_add_co_ci_u32 v2, vcc_lo, s1, 0, vcc_lo
            0x7E04_0501, // v_readfirstlane_b32 s2, v1
            0x7E06_0502, // v_readfirstlane_b32 s3, v2
            0xF400_0181,
            0xFA00_0000, // s_load_dword s6, s[2:3], null
            S_ENDPGM,
        ];
        assert!(read_from_the_first_lane(&words));
    }

    /// An SDWA move of a word keeps the high half of v7, each lane's own
    /// input. b8714dd took v7 as uniform (its operand is) and traced the
    /// address, whose computed term the lanes then compute apart.
    #[test]
    fn an_sdwa_write_of_a_word_keeps_per_lane_bits() {
        let words = code_table_at_v7(&[
            0x7E10_0283, // v_mov_b32 v8, 3
            0x7E0E_02F9,
            0x0006_1408, // v_mov_b32_sdwa v7, v8 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE
        ]);
        assert!(read_from_the_first_lane(&words));
    }

    /// EXEC named as an operand is a lane mask read as a number, which the
    /// per-lane form builds with a ballot of the lanes there: no value of
    /// EXEC is uniform. b8714dd skipped every read of EXEC, this one too,
    /// and took v7 as uniform.
    #[test]
    fn an_exec_operand_is_not_uniform() {
        let words = code_table_at_v7(&[
            0x7E0E_027E, // v_mov_b32 v7, exec_lo
        ]);
        assert!(read_from_the_first_lane(&words));
    }

    /// A carry-in is read lane by lane: from a mask copied out of an SGPR,
    /// each lane adds its own bit of s5. (VIctory#1225 took the carry as
    /// uniform because the SGPR value is.)
    #[test]
    fn a_carry_in_from_an_sgpr_mask_is_per_lane() {
        let words = code_table_at_v7(&[
            0x7E0E_0280, // v_mov_b32 v7, 0
            0xBEEA_0305, // s_mov_b32 vcc_lo, s5
            0xD528_6A07,
            0x01AA_0E80, // v_add_co_ci_u32 v7, vcc_lo, 0, v7, vcc_lo
        ]);
        assert!(read_from_the_first_lane(&words));
    }

    /// The other half of that rule: a branch on VCC copied from an SGPR.
    /// The loop of `code_table_loop`, with the exit test VCC = s5 shifted
    /// right once per pass: per lane, each lane leaves when its own bit is
    /// clear, so the counter v7 differs between lanes. (VIctory#1231 took
    /// the branch as uniform because the SGPR value is.)
    #[test]
    fn a_loop_left_on_the_lane_bits_of_an_sgpr_is_per_lane() {
        let words = [
            0xBE80_1F00, // s_getpc_b64 s[0:1]
            0x8000_FF00,
            0x0000_0DDC, // s_add_u32 s0, s0, 0xddc
            0x8201_8001, // s_addc_u32 s1, s1, 0
            0x7E0E_0280, // v_mov_b32 v7, 0
            0x3402_0E82, // loop: v_lshlrev_b32 v1, 2, v7
            0xD70F_6A01,
            0x0000_0101, // v_add_co_u32 v1, vcc_lo, v1, s0
            0xD528_6A02,
            0x01A9_0001, // v_add_co_ci_u32 v2, vcc_lo, s1, 0, vcc_lo
            0x7E04_0501, // v_readfirstlane_b32 s2, v1
            0x7E06_0502, // v_readfirstlane_b32 s3, v2
            0xF400_0181,
            0xFA00_0000, // s_load_dword s6, s[2:3], null
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0x4A0E_0E81, // v_add_nc_u32 v7, 1, v7
            0x9005_8105, // s_lshr_b32 s5, s5, 1
            0xBEEA_0305, // s_mov_b32 vcc_lo, s5
            0xBF87_FFF2, // s_cbranch_vccnz loop
            S_ENDPGM,
        ];
        assert!(read_from_the_first_lane(&words));
    }

    /// A DPP move reads another lane's v8; the first lane of each row has
    /// none and keeps its old v7 (5, where the others get 3).
    #[test]
    fn a_dpp_move_of_a_uniform_value_is_per_lane() {
        let words = code_table_at_v7(&[
            0x7E0E_0285, // v_mov_b32 v7, 5
            0x7E10_0283, // v_mov_b32 v8, 3
            0x7E0E_02FA,
            0xFF01_1108, // v_mov_b32_dpp v7, v8 row_shr:1 row_mask:0xf bank_mask:0xf
        ]);
        assert!(read_from_the_first_lane(&words));
    }

    /// A code table read in a block only lanes 0 to 4 enter, at index v7:
    /// `set` sets v7 there, then the address is built and read from the
    /// first lane, and s6 is read after the join.
    fn narrowed_code_table(set: u32) -> [u32; 20] {
        [
            0xBE80_1F00, // s_getpc_b64 s[0:1]
            0x8000_FF00,
            0x0000_0DDC, // s_add_u32 s0, s0, 0xddc
            0x8201_8001, // s_addc_u32 s1, s1, 0
            0x7D88_0085, // v_cmp_gt_u32 vcc_lo, 5, v0
            0xBE88_3C6A, // s_and_saveexec_b32 s8, vcc_lo
            0xBF88_000A, // s_cbranch_execz end
            set,
            0x3402_0E82, // v_lshlrev_b32 v1, 2, v7
            0xD70F_6A01,
            0x0000_0101, // v_add_co_u32 v1, vcc_lo, v1, s0
            0xD528_6A02,
            0x01A9_0001, // v_add_co_ci_u32 v2, vcc_lo, s1, 0, vcc_lo
            0x7E04_0501, // v_readfirstlane_b32 s2, v1
            0x7E06_0502, // v_readfirstlane_b32 s3, v2
            0xF400_0181,
            0xFA00_0000, // s_load_dword s6, s[2:3], null
            0x887E_087E, // end: s_or_b32 exec_lo, exec_lo, s8
            0x7E12_0206, // v_mov_b32 v9, s6
            S_ENDPGM,
        ]
    }

    /// Per lane, only lanes 0 to 4 enter the block and all of them set v7
    /// alike (to s9). In the console flow every lane runs it, and lanes 5 to
    /// 31 keep their own v7, load other dwords into s6 and read them after
    /// the join: the claim must hold in both lowerings, so the address is
    /// the first lane's read, which both lowerings compute.
    #[test]
    fn a_code_table_index_set_under_a_narrowed_exec_is_per_lane_in_the_console_flow() {
        // v_mov_b32 v7, s9
        let words = narrowed_code_table(0x7E0E_0209);
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        assert!(crate::structure::structure_console(&p).is_ok());
        assert!(read_from_the_first_lane(&words));
    }

    /// The same with a constant index: the first active lane computed the
    /// address from constants in the read's block, under the EXEC the read
    /// runs with, so it is entry 3 whatever the other lanes hold, and the
    /// load needs no lane's VGPR in either lowering (the console compiler's
    /// form in the exposure and irradiance passes).
    #[test]
    fn a_constant_code_table_index_under_a_narrowed_exec_is_folded() {
        // v_mov_b32 v7, 3
        let r = code_table_trace(&narrowed_code_table(0x7E0E_0283)).unwrap();
        let (_, a) = r.scalar.iter().find(|s| s.1.param == CODE).unwrap();
        assert_eq!(a.offset, 4 + 0xDDC + 3 * 4);
        assert_eq!(a.dynamic, None);
    }

    /// A waterfall loop over an image descriptor each lane loads from the
    /// table at s0 (wave32, assembled with llvm-mc for gfx1030): lane i
    /// reads entry i with two `global_load_dwordx4`, the loop reads the
    /// first active lane's descriptor into s[12:19], samples with the lanes
    /// whose descriptor it is, and repeats for the others. The sampler comes
    /// from the table at s1; the result goes to LDS so the sample is needed.
    pub(crate) const WATERFALL_FETCHED_WAVE32: &[u32] = &[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0x3402_0085, // v_lshlrev_b32 v1, 5, v0
        0xDC38_8000,
        0x0404_0001, // global_load_dwordx4 v[4:7], v1, s[4:5]
        0xDC38_8010,
        0x0804_0001, // global_load_dwordx4 v[8:11], v1, s[4:5] offset:16
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0x7E04_0D00, // v_cvt_f32_u32 v2, v0
        0x7E06_0302, // v_mov_b32 v3, v2
        0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
        0xBE94_037E, // s_mov_b32 s20, exec_lo
        0x7E18_0504, // 0x3c: v_readfirstlane_b32 s12, v4
        0x7E1A_0505, // v_readfirstlane_b32 s13, v5
        0x7E1C_0506, // v_readfirstlane_b32 s14, v6
        0x7E1E_0507, // v_readfirstlane_b32 s15, v7
        0x7E20_0508, // v_readfirstlane_b32 s16, v8
        0x7E22_0509, // v_readfirstlane_b32 s17, v9
        0x7E24_050A, // v_readfirstlane_b32 s18, v10
        0x7E26_050B, // v_readfirstlane_b32 s19, v11
        0xD4E2_0015,
        0x0002_080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
        0xD4E2_0016,
        0x0002_0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1010, // v_cmp_eq_u64_e64 s22, s[16:17], v[8:9]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1412, // v_cmp_eq_u64_e64 s22, s[18:19], v[10:11]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xBE95_3C15, // s_and_saveexec_b32 s21, s21
        0xBF80_0000, // s_nop 0
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
        0xBF89_FFE6, // s_cbranch_execnz -> 0x3c
        0xBEFE_0314, // s_mov_b32 exec_lo, s20
        0xD834_0000,
        0x0000_0C00, // ds_write_b32 v0, v12
        0xBF81_0000, // s_endpgm
    ];

    /// [`WATERFALL_FETCHED_WAVE32`] in wave64: 64-bit lane masks.
    pub(crate) const WATERFALL_FETCHED_WAVE64: &[u32] = &[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0x3402_0085, // v_lshlrev_b32 v1, 5, v0
        0xDC38_8000,
        0x0404_0001, // global_load_dwordx4 v[4:7], v1, s[4:5]
        0xDC38_8010,
        0x0804_0001, // global_load_dwordx4 v[8:11], v1, s[4:5] offset:16
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0x7E04_0D00, // v_cvt_f32_u32 v2, v0
        0x7E06_0302, // v_mov_b32 v3, v2
        0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
        0xBE94_047E, // s_mov_b64 s[20:21], exec
        0x7E18_0504, // 0x3c: v_readfirstlane_b32 s12, v4
        0x7E1A_0505, // v_readfirstlane_b32 s13, v5
        0x7E1C_0506, // v_readfirstlane_b32 s14, v6
        0x7E1E_0507, // v_readfirstlane_b32 s15, v7
        0x7E20_0508, // v_readfirstlane_b32 s16, v8
        0x7E22_0509, // v_readfirstlane_b32 s17, v9
        0x7E24_050A, // v_readfirstlane_b32 s18, v10
        0x7E26_050B, // v_readfirstlane_b32 s19, v11
        0xD4E2_0016,
        0x0002_080C, // v_cmp_eq_u64_e64 s[22:23], s[12:13], v[4:5]
        0xD4E2_0018,
        0x0002_0C0E, // v_cmp_eq_u64_e64 s[24:25], s[14:15], v[6:7]
        0x8796_1816, // s_and_b64 s[22:23], s[22:23], s[24:25]
        0xD4E2_0018,
        0x0002_1010, // v_cmp_eq_u64_e64 s[24:25], s[16:17], v[8:9]
        0x8796_1816, // s_and_b64 s[22:23], s[22:23], s[24:25]
        0xD4E2_0018,
        0x0002_1412, // v_cmp_eq_u64_e64 s[24:25], s[18:19], v[10:11]
        0x8796_1816, // s_and_b64 s[22:23], s[22:23], s[24:25]
        0xBE96_2416, // s_and_saveexec_b64 s[22:23], s[22:23]
        0xBF80_0000, // s_nop 0
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x89FE_167E, // s_xor_b64 exec, exec, s[22:23]
        0xBF89_FFE6, // s_cbranch_execnz -> 0x3c
        0xBEFE_0414, // s_mov_b64 exec, s[20:21]
        0xD834_0000,
        0x0000_0C00, // ds_write_b32 v0, v12
        0xBF81_0000, // s_endpgm
    ];

    /// The same waterfall over one of two image descriptors loaded from the
    /// table at s0, picked per lane by `v_cndmask_b32` (lanes below 16 take
    /// the first).
    pub(crate) const WATERFALL_PICKED_WAVE32: &[u32] = &[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF40C_0602,
        0xFA00_0000, // s_load_dwordx8 s[24:31], s[4:5], 0x0
        0xF40C_0802,
        0xFA00_0020, // s_load_dwordx8 s[32:39], s[4:5], 0x20
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0x7E04_0D00, // v_cvt_f32_u32 v2, v0
        0x7E06_0302, // v_mov_b32 v3, v2
        0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x7E08_0218, // v_mov_b32 v4, s24
        0x0208_0820, // v_cndmask_b32 v4, s32, v4, vcc_lo
        0x7E0A_0219, // v_mov_b32 v5, s25
        0x020A_0A21, // v_cndmask_b32 v5, s33, v5, vcc_lo
        0x7E0C_021A, // v_mov_b32 v6, s26
        0x020C_0C22, // v_cndmask_b32 v6, s34, v6, vcc_lo
        0x7E0E_021B, // v_mov_b32 v7, s27
        0x020E_0E23, // v_cndmask_b32 v7, s35, v7, vcc_lo
        0x7E10_021C, // v_mov_b32 v8, s28
        0x0210_1024, // v_cndmask_b32 v8, s36, v8, vcc_lo
        0x7E12_021D, // v_mov_b32 v9, s29
        0x0212_1225, // v_cndmask_b32 v9, s37, v9, vcc_lo
        0x7E14_021E, // v_mov_b32 v10, s30
        0x0214_1426, // v_cndmask_b32 v10, s38, v10, vcc_lo
        0x7E16_021F, // v_mov_b32 v11, s31
        0x0216_1627, // v_cndmask_b32 v11, s39, v11, vcc_lo
        0xBE94_037E, // s_mov_b32 s20, exec_lo
        0x7E18_0504, // 0x7c: v_readfirstlane_b32 s12, v4
        0x7E1A_0505, // v_readfirstlane_b32 s13, v5
        0x7E1C_0506, // v_readfirstlane_b32 s14, v6
        0x7E1E_0507, // v_readfirstlane_b32 s15, v7
        0x7E20_0508, // v_readfirstlane_b32 s16, v8
        0x7E22_0509, // v_readfirstlane_b32 s17, v9
        0x7E24_050A, // v_readfirstlane_b32 s18, v10
        0x7E26_050B, // v_readfirstlane_b32 s19, v11
        0xD4E2_0015,
        0x0002_080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
        0xD4E2_0016,
        0x0002_0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1010, // v_cmp_eq_u64_e64 s22, s[16:17], v[8:9]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1412, // v_cmp_eq_u64_e64 s22, s[18:19], v[10:11]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xBE95_3C15, // s_and_saveexec_b32 s21, s21
        0xBF80_0000, // s_nop 0
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
        0xBF89_FFE6, // s_cbranch_execnz -> 0x7c
        0xBEFE_0314, // s_mov_b32 exec_lo, s20
        0xD834_0000,
        0x0000_0C00, // ds_write_b32 v0, v12
        0xBF81_0000, // s_endpgm
    ];

    /// The waterfall loop of [`WATERFALL_PICKED_WAVE32`] over one of two
    /// image descriptors loaded from the table at s0, chosen by control flow
    /// as the compiler lowers an if/else on a per-lane condition: lanes below
    /// 16 move the first into v[4:11], the others the second, each arm
    /// skipped when no lane takes it. The sample goes to the buffer at dword
    /// 16 of the table, at lane * 4.
    pub(crate) const WATERFALL_JOINED_WAVE32: &[u32] = &[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF40C_0602,
        0xFA00_0000, // s_load_dwordx8 s[24:31], s[4:5], 0x0
        0xF40C_0802,
        0xFA00_0020, // s_load_dwordx8 s[32:39], s[4:5], 0x20
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0x7E04_0D00, // v_cvt_f32_u32 v2, v0
        0x7E06_0302, // v_mov_b32 v3, v2
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBE94_3C6A, // s_and_saveexec_b32 s20, vcc_lo
        0x8914_147E, // s_xor_b32 s20, exec_lo, s20
        0xBF88_0008, // s_cbranch_execz -> 0x70
        0x7E08_0218, // v_mov_b32 v4, s24
        0x7E0A_0219, // v_mov_b32 v5, s25
        0x7E0C_021A, // v_mov_b32 v6, s26
        0x7E0E_021B, // v_mov_b32 v7, s27
        0x7E10_021C, // v_mov_b32 v8, s28
        0x7E12_021D, // v_mov_b32 v9, s29
        0x7E14_021E, // v_mov_b32 v10, s30
        0x7E16_021F, // v_mov_b32 v11, s31
        0xBE94_3D14, // 0x70: s_or_saveexec_b32 s20, s20
        0x897E_147E, // s_xor_b32 exec_lo, exec_lo, s20
        0xBF88_0008, // s_cbranch_execz -> 0x9c
        0x7E08_0220, // v_mov_b32 v4, s32
        0x7E0A_0221, // v_mov_b32 v5, s33
        0x7E0C_0222, // v_mov_b32 v6, s34
        0x7E0E_0223, // v_mov_b32 v7, s35
        0x7E10_0224, // v_mov_b32 v8, s36
        0x7E12_0225, // v_mov_b32 v9, s37
        0x7E14_0226, // v_mov_b32 v10, s38
        0x7E16_0227, // v_mov_b32 v11, s39
        0x887E_147E, // 0x9c: s_or_b32 exec_lo, exec_lo, s20
        0xBE97_037E, // s_mov_b32 s23, exec_lo
        0x7E18_0504, // 0xa4: v_readfirstlane_b32 s12, v4
        0x7E1A_0505, // v_readfirstlane_b32 s13, v5
        0x7E1C_0506, // v_readfirstlane_b32 s14, v6
        0x7E1E_0507, // v_readfirstlane_b32 s15, v7
        0x7E20_0508, // v_readfirstlane_b32 s16, v8
        0x7E22_0509, // v_readfirstlane_b32 s17, v9
        0x7E24_050A, // v_readfirstlane_b32 s18, v10
        0x7E26_050B, // v_readfirstlane_b32 s19, v11
        0xD4E2_0015,
        0x0002_080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
        0xD4E2_0016,
        0x0002_0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1010, // v_cmp_eq_u64_e64 s22, s[16:17], v[8:9]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1412, // v_cmp_eq_u64_e64 s22, s[18:19], v[10:11]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xBE95_3C15, // s_and_saveexec_b32 s21, s21
        0xBF80_0000, // s_nop 0
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
        0xBF89_FFE6, // s_cbranch_execnz -> 0xa4
        0xBEFE_0317, // s_mov_b32 exec_lo, s23
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ];

    /// The console compiler's waterfall (BTP 6's ray-tracing
    /// resolve and others): each lane holds its entry's byte offset in the
    /// table at s0 (v7 = (lane & 1) * 32); the loop reads the first active
    /// lane's offset, narrows EXEC to the lanes with the same one
    /// (`v_cmpx_eq_u32`), loads that entry with `s_load_dwordx8` and samples,
    /// then drops those lanes (`s_andn2_wrexec_b32`). The sample goes to the
    /// buffer at dword 16 of the table, at lane * 4.
    pub(crate) const WATERFALL_INDEXED_WAVE32: &[u32] = &[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0x7E04_0D00, // v_cvt_f32_u32 v2, v0
        0x7E06_0302, // v_mov_b32 v3, v2
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0x360E_0081, // v_and_b32 v7, 1, v0
        0x340E_0E85, // v_lshlrev_b32 v7, 5, v7
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBE97_037E, // s_mov_b32 s23, exec_lo
        0xBE95_037E, // s_mov_b32 s21, exec_lo
        0x7E28_0507, // 0x40: v_readfirstlane_b32 s20, v7
        0x7DA4_0E14, // v_cmpx_eq_u32 exec_lo, s20, v7
        0xF40C_0302,
        0x2800_0000, // s_load_dwordx8 s[12:19], s[4:5], s20
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xBE95_4715, // s_andn2_wrexec_b32 s21, s21
        0xBF89_FFF6, // s_cbranch_execnz -> 0x40
        0xBEFE_0317, // s_mov_b32 exec_lo, s23
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ];

    #[test]
    fn the_console_waterfall_over_a_first_lane_offset_is_one() {
        let r = waterfall(WATERFALL_INDEXED_WAVE32, false).unwrap();
        let image = r
            .descriptors
            .iter()
            .find(|d| d.role == Role::Resource)
            .unwrap();
        let Source::Loaded { load, first: 0 } = image.source else {
            panic!("{:?}", image.source)
        };
        let (_, address) = r.scalar.iter().find(|s| s.0 == load).unwrap();
        assert!(address.dynamic.is_some(), "{address:?}");
        assert_eq!(r.waterfall.compares.len(), 1);
        assert_eq!(r.waterfall.reads.len(), 1);
        assert_eq!(r.waterfall.loads, [load]);
    }

    /// [`WATERFALL_JOINED_WAVE32`] with other sources on each path: lanes
    /// below 16 load table entry (lane & 1) with two `global_load_dwordx4`,
    /// the others pick per lane with `v_cndmask_b32` (lanes below 24 the
    /// first entry, the rest the second).
    pub(crate) const WATERFALL_JOINED_MIXED_WAVE32: &[u32] = &[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF40C_0602,
        0xFA00_0000, // s_load_dwordx8 s[24:31], s[4:5], 0x0
        0xF40C_0802,
        0xFA00_0020, // s_load_dwordx8 s[32:39], s[4:5], 0x20
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0x7E04_0D00, // v_cvt_f32_u32 v2, v0
        0x7E06_0302, // v_mov_b32 v3, v2
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0x3602_0081, // v_and_b32 v1, 1, v0
        0x3402_0285, // v_lshlrev_b32 v1, 5, v1
        0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBE94_3C6A, // s_and_saveexec_b32 s20, vcc_lo
        0x8914_147E, // s_xor_b32 s20, exec_lo, s20
        0xBF88_0004, // s_cbranch_execz -> 0x68
        0xDC38_8000,
        0x0404_0001, // global_load_dwordx4 v[4:7], v1, s[4:5]
        0xDC38_8010,
        0x0804_0001, // global_load_dwordx4 v[8:11], v1, s[4:5] offset:16
        0xBE94_3D14, // 0x68: s_or_saveexec_b32 s20, s20
        0x897E_147E, // s_xor_b32 exec_lo, exec_lo, s20
        0xBF88_0011, // s_cbranch_execz -> 0xb8
        0x7D88_0098, // v_cmp_gt_u32 vcc_lo, 24, v0
        0x7E08_0218, // v_mov_b32 v4, s24
        0x0208_0820, // v_cndmask_b32 v4, s32, v4, vcc_lo
        0x7E0A_0219, // v_mov_b32 v5, s25
        0x020A_0A21, // v_cndmask_b32 v5, s33, v5, vcc_lo
        0x7E0C_021A, // v_mov_b32 v6, s26
        0x020C_0C22, // v_cndmask_b32 v6, s34, v6, vcc_lo
        0x7E0E_021B, // v_mov_b32 v7, s27
        0x020E_0E23, // v_cndmask_b32 v7, s35, v7, vcc_lo
        0x7E10_021C, // v_mov_b32 v8, s28
        0x0210_1024, // v_cndmask_b32 v8, s36, v8, vcc_lo
        0x7E12_021D, // v_mov_b32 v9, s29
        0x0212_1225, // v_cndmask_b32 v9, s37, v9, vcc_lo
        0x7E14_021E, // v_mov_b32 v10, s30
        0x0214_1426, // v_cndmask_b32 v10, s38, v10, vcc_lo
        0x7E16_021F, // v_mov_b32 v11, s31
        0x0216_1627, // v_cndmask_b32 v11, s39, v11, vcc_lo
        0x887E_147E, // 0xb8: s_or_b32 exec_lo, exec_lo, s20
        0xBE97_037E, // s_mov_b32 s23, exec_lo
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x7E18_0504, // 0xc4: v_readfirstlane_b32 s12, v4
        0x7E1A_0505, // v_readfirstlane_b32 s13, v5
        0x7E1C_0506, // v_readfirstlane_b32 s14, v6
        0x7E1E_0507, // v_readfirstlane_b32 s15, v7
        0x7E20_0508, // v_readfirstlane_b32 s16, v8
        0x7E22_0509, // v_readfirstlane_b32 s17, v9
        0x7E24_050A, // v_readfirstlane_b32 s18, v10
        0x7E26_050B, // v_readfirstlane_b32 s19, v11
        0xD4E2_0015,
        0x0002_080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
        0xD4E2_0016,
        0x0002_0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1010, // v_cmp_eq_u64_e64 s22, s[16:17], v[8:9]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1412, // v_cmp_eq_u64_e64 s22, s[18:19], v[10:11]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xBE95_3C15, // s_and_saveexec_b32 s21, s21
        0xBF80_0000, // s_nop 0
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
        0xBF89_FFE6, // s_cbranch_execnz -> 0xc4
        0xBEFE_0317, // s_mov_b32 exec_lo, s23
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ];

    /// A waterfall loop whose descriptor changes on each pass: v[4:11] holds
    /// the first table entry on entry and the second after every pass, so
    /// the first lane's dwords are phis of the loop header.
    pub(crate) const WATERFALL_LOOPED_WAVE32: &[u32] = &[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE86_0301, // s_mov_b32 s6, s1
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF40C_0602,
        0xFA00_0000, // s_load_dwordx8 s[24:31], s[4:5], 0x0
        0xF40C_0802,
        0xFA00_0020, // s_load_dwordx8 s[32:39], s[4:5], 0x20
        0xF408_0203,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[6:7], 0x0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0x7E04_0D00, // v_cvt_f32_u32 v2, v0
        0x7E06_0302, // v_mov_b32 v3, v2
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0x7E08_0218, // v_mov_b32 v4, s24
        0x7E0A_0219, // v_mov_b32 v5, s25
        0x7E0C_021A, // v_mov_b32 v6, s26
        0x7E0E_021B, // v_mov_b32 v7, s27
        0x7E10_021C, // v_mov_b32 v8, s28
        0x7E12_021D, // v_mov_b32 v9, s29
        0x7E14_021E, // v_mov_b32 v10, s30
        0x7E16_021F, // v_mov_b32 v11, s31
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBE97_037E, // s_mov_b32 s23, exec_lo
        0x7E18_0504, // 0x64: v_readfirstlane_b32 s12, v4
        0x7E1A_0505, // v_readfirstlane_b32 s13, v5
        0x7E1C_0506, // v_readfirstlane_b32 s14, v6
        0x7E1E_0507, // v_readfirstlane_b32 s15, v7
        0x7E20_0508, // v_readfirstlane_b32 s16, v8
        0x7E22_0509, // v_readfirstlane_b32 s17, v9
        0x7E24_050A, // v_readfirstlane_b32 s18, v10
        0x7E26_050B, // v_readfirstlane_b32 s19, v11
        0xD4E2_0015,
        0x0002_080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
        0xD4E2_0016,
        0x0002_0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1010, // v_cmp_eq_u64_e64 s22, s[16:17], v[8:9]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xD4E2_0016,
        0x0002_1412, // v_cmp_eq_u64_e64 s22, s[18:19], v[10:11]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xBE95_3C15, // s_and_saveexec_b32 s21, s21
        0xBF80_0000, // s_nop 0
        0xF09C_0F08,
        0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
        0x7E08_0220, // v_mov_b32 v4, s32
        0x7E0A_0221, // v_mov_b32 v5, s33
        0x7E0C_0222, // v_mov_b32 v6, s34
        0x7E0E_0223, // v_mov_b32 v7, s35
        0x7E10_0224, // v_mov_b32 v8, s36
        0x7E12_0225, // v_mov_b32 v9, s37
        0x7E14_0226, // v_mov_b32 v10, s38
        0x7E16_0227, // v_mov_b32 v11, s39
        0xBF89_FFDE, // s_cbranch_execnz -> 0x64
        0xBEFE_0317, // s_mov_b32 exec_lo, s23
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ];

    /// A waterfall loop over one of two buffer descriptors (four dwords)
    /// loaded from the table at s0, chosen by nested if/else: lanes below 16
    /// move the first into v[4:7], lanes from 16 to 23 the second, and the
    /// others keep what v[4:7] held. Only lanes below 24 (the same mask)
    /// run the loop, which loads dword `lane` of the chosen buffer, and
    /// store it to the buffer at dword 16 of the table.
    pub(crate) const WATERFALL_JOINED_BUFFER_WAVE32: &[u32] = &[
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0602,
        0xFA00_0000, // s_load_dwordx4 s[24:27], s[4:5], 0x0
        0xF408_0802,
        0xFA00_0020, // s_load_dwordx4 s[32:35], s[4:5], 0x20
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0x3420_0082, // v_lshlrev_b32 v16, 2, v0
        0xD4C4_001E,
        0x0002_0098, // v_cmp_gt_u32_e64 s30, 24, v0
        0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBE94_3C6A, // s_and_saveexec_b32 s20, vcc_lo
        0x8914_147E, // s_xor_b32 s20, exec_lo, s20
        0xBF88_0004, // s_cbranch_execz -> 0x50
        0x7E08_0218, // 0x40: v_mov_b32 v4, s24
        0x7E0A_0219, // v_mov_b32 v5, s25
        0x7E0C_021A, // v_mov_b32 v6, s26
        0x7E0E_021B, // v_mov_b32 v7, s27
        0xBE94_3D14, // 0x50: s_or_saveexec_b32 s20, s20
        0x897E_147E, // s_xor_b32 exec_lo, exec_lo, s20
        0xBF88_0007, // s_cbranch_execz -> 0x78
        0xBE9F_3C1E, // s_and_saveexec_b32 s31, s30
        0xBF88_0004, // s_cbranch_execz -> 0x74
        0x7E08_0220, // 0x64: v_mov_b32 v4, s32
        0x7E0A_0221, // v_mov_b32 v5, s33
        0x7E0C_0222, // v_mov_b32 v6, s34
        0x7E0E_0223, // v_mov_b32 v7, s35
        0x887E_1F7E, // 0x74: s_or_b32 exec_lo, exec_lo, s31
        0x887E_147E, // 0x78: s_or_b32 exec_lo, exec_lo, s20
        0xBE97_3C1E, // s_and_saveexec_b32 s23, s30
        0xBF88_0013, // s_cbranch_execz -> 0xd0
        0xBE9D_037E, // s_mov_b32 s29, exec_lo
        0x7E18_0504, // 0x88: v_readfirstlane_b32 s12, v4
        0x7E1A_0505, // v_readfirstlane_b32 s13, v5
        0x7E1C_0506, // v_readfirstlane_b32 s14, v6
        0x7E1E_0507, // v_readfirstlane_b32 s15, v7
        0xD4E2_0015,
        0x0002_080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
        0xD4E2_0016,
        0x0002_0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
        0x8715_1615, // s_and_b32 s21, s21, s22
        0xBE95_3C15, // s_and_saveexec_b32 s21, s21
        0xE030_1000,
        0x8003_0C10, // buffer_load_dword v12, v16, s[12:15], 0 offen
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
        0xBF89_FFF1, // s_cbranch_execnz -> 0x88
        0xBEFE_031D, // s_mov_b32 exec_lo, s29
        0xE070_1000,
        0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
        0x887E_177E, // 0xd0: s_or_b32 exec_lo, exec_lo, s23
        0xBF81_0000, // s_endpgm
    ];

    fn waterfall(words: &[u32], wave64: bool) -> Result<Resources, ShaderError> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, wave64, 256).unwrap();
        let layout = [param(Table, 0, 1), param(Table, 1, 1)];
        track(&p, &layout, HwStage::Cs, lanes(&p).as_ref())
    }

    #[test]
    fn descriptors_read_from_the_first_lane_of_per_lane_loads_are_traced_to_them() {
        for (words, wave64) in [
            (WATERFALL_FETCHED_WAVE32, false),
            (WATERFALL_FETCHED_WAVE64, true),
        ] {
            let r = waterfall(words, wave64).unwrap();
            let image = r
                .descriptors
                .iter()
                .find(|d| d.role == Role::Resource)
                .unwrap();
            let Source::Fetched { load, first: 0 } = image.source else {
                panic!("{:?}", image.source)
            };
            // The first of the two loads: entry v0 * 32 of the table at s0.
            let a = r.scalar.iter().find(|s| s.0 == load).unwrap().1;
            assert_eq!((a.param, a.offset), (0, 0));
            assert!(a.dynamic.is_some());
            // The sampler is the same for every lane.
            let sampler = r.descriptors.iter().find(|d| d.role == Role::Sampler);
            assert!(matches!(
                sampler.unwrap().source,
                Source::Loaded { first: 0, .. }
            ));
            // Four 64-bit compares of the first lane's dwords with the lane's
            // own; the eight reads become the lane's own dwords.
            assert_eq!(r.waterfall.compares.len(), 4);
            assert_eq!(r.waterfall.reads.len(), 8);
        }
    }

    #[test]
    fn descriptors_read_from_the_first_lane_of_a_per_lane_choice_are_traced_per_lane() {
        let r = waterfall(WATERFALL_PICKED_WAVE32, false).unwrap();
        let image = &r.descriptors[0];
        let Source::Lanes { args, .. } = &image.source else {
            panic!("{:?}", image.source)
        };
        // Lanes with their bit set take the first load (offset 0), the
        // others the second (0x20).
        let offsets: Vec<u32> = args
            .iter()
            .map(|a| match *a {
                Source::Loaded { load, first: 0 } => {
                    r.scalar.iter().find(|s| s.0 == load).unwrap().1.offset
                }
                ref a => panic!("{a:?}"),
            })
            .collect();
        assert_eq!(offsets, [0, 0x20]);
        assert_eq!(r.waterfall.compares.len(), 4);
    }

    /// What the second arm of [`kept_per_lane`] puts in v[4:7].
    #[derive(Clone, Copy)]
    pub(crate) enum SecondArm {
        /// A copy of table entry 0, loaded again (the console form).
        SameEntry,
        /// A copy of table entry 1.
        OtherEntry,
        /// A copy of a constant in s32 (and of s[33:35], never written).
        Constant,
        /// A copy of entry 0 with dwords 0 and 1 swapped.
        Swapped,
        /// Entry 1's dwords, each ORed with 0 (`v_or_b32_e64 v4, s32, 0`):
        /// the same bits, by an instruction that is no move.
        OredOtherEntry,
    }

    /// A buffer descriptor kept per lane in VGPRs, as the console compiler
    /// leaves the sampler of the G-buffer's static and foliage meshes
    /// (903776ba469c8bac at 0x5780): v[4:7] hold numbers, then lanes below
    /// 16 load table entry 0 and copy it there, lanes from 16 to 23 load it
    /// again (or what `second` says) and copy it, and both set a flag in v8.
    /// The lanes whose flag is set run a waterfall loop over v[4:7] that
    /// loads dword `lane` of the buffer, and store it to the UAV at dword 16
    /// of the table at s0. Which lanes hold numbers is told by the flag, a
    /// VGPR the lanes' paths do not follow. (The offsets in the comments
    /// are those of the one-word copies.)
    pub(crate) fn kept_per_lane(second: SecondArm) -> Vec<u32> {
        kept_per_lane_flagged(second, 0)
    }

    /// [`kept_per_lane`] with v8 set to `flag` before the arms: with 1, the
    /// lanes that take neither arm run the loop too, over their numbers.
    fn kept_per_lane_flagged(second: SecondArm, flag: u32) -> Vec<u32> {
        // v_mov_b32 v4..v7, the second arm's SGPRs
        let moves = |sgprs: [u32; 4]| -> Vec<u32> {
            (0..4)
                .map(|k| 0x7E08_0200 | (k << 17) | sgprs[k as usize])
                .collect()
        };
        let (load, copies): (&[u32], Vec<u32>) = match second {
            // s_load_dwordx4 s[32:35], s[4:5], 0x0 or 0x20
            SecondArm::SameEntry => (&[0xF408_0802, 0xFA00_0000], moves([0x20, 0x21, 0x22, 0x23])),
            SecondArm::OtherEntry => (&[0xF408_0802, 0xFA00_0020], moves([0x20, 0x21, 0x22, 0x23])),
            // s_mov_b32 s32, 0x5a; s_nop 0
            SecondArm::Constant => (&[0xBEA0_03FF, 0x0000_005A], moves([0x20, 0x21, 0x22, 0x23])),
            SecondArm::Swapped => (&[0xF408_0802, 0xFA00_0000], moves([0x21, 0x20, 0x22, 0x23])),
            // v_or_b32_e64 v(4 + k), s(32 + k), 0
            SecondArm::OredOtherEntry => (
                &[0xF408_0802, 0xFA00_0020],
                (0..4)
                    .flat_map(|k| [0xD51C_0004 + k, 0x0001_0020 + k])
                    .collect(),
            ),
        };
        // The second arm after its s_cbranch_execz: the load, the wait, the
        // copies and the flag; then the end of the arm's own mask.
        let arm = load.len() as u32 + 1 + copies.len() as u32 + 1;
        let mut words = vec![
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0A02,
            0xFA00_0040,                // s_load_dwordx4 s[40:43], s[4:5], 0x40
            0x3420_0082,                // v_lshlrev_b32 v16, 2, v0
            0x7E08_0D00,                // v_cvt_f32_u32 v4, v0
            0x7E0A_0D00,                // v_cvt_f32_u32 v5, v0
            0x7E0C_0D00,                // v_cvt_f32_u32 v6, v0
            0x7E0E_0D00,                // v_cvt_f32_u32 v7, v0
            0x7E10_0200 | (128 + flag), // v_mov_b32 v8, 0 (or `flag`)
            0xD4C4_001E,
            0x0002_0098, // v_cmp_gt_u32_e64 s30, 24, v0
            0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xBE94_3C6A, // s_and_saveexec_b32 s20, vcc_lo
            0x8914_147E, // s_xor_b32 s20, exec_lo, s20
            0xBF88_0008, // s_cbranch_execz -> 0x64
            0xF408_0602,
            0xFA00_0000,             // s_load_dwordx4 s[24:27], s[4:5], 0x0
            0xBF8C_C07F,             // s_waitcnt lgkmcnt(0)
            0x7E08_0218,             // v_mov_b32 v4, s24
            0x7E0A_0219,             // v_mov_b32 v5, s25
            0x7E0C_021A,             // v_mov_b32 v6, s26
            0x7E0E_021B,             // v_mov_b32 v7, s27
            0x7E10_0281,             // v_mov_b32 v8, 1
            0xBE94_3D14,             // 0x64: s_or_saveexec_b32 s20, s20
            0x897E_147E,             // s_xor_b32 exec_lo, exec_lo, s20
            0xBF88_0000 | (arm + 3), // s_cbranch_execz -> 0x9c
            0xBE9F_3C1E,             // s_and_saveexec_b32 s31, s30
            0xBF88_0000 | arm,       // s_cbranch_execz -> 0x98
        ];
        words.extend(load);
        words.push(0xBF8C_C07F); // s_waitcnt lgkmcnt(0)
        words.extend(copies);
        words.extend([
            0x7E10_0281, // v_mov_b32 v8, 1
            0x887E_1F7E, // 0x98: s_or_b32 exec_lo, exec_lo, s31
            0x887E_147E, // 0x9c: s_or_b32 exec_lo, exec_lo, s20
            0x7D84_1081, // v_cmp_eq_u32 vcc_lo, 1, v8
            0xBE97_3C6A, // s_and_saveexec_b32 s23, vcc_lo
            0xBF88_0013, // s_cbranch_execz -> 0xf8
            0xBE9D_037E, // s_mov_b32 s29, exec_lo
            0x7E18_0504, // 0xb0: v_readfirstlane_b32 s12, v4
            0x7E1A_0505, // v_readfirstlane_b32 s13, v5
            0x7E1C_0506, // v_readfirstlane_b32 s14, v6
            0x7E1E_0507, // v_readfirstlane_b32 s15, v7
            0xD4E2_0015,
            0x0002_080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
            0xD4E2_0016,
            0x0002_0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
            0x8715_1615, // s_and_b32 s21, s21, s22
            0xBE95_3C15, // s_and_saveexec_b32 s21, s21
            0xE030_1000,
            0x8003_0C10, // buffer_load_dword v12, v16, s[12:15], 0 offen
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
            0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
            0xBF89_FFF1, // s_cbranch_execnz -> 0xb0
            0xBEFE_031D, // s_mov_b32 exec_lo, s29
            0xE070_1000,
            0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
            0x887E_177E, // 0xf8: s_or_b32 exec_lo, exec_lo, s23
            0xBF81_0000, // s_endpgm
        ]);
        words
    }

    /// [`kept_per_lane`]: the lanes that take the loop came by one of the
    /// arms (their flag is set), but the lanes' paths do not tell, so the
    /// per-path trace finds the numbers of the lanes that took neither.
    /// Walked back path by path with the flag, the numbers come with a
    /// flag of 0, which no lane in the loop has: every definition there is
    /// a copy of the same dwords of loads from one address, the descriptor
    /// is the first load, and the loop is a proven waterfall over it (one
    /// pass). Not where the second arm copies another entry, a constant, or
    /// the same entry's dwords in another order.
    #[test]
    fn a_descriptor_kept_per_lane_in_vgprs_is_its_load() {
        let r = waterfall(&kept_per_lane(SecondArm::SameEntry), false).unwrap();
        // The loop's load, then the store.
        let buffer = &r.descriptors[0];
        let loads: Vec<Site> = r
            .scalar
            .iter()
            .filter(|(_, a)| (a.param, a.offset, a.dynamic) == (0, 0, None))
            .map(|s| s.0)
            .collect();
        assert_eq!(loads.len(), 2, "{:?}", r.scalar);
        let Source::Loaded { load, first: 0 } = buffer.source else {
            panic!("{:?}", buffer.source)
        };
        assert!(loads.contains(&load), "{load:?} of {loads:?}");
        assert_eq!(r.waterfall.compares.len(), 2);
        assert_eq!(r.waterfall.reads.len(), 4);
        assert!(r.waterfall.loads.is_empty());
        for second in [
            SecondArm::OtherEntry,
            SecondArm::Constant,
            SecondArm::Swapped,
        ] {
            assert_eq!(
                refusal(&kept_per_lane(second), false),
                "a descriptor read from the first lane of computed VGPRs"
            );
        }
    }

    /// The review of meshdata's landing (P1): lanes 0 to 15 copy entry 0
    /// (A), lanes 16 to 23 load entry 1 (B) and OR each of its dwords with
    /// 0 into v[4:7], which keeps B's bits, and set the flag. One copy of A
    /// per dword was enough before, and the OR was taken for a number: the
    /// loop read A's handle, and lane 16 116 instead of B's 216. Every
    /// definition on the paths of the lanes in the loop must be a copy of
    /// A: refused. So with A copied in both arms, where the lanes that take
    /// neither have their flag set too and run the loop over their numbers.
    #[test]
    fn a_descriptor_kept_per_lane_with_other_bits_on_one_path_is_refused() {
        let ored = kept_per_lane(SecondArm::OredOtherEntry);
        assert!(ored.contains(&0xD51C_0004), "v_or_b32_e64 v4, s32, 0");
        assert_eq!(
            refusal(&ored, false),
            "a descriptor read from the first lane of computed VGPRs"
        );
        assert_eq!(
            refusal(&kept_per_lane_flagged(SecondArm::SameEntry, 1), false),
            "a descriptor read from the first lane of computed VGPRs"
        );
    }

    #[test]
    fn a_first_lane_read_outside_a_loop_is_no_waterfall() {
        // The waterfall without its back edge: the s_cbranch_execnz is an
        // s_nop, so lanes the first one does not match never get a pass.
        let mut words = WATERFALL_FETCHED_WAVE32.to_vec();
        let back = words.iter().position(|&w| w == 0xBF89_FFE6).unwrap();
        words[back] = 0xBF80_0000;
        let err = waterfall(&words, false).unwrap_err();
        assert!(
            matches!(
                err,
                ShaderError::Unsupported {
                    what: "a descriptor read from the first lane outside a waterfall loop",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    /// Why resource tracking refuses `words`.
    fn refusal(words: &[u32], wave64: bool) -> &'static str {
        match waterfall(words, wave64) {
            Err(ShaderError::Unsupported { what, .. }) => what,
            other => panic!("not refused: {other:?}"),
        }
    }

    const NO_LOOP: &str = "a descriptor read from the first lane outside a waterfall loop";
    const NOT_SELECTED: &str = "a waterfall loop whose compares do not select the access's lanes";
    const BAD_RETRY: &str = "a waterfall loop whose retry does not keep the lanes still waiting";
    const SPLIT: &str = "a waterfall loop nested in another or split over blocks";

    /// The index of `word` in `words`.
    fn word_at(words: &[u32], word: u32) -> usize {
        words.iter().position(|&w| w == word).unwrap()
    }

    #[test]
    fn a_first_lane_read_without_its_compare_is_no_waterfall() {
        // The loop without the compare of dwords 0 and 1: nothing makes
        // the first lane's dwords the lane's own there.
        let mut words = WATERFALL_FETCHED_WAVE32.to_vec();
        let cmp = word_at(&words, 0x0002_080C);
        words[cmp - 1..=cmp].copy_from_slice(&[0xBF80_0000, 0xBF80_0000]);
        assert_eq!(refusal(&words, false), NOT_SELECTED);
    }

    #[test]
    fn a_compare_that_does_not_mask_the_access_is_no_waterfall() {
        // The compares are all there, but their result never narrows EXEC:
        // every lane samples with the first lane's descriptor (reported by
        // ifBars on upstream PR #566).
        let mut words = WATERFALL_JOINED_WAVE32.to_vec();
        let save = word_at(&words, 0xBE95_3C15);
        words[save] = 0xBE95_037E; // s_mov_b32 s21, exec_lo
        assert_eq!(refusal(&words, false), NOT_SELECTED);
    }

    #[test]
    fn an_access_under_a_further_mask_is_no_waterfall() {
        // The access runs for the matching lanes that VCC also holds, yet
        // the retry removes every matching lane.
        let mut words = WATERFALL_JOINED_WAVE32.to_vec();
        let nop = word_at(&words, 0xBE95_3C15) + 1;
        assert_eq!(words[nop], 0xBF80_0000);
        words[nop] = 0x877E_6A7E; // s_and_b32 exec_lo, exec_lo, vcc_lo
        assert_eq!(refusal(&words, false), NOT_SELECTED);
    }

    #[test]
    fn a_loop_that_drops_unmatched_lanes_is_no_waterfall() {
        let mut words = WATERFALL_JOINED_WAVE32.to_vec();
        let retry = word_at(&words, 0x897E_157E);
        words[retry] = 0xBEFE_0380; // s_mov_b32 exec_lo, 0
        assert_eq!(refusal(&words, false), BAD_RETRY);
    }

    #[test]
    fn a_retry_that_keeps_other_lanes_than_those_waiting_is_no_waterfall() {
        // The retry keeps the lanes of VCC that did not just sample, not
        // the waiting ones: lanes 16 and up never sample on the console.
        let mut words = WATERFALL_JOINED_WAVE32.to_vec();
        let retry = word_at(&words, 0x897E_157E);
        words[retry] = 0x8A7E_7E6A; // s_andn2_b32 exec_lo, vcc_lo, exec_lo
        assert_eq!(refusal(&words, false), BAD_RETRY);
    }

    #[test]
    fn a_wave64_retry_of_one_half_is_no_waterfall() {
        // Only the low half of EXEC loses the lanes just served.
        let mut words = WATERFALL_FETCHED_WAVE64.to_vec();
        let retry = word_at(&words, 0x89FE_167E);
        words[retry] = 0x897E_167E; // s_xor_b32 exec_lo, exec_lo, s22
        assert_eq!(refusal(&words, true), BAD_RETRY);
    }

    #[test]
    fn a_descriptor_used_after_exec_is_restored_is_no_waterfall() {
        let mut words = WATERFALL_JOINED_WAVE32.to_vec();
        let sample = word_at(&words, 0xF09C_0F08);
        let restore = word_at(&words, 0xBEFE_0317);
        let image = [words[sample], words[sample + 1]];
        words[sample..sample + 2].fill(0xBF80_0000); // Keep the branch offsets.
        words.splice(restore + 1..restore + 1, image);
        assert_eq!(refusal(&words, false), NO_LOOP);
    }

    #[test]
    fn a_wave64_access_with_exec_hi_overwritten_is_no_waterfall() {
        let mut words = WATERFALL_FETCHED_WAVE64.to_vec();
        let save = word_at(&words, 0xBE96_2416);
        words[save + 1] = 0xBEFF_03C1; // s_mov_b32 exec_hi, -1
        assert_eq!(refusal(&words, true), NOT_SELECTED);
    }

    /// The waterfall with `v_cmpx_eq` narrowing EXEC itself, and the retry
    /// as `todo & !EXEC` (`folded`: in one `s_andn2_wrexec`), as LLVM's
    /// AMDGPU backend writes it for GFX10 and later.
    pub(crate) fn cmpx_waterfall(wave64: bool, folded: bool) -> Vec<u32> {
        let (base, todo, restore): (&[u32], u32, u32) = if wave64 {
            (WATERFALL_FETCHED_WAVE64, 0xBE94_047E, 0xBEFE_0414)
        } else {
            (WATERFALL_JOINED_WAVE32, 0xBE97_037E, 0xBEFE_0317)
        };
        let reads = word_at(base, 0x7E18_0504);
        let sample = word_at(base, 0xF09C_0F08);
        let mut w = base[..=word_at(base, todo)].to_vec();
        // s_mov_b32 s21, exec_lo / s_mov_b64 s[22:23], exec
        w.push(if wave64 { 0xBE96_047E } else { 0xBE95_037E });
        w.extend(&base[reads..reads + 8]); // v_readfirstlane_b32 s12..s19
        w.extend([
            0x7DE4_080C, // v_cmpx_eq_u64 s[12:13], v[4:5]
            0x7DE4_0C0E, // v_cmpx_eq_u64 s[14:15], v[6:7]
            0x7DE4_1010, // v_cmpx_eq_u64 s[16:17], v[8:9]
            0x7DE4_1412, // v_cmpx_eq_u64 s[18:19], v[10:11]
        ]);
        w.extend(&base[sample..sample + 2]); // image_sample_lz
        match (wave64, folded) {
            // s_andn2_wrexec_b32 s21, s21; s_cbranch_execnz
            (false, true) => w.extend([0xBE95_4715, 0xBF89_FFF0]),
            // s_andn2_b32 s21, s21, exec_lo; s_mov_b32 exec_lo, s21
            (false, false) => w.extend([0x8A15_7E15, 0xBEFE_0315, 0xBF89_FFEF]),
            // s_andn2_wrexec_b64 s[22:23], s[22:23]
            (true, true) => w.extend([0xBE96_3A16, 0xBF89_FFF0]),
            // s_andn2_b64 s[22:23], s[22:23], exec; s_mov_b64 exec, s[22:23]
            (true, false) => w.extend([0x8A96_7E16, 0xBEFE_0416, 0xBF89_FFEF]),
        }
        w.extend(&base[word_at(base, restore)..]);
        w
    }

    #[test]
    fn a_waterfall_narrowed_by_v_cmpx_is_one() {
        for wave64 in [false, true] {
            for folded in [false, true] {
                let r = waterfall(&cmpx_waterfall(wave64, folded), wave64)
                    .unwrap_or_else(|e| panic!("wave64 {wave64} folded {folded}: {e:?}"));
                assert_eq!(r.waterfall.compares.len(), 4, "{wave64} {folded}");
                assert_eq!(r.waterfall.reads.len(), 8, "{wave64} {folded}");
            }
        }
    }

    #[test]
    fn a_v_cmpx_waterfall_whose_retry_drops_lanes_is_none() {
        let mut words = cmpx_waterfall(false, false);
        let retry = word_at(&words, 0x8A15_7E15);
        words[retry] = 0x8A15_7E6A; // s_andn2_b32 s21, vcc_lo, exec_lo
        assert_eq!(refusal(&words, false), BAD_RETRY);
    }

    #[test]
    fn a_waterfall_nested_in_another_is_refused_by_name() {
        // An image whose resource and sampler are both per lane, with one
        // loop per operand, the one for the sampler inside the one for the
        // resource (what the SelectionDAG legalisation in LLVM would give).
        // The console gives each lane its own pair. Refused for now: the
        // access of the inner loop reads the first-lane dwords of the outer
        // loop, which another block holds.
        let words = [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xBE86_0301, // s_mov_b32 s6, s1
            0xBE87_0380, // s_mov_b32 s7, 0
            0xF408_0A02,
            0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
            0x7E04_0D00, // v_cvt_f32_u32 v2, v0
            0x7E06_0302, // v_mov_b32 v3, v2
            0x3420_0082, // v_lshlrev_b32 v16, 2, v0
            0x3602_0081, // v_and_b32 v1, 1, v0
            0x3402_0285, // v_lshlrev_b32 v1, 5, v1
            0x3622_0080, // v_and_b32 v17, 0, v0
            0x3422_2284, // v_lshlrev_b32 v17, 4, v17
            0xDC38_8000,
            0x0404_0001, // global_load_dwordx4 v[4:7], v1, s[4:5]
            0xDC38_8010,
            0x0804_0001, // global_load_dwordx4 v[8:11], v1, s[4:5] offset:16
            0xDC38_8000,
            0x1406_0011, // global_load_dwordx4 v[20:23], v17, s[6:7]
            0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
            0xBE94_037E, // s_mov_b32 s20, exec_lo
            // L1, the loop over the resource:
            0x7E18_0504, // v_readfirstlane_b32 s12, v4
            0x7E1A_0505,
            0x7E1C_0506,
            0x7E1E_0507,
            0x7E20_0508,
            0x7E22_0509,
            0x7E24_050A,
            0x7E26_050B, // v_readfirstlane_b32 s19, v11
            0xD4E2_0015,
            0x0002_080C, // v_cmp_eq_u64_e64 s21, s[12:13], v[4:5]
            0xD4E2_0016,
            0x0002_0C0E, // v_cmp_eq_u64_e64 s22, s[14:15], v[6:7]
            0x8715_1615, // s_and_b32 s21, s21, s22
            0xD4E2_0016,
            0x0002_1010, // v_cmp_eq_u64_e64 s22, s[16:17], v[8:9]
            0x8715_1615, // s_and_b32 s21, s21, s22
            0xD4E2_0016,
            0x0002_1412, // v_cmp_eq_u64_e64 s22, s[18:19], v[10:11]
            0x8715_1615, // s_and_b32 s21, s21, s22
            0xBE95_3C15, // s_and_saveexec_b32 s21, s21
            0xBE9A_037E, // s_mov_b32 s26, exec_lo
            // L2, the loop over the sampler:
            0x7E10_0514, // v_readfirstlane_b32 s8, v20
            0x7E12_0515, // v_readfirstlane_b32 s9, v21
            0x7E14_0516, // v_readfirstlane_b32 s10, v22
            0x7E16_0517, // v_readfirstlane_b32 s11, v23
            0xD4E2_0018,
            0x0002_2808, // v_cmp_eq_u64_e64 s24, s[8:9], v[20:21]
            0xD4E2_0019,
            0x0002_2C0A, // v_cmp_eq_u64_e64 s25, s[10:11], v[22:23]
            0x8718_1918, // s_and_b32 s24, s24, s25
            0xBE98_3C18, // s_and_saveexec_b32 s24, s24
            0xF09C_0F08,
            0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11]
            0x897E_187E, // s_xor_b32 exec_lo, exec_lo, s24
            0xBF89_FFF2, // s_cbranch_execnz L2
            0xBEFE_031A, // s_mov_b32 exec_lo, s26
            0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
            0xBF89_FFDA, // s_cbranch_execnz L1
            0xBEFE_0314, // s_mov_b32 exec_lo, s20
            0xE070_1000,
            0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
            0xBF81_0000, // s_endpgm
        ];
        assert_eq!(refusal(&words, false), SPLIT);
    }

    #[test]
    fn descriptors_read_from_the_first_lane_of_lds_fail_naming_it() {
        // The waterfall over descriptors each lane read from LDS:
        // ds_read_b128 v[4:7], v1 and ds_read_b128 v[8:11], v1 offset:16
        // in place of the global loads.
        let mut words = WATERFALL_FETCHED_WAVE32.to_vec();
        words[5..9].copy_from_slice(&[0xDBFC_0000, 0x0400_0001, 0xDBFC_0010, 0x0800_0001]);
        let err = waterfall(&words, false).unwrap_err();
        assert!(
            matches!(
                err,
                ShaderError::Unsupported {
                    what: "a descriptor read from the first lane of an LDS read",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    /// The block of `p` at byte `offset`.
    fn block_at(words: &[u32], offset: usize) -> usize {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        p.blocks.iter().position(|b| b.offset == offset).unwrap()
    }

    /// The table offset of what the scalar load of `s` reads.
    fn table_offset(r: &Resources, s: &Source) -> u32 {
        match *s {
            Source::Loaded { load, first: 0 } => {
                r.scalar.iter().find(|x| x.0 == load).unwrap().1.offset
            }
            ref s => panic!("{s:?}"),
        }
    }

    #[test]
    fn descriptors_read_from_the_first_lane_of_an_if_else_join_are_traced_per_path() {
        let words = WATERFALL_JOINED_WAVE32;
        let r = waterfall(words, false).unwrap();
        let image = r
            .descriptors
            .iter()
            .find(|d| matches!(d.source, Source::LanePhi { .. }))
            .unwrap();
        // The join after the else arm: lanes that ran it (lanes from 16 on)
        // take the second entry (0x20); the others come through the flow
        // block, whose join gives lanes that ran the then arm the first one
        // (0). No lane that skipped the then arm comes that way, so the
        // dwords they held before (the inputs) are not traced.
        let Source::LanePhi { block, args } = &image.source else {
            unreachable!()
        };
        assert_eq!(*block, block_at(words, 0x9c));
        let [Some(flow), Some(other)] = &args[..] else {
            panic!("{args:?}")
        };
        assert_eq!(table_offset(&r, other), 0x20);
        let Source::LanePhi { block, args } = flow else {
            panic!("{flow:?}")
        };
        assert_eq!(*block, block_at(words, 0x70));
        let [None, Some(then)] = &args[..] else {
            panic!("{args:?}")
        };
        assert_eq!(table_offset(&r, then), 0);
        // The loop's compares are of the joined dwords; still recognised.
        assert_eq!(r.waterfall.compares.len(), 4);
        assert_eq!(r.waterfall.reads.len(), 8);
        // Without the lanes' paths the join is refused.
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        let layout = [param(Table, 0, 1), param(Table, 1, 1)];
        let err = track(&p, &layout, HwStage::Cs, None).unwrap_err();
        assert!(
            matches!(
                err,
                ShaderError::Unsupported {
                    what: "a descriptor read from the first lane of a phi, without the lanes' paths",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn descriptors_joined_from_a_per_lane_load_and_a_per_lane_choice_are_traced_per_path() {
        let words = WATERFALL_JOINED_MIXED_WAVE32;
        let r = waterfall(words, false).unwrap();
        let image = r
            .descriptors
            .iter()
            .find(|d| matches!(d.source, Source::LanePhi { .. }))
            .unwrap();
        let Source::LanePhi { block, args } = &image.source else {
            unreachable!()
        };
        assert_eq!(*block, block_at(words, 0xb8));
        let [
            Some(Source::LanePhi { args: flow, .. }),
            Some(Source::Lanes { args: picked, .. }),
        ] = &args[..]
        else {
            panic!("{args:?}")
        };
        // The then arm loads each lane's entry from the table at s0.
        let [None, Some(Source::Fetched { load, first: 0 })] = flow[..] else {
            panic!("{flow:?}")
        };
        let a = r.scalar.iter().find(|s| s.0 == load).unwrap().1;
        assert_eq!((a.param, a.offset), (0, 0));
        assert!(a.dynamic.is_some());
        // The else arm picks the first entry where vcc is set.
        let offsets: Vec<u32> = picked.iter().map(|s| table_offset(&r, s)).collect();
        assert_eq!(offsets, [0, 0x20]);
        assert_eq!(r.waterfall.compares.len(), 4);
    }

    #[test]
    fn a_descriptor_read_from_the_first_lane_of_a_loop_phi_fails_naming_it() {
        let err = waterfall(WATERFALL_LOOPED_WAVE32, false).unwrap_err();
        assert!(
            matches!(
                err,
                ShaderError::Unsupported {
                    what: "a descriptor read from the first lane of a phi at a loop header",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn a_per_lane_descriptor_some_lanes_do_not_write_fails() {
        // The else arm of WATERFALL_JOINED_WAVE32 not skipped (its
        // s_cbranch_execz an s_nop): it runs with EXEC clear in the lanes
        // of the then arm, which keep the first entry; the second is not
        // every lane's.
        let mut words = WATERFALL_JOINED_WAVE32.to_vec();
        let skip = words.iter().rposition(|&w| w == 0xBF88_0008).unwrap();
        words[skip] = 0xBF80_0000;
        let err = waterfall(&words, false).unwrap_err();
        assert!(
            matches!(
                err,
                ShaderError::Unsupported {
                    what: "a per-lane descriptor written where some lanes may keep the old one",
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn a_buffer_descriptor_joined_per_lane_is_traced_on_the_paths_of_lanes_that_load() {
        let words = WATERFALL_JOINED_BUFFER_WAVE32;
        let r = waterfall(words, false).unwrap();
        let buffer = r
            .descriptors
            .iter()
            .find(|d| matches!(d.source, Source::LanePhi { .. }))
            .unwrap();
        // The outer join: lanes from the else arm's inner join, or (below
        // 16) through the flow block.
        let Source::LanePhi { block, args } = &buffer.source else {
            unreachable!()
        };
        assert_eq!(*block, block_at(words, 0x78));
        let [
            Some(Source::LanePhi { args: flow, .. }),
            Some(Source::LanePhi {
                block: inner,
                args: nested,
            }),
        ] = &args[..]
        else {
            panic!("{args:?}")
        };
        let [None, Some(then)] = &flow[..] else {
            panic!("{flow:?}")
        };
        assert_eq!(table_offset(&r, then), 0);
        // Lanes from 24 on skip the inner arm and keep the inputs, but do
        // not load: that edge is not traced.
        assert_eq!(*inner, block_at(words, 0x74));
        let [None, Some(other)] = &nested[..] else {
            panic!("{nested:?}")
        };
        assert_eq!(table_offset(&r, other), 0x20);
        assert_eq!(r.waterfall.compares.len(), 2);
        assert_eq!(r.waterfall.reads.len(), 4);
    }

    #[test]
    fn a_guarded_access_excludes_dead_lane_edges_at_a_join() {
        // Lanes failing `v_cmp_gt_u32` clear EXEC and take `s_cbranch_execz`
        // directly to the join block without initializing the immediate
        // sampler `s[8:11]`. The join's `image_sample_lz` is guarded by EXEC,
        // so only the arm where `s[8:11]` was built can execute it.
        let code: Vec<u8> = [
            0xBE84_0300u32, // s_mov_b32 s4, s0
            0xBE85_0380,    // s_mov_b32 s5, 0
            0xF40C_0302,
            0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
            0x7E04_0D00, // v_cvt_f32_u32 v2, v0
            0x7E06_0302, // v_mov_b32 v3, v2
            0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0x877E_6A7E, // s_and_b32 exec_lo, exec_lo, vcc_lo
            0xBF88_0004, // s_cbranch_execz -> join (skips 4 dwords)
            0xBE88_0380, // s_mov_b32 s8, 0
            0xBE89_0380, // s_mov_b32 s9, 0
            0xBE8A_0380, // s_mov_b32 s10, 0
            0xBE8B_0380, // s_mov_b32 s11, 0
            0xF09C_0F08,
            0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11]
            S_ENDPGM,
        ]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        let structured = lanes(&p).expect("structured form");
        assert!(
            structured.guards[2][0].is_some(),
            "join access must carry an EXEC guard"
        );
        let r = track(&p, &[param(Table, 0, 1)], HwStage::Cs, Some(&structured)).unwrap();
        assert!(
            r.descriptors
                .iter()
                .any(|d| d.role == Role::Sampler && d.source == Source::Constant(vec![0; 4])),
            "guarded join should resolve the immediate sampler from the live arm: {:?}",
            r.descriptors
        );
    }

    #[test]
    fn a_guarded_access_still_refuses_a_descriptor_rewritten_by_other_lanes() {
        // Removing the second `s_cbranch_execz` puts the odd lanes'
        // `image_sample_lz` directly in the join block under an EXEC guard.
        // Even lanes still rewrite `s8` in the first arm, so a mixed wave
        // samples with the rewritten descriptor: must remain refused.
        let mut words = rewritten_in_an_arm();
        let second_skip = word_at(&words, 0xBF88_0005);
        words[second_skip] = 0xBF80_0000; // s_nop 0
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        let st = lanes(&p).expect("structured form");
        let join = block_at(&words, 0x50);
        let sample = p.blocks[join]
            .insts
            .iter()
            .position(|inst| inst.name() == "image_sample_lz")
            .unwrap();
        assert!(
            st.guards[join][sample].is_some(),
            "join access must carry an EXEC guard"
        );
        assert_eq!(refusal(&words, false), "a descriptor changed after loading");
    }

    #[test]
    fn a_guarded_access_excludes_dead_lane_edges_after_a_long_reach_chain() {
        // 17 compound early-out clauses make `reach` at the guarded join carry
        // 3^17 > MAX_MEET satisfying assignments. `only_edge` must still prove
        // that the join's EXEC guard contradicts the dead skip edge.
        let mut words = vec![
            0xBE84_0300u32, // s_mov_b32 s4, s0
            0xBE85_0380,    // s_mov_b32 s5, 0
            0xF40C_0302,
            0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
            0x7E04_0D00, // v_cvt_f32_u32 v2, v0
            0x7E06_0302, // v_mov_b32 v3, v2
        ];
        const CLAUSES: usize = 17;
        const TAIL_WORDS: usize = 11; // compare + waitcnt + and + branch + 4 movs + sample(2) + endpgm
        for k in 0..CLAUSES {
            let remaining = (CLAUSES - 1 - k) * 6 + TAIL_WORDS - 1;
            words.extend([
                0x7D84_0080 + k as u32,         // v_cmp_eq_u32 vcc_lo, k, v0
                0xBE94_036A,                    // s_mov_b32 s20, vcc_lo
                0x7D84_0280 + k as u32,         // v_cmp_eq_u32 vcc_lo, k, v1
                0x8714_6A14,                    // s_and_b32 s20, s20, vcc_lo
                0x8A7E_147E,                    // s_andn2_b32 exec_lo, exec_lo, s20
                0xBF88_0000 | remaining as u32, // s_cbranch_execz -> s_endpgm
            ]);
        }
        words.extend([
            0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0x877E_6A7E, // s_and_b32 exec_lo, exec_lo, vcc_lo
            0xBF88_0004, // s_cbranch_execz -> join (skips 4 dwords)
            0xBE88_0380, // s_mov_b32 s8, 0
            0xBE89_0380, // s_mov_b32 s9, 0
            0xBE8A_0380, // s_mov_b32 s10, 0
            0xBE8B_0380, // s_mov_b32 s11, 0
            0xF09C_0F08,
            0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11]
            S_ENDPGM,
        ]);
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        let structured = lanes(&p).expect("structured form");
        let join = p
            .blocks
            .iter()
            .position(|b| b.insts.iter().any(|i| i.name() == "image_sample_lz"))
            .unwrap();
        assert!(
            structured.guards[join][0].is_some(),
            "join access must carry an EXEC guard"
        );
        assert!(
            structured.bools.support(structured.reach[join]).len() >= CLAUSES * 2,
            "reach at the join must retain all {CLAUSES} compound clauses"
        );
        let r = track(&p, &[param(Table, 0, 1)], HwStage::Cs, Some(&structured)).unwrap();
        assert!(
            r.descriptors
                .iter()
                .any(|d| d.role == Role::Sampler && d.source == Source::Constant(vec![0; 4])),
            "guarded join after a long reach chain must resolve the live arm: {:?}",
            r.descriptors
        );
    }

    #[test]
    fn a_guarded_wave64_access_excludes_dead_lane_edges_at_a_join() {
        let words = [
            0xBE84_0300u32, // s_mov_b32 s4, s0
            0xBE85_0380,    // s_mov_b32 s5, 0
            0xF40C_0302,
            0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
            0x7E04_0D00, // v_cvt_f32_u32 v2, v0
            0x7E06_0302, // v_mov_b32 v3, v2
            0x7D88_0090, // v_cmp_gt_u32 vcc, 16, v0
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0x87FE_6A7E, // s_and_b64 exec, exec, vcc
            0xBF88_0004, // s_cbranch_execz -> join
            0xBE88_0380, // s_mov_b32 s8, 0
            0xBE89_0380, // s_mov_b32 s9, 0
            0xBE8A_0380, // s_mov_b32 s10, 0
            0xBE8B_0380, // s_mov_b32 s11, 0
            0xF09C_0F08,
            0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11]
            S_ENDPGM,
        ];
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, true, 256).unwrap();
        let structured = lanes(&p).expect("structured form");
        assert!(structured.guards[2][0].is_some());
        let r = track(&p, &[param(Table, 0, 1)], HwStage::Cs, Some(&structured)).unwrap();
        assert!(
            r.descriptors
                .iter()
                .any(|d| d.role == Role::Sampler && d.source == Source::Constant(vec![0; 4])),
            "{:?}",
            r.descriptors
        );
    }

    #[test]
    fn a_scalar_buffer_load_at_a_guarded_join_still_refuses_dead_edge() {
        // `s_buffer_load_dword` is an SMEM instruction and ignores EXEC, so a
        // wave taking `s_cbranch_execz` still executes it with uninitialized
        // descriptor SGPRs `s[8:11]`. Must remain refused.
        let code: Vec<u8> = [
            0xBE84_0300u32, // s_mov_b32 s4, s0
            0xBE85_0380,    // s_mov_b32 s5, 0
            0x7D88_0090,    // v_cmp_gt_u32 vcc_lo, 16, v0
            0x877E_6A7E,    // s_and_b32 exec_lo, exec_lo, vcc_lo
            0xBF88_0002,    // s_cbranch_execz -> join (skips 2 dwords)
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
            0xF420_0504,
            0xFA00_0000, // join: s_buffer_load_dword s20, s[8:11], null
            S_ENDPGM,
        ]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        let structured = lanes(&p).expect("structured form");
        assert_eq!(structured.guards[2][0], None);
        let err = track(&p, &[param(Table, 0, 1)], HwStage::Cs, Some(&structured)).unwrap_err();
        assert!(matches!(
            err,
            ShaderError::Unsupported {
                what: "a descriptor neither loaded nor built",
                ..
            }
        ));
    }

    #[test]
    fn a_guarded_access_with_zero_exec_still_refuses_uninitialized_join() {
        // Setting EXEC to 0 before the join access yields `guard == Bool::FALSE`;
        // `only_edge` must see zero live edges and refuse cleanly without panicking.
        let code: Vec<u8> = [
            0xBE84_0300u32, // s_mov_b32 s4, s0
            0xBE85_0380,    // s_mov_b32 s5, 0
            0xF40C_0302,
            0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
            0x7E04_0D00, // v_cvt_f32_u32 v2, v0
            0x7E06_0302, // v_mov_b32 v3, v2
            0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0x877E_6A7E, // s_and_b32 exec_lo, exec_lo, vcc_lo
            0xBF88_0004, // s_cbranch_execz -> join
            0xBE88_0380, // s_mov_b32 s8, 0
            0xBE89_0380, // s_mov_b32 s9, 0
            0xBE8A_0380, // s_mov_b32 s10, 0
            0xBE8B_0380, // s_mov_b32 s11, 0
            0xBEFE_0380, // join: s_mov_b32 exec_lo, 0
            0xF09C_0F08,
            0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11]
            S_ENDPGM,
        ]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        let structured = lanes(&p).expect("structured form");
        assert_eq!(structured.guards[2][1], Some(Bool::FALSE));
        let err = track(&p, &[param(Table, 0, 1)], HwStage::Cs, Some(&structured)).unwrap_err();
        assert!(matches!(
            err,
            ShaderError::Unsupported {
                what: "a descriptor neither loaded nor built",
                ..
            }
        ));
    }

    #[test]
    fn a_guarded_global_load_resolves_saddr_high_word_from_live_arm() {
        // `global_load_dwordx4` is EXEC-masked and takes an SGPR base `s[4:5]`.
        // Clearing `s5 = 0` only in the `s_cbranch_execz` arm resolves cleanly
        // when the join's load is guarded by EXEC, and still fails when `s5 = 1`.
        let make = |high: u32| {
            let code: Vec<u8> = [
                0xBE84_0300u32,     // s_mov_b32 s4, s0
                0x7D88_0090,        // v_cmp_gt_u32 vcc_lo, 16, v0
                0x877E_6A7E,        // s_and_b32 exec_lo, exec_lo, vcc_lo
                0xBF88_0001,        // s_cbranch_execz -> join (skips 1 dword)
                0xBE85_0300 | high, // s_mov_b32 s5, high
                0xDC38_8000,
                0x0404_0001, // join: global_load_dwordx4 v[4:7], v1, s[4:5]
                S_ENDPGM,
            ]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
            let p = crate::ir::build(&code, false, 256).unwrap();
            let st = lanes(&p).expect("structured form");
            assert!(st.guards[2][0].is_some());
            track(&p, &[param(Table, 0, 1)], HwStage::Cs, Some(&st))
        };
        let r = make(128).unwrap(); // inline 0
        assert_eq!(r.scalar.len(), 1);
        assert_eq!(r.scalar[0].1.param, 0);
        assert!(matches!(
            make(129), // inline 1
            Err(ShaderError::Unsupported {
                what: "an address with an untracked high half",
                ..
            })
        ));
    }

    #[test]
    fn a_guarded_built_descriptor_resolves_high_word_through_set_bits() {
        // The high half of a root buffer descriptor is copied into `s5` only in
        // the live arm, and `s_or_b32` sets descriptor bits in the join block
        // before a guarded `buffer_store_dword`.
        let code: Vec<u8> = [
            0xBE84_0300u32, // s_mov_b32 s4, s0
            0xBE86_03C1,    // s_mov_b32 s6, -1
            0xBE87_03FF,
            0x1101_4FAC, // s_mov_b32 s7, 0x11014fac
            0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
            0x877E_6A7E, // s_and_b32 exec_lo, exec_lo, vcc_lo
            0xBF88_0001, // s_cbranch_execz -> join (skips 1 dword)
            0xBE85_0301, // s_mov_b32 s5, s1
            0x8805_FF05,
            0x0010_0000, // join: s_or_b32 s5, s5, 0x100000
            0xE070_1000,
            0x8001_0102, // buffer_store_dword v1, v2, s[4:7], 0 offen
            S_ENDPGM,
        ]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        let st = lanes(&p).expect("structured form");
        assert!(st.guards[2][1].is_some());
        let r = track(&p, &[param(Descriptor, 0, 2)], HwStage::Cs, Some(&st)).unwrap();
        assert_eq!(
            r.descriptors[0].source,
            Source::Built {
                address: at(0, 0),
                bits: 0x10_0000,
                words: [u32::MAX, 0x1101_4FAC],
            }
        );
    }
}
