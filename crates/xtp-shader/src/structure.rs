//! Control-flow structurisation (decision 0003): the program as one lane
//! runs it, without EXEC.
//!
//! Lane masks (EXEC, compare and carry results, the SGPRs they are saved
//! in) become per-lane booleans ([`Bools`]). A branch on a wave-wide test of
//! a mask (`s_cbranch_execz`, `vccz`, SCC of a mask operation) becomes a
//! branch on the lane's own bit: compiled code only skips work that lanes
//! outside the mask would not change, or tests a value that is the same in
//! every lane, so either way each lane can decide alone. A path-sensitive
//! pass then proves, for each vector instruction, that EXEC holds in every
//! lane that reaches it; what it cannot prove stays as the instruction's
//! guard, a per-lane `if` around it.
//!
//! The control-flow graph keeps its shape and must be reducible (DXIL's
//! only requirement). Helper lanes of a pixel shader count as live: on PC
//! their side effects are dropped too, and what they compute is only read
//! through derivatives.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

use crate::ShaderError;
use crate::gfx10::{Fields, NULL, Operand};
use crate::ir::{EXEC, Exit, Inst, Program, Reg, VCC, Value};

/// A per-lane boolean: a node of [`Bools`]. Equal functions are equal nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Bool(u32);

impl Bool {
    pub const FALSE: Bool = Bool(0);
    pub const TRUE: Bool = Bool(1);
}

/// What a variable of [`Bools`] stands for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Atom {
    /// The lane's bit of the compare or carry result in this VALU-written
    /// value, before EXEC masking.
    Lane(Value),
    /// An SCC value, the same in every lane.
    Uniform(Value),
    /// A lane mask joined at a block entry, by its low dword's phi.
    Phi(Value),
    /// `s_wqm` result: some lane of the quad is in the source mask.
    Quad(Value),
    /// A mask the pass cannot see into (loaded, built from scalars, an
    /// EXEC write it does not follow): the lane's bit of the register pair
    /// this value is a dword of.
    Opaque(Value),
    /// A constant mask other than all lanes or none.
    Constant(u64),
}

/// Reduced ordered binary decision diagrams over [`Atom`]s.
#[derive(Debug, Clone)]
pub struct Bools {
    /// (variable, low, high); nodes 0 and 1 are the constants.
    nodes: Vec<(u32, Bool, Bool)>,
    unique: Map<(u32, Bool, Bool), Bool>,
    memo: Map<(Bool, Bool, Bool), Bool>,
    atoms: Vec<Atom>,
    /// Reverse-postorder index of the block that defines each atom.
    born: Vec<usize>,
    ids: Map<Atom, u32>,
    overflow: bool,
}

/// The tables below have small integer keys; SipHash dominated run time.
type Map<K, V> = HashMap<K, V, BuildHasherDefault<Fx>>;

/// FxHash's multiply-rotate step.
#[derive(Default)]
struct Fx(u64);

impl Hasher for Fx {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(u64::from(b));
        }
    }

    fn write_u32(&mut self, n: u32) {
        self.write_u64(u64::from(n));
    }

    fn write_u64(&mut self, n: u64) {
        self.0 = (self.0.rotate_left(5) ^ n).wrapping_mul(0x517c_c1b7_2722_0a95);
    }

    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }
}

const TERMINAL: u32 = u32::MAX;
const MAX_NODES: usize = 1 << 21;
/// Case splits [`Bools::meet`] makes before it gives up.
const MAX_MEET: usize = 1 << 16;

impl Bools {
    fn new() -> Self {
        Bools {
            nodes: vec![
                (TERMINAL, Bool::FALSE, Bool::FALSE),
                (TERMINAL, Bool::TRUE, Bool::TRUE),
            ],
            unique: Map::default(),
            memo: Map::default(),
            atoms: Vec::new(),
            born: Vec::new(),
            ids: Map::default(),
            overflow: false,
        }
    }

    /// The atom tested at the top of `b` and the results when it is false
    /// and true; `None` for the constants.
    pub fn node(&self, b: Bool) -> Option<(Atom, Bool, Bool)> {
        let (v, lo, hi) = self.nodes[b.0 as usize];
        (v != TERMINAL).then(|| (self.atoms[v as usize], lo, hi))
    }

    /// The boolean that is `atom` itself, if some boolean tests it.
    pub fn var(&self, atom: Atom) -> Option<Bool> {
        let &v = self.ids.get(&atom)?;
        self.unique.get(&(v, Bool::FALSE, Bool::TRUE)).copied()
    }

    /// The atoms `b` depends on, in variable order.
    pub fn support(&self, b: Bool) -> Vec<Atom> {
        let mut vars = self.vars(b);
        vars.sort_unstable();
        vars.into_iter().map(|v| self.atoms[v as usize]).collect()
    }

    fn vars(&self, b: Bool) -> Vec<u32> {
        let (mut seen, mut vars, mut stack) = (Vec::new(), Vec::new(), vec![b]);
        while let Some(b) = stack.pop() {
            let (v, lo, hi) = self.nodes[b.0 as usize];
            if v == TERMINAL || seen.contains(&b) {
                continue;
            }
            seen.push(b);
            if !vars.contains(&v) {
                vars.push(v);
            }
            stack.extend([lo, hi]);
        }
        vars
    }

    fn atom(&mut self, atom: Atom, born: usize) -> Bool {
        let v = *self.ids.entry(atom).or_insert_with(|| {
            self.atoms.push(atom);
            self.born.push(born);
            (self.atoms.len() - 1) as u32
        });
        self.mk(v, Bool::FALSE, Bool::TRUE)
    }

    fn mk(&mut self, v: u32, lo: Bool, hi: Bool) -> Bool {
        if lo == hi {
            return lo;
        }
        if let Some(&b) = self.unique.get(&(v, lo, hi)) {
            return b;
        }
        if self.nodes.len() >= MAX_NODES {
            // The caller fails once it sees `overflow`.
            self.overflow = true;
            return Bool::FALSE;
        }
        let b = Bool(self.nodes.len() as u32);
        self.nodes.push((v, lo, hi));
        self.unique.insert((v, lo, hi), b);
        b
    }

    /// Cofactors of `b` for `v`, which is `b`'s top variable or above it.
    fn split(&self, b: Bool, v: u32) -> (Bool, Bool) {
        match self.nodes[b.0 as usize] {
            (bv, lo, hi) if bv == v => (lo, hi),
            _ => (b, b),
        }
    }

    fn ite(&mut self, f: Bool, g: Bool, h: Bool) -> Bool {
        if f == Bool::TRUE || g == h {
            return g;
        }
        if f == Bool::FALSE {
            return h;
        }
        if (g, h) == (Bool::TRUE, Bool::FALSE) {
            return f;
        }
        if let Some(&r) = self.memo.get(&(f, g, h)) {
            return r;
        }
        let top = |b: Bool| self.nodes[b.0 as usize].0;
        let v = top(f).min(top(g)).min(top(h));
        let ((f0, f1), (g0, g1), (h0, h1)) = (self.split(f, v), self.split(g, v), self.split(h, v));
        let lo = self.ite(f0, g0, h0);
        let hi = self.ite(f1, g1, h1);
        let r = self.mk(v, lo, hi);
        self.memo.insert((f, g, h), r);
        r
    }

    fn and(&mut self, a: Bool, b: Bool) -> Bool {
        self.ite(a, b, Bool::FALSE)
    }

    fn or(&mut self, a: Bool, b: Bool) -> Bool {
        self.ite(a, Bool::TRUE, b)
    }

    fn not(&mut self, a: Bool) -> Bool {
        self.ite(a, Bool::FALSE, Bool::TRUE)
    }

    fn xor(&mut self, a: Bool, b: Bool) -> Bool {
        let nb = self.not(b);
        self.ite(a, nb, b)
    }

    /// Whether some lane can satisfy every `(b, holds)` of `all` at once:
    /// `b` where `holds`, its negation where not. Yes when the search grows
    /// too large to finish.
    pub fn meet(&self, all: &[(Bool, bool)]) -> bool {
        let mut budget = MAX_MEET;
        let mut failed: HashSet<Vec<u32>, BuildHasherDefault<Fx>> = HashSet::default();
        let codes: Vec<u32> = all
            .iter()
            .map(|&(b, holds)| {
                debug_assert!(b.0 < (1 << 31));
                (b.0 << 1) | u32::from(holds)
            })
            .collect();
        self.meet_in(codes, &mut budget, &mut failed)
    }

    fn meet_in(
        &self,
        mut all: Vec<u32>,
        budget: &mut usize,
        failed: &mut HashSet<Vec<u32>, BuildHasherDefault<Fx>>,
    ) -> bool {
        let mut kept = 0;
        for i in 0..all.len() {
            let code = all[i];
            let (b, holds) = (code >> 1, code & 1 != 0);
            if b <= Bool::TRUE.0 {
                if (b == Bool::TRUE.0) != holds {
                    return false;
                }
            } else {
                all[kept] = code;
                kept += 1;
            }
        }
        all.truncate(kept);
        if all.is_empty() || *budget == 0 {
            return true;
        }
        all.sort_unstable();
        all.dedup();
        if all.windows(2).any(|w| (w[0] ^ w[1]) == 1) {
            return false;
        }
        if failed.contains(&all) {
            return false;
        }
        *budget -= 1;
        let top = |code: u32| self.nodes[(code >> 1) as usize].0;
        let v = all.iter().map(|&c| top(c)).min().expect("not empty");
        let found = [false, true].into_iter().any(|value| {
            let next = all
                .iter()
                .map(|&code| {
                    let (lo, hi) = self.split(Bool(code >> 1), v);
                    let next = if value { hi } else { lo };
                    (next.0 << 1) | (code & 1)
                })
                .collect();
            self.meet_in(next, budget, failed)
        });
        if !found {
            failed.insert(all);
        }
        found
    }

    fn implies(&mut self, a: Bool, b: Bool) -> bool {
        let nb = self.not(b);
        self.and(a, nb) == Bool::FALSE
    }

    /// `b` with every atom born at reverse-postorder index `from` or later
    /// existentially quantified: what still holds once those are redefined.
    fn forget(&mut self, b: Bool, from: usize, memo: &mut Map<Bool, Bool>) -> Bool {
        let (v, lo, hi) = self.nodes[b.0 as usize];
        if v == TERMINAL {
            return b;
        }
        if let Some(&r) = memo.get(&b) {
            return r;
        }
        let lo = self.forget(lo, from, memo);
        let hi = self.forget(hi, from, memo);
        let r = if self.born[v as usize] >= from {
            self.or(lo, hi)
        } else {
            self.mk(v, lo, hi)
        };
        memo.insert(b, r);
        r
    }

    fn restrict(&mut self, b: Bool, v: u32, value: bool, memo: &mut Map<Bool, Bool>) -> Bool {
        let (bv, lo, hi) = self.nodes[b.0 as usize];
        if bv == TERMINAL || bv > v {
            return b;
        }
        if bv == v {
            return if value { hi } else { lo };
        }
        if let Some(&r) = memo.get(&b) {
            return r;
        }
        let l = self.restrict(lo, v, value, memo);
        let h = self.restrict(hi, v, value, memo);
        let r = self.mk(bv, l, h);
        memo.insert(b, r);
        r
    }

    /// `b` with each atom that `known` decides replaced by its value.
    fn given(&mut self, mut b: Bool, known: Bool) -> Bool {
        for v in self.vars(b) {
            let x = self.mk(v, Bool::FALSE, Bool::TRUE);
            let nx = self.not(x);
            if self.implies(known, x) {
                b = self.restrict(b, v, true, &mut Map::default());
            } else if self.implies(known, nx) {
                b = self.restrict(b, v, false, &mut Map::default());
            }
        }
        b
    }
}

/// How a lane leaves a block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaneExit {
    Jump(usize),
    /// To `taken` in lanes where `when` holds. A condition over
    /// [`Atom::Uniform`]s only is a uniform branch.
    Branch {
        when: Bool,
        taken: usize,
        not_taken: usize,
    },
    End,
    Indirect,
}

/// A [`Program`] as one lane runs it: same blocks, per-lane branches.
#[derive(Debug, Clone)]
pub struct Structured {
    pub bools: Bools,
    /// Per block.
    pub exits: Vec<LaneExit>,
    /// Per block and instruction: the condition, beyond reaching it, under
    /// which a lane runs a vector instruction; `None` when reaching it is
    /// enough (lanes outside EXEC then never get there, so a VGPR write
    /// needs no old value). Always `None` for scalar instructions.
    pub guards: Vec<Vec<Option<Bool>>>,
    /// Per block: what holds in every lane that enters it (over forward
    /// edges; a loop header forgets what its loop redefines).
    pub reach: Vec<Bool>,
    /// The lane's bit of each lane-mask value, by its low dword's value.
    pub masks: HashMap<Value, Bool>,
    /// Exact Wave64 pairs for checking masks at their uses.
    paired_masks: Map<Key, Bool>,
    /// Wave64 masks: low dword's value by the high dword's.
    pub highs: HashMap<Value, Value>,
    /// Masks (low dwords) set only in lanes whose EXEC bit is set where
    /// they are written: every such lane reaches the writing block, so a
    /// ballot there gives the mask as data.
    pub active: HashSet<Value>,
    /// Masks (low dwords) whose value comes from a mask join resolved to a
    /// constant ("every lane arriving here has its bit clear", or set),
    /// itself or through mask operations. Their per-lane bits hold in the
    /// lanes that reach where they are read, not in the others: a loop's
    /// exit mask is 0 in every lane still in the loop, but holds the lanes
    /// that left it. None of them is in [`Self::active`]; read as data they
    /// need every lane of the wave (the console flow). EXEC is never one:
    /// it holds only the lanes that run where it is read.
    pub resolved: HashSet<Value>,
    /// Per block and instruction: the lane's EXEC bit at each vector
    /// instruction and scalar memory read, whatever path the lane took
    /// (`None` for other scalar instructions). What a program run as the
    /// console runs it, every lane through every block, guards its vector
    /// instructions with ([`structure_console`]); a scalar read through a
    /// descriptor that differs between lanes needs it too.
    pub execs: Vec<Vec<Option<Bool>>>,
    /// The lane's bit of values read as masks that derive from neither
    /// EXEC nor a VALU result (constants, uniform selects, opaque ones), by
    /// low dword. What a mask join takes on an edge such a value arrives by:
    /// a loop's exit mask starts as a constant 0 before lanes are added.
    pub others: HashMap<Value, Bool>,
    /// The high dword read with each low dword of a wave64 mask in
    /// [`Self::masks`] or [`Self::others`], where the two are different
    /// values: the halves of a mask held as a number need not come from
    /// one instruction.
    pub pairs: HashMap<Value, Value>,
    /// The SCC of each lane-mask operation as the console wave sets it:
    /// whether some lane has the lane bit given, negated where the flag
    /// says. A mask operation's SCC is "the result is not zero" (for
    /// `s_and_saveexec` and the like, the new EXEC's, not the saved one's);
    /// `s_cmp_eq` of two masks is "no lane differs", `s_cmp_lg` "some lane
    /// does". What the console flow branches on (`s_cbranch_scc*`).
    pub sccs: HashMap<Value, (Bool, bool)>,
}

impl Structured {
    pub(crate) fn mask_bits(&self, low: Value, high: Option<Value>) -> Option<Bool> {
        match high {
            Some(high) => self.paired_masks.get(&(low, high)),
            None => self.masks.get(&low),
        }
        .copied()
    }
}

/// Builds the per-lane form of `program`.
pub fn structure(program: &Program) -> Result<Structured, ShaderError> {
    run(Lanes::new(program, false)?)
}

/// The same for a program run as the console runs it: every lane of the
/// console wave through every block, the branches taken wave-wide. Masks
/// joined at a block are never resolved from what the lanes arriving there
/// know, since lanes outside EXEC arrive too: each join that differs
/// between edges is a phi ([`Atom::Phi`]). Guard vector instructions with
/// [`Structured::execs`] and ignore [`Structured::exits`]' conditions.
pub fn structure_console(program: &Program) -> Result<Structured, ShaderError> {
    match run(Lanes::new(program, true)?) {
        // The console flow takes every branch with the whole wave and guards
        // each vector instruction with EXEC (`execs`): what a lane knows on
        // entering a block only narrows guards and branch conditions that
        // flow does not use. Where that knowledge grows too large (a long
        // program, every mask join a phi there), the console form goes
        // without it: each block entered knowing nothing.
        Err(e) if e == TOO_LARGE => {
            let mut l = Lanes::new(program, true)?;
            l.unknowing = true;
            run(l)
        }
        r => r,
    }
}

fn run(mut l: Lanes) -> Result<Structured, ShaderError> {
    loop {
        let execs = l.masks_in_order();
        while let Some(key) = l.console_pending.pop() {
            l.mask(key);
        }
        let cond = if l.unknowing {
            vec![Bool::TRUE; l.p.blocks.len()]
        } else {
            l.paths()?
        };
        let wrong = l.wrong_joins(&cond);
        if wrong.is_empty() {
            return l.finish(&execs, &cond);
        }
        l.unresolved.extend(wrong);
        l.reset();
    }
}

/// A lane mask by the values of its low and high dword (both the low one in
/// wave32).
type Key = (Value, Value);

struct Lanes<'a> {
    p: &'a Program,
    /// See [`structure_console`].
    console: bool,
    /// The console form without what lanes know on entering each block
    /// (see [`structure_console`]).
    unknowing: bool,
    console_pending: Vec<Key>,
    order: Vec<usize>,
    /// Reverse-postorder index per block.
    index: Vec<usize>,
    /// Per block, in predecessor order: (block, over its taken edge).
    incoming: Vec<Vec<(usize, bool)>>,
    /// Block and instruction index writing each value.
    defs: Map<Value, (usize, usize)>,
    /// Block and index of each phi value.
    phis: Map<Value, (usize, usize)>,
    /// Mask joins found wrong when resolved to a constant: atoms instead.
    unresolved: HashSet<Value>,
    b: Bools,
    /// Masks derived from EXEC and VALU results.
    masks: Map<Key, Bool>,
    /// Other values read as masks: constants, uniform selects, opaque ones.
    other: Map<Key, Bool>,
    /// For branches: SCC of a mask operation ("some lane is set") as the
    /// lane's own bit.
    scc_lane: Map<Value, Bool>,
    /// See [`Structured::sccs`].
    scc_wave: Map<Value, (Bool, bool)>,
    /// Joins resolved to a constant, to check once all edges are known.
    joins: Vec<Join>,
    /// Per block: what a lane knows on entry, over forward edges only.
    first: Vec<Bool>,
    /// Per block ending in a branch: when a lane takes it.
    when: Vec<Option<Bool>>,
    /// See [`Structured::active`].
    active: HashSet<Value>,
    /// See [`Structured::resolved`].
    resolved: HashSet<Key>,
    /// Values consumed (directly or through mask SOPs/phis) by an actual
    /// lane-mask consumer.
    mask_used: HashSet<Value>,
    /// Values flowing into a non-EXEC mask op/compare against a partial constant
    /// (`Atom::Constant` candidate).
    const_bit_tested: HashSet<Value>,
    /// Non-EXEC joins mixing a lane mask with computed/loaded scalar data
    /// without feeding a lane-mask consumer; kept out of `masks`.
    mixed_data: HashSet<Key>,
}

/// A mask join taken as `value` in every lane that reaches `block`: each
/// incoming mask held (or failed) on its edge.
struct Join {
    block: usize,
    /// The phi value naming the join.
    id: Value,
    value: bool,
    args: Vec<Key>,
}

struct SopMasks {
    dst: Option<(Key, Bool)>,
    exec: Option<(Key, Bool)>,
    /// Some operand derives from EXEC or a VALU result.
    derived: bool,
    /// For branches: SCC ("result is not zero") per lane.
    scc: Option<Bool>,
    /// `dst` comes from a join resolved to a constant ([`Structured::resolved`]).
    resolved: bool,
}

const MAX_ROUNDS: usize = 256;

impl<'a> Lanes<'a> {
    fn new(p: &'a Program, console: bool) -> Result<Self, ShaderError> {
        let blocks = &p.blocks;
        let order = crate::ir::reverse_postorder(blocks);
        let mut index = vec![usize::MAX; blocks.len()];
        for (i, &b) in order.iter().enumerate() {
            index[b] = i;
        }
        check_reducible(p, &order, &index)?;
        let mut incoming = vec![Vec::new(); blocks.len()];
        let (mut defs, mut phis) = (Map::default(), Map::default());
        for (b, block) in blocks.iter().enumerate() {
            for (k, s) in block.exit.successors().into_iter().enumerate() {
                incoming[s].push((b, k == 0));
            }
            for (i, phi) in block.phis.iter().enumerate() {
                phis.insert(phi.value, (b, i));
            }
            for (i, inst) in block.insts.iter().enumerate() {
                for &(_, v) in &inst.writes {
                    defs.insert(v, (b, i));
                }
            }
        }
        let mut mask_used = HashSet::new();
        let mut mask_work = Vec::new();
        let mut const_bit_tested = HashSet::new();
        let mut test_work = Vec::new();
        for block in blocks {
            for inst in &block.insts {
                let name = inst.name();
                match &inst.fields {
                    Fields::Sop { src, .. }
                        if inst.write(Reg::S(EXEC)).is_some()
                            || inst.write(Reg::S(EXEC + 1)).is_some()
                            || name.starts_with("s_wqm_")
                            || is_mask_query(name) =>
                    {
                        for s in src {
                            push_scalar_op(inst, s, p.wave64, &mut mask_used, &mut mask_work);
                        }
                    }
                    Fields::Sopp { .. } if name.starts_with("s_cbranch_vcc") => {
                        for reg in [Reg::S(VCC), Reg::S(VCC + 1)] {
                            if let Some(v) = inst.read(reg)
                                && mask_used.insert(v)
                            {
                                mask_work.push(v);
                            }
                        }
                    }
                    Fields::Valu(v) if name == "v_cndmask_b32" || name.contains("_co_ci_") => {
                        if let Some(s) = v.src.get(2) {
                            push_scalar_op(inst, s, p.wave64, &mut mask_used, &mut mask_work);
                        }
                    }
                    Fields::Valu(v) if name.starts_with("v_mbcnt_") => {
                        if let Some(s) = v.src.first() {
                            push_scalar_op(inst, s, p.wave64, &mut mask_used, &mut mask_work);
                        }
                    }
                    _ => {}
                }
                if let Fields::Sop { src, .. } = &inst.fields
                    && (is_mask_sop(name, p.wave64) || is_mask_cmp(name, p.wave64))
                    && inst.write(Reg::S(EXEC)).is_none()
                    && inst.write(Reg::S(EXEC + 1)).is_none()
                    && src.iter().any(|s| is_partial_const(s, p.wave64))
                {
                    for s in src {
                        push_scalar_op(inst, s, p.wave64, &mut const_bit_tested, &mut test_work);
                    }
                }
            }
        }
        propagate_mask_uses(p, &phis, &defs, &mut mask_used, &mut mask_work);
        propagate_mask_uses(p, &phis, &defs, &mut const_bit_tested, &mut test_work);
        Ok(Lanes {
            p,
            console,
            unknowing: false,
            console_pending: Vec::new(),
            order,
            index,
            incoming,
            defs,
            phis,
            unresolved: HashSet::new(),
            b: Bools::new(),
            masks: Map::default(),
            other: Map::default(),
            scc_lane: Map::default(),
            scc_wave: Map::default(),
            joins: Vec::new(),
            first: vec![Bool::FALSE; blocks.len()],
            when: vec![None; blocks.len()],
            active: HashSet::new(),
            resolved: HashSet::new(),
            mask_used,
            const_bit_tested,
            mixed_data: HashSet::new(),
        })
    }

    fn reset(&mut self) {
        self.b = Bools::new();
        self.masks.clear();
        self.other.clear();
        self.scc_lane.clear();
        self.scc_wave.clear();
        self.joins.clear();
        self.first.fill(Bool::FALSE);
        self.when.fill(None);
        self.active.clear();
        self.resolved.clear();
        self.mixed_data.clear();
    }

    /// Lane masks, branch conditions and forward-edge knowledge, block by
    /// block in reverse postorder; returns EXEC at each instruction reading it.
    fn masks_in_order(&mut self) -> Vec<Vec<Option<Bool>>> {
        let p = self.p;
        let input = |reg| p.inputs.iter().find(|i| i.0 == reg).map(|i| i.1);
        if let Some(lo) = input(Reg::S(EXEC)) {
            let hi = if p.wave64 {
                input(Reg::S(EXEC + 1)).unwrap_or(lo)
            } else {
                lo
            };
            self.masks.insert((lo, hi), Bool::TRUE);
            self.active.insert(lo);
        }
        let mut execs: Vec<Vec<Option<Bool>>> =
            p.blocks.iter().map(|b| vec![None; b.insts.len()]).collect();
        // EXEC's values (low and high dword, where known) at the end of each
        // block, for the scalar memory reads, which do not read it.
        type Halves = (Option<Value>, Option<Value>);
        let high = if p.wave64 { EXEC + 1 } else { EXEC };
        let entry: Halves = (input(Reg::S(EXEC)), input(Reg::S(high)));
        let mut exec_out: Vec<Halves> = vec![(None, None); p.blocks.len()];
        for n in 0..self.order.len() {
            let b = self.order[n];
            let mut c = if n == 0 || self.unknowing {
                Bool::TRUE
            } else {
                Bool::FALSE
            };
            for k in 0..self.incoming[b].len() {
                let (q, taken) = self.incoming[b][k];
                if self.index[q] < n && !self.unknowing {
                    let e = self.edge(self.first[q], q, taken);
                    c = self.b.or(c, e);
                }
            }
            self.first[b] = c;
            // EXEC joined at the block's entry holds the lanes that arrive.
            for phi in &p.blocks[b].phis {
                if phi.reg == Reg::S(EXEC) {
                    self.active.insert(phi.value);
                }
            }
            // EXEC's values entering the block: its phis', else those a
            // forward predecessor leaves with (they all agree then).
            let half = |reg: u8, pick: fn(Halves) -> Option<Value>| {
                let phi = p.blocks[b].phis.iter().find(|q| q.reg == Reg::S(reg));
                match phi {
                    Some(phi) => Some(phi.value),
                    None if n == 0 => pick(entry),
                    None => self.incoming[b]
                        .iter()
                        .filter(|&&(q, _)| self.index[q] < n)
                        .find_map(|&(q, _)| pick(exec_out[q])),
                }
            };
            let mut values: Halves = (half(EXEC, |h| h.0), half(high, |h| h.1));
            // EXEC where known in this block (read or written so far).
            let mut now = None;
            for (i, inst) in p.blocks[b].insts.iter().enumerate() {
                if inst.read(Reg::S(EXEC)).is_some() {
                    let e = self.exec(inst);
                    execs[b][i] = Some(e);
                    now = Some(e);
                } else if matches!(inst.fields, Fields::Smem { .. })
                    && let (Some(lo), Some(hi)) = values
                {
                    execs[b][i] = Some(self.mask((lo, hi)).0);
                }
                self.step(inst, now);
                if let Some(k) = self.key(|r| inst.write(r), EXEC) {
                    now = self.masks.get(&k).or(self.other.get(&k)).copied();
                }
                // A write of one dword keeps the other's value.
                values = (
                    inst.write(Reg::S(EXEC)).or(values.0),
                    inst.write(Reg::S(high)).or(values.1),
                );
            }
            exec_out[b] = values;
            if let Exit::Branch { .. } = p.blocks[b].exit
                && let Some(last) = p.blocks[b].insts.last()
            {
                self.when[b] = Some(self.branch(last));
            }
        }
        execs
    }

    /// `c` narrowed by the branch at the end of `q` towards one successor.
    fn edge(&mut self, c: Bool, q: usize, taken: bool) -> Bool {
        match self.when[q] {
            Some(w) if taken => self.b.and(c, w),
            Some(w) => {
                let nw = self.b.not(w);
                self.b.and(c, nw)
            }
            None => c,
        }
    }

    /// Joins whose constant fails on some edge, given what lanes know.
    fn wrong_joins(&mut self, cond: &[Bool]) -> Vec<Value> {
        let mut wrong = Vec::new();
        let mut j = 0;
        // Checking an argument can resolve another join; it is checked too.
        while j < self.joins.len() {
            let (block, id, value) = (self.joins[j].block, self.joins[j].id, self.joins[j].value);
            for (n, a) in self.joins[j].args.clone().into_iter().enumerate() {
                let (q, taken) = self.incoming[block][n];
                let e = self.edge(cond[q], q, taken);
                let mut m = self.mask(a).0;
                if !value {
                    m = self.b.not(m);
                }
                if !self.b.implies(e, m) {
                    wrong.push(id);
                    break;
                }
            }
            j += 1;
        }
        wrong
    }

    fn finish(
        mut self,
        execs: &[Vec<Option<Bool>>],
        cond: &[Bool],
    ) -> Result<Structured, ShaderError> {
        let blocks = &self.p.blocks;
        let b = &mut self.b;
        let guards = blocks
            .iter()
            .enumerate()
            .map(|(n, block)| {
                block
                    .insts
                    .iter()
                    .zip(&execs[n])
                    .map(|(inst, exec)| match (exec, &inst.fields) {
                        (None, _)
                        | (_, Fields::Sop { .. } | Fields::Sopp { .. } | Fields::Smem { .. }) => {
                            None
                        }
                        (&Some(e), _) if b.implies(cond[n], e) => None,
                        (&Some(e), _) => Some(b.given(e, cond[n])),
                    })
                    .collect()
            })
            .collect();
        let exits = blocks
            .iter()
            .enumerate()
            .map(|(n, block)| match block.exit {
                Exit::Jump(t) => LaneExit::Jump(t),
                Exit::End => LaneExit::End,
                Exit::Indirect => LaneExit::Indirect,
                Exit::Branch { taken, not_taken } => LaneExit::Branch {
                    when: b.given(self.when[n].unwrap_or(Bool::FALSE), cond[n]),
                    taken,
                    not_taken,
                },
            })
            .collect();
        if self.b.overflow {
            return Err(TOO_LARGE);
        }
        Ok(Structured {
            masks: self.masks.iter().map(|(&(lo, _), &m)| (lo, m)).collect(),
            highs: self
                .masks
                .keys()
                .filter(|(lo, hi)| lo != hi)
                .map(|&(lo, hi)| (hi, lo))
                .collect(),
            active: self.active,
            resolved: self.resolved.iter().map(|k| k.0).collect(),
            bools: self.b,
            exits,
            guards,
            execs: execs.to_vec(),
            others: self.other.iter().map(|(&(lo, _), &m)| (lo, m)).collect(),
            pairs: self
                .masks
                .keys()
                .chain(self.other.keys())
                .filter(|(lo, hi)| lo != hi)
                .copied()
                .collect(),
            sccs: self.scc_wave.into_iter().collect(),
            paired_masks: if self.p.wave64 {
                self.masks
            } else {
                Map::default()
            },
            reach: cond.to_vec(),
        })
    }

    fn key(&self, value: impl Fn(Reg) -> Option<Value>, code: u8) -> Option<Key> {
        let lo = value(Reg::S(code))?;
        let hi = if self.p.wave64 {
            value(Reg::S(code + 1))?
        } else {
            lo
        };
        Some((lo, hi))
    }

    fn born(&self, v: Value) -> usize {
        match (self.defs.get(&v), self.phis.get(&v)) {
            (Some(&(b, _)), _) | (_, Some(&(b, _))) => self.index[b],
            _ => 0,
        }
    }

    fn exec(&mut self, inst: &Inst) -> Bool {
        match self.key(|r| inst.read(r), EXEC) {
            Some(k) => self.mask(k).0,
            None => Bool::TRUE,
        }
    }

    /// The mask a value pair holds, and whether it derives from EXEC or a
    /// VALU result.
    fn mask(&mut self, k: Key) -> (Bool, bool) {
        if let Some(&m) = self.masks.get(&k) {
            return (m, true);
        }
        if let Some(&m) = self.other.get(&k) {
            return (m, false);
        }
        let p = self.p;
        let (m, derived) = match self.join(k) {
            Some(join) => join,
            None => {
                let (lo, hi) = (self.defs.get(&k.0).copied(), self.defs.get(&k.1).copied());
                let def = |d: Option<(usize, usize)>| d.map(|(b, i)| &p.blocks[b].insts[i]);
                let m = match (lo, hi) {
                    (Some(at), Some(hi)) if at == hi => {
                        def(lo).and_then(|i| self.sop(i)).and_then(|r| {
                            if r.resolved {
                                self.resolved.insert(k);
                            }
                            r.dst.map(|d| d.1)
                        })
                    }
                    _ => None,
                };
                // What an EXEC write the pass does not follow leaves there.
                let exec_write = |d: Option<(usize, usize)>, v: Value| {
                    def(d).is_some_and(|i| {
                        i.write(Reg::S(EXEC)) == Some(v) || i.write(Reg::S(EXEC + 1)) == Some(v)
                    })
                };
                let derived = exec_write(lo, k.0) || exec_write(hi, k.1);
                let m = m.unwrap_or_else(|| {
                    let born = self.born(k.0);
                    self.b.atom(Atom::Opaque(k.0), born)
                });
                (m, derived)
            }
        };
        let cache = if derived {
            &mut self.masks
        } else {
            &mut self.other
        };
        cache.insert(k, m);
        (m, derived)
    }

    /// A mask joined by phis at a block entry, and whether a mask coming in
    /// on a forward edge derives from EXEC or a VALU result. Where every
    /// lane arriving so far has the incoming mask set (or clear), the join
    /// is that constant, for a lane can only read it after arriving.
    fn join(&mut self, k: Key) -> Option<(Bool, bool)> {
        let p = self.p;
        let (lo, hi) = (self.phis.get(&k.0).copied(), self.phis.get(&k.1).copied());
        let block = match (lo, hi) {
            (Some((a, _)), Some((b, _))) if a != b => return None,
            (Some((b, _)), _) | (_, Some((b, _))) => b,
            _ => return None,
        };
        let target = &p.blocks[block];
        let arg = |phi: Option<(usize, usize)>, own: Value, n: usize| {
            phi.map_or(own, |(_, i)| target.phis[i].args[n])
        };
        let args: Vec<Key> = (0..target.preds.len())
            .map(|n| (arg(lo, k.0, n), arg(hi, k.1, n)))
            .collect();
        let id = lo.map_or(k.1, |_| k.0);
        // EXEC holds only the lanes that run where it is read: what holds in
        // every lane arriving here is all of it (see `Structured::resolved`).
        let exec = [lo, hi]
            .into_iter()
            .flatten()
            .any(|(_, i)| matches!(target.phis[i].reg, Reg::S(c) if c == EXEC || c == EXEC + 1));
        let mut derived = false;
        let mut opaque_data = false;
        // Some mask coming in is one of `Structured::resolved`.
        let mut pinned = false;
        // (edge condition, mask) per forward edge; back edges come later.
        let mut forward = Vec::new();
        for (n, &a) in args.iter().enumerate() {
            let (q, taken) = self.incoming[block][n];
            if self.index[q] < self.index[block] {
                let (m, d) = self.mask(a);
                derived |= d;
                opaque_data |= (!d && has_defined_opaque(&self.b, &self.defs, m))
                    || self.mixed_data.contains(&a);
                pinned |= self.resolved.contains(&a);
                let e = self.edge(self.first[q], q, taken);
                forward.push((e, m));
            }
        }
        if !exec
            && opaque_data
            && (self.const_bit_tested.contains(&k.0) || self.const_bit_tested.contains(&k.1))
            && !self.mask_used.contains(&k.0)
            && !self.mask_used.contains(&k.1)
        {
            self.mixed_data.insert(k);
            let born = self.index[block];
            return Some((self.b.atom(Atom::Opaque(id), born), false));
        }
        if forward.len() == args.len()
            && let Some(&(_, m)) = forward.first()
            && forward.iter().all(|f| f.1 == m)
        {
            if pinned && !exec {
                self.resolved.insert(k);
            }
            return Some((m, derived));
        }
        if !self.console && !self.unresolved.contains(&id) {
            for value in [true, false] {
                let b = &mut self.b;
                let holds = forward.iter().all(|&(e, m)| {
                    let m = if value { m } else { b.not(m) };
                    b.implies(e, m)
                });
                if holds {
                    self.joins.push(Join {
                        block,
                        id,
                        value,
                        args,
                    });
                    if !exec {
                        self.resolved.insert(k);
                    }
                    let m = if value { Bool::TRUE } else { Bool::FALSE };
                    return Some((m, derived));
                }
            }
        }
        if self.console {
            self.console_pending.extend(args);
        }
        let born = self.index[block];
        Some((self.b.atom(Atom::Phi(id), born), derived))
    }

    fn operand(&mut self, inst: &Inst, op: &Operand) -> Option<(Bool, bool)> {
        let wave64 = self.p.wave64;
        let constant = |b: &mut Bools, v: u64| {
            let v = if wave64 { v } else { v & 0xFFFF_FFFF };
            match v {
                0 => Bool::FALSE,
                _ if v == u64::MAX >> (32 * u8::from(!wave64)) => Bool::TRUE,
                _ => b.atom(Atom::Constant(v), 0),
            }
        };
        Some(match *op {
            Operand::Scalar { code, .. } if code != NULL => {
                let k = self.key(|r| inst.read(r), code)?;
                self.mask(k)
            }
            Operand::Const(c @ 128..=208) => {
                let v = if c <= 192 {
                    i64::from(c) - 128
                } else {
                    192 - i64::from(c)
                };
                (constant(&mut self.b, v as u64), false)
            }
            Operand::Literal(x) => (constant(&mut self.b, u64::from(x)), false),
            _ => return None,
        })
    }

    /// Lane-mask meaning of a lane-width scalar operation.
    fn sop(&mut self, inst: &Inst) -> Option<SopMasks> {
        let Fields::Sop { dst, src, .. } = &inst.fields else {
            return None;
        };
        let width = if self.p.wave64 { "_b64" } else { "_b32" };
        let op = inst.name().strip_prefix("s_")?.strip_suffix(width)?;
        let dst = match dst {
            Some(Operand::Scalar { code, .. }) if *code != NULL => {
                self.key(|r| inst.write(r), *code)
            }
            _ => None,
        };
        let exec_out = self.key(|r| inst.write(r), EXEC);
        let (save, f) = match (op.strip_suffix("_saveexec"), op.strip_suffix("_wrexec")) {
            (Some(f), _) => (Some(true), f),
            (_, Some(f)) => (Some(false), f),
            _ => (None, op),
        };
        let mut derived = exec_out.is_some() || save.is_some();
        let mut arg = |n: usize, l: &mut Self| {
            let (m, d) = l.operand(inst, src.get(n)?)?;
            derived |= d;
            Some(m)
        };
        if let Some(saveexec) = save {
            let s = arg(0, self)?;
            let exec = self.exec(inst);
            let new = self.combine(f, s, exec)?;
            let saved = if saveexec { exec } else { new };
            // The saved EXEC is EXEC; a new one ANDed with EXEC keeps no lane
            // outside it.
            let resolved =
                !saveexec && !matches!(f, "and" | "andn1") && self.pinned(inst, src.first());
            return Some(SopMasks {
                dst: dst.map(|d| (d, saved)),
                exec: exec_out.map(|e| (e, new)),
                derived,
                scc: Some(new),
                resolved,
            });
        }
        let (result, scc) = match op {
            "mov" => (arg(0, self)?, None),
            "not" => {
                let a = arg(0, self)?;
                let r = self.b.not(a);
                (r, Some(r))
            }
            "wqm" => {
                let a = arg(0, self)?;
                let lo = dst.or(exec_out)?.0;
                let born = self.born(lo);
                let q = self.b.atom(Atom::Quad(lo), born);
                let r = self.b.or(a, q);
                (r, Some(r))
            }
            "cselect" => {
                let (a, c) = (arg(0, self)?, arg(1, self)?);
                let scc = inst.read(Reg::Scc)?;
                let born = self.born(scc);
                let u = self.b.atom(Atom::Uniform(scc), born);
                (self.b.ite(u, a, c), None)
            }
            _ => {
                let (a, c) = (arg(0, self)?, arg(1, self)?);
                let r = self.combine(op, a, c)?;
                (r, Some(r))
            }
        };
        // ANDed with EXEC itself, a mask has no lane outside EXEC, which
        // every lane that runs here holds: its bits are right in every lane.
        let exec_src =
            |n: usize| matches!(src.get(n), Some(&Operand::Scalar { code, .. }) if code == EXEC);
        let masked = match op {
            "and" => exec_src(0) || exec_src(1),
            "andn2" => exec_src(0),
            _ => false,
        };
        let resolved = dst != exec_out && !masked && src.iter().any(|o| self.pinned(inst, Some(o)));
        Some(SopMasks {
            dst: dst.map(|d| (d, result)),
            exec: exec_out.map(|e| (e, result)),
            derived,
            scc,
            resolved,
        })
    }

    /// Whether operand `op` of `inst` reads a mask of
    /// [`Structured::resolved`].
    fn pinned(&self, inst: &Inst, op: Option<&Operand>) -> bool {
        match op {
            Some(&Operand::Scalar { code, .. }) if code != NULL => self
                .key(|r| inst.read(r), code)
                .is_some_and(|k| self.resolved.contains(&k)),
            _ => false,
        }
    }

    fn combine(&mut self, f: &str, a: Bool, c: Bool) -> Option<Bool> {
        let b = &mut self.b;
        Some(match f {
            "and" => b.and(a, c),
            "or" => b.or(a, c),
            "xor" => b.xor(a, c),
            "andn2" => {
                let nc = b.not(c);
                b.and(a, nc)
            }
            "orn2" => {
                let nc = b.not(c);
                b.or(a, nc)
            }
            "andn1" => {
                let na = b.not(a);
                b.and(na, c)
            }
            "orn1" => {
                let na = b.not(a);
                b.or(na, c)
            }
            "nand" => {
                let r = b.and(a, c);
                b.not(r)
            }
            "nor" => {
                let r = b.or(a, c);
                b.not(r)
            }
            "xnor" => {
                let r = b.xor(a, c);
                b.not(r)
            }
            _ => return None,
        })
    }

    /// Records the masks `inst` defines; `exec` is EXEC there, if known.
    fn step(&mut self, inst: &Inst, exec: Option<Bool>) {
        let name = inst.name();
        match &inst.fields {
            Fields::Sop { src, .. } => {
                let mut takes_mask = false;
                for s in src {
                    if let Operand::Scalar { code, .. } = *s
                        && code != NULL
                        && let Some(k) = self.key(|r| inst.read(r), code)
                    {
                        let phi = self.phis.contains_key(&k.0) || self.phis.contains_key(&k.1);
                        takes_mask |= self.masks.contains_key(&k) || phi && self.mask(k).1;
                    }
                }
                let touches_exec =
                    inst.write(Reg::S(EXEC)).is_some() || inst.read(Reg::S(EXEC)).is_some();
                if (takes_mask || touches_exec)
                    && let Some(r) = self.sop(inst).filter(|r| r.derived)
                {
                    let pinned = r.dst.map(|d| d.0).filter(|_| r.resolved);
                    for (k, m) in r.dst.into_iter().chain(r.exec) {
                        self.masks.insert(k, m);
                        if pinned == Some(k) {
                            // Its bits are right only in the lanes here
                            // (see `Structured::resolved`): no ballot here
                            // holds the others'.
                            self.resolved.insert(k);
                        } else if r.exec.is_some_and(|e| e.0 == k)
                            || exec.is_some_and(|e| self.b.implies(m, e))
                        {
                            // EXEC itself holds the lanes that run here.
                            self.active.insert(k.0);
                        }
                    }
                    if let (Some(scc), Some(m)) = (inst.write(Reg::Scc), r.scc) {
                        self.scc_lane.insert(scc, m);
                        self.scc_wave.insert(scc, (m, false));
                    }
                }
                // `s_cmp_eq/lg` of a mask: "no lane differs" per lane.
                let width = if self.p.wave64 { "64" } else { "32" };
                if takes_mask
                    && let (Some(rest), Some(scc)) =
                        (name.strip_prefix("s_cmp_"), inst.write(Reg::Scc))
                    && rest
                        .strip_suffix(width)
                        .is_some_and(|r| r == "eq_u" || r == "lg_u")
                    && let [a, c] = &src[..]
                    && let (Some((a, _)), Some((c, _))) =
                        (self.operand(inst, a), self.operand(inst, c))
                {
                    let ne = self.b.xor(a, c);
                    let eq = rest.starts_with("eq");
                    let m = if eq { self.b.not(ne) } else { ne };
                    self.scc_lane.insert(scc, m);
                    self.scc_wave.insert(scc, (ne, eq));
                }
            }
            Fields::Valu(v) => {
                let exec = self.exec(inst);
                let lane = |l: &mut Self, k: Key| {
                    let born = l.born(k.0);
                    let a = l.b.atom(Atom::Lane(k.0), born);
                    let m = l.b.and(exec, a);
                    l.masks.insert(k, m);
                    l.active.insert(k.0);
                };
                // A select reads a mask nothing else may: a join of compare
                // results is resolved here, as for EXEC.
                if name == "v_cndmask_b32"
                    && let Some(&Operand::Scalar { code, .. }) = v.src.get(2)
                    && code != NULL
                    && let Some(k) = self.key(|r| inst.read(r), code)
                {
                    self.mask(k);
                }
                if name.starts_with("v_cmpx") {
                    if let Some(k) = self.key(|r| inst.write(r), EXEC) {
                        lane(self, k);
                    }
                } else {
                    let compare = v.dst.filter(|_| name.starts_with("v_cmp"));
                    for op in compare.iter().chain(&v.carry) {
                        if let Operand::Scalar { code, .. } = *op
                            && let Some(k) = self.key(|r| inst.write(r), code)
                        {
                            lane(self, k);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// The per-lane condition under which a block's closing branch is taken.
    fn branch(&mut self, inst: &Inst) -> Bool {
        let exec = self.exec(inst);
        match inst.name() {
            "s_cbranch_execz" => self.b.not(exec),
            "s_cbranch_execnz" => exec,
            n @ ("s_cbranch_vccz" | "s_cbranch_vccnz") => {
                let vcc = match self.key(|r| inst.read(r), VCC) {
                    Some(k) => self.mask(k).0,
                    None => Bool::FALSE,
                };
                let any = self.b.and(vcc, exec);
                if n.ends_with("nz") {
                    any
                } else {
                    self.b.not(any)
                }
            }
            n => {
                let Some(scc) = inst.read(Reg::Scc) else {
                    return Bool::FALSE;
                };
                let set = match self.scc_lane.get(&scc) {
                    Some(&m) => m,
                    None => {
                        let born = self.born(scc);
                        self.b.atom(Atom::Uniform(scc), born)
                    }
                };
                if n == "s_cbranch_scc1" {
                    set
                } else {
                    self.b.not(set)
                }
            }
        }
    }

    /// What a lane knows on entering each block, to a fixpoint. A loop
    /// header forgets what the back edge says about atoms the loop redefines.
    fn paths(&mut self) -> Result<Vec<Bool>, ShaderError> {
        let n = self.p.blocks.len();
        let mut cond = vec![Bool::FALSE; n];
        if let Some(&entry) = self.order.first() {
            cond[entry] = Bool::TRUE;
        }
        for _ in 0..MAX_ROUNDS {
            let mut changed = false;
            for i in 1..self.order.len() {
                let s = self.order[i];
                let mut c = Bool::FALSE;
                for k in 0..self.incoming[s].len() {
                    let (q, taken) = self.incoming[s][k];
                    let mut e = self.edge(cond[q], q, taken);
                    if i <= self.index[q] {
                        e = self.b.forget(e, i, &mut Map::default());
                    }
                    c = self.b.or(c, e);
                }
                if c != cond[s] {
                    cond[s] = c;
                    changed = true;
                }
            }
            if self.b.overflow {
                return Err(TOO_LARGE);
            }
            if !changed {
                return Ok(cond);
            }
        }
        Err(ShaderError::Isa {
            offset: 0,
            reason: "lane conditions do not settle",
        })
    }
}

const TOO_LARGE: ShaderError = ShaderError::Isa {
    offset: 0,
    reason: "lane conditions too large",
};

/// `ir::build` splits irreducible loops; checked again because everything
/// here assumes a reducible graph.
fn check_reducible(p: &Program, order: &[usize], index: &[usize]) -> Result<(), ShaderError> {
    match crate::ir::irreducible_edge(&p.blocks, order, index) {
        Some((from, _)) => Err(ShaderError::Unsupported {
            offset: p.blocks[from].offset,
            what: "irreducible control flow",
        }),
        None => Ok(()),
    }
}

fn is_mask_combine(op: &str) -> bool {
    matches!(
        op,
        "and" | "or" | "xor" | "andn2" | "orn2" | "andn1" | "orn1" | "nand" | "nor" | "xnor"
    )
}

fn is_mask_sop(name: &str, wave64: bool) -> bool {
    let width = if wave64 { "_b64" } else { "_b32" };
    let Some(op) = name.strip_prefix("s_").and_then(|s| s.strip_suffix(width)) else {
        return false;
    };
    if let Some(op) = op
        .strip_suffix("_saveexec")
        .or_else(|| op.strip_suffix("_wrexec"))
    {
        return is_mask_combine(op);
    }
    matches!(op, "mov" | "not" | "wqm" | "cselect") || is_mask_combine(op)
}

fn is_mask_cmp(name: &str, wave64: bool) -> bool {
    let width = if wave64 { "64" } else { "32" };
    name.strip_prefix("s_cmp_")
        .and_then(|r| r.strip_suffix(width))
        .is_some_and(|r| r == "eq_u" || r == "lg_u")
}

fn is_mask_query(name: &str) -> bool {
    name.starts_with("s_bcnt") || name.starts_with("s_ff") || name.starts_with("s_flbit_i32_b")
}

fn is_partial_const(op: &Operand, wave64: bool) -> bool {
    let raw = match *op {
        Operand::Const(c @ 128..=208) => {
            let v = if c <= 192 {
                i64::from(c) - 128
            } else {
                192 - i64::from(c)
            };
            v as u64
        }
        Operand::Literal(x) => u64::from(x),
        _ => return false,
    };
    let v = if wave64 { raw } else { raw & 0xFFFF_FFFF };
    let all = u64::MAX >> (32 * u8::from(!wave64));
    v != 0 && v != all
}

fn push_scalar_op(
    inst: &Inst,
    op: &Operand,
    wave64: bool,
    used: &mut HashSet<Value>,
    work: &mut Vec<Value>,
) {
    if let Operand::Scalar { code, .. } = *op
        && code != NULL
    {
        if let Some(v) = inst.read(Reg::S(code))
            && used.insert(v)
        {
            work.push(v);
        }
        if wave64
            && let Some(v) = inst.read(Reg::S(code + 1))
            && used.insert(v)
        {
            work.push(v);
        }
    }
}

fn propagate_mask_uses(
    p: &Program,
    phis: &Map<Value, (usize, usize)>,
    defs: &Map<Value, (usize, usize)>,
    used: &mut HashSet<Value>,
    work: &mut Vec<Value>,
) {
    while let Some(v) = work.pop() {
        if let Some(&(b, i)) = phis.get(&v) {
            for &a in &p.blocks[b].phis[i].args {
                if used.insert(a) {
                    work.push(a);
                }
            }
        } else if let Some(&(b, i)) = defs.get(&v) {
            let inst = &p.blocks[b].insts[i];
            if is_mask_sop(inst.name(), p.wave64)
                && let Fields::Sop { src, .. } = &inst.fields
            {
                for s in src {
                    push_scalar_op(inst, s, p.wave64, used, work);
                }
            }
        }
    }
}

fn has_defined_opaque(b: &Bools, defs: &Map<Value, (usize, usize)>, x: Bool) -> bool {
    let Some((atom, lo, hi)) = b.node(x) else {
        return false;
    };
    if lo == Bool::FALSE && hi == Bool::TRUE {
        return matches!(atom, Atom::Opaque(v) if defs.contains_key(&v));
    }
    let mut stack = vec![x];
    let mut seen = HashSet::new();
    while let Some(cur) = stack.pop() {
        if !seen.insert(cur) {
            continue;
        }
        let Some((atom, lo, hi)) = b.node(cur) else {
            continue;
        };
        if let Atom::Opaque(v) = atom
            && defs.contains_key(&v)
        {
            return true;
        }
        stack.push(lo);
        stack.push(hi);
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir;

    const S_ENDPGM: u32 = 0xBF81_0000;

    fn build(words: &[u32], wave64: bool) -> Program {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        ir::build(&code, wave64, 256).unwrap()
    }

    fn guarded(s: &Structured) -> usize {
        s.guards.iter().flatten().filter(|g| g.is_some()).count()
    }

    #[test]
    fn divergent_if_becomes_a_lane_branch() {
        let p = build(
            &[
                0x7D84_0080, // v_cmp_eq_u32 vcc, 0, v0
                0xBE80_3C6A, // s_and_saveexec_b32 s0, vcc_lo
                0xBF88_0001, // s_cbranch_execz -> 16
                0x7E02_0280, // v_mov_b32 v1, 0
                0x887E_007E, // 16: s_or_b32 exec_lo, exec_lo, s0
                0x7E04_0280, // v_mov_b32 v2, 0
                S_ENDPGM,
            ],
            false,
        );
        let s = structure(&p).unwrap();
        let LaneExit::Branch {
            when,
            taken: 2,
            not_taken: 1,
        } = s.exits[0]
        else {
            panic!("{:?}", s.exits)
        };
        // Skipped exactly where the compare fails.
        let cmp = p.blocks[0].insts[0].write(Reg::S(VCC)).unwrap();
        assert_eq!(
            s.bools.node(when),
            Some((Atom::Lane(cmp), Bool::TRUE, Bool::FALSE))
        );
        assert_eq!(guarded(&s), 0, "{:?}", s.guards);
    }

    #[test]
    fn exec_narrowed_without_a_branch_leaves_a_guard() {
        let p = build(
            &[
                0x7DA4_0080, // v_cmpx_eq_u32 exec, 0, v0
                0x7E02_0280, // v_mov_b32 v1, 0
                S_ENDPGM,
            ],
            true,
        );
        let s = structure(&p).unwrap();
        assert_eq!(s.guards[0][0], None, "the compare runs in every lane");
        let g = s.guards[0][1].expect("guard");
        assert!(matches!(s.bools.support(g)[..], [Atom::Lane(_)]));
    }

    #[test]
    fn divergent_loop_exits_per_lane() {
        let p = build(
            &[
                0xBE82_0480, // s_mov_b64 s[2:3], 0
                0x4A00_0081, // 4: v_add_nc_u32 v0, 1, v0
                0x7D84_0080, // v_cmp_eq_u32 vcc, 0, v0
                0x8882_026A, // s_or_b64 s[2:3], vcc, s[2:3]
                0x8AFE_027E, // s_andn2_b64 exec, exec, s[2:3]
                0xBF89_FFFB, // s_cbranch_execnz -> 4
                0x88FE_027E, // s_or_b64 exec, exec, s[2:3]
                0x7E02_0280, // v_mov_b32 v1, 0
                S_ENDPGM,
            ],
            true,
        );
        let s = structure(&p).unwrap();
        assert_eq!(guarded(&s), 0, "{:?}", s.guards);
        let LaneExit::Branch {
            when,
            taken: 1,
            not_taken: 2,
        } = s.exits[1]
        else {
            panic!("{:?}", s.exits)
        };
        // Back while the lane's compare fails.
        assert!(matches!(
            s.bools.node(when),
            Some((Atom::Lane(_), Bool::TRUE, Bool::FALSE))
        ));
    }

    /// A loop's exit mask grown from 0: its join at the header resolves to
    /// "no lane" (every lane still in the loop has its bit clear), but the
    /// mask holds the lanes that left. What the loop makes of it is no mask
    /// the lanes there hold whole: `resolved`, not `active`. EXEC stays
    /// active, and the console form resolves no join.
    #[test]
    fn a_loop_exit_mask_is_resolved_not_active() {
        let p = build(
            &[
                0xBE82_0480, // s_mov_b64 s[2:3], 0
                0x4A00_0081, // 4: v_add_nc_u32 v0, 1, v0
                0x7D84_0080, // v_cmp_eq_u32 vcc, 0, v0
                0x8882_026A, // s_or_b64 s[2:3], vcc, s[2:3]
                0x8AFE_027E, // s_andn2_b64 exec, exec, s[2:3]
                0xBF89_FFFB, // s_cbranch_execnz -> 4
                0x88FE_027E, // s_or_b64 exec, exec, s[2:3]
                S_ENDPGM,
            ],
            true,
        );
        let s = structure(&p).unwrap();
        let body = &p.blocks[1].insts;
        assert_eq!(body[2].name(), "s_or_b64");
        let exit = body[2].write(Reg::S(2)).unwrap();
        assert!(s.resolved.contains(&exit), "{:?}", s.resolved);
        assert!(!s.active.contains(&exit));
        let exec = body[3].write(Reg::S(EXEC)).unwrap();
        assert!(s.active.contains(&exec));
        assert!(!s.resolved.contains(&exec));
        assert!(structure_console(&p).unwrap().resolved.is_empty());
    }

    /// In wave64 a write of EXEC's low dword alone keeps the high one: the
    /// lanes after it are the pair (new low, old high), a mask held as a
    /// number whose high dword is EXEC's from before, for vector
    /// instructions and scalar memory reads alike.
    #[test]
    fn a_write_of_exec_lo_keeps_exec_hi() {
        let p = build(
            &[
                0xBEFE_0380, // s_mov_b32 exec_lo, 0
                0x7E02_0280, // v_mov_b32 v1, 0
                0xF400_0100,
                0xFA00_0000, // s_load_dword s4, s[0:1], 0x0
                S_ENDPGM,
            ],
            true,
        );
        let s = structure(&p).unwrap();
        let insts = &p.blocks[0].insts;
        let lo = insts[0].write(Reg::S(EXEC)).unwrap();
        assert_eq!(insts[0].write(Reg::S(EXEC + 1)), None);
        let hi = p.inputs.iter().find(|i| i.0 == Reg::S(EXEC + 1)).unwrap().1;
        assert_eq!(insts[1].read(Reg::S(EXEC + 1)), Some(hi));
        let e = s.execs[0][1].unwrap();
        assert_eq!(s.bools.support(e), [Atom::Opaque(lo)]);
        assert_eq!(s.pairs.get(&lo), Some(&hi));
        assert_eq!(s.execs[0][2], Some(e));
    }

    #[test]
    fn scalar_branches_stay_uniform() {
        let p = build(
            &[
                0xBF06_0100, // s_cmp_eq_u32 s0, s1
                0xBF85_0001, // s_cbranch_scc1 -> 12
                0x7E02_0280, // v_mov_b32 v1, 0
                S_ENDPGM,    // 12
            ],
            true,
        );
        let s = structure(&p).unwrap();
        let LaneExit::Branch { when, .. } = s.exits[0] else {
            panic!()
        };
        assert!(matches!(s.bools.support(when)[..], [Atom::Uniform(_)]));
        assert_eq!(guarded(&s), 0);
    }

    /// The SCC a mask operation sets is its result's: after
    /// `s_and_saveexec`, the new EXEC's ("some lane is left"), not that of
    /// the old EXEC it saves first. A console flow branching on it skips
    /// the block when no lane is left (wave64, both forms).
    #[test]
    fn the_scc_of_a_saveexec_is_the_new_execs() {
        let p = build(
            &[
                0x7D84_0080, // v_cmp_eq_u32 vcc, 0, v0
                0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
                0xBF84_0001, // s_cbranch_scc0 -> 20
                0x7E02_0280, // v_mov_b32 v1, 0
                0x88FE_0A7E, // 20: s_or_b64 exec, exec, s[10:11]
                0xBF12_7E6A, // s_cmp_eq_u64 vcc, exec
                S_ENDPGM,
            ],
            true,
        );
        let saveexec = &p.blocks[0].insts[1];
        let scc = saveexec.write(Reg::Scc).unwrap();
        let saved = saveexec.write(Reg::S(10)).unwrap();
        let exec = saveexec.write(Reg::S(EXEC)).unwrap();
        let cmp = p
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .find(|i| i.name() == "s_cmp_eq_u64")
            .unwrap();
        let eq = cmp.write(Reg::Scc).unwrap();
        let vcc = cmp.read(Reg::S(VCC)).unwrap();
        let exec_after = cmp.read(Reg::S(EXEC)).unwrap();
        for s in [structure(&p).unwrap(), structure_console(&p).unwrap()] {
            assert_ne!(s.masks[&exec], s.masks[&saved]);
            assert_eq!(s.sccs.get(&scc), Some(&(s.masks[&exec], false)));
            // `s_cmp_eq` of two masks: no lane differs.
            let mut b = s.bools.clone();
            let ne = b.xor(s.masks[&vcc], s.masks[&exec_after]);
            assert_eq!(s.sccs.get(&eq), Some(&(ne, true)));
        }
    }

    /// The console form without what lanes know on entering each block (its
    /// fallback when that grows too large) has the same masks, EXECs and
    /// SCCs; each block is entered knowing nothing.
    #[test]
    fn the_console_form_without_path_knowledge_keeps_its_masks() {
        let p = build(
            &[
                0x7D84_0080, // v_cmp_eq_u32 vcc, 0, v0
                0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
                0xBF84_0001, // s_cbranch_scc0 -> 20
                0x7E02_0280, // v_mov_b32 v1, 0
                0x88FE_0A7E, // 20: s_or_b64 exec, exec, s[10:11]
                0xBF12_7E6A, // s_cmp_eq_u64 vcc, exec
                S_ENDPGM,
            ],
            true,
        );
        // Booleans of two runs compared by their diagrams (the same atoms,
        // made in the same order: the same reduced diagrams).
        fn show(b: &Bools, x: Bool) -> String {
            match b.node(x) {
                None => format!("{}", x == Bool::TRUE),
                Some((a, lo, hi)) => format!("({a:?} ? {} : {})", show(b, hi), show(b, lo)),
            }
        }
        let full = structure_console(&p).unwrap();
        let mut l = Lanes::new(&p, true).unwrap();
        l.unknowing = true;
        let bare = run(l).unwrap();
        assert!(bare.reach.iter().all(|&r| r == Bool::TRUE));
        let execs = |s: &Structured| -> Vec<Vec<Option<String>>> {
            s.execs
                .iter()
                .map(|b| b.iter().map(|e| e.map(|e| show(&s.bools, e))).collect())
                .collect()
        };
        assert_eq!(execs(&bare), execs(&full));
        let masks = |s: &Structured| -> Vec<(Value, String)> {
            let mut m: Vec<_> = s
                .masks
                .iter()
                .map(|(&v, &b)| (v, show(&s.bools, b)))
                .collect();
            m.sort_unstable();
            m
        };
        assert_eq!(masks(&bare), masks(&full));
        let sccs = |s: &Structured| -> Vec<(Value, String, bool)> {
            let mut m: Vec<_> = s
                .sccs
                .iter()
                .map(|(&v, &(b, n))| (v, show(&s.bools, b), n))
                .collect();
            m.sort_unstable();
            m
        };
        assert_eq!(sccs(&bare), sccs(&full));
    }

    #[test]
    fn irreducible_loops_are_split() {
        let p = build(
            &[
                0xBF06_0100, // s_cmp_eq_u32 s0, s1
                0xBF85_0001, // s_cbranch_scc1 -> 12, into the loop's middle
                0x8002_8102, // 8: s_add_u32 s2, s2, 1
                0xBF84_FFFE, // 12: s_cbranch_scc0 -> 8
                S_ENDPGM,
            ],
            false,
        );
        // The loop keeps 12 as its header; the entry into 8 gets a copy of it.
        let copy = &p.blocks[4];
        assert_eq!((copy.offset, copy.exit), (8, ir::Exit::Jump(2)));
        assert_eq!(p.blocks[0].exit.successors(), [2, 4]);
        assert_eq!(p.blocks[1].preds, [2]);
        assert!(structure(&p).is_ok());
    }

    #[test]
    fn a_reused_mask_sgpr_bit_test_stays_uniform() {
        let p = build(
            &[
                0xBF07_8000, // s_cmp_lg_u32 s0, 0
                0xBF84_0003, // s_cbranch_scc0 -> 20
                0x7D84_0080, // v_cmp_eq_u32 vcc_lo, 0, v0
                0x8502_807E, // s_cselect_b32 s2, exec_lo, 0
                0xBF82_0001, // s_branch -> 24
                0x8102_8101, // 20: s_add_i32 s2, s1, 1
                0x8703_FF02,
                0x0010_0000, // 24: s_and_b32 s3, s2, 0x00100000
                0xBF06_8003, // s_cmp_eq_u32 s3, 0
                0xBF84_0002, // s_cbranch_scc0 -> 48
                0x7E02_0281, // v_mov_b32 v1, 1
                S_ENDPGM,
                0x7E02_0282, // 48: v_mov_b32 v1, 2
                S_ENDPGM,
            ],
            false,
        );
        let (join, phi_s2) = p
            .blocks
            .iter()
            .enumerate()
            .find_map(|(b, blk)| {
                blk.phis
                    .iter()
                    .find(|phi| phi.reg == Reg::S(2))
                    .map(|phi| (b, phi.value))
            })
            .unwrap();
        let s3 = p.blocks[join]
            .insts
            .iter()
            .find(|i| i.name() == "s_and_b32")
            .unwrap()
            .write(Reg::S(3))
            .unwrap();
        let scc = p.blocks[join]
            .insts
            .iter()
            .find(|i| i.name() == "s_cmp_eq_u32")
            .unwrap()
            .write(Reg::Scc)
            .unwrap();
        let s = structure(&p).unwrap();
        let sc = structure_console(&p).unwrap();
        for res in [&s, &sc] {
            assert!(!res.masks.contains_key(&phi_s2));
            assert!(!res.masks.contains_key(&s3));
            assert!(!res.sccs.contains_key(&scc));
        }
        let LaneExit::Branch { when, .. } = s.exits[join] else {
            panic!("{:?}", s.exits);
        };
        assert!(matches!(s.bools.support(when)[..], [Atom::Uniform(_)]));
    }

    #[test]
    fn a_mixed_mask_phi_used_as_a_mask_stays_tracked() {
        let p = build(
            &[
                0xBF07_8000, // s_cmp_lg_u32 s0, 0
                0xBF84_0003, // s_cbranch_scc0 -> 20
                0x7D84_0080, // v_cmp_eq_u32 vcc_lo, 0, v0
                0x8502_807E, // s_cselect_b32 s2, exec_lo, 0
                0xBF82_0001, // s_branch -> 24
                0x8102_8101, // 20: s_add_i32 s2, s1, 1
                0x8703_FF02,
                0x0010_0000, // 24: s_and_b32 s3, s2, 0x00100000
                0xBE84_3C03, // s_and_saveexec_b32 s4, s3
                0x7E02_0281, // v_mov_b32 v1, 1
                0xBEFE_0304, // s_mov_b32 exec_lo, s4
                S_ENDPGM,
            ],
            false,
        );
        let (join, phi_s2) = p
            .blocks
            .iter()
            .enumerate()
            .find_map(|(b, blk)| {
                blk.phis
                    .iter()
                    .find(|phi| phi.reg == Reg::S(2))
                    .map(|phi| (b, phi.value))
            })
            .unwrap();
        let s3 = p.blocks[join]
            .insts
            .iter()
            .find(|i| i.name() == "s_and_b32")
            .unwrap()
            .write(Reg::S(3))
            .unwrap();
        for res in [structure(&p).unwrap(), structure_console(&p).unwrap()] {
            assert!(res.masks.contains_key(&phi_s2));
            assert!(res.masks.contains_key(&s3));
        }
    }

    #[test]
    fn a_zero_compare_on_a_mixed_phi_stays_tracked() {
        let p = build(
            &[
                0xBF07_8000, // s_cmp_lg_u32 s0, 0
                0xBF84_0003, // s_cbranch_scc0 -> 20
                0x7D84_0080, // v_cmp_eq_u32 vcc_lo, 0, v0
                0x8502_807E, // s_cselect_b32 s2, exec_lo, 0
                0xBF82_0001, // s_branch -> 24
                0x8102_8101, // 20: s_add_i32 s2, s1, 1
                0xBF06_8002, // 24: s_cmp_eq_u32 s2, 0
                0xBF85_0001, // s_cbranch_scc1 -> 36
                0x7E02_0281, // v_mov_b32 v1, 1
                S_ENDPGM,    // 36
            ],
            false,
        );
        let (join, phi_s2) = p
            .blocks
            .iter()
            .enumerate()
            .find_map(|(b, blk)| {
                blk.phis
                    .iter()
                    .find(|phi| phi.reg == Reg::S(2))
                    .map(|phi| (b, phi.value))
            })
            .unwrap();
        let scc = p.blocks[join]
            .insts
            .iter()
            .find(|i| i.name() == "s_cmp_eq_u32")
            .unwrap()
            .write(Reg::Scc)
            .unwrap();
        for res in [structure(&p).unwrap(), structure_console(&p).unwrap()] {
            assert!(res.masks.contains_key(&phi_s2));
            assert!(res.sccs.contains_key(&scc));
        }
    }

    #[test]
    fn meet_memoizes_unsat_cofactors_behind_a_long_reach_chain() {
        // 17 independent (!a_k | !b_k) clauses have 3^17 > MAX_MEET satisfying
        // assignments on only 34 BDD nodes. An unmemoized search exhausts
        // MAX_MEET before proving the trailing guard contradicts its branch.
        let mut b = Bools::new();
        let mut reach = Bool::TRUE;
        for k in 0..17u32 {
            let ak = b.atom(Atom::Lane(Value(2 * k)), 0);
            let bk = b.atom(Atom::Lane(Value(2 * k + 1)), 0);
            let nak = b.not(ak);
            let nbk = b.not(bk);
            let clause = b.or(nak, nbk);
            reach = b.and(reach, clause);
        }
        let guard = b.atom(Atom::Lane(Value(100)), 0);
        let skip = b.not(guard);
        assert!(!b.meet(&[(reach, true), (guard, true), (skip, true)]));
        assert!(!b.meet(&[(reach, true), (guard, true), (guard, false)]));
        assert!(b.meet(&[(reach, true), (guard, true)]));
    }
}
