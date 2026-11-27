//! Which values of a mesh program are the same in every lane, as the
//! per-lane lowering computes them: in the whole thread group, in each
//! console wave, or neither. Backlog L4: wave64 mesh programs on hosts
//! without 64-lane waves (`docs/analysis/mesh-wave64.md`).
//!
//! The console's `v_readfirstlane_b32` gives every lane of its wave the
//! value its wave's first lane in EXEC holds. Lowered as the host's
//! `WaveReadLaneFirst`, it reads the first lane of the host wave instead,
//! and on a 32-lane host the upper half of a console wave64 is a host wave
//! of its own. Where the value read is the same in every lane of the
//! console wave that runs the instruction, each lane's own value is the
//! first lane's: the read needs no wave operation, whatever waves the host
//! forms. [`Uniformity`] says where that holds.
//!
//! Two lanes of a console wave hold the same value when they run the same
//! console instance of its definition from the same operands. They stop
//! agreeing where:
//!
//! * the value comes from the lane itself: thread ids, `v_mbcnt`, lane
//!   reads (DPP, swizzles), LDS, atomics, the clock; the console wave's
//!   index (`s3`, the merged wave info) differs between the group's waves
//!   only;
//! * an instruction runs under a guard that holds in some of the lanes
//!   only: the others keep the register's old value;
//! * lanes join after a branch they took differently: the phis of the
//!   branch's join points (blocks where two paths from its two successors
//!   first meet), found by labelling the blocks after it;
//! * lanes leave a loop at different iterations or by different exits:
//!   values the loop defines, read after it, and the phis where its exit
//!   edges meet.
//!
//! The analysis is optimistic (every value starts the same in every lane)
//! and raises levels until nothing changes; each step only raises, so it
//! ends. It also gives, for each block, the level of the branches that
//! decide whether, and how often, a lane runs it: a group barrier needs
//! every lane of the group there ([`Uniformity::control`]).
//!
//! Memory is read alike only where the program writes none: in two host
//! waves, two halves of one console load read at different times, so a
//! store between them (this group's or another's) would part them. A
//! program with a store or an atomic has every load per lane. (None of
//! BTP 6's mesh programs writes memory.)
//!
//! What it assumes: the per-lane form itself (a lane in EXEC at a vector
//! instruction has reached it, holding the console's registers), memory no
//! other dispatch changes meanwhile, and values used in their own console
//! instance (lanes that skip a block do not read the scalars it sets, as
//! structurisation already assumes).

use std::collections::{HashMap, HashSet};

use super::Parts;
use crate::gfx10::{Fields, NULL, Operand};
use crate::io::Input;
use crate::ir::{EXEC, Inst, M0, Program, Reg, Value};
use crate::structure::{Atom, Bool, Bools, LaneExit, Structured};

/// How widely a value is the same.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum Level {
    /// In every lane of the thread group.
    Group,
    /// In every lane of a console wave; the group's waves may differ.
    Wave,
    /// Per lane.
    Lane,
}

/// The result of the analysis (see the module notes).
#[derive(Debug, Clone)]
pub(super) struct Uniformity {
    /// By value: its level where it is defined.
    #[cfg(test)]
    level: Vec<Level>,
    /// By `v_readfirstlane_b32` result: the level of the value it reads,
    /// where it reads it.
    first_lane: HashMap<Value, Level>,
    /// By block: the highest level of the branches that decide whether,
    /// and how often, a lane runs it.
    control: Vec<Level>,
}

impl Uniformity {
    /// Analyses the program of `x` as structurised in `x.st`.
    pub(super) fn of(x: &Parts) -> Uniformity {
        Run::new(x).solve()
    }

    /// The level of the value that the `v_readfirstlane_b32` writing `w`
    /// reads; `None` for another value.
    pub(super) fn first_lane(&self, w: Value) -> Option<Level> {
        self.first_lane.get(&w).copied()
    }

    /// The level of the branches deciding whether and how often a lane
    /// runs block `b` ([`Level::Group`]: every lane of the group runs it
    /// as often as the others).
    pub(super) fn control(&self, b: usize) -> Level {
        self.control.get(b).copied().unwrap_or(Level::Lane)
    }

    /// The level of value `v` where it is defined.
    #[cfg(test)]
    pub(super) fn level(&self, v: Value) -> Level {
        self.level.get(v.0 as usize).copied().unwrap_or(Level::Lane)
    }
}

/// No block: values on entry.
const ENTRY: usize = usize::MAX;

/// A natural loop.
struct Loop {
    /// By block: inside the loop.
    body: Vec<bool>,
    /// Branches inside whose two sides meet again only after the loop:
    /// lanes that take them differently leave at different iterations, or
    /// by different exits.
    leaving: Vec<usize>,
    /// Branches inside whose two sides meet again only at the next
    /// iteration (or after the loop): with several back edges, lanes that
    /// take them differently come back by different ones.
    returning: Vec<usize>,
    /// Back edges into the header.
    back_edges: usize,
}

/// The analysis in progress.
struct Run<'a> {
    p: &'a Program,
    st: &'a Structured,
    x: &'a Parts<'a>,
    order: Vec<usize>,
    rpo: Vec<usize>,
    /// Block defining each value ([`ENTRY`] for inputs).
    def: Vec<usize>,
    level: Vec<Level>,
    /// By VALU-written SGPR value: the level of the lane bit it holds
    /// (compare and carry results), and whether one was written.
    bit: Vec<Level>,
    has_bit: Vec<bool>,
    first_lane: HashMap<Value, Level>,
    /// The program stores to memory or runs atomics on it: loads differ.
    writes_memory: bool,
    loops: Vec<Loop>,
    /// By block: the loops containing it.
    loops_of: Vec<Vec<usize>>,
    /// By block: the loop it heads.
    header_of: Vec<Option<usize>>,
    /// By block: the branches it is a join point of.
    joins_of: Vec<Vec<usize>>,
    /// By block: the loops whose exit edges first meet there.
    exit_joins_of: Vec<Vec<usize>>,
    /// By block: the literals its reach implies (computed on first use).
    implied: HashMap<usize, HashMap<Atom, bool>>,
    /// The current levels of branches and loops.
    branch: Vec<Level>,
    leave: Vec<Level>,
    come_back: Vec<Level>,
    changed: bool,
}

impl<'a> Run<'a> {
    fn new(x: &'a Parts<'a>) -> Self {
        let p = x.p;
        let n = p.blocks.len();
        let order = crate::ir::reverse_postorder(&p.blocks);
        let mut rpo = vec![usize::MAX; n];
        for (k, &b) in order.iter().enumerate() {
            rpo[b] = k;
        }
        let mut values = 0usize;
        let mut see = |v: Value| values = values.max(v.0 as usize + 1);
        for &(_, v) in &p.inputs {
            see(v);
        }
        for block in &p.blocks {
            for phi in &block.phis {
                see(phi.value);
                phi.args.iter().copied().for_each(&mut see);
            }
            for inst in &block.insts {
                for &(_, v) in inst.reads.iter().chain(&inst.writes) {
                    see(v);
                }
            }
        }
        let mut def = vec![ENTRY; values];
        for (b, block) in p.blocks.iter().enumerate() {
            for phi in &block.phis {
                def[phi.value.0 as usize] = b;
            }
            for inst in &block.insts {
                for &(_, w) in &inst.writes {
                    def[w.0 as usize] = b;
                }
            }
        }
        let writes_memory = p.blocks.iter().flat_map(|b| &b.insts).any(|i| {
            let n = i.name();
            matches!(
                i.fields,
                Fields::Smem { .. }
                    | Fields::Buffer { .. }
                    | Fields::Image { .. }
                    | Fields::Flat { .. }
            ) && (n.contains("store") || n.contains("atomic"))
        });
        let mut run = Run {
            p,
            st: x.st,
            x,
            order,
            rpo,
            def,
            level: vec![Level::Group; values],
            bit: vec![Level::Group; values],
            has_bit: vec![false; values],
            first_lane: HashMap::new(),
            writes_memory,
            loops: Vec::new(),
            loops_of: vec![Vec::new(); n],
            header_of: vec![None; n],
            joins_of: vec![Vec::new(); n],
            exit_joins_of: vec![Vec::new(); n],
            implied: HashMap::new(),
            branch: vec![Level::Group; n],
            leave: Vec::new(),
            come_back: Vec::new(),
            changed: false,
        };
        run.find_loops();
        run.find_joins();
        run.inputs();
        run
    }

    fn reachable(&self, b: usize) -> bool {
        self.rpo[b] != usize::MAX
    }

    /// Whether the edge `from` → `to` goes back to a loop header.
    fn back(&self, from: usize, to: usize) -> bool {
        self.rpo[to] <= self.rpo[from]
    }

    fn succs(&self, b: usize) -> Vec<usize> {
        let mut s = self.p.blocks[b].exit.successors();
        s.dedup();
        s
    }

    /// The two successors of a branch whose sides differ.
    fn sides(&self, b: usize) -> Option<(usize, usize)> {
        match self.st.exits[b] {
            LaneExit::Branch {
                taken, not_taken, ..
            } if taken != not_taken => Some((taken, not_taken)),
            _ => None,
        }
    }

    fn find_loops(&mut self) {
        let n = self.p.blocks.len();
        let mut latches: HashMap<usize, Vec<usize>> = HashMap::new();
        for &q in &self.order {
            for s in self.succs(q) {
                if self.reachable(s) && self.back(q, s) {
                    latches.entry(s).or_default().push(q);
                }
            }
        }
        let mut headers: Vec<usize> = latches.keys().copied().collect();
        headers.sort_unstable_by_key(|&h| self.rpo[h]);
        for h in headers {
            let tails = &latches[&h];
            let mut body = vec![false; n];
            body[h] = true;
            let mut work: Vec<usize> = tails.clone();
            while let Some(b) = work.pop() {
                if body[b] || !self.reachable(b) {
                    continue;
                }
                body[b] = true;
                work.extend(self.p.blocks[b].preds.iter().copied());
            }
            let (leaving, returning) = self.loop_branches(h, &body);
            let l = self.loops.len();
            for b in (0..n).filter(|&b| body[b]) {
                self.loops_of[b].push(l);
            }
            self.header_of[h] = Some(l);
            self.loops.push(Loop {
                body,
                leaving,
                returning,
                back_edges: tails.len(),
            });
        }
        self.leave = vec![Level::Group; self.loops.len()];
        self.come_back = vec![Level::Group; self.loops.len()];
    }

    /// The branches of loop `h` ([`Loop::leaving`], [`Loop::returning`]):
    /// post-dominators over one iteration, where back edges go to a node
    /// NEXT, exit edges to a node OUT, and both to END.
    fn loop_branches(&self, h: usize, body: &[bool]) -> (Vec<usize>, Vec<usize>) {
        let inside: Vec<usize> = (0..body.len()).filter(|&b| body[b]).collect();
        let index: HashMap<usize, usize> =
            inside.iter().enumerate().map(|(k, &b)| (b, k)).collect();
        let (next, out, end) = (inside.len(), inside.len() + 1, inside.len() + 2);
        let mut succ: Vec<Vec<usize>> = inside
            .iter()
            .map(|&b| {
                let mut s: Vec<usize> = self
                    .succs(b)
                    .into_iter()
                    .filter(|&t| self.reachable(t))
                    .map(|t| match index.get(&t) {
                        _ if t == h => next,
                        Some(&k) => k,
                        None => out,
                    })
                    .collect();
                s.sort_unstable();
                s.dedup();
                s
            })
            .collect();
        succ.push(vec![end]);
        succ.push(vec![end]);
        succ.push(Vec::new());
        let pdom = post_dominators(&succ, end);
        let (mut leaving, mut returning) = (Vec::new(), Vec::new());
        for (k, &d) in inside.iter().enumerate() {
            if self.sides(d).is_none() {
                continue;
            }
            let strict: Vec<usize> = (0..succ.len())
                .filter(|&m| m != k && pdom[k].has(m))
                .collect();
            if strict == [end] {
                leaving.push(d);
                returning.push(d);
            } else if strict == [next, end] {
                returning.push(d);
            }
        }
        (leaving, returning)
    }

    /// Join points of every branch, and blocks where a loop's exit edges
    /// meet: forward labelling from the branch's (the loop's) out-edges,
    /// over edges that are not back edges.
    fn find_joins(&mut self) {
        let n = self.p.blocks.len();
        for &d in &self.order.clone() {
            let Some((a, b)) = self.sides(d) else {
                continue;
            };
            let starts = [(d, a), (d, b)];
            for j in self.label(&starts) {
                self.joins_of[j].push(d);
            }
        }
        for l in 0..self.loops.len() {
            let body = &self.loops[l].body;
            let mut exits = Vec::new();
            for q in (0..n).filter(|&q| body[q] && self.reachable(q)) {
                for t in self.succs(q) {
                    if !body[t] && self.reachable(t) {
                        exits.push((q, t));
                    }
                }
            }
            if exits.len() < 2 {
                continue;
            }
            for j in self.label(&exits) {
                self.exit_joins_of[j].push(l);
            }
        }
    }

    /// Labels the blocks reached forward from the edges `starts`, each
    /// edge its own label; returns the blocks where two labels meet.
    fn label(&self, starts: &[(usize, usize)]) -> Vec<usize> {
        const NONE: usize = usize::MAX;
        let n = self.p.blocks.len();
        let first = starts
            .iter()
            .map(|&(_, t)| self.rpo[t])
            .min()
            .unwrap_or(usize::MAX);
        let mut label = vec![NONE; n];
        let mut joins = Vec::new();
        for &x in self.order.iter().skip(first) {
            let mut seen = NONE;
            let mut join = false;
            let mut preds = self.p.blocks[x].preds.clone();
            preds.sort_unstable();
            preds.dedup();
            for p in preds {
                if !self.reachable(p) || self.back(p, x) {
                    continue;
                }
                let mut ls: Vec<usize> = starts
                    .iter()
                    .enumerate()
                    .filter(|&(_, &e)| e == (p, x))
                    .map(|(k, _)| k)
                    .collect();
                if label[p] != NONE && !starts.iter().any(|&(q, _)| q == p) {
                    ls.push(label[p]);
                }
                for l in ls {
                    if seen == NONE {
                        seen = l;
                    } else if seen != l {
                        join = true;
                    }
                }
            }
            label[x] = if join {
                joins.push(x);
                starts.len() + x
            } else {
                seen
            };
        }
        joins
    }

    /// Levels of the program's inputs: user data and the group id are the
    /// group's, the merged wave info the console wave's; other SGPRs are
    /// taken as per lane. VGPRs the hardware does not load read as zero.
    fn inputs(&mut self) {
        let io: HashMap<Value, Input> = self.x.io.inputs.iter().copied().collect();
        let params: HashSet<Value> = self.x.res.params.iter().map(|p| p.0).collect();
        for &(reg, v) in &self.p.inputs {
            let level = match (reg, io.get(&v)) {
                (_, Some(Input::MeshGroup)) => Level::Group,
                (_, Some(Input::MergedWaveInfo)) => Level::Wave,
                (_, Some(_)) => Level::Lane,
                (Reg::S(_), None) if params.contains(&v) => Level::Group,
                (Reg::V(_), None) => Level::Group,
                _ => Level::Lane,
            };
            self.level[v.0 as usize] = level;
        }
    }

    fn solve(mut self) -> Uniformity {
        loop {
            self.changed = false;
            let n = self.p.blocks.len();
            self.branch = (0..n).map(|b| self.exit_level(b)).collect();
            self.leave = (0..self.loops.len())
                .map(|l| self.max_branch(&self.loops[l].leaving))
                .collect();
            self.come_back = (0..self.loops.len())
                .map(|l| self.max_branch(&self.loops[l].returning))
                .collect();
            for k in 0..self.order.len() {
                let b = self.order[k];
                self.block(b);
            }
            if !self.changed {
                break;
            }
        }
        let control = self.control();
        Uniformity {
            #[cfg(test)]
            level: self.level,
            first_lane: self.first_lane,
            control,
        }
    }

    fn max_branch(&self, ds: &[usize]) -> Level {
        ds.iter()
            .map(|&d| self.branch[d])
            .max()
            .unwrap_or(Level::Group)
    }

    fn raise(&mut self, v: Value, l: Level) {
        let at = &mut self.level[v.0 as usize];
        if l > *at {
            *at = l;
            self.changed = true;
        }
    }

    fn raise_bit(&mut self, v: Value, l: Level) {
        let k = v.0 as usize;
        self.has_bit[k] = true;
        if l > self.bit[k] {
            self.bit[k] = l;
            self.changed = true;
        }
    }

    /// How differently lanes may leave the loops that hold block `d` but
    /// not block `at`: the level a value defined in `d` takes when read in
    /// `at`.
    fn leaving(&self, d: usize, at: usize) -> Level {
        if d == ENTRY {
            return Level::Group;
        }
        self.loops_of[d]
            .iter()
            .filter(|&&l| !self.loops[l].body[at])
            .map(|&l| self.leave[l])
            .max()
            .unwrap_or(Level::Group)
    }

    /// The level of `v` as block `at` reads it.
    fn read(&self, v: Value, at: usize) -> Level {
        let k = v.0 as usize;
        self.level[k].max(self.leaving(self.def[k], at))
    }

    /// The level of the lane bit `b` (a per-lane boolean) in block `at`;
    /// `given`: first replace the atoms the block's reach decides.
    fn bool_level(&mut self, b: Bool, at: usize, given: bool) -> Level {
        let atoms = if given {
            let known = self.known(at);
            support_given(&self.st.bools, b, &known)
        } else {
            self.st.bools.support(b)
        };
        atoms
            .into_iter()
            .map(|a| match a {
                Atom::Uniform(v) => self.read(v, at),
                Atom::Lane(v) if self.has_bit[v.0 as usize] => {
                    self.bit[v.0 as usize].max(self.leaving(self.def[v.0 as usize], at))
                }
                _ => Level::Lane,
            })
            .max()
            .unwrap_or(Level::Group)
    }

    /// The literals the reach of block `at` implies.
    fn known(&mut self, at: usize) -> HashMap<Atom, bool> {
        if let Some(k) = self.implied.get(&at) {
            return k.clone();
        }
        let mut memo = HashMap::new();
        let lits = implied(&self.st.bools, self.st.reach[at], &mut memo).unwrap_or_default();
        let k: HashMap<Atom, bool> = lits.into_iter().collect();
        self.implied.insert(at, k.clone());
        k
    }

    fn exit_level(&mut self, b: usize) -> Level {
        if !self.reachable(b) || self.sides(b).is_none() {
            return Level::Group;
        }
        match self.st.exits[b] {
            LaneExit::Branch { when, .. } => self.bool_level(when, b, false),
            _ => Level::Group,
        }
    }

    /// The lane bit of an SGPR value read as a lane mask, in block `at`.
    fn mask_level(&mut self, v: Value, at: usize) -> Option<Level> {
        let low = self.st.highs.get(&v).copied().unwrap_or(v);
        let m = self
            .st
            .masks
            .get(&low)
            .or_else(|| self.st.others.get(&low))
            .copied()?;
        Some(self.bool_level(m, at, true))
    }

    fn is_mask(&self, v: Value) -> bool {
        self.st.masks.contains_key(&v)
            || self.st.highs.contains_key(&v)
            || self.st.others.contains_key(&v)
    }

    fn block(&mut self, b: usize) {
        let p = self.p;
        let block = &p.blocks[b];
        for phi in &block.phis {
            let mut l = Level::Group;
            for &arg in &phi.args {
                l = l.max(self.read(arg, b));
            }
            for &d in &self.joins_of[b] {
                l = l.max(self.branch[d]);
            }
            for &lp in &self.exit_joins_of[b] {
                l = l.max(self.leave[lp]);
            }
            if let Some(lp) = self.header_of[b]
                && self.loops[lp].back_edges > 1
            {
                l = l.max(self.come_back[lp]);
            }
            self.raise(phi.value, l);
        }
        for i in 0..block.insts.len() {
            self.inst(b, i);
        }
    }

    fn inst(&mut self, b: usize, i: usize) {
        let p = self.p;
        let inst = &p.blocks[b].insts[i];
        let name = inst.name();
        let all = |l: &mut Self, out: Level| {
            for &(_, w) in &inst.writes {
                l.raise(w, out);
            }
        };
        match &inst.fields {
            Fields::Sopp { .. } | Fields::Exp { .. } => {}
            Fields::Sop { .. } => {
                let l = if name.starts_with("s_getreg") || source_regs(inst) == Sources::Lane {
                    Level::Lane
                } else {
                    self.reads(inst, b)
                };
                all(self, l);
            }
            Fields::Smem { .. } => {
                let l = match name {
                    _ if self.x.res.spill_loads.binary_search(&(b, i)).is_ok() => Level::Group,
                    "s_memtime" | "s_memrealtime" => Level::Lane,
                    "s_get_waveid_in_workgroup" => Level::Wave,
                    _ if self.writes_memory => Level::Lane,
                    _ => self.reads(inst, b),
                };
                all(self, l);
            }
            Fields::Ds { .. } | Fields::Vintrp { .. } => all(self, Level::Lane),
            Fields::Valu(v) => {
                let lane_read = v.dpp.is_some()
                    || name.starts_with("v_mbcnt")
                    || name.starts_with("v_interp")
                    || matches!(name, "v_writelane_b32" | "v_swap_b32")
                    || name.contains("movrel")
                    || name.contains("swaprel")
                    || name.contains("permlane");
                if name == "v_readfirstlane_b32" {
                    self.first_lane_read(inst, b);
                } else if name == "v_readlane_b32" {
                    all(self, Level::Wave);
                } else if lane_read {
                    all(self, Level::Lane);
                } else {
                    self.valu(inst, b, i);
                }
            }
            Fields::Buffer { .. } | Fields::Image { .. } | Fields::Flat { .. } => {
                if name.contains("atomic") || self.writes_memory {
                    all(self, Level::Lane);
                } else {
                    let l = self.computed(inst, b, i);
                    all(self, l);
                }
            }
        }
    }

    /// The highest level among the reads of the scalar instruction `inst`,
    /// as block `at` reads them.
    fn reads(&self, inst: &Inst, at: usize) -> Level {
        inst.reads
            .iter()
            .map(|&(_, v)| self.read(v, at))
            .max()
            .unwrap_or(Level::Group)
    }

    /// A vector instruction's result: its sources, and under a guard, or
    /// where part of the register stays, the old value and the guard.
    fn computed(&mut self, inst: &Inst, b: usize, i: usize) -> Level {
        let sources = source_regs(inst);
        if sources == Sources::Lane {
            return Level::Lane;
        }
        let written: HashSet<Reg> = inst.writes.iter().map(|w| w.0).collect();
        let guard = self.st.guards[b][i];
        let keeps = guard.is_some() || keeps_old(inst);
        let mut l = Level::Group;
        for &(r, v) in &inst.reads {
            let x = match r {
                _ if sources.has(r) => self.operand_level(inst, r, v, b),
                // The guard, below.
                _ if is_exec(r) => continue,
                _ if keeps && written.contains(&r) => self.read(v, b),
                _ => continue,
            };
            l = l.max(x);
        }
        if let Some(g) = guard {
            l = l.max(self.bool_level(g, b, false));
        }
        l
    }

    /// The level of operand register `r` of the vector instruction `inst`,
    /// holding `v`, in block `at`: the select or carry-in by its lane bit
    /// ([`lane_bit_operand`]); EXEC and any other lane mask read as a number
    /// the lane's (the per-lane form holds no lane's mask as the console
    /// has it); other registers by their value.
    fn operand_level(&mut self, inst: &Inst, r: Reg, v: Value, at: usize) -> Level {
        let bit = match lane_bit_operand(inst) {
            Some(Operand::Scalar { code, dwords }) => (code..code.saturating_add(dwords))
                .map(Reg::S)
                .any(|s| s == r),
            _ => false,
        };
        match r {
            Reg::S(_) if bit => self.mask_level(v, at).unwrap_or(Level::Lane),
            _ if is_exec(r) => Level::Lane,
            Reg::S(_) if self.is_mask(v) => Level::Lane,
            _ => self.read(v, at),
        }
    }

    fn valu(&mut self, inst: &Inst, b: usize, i: usize) {
        let l = self.computed(inst, b, i);
        // The lane bits a compare or a carry writes do not depend on EXEC
        // (the guard): structurisation masks them.
        let bits = {
            let sources = source_regs(inst);
            let mut m = if sources == Sources::Lane {
                Level::Lane
            } else {
                Level::Group
            };
            for &(r, v) in &inst.reads {
                if sources.has(r) {
                    let x = self.operand_level(inst, r, v, b);
                    m = m.max(x);
                }
            }
            m
        };
        for &(r, w) in &inst.writes {
            match r {
                Reg::V(_) => self.raise(w, l),
                _ => {
                    self.raise_bit(w, bits);
                    // As a number, a lane mask is no value of the lane's.
                    self.raise(w, Level::Lane);
                }
            }
        }
    }

    /// `v_readfirstlane_b32`: records the level of the value read. The
    /// result is the console wave's either way; a waterfall loop's read
    /// is the lane's own (`resource::Waterfall`).
    fn first_lane_read(&mut self, inst: &Inst, b: usize) {
        let sources = source_regs(inst);
        let mut l = if sources == Sources::Lane {
            Level::Lane
        } else {
            Level::Group
        };
        for &(r, v) in &inst.reads {
            if sources.has(r) {
                let x = self.operand_level(inst, r, v, b);
                l = l.max(x);
            }
        }
        for &(_, w) in &inst.writes {
            self.first_lane
                .entry(w)
                .and_modify(|x| *x = (*x).max(l))
                .or_insert(l);
            let waterfall = self.x.res.waterfall.reads.contains(&w)
                || (self.x.native.wave64
                    && !self.x.target.runs(64)
                    && self.x.res.waterfall.indices.contains(&w));
            let out = if waterfall || l <= Level::Wave {
                l
            } else {
                Level::Wave
            };
            self.raise(w, out);
        }
    }

    /// [`Uniformity::control`]: the level of the branches each block is
    /// control dependent on, through the branches they depend on in turn.
    fn control(&self) -> Vec<Level> {
        let n = self.p.blocks.len();
        let end = n;
        let mut succ: Vec<Vec<usize>> = (0..n)
            .map(|b| {
                if !self.reachable(b) {
                    return vec![end];
                }
                let s: Vec<usize> = self
                    .succs(b)
                    .into_iter()
                    .filter(|&t| self.reachable(t))
                    .collect();
                if s.is_empty() { vec![end] } else { s }
            })
            .collect();
        succ.push(Vec::new());
        let pdom = post_dominators(&succ, end);
        // depends[b]: the branches b is directly control dependent on: b
        // post-dominates one side of the branch, not the branch itself.
        let mut depends: Vec<Vec<usize>> = vec![Vec::new(); n];
        for d in (0..n).filter(|&d| self.reachable(d)) {
            let Some((s1, s2)) = self.sides(d) else {
                continue;
            };
            for b in (0..n).filter(|&b| self.reachable(b)) {
                let strict = b != d && pdom[d].has(b);
                if !strict && (pdom[s1].has(b) || pdom[s2].has(b)) {
                    depends[b].push(d);
                }
            }
        }
        let mut control = vec![Level::Group; n];
        loop {
            let mut changed = false;
            for b in 0..n {
                let l = depends[b]
                    .iter()
                    .map(|&d| self.branch[d].max(control[d]))
                    .max()
                    .unwrap_or(Level::Group);
                if l > control[b] {
                    control[b] = l;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        for (b, c) in control.iter_mut().enumerate() {
            if !self.reachable(b) {
                *c = Level::Lane;
            }
        }
        control
    }
}

/// What an instruction reads as its operands ([`source_regs`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Sources {
    /// Every read counts (`v_movrel*`, `v_swap_b32`, encodings without a
    /// list of operands).
    All,
    /// The registers it names as operands: not the EXEC that guards it, nor
    /// the old value of a VGPR it writes, which [`crate::ir`] adds.
    Regs(Vec<Reg>),
    /// It names `vccz`, `execz` or `lds_direct`. The first two are a bit
    /// of a whole wave's mask, which no lane holds in the per-lane form; the
    /// third is a dword of LDS at M0, which another lane may have written.
    /// Per lane, whatever the registers behind them (VCC, EXEC, M0) hold.
    Lane,
}

impl Sources {
    /// Whether a read of `r` counts as an operand. A read of EXEC counts
    /// only where the instruction names EXEC; where every read counts,
    /// EXEC is still only the guard.
    pub(crate) fn has(&self, r: Reg) -> bool {
        match self {
            Sources::Regs(s) => s.contains(&r),
            Sources::All | Sources::Lane => !is_exec(r),
        }
    }
}

/// Whether `r` is a dword of EXEC.
pub(crate) fn is_exec(r: Reg) -> bool {
    matches!(r, Reg::S(c) if c == EXEC || c == EXEC + 1)
}

/// Operand codes of `vccz`, `execz` and `lds_direct` ([`Sources::Lane`]).
const PER_LANE_OPERANDS: [u8; 3] = [251, 252, 254];

/// The operands `inst` reads ([`Sources`]).
pub(crate) fn source_regs(inst: &Inst) -> Sources {
    let mut out = Vec::new();
    let mut lane = false;
    let mut add = |op: &Operand| match *op {
        Operand::Scalar { code, .. } if code == NULL => {}
        Operand::Scalar { code, dwords } => {
            out.extend((0..dwords).map(|k| Reg::S(code.wrapping_add(k))));
        }
        Operand::Vector { reg, dwords } => {
            out.extend((0..dwords).map(|k| Reg::V(reg.wrapping_add(k))));
        }
        Operand::Const(253) => out.push(Reg::Scc),
        Operand::Const(c) if PER_LANE_OPERANDS.contains(&c) => lane = true,
        _ => {}
    };
    let name = inst.name();
    let store = name.contains("store") || name.contains("atomic");
    match &inst.fields {
        Fields::Valu(v) => {
            if name.contains("movrel") || name.contains("swaprel") || name == "v_swap_b32" {
                return Sources::All;
            }
            v.src.iter().for_each(&mut add);
        }
        Fields::Buffer {
            data,
            addr,
            rsrc,
            soffset,
            offen,
            idxen,
            ..
        } => {
            if *offen || *idxen {
                add(addr);
            }
            add(rsrc);
            add(soffset);
            if store && let Some(d) = data {
                add(d);
            }
        }
        Fields::Image {
            data,
            addr,
            rsrc,
            sampler,
            ..
        } => {
            addr.iter().chain([rsrc]).chain(sampler).for_each(&mut add);
            if store {
                add(data);
            }
        }
        Fields::Flat {
            addr, data, saddr, ..
        } => {
            [addr, data, saddr].into_iter().flatten().for_each(&mut add);
        }
        // Every read of a scalar instruction counts, its special operands
        // aside.
        Fields::Sop { src, .. } => {
            src.iter().for_each(&mut add);
            return if lane { Sources::Lane } else { Sources::All };
        }
        _ => return Sources::All,
    }
    if lane {
        return Sources::Lane;
    }
    if matches!(inst.fields, Fields::Buffer { lds: true, .. }) {
        out.push(Reg::S(M0));
    }
    Sources::Regs(out)
}

/// The lane mask `inst` reads bit by bit, the lane's own bit: the select
/// of `v_cndmask_b32` and the carry-in of the `*_co_ci_u32` forms, their
/// third operand (VOP2's implicit VCC included). A VALU instruction reads
/// any other SGPR, a mask too, as a number.
pub(crate) fn lane_bit_operand(inst: &Inst) -> Option<Operand> {
    let Fields::Valu(v) = &inst.fields else {
        return None;
    };
    let name = inst.name();
    if name != "v_cndmask_b32" && !name.contains("_co_ci_") {
        return None;
    }
    v.src.get(2).copied()
}

/// Whether `inst` keeps part of a register it writes even in the lanes
/// that run it (accumulators, half writes, DPP's masked rows, SDWA's
/// partial destinations, `tfe` status dwords): its result then depends on
/// the old value.
pub(crate) fn keeps_old(inst: &Inst) -> bool {
    let n = inst.name();
    match &inst.fields {
        Fields::Valu(v) => {
            v.dpp.is_some()
                || v.sdwa.is_some_and(|s| s.dst_sel < 6)
                || n.contains("mac")
                || n.contains("dot")
                || n.contains("16")
                || n.contains("mix")
                || n.contains("qsad")
                || n.starts_with("v_pk_")
        }
        Fields::Buffer { tfe, .. } => *tfe || n.contains("d16") || n.contains("_hi"),
        Fields::Image { d16, tfe, lwe, .. } => *d16 || *tfe || *lwe,
        Fields::Flat { .. } => n.contains("d16") || n.contains("_hi"),
        _ => true,
    }
}

/// Post-dominator sets of a graph every node of which reaches `end`.
fn post_dominators(succ: &[Vec<usize>], end: usize) -> Vec<Bits> {
    let n = succ.len();
    let mut pdom: Vec<Bits> = (0..n)
        .map(|k| {
            if k == end {
                Bits::one(n, end)
            } else {
                Bits::all(n)
            }
        })
        .collect();
    loop {
        let mut changed = false;
        for k in (0..n).rev() {
            if k == end {
                continue;
            }
            let mut s = match succ[k].first() {
                Some(&f) => pdom[f].clone(),
                None => Bits::all(n),
            };
            for &t in succ[k].iter().skip(1) {
                s.and(&pdom[t]);
            }
            s.set(k);
            if s != pdom[k] {
                pdom[k] = s;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    pdom
}

/// A set of graph nodes.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Bits(Vec<u64>);

impl Bits {
    fn all(n: usize) -> Bits {
        Bits(vec![u64::MAX; n.div_ceil(64)])
    }

    fn one(n: usize, k: usize) -> Bits {
        let mut b = Bits(vec![0; n.div_ceil(64)]);
        b.set(k);
        b
    }

    fn set(&mut self, k: usize) {
        self.0[k / 64] |= 1 << (k % 64);
    }

    fn has(&self, k: usize) -> bool {
        self.0[k / 64] >> (k % 64) & 1 != 0
    }

    fn and(&mut self, o: &Bits) {
        for (a, b) in self.0.iter_mut().zip(&o.0) {
            *a &= b;
        }
    }
}

/// The literals every lane where `b` holds has: `None` where `b` never
/// holds.
fn implied(
    bools: &Bools,
    b: Bool,
    memo: &mut HashMap<Bool, Option<Vec<(Atom, bool)>>>,
) -> Option<Vec<(Atom, bool)>> {
    if b == Bool::TRUE {
        return Some(Vec::new());
    }
    if let Some(r) = memo.get(&b) {
        return r.clone();
    }
    let r = match bools.node(b) {
        None => None,
        Some((atom, lo, hi)) => {
            let with = |side: Option<Vec<(Atom, bool)>>, value: bool| {
                side.map(|mut v| {
                    v.push((atom, value));
                    v
                })
            };
            let lo = with(implied(bools, lo, memo), false);
            let hi = with(implied(bools, hi, memo), true);
            match (lo, hi) {
                (None, x) | (x, None) => x,
                (Some(a), Some(c)) => Some(a.into_iter().filter(|l| c.contains(l)).collect()),
            }
        }
    };
    memo.insert(b, r.clone());
    r
}

/// The atoms `b` depends on once those in `known` take their values.
fn support_given(bools: &Bools, b: Bool, known: &HashMap<Atom, bool>) -> Vec<Atom> {
    let (mut seen, mut out, mut stack) = (HashSet::new(), Vec::new(), vec![b]);
    while let Some(b) = stack.pop() {
        let Some((atom, lo, hi)) = bools.node(b) else {
            continue;
        };
        if !seen.insert(b) {
            continue;
        }
        match known.get(&atom) {
            Some(true) => stack.push(hi),
            Some(false) => stack.push(lo),
            None => {
                if !out.contains(&atom) {
                    out.push(atom);
                }
                stack.extend([lo, hi]);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::{MeshIo, NativeProgram, RootParam, RootType, StageIo};

    const S_ENDPGM: u32 = 0xBF81_0000;
    /// `exp prim, v9, off, off, off done`.
    const EXP_PRIM: [u32; 2] = [0xF800_0941, 0x0000_0009];
    /// `v_mbcnt_lo_u32_b32 v0, -1, 0`: the lane number (below 32).
    const LANE: [u32; 2] = [0xD765_0000, 0x0001_00C1];
    const S_MOV_S4_7: u32 = 0xBE84_0387; // s_mov_b32 s4, 7
    const V_MOV_V1_S4: u32 = 0x7E02_0204; // v_mov_b32 v1, s4
    const RFL_S5_V1: u32 = 0x7E0A_0501; // v_readfirstlane_b32 s5, v1

    /// A wave64 mesh program of two console waves (128 lanes): `words`,
    /// then a primitive export; user SGPR 0 holds a descriptor table.
    fn mesh(words: &[u32]) -> NativeProgram {
        let mut code: Vec<u32> = words.to_vec();
        code.extend(EXP_PRIM);
        code.push(S_ENDPGM);
        NativeProgram {
            stage: crate::HwStage::Ms,
            wave64: true,
            float_mode: 0,
            vgprs: 16,
            layout: vec![RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            }],
            used: Vec::new(),
            io: StageIo::Ms(MeshIo {
                outputs: Vec::new(),
                clip_cntl: 0,
                threads: 128,
                max_vertices: 128,
                max_primitives: 128,
                topology: 2,
                lds_bytes: 0,
            }),
            code: code.iter().flat_map(|w| w.to_le_bytes()).collect(),
        }
    }

    /// The program of `words` ([`mesh`]) and its analysis.
    fn analyse(words: &[u32]) -> (Program, Uniformity) {
        let native = mesh(words);
        let p = crate::ir::build(&native.patched_code(), true, native.vgprs).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        let u = Uniformity::of(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 128,
            target: crate::emit::Target {
                lanes_min: 32,
                lanes_max: 32,
                stride_table: false,
            },
            console_cf: false,
            per_lane_only: false,
        });
        (p, u)
    }

    /// The levels of the values the program's first-lane reads read, in
    /// code order.
    fn first_lanes(words: &[u32]) -> Vec<Level> {
        let (p, u) = analyse(words);
        let mut reads: Vec<(usize, Value)> = p
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .filter(|i| i.name() == "v_readfirstlane_b32")
            .map(|i| (i.machine.offset, i.writes[0].1))
            .collect();
        reads.sort();
        reads
            .into_iter()
            .map(|(_, w)| {
                let l = u.first_lane(w).expect("analysed");
                // The read's result is the console wave's either way.
                assert_eq!(u.level(w), l.min(Level::Wave));
                l
            })
            .collect()
    }

    /// The control level of each block holding an `s_barrier`, in code
    /// order.
    fn barriers(words: &[u32]) -> Vec<Level> {
        let (p, u) = analyse(words);
        let mut out: Vec<(usize, Level)> = Vec::new();
        for (b, block) in p.blocks.iter().enumerate() {
            for i in block.insts.iter().filter(|i| i.name() == "s_barrier") {
                out.push((i.machine.offset, u.control(b)));
            }
        }
        out.sort();
        out.into_iter().map(|(_, l)| l).collect()
    }

    /// Constants, the group id and what they compute are the group's; the
    /// console wave's index is its wave's; the lane number is the lane's.
    #[test]
    fn inputs_and_what_they_compute() {
        let lane = [&LANE[..], &[0x7E0A_0500]].concat(); // v_readfirstlane_b32 s5, v0
        assert_eq!(first_lanes(&lane), [Level::Lane]);
        let words = [
            0x7E04_0505, // v_readfirstlane_b32 s2, v5 (the group id)
            S_MOV_S4_7,
            V_MOV_V1_S4,
            RFL_S5_V1,
            0x7E04_0203, // v_mov_b32 v2, s3 (the merged wave info)
            0x7E0C_0502, // v_readfirstlane_b32 s6, v2
        ];
        assert_eq!(
            first_lanes(&words),
            [Level::Group, Level::Group, Level::Wave]
        );
        // With the lane number added, the value is the lane's.
        let mixed = [
            &LANE[..],
            &[
                S_MOV_S4_7,
                0x4A02_0004, // v_add_nc_u32 v1, s4, v0
                RFL_S5_V1,
            ],
        ]
        .concat();
        assert_eq!(first_lanes(&mixed), [Level::Lane]);
    }

    /// A lane bit selected from compares of the group's values is the
    /// group's (BTP 6's validity flags of its mesh counts); from a lane's
    /// compare, the lane's.
    #[test]
    fn selects_on_masks_take_the_masks_level() {
        let uniform = [
            S_MOV_S4_7,
            0xBF06_8004, // s_cmp_eq_u32 s4, 0
            0x8580_807E, // s_cselect_b64 s[0:1], exec, 0
            0xD501_0001,
            0x0001_0280, // v_cndmask_b32 v1, 0, 1, s[0:1]
            RFL_S5_V1,
        ];
        assert_eq!(first_lanes(&uniform), [Level::Group]);
        let lane = [
            &LANE[..],
            &[
                0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
                0xD501_0001,
                0x01A9_0280, // v_cndmask_b32 v1, 0, 1, vcc
                RFL_S5_V1,
            ],
        ]
        .concat();
        assert_eq!(first_lanes(&lane), [Level::Lane]);
    }

    /// After a branch lanes take differently, a register one side writes is
    /// the lane's where the sides meet; after a scalar branch, the group's.
    /// A write under a narrowed EXEC with no branch (a guard) is the lane's
    /// too.
    #[test]
    fn joins_and_guards() {
        let lane_branch = [
            &LANE[..],
            &[
                0x7E02_0280, // v_mov_b32 v1, 0
                0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
                0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
                0xBF88_0001, // s_cbranch_execz +1
                0x7E02_0281, // v_mov_b32 v1, 1
                0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
                RFL_S5_V1,
            ],
        ]
        .concat();
        assert_eq!(first_lanes(&lane_branch), [Level::Lane]);
        let scalar_branch = [
            0x7E02_0280, // v_mov_b32 v1, 0
            S_MOV_S4_7,
            0xBF06_8004, // s_cmp_eq_u32 s4, 0
            0xBF85_0001, // s_cbranch_scc1 +1
            0x7E02_0281, // v_mov_b32 v1, 1
            RFL_S5_V1,
        ];
        assert_eq!(first_lanes(&scalar_branch), [Level::Group]);
        let guard = [
            &LANE[..],
            &[
                0x7E02_0280, // v_mov_b32 v1, 0
                0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
                0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
                0x7E02_0281, // v_mov_b32 v1, 1
                0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
                RFL_S5_V1,
            ],
        ]
        .concat();
        assert_eq!(first_lanes(&guard), [Level::Lane]);
    }

    /// A scalar counted in a loop: the group's after it where every lane
    /// leaves at the same iteration, the lane's where each leaves at its
    /// own; inside the loop, the group's either way.
    #[test]
    fn loops_lanes_leave_apart() {
        let uniform = [
            0xBE84_0380, // s_mov_b32 s4, 0
            // loop:
            0x8104_8104, // s_add_i32 s4, s4, 1
            0xBF0A_8404, // s_cmp_lt_u32 s4, 4
            0xBF85_FFFD, // s_cbranch_scc1 loop
            V_MOV_V1_S4,
            RFL_S5_V1,
        ];
        assert_eq!(first_lanes(&uniform), [Level::Group]);
        let apart = [
            &LANE[..],
            &[
                0xBE84_0380, // s_mov_b32 s4, 0
                0xBE8A_047E, // s_mov_b64 s[10:11], exec
                // loop:
                0x8104_8104, // s_add_i32 s4, s4, 1
                V_MOV_V1_S4,
                RFL_S5_V1,   // every lane still looping has the same s4
                0x7D82_0004, // v_cmp_lt_u32 vcc, s4, v0
                0x8AFE_6A7E, // s_andn2_b64 exec, exec, vcc
                0xBF89_FFFA, // s_cbranch_execnz loop
                0xBEFE_040A, // s_mov_b64 exec, s[10:11]
                0x7E04_0204, // v_mov_b32 v2, s4
                0x7E0C_0502, // v_readfirstlane_b32 s6, v2
            ],
        ]
        .concat();
        assert_eq!(first_lanes(&apart), [Level::Group, Level::Lane]);
    }

    /// VIctory issue #1498's program (by r4tedx), in its wave64 form: lanes
    /// that are done join an exit mask, EXEC loses them, and the mask is
    /// counted after the loop. On the console every lane counts 64; each
    /// lane of the per-lane form knows only its own path. The count (and
    /// any value made from it) is the lane's, never proven alike. What
    /// pins it is the rule for masks read as numbers (the exit mask is made
    /// of compares' lane bits); the loop-exit rules alone are pinned by
    /// the tests after it.
    #[test]
    fn a_mask_counted_after_a_loop_lanes_leave_apart() {
        let words = [
            &LANE[..],
            &[
                0x4A02_0081, // v_add_nc_u32 v1, 1, v0 (lane i loops i + 1 times)
                0xBE8A_0480, // s_mov_b64 s[10:11], 0 (the exit mask)
                // loop:
                0x4E02_0281, // v_subrev_nc_u32 v1, 1, v1
                0x7D84_0280, // v_cmp_eq_u32 vcc, 0, v1
                0x888A_0A6A, // s_or_b64 s[10:11], vcc, s[10:11]
                0x8AFE_0A7E, // s_andn2_b64 exec, exec, s[10:11]
                0xBF89_FFFB, // s_cbranch_execnz loop
                0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
                0xBE89_100A, // s_bcnt1_i32_b64 s9, s[10:11]
                0x7E04_0209, // v_mov_b32 v2, s9
                0x7E18_0502, // v_readfirstlane_b32 s12, v2
            ],
        ]
        .concat();
        assert_eq!(first_lanes(&words), [Level::Lane]);
    }

    /// A barrier inside a loop that lanes leave at different iterations
    /// runs a different number of times in each lane: not the group's,
    /// though every lane enters the loop. The loop's own branch decides how
    /// often its blocks run (control dependence through the back edge).
    /// After the loop every lane gets there once; with a scalar exit the
    /// barrier inside is the group's too.
    #[test]
    fn barriers_in_a_loop_lanes_leave_apart() {
        let body = |exit: &[u32]| -> Vec<u32> {
            [
                &LANE[..],
                &[
                    0xBE84_0380, // s_mov_b32 s4, 0
                    0xBE8A_047E, // s_mov_b64 s[10:11], exec
                    // loop:
                    0x8104_8104, // s_add_i32 s4, s4, 1
                    0xBF8A_0000, // s_barrier
                ],
                exit,
                &[
                    0xBEFE_040A, // s_mov_b64 exec, s[10:11]
                    0xBF8A_0000, // s_barrier
                ],
            ]
            .concat()
        };
        let apart = body(&[
            0x7D82_0004, // v_cmp_lt_u32 vcc, s4, v0
            0x8AFE_6A7E, // s_andn2_b64 exec, exec, vcc
            0xBF89_FFFB, // s_cbranch_execnz loop
        ]);
        holds(&apart, &["s_barrier", "s_cbranch_execnz label_0010"]);
        assert_eq!(barriers(&apart), [Level::Lane, Level::Group]);
        let together = body(&[
            0xBF0A_8404, // s_cmp_lt_u32 s4, 4
            0xBF85_FFFC, // s_cbranch_scc1 loop
        ]);
        holds(&together, &["s_cbranch_scc1 label_0010"]);
        assert_eq!(barriers(&together), [Level::Group, Level::Group]);
    }

    /// Lanes that end the program on their own (`s_endpgm` behind a branch
    /// on their EXEC bit) never reach a barrier after it: not the group's.
    /// Behind a branch on a scalar every lane ends or none does. (Ends lead
    /// to one exit node in the post-dominators.)
    #[test]
    fn a_barrier_after_lanes_end_the_program() {
        let lane = [
            &LANE[..],
            &[
                0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
                0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
                0xBF89_0001, // s_cbranch_execnz +1
                S_ENDPGM,    // the lanes from 20 on end here
                0xBF8A_0000, // s_barrier
            ],
        ]
        .concat();
        holds(&lane, &["s_cbranch_execnz label_0018", "s_barrier"]);
        assert_eq!(barriers(&lane), [Level::Lane]);
        let scalar = [
            S_MOV_S4_7,
            0xBF06_8004, // s_cmp_eq_u32 s4, 0
            0xBF85_0001, // s_cbranch_scc1 +1
            S_ENDPGM,    // every lane ends here or none does
            0xBF8A_0000, // s_barrier
        ];
        holds(&scalar, &["s_cbranch_scc1 label_0010"]);
        assert_eq!(barriers(&scalar), [Level::Group]);
    }

    /// A loop header reached by two back edges, from the two sides of a
    /// branch on the lane: each side sets s6 to a constant of its own, so
    /// the header's phi has the group's arguments only, yet lanes come back
    /// with different values. The header's phi takes the level of such
    /// branches ([`Loop::returning`]). With one back edge after the sides
    /// meet, s6 is the lane's by the join's phi instead, and with a branch
    /// on the loop counter, which every lane takes alike, the group's.
    /// (A branch on VCC tests the lane's EXEC bit too: never the group's.)
    #[test]
    fn a_header_reached_by_two_back_edges() {
        // `branch`: a compare, and a branch to `other` on it.
        let words = |branch: [u32; 2], side: u32| -> Vec<u32> {
            [
                &LANE[..],
                &[
                    0xBE84_0380, // s_mov_b32 s4, 0
                    0xBE86_0380, // s_mov_b32 s6, 0
                    // head:
                    0x7E02_0206, // v_mov_b32 v1, s6
                    RFL_S5_V1,
                    0x8104_8104, // s_add_i32 s4, s4, 1
                    0xBF09_8404, // s_cmp_ge_u32 s4, 4
                    0xBF85_0006, // s_cbranch_scc1 exit
                    branch[0],
                    branch[1],
                    0xBE86_0381, // s_mov_b32 s6, 1
                    side,        // back to head, or to the join
                    // other:
                    0xBE86_0382, // s_mov_b32 s6, 2
                    0xBF82_FFF5, // s_branch head
                                 // exit:
                ],
            ]
            .concat()
        };
        let lane = [
            0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
            0xBF86_0002, // s_cbranch_vccz other
        ];
        let scalar = [
            0xBF06_8204, // s_cmp_eq_u32 s4, 2
            0xBF85_0002, // s_cbranch_scc1 other
        ];
        let back = 0xBF82_FFF7; // s_branch head
        let to_join = 0xBF82_0001; // s_branch +1 (the other side's branch)
        holds(
            &words(lane, back),
            &[
                "v_cmp_gt_u32 vcc, 20, v0",
                "s_cbranch_scc1 label_003C",
                "s_cbranch_vccz label_0034",
                "s_branch label_0010",
            ],
        );
        holds(
            &words(scalar, to_join),
            &[
                "s_cmp_eq_u32 s4, 2",
                "s_cbranch_scc1 label_0034",
                "s_branch label_0038",
            ],
        );
        assert_eq!(first_lanes(&words(lane, back)), [Level::Lane]);
        assert_eq!(first_lanes(&words(lane, to_join)), [Level::Lane]);
        assert_eq!(first_lanes(&words(scalar, back)), [Level::Group]);
    }

    /// Two exits of a loop, one on the lane (leaving at the first
    /// iteration) and one on a scalar (at the fourth): each exit's block
    /// sets s6 to a constant of its own, outside the loop, so the phi
    /// where they meet has the group's arguments only, and the only branch
    /// it is a join point of is the scalar one. Lanes still arrive by
    /// different exits: the phi takes the loop's leaving level
    /// ([`Run::exit_joins_of`]). With the first exit on the loop counter
    /// too (at the second iteration), the group's.
    #[test]
    fn exits_that_meet_after_the_loop() {
        // `branch`: a compare, and a branch back to `head` on it.
        let words = |branch: [u32; 2]| -> Vec<u32> {
            [
                &LANE[..],
                &[
                    0xBE84_0380, // s_mov_b32 s4, 0
                    // head:
                    0xBF09_8404, // s_cmp_ge_u32 s4, 4
                    0xBF85_0005, // s_cbranch_scc1 late
                    0x8104_8104, // s_add_i32 s4, s4, 1
                    branch[0],
                    branch[1],
                    // early:
                    0xBE86_0381, // s_mov_b32 s6, 1
                    0xBF82_0001, // s_branch join
                    // late:
                    0xBE86_0382, // s_mov_b32 s6, 2
                    // join:
                    0x7E02_0206, // v_mov_b32 v1, s6
                    RFL_S5_V1,
                ],
            ]
            .concat()
        };
        let lane = [
            0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
            0xBF86_FFFB, // s_cbranch_vccz head
        ];
        let scalar = [
            0xBF0A_8204, // s_cmp_lt_u32 s4, 2
            0xBF85_FFFB, // s_cbranch_scc1 head
        ];
        holds(
            &words(lane),
            &[
                "v_cmp_gt_u32 vcc, 20, v0",
                "s_cbranch_scc1 label_0028",
                "s_cbranch_vccz label_000C",
                "s_branch label_002C",
            ],
        );
        holds(
            &words(scalar),
            &["s_cmp_lt_u32 s4, 2", "s_cbranch_scc1 label_000C"],
        );
        assert_eq!(first_lanes(&words(lane)), [Level::Lane]);
        assert_eq!(first_lanes(&words(scalar)), [Level::Group]);
    }

    /// `vccz`, `execz` and `lds_direct` are the lane's, whatever VCC, EXEC
    /// and M0 hold: a bit of a whole wave's mask, which no lane holds in
    /// the per-lane form, and a dword of LDS. (VCC here has the group's
    /// lane bit, M0 the group's value.) Scalar instructions too.
    #[test]
    fn special_operands_are_the_lanes() {
        let vcc = [
            S_MOV_S4_7,
            0xBF06_8004, // s_cmp_eq_u32 s4, 0
            0x85EA_807E, // s_cselect_b64 vcc, exec, 0
        ];
        for (words, what) in [
            (&[0x7E02_02FB][..], "v_mov_b32 v1, src_vccz"),
            (&[0x7E02_02FC], "v_mov_b32 v1, src_execz"),
            (
                &[
                    0xBEFC_0380, // s_mov_b32 m0, 0
                    0x7E02_02FE, // v_mov_b32 v1, src_lds_direct
                ],
                "v_mov_b32 v1, src_lds_direct",
            ),
            (
                &[
                    0xBEEA_0480, // s_mov_b64 vcc, 0
                    0xBE84_03FB, // s_mov_b32 s4, src_vccz
                    0x7E02_0204, // v_mov_b32 v1, s4
                ],
                "s_mov_b32 s4, src_vccz",
            ),
        ] {
            let p = [&vcc[..], words, &[RFL_S5_V1]].concat();
            holds(&p, &[what]);
            assert_eq!(first_lanes(&p), [Level::Lane], "{what}");
        }
    }

    /// A lane mask read lane by lane, as a select or a carry-in, counts by
    /// its lane bit (the group's here, as in BTP 6's validity flags); read
    /// as a number by any other VALU operation, it is the lane's.
    #[test]
    fn masks_count_by_their_lane_bit_only_as_selects_and_carries() {
        let mask = [
            S_MOV_S4_7,
            0xBF06_8004, // s_cmp_eq_u32 s4, 0
            0x8580_807E, // s_cselect_b64 s[0:1], exec, 0
        ];
        for (words, what, want) in [
            (
                [0xD501_0001, 0x0001_0280],
                "v_cndmask_b32 v1, 0, 1, s[0:1]",
                Level::Group,
            ),
            (
                [0xD528_6A01, 0x0001_0080],
                "v_add_co_ci_u32 v1, vcc, 0, 0, s[0:1]",
                Level::Group,
            ),
            (
                [0xD525_0001, 0x0001_0000],
                "v_add_nc_u32 v1, s0, 0",
                Level::Lane,
            ),
        ] {
            let p = [&mask[..], &words, &[RFL_S5_V1]].concat();
            holds(&p, &[what]);
            assert_eq!(first_lanes(&p), [want], "{what}");
        }
    }

    /// Asserts that the program of `words` ([`mesh`]) holds each
    /// instruction of `want`: the hand-assembled words read as written (a
    /// wrong word would test another program).
    fn holds(words: &[u32], want: &[&str]) {
        let p = mesh(words);
        let text: Vec<String> = crate::gfx10::decode(&p.code)
            .unwrap()
            .iter()
            .map(|i| i.text(true))
            .collect();
        for w in want {
            assert!(text.iter().any(|t| t == w), "{w}: {text:#?}");
        }
    }

    /// Memory the program writes may change between the loads of two host
    /// waves: a load at the group's address is the group's only in a
    /// program without stores.
    #[test]
    fn loads_of_memory_the_program_writes() {
        let load = [
            0xBE82_0308, // s_mov_b32 s2, s8 (user SGPR 0 of a mesh program)
            0xBE83_0380, // s_mov_b32 s3, 0
            0xF408_0101,
            0xFA00_0000, // s_load_dwordx4 s[4:7], s[2:3], null
            0x7E06_0280, // v_mov_b32 v3, 0
            0xE030_1000,
            0x8001_0103, // buffer_load_dword v1, v3, s[4:7], 0 offen
            0x7E18_0501, // v_readfirstlane_b32 s12, v1
        ];
        assert_eq!(first_lanes(&load), [Level::Group]);
        let store = [
            &load[..],
            &[
                0xE070_1000,
                0x8001_0203, // buffer_store_dword v2, v3, s[4:7], 0 offen
            ],
        ]
        .concat();
        assert_eq!(first_lanes(&store), [Level::Lane]);
    }

    /// A barrier in a block only some lanes enter is not reached by the
    /// whole group; one where the sides have met again is, and so is one
    /// behind a branch on the group's values.
    #[test]
    fn barriers_where_lanes_branch_apart() {
        let words = [
            &LANE[..],
            &[
                0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
                0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
                0xBF88_0001, // s_cbranch_execz +1
                0xBF8A_0000, // s_barrier
                0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
                0xBF8A_0000, // s_barrier
            ],
        ]
        .concat();
        assert_eq!(barriers(&words), [Level::Lane, Level::Group]);
        let scalar = [
            S_MOV_S4_7,
            0xBF06_8004, // s_cmp_eq_u32 s4, 0
            0xBF85_0001, // s_cbranch_scc1 +1
            0xBF8A_0000, // s_barrier
            0xBF8A_0000, // s_barrier
        ];
        assert_eq!(barriers(&scalar), [Level::Group, Level::Group]);
    }
}
