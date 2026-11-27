//! IR to DXIL, one machine instruction at a time (decision 0003).
//!
//! Every register dword is an `i32` (SCC an `i1`); float operations bitcast
//! their operands. Lane masks exist only as the per-lane booleans of
//! structurisation; branches and guards are built from them. A value the
//! emitter cannot express is `Missing` with a reason, and only fails the
//! program when something with an effect (an export, a branch) needs it.
//!
//! # Files
//!
//! The lowering is one `Lower` value whose methods are spread over this
//! folder by what they lower, so that the backlog items of
//! `docs/analysis/shader.md` (section 6) each edit files of their own. A
//! method defined outside `mod.rs` is `pub(super)`: all of `lower` sees it.
//!
//! * `mod.rs`: `Lower`, its construction (`new`), the walk over blocks
//!   (`run`) and the dispatch by encoding (`inst`). An item that adds an
//!   encoding or a stage adds an arm there and nothing more.
//! * `operand.rs`: operands, results and the lane guards of writes.
//! * `flow.rs`: block exits, loops and their iteration limit, phis.
//! * `spin.rs`: loops that only wait for memory to change, which stop
//!   sooner.
//! * `mask.rs`: lane masks as booleans and as numbers. **M1**, **M2**.
//! * `compare.rs`: `v_cmp_*`, `v_cmpx_*`. **M1**.
//! * `lanes.rs`: whole-wave sections, DPP, lane reads, thread ids and
//!   console lanes. **L1**, **L2**, **L3**, **M2**; **S3** for
//!   `v_permlanex16_b32`.
//! * `mesh.rs`: mesh output counts and thread index. **G2**, **L4**.
//! * `salu.rs`: SOP1/SOP2/SOPC/SOPK. **S3** (`s_getpc_b64`), **S4**
//!   (`s_setreg*`, `s_round_mode`, `s_denorm_mode`), **M2** (`s_bcnt1`,
//!   `s_ff1` on masks).
//! * `smem.rs`: SMEM: constant buffers, scalar loads read as descriptor
//!   words or as data (`descriptor_data`), the constant table. **R1**,
//!   **R2**.
//! * `descriptor.rs`: host handles, heap indices, root descriptors,
//!   per-lane and waterfall descriptors. **R4**, **R3** (`used_kind`).
//! * `buffer.rs`: MUBUF/MTBUF and the buffer formats, `global_load_*`.
//!   **R3**.
//! * `valu.rs`: the VALU dispatch, float and conversion operations. **S3**
//!   (`v_cvt_f32_f64`), **S4** (non-fused MAD), **S5** (f64), **M1**
//!   (`v_cndmask_b32` on an untracked mask).
//! * `valu16.rs`: 16-bit and packed VALU. **S5**.
//! * `integer.rs`: integer VALU. **S5**.
//! * `image.rs`: MIMG. **S1** (A16 addressing and the sample forms: the
//!   address and sample parts of `image`), **I1** (64-bit atomics: its
//!   atomic branch).
//! * `lds.rs`: `ds_*`. **S3** (byte and half accesses), **G2** (the ESGS
//!   ring).
//! * `input.rs`: user data, system values, vertex and pixel inputs,
//!   interpolation. **P1**, **P2**, **S3** (hardware SGPR inputs).
//! * `export.rs`: `exp`, 16-bit outputs. **S3** (conditional exports),
//!   **G2**.
//! * `tests.rs`: unit tests of the lowering.
//!
//! Outside this folder: `structure.rs` (M1, M2), `resource.rs` (R1-R4),
//! `io.rs` (P1, P2, G2), `native.rs` (P1, G1), `emit/wave.rs` (M2, L1,
//! L3), `emit/mod.rs` (I1's binding notes, G1's entry point, L1's
//! translation option), `emit/shader.rs` and `emit/container.rs` (S2, P2),
//! `emit/bitcode.rs` (S4). Each item's gain is what `xtp shader-survey`
//! measures before and after it (`apps/xtp/src/survey.rs`).

mod buffer;
#[cfg(test)]
pub(crate) use buffer::{INTEGER_LOAD_PROGRAM_TEST, NORM_QUOTIENTS, code_fingerprint};
mod compare;
mod descriptor;
mod export;
mod flow;
pub(super) mod geometry;
mod image;
mod input;
mod integer;
mod lanes;
mod lds;
mod mask;
mod mesh;
mod operand;
// Inline ray queries in place of traversal walks (RT phase 2b).
pub(super) mod rayquery;
mod salu;
mod smem;
mod smem_data;
mod spin;
#[cfg(test)]
mod tests;
pub(crate) mod uniform;
mod valu;
mod valu16;

use std::collections::{BTreeMap, HashMap, HashSet};

use super::UserData;
use super::bitcode::{Attr, V};
use super::container::SigElement;
use super::shader::Shader;
use crate::ShaderError;
use crate::gfx10::{Fields, Operand};
use crate::io::{Export, Interp, Io};
use crate::ir::{EXEC, Inst, Program, Reg, Value};
use crate::native::{HwStage, NativeProgram, StageIo};
use crate::resource::{Address, Resources, Role, Site, Source};
use crate::structure::{Atom, Bool, LaneExit, Structured};
use export::half_outputs;
use mask::{MaskData, reads_mask_as_data};

type Why = &'static str;

#[derive(Debug, Clone, Copy)]
enum Val {
    I32(V),
    Bit(V),
    Missing(Why),
}

pub(crate) struct Lowered {
    pub dxil: Vec<u8>,
    /// The form the program was lowered in, for diagnostics
    /// (`emit::last_module_summary`): "per lane", "per lane in halves",
    /// "console flow" or "console flow in halves".
    pub flow: &'static str,
    /// Bit n: immediate sampler n is used for comparisons.
    pub cmp_samplers: u64,
    pub root_cbvs: Vec<u8>,
    pub root_srvs: Vec<u8>,
    pub root_uavs: Vec<u8>,
    /// The program reads the host's runtime stride table.
    pub stride_table: bool,
    /// Buffers read (and written) through view-heap slots the program
    /// picks at run time ([`super::RUN_TIME_READS`], [`super::RUN_TIME_WRITES`]).
    pub run_time_buffers: u8,
    pub user_data: Vec<UserData>,
    pub group_base: Option<u32>,
    pub samplers: Vec<[u32; 4]>,
    /// Heap handles made with a non-uniform index.
    #[cfg(test)]
    pub nonuniform_handles: u32,
    /// Typed buffer loads that move 16-bit integers (`bufferLoad.i32`):
    /// d16 ones, and plain ones of `buffer::INTEGER_LOAD_PROGRAMS`.
    #[cfg(test)]
    pub integer_loads: u32,
    /// Whether the program uses wave operations.
    #[cfg(test)]
    pub wave_ops: bool,
    /// Loops given the spin-wait limit.
    #[cfg(test)]
    pub spin_loops: usize,
    /// The limit each back edge compares its loop's count with, in the
    /// order they were lowered.
    #[cfg(test)]
    pub loop_limits: Vec<u32>,
    /// Scalar memory reads through a per-lane descriptor put under a lane
    /// guard (`Lower::scalar_guard`).
    #[cfg(test)]
    pub scalar_guards: u32,
}

#[derive(Clone)]
pub(crate) struct Parts<'a> {
    pub native: &'a NativeProgram,
    pub p: &'a Program,
    pub st: &'a Structured,
    pub io: &'a Io,
    pub res: &'a Resources,
    pub inputs: Vec<SigElement>,
    pub outputs: Vec<SigElement>,
    pub prims: Vec<SigElement>,
    /// Mesh: what the group id arrives multiplied by.
    pub mesh_stride: u32,
    /// The wave sizes the host runs.
    pub target: super::Target,
    /// Run the program as the console does (see [`CONSOLE_FLOW`]); `st`
    /// is then [`crate::structure::structure_console`]'s.
    pub console_cf: bool,
    /// Never ask for the console flow: the per-lane form or nothing (what
    /// a program falls back to when the console flow cannot lower it).
    pub per_lane_only: bool,
}

/// What [`lower`] stops with when a program should run as the console runs
/// it: lower it again from [`crate::structure::structure_console`] with
/// [`Parts::console_cf`]. Every lane of a console wave then runs every
/// block, branches are the console's (taken wave-wide) and EXEC guards the
/// vector instructions. Two kinds of program ask for it:
///
/// * a wave64 program whose two 32-lane halves must exchange values on a
///   host without 64-lane waves (a group of one console wave): the wave-wide
///   tests come from masks both halves exchange through group-shared
///   memory, and both halves reach every exchange, so a group barrier
///   orders it;
/// * on any host, a program that reads or writes a named lane
///   (`v_readlane`, `v_writelane`) and has branches its lanes take
///   differently. Those instructions ignore EXEC on the console, and a wave
///   runs a loop while any lane does; in the per-lane form a lane that left
///   a loop (or skipped a branch) is not there to be read, and never sees
///   the scalars set inside. One host wave is the console wave here, so
///   nothing is exchanged. If the console flow cannot lower such a program
///   it keeps the per-lane form ([`Parts::per_lane_only`]).
pub(crate) const CONSOLE_FLOW: &str = "a wave64 program whose halves exchange lanes (console flow)";

enum Stop {
    /// A phi whose incoming value has no DXIL form was read: redo with it missing.
    Retry(Value, Why),
    Fail(ShaderError),
}

pub(super) const UNTRACKED_SELECT: Why = "a select on an untracked mask";

pub(crate) fn lower(parts: &Parts) -> Result<Lowered, ShaderError> {
    match lower_program(parts, false) {
        Err(ShaderError::Unsupported {
            what: UNTRACKED_SELECT,
            ..
        }) if parts.native.stage == HwStage::Cs => lower_program(parts, true),
        Err(original @ ShaderError::Unsupported { what, .. })
            if parts.native.stage == HwStage::Ps
                && matches!(
                    what,
                    "the first half of an interpolation" | "a hardware SGPR input"
                ) =>
        {
            let retry = |io: &mut Io| {
                io.initialized_sgprs = parts.native.io.ps_initialized_sgprs();
                lower_program(
                    &Parts {
                        io,
                        ..parts.clone()
                    },
                    false,
                )
            };
            if what == "a hardware SGPR input" {
                // Le nombre de SGPR définis à l'entrée ne change aucune paire.
                match retry(&mut parts.io.clone()) {
                    Ok(result) => return Ok(result),
                    Err(ShaderError::Unsupported {
                        what: "the first half of an interpolation",
                        ..
                    }) => {}
                    Err(_) => return Err(original),
                }
            }
            crate::io::raise_inputs(parts.p, parts.native, true)
                .and_then(|mut io| retry(&mut io))
                .map_err(|_| original)
        }
        result => result,
    }
}

fn lower_program(parts: &Parts, numeric_selects: bool) -> Result<Lowered, ShaderError> {
    let mut bad = HashMap::new();
    loop {
        match Lower::new(parts, &bad, numeric_selects).and_then(Lower::run) {
            Ok(l) => return Ok(l),
            Err(Stop::Retry(v, why)) => {
                if bad.insert(v, why).is_some() {
                    return Err(ShaderError::Unsupported {
                        offset: 0,
                        what: "a phi that does not settle",
                    });
                }
            }
            Err(Stop::Fail(e)) => return Err(e),
        }
    }
}

/// `s_sendmsg` id of a mesh or NGG stage's output allocation request.
const GS_ALLOC_REQ: u16 = 9;

struct Lower<'a> {
    s: Shader,
    x: &'a Parts<'a>,
    bad: &'a HashMap<Value, Why>,
    lane_fail: HashMap<Value, Why>,
    vals: HashMap<Value, Val>,
    floats: HashMap<Value, V>,
    /// DXIL block per IR block (unreachable: none).
    block: Vec<Option<usize>>,
    cur: usize,
    bools: HashMap<(usize, Bool), V>,
    lane_bits: HashMap<Value, V>,
    /// IR phi location by value.
    phi_at: HashMap<Value, (usize, usize)>,
    mask_phis: HashMap<Value, V>,
    pending_masks: Vec<(V, usize, usize)>,
    data_phis: Vec<(V, usize, usize)>,
    read_phis: HashSet<Value>,
    /// Une somme f64 de lanes masquées lit les votes de la vague console.
    emitted_double_count: bool,
    params: HashMap<Value, (usize, u32)>,
    user_data: Option<V>,
    group_base_row: Option<V>,
    host_words: Vec<(UserData, V)>,
    host_rows: Vec<V>,
    partial_active: Option<V>,
    user_reads: BTreeMap<u32, UserData>,
    samplers: Vec<[u32; 4]>,
    descriptors: HashMap<(Site, bool), &'a Source>,
    scalar: HashMap<Site, Address>,
    exports: HashMap<Site, &'a Export>,
    interps: HashMap<Site, &'a Interp>,
    handles: HashMap<(usize, Site, u8, bool, u32), V>,
    /// Runtime stride table entries read in a DXIL block, by the
    /// descriptor's load (`descriptor::Entry`).
    entries: HashMap<(usize, Site, u8), descriptor::Entry>,
    /// [`Lowered::run_time_buffers`] so far.
    run_time_buffers: u8,
    /// [`Lowered::integer_loads`] so far.
    #[cfg(test)]
    integer_loads: u32,
    /// Blocks splitting the edge from one IR block to another.
    edges: HashMap<(usize, usize), usize>,
    /// Per IR block: a `done` export is behind every lane entering it.
    exported: Vec<bool>,
    /// Reverse-postorder index per IR block.
    rpo: Vec<usize>,
    /// DXIL block each IR block ends in (guarded stores split blocks).
    tail: Vec<Option<usize>>,
    /// The program writes UAVs: barriers fence them too.
    writes_uavs: bool,
    /// src0 of the current instruction as DPP fetched it.
    src0: Option<V>,
    /// Compute code that reads other lanes: thread ids follow the host
    /// lanes, so lane n is console lane n.
    console_lanes: bool,
    /// A wave64 program lowered in 32-lane halves. In paired mode both
    /// register sets belong to the same host wave.
    split: bool,
    /// Split, and run as the console runs it ([`CONSOLE_FLOW`]).
    console_cf: bool,
    /// Split without [`Self::console_cf`] cannot run it, the console flow can.
    wants_console_flow: bool,
    /// Why a wave64 program that needs its whole wave cannot take the
    /// console flow on a host without 64-lane waves.
    no_console_flow: Option<Why>,
    /// The console thread's flat id in the group (relabelled; see
    /// [`Self::thread_id`]).
    console_flat: Option<V>,
    /// Compute: the console's group shape where the host's is declared as
    /// one row of as many threads, for the quads of implicit derivatives
    /// (see `Lower::new`); the thread ids then come from the flat one
    /// ([`Self::flat_thread_id`]).
    flat_group: Option<[u32; 3]>,
    /// Group-shared slots the two halves exchange a dword through.
    xchg: Option<V>,
    /// Split, in a group of several console waves, each lane on its own
    /// path: lane reads at a constant lane go through group-shared memory
    /// (see `Lower::new`).
    group_reads: bool,
    /// The group-shared memory of those lane reads
    /// ([`Self::group_read_lane`]).
    lane_slots: Option<V>,
    /// Mesh: the thread index in the group (made in the entry block).
    flat: Option<V>,
    /// Mesh: le dernier export d'indices de chaque lane, dans un dword privé.
    mesh_primitive: Option<V>,
    /// Mesh: IR blocks whose `s_barrier` is left out.
    no_barrier: HashSet<usize>,
    /// Mesh: which values are the same in every lane (`mesh_prologue`).
    uniform: Option<uniform::Uniformity>,
    /// Vertex and mesh outputs holding 16-bit values: row, whether two per
    /// dword, and the row packed after it.
    half_outputs: HashMap<u8, (bool, Option<u8>)>,
    /// How lane masks become data.
    mask_data: MaskData,
    /// A pixel program's masks' helper lanes (empty in other programs).
    helper_bits: mask::HelperBits,
    /// The instruction being lowered may read lane masks as data.
    mask_ok: bool,
    /// Ballots of masks read as data, by low dword: (low, high).
    ballots: HashMap<Value, (V, V)>,
    /// Masks held as numbers whose number is being read (see `atom`).
    opaque_open: HashSet<Value>,
    /// Masks whose ballot is being built: a mask whose lane bit needs its
    /// own value as a number (an opaque mask keyed by a dword it shares
    /// with another) would otherwise recurse until the stack overflows.
    balloting: HashSet<Value>,
    /// Block writing each value.
    def_block: HashMap<Value, usize>,
    /// IR block being lowered.
    cur_ir: usize,
    /// Whole-wave sections by block and start index.
    sections: HashMap<usize, HashMap<usize, super::wave::Section>>,
    /// Values written in a section that code outside it reads.
    section_out: HashSet<Value>,
    /// The program's constant table (the code from the first byte read
    /// through `s_getpc_b64` on): global, first byte, dwords.
    code_table: Option<(V, u32, u32)>,
    /// Loop headers' iteration counters (see [`LOOP_LIMIT`]): the phi by
    /// IR block, and its incoming (value, DXIL block) pairs.
    counters: HashMap<usize, (V, Vec<(V, usize)>)>,
    /// The block a loop stopped by [`LOOP_LIMIT`] leaves the program from.
    abort: Option<usize>,
    /// Headers of loops that only wait for memory to change: they stop at
    /// [`super::spin_limit`] instead ([`spin`]).
    spin: HashSet<usize>,
    /// Headers of counted loops: they stop at [`super::counted_loop_limit`]
    /// instead ([`flow::counted_loops`]).
    counted: HashSet<usize>,
    /// Lanes that end keep passing group barriers ([`flow::Drain`]).
    drain: flow::Drain,
    /// The back edges of all loops together ([`flow::Budget`]).
    budget: flow::Budget,
    /// See [`Lowered::loop_limits`].
    #[cfg(test)]
    limits: Vec<u32>,
    /// See [`Lowered::scalar_guards`].
    #[cfg(test)]
    scalar_guards: u32,
    /// Values read other than as a resource access's descriptor: by an
    /// instruction that accesses no resource, or by a phi.
    data_reads: HashSet<Value>,
    /// A merged ES or GS half's plan ([`geometry::Merged`]).
    merged: Option<Box<geometry::Merged>>,
}

/// Iterations after which a loop is left (per entry into it), unless
/// [`super::loop_limit`] says otherwise. The console runs a loop as long as
/// its data says; data the host has not mirrored yet (a null view reads
/// zeros) can make that four billion iterations, which hangs the host GPU
/// until the driver resets it. A lane that reaches the limit leaves the
/// program as killed (a pixel discards).
///
/// 2^14 (16,384) because the host's GPU watchdog (TDR) removes the device
/// once a submission runs for about 2 s. In VIctory #452's world runs on an
/// RTX 5070, with 2^20 submissions took up to 10.6 s and the device was
/// removed; with 2^16 the longest was 1.1 s; with 2^14, 371 ms with no
/// device removal and the same frame rate. Loops that only wait for memory
/// stop sooner ([`spin`]); counted loops, whose count is the program's own,
/// later ([`COUNTED_LOOP_LIMIT`]).
// shortcut: a legitimate loop of more than 16,384 iterations per lane that
// is no counted loop (one that walks a list, say) is cut short, so its
// result is wrong there; raising `XTP_SHADER_LOOP_LIMIT` shows whether a
// wrong result comes from it.
pub(super) const LOOP_LIMIT: u32 = 1 << 14;

/// The limit of counted loops ([`flow::counted_loops`]): 2^20 (1,048,576),
/// the old general limit. Their iteration count comes from an induction
/// variable and a bound the loop does not change, so 2^14 would cut short
/// legitimate loops of 16,385 to 2^20 iterations.
// shortcut: #452's device removals were measured with 2^20 for every loop;
// a counted loop whose bound comes from data the host has not mirrored can
// still run that long.
pub(super) const COUNTED_LOOP_LIMIT: u32 = 1 << 20;

/// MODE's f32 denormal field (bits 4-5): bit 4 keeps input denormals, bit
/// 5 output ones.
const F32_DENORMALS: u32 = 0x30;

/// The `"fp32-denorm-mode"` of `main` (S11): how the program's f32
/// operations treat denormals, which DXIL declares once per program. The
/// program starts with the record's float mode (`native.float_mode`, 0xC0
/// in every record read so far: f32 denormals flushed, so `"ftz"`), and
/// `ir::float_modes` has refused a program whose f32 instructions run under
/// two denormal modes. MODE's f32 field 0 (flush both ways) is `"ftz"`, 3
/// (keep both) `"preserve"`.
// shortcut: none (DXIL's "any", as before) where the field keeps only one
// way (1 or 2: no DXIL mode), and where the program writes the field
// (`s_denorm_mode`, an `s_setreg*` of MODE over bits 4-5) other than with a
// constant equal to the entry's: the mode its f32 instructions then run
// under needs `ir::float_modes`'s flow to say. No program of the pack writes
// the field (shcuts' count of 606 records, task f3d80097).
pub(crate) fn fp32_denorm_mode(native: &NativeProgram, p: &Program) -> Option<&'static str> {
    let entry = u32::from(native.float_mode) & F32_DENORMALS;
    for inst in p.blocks.iter().flat_map(|b| &b.insts) {
        // What the instruction writes of MODE (none: a value it reads from a
        // register), and which bits.
        let (value, mask) = match (inst.name(), &inst.fields) {
            ("s_denorm_mode", Fields::Sopp { imm }) => (Some(u32::from(*imm) << 4), 0xF0),
            (
                "s_setreg_imm32_b32" | "s_setreg_b32",
                Fields::Sop {
                    imm: Some(hw), src, ..
                },
            ) if hw & 63 == 1 => {
                let offset = u32::from(hw >> 6 & 31);
                let width = u32::from(hw >> 11 & 31) + 1;
                let mask = (u32::MAX >> (32 - width)) << offset;
                let value = match src.first() {
                    Some(&Operand::Literal(x)) if inst.name() == "s_setreg_imm32_b32" => {
                        Some(x << offset & mask)
                    }
                    _ => None,
                };
                (value, mask)
            }
            _ => continue,
        };
        if mask & F32_DENORMALS == 0 {
            continue;
        }
        if mask & F32_DENORMALS != F32_DENORMALS || value.is_none_or(|v| v & F32_DENORMALS != entry)
        {
            return None;
        }
    }
    match entry >> 4 {
        0 => Some("ftz"),
        3 => Some("preserve"),
        _ => None,
    }
}

/// Where half `h` (dword h / 2, the high half when odd) of a parameter
/// packing 16-bit rows `a` and `b` two per dword comes from: a row and
/// channel of `elements`, none for padding. Channel c of `a` is half c;
/// `b` starts at the dword after `a`'s last channel.
fn packed_half(elements: &[SigElement], a: u8, b: Option<u8>, h: u8) -> Option<(u8, u8)> {
    let extent = |r: u8| {
        elements
            .iter()
            .filter(|e| e.row == Some(u32::from(r)))
            .map(|e| e.col + e.cols)
            .max()
    };
    let na = extent(a)?;
    let a_halves = na.div_ceil(2) * 2;
    if h < a_halves {
        return (h < na).then_some((a, h));
    }
    let b = b?;
    let c = h - a_halves;
    (c < extent(b)?).then_some((b, c))
}

/// The second row of each 16-bit row in `packed` that packs two per dword,
/// by first row: the next 16-bit row of `elements` that no entry names
/// (`named`), the packed rows taking theirs in row order. Vertex and mesh
/// outputs ([`export::half_outputs`]) and pixel inputs
/// ([`input::half_input_partner`]) pair rows by this one rule, so that what
/// a vertex program packs into a dword is what its pixel program unpacks:
/// the vertex metadata names no second row.
fn packed_partners(
    elements: &[SigElement],
    named: impl Fn(u8) -> bool,
    packed: impl IntoIterator<Item = u8>,
) -> HashMap<u8, u8> {
    let mut free: Vec<u8> = elements
        .iter()
        .filter(|e| e.half && e.kind == 0)
        .filter_map(|e| u8::try_from(e.row?).ok())
        .filter(|&r| !named(r))
        .collect();
    free.sort_unstable();
    free.dedup();
    let mut rows: Vec<u8> = packed.into_iter().collect();
    rows.sort_unstable();
    let mut partners = HashMap::new();
    for row in rows {
        if let Some(k) = free.iter().position(|&r| r > row) {
            partners.insert(row, free.remove(k));
        }
    }
    partners
}

fn unsupported(offset: usize, what: Why) -> Stop {
    Stop::Fail(ShaderError::Unsupported { offset, what })
}

impl<'a> Lower<'a> {
    /// Identifiant du titre. La base XYZ partage le bloc de constantes du programme.
    fn group_id(&mut self, component: u8) -> V {
        let Val::I32(id) = self.compute_id("groupId.i32", 94, component) else {
            unreachable!()
        };
        let row = *self
            .group_base_row
            .get_or_insert_with(|| self.s.m.pending_i32());
        let handle = *self
            .user_data
            .get_or_insert_with(|| self.s.user_data_handle());
        let entry = self.s.entry;
        let values = self.s.op(
            entry,
            "cbufferLoadLegacy.i32",
            Attr::ReadOnly,
            self.s.t.cbuf_ret,
            59,
            &[handle, row],
        );
        let offset = self.s.m.extract(entry, values, u32::from(component));
        self.s.m.bin(entry, super::bitcode::Bin::Add, id, offset)
    }

    fn new(
        x: &'a Parts<'a>,
        bad: &'a HashMap<Value, Why>,
        numeric_selects: bool,
    ) -> Result<Self, Stop> {
        let p = x.p;
        let mut phi_at = HashMap::new();
        for (b, block) in p.blocks.iter().enumerate() {
            for (i, phi) in block.phis.iter().enumerate() {
                phi_at.insert(phi.value, (b, i));
            }
        }
        let mut s = Shader::new(x.native.stage == HwStage::Ps);
        s.ordered_uavs = x.native.io.pops();
        // Portage de VIctory #2085 (0xkn0wh3r3, 1f6fa41, GPL-3.0).
        // DEPTH_BEFORE_SHADER fait rejeter les pixels masqués avant leurs écritures UAV.
        if let StageIo::Ps {
            db_shader_control, ..
        } = x.native.io
            && db_shader_control & (1 << 12) != 0
        {
            s.flags |= super::shader::flags::FORCE_EARLY_DEPTH_STENCIL;
        }
        s.inputs = x.inputs.clone();
        s.outputs = x.outputs.clone();
        let insts = || p.blocks.iter().flat_map(|b| &b.insts);
        let writes_uavs = insts().any(|i| {
            matches!(i.fields, Fields::Image { .. } | Fields::Buffer { .. })
                && (i.name().contains("store") || i.name().contains("atomic"))
        });
        let mut console_lanes = false;
        let mut split = false;
        let mut group_reads = false;
        let mut wants_console_flow = false;
        let mut no_console_flow = None;
        let mut flat_group = None;
        if let StageIo::Cs(cs) = x.native.io {
            s.threads = Some(cs.threads);
            // Samples taking their level from their quad (implicit
            // derivatives, `Lower::image`): the console's quad is four
            // consecutive thread ids, its lanes following the thread order;
            // the host's, in a group whose X and Y are both even, is a 2 × 2
            // block of the thread id (SM 6.6 compute derivatives, which
            // quad reads follow too). One row whose size is a multiple of 4
            // has the console's quads (SM 6.6's 1D rule), so the group is
            // declared as that row and the console's ids come from the flat
            // one. Other sizes have no such row: the samples refuse them.
            let threads: u32 = cs.threads.iter().product();
            if threads.is_multiple_of(4)
                && cs.threads[1..] != [1, 1]
                && insts().any(|i| {
                    matches!(i.fields, Fields::Image { .. })
                        && image::implicit_derivatives(i.name())
                })
            {
                s.threads = Some([threads, 1, 1]);
                flat_group = Some(cs.threads);
            }
            s.lds_dwords = cs.lds_bytes() / 4;
            // Code reading other lanes needs the console's wave size and
            // lane order; so do lane masks read as numbers, and LDS shared
            // within one wave, where the console compiler leaves out barriers.
            // Swizzles and DPP read lanes of the same 32 (rows of 16, groups
            // of 8, 32 or 4): the rest may read the other half of a wave64.
            let in_half = |i: &Inst| {
                i.name().starts_with("ds_swizzle")
                    || i.name() == "ds_bpermute_b32"
                    || matches!(&i.fields, Fields::Valu(v) if v.dpp.is_some())
            };
            let whole = |i: &Inst| {
                let n = i.name();
                (n.contains("permute") && n != "ds_bpermute_b32")
                    || n.starts_with("v_mbcnt")
                    || matches!(n, "v_readlane_b32" | "v_writelane_b32")
                    || reads_mask_as_data(x.st, i)
            };
            let reads_whole = insts().any(whole);
            let masks_held = {
                // Masks held as numbers or constant lane sets: the lane
                // tests its own bit by its console lane.
                let whens = x.st.exits.iter().filter_map(|e| match *e {
                    LaneExit::Branch { when, .. } => Some(when),
                    _ => None,
                });
                // Numeric selects need fixed lane numbering only if the
                // first lowering attempt needs an otherwise untracked result.
                let selectors =
                    insts()
                        .filter(|_| numeric_selects)
                        .filter_map(|i| match &i.fields {
                            Fields::Valu(v) if i.name() == "v_cndmask_b32" => match v.src.get(2) {
                                Some(&Operand::Scalar { code, .. }) => i.read(Reg::S(code)),
                                _ => None,
                            },
                            _ => None,
                        });
                let selectors = selectors
                    .filter(|m| !x.st.masks.contains_key(m))
                    .filter_map(|m| x.st.others.get(&m).copied());
                let mut all: Vec<Bool> =
                    x.st.guards
                        .iter()
                        .flatten()
                        .flatten()
                        .copied()
                        .chain(whens)
                        .chain(selectors)
                        .collect();
                all.sort_unstable();
                all.dedup();
                all.into_iter().any(|g| {
                    x.st.bools
                        .support(g)
                        .iter()
                        .any(|a| matches!(a, Atom::Opaque(_) | Atom::Constant(_)))
                })
            };
            console_lanes = reads_whole || masks_held || insts().any(in_half);
            let wave = if x.native.wave64 { 64 } else { 32 };
            let one_wave = cs.threads.iter().product::<u32>() <= wave;
            // On a host without 64-lane waves, a wave64 program whose lanes
            // read only their own half runs as two 32-lane waves: what each
            // lane reads is in its host wave, at its host lane's place in the
            // row. Not with LDS (the console compiler may leave out barriers
            // between the halves) nor whole-wave sections (their reductions
            // become wave operations over the host's wave).
            // (`v_readfirstlane` reads the console wave's first lane, which
            // the upper half does not hold: such programs take the console
            // flow below.)
            split = console_lanes
                && x.native.wave64
                && !reads_whole
                && !insts().any(|i| i.name() == "v_readfirstlane_b32")
                && !masks_held
                && s.lds_dwords == 0
                && !x.target.runs(64)
                && x.target.runs(32)
                && (0..p.blocks.len()).all(|b| super::wave::sections(p, b).is_empty());
            // Une dernière wave incomplète peut échanger par LDS si chaque
            // thread publie sa valeur et si la lane source existe partout.
            group_reads = console_lanes
                && !split
                && x.native.wave64
                && !x.target.runs(64)
                && x.target.runs(32)
                && threads > 64
                && !threads.is_multiple_of(64)
                // Le dernier groupe physique de 32 publie aussi tous ses indices.
                && threads.is_multiple_of(32)
                && reads_whole
                && !masks_held
                && !insts().any(|i| i.name() == "v_readfirstlane_b32")
                && insts()
                    .filter(|&i| whole(i))
                    .all(|i| lanes::present_lane_read(i, threads))
                && (0..p.blocks.len()).all(|b| super::wave::sections(p, b).is_empty())
                && lanes::group_reads_meet(p);
            split |= group_reads;
            // The rest of such programs need the halves to exchange values
            // (masks read whole, lanes of the other half) or to share LDS in
            // the console's instruction order. A single console wave can
            // exchange through LDS; multiple complete waves use paired lanes.
            let needs_wave64 =
                (console_lanes || one_wave && s.lds_dwords > 0) && !split && x.native.wave64;
            if needs_wave64 && !x.target.runs(64) {
                no_console_flow = if threads > 64 && !threads.is_multiple_of(64) {
                    Some(
                        "console wave64 lanes in a group of more than 64 threads on a host without 64-lane waves",
                    )
                } else if insts()
                    .any(|i| i.name().contains("permute") && i.name() != "ds_bpermute_b32")
                {
                    Some("ds_permute across the halves of a wave64 on a host without 64-lane waves")
                } else if threads < 64
                    && !(0..p.blocks.len()).all(|b| super::wave::sections(p, b).is_empty())
                {
                    Some("a whole-wave section of a wave64 on a host without 64-lane waves")
                } else {
                    None
                };
                wants_console_flow = no_console_flow.is_none() && x.target.runs(32);
            }
            // On a host that runs the console's wave size: lane reads and
            // writes in a program whose lanes branch differently (see
            // [`CONSOLE_FLOW`]).
            let uniform = |when: Bool| {
                x.st.bools
                    .support(when)
                    .iter()
                    .all(|a| matches!(a, Atom::Uniform(_)))
            };
            // (The lane reads a whole-wave section makes are its own: the
            // section is lowered as wave operations in either form. One
            // outside a section is what needs the console flow.)
            let lane_access_outside_sections = p.blocks.iter().enumerate().any(|(b, block)| {
                let sections = super::wave::sections(p, b);
                block.insts.iter().enumerate().any(|(i, inst)| {
                    matches!(inst.name(), "v_readlane_b32" | "v_writelane_b32")
                        && !sections
                            .iter()
                            .any(|(&start, s)| (start..=s.end).contains(&i))
                })
            });
            wants_console_flow |= x.target.runs(wave)
                && lane_access_outside_sections
                && x.st
                    .exits
                    .iter()
                    .any(|e| matches!(*e, LaneExit::Branch { when, .. } if !uniform(when)))
                && !insts().any(|i| i.name().contains("permute") && i.name() != "ds_bpermute_b32");
            if x.console_cf {
                console_lanes = true;
                // In halves only where the host cannot run the console wave.
                split = x.native.wave64 && !x.target.runs(64);
            }
            if x.console_cf && split && threads > 64 {
                if !threads.is_multiple_of(64) {
                    return Err(unsupported(
                        0,
                        "paired lanes require complete console waves",
                    ));
                }
                // Each physical lane carries console lanes i and i+32. One
                // physical wave therefore follows one console program counter.
                group_reads = false;
                s.m.pair_lanes();
                flat_group = Some(cs.threads);
                s.threads = Some([threads / 2, 1, 1]);
            }
            if split {
                s.wave_size = Some(32);
            } else if console_lanes || one_wave && s.lds_dwords > 0 {
                s.wave_size = Some(wave);
            }
        }
        let mut no_barrier = HashSet::new();
        if let StageIo::Ms(ms) = &x.native.io {
            s.prims = x.prims.clone();
            s.threads = Some([ms.threads, 1, 1]);
            s.lds_dwords = ms.lds_bytes.div_ceil(4);
            s.mesh = Some(super::shader::Mesh {
                max_vertices: ms.max_vertices,
                max_primitives: ms.max_primitives,
                topology: ms.topology,
            });
            // Barriers order the lanes' LDS accesses; the console's
            // allocation request needs them too, but the host does not. One
            // lane sends that request, and what it dominates (the path for
            // counts out of range) runs in that lane alone, so its barrier
            // would hang the host.
            let order = crate::ir::reverse_postorder(&p.blocks);
            let mut index = vec![usize::MAX; p.blocks.len()];
            for (n, &b) in order.iter().enumerate() {
                index[b] = n;
            }
            let idom = crate::ir::idoms(&p.blocks, &order, &index);
            let alloc: Vec<usize> = (0..p.blocks.len())
                .filter(|&b| {
                    p.blocks[b].insts.iter().any(|i| {
                        i.name() == "s_sendmsg"
                            && matches!(i.fields, Fields::Sopp { imm } if imm & 0xF == GS_ALLOC_REQ)
                    })
                })
                .collect();
            for &b in &order {
                let mut d = b;
                let dominated = loop {
                    if alloc.contains(&d) {
                        break true;
                    }
                    if idom[d] == d || idom[d] == usize::MAX {
                        break false;
                    }
                    d = idom[d];
                };
                if ms.lds_bytes == 0 || dominated {
                    no_barrier.insert(b);
                }
            }
        }
        Ok(Lower {
            s,
            x,
            bad,
            lane_fail: HashMap::new(),
            vals: HashMap::new(),
            floats: HashMap::new(),
            block: vec![None; p.blocks.len()],
            cur: 0,
            bools: HashMap::new(),
            lane_bits: HashMap::new(),
            phi_at,
            mask_phis: HashMap::new(),
            pending_masks: Vec::new(),
            data_phis: Vec::new(),
            read_phis: HashSet::new(),
            emitted_double_count: false,
            params: x.res.params.iter().map(|&(v, p, d)| (v, (p, d))).collect(),
            user_data: None,
            group_base_row: None,
            host_words: Vec::new(),
            host_rows: Vec::new(),
            partial_active: None,
            user_reads: BTreeMap::new(),
            samplers: Vec::new(),
            descriptors: x
                .res
                .descriptors
                .iter()
                .map(|d| ((d.site, d.role == Role::Sampler), &d.source))
                .collect(),
            scalar: x.res.scalar.iter().copied().collect(),
            exports: x.io.exports.iter().map(|e| (e.site, e)).collect(),
            interps: x.io.interps.iter().map(|(s, i)| (*s, i)).collect(),
            handles: HashMap::new(),
            entries: HashMap::new(),
            run_time_buffers: 0,
            #[cfg(test)]
            integer_loads: 0,
            edges: HashMap::new(),
            exported: vec![false; p.blocks.len()],
            rpo: vec![usize::MAX; p.blocks.len()],
            tail: vec![None; p.blocks.len()],
            writes_uavs,
            src0: None,
            console_lanes,
            split,
            console_cf: x.console_cf,
            wants_console_flow,
            no_console_flow,
            console_flat: None,
            flat_group,
            xchg: None,
            group_reads,
            lane_slots: None,
            flat: None,
            mesh_primitive: None,
            no_barrier,
            uniform: None,
            half_outputs: half_outputs(&x.native.io, &x.outputs),
            mask_data: match x.native.stage {
                HwStage::Cs => MaskData::Console,
                HwStage::Ps | HwStage::Vs => MaskData::Host,
                _ => MaskData::None,
            },
            helper_bits: if x.native.stage == HwStage::Ps {
                mask::HelperBits::of(p)
            } else {
                mask::HelperBits::default()
            },
            mask_ok: false,
            ballots: HashMap::new(),
            opaque_open: HashSet::new(),
            balloting: HashSet::new(),
            def_block: p
                .blocks
                .iter()
                .enumerate()
                .flat_map(|(b, block)| {
                    block
                        .insts
                        .iter()
                        .flat_map(move |i| i.writes.iter().map(move |w| (w.1, b)))
                        .chain(block.phis.iter().map(move |phi| (phi.value, b)))
                })
                .collect(),
            cur_ir: 0,
            sections: HashMap::new(),
            section_out: HashSet::new(),
            code_table: None,
            counters: HashMap::new(),
            abort: None,
            spin: HashSet::new(),
            counted: HashSet::new(),
            drain: flow::Drain::default(),
            budget: flow::Budget::default(),
            #[cfg(test)]
            limits: Vec::new(),
            #[cfg(test)]
            scalar_guards: 0,
            data_reads: {
                let access: HashSet<Site> = x.res.descriptors.iter().map(|d| d.site).collect();
                let mut reads = HashSet::new();
                for (b, block) in p.blocks.iter().enumerate() {
                    for phi in &block.phis {
                        reads.extend(phi.args.iter().copied());
                    }
                    for (i, inst) in block.insts.iter().enumerate() {
                        if !access.contains(&(b, i)) {
                            reads.extend(inst.reads.iter().map(|r| r.1));
                        }
                    }
                }
                reads
            },
            merged: None,
        })
    }

    fn run(mut self) -> Result<Lowered, Stop> {
        if self.wants_console_flow && !self.x.console_cf && !self.x.per_lane_only {
            return Err(unsupported(0, CONSOLE_FLOW));
        }
        if let Some(n) = self.s.wave_size
            && !self.x.target.runs(n)
        {
            // shortcut: no form of these lanes on the host's wave size yet
            // (two 32-lane waves standing for one console wave64 need the
            // halves to exchange masks and lane values).
            return Err(unsupported(
                0,
                match (n, self.console_lanes) {
                    (64, _) if self.no_console_flow.is_some() => {
                        self.no_console_flow.expect("checked")
                    }
                    (64, true) => "console wave64 lanes on a host without 64-lane waves",
                    (64, false) => {
                        "a wave64 group sharing LDS without barriers on a host without 64-lane waves"
                    }
                    _ => "a console wave size the host does not run",
                },
            ));
        }
        self.exchange_fits().map_err(|w| unsupported(0, w))?;
        let p = self.x.p;
        let order = crate::ir::reverse_postorder(&p.blocks);
        for (n, &b) in order.iter().enumerate() {
            self.rpo[b] = n;
        }
        for (n, &b) in order.iter().enumerate() {
            self.block[b] = Some(if n == 0 {
                self.s.entry
            } else {
                self.s.m.new_block()
            });
        }
        // Loop headers count iterations (`LOOP_LIMIT`; spin waits less,
        // counted loops more).
        self.spin = spin::spin_loops(p);
        self.counted = flow::counted_loops(p, self.x.st);
        for &b in &order {
            let header = p.blocks[b]
                .preds
                .iter()
                .any(|&q| self.block[q].is_some() && self.rpo[b] <= self.rpo[q]);
            if header {
                let blk = self.block[b].expect("reachable");
                let phi = self.s.m.phi(blk, self.s.t.i32);
                self.counters.insert(b, (phi, Vec::new()));
            }
        }
        self.cur = self.s.entry;
        // All 64 logical lanes are present in a paired wave. Execute the
        // original steps, including FP16 operations and their rounding.
        if !self.s.m.paired_lanes() {
            self.find_sections();
        }
        if !self.x.res.spill_loads.is_empty() {
            self.dispatch_threads();
        }
        self.inputs()?;
        self.merged_start()?;
        if let Some(mesh) = self.s.mesh {
            self.mesh_prologue(mesh).map_err(|w| unsupported(0, w))?;
            self.block[order[0]] = Some(self.cur);
        }
        for &b in &order {
            self.cur = self.block[b].expect("reachable");
            self.cur_ir = b;
            let preds: Vec<usize> = p.blocks[b]
                .preds
                .iter()
                .copied()
                .filter(|&q| self.block[q].is_some())
                .collect();
            self.exported[b] = !preds.is_empty() && preds.iter().all(|&q| self.exported_out(q));
            for (i, phi) in p.blocks[b].phis.iter().enumerate() {
                let val = match phi.reg {
                    Reg::S(c) if c == EXEC || c == EXEC + 1 => Val::Missing("a lane mask"),
                    _ if self.bad.contains_key(&phi.value) => Val::Missing(self.bad[&phi.value]),
                    Reg::Scc => {
                        let v = self.s.m.phi(self.cur, self.s.t.i1);
                        self.data_phis.push((v, b, i));
                        Val::Bit(v)
                    }
                    _ => {
                        let v = self.s.m.phi(self.cur, self.s.t.i32);
                        self.data_phis.push((v, b, i));
                        Val::I32(v)
                    }
                };
                self.vals.insert(phi.value, val);
            }
            let mut i = 0;
            while i < p.blocks[b].insts.len() {
                self.inst(b, i)?;
                if let Some(end) = self.sections.get(&b).and_then(|s| s.get(&i)).map(|s| s.end) {
                    for j in i + 1..end {
                        self.section_inst(b, j, i);
                    }
                    i = end;
                    continue;
                }
                i += 1;
            }
            self.mask_ok = self.mask_data == MaskData::Console;
            self.tail[b] = Some(self.cur);
            if self.merged_end(b) {
                continue;
            }
            self.exit(b)?;
        }
        self.fill_phis()?;
        self.s.denormals = fp32_denorm_mode(self.x.native, self.x.p);
        if let Some(slot) = self.mesh_primitive {
            self.finish_mesh_indices(slot);
        }
        #[cfg(debug_assertions)]
        self.s.m.assert_paired_branches();
        let mut user_data: Vec<_> = self.user_reads.values().copied().collect();
        let first = self.s.user_data.next_multiple_of(4);
        for (i, row) in self.host_rows.iter().enumerate() {
            let actual = self.s.m.i32(first / 4 + i as u32);
            self.s.m.replace_const(*row, actual);
        }
        for (mut binding, _) in self.host_words {
            binding.index += first;
            self.s.user_data = binding.index + 1;
            user_data.push(binding);
        }
        let group_base = self.group_base_row.map(|row| {
            // La rangée suivante reste libre même si des constantes du titre sont déportées.
            let offset = self.s.user_data.next_multiple_of(4);
            let actual = self.s.m.i32(offset / 4);
            self.s.m.replace_const(row, actual);
            self.s.user_data = offset + 3;
            offset
        });
        self.s.samplers = self.samplers.len() as u32;
        let root_cbvs = self.s.root_cbvs.iter().map(|r| r.0).collect();
        let root_srvs = self.s.root_srvs.iter().map(|r| r.0).collect();
        let root_uavs = self.s.root_uavs.iter().map(|r| r.0).collect();
        let stride_table = self.s.stride_table.is_some();
        let cmp_samplers = self.s.cmp_samplers;
        Ok(Lowered {
            group_base,
            flow: match (self.console_cf, self.split) {
                (false, false) => "per lane",
                (false, true) if self.group_reads => {
                    "per lane in halves, lane reads through group-shared memory"
                }
                (false, true) => "per lane in halves",
                (true, false) => "console flow",
                (true, true) if self.s.m.paired_lanes() => "console flow with paired lanes",
                (true, true) => "console flow in halves",
            },
            cmp_samplers,
            root_cbvs,
            root_srvs,
            root_uavs,
            stride_table,
            run_time_buffers: self.run_time_buffers,
            #[cfg(test)]
            nonuniform_handles: self.s.nonuniform_handles,
            #[cfg(test)]
            integer_loads: self.integer_loads,
            #[cfg(test)]
            wave_ops: self.s.flags & super::shader::flags::WAVE_OPS != 0,
            #[cfg(test)]
            spin_loops: self.spin.len(),
            #[cfg(test)]
            loop_limits: std::mem::take(&mut self.limits),
            #[cfg(test)]
            scalar_guards: self.scalar_guards,
            dxil: self.s.finish(),
            user_data,
            samplers: self.samplers,
        })
    }

    fn inst(&mut self, b: usize, i: usize) -> Result<(), Stop> {
        if let Some(r) = self.merged_inst(b, i) {
            return r;
        }
        if let Some(r) = self.ray_query(b, i) {
            return r;
        }
        let inst: &'a Inst = &self.x.p.blocks[b].insts[i];
        let offset = inst.machine.offset;
        let name = inst.name();
        self.mask_ok = match self.mask_data {
            MaskData::Console => true,
            MaskData::Host => {
                name.starts_with("s_bcnt1_i32_")
                    || name.starts_with("s_ff1_i32_")
                    || name.starts_with("v_mbcnt_")
                    || matches!(
                        name,
                        "s_cmp_eq_u32" | "s_cmp_lg_u32" | "s_cmp_eq_u64" | "s_cmp_lg_u64"
                    )
            }
            MaskData::None => false,
        };
        let effect = matches!(inst.fields, Fields::Exp { .. })
            || matches!(inst.fields, Fields::Flat { .. }) && !name.starts_with("global_load_")
            || name.contains("store")
            || name.contains("atomic")
            || name == "s_barrier"
            || matches!(inst.fields, Fields::Ds { .. })
                && !(name.starts_with("ds_read")
                    || name.starts_with("ds_swizzle")
                    || name.contains("permute"));
        let r = match &inst.fields {
            _ if inst.write(Reg::FloatMode).is_some() => self.set_float_mode(inst),
            Fields::Sop { .. } => self.sop(inst),
            Fields::Sopp { .. } if name == "s_barrier" && self.no_barrier.contains(&b) => Ok(()),
            Fields::Sopp { .. } if name == "s_barrier" => {
                // Thread-group sync with a shared-memory fence, and a UAV
                // fence when the program writes UAVs (the waits before an
                // s_barrier make those writes visible on the console).
                let mode = if self.writes_uavs { 1 | 2 | 8 } else { 1 | 8 };
                let mode = self.s.m.i32(mode);
                let void = self.s.t.void;
                self.s
                    .op(self.cur, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
                Ok(())
            }
            Fields::Sopp { .. } => Ok(()),
            Fields::Smem { .. } => self.smem(inst, (b, i)),
            Fields::Valu(_) if name.starts_with("v_interp_") => self.interp(inst, (b, i)),
            Fields::Valu(_) if self.group_reads && name == "v_readlane_b32" => {
                self.group_lane_read(inst)
            }
            Fields::Valu(_) => self.valu(inst, b, i),
            Fields::Vintrp { .. } => self.interp(inst, (b, i)),
            Fields::Image { .. } => self.image(inst, b, i),
            Fields::Buffer { .. } => self.buffer(inst, b, i),
            Fields::Exp { .. } => self.export(b, i),
            Fields::Ds { .. } => self.ds(inst, b, i),
            Fields::Flat { .. } if name.starts_with("global_") => self.flat(inst, b, i),
            Fields::Flat { .. } => Err("flat memory"),
        };
        match r {
            Ok(()) => Ok(()),
            Err(why) if effect => Err(unsupported(offset, why)),
            Err(why) => {
                for &(_, v) in &inst.writes {
                    self.vals.insert(v, Val::Missing(why));
                }
                Ok(())
            }
        }?;
        if !self.bad.is_empty() {
            for &(_, w) in &inst.writes {
                if self.bad.get(&w) == Some(&mask::LANE_MASK)
                    && let Ok(word) = self.mask_dword(w)
                {
                    self.vals.insert(w, Val::I32(word));
                }
            }
        }
        Ok(())
    }
}
