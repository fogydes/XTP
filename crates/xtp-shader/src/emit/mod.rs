//! DXIL emission (decision 0003, step 3): a console vertex, pixel, compute
//! or mesh shader as a PC DXIL container (SM 6.6) that host D3D12 accepts.
//! Compute shaders keep the console's thread-group size; one that depends
//! on the lane layout (cross-lane operations) declares the console's wave
//! size (`WaveSize`), so the host device must support it: [`Target`] says
//! which wave sizes the host runs, and a shader that needs another one does
//! not translate for it (rather than being refused at pipeline creation).
//!
//! What the host binds for a translated shader ([`HostShader`]):
//!
//! * **User data** at `b0, space 0` (32-bit root constants or a constant
//!   buffer, per stage): dword `index` holds a root parameter's value, as
//!   [`UserData`] lists. A root-constant dword is the constant. A
//!   descriptor table is the byte offset of its first descriptor in its
//!   heap: the descriptor index times 32 for CBV/SRV/UAV tables, times 16
//!   for sampler tables (the console's descriptor sizes; the code adds
//!   offsets in those units). The heap bases of dynamic resources are 0
//!   and not listed.
//! * **Descriptors** through SM 6.6 `ResourceDescriptorHeap[]` and
//!   `SamplerDescriptorHeap[]`, at the index the console address gives, so
//!   the host heaps keep the title's descriptor order. The root signature
//!   needs `CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED` and
//!   `SAMPLER_HEAP_DIRECTLY_INDEXED`.
//! * **Immediate samplers** as static samplers `s<n>, space 0`, `n` the
//!   position in [`HostShader::samplers`] (the console sampler words) and
//!   [`HostShader::static_samplers`] (the same decoded into a D3D12 static
//!   sampler description). Comparison sampling (added 2026-11-23): an
//!   immediate sampler it uses has a comparison filter there (bit 0x80);
//!   a heap sampler it uses must be a comparison sampler on the host.
//! * **Root CBVs and root SRVs** (raw buffers) as root descriptors
//!   `b<i>` / `t<i>` in [`ROOT_CBV_SPACE`], `i` the title's root parameter
//!   index ([`HostShader::root_cbvs`], [`HostShader::root_srvs`]); the
//!   address the title sets is the buffer's.
//! * **Vertex fetch** (added 2026-11-23). BTP 6's mesh and
//!   vertex programs fetch vertices themselves with typed-format loads
//!   (`tbuffer_load_format_*`): the format (8_8_8_8 unorm, 16_16 snorm,
//!   2_10_10_10 unorm, 8_8_8_8 and 16_16_16_16 uint...) is in the
//!   instruction, so the translation reads raw bytes and converts them
//!   itself. Their descriptors are built around a root SRV (stride 1, no
//!   record limit): the host binds the root SRV `t<slot>` above with the
//!   address the title sets, nothing more. The vertex index, the stride
//!   and the attribute offset are the shader's own arithmetic on data in
//!   that buffer (per-stream headers), not descriptor fields.
//! * **Buffer views in the heap** are declared by how the shader reads
//!   them: raw (ByteAddressBuffer) for dword loads and stores, atomics and
//!   typed-format loads without an index; structured with the used-resource
//!   entry's stride when indexed; typed with four 32-bit float channels for
//!   `buffer_load_format_*` and `buffer_store_format_*`. The host's view at
//!   that index must be of that kind and start at the console descriptor's
//!   base address (raw: `R32_TYPELESS` with the raw flag), whatever kind the
//!   title's view description says. Typed-format loads from heap
//!   descriptors (1,024 world compute programs, ray-tracing material
//!   resolves) read raw views this way.
//!
//! * **Loop-limit counter** (with [`loop_count`]): a root UAV `u0` in
//!   [`LOOP_COUNTER_SPACE`], a raw buffer whose first dword a lane adds 1 to
//!   where it leaves a loop at XTP's limit or budget
//!   ([`HostShader::loop_counter`] says the shader has one). The host points
//!   it at the pipeline's own dword (`xtp-gpu`'s `loopstat`).
//!
//! * **Texture atomics** (added 2026-11-23): a texture UAV the shader uses
//!   atomically is declared a 32-bit unsigned single-channel view; the
//!   host's view at that index must be an R32 integer view (64-bit atomics
//!   stop the translation). Global loads from a root SRV read it raw.
//!
//! Entry points: [`translate`] (container bytes as the title passes them
//! at pipeline creation → [`HostShader`]), [`emit`] (the same with the
//! earlier stages already run). The root constants of each stage must
//! cover [`HostShader::user_data_dwords`]. Stability: these types and the
//! binding scheme are what pipeline creation builds on; changes are
//! announced.
//! * **Signatures** as the console container has them (same semantics,
//!   registers and masks), so the input layout and stage linking are the
//!   title's; pixel outputs are plain 32-bit values.
//! * **Mesh shaders** ([`HwStage::Ms`], `ms_6_6`): user data and
//!   descriptors as above (the root constants visible to the mesh stage);
//!   the pipeline is a mesh pipeline (pipeline-state stream, no input
//!   layout) with the title's pixel shader. The thread group is the
//!   console's in lanes (`[numthreads]` in the DXIL), the vertex and
//!   primitive maxima its own, triangles only, no wave size required. The
//!   console numbers groups in x only: the draw path passes the title's
//!   count as `DispatchMesh(x, 1, 1)`. Per-primitive outputs (`PSG1`) are
//!   the console's, without `SV_PrimitiveID` and `SV_CullPrimitive` (the
//!   console never exports them).

mod bitcode;
#[cfg(all(test, windows))]
mod buffer16_tests;
mod container;
#[cfg(all(test, windows))]
mod desc_data_tests;
#[cfg(all(test, windows))]
mod diff_tests;
#[cfg(all(test, windows))]
mod geometry_tests;
#[cfg(all(test, windows))]
mod group_id_tests;
#[cfg(all(test, windows))]
mod helper_tests;
#[cfg(all(windows, any(test, feature = "host")))]
pub mod host;
#[cfg(all(test, windows))]
mod image_tests;
#[cfg(all(test, windows))]
mod input_tests;
#[cfg(all(test, windows))]
mod lane_read_loop_tests;
#[cfg(all(test, windows))]
mod lightloop_tests;
#[cfg(all(test, windows))]
mod limits_tests;
#[cfg(all(test, windows))]
mod literal64_tests;
#[cfg(all(test, windows))]
mod loop_limit_tests;
#[cfg(all(test, windows))]
mod mask_tests;
#[cfg(all(test, windows))]
mod mesh_tests;
#[cfg(all(test, windows))]
mod meshidx_tests;
// IR to DXIL, one file per encoding or concern: which backlog item of
// docs/analysis/shader.md edits which file is at the top of `lower/mod.rs`.
pub(crate) mod lower;
#[cfg(all(test, windows))]
mod norm_tests;
#[cfg(all(test, windows))]
mod r1_tests;
#[cfg(all(test, windows))]
mod r4_tests;
#[cfg(all(test, windows))]
mod raw_tests;
#[cfg(all(test, windows))]
mod rayquery_tests;
pub mod rect_gs;
mod sampler;
#[cfg(test)]
mod sampler_tests;
#[cfg(all(test, windows))]
mod scalar_gap_tests;
mod shader;
#[cfg(all(test, windows))]
mod smem_data_tests;
#[cfg(all(test, windows))]
mod smem_tests;
#[cfg(all(test, windows))]
mod split_tests;
#[cfg(all(test, windows))]
mod stride_tests;
#[cfg(test)]
mod userdata_tests;
#[cfg(windows)]
pub mod validator;
#[cfg(all(test, windows))]
mod validator_tests;
#[cfg(all(test, windows))]
mod valu_tests;
#[cfg(all(test, windows))]
mod waterfall_tests;
mod wave;

use crate::dxbc::{Container, FourCC};
use crate::native::{HwStage, NativeProgram, RootType};
use crate::{ShaderError, io, ir, resource, structure};

pub use bitcode::opt::dxil_opt;
pub(crate) use lower::geometry::GsShape;
/// The root SRV register of an inline ray query's acceleration structure.
pub use lower::rayquery::TLAS_REGISTER;
/// Where a spliced program's dispatch finds the title's descriptor of its
/// structure.
pub use lower::rayquery::{TlasDescriptor, container_tlas_descriptor, tlas_descriptor};
pub use sampler::StaticSampler;
pub use shader::LOOP_COUNTER_SPACE;
pub use shader::{
    ROOT_CBV_SPACE, RawWindow, SAMPLER_SPACE, STRIDE_SPACE, StrideEntry, USER_DATA_SPACE,
};

/// The loop limit without an override: 2^14 (16,384), under the host's
/// GPU watchdog (see `lower::LOOP_LIMIT`).
pub const DEFAULT_LOOP_LIMIT: u32 = lower::LOOP_LIMIT;

/// The iteration count after which a translated loop is left, per lane and
/// per entry into the loop: [`DEFAULT_LOOP_LIMIT`].
///
/// `XTP_SHADER_LOOP_LIMIT` (decimal or `0x` hex, at least 1) sets it
/// instead, in every build: a release build can then show which loops make
/// a frame slow without a diagnostics build. A loop that hits the limit in
/// every lane of a full-screen pass still costs (2^14 times a 4K screen is
/// about 10^11 iterations); a lower value shows whether such loops are what
/// makes a frame slow. The default cuts short a legitimate loop of more
/// than 16,384 iterations per lane that is no counted loop (counted loops
/// have [`counted_loop_limit`]), which then gives a wrong result; a higher
/// value (the old default was `0x100000`) shows whether a wrong result
/// comes from that, at the risk of the watchdog removing the device. The
/// switch sets the counted loops' limit too. The value is part of the
/// translation, so the shader cache must key on it (`xtp-gpu`'s does).
pub fn loop_limit() -> u32 {
    limit_override().unwrap_or(lower::LOOP_LIMIT)
}

/// The counted loops' limit without an override: 2^20 (see
/// `lower::COUNTED_LOOP_LIMIT`).
pub const DEFAULT_COUNTED_LOOP_LIMIT: u32 = lower::COUNTED_LOOP_LIMIT;

/// The iteration count after which a counted loop (an induction variable
/// compared with a bound the loop does not change, see `lower/flow.rs`) is
/// left: [`DEFAULT_COUNTED_LOOP_LIMIT`], or `XTP_SHADER_LOOP_LIMIT` when
/// set, like [`loop_limit`] (so the shader cache's key on that covers it).
pub(crate) fn counted_loop_limit() -> u32 {
    limit_override().unwrap_or(DEFAULT_COUNTED_LOOP_LIMIT)
}

/// `XTP_SHADER_LOOP_LIMIT`'s value, read once.
fn limit_override() -> Option<u32> {
    static LIMIT: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        parse_limit_override(std::env::var("XTP_SHADER_LOOP_LIMIT").ok().as_deref())
    })
}

/// The spin-wait limit under the default loop limit: 4096.
pub const DEFAULT_SPIN_LIMIT: u32 = 1 << 12;

/// The iteration count after which a loop that only waits for memory to
/// change (a spin wait, see `lower/spin.rs`) is left: [`DEFAULT_SPIN_LIMIT`],
/// or [`loop_limit`] when that is lower. On the console the write such a
/// loop waits for comes within microseconds; on the host, when it never
/// comes (the writer's view is another host resource, null, or the load was
/// hoisted), 2^20 uncached reads per lane (the old general limit) cost
/// about a second per wave; 4096 reads are a few milliseconds. It follows
/// the loop limit and has no switch of its own, so the shader cache's key
/// on [`loop_limit`] covers it.
pub(crate) fn spin_limit() -> u32 {
    DEFAULT_SPIN_LIMIT.min(loop_limit())
}

/// The per-lane budget without an override: 2^20 (1,048,576) back edges.
pub const DEFAULT_LOOP_BUDGET: u32 = 1 << 20;

/// The back edges a lane may take in all of a program's loops together
/// before it leaves the program as the loop limit leaves it (0: no budget):
/// [`DEFAULT_LOOP_BUDGET`]. The limits count per loop and per entry into
/// it, so loops inside loops multiply theirs (2^14 inside 2^14 is 2^28
/// iterations, seconds of one wave: past the host's GPU watchdog); the
/// budget bounds their product. It is the counted loops' limit, so no
/// single loop is cut shorter than by its own limit.
///
/// `XTP_SHADER_LOOP_BUDGET` (decimal or `0x` hex; 0 turns it off) sets it
/// instead, in every build. The value is part of the translation, so the
/// shader cache must key on it (`xtp-gpu`'s does).
// shortcut: an outer loop of n passes over an inner loop of m is cut once
// n·m passes the budget, even where the console runs it to its end.
pub fn loop_budget() -> u32 {
    static BUDGET: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *BUDGET
        .get_or_init(|| parse_loop_budget(std::env::var("XTP_SHADER_LOOP_BUDGET").ok().as_deref()))
}

/// The budget `XTP_SHADER_LOOP_BUDGET`'s value gives: the default when it
/// is unset or unreadable. Pure, for testing.
fn parse_loop_budget(value: Option<&str>) -> u32 {
    value
        .map(str::trim)
        .and_then(|v| match v.strip_prefix("0x") {
            Some(hex) => u32::from_str_radix(hex, 16).ok(),
            None => v.parse().ok(),
        })
        .unwrap_or(DEFAULT_LOOP_BUDGET)
}

/// Whether a lane that leaves a loop at XTP's limit (the loop limit, the
/// spin-wait limit, a counted loop's) or at the budget adds itself to its
/// pipeline's loop-limit counter ([`LOOP_COUNTER_SPACE`]) before it leaves:
/// `XTP_SHADER_LOOP_COUNT` set to anything but `0` or nothing, in every
/// build, read once (or [`with_loop_count`] on this thread). `xtp-gpu`
/// binds each counting pipeline's dword and reads the counts per frame and
/// per pipeline into the frame line (`loopstat`). Off (the default), a
/// translation is byte for byte what it is without the counter. On, a
/// program with a limited loop declares the counter, and each stopped lane
/// costs an atomic. The switch is part of the translation, so the shader
/// cache must key on it (`xtp-gpu`'s does).
pub fn loop_count() -> bool {
    if LOOP_COUNT_HERE.get() {
        return true;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| parse_loop_count(std::env::var_os("XTP_SHADER_LOOP_COUNT").as_deref()))
}

/// Whether `XTP_SHADER_LOOP_COUNT`'s value turns [`loop_count`] on: set,
/// and neither empty nor `0`. Pure, for testing.
fn parse_loop_count(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_some_and(|v| !v.is_empty() && v != "0")
}

thread_local! {
    /// [`with_loop_count`] is running on this thread.
    static LOOP_COUNT_HERE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `f` with [`loop_count`] on for the translations this thread makes
/// in it, whatever `XTP_SHADER_LOOP_COUNT` says (tests; a shader cache
/// keyed on the switch alone must not keep what it translates).
pub fn with_loop_count<T>(f: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            LOOP_COUNT_HERE.set(self.0);
        }
    }
    let _restore = Restore(LOOP_COUNT_HERE.replace(true));
    f()
}

/// Every setting a translation depends on besides the container and the
/// [`Target`], each in the value the translator uses (not its switch's
/// spelling: `XTP_SHADER_LOOP_LIMIT` unset and set to 16384 give the same
/// loop limit but not the same counted loops' limit). A shader cache keys
/// on all of it (`xtp-gpu`'s does, field by field). A switch added to the
/// translator must be added here, and so to the key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Settings {
    /// [`loop_limit`].
    pub loop_limit: u32,
    /// The counted loops' limit (`counted_loop_limit`).
    pub counted_loop_limit: u32,
    /// The spin waits' limit (`spin_limit`).
    pub spin_limit: u32,
    /// [`loop_budget`].
    pub loop_budget: u32,
    /// [`dxil_opt`]'s passes.
    pub dxil_opt: u32,
    /// [`raw_structured`].
    pub raw_structured: bool,
    /// [`loop_count`], on this thread.
    pub loop_count: bool,
}

/// The [`Settings`] a translation made now on this thread runs with.
pub fn settings() -> Settings {
    Settings {
        loop_limit: loop_limit(),
        counted_loop_limit: counted_loop_limit(),
        spin_limit: spin_limit(),
        loop_budget: loop_budget(),
        dxil_opt: dxil_opt(),
        raw_structured: raw_structured(),
        loop_count: loop_count(),
    }
}

/// `XTP_SHADER_LOOP_LIMIT`'s value; none when it is unset, unreadable or
/// 0. Pure, for testing.
fn parse_limit_override(value: Option<&str>) -> Option<u32> {
    value
        .map(str::trim)
        .and_then(|v| match v.strip_prefix("0x") {
            Some(hex) => u32::from_str_radix(hex, 16).ok(),
            None => v.parse().ok(),
        })
        .filter(|&n| n > 0)
}

/// The loop limit `XTP_SHADER_LOOP_LIMIT`'s value gives (see
/// [`parse_limit_override`]).
#[cfg(test)]
fn parse_loop_limit(value: Option<&str>) -> u32 {
    parse_limit_override(value).unwrap_or(lower::LOOP_LIMIT)
}

/// A translated shader and what the host binds for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostShader {
    pub stage: HwStage,
    /// The DXIL container, hashed as D3D12 expects.
    pub dxil: Vec<u8>,
    /// The user-data dwords the shader reads.
    pub user_data: Vec<UserData>,
    /// Premier des trois dwords de la base de groupe, si le programme lit SV_GroupID.
    pub group_base: Option<u32>,
    /// Immediate samplers: the four sampler words of `s<n>`.
    pub samplers: Vec<[u32; 4]>,
    /// The same samplers decoded, `s<n>` at position n: the static sampler
    /// descriptions the host's root signature declares.
    pub static_samplers: Vec<StaticSampler>,
    /// Root CBVs the shader reads: root parameter indices, each bound as
    /// `b<index>` in [`ROOT_CBV_SPACE`].
    pub root_cbvs: Vec<u8>,
    /// Root SRVs (raw buffers) the shader reads: `t<index>` in
    /// [`ROOT_CBV_SPACE`].
    pub root_srvs: Vec<u8>,
    /// UAV racine lus ou écrits par le programme, au registre u du même numéro.
    pub root_uavs: Vec<u8>,
    /// Whether the shader reads the host's runtime stride table (`t0` in
    /// [`STRIDE_SPACE`], a [`StrideEntry`] per view-heap slot): only when
    /// translated for a [`Target`] that has one.
    pub stride_table: bool,
    /// Whether the shader reads ([`RUN_TIME_READS`]) or may write
    /// ([`RUN_TIME_WRITES`]) buffers through view-heap slots it picks when it
    /// runs: a descriptor loaded at an address with a run-time part, or per
    /// lane. No used-resource entry names such a slot, so the host makes and
    /// refreshes its views another way (`xtp-gpu`'s `XTP_RUN_TIME_VIEWS`).
    pub run_time_buffers: u8,
}

/// Whether buffers read by element with a stride known at translation are
/// read through raw handles (`index * stride + offset`) instead of
/// structured ones: on by default since 2026-11-25 (every image session ran
/// with it; without it the world draws black), `XTP_RAW_STRUCTURED=0` turns
/// it off, read once. `xtp-gpu` then makes
/// the title's structured views raw views of the same bytes where it can
/// (Marvisak/VIctory#1762, after fxceless's VIctory fork's `raw_window`):
/// on NVIDIA a raw read through a structured view returns other dwords or
/// zeros (the lighting's pre-exposure, byte 4 of a 16-byte structured view,
/// read 0: a black world; watch.md items 148, 160, 190), and a structured
/// handle over a raw view reads zeros too, so both sides change together.
/// With the runtime stride table such reads go through its raw views. Part
/// of the translation: the shader cache keys on it.
pub fn raw_structured() -> bool {
    #[cfg(test)]
    if let Some(on) = RAW_STRUCTURED_TEST.get() {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("XTP_RAW_STRUCTURED").is_ok_and(|v| v == "0"))
}

#[cfg(test)]
thread_local! {
    /// [`raw_structured`] for this thread's tests.
    pub(crate) static RAW_STRUCTURED_TEST: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

/// [`HostShader::run_time_buffers`]: buffers read through slots picked at
/// run time.
pub const RUN_TIME_READS: u8 = 1;
/// [`HostShader::run_time_buffers`]: unordered access views through slots
/// picked at run time (they may be written).
pub const RUN_TIME_WRITES: u8 = 2;

/// Octets entre le début aligné de la vue brute et l'adresse de l'UAV racine.
pub const ROOT_UAV_BIAS: RootType = RootType::Other(0x103);
/// Limite de chaque axe d'un DispatchX exprimé en threads, zéro pour un axe entier.
pub const DISPATCH_THREADS: RootType = RootType::Other(0x104);
/// Indices de SRV calculés à partir des constantes racine du dispatch.
pub const PACKED_ADDRESS_SRV: RootType = RootType::Other(0x105);
pub const SCALAR_ADDRESS_SRV: RootType = RootType::Other(0x106);
pub const VECTOR_ADDRESS_SRV: RootType = RootType::Other(0x107);
/// Slot d'une SRV brute alignée, suivi en bits 0..3 du décalage de la base du titre.
pub const INDEXED_ADDRESS_SRV: RootType = RootType::Other(0x108);
/// Dynamic raw SRV heap slot `(slot << 4) | bias` for a self-sized table SRV
/// (`Options::self_sized_loads_via_memory`), breaking self-sized
/// `getDimensions` -> `rawBufferLoad` cycles on the same descriptor handle.
pub const SELF_SIZED_RAW_SRV: RootType = RootType::Other(0x109);
/// Byte length of the view bound for [`SELF_SIZED_RAW_SRV`], avoiding
/// `getDimensions(h)` element-count vs byte-offset mismatches on structured SRVs.
pub const SELF_SIZED_VIEW_BYTES: RootType = RootType::Other(0x10a);

/// Choices a translation can be made with, per shader: the host asks for
/// another form where the default one fails on its driver. The default
/// (`Options::default()`) is what [`translate_for`] makes.
// Ported and adapted from PastiiJay/klbf-lab commit f73233c8 (author aidzz / @BrokeAsh, GPL-3.0-or-later).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Options {
    /// A scalar load from a buffer at an offset computed from that same
    /// buffer's record count (the console's "last record" idiom,
    /// `buf[(size / stride - 1) * stride]`) reads through a separate raw
    /// view supplied via [`SELF_SIZED_RAW_SRV`] instead of reusing the
    /// handle whose dimensions computed the offset; past the view's end it
    /// reads 0, as the console's load does. The RX 9070 XT's compiler
    /// (`amdxc64.dll` 32.0.32029.2010) crashes the process on some programs
    /// with the load through the same handle (null read at
    /// `amdxc64.dll+0x861b9f`), and compiles them when the load goes through
    /// a separate raw handle.
    pub self_sized_loads_via_memory: bool,
}

thread_local! {
    static OPTIONS: std::cell::Cell<Options> = const {
        std::cell::Cell::new(Options { self_sized_loads_via_memory: false })
    };
}

/// The options of the translation running on this thread.
pub(crate) fn options() -> Options {
    OPTIONS.with(std::cell::Cell::get)
}

/// Runs `f` with `options` active on this thread, restoring the previous
/// options even if `f` panics.
pub(crate) fn with_options<T>(options: Options, f: impl FnOnce() -> T) -> T {
    struct Restore(Options);
    impl Drop for Restore {
        fn drop(&mut self) {
            OPTIONS.with(|o| o.set(self.0));
        }
    }
    let _restore = Restore(OPTIONS.with(|o| o.replace(options)));
    f()
}

/// Entries before slot 0's in the host's runtime stride table
/// ([`StrideEntry`]): slot `i`'s entry is at `(i + STRIDE_HEADER) * 8`. The
/// first holds, in its high dword, the title slots the host heap mirrors;
/// the second, the entries after the header; both low dwords are 0, so
/// either reads as no view. The table is a root SRV, which has no size: a
/// shader reads an entry only below those counts, and the first otherwise.
pub const STRIDE_HEADER: u32 = 2;

/// The bit of a view-heap index (bit 31 of the byte offset in the user
/// data) that says the host remapped it: a table the host made afresh in
/// dynamic slots. Only for targets with the stride table, where every heap
/// index loses it before a handle is made. Through the table, an index
/// with it reads the entry of its dynamic slot; one without it is the
/// title's, read only below the slots the host mirrors (a title slot past
/// a capped host heap has no host slot, and dynamic slots hold other
/// tables there).
pub const REMAPPED_SLOT: u32 = 1 << 26;

impl HostShader {
    /// Size of the user-data buffer the shader declares, in dwords (a
    /// multiple of 4; 0 when it reads none). Root constants must cover it.
    pub fn user_data_dwords(&self) -> u32 {
        let n = self
            .user_data
            .iter()
            .map(|u| u.index + 1)
            .max()
            .unwrap_or(0);
        n.max(self.group_base.map_or(0, |base| base + 3))
            .next_multiple_of(4)
    }

    /// The wave size the shader requires (`WaveSize`), as its `PSV0`
    /// says; `None` when it runs in waves of any size.
    pub fn wave_size(&self) -> Option<u32> {
        let c = Container::parse(&self.dxil).ok()?;
        let psv = c.part(FourCC::PSV0)?.data;
        let at = |o: usize| crate::dxbc::u32_at(psv, o).ok();
        let (min, max) = (at(4 + 16)?, at(4 + 20)?);
        (min == max).then_some(min)
    }

    /// Whether the shader writes the loop-limit counter ([`loop_count`]):
    /// its `PSV0` binds a raw UAV at `u0` in [`LOOP_COUNTER_SPACE`], which
    /// the host must bind (a root UAV) for it to run.
    pub fn loop_counter(&self) -> bool {
        let Ok(c) = Container::parse(&self.dxil) else {
            return false;
        };
        let Some(psv) = c.part(FourCC::PSV0).map(|p| p.data) else {
            return false;
        };
        let at = |o: usize| crate::dxbc::u32_at(psv, o).ok();
        // The runtime information, then the bindings: their count and size.
        let Some(first) = at(0).map(|info| 4 + info as usize) else {
            return false;
        };
        let (Some(count), Some(size)) = (at(first), at(first + 4)) else {
            return false;
        };
        (0..count as usize).any(|i| {
            let r = first + 8 + i * size as usize;
            // PSV resource type 7: a raw UAV.
            (at(r), at(r + 4), at(r + 8)) == (Some(7), Some(LOOP_COUNTER_SPACE), Some(0))
        })
    }
}

/// One dword of user data and the root parameter that fills it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UserData {
    /// Dword in the host user-data buffer. Spilled root arguments are packed.
    pub index: u32,
    /// Root parameter index in the title's root signature.
    pub slot: u8,
    /// `Constants` or `Table` (see the module notes for the value).
    pub root: RootType,
    /// Dword within the parameter.
    pub dword: u32,
}

/// The wave sizes (lanes per wave) the host GPU runs, as D3D12 reports them
/// (`D3D12_FEATURE_DATA_D3D12_OPTIONS1::WaveLaneCountMin` / `Max`). A
/// compute shader that declares the console's wave size translates only
/// for a host whose range holds it: NVIDIA's range is 32..32, so a console
/// wave64 program that reads other lanes needs 64-lane waves it cannot get.
///
/// `stride_table`: the host binds a runtime stride table ([`StrideEntry`],
/// `t0` in [`STRIDE_SPACE`]) to every shader that reads one and keeps it
/// for its view heaps, so a buffer read by element without a stride known
/// here translates (`xtp-gpu`'s `XTP_STRIDE_TABLE=1`). Without it such a
/// program is refused, as before: a host that does not declare the table
/// would refuse its pipeline instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Target {
    pub lanes_min: u32,
    pub lanes_max: u32,
    pub stride_table: bool,
}

impl Target {
    /// No limit known: every wave size is taken to run (the host refuses
    /// a pipeline it cannot run); no stride table.
    pub const ANY: Target = Target {
        lanes_min: 4,
        lanes_max: 128,
        stride_table: false,
    };

    /// Whether the host runs waves of `lanes`.
    pub fn runs(self, lanes: u32) -> bool {
        (self.lanes_min..=self.lanes_max).contains(&lanes)
    }

    /// The target as bytes, for cache keys: the range, then the stride
    /// table.
    pub fn key(self) -> [u8; 9] {
        let mut k = [0; 9];
        k[..4].copy_from_slice(&self.lanes_min.to_le_bytes());
        k[4..8].copy_from_slice(&self.lanes_max.to_le_bytes());
        k[8] = u8::from(self.stride_table);
        k
    }
}

impl Default for Target {
    fn default() -> Self {
        Target::ANY
    }
}

/// A diagnostic: what the module this thread emitted last declares and
/// calls (thread-group size, wave size, `dx.op` operations), for reports of
/// a module the validator refuses, and the form its program was lowered in
/// (`; per lane`, `; console flow`, with ` in halves` where a console wave64
/// runs as two host waves of 32). Nothing of the console program is in it.
pub fn last_module_summary() -> String {
    let flow = LAST_FLOW.get();
    let summary = shader::last_summary();
    if flow.is_empty() {
        summary
    } else {
        format!("{summary}; {flow}")
    }
}

/// Joins where a descriptor's dword took an undefined value in the console
/// flow without the proof that no lane reads it there (the `shortcut:` of
/// `lower/flow.rs`'s descriptor-word rule), over every translation of the
/// process: what a survey or a session reports of that shortcut.
pub static UNPROVEN_DESCRIPTOR_JOINS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

thread_local! {
    /// The form of the program [`emit_for`] translated last on this thread
    /// (see [`last_module_summary`]); empty after a failure.
    static LAST_FLOW: std::cell::Cell<&'static str> = const { std::cell::Cell::new("") };
}

/// Translates the vertex, pixel, compute or mesh program of a console
/// shader container, for any host ([`Target::ANY`]).
pub fn translate(container: &[u8]) -> Result<HostShader, ShaderError> {
    translate_for(container, Target::ANY)
}

/// [`translate`] with `options`.
pub fn translate_with(container: &[u8], options: Options) -> Result<HostShader, ShaderError> {
    translate_for_with(container, Target::ANY, options)
}

/// [`translate`] for a host that runs the wave sizes of `target`.
pub fn translate_for(container: &[u8], target: Target) -> Result<HostShader, ShaderError> {
    translate_for_with(container, target, Options::default())
}

/// [`translate_for`] with per-shader `options`.
pub fn translate_for_with(
    container: &[u8],
    target: Target,
    options: Options,
) -> Result<HostShader, ShaderError> {
    with_options(options, || {
        let c = Container::parse(container)?;
        let dxil = c
            .part(FourCC::DXIL)
            .ok_or(ShaderError::Malformed("no DXIL part"))?;
        let native = crate::native_programs(dxil.data)?
            .into_iter()
            .find(|p| {
                matches!(
                    p.stage,
                    HwStage::Vs | HwStage::Ps | HwStage::Cs | HwStage::Ms | HwStage::Gs
                )
            })
            .ok_or(ShaderError::Unsupported {
                offset: 0,
                what: "a container without a vertex, pixel, compute or mesh program",
            })?;
        translate_program_for(&c, &native, target).map_err(|(_, e)| e)
    })
}

/// [`translate_for`] of the export shader (ES) of a vertex shader container
/// instead of its vertex shader: the vertex shader of a pipeline that has a
/// geometry shader (`docs/analysis/es-gs.md`).
pub fn translate_es_for(container: &[u8], target: Target) -> Result<HostShader, ShaderError> {
    translate_es_for_with(container, target, Options::default())
}

/// [`translate_es_for`] with per-shader `options`.
pub fn translate_es_for_with(
    container: &[u8],
    target: Target,
    options: Options,
) -> Result<HostShader, ShaderError> {
    with_options(options, || {
        let c = Container::parse(container)?;
        let dxil = c
            .part(FourCC::DXIL)
            .ok_or(ShaderError::Malformed("no DXIL part"))?;
        let native = crate::native_programs(dxil.data)?
            .into_iter()
            .find(|p| p.stage == HwStage::Es)
            .ok_or(ShaderError::Unsupported {
                offset: 0,
                what: "a container without an export shader",
            })?;
        translate_program_for(&c, &native, target).map_err(|(_, e)| e)
    })
}

/// The step of [`translate_program`] that stopped a program, in the order
/// they report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Step {
    /// [`ir::build`]: decoding and the SSA form.
    Decode,
    /// [`resource::track`].
    Resources,
    /// [`io::raise`].
    Io,
    /// [`structure::structure`] (it runs before the resources, which need
    /// it, but its failure is reported after theirs and the I/O's).
    Structure,
    /// [`emit`].
    Emit,
}

impl Step {
    pub fn name(self) -> &'static str {
        match self {
            Step::Decode => "decode",
            Step::Resources => "resources",
            Step::Io => "I/O",
            Step::Structure => "structure",
            Step::Emit => "emit",
        }
    }
}

/// Translates one vertex, pixel, compute or mesh program of a parsed
/// container, for any host ([`Target::ANY`]): what [`translate`] does with
/// the first of them. On failure, the step that stopped it and its error
/// (the first blocker: fixing it can reveal another one in a later step).
pub fn translate_program(
    c: &Container,
    native: &NativeProgram,
) -> Result<HostShader, (Step, ShaderError)> {
    translate_program_for(c, native, Target::ANY)
}

/// [`translate_program`] for a host that runs the wave sizes of `target`.
pub fn translate_program_for(
    c: &Container,
    native: &NativeProgram,
    target: Target,
) -> Result<HostShader, (Step, ShaderError)> {
    let p = ir::build(&native.patched_code(), native.wave64, native.vgprs)
        .map_err(|e| (Step::Decode, e))?;
    let p = lower::geometry::prepared(p, native.stage).map_err(|e| (Step::Decode, e))?;
    let p = lower::rayquery::prepared(p, native).map_err(|e| (Step::Decode, e))?;
    // Structurisation first: tracing per-lane descriptors needs it (its
    // failure is reported after the earlier stages').
    let st = structure::structure(&p);
    let tracked = lower::geometry::tracked(c, &p, native).map_err(|e| (Step::Resources, e))?;
    let res = resource::track_native(&tracked, native, st.as_ref().ok())
        .map_err(|e| (Step::Resources, e))?;
    let io = io::raise(&p, native).map_err(|e| (Step::Io, e))?;
    let st = st.map_err(|e| (Step::Structure, e))?;
    emit_for(c, native, &p, &st, &io, &res, target).map_err(|e| (Step::Emit, e))
}

/// [`translate_program_for`] with per-shader `options`.
pub fn translate_program_for_with(
    c: &Container,
    native: &NativeProgram,
    target: Target,
    options: Options,
) -> Result<HostShader, (Step, ShaderError)> {
    with_options(options, || translate_program_for(c, native, target))
}

/// Emits a program whose earlier stages already ran, for any host
/// ([`Target::ANY`]).
pub fn emit(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
    st: &structure::Structured,
    io: &io::Io,
    res: &resource::Resources,
) -> Result<HostShader, ShaderError> {
    emit_for(c, native, p, st, io, res, Target::ANY)
}

/// [`emit`] for a host that runs the wave sizes of `target`.
pub fn emit_for(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
    st: &structure::Structured,
    io: &io::Io,
    res: &resource::Resources,
    target: Target,
) -> Result<HostShader, ShaderError> {
    LAST_FLOW.set("");
    let io = &*lower::geometry::with_shape(c, native, io)?;
    let sig = |fourcc| -> Result<Vec<container::SigElement>, ShaderError> {
        match c.part(fourcc) {
            Some(part) => signature(part.data),
            None => Ok(Vec::new()),
        }
    };
    let pixel = native.stage == HwStage::Ps;
    let mut inputs = sig(FourCC::ISG1)?;
    let mut outputs = sig(FourCC::OSG1)?;
    // Interpolation as DXC declares it by default; pixel inputs that are
    // read get the mode their barycentrics say.
    let interpolated = if pixel { &mut inputs } else { &mut outputs };
    for e in interpolated {
        e.interp = match (e.kind, e.comp) {
            (3, _) => 4,
            // Constant, as DXC declares it (the AMD compiler crashed on a
            // front face without a mode).
            (13, _) => 1,
            // Until a read says which mode (`lower`).
            (28, _) => 2,
            (0 | 6 | 7, 9) => 2,
            (0, _) => 1,
            _ => 0,
        };
    }
    // With a conservative depth output (SV_DepthLessEqual, GreaterEqual) DXIL
    // wants the position centroid (or per sample), as DXC declares it then.
    if pixel && outputs.iter().any(|e| matches!(e.kind, 18 | 19)) {
        for e in inputs.iter_mut().filter(|e| e.kind == 3) {
            e.interp = 5;
        }
    }
    // Per-primitive outputs are constant. The console never exports a
    // primitive id or cull flag a mesh shader writes (it culls with the
    // primitive export), so those rows are left out.
    let mut prims = sig(FourCC::PSG1)?;
    prims.retain(|e| !matches!(e.kind, 10 | 30));
    for e in &mut prims {
        e.interp = 1;
    }
    // A mesh group's id arrives multiplied by max(threads, vertices,
    // primitives) (the code divides by that); the thread count is in PSV0
    // from runtime info version 2 on.
    // shortcut: older containers fall back to the group's lane count, right
    // when the HLSL group fills its waves (BTP 6's do).
    let mesh_stride = match &native.io {
        crate::native::StageIo::Ms(ms) => {
            let psv = c.part(FourCC::PSV0).map(|p| p.data);
            let threads = psv
                .filter(|d| crate::dxbc::u32_at(d, 0).is_ok_and(|n| n >= 48))
                .and_then(|d| crate::dxbc::u32_at(d, 4 + 36).ok());
            threads.map_or(ms.threads, |t| {
                t.max(ms.max_vertices).max(ms.max_primitives)
            })
        }
        _ => 0,
    };
    let mut parts = lower::Parts {
        native,
        p,
        st,
        io,
        res,
        inputs,
        outputs,
        prims,
        mesh_stride,
        target,
        console_cf: false,
        per_lane_only: false,
    };
    let l = match lower::lower(&parts) {
        Err(ShaderError::Unsupported { what, .. }) if what == lower::CONSOLE_FLOW => {
            let console = structure::structure_console(p);
            let lowered = match &console {
                Ok(console) => {
                    parts.st = console;
                    parts.console_cf = true;
                    lower::lower(&parts)
                }
                Err(e) => Err(ShaderError::Unsupported {
                    offset: 0,
                    what: match e {
                        ShaderError::Unsupported { what, .. } => what,
                        _ => "a program the console structure cannot hold",
                    },
                }),
            };
            match lowered {
                Ok(l) => l,
                // What the console flow cannot lower keeps the per-lane form
                // where that translates. Else the reason is the console
                // flow's where the program needs it to run at all (a wave64
                // on a host without 64-lane waves: the per-lane form's
                // reason is only that), and the per-lane form's elsewhere:
                // what removes that one translates the program there.
                Err(e) => {
                    parts.st = st;
                    parts.console_cf = false;
                    parts.per_lane_only = true;
                    let halves = native.wave64 && !target.runs(64);
                    lower::lower(&parts).map_err(|per_lane| if halves { e } else { per_lane })?
                }
            }
        }
        l => l?,
    };
    LAST_FLOW.set(l.flow);
    Ok(HostShader {
        stage: native.stage,
        dxil: l.dxil,
        group_base: l.group_base,
        user_data: l.user_data,
        static_samplers: l
            .samplers
            .iter()
            .enumerate()
            .map(|(n, &w)| {
                let mut s = StaticSampler::from_words(w);
                if l.cmp_samplers >> n & 1 != 0 {
                    // D3D12 encode comparaison, minimum et maximum dans le même champ.
                    s.filter = (s.filter & !(3 << 7)) | 0x80;
                }
                s
            })
            .collect(),
        samplers: l.samplers,
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        stride_table: l.stride_table,
        run_time_buffers: l.run_time_buffers,
    })
}

/// The rows of an `ISG1`/`OSG1` part.
fn signature(part: &[u8]) -> Result<Vec<container::SigElement>, ShaderError> {
    use crate::dxbc::u32_at;
    let count = u32_at(part, 0)? as usize;
    let base = u32_at(part, 4)? as usize;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let at = base + 32 * i;
        let name_at = u32_at(part, at + 4)? as usize;
        let name = part
            .get(name_at..)
            .and_then(|s| s.split(|&b| b == 0).next())
            .ok_or(ShaderError::Malformed("signature name out of bounds"))?;
        let mask = *part
            .get(at + 24)
            .ok_or(ShaderError::Malformed("truncated"))?;
        if u32_at(part, at + 28)? != 0 {
            return Err(ShaderError::Unsupported {
                offset: 0,
                what: "a minimum-precision signature element",
            });
        }
        if u32_at(part, at)? != 0 {
            return Err(ShaderError::Unsupported {
                offset: 0,
                what: "a signature element of another stream",
            });
        }
        let kind = match u32_at(part, at + 12)? {
            0 => 0,
            1 => 3,
            2 => 6,
            3 => 7,
            4 => 4,
            5 => 5,
            6 => 1,
            7 => 10,
            8 => 2,
            9 => 13,
            10 => 12,
            64 => 16,
            65 => 17,
            66 => 14,
            67 => 19,
            68 => 18,
            69 => 20,
            70 => 15,
            23 => 28, // barycentrics
            25 => 30, // cull primitive
            _ => {
                return Err(ShaderError::Unsupported {
                    offset: 0,
                    what: "a system value in the signature",
                });
            }
        };
        // 16-bit elements are declared 32-bit; reading or writing one
        // converts (16-bit values travel packed on the console).
        let raw = u32_at(part, at + 16)?;
        let comp = match (kind, raw) {
            // DXIL declares the front face a uint (the validator insists).
            (13, _) => 5,
            (_, 1 | 4) => 5,
            (_, 2 | 5) => 4,
            (_, 3 | 6) => 9,
            (_, c) => {
                return Err(ShaderError::Unsupported {
                    offset: 0,
                    what: match c {
                        0 => "a signature element of unknown type",
                        _ => "a 64-bit signature element",
                    },
                });
            }
        };
        let register = u32_at(part, at + 20)?;
        out.push(container::SigElement {
            name: String::from_utf8_lossy(name).into_owned(),
            index: u32_at(part, at + 8)?,
            kind,
            comp,
            interp: 0,
            row: (register != u32::MAX).then_some(register),
            col: mask.trailing_zeros().min(3) as u8,
            cols: mask.count_ones().max(1) as u8,
            usage: 0,
            half: (4..=6).contains(&raw),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::Target;
    use super::bitcode::Attr;
    use super::lower::LOOP_LIMIT;
    use super::parse_loop_limit;

    #[test]
    fn the_loop_limit_switch_parses_or_falls_back() {
        assert_eq!(LOOP_LIMIT, 16_384);
        // Spin waits never get more than the general limit, counted loops
        // the old one.
        const { assert!(super::DEFAULT_SPIN_LIMIT <= LOOP_LIMIT) };
        assert_eq!(super::DEFAULT_COUNTED_LOOP_LIMIT, 1 << 20);
        assert_eq!(parse_loop_limit(None), LOOP_LIMIT);
        assert_eq!(parse_loop_limit(Some("4096")), 4096);
        assert_eq!(parse_loop_limit(Some(" 0x1000 ")), 4096);
        assert_eq!(parse_loop_limit(Some("1")), 1);
        for bad in ["", "0", "-5", "lots", "0x", "4294967296"] {
            assert_eq!(parse_loop_limit(Some(bad)), LOOP_LIMIT, "{bad:?}");
            assert_eq!(super::parse_limit_override(Some(bad)), None, "{bad:?}");
        }
        // Without the switch the counted loops keep their own limit.
        assert_eq!(super::parse_limit_override(None), None);
        assert_eq!(super::parse_limit_override(Some("0x10")), Some(16));
    }

    /// The budget switch: a value, 0 to turn it off, the default otherwise.
    #[test]
    fn the_loop_budget_switch_parses_or_falls_back() {
        use super::{DEFAULT_LOOP_BUDGET, parse_loop_budget};
        assert_eq!(DEFAULT_LOOP_BUDGET, 1 << 20);
        assert_eq!(parse_loop_budget(None), DEFAULT_LOOP_BUDGET);
        assert_eq!(parse_loop_budget(Some("0")), 0);
        assert_eq!(parse_loop_budget(Some(" 0x400 ")), 1024);
        assert_eq!(parse_loop_budget(Some("5000")), 5000);
        for bad in ["", "-5", "lots", "0x", "4294967296"] {
            assert_eq!(parse_loop_budget(Some(bad)), DEFAULT_LOOP_BUDGET, "{bad:?}");
        }
    }
    use super::container::SigElement;
    use super::shader::Shader;
    use super::*;
    use crate::native::{ComputeIo, StageIo};

    /// A pixel shader writing SV_StencilRef (channel y of `exp mrtz`).
    /// DXC sets ShaderFlags.StencilRef (bit 11) for a pixel shader whose
    /// output signature has SV_StencilRef, and the feature bit
    /// D3D_SHADER_FEATURE_STENCIL_REF (0x200) in SFI0; the validator checks
    /// both against the module.
    #[test]
    fn a_stencil_export_declares_the_stencil_ref_feature() {
        let words = [
            0x7E00_02F2u32, // v_mov_b32 v0, 1.0
            0x7E02_0285,    // v_mov_b32 v1, 5
            0xF800_1883,
            0x0000_0100, // exp mrtz v0, v1, off, off done vm
            0xBF81_0000,
        ];
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            float_mode: 0,
            vgprs: 2,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![],
                input_addr: 0,
                rsrc2: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let el = |name: &str, kind, comp| SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp,
            interp: 0,
            row: None,
            col: 0,
            cols: 1,
            usage: 1,
            half: false,
        };
        let osg1 =
            container::signature(&[el("SV_Depth", 17, 9), el("SV_StencilRef", 20, 5)], false);
        let bytes = crate::dxbc::build(&[(FourCC::OSG1, &osg1)]);
        let c = Container::parse(&bytes).unwrap();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let out = emit(&c, &native, &p, &st, &io, &res).unwrap();
        // dxil.dll, where there is one, checks the flags against the module.
        #[cfg(windows)]
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut out.dxil.clone()).unwrap();
        }
        let host = Container::parse(&out.dxil).unwrap();
        let sfi0 = host.part(FourCC::SFI0).unwrap().data;
        let features = u64::from_le_bytes(sfi0.try_into().unwrap());
        assert_ne!(features & 0x200, 0, "SFI0 {features:#x}: no StencilRef");
    }

    /// A wave64 compute program that writes only EXEC's high dword
    /// (`s_mov_b32 exec_hi, 0`) before a vector instruction. The new mask is
    /// opaque and keyed by the unchanged low dword, so building its lane bit
    /// needed the mask's own value: lowering recursed until the stack
    /// overflowed, which takes the whole process down (found by
    /// random-input testing). Now the instruction fails like any other that
    /// cannot be expressed: its result is missing, and only a side effect
    /// that uses it would stop the translation.
    #[test]
    fn an_exec_high_write_does_not_overflow_the_stack() {
        // s_mov_b32 exec_hi, 0; v_mov_b32 v1, v0; s_endpgm
        let words = [0xBEFF_0380u32, 0x7E02_0300, 0xBF81_0000];
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: true,
            float_mode: 0,
            vgprs: 2,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 0,
                threads: [64, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let container = crate::dxbc::build(&[]);
        let c = Container::parse(&container).unwrap();
        if let Err(e) = emit(&c, &native, &p, &st, &io, &res) {
            assert!(e.to_string().contains("its own value"), "{e}");
        }
    }

    /// A pixel shader writing a constant colour.
    #[cfg(windows)]
    fn constant_color() -> Vec<u8> {
        let mut s = Shader::new(true);
        s.outputs.push(SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 15,
            half: false,
        });
        let b = s.entry;
        for (c, v) in [1.0f32, 0.5, 0.25, 1.0].into_iter().enumerate() {
            let args = [
                s.m.i32(0),
                s.m.i32(0),
                s.m.i8(c as u8),
                s.m.f32(v.to_bits()),
            ];
            s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
        }
        s.m.terminate(b, super::bitcode::Term::Ret);
        s.finish()
    }

    #[test]
    #[cfg(windows)]
    fn stub_pipeline_is_created() {
        let Some(host) = super::host::Host::test_device() else {
            return;
        };
        let ps = super::HostShader {
            stage: crate::HwStage::Ps,
            dxil: constant_color(),
            group_base: None,
            user_data: Vec::new(),
            samplers: Vec::new(),
            static_samplers: Vec::new(),
            root_cbvs: Vec::new(),
            root_srvs: Vec::new(),
            root_uavs: Vec::new(),
            stride_table: false,
            run_time_buffers: 0,
        };
        let file = |var: &str, stage| {
            std::env::var(var).ok().map(|p| super::HostShader {
                stage,
                dxil: std::fs::read(p).unwrap(),
                group_base: None,
                user_data: Vec::new(),
                samplers: Vec::new(),
                static_samplers: Vec::new(),
                root_cbvs: Vec::new(),
                root_srvs: Vec::new(),
                root_uavs: Vec::new(),
                stride_table: false,
                run_time_buffers: 0,
            })
        };
        let ps = file("XTP_REF_PS", crate::HwStage::Ps).unwrap_or(ps);
        let vs = file("XTP_REF_VS", crate::HwStage::Vs)
            .unwrap_or_else(|| super::host::stub_vs(&ps).unwrap());
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut vs.dxil.clone()).unwrap();
        }
        host.pipeline(&vs, &ps).unwrap();
        // The runtime checks the container hash: no hash or a wrong one is
        // refused. (Recent runtimes also take sixteen 0x01 bytes, a bypass
        // value; ours works with older ones too.)
        let mut wrong = ps.dxil[4..20].to_vec();
        wrong[0] ^= 1;
        for digest in [&[0u8; 16][..], &wrong] {
            let mut other = ps.clone();
            other.dxil[4..20].copy_from_slice(digest);
            assert!(
                host.pipeline(&vs, &other).is_err(),
                "digest {digest:02x?} accepted"
            );
        }
    }

    #[test]
    #[cfg(windows)]
    fn constant_color_validates() {
        let mut c = constant_color();
        let path = super::validator::Validator::default_path();
        let Ok(mut v) = super::validator::Validator::load(path.as_deref()) else {
            super::host::skip_once(xtp_testkit::reason::NO_DXIL);
            return;
        };
        let ours = c[4..20].to_vec();
        v.validate(&mut c).unwrap();
        assert_eq!(c[4..20], ours[..], "our hash differs from the validator's");
    }

    /// A pixel shader interpolating at sample positions (PERSP_SAMPLE
    /// barycentrics, `SPI_PS_INPUT_ADDR` bit 0) runs at sample frequency:
    /// DXC's PSV0 says so in PSInfo.SampleFrequency (runtime info byte 1)
    /// whenever an input is interpolated per sample.
    #[test]
    fn sample_interpolated_inputs_set_psv_sample_frequency() {
        let words = [
            0xC808_0000u32, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001,    // v_interp_p2_f32 v2, v1, attr0.x
            0xF800_180F,
            0x0202_0202, // exp mrt0, v2, v2, v2, v2 done vm
            0xBF81_0000,
        ];
        let psv_of = |input_addr: u32| {
            let native = super::NativeProgram {
                stage: super::HwStage::Ps,
                wave64: false,
                float_mode: 0,
                vgprs: 4,
                layout: vec![],
                used: vec![],
                io: crate::native::StageIo::Ps {
                    inputs: vec![crate::native::PsInput {
                        usage: crate::native::Usage::Generic,
                        row: 0,
                        mask: 0b1111,
                        attr: 0,
                        flat: false,
                        fp16: false,
                        pack2: false,
                        row2: 0,
                    }],
                    input_addr,
                    rsrc2: 0,
                    db_shader_control: 0,
                    patches: vec![],
                },
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let el = |name: &str, kind, usage| SigElement {
                name: name.into(),
                index: 0,
                kind,
                comp: 9,
                interp: 0,
                row: Some(0),
                col: 0,
                cols: 4,
                usage,
                half: false,
            };
            let isg1 = super::container::signature(&[el("TEXCOORD", 0, 0)], true);
            let osg1 = super::container::signature(&[el("SV_Target", 16, 15)], false);
            let bytes =
                crate::dxbc::build(&[(super::FourCC::ISG1, &isg1), (super::FourCC::OSG1, &osg1)]);
            let c = super::Container::parse(&bytes).unwrap();
            let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
            let st = crate::structure::structure(&p).unwrap();
            let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
            let io = crate::io::raise(&p, &native).unwrap();
            let out = super::emit(&c, &native, &p, &st, &io, &res).unwrap();
            // dxil.dll, where there is one, checks PSV0 against the module.
            #[cfg(windows)]
            if let Ok(mut v) = super::validator::Validator::load(
                super::validator::Validator::default_path().as_deref(),
            ) {
                v.validate(&mut out.dxil.clone()).unwrap();
            }
            let host = super::Container::parse(&out.dxil).unwrap();
            host.part(super::FourCC::PSV0).unwrap().data.to_vec()
        };
        // Runtime info starts after its size; PSInfo is {DepthOutput, SampleFrequency}.
        let center = psv_of(1 << 1);
        assert_eq!(center[4 + 1], 0, "center interpolation is per pixel");
        let sample = psv_of(1 << 0);
        assert_eq!(sample[4 + 1], 1, "PSInfo.SampleFrequency");
    }

    /// A pixel shader reading channel z of an input whose element starts at
    /// w. Looking the channel up subtracted the element's first channel
    /// before checking the channel lies in it, so in a build with overflow
    /// checks (tests, debug) any read below an element that starts at z or
    /// w panicked, whichever row it was in (found by random-input testing).
    #[test]
    fn reading_a_channel_below_an_input_element_does_not_underflow() {
        // v_interp_mov_f32 v3, p0, attr0.z; s_endpgm
        let words = [0xC80E_0202u32, 0xBF81_0000];
        let native = super::NativeProgram {
            stage: super::HwStage::Ps,
            wave64: false,
            float_mode: 0,
            vgprs: 4,
            layout: vec![],
            used: vec![],
            io: crate::native::StageIo::Ps {
                inputs: vec![crate::native::PsInput {
                    usage: crate::native::Usage::Generic,
                    row: 0,
                    mask: 0b1000,
                    attr: 0,
                    flat: false,
                    fp16: false,
                    pack2: false,
                    row2: 0,
                }],
                input_addr: 0,
                rsrc2: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let w_only = SigElement {
            name: "TEXCOORD".into(),
            index: 0,
            kind: 0,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 3,
            cols: 1,
            usage: 0,
            half: false,
        };
        let isg1 = super::container::signature(&[w_only], true);
        let bytes = crate::dxbc::build(&[(super::FourCC::ISG1, &isg1)]);
        let c = super::Container::parse(&bytes).unwrap();
        let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        // Translated or refused with a reason; it must not panic.
        let _ = super::emit(&c, &native, &p, &st, &io, &res);
    }

    /// A compute program of 64 threads in one group: `words` as GFX10 code.
    pub(super) fn compute(words: &[u32], wave64: bool) -> crate::NativeProgram {
        compute_of(words, wave64, 64)
    }

    /// [`compute`] with `threads` threads in the group.
    pub(super) fn compute_of(words: &[u32], wave64: bool, threads: u32) -> crate::NativeProgram {
        crate::NativeProgram {
            stage: crate::HwStage::Cs,
            wave64,
            float_mode: 0,
            vgprs: 8,
            layout: vec![],
            used: vec![],
            io: crate::native::StageIo::Cs(crate::native::ComputeIo {
                rsrc2: 0,
                threads: [threads, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        }
    }

    pub(super) fn emit_for(
        native: &crate::NativeProgram,
        target: Target,
    ) -> Result<super::HostShader, crate::ShaderError> {
        let c = crate::Container {
            digest: [0; 16],
            version: (1, 0),
            parts: vec![],
        };
        let p = crate::ir::build(&native.code, native.wave64, native.vgprs)?;
        let st = crate::structure::structure(&p);
        let res = crate::resource::track(&p, &native.layout, native.stage, st.as_ref().ok())?;
        let io = crate::io::raise(&p, native)?;
        let st = st?;
        super::emit_for(&c, native, &p, &st, &io, &res, target)
    }

    const NVIDIA: Target = Target {
        lanes_min: 32,
        lanes_max: 32,
        stride_table: false,
    };
    const RDNA: Target = Target {
        lanes_min: 32,
        lanes_max: 64,
        stride_table: false,
    };

    #[test]
    fn console_lanes_translate_only_for_hosts_with_that_wave_size() {
        let words = [
            0xD760_0000,
            0x0001_0B00, // v_readlane_b32 s0, v0, 5
            0x7E02_0200, // v_mov_b32 v1, s0
            0xBF81_0000, // s_endpgm
        ];
        // One console wave per group: in the console flow on NVIDIA.
        let wave64 = compute(&words, true);
        assert!(emit_for(&wave64, Target::ANY).is_ok());
        assert!(emit_for(&wave64, RDNA).is_ok());
        assert!(emit_for(&wave64, NVIDIA).is_ok());
        // A constant lane in this group can use its existing LDS exchange.
        let two = compute_of(&words, true, 128);
        assert!(emit_for(&two, RDNA).is_ok());
        assert_eq!(emit_for(&two, NVIDIA).unwrap().wave_size(), Some(32));
        // A computed lane needs both register sets. Keep a partial last
        // wave as a negative control.
        let computed = [
            0xD760_0000,
            0x0000_0300, // v_readlane_b32 s0, v0, s1
            0x7E02_0200, // v_mov_b32 v1, s0
            0xBF81_0000, // s_endpgm
        ];
        assert_eq!(
            emit_for(&compute_of(&computed, true, 96), NVIDIA),
            Err(crate::ShaderError::Unsupported {
                offset: 0,
                what: "console wave64 lanes in a group of more than 64 threads on a host without 64-lane waves",
            })
        );
        assert!(emit_for(&compute_of(&computed, true, 128), NVIDIA).is_ok());
        // The same code in waves of 32 runs on either.
        let wave32 = compute(&words, false);
        assert!(emit_for(&wave32, NVIDIA).is_ok());
        assert!(emit_for(&wave32, RDNA).is_ok());
    }

    #[test]
    fn the_wave_size_a_translation_requires_is_read_from_its_psv0() {
        let lane_read = [0xD760_0000, 0x0001_0B00, 0x7E02_0200, 0xBF81_0000];
        let wave = |words: &[u32], wave64: bool, target: Target| {
            emit_for(&compute(words, wave64), target)
                .unwrap()
                .wave_size()
        };
        assert_eq!(wave(&lane_read, true, RDNA), Some(64));
        // The console wave as two 32-lane halves.
        assert_eq!(wave(&lane_read, true, NVIDIA), Some(32));
        assert_eq!(wave(&lane_read, false, NVIDIA), Some(32));
        // Code reading no other lane runs in waves of any size.
        let own = [0x7E02_0300, 0xBF81_0000];
        assert_eq!(wave(&own, true, NVIDIA), None);
    }

    #[test]
    #[cfg(windows)]
    fn what_translates_for_the_host_creates_a_pipeline() {
        let Some(host) = super::host::Host::test_device() else {
            return;
        };
        let target = host.target();
        eprintln!("host waves {}-{}", target.lanes_min, target.lanes_max);
        let words = [0xD760_0000, 0x0001_0B00, 0x7E02_0200, 0xBF81_0000];
        for wave64 in [false, true] {
            match emit_for(&compute_of(&words, wave64, 128), target) {
                Ok(cs) => {
                    if let Err(e) = host.compute_pipeline(&cs) {
                        panic!("wave64 {wave64}: translated for the host, refused by it: {e}");
                    }
                }
                Err(e) => {
                    // The host has no waves of the program's size. (The
                    // software adapter of a machine without a GPU has
                    // 4-lane waves: not the wave32 program either.)
                    let wave = if wave64 { 64 } else { 32 };
                    assert!(!target.runs(wave), "{e}");
                    // What it replaces: the host refuses the console's
                    // wave size at pipeline creation.
                    let any = emit_for(&compute_of(&words, wave64, 128), Target::ANY).unwrap();
                    assert!(host.compute_pipeline(&any).is_err());
                }
            }
        }
    }

    /// Groups of one or two waves can read a console lane on either host.
    #[test]
    #[cfg(windows)]
    fn each_adapter_gets_what_it_runs() {
        if super::host::test_adapter().as_deref() == Some("warp") {
            super::host::skip_once("hardware adapters kept out (XTP_TEST_ADAPTER=warp)");
            return;
        }
        if super::host::test_adapter().is_some() {
            let _ = super::host::Host::test_device();
            return;
        }
        let names = match super::host::adapters() {
            Ok(names) => names,
            Err(e) => {
                super::host::skip_gpu(&format!("no DXGI ({e})"));
                return;
            }
        };
        let words = [0xD760_0000, 0x0001_0B00, 0x7E02_0200, 0xBF81_0000];
        let mut ran = 0;
        for name in names {
            let host = match super::host::Host::on(&name) {
                Ok(host) => host,
                Err(e) => {
                    eprintln!("{name}: no SM 6.6 device ({e})");
                    continue;
                }
            };
            ran += 1;
            let target = host.target();
            eprintln!("{name}: waves {}-{}", target.lanes_min, target.lanes_max);
            let one = match emit_for(&compute(&words, true), target) {
                Ok(one) => one,
                Err(e) => {
                    assert!(!target.runs(64) && !target.runs(32), "{name}: {e}");
                    continue;
                }
            };
            host.compute_pipeline(&one)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let two = emit_for(&compute_of(&words, true, 128), target).unwrap();
            host.compute_pipeline(&two)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
        }
        if ran == 0 {
            super::host::skip_gpu("no hardware adapter with an SM 6.6 device");
        }
    }

    #[test]
    fn programs_reading_no_other_lanes_translate_for_any_wave_size() {
        let words = [
            0x7E02_0300, // v_mov_b32 v1, v0
            0xBF81_0000, // s_endpgm
        ];
        assert!(emit_for(&compute(&words, true), NVIDIA).is_ok());
    }
}
