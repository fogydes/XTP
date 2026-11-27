//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver.
//!
//! Une passe de calcul borne une série d’exécutions à partir de sa table
//! d’upload. Chaque commande et son compteur occupent une tranche privée.
//! Les dispatches ordinaires y gardent le format natif XYZ, les morceaux
//! qui utilisent une origine de groupe ont une signature distincte.
//!
//! Each list also has a budget (2^27 vertex instances or threads, groups
//! counted as 64 threads, mesh groups as 128): records past it run empty,
//! so a list of thousands of garbage records cannot run for seconds
//! either. One group charges it in record order (VIctory #1510): by
//! default a record that does not fit runs empty and leaves the remaining
//! budget for later records that do fit, so one oversized record does not
//! empty the rest of the list. Setting `XTP_INDIRECT_REST=1` restores the
//! earlier behaviour where the first record that does not fit spends the
//! rest of the budget and every record after it in the list runs empty.
//! A word that every thread adds to wraps once enough rejected records
//! have charged it, and lets work in again.
//!
//! shortcut: one group walks every record, 64 at a time, so the clamp's
//! time grows with the records a title passes (43 ms for 2^20 records on
//! an RTX 4080, by #1510's author).
//!
//! Indirect rectangle-list draws follow the clamp with an expansion on the
//! GPU ([`crate::rectangles`]).
//!
//! Idea from VIctory #1820 (report by Exopus-0x65f7ad0).
//! Le GPU produit des commandes hôtes (base XYZ, dimensions XYZ) dans
//! l’ordre du titre. Le budget porte sur le dispatch entier.
//! Avec XTP_DISPATCH_COUNTS=1, chaque bilan de trame regroupe par shader les
//! dimensions originales, les limites atteintes et les rejets de budget.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{Interface, s};

use crate::host::{HostList, host, transition};

const SOURCE: &str = r#"
#define RS "SRV(t0), UAV(u0), UAV(u1), UAV(u2), UAV(u3), UAV(u4)"
ByteAddressBuffer run : register(t0);
RWByteAddressBuffer args : register(u0);
RWByteAddressBuffer budget : register(u1);
RWByteAddressBuffer pieces : register(u2);
RWByteAddressBuffer countBuffer : register(u3);
RWByteAddressBuffer stats : register(u4);
static const uint BUDGET = 1u << 27;
static const uint LIMIT = 65535;
static const uint MAX_GROUPS = BUDGET / 64;
groupshared uint costs[64];
groupshared uint sizes[64];
groupshared uint starts[64];
groupshared uint metrics[16];

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint lane : SV_GroupIndex) {
    uint left = 0;
    if (lane == 0) left = BUDGET - min(budget.Load(0), BUDGET);
    uint executions = run.Load(0);
    for (uint e = 0; e < executions; ++e) {
        uint4 spec = run.Load4(16 + e * 32);
        uint kind = spec.x, stride = spec.y, records = spec.z, flags = spec.w;
        uint4 offsets = run.Load4(32 + e * 32);
        uint active = records;
        if (flags & 1) active = min(active, countBuffer.Load(offsets.y));
        bool measure = (flags & 2) != 0;
        if (measure && lane < 16) metrics[lane] = 0;
        GroupMemoryBarrierWithGroupSync();
        bool compact = (flags & 4) != 0;
        bool prefix = (flags & 8) != 0;
        uint nativeAt = offsets.z + 16 + 24 * min(records + MAX_GROUPS / LIMIT, MAX_GROUPS);
        uint nativeCount = 0, written = 0;
        for (uint tile = 0; tile < records; tile += 64) {
            uint n = min(records - tile, 64u);
            uint at = offsets.x + (tile + lane) * stride;
            uint cost = 0, size = 0;
            uint3 g = 0;
            uint2 draw = 0;
            if (tile + lane < active) {
                if (kind <= 1) {
                    uint2 raw = args.Load2(at);
                    uint vertices = min(raw.x, 1u << 24);
                    uint instances = min(raw.y, (1u << 26) / max(vertices, 1u));
                    if (measure) {
                        InterlockedMax(metrics[8], raw.x);
                        InterlockedMax(metrics[9], raw.y);
                        if (vertices != raw.x) InterlockedAdd(metrics[10], 1);
                        if (instances != raw.y) InterlockedAdd(metrics[11], 1);
                        if (any(raw != uint2(vertices, instances))) InterlockedAdd(metrics[4], 1);
                    }
                    draw = all(raw == uint2(vertices, instances)) ? raw : uint2(0, 0);
                    cost = draw.x * draw.y;
                } else {
                    g = args.Load3(at);
                    if (measure) {
                        InterlockedMax(metrics[0], g.x);
                        InterlockedMax(metrics[1], g.y);
                        InterlockedMax(metrics[2], g.z);
                        if (any(g > LIMIT)) InterlockedAdd(metrics[3], 1);
                    }
                    if (kind == 10 && measure && any(g == 0)) InterlockedAdd(metrics[14], 1);
                    if (all(g != 0)) {
                        uint weight = kind == 2 ? 64 : 128;
                        if (measure && g.x > (1u << 20) / g.y / g.z) InterlockedAdd(metrics[12], 1);
                        // Division avant multiplication, même pour trois UINT_MAX.
                        if (g.x > BUDGET / weight / g.y / g.z) {
                            if (measure) InterlockedAdd(metrics[5], 1);
                        } else {
                            uint groups = g.x * g.y * g.z;
                            cost = groups * weight;
                            // MeshGroup ne lit que X. Une origine X par morceau
                            // conserve tous les identifiants de la grille linéarisée.
                            if (kind == 10) g = uint3(groups, 1, 1);
                            uint3 tiles = (g - 1) / LIMIT + 1;
                            size = tiles.x * tiles.y * tiles.z;
                        }
                    }
                }
            }
            costs[lane] = cost;
            sizes[lane] = size;
            GroupMemoryBarrierWithGroupSync();
            if (lane == 0) {
                for (uint i = 0; i < n; i++) {
                    starts[i] = written;
                    if (costs[i] > left) {
#if REST
                        left = 0;
#endif
                        costs[i] = 0xffffffff;
                        if (measure) InterlockedAdd(metrics[6], 1);
                    } else {
                        left -= costs[i];
                        if (measure && sizes[i] > 1) InterlockedAdd(metrics[15], 1);
                        // Préfixe au format D3D12 natif jusqu’au premier
                        // découpage. Le suffixe garde ensuite l’ordre du titre.
                        if (prefix && written == 0 && sizes[i] == 1)
                            starts[i] = 0x80000000 | nativeCount++;
                        else
                            written += sizes[i];
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
            if (kind <= 1 && tile + lane < active) {
                uint dst = offsets.z + 4 + (tile + lane) * 24;
                if (costs[lane] == 0xffffffff) draw = 0;
                pieces.Store4(dst, uint4(draw, args.Load2(at + 8)));
                if (kind == 1) pieces.Store(dst + 16, args.Load(at + 16));
            } else if (size != 0 && costs[lane] != 0xffffffff && (starts[lane] & 0x80000000)) {
                pieces.Store3(nativeAt + 4 + 12 * (starts[lane] & 0x7fffffff), g);
            } else if (size != 0 && costs[lane] != 0xffffffff) {
                // Deux axes > LIMIT dépasseraient déjà le budget de toute la liste.
                uint3 step = uint3(g.x > LIMIT, g.y > LIMIT, g.z > LIMIT) * LIMIT;
                for (uint j = 0; j < size; j++) {
                    uint3 base = step * j;
                    uint index = starts[lane] + j;
                    uint3 dims = min(g - base, LIMIT);
                    if (compact) {
                        pieces.Store3(offsets.z + 4 + 12 * index, dims);
                    } else {
                        uint dst = offsets.z + 4 + 24 * index;
                        pieces.Store3(dst, base);
                        pieces.Store3(dst + 12, dims);
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if (lane == 0) {
            pieces.Store(offsets.z, kind > 1 ? written : active);
            if (prefix) pieces.Store(nativeAt, nativeCount);
            if (measure) {
                stats.Store4(offsets.w + 0, uint4(metrics[0], metrics[1], metrics[2], metrics[3]));
                stats.Store4(offsets.w + 16, uint4(metrics[4], metrics[5], metrics[6], active));
                stats.Store4(offsets.w + 32, uint4(metrics[8], metrics[9], metrics[10], metrics[11]));
                stats.Store4(offsets.w + 48, uint4(metrics[12], BUDGET - left, metrics[14], metrics[15]));
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0) budget.Store(0, BUDGET - left);
}
"#;

/// Résultats écrits par le GPU avant et pendant le bridage.
#[cfg(feature = "diagnostics")]
#[derive(Clone)]
pub(crate) enum Sample {
    Indirect {
        _buffer: ID3D12Resource,
        data: usize,
        hash: String,
    },
    Direct {
        groups: [u32; 3],
        hash: String,
    },
}

#[cfg(feature = "diagnostics")]
fn measuring() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XTP_DISPATCH_COUNTS").is_ok_and(|v| v == "1"))
}

#[cfg(feature = "diagnostics")]
pub(crate) fn note_direct(list: &HostList, pipeline: usize, groups: [u32; 3]) {
    if measuring() {
        let hash = crate::pipeline::dispatch_hash(pipeline as _).unwrap_or_default();
        list.dispatch_sample(Sample::Direct { groups, hash });
    }
}

#[cfg(feature = "diagnostics")]
#[derive(Debug, Default)]
struct Counts {
    max: [u32; 3],
    oversized: u64,
    clamped: u64,
    zeroed: u64,
    past_budget: u64,
    records: u64,
    direct: u64,
    max_vertices: u32,
    max_instances: u32,
    vertex_limit: u64,
    instance_limit: u64,
    over_million_groups: u64,
    max_list_cost: u32,
    empty_mesh: u64,
    split_records: u64,
}

#[cfg(feature = "diagnostics")]
static COUNTS: std::sync::Mutex<std::collections::BTreeMap<String, Counts>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());

#[cfg(feature = "diagnostics")]
impl Sample {
    pub(crate) fn report(&self) {
        let mut counts = COUNTS.lock().unwrap_or_else(|e| e.into_inner());
        let (data, hash) = match self {
            Self::Direct { groups, hash } => {
                let c = counts.entry(hash.clone()).or_default();
                c.direct += 1;
                c.oversized += u64::from(groups.iter().any(|&n| n > LIMIT));
                for (high, n) in c.max.iter_mut().zip(groups) {
                    *high = (*high).max(*n);
                }
                return;
            }
            Self::Indirect { data, hash, .. } => (*data, hash),
        };
        // SAFETY: le tampon mappé est conservé par l’enregistrement soumis.
        let read = |i| unsafe { (data as *const u32).add(i).read_volatile() };
        let c = counts.entry(hash.clone()).or_default();
        for (i, v) in c.max.iter_mut().enumerate() {
            *v = (*v).max(read(i));
        }
        c.oversized += u64::from(read(3));
        c.clamped += u64::from(read(4));
        c.zeroed += u64::from(read(5));
        c.past_budget += u64::from(read(6));
        c.records += u64::from(read(7));
        c.max_vertices = c.max_vertices.max(read(8));
        c.max_instances = c.max_instances.max(read(9));
        c.vertex_limit += u64::from(read(10));
        c.instance_limit += u64::from(read(11));
        c.over_million_groups += u64::from(read(12));
        c.max_list_cost = c.max_list_cost.max(read(13));
        c.empty_mesh += u64::from(read(14));
        c.split_records += u64::from(read(15));
    }
}

#[cfg(feature = "diagnostics")]
pub(crate) fn log_counts() {
    let counts = std::mem::take(&mut *COUNTS.lock().unwrap_or_else(|e| e.into_inner()));
    for (hash, counts) in counts {
        tracing::info!(%hash, ?counts, "dispatch and draw counts per shader");
    }
}

/// Les compteurs de diagnostic occupent 64 octets par exécution dans une page de la liste.
#[derive(Default)]
pub(crate) struct Samples {
    #[cfg(feature = "diagnostics")]
    page: Option<(ID3D12Resource, usize, usize)>,
}

impl Samples {
    pub(crate) fn next(&mut self, list: &HostList, pipeline: usize) -> Option<(u64, u32)> {
        #[cfg(feature = "diagnostics")]
        {
            if measuring() {
                const PAGE: usize = 65536;
                if self.page.as_ref().is_none_or(|p| p.2 == PAGE) {
                    let (buffer, data) = host()?.shared_buffer(PAGE as u64)?;
                    self.page = Some((buffer, data as usize, 0));
                }
                let (buffer, data, offset) = self.page.as_mut()?;
                let hash = crate::pipeline::dispatch_hash(pipeline as _).unwrap_or_default();
                list.dispatch_sample(Sample::Indirect {
                    _buffer: buffer.clone(),
                    data: *data + *offset,
                    hash,
                });
                // SAFETY: l’enregistrement soumis conserve la page et chaque plage est distincte.
                let gpu = (unsafe { buffer.GetGPUVirtualAddress() }, *offset as u32);
                *offset += 64;
                return Some(gpu);
            }
        }
        let _ = (list, pipeline);
        None
    }
}

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

/// Whether a record that does not fit in the list's remaining budget spends
/// the rest of it so every later record in the list runs empty too
/// (`XTP_INDIRECT_REST=1`; off by default so one oversized record runs empty
/// and later records that fit still run).
fn rest() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| rest_from(std::env::var_os("XTP_INDIRECT_REST").as_deref()))
}

fn rest_from(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_some_and(|v| v == "1")
}

fn clamp_for(take_rest: bool) -> Option<&'static Clamp> {
    static DEFAULT: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    static TAKE_REST: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    let slot = if take_rest { &TAKE_REST } else { &DEFAULT };
    slot.get_or_init(|| match make(take_rest) {
        Ok(c) => Some(c),
        Err(e) => {
            tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
            None
        }
    })
    .as_ref()
}

fn make(take_rest: bool) -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    let source = format!("#define REST {}\n{SOURCE}", u32::from(take_rest));
    compile_compute(device, &source)
}

fn compile_compute(device: &ID3D12Device, source: &str) -> windows::core::Result<Clamp> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            s!("clamp"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                )
            };
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes =
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Clamp { root, pso })
    }
}

const LIMIT: u32 = 65535;
const RECORD_BYTES: u32 = 24;
// Chaque morceau supplémentaire consomme au moins LIMIT groupes du budget.
const MAX_GROUPS: u32 = (1 << 27) / 64;
const EXTRA: u32 = MAX_GROUPS / LIMIT;

const RUN_LENGTH: usize = 128;
const PAGE_BYTES: u32 = 64 << 10;

/// Une écriture dans l’un des deux tampons ferme la série. Seules les écritures
/// UAV demandent aussi une barrière, les copies ont leurs transitions D3D12.
#[derive(Clone, Copy, Default)]
pub(crate) struct Writes {
    series: u64,
    revisions: [u64; 2],
}

#[derive(Default)]
pub(crate) struct Writers {
    revisions: std::collections::HashMap<usize, u64>,
    unknown: u64,
    series: u64,
    watched: [usize; 2],
}

impl Writers {
    pub(crate) fn record(&mut self, buffer: &ID3D12Resource, uav: bool) {
        let id = buffer.as_raw() as usize;
        if uav {
            *self.revisions.entry(id).or_default() += 1;
        }
        if self.watched.contains(&id) {
            self.series += 1;
        }
    }

    pub(crate) fn unknown(&mut self) {
        self.unknown += 1;
        self.series += 1;
    }

    pub(crate) fn before(
        &mut self,
        args: &ID3D12Resource,
        count: Option<&ID3D12Resource>,
    ) -> Writes {
        let watched = [
            args.as_raw() as usize,
            count.map_or(0, |b| b.as_raw() as usize),
        ];
        if self.watched != watched {
            self.watched = watched;
            self.series += 1;
        }
        Writes {
            series: self.series,
            revisions: watched
                .map(|id| self.unknown + self.revisions.get(&id).copied().unwrap_or(0)),
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct Inputs {
    args: u64,
    count: u64,
    stats: u64,
    series: u64,
    take_rest: bool,
}

struct Run {
    inputs: Inputs,
    table: usize,
    len: usize,
}

impl Run {
    fn append(&mut self, row: [u32; 8]) {
        // SAFETY: zone d’upload privée de la liste, remplie pendant l’enregistrement.
        // Une remise à zéro conserve la table de toute soumission en vol.
        unsafe {
            let table = self.table as *mut u32;
            table
                .add(4 + 8 * self.len)
                .copy_from_nonoverlapping(row.as_ptr(), 8);
            self.len += 1;
            table.write(self.len as u32);
        }
    }
}

struct Page {
    buffer: ID3D12Resource,
    used: u32,
    bytes: u32,
    indirect: bool,
}

impl Page {
    fn state(&mut self, list: &HostList, indirect: bool) {
        if self.indirect == indirect {
            return;
        }
        let states = [
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        ];
        // SAFETY: la liste conserve la page jusqu’à la fin de sa soumission.
        unsafe {
            list.list.ResourceBarrier(&[transition(
                &self.buffer,
                states[self.indirect as usize],
                states[indirect as usize],
            )])
        };
        self.indirect = indirect;
    }
}

#[derive(Default)]
pub(crate) struct Runs {
    open: Option<Run>,
    page: Option<Page>,
    ordered: std::collections::HashMap<usize, u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Output {
    /// Sortie lue via UAV ou copiée pour un prédicat GPU.
    Uav,
    Draw,
    Expanded,
    Native,
    NativePrefix,
}

impl Output {
    pub(crate) fn dispatch(has_base: bool) -> Self {
        if expansion_control() {
            Self::Expanded
        } else if has_base {
            Self::NativePrefix
        } else {
            Self::Native
        }
    }
}

// Témoin de mesure : mêmes séries et même budget, tous les dispatches au
// format de cxhair (base + dimensions), dans sa mémoire système mappée.
fn expansion_control() -> bool {
    #[cfg(feature = "diagnostics")]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("XTP_INDIRECT_EXPANSION_CONTROL").is_ok_and(|s| s == "1"))
    }
    #[cfg(not(feature = "diagnostics"))]
    {
        false
    }
}

pub(crate) struct Bounded {
    pub(crate) buffer: ID3D12Resource,
    pub(crate) offset: u64,
    pub(crate) capacity: u32,
    pub(crate) dispatched: bool,
    pub(crate) native_offset: Option<u64>,
}

/// Une ressource peut dépasser la fenêtre de 4 Gio d’une vue HLSL brute.
fn window(buffer: &ID3D12Resource, offset: u64, bytes: u32) -> Option<(u64, u32)> {
    // SAFETY: la ressource est vivante et ses dimensions sont immuables.
    let (gpu, width) = unsafe { (buffer.GetGPUVirtualAddress(), buffer.GetDesc().Width) };
    if !offset.is_multiple_of(4) || offset.checked_add(u64::from(bytes))? > width {
        return None;
    }
    let base = if offset + u64::from(bytes) > u64::from(u32::MAX) {
        offset
    } else {
        0
    };
    Some((gpu + base, (offset - base) as u32))
}

/// Une seule passe prépare les commandes d’une série. Les arguments et le compteur
/// sont lus ensemble, et chaque exécution reçoit une tranche de sortie distincte.
/// Une écriture intermédiaire sur l’une des deux entrées ouvre une nouvelle série.
pub(crate) fn bound(
    list: &HostList,
    args: (&ID3D12Resource, u64),
    spec: (u32, u32, u32),
    count: Option<(&ID3D12Resource, u64)>,
    stats: Option<(u64, u32)>,
    writes: Writes,
    output: Output,
) -> Option<Bounded> {
    bound_with(list, args, spec, count, stats, (writes, output), rest())
}

fn bound_with(
    list: &HostList,
    (buffer, offset): (&ID3D12Resource, u64),
    (kind, stride, records): (u32, u32, u32),
    count: Option<(&ID3D12Resource, u64)>,
    stats: Option<(u64, u32)>,
    (writes, output): (Writes, Output),
    take_rest: bool,
) -> Option<Bounded> {
    let min_stride = match kind {
        0 => 16,
        1 => 20,
        2 | 10 => 12,
        _ => return None,
    };
    if stride < min_stride || !stride.is_multiple_of(4) || records == 0 {
        return None;
    }
    let (args_base, args_at) = window(buffer, offset, stride.checked_mul(records)?)?;
    let (count_base, count_at) = match count {
        Some((b, at)) => window(b, at, 4)?,
        None => (args_base, 0),
    };
    let (stats_base, stats_at) = stats.unwrap_or((0, 0));
    let inputs = Inputs {
        args: args_base,
        count: count_base,
        stats: stats_base,
        series: writes.series,
        take_rest,
    };
    let capacity = if kind <= 1 {
        records
    } else {
        records.saturating_add(EXTRA).min(MAX_GROUPS)
    };
    // Douze octets en fin de tranche pour la signature qui saute la base XYZ.
    let stride_out = if output == Output::Native {
        12
    } else {
        RECORD_BYTES
    };
    let base_bytes = capacity.checked_mul(stride_out)?.checked_add(16)?;
    let native_bytes = if output == Output::NativePrefix {
        records.min(MAX_GROUPS).checked_mul(12)?.checked_add(4)?
    } else {
        0
    };
    let bytes = base_bytes.checked_add(native_bytes)?;
    let mut runs = list.clamps.lock().unwrap_or_else(|e| e.into_inner());
    let join = runs
        .open
        .as_ref()
        .is_some_and(|r| r.inputs == inputs && r.len < RUN_LENGTH)
        && runs
            .page
            .as_ref()
            .is_some_and(|p| bytes <= p.bytes - p.used);
    let at = if join { runs.page.as_ref()?.used } else { 0 };
    let flags = u32::from(count.is_some())
        | (u32::from(stats.is_some()) << 1)
        | (u32::from(output == Output::Native) << 2)
        | (u32::from(output == Output::NativePrefix) << 3);
    let row = [
        kind, stride, records, flags, args_at, count_at, at, stats_at,
    ];
    let _timed;
    let dispatched = match &mut runs.open {
        Some(run) if join => {
            run.append(row);
            false
        }
        _ => {
            let c = clamp_for(take_rest)?;
            let budget = list.budget()?;
            let (table, data) = list.scratch(16 + RUN_LENGTH as u64 * 32)?;
            let mut run = Run {
                inputs,
                table: data as usize,
                len: 0,
            };
            run.append(row);
            let new_page = runs.page.as_ref().is_none_or(|p| bytes > p.bytes);
            if new_page {
                let bytes = bytes.max(PAGE_BYTES);
                let buffer = if expansion_control() {
                    host()?.shared_buffer(bytes.into())?.0
                } else {
                    crate::host::buffers::card_buffer(&host()?.device, bytes.into())?
                };
                list.keep(&buffer);
                runs.page = Some(Page {
                    buffer,
                    used: 0,
                    bytes,
                    indirect: false,
                });
            }
            _timed = crate::timestamps::command(list, crate::timestamps::What::Clamp, None);
            let mut barriers = Vec::new();
            // La première lecture ordonne aussi les écritures des listes précédentes.
            for (b, revision) in [
                (Some(buffer), writes.revisions[0]),
                (count.map(|(b, _)| b), writes.revisions[1]),
            ] {
                if let Some(b) = b
                    && runs.ordered.insert(b.as_raw() as usize, revision) != Some(revision)
                {
                    barriers.push(uav_barrier(b));
                }
            }
            if runs.open.is_some() {
                barriers.push(uav_barrier(&budget));
            }
            let page = runs.page.as_mut()?;
            if !new_page && !page.indirect {
                // La dernière expansion de rectangles a lu cette page en UAV.
                barriers.push(uav_barrier(&page.buffer));
            }
            page.state(list, false);
            let l = &list.list;
            // SAFETY: les entrées, la table et la sortie sont gardées par l’enregistrement.
            unsafe {
                if !barriers.is_empty() {
                    l.ResourceBarrier(&barriers);
                }
                let budget = budget.GetGPUVirtualAddress();
                l.SetComputeRootSignature(&c.root);
                l.SetPipelineState(&c.pso);
                l.SetComputeRootShaderResourceView(0, table);
                l.SetComputeRootUnorderedAccessView(1, args_base);
                l.SetComputeRootUnorderedAccessView(2, budget);
                l.SetComputeRootUnorderedAccessView(3, page.buffer.GetGPUVirtualAddress());
                l.SetComputeRootUnorderedAccessView(4, count_base);
                l.SetComputeRootUnorderedAccessView(5, stats.map_or(budget, |s| s.0));
                l.Dispatch(1, 1, 1);
            }
            runs.open = Some(run);
            true
        }
    };
    let page = runs.page.as_mut()?;
    page.used = at + bytes;
    page.state(list, output != Output::Uav);
    if output == Output::Uav && dispatched {
        // SAFETY: la passe des rectangles lit la sortie via une UAV.
        unsafe { list.list.ResourceBarrier(&[uav_barrier(&page.buffer)]) };
    }
    Some(Bounded {
        buffer: page.buffer.clone(),
        offset: at.into(),
        capacity,
        dispatched,
        native_offset: (output == Output::NativePrefix)
            .then(|| u64::from(at) + u64::from(base_bytes)),
    })
}

/// La signature des morceaux installe trois constantes avant la commande.
/// Pour un programme sans lecture du groupe, on saute ces douze octets.
pub(crate) fn signature(
    kind: u32,
    root: &ID3D12RootSignature,
    base: Option<(u32, u32)>,
) -> Option<ID3D12CommandSignature> {
    let mut args = [D3D12_INDIRECT_ARGUMENT_DESC::default(); 2];
    if let Some((param, offset)) = base {
        args[0].Type = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT;
        args[0].Anonymous.Constant = D3D12_INDIRECT_ARGUMENT_DESC_0_1 {
            RootParameterIndex: param,
            DestOffsetIn32BitValues: offset,
            Num32BitValuesToSet: 3,
        };
    }
    args[1].Type = D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32);
    let args = &args[usize::from(base.is_none())..];
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: RECORD_BYTES,
        NumArgumentDescs: args.len() as u32,
        pArgumentDescs: args.as_ptr(),
        NodeMask: 0,
    };
    let mut signature = None;
    // SAFETY: les descriptions et la racine restent valables pendant la création.
    unsafe {
        host()?
            .device
            .CreateCommandSignature(&desc, base.map(|_| root), &mut signature)
    }
    .ok()?;
    signature
}

/// Limite le travail avant de parcourir les morceaux sur le CPU. La multiplication
/// protégée conserve les trois axes du calcul et linéarise ceux du mesh sur X.
fn direct_groups(groups: [u32; 3], mesh: bool) -> Option<[u32; 3]> {
    if groups.contains(&0) {
        return Some([0, 1, 1]);
    }
    let [x, y, z] = groups;
    if x > MAX_GROUPS / y / z {
        return None;
    }
    Some(if mesh { [x * y * z, 1, 1] } else { groups })
}

pub(crate) fn direct(
    list: &HostList,
    groups: [u32; 3],
    base: Option<(u32, u32)>,
    mesh: bool,
    mut run: impl FnMut([u32; 3]),
) {
    let Some(groups) = direct_groups(groups, mesh) else {
        return;
    };
    let extent = groups.map(|n| n.min(LIMIT));
    if groups.contains(&0) {
        run(extent);
        return;
    }
    if groups == extent {
        run(groups);
        return;
    }
    for z in (0..groups[2]).step_by(extent[2] as usize) {
        for y in (0..groups[1]).step_by(extent[1] as usize) {
            for x in (0..groups[0]).step_by(extent[0] as usize) {
                let origin = [x, y, z];
                if let Some((param, offset)) = base {
                    // SAFETY: trois dwords réservés dans la racine du programme.
                    unsafe {
                        if mesh {
                            list.list.SetGraphicsRoot32BitConstants(
                                param,
                                3,
                                origin.as_ptr().cast(),
                                offset,
                            );
                        } else {
                            list.list.SetComputeRoot32BitConstants(
                                param,
                                3,
                                origin.as_ptr().cast(),
                                offset,
                            );
                        }
                    }
                }
                run(std::array::from_fn(|i| {
                    (groups[i] - origin[i]).min(extent[i])
                }));
            }
        }
    }
}

fn uav_barrier(buffer: &ID3D12Resource) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                // SAFETY: la barrière emprunte la ressource pour ResourceBarrier.
                pResource: unsafe { std::mem::transmute_copy(buffer) },
            }),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testgpu::{TitleBuffer, TitleList};

    fn read_commands(list: &HostList, commands: &Bounded, bytes: u64) -> (ID3D12Resource, *mut u8) {
        let (result, data) = host().unwrap().shared_buffer(bytes).unwrap();
        let source = D3D12_RESOURCE_STATE_COPY_SOURCE;
        let indirect = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
        // SAFETY: les deux plages vivent jusqu’au retour de drain().
        unsafe {
            list.list
                .ResourceBarrier(&[transition(&commands.buffer, indirect, source)]);
            list.list
                .CopyBufferRegion(&result, 0, &commands.buffer, commands.offset, bytes);
            list.list
                .ResourceBarrier(&[transition(&commands.buffer, source, indirect)]);
        }
        (result, data)
    }

    /// Both clamp variants compile with the system's compiler and the host takes them.
    #[test]
    fn argument_clamp_builds() {
        let test = "indirect::tests::argument_clamp_builds";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        assert!(super::clamp_for(false).is_some());
        assert!(super::clamp_for(true).is_some());
    }

    #[test]
    fn rest_switch_only_enables_on_one() {
        use std::ffi::OsStr;
        assert!(!rest_from(None));
        assert!(!rest_from(Some(OsStr::new(""))));
        assert!(!rest_from(Some(OsStr::new("0"))));
        assert!(!rest_from(Some(OsStr::new("true"))));
        assert!(rest_from(Some(OsStr::new("1"))));
    }

    /// Clamps `records` records of `kind`, filled by repeating `words`,
    /// `calls` times in one list: each record's first word after (zero when
    /// it ran empty). One more record follows them and must stay as
    /// written. `None` without a host GPU (`test`'s skip is reported).
    fn clamped(test: &str, kind: u32, words: &[u32], records: u32, calls: u32) -> Option<Vec<u32>> {
        clamped_mode(test, kind, words, records, calls, false)
    }

    fn clamped_mode(
        test: &str,
        kind: u32,
        words: &[u32],
        records: u32,
        calls: u32,
        take_rest: bool,
    ) -> Option<Vec<u32>> {
        let gpu = crate::testgpu::device(test)?;
        let host = gpu.host();
        // Words in a record of `kind`.
        let per: usize = match kind {
            0 => 4,
            1 => 5,
            _ => 3,
        };
        let spec = (kind, per as u32 * 4, records);
        let all = per * (records as usize + 1);
        let (buffer, data) = host.shared_buffer(all as u64 * 4).expect("a shared buffer");
        let data = data.cast::<u32>();
        let list = host.new_list().expect("a host list");
        // SAFETY: our mapped buffer, `all` words, written before the list
        // runs; a live buffer the test holds until the list has run.
        unsafe {
            for at in 0..all {
                data.add(at).write_volatile(words[at % words.len()]);
            }
        }
        let mut commands = None;
        for _ in 0..calls {
            commands = super::bound_with(
                &list,
                (&buffer, 0),
                spec,
                None,
                None,
                (Writes::default(), Output::Draw),
                take_rest,
            );
            assert!(commands.is_some());
        }
        let (_out, result) = read_commands(&list, &commands.unwrap(), 4 + u64::from(records) * 24);
        list.close();
        host.execute(&[&list]);
        host.drain();
        // SAFETY: as above, read once the list has run.
        unsafe {
            for at in all - per..all {
                assert_eq!(
                    data.add(at).read_volatile(),
                    words[at % words.len()],
                    "the word {at} past the records"
                );
            }
            Some(
                (0..records as usize)
                    .map(|i| result.cast::<u32>().add(1 + i * 6).read_volatile())
                    .collect(),
            )
        }
    }

    /// Sur 128 dessins coûtant chacun 2^26 instances de sommet, seuls deux passent.
    #[test]
    fn rejected_records_do_not_wrap_the_budget() {
        let test = "indirect::tests::rejected_records_do_not_wrap_the_budget";
        let Some(draws) = clamped(test, 0, &[1 << 24, 4, 0, 0], 128, 1) else {
            return;
        };
        assert_eq!(draws.iter().filter(|&&count| count != 0).count(), 2);
    }

    /// Chaque dispatch mesh de 70 000 groupes consomme deux commandes et son coût entier.
    /// Le compte GPU exclut la dernière ligne. Une dimension corrompue ne déborde pas.
    #[test]
    fn mesh_pieces_share_the_original_budget() {
        let test = "indirect::tests::mesh_pieces_share_the_original_budget";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let list = host.new_list().unwrap();
        let (args, data) = host.shared_buffer(16 * 18).unwrap();
        let (stats, counts) = host.shared_buffer(64).unwrap();
        // SAFETY: les tampons mappés appartiennent au test jusqu’à drain().
        unsafe {
            let words = std::slice::from_raw_parts_mut(data.cast::<u32>(), 18 * 4);
            for record in words.as_chunks_mut::<4>().0 {
                record.copy_from_slice(&[7000, 2, 5, 0]);
            }
            words[..3].fill(u32::MAX);
            words[3] = 17;
            // An active empty record and one beyond the count. Only the first is measured.
            words[4 * 16 + 1] = 0;
            words[4 * 17 + 1] = 0;
            let commands = super::bound(
                &list,
                (&args, 0),
                (10, 16, 18),
                Some((&args, 12)),
                Some((stats.GetGPUVirtualAddress(), 0)),
                Writes::default(),
                Output::Expanded,
            )
            .unwrap();
            let (_out, written) = read_commands(&list, &commands, 4 + 24 * 28);
            list.close();
            host.execute(&[&list]);
            host.drain();
            let count = written.cast::<u32>().read_volatile();
            assert_eq!(count, 28);
            let pieces =
                std::slice::from_raw_parts(written.cast::<u32>().add(1), count as usize * 6);
            for &pair in pieces.as_chunks::<12>().0 {
                assert_eq!(pair, [0, 0, 0, 65535, 1, 1, 65535, 0, 0, 4465, 1, 1]);
            }
            let metrics = std::slice::from_raw_parts(counts.cast::<u32>(), 8);
            assert_eq!(metrics, [u32::MAX, u32::MAX, u32::MAX, 1, 0, 1, 1, 17]);
            assert_eq!(counts.cast::<u32>().add(14).read_volatile(), 1);
            assert_eq!(counts.cast::<u32>().add(15).read_volatile(), 14);
            assert_eq!(words[4], 7000);
        }
    }

    /// The budget goes to the first records, in order: of 200 draws of
    /// 3 * 2^18 vertices the first 170 run (it ends inside the third tile);
    /// small records that all fit all run.
    #[test]
    fn the_budget_goes_to_the_first_records() {
        let test = "indirect::tests::the_budget_goes_to_the_first_records";
        let Some(draws) = clamped(test, 0, &[3 << 18, 1, 0, 0], 200, 1) else {
            return;
        };
        assert!(draws[..170].iter().all(|&count| count != 0));
        assert!(draws[170..].iter().all(|&count| count == 0));
        let small = clamped(test, 0, &[1], 1 << 16, 1).expect("a host GPU");
        assert!(small.iter().all(|&count| count == 1));
    }

    /// By default a record that does not fit runs empty and leaves the
    /// remaining budget for later records that fit; with `XTP_INDIRECT_REST=1`
    /// it spends the rest so later records run empty too.
    #[test]
    fn a_record_that_does_not_fit_leaves_the_budget_by_default() {
        let test = "indirect::tests::a_record_that_does_not_fit_leaves_the_budget_by_default";
        let mut words = [1 << 24, 3, 0, 0].repeat(3);
        words.extend([3, 1, 0, 0].repeat(6));
        let Some(draws) = clamped(test, 0, &words, 8, 1) else {
            return;
        };
        let ran: Vec<bool> = draws.iter().map(|&count| count != 0).collect();
        assert_eq!(ran, [true, true, false, true, true, true, true, true]);
        let rest_draws = clamped_mode(test, 0, &words, 8, 1, true).expect("a host GPU");
        let rest_ran: Vec<bool> = rest_draws.iter().map(|&count| count != 0).collect();
        assert_eq!(
            rest_ran,
            [true, true, false, false, false, false, false, false]
        );
    }

    /// One list of indirect compute dispatches where an early record exceeds
    /// the remaining budget and later small records fit: by default the later
    /// records run (writing their marker), budget rejections are still counted,
    /// and a later >65,535-group record still chunks; with `XTP_INDIRECT_REST=1`
    /// every later record runs empty.
    #[test]
    fn oversized_indirect_dispatch_leaves_budget_for_later_record() {
        const MARKER_CS: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer marker : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint prev;
    marker.InterlockedAdd(0, 1, prev);
}
"#;
        let test = "indirect::tests::oversized_indirect_dispatch_leaves_budget_for_later_record";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let worker =
            compile_compute(&host.device, MARKER_CS).expect("marker compute shader must compile");
        let signature = crate::draw::host_signature(2, 12).expect("native dispatch signature");

        // Six 12-byte dispatch records:
        // 0: 65,535 * 24 = 1,572,840 groups (100,661,760 threads), leaving 33,555,968 threads.
        // 1: 40,000 * 14 = 560,000 groups (35,840,000 threads) -> over remaining budget.
        // 2: (1, 1, 1) -> 64 threads -> fits and writes +1 to marker.
        // 3: 40,000 * 14 -> second over-budget record -> counted in past_budget.
        // 4: (2, 1, 1) -> 128 threads -> fits and writes +2 to marker.
        // 5: (70,000, 1, 1) -> 4,480,000 threads -> fits and splits into 65,535 + 4,465.
        let records: [[u32; 3]; 6] = [
            [65_535, 24, 1],
            [40_000, 14, 1],
            [1, 1, 1],
            [40_000, 14, 1],
            [2, 1, 1],
            [70_000, 1, 1],
        ];
        let initial_cost = 65_535 * 24 * 64;

        for take_rest in [false, true] {
            let list = host.new_list().expect("a host list");
            let (args, args_data) = host.shared_buffer(6 * 12).expect("args buffer");
            let (stats, stats_data) = host.shared_buffer(3 * 64).expect("stats buffer");
            let (marker, marker_data) = host.shared_buffer(4).expect("marker buffer");
            // SAFETY: mapped buffers owned by this test until after `host.drain()`.
            let (stats_gpu, marker_gpu) = unsafe {
                std::ptr::copy_nonoverlapping(records.as_ptr().cast::<u8>(), args_data, 6 * 12);
                std::ptr::write_bytes(stats_data, 0, 3 * 64);
                marker_data.cast::<u32>().write_volatile(0);
                (stats.GetGPUVirtualAddress(), marker.GetGPUVirtualAddress())
            };

            let first = bound_with(
                &list,
                (&args, 0),
                (2, 12, 1),
                None,
                Some((stats_gpu, 0)),
                (Writes::default(), Output::Native),
                take_rest,
            )
            .expect("initial budget spend bounded");
            let second = bound_with(
                &list,
                (&args, 12),
                (2, 12, 4),
                None,
                Some((stats_gpu, 64)),
                (Writes::default(), Output::Native),
                take_rest,
            )
            .expect("mixed records bounded");
            let third = bound_with(
                &list,
                (&args, 60),
                (2, 12, 1),
                None,
                Some((stats_gpu, 128)),
                (Writes::default(), Output::Expanded),
                take_rest,
            )
            .expect("chunked record bounded");
            assert!(first.dispatched && !second.dispatched && !third.dispatched);

            let (_chunked_buf, chunked_data) = read_commands(&list, &third, 4 + 2 * 24);
            list.keep(&signature);
            // SAFETY: open command list, live pipeline and buffers; `second.buffer` is
            // in `INDIRECT_ARGUMENT` and `marker` promotes from `COMMON` to `UAV`.
            unsafe {
                list.list.SetComputeRootSignature(&worker.root);
                list.list.SetPipelineState(&worker.pso);
                list.list.SetComputeRootUnorderedAccessView(0, marker_gpu);
                list.list.ExecuteIndirect(
                    &signature,
                    second.capacity,
                    &second.buffer,
                    second.offset + 4,
                    &second.buffer,
                    second.offset,
                );
            }

            list.close();
            host.execute(&[&list]);
            host.drain();

            // SAFETY: GPU execution finished; buffers are still live and mapped.
            unsafe {
                let hits = marker_data.cast::<u32>().read_volatile();
                let s = stats_data.cast::<u32>();
                let second_past_budget = s.add(16 + 6).read_volatile();
                let second_spent = s.add(16 + 13).read_volatile();
                let third_past_budget = s.add(32 + 6).read_volatile();
                let third_spent = s.add(32 + 13).read_volatile();
                let third_split = s.add(32 + 15).read_volatile();
                let chunk_count = chunked_data.cast::<u32>().read_volatile();
                if take_rest {
                    assert_eq!(
                        hits, 0,
                        "later dispatches must not run with XTP_INDIRECT_REST=1"
                    );
                    assert_eq!(second_past_budget, 4);
                    assert_eq!(second_spent, 1 << 27);
                    assert_eq!(third_past_budget, 1);
                    assert_eq!(third_spent, 1 << 27);
                    assert_eq!(third_split, 0);
                    assert_eq!(chunk_count, 0);
                } else {
                    assert_eq!(hits, 3, "later small dispatches must run by default");
                    assert_eq!(second_past_budget, 2);
                    assert_eq!(second_spent, initial_cost + 3 * 64);
                    assert_eq!(third_past_budget, 0);
                    assert_eq!(third_spent, initial_cost + 3 * 64 + 70_000 * 64);
                    assert_eq!(third_split, 1);
                    assert_eq!(chunk_count, 2);
                    let pieces = std::slice::from_raw_parts(chunked_data.cast::<u32>().add(1), 12);
                    assert_eq!(pieces, [0, 0, 0, 65_535, 1, 1, 65_535, 0, 0, 4_465, 1, 1]);
                }
            }
        }
    }

    /// Changing `take_rest` on the same list must start a new clamp run rather
    /// than joining an open run compiled with the opposite mode.
    #[test]
    fn switching_rest_mode_on_one_list_starts_a_new_run() {
        let test = "indirect::tests::switching_rest_mode_on_one_list_starts_a_new_run";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let list = host.new_list().expect("a host list");
        // Record 0 spends 2^26 (leaves 2^26); record 1 asks for 3 * 2^24 (fits);
        // record 2 asks for 2^26 (> 2^24 remaining -> rejected); record 3 asks for 5.
        let words: [u32; 16] = [
            1 << 24,
            4,
            0,
            0,
            1 << 24,
            3,
            0,
            0,
            1 << 24,
            4,
            0,
            0,
            5,
            1,
            0,
            0,
        ];
        let (args, data) = host.shared_buffer(16 * 4).expect("args buffer");
        // SAFETY: mapped buffer owned by this test until after `host.drain()`.
        unsafe {
            std::ptr::copy_nonoverlapping(words.as_ptr().cast::<u8>(), data, 16 * 4);
        }
        let first = bound_with(
            &list,
            (&args, 0),
            (0, 16, 2),
            None,
            None,
            (Writes::default(), Output::Draw),
            false,
        )
        .expect("first run bounded");
        let second = bound_with(
            &list,
            (&args, 32),
            (0, 16, 2),
            None,
            None,
            (Writes::default(), Output::Draw),
            true,
        )
        .expect("second run bounded");
        assert!(first.dispatched);
        assert!(
            second.dispatched,
            "different take_rest must not join an open run"
        );
        let (_out, result) = read_commands(&list, &second, 4 + 2 * 24);
        list.close();
        host.execute(&[&list]);
        host.drain();
        // SAFETY: GPU execution finished; buffer is still live and mapped.
        unsafe {
            let out = result.cast::<u32>();
            assert_eq!(out.add(1).read_volatile(), 0);
            assert_eq!(
                out.add(1 + 6).read_volatile(),
                0,
                "second run uses take_rest=true, so record 3 must run empty"
            );
        }
    }

    /// Remaining budget after an oversized record survives across separate
    /// clamp runs on the same list, and preserves `Output::NativePrefix`
    /// ordering when a >65,535-group record is rejected before a small one.
    #[test]
    fn oversized_record_preserves_budget_across_runs_and_native_prefix() {
        const MARKER_CS: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer marker : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint prev;
    marker.InterlockedAdd(0, 1, prev);
}
"#;
        let test =
            "indirect::tests::oversized_record_preserves_budget_across_runs_and_native_prefix";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let worker =
            compile_compute(&host.device, MARKER_CS).expect("marker compute shader must compile");
        let signature = crate::draw::host_signature(2, 12).expect("native dispatch signature");

        // Run 1 (args_a):
        //   0: (65_535, 24, 1) -> spends 100_661_760 threads, leaving 33_555_968.
        //   1: (600_000, 1, 1) -> 38_400_000 threads (> left) and > 65_535 groups -> rejected.
        // Run 2 (args_b, separate buffer -> forces a second clamp dispatch and budget handoff):
        //   0: (1, 1, 1)       -> fits, stays in NativePrefix because rejected record did not split.
        //   1: (70_000, 1, 1)  -> fits, splits into 2 chunks in expanded suffix.
        //   2: (2, 1, 1)       -> fits, follows split in expanded suffix to preserve order.
        let run1_records: [[u32; 3]; 2] = [[65_535, 24, 1], [600_000, 1, 1]];
        let run2_records: [[u32; 3]; 3] = [[1, 1, 1], [70_000, 1, 1], [2, 1, 1]];

        for take_rest in [false, true] {
            let list = host.new_list().expect("a host list");
            let (args_a, data_a) = host.shared_buffer(2 * 12).expect("args_a");
            let (args_b, data_b) = host.shared_buffer(3 * 12).expect("args_b");
            let (stats, stats_data) = host.shared_buffer(2 * 64).expect("stats");
            let (marker, marker_data) = host.shared_buffer(4).expect("marker");
            // SAFETY: mapped buffers owned by this test until after `host.drain()`.
            let (stats_gpu, marker_gpu) = unsafe {
                std::ptr::copy_nonoverlapping(run1_records.as_ptr().cast::<u8>(), data_a, 2 * 12);
                std::ptr::copy_nonoverlapping(run2_records.as_ptr().cast::<u8>(), data_b, 3 * 12);
                std::ptr::write_bytes(stats_data, 0, 2 * 64);
                marker_data.cast::<u32>().write_volatile(0);
                (stats.GetGPUVirtualAddress(), marker.GetGPUVirtualAddress())
            };

            let first = bound_with(
                &list,
                (&args_a, 0),
                (2, 12, 2),
                None,
                Some((stats_gpu, 0)),
                (Writes::default(), Output::NativePrefix),
                take_rest,
            )
            .expect("run 1 bounded");
            let second = bound_with(
                &list,
                (&args_b, 0),
                (2, 12, 3),
                None,
                Some((stats_gpu, 64)),
                (Writes::default(), Output::NativePrefix),
                take_rest,
            )
            .expect("run 2 bounded");
            assert!(first.dispatched && second.dispatched);

            let native_off = second.native_offset.expect("native prefix offset");
            let skip = (native_off - second.offset) as usize / 4;
            let (_read, prefix_data) = read_commands(&list, &second, (skip as u64 + 4) * 4);
            list.keep(&signature);
            // SAFETY: open list, live pipeline/buffers; `second.buffer` is in `INDIRECT_ARGUMENT`.
            unsafe {
                list.list.SetComputeRootSignature(&worker.root);
                list.list.SetPipelineState(&worker.pso);
                list.list.SetComputeRootUnorderedAccessView(0, marker_gpu);
                list.list.ExecuteIndirect(
                    &signature,
                    3,
                    &second.buffer,
                    native_off + 4,
                    &second.buffer,
                    native_off,
                );
            }

            list.close();
            host.execute(&[&list]);
            host.drain();

            // SAFETY: GPU execution finished; buffers are still live and mapped.
            unsafe {
                let hits = marker_data.cast::<u32>().read_volatile();
                let s = stats_data.cast::<u32>();
                let run1_past_budget = s.add(6).read_volatile();
                let run1_split = s.add(15).read_volatile();
                let run2_past_budget = s.add(16 + 6).read_volatile();
                let run2_split = s.add(16 + 15).read_volatile();
                let words = std::slice::from_raw_parts(prefix_data.cast::<u32>(), skip + 4);
                assert_eq!(run1_past_budget, 1);
                assert_eq!(
                    run1_split, 0,
                    "rejected >65,535 record must not count as split"
                );
                if take_rest {
                    assert_eq!(hits, 0);
                    assert_eq!(run2_past_budget, 3);
                    assert_eq!(run2_split, 0);
                    assert_eq!(words[skip], 0);
                    assert_eq!(words[0], 0);
                } else {
                    assert_eq!(hits, 1, "native prefix record in run 2 must execute");
                    assert_eq!(run2_past_budget, 0);
                    assert_eq!(run2_split, 1);
                    assert_eq!(&words[skip..], [1, 1, 1, 1]);
                    assert_eq!(
                        &words[..19],
                        [
                            3, 0, 0, 0, 65_535, 1, 1, 65_535, 0, 0, 4_465, 1, 1, 0, 0, 0, 2, 1, 1
                        ]
                    );
                }
            }
        }
    }

    /// Exact boundary (`cost == left` fits, `cost == left + 1` is rejected
    /// without poisoning), zero-cost records after `left == 0` are not counted
    /// as budget rejections, and remaining budget carries across 64-record tiles.
    #[test]
    fn exact_budget_boundary_and_multi_tile_records() {
        let test = "indirect::tests::exact_budget_boundary_and_multi_tile_records";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        // 70 draw records (spans 2 tiles of 64):
        //   0: (1 << 24, 4)     = 67_108_864 (leaves 67_108_864)
        //   1: (1 << 24, 3)     = 50_331_648 (leaves 16_777_216)
        //   2: (1, 1)           = 1          (leaves 16_777_215)
        //   3: (16_777_216, 1)  = left + 1   -> rejected
        //   4..=64: (0, 0) empty draws
        //   65 (tile 1): (16_777_215, 1) = left -> exact fit, leaves 0
        //   66 (tile 1): (1, 1) -> 1 > 0 -> rejected (past_budget)
        //   67 (tile 1): (0, 0) -> 0 > 0 is false -> NOT counted in past_budget
        //   68 (tile 1): (2, 1) -> 2 > 0 -> rejected (past_budget)
        //   69 (tile 1): (0, 0) -> empty
        let mut records = [[0u32; 4]; 70];
        records[0] = [1 << 24, 4, 0, 0];
        records[1] = [1 << 24, 3, 0, 0];
        records[2] = [1, 1, 0, 0];
        records[3] = [1 << 24, 1, 0, 0];
        records[65] = [(1 << 24) - 1, 1, 0, 0];
        records[66] = [1, 1, 0, 0];
        records[68] = [2, 1, 0, 0];

        for take_rest in [false, true] {
            let list = host.new_list().expect("a host list");
            let (args, data) = host.shared_buffer(70 * 16).expect("args");
            let (stats, stats_data) = host.shared_buffer(64).expect("stats");
            // SAFETY: mapped buffers owned by this test until after `host.drain()`.
            let stats_gpu = unsafe {
                std::ptr::copy_nonoverlapping(records.as_ptr().cast::<u8>(), data, 70 * 16);
                std::ptr::write_bytes(stats_data, 0, 64);
                stats.GetGPUVirtualAddress()
            };
            let bounded = bound_with(
                &list,
                (&args, 0),
                (0, 16, 70),
                None,
                Some((stats_gpu, 0)),
                (Writes::default(), Output::Draw),
                take_rest,
            )
            .expect("draws bounded");
            let (_out, result) = read_commands(&list, &bounded, 4 + 70 * 24);
            list.close();
            host.execute(&[&list]);
            host.drain();
            // SAFETY: GPU execution finished; buffers are still live and mapped.
            unsafe {
                let out = result.cast::<u32>();
                let vertices = |i: usize| out.add(1 + i * 6).read_volatile();
                let s = stats_data.cast::<u32>();
                let past_budget = s.add(6).read_volatile();
                let spent = s.add(13).read_volatile();
                assert_eq!(vertices(0), 1 << 24);
                assert_eq!(vertices(1), 1 << 24);
                assert_eq!(vertices(2), 1);
                assert_eq!(vertices(3), 0);
                assert_eq!(vertices(66), 0);
                assert_eq!(vertices(68), 0);
                assert_eq!(spent, 1 << 27);
                if take_rest {
                    assert_eq!(vertices(65), 0);
                    // Rejected: records 3, 65, 66, 68 (empty records 4..=64, 67, 69 are NOT counted).
                    assert_eq!(past_budget, 4);
                } else {
                    assert_eq!(vertices(65), (1 << 24) - 1);
                    // Rejected: records 3, 66, 68 (record 65 fit exactly; empties NOT counted).
                    assert_eq!(past_budget, 3);
                }
            }
        }
    }

    /// The budget is the list's: a second execution of the same records
    /// finds it spent, and they all run empty.
    #[test]
    fn a_spent_budget_stays_spent() {
        let test = "indirect::tests::a_spent_budget_stays_spent";
        let Some(twice) = clamped(test, 1, &[1 << 24, 4, 0, 0, 0], 64, 2) else {
            return;
        };
        assert!(twice.iter().all(|&count| count == 0));
    }
    #[test]
    fn one_dispatch_bounds_two_executions_in_budget_order() {
        let test = "indirect::tests::one_dispatch_bounds_two_executions_in_budget_order";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let list = host.new_list().unwrap();
        let (args, data) = host.shared_buffer(48).unwrap();
        // SAFETY: trois dessins dans notre tampon mappé, initialisés avant soumission.
        unsafe {
            let words = std::slice::from_raw_parts_mut(data.cast::<u32>(), 12);
            words.copy_from_slice(&[1 << 24, 4, 3, 8, 1 << 24, 4, 1, 7, 3, 1, 9, 4]);
        }
        let first = bound(
            &list,
            (&args, 0),
            (0, 16, 2),
            None,
            None,
            Writes::default(),
            Output::Draw,
        )
        .unwrap();
        let second = bound(
            &list,
            (&args, 32),
            (0, 16, 1),
            None,
            None,
            Writes::default(),
            Output::Draw,
        )
        .unwrap();
        assert!(first.dispatched && !second.dispatched);
        let (_one, a) = read_commands(&list, &first, 52);
        let (_two, b) = read_commands(&list, &second, 20);
        list.close();
        host.execute(&[&list]);
        host.drain();
        // SAFETY: la copie est terminée, les deux tampons restent mappés.
        unsafe {
            assert_eq!(*a.cast::<[u32; 5]>(), [2, 1 << 24, 4, 3, 8]);
            assert_eq!(*b.cast::<[u32; 5]>(), [1, 0, 0, 9, 4]);
        }
    }

    #[test]
    fn a_gpu_write_to_the_count_ends_the_run() {
        let test = "indirect::tests::a_gpu_write_to_the_count_ends_the_run";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let _host = gpu.host();
        let title = TitleList::new();
        let list = title.host();
        let args = TitleBuffer::new(256);
        let count = TitleBuffer::new(256);
        args.set_words(|_| 1);
        count.set_words(|_| 1);
        let (a, c) = (args.shadow(), count.shadow());
        a.refresh(list, args.address(), 16);
        c.refresh(list, count.address(), 4);
        let run = || {
            let writes = list.indirect_writes(&a.buffer, Some(&c.buffer));
            bound(
                list,
                (&a.buffer, a.offset(args.address())),
                (0, 16, 1),
                Some((&c.buffer, c.offset(count.address()))),
                None,
                writes,
                Output::Draw,
            )
            .unwrap()
        };
        let first = run();
        let (_before, one) = read_commands(list, &first, 4);
        assert!(crate::shadow::write(list, count.address() + 4, 7, 4));
        assert!(run().dispatched, "autre mot du même tampon surveillé");
        assert!(crate::shadow::write(list, count.address(), 0, 4));
        let next = run();
        assert!(first.dispatched && next.dispatched);
        let (_after, zero) = read_commands(list, &next, 4);
        title.run();
        // SAFETY: la liste et les deux copies de leur compteur ont terminé.
        unsafe {
            assert_eq!(one.cast::<u32>().read_volatile(), 1);
            assert_eq!(zero.cast::<u32>().read_volatile(), 0);
        }
    }

    #[test]
    fn a_dropped_execution_records_no_clamp() {
        let test = "indirect::tests::a_dropped_execution_records_no_clamp";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let _host = gpu.host();
        let list = TitleList::new();
        let args = TitleBuffer::new(256);
        args.set_words(|_| u32::MAX);
        let mut state = crate::draw::State::default();
        assert!(!state.execute_indirect(list.host(), (10, 12, 1), args.address(), None));
        assert!(list.host().clamps.lock().unwrap().open.is_none());
        list.run();
    }

    #[test]
    fn native_prefix_precedes_the_split_and_its_suffix() {
        let test = "indirect::tests::native_prefix_precedes_the_split_and_its_suffix";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let list = host.new_list().unwrap();
        let (args, data) = host.shared_buffer(36).unwrap();
        // SAFETY: trois dispatches du test, lus après leur initialisation.
        unsafe {
            data.cast::<[u32; 9]>()
                .write([5, 2, 1, 70000, 1, 1, 2, 3, 4]);
        }
        let prefix = bound(
            &list,
            (&args, 0),
            (2, 12, 3),
            None,
            None,
            Writes::default(),
            Output::NativePrefix,
        )
        .unwrap();
        let skip = (prefix.native_offset.unwrap() - prefix.offset) as usize / 4;
        let (_read, data) = read_commands(&list, &prefix, (skip as u64 + 4) * 4);
        let compact = bound(
            &list,
            (&args, 0),
            (2, 12, 3),
            None,
            None,
            Writes::default(),
            Output::Native,
        )
        .unwrap();
        let (_compact, packed) = read_commands(&list, &compact, 4 + 4 * 12);
        list.close();
        host.execute(&[&list]);
        host.drain();
        // SAFETY: les deux tranches copiées sont mappées et la soumission est terminée.
        unsafe {
            let words = std::slice::from_raw_parts(data.cast::<u32>(), skip + 4);
            assert_eq!(&words[skip..], [1, 5, 2, 1]);
            assert_eq!(
                &words[..19],
                [
                    3, 0, 0, 0, 65535, 1, 1, 65535, 0, 0, 4465, 1, 1, 0, 0, 0, 2, 3, 4
                ]
            );
            assert_eq!(
                *packed.cast::<[u32; 13]>(),
                [4, 5, 2, 1, 65535, 1, 1, 4465, 1, 1, 2, 3, 4]
            );
        }
    }

    #[test]
    fn direct_grids_are_bounded_before_splitting() {
        assert_eq!(direct_groups([u32::MAX; 3], false), None);
        assert_eq!(
            direct_groups([MAX_GROUPS, 1, 1], false),
            Some([MAX_GROUPS, 1, 1])
        );
        assert_eq!(direct_groups([1, MAX_GROUPS + 1, 1], true), None);
        assert_eq!(direct_groups([1000, 2, 3], true), Some([6000, 1, 1]));
        assert_eq!(direct_groups([1000, 2, 3], false), Some([1000, 2, 3]));
    }

    #[test]
    fn corrupt_draws_run_empty() {
        let test = "indirect::tests::corrupt_draws_run_empty";
        let Some(out) = clamped(test, 0, &[u32::MAX, u32::MAX, 0, 0], 1, 1) else {
            return;
        };
        assert_eq!(out, [0]);
    }
}
