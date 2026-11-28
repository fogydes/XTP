//! Pipeline states. Both descriptions start as on PC: the root signature,
//! then the shader stages (five for graphics, one for compute); streams
//! carry the rest of the graphics state as subobjects of the standard
//! types.
//!
//! Host pipelines are made on a pipeline's first draw, not at creation:
//! BTP 6 creates some 60,000 pipelines in its first 20
//! seconds and draws with a few dozen (decision 0007, step 3). Shader
//! containers are shared by content hash and translated once, on first
//! use (decision 0003).
//!
//! shortcut: only vertex plus pixel pipelines from streams and compute
//! pipelines get host pipelines; mesh, tessellation and geometry stages,
//! and the plain graphics description, keep their containers only. The 32-byte
//! packet after the header stays zero: titles only copy it inline, which
//! our null command-list limits keep from running.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::HashMap;
use std::mem::offset_of;
use std::sync::{Arc, Mutex, OnceLock};

use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, PIPELINE, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

const PIPELINE_IIDS: [Iid; 4] = [
    guid(
        0x765a30f3,
        0xf624,
        0x4c6f,
        [0xa8, 0x28, 0xac, 0xe9, 0x48, 0x62, 0x24, 0x45],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
pub(crate) struct Bytecode {
    code: *const u8,
    size: usize,
}

#[repr(C)]
pub(crate) struct GraphicsHead {
    root: This,
    /// Vertex, pixel, domain, hull, geometry.
    stages: [Bytecode; 5],
}

#[repr(C)]
pub(crate) struct ComputeHead {
    root: This,
    stage: Bytecode,
}

/// A shader container a title passed, shared by every pipeline with the
/// same bytes.
pub(crate) struct Shader {
    pub(crate) stage: &'static str,
    /// The first 8 bytes of its BLAKE3, as logged.
    pub(crate) hash: String,
    #[cfg(windows)]
    pub(crate) ambient: Option<crate::ambient::Program>,
    container: Vec<u8>,
    /// What its console program says of itself, read on first use without
    /// translating it.
    #[cfg(windows)]
    native: OnceLock<Result<Native, String>>,
    #[cfg(windows)]
    translated: OnceLock<Option<Translated>>,
}

impl std::fmt::Debug for Shader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.stage, self.hash)
    }
}

/// Every container seen, by full content hash.
static SHADERS: Mutex<Option<HashMap<[u8; 32], Arc<Shader>>>> = Mutex::new(None);

/// The shared container for `bytes`.
fn shader(stage: &'static str, bytes: &[u8]) -> Arc<Shader> {
    let full = blake3::hash(bytes);
    let mut shaders = SHADERS.lock().unwrap_or_else(|e| e.into_inner());
    shaders
        .get_or_insert_with(HashMap::new)
        .entry(*full.as_bytes())
        .or_insert_with(|| {
            Arc::new(Shader {
                stage,
                hash: full.to_hex()[..16].to_owned(),
                #[cfg(windows)]
                ambient: if stage == "compute" {
                    crate::ambient::program(full.as_bytes(), bytes)
                } else {
                    None
                },
                container: bytes.to_vec(),
                #[cfg(windows)]
                native: OnceLock::new(),
                #[cfg(windows)]
                translated: OnceLock::new(),
            })
        })
        .clone()
}

/// A translated shader and the console resources its code reads.
#[cfg(windows)]
pub(crate) struct Translated {
    pub(crate) host: xtp_shader::emit::HostShader,
    pub(crate) used: Vec<xtp_shader::UsedResource>,
    /// Whether the console program runs in 64-lane waves (its lane masks
    /// and lane numbers are a wave64's). Of a mesh program, the translator
    /// says whether it runs on the host's waves (`host::Needs`).
    pub(crate) wave64: bool,
}

/// What a console program says of itself: what [`Translated`] holds besides
/// the host shader, read from the container without translating it.
#[cfg(windows)]
#[derive(Clone)]
pub(crate) struct Native {
    pub(crate) used: Vec<xtp_shader::UsedResource>,
    pub(crate) wave64: bool,
    /// A compute program's thread group (`numthreads`), as the console
    /// runs it: `DispatchX` turns thread counts into groups with it.
    pub(crate) threads: Option<[u32; 3]>,
}

#[cfg(windows)]
impl Shader {
    /// What its console program says of itself (cheap next to a
    /// translation: no structuring), read once; an error is the
    /// translation's error too.
    pub(crate) fn native(&self) -> Result<&Native, &str> {
        self.native
            .get_or_init(|| native(&self.container))
            .as_ref()
            .map_err(String::as_str)
    }

    /// The translation, made on first use; a shader that does not
    /// translate is logged once. Both say how long the translation took
    /// (or the cache took to answer): a pipeline worker's time.
    pub(crate) fn translated(&self) -> Option<&Translated> {
        self.translated
            .get_or_init(|| {
                let started = std::time::Instant::now();
                let done = translate(
                    &self.container,
                    self.native().map_err(str::to_owned),
                    self_sized_options(&self.hash),
                );
                let ms = started.elapsed().as_millis() as u64;
                match &done {
                    Ok(t) => tracing::info!(
                        stage = self.stage,
                        hash = %self.hash,
                        user_data = t.host.user_data.len(),
                        samplers = t.host.samplers.len(),
                        ms,
                        "shader translated"
                    ),
                    Err(e) => tracing::warn!(
                        stage = self.stage,
                        hash = %self.hash,
                        error = %e,
                        ms,
                        "shader does not translate; its pipelines' draws are dropped"
                    ),
                }
                done.ok()
            })
            .as_ref()
    }

    /// Where a dispatch of its program, spliced with an inline ray query,
    /// finds the title's descriptor of the structure the walk read
    /// (`xtp_shader::emit::tlas_descriptor`): only the few trace programs
    /// whose translation reads the host TLAS ask, once per dispatch.
    pub(crate) fn tlas_descriptor(&self) -> Option<xtp_shader::emit::TlasDescriptor> {
        xtp_shader::emit::container_tlas_descriptor(&self.container)
    }
}

/// PCI vendor ID and RDNA 4 device IDs (Navi 48: RX 9070 XT, RX 9070, Radeon
/// AI PRO R9700; Navi 44: RX 9060 XT / RX 9060) affected by the `amdxc64.dll`
/// null-read compiler crash on self-sized scalar buffer loads.
#[cfg(windows)]
const AMD_VENDOR: u32 = 0x1002;
#[cfg(windows)]
const RDNA4_DEVICES: [u32; 3] = [0x7550, 0x7551, 0x7590];

#[cfg(windows)]
fn is_rdna4(ids: Option<(u32, u32)>) -> bool {
    ids.is_some_and(|(vendor, device)| vendor == AMD_VENDOR && RDNA4_DEVICES.contains(&device))
}

#[cfg(windows)]
enum SelfSizedMode {
    Off,
    All,
    Hashes(Vec<String>),
}

#[cfg(windows)]
fn parse_self_sized_mode(ids: Option<(u32, u32)>, override_var: Option<&str>) -> SelfSizedMode {
    let Some(value) = override_var else {
        return if is_rdna4(ids) {
            SelfSizedMode::All
        } else {
            SelfSizedMode::Off
        };
    };
    let trimmed = value.trim();
    match trimmed {
        "" | "0" => SelfSizedMode::Off,
        _ if trimmed.eq_ignore_ascii_case("off") || trimmed.eq_ignore_ascii_case("none") => {
            SelfSizedMode::Off
        }
        "1" => SelfSizedMode::All,
        _ if trimmed.eq_ignore_ascii_case("all") => SelfSizedMode::All,
        _ if trimmed.eq_ignore_ascii_case("rdna4") => {
            if is_rdna4(ids) {
                SelfSizedMode::All
            } else {
                SelfSizedMode::Off
            }
        }
        _ => {
            let hashes: Vec<String> = trimmed
                .split(',')
                .map(str::trim)
                .filter(|h| h.len() == 16 && h.bytes().all(|b| b.is_ascii_hexdigit()))
                .map(str::to_ascii_lowercase)
                .collect();
            SelfSizedMode::Hashes(hashes)
        }
    }
}

#[cfg(windows)]
fn options_for_mode(mode: &SelfSizedMode, hash: &str) -> xtp_shader::emit::Options {
    let via_memory = match mode {
        SelfSizedMode::Off => false,
        SelfSizedMode::All => true,
        SelfSizedMode::Hashes(hashes) => hashes.iter().any(|h| h.eq_ignore_ascii_case(hash)),
    };
    xtp_shader::emit::Options {
        self_sized_loads_via_memory: via_memory,
    }
}

#[cfg(all(windows, test))]
fn self_sized_options_from(
    ids: Option<(u32, u32)>,
    override_var: Option<&str>,
    hash: &str,
) -> xtp_shader::emit::Options {
    options_for_mode(&parse_self_sized_mode(ids, override_var), hash)
}

/// Per-shader translation options for shader `hash`.
///
/// Unset `XTP_SELF_SIZED_LOADS` enables `self_sized_loads_via_memory` on AMD
/// RDNA 4 adapters (avoiding `amdxc64.dll` crashes without hardcoding title
/// hashes in platform crates); setting `XTP_SELF_SIZED_LOADS` to `0`/`off`,
/// `1`/`all`, `rdna4`, or a comma-separated list of 16-hex-digit hashes
/// overrides it on any adapter.
///
/// The env override is cached once, while RDNA 4 detection is only latched
/// once `crate::host::host()` has been initialized so early callers (such as
/// `precook` or pre-device setup) never latch `Off` prematurely.
// Ported and adapted from PastiiJay/klbf-lab commit f73233c8 (author aidzz / @BrokeAsh, GPL-3.0-or-later).
#[cfg(windows)]
pub(crate) fn self_sized_options(hash: &str) -> xtp_shader::emit::Options {
    static OVERRIDE: OnceLock<Option<String>> = OnceLock::new();
    static RDNA4: OnceLock<bool> = OnceLock::new();
    let override_var = OVERRIDE.get_or_init(|| std::env::var("XTP_SELF_SIZED_LOADS").ok());
    let ids = if override_var
        .as_deref()
        .is_none_or(|v| v.trim().eq_ignore_ascii_case("rdna4"))
    {
        let rdna4 =
            crate::host::host().is_some_and(|h| *RDNA4.get_or_init(|| is_rdna4(h.adapter_ids())));
        rdna4.then_some((AMD_VENDOR, RDNA4_DEVICES[0]))
    } else {
        None
    };
    let mode = parse_self_sized_mode(ids, override_var.as_deref());
    options_for_mode(&mode, hash)
}

/// The translation of `container` for the host's wave sizes
/// ([`crate::device::host_target`]), with what its console program says
/// of itself (`native`, [`native`]'s).
#[cfg(windows)]
fn translate(
    container: &[u8],
    native: Result<&Native, String>,
    options: xtp_shader::emit::Options,
) -> Result<Translated, String> {
    let host =
        crate::shader_cache::host_shader_with(container, crate::device::host_target(), options)?;
    let native = native?.clone();
    Ok(Translated {
        host,
        used: native.used,
        wave64: native.wave64,
    })
}

/// What the console program in `container` says of itself; a container
/// without one says nothing (no lanes, no resources).
#[cfg(windows)]
fn native(container: &[u8]) -> Result<Native, String> {
    use xtp_shader::{Container, FourCC, HwStage};
    let c = Container::parse(container).map_err(|e| e.to_string())?;
    let native = match c.part(FourCC::DXIL) {
        Some(part) => xtp_shader::native_programs(part.data)
            .map_err(|e| e.to_string())?
            .into_iter()
            .find(|p| {
                matches!(
                    p.stage,
                    HwStage::Vs | HwStage::Ps | HwStage::Ms | HwStage::Cs
                )
            }),
        None => None,
    };
    #[cfg(feature = "diagnostics")]
    if matches!(native.as_ref().map(|p| &p.io),
        Some(xtp_shader::native::StageIo::Ps { db_shader_control, .. }) if db_shader_control & (1 << 16) != 0)
    {
        // GFX10 DB_SHADER_CONTROL.PRIMITIVE_ORDERED_PIXEL_SHADER.
        crate::list::cuts::POPS_SHADERS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    Ok(match native {
        Some(p) => Native {
            wave64: p.wave64,
            threads: match p.io {
                xtp_shader::native::StageIo::Cs(cs) => Some(cs.threads),
                _ => None,
            },
            used: p.used,
        },
        None => Native {
            used: Vec::new(),
            wave64: false,
            threads: None,
        },
    })
}

#[repr(C)]
struct Pipeline {
    object: Object,
    packet: [u32; 8],
    /// Each used stage's container.
    shaders: Vec<Arc<Shader>>,
    /// Whether a list has bound it yet (its shaders are logged then).
    bound: std::sync::atomic::AtomicBool,
    /// What its host pipeline is made from, and the host pipeline once made
    /// (by a worker: the title may release the pipeline meanwhile).
    #[cfg(windows)]
    host: Arc<make::Pending>,
    #[cfg(windows)]
    builder: bool,
    #[cfg(windows)]
    missing_targets: OnceLock<Box<[OnceLock<Arc<make::Pending>>]>>,
}
const _: () = assert!(offset_of!(Pipeline, packet) == 0x10);

/// Stage names, in description order.
const GRAPHICS_STAGES: [&str; 5] = ["vertex", "pixel", "domain", "hull", "geometry"];

fn pipeline<'a>(this: This) -> &'a Pipeline {
    // SAFETY: the title passes one of our pipelines.
    unsafe { &*this.cast::<Pipeline>() }
}

#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn dispatch_hash(this: This) -> Option<String> {
    if this.is_null() {
        return None;
    }
    let p = pipeline(this);
    let shader = p
        .shaders
        .iter()
        .find(|s| matches!(s.stage, "mesh" | "compute" | "vertex"))
        .or_else(|| p.shaders.first())?;
    Some(shader.hash.clone())
}

/// List method: logs each pipeline's shaders the first time a list binds
/// it, so a run lists the shaders its frames need first.
pub(crate) fn bound(this: This) {
    if this.is_null() {
        return;
    }
    let p = pipeline(this);
    if p.bound.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let frame = crate::frame::current();
    for shader in &p.shaders {
        tracing::info!(pipeline = ?this, frame, stage = shader.stage, size = shader.container.len(), hash = %shader.hash, "first bind");
    }
}

/// The host pipeline of a title pipeline; `None` if one of its shaders does
/// not translate, the host refuses it, or it is still being made (its first
/// use starts that, see [`make`]).
#[cfg(windows)]
pub(crate) fn host_pipeline<'a>(this: This) -> Option<&'a host::HostPipeline> {
    if this.is_null() {
        return None;
    }
    make::ready(&pipeline(this).host)
}

#[cfg(windows)]
pub(crate) fn graphics_pipeline<'a>(this: This, missing: u8) -> Option<&'a host::HostPipeline> {
    if missing == 0 || this.is_null() {
        return host_pipeline(this);
    }
    make::ready(graphics_pending(pipeline(this), missing)?)
}

#[cfg(windows)]
fn graphics_pending(p: &Pipeline, missing: u8) -> Option<&Arc<make::Pending>> {
    let g = p.host.graphics.as_deref()?;
    let mask = missing & g.typed_slots();
    if mask == 0 {
        return Some(&p.host);
    }
    // Le tas de variantes n’existe que si le titre laisse un trou dans
    // des sorties typées. Chaque combinaison garde sa clé et sa compilation.
    let variants = p
        .missing_targets
        .get_or_init(|| (0..256).map(|_| OnceLock::new()).collect());
    let variant = variants[usize::from(mask)].get_or_init(|| {
        Arc::new(make::Pending::new(
            p.shaders.clone(),
            Some(Box::new(g.without_targets(mask))),
            None,
        ))
    });
    Some(variant)
}

#[cfg(windows)]
pub(crate) fn null_target_view(
    this: This,
    slot: usize,
) -> Option<(windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT, bool)> {
    if this.is_null() {
        return None;
    }
    pipeline(this).host.graphics.as_ref()?.null_view(slot)
}

/// The compute program of a title pipeline, made or not (`None` for a null
/// pipeline or a graphics one): how [`crate::rt_standin`] tells a dropped
/// dispatch's program, which has no host pipeline to ask.
#[cfg(windows)]
pub(crate) fn compute_shader<'a>(this: This) -> Option<&'a Shader> {
    if this.is_null() {
        return None;
    }
    match pipeline(this).shaders.as_slice() {
        [shader] if shader.stage == "compute" => Some(shader),
        _ => None,
    }
}

/// Host pipelines queued and not made yet.
#[cfg(windows)]
pub(crate) fn making() -> u64 {
    make::waiting()
}

/// Microseconds title threads waited for host pipelines (graphics,
/// compute) since the last call.
#[cfg(windows)]
pub(crate) fn take_wait_us() -> [u64; 2] {
    make::take_wait_us()
}

/// The pipelines refused since the start, by reason (`None` if none was).
#[cfg(windows)]
pub(crate) fn refusals() -> Option<String> {
    host::refusals()
}

/// A drop's cause (where in the list and why: "indirect kind 10 (mesh
/// dispatch): being made") and detail (the pipeline's shaders, or none).
#[cfg(all(windows, feature = "diagnostics"))]
type DropKey = (String, Option<String>);

/// Draws and dispatches dropped since the last [`take_drops`], by
/// [`DropKey`] (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
static DROPS: Mutex<Option<HashMap<DropKey, u64>>> = Mutex::new(None);

/// Shader hashes as printed in the pipeline diagnostics.
#[cfg(all(windows, feature = "diagnostics"))]
fn parse_skip_hashes(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|hash| !hash.is_empty())
        .filter_map(|hash| {
            if hash.len() == 16 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                Some(hash.to_ascii_lowercase())
            } else {
                tracing::warn!(hash, "XTP_SKIP_PIPELINES expects 16-digit shader hashes");
                None
            }
        })
        .collect()
}

#[cfg(all(windows, feature = "diagnostics"))]
fn skip_hashes() -> &'static [String] {
    static HASHES: OnceLock<Vec<String>> = OnceLock::new();
    HASHES.get_or_init(|| {
        let value = std::env::var("XTP_SKIP_PIPELINES").unwrap_or_default();
        parse_skip_hashes(&value)
    })
}

#[cfg(all(windows, feature = "diagnostics"))]
fn skipped_shader(p: &Pipeline) -> Option<&str> {
    let hashes = skip_hashes();
    if hashes.is_empty() {
        return None;
    }
    p.shaders
        .iter()
        .find(|shader| hashes.contains(&shader.hash))
        .map(|shader| shader.hash.as_str())
}

#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn is_skipped(this: This) -> bool {
    !this.is_null() && skipped_shader(pipeline(this)).is_some()
}

/// Counts a draw or dispatch of `this`, recorded at `path` ("draw",
/// "indirect kind 0 (draw)", ...), intentionally skipped or dropped for
/// want of a host pipeline: why, with the pipeline's shaders as the
/// detail (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn note_drop(path: &str, this: This) {
    let (why, detail) = if this.is_null() {
        ("no pipeline set".to_owned(), None)
    } else {
        let p = pipeline(this);
        match skipped_shader(p) {
            Some(hash) => (format!("skipped shader {hash}"), None),
            None => match make::missing(&p.host) {
                Some(missing) => missing,
                // It has one now (made since the draw looked).
                None => ("made meanwhile".to_owned(), None),
            },
        }
    };
    note(format!("{path}: {why}"), detail);
}

/// Counts a draw or dispatch dropped for `why` (diagnostics).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn note_why(why: String) {
    note(why, None);
}

#[cfg(all(windows, feature = "diagnostics"))]
fn note(cause: String, detail: Option<String>) {
    let mut drops = DROPS.lock().unwrap_or_else(|e| e.into_inner());
    *drops
        .get_or_insert_with(HashMap::new)
        .entry((cause, detail))
        .or_default() += 1;
}

/// The drops counted since the last call (diagnostics): first each cause's
/// total ("by cause: <cause>"), most first, so that the frame report's
/// first lines add up to every drop; then the causes that have details
/// (shaders), one entry per detail ("<cause>: <detail>"), most first. An
/// entry of the second kind is part of one of the first.
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn take_drops() -> Vec<(String, u64)> {
    let drops = DROPS.lock().unwrap_or_else(|e| e.into_inner()).take();
    by_cause(drops.into_iter().flatten())
}

#[cfg(all(windows, feature = "diagnostics"))]
fn by_cause(drops: impl Iterator<Item = (DropKey, u64)>) -> Vec<(String, u64)> {
    let mut causes: HashMap<String, u64> = HashMap::new();
    let mut details = Vec::new();
    for ((cause, detail), n) in drops {
        *causes.entry(cause.clone()).or_default() += n;
        if let Some(detail) = detail {
            details.push((format!("{cause}: {detail}"), n));
        }
    }
    let mut causes: Vec<(String, u64)> = causes
        .into_iter()
        .map(|(cause, n)| (format!("by cause: {cause}"), n))
        .collect();
    // Frame reports only show the first 40 lines. Keep ties ordered so a
    // HashMap iteration order change cannot change which ones appear.
    let order = |a: &(String, u64), b: &(String, u64)| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0));
    causes.sort_unstable_by(order);
    details.sort_unstable_by(order);
    causes.extend(details);
    causes
}

/// The stage's container, if it has one.
fn copy(stage: &'static str, code: &Bytecode) -> Option<Arc<Shader>> {
    if code.code.is_null() || code.size == 0 {
        return None;
    }
    // SAFETY: the title passes `size` bytes of shader code.
    Some(shader(stage, unsafe {
        std::slice::from_raw_parts(code.code, code.size)
    }))
}

fn hand_out(
    shaders: Vec<Arc<Shader>>,
    #[cfg(windows)] graphics: Option<(Box<host::GraphicsState>, Option<record::Stream>)>,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    tracing::trace!(?shaders, "pipeline state");
    #[cfg(windows)]
    let member = builder::member(&shaders);
    #[cfg(windows)]
    let builder = member.is_some();
    #[cfg(windows)]
    let host = member.unwrap_or_else(|| {
        let (graphics, stream) = graphics.unzip();
        Arc::new(make::Pending::new(
            shaders.clone(),
            graphics,
            stream.flatten(),
        ))
    });
    #[cfg(windows)]
    make::early(&host);
    let pipeline = Box::new(Pipeline {
        object: Object::new(&FUNCTIONS),
        packet: [0; 8],
        #[cfg(windows)]
        host,
        #[cfg(windows)]
        builder,
        #[cfg(windows)]
        missing_targets: OnceLock::new(),
        shaders,
        bound: std::sync::atomic::AtomicBool::new(false),
    });
    object::hand_out(PIPELINE, pipeline, iid, &PIPELINE_IIDS, out)
}

/// Device method.
pub(crate) extern "system" fn create_graphics(
    _: This,
    desc: *const GraphicsHead,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    let stages = GRAPHICS_STAGES.iter().zip(&desc.stages);
    let shaders = stages.filter_map(|(s, c)| copy(s, c)).collect();
    hand_out(
        shaders,
        #[cfg(windows)]
        None,
        iid,
        out,
    )
}

/// Device method.
pub(crate) extern "system" fn create_compute(
    _: This,
    desc: *const ComputeHead,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    hand_out(
        copy("compute", &desc.stage).into_iter().collect(),
        #[cfg(windows)]
        None,
        iid,
        out,
    )
}

/// Device method 0x1f0: the plain compute description plus console
/// extensions `(desc, extension count, extensions, iid, out)`.
/// shortcut: the extensions (console shader settings) are ignored, as is
/// the rest of the description.
pub(crate) extern "system" fn create_compute_x(
    this: This,
    desc: *const ComputeHead,
    _extensions: u32,
    _extended: *const u8,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    create_compute(this, desc, iid, out)
}

/// A pipeline described as a stream: its size and the subobjects, each a
/// type then its data, padded to 8 bytes.
#[repr(C)]
pub(crate) struct StreamDesc {
    size: usize,
    stream: *const u8,
}

/// A subobject type's size (type included), and the stage it is if it is a
/// shader. The console numbers amplification and mesh shaders 23 and 24
/// (PC: 24 and 25); its own types from 25 on are not sized yet.
fn subobject(kind: u32) -> Option<(usize, Option<&'static str>)> {
    let stage = |name| Some((24, Some(name)));
    match kind {
        1 => stage("vertex"),
        2 => stage("pixel"),
        3 => stage("domain"),
        4 => stage("hull"),
        5 => stage("geometry"),
        6 => stage("compute"),
        23 => stage("amplification"),
        24 => stage("mesh"),
        0 => Some((16, None)),
        12 | 19 => Some((24, None)),
        7 | 15 => Some((40, None)),
        8 => Some((336, None)),
        10 => Some((48, None)),
        11 => Some((56, None)),
        17 => Some((16, None)),
        21 => Some((64, None)),
        22 => Some((32, None)),
        9 | 13 | 14 | 16 | 18 | 20 => Some((8, None)),
        _ => None,
    }
}

/// A stream subobject's size and whether it is a shader, for the frame
/// recorder (`crate::record`), which copies what subobjects point at.
#[cfg(all(windows, any(test, feature = "diagnostics")))]
pub(crate) fn recorded_subobject(kind: u32) -> Option<(usize, bool)> {
    subobject(kind).map(|(size, stage)| (size, stage.is_some()))
}

/// Device method: a pipeline from a stream; the shaders are kept, and the
/// graphics state for the host pipeline.
pub(crate) extern "system" fn create_stream(
    _: This,
    desc: *const StreamDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    if desc.stream.is_null() {
        return E_INVALIDARG;
    }
    let mut shaders = Vec::new();
    #[cfg(windows)]
    let mut graphics = host::GraphicsState::default();
    // The subobjects as the title gave them, kept for the precook.
    #[cfg(windows)]
    let mut kept = record::Capture::new();
    let mut at = 0;
    while at + 8 <= desc.size {
        // SAFETY: inside the stream the title passes.
        let kind = unsafe { desc.stream.add(at).cast::<u32>().read_unaligned() };
        let Some((size, stage)) = subobject(kind) else {
            tracing::warn!(
                kind,
                at,
                "pipeline subobject of unknown size; the rest is ignored"
            );
            #[cfg(windows)]
            kept.unknown();
            break;
        };
        if let Some(stage) = stage {
            // SAFETY: a shader subobject has its bytecode at +8.
            let code = unsafe { &*desc.stream.add(at + 8).cast::<Bytecode>() };
            shaders.extend(copy(stage, code));
        }
        #[cfg(windows)]
        // SAFETY: a whole subobject of this kind and size lies at `at`.
        unsafe {
            graphics.read(kind, desc.stream.add(at));
            kept.take(kind, desc.stream.add(at), size);
        };
        at += size;
    }
    hand_out(
        shaders,
        #[cfg(windows)]
        Some((Box::new(graphics), kept.finish())),
        iid,
        out,
    )
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let found = object::is(iid, &PIPELINE_IIDS).then_some(this);
    object::answer(PIPELINE, iid, found, out)
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Pipeline>(this)
}

/// IUnknown 3, object 4, device child 1, pipeline 3.
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<PIPELINE, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    Table(f)
};

#[cfg(windows)]
mod builder;
pub mod effects;
#[cfg(windows)]
pub use builder::register_builder;
#[cfg(windows)]
pub use builder::{BuilderCall, builder_call};
#[cfg(windows)]
pub(crate) fn builder_held(this: This) -> bool {
    !this.is_null() && pipeline(this).builder && !builder::admitted()
}
#[cfg(windows)]
pub(crate) mod host;
#[cfg(windows)]
mod library;
#[cfg(windows)]
mod make;
#[cfg(windows)]
pub use make::configure_priority_pipelines;
#[cfg(windows)]
pub mod precook;
#[cfg(windows)]
mod record;
#[cfg(windows)]
pub(crate) mod samplers;

/// Keeps compiled host pipelines, and the list of title pipelines to start
/// early, in `dir` (a folder of the cache) across runs.
#[cfg(windows)]
pub fn use_pipeline_cache(dir: std::path::PathBuf) {
    library::use_pipeline_cache(dir);
}

/// Saves the pipeline cache now if anything was added (at the end of a run).
#[cfg(windows)]
pub fn save_pipeline_cache() {
    library::save(true);
}

/// Lets the mapped pipeline library's pages leave the working set once loads
/// go idle. Called periodically from host maintenance.
#[cfg(windows)]
pub(crate) fn trim_library() {
    library::trim();
}

#[cfg(all(test, windows))]
pub(crate) fn test_graphics_pipeline(
    shaders: Vec<Arc<Shader>>,
    graphics: host::GraphicsState,
    masks: &[u8],
) -> This {
    let mut title = std::ptr::null();
    assert_eq!(
        hand_out(
            shaders,
            Some((Box::new(graphics), None)),
            &PIPELINE_IIDS[0],
            &mut title
        ),
        object::S_OK
    );
    for &mask in masks {
        graphics_pending(pipeline(title), mask).unwrap().make(|| {});
    }
    title
}

#[cfg(all(test, windows))]
pub(crate) fn test_shader(stage: &'static str, host: xtp_shader::emit::HostShader) -> Arc<Shader> {
    Arc::new(Shader {
        stage,
        hash: blake3::hash(&host.dxil).to_hex()[..16].to_owned(),
        ambient: None,
        container: Vec::new(),
        native: OnceLock::new(),
        translated: OnceLock::from(Some(Translated {
            host,
            used: vec![],
            wave64: false,
        })),
    })
}

#[cfg(test)]
mod tests {
    #[cfg(all(windows, feature = "diagnostics"))]
    #[test]
    fn skip_hashes_accepts_full_hashes_only() {
        assert!(super::parse_skip_hashes(" , ").is_empty());
        assert_eq!(
            super::parse_skip_hashes(" 0123456789ABCDEF, 0123456789abcde, xyz, 0011223344556677,"),
            ["0123456789abcdef", "0011223344556677"]
        );
    }

    /// The frame report's drops: each cause's total first (they add up to
    /// every drop), then the details of the causes that have some.
    #[cfg(all(windows, feature = "diagnostics"))]
    #[test]
    fn drops_come_by_cause_then_by_detail() {
        let entry = |cause: &str, detail: Option<&str>, n| {
            ((cause.to_owned(), detail.map(str::to_owned)), n)
        };
        let drops = super::by_cause(
            [
                entry("mesh dispatch: refused, wave64", Some("[mesh a]"), 3),
                entry("mesh dispatch: refused, wave64", Some("[mesh b]"), 5),
                entry("indirect kind 0 (draw): rectangles", None, 6),
                entry("dispatch: being made", None, 1),
                entry("dispatch: does not translate", Some("compute c"), 2),
            ]
            .into_iter(),
        );
        let lines: Vec<(&str, u64)> = drops.iter().map(|(w, n)| (w.as_str(), *n)).collect();
        assert_eq!(
            lines,
            [
                ("by cause: mesh dispatch: refused, wave64", 8),
                ("by cause: indirect kind 0 (draw): rectangles", 6),
                ("by cause: dispatch: does not translate", 2),
                ("by cause: dispatch: being made", 1),
                ("mesh dispatch: refused, wave64: [mesh b]", 5),
                ("mesh dispatch: refused, wave64: [mesh a]", 3),
                ("dispatch: does not translate: compute c", 2),
            ]
        );
    }

    use super::*;
    use crate::object::{S_OK, slot};

    fn shaders(p: This) -> Vec<(&'static str, Vec<u8>)> {
        pipeline(p)
            .shaders
            .iter()
            .map(|s| (s.stage, s.container.clone()))
            .collect()
    }

    #[test]
    fn pipelines_keep_their_shaders() {
        let (vs, ps) = ([1u8, 2, 3], [4u8; 5]);
        let none = || Bytecode {
            code: std::ptr::null(),
            size: 0,
        };
        let desc = GraphicsHead {
            root: std::ptr::null(),
            stages: [
                Bytecode {
                    code: vs.as_ptr(),
                    size: vs.len(),
                },
                Bytecode {
                    code: ps.as_ptr(),
                    size: ps.len(),
                },
                none(),
                none(),
                none(),
            ],
        };
        let mut p = std::ptr::null();
        assert_eq!(
            create_graphics(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut p),
            S_OK
        );
        assert_eq!(
            shaders(p),
            [("vertex", vs.to_vec()), ("pixel", ps.to_vec())]
        );
        // A second pipeline with the same pixel shader shares its container.
        let mut q = std::ptr::null();
        create_graphics(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut q);
        assert!(Arc::ptr_eq(
            &pipeline(p).shaders[1],
            &pipeline(q).shaders[1]
        ));
        assert_eq!(slot::<extern "system" fn(This) -> u32>(p, 0x10)(p), 0);
    }

    /// The stream layout BTP 6 builds (title+0x86f090): root
    /// signature, five shader slots (the first a mesh shader), blend, sample
    /// mask, rasterizer, depth-stencil 1, sample description, input layout,
    /// topology, render target and depth formats: 0x2b0 bytes.
    #[test]
    fn streams_yield_their_shaders() {
        let (ms, ps) = ([9u8; 7], [4u8; 5]);
        let mut stream = vec![0u8; 0x2b0];
        let mut put = |at: usize, v: u64| stream[at..at + 8].copy_from_slice(&v.to_le_bytes());
        for (at, kind) in [
            (0x10, 24),
            (0x28, 2),
            (0x40, 3),
            (0x58, 4),
            (0x70, 5),
            (0x88, 8),
            (0x1d8, 9),
            (0x1e0, 10),
            (0x210, 21),
            (0x250, 17),
            (0x260, 12),
            (0x278, 14),
            (0x280, 15),
            (0x2a8, 16),
        ] {
            put(at, kind);
        }
        put(0x18, ms.as_ptr() as u64);
        put(0x20, ms.len() as u64);
        put(0x30, ps.as_ptr() as u64);
        put(0x38, ps.len() as u64);
        // Topology type 3 (triangles), one render target of format 28.
        put(0x278, 14 | 3 << 32);
        put(0x280, 15 | 28 << 32);
        put(0x2a0, 1 << 32);
        let desc = StreamDesc {
            size: stream.len(),
            stream: stream.as_ptr(),
        };
        let mut p = std::ptr::null();
        let hr = create_stream(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut p);
        assert_eq!(hr, S_OK);
        assert_eq!(shaders(p), [("mesh", ms.to_vec()), ("pixel", ps.to_vec())]);
        #[cfg(windows)]
        {
            let g = pipeline(p).host.graphics.as_deref().unwrap();
            assert_eq!(g.topology.0, 3);
            assert_eq!(
                (g.targets.NumRenderTargets, g.targets.RTFormats[0].0),
                (1, 28)
            );
        }
    }

    #[cfg(windows)]
    #[test]
    fn self_sized_options_detect_rdna4_and_respect_overrides() {
        for device in [0x7550, 0x7551, 0x7590] {
            assert!(
                self_sized_options_from(Some((AMD_VENDOR, device)), None, "0123456789abcdef")
                    .self_sized_loads_via_memory
            );
            assert!(
                !self_sized_options_from(Some((AMD_VENDOR, device)), Some("0"), "0123456789abcdef")
                    .self_sized_loads_via_memory
            );
            assert!(
                !self_sized_options_from(
                    Some((AMD_VENDOR, device)),
                    Some("off"),
                    "0123456789abcdef"
                )
                .self_sized_loads_via_memory
            );
        }
        assert!(
            !self_sized_options_from(Some((AMD_VENDOR, 0x744c)), None, "0123456789abcdef")
                .self_sized_loads_via_memory
        );
        assert!(
            !self_sized_options_from(Some((0x10de, 0x2684)), None, "0123456789abcdef")
                .self_sized_loads_via_memory
        );
        assert!(
            !self_sized_options_from(None, Some("rdna4"), "0123456789abcdef")
                .self_sized_loads_via_memory
        );
        assert!(
            self_sized_options_from(None, Some("1"), "0123456789abcdef")
                .self_sized_loads_via_memory
        );
        assert!(
            self_sized_options_from(
                None,
                Some("0123456789ABCDEF, fedcba9876543210"),
                "fedcba9876543210"
            )
            .self_sized_loads_via_memory
        );
        assert!(
            !self_sized_options_from(
                None,
                Some("0123456789abcdef,fedcba9876543210"),
                "aaaaaaaaaaaaaaaa"
            )
            .self_sized_loads_via_memory
        );
    }
}
