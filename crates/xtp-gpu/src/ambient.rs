//! Optional indirect lighting from visible surfaces. A radiance pass saves
//! the surface light in 64 by 36 cells and updates a slow mean. A later
//! resolve places those cells in view space using the current reverse-Z
//! depth and gathers first-order harmonics at 160 by 90 points. The final
//! pass interpolates by depth, shades creases and bounds the old diffuse
//! light as it raises it. An optional pass caps in-scattering using the
//! slow mean and the estimate along each sightline.
//!
//! The GPU runs all title lists on one host queue. History validity lives
//! in the UAVs, so it follows execution order even when lists are recorded
//! in parallel, discarded or replayed. The passes share the title's bound
//! view heap and leave its resource states in place.
//!
//! Adapted from the GPL-3.0-or-later ambient stand-in in a VIctory fork.

use std::collections::HashMap;
use std::mem::{ManuallyDrop, size_of};
use std::sync::OnceLock;

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{Error, PCSTR, Result, s};

use crate::host::{HostList, host, uav_barrier};

/// The root parameter and offset in dwords of a title table entry.
pub type Binding = (u8, u32);

/// A compute program's role in the estimate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// Resolves the probe diffuse light.
    Resolve,
    /// Saves the radiance of visible surfaces for a later frame.
    Radiance,
    /// Finishes writing the in-scattering volume.
    Air,
    /// Supplies sky light for reflection rays that missed.
    Specular,
}

/// The table entries used by the compute programs for each role.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    /// A 2D R32_UINT UAV holding RGB9E5 light.
    pub light: Binding,
    /// A single-channel 2D reverse-Z depth SRV at the light's resolution.
    pub depth: Binding,
    /// A 2D R11G11B10_FLOAT UAV holding surface radiance.
    pub radiance: Binding,
    /// A 3D R11G11B10_FLOAT UAV holding in-scattered light.
    pub air: Binding,
    /// Reflection-miss output and preferred scene depth.
    pub specular: Binding,
    /// The reverse-Z SRV at twice the reflection resolution.
    pub specular_depth: Binding,
    /// Surface layers with usable depth, selected by the first root constant.
    pub reflection_layers: &'static [u32],
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    Off,
    Estimate(f32),
    Constant(f32),
}

fn positive(text: &str) -> Option<f32> {
    text.trim()
        .parse()
        .ok()
        .filter(|value: &f32| value.is_finite() && *value > 0.0)
}

impl Mode {
    fn parse(text: &str) -> Self {
        if let Some(level) = text.trim().strip_prefix("constant:") {
            positive(level).map_or(Self::Off, Self::Constant)
        } else {
            positive(text).map_or(Self::Off, Self::Estimate)
        }
    }
}

#[derive(Clone, Copy)]
struct Settings {
    mode: Mode,
    air: f32,
    lamps: f32,
    cut: f32,
    floor: f32,
    specular: Option<(f32, f32)>,
}

impl Settings {
    /// XTP_AMBIENT is off when unset, 0 or off. A positive number scales
    /// the estimate, and constant:<level> uses fixed light with crease shade.
    /// XTP_AMBIENT_AIR sets a positive cap multiplier in estimate mode.
    /// XTP_AMBIENT_LAMPS sets a positive per-sample cap multiple of the slow
    /// frame mean in estimate mode (bounded by 20).
    fn read(get: impl Fn(&str) -> Option<String>) -> Self {
        let number = |name| get(name).as_deref().and_then(positive).unwrap_or_default();
        Self {
            mode: Mode::parse(&get("XTP_AMBIENT").unwrap_or_default()),
            air: number("XTP_AMBIENT_AIR"),
            lamps: number("XTP_AMBIENT_LAMPS"),
            cut: number("XTP_AMBIENT_CUT").max(1.0).ln(),
            floor: number("XTP_AMBIENT_FLOOR"),
            specular: get("XTP_AMBIENT_SPECULAR").and_then(|value| {
                let mut parts = value.split(',');
                let share = positive(parts.next()?)?;
                let sky = parts.next().and_then(positive).unwrap_or_default().min(1.0);
                Some((share, sky))
            }),
        }
    }

    fn runs(self, role: Role) -> bool {
        let estimate = matches!(self.mode, Mode::Estimate(_));
        match role {
            Role::Resolve => self.mode != Mode::Off,
            Role::Air => estimate && self.air > 0.0,
            Role::Radiance => estimate || self.specular.is_some(),
            Role::Specular => self.specular.is_some(),
        }
    }

    fn record(
        self,
        list: &HostList,
        bound: &Bound,
        get_passes: impl FnOnce() -> Option<&'static Passes>,
    ) -> bool {
        self.runs(bound.role) && get_passes().is_some_and(|passes| passes.record(list, self, bound))
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Program {
    pub(crate) role: Role,
    pub(crate) layout: Layout,
}

struct Configuration {
    reflection_miss: Option<fn(&[u8]) -> bool>,
    settings: Settings,
    layout: Layout,
    roles: HashMap<u64, Role>,
}

static CONFIGURATION: OnceLock<Configuration> = OnceLock::new();

fn log_hash(text: &str) -> Option<u64> {
    (text.len() == 16 && text.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .then(|| u64::from_str_radix(text, 16).ok())
        .flatten()
}

fn override_roles(roles: &mut HashMap<u64, Role>, text: &str, mut invalid: impl FnMut(&str)) {
    for group in text
        .split(';')
        .map(str::trim)
        .filter(|part| !part.is_empty())
    {
        let Some((role, hashes)) = group.split_once(':') else {
            invalid(group);
            continue;
        };
        let role = match role.trim() {
            "resolve" => Role::Resolve,
            "radiance" => Role::Radiance,
            "air" => Role::Air,
            "specular" => Role::Specular,
            _ => {
                invalid(group);
                continue;
            }
        };
        for hash in hashes.split(',').map(str::trim) {
            match log_hash(hash) {
                Some(hash) => {
                    roles.insert(hash, role);
                }
                None => invalid(hash),
            }
        }
    }
}

/// Supplies a title's bindings and known compute hashes before it creates
/// pipelines. XTP_AMBIENT_PROGRAMS adds or reassigns hashes using
/// `resolve:<hash>,...;radiance:<hash>,...;air:<hash>,...`. Hashes are the
/// 16 hex digits printed by the runtime. The last assignment wins.
pub fn configure(
    layout: Layout,
    defaults: &[(&str, Role)],
    reflection_miss: Option<fn(&[u8]) -> bool>,
) {
    CONFIGURATION.get_or_init(|| {
        let settings = Settings::read(|name| std::env::var(name).ok());
        let mut roles = HashMap::new();
        if settings.mode != Mode::Off || settings.specular.is_some() {
            roles.extend(
                defaults
                    .iter()
                    .filter_map(|&(hash, role)| log_hash(hash).map(|hash| (hash, role))),
            );
            if let Ok(text) = std::env::var("XTP_AMBIENT_PROGRAMS") {
                override_roles(&mut roles, &text, |entry| {
                    tracing::warn!(entry, "invalid XTP_AMBIENT_PROGRAMS entry");
                });
            }
            if settings.runs(Role::Radiance) && !roles.values().any(|r| *r == Role::Radiance) {
                tracing::warn!("ambient estimate needs a radiance program in XTP_AMBIENT_PROGRAMS");
                roles.retain(|_, role| {
                    *role == Role::Resolve && matches!(settings.mode, Mode::Constant(_))
                });
            }
            if settings.runs(Role::Air)
                && !roles.is_empty()
                && !roles.values().any(|r| *r == Role::Air)
            {
                tracing::warn!("ambient air cap needs an air program in XTP_AMBIENT_PROGRAMS");
            }
        }
        Configuration {
            reflection_miss: reflection_miss.filter(|_| {
                settings.specular.is_some() && roles.values().any(|role| *role == Role::Radiance)
            }),
            settings,
            layout,
            roles,
        }
    });
}

/// Looks up the first eight BLAKE3 bytes in the printed byte order.
pub(crate) fn program(hash: &[u8; 32], bytes: &[u8]) -> Option<Program> {
    let config = CONFIGURATION.get()?;
    let key = u64::from_be_bytes(hash[..8].try_into().ok()?);
    let role = config.roles.get(&key).copied().or_else(|| {
        config
            .reflection_miss
            .is_some_and(|matches| matches(bytes))
            .then_some(Role::Specular)
    })?;
    config.settings.runs(role).then_some(Program {
        role,
        layout: config.layout,
    })
}

/// A view already bound by the translated dispatch, kept alive with it.
pub(crate) struct Texture {
    pub(crate) handle: D3D12_GPU_DESCRIPTOR_HANDLE,
    pub(crate) resource: ID3D12Resource,
}

pub(crate) struct Bound {
    pub(crate) role: Role,
    pub(crate) image: Texture,
    pub(crate) depth: Option<Texture>,
    pub(crate) size: [u32; 3],
}

const SOURCE: &str = include_str!("ambient.hlsl");
const CELL_BYTES: u64 = (2 * 64 * 36 + 2) * 16;
const POINT_BYTES: u64 = (160 * 90 * 3 + 1) * 16;

struct Passes {
    root: ID3D12RootSignature,
    keep: Option<(ID3D12PipelineState, ID3D12PipelineState)>,
    place: ID3D12PipelineState,
    gather: ID3D12PipelineState,
    raise: ID3D12PipelineState,
    air: Option<ID3D12PipelineState>,
    reflect: Option<ID3D12PipelineState>,
    cells: ID3D12Resource,
    points: ID3D12Resource,
}

fn compile(part: &str, entry: PCSTR) -> Result<ID3DBlob> {
    let source = format!("#define {part}\n{SOURCE}");
    let mut code = None;
    let mut errors = None;
    // SAFETY: the source and names live through the call, and the output slots are ours.
    let result = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            s!("ambient.hlsl"),
            None,
            None,
            entry,
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(error) = result {
        if let Some(errors) = errors {
            // SAFETY: the compiler blob owns these bytes.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                )
            };
            tracing::warn!(part, message = %String::from_utf8_lossy(bytes), "ambient shader compilation failed");
        }
        return Err(error);
    }
    code.ok_or_else(Error::empty)
}

fn typed_load_r11(device: &ID3D12Device) -> bool {
    let mut support = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
        Format: DXGI_FORMAT_R11G11B10_FLOAT,
        ..Default::default()
    };
    // SAFETY: the feature selector matches the writable structure and its size.
    unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_FORMAT_SUPPORT,
            std::ptr::from_mut(&mut support).cast(),
            size_of::<D3D12_FEATURE_DATA_FORMAT_SUPPORT>() as u32,
        )
    }
    .is_ok()
        && support
            .Support2
            .contains(D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD)
}

fn scratch(device: &ID3D12Device, bytes: u64) -> Result<ID3D12Resource> {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut buffer = None;
    // SAFETY: plain committed buffer creation. Without CREATE_NOT_ZEROED,
    // the new heap starts zeroed, including the GPU history flags.
    unsafe {
        device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            None,
            &mut buffer,
        )?;
    }
    buffer.ok_or_else(Error::empty)
}

impl Passes {
    fn make(device: &ID3D12Device, settings: Settings) -> Result<Self> {
        Self::make_with_r11(device, settings, typed_load_r11(device))
    }

    fn make_with_r11(
        device: &ID3D12Device,
        settings: Settings,
        r11_supported: bool,
    ) -> Result<Self> {
        let raise_code = compile("RAISE", s!("raise"))?;
        // SAFETY: the compiler blob contains the shared root signature.
        let root: ID3D12RootSignature = unsafe {
            let bytes = std::slice::from_raw_parts(
                raise_code.GetBufferPointer().cast::<u8>(),
                raise_code.GetBufferSize(),
            );
            device.CreateRootSignature(0, bytes)?
        };
        let pipeline = |code: &ID3DBlob| {
            // SAFETY: the root signature and bytecode live through PSO creation.
            unsafe {
                let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: ManuallyDrop::new(std::mem::transmute_copy(&root)),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.GetBufferPointer(),
                        BytecodeLength: code.GetBufferSize(),
                    },
                    ..Default::default()
                };
                device.CreateComputePipelineState(&desc)
            }
        };
        let make = |part, entry| pipeline(&compile(part, entry)?);
        let raise = pipeline(&raise_code)?;
        let place = make("PLACE", s!("place"))?;
        let gather = make("GATHER", s!("gather"))?;
        let r11 = settings.runs(Role::Radiance) && r11_supported;
        if settings.runs(Role::Radiance) && !r11 {
            tracing::warn!("ambient estimate needs R11G11B10_FLOAT typed UAV loads");
        }
        let keep = r11
            .then(|| Ok::<_, Error>((make("KEEP", s!("keep"))?, make("KEEP", s!("mean"))?)))
            .transpose()?;
        let air = (r11 && settings.runs(Role::Air))
            .then(|| make("AIR", s!("clear_air")))
            .transpose()?;
        let reflect = (r11 && settings.runs(Role::Specular))
            .then(|| make("REFLECT", s!("reflect_light")))
            .transpose()?;
        Ok(Self {
            root,
            keep,
            place,
            gather,
            raise,
            air,
            reflect,
            cells: scratch(device, CELL_BYTES)?,
            points: scratch(device, POINT_BYTES)?,
        })
    }

    fn record(&self, list: &HostList, settings: Settings, bound: &Bound) -> bool {
        if !settings.runs(bound.role) {
            return false;
        }
        let (level, constant) = match (bound.role, settings.mode) {
            (Role::Specular, _) => settings.specular.unwrap_or_default(),
            (_, Mode::Off) => (0.0, 0.0),
            (_, Mode::Estimate(level)) => (level, 0.0),
            (_, Mode::Constant(level)) => (0.0, level),
        };
        let [width, height, slices] = bound.size;
        let (passes, groups) = match bound.role {
            Role::Radiance => {
                let Some((keep, mean)) = &self.keep else {
                    return false;
                };
                (
                    [Some(keep), Some(mean), None],
                    [[8, 9, 1], [1, 1, 1], [0; 3]],
                )
            }
            Role::Resolve => {
                let estimate = matches!(settings.mode, Mode::Estimate(_));
                if estimate && self.keep.is_none() {
                    return false;
                }
                (
                    [
                        estimate.then_some(&self.place),
                        estimate.then_some(&self.gather),
                        Some(&self.raise),
                    ],
                    [
                        [8, 9, 1],
                        [20, 45, 1],
                        [width.div_ceil(8), height.div_ceil(8), 1],
                    ],
                )
            }
            Role::Air => {
                let Some(air) = &self.air else { return false };
                (
                    [Some(air), None, None],
                    [
                        [width.div_ceil(8), height.div_ceil(8), slices.div_ceil(2)],
                        [0; 3],
                        [0; 3],
                    ],
                )
            }
            Role::Specular => {
                let Some(reflect) = &self.reflect else {
                    return false;
                };
                (
                    [Some(&self.place), Some(&self.gather), Some(reflect)],
                    [
                        [8, 9, 1],
                        [20, 45, 1],
                        [width.div_ceil(8), height.div_ceil(8), 1],
                    ],
                )
            }
        };
        // The near plane and tangent of the vertical half-FOV approximate
        // the projection for converting reverse Z to view-space distances.
        let constants = [
            level.to_bits(),
            constant.to_bits(),
            1.0f32.to_bits(),
            0.6f32.to_bits(),
            0.1f32.to_bits(),
            0.45f32.to_bits(),
            width,
            height,
            0.04f32.to_bits(),
            4.0f32.to_bits(),
            settings.air.to_bits(),
            settings.lamps.to_bits(),
            settings.cut.to_bits(),
            settings.floor.to_bits(),
        ];
        list.keep(&bound.image.resource);
        if let Some(depth) = &bound.depth {
            list.keep(&depth.resource);
        }
        let l = &list.list;
        // SAFETY: the open list has the validated views' heap bound and the
        // title's dispatch prepared their states. The scratch buffers are
        // persistent UAVs. A barrier orders the title's writes and each
        // pass's writes before subsequent accesses on the host queue.
        unsafe {
            l.ResourceBarrier(&[uav_barrier()]);
            l.SetComputeRootSignature(&self.root);
            l.SetComputeRoot32BitConstants(0, constants.len() as u32, constants.as_ptr().cast(), 0);
            if let Some(depth) = &bound.depth {
                l.SetComputeRootDescriptorTable(1, depth.handle);
            }
            l.SetComputeRootDescriptorTable(2, bound.image.handle);
            l.SetComputeRootUnorderedAccessView(3, self.cells.GetGPUVirtualAddress());
            l.SetComputeRootUnorderedAccessView(4, self.points.GetGPUVirtualAddress());
            for (pass, [x, y, z]) in passes.into_iter().zip(groups) {
                if let Some(pass) = pass {
                    l.SetPipelineState(pass);
                    l.Dispatch(x, y, z);
                    l.ResourceBarrier(&[uav_barrier()]);
                }
            }
        }
        true
    }
}

/// Records after a successful title dispatch. True means the caller must
/// rebind the title's compute root signature and pipeline.
pub(crate) fn after(list: &HostList, bound: &Bound) -> bool {
    let Some(config) = CONFIGURATION.get() else {
        return false;
    };
    let ran = config.settings.record(list, bound, || {
        static PASSES: OnceLock<Option<Passes>> = OnceLock::new();
        PASSES
            .get_or_init(|| {
                Passes::make(&host()?.device, config.settings)
                    .inspect_err(|error| tracing::warn!(%error, "ambient passes unavailable"))
                    .ok()
            })
            .as_ref()
    });
    if ran {
        use std::sync::atomic::{AtomicU8, Ordering};
        static LOGGED: AtomicU8 = AtomicU8::new(0);
        let bit = 1 << bound.role as u8;
        if LOGGED.fetch_or(bit, Ordering::Relaxed) & bit == 0 {
            tracing::info!(role = ?bound.role, size = ?bound.size, "recorded ambient pass");
        }
    }
    ran
}

#[cfg(test)]
mod tests;
