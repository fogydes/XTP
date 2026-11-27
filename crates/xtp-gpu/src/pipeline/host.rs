//! Host pipelines for title pipelines (decision 0007, step 3): the
//! title's graphics state with the translated shaders, under a root
//! signature of our own that the translated shaders' bindings define
//! (`xtp_shader::emit`): per stage one block of root constants for the
//! user data, root views for the title's root views, static samplers for
//! the immediate samplers, and directly indexed heaps.

use std::ffi::{CStr, CString};
use std::sync::Arc;

use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::PCSTR;
use xtp_shader::emit::{
    LOOP_COUNTER_SPACE, ROOT_CBV_SPACE, SAMPLER_SPACE, STRIDE_SPACE, TLAS_REGISTER, USER_DATA_SPACE,
};

use super::{Shader, Translated};
use crate::host::host;

#[cfg(feature = "diagnostics")]
pub(crate) static BOUNDED_PIPELINES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

trait PipelineCaps: crate::resource::format::Caps {
    fn logic_op(&self, format: DXGI_FORMAT) -> bool;
}

impl PipelineCaps for crate::resource::format::HostCaps {
    fn logic_op(&self, format: DXGI_FORMAT) -> bool {
        let Some(host) = host() else { return false };
        let mut data = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
            Format: format,
            ..Default::default()
        };
        // SAFETY: the query writes the format support structure of the given size.
        let result = unsafe {
            host.device.CheckFeatureSupport(
                D3D12_FEATURE_FORMAT_SUPPORT,
                (&raw mut data).cast(),
                size_of_val(&data) as u32,
            )
        };
        result.is_ok()
            && data
                .Support2
                .contains(D3D12_FORMAT_SUPPORT2_OUTPUT_MERGER_LOGIC_OP)
    }
}

/// A stream's graphics state, owned (the title's stream lives only for the
/// creation call). Subobjects the stream leaves out keep D3D12's defaults.
#[derive(Clone)]
pub(crate) struct GraphicsState {
    blend: D3D12_BLEND_DESC,
    sample_mask: u32,
    rasterizer: D3D12_RASTERIZER_DESC,
    depth: D3D12_DEPTH_STENCIL_DESC,
    depth_bounds: bool,
    /// Input elements, their names pointing into `names`.
    inputs: Vec<D3D12_INPUT_ELEMENT_DESC>,
    names: Arc<[CString]>,
    strip_cut: D3D12_INDEX_BUFFER_STRIP_CUT_VALUE,
    pub(crate) topology: D3D12_PRIMITIVE_TOPOLOGY_TYPE,
    pub(crate) targets: D3D12_RT_FORMAT_ARRAY,
    /// The formats used to adjust the title's blend state before host substitution.
    title_targets: D3D12_RT_FORMAT_ARRAY,
    depth_format: DXGI_FORMAT,
    samples: DXGI_SAMPLE_DESC,
}

impl Default for GraphicsState {
    fn default() -> Self {
        let target = D3D12_RENDER_TARGET_BLEND_DESC {
            SrcBlend: D3D12_BLEND_ONE,
            DestBlend: D3D12_BLEND_ZERO,
            BlendOp: D3D12_BLEND_OP_ADD,
            SrcBlendAlpha: D3D12_BLEND_ONE,
            DestBlendAlpha: D3D12_BLEND_ZERO,
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            LogicOp: D3D12_LOGIC_OP_NOOP,
            RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
            ..Default::default()
        };
        let keep = D3D12_DEPTH_STENCILOP_DESC {
            StencilFailOp: D3D12_STENCIL_OP_KEEP,
            StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
            StencilPassOp: D3D12_STENCIL_OP_KEEP,
            StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        };
        GraphicsState {
            blend: D3D12_BLEND_DESC {
                RenderTarget: [target; 8],
                ..Default::default()
            },
            sample_mask: u32::MAX,
            rasterizer: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_BACK,
                DepthClipEnable: true.into(),
                ..Default::default()
            },
            depth: D3D12_DEPTH_STENCIL_DESC {
                DepthEnable: true.into(),
                DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
                DepthFunc: D3D12_COMPARISON_FUNC_LESS,
                StencilReadMask: 0xff,
                StencilWriteMask: 0xff,
                FrontFace: keep,
                BackFace: keep,
                ..Default::default()
            },
            depth_bounds: false,
            inputs: Vec::new(),
            names: Arc::default(),
            strip_cut: D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED,
            topology: D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED,
            targets: D3D12_RT_FORMAT_ARRAY::default(),
            title_targets: D3D12_RT_FORMAT_ARRAY::default(),
            depth_format: DXGI_FORMAT_UNKNOWN,
            samples: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
        }
    }
}

impl GraphicsState {
    pub(super) fn typed_slots(&self) -> u8 {
        self.targets
            .RTFormats
            .iter()
            .take(self.targets.NumRenderTargets as usize)
            .enumerate()
            .fold(0, |mask, (slot, f)| {
                mask | (u8::from(*f != DXGI_FORMAT_UNKNOWN) << slot)
            })
    }

    pub(crate) fn without_targets(&self, missing: u8) -> Self {
        // Les pointeurs des éléments restent valides grâce aux noms partagés.
        let mut state = self.clone();
        for (slot, f) in state.targets.RTFormats.iter_mut().enumerate() {
            if missing & (1 << slot) != 0 {
                *f = DXGI_FORMAT_UNKNOWN;
            }
        }
        state
    }

    pub(super) fn null_view(&self, slot: usize) -> Option<(DXGI_FORMAT, bool)> {
        let format = *self.targets.RTFormats.get(slot)?;
        Some((format, self.samples.Count > 1))
    }

    /// Takes in one subobject of a stream: its type, then its data at +4
    /// (+8 for those holding pointers).
    ///
    /// # Safety
    /// `at` points at a whole subobject of type `kind`.
    pub(crate) unsafe fn read(&mut self, kind: u32, at: *const u8) {
        // SAFETY: the caller passes a whole subobject.
        unsafe {
            let data = at.add(4);
            match kind {
                8 => {
                    self.blend = data.cast::<D3D12_BLEND_DESC>().read_unaligned();
                    self.adjust_blend();
                }
                9 => self.sample_mask = data.cast::<u32>().read_unaligned(),
                10 => self.rasterizer = data.cast::<D3D12_RASTERIZER_DESC>().read_unaligned(),
                11 => {
                    self.depth = data.cast::<D3D12_DEPTH_STENCIL_DESC>().read_unaligned();
                    self.depth_bounds = false;
                }
                21 => {
                    let d = data.cast::<D3D12_DEPTH_STENCIL_DESC1>().read_unaligned();
                    self.depth_bounds = d.DepthBoundsTestEnable.as_bool();
                    #[cfg(feature = "diagnostics")]
                    if self.depth_bounds {
                        BOUNDED_PIPELINES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    self.depth = D3D12_DEPTH_STENCIL_DESC {
                        DepthEnable: d.DepthEnable,
                        DepthWriteMask: d.DepthWriteMask,
                        DepthFunc: d.DepthFunc,
                        StencilEnable: d.StencilEnable,
                        StencilReadMask: d.StencilReadMask,
                        StencilWriteMask: d.StencilWriteMask,
                        FrontFace: d.FrontFace,
                        BackFace: d.BackFace,
                    };
                }
                12 => {
                    let layout = at.add(8).cast::<D3D12_INPUT_LAYOUT_DESC>().read_unaligned();
                    if layout.pInputElementDescs.is_null() {
                        return;
                    }
                    let elements = std::slice::from_raw_parts(
                        layout.pInputElementDescs,
                        layout.NumElements as usize,
                    );
                    self.names = elements
                        .iter()
                        .map(|e| CStr::from_ptr(e.SemanticName.0.cast()).to_owned())
                        .collect();
                    self.inputs = elements
                        .iter()
                        .zip(self.names.iter())
                        .map(|(e, name)| D3D12_INPUT_ELEMENT_DESC {
                            SemanticName: PCSTR(name.as_ptr().cast()),
                            Format: DXGI_FORMAT(
                                crate::resource::pc_vertex_format(e.Format.0 as u32) as i32,
                            ),
                            ..*e
                        })
                        .collect();
                }
                13 => {
                    self.strip_cut =
                        D3D12_INDEX_BUFFER_STRIP_CUT_VALUE(data.cast::<i32>().read_unaligned())
                }
                14 => {
                    self.topology =
                        D3D12_PRIMITIVE_TOPOLOGY_TYPE(data.cast::<i32>().read_unaligned())
                }
                15 => {
                    self.targets = data.cast::<D3D12_RT_FORMAT_ARRAY>().read_unaligned();
                    // The blend and the formats come in either order: each
                    // applies it, and twice changes nothing more.
                    self.title_targets = self.targets;
                    self.adjust_blend();
                    for f in &mut self.targets.RTFormats {
                        f.0 = crate::resource::pc_format(f.0 as u32) as i32;
                    }
                }
                16 => {
                    let f = data.cast::<u32>().read_unaligned();
                    self.depth_format = DXGI_FORMAT(crate::resource::pc_format(f) as i32);
                }
                17 => self.samples = data.cast::<DXGI_SAMPLE_DESC>().read_unaligned(),
                _ => {}
            }
        }
    }

    fn adjust_blend(&mut self) {
        bypass_integer_blend(&mut self.blend, &self.title_targets);
        blend_without_alpha(&mut self.blend, &self.title_targets);
    }

    /// Everything the state holds, as text: part of a pipeline's name in
    /// the library and of the key a title pipeline is started early by.
    pub(crate) fn describe(&self) -> String {
        self.describe_with(self.depth_bounds)
    }

    /// [`Self::describe`] of the state a host makes, for the library's
    /// name: the depth bounds test only where the host has one, as the
    /// pipeline's description sets it (`g.depth_bounds &&
    /// host.depth_bounds`). The title's ask alone named two descriptions
    /// alike on hosts with and without it (Codex review of d286ffd, point
    /// 5).
    pub(crate) fn describe_on(&self, host_depth_bounds: bool) -> String {
        self.describe_with(self.depth_bounds && host_depth_bounds)
    }

    fn describe_with(&self, depth_bounds: bool) -> String {
        let inputs: Vec<String> = self
            .inputs
            .iter()
            .zip(self.names.iter())
            .map(|(e, name)| {
                format!(
                    "{name:?} {} {} {} {} {} {}",
                    e.SemanticIndex,
                    e.Format.0,
                    e.InputSlot,
                    e.AlignedByteOffset,
                    e.InputSlotClass.0,
                    e.InstanceDataStepRate
                )
            })
            .collect();
        let bounds = if depth_bounds { " depth-bounds" } else { "" };
        format!(
            "{:?} {} {:?} {:?} {inputs:?} {} {} {:?} {} {:?}{bounds}",
            self.blend,
            self.sample_mask,
            self.rasterizer,
            self.depth,
            self.strip_cut.0,
            self.topology.0,
            self.targets,
            self.depth_format.0,
            self.samples
        )
    }

    /// Why the host cannot make a pipeline with this state (its target,
    /// depth and vertex formats, sample count, blending); `None` if it can.
    fn refusal(&self, caps: &impl PipelineCaps) -> Option<String> {
        use crate::resource::format::uses::*;
        let count = self.samples.Count;
        let targets = (self.targets.NumRenderTargets as usize).min(8);
        for (i, &format) in self.targets.RTFormats[..targets].iter().enumerate() {
            let f = format.0 as u32;
            if f == 0 {
                continue;
            }
            let support = caps.support1(f);
            let blend = if self.blend.IndependentBlendEnable.as_bool() {
                self.blend.RenderTarget[i]
            } else {
                self.blend.RenderTarget[0]
            };
            if support & RENDER_TARGET == 0 {
                return Some(format!("render target format {f}: no render targets"));
            }
            if blend.BlendEnable.as_bool() && support & BLENDABLE == 0 {
                return Some(format!("render target format {f}: no blending"));
            }
            if self.blend.IndependentBlendEnable.as_bool() && blend.LogicOpEnable.as_bool() {
                return Some(format!(
                    "render target format {f}: no per-target logic operations"
                ));
            }
            if blend.LogicOpEnable.as_bool() && !caps.logic_op(format) {
                return Some(format!("render target format {f}: no logic operations"));
            }
            if !caps.samples(f, count) {
                return Some(format!("render target format {f}: no {count} samples"));
            }
        }
        let depth = self.depth_format.0 as u32;
        if depth != 0 {
            if caps.support1(depth) & DEPTH_STENCIL == 0 {
                return Some(format!("depth format {depth}: no depth-stencil targets"));
            }
            if !caps.samples(depth, count) {
                return Some(format!("depth format {depth}: no {count} samples"));
            }
        }
        self.inputs.iter().find_map(|e| {
            let f = e.Format.0 as u32;
            (caps.support1(f) & VERTEX_BUFFER == 0)
                .then(|| format!("vertex format {f}: not a vertex buffer format"))
        })
    }

    /// The vertex buffer slots the input layout reads.
    fn slots(&self) -> u32 {
        self.inputs
            .iter()
            .fold(0, |m, e| m | 1 << e.InputSlot.min(31))
    }

    /// The input elements with their byte offsets worked out.
    fn elements(&self) -> Vec<Input> {
        let mut next = [0u32; 32];
        self.inputs
            .iter()
            .map(|e| {
                let slot = e.InputSlot.min(31) as usize;
                let format = e.Format.0 as u32;
                let offset = if e.AlignedByteOffset == D3D12_APPEND_ALIGNED_ELEMENT {
                    next[slot]
                } else {
                    e.AlignedByteOffset
                };
                let bytes = crate::resource::element(format).map_or(0, |(b, _)| b as u32);
                next[slot] = offset + bytes;
                Input {
                    slot: slot as u32,
                    offset,
                    format,
                    per_instance: e.InputSlotClass == D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA,
                }
            })
            .collect()
    }
}

/// One input element: vertex buffer slot, byte offset, format.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Input {
    pub(crate) slot: u32,
    pub(crate) offset: u32,
    pub(crate) format: u32,
    pub(crate) per_instance: bool,
}

/// The console's own primitive topology types.
const RECTANGLES: i32 = 19;
const QUADS: i32 = 32;

/// Typed integer colour formats. Depth/stencil views and typeless resources
/// are not integer colour targets.
fn integer_target(format: DXGI_FORMAT) -> bool {
    matches!(
        format,
        DXGI_FORMAT_R32G32B32A32_UINT
            | DXGI_FORMAT_R32G32B32A32_SINT
            | DXGI_FORMAT_R32G32B32_UINT
            | DXGI_FORMAT_R32G32B32_SINT
            | DXGI_FORMAT_R16G16B16A16_UINT
            | DXGI_FORMAT_R16G16B16A16_SINT
            | DXGI_FORMAT_R32G32_UINT
            | DXGI_FORMAT_R32G32_SINT
            | DXGI_FORMAT_R10G10B10A2_UINT
            | DXGI_FORMAT_R8G8B8A8_UINT
            | DXGI_FORMAT_R8G8B8A8_SINT
            | DXGI_FORMAT_R16G16_UINT
            | DXGI_FORMAT_R16G16_SINT
            | DXGI_FORMAT_R32_UINT
            | DXGI_FORMAT_R32_SINT
            | DXGI_FORMAT_R8G8_UINT
            | DXGI_FORMAT_R8G8_SINT
            | DXGI_FORMAT_R16_UINT
            | DXGI_FORMAT_R16_SINT
            | DXGI_FORMAT_R8_UINT
            | DXGI_FORMAT_R8_SINT
    )
}

/// GFX10 sets CB_COLORn_INFO.BLEND_BYPASS for UINT and SINT. ROP3 still
/// applies to these targets. Split shared blending only for a mixed set.
fn bypass_integer_blend(blend: &mut D3D12_BLEND_DESC, targets: &D3D12_RT_FORMAT_ARRAY) {
    let formats = &targets.RTFormats[..targets.NumRenderTargets.min(8) as usize];
    if !blend.IndependentBlendEnable.as_bool() {
        let shared = blend.RenderTarget[0];
        if !shared.BlendEnable.as_bool() || !formats.iter().any(|&f| integer_target(f)) {
            return;
        }
        if formats
            .iter()
            .all(|&f| f == DXGI_FORMAT_UNKNOWN || integer_target(f))
        {
            blend.RenderTarget[0].BlendEnable = false.into();
            return;
        }
        blend.RenderTarget = [shared; 8];
        blend.IndependentBlendEnable = true.into();
    }
    for (&f, t) in formats.iter().zip(&mut blend.RenderTarget) {
        if integer_target(f) {
            t.BlendEnable = false.into();
        }
    }
}

/// A pipeline's blending for render targets of title formats `targets`,
/// with those whose stand-in has an alpha the title's format lacks
/// ([`crate::resource::format::alpha_one`]: R9G9B9E5 as RGBA16F, B5G6R5
/// as RGBA8) neither writing alpha nor blending with the target's alpha,
/// as on the console, where that alpha reads one: in the colour factors,
/// destination alpha becomes one, its inverse zero, and saturated source
/// alpha (`min(As, 1 - Ad)`) zero. The alpha factors stay: what they make
/// is not written. Applied as a stream is read, so pipeline names in the
/// library change with it (the review of the #492 port, from dark-world).
fn blend_without_alpha(blend: &mut D3D12_BLEND_DESC, targets: &D3D12_RT_FORMAT_ARRAY) {
    blend_without(blend, targets, crate::resource::format::alpha_one);
}

/// [`blend_without_alpha`] for the title formats `lacks_alpha` names. One
/// blending for every target stays one where every target lacks alpha,
/// and becomes one per target where some target keeps its alpha, unless
/// it has a logic operation: D3D12 takes none with blending per target
/// (`IndependentBlendEnable` must be false), so such a pipeline keeps its
/// blending and writes the alpha of its targets without one.
// shortcut: a logic operation on targets with and without alpha still
// writes the stand-in's alpha (rare: no logic operation on float targets,
// R9G9B9E5's stand-in is one).
fn blend_without(
    blend: &mut D3D12_BLEND_DESC,
    targets: &D3D12_RT_FORMAT_ARRAY,
    lacks_alpha: impl Fn(u32) -> bool,
) {
    let count = (targets.NumRenderTargets as usize).min(8);
    let bound = || (0..count).filter(|&i| targets.RTFormats[i].0 != 0);
    let lacking: Vec<usize> = bound()
        .filter(|&i| lacks_alpha(targets.RTFormats[i].0 as u32))
        .collect();
    if lacking.is_empty() {
        return;
    }
    let keeping = bound().any(|i| !lacks_alpha(targets.RTFormats[i].0 as u32));
    if !blend.IndependentBlendEnable.as_bool() && keeping {
        if blend.RenderTarget[0].LogicOpEnable.as_bool() {
            return;
        }
        blend.IndependentBlendEnable = true.into();
        blend.RenderTarget = [blend.RenderTarget[0]; 8];
    }
    let slots = if blend.IndependentBlendEnable.as_bool() {
        lacking
    } else {
        vec![0]
    };
    let factor = |f: D3D12_BLEND| match f {
        D3D12_BLEND_DEST_ALPHA => D3D12_BLEND_ONE,
        D3D12_BLEND_INV_DEST_ALPHA | D3D12_BLEND_SRC_ALPHA_SAT => D3D12_BLEND_ZERO,
        other => other,
    };
    for i in slots {
        let t = &mut blend.RenderTarget[i];
        t.SrcBlend = factor(t.SrcBlend);
        t.DestBlend = factor(t.DestBlend);
        t.RenderTargetWriteMask &= !(D3D12_COLOR_WRITE_ENABLE_ALPHA.0 as u8);
    }
}

/// Whether root SRVs are bound as one-view tables of the buffer's host
/// copy (reads past its end return zeros) rather than root descriptors
/// (unbounded: a mesh shader's vertex fetch, which has no record limit,
/// read past the copy into memory the host had freed). Set
/// `XTP_ROOT_SRV_DIRECT=1` to go back.
pub(crate) fn bounded_root_srvs() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("XTP_ROOT_SRV_DIRECT").is_none())
}

/// Whether the root SRV at `slot` of a translated shader is a one-view
/// table (bound with a descriptor table) rather than a root descriptor:
/// with [`bounded_root_srvs`], every slot but the RayQuery TLAS
/// ([`TLAS_REGISTER`]), which always stays a root SRV. Its trace dispatch
/// passes the host TLAS as an indirect root SRV argument
/// ([`crate::rt::dispatch_descriptor_tlas`]), which a table cannot take,
/// and an acceleration structure has no byte range for a view to bound.
pub(crate) fn root_srv_in_table(slot: u8) -> bool {
    slot != TLAS_REGISTER && bounded_root_srvs()
}

/// The root parameter of a translated shader's root view at `slot` of
/// `kind` (a root CBV, SRV or UAV in [`ROOT_CBV_SPACE`]): a one-view table
/// for root UAVs and the root SRVs [`root_srv_in_table`] names (its range
/// pushed to `ranges`, which must outlive the parameter), else a root
/// descriptor.
#[expect(
    clippy::vec_box,
    reason = "a table parameter points at its range, which must stay in place as the vector grows"
)]
pub(crate) fn root_view(
    kind: D3D12_ROOT_PARAMETER_TYPE,
    slot: u8,
    visibility: D3D12_SHADER_VISIBILITY,
    ranges: &mut Vec<Box<D3D12_DESCRIPTOR_RANGE1>>,
) -> D3D12_ROOT_PARAMETER1 {
    if kind == D3D12_ROOT_PARAMETER_TYPE_UAV
        || kind == D3D12_ROOT_PARAMETER_TYPE_SRV && root_srv_in_table(slot)
    {
        // A one-view table: the view's size bounds the reads, which a root
        // descriptor leaves unbounded.
        ranges.push(Box::new(D3D12_DESCRIPTOR_RANGE1 {
            RangeType: if kind == D3D12_ROOT_PARAMETER_TYPE_UAV {
                D3D12_DESCRIPTOR_RANGE_TYPE_UAV
            } else {
                D3D12_DESCRIPTOR_RANGE_TYPE_SRV
            },
            NumDescriptors: 1,
            BaseShaderRegister: u32::from(slot),
            RegisterSpace: ROOT_CBV_SPACE,
            Flags: D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE
                | D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE,
            OffsetInDescriptorsFromTableStart: 0,
        }));
        // The box keeps the range in place when the vector grows.
        let range: *const D3D12_DESCRIPTOR_RANGE1 = &**ranges.last().expect("pushed");
        return D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE1 {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: range,
                },
            },
            ShaderVisibility: visibility,
        };
    }
    D3D12_ROOT_PARAMETER1 {
        ParameterType: kind,
        Anonymous: D3D12_ROOT_PARAMETER1_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                ShaderRegister: u32::from(slot),
                RegisterSpace: ROOT_CBV_SPACE,
                Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
            },
        },
        ShaderVisibility: visibility,
    }
}

/// One stage of a host pipeline and where its bindings go.
pub(crate) struct Stage {
    pub(crate) shader: Arc<Shader>,
    /// A stage of a pipeline with a geometry shader, a merged ES + GS
    /// half ([`merged_half`]): the vertex stage's translation of its
    /// container's export shader (ES), which runs there instead of its
    /// vertex shader, and the geometry stage's own.
    pub(crate) merged: Option<Arc<Translated>>,
    /// The root parameter of its user data, and its size in dwords.
    pub(crate) user_data: Option<(u32, u32)>,
    /// Root parameters of its root views: (title root parameter, ours).
    pub(crate) root_cbvs: Vec<(u8, u32)>,
    pub(crate) root_srvs: Vec<(u8, u32)>,
    pub(crate) root_uavs: Vec<(u8, u32)>,
    /// The root parameter of the runtime stride table (`t0` in
    /// [`STRIDE_SPACE`]), if its shader reads one: a root SRV of the view
    /// heap's table, which draws bind or are dropped without.
    pub(crate) stride_table: Option<u32>,
}

impl Stage {
    pub(crate) fn translated(&self) -> &Translated {
        match &self.merged {
            Some(t) => t,
            None => self
                .shader
                .translated()
                .expect("host pipelines are made from translated shaders"),
        }
    }
}

/// Merged halves' translations ([`merged_half`]), by container hash and
/// half (true: the export shader).
type MergedHalves =
    std::collections::HashMap<(String, bool), Arc<std::sync::OnceLock<Option<Arc<Translated>>>>>;
static MERGED_HALVES: std::sync::Mutex<Option<MergedHalves>> = std::sync::Mutex::new(None);

/// A half of the console's merged ES + GS stage, translated for the host
/// (`docs/analysis/es-gs.md`): of a vertex shader container its export
/// shader (`es`), which a pipeline with a geometry shader runs as its
/// vertex shader, the console's ES writing the ring its GS reads; of a
/// geometry shader container its GS. Each with what its own console
/// program says of itself (the resources its tables hold): the container's
/// [`Shader::native`] reads its vertex shader, and nothing of a GS
/// container. Made once per container and half; `None` (logged once) when
/// it does not translate.
fn merged_half(shader: &Shader, es: bool) -> Option<Arc<Translated>> {
    let once = {
        let mut made = MERGED_HALVES.lock().unwrap_or_else(|e| e.into_inner());
        made.get_or_insert_with(Default::default)
            .entry((shader.hash.clone(), es))
            .or_default()
            .clone()
    };
    once.get_or_init(|| {
        let half = if es { "export" } else { "geometry" };
        let started = std::time::Instant::now();
        let done = merged_translation(&shader.container, es);
        let ms = started.elapsed().as_millis() as u64;
        match &done {
            Ok(_) => tracing::info!(hash = %shader.hash, half, ms, "merged half translated"),
            Err(e) => tracing::warn!(
                hash = %shader.hash,
                half,
                error = %e,
                ms,
                "merged half does not translate; its pipelines with a geometry shader are dropped"
            ),
        }
        done.ok().map(Arc::new)
    })
    .clone()
}

/// [`merged_half`]'s translation of `container`.
fn merged_translation(container: &[u8], es: bool) -> Result<Translated, String> {
    use xtp_shader::HwStage;
    let target = crate::device::host_target();
    let hash = blake3::hash(container).to_hex()[..16].to_owned();
    let options = super::self_sized_options(&hash);
    let (host, stage) = if es {
        let host = crate::shader_cache::host_es_shader_with(container, target, options)?;
        (host, HwStage::Es)
    } else {
        let host = crate::shader_cache::host_shader_with(container, target, options)?;
        (host, HwStage::Gs)
    };
    if host.stage != stage {
        return Err(format!(
            "the container's {:?} program translated",
            host.stage
        ));
    }
    let c = xtp_shader::Container::parse(container).map_err(|e| e.to_string())?;
    let dxil = c.part(xtp_shader::FourCC::DXIL).ok_or("no DXIL part")?.data;
    let native = xtp_shader::native_programs(dxil)
        .map_err(|e| e.to_string())?
        .into_iter()
        .find(|p| p.stage == stage)
        .ok_or("no program of its half")?;
    Ok(Translated {
        host,
        used: native.used,
        wave64: native.wave64,
    })
}

pub(crate) struct HostPipeline {
    pub(crate) dispatch_signature: std::sync::OnceLock<Option<ID3D12CommandSignature>>,
    /// A compute pipeline (else graphics).
    pub(crate) compute: bool,
    /// A mesh pipeline (graphics without vertex input).
    pub(crate) mesh: bool,
    pub(crate) pso: ID3D12PipelineState,
    pub(crate) root: ID3D12RootSignature,
    /// Vertex (or mesh) stage, then pixel and geometry stages, or the
    /// compute stage.
    pub(crate) stages: Vec<Stage>,
    /// The vertex buffer slots it reads (bit per slot).
    pub(crate) slots: u32,
    pub(crate) inputs: Vec<Input>,
    /// Its shaders, for reports of slow submissions.
    pub(crate) name: String,
    /// With `XTP_SHADER_LOOP_COUNT`, where a shader of it counts the lanes
    /// stopped at a loop's limit or budget: the root parameter of the
    /// counter (a root UAV, `u0` in [`LOOP_COUNTER_SPACE`], for every
    /// stage) and the pipeline's slot, whose counter (16-byte aligned)
    /// draws and dispatches bind there (`crate::loopstat`).
    pub(crate) loop_counter: Option<(u32, u32)>,
}

/// Pipelines that got no host pipeline since the start, by reason.
static REFUSED: std::sync::Mutex<std::collections::BTreeMap<String, u64>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());

#[cfg(feature = "diagnostics")]
thread_local! {
    /// The reason of the last refusal on this thread (diagnostics: what
    /// draws dropped for want of the pipeline are counted under).
    static LAST_REFUSAL: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

/// The reason of the last refusal on this thread since the last call
/// (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) fn take_refusal() -> Option<String> {
    LAST_REFUSAL.with(|r| r.borrow_mut().take())
}

/// Counts a pipeline that gets no host pipeline for `reason`; true for the
/// first of that reason.
fn note_refusal(reason: String) -> bool {
    #[cfg(feature = "diagnostics")]
    LAST_REFUSAL.with(|r| *r.borrow_mut() = Some(reason.clone()));
    let mut refused = REFUSED.lock().unwrap_or_else(|e| e.into_inner());
    let n = refused.entry(reason).or_default();
    *n += 1;
    *n == 1
}

/// Counts a pipeline that gets no host pipeline, before any driver call;
/// logs the first of each reason.
fn refuse(reason: impl Into<String>, shaders: &[Arc<Shader>]) -> Option<HostPipeline> {
    let reason = reason.into();
    if note_refusal(reason.clone()) {
        tracing::warn!(?shaders, "no host pipeline: {reason} (logged once)");
    }
    None
}

/// The pipelines refused since the start, by reason, most first (`None`
/// if none was).
pub(crate) fn refusals() -> Option<String> {
    let refused = REFUSED.lock().unwrap_or_else(|e| e.into_inner());
    if refused.is_empty() {
        return None;
    }
    let mut counts: Vec<(&String, &u64)> = refused.iter().collect();
    counts.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
    let counts: Vec<String> = counts.iter().map(|(r, n)| format!("{n} {r}")).collect();
    Some(format!(
        "host pipelines refused so far: {}",
        counts.join("; ")
    ))
}

/// What a translated shader asks of the host: the wave lanes it runs in
/// (`PSV0`'s expected lane counts; 0 and `u32::MAX`: any), its shader
/// model (the `DXIL` program header's version), and whether it is a mesh
/// program that needs the console's 64-lane waves whole.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Needs {
    pub(crate) lanes: (u32, u32),
    pub(crate) model: (u32, u32),
    /// Why a console mesh program that runs in 64-lane waves cannot run on
    /// a host without them, if it cannot: its translation declares no wave
    /// size (only compute translations do), so `lanes` does not say so.
    ///
    /// The translator decides (`xtp-shader` `emit/lower/mesh.rs`,
    /// `docs/analysis/mesh-wave64.md`). Translated for a target without
    /// 64-lane waves, which the host's is on NVIDIA, a wave64 mesh program
    /// gets its console wave's first lane by proof or through group-shared
    /// memory, and is refused there when neither holds. What stays here:
    ///
    /// * `XTP_WAVE64_MESH=refuse` ([`wave64_mesh_refused`]): VIctory #578's
    ///   refusal of every such program, for an A/B in game (the RTX 5090
    ///   VIctory saw removed as hung ran a translator without that form);
    /// * a translation made for a target with 64-lane waves
    ///   (`XTP_HOST_LANES`), which takes the console waves to be whole, on
    ///   a host without them.
    pub(crate) wave64_mesh: Option<&'static str>,
}

/// What a wave64 mesh program is refused as with `XTP_WAVE64_MESH=refuse`
/// ([`Needs::wave64_mesh`]).
const MESH_REFUSED: &str =
    "a mesh program written for 64-lane waves (XTP_WAVE64_MESH=refuse, as VIctory #578)";

/// What a wave64 mesh program translated for a target with 64-lane waves
/// is refused as on a host without them ([`Needs::wave64_mesh`]).
const MESH_WAVE64: &str = "a mesh program translated for 64-lane waves (XTP_HOST_LANES)";

/// Whether `XTP_WAVE64_MESH=refuse` refuses every wave64 mesh program on
/// hosts without 64-lane waves ([`Needs::wave64_mesh`]); read once, and
/// logged with what it means on this host.
pub(crate) fn wave64_mesh_refused() -> bool {
    static REFUSE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *REFUSE.get_or_init(|| {
        let refuse = std::env::var("XTP_WAVE64_MESH").is_ok_and(|v| v == "refuse");
        // The host's own lanes, not the translations' target: mesh
        // translations declare no wave size.
        match crate::device::host_wave_lanes() {
            Some((min, max)) if max < 64 && refuse => tracing::warn!(
                lanes = format_args!("{min} to {max}"),
                "XTP_WAVE64_MESH=refuse: wave64 mesh programs are refused on a host without \
                 64-lane waves (VIctory #578)"
            ),
            Some((min, max)) if max < 64 => tracing::info!(
                lanes = format_args!("{min} to {max}"),
                "the host has no 64-lane waves: wave64 mesh programs read their console wave's \
                 first lane by proof or through group-shared memory, or do not translate \
                 (XTP_WAVE64_MESH=refuse refuses them all)"
            ),
            _ => {}
        }
        refuse
    })
}

impl Needs {
    /// What a translated `stage` shader needs; `None` if its container does
    /// not say and it is no wave64 mesh program the host cannot run (that
    /// check does not depend on the container).
    pub(crate) fn of_translated(stage: &str, shader: &Translated) -> Option<Needs> {
        Self::translated_with(stage, shader, Self::mesh_switches())
    }

    /// [`Needs::of_translated`] with `switches` ([`Needs::mesh_switches`]).
    fn translated_with(stage: &str, shader: &Translated, switches: (bool, bool)) -> Option<Needs> {
        let wave64_mesh = Self::mesh_reason(stage, shader.wave64, switches);
        match Needs::of(&shader.host.dxil) {
            Some(needs) => Some(Needs {
                wave64_mesh,
                ..needs
            }),
            None => wave64_mesh.map(|_| Needs {
                lanes: (0, u32::MAX),
                model: (0, 0),
                wave64_mesh,
            }),
        }
    }

    /// What a `stage` shader whose console program says it of itself
    /// (`native`) needs, as far as that decides: whether it is a wave64 mesh
    /// program the host cannot run, with any lanes and shader model (mesh
    /// translations declare no wave size, so the translation would not say
    /// more on that); `None` if it is none.
    pub(crate) fn of_native(stage: &str, native: &super::Native) -> Option<Needs> {
        Self::native_with(stage, native, Self::mesh_switches())
    }

    /// [`Needs::of_native`] with `switches` ([`Needs::mesh_switches`]).
    fn native_with(stage: &str, native: &super::Native, switches: (bool, bool)) -> Option<Needs> {
        Self::mesh_reason(stage, native.wave64, switches).map(|what| Needs {
            lanes: (0, u32::MAX),
            model: (0, 0),
            wave64_mesh: Some(what),
        })
    }

    /// What decides [`Needs::wave64_mesh`] besides the program: whether
    /// `XTP_WAVE64_MESH=refuse` is set, and whether shaders are translated
    /// for a target with 64-lane waves ([`crate::device::host_target`]).
    fn mesh_switches() -> (bool, bool) {
        (wave64_mesh_refused(), crate::device::host_target().runs(64))
    }

    /// [`Needs::wave64_mesh`] for a `stage` shader whose console program
    /// runs in 64-lane waves or not (`wave64`), with `(refuse, target_64)`
    /// ([`Needs::mesh_switches`]).
    fn mesh_reason(
        stage: &str,
        wave64: bool,
        (refuse, target_64): (bool, bool),
    ) -> Option<&'static str> {
        if stage != "mesh" || !wave64 {
            None
        } else if refuse {
            Some(MESH_REFUSED)
        } else if target_64 {
            Some(MESH_WAVE64)
        } else {
            None
        }
    }

    /// What the shader in a DXIL container needs; `None` if the container
    /// does not say.
    pub(crate) fn of(dxil: &[u8]) -> Option<Needs> {
        use xtp_shader::{Container, FourCC};
        let c = Container::parse(dxil).ok()?;
        let word = |data: &[u8], at: usize| {
            Some(u32::from_le_bytes(data.get(at..at + 4)?.try_into().ok()?))
        };
        let psv = c.part(FourCC::PSV0)?.data;
        // The runtime information's size, then its stage-specific 16
        // bytes, then the lane counts.
        let lanes = if word(psv, 0)? >= 24 {
            (word(psv, 20)?, word(psv, 24)?)
        } else {
            (0, u32::MAX)
        };
        let version = word(c.part(FourCC::DXIL)?.data, 0)?;
        Some(Needs {
            lanes,
            model: (version >> 4 & 0xf, version & 0xf),
            wave64_mesh: None,
        })
    }

    /// Why a host with waves of `host` lanes (min, max) and shader model
    /// `model` cannot run it; `None` if it can. One reason, the first of:
    /// the wave size its translation declares (a wave64 program the
    /// translator made 32-lane declares `WaveSize(32)` and runs on a 32-lane
    /// host); a wave64 mesh program that needs its waves whole
    /// ([`Needs::wave64_mesh`]), which declares none, on a host without
    /// 64-lane waves; its shader model.
    pub(crate) fn refusal(&self, host: (u32, u32), model: (u32, u32)) -> Option<String> {
        let (min, max) = self.lanes;
        if min != 0 && (max < host.0 || min > host.1) {
            let wanted = if min == max {
                format!("{min}-lane waves")
            } else {
                format!("waves of {min} to {max} lanes")
            };
            return Some(format!(
                "needs {wanted}; the host has {} to {}",
                host.0, host.1
            ));
        }
        // shortcut: a host whose range holds 64 lanes (AMD RDNA: 32 to 64)
        // is taken to run mesh programs in 64-lane waves. D3D12 gives mesh
        // shaders no wave size to ask for, so its driver may pick 32 (VIctory
        // #578 takes the same; refusing there would drop every mesh program
        // on AMD too).
        if let Some(what) = self.wave64_mesh
            && min == 0
            && host.1 < 64
        {
            return Some(format!("is {what}; the host has {} to {}", host.0, host.1));
        }
        (self.model > model).then(|| {
            format!(
                "needs shader model {}.{}; the host has {}.{}",
                self.model.0, self.model.1, model.0, model.1
            )
        })
    }
}

/// Why the host cannot run the translation of a `stage` shader; `None` if
/// it can (or the host's lanes are unknown).
fn shader_refusal(stage: &str, shader: &Translated) -> Option<String> {
    let needs = Needs::of_translated(stage, shader)?;
    let lanes = crate::device::host_wave_lanes()?;
    let model = crate::device::host_shader_model()?;
    needs.refusal(lanes, model)
}

/// Why the host cannot run a `stage` shader, known from its console
/// program alone ([`Needs::of_native`]): the reason [`shader_refusal`]
/// would give once it is translated. `None` if only the translation can
/// say (or the host's lanes are unknown).
fn native_refusal(stage: &str, shader: &Shader) -> Option<String> {
    let needs = Needs::of_native(stage, shader.native().ok()?)?;
    let lanes = crate::device::host_wave_lanes()?;
    let model = crate::device::host_shader_model()?;
    needs.refusal(lanes, model)
}

impl HostPipeline {
    pub(crate) fn group_base(&self) -> Option<(u32, u32)> {
        let stage = self.stages.first()?;
        Some((stage.user_data?.0, stage.translated().host.group_base?))
    }

    /// A host pipeline for vertex and pixel shaders with the graphics
    /// state `g`, or for one compute shader.
    pub(crate) fn new(shaders: &[Arc<Shader>], g: Option<&GraphicsState>) -> Option<HostPipeline> {
        let find = |stage| shaders.iter().find(|s| s.stage == stage);
        let compute = match shaders {
            [cs] if cs.stage == "compute" => Some(cs),
            _ => None,
        };
        if compute.is_none()
            && shaders
                .iter()
                .any(|s| !matches!(s.stage, "vertex" | "pixel" | "mesh" | "geometry"))
        {
            return refuse(
                "only vertex, mesh, geometry, pixel and compute stages are translated",
                shaders,
            );
        }
        let mesh = find("mesh");
        // A geometry shader: the vertex stage runs its container's export
        // shader (ES), which writes the ring the GS reads on the console.
        let geometry = find("geometry");
        if geometry.is_some() && mesh.is_some() {
            return refuse("a geometry shader beside a mesh shader", shaders);
        }
        let stages = if let Some(cs) = compute {
            vec![(cs, D3D12_SHADER_VISIBILITY_ALL)]
        } else {
            let first = match (mesh, find("vertex")) {
                (Some(ms), _) => (ms, D3D12_SHADER_VISIBILITY_MESH),
                (None, Some(vs)) => (vs, D3D12_SHADER_VISIBILITY_VERTEX),
                (None, None) => return refuse("no vertex or mesh shader", shaders),
            };
            let mut stages = vec![first];
            stages.extend(find("pixel").map(|ps| (ps, D3D12_SHADER_VISIBILITY_PIXEL)));
            stages.extend(geometry.map(|gs| (gs, D3D12_SHADER_VISIBILITY_GEOMETRY)));
            stages
        };
        if mesh.is_some() && !host()?.mesh_shaders() {
            return refuse(
                "the host GPU has no mesh shaders (or no shader model 6.6)",
                shaders,
            );
        }
        // What the console programs refuse by themselves, before any
        // translation: one takes a worker up to seconds, and in the world
        // the workers spent them on wave64 mesh programs refused right
        // after (a quarter of the world's shaders, docs/analysis/drops.md),
        // while the pipelines that would run waited behind them.
        for (s, _) in &stages {
            if let Some(why) = native_refusal(s.stage, s) {
                return refuse(format!("{} shader {why}", s.stage), shaders);
            }
        }
        // With a geometry shader, the vertex and geometry stages are a
        // merged ES + GS pair's halves: their own translations.
        let mut merged: Vec<Option<Arc<Translated>>> = vec![None; stages.len()];
        if geometry.is_some() {
            for ((s, _), m) in stages.iter().zip(&mut merged) {
                if matches!(s.stage, "vertex" | "geometry") {
                    *m = merged_half(s, s.stage == "vertex");
                    if m.is_none() {
                        let why =
                            format!("its {} shader's merged half does not translate", s.stage);
                        return refuse(why, shaders);
                    }
                }
            }
        }
        let translations: Vec<&Translated> = stages
            .iter()
            .zip(&merged)
            .map(|((s, _), m)| m.as_deref().or_else(|| s.translated()))
            .collect::<Option<_>>()?;
        for ((s, _), t) in stages.iter().zip(&translations) {
            if let Some(why) = shader_refusal(s.stage, t) {
                return refuse(format!("{} shader {why}", s.stage), shaders);
            }
        }
        let host = host()?;

        // The root signature the translated shaders' bindings define.
        let mut params: Vec<D3D12_ROOT_PARAMETER1> = Vec::new();
        // Ranges the table parameters point at, alive until the root
        // signature is made.
        let mut ranges: Vec<Box<D3D12_DESCRIPTOR_RANGE1>> = Vec::new();
        let mut samplers = Vec::new();
        let mut bound = Vec::new();
        for (k, &(shader, visibility)) in stages.iter().enumerate() {
            let t = &translations[k].host;
            let mut stage = Stage {
                shader: shader.clone(),
                merged: merged[k].clone(),
                user_data: None,
                root_cbvs: Vec::new(),
                root_srvs: Vec::new(),
                root_uavs: Vec::new(),
                stride_table: None,
            };
            let dwords = t.user_data_dwords();
            if dwords > 0 {
                stage.user_data = Some((params.len() as u32, dwords));
                params.push(D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Constants: D3D12_ROOT_CONSTANTS {
                            ShaderRegister: 0,
                            RegisterSpace: USER_DATA_SPACE,
                            Num32BitValues: dwords,
                        },
                    },
                    ShaderVisibility: visibility,
                });
            }
            for (slots, kind, out) in [
                (
                    &t.root_cbvs,
                    D3D12_ROOT_PARAMETER_TYPE_CBV,
                    &mut stage.root_cbvs,
                ),
                (
                    &t.root_srvs,
                    D3D12_ROOT_PARAMETER_TYPE_SRV,
                    &mut stage.root_srvs,
                ),
                (
                    &t.root_uavs,
                    D3D12_ROOT_PARAMETER_TYPE_UAV,
                    &mut stage.root_uavs,
                ),
            ] {
                for &slot in slots {
                    out.push((slot, params.len() as u32));
                    params.push(root_view(kind, slot, visibility, &mut ranges));
                }
            }
            if t.stride_table {
                // A root SRV: the table covers every slot of the heap, and a
                // shader reads it at the slot of a handle it makes there.
                // The host rewrites entries while lists that read it run.
                stage.stride_table = Some(params.len() as u32);
                params.push(D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                            ShaderRegister: 0,
                            RegisterSpace: STRIDE_SPACE,
                            Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                        },
                    },
                    ShaderVisibility: visibility,
                });
            }
            for (n, s) in t.static_samplers.iter().enumerate() {
                samplers.push(D3D12_STATIC_SAMPLER_DESC {
                    Filter: D3D12_FILTER(s.filter as i32),
                    AddressU: D3D12_TEXTURE_ADDRESS_MODE(s.address[0] as i32),
                    AddressV: D3D12_TEXTURE_ADDRESS_MODE(s.address[1] as i32),
                    AddressW: D3D12_TEXTURE_ADDRESS_MODE(s.address[2] as i32),
                    MipLODBias: s.mip_lod_bias,
                    MaxAnisotropy: s.max_anisotropy,
                    ComparisonFunc: D3D12_COMPARISON_FUNC(s.comparison as i32),
                    BorderColor: D3D12_STATIC_BORDER_COLOR(s.border as i32),
                    MinLOD: s.min_lod,
                    MaxLOD: s.max_lod,
                    ShaderRegister: n as u32,
                    RegisterSpace: SAMPLER_SPACE,
                    ShaderVisibility: visibility,
                });
            }
            bound.push(stage);
        }
        // XTP_SHADER_LOOP_COUNT: a shader that counts the lanes it stops at a
        // limit writes the pipeline's counter, a root UAV all stages see
        // (`crate::loopstat`).
        let mut loop_counter = None;
        if translations.iter().any(|t| t.host.loop_counter()) {
            let Some(slot) = crate::loopstat::slot(&format!("{shaders:?}")) else {
                return refuse("its loop-limit counter has no buffer", shaders);
            };
            loop_counter = Some((params.len() as u32, slot));
            params.push(D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                        ShaderRegister: 0,
                        RegisterSpace: LOOP_COUNTER_SPACE,
                        Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            });
        }
        let (root, root_blob) =
            match root_signature(&host.device, &params, &samplers, compute.is_some()) {
                Ok(root) => root,
                Err(e) => {
                    tracing::warn!(error = %e, ?shaders, "host root signature refused");
                    note_refusal(format!(
                        "the driver refused its root signature ({})",
                        e.code()
                    ));
                    return None;
                }
            };
        let code = |i: usize| {
            translations
                .get(i)
                .map_or(D3D12_SHADER_BYTECODE::default(), |t| {
                    D3D12_SHADER_BYTECODE {
                        pShaderBytecode: t.host.dxil.as_ptr().cast(),
                        BytecodeLength: t.host.dxil.len(),
                    }
                })
        };
        if compute.is_some() {
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                // SAFETY: a borrowed pointer the description never releases.
                pRootSignature: unsafe { std::mem::transmute_copy(&root) },
                CS: code(0),
                ..Default::default()
            };
            let name = super::library::name("compute", &root_blob, &[&code_bytes(&code(0))], "");
            let made = super::library::pipeline(
                &name,
                // SAFETY: the description points at live data.
                |library, name| unsafe { library.LoadComputePipeline(name, &desc) },
                // SAFETY: as above.
                || unsafe { host.device.CreateComputePipelineState(&desc) },
            );
            let pso = match made {
                Ok(pso) => pso,
                Err(e) => {
                    tracing::warn!(error = %e, ?shaders, "host pipeline refused");
                    host.log_messages();
                    note_refusal(format!("the driver refused it ({})", e.code()));
                    return None;
                }
            };
            tracing::info!(?shaders, "host pipeline");
            crate::host::name(&pso, || format!("{shaders:?}"));
            return Some(HostPipeline {
                dispatch_signature: Default::default(),
                compute: true,
                mesh: false,
                pso,
                root,
                stages: bound,
                name: format!("{shaders:?}"),
                slots: 0,
                inputs: Vec::new(),
                loop_counter,
            });
        }
        let Some(g) = g else {
            return refuse("graphics state not kept (plain description)", shaders);
        };
        if let Some(why) = g.refusal(&crate::resource::format::HostCaps) {
            return refuse(why, shaders);
        }
        // The state the host makes names the pipeline in the library.
        let state = g.describe_on(host.depth_bounds);
        if mesh.is_some() {
            let name = super::library::name(
                "mesh",
                &root_blob,
                &[&code_bytes(&code(0)), &code_bytes(&code(1))],
                &state,
            );
            let stages = MeshStages::new(code(0), code(1), &translations[0].host.dxil);
            // SAFETY: trois sous-objets alignés, les shaders restent vivants pendant l'appel.
            let made = unsafe { stream_pipeline(host, &root, g, stages, &name) };
            let pso = match made {
                Ok(pso) => pso,
                Err(e) => {
                    tracing::warn!(error = %e, ?shaders, "host mesh pipeline refused");
                    host.log_messages();
                    note_refusal(format!("the driver refused it ({})", e.code()));
                    return None;
                }
            };
            tracing::info!(?shaders, "host mesh pipeline");
            crate::host::name(&pso, || format!("{shaders:?}"));
            return Some(HostPipeline {
                dispatch_signature: Default::default(),
                compute: false,
                mesh: true,
                pso,
                root,
                stages: bound,
                name: format!("{shaders:?}"),
                slots: 0,
                inputs: Vec::new(),
                loop_counter,
            });
        }

        // The stages after the vertex stage, by name: a pipeline may have no
        // pixel stage, and the geometry stage comes after it.
        let at = |name: &str| stages.iter().position(|(s, _)| s.stage == name);
        let ps_at = at("pixel");
        let pixel = ps_at.map_or_else(D3D12_SHADER_BYTECODE::default, code);
        // The console's rectangles whose vertex shader has no per-vertex
        // input: a geometry stage makes each one's fourth corner.
        let gs = if g.topology.0 == RECTANGLES {
            if geometry.is_some() {
                return refuse("rectangles drawn through a geometry shader", shaders);
            }
            let dxil = |i: Option<usize>| {
                let t = i.and_then(|i| translations.get(i));
                t.map(|t| t.host.dxil.as_slice())
            };
            dxil(Some(0))
                .and_then(|vs| crate::rectangles::geometry_stage(&g.elements(), vs, dxil(ps_at)))
        } else {
            None
        };
        // The title's geometry shader, else the rectangles'.
        let title_gs = at("geometry").map(code);
        let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            // SAFETY: a borrowed pointer the description never releases.
            pRootSignature: unsafe { std::mem::transmute_copy(&root) },
            VS: code(0),
            GS: title_gs.unwrap_or_else(|| {
                gs.as_deref()
                    .map_or_else(D3D12_SHADER_BYTECODE::default, |b| D3D12_SHADER_BYTECODE {
                        pShaderBytecode: b.as_ptr().cast(),
                        BytecodeLength: b.len(),
                    })
            }),
            PS: pixel,
            BlendState: g.blend,
            SampleMask: g.sample_mask,
            RasterizerState: g.rasterizer,
            DepthStencilState: g.depth,
            InputLayout: D3D12_INPUT_LAYOUT_DESC {
                pInputElementDescs: g.inputs.as_ptr(),
                NumElements: g.inputs.len() as u32,
            },
            IBStripCutValue: g.strip_cut,
            // The console's rectangle and quad types draw as triangles on
            // the host (see `draw`).
            PrimitiveTopologyType: match g.topology.0 {
                RECTANGLES | QUADS => D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                _ => g.topology,
            },
            NumRenderTargets: g.targets.NumRenderTargets,
            RTVFormats: g.targets.RTFormats,
            DSVFormat: g.depth_format,
            SampleDesc: g.samples,
            ..Default::default()
        };
        let (vs_code, ps_code) = (code_bytes(&code(0)), code_bytes(&pixel));
        let gs_code = title_gs.as_ref().map(code_bytes);
        // The stage's bytes are in the name only when there is one: the other
        // pipelines keep their names in the pipeline library.
        let mut codes = vec![vs_code.as_slice(), ps_code.as_slice()];
        codes.extend(gs.as_deref());
        codes.extend(gs_code.as_deref());
        let name = super::library::name("graphics", &root_blob, &codes, &state);
        let make = |desc: &D3D12_GRAPHICS_PIPELINE_STATE_DESC, name: &str| {
            if g.depth_bounds && host.depth_bounds {
                #[repr(C)]
                struct VertexStages {
                    vs: Sub<D3D12_SHADER_BYTECODE>,
                    gs: Sub<D3D12_SHADER_BYTECODE>,
                    ps: Sub<D3D12_SHADER_BYTECODE>,
                    layout: Sub<D3D12_INPUT_LAYOUT_DESC>,
                    strip_cut: Sub<D3D12_INDEX_BUFFER_STRIP_CUT_VALUE>,
                    topology: Sub<D3D12_PRIMITIVE_TOPOLOGY_TYPE>,
                }
                let stages = VertexStages {
                    vs: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, desc.VS),
                    gs: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_GS, desc.GS),
                    ps: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, desc.PS),
                    layout: Sub(
                        D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT,
                        desc.InputLayout,
                    ),
                    strip_cut: Sub(
                        D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_IB_STRIP_CUT_VALUE,
                        desc.IBStripCutValue,
                    ),
                    topology: Sub(
                        D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY,
                        desc.PrimitiveTopologyType,
                    ),
                };
                // SAFETY: VertexStages contient les sous-objets alignés de cette description.
                return unsafe { stream_pipeline(host, &root, g, stages, name) };
            }
            super::library::pipeline(
                name,
                // SAFETY: la racine et les données de la description sont vivantes.
                |library, name| unsafe { library.LoadGraphicsPipeline(name, desc) },
                // SAFETY: mêmes données que pour le chargement.
                || unsafe { host.device.CreateGraphicsPipelineState(desc) },
            )
        };
        let made = make(&desc, &name);
        // Refused with the stage: made without it, as before it (each
        // rectangle one triangle).
        let made = match made {
            Err(e) if gs.is_some() => {
                tracing::warn!(
                    error = %e,
                    ?shaders,
                    "host pipeline refused with the rectangles' geometry stage: made without it"
                );
                host.log_messages();
                let plain = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                    GS: D3D12_SHADER_BYTECODE::default(),
                    ..desc
                };
                let codes = [vs_code.as_slice(), ps_code.as_slice()];
                let name = super::library::name("graphics", &root_blob, &codes, &state);
                make(&plain, &name)
            }
            made => made,
        };
        let pso = match made {
            Ok(pso) => pso,
            Err(e) => {
                tracing::warn!(error = %e, ?shaders, "host pipeline refused");
                host.log_messages();
                note_refusal(format!("the driver refused it ({})", e.code()));
                return None;
            }
        };
        tracing::info!(?shaders, "host pipeline");
        crate::host::name(&pso, || format!("{shaders:?}"));
        tracing::debug!(
            ?shaders,
            depth = ?g.depth,
            dsv = g.depth_format.0,
            targets = ?&g.targets.RTFormats[..g.targets.NumRenderTargets.min(8) as usize],
            blend = ?g.blend.RenderTarget[0],
            cull = g.rasterizer.CullMode.0,
            "host graphics state"
        );
        Some(HostPipeline {
            dispatch_signature: Default::default(),
            compute: false,
            mesh: false,
            pso,
            root,
            stages: bound,
            name: format!("{shaders:?}"),
            slots: g.slots(),
            inputs: g.elements(),
            loop_counter,
        })
    }
}

/// Chaque sous-objet D3D12 commence sur une frontière de la taille d’un pointeur.
#[repr(C, align(8))]
struct Sub<T>(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE, T);

/// A pipeline as a subobject stream: its root signature, its `stages`,
/// and the graphics state the mesh and depth-bounds pipelines share.
#[repr(C)]
struct Stream<S> {
    root: Sub<*mut std::ffi::c_void>,
    stages: S,
    blend: Sub<D3D12_BLEND_DESC>,
    mask: Sub<u32>,
    raster: Sub<D3D12_RASTERIZER_DESC>,
    depth: Sub<D3D12_DEPTH_STENCIL_DESC1>,
    targets: Sub<D3D12_RT_FORMAT_ARRAY>,
    dsv: Sub<DXGI_FORMAT>,
    samples: Sub<DXGI_SAMPLE_DESC>,
}

/// Portage de victory-fork#82 de Kaivyyy (c96b4f2, GPL-3.0).
/// Les pipelines de maillage et de sommets partagent l’état de rastérisation.
fn stream_of<S>(
    host: &crate::host::Host,
    root: &ID3D12RootSignature,
    g: &GraphicsState,
    stages: S,
) -> Stream<S> {
    use windows::core::Interface;
    Stream {
        root: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE,
            root.as_raw(),
        ),
        stages,
        blend: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, g.blend),
        mask: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK,
            g.sample_mask,
        ),
        raster: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, g.rasterizer),
        depth: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1,
            D3D12_DEPTH_STENCIL_DESC1 {
                DepthEnable: g.depth.DepthEnable,
                DepthWriteMask: g.depth.DepthWriteMask,
                DepthFunc: g.depth.DepthFunc,
                StencilEnable: g.depth.StencilEnable,
                StencilReadMask: g.depth.StencilReadMask,
                StencilWriteMask: g.depth.StencilWriteMask,
                FrontFace: g.depth.FrontFace,
                BackFace: g.depth.BackFace,
                DepthBoundsTestEnable: (g.depth_bounds && host.depth_bounds).into(),
            },
        ),
        targets: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS,
            g.targets,
        ),
        dsv: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT,
            g.depth_format,
        ),
        samples: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, g.samples),
    }
}

impl<S> Stream<S> {
    /// The description of this stream, which must stay in place while it
    /// is used.
    fn desc(&mut self) -> D3D12_PIPELINE_STATE_STREAM_DESC {
        D3D12_PIPELINE_STATE_STREAM_DESC {
            SizeInBytes: size_of::<Self>(),
            pPipelineStateSubobjectStream: std::ptr::from_mut(self).cast(),
        }
    }
}

/// A mesh pipeline's stages: its mesh and pixel programs, and the
/// primitive topology its mesh program outputs. D3D12 keeps that topology
/// in the pipeline it makes from a stream without one, so that a load from
/// the library with the stream without it did not match the entry (D3D12
/// message 972, then 970 at the store under the same name): every mesh
/// pipeline was compiled again in every session (on the L4, 2026-11-25,
/// about as many made as mesh pipelines drawn, before and after 5943ac5).
#[repr(C)]
struct MeshStages(
    Sub<D3D12_SHADER_BYTECODE>,
    Sub<D3D12_SHADER_BYTECODE>,
    Sub<D3D12_PRIMITIVE_TOPOLOGY_TYPE>,
);

impl MeshStages {
    /// The mesh program `ms`, whose translation is `ms_dxil`, and the pixel
    /// program `ps`.
    fn new(ms: D3D12_SHADER_BYTECODE, ps: D3D12_SHADER_BYTECODE, ms_dxil: &[u8]) -> MeshStages {
        MeshStages(
            Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, ms),
            Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, ps),
            Sub(
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY,
                mesh_topology(ms_dxil),
            ),
        )
    }
}

/// The primitive topology the mesh program in the DXIL container `dxil`
/// outputs: its `PSV0` runtime information's `MeshOutputTopology` (1 lines,
/// 2 triangles, at byte 27 of the runtime information, after its size), as
/// the translator writes it (`xtp-shader` `emit/container.rs`). Undefined
/// if the container does not say.
fn mesh_topology(dxil: &[u8]) -> D3D12_PRIMITIVE_TOPOLOGY_TYPE {
    use xtp_shader::{Container, FourCC};
    let topology = Container::parse(dxil).ok().and_then(|c| {
        let psv = c.part(FourCC::PSV0)?.data;
        let size = u32::from_le_bytes(psv.get(..4)?.try_into().ok()?);
        (size >= 28).then_some(())?;
        psv.get(4 + 27).copied()
    });
    match topology {
        Some(1) => D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE,
        Some(2) => D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        _ => D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED,
    }
}

/// The pipeline of `stages` with the state of `g`, as a stream
/// ([`stream_of`]), from the library under `name`.
///
/// # Safety
/// `stages` doit contenir une suite de sous-objets D3D12 alignés dont les
/// pointeurs restent valides jusqu'à la création ou au chargement du pipeline.
unsafe fn stream_pipeline<S>(
    host: &crate::host::Host,
    root: &ID3D12RootSignature,
    g: &GraphicsState,
    stages: S,
    name: &str,
) -> windows::core::Result<ID3D12PipelineState> {
    use windows::core::Interface;
    let mut stream = stream_of(host, root, g, stages);
    let desc = stream.desc();
    let device = host.device.cast::<ID3D12Device2>()?;
    super::library::pipeline(
        name,
        // SAFETY: the stream holds whole subobjects pointing at live data.
        |library, name| unsafe { library.LoadPipeline(name, &desc) },
        // SAFETY: as above.
        || unsafe { device.CreatePipelineState(&desc) },
    )
}

/// The bytes of shader code (none for an absent stage).
fn code_bytes(code: &D3D12_SHADER_BYTECODE) -> Vec<u8> {
    if code.pShaderBytecode.is_null() {
        return Vec::new();
    }
    // SAFETY: a translation's own bytes, alive while its shader is.
    unsafe { std::slice::from_raw_parts(code.pShaderBytecode.cast::<u8>(), code.BytecodeLength) }
        .to_vec()
}

pub(crate) fn root_signature(
    device: &ID3D12Device,
    params: &[D3D12_ROOT_PARAMETER1],
    samplers: &[D3D12_STATIC_SAMPLER_DESC],
    compute: bool,
) -> windows::core::Result<(ID3D12RootSignature, Vec<u8>)> {
    let input = if compute {
        D3D12_ROOT_SIGNATURE_FLAG_NONE
    } else {
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
    };
    let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
        Version: windows::Win32::Graphics::Direct3D12::D3D_ROOT_SIGNATURE_VERSION_1_1,
        Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
            Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                NumParameters: params.len() as u32,
                pParameters: params.as_ptr(),
                NumStaticSamplers: samplers.len() as u32,
                pStaticSamplers: samplers.as_ptr(),
                Flags: input
                    | D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED
                    | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED,
            },
        },
    };
    // SAFETY: the description points at live arrays; the blob is read in
    // full before it is released.
    unsafe {
        let mut blob: Option<ID3DBlob> = None;
        let mut error: Option<ID3DBlob> = None;
        if let Err(e) = D3D12SerializeVersionedRootSignature(&desc, &mut blob, Some(&mut error)) {
            if let Some(error) = error {
                let text = std::slice::from_raw_parts(
                    error.GetBufferPointer().cast::<u8>(),
                    error.GetBufferSize(),
                );
                tracing::warn!("root signature: {}", String::from_utf8_lossy(text));
            }
            return Err(e);
        }
        let blob = blob.ok_or_else(windows::core::Error::empty)?;
        let bytes =
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
        Ok((device.CreateRootSignature(0, bytes)?, bytes.to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A DXIL container with a `PSV0` part (runtime information of
    /// `info_size` bytes, the lane counts at 16) and a `DXIL` part starting
    /// with the program version.
    fn container(info_size: u32, lanes: (u32, u32), version: u32) -> Vec<u8> {
        let mut psv = info_size.to_le_bytes().to_vec();
        let mut info = vec![0u8; info_size as usize];
        if info_size >= 24 {
            info[16..20].copy_from_slice(&lanes.0.to_le_bytes());
            info[20..24].copy_from_slice(&lanes.1.to_le_bytes());
        }
        psv.extend(info);
        let dxil = [version.to_le_bytes(), [0; 4]].concat();
        let parts: [(&[u8; 4], &[u8]); 2] = [(b"PSV0", &psv), (b"DXIL", &dxil)];
        let header = 0x20 + 4 * parts.len();
        let mut body = Vec::new();
        let mut offsets = Vec::new();
        for (fourcc, data) in parts {
            offsets.push((header + body.len()) as u32);
            body.extend(*fourcc);
            body.extend((data.len() as u32).to_le_bytes());
            body.extend(data);
        }
        let mut out = b"DXBC".to_vec();
        out.extend([0; 16]);
        out.extend(1u16.to_le_bytes());
        out.extend(0u16.to_le_bytes());
        out.extend(((header + body.len()) as u32).to_le_bytes());
        out.extend((parts.len() as u32).to_le_bytes());
        for o in offsets {
            out.extend(o.to_le_bytes());
        }
        out.extend(body);
        out
    }

    /// Compute shader (kind 5), shader model 6.6.
    const CS_6_6: u32 = 5 << 16 | 6 << 4 | 6;

    /// A real console wave64 mesh program of two console waves (it reads
    /// its group id, `v5`, from the first lane and exports a primitive),
    /// translated for a host running waves of `lanes` lanes.
    fn wave64_mesh(lanes: (u32, u32)) -> xtp_shader::emit::HostShader {
        use xtp_shader::native::{MeshIo, StageIo};
        let words: [u32; 4] = [
            0x7E04_0505, // v_readfirstlane_b32 s2, v5
            0xF800_0941,
            0x0000_0009, // exp prim, v9, off, off, off done
            0xBF81_0000, // s_endpgm
        ];
        let native = xtp_shader::NativeProgram {
            stage: xtp_shader::HwStage::Ms,
            wave64: true,
            float_mode: 0,
            vgprs: 16,
            layout: Vec::new(),
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
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let c = xtp_shader::Container {
            digest: [0; 16],
            version: (1, 0),
            parts: Vec::new(),
        };
        let target = xtp_shader::emit::Target {
            lanes_min: lanes.0,
            lanes_max: lanes.1,
            stride_table: false,
        };
        xtp_shader::emit::translate_program_for(&c, &native, target).expect("it translates")
    }

    /// A refusal known from the console program alone is the one its
    /// translation gets, for real mesh translations: they declare no wave
    /// size, so their container adds nothing to what decides. For every
    /// switch, target and host; and there is none for other stages and
    /// wave32 programs, which only their translation can refuse.
    #[test]
    fn missing_output_keeps_multisampling_and_other_formats() {
        let mut state = GraphicsState::default();
        state.samples.Count = 4;
        state.targets.NumRenderTargets = 2;
        state.targets.RTFormats[..2].copy_from_slice(&[DXGI_FORMAT(10), DXGI_FORMAT(28)]);
        assert_eq!(state.null_view(0), Some((DXGI_FORMAT(10), true)));
        let variant = state.without_targets(1);
        assert_eq!(variant.samples.Count, 4);
        assert_eq!(
            variant.targets.RTFormats[..2],
            [DXGI_FORMAT_UNKNOWN, DXGI_FORMAT(28)]
        );
    }

    #[test]
    fn mesh_refusals_before_and_after_translation_agree() {
        for (target, target_64) in [((32, 32), false), ((4, 4), false), ((32, 64), true)] {
            let host = wave64_mesh(target);
            assert_eq!(
                Needs::of(&host.dxil).map(|n| n.lanes),
                Some((0, u32::MAX)),
                "a mesh translation for {target:?} declares no wave size"
            );
            for (stage, wave64, refuse) in ["mesh", "pixel", "compute"]
                .into_iter()
                .flat_map(|s| [(s, true), (s, false)])
                .flat_map(|(s, w)| [(s, w, false), (s, w, true)])
            {
                let switches = (refuse, target_64);
                let native = super::super::Native {
                    used: Vec::new(),
                    wave64,
                    threads: None,
                };
                let translated = Translated {
                    host: host.clone(),
                    used: Vec::new(),
                    wave64,
                };
                for lanes in [(32, 32), (4, 4), (32, 64), (64, 64)] {
                    let before = Needs::native_with(stage, &native, switches)
                        .and_then(|n| n.refusal(lanes, (6, 8)));
                    let after = Needs::translated_with(stage, &translated, switches)
                        .and_then(|n| n.refusal(lanes, (6, 8)));
                    let what = format!("{stage} {wave64} {switches:?} {target:?} {lanes:?}");
                    if before.is_some() {
                        assert_eq!(before, after, "{what}");
                    }
                    if stage != "mesh" || !wave64 || lanes.1 >= 64 {
                        assert_eq!(before, None, "{what}");
                    }
                }
            }
        }
    }

    /// A wave64 mesh pipeline is refused before any of its shaders is
    /// translated only with `XTP_WAVE64_MESH=refuse` (or a target with
    /// 64-lane waves forced on a host without them): by default the
    /// translator decides, so on a host without 64-lane waves (WARP,
    /// NVIDIA) its shaders are translated first.
    #[test]
    fn wave64_mesh_pipelines_are_refused_before_translation() {
        let test = "pipeline::host::tests::wave64_mesh_pipelines_are_refused_before_translation";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let lanes = crate::device::host_wave_lanes().expect("the device's lanes");
        if !host.mesh_shaders() || lanes.1 >= 64 {
            // Nothing is refused for its waves on this host: not a missing
            // device, so no failure in strict mode either.
            eprintln!("SKIPPED {test}: no mesh shaders, or 64-lane waves");
            return;
        }
        let shader = |stage, hash: &str| {
            Arc::new(Shader {
                stage,
                hash: hash.into(),
                ambient: None,
                container: b"not translated".to_vec(),
                native: std::sync::OnceLock::from(Ok(super::super::Native {
                    used: Vec::new(),
                    wave64: true,
                    threads: None,
                })),
                translated: std::sync::OnceLock::new(),
            })
        };
        let (ms, ps) = (
            shader("mesh", "6d65736836340000"),
            shader("pixel", "7069786c36340000"),
        );
        let made = HostPipeline::new(&[ms.clone(), ps.clone()], Some(&GraphicsState::default()));
        assert!(made.is_none(), "refused, or not translated");
        let refused = refusals().unwrap_or_default();
        let early = refused.contains("mesh shader is a mesh program written for 64-lane waves")
            || refused.contains("mesh shader is a mesh program translated for 64-lane waves");
        if Needs::mesh_switches() == (false, false) {
            assert!(
                ms.translated.get().is_some(),
                "the mesh shader was translated"
            );
            assert!(!early, "{refused}");
        } else {
            assert!(
                ms.translated.get().is_none(),
                "the mesh shader was not translated"
            );
            assert!(ps.translated.get().is_none(), "nor the pixel shader");
            assert!(early, "{refused}");
        }
    }

    #[test]
    fn wave64_programs_are_refused_on_32_lane_hosts() {
        let wave64 = Needs::of(&container(48, (64, 64), CS_6_6)).unwrap();
        assert_eq!(
            wave64,
            Needs {
                lanes: (64, 64),
                model: (6, 6),
                wave64_mesh: None,
            }
        );
        let nvidia = (32, 32);
        assert_eq!(
            wave64.refusal(nvidia, (6, 8)).as_deref(),
            Some("needs 64-lane waves; the host has 32 to 32")
        );
        // RDNA: 32 to 64 lanes.
        assert_eq!(wave64.refusal((32, 64), (6, 8)), None);
        // No wave size asked: anywhere.
        let any = Needs::of(&container(48, (0, u32::MAX), CS_6_6)).unwrap();
        assert_eq!(any.refusal(nvidia, (6, 8)), None);
        // Older runtime information has no lane counts.
        let old = Needs::of(&container(16, (64, 64), CS_6_6)).unwrap();
        assert_eq!(old.lanes, (0, u32::MAX));
    }

    /// What the wave64 port emits for a 32-lane host, `WaveSize(32)`, runs
    /// there; WARP's 4-lane waves run neither (VIctory #578's PSV0 check,
    /// which fb3db31's [`Needs`] already made).
    #[test]
    fn wave32_translations_run_on_32_lane_hosts() {
        let wave32 = Needs::of(&container(48, (32, 32), CS_6_6)).unwrap();
        assert_eq!(wave32.refusal((32, 32), (6, 8)), None);
        assert_eq!(wave32.refusal((32, 64), (6, 8)), None);
        assert_eq!(
            wave32.refusal((4, 4), (6, 8)).as_deref(),
            Some("needs 32-lane waves; the host has 4 to 4")
        );
        // An SM 6.8 range overlapping the host's runs.
        let range = Needs::of(&container(48, (32, 64), CS_6_6)).unwrap();
        assert_eq!(range.refusal((32, 32), (6, 8)), None);
    }

    /// A wave64 mesh translation declares no wave size. The translator
    /// makes it run in the waves of the target it is made for, so it is
    /// refused only where that target has 64-lane waves and the host none,
    /// or with `XTP_WAVE64_MESH=refuse` (VIctory #578's refusal); one
    /// reason only, and the same without a container that says more.
    #[test]
    fn wave64_mesh_programs_run_where_their_target_runs() {
        let any = Needs::of(&container(48, (0, u32::MAX), CS_6_6)).unwrap();
        let reason = |stage, wave64, switches| Needs::mesh_reason(stage, wave64, switches);
        assert_eq!(reason("mesh", true, (false, false)), None, "by default");
        assert_eq!(reason("mesh", true, (true, false)), Some(MESH_REFUSED));
        assert_eq!(reason("mesh", true, (true, true)), Some(MESH_REFUSED));
        assert_eq!(reason("mesh", true, (false, true)), Some(MESH_WAVE64));
        for switches in [(false, false), (true, false), (false, true)] {
            assert_eq!(reason("mesh", false, switches), None, "wave32");
            assert_eq!(reason("compute", true, switches), None, "compute");
        }
        for (what, text) in [
            (
                MESH_REFUSED,
                "is a mesh program written for 64-lane waves (XTP_WAVE64_MESH=refuse, as \
                 VIctory #578); the host has 32 to 32",
            ),
            (
                MESH_WAVE64,
                "is a mesh program translated for 64-lane waves (XTP_HOST_LANES); the host \
                 has 32 to 32",
            ),
        ] {
            let mesh = Needs {
                wave64_mesh: Some(what),
                ..any
            };
            assert_eq!(mesh.refusal((32, 32), (6, 8)).as_deref(), Some(text));
            assert!(
                mesh.refusal((4, 4), (6, 5))
                    .is_some_and(|r| r.ends_with("the host has 4 to 4")),
                "the lanes are the reason, not the shader model too"
            );
            assert_eq!(mesh.refusal((32, 64), (6, 8)), None);
            // A translation that declares its wave size is judged by it.
            let declared = Needs {
                lanes: (32, 32),
                ..mesh
            };
            assert_eq!(declared.refusal((32, 32), (6, 8)), None);
        }
        assert_eq!(any.refusal((32, 32), (6, 8)), None, "no mesh refusal");

        // A container without a PSV0 part does not let it through (#578's
        // check depended on one).
        let unsaid = Translated {
            host: xtp_shader::emit::HostShader {
                stage: xtp_shader::HwStage::Ms,
                dxil: b"not a container".to_vec(),
                group_base: None,
                user_data: Vec::new(),
                samplers: Vec::new(),
                static_samplers: Vec::new(),
                root_cbvs: Vec::new(),
                root_srvs: Vec::new(),
                root_uavs: Vec::new(),
                stride_table: false,
                run_time_buffers: 0,
            },
            used: Vec::new(),
            wave64: true,
        };
        let needs = Needs::translated_with("mesh", &unsaid, (true, false)).expect("the mesh check");
        let refused = needs.refusal((32, 32), (6, 8)).unwrap_or_default();
        assert!(refused.starts_with("is a mesh program"), "{refused}");
        assert_eq!(needs.refusal((32, 64), (6, 5)), None, "nor anything else");
        assert_eq!(
            Needs::translated_with("mesh", &unsaid, (false, false)),
            None
        );
    }

    #[test]
    fn programs_of_a_newer_shader_model_are_refused() {
        let needs = Needs::of(&container(48, (0, u32::MAX), CS_6_6)).unwrap();
        assert_eq!(
            needs.refusal((32, 32), (6, 5)).as_deref(),
            Some("needs shader model 6.6; the host has 6.5")
        );
        assert_eq!(needs.refusal((32, 32), (6, 6)), None);
        assert_eq!(Needs::of(b"not a container"), None);
    }

    /// A host that renders, blends and samples only `targets`, has depth in
    /// `depth`, vertex input in `vertex`, and 4 samples of nothing.
    struct Fake {
        targets: &'static [u32],
        blend: &'static [u32],
        depth: &'static [u32],
        vertex: &'static [u32],
    }

    impl crate::resource::format::Caps for Fake {
        fn support1(&self, format: u32) -> u32 {
            use crate::resource::format::uses::*;
            let has = |list: &[u32], bit| if list.contains(&format) { bit } else { 0 };
            has(self.targets, RENDER_TARGET)
                | has(self.blend, BLENDABLE)
                | has(self.depth, DEPTH_STENCIL)
                | has(self.vertex, VERTEX_BUFFER)
        }
        fn samples(&self, _: u32, count: u32) -> bool {
            count <= 1
        }
    }

    impl PipelineCaps for Fake {
        fn logic_op(&self, format: DXGI_FORMAT) -> bool {
            matches!(format, DXGI_FORMAT_R8_UINT | DXGI_FORMAT_R16_UINT)
        }
    }

    const HOST: Fake = Fake {
        targets: &[28, 54, 57, 62, 64],
        blend: &[28],
        depth: &[40],
        vertex: &[2, 28],
    };

    fn state(targets: &[u32], depth: u32) -> GraphicsState {
        let mut g = GraphicsState::default();
        g.targets.NumRenderTargets = targets.len() as u32;
        for (slot, &f) in g.targets.RTFormats.iter_mut().zip(targets) {
            slot.0 = f as i32;
        }
        g.depth_format = DXGI_FORMAT(depth as i32);
        g
    }

    #[test]
    fn graphics_states_the_host_lacks_are_refused() {
        assert_eq!(state(&[28, 0], 40).refusal(&HOST), None);
        assert_eq!(
            state(&[115], 0).refusal(&HOST).as_deref(),
            Some("render target format 115: no render targets")
        );
        assert_eq!(
            state(&[28], 55).refusal(&HOST).as_deref(),
            Some("depth format 55: no depth-stencil targets")
        );
        // A floating-point target on a host that cannot blend it.
        let mut blended = state(&[28, 54], 0);
        blended.blend.RenderTarget[0].BlendEnable = true.into();
        assert_eq!(
            blended.refusal(&HOST).as_deref(),
            Some("render target format 54: no blending")
        );
        blended.blend.IndependentBlendEnable = true.into();
        assert_eq!(blended.refusal(&HOST), None, "only target 0 blends");
        let mut msaa = state(&[28], 0);
        msaa.samples.Count = 4;
        assert_eq!(
            msaa.refusal(&HOST).as_deref(),
            Some("render target format 28: no 4 samples")
        );
        let mut inputs = state(&[28], 0);
        inputs.inputs.push(D3D12_INPUT_ELEMENT_DESC {
            Format: DXGI_FORMAT(86),
            ..Default::default()
        });
        assert_eq!(
            inputs.refusal(&HOST).as_deref(),
            Some("vertex format 86: not a vertex buffer format")
        );
    }

    fn blend_stream(
        formats: &[u32],
        blend: D3D12_BLEND_DESC,
        formats_first: bool,
    ) -> GraphicsState {
        #[repr(C)]
        struct Sub<T>(u32, T);
        let formats = Sub(15, state(formats, 0).targets);
        let blend = Sub(8, blend);
        let mut objects = [
            (formats.0, std::ptr::from_ref(&formats).cast()),
            (blend.0, std::ptr::from_ref(&blend).cast()),
        ];
        if !formats_first {
            objects.reverse();
        }
        let mut g = GraphicsState::default();
        for (kind, at) in objects {
            // SAFETY: both subobjects have a four-byte-aligned payload at +4.
            unsafe { g.read(kind, at) };
        }
        g
    }

    #[test]
    fn integer_targets_bypass_only_their_own_blend() {
        let mut shared = GraphicsState::default().blend;
        shared.AlphaToCoverageEnable = true.into();
        shared.RenderTarget[0].BlendEnable = true.into();
        shared.RenderTarget[0].DestBlend = D3D12_BLEND_ONE;
        shared.RenderTarget[0].RenderTargetWriteMask = 5;
        let t = shared.RenderTarget[0];
        let off = D3D12_RENDER_TARGET_BLEND_DESC {
            BlendEnable: false.into(),
            ..t
        };
        for first in [true, false] {
            for (formats, integer, colour) in [([57, 0, 28], 0, 2), ([28, 0, 64], 2, 0)] {
                let g = blend_stream(&formats, shared, first);
                assert_eq!(g.refusal(&HOST), None);
                assert!(g.blend.IndependentBlendEnable.as_bool());
                assert!(g.blend.AlphaToCoverageEnable.as_bool());
                assert_eq!(g.blend.RenderTarget[integer], off);
                assert_eq!(g.blend.RenderTarget[colour], t);
            }
            let mut independent = shared;
            independent.IndependentBlendEnable = true.into();
            independent.RenderTarget[1].BlendEnable = true.into();
            let g = blend_stream(&[28, 62], independent, first);
            independent.RenderTarget[1].BlendEnable = false.into();
            assert_eq!(g.blend, independent);
            assert_eq!(g.refusal(&HOST), None);
        }
    }

    #[test]
    fn shared_logic_op_is_preserved_and_checked_for_each_target() {
        let mut logic = GraphicsState::default().blend;
        logic.RenderTarget[0].LogicOpEnable = true.into();
        logic.RenderTarget[0].LogicOp = D3D12_LOGIC_OP_XOR;
        for first in [true, false] {
            let supported = blend_stream(&[0, 62, 57], logic, first);
            assert_eq!(supported.blend, logic);
            assert_eq!(supported.refusal(&HOST), None);

            // The integer blend bypass also applies when the title sets both enables.
            let mut both = logic;
            both.RenderTarget[0].BlendEnable = true.into();
            assert_eq!(blend_stream(&[62], both, first).blend, logic);

            for unsupported in [28, 64] {
                let mixed = blend_stream(&[62, unsupported], logic, first);
                assert_eq!(mixed.blend, logic);
                assert_eq!(
                    mixed.refusal(&HOST),
                    Some(format!(
                        "render target format {unsupported}: no logic operations"
                    ))
                );
            }

            let mut independent = GraphicsState::default().blend;
            independent.IndependentBlendEnable = true.into();
            independent.RenderTarget[1] = logic.RenderTarget[0];
            let per_target = blend_stream(&[28, 62], independent, first);
            assert_eq!(per_target.blend, independent);
            assert_eq!(
                per_target.refusal(&HOST).as_deref(),
                Some("render target format 62: no per-target logic operations")
            );
        }
    }

    #[test]
    fn r8_uint_pipeline_writes_the_pixel_shader_value_unblended() {
        let mut blend = GraphicsState::default().blend;
        blend.RenderTarget[0].BlendEnable = true.into();
        // With these factors the destination would keep its clear value.
        blend.RenderTarget[0].SrcBlend = D3D12_BLEND_ZERO;
        blend.RenderTarget[0].DestBlend = D3D12_BLEND_ONE;
        check_r8_uint_draw(
            "r8_uint_pipeline_writes_the_pixel_shader_value_unblended",
            blend,
            73,
        );
    }

    #[test]
    fn r8_uint_pipeline_xors_the_pixel_shader_value_with_the_target() {
        let mut blend = GraphicsState::default().blend;
        blend.RenderTarget[0].LogicOpEnable = true.into();
        blend.RenderTarget[0].LogicOp = D3D12_LOGIC_OP_XOR;
        check_r8_uint_draw(
            "r8_uint_pipeline_xors_the_pixel_shader_value_with_the_target",
            blend,
            73 ^ 11,
        );
    }

    fn check_r8_uint_draw(test: &str, blend: D3D12_BLEND_DESC, expected: u8) {
        use windows::Win32::Foundation::RECT;
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let mut g = blend_stream(&[62], blend, false);
        g.depth.DepthEnable = false.into();
        let bytes = render_r8_uint(host, g, 2, |list, _| {
            // SAFETY: la liste est ouverte, le premier pixel est dans la cible.
            unsafe {
                list.host().list.RSSetScissorRects(&[RECT {
                    left: 0,
                    top: 0,
                    right: 1,
                    bottom: 1,
                }])
            };
        });
        assert_eq!(
            &bytes[..2],
            &[expected, 11],
            "drawn pixel and untouched neighbour"
        );
    }

    /// A pipeline with a geometry shader runs its vertex shader
    /// container's export shader (ES) as its vertex shader, and its
    /// geometry shader's own translation (`merged_half`'s, given here as
    /// FXC's programs): the ES writes the value the GS passes on and the
    /// pixel shader writes. The container's plain vertex shader (its own
    /// translation here) writes no such value, and the geometry shader's
    /// container has no translation of its own: both stages must take the
    /// merged halves.
    #[test]
    fn a_pipeline_with_a_geometry_shader_runs_the_export_shader() {
        let Some(gpu) =
            crate::testgpu::device("a_pipeline_with_a_geometry_shader_runs_the_export_shader")
        else {
            return;
        };
        let host = gpu.host();
        use windows::core::s;
        use xtp_shader::HwStage;
        const ITEM: &str = "struct V {
            float4 pos : SV_Position;
            nointerpolation uint value : TEXCOORD0;
        };";
        let es = format!(
            "{ITEM} V main(uint id : SV_VertexID) {{
                V v;
                v.pos = float4(id == 1 ? 3 : -1, id == 2 ? 3 : -1, 0, 1);
                v.value = 73;
                return v;
            }}"
        );
        let gs = format!(
            "{ITEM} [maxvertexcount(3)]
            void main(triangle V input[3], inout TriangleStream<V> s) {{
                for (uint i = 0; i < 3; i++) {{
                    s.Append(input[i]);
                }}
            }}"
        );
        let plain = "float4 main(uint id : SV_VertexID) : SV_Position {
            return float4(id == 1 ? 3 : -1, id == 2 ? 3 : -1, 0, 1);
        }";
        let ps = format!("{ITEM} uint main(V v) : SV_Target {{ return v.value; }}");
        let (vs_hash, gs_hash) = ("merged-es-test-vertex", "merged-es-test-geometry");
        {
            let mut made = MERGED_HALVES.lock().unwrap_or_else(|e| e.into_inner());
            let made = made.get_or_insert_with(Default::default);
            for (hash, half, t) in [
                (vs_hash, true, compiled(HwStage::Es, &es, s!("vs_5_1"))),
                (gs_hash, false, compiled(HwStage::Gs, &gs, s!("gs_5_1"))),
            ] {
                let once = std::sync::OnceLock::from(Some(Arc::new(t)));
                made.insert((hash.to_owned(), half), Arc::new(once));
            }
        }
        let shaders = [
            title_shader(
                "vertex",
                vs_hash,
                Some(compiled(HwStage::Vs, plain, s!("vs_5_1"))),
            ),
            title_shader(
                "pixel",
                "merged-es-test-pixel",
                Some(compiled(HwStage::Ps, &ps, s!("ps_5_1"))),
            ),
            title_shader("geometry", gs_hash, None),
        ];
        let mut g = blend_stream(&[62], GraphicsState::default().blend, false);
        g.depth.DepthEnable = false.into();
        g.topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        g.rasterizer.CullMode = D3D12_CULL_MODE_NONE;
        let p = HostPipeline::new(&shaders, Some(&g)).expect("a pipeline with a geometry shader");
        let stages: Vec<_> = p.stages.iter().map(|s| s.translated().host.stage).collect();
        assert_eq!(stages, [HwStage::Es, HwStage::Ps, HwStage::Gs]);
        assert_eq!(draw_r8_uint(host, &p, 2, |_, _| {})[..2], [73, 73]);
    }

    #[test]
    fn depth_bounds_distinguish_pipeline_keys() {
        let mut g = GraphicsState::default();
        let mut depth = Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1,
            D3D12_DEPTH_STENCIL_DESC1::default(),
        );
        // SAFETY: le type et la disposition des deux sous-objets suivent D3D12.
        unsafe {
            g.read(21, (&raw const depth).cast());
            let disabled = g.describe();
            depth.1.DepthBoundsTestEnable = true.into();
            g.read(21, (&raw const depth).cast());
            assert_ne!(g.describe(), disabled);
            // The library's name: the state the host makes (Codex review
            // of d286ffd, point 5): without the test on a host without it,
            // as the pipeline's description has it there.
            assert_eq!(g.describe_on(true), g.describe());
            assert_eq!(g.describe_on(false), disabled);
            let legacy = Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, g.depth);
            g.read(11, (&raw const legacy).cast());
            assert_eq!(g.describe(), disabled);
        }
    }

    #[test]
    fn depth_bounds_filter_stored_depth() {
        use crate::object::{This, slot};
        use windows::Win32::Foundation::RECT;
        let test = "depth_bounds_filter_stored_depth";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        if !host.depth_bounds {
            crate::testgpu::skip(test, "DepthBoundsTestSupported = FALSE");
            return;
        }
        let mut g = state(&[62], DXGI_FORMAT_D32_FLOAT.0 as u32);
        let sub = Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1,
            D3D12_DEPTH_STENCIL_DESC1 {
                DepthEnable: true.into(),
                DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
                DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
                DepthBoundsTestEnable: true.into(),
                ..Default::default()
            },
        );
        // SAFETY: sous-objet complet, dans la disposition du flux du titre.
        unsafe { g.read(21, (&raw const sub).cast()) };
        let mut depth: Option<ID3D12Resource> = None;
        // SAFETY: description d'une texture de quatre pixels, sans données initiales.
        unsafe {
            host.device.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_DEFAULT,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                    Width: 4,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    Format: DXGI_FORMAT_D32_FLOAT,
                    SampleDesc: g.samples,
                    Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                    ..Default::default()
                },
                D3D12_RESOURCE_STATE_DEPTH_WRITE,
                None,
                &mut depth,
            )
        }
        .unwrap();
        let depth = depth.unwrap();
        let bytes = render_r8_uint(host, g, 4, |list, rtv| {
            let dsv = host.dsv(&depth, None);
            let l = &list.host().list;
            // SAFETY: les vues et leurs ressources restent vivantes jusqu'à la lecture.
            unsafe {
                for (x, z) in [0.125, 0.25, 0.75, 0.875].into_iter().enumerate() {
                    let pixel = RECT {
                        left: x as i32,
                        top: 0,
                        right: x as i32 + 1,
                        bottom: 1,
                    };
                    l.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, z, 0, Some(&[pixel]));
                }
                l.OMSetRenderTargets(1, Some(&rtv), false, Some(&dsv));
            }
            let offset = if crate::device::newer() { 0x230 } else { 0x220 };
            slot::<extern "system" fn(This, f32, f32)>(list.this(), offset)(
                list.this(),
                0.25,
                0.75,
            );
        });
        assert_eq!(&bytes[..4], &[11, 73, 73, 11]);
    }

    /// `source` compiled by FXC for `profile`, as the translation of a
    /// `kind` console program.
    fn compiled(kind: xtp_shader::HwStage, source: &str, profile: PCSTR) -> Translated {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::core::s;
        let mut blob = None;
        // SAFETY: the source lives through compilation and the blob owns its bytes.
        let dxil = unsafe {
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                PCSTR::null(),
                None,
                None,
                s!("main"),
                profile,
                0,
                0,
                &mut blob,
                None,
            )
            .unwrap();
            let blob = blob.unwrap();
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
                .to_vec()
        };
        Translated {
            host: xtp_shader::emit::HostShader {
                stage: kind,
                dxil,
                group_base: None,
                user_data: vec![],
                samplers: vec![],
                static_samplers: vec![],
                root_cbvs: vec![],
                root_srvs: vec![],
                root_uavs: vec![],
                stride_table: false,
                run_time_buffers: 0,
            },
            used: vec![],
            wave64: false,
        }
    }

    /// A title's `stage` shader named `hash`, already translated (or not:
    /// `None`), without a container.
    fn title_shader(
        stage: &'static str,
        hash: &str,
        translated: Option<Translated>,
    ) -> Arc<Shader> {
        Arc::new(Shader {
            stage,
            hash: hash.to_owned(),
            ambient: None,
            container: Vec::new(),
            native: Default::default(),
            translated: std::sync::OnceLock::from(translated),
        })
    }

    fn render_r8_uint(
        host: &crate::host::Host,
        mut g: GraphicsState,
        width: u32,
        before_draw: impl FnOnce(&crate::testgpu::TitleList, D3D12_CPU_DESCRIPTOR_HANDLE),
    ) -> Vec<u8> {
        use windows::core::s;
        let vs = title_shader(
            "vertex",
            "r8-uint-vertex",
            Some(compiled(
                xtp_shader::HwStage::Vs,
                "float4 main(uint id : SV_VertexID) : SV_Position {
                return float4(id == 1 ? 3 : -1, id == 2 ? 3 : -1, 0, 1);
            }",
                s!("vs_5_1"),
            )),
        );
        let ps = title_shader(
            "pixel",
            "r8-uint-pixel",
            Some(compiled(
                xtp_shader::HwStage::Ps,
                "uint main() : SV_Target { return 73u; }",
                s!("ps_5_1"),
            )),
        );
        g.topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        g.rasterizer.CullMode = D3D12_CULL_MODE_NONE;
        let p = HostPipeline::new(&[vs, ps], Some(&g)).expect("R8_UINT pipeline");
        draw_r8_uint(host, &p, width, before_draw)
    }

    /// One triangle drawn with `p` over an R8_UINT target `width` pixels
    /// wide and one high, cleared to 11; the target's bytes.
    fn draw_r8_uint(
        host: &crate::host::Host,
        p: &HostPipeline,
        width: u32,
        before_draw: impl FnOnce(&crate::testgpu::TitleList, D3D12_CPU_DESCRIPTOR_HANDLE),
    ) -> Vec<u8> {
        use crate::object::{This, slot};
        use crate::resource::tests_support::{committed, read_back, target};
        use windows::Win32::Foundation::RECT;
        use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;

        let image = committed(&target(width.into(), 1, 62));
        let texture = crate::resource::host_resource(image).expect("R8_UINT texture");
        let rtv = host.rtv(texture, None);
        let list = crate::testgpu::TitleList::new();
        let l = &list.host().list;
        let render = D3D12_RESOURCE_STATE_RENDER_TARGET;
        // SAFETY: the pipeline, target and view live through the draw and readback.
        unsafe {
            l.ResourceBarrier(&[crate::host::transition(
                texture,
                D3D12_RESOURCE_STATE_COMMON,
                render,
            )]);
            l.ClearRenderTargetView(rtv, &[11.0; 4], None);
            l.OMSetRenderTargets(1, Some(&rtv), false, None);
            l.SetDescriptorHeaps(&crate::descriptor::heaps_with_own_samplers(None));
            l.SetGraphicsRootSignature(&p.root);
            l.SetPipelineState(&p.pso);
            l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            l.RSSetViewports(&[D3D12_VIEWPORT {
                Width: width as f32,
                Height: 1.0,
                MaxDepth: 1.0,
                ..Default::default()
            }]);
            l.RSSetScissorRects(&[RECT {
                left: 0,
                top: 0,
                right: width as i32,
                bottom: 1,
            }]);
            before_draw(&list, rtv);
            l.DrawInstanced(3, 1, 0, 0);
        }
        list.run();
        let bytes = read_back(host, texture, render.0 as u32);
        slot::<extern "system" fn(This) -> u32>(image, 0x10)(image);
        bytes
    }

    /// A target whose stand-in has an alpha the title's format lacks
    /// neither writes alpha nor blends with it; the others keep their
    /// blending, per target only where some target keeps its alpha
    /// (dark-world's test from the review of the #492 port, extended).
    #[test]
    fn targets_without_alpha_blend_with_alpha_one() {
        let target = D3D12_RENDER_TARGET_BLEND_DESC {
            BlendEnable: true.into(),
            SrcBlend: D3D12_BLEND_INV_DEST_ALPHA,
            DestBlend: D3D12_BLEND_DEST_ALPHA,
            BlendOp: D3D12_BLEND_OP_ADD,
            SrcBlendAlpha: D3D12_BLEND_SRC_ALPHA_SAT,
            DestBlendAlpha: D3D12_BLEND_SRC_ALPHA,
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            RenderTargetWriteMask: 0xf,
            ..Default::default()
        };
        let shared = D3D12_BLEND_DESC {
            RenderTarget: [target; 8],
            ..Default::default()
        };
        let formats = |list: &[i32]| {
            let mut formats = [DXGI_FORMAT(0); 8];
            for (f, &title) in formats.iter_mut().zip(list) {
                f.0 = title;
            }
            D3D12_RT_FORMAT_ARRAY {
                RTFormats: formats,
                NumRenderTargets: list.len() as u32,
            }
        };
        let lacks = |f| f == 67 || f == 85;
        let text = |b: &D3D12_BLEND_DESC| format!("{b:?}");

        // Nothing to change for formats with their own alpha.
        let mut same = shared;
        blend_without(&mut same, &formats(&[28, 0]), lacks);
        assert_eq!(text(&same), text(&shared));

        // A target keeping its alpha beside one without: per target.
        let mut mixed = shared;
        blend_without(&mut mixed, &formats(&[28, 67]), lacks);
        assert!(mixed.IndependentBlendEnable.as_bool());
        let (kept, changed) = (mixed.RenderTarget[0], mixed.RenderTarget[1]);
        assert_eq!(
            text(&D3D12_BLEND_DESC {
                RenderTarget: [kept; 8],
                ..shared
            }),
            text(&shared)
        );
        assert_eq!(changed.SrcBlend, D3D12_BLEND_ZERO);
        assert_eq!(changed.DestBlend, D3D12_BLEND_ONE);
        assert_eq!(
            changed.SrcBlendAlpha, D3D12_BLEND_SRC_ALPHA_SAT,
            "not written"
        );
        assert_eq!(changed.RenderTargetWriteMask, 0x7);
        // The blend and the formats come in either order: twice is once.
        let mut twice = mixed;
        blend_without(&mut twice, &formats(&[28, 67]), lacks);
        assert_eq!(text(&twice), text(&mixed));

        // Every bound target without alpha: still one blending for all.
        let mut alone = shared;
        blend_without(&mut alone, &formats(&[67, 0, 85]), lacks);
        assert!(!alone.IndependentBlendEnable.as_bool());
        assert_eq!(alone.RenderTarget[0].SrcBlend, D3D12_BLEND_ZERO);
        assert_eq!(alone.RenderTarget[0].DestBlend, D3D12_BLEND_ONE);
        assert_eq!(alone.RenderTarget[0].RenderTargetWriteMask, 0x7);

        // A logic operation never becomes per target (D3D12 refuses it):
        // the alpha goes out of the writes only where every target lacks it.
        let logic = D3D12_BLEND_DESC {
            RenderTarget: [D3D12_RENDER_TARGET_BLEND_DESC {
                BlendEnable: false.into(),
                LogicOpEnable: true.into(),
                LogicOp: D3D12_LOGIC_OP_XOR,
                ..target
            }; 8],
            ..Default::default()
        };
        let mut kept = logic;
        blend_without(&mut kept, &formats(&[28, 85]), lacks);
        assert_eq!(text(&kept), text(&logic), "unchanged");
        let mut masked = logic;
        blend_without(&mut masked, &formats(&[85]), lacks);
        assert!(!masked.IndependentBlendEnable.as_bool());
        assert!(masked.RenderTarget[0].LogicOpEnable.as_bool());
        assert_eq!(masked.RenderTarget[0].RenderTargetWriteMask, 0x7);
    }

    /// The acceptance test of the wave64 wiring (VIctory #223), on the
    /// test's device (WARP on the person's PC, whose waves are 4 lanes):
    /// translations for a target forced to an NVIDIA host's 32 to 32 lanes
    /// (what `XTP_HOST_LANES=32-32` makes it), kept in a shader cache under
    /// that target's key, made into host pipelines as a title's are.
    /// - A console wave64 compute program whose lanes read only inside
    ///   their half declares `WaveSize(32)`, and is found again under its
    ///   own key, apart from its translation for any host (`WaveSize(64)`).
    ///   Where the device runs 32-lane waves its pipeline is made and gives
    ///   the console's result. Where it does not (WARP) it is refused for
    ///   its lanes before any driver call, and the driver refuses it too:
    ///   the early refusal drops nothing the device would run.
    /// - The same program without lane reads declares no wave size and runs
    ///   on every device, which also checks the dispatch read back here.
    #[test]
    fn wave64_compute_for_a_forced_32_lane_target() {
        use std::sync::atomic::{AtomicU64, Ordering};
        use xtp_shader::emit::{HostShader, Target};
        let test = "pipeline::host::tests::wave64_compute_for_a_forced_32_lane_target";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let device = gpu.host();
        let lanes = crate::device::host_wave_lanes().expect("the device's lanes");
        let target = crate::device::target_of(Some(lanes), Some("32-32")).unwrap();
        assert_eq!(
            target,
            Target {
                lanes_min: 32,
                lanes_max: 32,
                stride_table: false,
            }
        );

        let dir = tempfile::tempdir().unwrap();
        let cache = xtp_pipeline::Cache::open(dir.path()).unwrap();
        let runs = AtomicU64::new(0);
        let _serial = crate::shader_cache::serial();
        let translate = |in_half: bool, target: Target| {
            let (c, native) = crate::shader_cache::wave64_program(in_half);
            let container: &[u8] = if in_half {
                b"wave64 reading in half"
            } else {
                b"wave64 reading its own lane"
            };
            crate::shader_cache::kept(&cache, container, target, |_| {
                runs.fetch_add(1, Ordering::Relaxed);
                xtp_shader::emit::translate_program_for(&c, &native, target)
                    .map_err(|(step, e)| format!("{}: {e}", step.name()))
            })
            .unwrap()
        };
        let halves = translate(true, target);
        assert_eq!(halves.wave_size(), Some(32), "two 32-lane waves");
        assert_eq!(translate(true, target), halves, "found again");
        assert_eq!(runs.load(Ordering::Relaxed), 1, "under its own key");
        let whole = translate(true, Target::ANY);
        assert_eq!(whole.wave_size(), Some(64), "for any host: one wave64");
        assert_eq!(runs.load(Ordering::Relaxed), 2, "under another key");
        let local = translate(false, target);
        assert_eq!(local.wave_size(), None);

        let pipeline = |host: HostShader, hash: &str| {
            let shader = Arc::new(Shader {
                stage: "compute",
                hash: hash.into(),
                ambient: None,
                container: Vec::new(),
                native: std::sync::OnceLock::new(),
                translated: std::sync::OnceLock::from(Some(Translated {
                    host,
                    used: Vec::new(),
                    wave64: true,
                })),
            });
            HostPipeline::new(&[shader], None)
        };
        // The driver check below tells a refusal from a description it takes.
        assert!(
            !driver_refuses(device, &local),
            "the driver takes a pipeline of no wave size"
        );
        let groups = 2;
        let made = pipeline(local, "forced-32-own-lane").expect("a pipeline of no wave size");
        let own: Vec<u32> = (0..groups).flat_map(|_| 0..64).collect();
        assert_eq!(run_wave64_program(device, &made, groups), own);

        let made = pipeline(halves.clone(), "forced-32-in-half");
        if lanes.0 <= 32 && 32 <= lanes.1 {
            let made = made.unwrap_or_else(|| panic!("a host of {lanes:?} lanes makes it"));
            let console: Vec<u32> = (0..groups).flat_map(|_| (0..64).map(|t| t ^ 31)).collect();
            assert_eq!(run_wave64_program(device, &made, groups), console);
        } else {
            assert!(made.is_none(), "{lanes:?}");
            let why = format!(
                "compute shader needs 32-lane waves; the host has {} to {}",
                lanes.0, lanes.1
            );
            let refused = refusals().unwrap_or_default();
            assert!(refused.contains(&why), "{refused}");
            assert!(
                driver_refuses(device, &halves),
                "the device makes WaveSize(32) pipelines though its waves are {lanes:?} lanes: \
                 Needs refuses them wrongly"
            );
        }
    }

    /// Runs `p`, a pipeline of [`crate::shader_cache::wave64_program`],
    /// over `groups` groups: the dwords its threads store.
    fn run_wave64_program(host: &crate::host::Host, p: &HostPipeline, groups: u32) -> Vec<u32> {
        use crate::testgpu::{TitleBuffer, TitleList};
        let words = 64 * groups;
        let bytes = 4 * u64::from(words);
        let dst = TitleBuffer::new(1 << 16);
        dst.set_words(|_| 0xdead_beef);
        let to = dst.shadow();
        // SAFETY: plain heap creation.
        let heap: ID3D12DescriptorHeap = unsafe {
            host.device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 2,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
                .expect("a view heap")
        };
        // The raw UAV at slot 1: the table's user data is its offset in the
        // heap at the console's 32 bytes per descriptor.
        let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    FirstElement: to.offset(dst.address()) / 4,
                    NumElements: words,
                    StructureByteStride: 0,
                    CounterOffsetInBytes: 0,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                },
            },
        };
        // SAFETY: a slot of the heap made above, a live buffer covering the
        // view.
        unsafe {
            let step = host
                .device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            let start = heap.GetCPUDescriptorHandleForHeapStart();
            host.device.CreateUnorderedAccessView(
                &to.buffer,
                None,
                Some(&uav),
                D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: start.ptr + step as usize,
                },
            );
        }
        let (param, dwords) = p.stages[0].user_data.expect("the table's user data");
        let mut data = vec![0u32; dwords as usize];
        data[0] = 32;
        let list = TitleList::new();
        let l = list.host();
        to.refresh(l, dst.address(), bytes);
        to.gpu_writes(l, dst.address(), bytes);
        l.keep(&heap);
        l.keep(&p.root);
        l.keep(&p.pso);
        // SAFETY: an open list; live objects kept until it is reset.
        unsafe {
            l.list
                .SetDescriptorHeaps(&crate::descriptor::heaps_with_own_samplers(Some(&heap)));
            l.list.SetComputeRootSignature(&p.root);
            l.list.SetPipelineState(&p.pso);
            l.list
                .SetComputeRoot32BitConstants(param, dwords, data.as_ptr().cast(), 0);
            l.list.Dispatch(groups, 1, 1);
        }
        list.run();
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
        dst.words()[..words as usize].to_vec()
    }

    /// Whether the device's driver refuses a compute pipeline of `shader`,
    /// made without [`Needs`]' checks.
    fn driver_refuses(host: &crate::host::Host, shader: &xtp_shader::emit::HostShader) -> bool {
        let params = [D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: USER_DATA_SPACE,
                    Num32BitValues: shader.user_data_dwords().max(1),
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        }];
        let (root, _) =
            root_signature(&host.device, &params, &[], true).expect("its root signature");
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            // SAFETY: a borrowed pointer the description never releases.
            pRootSignature: unsafe { std::mem::transmute_copy(&root) },
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: shader.dxil.as_ptr().cast(),
                BytecodeLength: shader.dxil.len(),
            },
            ..Default::default()
        };
        // A probe: the layer reports a description the driver refuses.
        let made: windows::core::Result<ID3D12PipelineState> = crate::testgpu::expect_layer_errors(
            // SAFETY: the description points at live data.
            || unsafe { host.device.CreateComputePipelineState(&desc) },
        );
        made.is_err()
    }

    /// `XTP_SHADER_LOOP_COUNT` end to end (S12): a compute program whose
    /// lanes 0 to 4 loop past the loop limit, translated with the counter,
    /// gets the counter's root UAV and a slot, and each group it runs adds
    /// the 5 lanes the limit stops to that slot's counter. Translated
    /// without the counter, its pipeline has neither.
    ///
    /// Four pipelines of it take four slots, each residue modulo 4 for
    /// consecutive ones: each binds its counter at a 16-byte aligned
    /// address, which the debug layer checks of a raw view's
    /// (`XTP_D3D12_DEBUG=1`), and counts its own groups alone. A snapshot
    /// recorded between two lists, as the present's list records one
    /// (`loopstat::record`), holds what the lists before it counted, not
    /// what the list after it adds to the live counters.
    #[test]
    fn a_loop_counter_counts_the_lanes_a_limit_stops() {
        use crate::testgpu::TitleList;
        use xtp_shader::native::{ComputeIo, StageIo};
        let test = "pipeline::host::tests::a_loop_counter_counts_the_lanes_a_limit_stops";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let device = gpu.host();
        // Lane t loops to 1, or past any limit where t < 5. The loop steps
        // by a register, so it is no counted loop: the general limit.
        let words = [
            0x7E02_0280u32, // v_mov_b32 v1, 0
            0x7E0A_0281,    // v_mov_b32 v5, 1
            0x7E08_0281,    // v_mov_b32 v4, 1
            0x7E0C_02FF,
            0x7FFF_FFFF, // v_mov_b32 v6, 0x7fffffff
            0x7D88_0085, // v_cmp_gt_u32 vcc_lo, 5, v0
            0x0208_0D04, // v_cndmask_b32 v4, v4, v6, vcc_lo
            0xBE88_0380, // s_mov_b32 s8, 0
            0x4A02_0305, // loop: v_add_nc_u32 v1, v5, v1
            0x7D8C_0901, // v_cmp_ge_u32 vcc_lo, v1, v4
            0x8808_086A, // s_or_b32 s8, vcc_lo, s8
            0x8A7E_087E, // s_andn2_b32 exec_lo, exec_lo, s8
            0xBF89_FFFB, // s_cbranch_execnz loop
            0x887E_087E, // s_or_b32 exec_lo, exec_lo, s8
            0xBF81_0000, // s_endpgm
        ];
        let (container, template) = crate::shader_cache::wave64_program(false);
        let native = xtp_shader::NativeProgram {
            wave64: false,
            vgprs: 8,
            layout: Vec::new(),
            used: Vec::new(),
            io: StageIo::Cs(ComputeIo {
                rsrc2: 0,
                threads: [64, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            ..template
        };
        let translate = || {
            xtp_shader::emit::translate_program(&container, &native)
                .unwrap_or_else(|(step, e)| panic!("{}: {e}", step.name()))
        };
        let pipeline = |host: xtp_shader::emit::HostShader, hash: &str| {
            let shader = Arc::new(Shader {
                stage: "compute",
                hash: hash.into(),
                ambient: None,
                container: Vec::new(),
                native: std::sync::OnceLock::new(),
                translated: std::sync::OnceLock::from(Some(Translated {
                    host,
                    used: Vec::new(),
                    wave64: false,
                })),
            });
            HostPipeline::new(&[shader], None).expect("a pipeline")
        };
        if !xtp_shader::emit::loop_count() {
            let plain = translate();
            assert!(!plain.loop_counter());
            assert_eq!(pipeline(plain, "loop-count-off").loop_counter, None);
        }
        let counting = xtp_shader::emit::with_loop_count(translate);
        assert!(counting.loop_counter());
        let pipelines: Vec<HostPipeline> = (0..4)
            .map(|n| pipeline(counting.clone(), &format!("loop-count-on-{n}")))
            .collect();
        let slots: Vec<u32> = pipelines
            .iter()
            .map(|p| p.loop_counter.expect("the counter's root parameter").1)
            .collect();
        let distinct: std::collections::HashSet<u32> = slots.iter().copied().collect();
        assert_eq!(distinct.len(), 4, "one slot per pipeline: {slots:?}");
        let counted = || -> Vec<u32> {
            slots
                .iter()
                .map(|&slot| crate::loopstat::counted(slot).expect("the counters"))
                .collect()
        };
        // Runs `groups` groups of `p` in `l`, its counter bound as the draw
        // state binds it.
        let dispatch = |l: &crate::host::HostList, p: &HostPipeline, groups: u32| {
            let (param, slot) = p.loop_counter.expect("the counter's root parameter");
            let at = crate::loopstat::address(slot).expect("the slot's address");
            assert_eq!(at % 16, 0, "slot {slot}: a raw view's address");
            l.keep(&p.root);
            l.keep(&p.pso);
            let user = p.stages[0].user_data;
            let zeros = vec![0u32; user.map_or(0, |(_, n)| n as usize)];
            // SAFETY: an open list; live objects kept until it is reset;
            // the counters live for good.
            unsafe {
                l.list.SetComputeRootSignature(&p.root);
                l.list.SetPipelineState(&p.pso);
                if let Some((param, n)) = user {
                    l.list
                        .SetComputeRoot32BitConstants(param, n, zeros.as_ptr().cast(), 0);
                }
                l.list.SetComputeRootUnorderedAccessView(param, at);
                l.list.Dispatch(groups, 1, 1);
            }
        };
        let before = counted();
        let list = TitleList::new();
        let l = list.host();
        // The root signatures index both heaps directly: both are bound
        // before one is set.
        crate::testgpu::bind_indexed_heaps(l);
        for (n, p) in pipelines.iter().enumerate() {
            dispatch(l, p, n as u32 + 1);
        }
        list.run();
        // SAFETY: a plain device query.
        assert!(unsafe { device.device.GetDeviceRemovedReason() }.is_ok());
        let after = counted();
        for n in 0..4 {
            assert_eq!(
                after[n].wrapping_sub(before[n]),
                5 * (n as u32 + 1),
                "slot {}, of slots {slots:?}",
                slots[n]
            );
        }
        // The snapshot, then more lanes stopped after it.
        let list = TitleList::new();
        let snapshot = crate::loopstat::snapshot_in(device, &list.host().list)
            .expect("a snapshot of the counters");
        list.run();
        let list = TitleList::new();
        let l = list.host();
        crate::testgpu::bind_indexed_heaps(l);
        dispatch(l, &pipelines[0], 2);
        list.run();
        // SAFETY: a plain device query.
        assert!(unsafe { device.device.GetDeviceRemovedReason() }.is_ok());
        let copied: Vec<u32> = slots
            .iter()
            .map(|&slot| snapshot.counted(slot).expect("copied"))
            .collect();
        assert_eq!(copied, after, "the snapshot: slots {slots:?}");
        let live = counted();
        assert_eq!(live[0].wrapping_sub(after[0]), 10, "slot {}", slots[0]);
        assert_eq!(live[1..], after[1..], "slots {slots:?}");
    }

    /// Where the parent's library file is, for [`library_child`].
    const LIBRARY_CHILD: &str = "XTP_LIBRARY_CHILD";

    /// A translated console mesh program: three corners from the lane
    /// number (`xtp-shader` `emit/mesh_tests.rs`' triangle program), their
    /// position exported, then triangle (0, 1, 2); none on a host without
    /// mesh shaders.
    fn triangle_mesh(host: &crate::host::Host) -> Option<xtp_shader::emit::HostShader> {
        use xtp_shader::native::{MeshIo, NativeProgram, StageIo};
        if !host.mesh_shaders() {
            return None;
        }
        let words: [u32; 19] = [
            0xD765_0001,
            0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
            0x3604_0281, // v_and_b32 v2, 1, v1
            0x3606_0282, // v_and_b32 v3, 2, v1
            0x7E04_0D02, // v_cvt_f32_u32 v2, v2
            0x7E06_0D03, // v_cvt_f32_u32 v3, v3
            0x1008_04F6, // v_mul_f32 v4, 4.0, v2
            0x0608_08F3, // v_add_f32 v4, -1.0, v4
            0x100A_06F4, // v_mul_f32 v5, 2.0, v3
            0x060A_0AF3, // v_add_f32 v5, -1.0, v5
            0x7E0C_02F0, // v_mov_b32 v6, 0.5
            0x7E0E_02F2, // v_mov_b32 v7, 1.0
            0xF800_08CF,
            0x0706_0504, // exp pos0, v4, v5, v6, v7 done
            0x7E12_02FF,
            1 << 10 | 2 << 20, // v_mov_b32 v9, triangle (0, 1, 2)
            0xF800_0941,
            0x0000_0009, // exp prim, v9 done
            0xBF81_0000, // s_endpgm
        ];
        let native = NativeProgram {
            stage: xtp_shader::HwStage::Ms,
            wave64: false,
            float_mode: 0,
            vgprs: 16,
            layout: Vec::new(),
            used: Vec::new(),
            io: StageIo::Ms(MeshIo {
                outputs: Vec::new(),
                clip_cntl: 0,
                threads: 32,
                max_vertices: 32,
                max_primitives: 32,
                topology: 2,
                lds_bytes: 0,
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        // Its output signature: SV_Position, a float4 in row 0.
        let mut osg1 = Vec::new();
        for v in [1u32, 8, 0, 40, 0, 1, 3, 0] {
            osg1.extend(v.to_le_bytes());
        }
        osg1.extend([0xf, 0, 0, 0, 0, 0, 0, 0]);
        osg1.extend(b"SV_Position\0");
        let container = xtp_shader::Container {
            digest: [0; 16],
            version: (1, 0),
            parts: vec![xtp_shader::Part {
                fourcc: xtp_shader::FourCC::OSG1,
                offset: 0,
                data: &osg1,
            }],
        };
        let target = crate::device::host_target();
        let ms = xtp_shader::emit::translate_program_for(&container, &native, target)
            .map_err(|(_, e)| e.to_string())
            .expect("the mesh program translates");
        Some(ms)
    }

    /// Three pipelines of the same state, each as a session describes it:
    /// a vertex one as a classic description, the same as a subobject
    /// stream ([`stream_of`], as the depth-bounds ones), and a mesh one
    /// ([`MeshStages`], where the host has mesh shaders). `f` gets their
    /// descriptions, each pointing at live data.
    fn with_descs<R>(
        host: &crate::host::Host,
        f: impl FnOnce(
            &D3D12_GRAPHICS_PIPELINE_STATE_DESC,
            &D3D12_PIPELINE_STATE_STREAM_DESC,
            Option<&D3D12_PIPELINE_STATE_STREAM_DESC>,
        ) -> R,
    ) -> R {
        use windows::core::s;
        let vs = compiled(
            xtp_shader::HwStage::Vs,
            "float4 main(uint id : SV_VertexID) : SV_Position { return float4(id, 0, 0, 1); }",
            s!("vs_5_1"),
        );
        let ps = compiled(
            xtp_shader::HwStage::Ps,
            "float4 main() : SV_Target { return float4(0.25, 0.5, 0.75, 1); }",
            s!("ps_5_1"),
        );
        let (root, _) = root_signature(&host.device, &[], &[], false).unwrap();
        let g = state(&[28], 0);
        let code = |dxil: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: dxil.as_ptr().cast(),
            BytecodeLength: dxil.len(),
        };
        let classic = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            // SAFETY: a borrowed pointer the description never releases.
            pRootSignature: unsafe { std::mem::transmute_copy(&root) },
            VS: code(&vs.host.dxil),
            PS: code(&ps.host.dxil),
            BlendState: g.blend,
            SampleMask: g.sample_mask,
            RasterizerState: g.rasterizer,
            DepthStencilState: g.depth,
            IBStripCutValue: g.strip_cut,
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: g.targets.NumRenderTargets,
            RTVFormats: g.targets.RTFormats,
            DSVFormat: g.depth_format,
            SampleDesc: g.samples,
            ..Default::default()
        };
        #[repr(C)]
        struct VertexStages(
            Sub<D3D12_SHADER_BYTECODE>,
            Sub<D3D12_SHADER_BYTECODE>,
            Sub<D3D12_PRIMITIVE_TOPOLOGY_TYPE>,
        );
        let vertex = VertexStages(
            Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, classic.VS),
            Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, classic.PS),
            Sub(
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY,
                D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            ),
        );
        let mut streamed = stream_of(host, &root, &g, vertex);
        // The mesh pipeline, with the root signature a session makes for
        // its translation.
        let ms = triangle_mesh(host);
        let mesh_root = ms.as_ref().map(|ms| {
            let shader = title_shader(
                "mesh",
                "library test mesh",
                Some(Translated {
                    host: ms.clone(),
                    used: Vec::new(),
                    wave64: false,
                }),
            );
            HostPipeline::new(&[shader], Some(&g))
                .expect("a mesh pipeline")
                .root
        });
        let mut meshed = ms.as_ref().zip(mesh_root.as_ref()).map(|(ms, root)| {
            let stages =
                MeshStages::new(code(&ms.dxil), D3D12_SHADER_BYTECODE::default(), &ms.dxil);
            stream_of(host, root, &g, stages)
        });
        let meshed = meshed.as_mut().map(Stream::desc);
        f(&classic, &streamed.desc(), meshed.as_ref())
    }

    /// The mesh stages as builds before 284010f streamed them: the mesh and
    /// pixel programs, no primitive topology.
    #[repr(C)]
    struct Before(Sub<D3D12_SHADER_BYTECODE>, Sub<D3D12_SHADER_BYTECODE>);

    /// The mesh pipeline of the stream description `meshed` (made by
    /// [`with_descs`] from a `Stream<MeshStages>`) as a build before
    /// 284010f streamed it: without its primitive topology.
    ///
    /// # Safety
    /// `meshed` points at a live `Stream<MeshStages>`.
    unsafe fn before(meshed: &D3D12_PIPELINE_STATE_STREAM_DESC) -> Stream<Before> {
        // SAFETY: the caller's.
        let now = unsafe {
            &*meshed
                .pPipelineStateSubobjectStream
                .cast::<Stream<MeshStages>>()
        };
        Stream {
            root: Sub(now.root.0, now.root.1),
            stages: Before(
                Sub(now.stages.0.0, now.stages.0.1),
                Sub(now.stages.1.0, now.stages.1.1),
            ),
            blend: Sub(now.blend.0, now.blend.1),
            mask: Sub(now.mask.0, now.mask.1),
            raster: Sub(now.raster.0, now.raster.1),
            depth: Sub(now.depth.0, now.depth.1),
            targets: Sub(now.targets.0, now.targets.1),
            dsv: Sub(now.dsv.0, now.dsv.1),
            samples: Sub(now.samples.0, now.samples.1),
        }
    }

    /// What loads, in a library of `blob`, of [`with_descs`]' three
    /// pipelines stored under the names "classic", "stream" and "mesh",
    /// and under "before" the mesh pipeline a build before 284010f made
    /// and stored ([`before`]), loaded with this build's stream.
    fn loads(host: &crate::host::Host, blob: &[u8]) -> String {
        use windows::core::{HSTRING, Interface, PCWSTR};
        let device = host.device.cast::<ID3D12Device1>().unwrap();
        // SAFETY: `blob` outlives the library, dropped first.
        let library: ID3D12PipelineLibrary1 = unsafe {
            device
                .CreatePipelineLibrary::<ID3D12PipelineLibrary>(blob)
                .unwrap()
                .cast()
                .unwrap()
        };
        let said = |r: windows::core::Result<ID3D12PipelineState>| match r {
            Ok(_) => "ok".to_owned(),
            Err(e) => format!("{:?}", e.code()),
        };
        let name = |name: &str| HSTRING::from(name);
        let (classic, streamed, meshed) = with_descs(host, |classic, streamed, meshed| {
            // SAFETY: a live library; the descriptions point at live data.
            unsafe {
                (
                    library.LoadGraphicsPipeline(PCWSTR(name("classic").as_ptr()), classic),
                    library.LoadPipeline(PCWSTR(name("stream").as_ptr()), streamed),
                    meshed.map(|m| {
                        (
                            library.LoadPipeline(PCWSTR(name("mesh").as_ptr()), m),
                            library.LoadPipeline(PCWSTR(name("before").as_ptr()), m),
                        )
                    }),
                )
            }
        });
        let meshed = meshed.map_or_else(
            || "mesh=none".to_owned(),
            |(now, before)| format!("mesh={} before={}", said(now), said(before)),
        );
        format!(
            "classic={} stream={} {meshed}",
            said(classic),
            said(streamed)
        )
    }

    /// The child of [`pipelines_load_from_the_library_file`]: run by it
    /// alone, it loads the parent's library file and writes what loads.
    #[test]
    #[ignore = "run by pipelines_load_from_the_library_file"]
    fn library_child() {
        let Some(dir) = std::env::var_os(LIBRARY_CHILD).map(std::path::PathBuf::from) else {
            return;
        };
        let Some(gpu) = crate::testgpu::device("pipeline::host::tests::library_child") else {
            return;
        };
        let blob = std::fs::read(dir.join("library.bin")).unwrap();
        let said = loads(gpu.host(), &blob);
        std::fs::write(dir.join("loads.txt"), said).unwrap();
    }

    /// The pipelines a session stores load from the library file, in it
    /// and in the next process: a classic description, a subobject stream,
    /// and a mesh one. The mesh stream had no primitive topology; D3D12
    /// keeps the one its program outputs in the pipeline it makes, so the
    /// stream without it did not load what it had stored (D3D12 message
    /// 972, then 970 at the store under the same name: on the L4, every
    /// mesh pipeline compiled again in every session). The stream without
    /// it is checked to fail here too, so that this test tells the cause.
    #[test]
    fn pipelines_load_from_the_library_file() {
        use windows::core::{HSTRING, Interface, PCWSTR};
        let test = "pipeline::host::tests::pipelines_load_from_the_library_file";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let device = host.device.cast::<ID3D12Device1>().unwrap();
        let device2 = host.device.cast::<ID3D12Device2>().unwrap();
        // SAFETY: an empty library, which keeps no pointer to the slice.
        let library: ID3D12PipelineLibrary1 = unsafe {
            device
                .CreatePipelineLibrary::<ID3D12PipelineLibrary>(&[])
                .unwrap()
                .cast()
                .unwrap()
        };
        let name = |name: &str| HSTRING::from(name);
        let mesh = with_descs(host, |classic, streamed, meshed| {
            // SAFETY: live descriptions, a live library.
            unsafe {
                let made: ID3D12PipelineState =
                    host.device.CreateGraphicsPipelineState(classic).unwrap();
                library
                    .StorePipeline(PCWSTR(name("classic").as_ptr()), &made)
                    .unwrap();
                let made: ID3D12PipelineState = device2.CreatePipelineState(streamed).unwrap();
                library
                    .StorePipeline(PCWSTR(name("stream").as_ptr()), &made)
                    .unwrap();
                if let Some(m) = meshed {
                    let made: ID3D12PipelineState = device2.CreatePipelineState(m).unwrap();
                    library
                        .StorePipeline(PCWSTR(name("mesh").as_ptr()), &made)
                        .unwrap();
                    // The library a build before 284010f left: its stream
                    // made and stored the pipeline (Codex review of
                    // d286ffd, point 4). The name a session gives it is the
                    // same in both builds (the topology is not in it, the
                    // program's `PSV0` is, unchanged); D3D12 matches the
                    // description, which this build's stream must load.
                    let made: ID3D12PipelineState =
                        device2.CreatePipelineState(&before(m).desc()).unwrap();
                    library
                        .StorePipeline(PCWSTR(name("before").as_ptr()), &made)
                        .unwrap();
                }
            }
            meshed.is_some()
        });
        if !mesh {
            // Strict mode fails here: the mesh pipeline is what it checks.
            crate::testgpu::skip(test, "no mesh shaders on this host");
        }
        let all = if mesh {
            "classic=ok stream=ok mesh=ok before=ok"
        } else {
            "classic=ok stream=ok mesh=none"
        };
        // SAFETY: a live library; the buffer is its serialised size.
        let blob = unsafe {
            let mut blob = vec![0u8; library.GetSerializedSize()];
            library.Serialize(&mut blob).unwrap();
            blob
        };
        assert_eq!(loads(host, &blob), all, "this process");

        // The mesh stream as it was, without its topology, does not load
        // the entry: the cause, not only the cure.
        if let Some(ms) = triangle_mesh(host) {
            with_descs(host, |_, _, meshed| {
                let meshed = meshed.expect("a mesh pipeline");
                // SAFETY: the stream description was made by `with_descs`
                // from a `Stream<MeshStages>`, alive meanwhile.
                let mut before = unsafe { before(meshed) };
                assert_eq!(
                    mesh_topology(&ms.dxil),
                    D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE
                );
                let desc = before.desc();
                let refused = crate::testgpu::expect_layer_errors(|| {
                    // SAFETY: a live library; the stream points at live data.
                    unsafe {
                        library.LoadPipeline::<_, ID3D12PipelineState>(
                            PCWSTR(name("mesh").as_ptr()),
                            &desc,
                        )
                    }
                });
                assert!(refused.is_err(), "the stream without its topology");
            });
        }

        // The next process, from the file.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("library.bin"), &blob).unwrap();
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "pipeline::host::tests::library_child",
                "--exact",
                "--ignored",
                "--test-threads=1",
            ])
            .env(LIBRARY_CHILD, dir.path())
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stdout)
        );
        // The same machine: a child without the parent's device is a
        // failure, not a pass without the next process's answer.
        let said = std::fs::read_to_string(dir.path().join("loads.txt")).unwrap_or_else(|_| {
            panic!(
                "the child answered nothing: {}",
                String::from_utf8_lossy(&out.stderr)
            )
        });
        assert_eq!(said, all, "the next process");
    }

    /// A mesh program's topology is its `PSV0`'s: lines, triangles, or
    /// undefined without one.
    #[test]
    fn mesh_topologies_are_read_from_the_container() {
        let with = |topology: u8| {
            let mut psv = 48u32.to_le_bytes().to_vec();
            let mut info = [0u8; 48];
            info[27] = topology;
            psv.extend(info);
            // A DXBC container of that part alone.
            let header = 0x20 + 4;
            let mut out = b"DXBC".to_vec();
            out.extend([0; 16]);
            out.extend(1u16.to_le_bytes());
            out.extend(0u16.to_le_bytes());
            out.extend(((header + 8 + psv.len()) as u32).to_le_bytes());
            out.extend(1u32.to_le_bytes());
            out.extend((header as u32).to_le_bytes());
            out.extend(b"PSV0");
            out.extend((psv.len() as u32).to_le_bytes());
            out.extend(psv);
            out
        };
        assert_eq!(mesh_topology(&with(1)), D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE);
        assert_eq!(
            mesh_topology(&with(2)),
            D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE
        );
        assert_eq!(
            mesh_topology(&with(0)),
            D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED
        );
        assert_eq!(
            mesh_topology(b"not a container"),
            D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED
        );
    }
}

#[cfg(test)]
#[path = "../dispatch_split_tests.rs"]
mod indirect_dispatch_tests;
