//! Draws on the host GPU (decision 0007, step 3). A list keeps the state
//! the title sets for its draws (pipeline, descriptor heaps, root
//! arguments, vertex and index buffers, targets); each draw then brings the
//! host list up to date: the host pipeline and our root signature, the
//! user data the translated shaders read, the host copies of the buffers
//! and descriptors they read, and the states of the textures involved.
//! State the host keeps across draws (viewports, scissors, blend factor,
//! stencil reference, topology, targets) is set on the host list as the
//! title sets it.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use xtp_shader::{ResourceType, RootType, UsedResource};

use crate::descriptor::View;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows::core::Interface;

use crate::host::HostList;
use crate::object::This;
use crate::pipeline::host::HostPipeline;
use crate::rectangles::Records;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// Counts an event and logs it at the first and each power of two after.
macro_rules! counted {
    ($level:ident, $($arg:tt)*) => {{
        static N: AtomicU64 = AtomicU64::new(0);
        let count = N.fetch_add(1, Ordering::Relaxed) + 1;
        if count.is_power_of_two() {
            tracing::$level!(count, $($arg)*);
        }
    }};
}

/// A root argument as the title set it.
#[derive(Debug, Clone, Default)]
enum Root {
    #[default]
    None,
    /// A descriptor table's first descriptor (an address in a heap).
    Table(u64),
    Constants(Vec<u32>),
    /// A root view's GPU address.
    View(u64),
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct VertexView {
    address: u64,
    size: u32,
    stride: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct IndexView {
    pub(crate) address: u64,
    pub(crate) size: u32,
    pub(crate) format: u32,
}

/// Copies des vues liées quand une sortie manque. Le titre peut écraser ses
/// descripteurs ou changer de pipeline entre OMSetRenderTargets et le dessin.
#[derive(Default)]
struct IncompleteTargets {
    colors: Vec<SavedTarget>,
    depth: Option<(ID3D12Resource, Option<D3D12_DEPTH_STENCIL_VIEW_DESC>)>,
    pipeline: usize,
}

enum SavedTarget {
    Real(ID3D12Resource, Option<D3D12_RENDER_TARGET_VIEW_DESC>),
    Missing(D3D12_RENDER_TARGET_VIEW_DESC),
}

fn null_view(
    mut desc: D3D12_RENDER_TARGET_VIEW_DESC,
    pipeline: usize,
    slot: usize,
) -> D3D12_RENDER_TARGET_VIEW_DESC {
    if let Some((format, multisampled)) = crate::pipeline::null_target_view(pipeline as This, slot)
    {
        if format.0 != 0 {
            desc.Format = format;
        }
        desc.ViewDimension = if multisampled {
            D3D12_RTV_DIMENSION_TEXTURE2DMS
        } else {
            D3D12_RTV_DIMENSION_TEXTURE2D
        };
    }
    desc
}

impl IncompleteTargets {
    fn rebind(&mut self, list: &HostList, pipeline: usize) {
        if self.pipeline == pipeline {
            return;
        }
        let Some(host) = crate::host::host() else {
            return;
        };
        let rtvs: Vec<_> = self
            .colors
            .iter()
            .enumerate()
            .map(|(slot, view)| match view {
                SavedTarget::Real(texture, desc) => host.rtv(texture, desc.as_ref()),
                SavedTarget::Missing(desc) => host.rtv(
                    None::<&ID3D12Resource>,
                    Some(&null_view(*desc, pipeline, slot)),
                ),
            })
            .collect();
        let dsv = self
            .depth
            .as_ref()
            .map(|(texture, desc)| host.dsv(texture, desc.as_ref()));
        // SAFETY: ces vues viennent d’être recréées dans les tas CPU de l’hôte.
        unsafe {
            list.list.OMSetRenderTargets(
                rtvs.len() as u32,
                Some(rtvs.as_ptr()),
                false,
                dsv.as_ref().map(std::ptr::from_ref),
            )
        };
        self.pipeline = pipeline;
    }
}

/// Cache entries first created while preparing one bind.
enum NewDescriptor {
    Table((u64, u64)),
    RootSrv((usize, u64, usize, u64)),
    RootUav((usize, usize, u64, u64)),
}

/// What one title list has set, and what its host list was last given.
#[derive(Default)]
pub(crate) struct State {
    pub(crate) occlusion: crate::query::Active,
    pipeline: usize,
    /// View heap, sampler heap.
    heaps: [usize; 2],
    graphics: Vec<Root>,
    compute: Vec<Root>,
    /// A spliced trace's structure whose host TLAS the dispatch takes at
    /// its submission, not found when it was bound ([`State::bind_tlas`]):
    /// the title's view of it and the TLAS's root parameter.
    tlas_at_submission: Option<(u64, u32)>,
    /// The title's graphics and compute root signatures, for
    /// [`State::set_root_signature`].
    title_roots: [usize; 2],
    vertex: [VertexView; 32],
    index: IndexView,
    /// Bound render target and depth resources, for their host states.
    targets: Vec<usize>,
    missing_targets: u8,
    incomplete_targets: Option<Box<IncompleteTargets>>,
    /// The targets and depth the title set last, as the host made their
    /// views (`set_targets`): set again after a pass of ours between its
    /// commands (`State::after_our_pass`).
    title_target_views: Option<Box<IncompleteTargets>>,
    depth: usize,
    /// The title's primitive topology (the host's may differ).
    topology: u32,
    /// What the host list has: pipeline, root signature, heaps.
    host_pipeline: usize,
    /// Graphics, compute.
    host_roots: [usize; 2],
    // Racines validées avant le bridage, posées après sa passe de calcul.
    pending_compute: Option<Vec<(u32, ComputeRoot)>>,
    host_heaps: [usize; 2],
    /// The host heap behind each key of [`State::host_heaps`] this
    /// recording has used, by object: (key, heap). A title heap released
    /// while the list records and another made at its address keep the
    /// key, not the host heap ([`State::bind_host_heaps`]).
    heap_objects: Vec<(usize, usize)>,
    /// Tables made afresh in this list, by title table and descriptor words.
    /// GPU-filled tables also depend on the writes preceding their copies.
    fresh_tables: std::collections::HashMap<(u64, u64), FreshTable>,
    /// What the recording holds in host view heaps (its fresh tables' and
    /// own copies' dynamic slots, the shared root SRV views it binds), and
    /// with it [`State::own`]: given back in one batch once the list is
    /// reset or released and the recording has gone (after its submission,
    /// if the list is reset while it runs).
    recording: crate::descriptor::Recording,
    new_descriptors: Vec<NewDescriptor>,
    /// View heaps whose buffer views of slots picked at run time this
    /// recording brought up to date ([`State::picked_buffers`]).
    picked: Vec<usize>,
    /// The pipeline being bound writes buffers through view-heap slots its
    /// shaders pick at run time (`RUN_TIME_WRITES`).
    picking_writes: bool,
    /// For each heap of `picked`, what the recording refreshed and
    /// declared written of its listings (shared with the work the list
    /// runs as it closes).
    picked_uses: Vec<(usize, PickedUse)>,
    /// View heaps this recording asked for a pass over the textures shaders
    /// pick by index ([`State::heap_textures`]).
    heap_textured: Vec<usize>,
    /// Root SRVs bound as one-view tables in this list: (view heap, title
    /// address, host copy, 0 for a view in place or the source write serial + 1
    /// for a copy of the list's own) → slot.
    root_srv_slots: std::collections::HashMap<(usize, u64, usize, u64), u32>,
    /// Une vue brute par segment de tampon et par tas pour cet enregistrement.
    root_uav_slots: std::collections::HashMap<(usize, usize, u64, u64), u32>,
    /// Latest CBV fill for each source buffer and title address.
    cbv_copies: std::collections::HashMap<(usize, u64), CbvCopy>,
    /// A dispatch was dropped since the list's last one that ran: the next
    /// that runs starts with a UAV barrier ([`State::dispatch_dropped`]).
    dropped_dispatch: bool,
    /// Memory of the list's own that root views are copied into.
    own: Own,
}

enum ComputeRoot {
    Constants(Vec<u32>),
    Cbv(u64),
    Srv(u64),
    Table(D3D12_GPU_DESCRIPTOR_HANDLE),
    /// A root UAV: the loop-limit counter's dword (`loopstat`).
    Uav(u64),
}

impl ComputeRoot {
    fn set(self, list: &HostList, param: u32) {
        let l = &list.list;
        // SAFETY: bind a validé les ressources et la racine de ce pipeline.
        unsafe {
            match self {
                Self::Constants(v) => {
                    l.SetComputeRoot32BitConstants(param, v.len() as u32, v.as_ptr().cast(), 0)
                }
                Self::Cbv(gpu) => l.SetComputeRootConstantBufferView(param, gpu),
                Self::Srv(gpu) => l.SetComputeRootShaderResourceView(param, gpu),
                Self::Table(handle) => l.SetComputeRootDescriptorTable(param, handle),
                Self::Uav(gpu) => l.SetComputeRootUnorderedAccessView(param, gpu),
            }
        }
    }
}

impl Drop for State {
    fn drop(&mut self) {
        let own = std::mem::take(&mut self.own);
        self.recording.finish(Box::new(own));
    }
}

/// A root CBV's reach: shaders read up to 64 KiB through one (the
/// translator declares 64 KiB).
const CBV_SPAN: u64 = 1 << 16;

/// Bytes of each root CBV [`State::dump_at_submission`] logs.
#[cfg(feature = "diagnostics")]
const DUMPED_CBV_BYTES: u64 = 384;

/// The way this CBV is filled and the last list write affecting its bytes.
#[derive(Clone, Copy, PartialEq, Eq)]
struct CbvInput {
    gpu_source: Option<usize>,
    writes: u64,
}

impl CbvInput {
    fn at(
        list: &HostList,
        address: u64,
        len: u64,
        shadow: Option<&std::sync::Arc<crate::shadow::Shadow>>,
    ) -> Self {
        let copy = shadow.map(|s| std::sync::Arc::as_ptr(s) as usize);
        let (title, gpu) = list.input_writes(copy, address, len);
        let gpu_source = copy.filter(|&c| {
            gpu != 0 || crate::host::host().and_then(|h| h.fill_writer(address, len)) == Some(c)
        });
        Self {
            gpu_source,
            writes: title.max(gpu),
        }
    }
}

struct FreshTable {
    first: u32,
    constants: Vec<(u64, u64, CbvInput)>,
}

struct CbvCopy {
    input: CbvInput,
    gpu: u64,
    bytes: u64,
}

/// The bytes of a root SRV not on 16 bytes of its host copy that are
/// copied for its view: what was brought up to date for root SRVs before
/// they were bounded (the view has no size of its own).
const MISALIGNED_SPAN: u64 = xtp_shader::resource::MISALIGNED_SRV_BYTES as u64;

/// Bytes of the list's own memory per chunk, past which nothing is placed:
/// [`CBV_SPAN`] more follow, so a root CBV copied there never reads past
/// the chunk.
const OWN_CHUNK: u64 = 1 << 20;

/// Bytes one list copies on the GPU for its root views at most: past it,
/// a bind requiring a new GPU copy is rejected and logged.
const LIST_COPIES: u64 = 128 << 20;

/// GPU memory the lists' own chunks take at most, together.
const ALL_COPIES: u64 = 512 << 20;

/// Chunks of each kind kept for later lists when no list uses them.
const POOLED: usize = 64;

/// GPU memory the lists' own chunks take, used or pooled.
static GPU_OWN: AtomicU64 = AtomicU64::new(0);

/// Chunks no list uses (GPU, upload): a list takes them as it needs them
/// and gives them back when reset, so lists share them.
static POOL: std::sync::Mutex<(Vec<Chunk>, Vec<Chunk>)> =
    std::sync::Mutex::new((Vec::new(), Vec::new()));

/// A root SRV with less of its buffer left than this is logged: a shader
/// reading as much as on the console reads zeros past the buffer's end.
const SMALL_LEFT: u64 = 4 << 10;

/// Memory of a list's own, for root views the host cannot read in place:
/// on the GPU (default heap) what CopyBufferRegion copies from host copies
/// at the list's place, and upload memory filled from title memory at
/// submission. It goes with the list's recording ([`State::recording`]),
/// and its chunks back to [`POOL`] once the recording has gone (its
/// submissions have run).
#[derive(Default)]
struct Own {
    gpu: Vec<Chunk>,
    upload: Vec<Chunk>,
    /// Bytes copied on the GPU since the reset.
    copied: u64,
    /// What the list's indirect rectangle draws are expanded into.
    rects: Option<crate::rectangles::Scratch>,
    indirect_samples: crate::indirect::Samples,
    /// The holds on the raw views of the runtime stride tables the list's
    /// shaders read, by view heap ([`State::stride_table`]).
    strides: Vec<(usize, crate::descriptor::host::StrideHold)>,
    /// Where [`Own::zeros`] put its zeros: upload chunk and offset.
    zeros: Option<(usize, u64)>,
}

struct Chunk {
    buffer: ID3D12Resource,
    /// The CPU address of an upload chunk.
    data: usize,
    gpu: u64,
    /// Bytes things may be placed in (the chunk has [`CBV_SPAN`] more), and
    /// those placed since the reset.
    size: u64,
    used: u64,
    /// The state the list's commands so far leave a GPU chunk in: common
    /// when the list starts (buffers decay to it when a submission ends).
    state: D3D12_RESOURCE_STATES,
    /// Whether the host list keeps it (until its reset).
    kept: bool,
}

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only by the list holding the chunk.
unsafe impl Send for Chunk {}

impl Drop for Own {
    fn drop(&mut self) {
        if let Some(rects) = self.rects.take() {
            crate::rectangles::give_back(rects);
        }
        let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        let pool = &mut *pool;
        for (mine, pooled, gpu) in [
            (&mut self.gpu, &mut pool.0, true),
            (&mut self.upload, &mut pool.1, false),
        ] {
            for mut c in mine.drain(..) {
                if c.size == OWN_CHUNK && pooled.len() < POOLED {
                    c.used = 0;
                    c.state = D3D12_RESOURCE_STATE_COMMON;
                    c.kept = false;
                    pooled.push(c);
                } else if gpu {
                    GPU_OWN.fetch_sub(c.size + CBV_SPAN, Ordering::Relaxed);
                }
            }
        }
    }
}

impl Own {
    /// Room for `len` bytes (256-byte aligned) in the list's GPU or upload
    /// chunks: one with room, one from [`POOL`], or a new one (GPU ones up
    /// to [`ALL_COPIES`] in all). The chunk and the offset.
    fn place(&mut self, gpu: bool, list: &HostList, len: u64) -> Option<(usize, u64)> {
        let len = len.max(1).next_multiple_of(256);
        let chunks = if gpu { &mut self.gpu } else { &mut self.upload };
        let i = match chunks.iter().position(|c| c.used + len <= c.size) {
            Some(i) => i,
            None => {
                let pooled = (len <= OWN_CHUNK)
                    .then(|| {
                        let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
                        if gpu { pool.0.pop() } else { pool.1.pop() }
                    })
                    .flatten();
                let c = match pooled {
                    Some(c) => c,
                    None => new_chunk(gpu, len.max(OWN_CHUNK))?,
                };
                chunks.push(c);
                chunks.len() - 1
            }
        };
        let c = &mut chunks[i];
        if !c.kept {
            list.keep(&c.buffer);
            c.kept = true;
        }
        let at = c.used;
        c.used += len;
        Some((i, at))
    }

    /// Copies `len` bytes of host copy `from` from byte `offset` into GPU
    /// memory of the list's own, at this place in the list (after what
    /// earlier commands wrote there, before later ones read it): the buffer
    /// and the offset in it. The `pad` bytes after them are zeroed (at
    /// most 255: a constant view's [`padding`]). `None` past
    /// [`LIST_COPIES`] or [`ALL_COPIES`] (logged).
    fn gpu_copy(
        &mut self,
        list: &HostList,
        from: &ID3D12Resource,
        offset: u64,
        len: u64,
        pad: u64,
    ) -> Option<(ID3D12Resource, u64)> {
        use crate::host::{transition, uav_barrier};
        debug_assert!(pad < 256);
        let zeros = if pad > 0 {
            Some(self.zeros(list)?)
        } else {
            None
        };
        if self.copied + len > LIST_COPIES {
            counted!(
                warn,
                budget = LIST_COPIES,
                "a list's root view copies on the GPU reached their budget: the next is not made"
            );
            return None;
        }
        let (i, at) = self.place(true, list, len)?;
        self.copied += len;
        let c = &mut self.gpu[i];
        // Pixel shader resources are a state of direct lists only.
        // SAFETY: an open list.
        let direct = unsafe { list.list.GetType() } == D3D12_COMMAND_LIST_TYPE_DIRECT;
        let read = D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER
            | if direct {
                D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE
            } else {
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
            };
        let (common, source, dest) = (
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        #[cfg(feature = "diagnostics")]
        list.note_read(
            || format!("root view copy of {len:#x} bytes on the GPU"),
            // SAFETY: a live buffer.
            (unsafe { from.GetGPUVirtualAddress() } + offset, len),
        );
        // SAFETY: an open list; live buffers the list keeps (the host copy
        // through its refresh), ranges inside both. The host copy is taken
        // as common, as every other use takes it (see `shadow`), after an
        // unordered access barrier so that shader writes before are done.
        unsafe {
            list.list.ResourceBarrier(&[
                uav_barrier(),
                transition(from, common, source),
                transition(&c.buffer, c.state, dest),
            ]);
            list.list.CopyBufferRegion(&c.buffer, at, from, offset, len);
            if let Some((zeros, zeros_at)) = &zeros {
                // Within the 256-byte steps `place` gives.
                list.list
                    .CopyBufferRegion(&c.buffer, at + len, zeros, *zeros_at, pad);
            }
            list.list.ResourceBarrier(&[
                transition(from, source, common),
                transition(&c.buffer, dest, read),
            ]);
        }
        c.state = read;
        Some((c.buffer.clone(), at))
    }

    /// Upload memory of the list's own holding `len` bytes of title memory
    /// at `address` once the list is submitted: its host address.
    /// The [`padding`] after them reads zeros.
    fn upload_fill(&mut self, list: &HostList, address: u64, len: u64) -> Option<u64> {
        let (i, at) = self.place(false, list, len)?;
        let c = &self.upload[i];
        // SAFETY: the padding lies in the place just taken in this mapped
        // chunk, which no GPU work reads before the list's submission; the
        // fill writes only the first `len` bytes.
        unsafe {
            ((c.data + (at + len) as usize) as *mut u8).write_bytes(0, padding(len) as usize)
        };
        list.defer_fill_at(c.data, at as usize, address as usize, len as usize);
        Some(c.gpu + at)
    }

    /// 256 bytes of zeros in the list's upload memory, to copy from on the
    /// GPU: the buffer and the offset in it. Made once per recording.
    fn zeros(&mut self, list: &HostList) -> Option<(ID3D12Resource, u64)> {
        let (i, at) = match self.zeros {
            Some(place) => place,
            None => {
                let (i, at) = self.place(false, list, 256)?;
                let c = &self.upload[i];
                // SAFETY: the place just taken in this mapped chunk; the
                // GPU reads it only after the list's submission.
                unsafe { ((c.data + at as usize) as *mut u8).write_bytes(0, 256) };
                self.zeros = Some((i, at));
                (i, at)
            }
        };
        Some((self.upload[i].buffer.clone(), at))
    }
}

/// The bytes past `len` up to the next 256 of a constant view of the list's
/// own memory: the host view takes whole 256s, and there the console reads
/// zeros (past the view's size), not what earlier lists left.
fn padding(len: u64) -> u64 {
    len.next_multiple_of(256) - len
}

/// Scratch memory of the list's holding `len` bytes of title memory at
/// `address` once it is submitted, for a table's constant view: its host
/// address. The [`padding`] after them reads zeros.
fn scratch_constant(list: &HostList, address: u64, len: u64) -> Option<u64> {
    let pad = padding(len);
    if pad == 0 {
        return list.scratch_fill(address, len);
    }
    let (gpu, to) = list.scratch(len + pad)?;
    // SAFETY: `len + pad` bytes of the list's mapped scratch memory, which
    // no GPU work reads before the list's submission (a list reset while it
    // runs records into fresh chunks); the fill writes only the first `len`.
    unsafe { to.add(len as usize).write_bytes(0, pad as usize) };
    list.defer_fill(to, address as *const u8, len as usize);
    Some(gpu)
}

/// A chunk of `size` bytes and [`CBV_SPAN`] more: in GPU memory (`gpu`,
/// counted in [`GPU_OWN`], up to [`ALL_COPIES`]) or upload memory.
fn new_chunk(gpu: bool, size: u64) -> Option<Chunk> {
    let bytes = size + CBV_SPAN;
    let (buffer, data) = if gpu {
        if GPU_OWN.fetch_add(bytes, Ordering::Relaxed) + bytes > ALL_COPIES {
            GPU_OWN.fetch_sub(bytes, Ordering::Relaxed);
            counted!(
                warn,
                cap = ALL_COPIES,
                "GPU memory for lists' root view copies at its cap: a copy is not made"
            );
            return None;
        }
        match default_buffer(bytes) {
            Some(buffer) => (buffer, 0),
            None => {
                GPU_OWN.fetch_sub(bytes, Ordering::Relaxed);
                return None;
            }
        }
    } else {
        let (buffer, data) = crate::host::host()?.upload_buffer(bytes)?;
        (buffer, data as usize)
    };
    // SAFETY: a live buffer.
    let at = unsafe { buffer.GetGPUVirtualAddress() };
    Some(Chunk {
        buffer,
        data,
        gpu: at,
        size,
        used: 0,
        state: D3D12_RESOURCE_STATE_COMMON,
        kept: false,
    })
}

/// A buffer of `size` bytes in GPU memory (zeroed), in the common state.
fn default_buffer(size: u64) -> Option<ID3D12Resource> {
    let host = crate::host::host()?;
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation of a buffer.
    let made = unsafe {
        host.device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut buffer,
        )
    };
    if let Err(e) = made {
        once!(error = %e, size, "no GPU memory of a list's own for root views");
        return None;
    }
    let buffer = buffer?;
    crate::host::name(&buffer, || format!("list root views {size:#x}"));
    Some(buffer)
}

/// What a recording did with a view heap's listing for shaders that pick
/// slots at run time: its step for the list.
type PickedUse = std::sync::Arc<std::sync::Mutex<crate::descriptor::host::PickedUse>>;

/// The views of `host`'s slots (title heap at `start`) made, and the list's
/// step for them ([`crate::descriptor::host::PickedUse::defer`]): at each
/// submission what they read is brought up to date, then (`writes`, after
/// the refresh) what they may write declared written, so that what the GPU
/// writes goes back to title memory and no later refresh copies title
/// memory over it; once per submission whichever lists ask.
fn picked_refresh(
    list: &HostList,
    host: &crate::descriptor::host::HostHeap,
    start: u64,
    used: &PickedUse,
    writes: bool,
) {
    let Some(picked) = host.picked_spans(start) else {
        return;
    };
    used.lock()
        .unwrap_or_else(|e| e.into_inner())
        .defer(list, &picked, writes);
}

/// The host heap behind a key of [`State::host_heaps`]: a title heap, or
/// XTP's own view or sampler heap.
fn host_heap(key: usize) -> Option<&'static crate::descriptor::host::HostHeap> {
    match key {
        0 => None,
        crate::descriptor::OWN_VIEW_HEAP => crate::descriptor::own_view_heap(),
        crate::descriptor::OWN_SAMPLER_HEAP => crate::descriptor::own_sampler_heap(),
        h => crate::descriptor::heap(h as This).host(),
    }
}

/// Logs where root SRVs land (diagnostics for VIctory #493's bound): once
/// per host copy, the address, the buffer it is viewed in (the innermost
/// holding it, where the view stops) and any buffer around that one, whose
/// memory past the inner buffer the console may read; counted as they
/// come, root SRVs in memory outside every buffer (viewed in a window copy
/// of title memory) and those near their buffer's end.
fn note_root_srv(address: u64, copy: &crate::shadow::Shadow) {
    type Seen = std::collections::HashMap<(usize, u64, u64), bool>;
    static SEEN: std::sync::Mutex<Option<Seen>> = std::sync::Mutex::new(None);
    let left = copy.left(address);
    let start = address - copy.offset(address);
    let len = copy.offset(address) + left;
    let key = (copy.buffer.as_raw() as usize, start, len);
    let outside = {
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        let seen = seen.get_or_insert_with(Default::default);
        match seen.get(&key) {
            Some(&outside) => outside,
            None => {
                let holders = crate::shadow::holders(address);
                if seen.len() >= 1 << 16 {
                    seen.clear();
                }
                seen.insert(key, holders.is_empty());
                let buffer = format!("{start:#x}+{len:#x}");
                match holders.get(1..) {
                    None => {}
                    Some([]) => tracing::debug!(
                        address = format_args!("{address:#x}"),
                        %buffer,
                        "a root SRV's buffer"
                    ),
                    Some(around) => {
                        let around: Vec<String> = around
                            .iter()
                            .map(|(s, l)| format!("{s:#x}+{l:#x}"))
                            .collect();
                        tracing::info!(
                            address = format_args!("{address:#x}"),
                            %buffer,
                            ?around,
                            "a root SRV in a buffer inside another: its view stops at the inner \
                             buffer's end (once per buffer)"
                        );
                    }
                }
                holders.is_empty()
            }
        }
    };
    if outside {
        counted!(
            info,
            address = format_args!("{address:#x}"),
            window = format_args!("{start:#x}+{len:#x}"),
            "a root SRV outside every title buffer: viewed in a window copy of title memory"
        );
    }
    if left < SMALL_LEFT {
        counted!(
            info,
            address = format_args!("{address:#x}"),
            left,
            buffer = format_args!("{start:#x}+{len:#x}"),
            "a root SRV near its buffer's end: reads past it return zeros"
        );
    }
}

/// Logs a root SRV's one-view table (Watch 443, VIctory #1671): the bytes
/// it views from `address` against those its host copy holds from there,
/// to the end of the title buffer or of the 4 MiB window copy of memory
/// outside every buffer. A table that stops before its copy's end is cut
/// (counted at info): a shader reading past the cut reads zeros where the
/// console reads the buffer. Past a window's end the title's memory goes
/// on as well. Each table at debug level.
fn note_root_srv_extent(address: u64, copy: &crate::shadow::Shadow, viewed: u64) {
    let left = copy.left(address);
    let start = address - copy.offset(address);
    let len = copy.offset(address) + left;
    if viewed < left {
        counted!(
            info,
            address = format_args!("{address:#x}"),
            viewed,
            left,
            copy = format_args!("{start:#x}+{len:#x}"),
            "a root SRV's one-view table stops before its host copy's end: reads past it \
             return zeros"
        );
    }
    if tracing::enabled!(tracing::Level::DEBUG) {
        let window = crate::shadow::holders(address).is_empty();
        tracing::debug!(
            address = format_args!("{address:#x}"),
            viewed,
            left,
            copy = format_args!("{start:#x}+{len:#x}"),
            holds = if window {
                "a 4 MiB window"
            } else {
                "a title buffer"
            },
            "a root SRV's one-view table"
        );
    }
}

/// Draws recorded on the host GPU since the last frame line.
static DRAWN: AtomicU64 = AtomicU64::new(0);

/// GPU constant-buffer copies recorded and binds served by the copy cache.
static CBV_COPIES: AtomicU64 = AtomicU64::new(0);
static CBV_COPY_HITS: AtomicU64 = AtomicU64::new(0);

pub(crate) fn take_cbv_copies() -> [u64; 2] {
    [&CBV_COPIES, &CBV_COPY_HITS].map(|c| c.swap(0, Ordering::Relaxed))
}

/// Draws run on the host so far.
pub(crate) fn drawn() -> u64 {
    DRAWN.load(Ordering::Relaxed)
}

/// The draws traced since the last present (diagnostics builds, with
/// debug logging for this module): logged when the frame comes out black.
#[cfg(feature = "diagnostics")]
static FRAME: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// The draws traced since the last call.
#[cfg(feature = "diagnostics")]
pub(crate) fn take_frame() -> Vec<String> {
    std::mem::take(&mut *FRAME.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Draws, mesh dispatches and dispatches recorded but not run on the host
/// (no host pipeline) since the last call.
pub(crate) fn take_dropped() -> u64 {
    DROPPED.swap(0, Ordering::Relaxed)
}
static DROPPED: AtomicU64 = AtomicU64::new(0);

/// Why a draw or dispatch cannot run on the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Unready {
    /// Its pipeline has no host pipeline (yet, or for good):
    /// `pipeline::note_drop` tells which.
    NoPipeline,
    /// Anything else.
    Because(&'static str),
}

impl From<&'static str> for Unready {
    fn from(why: &'static str) -> Unready {
        Unready::Because(why)
    }
}

/// Counts a draw or dispatch of `pipeline`, recorded at `path`, that did
/// not run on the host for `why` (diagnostics).
#[cfg(feature = "diagnostics")]
fn note_unready(path: &str, pipeline: usize, why: Unready) {
    crate::passdump::dropped(path, pipeline, &why);
    match why {
        Unready::NoPipeline => crate::pipeline::note_drop(path, pipeline as This),
        Unready::Because(why) => crate::pipeline::note_why(format!("{path}: {why}")),
    }
}

/// Counts a draw, mesh dispatch or dispatch recorded at `path` that did
/// not run on the host for `why`, and (diagnostics) marks the last traced
/// draw as not run.
fn dropped(path: &str, pipeline: usize, why: Unready) {
    DROPPED.fetch_add(1, Ordering::Relaxed);
    #[cfg(feature = "diagnostics")]
    {
        note_unready(path, pipeline, why);
        if let Some(last) = FRAME.lock().unwrap_or_else(|e| e.into_inner()).last_mut() {
            last.push_str(" DROPPED");
        }
    }
    #[cfg(not(feature = "diagnostics"))]
    let _ = (path, pipeline, why);
}

/// Dispatches run on the host so far.
static DISPATCHED: AtomicU64 = AtomicU64::new(0);

pub(crate) fn dispatched() -> u64 {
    DISPATCHED.load(Ordering::Relaxed)
}

/// The host address of a root SRV bound as a root descriptor (with
/// `XTP_ROOT_SRV_DIRECT=1`): its buffer's host copy.
/// shortcut: a root descriptor has no size; the view is taken as 1 MiB (cut
/// at its buffer's end), and shaders reading past the copy read whatever
/// the host placed after it.
fn root_srv_direct(list: &HostList, address: u64) -> Option<u64> {
    let shadow = crate::shadow::find(address, 1)?;
    let len = shadow.left(address).min(MISALIGNED_SPAN);
    shadow.refresh(list, address, len);
    Some(shadow.gpu(address))
}

/// Why a dispatch of a program spliced with an inline ray query (RT phase
/// 2b, its root SRV at [`xtp_shader::emit::TLAS_REGISTER`]) does not run:
/// the title's view of the structure its walk read is not where the walk
/// read it (`Shader::tlas_descriptor`): no descriptor table bound at that
/// root parameter, or no view at that offset of it.
const NO_TLAS_VIEW: &str = "rt: no title view of the structure";

/// Why it does not run either: the host has no TLAS for the structure the
/// view names that this recording may read (`rt::descriptor_tlas`): none
/// built, or built in a list not submitted yet. A direct dispatch dropped
/// for either is one of a trace program without a host pipeline, as before
/// the splice ([`State::rt_dropped`]).
const NO_HOST_TLAS: &str = "rt: no host TLAS";

fn set(roots: &mut Vec<Root>, slot: u32, root: Root) {
    let slot = slot as usize;
    if roots.len() <= slot {
        roots.resize(slot + 1, Root::None);
    }
    roots[slot] = root;
}

/// Byte-budget tests disable deadlines on their thread (`tests::Untimed`).
fn pass_time_spent(started: std::time::Instant, limit: std::time::Duration) -> bool {
    #[cfg(test)]
    if tests::UNTIMED.get() {
        return false;
    }
    started.elapsed() > limit
}

impl State {
    pub(crate) fn reset(&mut self, pipeline: This) {
        // The old state gives back what it held as it goes: slots, and its
        // own memory to the pool.
        *self = State::default();
        self.pipeline = pipeline as usize;
    }

    /// The title's topology; console-only ones become host lists (see
    /// [`crate::primitive`]).
    pub(crate) fn set_topology(&mut self, list: &HostList, topology: u32) {
        self.topology = topology;
        let host = match topology {
            RECTANGLES => 4,
            TRIANGLE_FAN | LINE_LOOP | QUADS | QUAD_STRIP => {
                #[cfg(feature = "diagnostics")]
                crate::list::cuts::TOPOLOGIES.fetch_add(1, Ordering::Relaxed);
                once!(topology, "console topology drawn as a triangle list");
                4
            }
            t => t,
        };
        // SAFETY: an open list.
        unsafe {
            list.list
                .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(host as i32))
        };
    }

    pub(crate) fn set_pipeline(&mut self, pipeline: This) {
        self.pipeline = pipeline as usize;
    }

    /// The pipeline the title set (tests).
    #[cfg(test)]
    pub(crate) fn pipeline(&self) -> This {
        self.pipeline as This
    }

    /// Work of XTP's own bound its heaps, root signature and pipeline on
    /// the host list (`crate::list`'s upscale and UAV clears): the next draw
    /// or dispatch binds the title's again.
    pub(crate) fn forget_host(&mut self) {
        self.host_pipeline = 0;
        self.host_roots = [0; 2];
        self.host_heaps = [0; 2];
    }

    /// A pass of ours was recorded between the title's commands
    /// (`HostList::planes_now`, which set the viewports and scissors back):
    /// it set a descriptor heap, a graphics root signature, a pipeline, a
    /// depth target and the topology of its own. The title's targets and
    /// topology are set again here; heaps, root signatures and pipeline by
    /// the next `bind`, which also sets every root argument
    /// (`forget_host`). The pass leaves the blend factor, stencil reference
    /// and depth bounds alone.
    fn after_our_pass(&mut self, list: &HostList) {
        self.forget_host();
        // `forget_host` notes no heap, which `bind_host_heaps` takes for
        // what the title wants when it binds none: the pass's own heap
        // stayed. With a key no heap has, the next bind sets the heaps it
        // wants, none included.
        self.host_heaps = [AFTER_OUR_PASS; 2];
        // No topology since the list's reset: none again, as before the
        // pass (a draw then draws nothing).
        if self.topology == 0 {
            // SAFETY: an open list.
            unsafe { list.list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(0)) };
        }
        // The pass may come before a dispatch, under a compute pipeline:
        // the views of missing targets set below then have its formats,
        // and the next draw makes them again for its own pipeline. Before,
        // the draw's own cache still named that pipeline and kept them.
        if let Some(saved) = &mut self.incomplete_targets {
            saved.pipeline = usize::MAX;
        }
        // The pass leaves a triangle list: what the console's own
        // topologies are drawn as too (`set_topology`).
        if !matches!(
            self.topology,
            0 | 4 | RECTANGLES | TRIANGLE_FAN | LINE_LOOP | QUADS | QUAD_STRIP
        ) {
            self.set_topology(list, self.topology);
        }
        match &mut self.title_target_views {
            Some(views) => {
                views.pipeline = usize::MAX;
                views.rebind(list, self.pipeline);
            }
            // SAFETY: an open list.
            None => unsafe { list.list.OMSetRenderTargets(0, None, false, None) },
        }
    }

    pub(crate) fn set_heaps(&mut self, heaps: &[This]) {
        self.heaps = [0; 2];
        for &h in heaps {
            if !h.is_null() {
                let samplers = crate::descriptor::heap(h).samplers();
                self.heaps[usize::from(samplers)] = h as usize;
            }
        }
    }

    /// The title sets its graphics or compute root signature: another one
    /// than the stage has leaves the stage's root arguments unset, as D3D12
    /// does (they are stale until set again); the same one keeps them. The
    /// host's root signature is the translated pipeline's, set by `bind`.
    pub(crate) fn set_root_signature(&mut self, compute: bool, signature: This) {
        let title = &mut self.title_roots[usize::from(compute)];
        if *title != signature as usize {
            *title = signature as usize;
            self.roots(compute).clear();
        }
    }

    fn roots(&mut self, compute: bool) -> &mut Vec<Root> {
        if compute {
            &mut self.compute
        } else {
            &mut self.graphics
        }
    }

    pub(crate) fn set_table(&mut self, compute: bool, slot: u32, handle: u64) {
        set(self.roots(compute), slot, Root::Table(handle));
    }

    pub(crate) fn set_view(&mut self, compute: bool, slot: u32, address: u64) {
        set(self.roots(compute), slot, Root::View(address));
    }

    pub(crate) fn set_constants(&mut self, compute: bool, slot: u32, values: &[u32], offset: u32) {
        let roots = self.roots(compute);
        let s = slot as usize;
        if !matches!(roots.get(s), Some(Root::Constants(_))) {
            set(roots, slot, Root::Constants(Vec::new()));
        }
        if let Root::Constants(c) = &mut roots[s] {
            let end = offset as usize + values.len();
            if c.len() < end {
                c.resize(end, 0);
            }
            c[offset as usize..end].copy_from_slice(values);
        }
    }

    pub(crate) fn set_vertex_buffers(&mut self, start: u32, views: &[VertexView]) {
        for (i, v) in views.iter().enumerate() {
            if let Some(slot) = self.vertex.get_mut(start as usize + i) {
                *slot = *v;
            }
        }
    }

    pub(crate) fn clear_vertex_buffers(&mut self, start: u32, count: u32) {
        for slot in self
            .vertex
            .iter_mut()
            .skip(start as usize)
            .take(count as usize)
        {
            *slot = VertexView::default();
        }
    }

    pub(crate) fn set_index_buffer(&mut self, view: Option<&IndexView>) {
        self.index = view.copied().unwrap_or_default();
    }

    /// Render targets and depth, set on the host list at once.
    pub(crate) fn set_targets(&mut self, list: &HostList, handles: &[u64], depth: u64) {
        let Some(host) = crate::host::host() else {
            return;
        };
        self.targets.clear();
        self.missing_targets = 0;
        let mut title_target_views = IncompleteTargets::default();
        let mut rtvs = Vec::new();
        #[cfg(feature = "diagnostics")]
        let mut hole = false;
        // AgneyaK identified this slot shift in Marvisak/VIctory#1613 (736fa790).
        // Each shader output keeps its slot; only real resources need transitions.
        let views: [_; 8] = std::array::from_fn(|slot| {
            let record = handles
                .get(slot)
                .and_then(|&h| crate::descriptor::rtv_record(h));
            let texture = record.and_then(|r| crate::resource::host_resource(r.resource));
            (record, texture)
        });
        let views = &views[..handles.len().min(8)];
        self.incomplete_targets = views.iter().any(|(_, t)| t.is_none()).then(Box::default);
        for (slot, &(record, texture)) in views.iter().enumerate() {
            if let (Some(record), Some(texture)) = (record, texture) {
                #[cfg(feature = "diagnostics")]
                if hole {
                    crate::list::cuts::SHIFTED_TARGET.fetch_add(1, Ordering::Relaxed);
                }
                self.targets.push(record.resource as usize);
                #[cfg(feature = "diagnostics")]
                crate::capture::note_target(record.resource);
                crate::resource::note_target_view(
                    texture,
                    record.desc().map(|d| d.Format.0 as u32),
                );
                let desc = record.desc();
                rtvs.push(host.rtv(texture, desc.as_ref()));
                title_target_views
                    .colors
                    .push(SavedTarget::Real(texture.clone(), desc));
                if let Some(saved) = &mut self.incomplete_targets {
                    saved.colors.push(SavedTarget::Real(texture.clone(), desc));
                }
            } else {
                use crate::resource::format::{Caps, HostCaps, uses};
                #[cfg(feature = "diagnostics")]
                {
                    hole = true;
                    crate::list::cuts::MISSING_TARGET.fetch_add(1, Ordering::Relaxed);
                }
                self.missing_targets |= 1 << slot;
                let fallback = record
                    .and_then(|r| {
                        r.desc().map(|d| d.Format).or_else(|| {
                            crate::resource::texture_memory(r.resource).map(|(_, d)| {
                                DXGI_FORMAT(crate::resource::pc_format(d.format()) as i32)
                            })
                        })
                    })
                    .filter(|f| HostCaps.support1(f.0 as u32) & uses::RENDER_TARGET != 0)
                    .unwrap_or(windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM);
                let null = D3D12_RENDER_TARGET_VIEW_DESC {
                    Format: fallback,
                    ViewDimension: match record.and_then(|r| r.desc()).map(|d| d.ViewDimension) {
                        Some(
                            D3D12_RTV_DIMENSION_TEXTURE2DMS | D3D12_RTV_DIMENSION_TEXTURE2DMSARRAY,
                        ) => D3D12_RTV_DIMENSION_TEXTURE2DMS,
                        _ => D3D12_RTV_DIMENSION_TEXTURE2D,
                    },
                    ..Default::default()
                };
                rtvs.push(host.rtv(
                    None::<&ID3D12Resource>,
                    Some(&null_view(null, self.pipeline, slot)),
                ));
                title_target_views.colors.push(SavedTarget::Missing(null));
                if let Some(saved) = &mut self.incomplete_targets {
                    saved.colors.push(SavedTarget::Missing(null));
                }
            }
        }
        self.depth = 0;
        let dsv = crate::descriptor::dsv_record(depth).and_then(|record| {
            let texture = crate::resource::host_resource(record.resource)?;
            self.depth = record.resource as usize;
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(record.resource);
            let desc = record.depth_desc();
            title_target_views.depth = Some((texture.clone(), desc));
            if let Some(saved) = &mut self.incomplete_targets {
                saved.depth = Some((texture.clone(), desc));
            }
            Some(host.dsv(texture, desc.as_ref()))
        });
        title_target_views.pipeline = self.pipeline;
        self.title_target_views = Some(Box::new(title_target_views));
        if let Some(saved) = &mut self.incomplete_targets {
            saved.pipeline = self.pipeline;
        }
        // SAFETY: an open list; the views were just made.
        unsafe {
            list.list.OMSetRenderTargets(
                rtvs.len() as u32,
                (!rtvs.is_empty()).then_some(rtvs.as_ptr()),
                false,
                dsv.as_ref().map(std::ptr::from_ref),
            )
        };
        #[cfg(feature = "diagnostics")]
        crate::passdump::targets(list, &self.targets, self.depth);
    }

    /// The user-data value of a root argument, as translated shaders read
    /// it: a table's byte offset in its heap, a constant's value.
    fn user_value(&self, compute: bool, slot: u8, root: RootType, dword: u32) -> u32 {
        match (self.roots_ref(compute).get(usize::from(slot)), root) {
            (Some(Root::Table(handle)), RootType::Table) => match self.heap_of(*handle) {
                Some(heap) => (handle - heap.start) as u32,
                None => {
                    once!(
                        handle = format_args!("{handle:#x}"),
                        "descriptor table outside the bound heaps"
                    );
                    0
                }
            },
            (Some(Root::Constants(c)), RootType::Constants) => {
                c.get(dword as usize).copied().unwrap_or(0)
            }
            _ => {
                once!(
                    slot,
                    ?root,
                    "a shader reads a root argument the title did not set"
                );
                0
            }
        }
    }

    /// The bound heap holding `handle`.
    fn heap_of(&self, handle: u64) -> Option<&'static crate::descriptor::Heap> {
        self.heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| h.index(handle).is_some())
    }

    /// Brings the host list up to date for a draw; why the draw cannot run
    /// on the host if it cannot.
    fn prepare(&mut self, list: &HostList, indexed: bool) -> Result<(), Unready> {
        self.prepare_for(list, indexed, false)
    }

    /// As [`State::prepare`], for draws (`mesh` false) or mesh dispatches.
    fn prepare_for(&mut self, list: &HostList, indexed: bool, mesh: bool) -> Result<(), Unready> {
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            return Err(Unready::NoPipeline);
        }
        let Some(p) =
            crate::pipeline::graphics_pipeline(self.pipeline as This, self.missing_targets)
        else {
            return Err(Unready::NoPipeline);
        };
        if p.compute || p.mesh != mesh {
            once!(
                mesh,
                "a draw or mesh dispatch with another kind of pipeline"
            );
            return Err(match (p.compute, p.mesh) {
                (true, _) => "a compute pipeline bound",
                (false, true) => "a mesh pipeline bound",
                (false, false) => "a vertex pipeline bound",
            }
            .into());
        }
        if let Some(saved) = &mut self.incomplete_targets {
            saved.rebind(list, self.pipeline);
        }
        self.bind(list, p, false, [0; 3])?;
        let l = &list.list;
        // Vertex buffers the input layout reads.
        let mut views = [D3D12_VERTEX_BUFFER_VIEW::default(); 32];
        let mut count = 0;
        for (slot, (view, &v)) in views.iter_mut().zip(&self.vertex).enumerate() {
            if p.slots & 1 << slot == 0 {
                continue;
            }
            count = slot + 1;
            if v.address == 0 || v.size == 0 {
                continue;
            }
            let Some(shadow) = crate::shadow::find(v.address, u64::from(v.size)) else {
                continue;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            #[cfg(feature = "diagnostics")]
            crate::meshstat::read(
                crate::meshstat::What::Vertex { stride: v.stride },
                &shadow,
                v.address,
                u64::from(v.size),
            );
            *view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                StrideInBytes: v.stride,
            };
        }
        // SAFETY: an open list; the views name live host buffers.
        unsafe {
            if count > 0 {
                l.IASetVertexBuffers(0, Some(&views[..count]));
            }
        }
        if indexed {
            let v = self.index;
            let Some(shadow) = crate::shadow::find(v.address, u64::from(v.size)) else {
                return Err("index buffer outside title buffers".into());
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            #[cfg(feature = "diagnostics")]
            crate::meshstat::read(
                crate::meshstat::What::Index { format: v.format },
                &shadow,
                v.address,
                u64::from(v.size),
            );
            let view = D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                Format: DXGI_FORMAT(v.format as i32),
            };
            // SAFETY: an open list; the view names a live host buffer.
            unsafe { l.IASetIndexBuffer(Some(&view)) };
        }
        const RENDER_TARGET: u32 = 0x4;
        const DEPTH_WRITE: u32 = 0x10;
        for &t in &self.targets {
            crate::resource::decode::wrote(list, t as This);
            crate::list::ensure_state(list, t as This, RENDER_TARGET);
        }
        if self.depth != 0 {
            // The draw's pipeline is set: no pass of ours before its depth.
            list.twin_target();
            list.twin_use(self.depth as This, false, true);
            crate::list::ensure_state(list, self.depth as This, DEPTH_WRITE);
        }
        Ok(())
    }

    /// A descriptor a shader reads through a table: its host slot made if
    /// it changed, constant data copied, textures made readable.
    /// What draws and dispatches share: heaps, our root signature, the
    /// pipeline, and per stage the user data, root views and the
    /// descriptors its tables hold.
    /// False if the draw or dispatch must not run: a table it reads could
    /// not be made.
    /// The heaps the host list is to have: the title's, with XTP's own in
    /// place of each the title binds none of (or one without a host heap).
    /// Pipelines' root signatures index both heaps directly, and D3D12 lets
    /// such a root signature be set only once both are bound (debug layer
    /// message 1419); root SRVs, bound as tables, also need a view heap.
    fn wanted_heaps(&self) -> [usize; 2] {
        let mut wanted = self.heaps;
        let hosted = |h: usize| h != 0 && crate::descriptor::heap(h as This).host().is_some();
        if !hosted(wanted[0]) {
            wanted[0] = crate::descriptor::OWN_VIEW_HEAP;
        }
        if !hosted(wanted[1]) {
            wanted[1] = crate::descriptor::OWN_SAMPLER_HEAP;
        }
        wanted
    }

    /// Sets `wanted` (see [`State::wanted_heaps`]) on the host list if it
    /// has other heaps.
    fn set_host_heaps(&mut self, list: &HostList, wanted: [usize; 2]) {
        let heaps = wanted.map(|h| host_heap(h).map(|h| &h.heap));
        self.bind_host_heaps(list, wanted, heaps);
    }

    /// Sets `heaps`, the host heaps behind keys `wanted`, on the host list
    /// unless it has them, compared by host object as pipelines are
    /// ([`State::bind_pipeline`]). A title heap released while the list
    /// records and another made at its address keep the key: the list
    /// keeps the old host heap alive, so the new one is another object.
    /// Before, the key alone was compared: the host list kept the old heap
    /// while tables were made in the new one. What the recording keeps by
    /// that key goes with the old heap ([`State::forget_heap`]).
    fn bind_host_heaps(
        &mut self,
        list: &HostList,
        wanted: [usize; 2],
        heaps: [Option<&ID3D12DescriptorHeap>; 2],
    ) {
        let mut replaced = false;
        for (key, heap) in wanted.into_iter().zip(heaps) {
            if key == 0 {
                continue;
            }
            let object = heap.map_or(0, |h| h.as_raw() as usize);
            match self.heap_objects.iter().position(|&(k, _)| k == key) {
                None => self.heap_objects.push((key, object)),
                Some(i) if self.heap_objects[i].1 == object => {}
                Some(i) => {
                    self.heap_objects[i].1 = object;
                    self.forget_heap(key);
                    replaced = true;
                }
            }
        }
        if self.host_heaps == wanted && !replaced {
            return;
        }
        let heaps: Vec<Option<ID3D12DescriptorHeap>> = wanted
            .into_iter()
            .zip(heaps)
            .filter_map(|(key, heap)| {
                if key != 0 && heap.is_none() {
                    // A root signature that indexes the heaps directly then
                    // meets message 1419 again.
                    once!(key, "no host descriptor heap to bind in a heap slot");
                }
                heap
            })
            .map(|h| Some(h.clone()))
            .collect();
        heaps.iter().flatten().for_each(|h| list.keep(h));
        // SAFETY: an open list and shader-visible heaps.
        unsafe { list.list.SetDescriptorHeaps(&heaps) };
        self.host_heaps = wanted;
        self.host_roots = [0; 2];
    }

    /// Title heap `key` has another host heap than this recording used
    /// (see [`State::bind_host_heaps`]): what the recording keeps by the
    /// key goes, slots of the old heap and its passes. Fresh tables all go
    /// (they are kept by table, not heap): made again in the new heap.
    /// Stride holds stay under no key until the recording goes (the old
    /// heap's views may still be read), and the new heap gets its own.
    fn forget_heap(&mut self, key: usize) {
        self.fresh_tables.clear();
        self.root_srv_slots.retain(|k, _| k.0 != key);
        self.root_uav_slots.retain(|k, _| k.0 != key);
        self.picked.retain(|&h| h != key);
        // The work the list runs as it closes leaves the old heap alone.
        for (heap, used) in &self.picked_uses {
            if *heap == key {
                used.lock().unwrap_or_else(|e| e.into_inner()).forgotten = true;
            }
        }
        self.picked_uses.retain(|(h, _)| *h != key);
        self.heap_textured.retain(|&h| h != key);
        for (heap, _) in &mut self.own.strides {
            if *heap == key {
                *heap = 0;
            }
        }
    }

    fn compute_root(&mut self, list: &HostList, param: u32, value: ComputeRoot) {
        if let Some(pending) = &mut self.pending_compute {
            pending.push((param, value));
        } else {
            value.set(list, param);
        }
    }

    fn bind_pipeline(&mut self, list: &HostList, p: &HostPipeline, compute: bool) {
        let l = &list.list;
        // Heaps first: a root signature with directly indexed heaps must be
        // set after them.
        let wanted = self.wanted_heaps();
        self.set_host_heaps(list, wanted);
        let root = p.root.as_raw() as usize;
        if (!compute || self.pending_compute.is_none())
            && self.host_roots[usize::from(compute)] != root
        {
            // SAFETY: an open list and a live root signature.
            unsafe {
                if compute {
                    l.SetComputeRootSignature(&p.root);
                } else {
                    l.SetGraphicsRootSignature(&p.root);
                }
            }
            self.host_roots[usize::from(compute)] = root;
        }
        // By host object: a title pipeline released and another made at its
        // address is another host pipeline (the list keeps the old one).
        if self.pending_compute.is_none() && self.host_pipeline != p.pso.as_raw() as usize {
            // SAFETY: an open list and a live pipeline.
            unsafe { l.SetPipelineState(&p.pso) };
            self.host_pipeline = p.pso.as_raw() as usize;
            list.used_pipeline(&p.name);
            // The title may release the pipeline before it submits the list.
            list.keep(&p.pso);
            list.keep(&p.root);
        }
    }

    /// Binds `p` and what its stages read; why not if the draw or dispatch
    /// must not run.
    fn bind(
        &mut self,
        list: &HostList,
        p: &HostPipeline,
        compute: bool,
        limits: [u32; 3],
    ) -> Result<(), &'static str> {
        self.tlas_at_submission = None;
        #[cfg(feature = "diagnostics")]
        list.twin_reader(&p.name);
        // Depth planes whose twins shaders of the list wrote are drawn
        // before anything of this draw or dispatch is set
        // (`HostList::planes_now`).
        if list.planes_now() {
            self.after_our_pass(list);
        }
        list.begin_shader_outputs();
        let result =
            self.with_bind_descriptors(|state| state.bind_stages(list, p, compute, limits));
        list.end_shader_outputs(result.is_ok());
        result
    }

    fn with_bind_descriptors(
        &mut self,
        bind: impl FnOnce(&mut Self) -> Result<(), &'static str>,
    ) -> Result<(), &'static str> {
        let held = self.recording.checkpoint();
        self.new_descriptors.clear();
        let result = bind(self);
        if result.is_err() {
            for entry in self.new_descriptors.drain(..) {
                match entry {
                    NewDescriptor::Table(key) => {
                        self.fresh_tables.remove(&key);
                    }
                    NewDescriptor::RootSrv(key) => {
                        self.root_srv_slots.remove(&key);
                    }
                    NewDescriptor::RootUav(key) => {
                        self.root_uav_slots.remove(&key);
                    }
                };
            }
            self.recording.reject(held);
        }
        result
    }

    fn bind_stages(
        &mut self,
        list: &HostList,
        p: &HostPipeline,
        compute: bool,
        limits: [u32; 3],
    ) -> Result<(), &'static str> {
        let l = &list.list;
        use xtp_shader::emit::{
            INDEXED_ADDRESS_SRV, PACKED_ADDRESS_SRV, SCALAR_ADDRESS_SRV, SELF_SIZED_RAW_SRV,
            SELF_SIZED_VIEW_BYTES, VECTOR_ADDRESS_SRV,
        };
        list.decode_uses
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .begin_bind();
        // A trace's empty TLAS, its fallback where no title root view sets
        // it: built (when it must be) before the pipeline's state is bound.
        let empty_tlas = p
            .stages
            .iter()
            .any(|s| {
                s.root_srvs
                    .iter()
                    .any(|r| r.0 == xtp_shader::emit::TLAS_REGISTER)
            })
            .then(|| crate::rt::empty_tlas_in(list))
            .flatten();
        self.bind_pipeline(list, p, compute);
        let picked = p
            .stages
            .iter()
            .fold(0, |bits, s| bits | s.translated().host.run_time_buffers);
        if picked & xtp_shader::emit::RUN_TIME_WRITES != 0 {
            list.unknown_buffer_write();
        }
        if picked != 0 && crate::device::run_time_views_on() {
            self.picking_writes = picked & xtp_shader::emit::RUN_TIME_WRITES != 0;
            self.picked_buffers(list, &p.name);
        }
        // Textures shaders may pick from the view heap by index: a pass
        // once a frame ([`State::heap_textures`]).
        self.heap_textures(list);
        for stage in &p.stages {
            let t = stage.translated();
            #[cfg(feature = "diagnostics")]
            crate::frameprobe::begin(&stage.shader.hash);
            let made = self.tables(list, &t.used, compute)?;
            // Decoding binds a compute PSO and root signature. Restore the descriptor
            // tables before setting the compute stage's root arguments.
            if list.decode_pipeline.swap(false, Ordering::Relaxed) {
                self.host_roots[1] = 0;
                self.host_pipeline = 0;
                self.bind_pipeline(list, p, compute);
            }
            let mut biases = Vec::new();
            for &(slot, param) in &stage.root_uavs {
                let Some(&Root::View(address)) = self.roots_ref(compute).get(usize::from(slot))
                else {
                    return Err("a root UAV has no bound address");
                };
                let (handle, bias) = self.root_uav(list, address)?;
                biases.push((slot, bias));
                // SAFETY: le tas actif conserve cette vue pendant la soumission.
                unsafe {
                    if compute {
                        self.compute_root(list, param, ComputeRoot::Table(handle));
                    } else {
                        l.SetGraphicsRootDescriptorTable(param, handle);
                    }
                }
            }
            // XTP_DIAG_DIR_LIGHT: a directional light colour of ours written
            // into the light record the deferred lighting reads, before the
            // op (`frameprobe::dir_light`); before the probe, which then
            // shows it.
            #[cfg(feature = "diagnostics")]
            if crate::frameprobe::dir_light_on() {
                let tables = self.probe_tables(&t.used, &made, compute);
                crate::frameprobe::dir_light(list, &tables);
            }
            // XTP_FRAME_PROBE: what the stage reads through its tables, and
            // the host copies' bytes just before the op (`frameprobe`).
            #[cfg(feature = "diagnostics")]
            if crate::frameprobe::wanted(&stage.shader.hash) {
                let tables = self.probe_tables(&t.used, &made, compute);
                crate::frameprobe::stage(list, &stage.shader.hash, tables);
            }
            if let Some((param, dwords)) = stage.user_data {
                let mut values = vec![0u32; dwords as usize];
                for u in &t.host.user_data {
                    let fresh = made
                        .iter()
                        .find(|m| m.0 == u.slot && u.root == RootType::Table);
                    if u.root == xtp_shader::emit::DISPATCH_THREADS {
                        values[u.index as usize] = *limits
                            .get(u.dword as usize)
                            .ok_or("invalid DispatchX axis")?;
                        continue;
                    }
                    if u.root == xtp_shader::emit::ROOT_UAV_BIAS {
                        values[u.index as usize] = biases
                            .iter()
                            .find(|b| b.0 == u.slot)
                            .ok_or("missing root UAV bias")?
                            .1;
                        continue;
                    }
                    if matches!(u.root, SELF_SIZED_RAW_SRV | SELF_SIZED_VIEW_BYTES) {
                        let table = match self.roots_ref(compute).get(usize::from(u.slot)) {
                            Some(&Root::Table(table)) => Some(table),
                            _ => None,
                        };
                        let (slot_and_bias, bytes) =
                            self.self_sized_raw_srv(list, table, u.dword)?;
                        values[u.index as usize] = if u.root == SELF_SIZED_RAW_SRV {
                            slot_and_bias
                        } else {
                            bytes
                        };
                        continue;
                    }
                    if matches!(
                        u.root,
                        INDEXED_ADDRESS_SRV
                            | PACKED_ADDRESS_SRV
                            | SCALAR_ADDRESS_SRV
                            | VECTOR_ADDRESS_SRV
                    ) {
                        let words = match self.roots_ref(compute).get(usize::from(u.slot)) {
                            Some(Root::Constants(words)) => words.as_slice(),
                            _ => &[],
                        };
                        let dword = u.dword as usize;
                        let address = words.get(dword).and_then(|&lo| match u.root {
                            PACKED_ADDRESS_SRV => Some(u64::from(lo) << 8),
                            _ => words.get(dword + 1).map(|&hi| {
                                (u64::from(hi) << 32)
                                    | u64::from(if u.root == SCALAR_ADDRESS_SRV {
                                        lo & !3
                                    } else {
                                        lo
                                    })
                            }),
                        });
                        values[u.index as usize] = if u.root == INDEXED_ADDRESS_SRV {
                            self.indexed_address_srv(list, address)?
                        } else {
                            self.root_srv_slot(list, address.filter(|&a| a != 0))?.1
                        };
                        continue;
                    }
                    values[u.index as usize] = match fresh {
                        Some(&(_, offset)) => offset,
                        None => self.user_value(compute, u.slot, u.root, u.dword),
                    };
                }
                // SAFETY: an open list; `values` covers the parameter.
                unsafe {
                    if compute {
                        self.compute_root(list, param, ComputeRoot::Constants(values));
                    } else {
                        l.SetGraphicsRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    }
                }
            }
            for (views, srv) in [(&stage.root_cbvs, false), (&stage.root_srvs, true)] {
                for &(slot, param) in views {
                    // Not a title root view: the host TLAS of a spliced walk.
                    if srv && slot == xtp_shader::emit::TLAS_REGISTER {
                        self.bind_tlas(list, stage, param, compute)?;
                        continue;
                    }
                    let tabled = srv && crate::pipeline::host::root_srv_in_table(slot);
                    let Some(&Root::View(address)) = self.roots_ref(compute).get(usize::from(slot))
                    else {
                        once!(slot, "a shader reads a root view the title did not set");
                        // An unset table is undefined: a null view instead.
                        if tabled {
                            self.root_srv_table(list, None, param, compute)?;
                        }
                        // The RayQuery TLAS, a root SRV no title root view
                        // sets: an empty TLAS, where every ray misses, unless
                        // the trace binds the host's (a null root SRV removed
                        // the WARP device).
                        if srv && slot == xtp_shader::emit::TLAS_REGISTER {
                            let gpu = empty_tlas.ok_or("no empty TLAS for a trace")?;
                            if compute {
                                self.compute_root(list, param, ComputeRoot::Srv(gpu));
                            } else {
                                // SAFETY: an open list; the list keeps the TLAS.
                                unsafe { l.SetGraphicsRootShaderResourceView(param, gpu) };
                            }
                        }
                        continue;
                    };
                    #[cfg(feature = "diagnostics")]
                    crate::meshstat::root_view(address);
                    if tabled {
                        self.root_srv_table(list, Some(address), param, compute)?;
                        continue;
                    }
                    let gpu = if srv {
                        root_srv_direct(list, address)
                    } else {
                        Some(self.root_cbv(list, address)?)
                    };
                    let Some(gpu) = gpu else {
                        continue;
                    };
                    // SAFETY: an open list; the address is in live memory.
                    unsafe {
                        match (compute, srv) {
                            (false, false) => l.SetGraphicsRootConstantBufferView(param, gpu),
                            (false, true) => l.SetGraphicsRootShaderResourceView(param, gpu),
                            (true, false) => self.compute_root(list, param, ComputeRoot::Cbv(gpu)),
                            (true, true) => self.compute_root(list, param, ComputeRoot::Srv(gpu)),
                        }
                    }
                }
            }
            if let Some(param) = stage.stride_table {
                let gpu = self.stride_table(list)?;
                // SAFETY: an open list; the table lives with its heap, which
                // the list keeps, and the list keeps the table too.
                unsafe {
                    if compute {
                        self.compute_root(list, param, ComputeRoot::Srv(gpu));
                    } else {
                        l.SetGraphicsRootShaderResourceView(param, gpu);
                    }
                }
            }
        }
        // XTP_SHADER_LOOP_COUNT: the pipeline's dword of the loop-limit
        // counters, which its shaders add their stopped lanes to.
        if let Some((param, slot)) = p.loop_counter {
            let gpu = crate::loopstat::address(slot).ok_or("no loop-limit counter buffer")?;
            // SAFETY: an open list; the counters' buffer lives for good.
            unsafe {
                if compute {
                    self.compute_root(list, param, ComputeRoot::Uav(gpu));
                } else {
                    l.SetGraphicsRootUnorderedAccessView(param, gpu);
                }
            }
        }
        crate::resource::decode::bound_writes(list);
        list.twins_written();
        Ok(())
    }

    /// The GPU address of the runtime stride table of the view heap set on
    /// the host list, for a shader that reads one (VIctory #193 and #704):
    /// the list keeps the table, and its recording holds the raw views the
    /// table names ([`crate::descriptor::host::StrideHold`]) until its
    /// submissions have run. Why not if that heap has no table (no view
    /// heap of the title's, or `XTP_STRIDE_TABLE` off): the draw or
    /// dispatch is dropped, as VIctory #193's head (`f99b792`) does, where
    /// #704 still logged and ran with the table unbound.
    fn stride_table(&mut self, list: &HostList) -> Result<u64, &'static str> {
        const NONE: &str = "a shader reads the runtime stride table its view heap has not";
        let heap = self.host_heaps[0];
        let host = host_heap(heap).ok_or(NONE)?;
        let (gpu, table) = host.stride_table().ok_or(NONE)?;
        if !self.own.strides.iter().any(|&(h, _)| h == heap) {
            let hold = host.stride_hold().ok_or(NONE)?;
            // The hold goes with the recording (`State::recording`).
            self.recording.lease(list);
            self.own.strides.push((heap, hold));
            list.keep(table);
        }
        Ok(gpu)
    }

    /// For shaders that pick buffer slots of the view heap when they run
    /// (`name`: the pipeline's): every buffer view the title wrote in the
    /// heap set on the host list made, and the title memory they read
    /// brought up to date for the list's submission
    /// ([`crate::descriptor::host::HostHeap::picked_spans`]).
    /// Ported from Marvisak/VIctory#1544 (head 78fd990) by n-romanovich, and
    /// once a frame from fxceless's VIctory fork's heap_buffers (578d459);
    /// see `picked_spans` for what changed.
    /// shortcut: what shaders write through such views stays in the host
    /// copies (as for every directly indexed view): declaring every
    /// unordered view of the heap written, once a recording, copied tens of
    /// thousands of ranges back after each submission, and the first world
    /// frame never ended (worldlight-16). Views the shader's tables name
    /// are declared written as before (`readable`). Buffers only: textures
    /// picked at run time are another matter (`texture.rs`).
    fn picked_buffers(&mut self, list: &HostList, name: &str) {
        let heap = self.host_heaps[0];
        // Since the Codex review of 54ab53d: at every draw or dispatch, and
        // in every list. The views the title wrote since are made, and each
        // list refreshes what they read for its own submission (whichever
        // the title submits first), declaring written what shaders that
        // write through picked slots may write. Since that of 244cf78:
        // only what changed since the list's last such draw, and once more
        // as the list closes, for the views the title wrote after it.
        if heap == 0 || heap == crate::descriptor::OWN_VIEW_HEAP {
            return;
        }
        let title = crate::descriptor::heap(heap as This);
        let Some(host) = title.host() else {
            return;
        };
        let used = match self.picked_uses.iter().find(|(h, _)| *h == heap) {
            Some((_, used)) => used.clone(),
            None => {
                if !self.picked.contains(&heap) {
                    self.picked.push(heap);
                    counted!(
                        info,
                        pipeline = name,
                        "a draw or dispatch reads buffer views of slots picked at run time"
                    );
                }
                let used = PickedUse::default();
                self.picked_uses.push((heap, used.clone()));
                let (start, closing) = (title.start, used.clone());
                let made = title.hold();
                list.at_close(move |list| {
                    let writes = {
                        let used = closing.lock().unwrap_or_else(|e| e.into_inner());
                        if used.forgotten {
                            return;
                        }
                        used.writing
                    };
                    // The same title heap, still there (as while the
                    // recording used it), with the same host heap.
                    // Held for the whole sweep: its release waits, and a heap
                    // made since at its address is another (Codex review of
                    // a090f22, P1). Then again at each of the list's
                    // submissions, for the views the title writes between
                    // its close and its submission (point 6).
                    let sweep = move |list: &HostList| {
                        made.with(|title| {
                            if let Some(host) = title.host() {
                                picked_refresh(list, host, start, &closing, writes);
                            }
                        });
                    };
                    sweep(list);
                    list.at_submit(sweep);
                });
                used
            }
        };
        picked_refresh(list, host, title.start, &used, self.picking_writes);
    }

    /// For shaders that pick textures from the view heap set on the host
    /// list by an index they compute: no used-resource entry names such a
    /// slot, so no draw's table made it (a placed texture's slot stayed
    /// null) nor uploaded its texture (the host copy kept what it was made
    /// with: "98 % of the drawn pixels had no colour texture" in the fork's
    /// first 3D frames). Once a frame, by the first list that binds a
    /// pipeline with the title's view heap set
    /// ([`crate::descriptor::host::HostHeap::texture_pass`]): the slots the
    /// title wrote a texture view into since are made and their textures
    /// uploaded as a table read does (`readable`), the newest writes first,
    /// within a budget of bytes and of time; then the textures uploaded
    /// before are checked for changes in turns, within a time, against the
    /// write watch or their guards (`texture::memory_change`, no hash), and
    /// the changed ones uploaded again (those whose memory neither covers
    /// within a budget of bytes, for the check at submission to hash).
    /// Room is checked per texture, with its bytes: a host texture made
    /// within a share of the video memory budget, a staging buffer within
    /// the staging limit and the pass's bytes (a texture that has one needs
    /// none). A texture is served once its upload is recorded, not before.
    /// What the pass records runs before the draw that asked and every list
    /// submitted after it; lists submitted before see the frame before's.
    /// Its uploads are decided at submission, as a table read's are: one
    /// submitted after a GPU write into the texture runs no copy
    /// (`texture::gpu_write`).
    /// Returns what the pass did, if this list made one.
    /// Ported from fxceless's VIctory fork's heap_textures (578d459); see
    /// `descriptor::host::TextureQueue` for what changed.
    /// shortcut: the translator does not say which shaders pick textures
    /// when they run (`HostShader::run_time_buffers` is about buffers), so
    /// the pass runs for the first pipeline of the frame whatever it reads;
    /// with such a bit it would run only for those that do.
    /// `XTP_HEAP_TEXTURES` sets the bytes a pass stages (0: no pass).
    fn heap_textures(&mut self, list: &HostList) -> Option<crate::descriptor::host::PassCounts> {
        use crate::descriptor::host::{
            PASS_CHECKING, PASS_HASHING, PASS_MAKING, PassCounts, stage_for_pass, video_room,
        };
        use crate::texture::{Change, Staged};
        let heap = self.host_heaps[0];
        if heap == 0
            || heap == crate::descriptor::OWN_VIEW_HEAP
            || self.heap_textured.contains(&heap)
        {
            return None;
        }
        self.heap_textured.push(heap);
        let budget = crate::descriptor::host::heap_textures_budget()?;
        let title = crate::descriptor::heap(heap as This);
        let host = title.host()?;
        let pass = host.texture_pass()?;
        let started = std::time::Instant::now();
        let mut counts = PassCounts::default();
        // The slots this pass uploads, first and again (`XTP_PASS_DUMP`).
        #[cfg(feature = "diagnostics")]
        let mut uploads: (Vec<u32>, Vec<u32>) = Default::default();
        // What a table entry of a texture a shader samples says.
        let sampled = UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 0,
        };
        let handle = |index: u32| title.start + u64::from(index) * 32;
        // A placed view's title memory and its bytes.
        let placed_memory = |resource: &[u32; 14], desc: &[u32; 16]| {
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let len = crate::texture::placed_len(resource).unwrap_or(1);
            (address, len)
        };
        // The video memory host textures may still take this pass: past a
        // share of its budget, the video memory manager would evict at
        // every submission. Each texture made counts its bytes; staging
        // buffers count theirs in `spent` (`stage_for_pass`).
        let video = crate::host::host().and_then(|h| crate::dxgi::video_memory(&h.device));
        let mut video_left = video_room(video);
        let mut spent = 0;
        for &(index, write) in &pass.due {
            if spent >= budget || pass_time_spent(started, PASS_MAKING) {
                break;
            }
            counts.looked += 1;
            let at = handle(index);
            let Some(written) = crate::descriptor::view(at) else {
                host.texture_slot_dropped(index, Some(write));
                continue;
            };
            if let View::Placed(false, resource, desc) = &written {
                let (address, len) = placed_memory(resource, desc);
                // None of its memory mapped (not streamed in yet, or streamed
                // out: the heap keeps many such views): no host texture,
                // looked at again later.
                if !crate::texture::any_committed(address as *const u8, len) {
                    counts.away += 1;
                    host.texture_slot_later(index, write, &pass);
                    continue;
                }
                // Its host texture is made below unless a draw made it: it
                // waits if the video memory left cannot take it.
                if len as u64 > video_left {
                    crate::texture::note_left_for_room(len as u64);
                    counts.left += 1;
                    counts.full = 1;
                    once!(
                        usage_mib = video.map_or(0, |v| v.1 >> 20),
                        budget_mib = video.map_or(0, |v| v.0 >> 20),
                        "textures shaders pick by index: video memory near its budget, the rest wait"
                    );
                    continue;
                }
                video_left -= len as u64;
            }
            let Some(view) = host.mirror(index, at, true) else {
                host.texture_slot_dropped(index, Some(write));
                continue;
            };
            // A placed texture whose host copy is not made or resident yet
            // stays a placed view; a texture object's has no host copy.
            let made = match &view {
                View::Shader(r, _) => {
                    crate::resource::host_resource(*r as This).map(|t| (*r as This, t))
                }
                _ => None,
            };
            let Some((resource, texture)) = made else {
                counts.away += 1;
                host.texture_slot_later(index, write, &pass);
                continue;
            };
            if !crate::texture::title_written(texture) {
                // A target, or a texture written through unordered access:
                // the host's own contents.
                host.texture_slot_dropped(index, Some(write));
                continue;
            }
            // Its staging buffer first: no upload is recorded without one,
            // and a slot is served only once its upload is.
            match stage_for_pass(resource, texture, budget, &mut spent, &mut counts) {
                Staged::Ready(_) => {}
                // Past the room left (this pass's bytes, the staging limit).
                Staged::NoRoom(_) => {
                    counts.left += 1;
                    continue;
                }
                // No upload memory now: looked at again later.
                Staged::NotNow => {
                    counts.away += 1;
                    host.texture_slot_later(index, write, &pass);
                    continue;
                }
                // Nothing of it to upload, or the host GPU's own contents.
                Staged::Never | Staged::Host => {
                    host.texture_slot_dropped(index, Some(write));
                    continue;
                }
            }
            readable(list, &sampled, &view);
            crate::descriptor::host::note_heap_texture(index, at, texture, "uploaded first");
            #[cfg(feature = "diagnostics")]
            uploads.0.push(index);
            counts.uploaded += 1;
            host.texture_slot_done(index, write);
        }
        counts.left += pass.due.len() as u64 - counts.looked;
        // The textures uploaded before, in turns: a check is a question to
        // the write watch or the guards, the changed ones are uploaded.
        let checking = std::time::Instant::now();
        let mut last = None;
        let mut seen = std::collections::HashSet::new();
        let mut hashing = 0;
        // The staging buffers made again for served textures that changed:
        // a share of the budget the first uploads above did not spend.
        let mut refreshes = crate::texture::Refreshes::of(budget);
        for &index in &pass.served {
            if pass_time_spent(checking, PASS_CHECKING) {
                break;
            }
            let before = last;
            last = Some(index);
            counts.checked += 1;
            let at = handle(index);
            let Some(written) = crate::descriptor::view(at) else {
                host.texture_slot_dropped(index, None);
                continue;
            };
            // A texture the title placed itself is looked at through the
            // resource its slot was made from, while that one lives (its
            // staging buffer goes with it): asking for the view again could
            // make a host texture, which only an upload does here, within
            // the video memory left. One that went (its memory remapped) gets
            // its first upload here, within the same budgets.
            let placed = match &written {
                View::Placed(false, resource, desc) if resource[0] != 1 => {
                    let (address, len) = placed_memory(resource, desc);
                    let alive = host
                        .placed_resource(index)
                        .filter(|&made| crate::texture::staged_at(made, address));
                    Some((alive, len as u64))
                }
                _ => None,
            };
            let (looked, mirrored) = match placed {
                Some((alive, _)) => (alive, None),
                None => match host.mirror(index, at, true) {
                    Some(view @ View::Shader(r, _)) => (Some(r), Some(view)),
                    // Not a texture view any more: the title's write of the
                    // slot queued it again.
                    Some(_) => continue,
                    None => {
                        host.texture_slot_dropped(index, None);
                        continue;
                    }
                },
            };
            let change = match looked {
                Some(r) => {
                    if !seen.insert(r) {
                        continue;
                    }
                    crate::texture::memory_change(r as This)
                }
                // The placed texture went: what is there now is due.
                None => Change::Yes,
            };
            match change {
                Change::No => continue,
                Change::Yes => {}
                // Only the check at submission's hash tells, within a budget
                // a pass: the rest from this one at the next pass.
                Change::Unknown(bytes) => {
                    if hashing > 0 && hashing + bytes > PASS_HASHING {
                        last = before;
                        break;
                    }
                }
            }
            // The view the upload reads: a placed texture's as a draw gets it
            // (the resource it names is held a while after it goes), which
            // makes a host texture if the slot's went.
            let view = match (mirrored, placed) {
                (Some(view), _) => view,
                (None, Some((alive, len))) => {
                    if len > video_left {
                        continue;
                    }
                    match host.mirror(index, at, true) {
                        Some(view @ View::Shader(r, _)) => {
                            if alive != Some(r) {
                                video_left -= len;
                            }
                            view
                        }
                        // Its host copy is not resident yet: next lap.
                        Some(_) => continue,
                        None => {
                            host.texture_slot_dropped(index, None);
                            continue;
                        }
                    }
                }
                (None, None) => continue,
            };
            let View::Shader(resource, _) = &view else {
                continue;
            };
            let resource = *resource as This;
            let Some(texture) = crate::resource::host_resource(resource) else {
                continue;
            };
            if !crate::texture::title_written(texture) {
                // Now a GPU-written texture's copy (#643): the host's own.
                host.texture_slot_dropped(index, None);
                continue;
            }
            // One never staged (none could be made then, or it took another's
            // place) gets its staging buffer here, within the room left.
            // One whose staging buffer was released gets it again from the
            // refreshes' share (`texture::Refreshes`): before, the first
            // uploads left it no room while the world loaded, and shaders
            // read its old texels (Codex's review of c44a716, 2).
            let staged = match refreshes.stage(resource) {
                // Past the share: the first the next pass looks at.
                Some(Staged::NoRoom(_)) => {
                    last = before;
                    break;
                }
                // Past what a pass of its own could make now: `NotNow`, left
                // for its next turn, the textures after it go on
                // (`texture::Refreshes::stage`).
                Some(staged) => staged,
                None => stage_for_pass(resource, texture, budget, &mut spent, &mut counts),
            };
            match staged {
                Staged::Ready(_) => {}
                Staged::NoRoom(_) | Staged::NotNow => continue,
                Staged::Never | Staged::Host => {
                    host.texture_slot_dropped(index, None);
                    continue;
                }
            }
            readable(list, &sampled, &view);
            if let Change::Unknown(bytes) = change {
                hashing += bytes;
                counts.hashed += 1;
            } else {
                counts.changed += 1;
                crate::descriptor::host::note_heap_texture(
                    index,
                    at,
                    texture,
                    "changed: uploaded again",
                );
                #[cfg(feature = "diagnostics")]
                uploads.1.push(index);
            }
        }
        if let Some(last) = last {
            host.textures_checked(last);
        }
        counts.bytes = spent + refreshes.made();
        counts.us = started.elapsed().as_micros() as u64;
        // In the pass dump (`XTP_PASS_DUMP`), as an op of this list: the
        // slots whose uploads run before this list's commands and after the
        // lists submitted before it, against the draws that pick them
        // (faces black in the G-buffer: Samuel's watch 38, item 490).
        #[cfg(feature = "diagnostics")]
        list.note_op(|| {
            format!(
                "texture pass of heap {:#x}, frame {}: uploaded first {:?}, uploaded again {:?}; {} due left",
                title.start,
                pass.frame,
                uploads.0,
                uploads.1,
                pass.left(&counts)
            )
        });
        host.texture_pass_done(&pass, counts);
        Some(counts)
    }

    /// Binds at root parameter `param`, a root SRV, the host TLAS a program
    /// spliced with an inline ray query reads
    /// ([`xtp_shader::emit::TLAS_REGISTER`]): the one the host built for the
    /// structure the title's view names, the view where the program's walk
    /// read it ([`State::tlas_view`]). `rt::descriptor_tlas` pins it for the
    /// recording and gives it only once its build is recorded earlier in
    /// this list or submitted. A compute dispatch whose TLAS the title
    /// builds in a list not submitted yet takes it at its submission instead
    /// ([`State::tlas_at_submission`], `rt::dispatch_descriptor_tlas`). Why
    /// not if it cannot be (the dispatch is dropped).
    fn bind_tlas(
        &mut self,
        list: &HostList,
        stage: &crate::pipeline::host::Stage,
        param: u32,
        compute: bool,
    ) -> Result<(), &'static str> {
        let handle = self.tlas_view(stage, compute).ok_or(NO_TLAS_VIEW)?;
        self.bind_tlas_view(list, handle, param, compute)
    }

    /// [`State::bind_tlas`] for the title's view at `handle`.
    fn bind_tlas_view(
        &mut self,
        list: &HostList,
        handle: u64,
        param: u32,
        compute: bool,
    ) -> Result<(), &'static str> {
        match crate::rt::descriptor_tlas_in(list, handle) {
            Some(gpu) if compute => self.compute_root(list, param, ComputeRoot::Srv(gpu)),
            // SAFETY: an open list, which keeps the structure.
            Some(gpu) => unsafe { list.list.SetGraphicsRootShaderResourceView(param, gpu) },
            None if compute => self.tlas_at_submission = Some((handle, param)),
            None => return Err(NO_HOST_TLAS),
        }
        Ok(())
    }

    /// Records a trace dispatch whose host TLAS it takes at its submission
    /// ([`State::tlas_at_submission`]): `rt::dispatch_descriptor_tlas_in`
    /// sets the TLAS's root SRV as an indirect argument then, the one
    /// published in queue order or the empty one (every ray misses). A
    /// dispatch past D3D12's group limit, which a direct one splits, is
    /// dropped.
    fn dispatch_tlas_at_submission(
        &mut self,
        list: &HostList,
        p: &HostPipeline,
        handle: u64,
        param: u32,
        groups: [u32; 3],
    ) {
        if groups.contains(&0) {
            return;
        }
        if groups.iter().any(|&n| n > 65_535) {
            self.dispatch_dropped(NO_HOST_TLAS.into());
            return;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| {
            let [x, y, z] = groups;
            self.op_text(format!("dispatch {x}x{y}x{z}, TLAS at submission"), true)
        });
        let _timed = crate::timestamps::command(list, crate::timestamps::What::Dispatch, Some(p));
        if std::mem::take(&mut self.dropped_dispatch) {
            // SAFETY: an open list; the barrier names no resource.
            unsafe { list.list.ResourceBarrier(&[crate::host::uav_barrier()]) };
        }
        if crate::rt::dispatch_descriptor_tlas_in(list, handle, &p.root, param, groups) {
            DISPATCHED.fetch_add(1, Ordering::Relaxed);
        } else {
            self.dispatch_dropped(NO_HOST_TLAS.into());
        }
    }

    /// The title's view of the structure a spliced program's walk reads:
    /// the descriptor at its site's offset in the table bound at the root
    /// parameter whose address the site's user SGPR holds
    /// (`Shader::tlas_descriptor`), in a heap the list has bound.
    fn tlas_view(&self, stage: &crate::pipeline::host::Stage, compute: bool) -> Option<u64> {
        self.tlas_view_at(stage.shader.tlas_descriptor()?, compute)
    }

    /// [`State::tlas_view`] where `d` says.
    fn tlas_view_at(&self, d: xtp_shader::emit::TlasDescriptor, compute: bool) -> Option<u64> {
        let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(d.slot)) else {
            return None;
        };
        let handle = table.checked_add(u64::from(d.offset))?;
        self.heap_of(handle)?;
        Some(handle)
    }

    /// Binds a root SRV as a one-view table (see
    /// [`crate::pipeline::host::bounded_root_srvs`] and
    /// [`State::root_srv_slot`]). Why not if it cannot be (the draw is
    /// dropped).
    fn root_srv_table(
        &mut self,
        list: &HostList,
        address: Option<u64>,
        param: u32,
        compute: bool,
    ) -> Result<(), &'static str> {
        let (host, slot) = self.root_srv_slot(list, address)?;
        let handle = host.gpu_handle(slot);
        // SAFETY: an open list; the slot is in the view heap set on it.
        unsafe {
            if compute {
                self.compute_root(list, param, ComputeRoot::Table(handle));
            } else {
                list.list.SetGraphicsRootDescriptorTable(param, handle);
            }
        }
        Ok(())
    }

    /// Ported from Marvisak/VIctory#1077 (head 873548f) by DuarttBR.
    /// Le décalage est celui de la vue liée, y compris pour un compteur
    /// situé à 4, 8 ou 12 octets du début aligné.
    fn root_uav(
        &mut self,
        list: &HostList,
        address: u64,
    ) -> Result<(D3D12_GPU_DESCRIPTOR_HANDLE, u32), &'static str> {
        let heap = self.host_heaps[0];
        let host = host_heap(heap).ok_or("a root UAV without a view heap")?;
        let copy = crate::shadow::find(address, 4).ok_or("a root UAV outside a title buffer")?;
        // SAFETY: la copie possède la ressource.
        let flags = unsafe { copy.buffer.GetDesc() }.Flags;
        if !flags.contains(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) {
            return Err("a root UAV on a buffer without unordered access");
        }
        let window =
            xtp_shader::emit::RawWindow::covering(copy.offset(address), copy.left(address) & !3)
                .ok_or("a root UAV off a dword")?;
        let key = (
            heap,
            copy.buffer.as_raw() as usize,
            window.first,
            u64::from(window.dwords),
        );
        let slot = match self.root_uav_slots.get(&key) {
            Some(&slot) => slot,
            None => {
                let slot = host.dynamic(1).ok_or("no dynamic slot for a root UAV")?;
                self.recording.range(list, heap, slot, 1);
                host.raw_uav_at(slot, &copy.buffer, window);
                self.root_uav_slots.insert(key, slot);
                self.new_descriptors.push(NewDescriptor::RootUav(key));
                slot
            }
        };
        let bytes = u64::from(window.elements(0)) * 4;
        copy.refresh(list, address, bytes);
        copy.gpu_writes(list, address, bytes);
        Ok((host.gpu_handle(slot), window.bias))
    }

    /// Lie la copie qui contient la base. Le shader conserve l'index GPU et
    /// l'identité du tampon, même pour une base décalée dans la copie.
    fn raw_slice_srv(
        &mut self,
        list: &HostList,
        copy: &Arc<crate::shadow::Shadow>,
        (title, offset, len): (u64, u64, u64),
        (too_large, no_slot): (&'static str, &'static str),
    ) -> Result<u32, &'static str> {
        let heap = self.host_heaps[0];
        let host = host_heap(heap).ok_or("a raw buffer slice without a view heap")?;
        let bias = offset % 16;
        let span = len
            .checked_add(bias)
            .filter(|&n| n <= u64::from(xtp_shader::emit::RawWindow::MAX_DWORDS) * 4)
            .ok_or(too_large)?;
        let slot = host.shared_raw(copy, offset - bias, span).ok_or(no_slot)?;
        self.recording.shared(list, heap, slot);
        copy.refresh(list, title, len);
        Ok((slot << 4) | bias as u32)
    }

    fn self_sized_raw_srv(
        &mut self,
        list: &HostList,
        table: Option<u64>,
        dword: u32,
    ) -> Result<(u32, u32), &'static str> {
        if let Some((copy, title, offset, len)) = table
            .and_then(|t| crate::descriptor::view(t + u64::from(dword) * 4))
            .and_then(|v| crate::descriptor::host::srv_range(&v))
        {
            let slot_and_bias = self.raw_slice_srv(
                list,
                &copy,
                (title, offset, len),
                (
                    "a self-sized buffer view too large for a raw view",
                    "no shared slot for a self-sized buffer load",
                ),
            )?;
            return Ok((slot_and_bias, u32::try_from(len).unwrap_or(u32::MAX)));
        }
        let heap = self.host_heaps[0];
        let host = host_heap(heap).ok_or("a self-sized buffer load without a view heap")?;
        let slot = host
            .null_raw()
            .ok_or("no dynamic slot for a null self-sized buffer load")?;
        Ok((slot << 4, 0))
    }

    fn indexed_address_srv(
        &mut self,
        list: &HostList,
        address: Option<u64>,
    ) -> Result<u32, &'static str> {
        let (a, copy) = address
            .filter(|&a| a != 0)
            .and_then(|a| crate::shadow::find(a, 1).map(|s| (a, s)))
            .ok_or("an indexed BVH buffer without a host copy")?;
        let offset = copy.offset(a);
        let len = copy.left(a);
        self.raw_slice_srv(
            list,
            &copy,
            (a, offset, len),
            (
                "an indexed BVH buffer too large for a raw view",
                "no shared slot for an indexed BVH buffer",
            ),
        )
    }

    /// The slot, in the view heap set on the host list (the title's, or
    /// XTP's own: [`State::wanted_heaps`]), of a root SRV's one-view table:
    /// a raw view from `address` to the end of its buffer's host copy, the
    /// range brought up to date; or a null view (no address, or no host
    /// copy).
    ///
    /// A view of the host copy in place is shared with every list that
    /// views the same range, made once. A raw view starts on 16 bytes and
    /// the title's root SRVs need only 4 (vertex pools of 12- or 20-byte
    /// vertices): such an SRV gets a copy of its range of the list's own,
    /// made on the GPU at this place in the list (so it holds what earlier
    /// work wrote), and a view of that, in a slot the list holds.
    /// Why not if no slot could be had.
    fn root_srv_slot(
        &mut self,
        list: &HostList,
        address: Option<u64>,
    ) -> Result<(&'static crate::descriptor::host::HostHeap, u32), &'static str> {
        let heap = self.host_heaps[0];
        let Some(host) = host_heap(heap) else {
            once!("a root SRV without a view heap (none of XTP's own either): the draw is dropped");
            return Err("a root SRV without a view heap");
        };
        let Some((a, s)) = address.and_then(|a| crate::shadow::find(a, 1).map(|s| (a, s))) else {
            let slot = host
                .null_raw()
                .ok_or("no dynamic slot for a null root SRV")?;
            return Ok((host, slot));
        };
        // By host copy too: a buffer the title makes at the address of one
        // it released while recording the list has a copy of its own.
        let copy = s.buffer.as_raw() as usize;
        let offset = s.offset(a);
        let in_place = offset.is_multiple_of(16);
        let limit = if in_place {
            crate::descriptor::host::MAX_RAW_VIEW_BYTES
        } else {
            MISALIGNED_SPAN
        };
        let bytes = s.left(a).min(limit);
        s.refresh(list, a, bytes);
        let serial = if in_place {
            0
        } else {
            let (title, gpu) =
                list.input_writes(Some(std::sync::Arc::as_ptr(&s) as usize), a, bytes);
            title.max(gpu) + 1
        };
        let key = (heap, a, copy, serial);
        if let Some(&slot) = self.root_srv_slots.get(&key) {
            return Ok((host, slot));
        }
        note_root_srv(a, &s);
        note_root_srv_extent(a, &s, bytes);
        let slot = if in_place {
            let slot = host.shared_raw(&s, offset, bytes);
            if let Some(slot) = slot {
                self.recording.shared(list, heap, slot);
            }
            slot
        } else {
            Some(self.misaligned_srv(list, host, heap, &s, a)?)
        };
        let Some(slot) = slot else {
            counted!(
                warn,
                "a draw or dispatch dropped: no dynamic slot for a root SRV"
            );
            return Err("no dynamic slot for a root SRV");
        };
        self.root_srv_slots.insert(key, slot);
        self.new_descriptors.push(NewDescriptor::RootSrv(key));
        Ok((host, slot))
    }

    /// A slot the list holds with a raw view of a copy of the list's own of
    /// the root SRV at `a` (not on 16 bytes of host copy `s`): its first
    /// [`MISALIGNED_SPAN`] bytes, copied on the GPU here. The bind fails
    /// if the copy or descriptor cannot be allocated.
    fn misaligned_srv(
        &mut self,
        list: &HostList,
        host: &crate::descriptor::host::HostHeap,
        heap: usize,
        s: &crate::shadow::Shadow,
        a: u64,
    ) -> Result<u32, &'static str> {
        let left = s.left(a);
        let len = left.min(MISALIGNED_SPAN);
        counted!(
            debug,
            address = format_args!("{a:#x}"),
            left,
            copied = len,
            "a root SRV not on 16 bytes of its host copy: copied on the GPU to view it"
        );
        if len < left {
            counted!(
                info,
                address = format_args!("{a:#x}"),
                left,
                copied = len,
                "a root SRV not on 16 bytes copied in part: reads past the copy return zeros"
            );
        }
        let slot = host
            .dynamic(1)
            .ok_or("no dynamic slot for a copied root SRV")?;
        self.recording.range(list, heap, slot, 1);
        // Whole dwords, the last one too: the copy is rounded up to 64 KiB.
        // SAFETY: a live buffer.
        let width = unsafe { s.buffer.GetDesc() }.Width;
        let offset = s.offset(a);
        let bytes = len.next_multiple_of(4).min(width - offset);
        let Some((buffer, at)) = self.own.gpu_copy(list, &s.buffer, offset, bytes, 0) else {
            counted!(
                warn,
                address = a,
                bytes,
                "bind dropped: could not copy a misaligned root SRV"
            );
            return Err("could not copy a misaligned root SRV");
        };
        host.raw_at(slot, &buffer, at, len);
        Ok(slot)
    }

    /// A root CBV must be aligned and backed by 64 KiB of list-owned or
    /// shadow memory. Misaligned inputs without a GPU writer can be filled
    /// from title memory at submission.
    fn root_cbv(&mut self, list: &HostList, address: u64) -> Result<u64, &'static str> {
        let s = crate::shadow::find(address, 1).ok_or("a root CBV without a host copy")?;
        let len = s.left(address).min(CBV_SPAN);
        let gpu = s.gpu(address);
        let aligned = gpu.is_multiple_of(256);
        // SAFETY: a live buffer.
        if aligned && s.offset(address) + CBV_SPAN <= unsafe { s.buffer.GetDesc() }.Width {
            s.refresh(list, address, len);
            return Ok(gpu);
        }
        let mut input = CbvInput::at(list, address, len, Some(&s));
        if aligned {
            // A short aligned shadow also needs padding for unbounded reads.
            input.gpu_source = Some(std::sync::Arc::as_ptr(&s) as usize);
        }
        self.fill_cbv(list, &s, address, len, input)
    }

    fn fill_cbv(
        &mut self,
        list: &HostList,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: u64,
        len: u64,
        input: CbvInput,
    ) -> Result<u64, &'static str> {
        let key = (shadow.buffer.as_raw() as usize, address);
        if let Some(copy) = self.cbv_copies.get(&key)
            && copy.input.gpu_source == input.gpu_source
            // A view of whole 256s reads no padding; one with padding must
            // find zeros there, not the longer copy's next bytes.
            && (len == copy.bytes || padding(len) == 0 && len <= copy.bytes)
            // A larger snapshot may also cover a newer write outside the
            // requested prefix. Every write after that snapshot is newer.
            && input.writes <= copy.input.writes
        {
            if input.gpu_source.is_some() {
                CBV_COPY_HITS.fetch_add(1, Ordering::Relaxed);
            }
            return Ok(copy.gpu);
        }
        let gpu = if input.gpu_source.is_some() {
            let Some((buffer, at)) = self.own.gpu_copy(
                list,
                &shadow.buffer,
                shadow.offset(address),
                len,
                padding(len),
            ) else {
                counted!(
                    warn,
                    address,
                    bytes = len,
                    "bind dropped: could not copy a CBV on the GPU"
                );
                return Err("could not copy a CBV on the GPU");
            };
            shadow.refresh(list, address, len);
            list.cbv_read(shadow, address, len);
            CBV_COPIES.fetch_add(1, Ordering::Relaxed);
            counted!(
                debug,
                address,
                bytes = len,
                "a CBV copied on the GPU into list memory"
            );
            // SAFETY: the recording holds the aligned chunk until its lists finish.
            unsafe { buffer.GetGPUVirtualAddress() + at }
        } else {
            self.own
                .upload_fill(list, address, len)
                .ok_or("could not upload a root CBV")?
        };
        self.recording.lease(list);
        self.cbv_copies.insert(
            key,
            CbvCopy {
                input,
                gpu,
                bytes: len,
            },
        );
        Ok(gpu)
    }

    /// Runs a dispatch on the host; the first that runs after dropped ones
    /// starts with a barrier ([`State::dispatch_dropped`]).
    pub(crate) fn dispatch(&mut self, list: &HostList, [x, y, z]: [u32; 3], limits: [u32; 3]) {
        if crate::pipeline::builder_held(self.pipeline as This) {
            self.dispatch_dropped("BVH builder held".into());
            return;
        }
        #[cfg(feature = "diagnostics")]
        crate::indirect::note_direct(list, self.pipeline, [x, y, z]);
        #[cfg(feature = "diagnostics")]
        let mut meshes = crate::meshstat::start(list, self.pipeline, crate::meshstat::Op::Dispatch);
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            self.dispatch_dropped(Unready::NoPipeline);
            return;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            once!(
                pipeline = format_args!("{:#x}", self.pipeline),
                "a dispatch without a host pipeline dropped"
            );
            // XTP_FRAME_PROBE: what it would read (`frameprobe::dropped`).
            #[cfg(feature = "diagnostics")]
            self.probe_dropped(list, None);
            self.rt_dropped(list, crate::rt_standin::Work::Groups([x, y, z]));
            self.dispatch_dropped(Unready::NoPipeline);
            return;
        };
        if !p.compute {
            once!("a dispatch with a graphics pipeline");
            self.dispatch_dropped("a graphics pipeline bound".into());
            return;
        }
        if let Err(why) = self.bind(list, p, true, limits) {
            if why == NO_TLAS_VIEW || why == NO_HOST_TLAS {
                self.rt_dropped(list, crate::rt_standin::Work::Groups([x, y, z]));
            }
            self.dispatch_dropped(why.into());
            return;
        }
        // A trace whose host TLAS was not there to bind: the dispatch takes
        // the one published at its submission, or the empty one (every ray
        // misses), and sets its root SRV itself.
        if let Some((handle, param)) = self.tlas_at_submission.take() {
            self.dispatch_tlas_at_submission(list, p, handle, param, [x, y, z]);
            #[cfg(feature = "diagnostics")]
            meshes.ran();
            return;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| self.op_text(format!("dispatch {x}x{y}x{z}"), true));
        #[cfg(feature = "diagnostics")]
        self.dump_at_submission(list, None);
        let _timed = crate::timestamps::command(list, crate::timestamps::What::Dispatch, Some(p));
        // SAFETY: an open list with everything set; the barrier names no
        // resource.
        unsafe {
            if std::mem::take(&mut self.dropped_dispatch) {
                list.list.ResourceBarrier(&[crate::host::uav_barrier()]);
            }
            crate::indirect::direct(list, [x, y, z], p.group_base(), false, |[x, y, z]| {
                list.list.Dispatch(x, y, z)
            });
        };
        DISPATCHED.fetch_add(1, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        meshes.ran();
    }

    /// Counts a dispatch that does not run; the next one of the list that
    /// runs, direct or indirect, starts with a global UAV barrier. From Marvisak/VIctory#1392 by
    /// kunalkhude99, for issue #1347: an RTX 4060 lost its device on a
    /// dispatch of `4c0ac499ad613809` right after three dropped ones, and
    /// an RTX 3060 Ti named the same program (the maintainers' private
    /// watch notes).
    /// The PR's reason does not hold here as such: XTP records the title's
    /// own barriers (`list::barrier`, `list::barrier_groups`) and a dropped
    /// dispatch records nothing, so no barrier the title asked for goes
    /// missing. The barrier is a guard, not a proven fix. The PR puts one
    /// after each dropped dispatch; the Series S world drops about 1,000
    /// direct dispatches a frame (most do not translate), so here one waits
    /// for the next dispatch that runs: one a run of drops.
    fn dispatch_dropped(&mut self, why: Unready) {
        self.dropped_dispatch = true;
        dropped("dispatch", self.pipeline, why);
    }

    /// A dispatch, direct or indirect, dropped for want of a host pipeline,
    /// which would have run `work`: if it runs one of the title's trace
    /// programs ([`crate::rt_standin`]), it is counted, and with the
    /// stand-in on, what its program writes through unordered access is
    /// handed over as the title's compute tables (in the bound heaps only)
    /// and root views hold it now, to be cleared to "every ray missed". A
    /// clear binds XTP's own view heap: the next draw or dispatch binds the
    /// title's.
    fn rt_dropped(&mut self, list: &HostList, work: crate::rt_standin::Work) {
        let Some(program) = crate::rt_standin::program(self.pipeline as This) else {
            return;
        };
        use crate::rt_standin::Bound;
        let bound = || {
            program
                .writes()
                .map(|u| match (u.root, self.compute.get(usize::from(u.slot))) {
                    (RootType::Table, Some(Root::Table(table))) => {
                        let handle = table + u64::from(u.table_offset_dwords) * 4;
                        self.heap_of(handle)
                            .and_then(|_| crate::descriptor::view(handle))
                            .map_or(Bound::Unset, Bound::View)
                    }
                    (RootType::Descriptor, Some(Root::View(address))) => Bound::Root(*address),
                    _ => Bound::Unset,
                })
                .collect()
        };
        if crate::rt_standin::dropped(list, &program, work, bound) {
            self.forget_host();
        }
    }

    /// What a draw or dispatch runs, for the device-removed report
    /// (diagnostics): the pipeline, then per stage the user data it gets
    /// (with the view at that heap index, if one is), the host addresses
    /// of its root views and the views its tables name.
    #[cfg(feature = "diagnostics")]
    fn op_text(&self, kind: String, compute: bool) -> String {
        use std::fmt::Write;
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return kind;
        };
        let mut text = format!("{kind} {}", p.name);
        let views = (self.heaps[0] != 0)
            .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
            .flatten();
        for stage in &p.stages {
            let t = stage.translated();
            let _ = write!(text, " | {}:", stage.shader.hash);
            for u in &t.host.user_data {
                let v = self.user_value(compute, u.slot, u.root, u.dword);
                let _ = write!(text, " u{}={v:#x}", u.index);
                let index = if u.root == RootType::Table { v / 32 } else { v };
                if let Some(slot) = views.and_then(|h| h.describe_slot(index)) {
                    let _ = write!(text, " [heap {index}: {slot}]");
                }
            }
            for &(slot, _) in stage.root_cbvs.iter().chain(&stage.root_srvs) {
                if let Some(Root::View(address)) = self.roots_ref(compute).get(usize::from(slot)) {
                    let host = crate::shadow::find(*address, 1)
                        .map(|s| (s.gpu(*address), s.left(*address)));
                    let _ = write!(text, " root{slot}={address:#x} host {host:#x?}");
                }
            }
        }
        let _ = write!(text, " tables {:?}", self.reads(compute));
        text
    }

    /// Diagnostics (`XTP_DUMP_DISPATCH_INPUTS=<program hash>,...`, each
    /// the start of a program's hash): for each dispatch of a matching
    /// program, logs at its list's submission what title memory holds
    /// then: the first [`DUMPED_CBV_BYTES`] bytes of each root CBV, and the
    /// 12 bytes of group counts an indirect one reads at `arguments` (what
    /// the CPU left there: a GPU writer runs after the submission). With
    /// `XTP_D3D12_DRED`, the device-removed report names the dispatches in
    /// flight by the same program and root addresses ([`State::op_text`]).
    #[cfg(feature = "diagnostics")]
    fn dump_at_submission(&self, list: &HostList, arguments: Option<u64>) {
        static PROGRAMS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
        let programs = PROGRAMS.get_or_init(|| {
            std::env::var("XTP_DUMP_DISPATCH_INPUTS")
                .unwrap_or_default()
                .split(',')
                .map(|p| p.trim().to_ascii_lowercase())
                .filter(|p| !p.is_empty())
                .collect()
        });
        if programs.is_empty() {
            return;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return;
        };
        for stage in &p.stages {
            let program = stage.shader.hash.to_string();
            if !programs.iter().any(|w| program.starts_with(w.as_str())) {
                continue;
            }
            let mut reads: Vec<(String, u64, u64)> = Vec::new();
            for &(slot, _) in &stage.root_cbvs {
                if let Some(&Root::View(address)) = self.roots_ref(true).get(usize::from(slot)) {
                    reads.push((format!("root CBV s{slot}"), address, DUMPED_CBV_BYTES));
                }
            }
            // CBVs in descriptor tables: the bytes their views name.
            let t = stage.translated();
            for u in t.used.iter().filter(|u| u.kind == ResourceType::Cbv) {
                if u.root != RootType::Table {
                    continue;
                }
                let Some(&Root::Table(table)) = self.roots_ref(true).get(usize::from(u.slot))
                else {
                    continue;
                };
                let handle = table + u64::from(u.table_offset_dwords) * 4;
                if let Some(View::Constant(address, size)) = crate::descriptor::view(handle) {
                    let what = format!("table CBV s{}+{}", u.slot, u.table_offset_dwords);
                    let bytes = u64::from(size).clamp(4, DUMPED_CBV_BYTES);
                    reads.push((what, address, bytes));
                }
            }
            if let Some(at) = arguments {
                reads.push(("indirect group counts".into(), at, 12));
            }
            // Inline constants: the user data the program reads, as set now.
            let user_data: Vec<String> = t
                .host
                .user_data
                .iter()
                .map(|u| {
                    let v = self.user_value(true, u.slot, u.root, u.dword);
                    format!("u{}={v:#x}", u.index)
                })
                .collect();
            tracing::info!(
                program,
                list = format_args!("{:#x}", list.list.as_raw() as usize),
                user_data = user_data.join(" "),
                "a dispatch's user data at its recording"
            );
            let host_list = list.list.as_raw() as usize;
            list.defer_host((0, 0), move || {
                for (what, address, len) in &reads {
                    // Read through the kernel: unmapped pages read as zeros.
                    let mut bytes = vec![0u8; *len as usize];
                    crate::host::copy_mapped(
                        bytes.as_mut_ptr(),
                        *address as *const u8,
                        bytes.len(),
                        false,
                    );
                    let (words, _) = bytes.as_chunks::<4>();
                    let words = words
                        .iter()
                        .map(|&w| format!("{:08x}", u32::from_le_bytes(w)))
                        .collect::<Vec<_>>()
                        .join(" ");
                    tracing::info!(
                        program,
                        list = format_args!("{host_list:#x}"),
                        what,
                        address = format_args!("{address:#x}"),
                        words,
                        "a dispatch's input at its submission"
                    );
                }
            });
        }
    }

    /// Runs a mesh dispatch of `x` groups on the host (the title's mesh
    /// programs are one-dimensional).
    pub(crate) fn dispatch_mesh(&mut self, list: &HostList, groups: [u32; 3]) {
        #[cfg(feature = "diagnostics")]
        {
            self.trace("mesh", groups[0], 0);
            crate::indirect::note_direct(list, self.pipeline, groups);
        }
        #[cfg(feature = "diagnostics")]
        let mut meshes = crate::meshstat::start(
            list,
            self.pipeline,
            crate::meshstat::Op::Mesh {
                groups: groups[0]
                    .saturating_mul(groups[1])
                    .saturating_mul(groups[2]),
            },
        );
        let ran = self.prepare_for(list, false, true).and_then(|()| {
            let l6 = list
                .list
                .cast::<ID3D12GraphicsCommandList6>()
                .map_err(|_| "no mesh dispatch on the host list")?;
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("mesh {groups:?}"), false));
            let _timed = crate::timestamps::command(
                list,
                crate::timestamps::What::Mesh,
                crate::pipeline::host_pipeline(self.pipeline as This),
            );
            let p =
                crate::pipeline::host_pipeline(self.pipeline as This).ok_or(Unready::NoPipeline)?;
            // SAFETY: la liste est ouverte et le pipeline mesh est lié.
            crate::indirect::direct(list, groups, p.group_base(), true, |[x, y, z]| unsafe {
                l6.DispatchMesh(x, y, z)
            });
            DRAWN.fetch_add(1, Ordering::Relaxed);
            Ok(())
        });
        #[cfg(feature = "diagnostics")]
        if ran.is_ok() {
            meshes.ran();
        }
        if let Err(why) = ran {
            self.occlusion.dropped_draw();
            dropped("mesh dispatch", self.pipeline, why);
        }
    }

    fn roots_ref(&self, compute: bool) -> &Vec<Root> {
        if compute {
            &self.compute
        } else {
            &self.graphics
        }
    }

    fn table_entry(&self, list: &HostList, u: &UsedResource, compute: bool) {
        let Some(Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
            return;
        };
        let handle = table + u64::from(u.table_offset_dwords) * 4;
        let Some(heap) = self.heap_of(handle) else {
            return;
        };
        let (Some(host), Some(index)) = (heap.host(), heap.index(handle)) else {
            return;
        };
        if let Some(view) = host.mirror(index, handle, true) {
            readable(list, u, &view);
        }
        // Brought up to date here: passes over the textures shaders pick
        // by index leave the slot to such draws.
        if u.kind == ResourceType::Srv {
            host.texture_slot_tabled(index);
        }
    }

    /// The descriptors a shader reads through tables. Tables holding a
    /// constant buffer are made afresh for the draw in dynamic slots, with
    /// the constants copied on the GPU for in-flight writers or into upload
    /// memory at submission. The host requires 256-byte alignment and the
    /// console requires 16. Returns each new table's byte offset by root
    /// parameter. The other tables use the title's slots.
    ///
    /// Within a list, the same table holding the same descriptors is made
    /// once unless intervening writes can change its GPU-filled constants.
    /// The world's large tables would otherwise take hundreds of slots per draw.
    fn tables(
        &mut self,
        list: &HostList,
        used: &[UsedResource],
        compute: bool,
    ) -> Result<Vec<(u8, u32)>, &'static str> {
        let tables = |u: &&UsedResource| u.root == RootType::Table;
        let mut fresh: Vec<u8> = used
            .iter()
            .filter(tables)
            .filter(|u| u.kind == ResourceType::Cbv)
            .map(|u| u.slot)
            .collect();
        // Tables reaching past a host heap smaller than the title's (the
        // host's limit, 1,000,000 views on NVIDIA) are made afresh too: the
        // title's slots there have no host slot of their own.
        for u in used.iter().filter(tables) {
            if fresh.contains(&u.slot) {
                continue;
            }
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
                continue;
            };
            let end = table + u64::from(u.table_offset_dwords) * 4;
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let (Some(host), Some(index)) = (heap.host(), heap.index(end)) else {
                continue;
            };
            if index >= host.count() {
                fresh.push(u.slot);
            }
        }
        fresh.sort_unstable();
        fresh.dedup();
        for u in used.iter().filter(tables) {
            if !fresh.contains(&u.slot) {
                self.table_entry(list, u, compute);
            }
        }
        let mut made = Vec::new();
        for slot in fresh {
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(slot)) else {
                continue;
            };
            let entries: Vec<&UsedResource> = used
                .iter()
                .filter(tables)
                .filter(|u| u.slot == slot)
                .collect();
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let Some(host) = heap.host() else {
                continue;
            };
            let n = entries
                .iter()
                .map(|u| u.table_offset_dwords / 8 + 1)
                .max()
                .unwrap_or(1);
            // The table and what its entries hold (record words).
            let key = {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                for k in 0..n {
                    let at = table + u64::from(k) * 32;
                    crate::descriptor::header(at).hash(&mut h);
                }
                (table, h.finish())
            };
            if let Some(cached) = self.fresh_tables.get(&key)
                && cached.constants.iter().all(|&(address, len, was)| {
                    let shadow = crate::shadow::existing(address, len);
                    was == CbvInput::at(list, address, len, shadow.as_ref())
                })
            {
                let first = cached.first;
                for u in entries.iter().filter(|u| u.kind != ResourceType::Cbv) {
                    let handle = table + u64::from(u.table_offset_dwords) * 4;
                    if let Some(view) = crate::descriptor::view(handle) {
                        readable(list, u, &view);
                    }
                }
                made.push((slot, first * 32));
                continue;
            }
            let Some(first) = host.dynamic(n) else {
                // Without its fresh table the shader would read the title's
                // slots, which for constant buffer tables hold no constant
                // view: a texture or null descriptor read as a buffer gives
                // a wild address on AMD and the device is removed (#59).
                // The draw or dispatch is dropped instead.
                counted!(
                    warn,
                    slots = n,
                    "bind dropped: no dynamic slots for its table"
                );
                return Err("no dynamic slots for its tables");
            };
            let mut constants = Vec::new();
            // Held until the recording goes (after its submission, if the
            // list is reset while it runs).
            self.recording
                .range(list, std::ptr::from_ref(heap) as usize, first, n);
            // The whole range: a shader may index past the entries its
            // used-resource list names (descriptor arrays).
            for k in 0..n {
                let index = first + k;
                let used = entries.iter().find(|u| u.table_offset_dwords / 8 == k);
                match crate::descriptor::view(table + u64::from(k) * 32) {
                    // No address or no bytes: a null constant view (zeros),
                    // not a view of 0 bytes or of title address 0.
                    Some(View::Constant(address, size)) if address == 0 || size == 0 => {
                        host.null_constant(index);
                    }
                    Some(View::Constant(address, size)) => {
                        // At most what a shader can index: the rest is not
                        // copied, and the host takes no larger view.
                        let size = crate::descriptor::host::capped_constant(address, size);
                        let len = u64::from(size);
                        let shadow = crate::shadow::existing(address, len);
                        let input = CbvInput::at(list, address, len, shadow.as_ref());
                        let gpu = match shadow.filter(|_| input.gpu_source.is_some()) {
                            Some(s) => self.fill_cbv(list, &s, address, len, input)?,
                            None => scratch_constant(list, address, len)
                                .ok_or("could not upload a table CBV")?,
                        };
                        constants.push((address, len, input));
                        host.constant_at(index, gpu, size);
                    }
                    Some(view) => {
                        host.make_at(index, &view);
                        if let Some(u) = used {
                            readable(list, u, &view);
                        }
                    }
                    None => {
                        host.make_null(index);
                        if used.is_some() {
                            once!(slot, "a table entry without a view");
                        }
                    }
                }
            }
            self.fresh_tables
                .insert(key, FreshTable { first, constants });
            self.new_descriptors.push(NewDescriptor::Table(key));
            made.push((slot, first * 32));
        }
        // A table made afresh in the dynamic slots of a view heap with a
        // stride table says so in its user data, for the shader to tell
        // it from a title table there (`emit::REMAPPED_SLOT`).
        for (slot, offset) in &mut made {
            let remapped = match self.roots_ref(compute).get(usize::from(*slot)) {
                Some(&Root::Table(table)) => self
                    .heap_of(table)
                    .and_then(|heap| heap.host())
                    .is_some_and(|host| host.stride_table().is_some()),
                _ => false,
            };
            if remapped {
                *offset |= xtp_shader::emit::REMAPPED_SLOT << 5;
            }
        }
        Ok(made)
    }

    /// The first draws, at debug level: what they bind and read
    /// (diagnostics builds).
    #[cfg(feature = "diagnostics")]
    fn trace(&self, kind: &str, count: u32, first: u32) {
        static TRACED: AtomicU64 = AtomicU64::new(0);
        if !tracing::enabled!(tracing::Level::DEBUG) {
            return;
        }
        let listed = crate::capture::listed(crate::queue::presented() + 1);
        let logged = listed || TRACED.fetch_add(1, Ordering::Relaxed) < 64;
        let targets: Vec<String> = self
            .targets
            .iter()
            .filter_map(|&t| crate::resource::host_resource(t as This))
            .map(|t| {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                format!("{}x{} f{}", d.Width, d.Height, d.Format.0)
            })
            .collect();
        let v = self.vertex[0];
        let line = format!(
            "{kind} {count} from {first} topology {} pipeline {:#x} targets {targets:?} vb {:#x}+{} stride {} reads {:?}",
            self.topology,
            self.pipeline,
            v.address,
            v.size,
            v.stride,
            self.reads(false)
        );
        if logged {
            tracing::debug!("draw {line}");
        }
        let mut frame = FRAME.lock().unwrap_or_else(|e| e.into_inner());
        if frame.len() < 256 {
            frame.push(line);
        }
    }

    /// What a draw's shaders read, one line per used resource (diagnostics
    /// builds).
    #[cfg(feature = "diagnostics")]
    fn reads(&self, compute: bool) -> Vec<String> {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return vec!["no host pipeline".into()];
        };
        let mut out = Vec::new();
        for stage in &p.stages {
            let t = stage.translated();
            for u in &t.used {
                let what = match (u.root, self.roots_ref(compute).get(usize::from(u.slot))) {
                    (RootType::Table, Some(Root::Table(table))) => {
                        let handle = table + u64::from(u.table_offset_dwords) * 4;
                        describe(crate::descriptor::view(handle).as_ref())
                    }
                    (root, _) => format!("{root:?}"),
                };
                out.push(format!(
                    "{} {:?} s{}+{}: {what}",
                    stage.shader.hash, u.kind, u.slot, u.table_offset_dwords
                ));
            }
        }
        out
    }

    /// What `XTP_FRAME_PROBE` reports of a stage's table reads
    /// (diagnostics): each entry's title descriptor and slot, and the host
    /// slot the shader reads, given the tables `made` afresh
    /// ([`State::tables`]).
    #[cfg(feature = "diagnostics")]
    fn probe_tables(
        &self,
        used: &[UsedResource],
        made: &[(u8, u32)],
        compute: bool,
    ) -> Vec<crate::frameprobe::Table> {
        used.iter()
            .filter(|u| u.root == RootType::Table)
            .filter_map(|u| {
                let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot))
                else {
                    return None;
                };
                let handle = table + u64::from(u.table_offset_dwords) * 4;
                let heap = self.heap_of(table);
                let host = heap.and_then(|h| h.host());
                let index = heap.and_then(|h| h.index(handle));
                Some(crate::frameprobe::Table {
                    slot: u.slot,
                    offset_dwords: u.table_offset_dwords,
                    kind: u.kind,
                    handle,
                    index,
                    mirrored: host.map(|h| h.count()),
                    title_offset: heap.map(|h| (table - h.start) as u32),
                    fresh_offset: made.iter().find(|m| m.0 == u.slot).map(|m| m.1),
                    host_view: match (host, index) {
                        (Some(h), Some(i)) if i < h.count() => h.describe_slot(i),
                        _ => None,
                    },
                })
            })
            .collect()
    }

    /// `XTP_FRAME_PROBE` for a dispatch dropped without a host pipeline
    /// (diagnostics): the tables its compute program says it uses, from its
    /// console program (`pipeline::Shader::native`), resolved from the
    /// roots the title bound, and for an indirect one its `arguments`.
    #[cfg(feature = "diagnostics")]
    fn probe_dropped(&self, list: &HostList, arguments: Option<(u64, u64)>) {
        let Some(shader) = crate::pipeline::compute_shader(self.pipeline as This) else {
            return;
        };
        if !crate::frameprobe::wanted(&shader.hash) {
            return;
        }
        let Ok(native) = shader.native() else {
            return;
        };
        let tables = self.probe_tables(&native.used, &[], true);
        crate::frameprobe::dropped(list, &shader.hash, tables);
        if let Some((address, len)) = arguments {
            crate::frameprobe::arguments(list, address, len);
        }
    }

    pub(crate) fn draw(
        &mut self,
        list: &HostList,
        vertices: u32,
        instances: u32,
        first: u32,
        first_instance: u32,
    ) {
        #[cfg(feature = "diagnostics")]
        self.trace("draw", vertices, first);
        #[cfg(feature = "diagnostics")]
        let mut meshes = crate::meshstat::start(
            list,
            self.pipeline,
            crate::meshstat::Op::Draw { vertices, first },
        );
        let rects = self.topology == RECTANGLES;
        let ran = self.prepare(list, false).and_then(|()| {
            if rects {
                self.rectangles(list, vertices, first)
            } else {
                Ok(false)
            }
        });
        if let Ok(expanded) = ran {
            let (count, from) = if rects {
                rectangle_vertices(vertices, first, expanded)
            } else {
                (vertices, first)
            };
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("draw {count}x{instances}"), false));
            let _timed = crate::timestamps::command(
                list,
                crate::timestamps::What::Draw,
                crate::pipeline::host_pipeline(self.pipeline as This),
            );
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawInstanced(count, instances, from, first_instance)
            };
            DRAWN.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            meshes.ran();
        }
        if let Err(why) = ran {
            self.occlusion.dropped_draw();
            dropped("draw", self.pipeline, why);
        }
    }

    /// Binds vertex buffers holding the rectangles of `count` vertices from
    /// `first` as triangles, made when the list is submitted, for the
    /// pipeline's per-vertex slots: whether it has any.
    fn rectangles(&self, list: &HostList, count: u32, first: u32) -> Result<bool, Unready> {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return Err(Unready::NoPipeline);
        };
        let rects = u64::from(count / 3);
        let mut expanded = false;
        for slot in 0..32 {
            let elements: Vec<(u32, u32)> = p
                .inputs
                .iter()
                .filter(|e| e.slot == slot && !e.per_instance)
                .map(|e| (e.offset, e.format))
                .collect();
            if elements.is_empty() {
                continue;
            }
            let v = self.vertex[slot as usize];
            let stride = u64::from(v.stride);
            let from = v.address + u64::from(first) * stride;
            if v.address == 0
                || stride == 0
                || (u64::from(first) + u64::from(count)) * stride > u64::from(v.size)
                || crate::shadow::find(from, rects * 3 * stride).is_none()
            {
                once!(slot, "rectangle vertices outside their buffer");
                return Err("rectangle vertices outside their buffer".into());
            }
            let Some((gpu, to)) = list.scratch(rects * 6 * stride) else {
                return Err("no scratch memory for rectangles".into());
            };
            let (from, to, stride) = (from as usize, to as usize, stride as usize);
            let n = rects as usize;
            list.defer_host((from, n * 3 * stride), move || {
                // SAFETY: the vertices lie in a title buffer (checked when
                // recorded) and the scratch memory is the list's.
                let (src, dst) = unsafe {
                    (
                        std::slice::from_raw_parts(from as *const u8, n * 3 * stride),
                        std::slice::from_raw_parts_mut(to as *mut u8, n * 6 * stride),
                    )
                };
                crate::primitive::expand_rects(src, dst, stride, &elements);
            });
            let view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: gpu,
                SizeInBytes: (rects * 6 * stride as u64) as u32,
                StrideInBytes: v.stride,
            };
            // SAFETY: an open list; the view names live scratch memory.
            unsafe { list.list.IASetVertexBuffers(slot, Some(&[view])) };
            expanded = true;
        }
        Ok(expanded)
    }

    pub(crate) fn draw_indexed(
        &mut self,
        list: &HostList,
        indices: u32,
        instances: u32,
        first: u32,
        base: i32,
        first_instance: u32,
    ) {
        #[cfg(feature = "diagnostics")]
        self.trace("indexed", indices, first);
        #[cfg(feature = "diagnostics")]
        let mut meshes = crate::meshstat::start(
            list,
            self.pipeline,
            crate::meshstat::Op::Indexed { indices, first },
        );
        if self.topology == RECTANGLES {
            #[cfg(feature = "diagnostics")]
            crate::list::cuts::INDEXED_RECT.fetch_add(1, Ordering::Relaxed);
            if let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This)
                && p.inputs.iter().any(|e| !e.per_instance)
            {
                let record =
                    Records::Direct([indices, instances, first, base as u32, first_instance]);
                match self.rectangles_indirect(
                    list,
                    &p.inputs,
                    (20, 1),
                    (record, None, Some(self.index)),
                    |s, l| s.prepare(l, false).map(|()| p.pso.clone()),
                ) {
                    Ok(()) => {
                        DRAWN.fetch_add(1, Ordering::Relaxed);
                        #[cfg(feature = "diagnostics")]
                        meshes.ran();
                    }
                    Err(why) => {
                        self.occlusion.dropped_draw();
                        dropped("indexed draw", self.pipeline, why);
                    }
                }
                return;
            }
            // Vertex-ID pipelines consume the original indexed triangles.
            // Their geometry stage emits the second half of each rectangle.
        }
        let ran = self.prepare(list, true);
        match ran {
            Ok(()) => {
                #[cfg(feature = "diagnostics")]
                list.note_op(|| self.op_text(format!("indexed {indices}x{instances}"), false));
                let _timed = crate::timestamps::command(
                    list,
                    crate::timestamps::What::Indexed,
                    crate::pipeline::host_pipeline(self.pipeline as This),
                );
                // SAFETY: an open list with everything set.
                unsafe {
                    list.list
                        .DrawIndexedInstanced(indices, instances, first, base, first_instance)
                };
                DRAWN.fetch_add(1, Ordering::Relaxed);
                #[cfg(feature = "diagnostics")]
                meshes.ran();
            }
            Err(why) => {
                self.occlusion.dropped_draw();
                dropped("indexed draw", self.pipeline, why);
            }
        }
    }
}

/// The vertices (count, first) a direct rectangle draw of `vertices` from
/// `first` runs as a triangle list: six a rectangle from 0, in the buffers
/// they were expanded into (`expanded`); for a pipeline without per-vertex
/// inputs (corners from the vertex id), each rectangle's first triangle:
/// the title's draw, with whole rectangles, whose fourth corners the
/// pipeline's geometry stage makes ([`crate::rectangles::geometry_stage`]).
/// Its shader sees vertex ids from 0, the start left out, on the host as on
/// the console (docs/analysis/drops.md §5.1.1).
fn rectangle_vertices(vertices: u32, first: u32, expanded: bool) -> (u32, u32) {
    let rects = vertices / 3;
    if expanded {
        (rects.saturating_mul(6), 0)
    } else {
        (rects * 3, first)
    }
}

impl State {
    /// Indirect execution of a one-argument signature of `kind` (PC
    /// numbering: 0 draw, 1 indexed draw, 2 dispatch, 10 mesh dispatch) with
    /// records `stride` bytes apart at title address `arguments`, and the
    /// record count at `count` if the title gives one: the GPU reads them
    /// from the buffer's host copy, where earlier dispatches may have
    /// written them. False if it cannot run on the host; true, and nothing
    /// drawn or dispatched, for no record.
    pub(crate) fn execute_indirect(
        &mut self,
        list: &HostList,
        (kind, stride, max): (u32, u32, u32),
        arguments: u64,
        count: Option<u64>,
    ) -> bool {
        const DRAW: u32 = 0;
        const INDEXED: u32 = 1;
        const DISPATCH: u32 = 2;
        const MESH: u32 = 10;
        let path = match kind {
            DRAW => "indirect kind 0 (draw)",
            INDEXED => "indirect kind 1 (indexed draw)",
            DISPATCH => "indirect kind 2 (dispatch)",
            MESH => "indirect kind 10 (mesh dispatch)",
            _ => "indirect, another kind",
        };
        if max == 0 {
            return true;
        }
        #[cfg(feature = "diagnostics")]
        let mut meshes = crate::meshstat::start(
            list,
            self.pipeline,
            crate::meshstat::Op::Indirect { kind, stride, max },
        );
        let pipeline = self.pipeline;
        // Counted in `list` (its indirect drops); here only why.
        let fail = |why: Unready| {
            #[cfg(feature = "diagnostics")]
            note_unready(path, pipeline, why);
            #[cfg(not(feature = "diagnostics"))]
            let _ = (path, pipeline, why);
            false
        };
        // Skip before the argument clamp can mutate the indirect buffer.
        if kind == DISPATCH && crate::pipeline::builder_held(self.pipeline as This) {
            return fail("BVH builder held".into());
        }
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            return fail(Unready::NoPipeline);
        }
        let indexed_rect = kind == INDEXED && self.topology == RECTANGLES;
        let indexed_expansion = if indexed_rect {
            #[cfg(feature = "diagnostics")]
            crate::list::cuts::INDIRECT_RECT.fetch_add(1, Ordering::Relaxed);
            let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
                return fail(Unready::NoPipeline);
            };
            p.inputs.iter().any(|e| !e.per_instance)
        } else {
            false
        };
        let len = u64::from(stride) * u64::from(max.max(1));
        let Some(args) = crate::shadow::find(arguments, len) else {
            return fail("arguments outside title buffers".into());
        };
        let counter = match count {
            Some(at) => match crate::shadow::find(at, 4) {
                Some(s) => Some((s, at)),
                None => return fail("count outside title buffers".into()),
            },
            None => None,
        };
        #[cfg(feature = "diagnostics")]
        {
            use crate::meshstat::{What, read};
            read(What::Args, &args, arguments, len);
            if let Some((s, at)) = &counter {
                read(What::Count, s, *at, 4);
            }
        }
        if self.topology == RECTANGLES && (kind == DRAW || indexed_expansion) {
            let rects = (&args, arguments);
            return match self.indirect_rectangles(
                list,
                (stride, max),
                rects,
                &counter,
                indexed_expansion,
            ) {
                Ok(()) => {
                    #[cfg(feature = "diagnostics")]
                    if indexed_rect {
                        crate::list::cuts::INDIRECT_RECT_RECORDED.fetch_add(1, Ordering::Relaxed);
                    }
                    DRAWN.fetch_add(1, Ordering::Relaxed);
                    #[cfg(feature = "diagnostics")]
                    meshes.ran();
                    true
                }
                Err(why) => fail(why),
            };
        }
        if !matches!(kind, DRAW | INDEXED | DISPATCH | MESH) {
            return fail("no host signature".into());
        }
        let Some(p) = crate::pipeline::host_pipeline(pipeline as This) else {
            // XTP_FRAME_PROBE: what it would read, and its arguments.
            #[cfg(feature = "diagnostics")]
            if kind == DISPATCH {
                self.probe_dropped(list, Some((arguments, len)));
            }
            if kind == DISPATCH {
                // Le substitut de ray tracing utilise le nombre réel de commandes
                // comme prédicat GPU. Lui seul a besoin du bridage en cas de rejet.
                let commands = if crate::rt_standin::active()
                    && crate::rt_standin::program(pipeline as This).is_some()
                {
                    args.refresh(list, arguments, len);
                    let count = counter.as_ref().map(|(s, at)| {
                        s.refresh(list, *at, 4);
                        (&s.buffer, s.offset(*at))
                    });
                    let writes = list.indirect_writes(&args.buffer, count.map(|(b, _)| b));
                    crate::indirect::bound(
                        list,
                        (&args.buffer, args.offset(arguments)),
                        (kind, stride, max),
                        count,
                        None,
                        writes,
                        crate::indirect::Output::Uav,
                    )
                    .map(|out| {
                        if out.dispatched {
                            self.host_pipeline = 0;
                            self.host_roots[1] = 0;
                        }
                        (out.buffer, out.offset)
                    })
                } else {
                    None
                };
                self.rt_dropped(list, crate::rt_standin::Work::Commands(commands));
            }
            return fail(Unready::NoPipeline);
        };
        if p.compute != (kind == DISPATCH) || p.mesh != (kind == MESH) {
            return fail("another kind of pipeline bound".into());
        }
        use crate::indirect::Output;
        let dispatch = matches!(kind, DISPATCH | MESH);
        let output = if dispatch {
            Output::dispatch(p.group_base().is_some())
        } else {
            Output::Draw
        };
        let signature = match output {
            Output::Native => host_signature(kind, 12),
            Output::Expanded | Output::NativePrefix => p
                .dispatch_signature
                .get_or_init(|| crate::indirect::signature(kind, &p.root, p.group_base()))
                .clone(),
            _ => host_signature(kind, 24),
        };
        let Some(signature) = signature else {
            return fail("no host signature".into());
        };
        let native = if output == Output::NativePrefix {
            let Some(sig) = host_signature(kind, 12) else {
                return fail("no native dispatch signature".into());
            };
            Some(sig)
        } else {
            None
        };

        let counter = counter.as_ref().map(|(s, at)| {
            s.refresh(list, *at, 4);
            (&s.buffer, s.offset(*at))
        });
        args.refresh(list, arguments, len);
        let writes = list.indirect_writes(&args.buffer, counter.map(|(b, _)| b));
        let compute = kind == DISPATCH;
        self.pending_compute = Some(Vec::new());
        let ready = if compute {
            self.bind(list, p, true, [0; 3]).map_err(Unready::from)
        } else {
            self.prepare_for(list, kind == INDEXED, kind == MESH)
        };
        let roots = self.pending_compute.take();
        if let Err(why) = ready {
            return fail(why);
        }
        // A trace's TLAS taken at submission has no indirect form here yet.
        if self.tlas_at_submission.take().is_some() {
            return fail(NO_HOST_TLAS.into());
        }
        // XTP_FRAME_PROBE: the argument records of a probed op.
        #[cfg(feature = "diagnostics")]
        crate::frameprobe::arguments(list, arguments, len);
        let stats = self.own.indirect_samples.next(list, pipeline);
        let Some(out) = crate::indirect::bound(
            list,
            (&args.buffer, args.offset(arguments)),
            (kind, stride, max),
            counter,
            stats,
            writes,
            output,
        ) else {
            return fail("not bounded (no clamp or invalid arguments)".into());
        };
        if out.dispatched {
            self.host_roots[1] = 0;
            self.host_pipeline = 0;
        }
        self.bind_pipeline(list, p, compute);
        if let Some(roots) = roots {
            for (param, value) in roots {
                value.set(list, param);
            }
        }
        list.keep(&signature);
        #[cfg(feature = "diagnostics")]
        list.note_op(|| {
            self.op_text(
                format!("indirect kind {kind} max {max} at {arguments:#x}"),
                kind == DISPATCH,
            )
        });
        #[cfg(feature = "diagnostics")]
        if kind == DISPATCH {
            self.dump_at_submission(list, Some(arguments));
        }
        let _timed = crate::timestamps::command(
            list,
            crate::timestamps::What::Indirect(kind),
            crate::pipeline::host_pipeline(self.pipeline as This),
        );
        let command = out.offset
            + if output == Output::Expanded && p.group_base().is_none() {
                16
            } else {
                4
            };
        // SAFETY: la passe a borné les commandes et leur nombre dans la même tranche.
        unsafe {
            if compute && std::mem::take(&mut self.dropped_dispatch) {
                list.list.ResourceBarrier(&[crate::host::uav_barrier()]);
            }
            if let (Some(sig), Some(offset)) = (&native, out.native_offset) {
                list.list.ExecuteIndirect(
                    sig,
                    max.min(out.capacity),
                    &out.buffer,
                    offset + 4,
                    &out.buffer,
                    offset,
                );
            }
            list.list.ExecuteIndirect(
                &signature,
                out.capacity,
                &out.buffer,
                command,
                &out.buffer,
                out.offset,
            );
        }
        #[cfg(feature = "diagnostics")]
        if indexed_rect {
            crate::list::cuts::INDIRECT_RECT_RECORDED.fetch_add(1, Ordering::Relaxed);
        }
        if kind == DISPATCH {
            DISPATCHED.fetch_add(1, Ordering::Relaxed);
        } else {
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        #[cfg(feature = "diagnostics")]
        meshes.ran();
        true
    }
}

/// The records of an indirect rectangle draw, and its count if the title
/// gives one: a host copy and a title address each.
type RectangleArguments<'a> = (
    Records<'a>,
    Option<(&'a Arc<crate::shadow::Shadow>, u64)>,
    Option<IndexView>,
);

impl State {
    /// An indirect draw of rectangle lists: `max` records `stride` bytes
    /// apart at title address `arguments` (host copy `args`), and the count
    /// at `counter` if the title gives one, expanded on the GPU
    /// ([`crate::rectangles`]) and run from the list's own arguments.
    fn indirect_rectangles(
        &mut self,
        list: &HostList,
        (stride, max): (u32, u32),
        (args, arguments): (&Arc<crate::shadow::Shadow>, u64),
        counter: &Option<(Arc<crate::shadow::Shadow>, u64)>,
        indexed: bool,
    ) -> Result<(), Unready> {
        let p = crate::pipeline::host_pipeline(self.pipeline as This).ok_or(Unready::NoPipeline)?;
        if p.compute || p.mesh {
            return Err(if p.compute {
                "a compute pipeline bound"
            } else {
                "a mesh pipeline bound"
            }
            .into());
        }
        if !p.inputs.iter().any(|e| !e.per_instance) {
            note_vertex_id_rectangles(p);
        }
        let counter = counter.as_ref().map(|(s, at)| (s, *at));
        self.rectangles_indirect(
            list,
            &p.inputs,
            (stride, max),
            (
                Records::Title(args, arguments),
                counter,
                indexed.then_some(self.index),
            ),
            |s, l| s.prepare(l, false).map(|()| p.pso.clone()),
        )
    }

    /// Valide les ressources du dessin avant le bridage, puis rétablit son PSO
    /// après les passes de calcul et exécute les rectangles développés.
    fn rectangles_indirect(
        &mut self,
        list: &HostList,
        inputs: &[crate::pipeline::host::Input],
        (stride, max): (u32, u32),
        arguments: RectangleArguments,
        bind: impl FnOnce(&mut State, &HostList) -> Result<ID3D12PipelineState, Unready>,
    ) -> Result<(), Unready> {
        // The rewritten records: draw arguments, 16 bytes apart.
        let signature = host_signature(0, 16).ok_or("no host signature")?;
        let writes = match arguments.0 {
            Records::Title(s, _) => {
                list.indirect_writes(&s.buffer, arguments.1.map(|(c, _)| &c.buffer))
            }
            Records::Direct(_) => Default::default(),
        };
        let pso = bind(self, list)?;
        let mut batches = crate::rectangles::Batches::new(writes);
        let mut first = 0;
        while first < max {
            let prepared = match self.expand_rectangles(
                list,
                inputs,
                (stride, first..max),
                arguments,
                &mut batches,
            ) {
                Ok(batch) => batch,
                Err(why) if first == 0 => return Err(why),
                Err(why) => {
                    // Les premiers lots sont déjà dans la liste. Le reste du
                    // dessin aurait pu contribuer à la requête d’occlusion.
                    self.occlusion.dropped_draw();
                    tracing::warn!(first, max, ?why, "dessin de rectangles partiel");
                    break;
                }
            };
            self.host_roots[1] = 0;
            // SAFETY: les passes de calcul ont remplacé le PSO, les racines graphiques restent valides.
            unsafe { list.list.SetPipelineState(&pso) };
            self.host_pipeline = pso.as_raw() as usize;
            prepared.bind_vertices(list);
            #[cfg(feature = "diagnostics")]
            list.note_op(|| {
                let text = match arguments.0 {
                    Records::Title(_, at) => format!("indirect rectangles max {max} at {at:#x}"),
                    Records::Direct([indices, instances, ..]) => {
                        format!("indexed rectangles {indices}x{instances}")
                    }
                };
                self.op_text(text, false)
            });
            let _timed = crate::timestamps::command(
                list,
                crate::timestamps::What::Rectangles,
                crate::pipeline::host_pipeline(self.pipeline as This),
            );
            // SAFETY: an open list with everything set; the arguments and their
            // count are the list's own, in the indirect argument state (records
            // past the title's count do not run); the signature is cached.
            unsafe {
                list.list.ExecuteIndirect(
                    &signature,
                    prepared.max,
                    &prepared.args,
                    prepared.offset,
                    &prepared.args,
                    prepared.count_offset,
                )
            };
            first += prepared.max;
        }
        Ok(())
    }

    /// Records the passes that expand an indirect rectangle draw into the
    /// list's scratch, which its recording holds until it has gone (the
    /// GPU may run it after the title resets the list).
    fn expand_rectangles(
        &mut self,
        list: &HostList,
        inputs: &[crate::pipeline::host::Input],
        (stride, range): (u32, std::ops::Range<u32>),
        (records, counter, index): RectangleArguments,
        batches: &mut crate::rectangles::Batches,
    ) -> Result<crate::rectangles::Prepared, Unready> {
        let vertices = self.vertex.map(|v| (v.address, v.size, v.stride));
        self.recording.lease(list);
        Ok(crate::rectangles::expand(
            list,
            &mut self.own.rects,
            inputs,
            &vertices,
            (records, stride, range.clone()),
            index,
            counter,
            if range.start == 0 && matches!(records, Records::Title(..)) {
                self.own.indirect_samples.next(list, self.pipeline)
            } else {
                None
            },
            batches,
        )?)
    }
}

/// Logs a pipeline whose indirect rectangles make their corners from the
/// vertex id (see [`crate::rectangles`]), once each: its geometry stage
/// makes their fourth corners. Cheap on every draw: a recording thread
/// looks in its own set first, so the shared one is locked once per thread
/// and pipeline.
fn note_vertex_id_rectangles(p: &HostPipeline) {
    thread_local! {
        /// The host pipelines this thread has seen, by address.
        static MINE: std::cell::RefCell<std::collections::HashSet<usize>> =
            std::cell::RefCell::new(std::collections::HashSet::new());
    }
    if !MINE.with_borrow_mut(|mine| mine.insert(std::ptr::from_ref(p) as usize)) {
        return;
    }
    static SEEN: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&p.name) {
        seen.push(p.name.clone());
        tracing::info!(
            pipeline = %p.name,
            "indirect rectangles without vertex inputs: each rectangle's three vertices, the fourth \
             corner made by the pipeline's geometry stage unless its creation said otherwise \
             (vertex ids from 0 in each record, as on the console)"
        );
    }
}

/// The host command signature for one argument of `kind` with records
/// `stride` bytes apart (no root arguments, so no root signature), made
/// once.
pub(crate) fn host_signature(kind: u32, stride: u32) -> Option<ID3D12CommandSignature> {
    static MADE: std::sync::Mutex<Vec<((u32, u32), ID3D12CommandSignature)>> =
        std::sync::Mutex::new(Vec::new());
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, s)) = made.iter().find(|(k, _)| *k == (kind, stride)) {
        return Some(s.clone());
    }
    let argument = D3D12_INDIRECT_ARGUMENT_DESC {
        Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
        ..Default::default()
    };
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: stride,
        NumArgumentDescs: 1,
        pArgumentDescs: &argument,
        NodeMask: 0,
    };
    let mut signature: Option<ID3D12CommandSignature> = None;
    // SAFETY: a description of one argument; no root arguments.
    let result = unsafe {
        crate::host::host()?.device.CreateCommandSignature(
            &desc,
            None::<&ID3D12RootSignature>,
            &mut signature,
        )
    };
    if let Err(e) = result {
        once!(error = %e, kind, stride, "host command signature refused");
        return None;
    }
    let signature = signature?;
    made.push(((kind, stride), signature.clone()));
    Some(signature)
}

/// Gets what a shader reads or writes through a view ready: buffer data
/// copied, textures moved to the state they need.
fn readable(list: &HostList, u: &UsedResource, view: &View) {
    const SHADER_RESOURCE: u32 = 0x40 | 0x80;
    const UNORDERED_ACCESS: u32 = 0x8;
    let placed = crate::descriptor::host::placed_texture(view);
    let view = placed.as_ref().unwrap_or(view);
    #[cfg(feature = "diagnostics")]
    if let View::Shader(r, _) | View::Unordered(r, ..) = view {
        crate::passdump::binding(list, *r as This, u.slot, u.table_offset_dwords);
    }
    // Buffer views read title memory: their range is copied at submission;
    // what shaders write through unordered access goes back to it after.
    if let Some((address, len)) = crate::descriptor::host::buffer_range(view)
        && len > 0
        && let Some(shadow) = crate::shadow::find(address, len)
    {
        shadow.refresh(list, address, len);
        if u.kind == ResourceType::Uav {
            shadow.gpu_writes(list, address, len);
        }
        #[cfg(feature = "diagnostics")]
        if u.kind == ResourceType::Uav {
            crate::meshstat::wrote(&shadow, address, len);
        } else {
            crate::meshstat::read(crate::meshstat::What::View, &shadow, address, len);
        }
    }
    match (view, u.kind) {
        (View::Shader(resource, _), ResourceType::Srv) => {
            if let Some(t) = crate::resource::host_resource(*resource as This)
                && crate::texture::title_written(t)
            {
                crate::texture::upload(list, *resource as This, t);
            }
            #[cfg(feature = "diagnostics")]
            crate::frameprobe::texture(list, *resource as This, false);
            crate::resource::alias_copy(list, *resource as This);
            list.twin_use(*resource as This, false, false);
            crate::list::ensure_state(list, *resource as This, SHADER_RESOURCE);
            #[cfg(feature = "diagnostics")]
            if let Some(t) = crate::resource::host_resource(*resource as This) {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                if d.Flags.0 & 7 == 0 {
                    tracing::debug!(
                        resource = format_args!("{resource:#x}"),
                        ?d,
                        "a shader reads a texture the title fills"
                    );
                }
            }
        }
        (View::Unordered(resource, _, _), ResourceType::Uav) => {
            #[cfg(feature = "diagnostics")]
            crate::frameprobe::texture(list, *resource as This, true);
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(*resource as This);
            #[cfg(feature = "diagnostics")]
            crate::passdump::written(list, *resource as This);
            crate::resource::decode::bound_uav(list, *resource as This);
            list.texture_written(*resource as This);
            list.twin_use(*resource as This, true, true);
            // Which twin the view writes, of a depth-stencil texture's two.
            list.twin_view_written(*resource as This, view);
            crate::list::ensure_state(list, *resource as This, UNORDERED_ACCESS);
        }
        _ => {}
    }
}

/// A view in a few words (diagnostics builds).
#[cfg(feature = "diagnostics")]
fn describe(view: Option<&View>) -> String {
    let texture = |r: This| {
        crate::resource::host_resource(r).map(|t| {
            // SAFETY: a live resource.
            let d = unsafe { t.GetDesc() };
            format!(
                "{}x{} f{} flags {:#x}",
                d.Width, d.Height, d.Format.0, d.Flags.0
            )
        })
    };
    match view {
        None => "no view".into(),
        Some(v @ View::Shader(r, _)) => texture(*r as This)
            .unwrap_or_else(|| format!("buffer {:?}", crate::descriptor::host::buffer_range(v))),
        // A buffer has no host texture: its title range, as for a shader view.
        Some(v @ View::Unordered(r, ..)) => texture(*r as This).map_or_else(
            || format!("uav buffer {:?}", crate::descriptor::host::buffer_range(v)),
            |t| format!("uav {t}"),
        ),
        Some(View::Placed(u, r, d)) => format!(
            "placed uav={u} dim {} {}x{} f{} layout {:#x} at {:#x}",
            r[0],
            u64::from(r[4]) | u64::from(r[5]) << 32,
            r[6],
            r[8],
            r[11],
            u64::from(d[10]) | u64::from(d[11]) << 32
        ),
        // The first four values, as a colour rectangle's constants are.
        Some(View::Constant(address, size))
            if *size >= 16 && crate::texture::committed(*address as *const u8, 16) =>
        {
            // SAFETY: 16 committed bytes of title memory.
            let v = unsafe { (*address as *const [f32; 4]).read_unaligned() };
            format!("Constant({address:#x}, {size}) {v:?}")
        }
        Some(v) => format!("{v:?}").chars().take(60).collect(),
    }
}

/// The console's own primitive topologies.
const TRIANGLE_FAN: u32 = 6;
const RECTANGLES: u32 = 17;
const LINE_LOOP: u32 = 18;
const QUADS: u32 = 19;
const QUAD_STRIP: u32 = 20;

/// `State::host_heaps` after a pass of ours: no heap's key, which is 0, a
/// title heap's address or `OWN_VIEW_HEAP` (all ones), so that the next
/// bind sets the heaps it wants (`State::after_our_pass`).
const AFTER_OUR_PASS: usize = 1;

/// Fixed-function state, straight onto the host list.
pub(crate) fn viewports(list: &HostList, viewports: &[D3D12_VIEWPORT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetViewports(viewports) };
    list.note_viewports(viewports);
}

pub(crate) fn scissors(list: &HostList, rects: &[RECT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetScissorRects(rects) };
    list.note_scissors(rects);
}

pub(crate) fn blend_factor(list: &HostList, factor: Option<&[f32; 4]>) {
    // SAFETY: an open list.
    unsafe { list.list.OMSetBlendFactor(factor) };
}

pub(crate) fn stencil_ref(list: &HostList, reference: u32) {
    // SAFETY: an open list.
    unsafe { list.list.OMSetStencilRef(reference) };
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::descriptor::host::SHARED_CHUNK;
    use crate::descriptor::{OWN_SAMPLER_HEAP, OWN_VIEW_HEAP, TestHeap};
    use crate::host::Host;
    use crate::testgpu::{Kernel, Pass, TitleBuffer, TitleList, pattern};
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    use windows::core::s;
    use xtp_shader::emit::ROOT_CBV_SPACE;

    /// Words a [`Reader`] dispatch reads.
    const WORDS: usize = 128;

    /// How a [`Reader`] dispatch gets its view.
    enum Bind {
        Table(D3D12_GPU_DESCRIPTOR_HANDLE),
        Cbv(u64),
    }

    enum ReaderView {
        RootCbv,
        TableCbv,
        TableSrv,
    }

    /// A compute kernel that copies the first [`WORDS`] words a root view
    /// reads (root parameter 0) into a root UAV (parameter 1). The view is
    /// bound as [`HostPipeline::new`] binds the translated shaders' root
    /// views: a root SRV as a table of one SRV, `t0` in the root views'
    /// space, with volatile descriptors and data; a root CBV as a root
    /// descriptor or a one-CBV table at `b0` in that space. Only the view
    /// heap is bound for these buffer reads.
    struct Reader {
        root: ID3D12RootSignature,
        pso: ID3D12PipelineState,
    }

    impl Reader {
        fn new(host: &Host, binding: ReaderView) -> Reader {
            let cbv = !matches!(binding, ReaderView::TableSrv);
            let source = if cbv {
                "cbuffer Src : register(b0, space1) { uint4 src[4096]; };
RWByteAddressBuffer dst : register(u0);
[numthreads(128, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    dst.Store(id.x * 4, src[id.x / 4][id.x % 4]);
}"
            } else {
                "ByteAddressBuffer src : register(t0, space1);
RWByteAddressBuffer dst : register(u0);
[numthreads(128, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    dst.Store(id.x * 4, src.Load(id.x * 4));
}"
            };
            assert_eq!(ROOT_CBV_SPACE, 1, "the kernels' register space");
            let mut code: Option<ID3DBlob> = None;
            // SAFETY: the source and names live through the call.
            unsafe {
                D3DCompile(
                    source.as_ptr().cast(),
                    source.len(),
                    s!("root view test"),
                    None,
                    None,
                    s!("main"),
                    s!("cs_5_1"),
                    0,
                    0,
                    &mut code,
                    None,
                )
            }
            .expect("the test kernel compiles");
            let code = code.expect("compiled code");
            let range = D3D12_DESCRIPTOR_RANGE1 {
                RangeType: if cbv {
                    D3D12_DESCRIPTOR_RANGE_TYPE_CBV
                } else {
                    D3D12_DESCRIPTOR_RANGE_TYPE_SRV
                },
                NumDescriptors: 1,
                BaseShaderRegister: 0,
                RegisterSpace: ROOT_CBV_SPACE,
                Flags: D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE
                    | D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE,
                OffsetInDescriptorsFromTableStart: 0,
            };
            let view = if matches!(binding, ReaderView::RootCbv) {
                D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                            ShaderRegister: 0,
                            RegisterSpace: ROOT_CBV_SPACE,
                            Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                        },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                }
            } else {
                D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE1 {
                            NumDescriptorRanges: 1,
                            pDescriptorRanges: &range,
                        },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                }
            };
            let params = [
                view,
                D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                            ShaderRegister: 0,
                            RegisterSpace: 0,
                            Flags: D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
                        },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                },
            ];
            let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
                Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
                Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                    Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                        NumParameters: params.len() as u32,
                        pParameters: params.as_ptr(),
                        NumStaticSamplers: 0,
                        pStaticSamplers: std::ptr::null(),
                        Flags: D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED,
                    },
                },
            };
            // SAFETY: the description and the code live through the calls.
            unsafe {
                let mut blob: Option<ID3DBlob> = None;
                D3D12SerializeVersionedRootSignature(&desc, &mut blob, None)
                    .expect("the root signature serializes");
                let blob = blob.expect("a root signature");
                let bytes = std::slice::from_raw_parts(
                    blob.GetBufferPointer().cast::<u8>(),
                    blob.GetBufferSize(),
                );
                let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes).unwrap();
                let pso = host
                    .device
                    .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&root),
                        CS: D3D12_SHADER_BYTECODE {
                            pShaderBytecode: code.GetBufferPointer(),
                            BytecodeLength: code.GetBufferSize(),
                        },
                        ..Default::default()
                    })
                    .unwrap();
                Reader { root, pso }
            }
        }

        /// Runs `list` with a dispatch reading the view `bind` gives (it
        /// runs first: heaps, then the root signature) and returns the
        /// words read.
        fn read(&self, list: &TitleList, bind: impl FnOnce(&HostList) -> Bind) -> Vec<u32> {
            let dst = TitleBuffer::new(1 << 16);
            dst.set_words(|_| 0xdead_beef);
            let to = dst.shadow();
            let l = list.host();
            let bytes = WORDS as u64 * 4;
            to.refresh(l, dst.address(), bytes);
            to.gpu_writes(l, dst.address(), bytes);
            let view = bind(l);
            self.dispatch(l, view, to.gpu(dst.address()));
            list.run();
            dst.words()[..WORDS].to_vec()
        }

        fn dispatch(&self, l: &HostList, view: Bind, dst: u64) {
            l.keep(&self.root);
            l.keep(&self.pso);
            // SAFETY: an open list; live objects kept until it is reset; the
            // view's slot is in the heap set on the list, its memory held by
            // the list.
            unsafe {
                l.list.SetComputeRootSignature(&self.root);
                l.list.SetPipelineState(&self.pso);
                match view {
                    Bind::Table(handle) => l.list.SetComputeRootDescriptorTable(0, handle),
                    Bind::Cbv(gpu) => l.list.SetComputeRootConstantBufferView(0, gpu),
                }
                l.list.SetComputeRootUnorderedAccessView(1, dst);
                l.list.Dispatch(1, 1, 1);
            }
        }
    }

    /// A shader that reads the runtime stride table is dropped where the
    /// view heap set on the list has none (XTP's own heap, or a title heap
    /// made with `XTP_STRIDE_TABLE` off, as in tests): no root SRV of
    /// another buffer, or none at all, is bound in its place.
    #[test]
    fn a_missing_stride_table_drops_the_draw() {
        let test = "draw::tests::a_missing_stride_table_drops_the_draw";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let list = crate::testgpu::TitleList::new();
        let l = list.host();
        // No view heap of the title's: XTP's own, which has no table.
        let mut own = state(std::ptr::null());
        heaps(&mut own, l);
        assert_eq!(own.host_heaps[0], crate::descriptor::OWN_VIEW_HEAP);
        assert!(own.stride_table(l).is_err());
        // A title heap whose host heap has no table.
        let heap = crate::descriptor::TestHeap::views(4);
        let mut titled = state(heap.0);
        heaps(&mut titled, l);
        assert_eq!(titled.host_heaps[0], heap.0 as usize);
        assert!(titled.stride_table(l).is_err());
        assert!(titled.own.strides.is_empty(), "nothing held");
    }

    /// A compute program whose two thread groups each load a heap slot from
    /// the buffer in table slot 0 (an index read at run time, as the
    /// instance culling does), read `[lane, 4]` of the structured view of
    /// 16 bytes there and store it at `[lane, 16 * group]` of the
    /// structured view of 32 bytes in table slot 1 (strides known).
    fn picking_program() -> xtp_shader::emit::HostShader {
        use xtp_shader::native::{ComputeIo, StageIo};
        use xtp_shader::{HwStage, NativeProgram, RootParam};
        let words: &[u32] = &[
            0xBE94_0304, // s_mov_b32 s20, s4
            0xBE95_0380, // s_mov_b32 s21, 0
            0xF408_040A,
            0xFA00_0000, // s_load_dwordx4 s[16:19], s[20:21], 0x0
            0x8F07_8205, // s_lshl_b32 s7, s5, 2
            0xF420_0188,
            0x0E00_0000, // s_buffer_load_dword s6, s[16:19], s7
            0x8F06_8506, // s_lshl_b32 s6, s6, 5
            0xF408_020A,
            0x0C00_0000, // s_load_dwordx4 s[8:11], s[20:21], s6
            0xF408_030A,
            0xFA00_0020, // s_load_dwordx4 s[12:15], s[20:21], 0x20
            0x7E02_0284, // v_mov_b32 v1, 4
            0xE030_3000,
            0x8002_0200, // buffer_load_dword v2, v[0:1], s[8:11], 0 idxen offen
            0x8F07_8405, // s_lshl_b32 s7, s5, 4
            0x7E02_0207, // v_mov_b32 v1, s7
            0xE070_3000,
            0x8003_0200, // buffer_store_dword v2, v[0:1], s[12:15], 0 idxen offen
            0xBF81_0000, // s_endpgm
        ];
        let used = |kind, table_offset_dwords, stride| UsedResource {
            kind,
            root: RootType::Table,
            table_offset_dwords,
            lower_bound: 0,
            stride,
            dwords: 4,
            user_data: 4,
            slot: 1,
        };
        let param = |root, sgprs, user_data, slot| RootParam {
            root,
            table_offset_dwords: 0,
            sgprs,
            base_register: 0,
            user_data,
            slot,
        };
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            float_mode: 0,
            vgprs: 4,
            layout: vec![
                param(RootType::Constants, 4, 0, 0),
                param(RootType::Table, 1, 4, 1),
            ],
            used: vec![
                used(ResourceType::Srv, 0, 16),
                used(ResourceType::Uav, 8, 32),
            ],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 5 << 1 | 1 << 7,
                threads: [4, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = xtp_shader::ir::build(&native.code, native.wave64, native.vgprs).unwrap();
        let st = xtp_shader::structure::structure(&p).unwrap();
        let res = xtp_shader::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = xtp_shader::io::raise(&p, &native).unwrap();
        // An empty container: the tag, a digest, version 1, its size.
        let mut container = b"DXBC".to_vec();
        container.extend([0; 16]);
        container.extend([1u32, 32, 0].into_iter().flat_map(u32::to_le_bytes));
        let c = xtp_shader::Container::parse(&container).unwrap();
        xtp_shader::emit::emit(&c, &native, &p, &st, &io, &res).unwrap()
    }

    /// Buffer views the title wrote into slots a shader picks when it runs
    /// (watch.md item 170): no draw's table names them, so they stay null
    /// and read zeros, and the program writes nothing; once
    /// `picked_buffers` has made them and brought their memory up to date,
    /// each group reads the bytes of the view in the slot it chose and its
    /// writes reach title memory.
    #[test]
    fn buffer_views_of_slots_picked_at_run_time_are_made_and_read() {
        use crate::testgpu::{TitleBuffer, TitleList};
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        let test = "draw::tests::buffer_views_of_slots_picked_at_run_time_are_made_and_read";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        // Views kept for slots picked at run time, as `XTP_RUN_TIME_VIEWS=1`
        // has them.
        crate::descriptor::host::RUN_TIME_VIEWS_TEST.set(Some(true));
        let host = gpu.host();
        let cs = picking_program();
        assert_eq!(cs.run_time_buffers, xtp_shader::emit::RUN_TIME_READS);
        // Root signature: the user data, the heaps indexed directly.
        let params = [D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: xtp_shader::emit::USER_DATA_SPACE,
                    Num32BitValues: cs.user_data_dwords(),
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        }];
        let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
            Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
            Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                    NumParameters: 1,
                    pParameters: params.as_ptr(),
                    NumStaticSamplers: 0,
                    pStaticSamplers: std::ptr::null(),
                    Flags: D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED,
                },
            },
        };
        // SAFETY: the descriptions point at live data; the blob is read
        // before it goes.
        let (root, pso) = unsafe {
            let mut blob: Option<ID3DBlob> = None;
            D3D12SerializeVersionedRootSignature(&desc, &mut blob, None).unwrap();
            let blob = blob.unwrap();
            let bytes = std::slice::from_raw_parts(
                blob.GetBufferPointer().cast::<u8>(),
                blob.GetBufferSize(),
            );
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes).unwrap();
            let pso: ID3D12PipelineState = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: cs.dxil.as_ptr().cast(),
                        BytecodeLength: cs.dxil.len(),
                    },
                    ..Default::default()
                })
                .unwrap();
            (root, pso)
        };
        // The title's buffers and the views it writes in its heap (made
        // null by the host: no draw named them).
        let heap = crate::descriptor::TestHeap::views(8);
        let title = crate::descriptor::heap(heap.0);
        assert!(title.host().is_some(), "a host heap");
        let (slots, out) = (TitleBuffer::new(1 << 12), TitleBuffer::new(1 << 12));
        let (a, b) = (TitleBuffer::new(1 << 12), TitleBuffer::new(1 << 12));
        slots.set_words(|i| [2, 3].get(i).copied().unwrap_or(0));
        a.set_words(|i| 100 + i as u32);
        b.set_words(|i| 200 + i as u32);
        // Format, buffer, (mapping,) first element, count, stride, flags.
        let srv = |format, count, stride, flags| {
            let mut w = [0u32; 10];
            (w[0], w[1], w[2], w[6], w[7], w[8]) = (format, 1, 0x1688, count, stride, flags);
            w
        };
        let mut uav = [0u32; 10];
        (uav[1], uav[4], uav[5]) = (1, 32, 32);
        let at = |i: u64| title.start + i * 32;
        let null = std::ptr::null();
        crate::descriptor::create_srv(null, slots.resource(), &srv(39, 1024, 0, 1), at(0));
        crate::descriptor::create_uav(null, out.resource(), null, &uav, at(1));
        crate::descriptor::create_srv(null, a.resource(), &srv(0, 64, 16, 0), at(2));
        crate::descriptor::create_srv(null, b.resource(), &srv(0, 64, 16, 0), at(3));
        // A constant buffer view waits for no pass: buffer views only.
        let cbv = [a.address() as u32, (a.address() >> 32) as u32, 256, 0];
        crate::descriptor::create_cbv(null, &cbv, at(4));
        assert_eq!(
            title.host().unwrap().waiting(),
            4,
            "the four buffer views wait"
        );
        let run = |pick: bool| -> Vec<u32> {
            out.set_words(|_| 0);
            let list = TitleList::new();
            let l = list.host();
            let mut state = state(heap.0);
            heaps(&mut state, l);
            if pick {
                state.picked_buffers(l, "test");
            }
            // The output is named by the program's table: declared written
            // as `readable` does for it.
            out.shadow().gpu_writes(l, out.address(), 1 << 12);
            l.keep(&root);
            l.keep(&pso);
            let zeros = vec![0u32; cs.user_data_dwords() as usize];
            // SAFETY: an open list with the heap and every root argument
            // set; what it names outlives the run.
            unsafe {
                l.list.SetComputeRootSignature(&root);
                l.list.SetPipelineState(&pso);
                l.list.SetComputeRoot32BitConstants(
                    0,
                    zeros.len() as u32,
                    zeros.as_ptr().cast(),
                    0,
                );
                l.list.Dispatch(2, 1, 1);
            }
            drop(state);
            list.run();
            out.words()[..32].to_vec()
        };
        assert_eq!(
            run(false),
            vec![0; 32],
            "null views: nothing read or written"
        );
        // Group g, lane l reads dword 4 l + 1 of its view and stores it at
        // dword 8 l + 4 g.
        let mut want = vec![0; 32];
        for l in 0..4 {
            want[8 * l] = 101 + 4 * l as u32;
            want[8 * l + 4] = 201 + 4 * l as u32;
        }
        // A pass makes waiting views for 8 ms and leaves the rest to the
        // next frames' passes (`picked_spans`): on a loaded machine one pass
        // may not make all four (xtp-farm-2 at 2a63867 made those of slots 0
        // to 2, and group 1 read zeros). So passes go on, as the first lists
        // of the frames that follow make them, until none waits: one view a
        // pass at least, four passes at most. Then the dispatch runs in the
        // first list of a frame of its own, which brings their memory up to
        // date.
        let views = title.host().unwrap();
        let mut passes = 0;
        while views.waiting() > 0 {
            passes += 1;
            assert!(passes <= 4, "each pass makes one waiting view at least");
            views.forget_picked_pass();
            run(true);
        }
        views.forget_picked_pass();
        assert_eq!(run(true), want);
        // Codex review of 54ab53d, P1: the frame's first list to sweep took
        // the frame's refresh for its own submission; one dropped unsubmitted
        // left the frame's other lists the data of before. Each list now
        // refreshes what it reads.
        a.set_words(|i| 300 + i as u32);
        b.set_words(|i| 400 + i as u32);
        {
            let list = TitleList::new();
            let mut first = state(heap.0);
            heaps(&mut first, list.host());
            first.picked_buffers(list.host(), "test");
        }
        let mut fresh = vec![0; 32];
        for l in 0..4 {
            fresh[8 * l] = 301 + 4 * l as u32;
            fresh[8 * l + 4] = 401 + 4 * l as u32;
        }
        assert_eq!(run(true), fresh, "the frame's second list");
        // Codex review of 54ab53d, P2: what shaders wrote through views of
        // the heap that no draw declared written went back to no title
        // memory. For a pipeline that writes through slots it picks, the
        // heap's unordered views are declared written by the sweep.
        let run_writing = |writes: bool| -> Vec<u32> {
            out.set_words(|_| 0);
            let list = TitleList::new();
            let l = list.host();
            let mut state = state(heap.0);
            heaps(&mut state, l);
            state.picking_writes = writes;
            state.picked_buffers(l, "test");
            l.keep(&root);
            l.keep(&pso);
            let zeros = vec![0u32; cs.user_data_dwords() as usize];
            // SAFETY: an open list with the heap and every root argument
            // set; what it names outlives the run.
            unsafe {
                l.list.SetComputeRootSignature(&root);
                l.list.SetPipelineState(&pso);
                l.list.SetComputeRoot32BitConstants(
                    0,
                    zeros.len() as u32,
                    zeros.as_ptr().cast(),
                    0,
                );
                l.list.Dispatch(2, 1, 1);
            }
            drop(state);
            list.run();
            out.words()[..32].to_vec()
        };
        assert_eq!(run_writing(false), vec![0; 32], "written nowhere");
        assert_eq!(run_writing(true), fresh, "declared written");
        crate::descriptor::host::RUN_TIME_VIEWS_TEST.set(None);
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Codex review of 244cf78, finding 6: a view the title wrote after a
    /// list's last draw or dispatch that picks slots at run time stayed
    /// null in that list's submission (the console reads descriptors when
    /// the GPU runs). The list makes it as it closes, and refreshes its
    /// bytes: the dispatch recorded before the write reads them.
    #[test]
    fn views_written_after_a_lists_last_pick_are_made_as_it_closes() {
        use crate::testgpu::{TitleBuffer, TitleList};
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        let test = "draw::tests::views_written_after_a_lists_last_pick_are_made_as_it_closes";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        crate::descriptor::host::RUN_TIME_VIEWS_TEST.set(Some(true));
        let host = gpu.host();
        let cs = picking_program();
        let params = [D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: xtp_shader::emit::USER_DATA_SPACE,
                    Num32BitValues: cs.user_data_dwords(),
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        }];
        let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
            Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
            Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                    NumParameters: 1,
                    pParameters: params.as_ptr(),
                    NumStaticSamplers: 0,
                    pStaticSamplers: std::ptr::null(),
                    Flags: D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED,
                },
            },
        };
        // SAFETY: the descriptions point at live data; the blob is read
        // before it goes.
        let (root, pso) = unsafe {
            let mut blob: Option<ID3DBlob> = None;
            D3D12SerializeVersionedRootSignature(&desc, &mut blob, None).unwrap();
            let blob = blob.unwrap();
            let bytes = std::slice::from_raw_parts(
                blob.GetBufferPointer().cast::<u8>(),
                blob.GetBufferSize(),
            );
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes).unwrap();
            let pso: ID3D12PipelineState = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: cs.dxil.as_ptr().cast(),
                        BytecodeLength: cs.dxil.len(),
                    },
                    ..Default::default()
                })
                .unwrap();
            (root, pso)
        };
        let heap = crate::descriptor::TestHeap::views(8);
        let title = crate::descriptor::heap(heap.0);
        assert!(title.host().is_some(), "a host heap");
        let (slots, out) = (TitleBuffer::new(1 << 12), TitleBuffer::new(1 << 12));
        let (a, b) = (TitleBuffer::new(1 << 12), TitleBuffer::new(1 << 12));
        slots.set_words(|i| [2, 3].get(i).copied().unwrap_or(0));
        a.set_words(|i| 100 + i as u32);
        b.set_words(|i| 200 + i as u32);
        // Format, buffer, (mapping,) first element, count, stride, flags.
        let srv = |format, count, stride, flags| {
            let mut w = [0u32; 10];
            (w[0], w[1], w[2], w[6], w[7], w[8]) = (format, 1, 0x1688, count, stride, flags);
            w
        };
        let mut uav = [0u32; 10];
        (uav[1], uav[4], uav[5]) = (1, 32, 32);
        let at = |i: u64| title.start + i * 32;
        let null = std::ptr::null();
        crate::descriptor::create_srv(null, slots.resource(), &srv(39, 1024, 0, 1), at(0));
        crate::descriptor::create_uav(null, out.resource(), null, &uav, at(1));
        crate::descriptor::create_srv(null, a.resource(), &srv(0, 64, 16, 0), at(2));
        out.set_words(|_| 0);
        let list = TitleList::new();
        let l = list.host();
        let mut state = state(heap.0);
        heaps(&mut state, l);
        // The list's only pick: slot 3 holds no view yet.
        state.picked_buffers(l, "test");
        out.shadow().gpu_writes(l, out.address(), 1 << 12);
        l.keep(&root);
        l.keep(&pso);
        let zeros = vec![0u32; cs.user_data_dwords() as usize];
        // SAFETY: an open list with the heap and every root argument set;
        // what it names outlives the run.
        unsafe {
            l.list.SetComputeRootSignature(&root);
            l.list.SetPipelineState(&pso);
            l.list
                .SetComputeRoot32BitConstants(0, zeros.len() as u32, zeros.as_ptr().cast(), 0);
            l.list.Dispatch(2, 1, 1);
        }
        drop(state);
        // The title writes slot 3 after it, before the list closes.
        crate::descriptor::create_srv(null, b.resource(), &srv(0, 64, 16, 0), at(3));
        list.run();
        // Group g, lane l reads dword 4 l + 1 of its view and stores it at
        // dword 8 l + 4 g.
        let mut want = [0u32; 32];
        for l in 0..4 {
            want[8 * l] = 101 + 4 * l as u32;
            want[8 * l + 4] = 201 + 4 * l as u32;
        }
        assert_eq!(out.words()[..32], want, "slot 3 made and read");
        crate::descriptor::host::RUN_TIME_VIEWS_TEST.set(None);
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// The root signature (user data, heaps indexed directly) and pipeline
    /// of `picking_program`'s shader `cs`.
    fn picking_pipeline(
        host: &Host,
        cs: &xtp_shader::emit::HostShader,
    ) -> (ID3D12RootSignature, ID3D12PipelineState) {
        let params = [D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: xtp_shader::emit::USER_DATA_SPACE,
                    Num32BitValues: cs.user_data_dwords(),
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        }];
        let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
            Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
            Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                    NumParameters: 1,
                    pParameters: params.as_ptr(),
                    NumStaticSamplers: 0,
                    pStaticSamplers: std::ptr::null(),
                    Flags: D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED,
                },
            },
        };
        // SAFETY: the descriptions point at live data; the blob is read
        // before it goes.
        unsafe {
            let mut blob: Option<ID3DBlob> = None;
            D3D12SerializeVersionedRootSignature(&desc, &mut blob, None).unwrap();
            let blob = blob.unwrap();
            let bytes = std::slice::from_raw_parts(
                blob.GetBufferPointer().cast::<u8>(),
                blob.GetBufferSize(),
            );
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes).unwrap();
            let pso: ID3D12PipelineState = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: cs.dxil.as_ptr().cast(),
                        BytecodeLength: cs.dxil.len(),
                    },
                    ..Default::default()
                })
                .unwrap();
            (root, pso)
        }
    }

    /// Codex review of a090f22, point 6: a list recorded a dispatch that
    /// picks slot 3 and closed; the title then wrote a buffer view there
    /// and submitted the list with nothing else sweeping. The view waited
    /// null and the dispatch read zeros. The list sweeps as it is
    /// submitted: the view is made and its bytes refreshed for it.
    #[test]
    fn views_written_between_a_lists_close_and_its_submission_are_made_for_it() {
        let test =
            "draw::tests::views_written_between_a_lists_close_and_its_submission_are_made_for_it";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        crate::descriptor::host::RUN_TIME_VIEWS_TEST.set(Some(true));
        let host = gpu.host();
        let cs = picking_program();
        let (root, pso) = picking_pipeline(host, &cs);
        let heap = TestHeap::views(8);
        let title = crate::descriptor::heap(heap.0);
        assert!(title.host().is_some(), "a host heap");
        let (slots, out) = (TitleBuffer::new(1 << 12), TitleBuffer::new(1 << 12));
        let (a, b) = (TitleBuffer::new(1 << 12), TitleBuffer::new(1 << 12));
        slots.set_words(|i| [2, 3].get(i).copied().unwrap_or(0));
        a.set_words(|i| 100 + i as u32);
        b.set_words(|i| 200 + i as u32);
        // Format, buffer, (mapping,) first element, count, stride, flags.
        let srv = |format, count, stride, flags| {
            let mut w = [0u32; 10];
            (w[0], w[1], w[2], w[6], w[7], w[8]) = (format, 1, 0x1688, count, stride, flags);
            w
        };
        let mut uav = [0u32; 10];
        (uav[1], uav[4], uav[5]) = (1, 32, 32);
        let at = |i: u64| title.start + i * 32;
        let null = std::ptr::null();
        crate::descriptor::create_srv(null, slots.resource(), &srv(39, 1024, 0, 1), at(0));
        crate::descriptor::create_uav(null, out.resource(), null, &uav, at(1));
        crate::descriptor::create_srv(null, a.resource(), &srv(0, 64, 16, 0), at(2));
        out.set_words(|_| 0);
        let list = TitleList::new();
        let l = list.host();
        let mut state = state(heap.0);
        heaps(&mut state, l);
        // The list's only pick: slot 3 holds no view yet.
        state.picked_buffers(l, "test");
        out.shadow().gpu_writes(l, out.address(), 1 << 12);
        l.keep(&root);
        l.keep(&pso);
        let zeros = vec![0u32; cs.user_data_dwords() as usize];
        // SAFETY: an open list with the heap and every root argument set;
        // what it names outlives the run.
        unsafe {
            l.list.SetComputeRootSignature(&root);
            l.list.SetPipelineState(&pso);
            l.list
                .SetComputeRoot32BitConstants(0, zeros.len() as u32, zeros.as_ptr().cast(), 0);
            l.list.Dispatch(2, 1, 1);
        }
        drop(state);
        // The title's close, then its view creation at slot 3, then its
        // submission (as `TitleList::run`, without the close).
        let close = crate::object::slot::<extern "system" fn(This) -> crate::object::HRESULT>;
        assert_eq!(close(list.this(), 0x48)(list.this()), crate::object::S_OK);
        crate::descriptor::create_srv(null, b.resource(), &srv(0, 64, 16, 0), at(3));
        assert_eq!(title.host().unwrap().waiting(), 1, "slot 3 waits, null");
        host.submit(&[l]);
        host.drain();
        assert_eq!(title.host().unwrap().waiting(), 0, "made at the submission");
        // Group g, lane l reads dword 4 l + 1 of its view and stores it at
        // dword 8 l + 4 g.
        let mut want = [0u32; 32];
        for l in 0..4 {
            want[8 * l] = 101 + 4 * l as u32;
            want[8 * l + 4] = 201 + 4 * l as u32;
        }
        assert_eq!(out.words()[..32], want, "slot 3 made and read");
        // Submitted again, the list reads what the title wrote since (what
        // the dispatch writes as before goes back to no title memory: the
        // output is left as it is).
        b.set_words(|i| 300 + i as u32);
        host.submit(&[l]);
        host.drain();
        for l in 0..4 {
            want[8 * l + 4] = 301 + 4 * l as u32;
        }
        assert_eq!(out.words()[..32], want, "refreshed at each submission");
        crate::descriptor::host::RUN_TIME_VIEWS_TEST.set(None);
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// A compute program of 16 lanes that picks a texture slot by an index
    /// it reads from the buffer in table slot 0 (as the world's materials
    /// do), loads texel (lane, 0) of the 2D texture there and stores it at
    /// element `lane` of the structured view of 16 bytes in table slot 1;
    /// and its table's entries, as the console program lists them (the
    /// texture's slot is in none).
    fn texture_picking_program() -> (xtp_shader::emit::HostShader, Vec<UsedResource>) {
        use xtp_shader::native::{ComputeIo, StageIo};
        use xtp_shader::{HwStage, NativeProgram, RootParam};
        let words: &[u32] = &[
            0xBE94_0304, // s_mov_b32 s20, s4
            0xBE95_0380, // s_mov_b32 s21, 0
            0xF408_040A,
            0xFA00_0000, // s_load_dwordx4 s[16:19], s[20:21], 0x0
            0xF408_060A,
            0xFA00_0020, // s_load_dwordx4 s[24:27], s[20:21], 0x20
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xF420_0188,
            0xFA00_0000, // s_buffer_load_dword s6, s[16:19], 0x0
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0x8F06_8506, // s_lshl_b32 s6, s6, 5
            0xF40C_020A,
            0x0C00_0000, // s_load_dwordx8 s[8:15], s[20:21], s6
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0x7E04_0300, // v_mov_b32 v2, v0
            0x7E06_0280, // v_mov_b32 v3, 0
            0xF000_0F08,
            0x0002_1402, // image_load v[20:23], v[2:3], s[8:15] dmask:0xf dim:2D
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
            0x7E02_0280, // v_mov_b32 v1, 0
            0xE078_3000,
            0x8006_1400, // buffer_store_dwordx4 v[20:23], v[0:1], s[24:27], 0 idxen offen
            0xBF81_0000, // s_endpgm
        ];
        let entry = |kind, table_offset_dwords, stride| UsedResource {
            kind,
            root: RootType::Table,
            table_offset_dwords,
            lower_bound: 0,
            stride,
            dwords: 4,
            user_data: 4,
            slot: 1,
        };
        let param = |root, sgprs, user_data, slot| RootParam {
            root,
            table_offset_dwords: 0,
            sgprs,
            base_register: 0,
            user_data,
            slot,
        };
        let used = vec![
            entry(ResourceType::Srv, 0, 0),
            entry(ResourceType::Uav, 8, 16),
        ];
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            float_mode: 0,
            vgprs: 32,
            layout: vec![
                param(RootType::Constants, 4, 0, 0),
                param(RootType::Table, 1, 4, 1),
            ],
            used: used.clone(),
            io: StageIo::Cs(ComputeIo {
                rsrc2: 5 << 1,
                threads: [16, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = xtp_shader::ir::build(&native.code, native.wave64, native.vgprs).unwrap();
        let st = xtp_shader::structure::structure(&p).unwrap();
        let res = xtp_shader::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = xtp_shader::io::raise(&p, &native).unwrap();
        // An empty container: the tag, a digest, version 1, its size.
        let mut container = b"DXBC".to_vec();
        container.extend([0; 16]);
        container.extend([1u32, 32, 0].into_iter().flat_map(u32::to_le_bytes));
        let c = xtp_shader::Container::parse(&container).unwrap();
        let shader = xtp_shader::emit::emit(&c, &native, &p, &st, &io, &res).unwrap();
        (shader, used)
    }

    /// A host compute pipeline of `cs` whose root signature holds its user
    /// data and indexes the heaps directly.
    fn compute_pipeline(
        host: &crate::host::Host,
        cs: &xtp_shader::emit::HostShader,
    ) -> (ID3D12RootSignature, ID3D12PipelineState) {
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        let params = [D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: xtp_shader::emit::USER_DATA_SPACE,
                    Num32BitValues: cs.user_data_dwords(),
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        }];
        let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
            Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
            Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                    NumParameters: 1,
                    pParameters: params.as_ptr(),
                    NumStaticSamplers: 0,
                    pStaticSamplers: std::ptr::null(),
                    Flags: D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED,
                },
            },
        };
        // SAFETY: the descriptions point at live data; the blob is read
        // before it goes.
        unsafe {
            let mut blob: Option<ID3DBlob> = None;
            D3D12SerializeVersionedRootSignature(&desc, &mut blob, None).unwrap();
            let blob = blob.unwrap();
            let bytes = std::slice::from_raw_parts(
                blob.GetBufferPointer().cast::<u8>(),
                blob.GetBufferSize(),
            );
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes).unwrap();
            let pso: ID3D12PipelineState = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: cs.dxil.as_ptr().cast(),
                        BytecodeLength: cs.dxil.len(),
                    },
                    ..Default::default()
                })
                .unwrap();
            (root, pso)
        }
    }

    /// Textures the title wrote into slots a shader picks when it runs (fork
    /// pick 6, texture side): no draw's table names them, so a placed
    /// texture's slot stays null and a texture object's host copy is never
    /// uploaded: the program reads zeros, or what the host copy was made
    /// with. Once `heap_textures` has made and uploaded them, it reads the
    /// title's texels; and when the title writes others, the next frame's
    /// pass finds the change and the program reads the latest.
    #[test]
    fn textures_of_slots_picked_at_run_time_are_made_and_kept_up_to_date() {
        use crate::resource::tests_support::{committed, sampled, target};
        use crate::testgpu::{TitleBuffer, TitleList};
        let test = "draw::tests::textures_of_slots_picked_at_run_time_are_made_and_kept_up_to_date";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        if crate::descriptor::host::heap_textures_budget().is_none() {
            return;
        }
        let (cs, used) = texture_picking_program();
        assert_eq!(cs.run_time_buffers, 0, "no buffer is picked at run time");
        let (root, pso) = compute_pipeline(host, &cs);
        let heap = crate::descriptor::TestHeap::views(8);
        let title = crate::descriptor::heap(heap.0);
        let views = title.host().expect("a host heap");
        let at = |i: u64| title.start + i * 32;
        let null = std::ptr::null();
        let (slots, out) = (TitleBuffer::new(1 << 12), TitleBuffer::new(1 << 12));
        // The index: a raw view (format, buffer, mapping, then from word 4
        // the first element, the count and the flags).
        let mut srv = [0u32; 10];
        (srv[0], srv[1], srv[2], srv[6], srv[8]) = (39, 1, 0x1688, 1024, 1);
        crate::descriptor::create_srv(null, slots.resource(), &srv, at(0));
        // The output: 16 elements of 16 bytes.
        let mut uav = [0u32; 10];
        (uav[1], uav[4], uav[5]) = (1, 16, 16);
        crate::descriptor::create_uav(null, out.resource(), null, &uav, at(1));
        // A texture object the title fills, in slot 2: format, 2D, the
        // default mapping, then (from word 4) the first mip and the count.
        let object = committed(&sampled(16, 16, 28));
        let started = std::time::Instant::now();
        while crate::resource::host_resource(object).is_none() {
            assert!(
                started.elapsed().as_secs() < 5,
                "the host copy never became resident"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let (memory, desc) = crate::resource::texture_memory(object).unwrap();
        let size = crate::resource::texture_layout(&desc).unwrap().size as usize;
        // SAFETY: the texture's own title memory, `size` bytes.
        let fill_object = |byte: u8| unsafe { memory.cast_mut().write_bytes(byte, size) };
        let texture_2d = [28, 4, 0x1688, 0, 0, 1, 0, 0, 0, 0];
        crate::descriptor::create_srv(null, object, &texture_2d, at(2));
        // A texture placed in title memory, write-watched as the runtime
        // reserves it, in slot 3 (freed only once nothing is placed in it
        // any more: a failing assertion leaks it).
        const KIB_64: u64 = 1 << 16;
        const MEM_WRITE_WATCH: u32 = 0x0020_0000;
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        // SAFETY: a fresh reservation of our own.
        let address = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                KIB_64 as usize,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as u64;
        assert_ne!(address, 0, "write-watched memory");
        let drawn = target(16, 16, 28);
        // SAFETY: a description is 56 bytes, 14 words.
        let mut placed: [u32; 14] = unsafe { std::mem::transmute_copy(&drawn) };
        placed[12] = 0; // not a render target: the title fills it
        let mut view = [0u32; 16];
        view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        (view[10], view[11]) = (address as u32, (address >> 32) as u32);
        crate::descriptor::create_placed_srv(null, &placed, &view, at(3));
        // SAFETY: the 64 KiB the texture is placed in.
        let fill_placed =
            |byte: u8| unsafe { (address as *mut u8).write_bytes(byte, KIB_64 as usize) };
        assert_eq!(views.texture_queue().waiting_slots(), [2, 3]);

        // The texel the 16 lanes read, each channel of it; `None` if they
        // differ.
        let texel = |words: &[u32]| -> Option<u8> {
            let first = f32::from_bits(words[0]);
            words
                .iter()
                .all(|&w| w == words[0])
                .then(|| (first * 255.0).round() as u8)
        };
        // One dispatch picking `slot`, after a pass if `pass`.
        let run = |slot: u32, pass: bool| -> Vec<u32> {
            slots.set_words(|i| if i == 0 { slot } else { 0 });
            out.set_words(|_| 0);
            let list = TitleList::new();
            let l = list.host();
            let mut state = state(heap.0);
            heaps(&mut state, l);
            // The program's table, read as a draw reads it.
            state.set_table(true, 1, title.start);
            state.tables(l, &used, true).unwrap();
            if pass {
                state.heap_textures(l);
            }
            l.keep(&root);
            l.keep(&pso);
            let zeros = vec![0u32; cs.user_data_dwords() as usize];
            // SAFETY: an open list with the heap and every root argument
            // set; what it names outlives the run.
            unsafe {
                l.list.SetComputeRootSignature(&root);
                l.list.SetPipelineState(&pso);
                l.list.SetComputeRoot32BitConstants(
                    0,
                    zeros.len() as u32,
                    zeros.as_ptr().cast(),
                    0,
                );
                l.list.Dispatch(1, 1, 1);
            }
            drop(state);
            list.run();
            out.words()[..64].to_vec()
        };
        // A pass in a frame of its own.
        let next_frame = || {
            crate::descriptor::host::PASS_LATER.fetch_add(1, Ordering::Relaxed);
        };

        fill_object(0x11);
        fill_placed(0x55);
        assert_eq!(texel(&run(3, false)), Some(0), "a null slot");
        assert_ne!(texel(&run(2, false)), Some(0x11), "never uploaded");
        // The placed texture's host copy may take passes to be resident.
        let mut passes = 0;
        while !views.texture_queue().waiting_slots().is_empty() {
            passes += 1;
            assert!(passes < 64, "still waiting after {passes} passes");
            next_frame();
            run(2, true);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(views.texture_queue().served_slots(), [2, 3]);
        assert_eq!(texel(&run(2, false)), Some(0x11), "uploaded");
        assert_eq!(texel(&run(3, false)), Some(0x55), "made and uploaded");

        // The title writes other texels: the next frames' passes check the
        // served textures and upload the changed ones. A pass checks them in
        // turns within `PASS_CHECKING` (1 ms), one at least: on a loaded
        // machine it may check one of the two, and the other waits for the
        // next pass (watch30 at c83e996: slot 3 still read 0x55 once). So
        // passes go on, a frame each, until both read the latest: two
        // passes check both.
        fill_object(0x22);
        fill_placed(0x66);
        let mut passes = 0;
        loop {
            passes += 1;
            next_frame();
            let read = (texel(&run(2, true)), texel(&run(3, false)));
            if read == (Some(0x22), Some(0x66)) {
                break;
            }
            assert!(
                passes < 4,
                "the latest bytes: {read:?} after {passes} passes"
            );
        }
        // Nothing changed since: the write-watched placed texture is found
        // clean and not uploaded; the texture object's memory (the test's
        // own, neither write-watched nor guarded) only the check at
        // submission's hash can tell, so it is uploaded for that. Passes go
        // on, as above, until both were checked.
        let (mut checked, mut changed, mut hashed) = (0, 0, 0);
        let mut passes = 0;
        while checked < 2 {
            passes += 1;
            assert!(passes < 4, "{checked} checked after {passes} passes");
            next_frame();
            let list = TitleList::new();
            let mut state = state(heap.0);
            heaps(&mut state, list.host());
            let counts = state.heap_textures(list.host()).expect("a pass");
            checked += counts.checked;
            changed += counts.changed;
            hashed += counts.hashed;
            drop(state);
            list.run();
        }
        assert_eq!(changed, 0, "the placed texture found clean");
        assert!(hashed >= 1, "the texture object hashed at submission");
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());

        type Release = extern "system" fn(This) -> u32;
        crate::object::slot::<Release>(object, 0x10)(object);
        drop(heap);
        crate::resource::memory_remapped(address as usize, KIB_64 as usize);
        // SAFETY: the reservation made above; nothing is placed in it now.
        unsafe { VirtualFree(address as *mut _, 0, MEM_RELEASE) };
    }

    /// A view of a texture the title fills, which the title copies into its
    /// placed heap as bytes (no call sees it) and a shader picks by index:
    /// its host slot is never made and no pass serves it, so the program
    /// reads zeros. Once the look before a submission has mirrored it
    /// (`descriptor::sync_heaps`, ported from Marvisak/VIctory (86a5ed6) by
    /// Marvisak), the slot is made and the pass over textures picked by
    /// index uploads it: the program reads the title's texels.
    #[test]
    fn textures_copied_into_the_heap_as_bytes_are_read_by_shaders_picking_them() {
        use crate::resource::tests_support::{committed, sampled};
        use crate::testgpu::{TitleBuffer, TitleList};
        let test =
            "draw::tests::textures_copied_into_the_heap_as_bytes_are_read_by_shaders_picking_them";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        if crate::descriptor::host::heap_textures_budget().is_none() {
            return;
        }
        let (cs, used) = texture_picking_program();
        let (root, pso) = compute_pipeline(host, &cs);
        let heap = crate::descriptor::PlacedTestHeap::views(8);
        let title = crate::descriptor::heap(heap.this);
        let at = |i: u64| title.start + i * 32;
        let null = std::ptr::null();
        let (slots, out) = (TitleBuffer::new(1 << 12), TitleBuffer::new(1 << 12));
        // The index and the output, as in the test above, through calls.
        let mut srv = [0u32; 10];
        (srv[0], srv[1], srv[2], srv[6], srv[8]) = (39, 1, 0x1688, 1024, 1);
        crate::descriptor::create_srv(null, slots.resource(), &srv, at(0));
        let mut uav = [0u32; 10];
        (uav[1], uav[4], uav[5]) = (1, 16, 16);
        crate::descriptor::create_uav(null, out.resource(), null, &uav, at(1));
        let views = title.host().expect("a host heap");
        // A texture object the title fills: its view made where no heap
        // sees it, then its 32 bytes copied into slot 5.
        let object = committed(&sampled(16, 16, 28));
        let started = std::time::Instant::now();
        while crate::resource::host_resource(object).is_none() {
            assert!(started.elapsed().as_secs() < 5, "never resident");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let (memory, desc) = crate::resource::texture_memory(object).unwrap();
        let size = crate::resource::texture_layout(&desc).unwrap().size as usize;
        // SAFETY: the texture's own title memory, `size` bytes.
        unsafe { memory.cast_mut().write_bytes(0x11, size) };
        let record = [0u64; 4];
        let texture_2d = [28, 4, 0x1688, 0, 0, 1, 0, 0, 0, 0];
        crate::descriptor::create_srv(null, object, &texture_2d, record.as_ptr() as u64);
        // SAFETY: slot 5 of the heap, in memory it holds.
        unsafe {
            std::ptr::copy_nonoverlapping(record.as_ptr().cast::<u8>(), at(5) as *mut u8, 32);
        }
        let texel = |words: &[u32]| -> Option<u8> {
            let first = f32::from_bits(words[0]);
            words
                .iter()
                .all(|&w| w == words[0])
                .then(|| (first * 255.0).round() as u8)
        };
        // One dispatch picking `slot`, after a pass if `pass`.
        let run = |slot: u32, pass: bool| -> Vec<u32> {
            slots.set_words(|i| if i == 0 { slot } else { 0 });
            out.set_words(|_| 0);
            let list = TitleList::new();
            let l = list.host();
            let mut state = state(heap.this);
            heaps(&mut state, l);
            state.set_table(true, 1, title.start);
            state.tables(l, &used, true).unwrap();
            if pass {
                state.heap_textures(l);
            }
            l.keep(&root);
            l.keep(&pso);
            let zeros = vec![0u32; cs.user_data_dwords() as usize];
            // SAFETY: an open list with the heap and every root argument
            // set; what it names outlives the run.
            unsafe {
                l.list.SetComputeRootSignature(&root);
                l.list.SetPipelineState(&pso);
                l.list.SetComputeRoot32BitConstants(
                    0,
                    zeros.len() as u32,
                    zeros.as_ptr().cast(),
                    0,
                );
                l.list.Dispatch(1, 1, 1);
            }
            drop(state);
            list.run();
            out.words()[..64].to_vec()
        };
        let next_frame = || {
            crate::descriptor::host::PASS_LATER.fetch_add(1, Ordering::Relaxed);
        };
        // Not mirrored: null, and no pass serves it.
        for _ in 0..2 {
            next_frame();
            assert_eq!(texel(&run(5, true)), Some(0), "a slot no call wrote");
        }
        assert_eq!(views.texture_queue().waiting_slots(), [] as [u32; 0]);
        // Mirrored by the look: made, and waiting for its first upload.
        let found = heap.look();
        assert_eq!(
            (found.copied, found.textures, found.tabled),
            (1, 1, 0),
            "{found:?}"
        );
        assert_eq!(views.texture_queue().waiting_slots(), [5]);
        // The host copy may take passes to be resident.
        let mut passes = 0;
        while !views.texture_queue().waiting_slots().is_empty() {
            passes += 1;
            assert!(passes < 64, "still waiting after {passes} passes");
            next_frame();
            run(5, true);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(texel(&run(5, false)), Some(0x11), "the title's texels");
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
        type Release = extern "system" fn(This) -> u32;
        crate::object::slot::<Release>(object, 0x10)(object);
        drop(heap);
    }

    /// What passes over the textures shaders pick by index cost the
    /// recording thread, out of the game: 2,048 textures placed in
    /// write-watched title memory (64x64 RGBA8, one per 64 KiB) and their
    /// views in a heap; the passes that make and upload them all for the
    /// first time (as many as their budgets take, each list then run), then
    /// passes over the served ones with none changed: per pass, how many it
    /// checked in its time and what a check costs. Run with
    /// `cargo test --release -p xtp-gpu -- --ignored --nocapture heap_texture_costs`.
    #[test]
    #[ignore = "benchmark"]
    fn heap_texture_costs() {
        use crate::resource::tests_support::target;
        use crate::testgpu::TitleList;
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const MEM_WRITE_WATCH: u32 = 0x0020_0000;
        const N: u32 = 2048;
        const STEP: u64 = 1 << 16;
        let test = "draw::tests::heap_texture_costs";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        if crate::descriptor::host::heap_textures_budget().is_none() {
            return;
        }
        // SAFETY: a fresh reservation of our own.
        let base = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                (u64::from(N) * STEP) as usize,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as u64;
        assert_ne!(base, 0);
        // SAFETY: the reservation, all of it.
        unsafe { (base as *mut u8).write_bytes(0x33, (u64::from(N) * STEP) as usize) };
        let heap = crate::descriptor::TestHeap::views(N);
        let title = crate::descriptor::heap(heap.0);
        let views = title.host().expect("a host heap");
        let drawn = target(64, 64, 28);
        // SAFETY: a description is 56 bytes, 14 words.
        let mut placed: [u32; 14] = unsafe { std::mem::transmute_copy(&drawn) };
        placed[12] = 0;
        for i in 0..N {
            let address = base + u64::from(i) * STEP;
            let mut view = [0u32; 16];
            view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
            (view[10], view[11]) = (address as u32, (address >> 32) as u32);
            let at = title.start + u64::from(i) * 32;
            crate::descriptor::create_placed_srv(std::ptr::null(), &placed, &view, at);
        }
        // One pass a frame, each in a list of its own, run before the next.
        let pass = || {
            crate::descriptor::host::PASS_LATER.fetch_add(1, Ordering::Relaxed);
            let list = TitleList::new();
            let mut state = state(heap.0);
            heaps(&mut state, list.host());
            let started = std::time::Instant::now();
            let counts = state.heap_textures(list.host()).expect("a pass");
            let recorded = started.elapsed();
            drop(state);
            let run = std::time::Instant::now();
            list.run();
            (counts, recorded, run.elapsed())
        };
        // The host textures made first, as a draw's tables would: what is
        // left to the passes is the staging buffers and their copies.
        let making = std::time::Instant::now();
        for i in 0..N {
            views.mirror(i, title.start + u64::from(i) * 32, true);
        }
        let made = making.elapsed().as_secs_f64() * 1e6 / f64::from(N);
        let (mut passes, mut recorded, mut ran, mut uploaded) = (0, 0.0, 0.0, 0);
        while !views.texture_queue().waiting_slots().is_empty() {
            let (counts, r, s) = pass();
            passes += 1;
            uploaded += counts.uploaded;
            recorded += r.as_secs_f64() * 1e3;
            ran += s.as_secs_f64() * 1e3;
            assert!(passes < 1000, "still waiting");
        }
        eprintln!(
            "first uploads of {N} placed textures (16 KiB each): their host textures made in \
             {made:.1} us each; {passes} passes, {uploaded} uploaded, {recorded:.1} ms recording \
             ({:.1} us a texture), {ran:.1} ms submitting and running ({:.1} us a texture)",
            recorded * 1e3 / f64::from(N),
            ran * 1e3 / f64::from(N)
        );
        let mut best = f64::INFINITY;
        for _ in 0..8 {
            let (counts, r, _) = pass();
            assert_eq!(counts.changed + counts.hashed, 0, "{counts:?}");
            best = best.min(r.as_secs_f64() * 1e6 / counts.checked.max(1) as f64);
            eprintln!(
                "a pass over the served ones, none changed: {} checked in {:.2} ms",
                counts.checked,
                r.as_secs_f64() * 1e3
            );
        }
        eprintln!("a served texture checked: {best:.2} us at best (a lap: {N} of them)");
        drop(heap);
        crate::resource::memory_remapped(base as usize, (u64::from(N) * STEP) as usize);
        // SAFETY: the reservation made above; nothing is placed in it now.
        unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
    }

    thread_local! {
        pub(super) static UNTIMED: std::cell::Cell<bool> =
            const { std::cell::Cell::new(false) };
    }

    /// Disables pass deadlines on this thread, restoring the previous setting.
    struct Untimed(bool);

    impl Untimed {
        fn new() -> Untimed {
            Untimed(UNTIMED.replace(true))
        }
    }

    impl Drop for Untimed {
        fn drop(&mut self) {
            UNTIMED.set(self.0);
        }
    }

    #[test]
    fn passes_are_timed_again_after_the_guard_drops() {
        let started = std::time::Instant::now()
            .checked_sub(std::time::Duration::from_secs(1))
            .unwrap();
        let limit = std::time::Duration::ZERO;
        assert!(pass_time_spent(started, limit));
        {
            let _outer = Untimed::new();
            assert!(!pass_time_spent(started, limit));
            {
                let _inner = Untimed::new();
                assert!(!pass_time_spent(started, limit));
            }
            assert!(!pass_time_spent(started, limit));
        }
        assert!(pass_time_spent(started, limit));
    }

    /// A `width` x `height` RGBA8 texture object the title fills with
    /// `byte`, once its host copy is resident, and a shader view of it in
    /// slot `slot` of the title heap `heap`.
    fn filled_texture(heap: This, slot: u64, width: u64, height: u32, byte: u8) -> This {
        use crate::resource::tests_support::{committed, sampled};
        let object = committed(&sampled(width, height, 28));
        let started = std::time::Instant::now();
        while crate::resource::host_resource(object).is_none() {
            assert!(started.elapsed().as_secs() < 5, "never resident");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        fill_texture(object, byte);
        // Format, 2D, the default mapping, then (from word 4) the first mip
        // and the count.
        let texture_2d = [28, 4, 0x1688, 0, 0, 1, 0, 0, 0, 0];
        let at = crate::descriptor::heap(heap).start + slot * 32;
        crate::descriptor::create_srv(std::ptr::null(), object, &texture_2d, at);
        object
    }

    /// Writes `byte` over all of `object`'s title memory, as the title does.
    fn fill_texture(object: This, byte: u8) {
        let (memory, desc) = crate::resource::texture_memory(object).unwrap();
        let size = crate::resource::texture_layout(&desc).unwrap().size as usize;
        // SAFETY: the texture's own title memory, `size` bytes.
        unsafe { memory.cast_mut().write_bytes(byte, size) };
    }

    /// The texels of the 4x4 RGBA8 host copy of `object`, row by row.
    fn texels_4x4(host: &crate::host::Host, object: This) -> Vec<u32> {
        let copy = crate::resource::host_copy(object).expect("a host copy");
        let state = copy.state.load(Ordering::Relaxed);
        let read = crate::resource::tests_support::read_back(host, &copy.texture, state);
        (0..16)
            .map(|i| {
                let at = (i / 4) * 256 + (i % 4) * 4;
                u32::from_le_bytes(read[at..at + 4].try_into().unwrap())
            })
            .collect()
    }

    /// A pass over the textures shaders pick from the title heap `heap` by
    /// index, in a frame of its own, recorded in `list`: what it did.
    fn pass_in(
        heap: This,
        list: &crate::testgpu::TitleList,
    ) -> crate::descriptor::host::PassCounts {
        crate::descriptor::host::PASS_LATER.fetch_add(1, Ordering::Relaxed);
        let mut state = state(heap);
        heaps(&mut state, list.host());
        state.heap_textures(list.host()).expect("a pass")
    }

    /// Records in `list`, as the title does, the copy of a 2x2 block of
    /// `texel`s from a title buffer (returned, to keep until the list ran)
    /// to texel (0, 0) of subresource 0 of `dst`.
    fn copy_block(
        list: &crate::testgpu::TitleList,
        dst: This,
        texel: u32,
    ) -> crate::testgpu::TitleBuffer {
        /// A copy location as a title passes one: a resource, the kind (0 a
        /// subresource index, 1 a placed footprint), the index or footprint.
        #[repr(C)]
        struct Location {
            resource: This,
            kind: u32,
            offset: u64,
            format: u32,
            width: u32,
            height: u32,
            depth: u32,
            row_pitch: u32,
        }
        let block = crate::testgpu::TitleBuffer::new(1 << 12);
        block.set_words(|i| if i % 64 < 2 && i < 128 { texel } else { 0 });
        let to = Location {
            resource: dst,
            kind: 0,
            offset: 0,
            format: 0,
            width: 0,
            height: 0,
            depth: 0,
            row_pitch: 0,
        };
        let from = Location {
            resource: block.resource(),
            kind: 1,
            offset: 0,
            format: 28,
            width: 2,
            height: 2,
            depth: 1,
            row_pitch: 256,
        };
        type Copy = extern "system" fn(
            This,
            *const Location,
            u32,
            u32,
            u32,
            *const Location,
            *const [u32; 6],
        );
        let this = list.this();
        crate::object::slot::<Copy>(this, 0x80)(this, &to, 0, 0, 0, &from, std::ptr::null());
        block
    }

    /// Releases the title's reference to `resource`.
    fn release(resource: This) {
        type Release = extern "system" fn(This) -> u32;
        crate::object::slot::<Release>(resource, 0x10)(resource);
    }

    /// A pass's first upload of a texture, recorded in a list that never
    /// asked for it, and a copy into part of it recorded after, in another
    /// list: whichever list is submitted first, the texture ends as the
    /// title filled it with the copied block over it, as on the console.
    /// Before, with the copy's list submitted first, the pass's upload ran
    /// after it and rewrote the block (Codex's review of heaptex, P1 1).
    #[test]
    fn a_pass_upload_submitted_after_a_copy_keeps_the_copy() {
        use crate::testgpu::TitleList;
        let test = "draw::tests::a_pass_upload_submitted_after_a_copy_keeps_the_copy";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        if crate::descriptor::host::heap_textures_budget().is_none()
            || !crate::texture::predication()
        {
            return;
        }
        let (background, block) = (0x1111_1111, 0xaaaa_aaaa);
        let mut want = vec![background; 16];
        for i in [0, 1, 4, 5] {
            want[i] = block;
        }
        for copy_first in [true, false] {
            let heap = crate::descriptor::TestHeap::views(4);
            let object = filled_texture(heap.0, 2, 4, 4, 0x11);
            let (pass, copy) = (TitleList::new(), TitleList::new());
            let counts = pass_in(heap.0, &pass);
            assert_eq!(counts.uploaded, 1, "{counts:?}");
            let _block = copy_block(&copy, object, block);
            let (first, second) = if copy_first {
                (&copy, &pass)
            } else {
                (&pass, &copy)
            };
            first.run();
            second.run();
            assert_eq!(
                texels_4x4(host, object),
                want,
                "the copy submitted first: {copy_first}"
            );
            drop(heap);
            release(object);
        }
        assert_eq!(host.log_messages(), None);
    }

    /// A served texture that never got a staging buffer (none could be
    /// made when a draw read it through a table) is due at the pass's
    /// check, and uploaded then. Before, the check found it "clean" for good
    /// (Codex's review of heaptex, P1 3).
    #[test]
    fn a_served_texture_never_staged_is_uploaded_at_its_check() {
        use crate::testgpu::TitleList;
        let test = "draw::tests::a_served_texture_never_staged_is_uploaded_at_its_check";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        if crate::descriptor::host::heap_textures_budget().is_none() {
            return;
        }
        let heap = crate::descriptor::TestHeap::views(4);
        let views = crate::descriptor::heap(heap.0).host().expect("a host heap");
        let object = filled_texture(heap.0, 2, 4, 4, 0x33);
        // A draw read it through a table, its upload not recorded.
        views.texture_slot_tabled(2);
        assert_eq!(views.texture_queue().served_slots(), [2]);
        // Frames on, no table reads it: the pass checks it.
        crate::descriptor::host::PASS_LATER.fetch_add(2, Ordering::Relaxed);
        let list = TitleList::new();
        let counts = pass_in(heap.0, &list);
        assert_eq!((counts.checked, counts.changed), (1, 1), "{counts:?}");
        list.run();
        assert_eq!(texels_4x4(host, object), vec![0x3333_3333; 16]);
        assert_eq!(host.log_messages(), None);
        drop(heap);
        release(object);
    }

    /// Served textures whose memory only a hash tells (the test's own
    /// memory: neither write-watched nor guarded) take turns within the
    /// pass's hashing budget: the one it left out is the first the next
    /// pass looks at. Before, the checks went on past it, and a texture
    /// after one as big as the budget was never uploaded again (Codex's
    /// review of heaptex). Disable deadlines: staging the 4 MiB texture can
    /// spend the 8 ms upload limit before the small one is reached (#54).
    #[test]
    fn textures_the_hashing_budget_left_out_come_first_next() {
        use crate::testgpu::TitleList;
        let test = "draw::tests::textures_the_hashing_budget_left_out_come_first_next";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        if crate::descriptor::host::heap_textures_budget().is_none() {
            return;
        }
        let _untimed = Untimed::new();
        let heap = crate::descriptor::TestHeap::views(4);
        let views = crate::descriptor::heap(heap.0).host().expect("a host heap");
        // 4 MiB of RGBA8: the whole budget.
        let big = filled_texture(heap.0, 1, 64, 16384, 0x44);
        let small = filled_texture(heap.0, 2, 4, 4, 0x55);
        let list = TitleList::new();
        let counts = pass_in(heap.0, &list);
        assert_eq!(counts.uploaded, 2, "{counts:?}");
        list.run();
        assert_eq!(views.texture_queue().served_slots(), [1, 2]);
        fill_texture(small, 0x66);
        // The big one takes the budget; the small one waits for the next.
        let list = TitleList::new();
        let counts = pass_in(heap.0, &list);
        list.run();
        assert_eq!(counts.hashed, 1, "{counts:?}");
        assert_eq!(texels_4x4(host, small), vec![0x5555_5555; 16]);
        let list = TitleList::new();
        let counts = pass_in(heap.0, &list);
        list.run();
        assert_eq!(counts.hashed, 1, "{counts:?}");
        assert_eq!(texels_4x4(host, small), vec![0x6666_6666; 16]);
        assert_eq!(host.log_messages(), None);
        drop(heap);
        release(big);
        release(small);
    }

    /// A served texture whose staging buffer was released, then changed by
    /// the title, is uploaded again at the pass's check even when the
    /// pass's first uploads spent its whole budget: its buffer is made
    /// again from the refreshes' share (`texture::Refreshes`). Before, the
    /// check found no room left and went on, and shaders read its old
    /// texels as long as first uploads took the budget (Codex's review of
    /// c44a716, 2). In a child process whose passes stage 1 MiB
    /// (`XTP_HEAP_TEXTURES`, read once), which a 4 MiB texture spends.
    #[test]
    fn a_released_served_texture_is_refreshed_when_first_uploads_spent_the_budget() {
        use crate::testgpu::TitleList;
        const CHILD: &str = "XTP_TEST_SMALL_PASS";
        let test = "draw::tests::a_released_served_texture_is_refreshed_when_first_uploads_spent_the_budget";
        if std::env::var_os(CHILD).is_none() {
            let Some(_gpu) = crate::testgpu::device(test) else {
                return;
            };
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args([test, "--exact", "--nocapture", "--test-threads=1"])
                .env(CHILD, "1")
                .env("XTP_HEAP_TEXTURES", "1")
                .output()
                .unwrap();
            let stdout = String::from_utf8_lossy(&out.stdout);
            assert!(
                out.status.success() && stdout.contains("1 passed"),
                "the child: {stdout}\n{}",
                String::from_utf8_lossy(&out.stderr)
            );
            return;
        }
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        assert_eq!(
            crate::descriptor::host::heap_textures_budget(),
            Some(1 << 20)
        );
        let heap = crate::descriptor::TestHeap::views(4);
        let views = crate::descriptor::heap(heap.0).host().expect("a host heap");
        let small = filled_texture(heap.0, 1, 4, 4, 0x11);
        let list = TitleList::new();
        assert_eq!(pass_in(heap.0, &list).uploaded, 1);
        list.run();
        // The list held its staging buffer until it went.
        drop(list);
        assert_eq!(views.texture_queue().served_slots(), [1]);
        assert_eq!(texels_4x4(host, small), vec![0x1111_1111; 16]);
        assert!(crate::texture::release_staging_now(small));
        fill_texture(small, 0x22);
        // 4 MiB of RGBA8, uploaded first: more than the pass's budget.
        let big = filled_texture(heap.0, 2, 64, 16384, 0x33);
        let list = TitleList::new();
        let counts = pass_in(heap.0, &list);
        list.run();
        assert_eq!((counts.uploaded, counts.hashed), (1, 1), "{counts:?}");
        assert!(counts.bytes > 4 << 20, "{counts:?}");
        assert_eq!(texels_4x4(host, small), vec![0x2222_2222; 16]);
        assert_eq!(host.log_messages(), None);
        drop(list);
        drop(heap);
        release(big);
        release(small);
    }

    /// A served texture no pass could stage again (4 MiB, past a 1 MiB
    /// staging limit and the refreshes' share of a 1 MiB pass: a table read
    /// uploaded it, then its staging buffer was released) is left for its
    /// next turn when it changes, and the checks go on past it: a changed
    /// texture after it is refreshed, pass after pass, while the big one
    /// keeps the texels it had. Before, it stopped the checks and the next
    /// pass started with it again, so the textures after it stayed stale for
    /// good (Codex's review of db37732, 3). In a child process whose passes
    /// stage 1 MiB within a 1 MiB staging limit (`XTP_HEAP_TEXTURES`,
    /// `XTP_PASS_STAGING`, read once).
    #[test]
    fn a_texture_no_pass_can_stage_again_leaves_the_next_ones_their_refresh() {
        use crate::testgpu::TitleList;
        const CHILD: &str = "XTP_TEST_SMALL_STAGING";
        let test =
            "draw::tests::a_texture_no_pass_can_stage_again_leaves_the_next_ones_their_refresh";
        if std::env::var_os(CHILD).is_none() {
            let Some(_gpu) = crate::testgpu::device(test) else {
                return;
            };
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args([test, "--exact", "--nocapture", "--test-threads=1"])
                .env(CHILD, "1")
                .env("XTP_HEAP_TEXTURES", "1")
                .env("XTP_PASS_STAGING", "1")
                .output()
                .unwrap();
            let stdout = String::from_utf8_lossy(&out.stdout);
            assert!(
                out.status.success() && stdout.contains("1 passed"),
                "the child: {stdout}\n{}",
                String::from_utf8_lossy(&out.stderr)
            );
            return;
        }
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        assert_eq!(crate::descriptor::host::pass_staging(), 1 << 20);
        let heap = crate::descriptor::TestHeap::views(4);
        let views = crate::descriptor::heap(heap.0).host().expect("a host heap");
        // The small one by a pass; the big one, 4 MiB of RGBA8 in the slot
        // before it, by a table read (no pass stages it).
        let small = filled_texture(heap.0, 2, 4, 4, 0x11);
        let list = TitleList::new();
        assert_eq!(pass_in(heap.0, &list).uploaded, 1);
        list.run();
        drop(list);
        let big = filled_texture(heap.0, 1, 64, 16384, 0x55);
        let list = TitleList::new();
        let texture = crate::resource::host_resource(big).expect("resident");
        assert!(crate::texture::upload(list.host(), big, texture));
        views.texture_slot_tabled(1);
        list.run();
        drop(list);
        assert_eq!(views.texture_queue().served_slots(), [1, 2]);
        assert_eq!(texels_4x4(host, big), vec![0x5555_5555; 16]);
        assert!(crate::texture::release_staging_now(big));
        assert!(crate::texture::release_staging_now(small));
        // Frames on, no table reads it: the passes check both, the big one
        // first. A pass checks within `PASS_CHECKING` (1 ms), one at least:
        // on a loaded machine the small one may wait for the next pass.
        crate::descriptor::host::PASS_LATER.fetch_add(2, Ordering::Relaxed);
        fill_texture(big, 0x66);
        for byte in [0x22u8, 0x33] {
            fill_texture(small, byte);
            let want = vec![u32::from_ne_bytes([byte; 4]); 16];
            let mut passes = 0;
            while texels_4x4(host, small) != want {
                passes += 1;
                assert!(passes <= 3, "still stale after {passes} passes");
                let list = TitleList::new();
                pass_in(heap.0, &list);
                list.run();
            }
            assert_eq!(texels_4x4(host, big), vec![0x5555_5555; 16], "as it was");
        }
        assert_eq!(host.log_messages(), None);
        drop(heap);
        release(big);
        release(small);
    }

    /// A list's state with the title view heap `heap` bound (none: null).
    fn state(heap: This) -> State {
        let mut state = State::default();
        if !heap.is_null() {
            state.set_heaps(&[heap]);
        }
        state
    }

    /// Sets the heaps a pipeline gets (the title's, XTP's own where it binds
    /// none), as `bind` does.
    fn heaps(state: &mut State, list: &HostList) {
        let wanted = state.wanted_heaps();
        state.set_host_heaps(list, wanted);
    }

    /// The table of the root SRV at `address` (none: unset).
    fn table(state: &mut State, list: &HostList, address: Option<u64>) -> Bind {
        let (host, slot) = state.root_srv_slot(list, address).expect("a slot");
        Bind::Table(host.gpu_handle(slot))
    }

    /// What a view of `n` words from word `first` of the pattern reads:
    /// those words, then zeros.
    fn bounded(first: usize, n: usize) -> Vec<u32> {
        (0..WORDS)
            .map(|i| if i < n { pattern(first + i) } else { 0 })
            .collect()
    }

    /// Words in a [`TitleBuffer`] of 64 KiB.
    const END: usize = 1 << 14;

    /// A root SRV is read through its one-view table: in place on 16 bytes
    /// of its host copy, through a copy on the GPU elsewhere (4-byte
    /// aligned, as vertex pools of 12-byte vertices give), bounded at its
    /// buffer's end either way; unset, it reads zeros.
    #[test]
    fn root_srv_tables_read_their_range() {
        let test = "draw::tests::root_srv_tables_read_their_range";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let reader = Reader::new(host, ReaderView::TableSrv);
        let heap = TestHeap::views(16);
        let src = TitleBuffer::new(1 << 16);
        src.set_words(pattern);
        let read = |at: Option<usize>, copied: u64| {
            let list = TitleList::new();
            let mut state = state(heap.0);
            state.own.copied = copied;
            let got = reader.read(&list, |l| {
                heaps(&mut state, l);
                table(&mut state, l, at.map(|a| src.address() + a as u64))
            });
            assert_eq!(state.host_heaps[0], heap.0 as usize, "the title's heap");
            got
        };
        let words = |at| read(at, 0);
        assert_eq!(words(Some(16)), bounded(4, WORDS), "in place");
        // Views in place consume no copy budget.
        assert_eq!(read(Some(16), LIST_COPIES), bounded(4, WORDS));
        assert_eq!(words(Some(4)), bounded(1, WORDS), "copied");
        assert_eq!(words(Some(20)), bounded(5, WORDS), "copied");
        assert_eq!(
            words(Some(4 * (END - 8))),
            bounded(END - 8, 8),
            "in place, at the buffer's end"
        );
        assert_eq!(
            words(Some(4 * (END - 6))),
            bounded(END - 6, 6),
            "copied, at the buffer's end"
        );
        assert_eq!(words(None), vec![0; WORDS], "unset");
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// A root SRV not on 16 bytes is copied at its place in the list: it
    /// reads what a dispatch recorded before it wrote there.
    #[test]
    fn copied_root_srvs_see_what_the_list_wrote_before() {
        let test = "draw::tests::copied_root_srvs_see_what_the_list_wrote_before";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let reader = Reader::new(host, ReaderView::TableSrv);
        let kernel = Kernel::new(host);
        let heap = TestHeap::views(16);
        let (scratch, out) = (TitleBuffer::new(256), TitleBuffer::new(1 << 16));
        let list = TitleList::new();
        let mut state = state(heap.0);
        let got = reader.read(&list, |l| {
            heaps(&mut state, l);
            // The pattern's first 64 words into `out`'s host copy.
            kernel.dispatch(l, Pass::Fill, scratch.address(), out.address());
            table(&mut state, l, Some(out.address() + 4))
        });
        assert_eq!(got, bounded(1, 63));
    }

    /// A root SRV's one-view table in place views the rest of its buffer,
    /// not 4 MiB of it (Watch 443, VIctory #1671): a shader reading 6 MiB
    /// past the SRV reads the buffer there, as on the console.
    #[test]
    fn root_srv_tables_read_past_4_mib() {
        const FAR: usize = 6 << 20;
        let test = "draw::tests::root_srv_tables_read_past_4_mib";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let near = Reader::new(host, ReaderView::TableSrv);
        let source = format!(
            "ByteAddressBuffer src : register(t0, space1);
RWByteAddressBuffer dst : register(u0);
[numthreads(128, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {{
    dst.Store(id.x * 4, src.Load({FAR} + id.x * 4));
}}"
        );
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call.
        unsafe {
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                s!("far root view test"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )
        }
        .expect("the test kernel compiles");
        let code = code.expect("compiled code");
        // SAFETY: the code and the root signature (the near reader's: the
        // same views) live through the call.
        let pso = unsafe {
            host.device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&near.root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.GetBufferPointer(),
                        BytecodeLength: code.GetBufferSize(),
                    },
                    ..Default::default()
                })
                .unwrap()
        };
        let far = Reader {
            root: near.root.clone(),
            pso,
        };
        let heap = TestHeap::views(16);
        let src = TitleBuffer::new(8 << 20);
        src.set_words(pattern);
        let list = TitleList::new();
        let mut state = state(heap.0);
        let got = far.read(&list, |l| {
            heaps(&mut state, l);
            table(&mut state, l, Some(src.address() + 16))
        });
        let want: Vec<u32> = (0..WORDS).map(|i| pattern(4 + FAR / 4 + i)).collect();
        assert_eq!(got, want);
    }

    /// A list binds XTP's own heap in each slot the title binds no hosted
    /// heap of (the console needs none for root views), both before a
    /// directly indexed root signature; root SRVs are read through tables
    /// in its own view heap. A title heap takes its slot over; setting the
    /// heaps again with the same ones does nothing.
    #[test]
    fn lists_without_a_view_heap_use_xtps_own() {
        let test = "draw::tests::lists_without_a_view_heap_use_xtps_own";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let reader = Reader::new(host, ReaderView::TableSrv);
        let src = TitleBuffer::new(1 << 16);
        src.set_words(pattern);
        let list = TitleList::new();
        let mut state = state(std::ptr::null());
        // Without heaps of the title's, XTP's own: a root signature that
        // indexes both heaps directly needs both bound.
        assert_eq!(state.wanted_heaps(), [OWN_VIEW_HEAP, OWN_SAMPLER_HEAP]);
        let got = reader.read(&list, |l| {
            heaps(&mut state, l);
            table(&mut state, l, Some(src.address() + 64))
        });
        assert_eq!(got, bounded(16, WORDS));
        assert_eq!(state.wanted_heaps(), [OWN_VIEW_HEAP, OWN_SAMPLER_HEAP]);
        let heap = TestHeap::views(16);
        state.set_heaps(&[heap.0]);
        assert_eq!(state.wanted_heaps(), [heap.0 as usize, OWN_SAMPLER_HEAP]);
    }

    /// A title sampler heap takes the sampler slot and XTP's own view heap
    /// the view one; with a view heap too, both are the title's. Setting the
    /// same heaps again does nothing (the root signatures stay set), other
    /// heaps reset them.
    #[test]
    fn title_sampler_heaps_take_their_slot() {
        let test = "draw::tests::title_sampler_heaps_take_their_slot";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let views = TestHeap::views(16);
        let samplers = TestHeap::samplers(16);
        let (v, s) = (views.0 as usize, samplers.0 as usize);
        let list = TitleList::new();
        let mut state = State::default();
        state.set_heaps(&[samplers.0]);
        assert_eq!(state.wanted_heaps(), [OWN_VIEW_HEAP, s], "samplers alone");
        heaps(&mut state, list.host());
        assert_eq!(state.host_heaps, [OWN_VIEW_HEAP, s]);
        state.set_heaps(&[views.0, samplers.0]);
        assert_eq!(state.wanted_heaps(), [v, s], "views and samplers");
        heaps(&mut state, list.host());
        assert_eq!(state.host_heaps, [v, s]);
        // The same heaps again: nothing is set, the root signatures stay.
        state.host_roots = [1, 2];
        heaps(&mut state, list.host());
        assert_eq!(state.host_heaps, [v, s]);
        assert_eq!(state.host_roots, [1, 2], "unchanged heaps");
        // Other heaps: set, and the root signatures with them.
        state.set_heaps(&[views.0]);
        heaps(&mut state, list.host());
        assert_eq!(state.host_heaps, [v, OWN_SAMPLER_HEAP]);
        assert_eq!(state.host_roots, [0; 2], "changed heaps");
    }

    /// Root SRV views in place are shared across lists, by host copy and
    /// address; copies of a list's own are its own, made again after work
    /// that may write memory; unset ones share a null view. A list gives
    /// back what it holds when it is reset or released.
    #[test]
    fn root_srv_views_are_shared_across_lists_and_given_back() {
        let test = "draw::tests::root_srv_views_are_shared_across_lists_and_given_back";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let heap = TestHeap::views(16);
        let views = crate::descriptor::heap(heap.0).host().expect("a host heap");
        let src = TitleBuffer::new(1 << 16);
        let (la, lb) = (TitleList::new(), TitleList::new());
        let (mut a, mut b) = (state(heap.0), state(heap.0));
        heaps(&mut a, la.host());
        heaps(&mut b, lb.host());
        let slot = |s: &mut State, l: &TitleList, at: Option<u64>| {
            s.root_srv_slot(l.host(), at.map(|a| src.address() + a))
                .map(|(_, slot)| slot)
                .expect("a slot")
        };
        let key = heap.0 as usize;
        let first = slot(&mut a, &la, Some(16));
        assert_eq!(slot(&mut a, &la, Some(16)), first);
        assert_eq!(slot(&mut b, &lb, Some(16)), first, "shared with list b");
        let other = slot(&mut a, &la, Some(32));
        assert_ne!(other, first);
        assert_eq!(a.recording.held().shared, [(key, first), (key, other)]);
        // Copies: the list's own, made again after work that may write.
        let copied = slot(&mut a, &la, Some(4));
        assert_eq!(slot(&mut a, &la, Some(4)), copied);
        src.shadow().gpu_writes(la.host(), src.address() + 4, 4);
        let again = slot(&mut a, &la, Some(4));
        assert_ne!(again, copied);
        assert_eq!(
            a.recording.held().ranges,
            [(key, copied, 1), (key, again, 1)]
        );
        assert_eq!(slot(&mut a, &la, None), slot(&mut b, &lb, None), "null");
        assert_eq!(views.shared_count(), (2, 0));
        // The shared views' chunk, the null view and the two copies.
        assert_eq!(views.dynamic_use().0, SHARED_CHUNK + 3);
        // List a is reset as the title resets one: its recording goes (the
        // host list), then its state.
        la.host().close();
        la.host().reset();
        assert_eq!(views.shared_count(), (2, 0), "the state still has it");
        a.reset(std::ptr::null());
        assert_eq!(a.recording.held(), &crate::descriptor::Held::default());
        assert_eq!(views.shared_count(), (2, 1), "list b holds the first");
        assert_eq!(views.dynamic_use().0, SHARED_CHUNK + 1);
        // List b's state goes first: nothing goes back before its recording.
        drop(b);
        assert_eq!(views.shared_count(), (2, 1), "its recording is still there");
        lb.host().close();
        lb.host().reset();
        assert_eq!(views.shared_count(), (2, 2), "kept for later lists");
        // The next list finds the view made for the first two.
        let mut c = state(heap.0);
        heaps(&mut c, la.host());
        assert_eq!(slot(&mut c, &la, Some(16)), first);
    }

    #[test]
    fn rejected_bind_releases_only_its_new_descriptors() {
        let test = "draw::tests::rejected_bind_releases_only_its_new_descriptors";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let heap = TestHeap::views(16);
        let views = crate::descriptor::heap(heap.0).host().unwrap();
        let src = TitleBuffer::new(1 << 16);
        let list = TitleList::new();
        let l = list.host();
        let mut state = state(heap.0);
        heaps(&mut state, l);
        let address = src.address();
        let kept = state.root_srv_slot(l, Some(address + 4)).unwrap().1;
        let before = state.recording.checkpoint();
        for _ in 0..3 {
            let result = state.with_bind_descriptors(|state| {
                assert_eq!(state.root_srv_slot(l, Some(address + 4))?.1, kept);
                state.root_srv_slot(l, Some(address + 8))?;
                state.root_srv_slot(l, Some(address + 16))?;
                Err("later stage rejected")
            });
            assert!(result.is_err());
            assert_eq!(state.root_srv_slots.len(), 1);
            assert_eq!(state.recording.checkpoint(), before);
            assert_eq!(views.shared_count(), (1, 1));
            assert_eq!(views.dynamic_use().0, 1 + SHARED_CHUNK);
        }
    }

    /// A misaligned CBV whose first 64 words have a queued shader writer.
    fn pending_cbv_writer(host: &Host, first: &HostList) -> (crate::host::submissions::Gate, u64) {
        use crate::host::submissions::{Gate, watched};
        let kernel = Kernel::new(host);
        let base = watched(4 << 20) as u64;
        let address = base + 16; // A valid title CBV, misaligned on the host.
        let shadow = crate::shadow::find(base, CBV_SPAN + 16).unwrap();
        shadow.refresh(first, base, crate::watch::PAGE as u64);
        kernel.dispatch(first, Pass::Fill, base + 2 * CBV_SPAN, address);
        first.close();
        let gate = Gate::close(host);
        host.execute(&[first]);
        (gate, address)
    }

    #[test]
    fn a_cbv_fill_reads_an_in_flight_shaders_output_on_the_gpu() {
        let _serial = crate::host::submissions::serial();
        let test = "draw::tests::a_cbv_fill_reads_an_in_flight_shaders_output_on_the_gpu";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let reader = Reader::new(host, ReaderView::RootCbv);
        let second = host.new_list().unwrap();
        let (out, data) = host.shared_buffer(WORDS as u64 * 4).unwrap();
        let mut state = state(std::ptr::null());
        let first = host.new_list().unwrap();
        let kernel = Kernel::new(host);
        let (mut gate, address) = pending_cbv_writer(host, &first);
        heaps(&mut state, &second);
        let cbv = state.root_cbv(&second, address).unwrap();
        // A dispatch writes a disjoint range in the same shadow.
        kernel.dispatch(
            &second,
            Pass::Fill,
            address + 2 * CBV_SPAN,
            address + CBV_SPAN,
        );
        // Without reuse, these binds alone spend the whole list's budget.
        for _ in 0..LIST_COPIES / CBV_SPAN {
            assert_eq!(state.root_cbv(&second, address), Ok(cbv));
        }
        assert_eq!(state.own.copied, CBV_SPAN);
        let shifted = state.root_cbv(&second, address + 16).unwrap();
        assert_ne!(shifted, cbv, "different source offset");
        assert_eq!(state.own.copied, 2 * CBV_SPAN);
        // SAFETY: the mapped output lives through the dispatch.
        reader.dispatch(&second, Bind::Cbv(cbv), unsafe {
            out.GetGPUVirtualAddress()
        });
        second.close();
        host.execute(&[&second]);
        assert!(first.running() && second.running());
        gate.open();
        host.drain();
        // SAFETY: the GPU is done writing the mapped result.
        let got = unsafe { std::slice::from_raw_parts(data.cast::<u32>(), WORDS) };
        assert_eq!(&got[..64], (0..64).map(pattern).collect::<Vec<_>>());
        assert_eq!(&got[64..], [0; WORDS - 64]);
    }

    #[test]
    fn table_cbv_copies_see_gpu_writes_and_cpu_updates_before_submission() {
        let _serial = crate::host::submissions::serial();
        let test = "draw::tests::table_cbv_copies_see_gpu_writes_and_cpu_updates_before_submission";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let reader = Reader::new(host, ReaderView::TableCbv);
        let list = host.new_list().unwrap();
        let heap = TestHeap::views(16);
        let title = crate::descriptor::heap(heap.0);
        let views = title.host().unwrap();
        let mut state = state(heap.0);
        heaps(&mut state, &list);
        let used = [UsedResource {
            kind: ResourceType::Cbv,
            root: RootType::Table,
            slot: 0,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
        }];
        let (out, data) = host.shared_buffer(WORDS as u64 * 4).unwrap();
        let first = host.new_list().unwrap();
        let kernel = Kernel::new(host);
        let (mut gate, address) = pending_cbv_writer(host, &first);
        let bind = |state: &mut State, index: u64, bytes: u32| {
            let handle = title.start + index * 32;
            let desc = [address as u32, (address >> 32) as u32, bytes, 0];
            crate::descriptor::create_cbv(std::ptr::null(), &desc, handle);
            state.set_table(true, 0, handle);
            let made = state.tables(&list, &used, true).unwrap();
            assert_eq!(made.len(), 1);
            (made[0].1 / 32) & !xtp_shader::emit::REMAPPED_SLOT
        };
        let short = bind(&mut state, 0, 256);
        assert_eq!(state.own.copied, 256);
        let long = bind(&mut state, 1, 512);
        assert_eq!(
            state.own.copied,
            256 + 512,
            "a short copy cannot serve a longer view"
        );
        assert_eq!(bind(&mut state, 0, 256), short);
        bind(&mut state, 2, 512);
        assert_eq!(
            state.own.copied,
            256 + 512,
            "another table reuses the same bytes"
        );
        let slots = views.dynamic_use().0;
        for _ in 0..8 {
            kernel.dispatch(&list, Pass::Fill, address + 2 * CBV_SPAN, address + 512);
            assert_eq!(bind(&mut state, 1, 512), long);
        }
        assert_eq!(views.dynamic_use().0, slots);
        assert_eq!(state.own.copied, 256 + 512);
        gate.open();
        host.drain();
        let uploaded = bind(&mut state, 1, 512);
        assert_ne!(uploaded, long, "the queued writer finished");
        assert_eq!(bind(&mut state, 1, 512), uploaded);
        assert_eq!(state.own.copied, 256 + 512);
        assert!(crate::shadow::write(&list, address, 0x1234, 4));
        let updated = bind(&mut state, 1, 512);
        assert_ne!(updated, uploaded, "the GPU write invalidates the upload");
        assert_eq!(state.own.copied, 256 + 2 * 512);
        // SAFETY: the mapped output lives through the dispatch.
        reader.dispatch(&list, Bind::Table(views.gpu_handle(updated)), unsafe {
            out.GetGPUVirtualAddress()
        });
        list.close();
        // SAFETY: word 100 belongs to the watched reservation, beyond the producer's writes.
        unsafe { (address as *mut u32).add(100).write_volatile(0x5678) };
        host.execute(&[&list]);
        host.drain();
        // SAFETY: the GPU finished writing the mapped output.
        let got = unsafe { std::slice::from_raw_parts(data.cast::<u32>(), WORDS) };
        let mut want = bounded(0, 64);
        want[0] = 0x1234;
        want[100] = 0x5678;
        assert_eq!(got, want);
    }

    /// A table's constant view reads its title bytes, then zeros to its
    /// 256-byte end (the console reads zeros past a view's size), not what
    /// earlier work left in the list's memory: in scratch memory filled at
    /// submission, or in a copy on the GPU (a source shaders write). Views
    /// of no address or no bytes are null ones and take no memory; views
    /// over 64 KiB take 64 KiB.
    #[test]
    fn table_cbvs_read_zeros_past_their_size() {
        let test = "draw::tests::table_cbvs_read_zeros_past_their_size";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let reader = Reader::new(host, ReaderView::TableCbv);
        let heap = TestHeap::views(16);
        let title = crate::descriptor::heap(heap.0);
        let views = title.host().unwrap();
        let src = TitleBuffer::new(1 << 20);
        src.set_words(pattern);
        let written = TitleBuffer::new(1 << 17);
        written.set_words(pattern);
        let used = [UsedResource {
            kind: ResourceType::Cbv,
            root: RootType::Table,
            slot: 0,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
        }];
        let list = host.new_list().unwrap();
        // What an earlier recording left in the list's scratch memory (a
        // list not running keeps its chunks through a reset).
        const STALE: u64 = 1 << 20;
        let (_, stale) = list.scratch(STALE).unwrap();
        // SAFETY: `STALE` bytes of the list's mapped scratch memory.
        unsafe { stale.write_bytes(0xff, STALE as usize) };
        list.close();
        list.reset();
        let mut state = state(heap.0);
        heaps(&mut state, &list);
        let bind = |state: &mut State, index: u64, address: u64, bytes: u32| {
            let handle = title.start + index * 32;
            let desc = [address as u32, (address >> 32) as u32, bytes, 0];
            crate::descriptor::create_cbv(std::ptr::null(), &desc, handle);
            state.set_table(true, 0, handle);
            let made = state.tables(&list, &used, true).unwrap();
            assert_eq!(made.len(), 1);
            (made[0].1 / 32) & !xtp_shader::emit::REMAPPED_SLOT
        };
        let scratch_at = |list: &HostList| list.scratch(256).unwrap().0;

        // 32 bytes through scratch memory.
        let scratch = bind(&mut state, 0, src.address(), 32);

        // 32 bytes copied on the GPU, into memory of the list's own that a
        // longer copy (a root CBV's) filled first.
        let at = written.address() + 4096 + 16;
        written.shadow().gpu_writes(&list, at, 4);
        state.root_cbv(&list, at).unwrap();
        assert_eq!(state.own.gpu.len(), 1);
        assert_eq!(state.own.copied, CBV_SPAN);
        (state.own.gpu[0].used, state.own.copied) = (0, 0);
        state.cbv_copies.clear();
        let copied = bind(&mut state, 1, at, 32);
        assert_eq!(state.own.copied, 32, "copied on the GPU");

        // Null views, which take no scratch memory.
        let before = scratch_at(&list);
        let nulls = [(2, 0, 0), (3, src.address(), 0), (4, 0, 256)]
            .map(|(index, address, bytes)| bind(&mut state, index, address, bytes));
        assert_eq!(scratch_at(&list), before + 256, "null views take no memory");

        // A view of 969,472 bytes (a size seen in a title's stale slot).
        let before = scratch_at(&list);
        let big = bind(&mut state, 5, src.address(), 969_472);
        assert_eq!(
            scratch_at(&list),
            before + 256 + CBV_SPAN,
            "64 KiB of scratch memory"
        );

        let slots = [scratch, copied, nulls[0], nulls[1], nulls[2], big];
        let bytes = WORDS as u64 * 4;
        let (out, data) = host.shared_buffer(slots.len() as u64 * bytes).unwrap();
        // SAFETY: a live buffer.
        let out_at = unsafe { out.GetGPUVirtualAddress() };
        for (k, &slot) in slots.iter().enumerate() {
            let to = out_at + k as u64 * bytes;
            reader.dispatch(&list, Bind::Table(views.gpu_handle(slot)), to);
        }
        list.close();
        host.execute(&[&list]);
        host.drain();
        // SAFETY: the GPU finished writing the mapped output.
        let got = unsafe { std::slice::from_raw_parts(data.cast::<u32>(), slots.len() * WORDS) };
        let got = |k: usize| &got[k * WORDS..(k + 1) * WORDS];
        assert_eq!(got(0), bounded(0, 8), "scratch: zeros past 32 bytes");
        // Its first word is the shader's (`gpu_writes`), which wrote none:
        // the host copy keeps its zeros there.
        let mut want = bounded((4096 + 16) / 4, 8);
        want[0] = 0;
        assert_eq!(got(1), want, "GPU copy: zeros past 32 bytes");
        for k in 2..5 {
            assert_eq!(got(k), [0; WORDS], "null view {k}");
        }
        assert_eq!(got(5), bounded(0, WORDS), "the first 64 KiB");
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    #[test]
    fn cbv_copy_budget_rejects_new_gpu_inputs_and_preserves_earlier_binds() {
        let test =
            "draw::tests::cbv_copy_budget_rejects_new_gpu_inputs_and_preserves_earlier_binds";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let src = TitleBuffer::new(2 << 16);
        let address = src.address() + 16;
        let heap = TestHeap::views(300);
        let title = crate::descriptor::heap(heap.0);
        let list = TitleList::new();
        let l = list.host();
        let mut state = state(heap.0);
        heaps(&mut state, l);
        let used = [UsedResource {
            kind: ResourceType::Cbv,
            root: RootType::Table,
            slot: 0,
            table_offset_dwords: 299 * 8,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
        }];
        let desc = [address as u32, (address >> 32) as u32, 256, 0];
        crate::descriptor::create_cbv(std::ptr::null(), &desc, title.start + 299 * 32);
        state.set_table(true, 0, title.start);
        src.shadow().gpu_writes(l, address, 4);
        // The root copy includes this newer write beyond the table's CBV.
        src.shadow().gpu_writes(l, address + 256, 4);
        let original = state.root_cbv(l, address).unwrap();
        let table = state.tables(l, &used, true).unwrap();
        let slots = title.host().unwrap().dynamic_use().0;
        assert_eq!(slots, 300);
        let kernel = Kernel::new(gpu.host());
        for _ in 0..8 {
            kernel.dispatch(l, Pass::Fill, address, address + CBV_SPAN);
            assert_eq!(state.root_cbv(l, address), Ok(original));
            assert_eq!(state.tables(l, &used, true).unwrap(), table);
        }
        assert_eq!(title.host().unwrap().dynamic_use().0, slots);
        assert_eq!(state.own.copied, CBV_SPAN);
        state.own.copied = LIST_COPIES;

        // The current shader's outputs follow its bindings, regardless of
        // the order its stages declare their views.
        l.begin_shader_outputs();
        src.shadow().gpu_writes(l, address, 4);
        assert_eq!(state.root_cbv(l, address), Ok(original));
        assert_eq!(state.tables(l, &used, true).unwrap(), table);
        l.end_shader_outputs(true);
        assert_eq!(
            state.root_cbv(l, address),
            Err("could not copy a CBV on the GPU")
        );
        let before = state.recording.checkpoint();
        assert_eq!(
            state.with_bind_descriptors(|s| s.tables(l, &used, true).map(|_| ())),
            Err("could not copy a CBV on the GPU")
        );
        assert_eq!(
            state.with_bind_descriptors(|s| s.root_srv_slot(l, Some(address + 4)).map(|_| ())),
            Err("could not copy a misaligned root SRV")
        );
        assert_eq!(state.recording.checkpoint(), before);
        assert!(state.root_srv_slots.is_empty());
        let upload_address = address + CBV_SPAN + 256;
        let upload = state.root_cbv(l, upload_address).expect("disjoint upload");
        assert_eq!(state.root_cbv(l, upload_address), Ok(upload));
        l.defer_copy(address as *mut u8, upload_address as *const u8, 4);
        assert_eq!(state.root_cbv(l, upload_address), Ok(upload));
        l.defer_copy(upload_address as *mut u8, address as *const u8, 4);
        assert_ne!(state.root_cbv(l, upload_address).unwrap(), upload);
        assert_eq!(state.own.copied, LIST_COPIES);
    }

    /// A root CBV reads its 64 KiB inside memory the list holds: in place
    /// when they lie in its host copy, else in a copy of the list's own
    /// (on the GPU when 256-byte aligned, from title memory when not).
    #[test]
    fn root_cbvs_read_inside_memory_the_list_holds() {
        let test = "draw::tests::root_cbvs_read_inside_memory_the_list_holds";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let reader = Reader::new(host, ReaderView::RootCbv);
        let src = TitleBuffer::new(1 << 16);
        src.set_words(pattern);
        let copy = src.shadow();
        let words = |offset: usize| {
            let list = TitleList::new();
            let mut state = state(std::ptr::null());
            let address = src.address() + offset as u64;
            let mut in_place = false;
            let got = reader.read(&list, |l| {
                heaps(&mut state, l);
                let gpu = state.root_cbv(l, address).expect("a CBV");
                in_place = gpu == copy.gpu(address);
                Bind::Cbv(gpu)
            });
            (got, in_place)
        };
        assert_eq!(words(0), (bounded(0, WORDS), true), "in place");
        // Reject a bind that would read past the end of the host copy.
        let list = TitleList::new();
        let mut spent = state(std::ptr::null());
        spent.own.copied = LIST_COPIES;
        let address = src.address() + 4 * (END as u64 - 64);
        assert_eq!(
            spent.root_cbv(list.host(), address),
            Err("could not copy a CBV on the GPU")
        );
        assert_eq!(
            words(16),
            (bounded(4, WORDS), false),
            "not 256-byte aligned"
        );
        // At the buffer's end: its last words, then the list's own memory
        // (chunks are reused, so not zeros necessarily).
        let (got, in_place) = words(4 * (END - 64));
        assert_eq!(
            (&got[..64], in_place),
            (&bounded(END - 64, 64)[..64], false),
            "copied on the GPU"
        );
        let (got, in_place) = words(4 * (END - 4));
        assert_eq!(
            (&got[..4], in_place),
            (&bounded(END - 4, 4)[..4], false),
            "not aligned"
        );
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// A direct rectangle draw runs six vertices a rectangle from 0 in its
    /// expanded buffers; for a pipeline without per-vertex inputs, each
    /// rectangle's first triangle, from the title's start (it drew twice
    /// the vertices from 0 before).
    #[test]
    fn direct_rectangles_without_vertex_inputs_keep_their_start() {
        assert_eq!(rectangle_vertices(7, 12, true), (12, 0));
        assert_eq!(rectangle_vertices(7, 12, false), (6, 12));
        assert_eq!(rectangle_vertices(2, 5, false), (0, 5));
    }

    /// Dropped dispatches (no pipeline) mark the list, so that its next
    /// dispatch that runs starts with a UAV barrier (VIctory #1392): one a
    /// run of drops; the mark goes with the title's reset. No test here
    /// runs a dispatch (no title pipeline), nor could one see what the
    /// barrier orders (WARP runs commands in order).
    #[test]
    fn dropped_dispatches_mark_the_list_for_one_barrier() {
        let test = "draw::tests::dropped_dispatches_mark_the_list_for_one_barrier";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let list = TitleList::new();
        let mut state = State::default();
        assert!(!state.dropped_dispatch);
        state.dispatch(list.host(), [1; 3], [0; 3]);
        state.dispatch(list.host(), [1; 3], [0; 3]);
        assert!(state.dropped_dispatch, "marked");
        state.reset(std::ptr::null());
        assert!(!state.dropped_dispatch, "gone with the reset");
        list.run();
    }

    /// An indirect execution of no record draws and dispatches nothing,
    /// and is not dropped either (whatever is bound).
    #[test]
    fn indirect_executions_without_records_draw_nothing() {
        let test = "draw::tests::indirect_executions_without_records_draw_nothing";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let list = TitleList::new();
        let args = TitleBuffer::new(256);
        let mut state = State::default();
        let before = drawn();
        state.topology = RECTANGLES;
        let rectangles = state.execute_indirect(list.host(), (0, 16, 0), args.address(), None);
        state.topology = 4;
        let draws = state.execute_indirect(list.host(), (0, 16, 0), args.address(), None);
        let dispatches = state.execute_indirect(list.host(), (2, 12, 0), args.address(), None);
        assert_eq!(
            (rectangles, draws, dispatches),
            (true, true, true),
            "not dropped"
        );
        assert_eq!(drawn(), before, "not drawn");
    }

    /// Pixels of a [`Painter`]'s target a side.
    const SIDE: u32 = 16;

    /// Black, where a [`Painter`] did not draw.
    const CLEAR: [u8; 4] = [0; 4];

    /// The colors of a [`Painter`]'s four cells.
    const COLORS: [[u8; 4]; 4] = [
        [255, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 0, 255, 255],
        [255, 255, 0, 255],
    ];

    const PAINTER: &str = r#"
#define RS "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT)"
struct V { float4 pos : SV_Position; float4 color : COLOR; };
[RootSignature(RS)]
V vs_inputs(float2 pos : POSITION, float4 color : COLOR) {
    V v;
    v.pos = float4(pos, 0.0, 1.0);
    v.color = color;
    return v;
}
[RootSignature(RS)]
V vs_ids(uint id : SV_VertexID) {
    float x = -1.0 + 0.5 * (float)(id / 3u);
    uint corner = id % 3u;
    V v;
    v.pos = float4(corner == 2u ? x + 0.5 : x, corner == 0u ? 1.0 : -1.0, 0.0, 1.0);
    v.color = float4(1.0, 1.0, 1.0, 1.0);
    return v;
}
float4 ps(V v) : SV_Target { return v.color; }
"#;

    /// A graphics pipeline standing in for a title's rectangle pipeline,
    /// drawing into a [`SIDE`]² RGBA8 target of four cells side by side
    /// (each a quarter of its width). With inputs: positions (R32G32_FLOAT)
    /// in slot 0, colors (R8G8B8A8_UNORM) in slot 1. Without: corners from
    /// the vertex id, in white: vertex `i` is corner `i % 3` (top left,
    /// bottom left, bottom right) of cell `i / 3`.
    pub(crate) struct Painter {
        root: ID3D12RootSignature,
        pso: ID3D12PipelineState,
        target: ID3D12Resource,
        views: ID3D12DescriptorHeap,
    }

    impl Painter {
        fn compile(entry: windows::core::PCSTR, target: windows::core::PCSTR) -> ID3DBlob {
            let mut code: Option<ID3DBlob> = None;
            let mut errors: Option<ID3DBlob> = None;
            // SAFETY: the source and names live through the call.
            let compiled = unsafe {
                D3DCompile(
                    PAINTER.as_ptr().cast(),
                    PAINTER.len(),
                    s!("painter"),
                    None,
                    None,
                    entry,
                    target,
                    0,
                    0,
                    &mut code,
                    Some(&mut errors),
                )
            };
            if let Err(e) = compiled {
                let text = errors.map_or_else(String::new, |errors| {
                    // SAFETY: the blob's own bytes.
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            errors.GetBufferPointer().cast::<u8>(),
                            errors.GetBufferSize(),
                        )
                    };
                    String::from_utf8_lossy(bytes).into_owned()
                });
                panic!("the painter did not compile ({e}): {text}");
            }
            code.expect("compiled code")
        }

        pub(crate) fn new(host: &Host, inputs: bool) -> Painter {
            use windows::Win32::Graphics::Dxgi::Common::*;
            let entry = if inputs {
                s!("vs_inputs")
            } else {
                s!("vs_ids")
            };
            let vs = Painter::compile(entry, s!("vs_5_1"));
            let ps = Painter::compile(s!("ps"), s!("ps_5_1"));
            let element = |name, format, slot| D3D12_INPUT_ELEMENT_DESC {
                SemanticName: name,
                SemanticIndex: 0,
                Format: format,
                InputSlot: slot,
                AlignedByteOffset: 0,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                InstanceDataStepRate: 0,
            };
            let elements = [
                element(s!("POSITION"), DXGI_FORMAT_R32G32_FLOAT, 0),
                element(s!("COLOR"), DXGI_FORMAT_R8G8B8A8_UNORM, 1),
            ];
            let layout: &[D3D12_INPUT_ELEMENT_DESC] = if inputs { &elements } else { &[] };
            let device = &host.device;
            // SAFETY: plain object creation from descriptions and code that
            // live through the calls; the root signature is in the vertex
            // shader.
            unsafe {
                let vs_bytes = std::slice::from_raw_parts(
                    vs.GetBufferPointer().cast::<u8>(),
                    vs.GetBufferSize(),
                );
                let root: ID3D12RootSignature = device.CreateRootSignature(0, vs_bytes).unwrap();
                let mut blend = D3D12_BLEND_DESC::default();
                blend.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
                let mut formats = [DXGI_FORMAT_UNKNOWN; 8];
                formats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
                let pso = device
                    .CreateGraphicsPipelineState(&D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&root),
                        VS: D3D12_SHADER_BYTECODE {
                            pShaderBytecode: vs.GetBufferPointer(),
                            BytecodeLength: vs.GetBufferSize(),
                        },
                        PS: D3D12_SHADER_BYTECODE {
                            pShaderBytecode: ps.GetBufferPointer(),
                            BytecodeLength: ps.GetBufferSize(),
                        },
                        BlendState: blend,
                        SampleMask: u32::MAX,
                        RasterizerState: D3D12_RASTERIZER_DESC {
                            FillMode: D3D12_FILL_MODE_SOLID,
                            CullMode: D3D12_CULL_MODE_NONE,
                            ..Default::default()
                        },
                        InputLayout: D3D12_INPUT_LAYOUT_DESC {
                            pInputElementDescs: layout.as_ptr(),
                            NumElements: layout.len() as u32,
                        },
                        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                        NumRenderTargets: 1,
                        RTVFormats: formats,
                        SampleDesc: DXGI_SAMPLE_DESC {
                            Count: 1,
                            Quality: 0,
                        },
                        ..Default::default()
                    })
                    .unwrap();
                let mut target: Option<ID3D12Resource> = None;
                device
                    .CreateCommittedResource(
                        &D3D12_HEAP_PROPERTIES {
                            Type: D3D12_HEAP_TYPE_DEFAULT,
                            ..Default::default()
                        },
                        D3D12_HEAP_FLAG_NONE,
                        &D3D12_RESOURCE_DESC {
                            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                            Width: u64::from(SIDE),
                            Height: SIDE,
                            DepthOrArraySize: 1,
                            MipLevels: 1,
                            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                            SampleDesc: DXGI_SAMPLE_DESC {
                                Count: 1,
                                Quality: 0,
                            },
                            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
                            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                            ..Default::default()
                        },
                        D3D12_RESOURCE_STATE_RENDER_TARGET,
                        None,
                        &mut target,
                    )
                    .unwrap();
                let target = target.expect("a render target");
                let views: ID3D12DescriptorHeap = device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                        NumDescriptors: 1,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                        NodeMask: 0,
                    })
                    .unwrap();
                device.CreateRenderTargetView(
                    &target,
                    None,
                    views.GetCPUDescriptorHandleForHeapStart(),
                );
                Painter {
                    root,
                    pso,
                    target,
                    views,
                }
            }
        }

        /// Clears the target to [`CLEAR`].
        fn clear(&self, list: &HostList) {
            // SAFETY: an open list; the view names our live target.
            unsafe {
                let rtv = self.views.GetCPUDescriptorHandleForHeapStart();
                list.list.ClearRenderTargetView(rtv, &[0.0f32; 4], None);
            }
        }

        /// Binds the pipeline and its target, as `prepare` binds a
        /// title's draw.
        pub(crate) fn bind(&self, list: &HostList) {
            let side = SIDE as f32;
            // SAFETY: an open list; our live objects, alive until it has run.
            unsafe {
                let rtv = self.views.GetCPUDescriptorHandleForHeapStart();
                let l = &list.list;
                l.SetGraphicsRootSignature(&self.root);
                l.SetPipelineState(&self.pso);
                l.OMSetRenderTargets(1, Some(&rtv), false, None);
                l.RSSetViewports(&[D3D12_VIEWPORT {
                    TopLeftX: 0.0,
                    TopLeftY: 0.0,
                    Width: side,
                    Height: side,
                    MinDepth: 0.0,
                    MaxDepth: 1.0,
                }]);
                l.RSSetScissorRects(&[RECT {
                    left: 0,
                    top: 0,
                    right: SIDE as i32,
                    bottom: SIDE as i32,
                }]);
                l.IASetPrimitiveTopology(
                    windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
                );
            }
        }

        /// Runs `list` and reads the target back: RGBA by pixel, rows top
        /// first.
        fn read(&self, host: &Host, list: &TitleList) -> Vec<[u8; 4]> {
            use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM;
            const PITCH: u32 = 256;
            let (buffer, data) = host
                .shared_buffer(u64::from(PITCH * SIDE))
                .expect("readback");
            let l = list.host();
            // SAFETY: an open list; our live target and buffer, the
            // footprint inside the buffer.
            unsafe {
                l.list.ResourceBarrier(&[crate::host::transition(
                    &self.target,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                let to = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&buffer),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                            Offset: 0,
                            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                                Width: SIDE,
                                Height: SIDE,
                                Depth: 1,
                                RowPitch: PITCH,
                            },
                        },
                    },
                };
                let from = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&self.target),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        SubresourceIndex: 0,
                    },
                };
                l.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
            }
            list.run();
            (0..SIDE * SIDE)
                .map(|i| {
                    let at = (i / SIDE * PITCH + i % SIDE * 4) as usize;
                    // SAFETY: the list has run; inside the mapped buffer.
                    unsafe { data.add(at).cast::<[u8; 4]>().read() }
                })
                .collect()
        }
    }

    /// The pixel at (`x`, `y`) of what [`Painter::read`] read.
    fn pixel(pixels: &[[u8; 4]], x: u32, y: u32) -> [u8; 4] {
        pixels[(y * SIDE + x) as usize]
    }

    /// A title buffer of 256 bytes holding `bytes`.
    fn title_bytes(bytes: &[u8]) -> TitleBuffer {
        let buffer = TitleBuffer::new(256);
        buffer.set_words(|i| {
            let mut w = [0u8; 4];
            for (k, b) in w.iter_mut().enumerate() {
                *b = bytes.get(i * 4 + k).copied().unwrap_or(0);
            }
            u32::from_le_bytes(w)
        });
        let _copy = buffer.shadow();
        buffer
    }

    /// Title draw arguments holding `records`.
    fn title_records(records: &[[u32; 4]]) -> TitleBuffer {
        let bytes: Vec<u8> = records
            .iter()
            .flatten()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        title_bytes(&bytes)
    }

    /// A [`Painter`] pipeline's vertex inputs, as the host pipeline lists
    /// them.
    fn painter_inputs() -> [crate::pipeline::host::Input; 2] {
        let input = |slot, format| crate::pipeline::host::Input {
            slot,
            offset: 0,
            format,
            per_instance: false,
        };
        [input(0, 16), input(1, 28)]
    }

    /// The vertex buffers of a rectangle in each of a [`Painter`]'s four
    /// cells, filling it: positions (top left, bottom left, bottom right)
    /// and colors ([`COLORS`]).
    fn cell_rectangles() -> (TitleBuffer, TitleBuffer, [VertexView; 2]) {
        let (mut positions, mut colors) = (Vec::new(), Vec::new());
        for (cell, color) in COLORS.iter().enumerate() {
            let x = -1.0 + 0.5 * cell as f32;
            for (px, py) in [(x, 1.0f32), (x, -1.0), (x + 0.5, -1.0)] {
                positions.extend(px.to_le_bytes());
                positions.extend(py.to_le_bytes());
                colors.extend(color);
            }
        }
        let (p, c) = (title_bytes(&positions), title_bytes(&colors));
        let views = [
            VertexView {
                address: p.address(),
                size: positions.len() as u32,
                stride: 8,
            },
            VertexView {
                address: c.address(),
                size: colors.len() as u32,
                stride: 4,
            },
        ];
        (p, c, views)
    }

    #[test]
    fn missing_first_target_preserves_the_second_slot() {
        use crate::resource::tests_support::{committed, read_back, target};
        let test = "draw::tests::missing_first_target_preserves_the_second_slot";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let null = std::ptr::null();

        use xtp_shader::{
            HwStage,
            emit::{HostShader, rect_gs::probe},
        };
        let shader = |name, stage, dxil| {
            crate::pipeline::test_shader(
                name,
                HostShader {
                    stage,
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
            )
        };
        let vs = probe::vertex(
            &[[-1., -1., 0., 1.], [-1., 3., 0., 1.], [3., -1., 0., 1.]],
            &[[1.; 4]; 3],
            &[1; 3],
        );
        let mut graphics = crate::pipeline::host::GraphicsState::default();
        graphics.topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        graphics.targets.NumRenderTargets = 2;
        graphics.targets.RTFormats[0] = DXGI_FORMAT(10);
        graphics.targets.RTFormats[1] = DXGI_FORMAT(28);
        // Désactive le test de profondeur de cette sonde couleur.
        #[repr(C)]
        struct Depth(u32, D3D12_DEPTH_STENCIL_DESC);
        let depth = Depth(
            11,
            D3D12_DEPTH_STENCIL_DESC {
                DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
                ..Default::default()
            },
        );
        // SAFETY: sous-objet complet dans le format du flux.
        unsafe { graphics.read(11, (&raw const depth).cast()) };
        let pipeline = crate::pipeline::test_graphics_pipeline(
            vec![
                shader("vertex", HwStage::Vs, vs),
                shader("pixel", HwStage::Ps, probe::pixel()),
            ],
            graphics,
            &[0, 1],
        );
        let painter = Painter::new(host, false);
        for typed in [false, true] {
            let title = committed(&target(u64::from(SIDE), SIDE, 28));
            let tex = crate::resource::host_resource(title).unwrap();
            let list = TitleList::new();
            let mut records = [[0u64; 16]; 2];
            let handles = records.each_mut().map(|slot| slot.as_mut_ptr() as u64);
            crate::descriptor::create_rtv(null, title, null.cast(), handles[1]);
            if typed {
                crate::descriptor::create_rtv(null, null, &[28, 4, 0, 0, 0, 0], handles[0]);
            }
            crate::list::ensure_state(list.host(), title, 4);
            // SAFETY: an open list and live RGBA8 target.
            unsafe {
                list.host()
                    .list
                    .ClearRenderTargetView(host.rtv(tex, None), &[0.; 4], None);
            }
            painter.bind(list.host());
            crate::testgpu::bind_indexed_heaps(list.host());
            let mut state = State::default();
            // Les deux ordres de liaison doivent sélectionner la même variante.
            if typed {
                state.set_pipeline(pipeline);
            }
            state.set_targets(list.host(), &handles, 0);
            state.set_pipeline(pipeline);
            state
                .prepare(list.host(), false)
                .expect("le dessin est prêt");
            // SAFETY: le pipeline et les vues sont ceux sélectionnés par la préparation réelle.
            unsafe { list.host().list.DrawInstanced(3, 1, 0, 0) };
            list.run();
            let image = read_back(host, tex, 4);
            crate::object::slot::<extern "system" fn(This) -> u32>(title, 0x10)(title);
            assert!(
                image
                    .as_chunks::<256>()
                    .0
                    .iter()
                    .all(|row| row[..SIDE as usize * 4].iter().all(|&v| v == 0)),
                "slot 1 overwritten (typed null: {typed})"
            );
        }
        crate::object::slot::<extern "system" fn(This) -> u32>(pipeline, 0x10)(pipeline);
    }

    /// A depth plane's pass of ours before a dispatch (`State::after_our_pass`),
    /// under a compute pipeline, sets the title's targets again, a missing
    /// one as a null view in a format of no graphics pipeline's: the next
    /// draw, under the graphics pipeline that had them, makes them again
    /// for its own formats (R16G16B16A16 for slot 0 here, where a null
    /// view made for no pipeline is RGBA8). Before, the draw's cache still
    /// named its pipeline and kept the pass's views (Codex review of
    /// 05db757, point 6).
    #[test]
    fn missing_targets_are_made_again_for_the_draw_after_our_pass() {
        use crate::resource::tests_support::{committed, target};
        let test = "draw::tests::missing_targets_are_made_again_for_the_draw_after_our_pass";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let null = std::ptr::null();
        use xtp_shader::{
            HwStage,
            emit::{HostShader, rect_gs::probe},
        };
        let shader = |name, stage, dxil| {
            crate::pipeline::test_shader(
                name,
                HostShader {
                    stage,
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
            )
        };
        let vs = probe::vertex(
            &[[-1., -1., 0., 1.], [-1., 3., 0., 1.], [3., -1., 0., 1.]],
            &[[1.; 4]; 3],
            &[1; 3],
        );
        let mut graphics = crate::pipeline::host::GraphicsState::default();
        graphics.topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        graphics.targets.NumRenderTargets = 2;
        graphics.targets.RTFormats[0] = DXGI_FORMAT(10);
        graphics.targets.RTFormats[1] = DXGI_FORMAT(28);
        let pipeline = crate::pipeline::test_graphics_pipeline(
            vec![
                shader("vertex", HwStage::Vs, vs),
                shader("pixel", HwStage::Ps, probe::pixel()),
            ],
            graphics,
            &[0, 1],
        );
        let title = committed(&target(u64::from(SIDE), SIDE, 28));
        let list = TitleList::new();
        let mut records = [[0u64; 16]; 2];
        let handles = records.each_mut().map(|slot| slot.as_mut_ptr() as u64);
        // Slot 0 has no view of the title's: missing.
        crate::descriptor::create_rtv(null, title, null.cast(), handles[1]);
        let mut state = State::default();
        state.set_pipeline(pipeline);
        state.set_targets(list.host(), &handles, 0);
        let saved = |state: &State| state.incomplete_targets.as_ref().map(|s| s.pipeline);
        assert_eq!(saved(&state), Some(pipeline as usize));
        // A dispatch's pipeline (none of graphics), and our pass before it.
        state.set_pipeline(null);
        state.after_our_pass(list.host());
        assert_ne!(
            saved(&state),
            Some(pipeline as usize),
            "the draw under the pipeline would keep the pass's views"
        );
        drop(list);
        crate::object::slot::<extern "system" fn(This) -> u32>(title, 0x10)(title);
        crate::object::slot::<extern "system" fn(This) -> u32>(pipeline, 0x10)(pipeline);
    }

    /// Régression de la veille 343 : deux triangles passent dans la même série.
    /// Une copie met à zéro le compteur ou le nombre de sommets avant le troisième.
    #[test]
    fn a_copy_between_two_indirect_executions_of_the_same_run() {
        let test = "draw::tests::a_copy_between_two_indirect_executions_of_the_same_run";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let signature = host_signature(0, 24).unwrap();
        let zero = title_bytes(&0u32.to_le_bytes());
        for copy_to_count in [true, false] {
            let painter = Painter::new(host, false);
            let args = title_records(&[[3, 1, 0, 0]]);
            let count = title_bytes(&1u32.to_le_bytes());
            let (a, c) = (args.shadow(), count.shadow());
            let title = TitleList::new();
            let list = title.host();
            a.refresh(list, args.address(), 16);
            c.refresh(list, count.address(), 4);
            painter.clear(list);
            for execution in 0..3 {
                if execution == 2 {
                    let dst = if copy_to_count {
                        count.address()
                    } else {
                        args.address()
                    };
                    assert!(crate::shadow::copy(list, dst, zero.address(), 4));
                }
                let bounded = crate::indirect::bound(
                    list,
                    (&a.buffer, a.offset(args.address())),
                    (0, 16, 1),
                    Some((&c.buffer, c.offset(count.address()))),
                    None,
                    list.indirect_writes(&a.buffer, Some(&c.buffer)),
                    crate::indirect::Output::Draw,
                )
                .unwrap();
                assert_eq!(bounded.dispatched, execution != 1);
                painter.bind(list);
                // SAFETY: la signature de dessin lit la tranche et son compteur bornés.
                unsafe {
                    // Même triangle, décalé dans une cellule propre à cet appel.
                    list.list.RSSetViewports(&[D3D12_VIEWPORT {
                        TopLeftX: (execution * SIDE / 4) as f32,
                        Width: SIDE as f32,
                        Height: SIDE as f32,
                        MaxDepth: 1.0,
                        ..Default::default()
                    }]);
                    list.list.ExecuteIndirect(
                        &signature,
                        1,
                        &bounded.buffer,
                        bounded.offset + 4,
                        &bounded.buffer,
                        bounded.offset,
                    );
                }
            }
            let pixels = painter.read(host, &title);
            for cell in 0..3 {
                let expected = if cell < 2 { [255; 4] } else { CLEAR };
                assert_eq!(
                    pixel(&pixels, cell * SIDE / 4, SIDE - 1),
                    expected,
                    "triangle {cell}, copie dans le compteur = {copy_to_count}"
                );
            }
        }
    }

    /// An indirect rectangle draw end to end, as `indirect_rectangles`
    /// records it: the expansion, the draw's own state bound again after
    /// the passes (here the test pipeline), the expanded vertices of two
    /// slots, and `ExecuteIndirect` with stride 16 at offset 16 and the
    /// list's record count. Records: two rectangles from vertex 0 (cells 0
    /// and 1), one from vertex 6 (cell 2), one from vertex 9 past the
    /// title's count (2): every rectangle fills its cell (its fourth corner
    /// made) but the last.
    #[test]
    fn indirect_rectangles_draw_from_the_lists_arguments() {
        let _pool = crate::rectangles::POOL_TESTS
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let test = "draw::tests::indirect_rectangles_draw_from_the_lists_arguments";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let painter = Painter::new(host, true);
        let (_positions, _colors, views) = cell_rectangles();
        let args = title_records(&[[6, 1, 0, 0], [3, 1, 6, 0], [3, 1, 9, 0]]);
        let count = title_bytes(&2u32.to_le_bytes());
        let (args_copy, count_copy) = (args.shadow(), count.shadow());
        let list = TitleList::new();
        let mut state = State::default();
        state.set_vertex_buffers(0, &views);
        // As a dispatch before would leave them.
        (state.host_pipeline, state.host_roots[1]) = (1, 1);
        painter.clear(list.host());
        state
            .rectangles_indirect(
                list.host(),
                &painter_inputs(),
                (16, 3),
                (
                    Records::Title(&args_copy, args.address()),
                    Some((&count_copy, count.address())),
                    None,
                ),
                |_, l| {
                    painter.bind(l);
                    Ok(painter.pso.clone())
                },
            )
            .expect("drawn");
        let pixels = painter.read(host, &list);
        for cell in 0..4 {
            let want = if cell < 3 {
                COLORS[cell as usize]
            } else {
                CLEAR
            };
            let x = cell * SIDE / 4;
            assert_eq!(
                pixel(&pixels, x, SIDE - 1),
                want,
                "cell {cell}'s first triangle"
            );
            assert_eq!(
                pixel(&pixels, x + SIDE / 4 - 1, 0),
                want,
                "cell {cell}'s fourth corner"
            );
        }
    }

    #[test]
    fn query_rectangle_batches_restore_the_graphics_pipeline() {
        let _pool = crate::rectangles::POOL_TESTS
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let test = "draw::tests::query_rectangle_batches_restore_the_graphics_pipeline";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let painter = Painter::new(host, true);
        let (positions, colors, mut views) = cell_rectangles();
        // Chaque commande peut produire 12 Mio de sommets. Le budget de 32 Mio
        // impose deux lots de deux commandes, chacun couvrant deux cellules.
        let padded = [positions, colors]
            .into_iter()
            .zip(&mut views)
            .map(|(source, view)| {
                view.size = (1 << 19) * view.stride;
                let buffer = TitleBuffer::new(view.size as usize);
                let words = source.words();
                buffer.set_words(|i| words.get(i).copied().unwrap_or(0));
                view.address = buffer.address();
                buffer
            })
            .collect::<Vec<_>>();
        let args = title_records(&std::array::from_fn::<_, 4, _>(|i| [3, 1, i as u32 * 3, 0]));
        let title = TitleList::new();
        let list = title.host();
        let mut state = State::default();
        state.set_vertex_buffers(0, &views);
        painter.clear(list);
        state
            .rectangles_indirect(
                list,
                &painter_inputs(),
                (16, 4),
                (Records::Title(&args.shadow(), args.address()), None, None),
                |_, l| {
                    painter.bind(l);
                    Ok(painter.pso.clone())
                },
            )
            .unwrap();
        let pixels = painter.read(host, &title);
        for (cell, color) in COLORS.iter().enumerate() {
            let x = cell as u32 * SIDE / 4;
            assert_eq!(&pixel(&pixels, x, SIDE - 1), color);
            assert_eq!(&pixel(&pixels, x + SIDE / 4 - 1, 0), color);
        }
        drop(padded);
    }

    /// A pipeline without vertex inputs draws each rectangle's first
    /// triangle, without its fourth corner: one rectangle from vertex 6 and
    /// two from vertex 9. The vertex ids count from 0 in each record, so
    /// they are cells 0, then 0 and 1: what the console draws. D3D12's
    /// `SV_VertexID` leaves out the start vertex, and so does the console's
    /// v0; the start reaches a console program only through a user SGPR,
    /// which none of the title's vertex programs read (docs/analysis/drops.md
    /// §5.1.1). They drew twice the vertices before: cells 0 to 3.
    #[test]
    fn indirect_rectangles_without_vertex_inputs_draw_their_first_triangles() {
        let _pool = crate::rectangles::POOL_TESTS
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let test =
            "draw::tests::indirect_rectangles_without_vertex_inputs_draw_their_first_triangles";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let painter = Painter::new(host, false);
        let args = title_records(&[[3, 1, 6, 0], [6, 1, 9, 0]]);
        let args_copy = args.shadow();
        let list = TitleList::new();
        let mut state = State::default();
        painter.clear(list.host());
        state
            .rectangles_indirect(
                list.host(),
                &[],
                (16, 2),
                (Records::Title(&args_copy, args.address()), None, None),
                |_, l| {
                    painter.bind(l);
                    Ok(painter.pso.clone())
                },
            )
            .expect("drawn");
        let pixels = painter.read(host, &list);
        for cell in 0..4 {
            let x = cell * SIDE / 4;
            let first = if cell < 2 { [255; 4] } else { CLEAR };
            assert_eq!(
                pixel(&pixels, x, SIDE - 1),
                first,
                "cell {cell}'s first triangle"
            );
            assert_eq!(
                pixel(&pixels, x + SIDE / 4 - 1, 0),
                CLEAR,
                "cell {cell}: no fourth corner"
            );
        }
    }

    /// A list whose only work is an indirect rectangle draw keeps its
    /// scratch until its recording has gone: the title may reset the list
    /// (its state goes) while the GPU still runs it. The scratch went back
    /// to the pool at that reset before, for later lists to write.
    #[test]
    fn a_lists_rectangle_scratch_waits_for_its_recording() {
        let _pool = crate::rectangles::POOL_TESTS
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let test = "draw::tests::a_lists_rectangle_scratch_waits_for_its_recording";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let (_positions, _colors, views) = cell_rectangles();
        let args = title_records(&[[3, 1, 0, 0]]);
        let args_copy = args.shadow();
        let list = TitleList::new();
        let mut state = State::default();
        state.set_vertex_buffers(0, &views);
        let prepared = state
            .expand_rectangles(
                list.host(),
                &painter_inputs(),
                (16, 0..1),
                (Records::Title(&args_copy, args.address()), None, None),
                &mut Default::default(),
            )
            .expect("expanded");
        let scratch = prepared.args.clone();
        drop(prepared);
        let held = state
            .own
            .rects
            .as_ref()
            .and_then(|s| s.arguments().cloned());
        assert_eq!(held.as_ref(), Some(&scratch), "the list's scratch");
        state.reset(std::ptr::null());
        assert!(
            !crate::rectangles::pooled(&scratch),
            "held by the recording"
        );
        list.host().close();
        list.host().reset();
        assert!(
            crate::rectangles::pooled(&scratch),
            "given back once the recording has gone"
        );
    }

    /// Root arguments of a stage last until the title sets another root
    /// signature for that stage: setting the same one keeps them, another
    /// one (null included) leaves them unset, and the other stage's stay.
    /// Before, the root signature calls did nothing and draws read
    /// arguments the title had set for an earlier signature.
    #[test]
    fn changing_root_signatures_invalidates_only_the_matching_bindings() {
        let mut state = State::default();
        let first = std::ptr::without_provenance(1);
        let second = std::ptr::without_provenance(2);
        state.set_root_signature(false, first);
        state.set_root_signature(true, first);
        state.set_table(false, 0, 0x1000);
        state.set_view(false, 1, 0x2000);
        state.set_constants(false, 2, &[42], 0);
        state.set_constants(true, 0, &[7], 0);

        state.set_root_signature(false, first);
        assert!(matches!(&state.graphics[..], [
            Root::Table(0x1000), Root::View(0x2000), Root::Constants(values)
        ] if values == &[42]));

        state.set_root_signature(false, second);
        assert!(state.graphics.is_empty());
        assert!(matches!(&state.compute[..], [Root::Constants(values)] if values == &[7]));

        state.set_root_signature(true, std::ptr::null());
        assert!(state.compute.is_empty());
    }

    /// A spliced program's dispatch finds the title's view of its structure
    /// at the descriptor's offset in the table bound at its root parameter,
    /// in a bound heap: none without a table there (another stage's, root
    /// constants), and none past the heap's end.
    #[test]
    fn the_tlas_view_is_in_the_table_the_walk_read() {
        let heap = TestHeap::views(16);
        let title = crate::descriptor::heap(heap.0);
        let mut state = state(heap.0);
        let d = xtp_shader::emit::TlasDescriptor {
            slot: 2,
            offset: 0x20,
        };
        assert_eq!(state.tlas_view_at(d, true), None);
        state.set_table(true, 2, title.start + 3 * 32);
        assert_eq!(state.tlas_view_at(d, true), Some(title.start + 4 * 32));
        assert_eq!(state.tlas_view_at(d, false), None);
        state.set_table(true, 2, title.start + 15 * 32);
        assert_eq!(state.tlas_view_at(d, true), None);
        state.set_constants(true, 2, &[1], 0);
        assert_eq!(state.tlas_view_at(d, true), None);
    }

    /// The host device for `test` if it runs inline ray queries (DXR tier
    /// 1.1); a skip otherwise.
    fn ray_query_gpu(test: &str) -> Option<crate::testgpu::Gpu> {
        let gpu = crate::testgpu::device(test)?;
        let host = gpu.host();
        let mut caps = D3D12_FEATURE_DATA_D3D12_OPTIONS5::default();
        // SAFETY: the output is the options structure the feature names.
        let tier = unsafe {
            host.device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS5,
                (&raw mut caps).cast(),
                size_of_val(&caps) as u32,
            )
        }
        .map_or(D3D12_RAYTRACING_TIER_NOT_SUPPORTED, |()| {
            caps.RaytracingTier
        });
        if tier.0 < D3D12_RAYTRACING_TIER_1_1.0 {
            crate::testgpu::skip(test, "inline ray queries require DXR tier 1.1");
            return None;
        }
        Some(gpu)
    }

    /// One triangle about x = `x` at z = 0, built by `producer` as a title
    /// TLAS at `at` (not submitted): of `ray_test.hlsl`'s two rays along +z
    /// from z = -2, at x = 0 and x = 4, the one at `x` hits it two units on.
    /// Its inputs, which must outlive the build.
    fn tlas_scene(producer: &TitleList, x: f32, at: u64) -> Vec<TitleBuffer> {
        // Plain DXR formats: the console builder reads R16_TYPELESS indices
        // as meshlet-compressed and R32G32_FLOAT vertices as quantized.
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_R16_UINT, DXGI_FORMAT_R32G32B32_FLOAT,
        };
        let (vertices, indices) = (TitleBuffer::new(36), TitleBuffer::new(8));
        let positions = [x - 1., -1., 0., x, 1., 0., x + 1., -1., 0.];
        vertices.set_words(|i| positions[i].to_bits());
        indices.set_words(|i| [1 << 16, 2][i]);
        let (blas, instances) = (TitleBuffer::new(1024), TitleBuffer::new(64));
        instances.set_words(|i| match i {
            0 | 5 | 10 => 1f32.to_bits(),
            12 => 0xff00_0000,
            14 => blas.address() as u32,
            15 => (blas.address() >> 32) as u32,
            _ => 0,
        });
        let geometry = D3D12_RAYTRACING_GEOMETRY_DESC {
            Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
            Flags: D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
            Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                    IndexFormat: DXGI_FORMAT_R16_UINT,
                    VertexFormat: DXGI_FORMAT_R32G32B32_FLOAT,
                    IndexCount: 3,
                    VertexCount: 3,
                    IndexBuffer: indices.address(),
                    VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                        StartAddress: vertices.address(),
                        StrideInBytes: 12,
                    },
                    ..Default::default()
                },
            },
        };
        let inputs = |kind, descs| D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: kind,
            NumDescs: 1,
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
            Anonymous: descs,
            ..Default::default()
        };
        let blas_inputs = inputs(
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                pGeometryDescs: &geometry,
            },
        );
        let tlas_inputs = inputs(
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                InstanceDescs: instances.address(),
            },
        );
        // SAFETY: the title list and the CPU arrays are live for each call.
        unsafe {
            for (dst, inputs) in [(blas.address(), blas_inputs), (at, tlas_inputs)] {
                assert!(crate::rt::Builder.build(
                    producer.this(),
                    &D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                        DestAccelerationStructureData: dst,
                        Inputs: inputs,
                        ..Default::default()
                    }
                ));
            }
        }
        vec![vertices, indices, blas, instances]
    }

    /// Which TLAS a spliced trace reads. The title's table at the trace's
    /// root parameter holds a decoy's descriptor at +0x00 and the trace's at
    /// +0x20 ([`xtp_shader::emit::TlasDescriptor`]), and the trace's names a
    /// TLAS the title built 0x100 bytes into its buffer. The binding
    /// ([`State::tlas_view_at`], then `rt::descriptor_tlas_in`) gives the
    /// host TLAS of that title address, the one its root address gives, not
    /// the decoy's: the same rays traced through either give the same words,
    /// and other words through the decoy.
    ///
    /// After the test of Xbox2Windows commit 1b753c5828 by tedin7, which
    /// traces rays through a TLAS published at a non-zero offset, against
    /// the root address and a decoy (MIT License, Copyright (c) 2026
    /// Xbox2Windows contributors); written anew here, none of its code.
    #[test]
    fn a_trace_reads_the_tlas_its_descriptor_names_not_a_decoy() {
        let test = "draw::tests::a_trace_reads_the_tlas_its_descriptor_names_not_a_decoy";
        let Some(gpu) = ray_query_gpu(test) else {
            return;
        };
        let host = gpu.host();
        let producer = TitleList::new();
        let scene = |x: f32, at: u64| tlas_scene(&producer, x, at);
        // The trace's TLAS 0x100 bytes into the title's buffer; the decoy's.
        let (placed, decoy) = (TitleBuffer::new(4096), TitleBuffer::new(1024));
        let at = placed.address() + 0x100;
        let _inputs = [scene(0., at), scene(4., decoy.address())];
        producer.run();
        // The title's table: the decoy's descriptor, then the trace's.
        let heap = TestHeap::views(16);
        let title = crate::descriptor::heap(heap.0);
        for (k, address) in [decoy.address(), at].into_iter().enumerate() {
            let mut srv = [0u32; 10];
            srv[1] = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE.0 as u32;
            srv[4] = address as u32;
            srv[5] = (address >> 32) as u32;
            let slot = title.start + k as u64 * 32;
            crate::descriptor::create_srv(std::ptr::null(), std::ptr::null(), &srv, slot);
        }
        let list = TitleList::new();
        let mut state = state(heap.0);
        state.set_table(true, 1, title.start);
        let d = xtp_shader::emit::TlasDescriptor {
            slot: 1,
            offset: 0x20,
        };
        let view = state.tlas_view_at(d, true).expect("the trace's view");
        assert_eq!(view, title.start + 0x20);
        let bound = crate::rt::descriptor_tlas_in(list.host(), view).expect("its host TLAS");
        // SAFETY: the test's live list.
        let (root, other) = unsafe {
            (
                crate::rt::host_tlas(list.this(), at).expect("the root address's TLAS"),
                crate::rt::host_tlas(list.this(), decoy.address()).expect("the decoy's"),
            )
        };
        assert_eq!(bound, root);
        assert_ne!(bound, other);
        // The same rays through each.
        let dxc = std::env::var_os("XTP_PC_DXC")
            .map(std::path::PathBuf::from)
            .or_else(|| {
                xtp_shader::emit::validator::Validator::default_path()
                    .map(|p| p.with_file_name("dxc.exe"))
            })
            .expect("Windows SDK DXC for the inline ray test");
        let dir = tempfile::tempdir().unwrap();
        let binary = dir.path().join("ray.dxil");
        let compiled = std::process::Command::new(dxc)
            .args(["-T", "cs_6_5", "-E", "main", "-Fo"])
            .arg(&binary)
            .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/src/rt/ray_test.hlsl"))
            .output()
            .unwrap();
        assert!(
            compiled.status.success(),
            "{}",
            String::from_utf8_lossy(&compiled.stderr)
        );
        let code = std::fs::read(binary).unwrap();
        let l = list.host();
        let outputs: Vec<TitleBuffer> = (0..3).map(|_| TitleBuffer::new(16)).collect();
        // SAFETY: the root signature is the one DXC embeds; the list keeps it,
        // the PSO and the outputs' copies, and the structures are pinned.
        unsafe {
            let root_signature: ID3D12RootSignature =
                host.device.CreateRootSignature(0, &code).unwrap();
            let pso: ID3D12PipelineState = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root_signature),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.as_ptr().cast(),
                        BytecodeLength: code.len(),
                    },
                    ..Default::default()
                })
                .unwrap();
            l.keep(&root_signature);
            l.keep(&pso);
            for (tlas, output) in [bound, root, other].into_iter().zip(&outputs) {
                output.set_words(|_| 77);
                let shadow = output.shadow();
                shadow.refresh(l, output.address(), 16);
                shadow.gpu_writes(l, output.address(), 16);
                l.list.SetPipelineState(&pso);
                l.list.SetComputeRootSignature(&root_signature);
                l.list.SetComputeRootShaderResourceView(0, tlas);
                l.list
                    .SetComputeRootUnorderedAccessView(1, shadow.gpu(output.address()));
                l.list.Dispatch(1, 1, 1);
                l.list.ResourceBarrier(&[crate::host::uav_barrier()]);
            }
        }
        list.run();
        let [through_table, through_root, through_decoy] = [0, 1, 2].map(|k| outputs[k].words());
        let two = 2.0f32.to_bits();
        assert_eq!(through_table, [1, two, 0, 0], "the trace's rays");
        assert_eq!(through_table, through_root);
        assert_eq!(through_decoy, [0, 0, 1, two], "the decoy's rays");
    }

    /// A trace whose TLAS the title builds in another list, not submitted
    /// yet when the trace is recorded, takes it at its submission
    /// ([`State::tlas_at_submission`], which `State::dispatch` hands to
    /// `rt::dispatch_descriptor_tlas_in`); once that list is submitted, the
    /// trace binds it as it is recorded, a root SRV. A draw has no such way
    /// round: it is refused.
    #[test]
    fn a_trace_takes_a_tlas_not_yet_submitted_at_its_submission() {
        let test = "draw::tests::a_trace_takes_a_tlas_not_yet_submitted_at_its_submission";
        let Some(_gpu) = ray_query_gpu(test) else {
            return;
        };
        let producer = TitleList::new();
        let tlas = TitleBuffer::new(1024);
        let _inputs = tlas_scene(&producer, 0., tlas.address());
        let heap = TestHeap::views(16);
        let title = crate::descriptor::heap(heap.0);
        let mut srv = [0u32; 10];
        srv[1] = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE.0 as u32;
        srv[4] = tlas.address() as u32;
        srv[5] = (tlas.address() >> 32) as u32;
        crate::descriptor::create_srv(std::ptr::null(), std::ptr::null(), &srv, title.start);
        let list = TitleList::new();
        let mut state = state(heap.0);
        state.pending_compute = Some(Vec::new());
        state
            .bind_tlas_view(list.host(), title.start, 5, true)
            .unwrap();
        assert_eq!(state.tlas_at_submission, Some((title.start, 5)));
        assert!(state.pending_compute.as_ref().unwrap().is_empty());
        assert_eq!(
            state.bind_tlas_view(list.host(), title.start, 5, false),
            Err(NO_HOST_TLAS)
        );
        producer.run();
        state.tlas_at_submission = None;
        state
            .bind_tlas_view(list.host(), title.start, 5, true)
            .unwrap();
        assert_eq!(state.tlas_at_submission, None);
        assert!(matches!(
            state.pending_compute.as_deref(),
            Some([(5, ComputeRoot::Srv(gpu))]) if *gpu != 0
        ));
    }

    /// A title heap released while a list records and another made at its
    /// address keep the key the state compares, not the host heap: the
    /// host list gets the new heap (and root signatures again after it),
    /// and what the recording kept by that key goes with the old heap, also
    /// when XTP's own work set other heaps in between. The same heap again
    /// sets nothing, and another heap's entries stay. Before, the key alone
    /// was compared: the host list kept the old heap.
    #[test]
    fn a_heap_made_again_at_its_address_is_bound_afresh() {
        let test = "draw::tests::a_heap_made_again_at_its_address_is_bound_afresh";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let list = TitleList::new();
        let l = list.host();
        let new_heap = || -> ID3D12DescriptorHeap {
            // SAFETY: a live device and a valid shader-visible view heap
            // description.
            unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        NumDescriptors: 4,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
            }
            .expect("a host heap")
        };
        // The title heap's address, and another heap's.
        let (key, other) = (0x1000, 0x2000);
        // What a recording keeps by heap.
        let keep = |state: &mut State| {
            for heap in [key, other] {
                state.root_srv_slots.insert((heap, 0, 0, 0), 1);
                state.root_uav_slots.insert((heap, 0, 0, 0), 2);
                state.picked.push(heap);
                state.heap_textured.push(heap);
            }
            state.fresh_tables.insert(
                (0x40, 0),
                FreshTable {
                    first: 3,
                    constants: vec![],
                },
            );
        };
        let only_other = |state: &State| {
            assert!(state.root_srv_slots.keys().all(|k| k.0 == other));
            assert!(state.root_uav_slots.keys().all(|k| k.0 == other));
            assert_eq!(state.root_srv_slots.len(), 1);
            assert_eq!(state.root_uav_slots.len(), 1);
            assert_eq!(
                (&state.picked[..], &state.heap_textured[..]),
                (&[other][..], &[other][..])
            );
            assert!(state.fresh_tables.is_empty());
        };
        let mut state = State::default();
        let original = new_heap();
        state.bind_host_heaps(l, [key, 0], [Some(&original), None]);
        keep(&mut state);
        state.host_roots = [1, 2];
        state.bind_host_heaps(l, [key, 0], [Some(&original), None]);
        assert_eq!(state.host_roots, [1, 2], "the same heap: nothing set");
        assert_eq!(state.picked, [key, other]);

        // The title releases its heap (the list holds the host heap) and
        // makes one at its address.
        drop(original);
        let replacement = new_heap();
        state.bind_host_heaps(l, [key, 0], [Some(&replacement), None]);
        assert_eq!(state.host_roots, [0; 2], "set again after the new heap");
        assert_eq!(state.host_heaps, [key, 0]);
        only_other(&state);

        // Again, with XTP's own heaps set in between.
        state.picked.clear();
        state.heap_textured.clear();
        keep(&mut state);
        state.forget_host();
        drop(replacement);
        let third = new_heap();
        state.bind_host_heaps(l, [key, 0], [Some(&third), None]);
        only_other(&state);
    }
}
