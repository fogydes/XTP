//! Textures shaders write that a title's CPU code reads back.
//!
//! On the console the CPU and GPU share memory, so what a compute pass writes
//! into a texture through an unordered-access view is immediately visible in
//! title memory. On Windows it stays in the host texture copy unless copied
//! back.
//!
//! For textures matching a configured kind (`(width, height, dxgi_format)`),
//! configured by `XTP_TEXTURE_BACK` (comma-separated `<width>x<height>f<format>`,
//! `1` for the build's defaults, `0`/`off`/`false` to disable; unset uses the
//! build's defaults supplied via [`configure_texture_readback`], or off if
//! none), a list that binds or clears one as a UAV copies it into a readback
//! buffer when the list closes. Once the submission completes, the linear
//! texels are tiled back into the texture's title memory in the console's
//! layout using [`xtp_formats::texture::Layout::tile`].
//!
//! Because UAV-writable textures carry `ALLOW_UNORDERED_ACCESS`,
//! [`crate::texture::title_written`] is false for them and their aliases, so
//! landing bytes into title memory never triggers an echo re-upload as a CPU
//! write.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use windows::Win32::Graphics::Direct3D12::*;

use crate::object::This;

/// Per-build default texture kinds `(width, height, dxgi_format)`, supplied
/// before GPU work starts (for example by `xtp-game` through the launcher).
static DEFAULT_KINDS: OnceLock<Vec<(u64, u32, u32)>> = OnceLock::new();

/// Sets the default texture kinds read back when `XTP_TEXTURE_BACK` is unset
/// or `1`. Must be called before the first UAV texture bind; later calls are
/// ignored.
pub fn configure_texture_readback(defaults: &[(u64, u32, u32)]) {
    let _ = DEFAULT_KINDS.set(defaults.to_vec());
}

fn parse_kinds(spec: &str, defaults: &[(u64, u32, u32)]) -> Vec<(u64, u32, u32)> {
    let trimmed = spec.trim();
    if trimmed.is_empty()
        || trimmed == "0"
        || trimmed.eq_ignore_ascii_case("off")
        || trimmed.eq_ignore_ascii_case("false")
    {
        return Vec::new();
    }
    if trimmed == "1" || trimmed.eq_ignore_ascii_case("on") || trimmed.eq_ignore_ascii_case("true")
    {
        return defaults.to_vec();
    }
    let mut kinds = Vec::new();
    for raw in trimmed.split(',') {
        let entry = raw.trim();
        if entry.is_empty() {
            continue;
        }
        let parsed = entry.split_once(['f', 'F']).and_then(|(size, format)| {
            let (w, h) = size.split_once(['x', 'X'])?;
            let w: u64 = w.trim().parse().ok()?;
            let h: u32 = h.trim().parse().ok()?;
            let format: u32 = format.trim().parse().ok()?;
            (w > 0 && h > 0 && format > 0).then_some((w, h, format))
        });
        match parsed {
            Some(kind) => kinds.push(kind),
            None => tracing::warn!(
                entry,
                "ignoring malformed XTP_TEXTURE_BACK entry (expected <width>x<height>f<format>)"
            ),
        }
    }
    kinds
}

/// The kinds copied back: `(width, height, dxgi_format)`.
///
/// Reads `XTP_TEXTURE_BACK`: `0`/`off`/`false` disables readback, `1` or unset
/// uses the defaults from [`configure_texture_readback`] (empty if unset), and
/// a comma-separated `<width>x<height>f<format>` list overrides them.
fn kinds() -> &'static [(u64, u32, u32)] {
    static KINDS: OnceLock<Vec<(u64, u32, u32)>> = OnceLock::new();
    KINDS.get_or_init(|| {
        let defaults = DEFAULT_KINDS.get().map_or(&[][..], Vec::as_slice);
        let kinds = match std::env::var("XTP_TEXTURE_BACK") {
            Ok(spec) => parse_kinds(&spec, defaults),
            Err(_) => defaults.to_vec(),
        };
        if !kinds.is_empty() {
            tracing::info!(
                ?kinds,
                "textures of these kinds that shaders write are copied back into title memory (XTP_TEXTURE_BACK)"
            );
        }
        kinds
    })
}

#[cfg(test)]
thread_local! {
    static TEST_KINDS: std::cell::RefCell<Option<Vec<(u64, u32, u32)>>> =
        const { std::cell::RefCell::new(None) };
}

/// Whether title texture `resource` is of a kind copied back.
pub(crate) fn wanted(resource: This) -> bool {
    #[cfg(test)]
    if let Some(hit) = TEST_KINDS.with_borrow(|over| {
        over.as_ref().map(|kinds| {
            !kinds.is_empty()
                && crate::resource::texture_kind(resource).is_some_and(|k| kinds.contains(&k))
        })
    }) {
        return hit;
    }
    let kinds = kinds();
    !kinds.is_empty() && crate::resource::texture_kind(resource).is_some_and(|k| kinds.contains(&k))
}

/// An outstanding readback to a specific title texture allocation. Mapping and
/// resource lifetime changes invalidate only the readbacks that could touch
/// replaced or released memory.
struct Ticket {
    owner: usize,
    at: usize,
    end: usize,
    valid: AtomicBool,
}

static TICKETS: Mutex<Vec<Weak<Ticket>>> = Mutex::new(Vec::new());
static TRACKING: AtomicBool = AtomicBool::new(false);

/// Whether any texture readback ticket has been tracked. Allows hot paths
/// (`hand_out_resource`, `Resource::drop`, `title_memory_remapped`,
/// `HostList::close`) to skip work when readback is off or unused.
pub(crate) fn tracking() -> bool {
    TRACKING.load(Ordering::Acquire)
}

fn track(owner: usize, at: usize, end: usize) -> Arc<Ticket> {
    let ticket = Arc::new(Ticket {
        owner,
        at,
        end,
        valid: AtomicBool::new(true),
    });
    let mut tickets = TICKETS.lock().unwrap_or_else(|e| e.into_inner());
    tickets.retain(|weak| weak.strong_count() != 0);
    tickets.push(Arc::downgrade(&ticket));
    TRACKING.store(true, Ordering::Release);
    ticket
}

fn invalidate(matches: impl Fn(&Ticket) -> bool) {
    if !tracking() {
        return;
    }
    TICKETS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .retain(|weak| {
            let Some(ticket) = weak.upgrade() else {
                return false;
            };
            if matches(&ticket) {
                ticket.valid.store(false, Ordering::Release);
                return false;
            }
            true
        });
}

pub(crate) fn released(owner: usize) {
    invalidate(|ticket| ticket.owner == owner);
}

pub(crate) fn replaced(at: usize, len: usize) {
    let end = at.saturating_add(len.max(1));
    invalidate(|ticket| at < ticket.end && ticket.at < end);
}

pub(crate) fn remapped(at: usize, len: usize) {
    let end = at.saturating_add(len);
    invalidate(|ticket| at < ticket.end && ticket.at < end);
}

pub(crate) struct Target {
    pub(crate) copy: Arc<crate::resource::HostTexture>,
    layout: xtp_formats::texture::Layout,
    ticket: Arc<Ticket>,
}

impl Target {
    pub(crate) fn new(resource: This) -> Option<Self> {
        // Keep the host copy alive independently of the title resource lifetime.
        let copy = crate::resource::host_copy(resource)?;
        let (layout, at, size) = crate::resource::title_texture(resource)?;
        if size > isize::MAX as usize {
            return None;
        }
        let end = at.checked_add(size).filter(|&end| at != 0 && end > at)?;
        let ticket = track(resource as usize, at, end);
        Some(Self {
            copy,
            layout,
            ticket,
        })
    }

    pub(crate) fn owner(&self) -> usize {
        self.ticket.owner
    }

    pub(crate) fn is_valid(&self) -> bool {
        self.ticket.valid.load(Ordering::Acquire)
    }
}

/// The readback buffer and console layout for a texture copied at list close.
pub(crate) struct Back {
    buffer: ID3D12Resource,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    mips: u32,
    layout: xtp_formats::texture::Layout,
    ticket: Arc<Ticket>,
}

impl Back {
    pub(crate) fn range(&self) -> (usize, usize) {
        (self.ticket.at, self.ticket.end)
    }
}

/// Unmaps a readback buffer with an empty written range on drop.
struct MappedReadback<'a>(&'a ID3D12Resource);

impl Drop for MappedReadback<'_> {
    fn drop(&mut self) {
        let written = D3D12_RANGE { Begin: 0, End: 0 };
        // SAFETY: mapped by `land` for CPU reading only; zero bytes were written.
        unsafe { self.0.Unmap(0, Some(&written)) };
    }
}

/// Records copies of every subresource of `target` into a readback buffer.
/// Calls `prepare` (to transition the host texture to `COPY_SOURCE`) only once
/// validation and buffer allocation succeed.
pub(crate) fn record(
    list: &ID3D12GraphicsCommandList,
    target: Target,
    prepare: impl FnOnce(),
) -> Option<Back> {
    if !target.is_valid() {
        return None;
    }
    let host = crate::host::host()?;
    let texture = &target.copy.texture;
    // SAFETY: a live host texture held by `target.copy`.
    let desc = unsafe { texture.GetDesc() };
    if desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
        || desc.Width != u64::from(target.layout.desc.width)
        || desc.Height != target.layout.desc.height
        || desc.SampleDesc.Count != 1
    {
        return None;
    }
    // Reject stand-in host formats whose texel or block layout differs from
    // the title layout (tiling raw host bytes would corrupt title memory).
    let host_format = xtp_formats::texture::format(desc.Format.0 as u32)?;
    if host_format.stencil
        || host_format.bytes != target.layout.desc.element_bytes
        || host_format.block != target.layout.desc.block
    {
        return None;
    }
    let mips = u32::from(desc.MipLevels.max(1));
    let slices = u32::from(desc.DepthOrArraySize.max(1));
    if mips as usize != target.layout.mips.len() || slices != target.layout.slices {
        return None;
    }
    let subresources = mips.checked_mul(slices)?;
    let n = subresources as usize;
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n];
    let mut rows = vec![0u32; n];
    let mut total = 0u64;
    // SAFETY: out arrays hold `n == subresources` entries.
    unsafe {
        host.device.GetCopyableFootprints(
            &desc,
            0,
            subresources,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    if total == 0 || total == u64::MAX {
        return None;
    }
    // The last footprint's required size can omit row-end padding, while
    // `Layout::tile` reads pitched rows; make sure the buffer covers them.
    for (footprint, &rows) in footprints.iter().zip(&rows) {
        let format = footprint.Footprint;
        if format.Depth == 0 {
            return None;
        }
        let pitched = u64::from(format.RowPitch)
            .checked_mul(u64::from(rows))?
            .checked_mul(u64::from(format.Depth))?;
        total = total.max(footprint.Offset.checked_add(pitched)?);
    }
    if usize::try_from(total)
        .ok()
        .is_none_or(|n| n > isize::MAX as usize)
    {
        return None;
    }
    let buffer = host.read_back_buffer(total)?;
    prepare();
    for (i, footprint) in footprints.iter().enumerate() {
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(&buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *footprint,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: i as u32,
            },
        };
        // SAFETY: an open list; `prepare` moved the texture to `COPY_SOURCE`
        // and `buffer` starts in `COPY_DEST`.
        unsafe { list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    Some(Back {
        buffer,
        footprints,
        rows,
        mips,
        layout: target.layout,
        ticket: target.ticket,
    })
}

/// Once the submission has run, tiles the copied texels back into the same
/// title allocation. Returns the number of bytes written.
pub(crate) fn land(back: &Back) -> usize {
    static COPIED: AtomicU64 = AtomicU64::new(0);
    static STALE: AtomicU64 = AtomicU64::new(0);
    let log_stale = || {
        let skipped = STALE.fetch_add(1, Ordering::Relaxed) + 1;
        if skipped.is_power_of_two() {
            tracing::info!(
                skipped,
                "skipping texture readbacks whose title allocation changed"
            );
        }
    };
    if !back.ticket.valid.load(Ordering::Acquire) {
        log_stale();
        return 0;
    }
    let (at, end) = back.range();
    let size = end - at;
    let mut tiled = vec![0u8; size];
    // Preserve title bytes that the subresources do not cover (padding/tail).
    if !crate::host::read_into(at as *const u8, &mut tiled) {
        return 0;
    }
    let mut data = std::ptr::null_mut();
    // SAFETY: our readback buffer, written by the completed submission.
    unsafe {
        if back.buffer.Map(0, None, Some(&mut data)).is_err() || data.is_null() {
            return 0;
        }
    }
    let mapped = MappedReadback(&back.buffer);
    for (i, (footprint, &rows)) in back.footprints.iter().zip(&back.rows).enumerate() {
        let (mip, slice) = (i as u32 % back.mips, i as u32 / back.mips);
        let f = footprint.Footprint;
        let pitch = f.RowPitch as usize;
        let len = pitch * rows as usize * f.Depth as usize;
        // SAFETY: subresource slice within the mapped readback buffer bounds
        // checked when computing `total` (<= isize::MAX) in `record`.
        let linear = unsafe {
            std::slice::from_raw_parts(data.cast::<u8>().add(footprint.Offset as usize), len)
        };
        if let Err(e) =
            back.layout
                .tile(&mut tiled, mip, slice, linear, pitch, pitch * rows as usize)
        {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(?e, mip, slice, "a texture copied back could not be tiled")
            });
            return 0;
        }
    }
    drop(mapped);
    // Serialize the final write into title memory with resource release,
    // replacement, and remap invalidations without holding `TICKETS` across
    // `Map` and CPU tiling.
    let _ticket_guard = TICKETS.lock().unwrap_or_else(|e| e.into_inner());
    if !back.ticket.valid.load(Ordering::Acquire) {
        log_stale();
        return 0;
    }
    // `host::write_mapped` holds `kernel_writing_range` when `readmap` is on;
    // when it is off, hold one here so guarded view blocks stay open and
    // cannot be re-guarded by a concurrent refresh during `WriteProcessMemory`.
    let _write_guard =
        (!crate::readmap::enabled()).then(|| crate::watch::kernel_writing_range(at, size, false));
    crate::host::write_mapped(at as *mut u8, tiled.as_ptr(), size);
    drop(_write_guard);
    drop(_ticket_guard);
    let copied = COPIED.fetch_add(1, Ordering::Relaxed) + 1;
    if copied == 1 || copied.is_multiple_of(128) {
        tracing::info!(copied, bytes = size, "texture readback landed");
    }
    size
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{HRESULT, S_OK, guid, slot};

    struct ResetKinds;
    impl Drop for ResetKinds {
        fn drop(&mut self) {
            TEST_KINDS.with_borrow_mut(|over| *over = None);
        }
    }

    fn with_test_kinds(kinds: Vec<(u64, u32, u32)>) -> ResetKinds {
        TEST_KINDS.with_borrow_mut(|over| *over = Some(kinds));
        ResetKinds
    }

    #[repr(C)]
    struct TexDesc {
        dimension: u32,
        alignment: u64,
        width: u64,
        height: u32,
        depth_or_array_size: u16,
        mip_levels: u16,
        format: u32,
        sample_count: u32,
        sample_quality: u32,
        layout: u32,
        flags: u32,
    }

    struct Release(This);
    impl Drop for Release {
        fn drop(&mut self) {
            slot::<extern "system" fn(This) -> u32>(self.0, 0x10)(self.0);
        }
    }

    fn resource_iid() -> crate::object::Iid {
        guid(
            0x6964_42be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        )
    }

    fn make_committed_uav_tex(w: u32, h: u32, array_size: u16, mips: u16, format: u32) -> Release {
        const DEFAULT_HEAP: [u32; 5] = [1, 0, 0, 1, 1];
        const TEXTURE_2D: u32 = 3;
        const ALLOW_UAV: u32 = 0x4;
        let desc = TexDesc {
            dimension: TEXTURE_2D,
            alignment: 0,
            width: u64::from(w),
            height: h,
            depth_or_array_size: array_size,
            mip_levels: mips,
            format,
            sample_count: 1,
            sample_quality: 0,
            layout: 0,
            flags: ALLOW_UAV,
        };
        let iid = resource_iid();
        let mut resource: This = std::ptr::null();
        let hr: HRESULT = crate::resource::create_committed(
            std::ptr::null(),
            DEFAULT_HEAP.as_ptr().cast(),
            0,
            (&raw const desc).cast(),
            0,
            std::ptr::null(),
            &iid,
            &mut resource,
        );
        assert_eq!(hr, S_OK);
        assert!(!resource.is_null());
        Release(resource)
    }

    fn clear_uav_float(list: This, gpu: u64, cpu: u64, resource: This, values: &[f32; 4]) {
        type Clear = extern "system" fn(This, u64, u64, This, *const [f32; 4], u32, *const u8);
        slot::<Clear>(list, 0x1b0)(list, gpu, cpu, resource, values, 0, std::ptr::null());
    }

    fn make_at_address_tex(addr: u64, w: u32, h: u32, format: u32, flags: u32) -> Release {
        const TEXTURE_2D: u32 = 3;
        let desc = TexDesc {
            dimension: TEXTURE_2D,
            alignment: 0,
            width: u64::from(w),
            height: h,
            depth_or_array_size: 1,
            mip_levels: 1,
            format,
            sample_count: 1,
            sample_quality: 0,
            layout: 0,
            flags,
        };
        let iid = resource_iid();
        let mut resource: This = std::ptr::null();
        let hr: HRESULT = crate::resource::create_at_address(
            std::ptr::null(),
            addr,
            (&raw const desc).cast(),
            0,
            std::ptr::null(),
            &iid,
            &mut resource,
        );
        assert_eq!(hr, S_OK);
        assert!(!resource.is_null());
        Release(resource)
    }

    #[test]
    fn only_readbacks_for_changed_allocations_are_invalidated() {
        let first = track(91, 0x1000, 0x2000);
        let next = track(92, 0x2000, 0x3000);
        let far = track(93, 0x4000, 0x5000);

        // Zero-length or non-overlapping remaps leave tickets intact.
        remapped(0x1000, 0);
        remapped(usize::MAX - 0x10, 0x100);
        assert!(first.valid.load(Ordering::Acquire));

        remapped(0x0f00, 0x200);
        assert!(!first.valid.load(Ordering::Acquire));
        assert!(next.valid.load(Ordering::Acquire));

        // Adjacent non-overlapping replacement leaves `next` intact; an
        // interior overlap invalidates it.
        replaced(0x3000, 0x1000);
        assert!(next.valid.load(Ordering::Acquire));
        replaced(0x2800, 0x200);
        assert!(!next.valid.load(Ordering::Acquire));
        assert!(far.valid.load(Ordering::Acquire));

        // Replacement starting before a ticket and overlapping into it.
        let overlap = track(95, 0x6000, 0x7000);
        replaced(0x5800, 0x1000);
        assert!(!overlap.valid.load(Ordering::Acquire));

        released(93);
        assert!(!far.valid.load(Ordering::Acquire));
        assert!(track(94, 0x2000, 0x3000).valid.load(Ordering::Acquire));
    }

    #[test]
    fn parses_texture_back_kinds() {
        let defaults = [(256, 256, 41), (64, 64, 16)];
        assert!(parse_kinds("0", &defaults).is_empty());
        assert!(parse_kinds("off", &defaults).is_empty());
        assert!(parse_kinds("FALSE", &defaults).is_empty());
        assert!(parse_kinds("  ", &defaults).is_empty());
        assert_eq!(parse_kinds("1", &defaults), defaults);
        assert_eq!(parse_kinds("on", &defaults), defaults);
        assert_eq!(parse_kinds("TRUE", &defaults), defaults);
        assert_eq!(
            parse_kinds(
                "16x16f41, 128X128F36, 0x16f41, 16x0f41, 16x16f0, bad",
                &defaults
            ),
            vec![(16, 16, 41), (128, 128, 36)]
        );
    }

    #[test]
    fn compute_uav_texture_lands_tiled_in_title_memory() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_FLOAT;
        use windows::core::s;

        use crate::testgpu::TitleList;

        let test = "texback::tests::compute_uav_texture_lands_tiled_in_title_memory";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };

        const W: u32 = 16;
        const H: u32 = 16;
        const R32_FLOAT: u32 = 41;
        const UNORDERED_ACCESS: u32 = 0x8;
        let _reset = with_test_kinds(vec![(u64::from(W), H, R32_FLOAT)]);

        let res = make_committed_uav_tex(W, H, 1, 1, R32_FLOAT);
        let resource = res.0;

        let host_tex = crate::resource::host_resource(resource).expect("host texture");
        assert!(
            !crate::texture::title_written(host_tex),
            "UAV textures are not re-uploaded from title memory"
        );

        let source = r#"
#define RS "DescriptorTable(UAV(u0))"
RWTexture2D<float> dst : register(u0);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    dst[id.xy] = float(id.y * 16u + id.x + 1u);
}
"#;
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: source string and entry names live across D3DCompile.
        unsafe {
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                s!("texback_test"),
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
        .expect("compute shader compiles");
        let code = code.expect("compiled bytecode");
        // SAFETY: compiled blob owns its buffer until dropped.
        let bytes = unsafe {
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
        };
        // SAFETY: live device and valid shader bytecode containing embedded root signature.
        let (root, pso, desc_heap) = unsafe {
            let root: ID3D12RootSignature = gpu.device.CreateRootSignature(0, bytes).expect("root");
            let pso: ID3D12PipelineState = gpu
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytes.as_ptr().cast(),
                        BytecodeLength: bytes.len(),
                    },
                    ..Default::default()
                })
                .expect("pso");
            let desc_heap: ID3D12DescriptorHeap = gpu
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 1,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
                .expect("uav heap");
            gpu.device.CreateUnorderedAccessView(
                host_tex,
                None,
                Some(&D3D12_UNORDERED_ACCESS_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_FLOAT,
                    ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
                    Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                        Texture2D: D3D12_TEX2D_UAV {
                            MipSlice: 0,
                            PlaneSlice: 0,
                        },
                    },
                }),
                desc_heap.GetCPUDescriptorHandleForHeapStart(),
            );
            (root, pso, desc_heap)
        };

        let list = TitleList::new();
        let hl = list.host();
        hl.texture_written(resource);
        crate::list::ensure_state(hl, resource, UNORDERED_ACCESS);
        // SAFETY: open host command list with live pipeline, root signature, and descriptor heap.
        unsafe {
            hl.list.SetDescriptorHeaps(&[Some(desc_heap.clone())]);
            hl.list.SetComputeRootSignature(&root);
            hl.list.SetPipelineState(&pso);
            hl.list
                .SetComputeRootDescriptorTable(0, desc_heap.GetGPUDescriptorHandleForHeapStart());
            hl.list.Dispatch(W / 8, H / 8, 1);
        }
        list.run();

        let (layout, at, size) =
            crate::resource::title_texture(resource).expect("title texture layout");
        // SAFETY: title resource is still alive and committed for `size` bytes.
        let tiled = unsafe { std::slice::from_raw_parts(at as *const u8, size) };
        let pitch = (W * 4) as usize;
        let depth_pitch = pitch * H as usize;
        let mut linear = vec![0u8; depth_pitch];
        layout
            .detile(tiled, 0, 0, &mut linear, pitch, depth_pitch)
            .expect("detile landed texture");

        let mut expected = Vec::with_capacity((W * H) as usize);
        for y in 0..H {
            for x in 0..W {
                expected.push((y * W + x + 1) as f32);
            }
        }
        let got: Vec<f32> = linear
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| f32::from_le_bytes(c))
            .collect();
        assert_eq!(got, expected);
        // The default 2D swizzle (`SW_4KB_S`) is tiled, not row-major linear.
        assert_ne!(&tiled[..linear.len()], linear.as_slice());
    }

    #[test]
    fn uav_clear_lands_multi_mip_array_and_preserves_tail_bytes() {
        use crate::testgpu::TitleList;

        let test = "texback::tests::uav_clear_lands_multi_mip_array_and_preserves_tail_bytes";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };

        const W: u32 = 16;
        const H: u32 = 16;
        const MIPS: u16 = 2;
        const SLICES: u16 = 2;
        const R32_FLOAT: u32 = 41;
        const UAV_TEX2D_ARRAY: u32 = 5;
        let _reset = with_test_kinds(vec![(u64::from(W), H, R32_FLOAT)]);

        let res = make_committed_uav_tex(W, H, SLICES, MIPS, R32_FLOAT);
        let resource = res.0;
        let (layout, at, size) =
            crate::resource::title_texture(resource).expect("title texture layout");

        // Fill title memory with a sentinel so uncovered padding bytes inside
        // the 4 KiB blocks prove `land` preserves them.
        // SAFETY: title resource owns `size` committed writable bytes at `at`.
        unsafe { std::slice::from_raw_parts_mut(at as *mut u8, size).fill(0xA5) };

        let heap = crate::descriptor::TestHeap::views(4);
        let base = crate::descriptor::heap(heap.0).start;
        let mut cpu = [0u64; 16];
        let list = TitleList::new();
        let values = [[1.25f32, 2.5f32], [3.75f32, 5.0f32]];
        for slice in 0..u32::from(SLICES) {
            for mip in 0..u32::from(MIPS) {
                let idx = (slice * u32::from(MIPS) + mip) as usize;
                let desc: [u32; 10] = [R32_FLOAT, UAV_TEX2D_ARRAY, mip, slice, 1, 0, 0, 0, 0, 0];
                let gpu_handle = base + idx as u64 * 32;
                let cpu_handle = (&raw mut cpu[idx * 4]) as u64;
                crate::descriptor::create_uav(
                    std::ptr::null(),
                    resource,
                    std::ptr::null(),
                    desc.as_ptr().cast(),
                    gpu_handle,
                );
                crate::descriptor::create_uav(
                    std::ptr::null(),
                    resource,
                    std::ptr::null(),
                    desc.as_ptr().cast(),
                    cpu_handle,
                );
                let v = values[slice as usize][mip as usize];
                let clear = [v, 0.0, 0.0, 0.0];
                clear_uav_float(list.this(), gpu_handle, cpu_handle, resource, &clear);
            }
        }
        list.run();

        // SAFETY: title resource is alive and committed for `size` bytes.
        let tiled = unsafe { std::slice::from_raw_parts(at as *const u8, size) };
        let mut covered = vec![false; size];
        for slice in 0..u32::from(SLICES) {
            for mip in 0..u32::from(MIPS) {
                let mw = W >> mip;
                let mh = H >> mip;
                let pitch = (mw * 4) as usize;
                let depth_pitch = pitch * mh as usize;
                let mut linear = vec![0u8; depth_pitch];
                layout
                    .detile(tiled, mip, slice, &mut linear, pitch, depth_pitch)
                    .expect("detile subresource");
                let expected = values[slice as usize][mip as usize];
                for &chunk in linear.as_chunks::<4>().0 {
                    assert_eq!(
                        f32::from_le_bytes(chunk),
                        expected,
                        "mip {mip} slice {slice}"
                    );
                }
                for y in 0..mh {
                    for x in 0..mw {
                        let off = layout.offset(mip, x, y, slice).unwrap() as usize;
                        covered[off..off + 4].fill(true);
                    }
                }
            }
        }
        // Every padding byte not covered by any subresource stays 0xA5.
        let mut padding_checked = 0usize;
        for (i, &byte) in tiled.iter().enumerate() {
            if !covered[i] {
                assert_eq!(byte, 0xA5, "uncovered padding byte {i} was clobbered");
                padding_checked += 1;
            }
        }
        assert!(padding_checked > 0, "expected uncovered tile padding bytes");
    }

    #[test]
    fn shared_alias_keeps_readback_while_overlap_and_release_invalidate() {
        use crate::testgpu::TitleList;

        let test =
            "texback::tests::shared_alias_keeps_readback_while_overlap_and_release_invalidate";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };

        const W: u32 = 16;
        const H: u32 = 16;
        const R32_FLOAT: u32 = 41;
        const ALLOW_UAV: u32 = 0x4;
        let _reset = with_test_kinds(vec![(u64::from(W), H, R32_FLOAT)]);

        let mem = crate::resource::Memory::zeroed(65536, 65536).expect("memory");
        let addr = mem.address();
        let res = make_at_address_tex(addr, W, H, R32_FLOAT, ALLOW_UAV);
        let resource = res.0;
        let (layout, at, size) =
            crate::resource::title_texture(resource).expect("title texture layout");

        let heap = crate::descriptor::TestHeap::views(2);
        let gpu_handle = crate::descriptor::heap(heap.0).start;
        let mut cpu = [0u64; 4];
        let cpu_handle = cpu.as_mut_ptr() as u64;
        crate::descriptor::create_uav(
            std::ptr::null(),
            resource,
            std::ptr::null(),
            std::ptr::null(),
            gpu_handle,
        );
        crate::descriptor::create_uav(
            std::ptr::null(),
            resource,
            std::ptr::null(),
            std::ptr::null(),
            cpu_handle,
        );

        // 1. A shared alias created at the same address must NOT invalidate `res`'s readback.
        let list1 = TitleList::new();
        clear_uav_float(
            list1.this(),
            gpu_handle,
            cpu_handle,
            resource,
            &[7.5, 0.0, 0.0, 0.0],
        );
        let alias = make_at_address_tex(addr, W, H, R32_FLOAT, 0);
        list1.run();
        drop(alias);

        let mut linear = vec![0u8; (W * H * 4) as usize];
        // SAFETY: `mem` owns `size` committed bytes at `at`.
        let tiled = unsafe { std::slice::from_raw_parts(at as *const u8, size) };
        layout
            .detile(
                tiled,
                0,
                0,
                &mut linear,
                (W * 4) as usize,
                (W * H * 4) as usize,
            )
            .expect("detile after shared alias");
        assert_eq!(f32::from_le_bytes(linear[..4].try_into().unwrap()), 7.5);

        // 2. An unrelated texture overlapping inside `[at, at + size)` MUST invalidate it.
        let list2 = TitleList::new();
        clear_uav_float(
            list2.this(),
            gpu_handle,
            cpu_handle,
            resource,
            &[9.0, 0.0, 0.0, 0.0],
        );
        let overlap = make_at_address_tex(addr + 256, 8, 8, R32_FLOAT, 0);
        list2.run();
        drop(overlap);

        // Title memory must still hold 7.5, not 9.0.
        layout
            .detile(
                tiled,
                0,
                0,
                &mut linear,
                (W * 4) as usize,
                (W * H * 4) as usize,
            )
            .expect("detile after overlap");
        assert_eq!(f32::from_le_bytes(linear[..4].try_into().unwrap()), 7.5);

        // 3. An invalidated entry in `back_wanted` must not block re-recording the same resource
        // later in the same command list.
        let list3 = TitleList::new();
        list3.host().texture_written(resource);
        remapped(at, size);
        clear_uav_float(
            list3.this(),
            gpu_handle,
            cpu_handle,
            resource,
            &[11.25, 0.0, 0.0, 0.0],
        );
        list3.run();
        layout
            .detile(
                tiled,
                0,
                0,
                &mut linear,
                (W * 4) as usize,
                (W * H * 4) as usize,
            )
            .expect("detile after re-record");
        assert_eq!(f32::from_le_bytes(linear[..4].try_into().unwrap()), 11.25);

        drop(res);
        crate::resource::memory_remapped(addr as usize, 65536);
    }

    #[test]
    fn landing_on_readonly_and_decommitted_memory_does_not_crash_or_zero() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READONLY, PAGE_READWRITE,
            VirtualAlloc, VirtualFree, VirtualProtect,
        };

        use crate::testgpu::TitleList;

        let test =
            "texback::tests::landing_on_readonly_and_decommitted_memory_does_not_crash_or_zero";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };

        const W: u32 = 16;
        const H: u32 = 16;
        const R32_FLOAT: u32 = 41;
        const ALLOW_UAV: u32 = 0x4;
        const LEN: usize = 65536;
        let _reset = with_test_kinds(vec![(u64::from(W), H, R32_FLOAT)]);

        // SAFETY: private committed allocation owned by this test.
        let base = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                LEN,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        };
        assert!(!base.is_null());
        struct FreeMem(*mut std::ffi::c_void);
        impl Drop for FreeMem {
            fn drop(&mut self) {
                // SAFETY: matches `VirtualAlloc` above.
                unsafe { VirtualFree(self.0, 0, MEM_RELEASE) };
            }
        }
        let _free = FreeMem(base);

        let res = make_at_address_tex(base as u64, W, H, R32_FLOAT, ALLOW_UAV);
        let resource = res.0;
        let (layout, at, size) =
            crate::resource::title_texture(resource).expect("title texture layout");

        let heap = crate::descriptor::TestHeap::views(2);
        let gpu_handle = crate::descriptor::heap(heap.0).start;
        let mut cpu = [0u64; 4];
        let cpu_handle = cpu.as_mut_ptr() as u64;
        crate::descriptor::create_uav(
            std::ptr::null(),
            resource,
            std::ptr::null(),
            std::ptr::null(),
            gpu_handle,
        );
        crate::descriptor::create_uav(
            std::ptr::null(),
            resource,
            std::ptr::null(),
            std::ptr::null(),
            cpu_handle,
        );

        // Fill with a sentinel before making the allocation read-only.
        // SAFETY: `base..base + LEN` is committed writable memory owned by this test.
        unsafe { std::slice::from_raw_parts_mut(at as *mut u8, size).fill(0x5A) };

        let list = TitleList::new();
        clear_uav_float(
            list.this(),
            gpu_handle,
            cpu_handle,
            resource,
            &[4.5, 0.0, 0.0, 0.0],
        );
        let mut old = 0u32;
        // SAFETY: `base..base + LEN` is our committed allocation.
        let protected = unsafe { VirtualProtect(base, LEN, PAGE_READONLY, &mut old) };
        assert_ne!(protected, 0);
        // Must not access-violate or zero title memory via `copy_mapped`'s fallback;
        // `WriteProcessMemory` refuses title-protected `PAGE_READONLY` pages cleanly.
        list.run();

        // SAFETY: `base` is still committed readable memory.
        let tiled = unsafe { std::slice::from_raw_parts(at as *const u8, size) };
        assert!(
            tiled.iter().all(|&b| b == 0x5A),
            "readonly landing must leave title-protected bytes untouched"
        );

        // Once writable again, the next submission lands normally.
        // SAFETY: `base..base + LEN` is our committed allocation.
        let restored = unsafe { VirtualProtect(base, LEN, PAGE_READWRITE, &mut old) };
        assert_ne!(restored, 0);
        let list_rw = TitleList::new();
        clear_uav_float(
            list_rw.this(),
            gpu_handle,
            cpu_handle,
            resource,
            &[4.5, 0.0, 0.0, 0.0],
        );
        list_rw.run();
        let mut linear = vec![0u8; (W * H * 4) as usize];
        // SAFETY: `base` is committed readable memory.
        let tiled = unsafe { std::slice::from_raw_parts(at as *const u8, size) };
        layout
            .detile(
                tiled,
                0,
                0,
                &mut linear,
                (W * 4) as usize,
                (W * H * 4) as usize,
            )
            .expect("detile after writable land");
        assert_eq!(f32::from_le_bytes(linear[..4].try_into().unwrap()), 4.5);

        // Decommitted memory must skip cleanly without crashing.
        let list2 = TitleList::new();
        clear_uav_float(
            list2.this(),
            gpu_handle,
            cpu_handle,
            resource,
            &[8.0, 0.0, 0.0, 0.0],
        );
        // SAFETY: decommitting our own private allocation; `res` is dropped before `FreeMem`.
        let freed = unsafe { VirtualFree(base, LEN, MEM_DECOMMIT) };
        assert_ne!(freed, 0);
        list2.run();

        drop(res);
    }

    #[test]
    fn standin_format_width_mismatch_is_skipped() {
        use crate::testgpu::TitleList;

        let test = "texback::tests::standin_format_width_mismatch_is_skipped";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };

        const W: u32 = 16;
        const H: u32 = 16;
        const R32_FLOAT: u32 = 41;
        const RGBA16F: u32 = 10;

        let res_4b = make_committed_uav_tex(W, H, 1, 1, R32_FLOAT);
        let res_8b = make_committed_uav_tex(W, H, 1, 1, RGBA16F);
        let copy_8b = crate::resource::host_copy(res_8b.0).expect("8-byte host copy");
        let (layout_4b, at, size) =
            crate::resource::title_texture(res_4b.0).expect("4-byte title layout");

        let list = TitleList::new();
        let target = Target {
            copy: copy_8b,
            layout: layout_4b.clone(),
            ticket: track(res_4b.0 as usize, at, at + size),
        };
        let mut prepared = false;
        let recorded = record(&list.host().list, target, || prepared = true);
        assert!(
            recorded.is_none(),
            "mismatched host texel width must be refused"
        );
        assert!(!prepared, "refused target must not emit a state transition");

        // Geometry mismatch must also be refused before any state transition.
        let res_wide = make_committed_uav_tex(W * 2, H, 1, 1, R32_FLOAT);
        let copy_wide = crate::resource::host_copy(res_wide.0).expect("wide host copy");
        let target = Target {
            copy: copy_wide,
            layout: layout_4b,
            ticket: track(res_4b.0 as usize, at, at + size),
        };
        let mut prepared = false;
        let recorded = record(&list.host().list, target, || prepared = true);
        assert!(
            recorded.is_none(),
            "mismatched host geometry must be refused"
        );
        assert!(!prepared, "refused target must not emit a state transition");
        list.run();
    }

    #[test]
    fn non_shared_placed_view_does_not_invalidate_live_readback() {
        use crate::testgpu::TitleList;

        let test = "texback::tests::non_shared_placed_view_does_not_invalidate_live_readback";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };

        const W: u32 = 16;
        const H: u32 = 16;
        const R32_FLOAT: u32 = 41;
        const ALLOW_UAV: u32 = 0x4;
        let _reset = with_test_kinds(vec![(u64::from(W), H, R32_FLOAT)]);

        let mem = crate::resource::Memory::zeroed(65536, 65536).expect("memory");
        let addr = mem.address();
        // A 2-mip UAV texture at `addr`.
        let desc2 = TexDesc {
            dimension: 3,
            alignment: 0,
            width: u64::from(W),
            height: H,
            depth_or_array_size: 1,
            mip_levels: 2,
            format: R32_FLOAT,
            sample_count: 1,
            sample_quality: 0,
            layout: 0,
            flags: ALLOW_UAV,
        };
        let iid = resource_iid();
        let mut resource: This = std::ptr::null();
        assert_eq!(
            crate::resource::create_at_address(
                std::ptr::null(),
                addr,
                (&raw const desc2).cast(),
                0,
                std::ptr::null(),
                &iid,
                &mut resource,
            ),
            S_OK
        );
        let res = Release(resource);
        let (layout, at, size) =
            crate::resource::title_texture(resource).expect("title texture layout");

        let heap = crate::descriptor::TestHeap::views(2);
        let gpu_handle = crate::descriptor::heap(heap.0).start;
        let mut cpu = [0u64; 4];
        let cpu_handle = cpu.as_mut_ptr() as u64;
        crate::descriptor::create_uav(
            std::ptr::null(),
            resource,
            std::ptr::null(),
            std::ptr::null(),
            gpu_handle,
        );
        crate::descriptor::create_uav(
            std::ptr::null(),
            resource,
            std::ptr::null(),
            std::ptr::null(),
            cpu_handle,
        );

        let list = TitleList::new();
        clear_uav_float(
            list.this(),
            gpu_handle,
            cpu_handle,
            resource,
            &[3.25, 0.0, 0.0, 0.0],
        );

        // Materialize an internal cached `PLACED_VIEW` with a different mip
        // count (1 != 2) so it is neither `shared` nor `written_beside`.
        let placed_desc = TexDesc {
            mip_levels: 1,
            flags: 0,
            ..desc2
        };
        // SAFETY: `TexDesc` is 56 bytes (`[u32; 14]`), matching `ResourceDesc`.
        let words: [u32; 14] = unsafe { std::mem::transmute(placed_desc) };
        assert!(
            crate::resource::texture_at(addr, &words).is_some(),
            "placed view created"
        );

        list.run();
        let mut linear = vec![0u8; (W * H * 4) as usize];
        // SAFETY: `mem` owns `size` committed bytes at `at`.
        let tiled = unsafe { std::slice::from_raw_parts(at as *const u8, size) };
        layout
            .detile(
                tiled,
                0,
                0,
                &mut linear,
                (W * 4) as usize,
                (W * H * 4) as usize,
            )
            .expect("detile after non-shared placed view");
        assert_eq!(f32::from_le_bytes(linear[..4].try_into().unwrap()), 3.25);

        drop(res);
        crate::resource::memory_remapped(addr as usize, 65536);
    }

    #[test]
    fn async_submissions_land_in_order_across_in_flight_reset_and_release() {
        use crate::host::submissions::{Gate, serial};
        use crate::testgpu::TitleList;

        let test =
            "texback::tests::async_submissions_land_in_order_across_in_flight_reset_and_release";
        let _serial = serial();
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();

        const W: u32 = 16;
        const H: u32 = 16;
        const R32_FLOAT: u32 = 41;
        const ALLOW_UAV: u32 = 0x4;
        let _reset = with_test_kinds(vec![(u64::from(W), H, R32_FLOAT)]);

        let mem = crate::resource::Memory::zeroed(65536, 65536).expect("memory");
        let addr = mem.address();
        let res = make_at_address_tex(addr, W, H, R32_FLOAT, ALLOW_UAV);
        let resource = res.0;
        let (layout, at, size) =
            crate::resource::title_texture(resource).expect("title texture layout");

        let heap = crate::descriptor::TestHeap::views(2);
        let gpu_handle = crate::descriptor::heap(heap.0).start;
        let mut cpu = [0u64; 4];
        let cpu_handle = cpu.as_mut_ptr() as u64;
        crate::descriptor::create_uav(
            std::ptr::null(),
            resource,
            std::ptr::null(),
            std::ptr::null(),
            gpu_handle,
        );
        crate::descriptor::create_uav(
            std::ptr::null(),
            resource,
            std::ptr::null(),
            std::ptr::null(),
            cpu_handle,
        );

        let read_first_texel = || {
            let mut linear = vec![0u8; (W * H * 4) as usize];
            // SAFETY: `mem` owns `size` committed bytes at `at`.
            let tiled = unsafe { std::slice::from_raw_parts(at as *const u8, size) };
            layout
                .detile(
                    tiled,
                    0,
                    0,
                    &mut linear,
                    (W * 4) as usize,
                    (W * H * 4) as usize,
                )
                .expect("detile");
            f32::from_le_bytes(linear[..4].try_into().unwrap())
        };

        // Two lists writing different values, both submitted while the GPU is
        // gated; list1 is reset while still in flight.
        let list1 = TitleList::new();
        clear_uav_float(
            list1.this(),
            gpu_handle,
            cpu_handle,
            resource,
            &[2.0, 0.0, 0.0, 0.0],
        );
        list1.host().close();

        let list2 = TitleList::new();
        clear_uav_float(
            list2.this(),
            gpu_handle,
            cpu_handle,
            resource,
            &[6.0, 0.0, 0.0, 0.0],
        );
        list2.host().close();

        let mut gate = Gate::close(host);
        host.submit(&[list1.host()]);
        host.submit(&[list2.host()]);
        assert!(list1.host().running() && list2.host().running());
        assert_eq!(read_first_texel(), 0.0, "nothing landed while gated");
        // Reset list1 while its submission is in flight; `Pending` retains its
        // `Arc<Back>` and the second submission lands after it.
        list1.host().reset();
        list1.host().close();
        gate.open();
        host.drain();
        assert_eq!(read_first_texel(), 6.0, "second submission lands last");

        // Releasing the resource while a third submission is in flight must
        // invalidate its readback before completion.
        let list3 = TitleList::new();
        clear_uav_float(
            list3.this(),
            gpu_handle,
            cpu_handle,
            resource,
            &[12.0, 0.0, 0.0, 0.0],
        );
        list3.host().close();
        let mut gate = Gate::close(host);
        host.submit(&[list3.host()]);
        assert!(list3.host().running());
        drop(res);
        gate.open();
        host.drain();
        assert_eq!(
            read_first_texel(),
            6.0,
            "in-flight release prevented stale landing"
        );
        crate::resource::memory_remapped(addr as usize, 65536);
    }

    #[test]
    fn unwanted_kind_leaves_title_memory_untouched() {
        use crate::testgpu::TitleList;

        let test = "texback::tests::unwanted_kind_leaves_title_memory_untouched";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };

        const W: u32 = 16;
        const H: u32 = 16;
        const R32_FLOAT: u32 = 41;
        // Explicitly configure an empty wanted list (equivalent to off).
        let _reset = with_test_kinds(Vec::new());

        let res = make_committed_uav_tex(W, H, 1, 1, R32_FLOAT);
        let resource = res.0;
        let (_, at, size) = crate::resource::title_texture(resource).expect("title texture layout");
        // SAFETY: committed resource memory owned by `res`.
        unsafe { std::slice::from_raw_parts_mut(at as *mut u8, size).fill(0x5A) };

        let heap = crate::descriptor::TestHeap::views(2);
        let gpu_handle = crate::descriptor::heap(heap.0).start;
        let mut cpu = [0u64; 4];
        let cpu_handle = cpu.as_mut_ptr() as u64;
        crate::descriptor::create_uav(
            std::ptr::null(),
            resource,
            std::ptr::null(),
            std::ptr::null(),
            gpu_handle,
        );
        crate::descriptor::create_uav(
            std::ptr::null(),
            resource,
            std::ptr::null(),
            std::ptr::null(),
            cpu_handle,
        );

        let list = TitleList::new();
        clear_uav_float(
            list.this(),
            gpu_handle,
            cpu_handle,
            resource,
            &[99.0, 0.0, 0.0, 0.0],
        );
        list.run();

        // SAFETY: committed resource memory owned by `res`.
        let bytes = unsafe { std::slice::from_raw_parts(at as *const u8, size) };
        assert!(
            bytes.iter().all(|&b| b == 0x5A),
            "off/unwanted texture must not be copied back"
        );
    }
}
