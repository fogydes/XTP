//! Heaps, committed and placed resources, and resource sizes.
//!
//! Heaps and resources are host memory and a resource's GPU address is its
//! CPU address, as on the console; textures also get a host copy, and
//! buffers a host copy when a draw reads them (`shadow`). [`format`] gives
//! the host format of each title format.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

#[cfg(windows)]
mod alias;
#[cfg(windows)]
pub(crate) mod decode;
pub(crate) mod format;
#[cfg(windows)]
pub(crate) mod pool;
#[cfg(windows)]
mod ranges;
#[cfg(windows)]
mod residency;

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::ffi::c_void;
use std::ptr::NonNull;
#[cfg(windows)]
use std::sync::Arc;

use xtp_formats::texture;

use crate::object::{
    self, E_INVALIDARG, E_OUTOFMEMORY, HEAP, HRESULT, Iid, Object, RESOURCE, S_OK, Table, This,
    guid,
};

/// The interfaces every device child also answers for.
pub(crate) const PAGEABLE_IIDS: [Iid; 3] = [
    guid(
        0x63ee58fb,
        0x1268,
        0x4835,
        [0x86, 0xda, 0xf0, 0x08, 0xce, 0x62, 0xf0, 0xd6],
    ),
    guid(
        0x905db94b,
        0xa00c,
        0x4140,
        [0x9d, 0xf5, 0x2b, 0x64, 0xca, 0x9e, 0xa3, 0x57],
    ),
    guid(
        0xc4fec28f,
        0x7966,
        0x4e95,
        [0x9f, 0x94, 0xf4, 0x31, 0xcb, 0x56, 0xc3, 0xb8],
    ),
];
const HEAP_IIDS: [Iid; 4] = [
    guid(
        0x6b3b2502,
        0x6e51,
        0x45b3,
        [0x90, 0xee, 0x98, 0x84, 0x26, 0x5e, 0x8d, 0xf3],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];
const RESOURCE_IIDS: [Iid; 4] = [
    guid(
        0x696442be,
        0xa72e,
        0x4059,
        [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

const KIB_64: u64 = 64 << 10;
const MIB_4: u64 = 4 << 20;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct HeapProperties {
    kind: u32,
    cpu_pages: u32,
    pool: u32,
    creation_nodes: u32,
    visible_nodes: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct HeapDesc {
    size: u64,
    properties: HeapProperties,
    alignment: u64,
    flags: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct ResourceDesc {
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

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AllocationInfo {
    size: u64,
    alignment: u64,
}

impl ResourceDesc {
    /// The title's format.
    pub(crate) fn format(&self) -> u32 {
        self.format
    }
}

const _: () = assert!(size_of::<HeapProperties>() == 20);
const _: () = assert!(size_of::<HeapDesc>() == 48);
const _: () = assert!(size_of::<ResourceDesc>() == 56);
const _: () = assert!(std::mem::offset_of!(ResourceDesc, alignment) == 8);
const _: () = assert!(std::mem::offset_of!(ResourceDesc, flags) == 48);

/// A resource description as words with its two padding words (bytes 4-7,
/// after the dimension, and 52-55, after the flags) zeroed: titles pass
/// descriptions from the stack, and equal ones must compare equal (after
/// VIctory #643, by nathanbzrr).
pub(crate) fn canonical_resource_desc(mut words: [u32; 14]) -> [u32; 14] {
    words[1] = 0;
    words[13] = 0;
    words
}

const BUFFER: u32 = 1;
const TEXTURE_3D: u32 = 4;

/// Bytes per element and the element's width in texels (4 for block
/// compression), by DXGI format number. `None` for 1-bit and video
/// formats. The console adds 116/117 (10-bit float color), 118-120 (16-bit
/// depth with 8-bit stencil, two planes: 3 bytes per texel), 189 (signed
/// 10-bit color), 190 (two 4-bit channels) and 191 (24/8 color).
pub(crate) fn element(format: u32) -> Option<(u64, u64)> {
    Some(match format {
        1..=4 => (16, 1),
        5..=8 => (12, 1),
        9..=22 => (8, 1),
        23..=47 | 67 | 87..=93 | 116 | 117 | 189 | 191 => (4, 1),
        118..=120 => (3, 1),
        48..=59 | 68 | 69 | 85 | 86 | 115 => (2, 1),
        60..=65 | 190 => (1, 1),
        70..=72 | 79..=81 => (8, 4),
        73..=78 | 82..=84 | 94..=99 => (16, 4),
        _ => return None,
    })
}

/// A texture's layout in console memory, as the console's layout library
/// gives it (`xtp-hle`'s `xg` serves the same): the description's
/// layout field is `0x100 | swizzle mode`, 1 linear, 0 the library's pick.
pub(crate) fn texture_layout(desc: &ResourceDesc) -> Option<texture::Texture> {
    use texture::{Dimension, Swizzle, TextureDesc};
    const TEXTURE_1D: u32 = 2;
    const TEXTURE_2D: u32 = 3;
    const RENDER_TARGET: u32 = 1;
    const DEPTH: u32 = 2;
    let dimension = match desc.dimension {
        TEXTURE_1D => Dimension::Tex1d,
        TEXTURE_2D => Dimension::Tex2d,
        TEXTURE_3D => Dimension::Tex3d,
        _ => return None,
    };
    let swizzle = match desc.layout {
        0 if dimension == Dimension::Tex1d => Swizzle::LINEAR,
        0 if desc.flags & DEPTH != 0 => Swizzle::SW_64KB_Z_X,
        0 if desc.flags & RENDER_TARGET != 0 => Swizzle::SW_64KB_R_X,
        0 => Swizzle::SW_4KB_S,
        1 => Swizzle::LINEAR,
        l if l & !0xff == 0x100 => Swizzle(l & 0xff),
        _ => return None,
    };
    texture::Texture::new(&TextureDesc {
        dimension,
        width: u32::try_from(desc.width).ok()?,
        height: desc.height,
        depth_or_array: u32::from(desc.depth_or_array_size),
        mips: u32::from(desc.mip_levels),
        format: desc.format,
        samples: desc.sample_count.max(1),
        swizzle,
    })
    .ok()
}

/// Bytes and alignment a resource takes: buffers their width (64 KiB
/// aligned), textures their console layout (formats it does not know:
/// every mip of every array slice and sample, packed, 64 KiB aligned),
/// rounded up to the alignment.
fn footprint(desc: &ResourceDesc) -> Option<AllocationInfo> {
    if desc.dimension != BUFFER
        && let Some(t) = texture_layout(desc)
    {
        // As the console: the layout's own alignment unless more is asked
        // (4 KiB for 4 KiB modes, where PC says 64 KiB). BTP 6
        // keeps these sizes in its texture tables, and its streaming
        // asserts when they disagree with the streamed layout.
        let alignment = desc.alignment.max(t.alignment);
        if !alignment.is_power_of_two() {
            return None;
        }
        return Some(AllocationInfo {
            size: t.size.checked_next_multiple_of(alignment)?,
            alignment,
        });
    }
    let bytes = if desc.dimension == BUFFER {
        desc.width
    } else {
        let (element, block) = element(desc.format)?;
        let (depth, slices) = if desc.dimension == TEXTURE_3D {
            (u64::from(desc.depth_or_array_size), 1)
        } else {
            (1, u64::from(desc.depth_or_array_size))
        };
        let height = u64::from(desc.height);
        let mips = mip_count(desc);
        let mut chain = 0u64;
        for mip in 0..mips.min(64) {
            let w = (desc.width >> mip).max(1).div_ceil(block);
            let h = (height >> mip).max(1).div_ceil(block);
            let d = (depth >> mip).max(1);
            chain = chain.checked_add(w.checked_mul(h)?.checked_mul(d)?.checked_mul(element)?)?;
        }
        chain
            .checked_mul(slices)?
            .checked_mul(u64::from(desc.sample_count.max(1)))?
    };
    if bytes == 0 {
        return None;
    }
    let minimum = if desc.sample_count > 1 { MIB_4 } else { KIB_64 };
    let alignment = desc.alignment.max(minimum);
    if !alignment.is_power_of_two() {
        return None;
    }
    Some(AllocationInfo {
        size: bytes.checked_next_multiple_of(KIB_64)?,
        alignment,
    })
}

/// A title format as the host has it, for textures, views, targets and
/// clear values: a format of [`format::TABLE`] the host lacks takes its
/// stand-in, and [`pc_vertex_format`] applies.
pub(crate) fn pc_format(format: u32) -> u32 {
    use format::Caps;
    format::HostCaps.host_format(format)
}

/// The host format of a title shader or unordered access view (`unordered`)
/// described by `words` (the host's layout): [`pc_format`], but a typed
/// buffer view keeps the title's element size ([`format::buffer_format`]),
/// as its bytes are the title's. (Raw and structured views read by their
/// stride; the host names their format itself, `repair_format`.)
pub(crate) fn pc_view_format(words: &[u32; 10], unordered: bool) -> u32 {
    const BUFFER_VIEW: u32 = 1;
    const RAW: u32 = 1;
    // A buffer view's stride and flags: words 7 and 8 of a shader view
    // (its buffer part starts at word 4), 5 and 8 of an unordered access
    // view (at word 2).
    let stride = if unordered { words[5] } else { words[7] };
    let flags = words[8];
    if words[1] == BUFFER_VIEW && stride == 0 && flags & RAW == 0 {
        format::buffer_format(words[0], unordered)
    } else {
        pc_format(words[0])
    }
}

/// A title format as the host has it without stand-ins (vertex elements,
/// whose bytes are the title's). The console's 16-bit depth with stencil
/// (and its typeless views) has no PC format: the 24-bit ones stand in.
pub(crate) fn pc_vertex_format(format: u32) -> u32 {
    match format {
        118 => 45,
        119 => 46,
        120 => 47,
        other => other,
    }
}

/// Format families: the typeless format first, then its typed ones.
const FAMILIES: [(u32, u32); 22] = [
    (1, 4),
    (5, 8),
    (9, 14),
    (15, 18),
    (19, 22),
    (23, 25),
    (27, 32),
    (33, 38),
    (39, 43),
    (44, 47),
    (48, 52),
    (53, 59),
    (60, 64),
    (70, 72),
    (73, 75),
    (76, 78),
    (79, 81),
    (82, 84),
    (90, 91),
    (92, 93),
    (94, 96),
    (97, 99),
];

/// The typeless format of a format's family (itself if it has none).
pub(crate) fn family(format: u32) -> u32 {
    match format {
        87 => 90,
        88 => 92,
        f => FAMILIES
            .iter()
            .find(|&&(first, last)| (first..=last).contains(&f))
            .map_or(f, |&(first, _)| first),
    }
}

/// The typed formats of a typeless format's family; `None` if `format` is
/// not typeless.
pub(crate) fn family_members(format: u32) -> Option<impl Iterator<Item = u32>> {
    let &(first, last) = FAMILIES.iter().find(|&&(first, _)| first == format)?;
    let outside = match first {
        90 => Some(87),
        92 => Some(88),
        _ => None,
    };
    Some((first + 1..=last).chain(outside))
}

/// sRGB formats and their linear twins.
pub(crate) const SRGB: [(u32, u32); 3] = [(29, 28), (91, 87), (93, 88)];

/// The format a view of a title texture must name when its host copy is
/// typeless (see [`DEPTH_FORMATS`], [`SRGB`]): the title's own; `None` if
/// the host copy is typed.
pub(crate) fn view_format(this: This) -> Option<u32> {
    let own = pc_format(resource(this).desc.format);
    let host = host_resource(this)?;
    // SAFETY: a live resource.
    let format = unsafe { host.GetDesc() }.Format.0 as u32;
    (format != own).then_some(own)
}

/// Depth formats with their typeless family: depth textures that shaders
/// read are made typeless on the host, which only allows views of
/// typeless depth resources.
pub(crate) const DEPTH_FORMATS: [(u32, u32); 4] = [(20, 19), (40, 39), (45, 44), (55, 53)];

/// Mip levels, the full chain for 0.
fn mip_count(desc: &ResourceDesc) -> u32 {
    let depth = if desc.dimension == TEXTURE_3D {
        u64::from(desc.depth_or_array_size)
    } else {
        1
    };
    match desc.mip_levels {
        0 => {
            64 - desc
                .width
                .max(u64::from(desc.height))
                .max(depth)
                .max(1)
                .leading_zeros()
        }
        n => u32::from(n),
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PlacedFootprint {
    pub(crate) offset: u64,
    format: u32,
    width: u32,
    height: u32,
    depth: u32,
    row_pitch: u32,
}

impl PlacedFootprint {
    /// Its format.
    pub(crate) fn format(&self) -> u32 {
        self.format
    }

    /// Its width, height and depth in texels, and its row pitch in bytes.
    pub(crate) fn shape(&self) -> (u32, u32, u32, u32) {
        (self.width, self.height, self.depth, self.row_pitch)
    }

    /// The bytes it spans in a buffer: every row pitched but the last.
    pub(crate) fn bytes(&self) -> Option<u64> {
        let (element, block) = element(self.format)?;
        let lines = u64::from(self.height).div_ceil(block) * u64::from(self.depth);
        let last = u64::from(self.width).div_ceil(block) * element;
        Some(u64::from(self.row_pitch) * lines.saturating_sub(1) + last)
    }
}
const _: () = assert!(size_of::<PlacedFootprint>() == 32);

const NO_FOOTPRINT: PlacedFootprint = PlacedFootprint {
    offset: u64::MAX,
    format: 0,
    width: 0,
    height: 0,
    depth: 0,
    row_pitch: 0,
};

/// One subresource's size in an upload buffer: footprint (offset 0), rows
/// and bytes per row. Planes of depth-stencil formats are not told apart.
fn subresource(desc: &ResourceDesc, index: u32) -> Option<(PlacedFootprint, u32, u64)> {
    const PITCH: u64 = 256;
    if desc.dimension == BUFFER {
        let footprint = PlacedFootprint {
            offset: 0,
            format: 0,
            width: u32::try_from(desc.width).ok()?,
            height: 1,
            depth: 1,
            row_pitch: u32::try_from(desc.width.next_multiple_of(PITCH)).ok()?,
        };
        return (index == 0).then_some((footprint, 1, desc.width));
    }
    let mips = mip_count(desc);
    let slices = if desc.dimension == TEXTURE_3D {
        1
    } else {
        u32::from(desc.depth_or_array_size).max(1)
    };
    let (mip, slice) = (index % mips, index / mips);
    if slice >= slices {
        return None;
    }
    let (element, block) = element(desc.format)?;
    let width = (desc.width >> mip).max(1).next_multiple_of(block);
    let height = (u64::from(desc.height) >> mip)
        .max(1)
        .next_multiple_of(block);
    let depth = if desc.dimension == TEXTURE_3D {
        (u32::from(desc.depth_or_array_size) >> mip).max(1)
    } else {
        1
    };
    let row_size = width / block * element;
    let footprint = PlacedFootprint {
        offset: 0,
        format: desc.format,
        width: u32::try_from(width).ok()?,
        height: u32::try_from(height).ok()?,
        depth,
        row_pitch: u32::try_from(row_size.next_multiple_of(PITCH)).ok()?,
    };
    Some((footprint, u32::try_from(height / block).ok()?, row_size))
}

/// Device method: where each subresource goes in an upload buffer, rows
/// 256-byte and subresources 512-byte aligned, as on PC. Unknown sizes give
/// the documented all-ones values.
pub(crate) extern "system" fn copyable_footprints(
    _: This,
    desc: *const ResourceDesc,
    first: u32,
    count: u32,
    base: u64,
    layouts: *mut PlacedFootprint,
    rows: *mut u32,
    row_sizes: *mut u64,
    total: *mut u64,
) {
    const PLACEMENT: u64 = 512;
    // SAFETY: the title passes a description (null is tolerated).
    let desc = unsafe { desc.as_ref() };
    let mut end = base;
    let mut known = desc.is_some();
    for i in 0..count {
        let one = desc.and_then(|d| subresource(d, first + i));
        let (layout, row_count, row_size) = match one {
            Some((mut footprint, row_count, row_size)) => {
                footprint.offset = end.next_multiple_of(PLACEMENT);
                let pitch = u64::from(footprint.row_pitch);
                let lines = u64::from(row_count) * u64::from(footprint.depth);
                end = footprint.offset + pitch * lines.saturating_sub(1) + row_size;
                (footprint, row_count, row_size)
            }
            None => {
                known = false;
                (NO_FOOTPRINT, u32::MAX, u64::MAX)
            }
        };
        let i = i as usize;
        // SAFETY: each output is null or holds `count` entries.
        unsafe {
            if !layouts.is_null() {
                layouts.add(i).write(layout);
            }
            if !rows.is_null() {
                rows.add(i).write(row_count);
            }
            if !row_sizes.is_null() {
                row_sizes.add(i).write(row_size);
            }
        }
    }
    if !known {
        tracing::warn!(?desc, "no footprint for this resource");
    }
    if !total.is_null() {
        // SAFETY: checked non-null.
        unsafe { total.write(if known { end - base } else { u64::MAX }) };
    }
}

/// Device method: the footprint of resources placed one after another.
/// Returns a 16-byte value, through a hidden first argument. Invalid input
/// gives the documented size `u64::MAX`.
pub(crate) extern "system" fn allocation_info(
    out: *mut AllocationInfo,
    _: This,
    _visible: u32,
    count: u32,
    descs: *const ResourceDesc,
) -> *mut AllocationInfo {
    let descs = if descs.is_null() {
        &[][..]
    } else {
        // SAFETY: the title passes `count` descriptions.
        unsafe { std::slice::from_raw_parts(descs, count as usize) }
    };
    // SAFETY: the caller passes room for the value.
    unsafe { out.write(place(descs.iter(), |_, _| ())) };
    out
}

/// The newer description: the plain one, then a sampler feedback mip
/// region (width, height, depth).
#[repr(C)]
pub(crate) struct ResourceDesc1 {
    desc: ResourceDesc,
    mip_region: [u32; 3],
}
const _: () = assert!(size_of::<ResourceDesc1>() == 72);

/// Where one resource goes in [`allocation_info2`]'s placement.
#[repr(C)]
pub(crate) struct AllocationInfo1 {
    offset: u64,
    alignment: u64,
    size: u64,
}

/// Device method 0x360 (`ID3D12Device8`): [`allocation_info`] for newer
/// descriptions, also telling where each resource goes.
pub(crate) extern "system" fn allocation_info2(
    out: *mut AllocationInfo,
    _: This,
    _visible: u32,
    count: u32,
    descs: *const ResourceDesc1,
    each: *mut AllocationInfo1,
) -> *mut AllocationInfo {
    let descs = if descs.is_null() {
        &[][..]
    } else {
        // SAFETY: the title passes `count` descriptions.
        unsafe { std::slice::from_raw_parts(descs, count as usize) }
    };
    let total = place(descs.iter().map(|d| &d.desc), |i, info| {
        if !each.is_null() {
            // SAFETY: the title passes room for `count` entries.
            unsafe { each.add(i).write(info) };
        }
    });
    // SAFETY: the caller passes room for the value.
    unsafe { out.write(total) };
    out
}

/// Resources placed one after another: the whole footprint, telling `each`
/// where every resource goes. Unknown sizes give the documented `u64::MAX`.
fn place<'a>(
    descs: impl Iterator<Item = &'a ResourceDesc>,
    mut each: impl FnMut(usize, AllocationInfo1),
) -> AllocationInfo {
    let mut total = AllocationInfo {
        size: 0,
        alignment: KIB_64,
    };
    for (i, desc) in descs.enumerate() {
        let Some(one) = footprint(desc) else {
            tracing::warn!(?desc, "no size for this resource");
            each(
                i,
                AllocationInfo1 {
                    offset: u64::MAX,
                    alignment: 0,
                    size: u64::MAX,
                },
            );
            return AllocationInfo {
                size: u64::MAX,
                alignment: 0,
            };
        };
        tracing::debug!(?desc, ?one, "allocation info");
        let offset = total.size.next_multiple_of(one.alignment);
        each(
            i,
            AllocationInfo1 {
                offset,
                alignment: one.alignment,
                size: one.size,
            },
        );
        total.size = offset + one.size;
        total.alignment = total.alignment.max(one.alignment);
    }
    total
}

/// Zeroed host memory standing in for GPU memory.
pub(crate) struct Memory {
    ptr: NonNull<u8>,
    layout: Layout,
    /// Set when the memory came from the title's allocator, with these
    /// allocation attributes (it goes back there).
    attributes: Option<u64>,
}

pub type AllocRoutine = extern "system" fn(usize, u64) -> *mut c_void;
pub type FreeRoutine = extern "system" fn(*mut c_void, u64);
static ALLOCATOR: std::sync::OnceLock<(AllocRoutine, FreeRoutine)> = std::sync::OnceLock::new();

/// Committed resources get their memory from the title's general
/// allocator, as on the console, where titles hook it to place graphics
/// memory themselves (BTP 6 places its back buffers in a
/// block of its own and checks they fill it). The runtime names it before
/// the device exists.
pub fn use_allocator(alloc: AllocRoutine, free: FreeRoutine) {
    let _ = ALLOCATOR.set((alloc, free));
}

/// Maps the physical pages behind one range of title memory at another,
/// `(to, from, len)` (`from` 0 unmaps); false on failure.
pub type PageMapper = fn(u64, u64, u64) -> bool;
static PAGE_MAPPER: std::sync::OnceLock<PageMapper> = std::sync::OnceLock::new();

/// The runtime's page mapper, for the queues' page-mapping copies (memory
/// is the runtime's, as on the console).
pub fn use_page_mapper(mapper: PageMapper) {
    let _ = PAGE_MAPPER.set(mapper);
}

pub(crate) fn page_mapper() -> Option<PageMapper> {
    PAGE_MAPPER.get().copied()
}

/// Allocation attributes for resource memory: allocator id, page size
/// (bits 14-15: 2 MiB pages from 2 MiB up, else 64 KiB), alignment as log2
/// (bits 24-28) and memory type (bits 29-32: write-combined for GPU-only
/// heaps, cacheable for the CPU-visible ones).
/// shortcut: the console runtime's own choice of page size and memory type
/// is not known; this is the plausible one.
fn resource_attributes(info: AllocationInfo, heap_kind: u32) -> u64 {
    const GRAPHICS_ALLOCATOR: u64 = 128 << 16;
    const DEFAULT_HEAP: u32 = 1;
    let pages: u64 = if info.size >= 2 << 20 { 2 } else { 1 };
    let memory: u64 = if heap_kind == DEFAULT_HEAP { 1 } else { 3 };
    let alignment = u64::from(info.alignment.max(1).trailing_zeros());
    GRAPHICS_ALLOCATOR | pages << 14 | alignment << 24 | memory << 29
}

impl Memory {
    /// A committed resource's memory: from the title's allocator if the
    /// runtime named one, else our own; zeroed either way.
    fn for_resource(info: AllocationInfo, heap_kind: u32) -> Result<Memory, HRESULT> {
        let Some(&(alloc, _)) = ALLOCATOR.get() else {
            return Memory::new(info);
        };
        let layout = Layout::from_size_align(info.size as usize, info.alignment as usize)
            .map_err(|_| E_INVALIDARG)?;
        let attributes = resource_attributes(info, heap_kind);
        let ptr =
            NonNull::new(alloc(layout.size(), attributes).cast::<u8>()).ok_or(E_OUTOFMEMORY)?;
        // SAFETY: the allocator handed out `size` bytes.
        unsafe { ptr.as_ptr().write_bytes(0, layout.size()) };
        Ok(Memory {
            ptr,
            layout,
            attributes: Some(attributes),
        })
    }

    /// `size` zeroed bytes aligned to `alignment` (a power of two).
    pub(crate) fn zeroed(size: u64, alignment: u64) -> Result<Memory, HRESULT> {
        Memory::new(AllocationInfo { size, alignment })
    }

    pub(crate) fn address(&self) -> u64 {
        self.ptr.as_ptr() as u64
    }

    fn new(info: AllocationInfo) -> Result<Memory, HRESULT> {
        let layout = usize::try_from(info.size)
            .ok()
            .zip(usize::try_from(info.alignment).ok())
            .filter(|&(size, _)| size > 0)
            .and_then(|(size, align)| Layout::from_size_align(size, align).ok())
            .ok_or(E_INVALIDARG)?;
        // SAFETY: the size is non-zero.
        let ptr = NonNull::new(unsafe { alloc_zeroed(layout) }).ok_or(E_OUTOFMEMORY)?;
        Ok(Memory {
            ptr,
            layout,
            attributes: None,
        })
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        match (self.attributes, ALLOCATOR.get()) {
            (Some(attributes), Some(&(_, free))) => free(self.ptr.as_ptr().cast(), attributes),
            // SAFETY: allocated in `new` with this layout.
            _ => unsafe { dealloc(self.ptr.as_ptr(), self.layout) },
        }
    }
}

#[repr(C)]
struct Heap {
    object: Object,
    desc: HeapDesc,
    memory: Memory,
}

pub(crate) extern "system" fn create_heap(
    _: This,
    desc: *const HeapDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(
        size = desc.size,
        kind = desc.properties.kind,
        flags = format_args!("{:#x}", desc.flags),
        "heap"
    );
    if desc.alignment != 0 && !desc.alignment.is_power_of_two() {
        return E_INVALIDARG;
    }
    let info = AllocationInfo {
        size: desc.size,
        alignment: desc.alignment.max(KIB_64),
    };
    let memory = match Memory::new(info) {
        Ok(memory) => memory,
        Err(hr) => return hr,
    };
    let heap = Box::new(Heap {
        object: Object::new(&HEAP_FUNCTIONS),
        desc,
        memory,
    });
    object::hand_out(HEAP, heap, iid, &HEAP_IIDS, out)
}

extern "system" fn heap_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(HEAP, iid, object::is(iid, &HEAP_IIDS).then_some(this), out)
}
extern "system" fn heap_release(this: This) -> u32 {
    object::release_boxed::<Heap>(this)
}
extern "system" fn heap_desc(this: This, out: *mut HeapDesc) {
    if !out.is_null() {
        // SAFETY: `this` is a heap; the title passes room for a description.
        unsafe { out.write((*this.cast::<Heap>()).desc) };
    }
}

/// A resource's memory: its own, or a range of a heap it holds a reference
/// to.
enum Backing {
    Committed(Memory),
    Placed {
        heap: This,
        offset: u64,
    },
    /// Memory the title mapped itself, at this address.
    Address(*mut u8),
}

impl Drop for Backing {
    fn drop(&mut self) {
        if let Backing::Placed { heap, .. } = *self {
            heap_release(heap);
        }
    }
}

#[repr(C)]
struct Resource {
    object: Object,
    desc: ResourceDesc,
    heap: HeapProperties,
    heap_flags: u32,
    backing: Backing,
    /// Textures also live on the host GPU (decision 0007); buffers do not
    /// yet.
    #[cfg(windows)]
    host: Option<std::sync::Arc<HostTexture>>,
    /// Whether `host` is another texture's copy (`alias`).
    #[cfg(windows)]
    shared: bool,
    /// The alias registry's count of changes (`alias::changes`) when
    /// [`alias_copy`] last found no GPU-written texture under it to copy
    /// from (0: not looked yet): it looks again only after a change.
    #[cfg(windows)]
    beside: std::sync::atomic::AtomicU64,
    /// For a depth texture the title also writes through unordered
    /// access, the colour texture shaders write instead ([`Twin`]): made
    /// with the first unordered access view of it ([`twin_view`]). Many
    /// depth textures ask for unordered access and never get a view (the
    /// shadow maps the title places anew every frame); a committed twin
    /// for each, at its creation, cost a video memory admission apiece
    /// (0.3 to 0.5 s once VidMm throttles): the world took 90 s to load.
    #[cfg(windows)]
    twin: std::sync::OnceLock<Option<Arc<Twin>>>,
}

/// A title texture's host copy and the state it is in. The title's barriers
/// name console states (looser than the host's), so we transition from this
/// one. Lists that use the texture hold it until they are reset: the title
/// may release a resource whose commands it recorded but not yet submitted
/// (console memory outlives the object; a host texture must outlive the
/// lists that name it). That also keeps a pooled texture's range (`pool`)
/// from going to another texture while lists still name it.
#[cfg(windows)]
pub(crate) struct HostTexture {
    pub(crate) texture: windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    pub(crate) state: std::sync::atomic::AtomicU32,
    #[cfg(test)]
    decode_count: std::sync::atomic::AtomicU64,
    decode_tracked: bool,
    content: std::sync::atomic::AtomicU64,
    /// Bytes of video memory it takes (counted in [`format::report`]).
    pub(crate) bytes: u64,
    /// Of them, what its stand-in format takes beyond the title's.
    extra: u64,
    /// Which title creation path allocated this copy. Shared views do not
    /// allocate another copy.
    origin: TextureOrigin,
    /// Whether its memory is resident yet (its own, or its pool heap's):
    /// until then nothing hands it to the GPU ([`HostTexture::ready`]).
    residency: Arc<residency::Residency>,
    /// Its range of a pool heap, given back when it is dropped (after the
    /// texture, which is released first).
    slot: Option<pool::Slot>,
}

#[cfg(windows)]
impl HostTexture {
    /// Whether the GPU may use it: its memory is resident. A texture is
    /// made without waiting for that (`residency`); until it is, lookups
    /// answer as if it had no host copy (null views, no draws through it).
    pub(crate) fn ready(&self) -> bool {
        self.residency.is_ready()
    }
}

#[cfg(windows)]
impl Drop for HostTexture {
    fn drop(&mut self) {
        format::note_texture(
            false,
            self.bytes,
            self.extra,
            self.slot.is_some(),
            self.origin,
        );
    }
}

/// The host copy of `this`, if it has one the GPU may use now.
#[cfg(windows)]
fn ready_host(this: This) -> Option<&'static std::sync::Arc<HostTexture>> {
    // SAFETY: callers keep the resource alive while using the host copy.
    // The background descriptor mirror acquires a reference from VIEWS.
    let r = unsafe { &*this.cast::<Resource>() };
    r.host.as_ref().filter(|h| h.ready())
}

/// Moves the host texture to `wanted` (a host texture state); the host
/// resource and the state it leaves, if that is a change.
#[cfg(windows)]
pub(crate) fn host_transition(this: This, wanted: u32) -> Option<(*mut c_void, u32)> {
    use windows::core::Interface;
    let host = ready_host(this)?;
    let before = host
        .state
        .swap(wanted, std::sync::atomic::Ordering::Relaxed);
    (before != wanted).then(|| (host.texture.as_raw(), before))
}

/// The shader view to make of title texture `this` for the title's view
/// `desc` (none: the whole texture): as it is, but for a texture whose host
/// format has an alpha its title format lacks ([`format::alpha_one`]),
/// whose alpha then reads one, as on the console (after VIctory #492's
/// stand-in, from the review of its port).
#[cfg(windows)]
pub(crate) fn view_alpha(
    this: This,
    desc: Option<windows::Win32::Graphics::Direct3D12::D3D12_SHADER_RESOURCE_VIEW_DESC>,
) -> Option<windows::Win32::Graphics::Direct3D12::D3D12_SHADER_RESOURCE_VIEW_DESC> {
    let Some((_, title)) = texture_memory(this) else {
        return desc;
    };
    if !format::alpha_one(title.format) {
        return desc;
    }
    let mut view = match desc {
        Some(view) => view,
        // SAFETY: a live resource.
        None => whole_view(&unsafe { host_resource(this)?.GetDesc() }),
    };
    view.Shader4ComponentMapping = format::mapping_alpha_one(view.Shader4ComponentMapping);
    Some(view)
}

/// A shader view of the whole of a texture described by `d`, as D3D12
/// makes one without a description (with its default component mapping).
#[cfg(windows)]
fn whole_view(
    d: &windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DESC,
) -> windows::Win32::Graphics::Direct3D12::D3D12_SHADER_RESOURCE_VIEW_DESC {
    use windows::Win32::Graphics::Direct3D12::*;
    let all = u32::MAX;
    let arrays = u32::from(d.DepthOrArraySize);
    let mut view = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: d.Format,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        ..Default::default()
    };
    match (d.Dimension, arrays > 1, d.SampleDesc.Count > 1) {
        (D3D12_RESOURCE_DIMENSION_TEXTURE1D, false, _) => {
            view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D;
            view.Anonymous.Texture1D.MipLevels = all;
        }
        (D3D12_RESOURCE_DIMENSION_TEXTURE1D, true, _) => {
            view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1DARRAY;
            view.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_SRV {
                MipLevels: all,
                ArraySize: arrays,
                ..Default::default()
            };
        }
        (D3D12_RESOURCE_DIMENSION_TEXTURE3D, ..) => {
            view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
            view.Anonymous.Texture3D.MipLevels = all;
        }
        (_, false, true) => view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMS,
        (_, true, true) => {
            view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY;
            view.Anonymous.Texture2DMSArray.ArraySize = arrays;
        }
        (_, false, false) => {
            view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            view.Anonymous.Texture2D.MipLevels = all;
        }
        (_, true, false) => {
            view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
            view.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_SRV {
                MipLevels: all,
                ArraySize: arrays,
                ..Default::default()
            };
        }
    }
    view
}

/// The shader view of the whole of title texture `this`, spelled out in
/// the title's format, for a view the title made without a description:
/// `None` where D3D12's own view of the host copy reads that format (a
/// host copy in it). A typeless host copy (depth that shaders read, sRGB
/// with unordered access: [`host_desc`]) has no view without a
/// description, and its slot read zeros; a host copy shared with a texture
/// of another format of the family (`alias`) read in that format. Depth is
/// read in the color format of its depth plane.
#[cfg(windows)]
pub(crate) fn whole_typed_view(
    this: This,
) -> Option<windows::Win32::Graphics::Direct3D12::D3D12_SHADER_RESOURCE_VIEW_DESC> {
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
    // SAFETY: a live resource.
    let d = unsafe { host_resource(this)?.GetDesc() };
    let host = d.Format.0 as u32;
    // Depth with stencil: its depth plane, which a view's plane 0 is.
    let read = match pc_format(resource(this).desc.format) {
        20 => 21,
        40 => 41,
        45 => 46,
        55 => 56,
        f => f,
    };
    if read == host || family(read) != family(host) {
        return None;
    }
    if family_members(host).is_some() {
        format::note_whole_view(read);
    }
    let mut view = whole_view(&d);
    view.Format = DXGI_FORMAT(read as i32);
    Some(view)
}

/// The host GPU's copy of a title resource, if it has one the GPU may use
/// now (resident: [`HostTexture::ready`]).
#[cfg(windows)]
pub(crate) fn host_resource<'a>(
    this: This,
) -> Option<&'a windows::Win32::Graphics::Direct3D12::ID3D12Resource> {
    if this.is_null() {
        return None;
    }
    ready_host(this).map(|h| &h.texture)
}

/// A title texture's host copy, shared, for a list to hold (once the GPU
/// may use it, as [`host_resource`]).
#[cfg(windows)]
pub(crate) fn host_copy(this: This) -> Option<std::sync::Arc<HostTexture>> {
    if this.is_null() {
        return None;
    }
    ready_host(this).cloned()
}

/// The host's resource flags for the title's: render target, depth,
/// unordered access and no shader access; a depth target loses unordered
/// access, which PC refuses with it (the console allows both; BTP 6
///'s shadow maps ask for it).
/// shortcut: the host copy loses UAV use; give it a second view-only copy
/// if draws write it as a UAV (VIctory #469).
#[cfg(windows)]
fn host_flags(flags: u32) -> u32 {
    const PC_FLAGS: u32 = 0xf;
    const DEPTH: u32 = 2;
    const UNORDERED: u32 = 4;
    let flags = flags & PC_FLAGS;
    if flags & DEPTH != 0 {
        flags & !UNORDERED
    } else {
        flags
    }
}

/// The host description of a title texture for a device that can do
/// `caps`: the same description without the console's layout and flags, in
/// the host's format. Its error says why the device cannot create it (no
/// driver call is made then).
#[cfg(windows)]
pub(crate) fn host_desc(
    desc: &ResourceDesc,
    caps: &impl format::Caps,
) -> Result<ResourceDesc, String> {
    const UNORDERED: u32 = 4;
    let mut d = *desc;
    d.layout = 0;
    d.alignment = 0;
    d.flags = host_flags(d.flags);
    if caps.choice(desc.format) == Some(format::Choice::Refused) {
        return Err("no host format".into());
    }
    d.format = caps.host_format(d.format);
    let typed = d.format;
    const DENY_SHADER_RESOURCE: u32 = 8;
    if d.flags & DENY_SHADER_RESOURCE == 0
        && let Some(&(_, typeless)) = DEPTH_FORMATS.iter().find(|&&(f, _)| f == d.format)
    {
        d.format = typeless;
    }
    // The host has no sRGB unordered access: such textures are typeless,
    // their unordered access views plain UNORM.
    if d.flags & UNORDERED != 0 && SRGB.iter().any(|&(srgb, _)| srgb == d.format) {
        d.format = family(d.format);
    }
    if let Some(why) = format::refusal(caps.support(d.format), d.dimension, d.flags) {
        return Err(format!("as {}: {why}", d.format));
    }
    if !caps.samples(typed, d.sample_count) {
        return Err(format!("as {typed}: no {} samples", d.sample_count));
    }
    Ok(d)
}

/// A host texture like the title's ([`host_desc`]), in GPU memory: placed
/// in the texture pool if it is one the title fills itself (`pool`), else
/// a committed resource. `clear` is the title's clear value (format, then
/// color or depth and stencil), read when the title made the resource: the
/// one entry point to make a host texture from a recorded description,
/// later than the title's call if need be.
///
/// The title's thread never waits on VidMm: a session stalled for good in
/// `CreateCommittedResource` (in `D3DKMTWaitForSynchronizationObjectFromCpu`,
/// the GPU idle) on the way into the world, 2026-11-24. Memory is asked to
/// become resident and the texture handed out at once; the GPU gets it once
/// it is resident (`residency`). Big, slow or over-budget committed
/// creations are logged before and after the call with their size and the
/// video memory budget, so a stall names its texture.
#[cfg(windows)]
fn host_texture(
    desc: &ResourceDesc,
    state: u32,
    clear: Option<[u32; 5]>,
    origin: TextureOrigin,
) -> Option<HostTexture> {
    use windows::Win32::Graphics::Direct3D12::*;
    if desc.dimension == BUFFER {
        return None;
    }
    let host = crate::host::host()?;
    let d = match host_desc(desc, &format::HostCaps) {
        Ok(d) => d,
        Err(why) => {
            refuse_texture(desc, why);
            return None;
        }
    };
    let start = D3D12_RESOURCE_STATES(crate::host::legal_for(
        d.flags,
        crate::host::texture_state(state),
    ) as i32);
    // SAFETY: the description has the host layout (checked size, same
    // fields).
    let hd = unsafe { std::ptr::from_ref(&d).cast::<D3D12_RESOURCE_DESC>().read() };
    let placed = match pooled(desc, &d).then(pool::global).flatten() {
        Some(pool) => {
            let started = std::time::Instant::now();
            let placed = pool.place(&hd, start);
            let took = started.elapsed();
            if took >= CreationWatch::SLOW {
                tracing::warn!(
                    ms = took.as_secs_f64() * 1000.0,
                    desc = ?d,
                    "a host texture was slow to place in the pool"
                );
            }
            match placed {
                Ok(p) => Some(p),
                Err(why) => {
                    note_not_placed(desc, why);
                    None
                }
            }
        }
        None => None,
    };
    let (texture, bytes, residency, slot) = match placed {
        Some(p) => (p.texture, p.bytes, p.residency, Some(p.slot)),
        None => {
            let (texture, bytes, residency) = committed(host, desc, &d, &hd, start, clear)?;
            (texture, bytes, residency, None)
        }
    };
    let stand_in = pc_format(desc.format);
    let mut extra = 0;
    if stand_in != pc_vertex_format(desc.format) {
        format::note_stand_in(desc.format, stand_in);
        // The stand-in's bytes beyond the title format's (twice as many
        // for every stand-in of the table).
        if let (Some((title, _)), Some((own, _))) = (element(desc.format), element(stand_in))
            && own > title
        {
            extra = bytes / own * (own - title);
        }
    }
    crate::host::name(&texture, || {
        format!(
            "texture {}x{} f{} flags {:#x}{}",
            d.width,
            d.height,
            d.format,
            d.flags,
            if slot.is_some() { " pooled" } else { "" }
        )
    });
    format::note_texture(true, bytes, extra, slot.is_some(), origin);
    let state = crate::host::legal_state(&texture, crate::host::texture_state(state));
    Some(HostTexture {
        texture,
        state: state.into(),
        #[cfg(test)]
        decode_count: Default::default(),
        decode_tracked: desc.sample_count == 1
            && ((d.flags & 5 != 0 && element(stand_in) == Some((4, 1)))
                || (desc.format == 67 && stand_in == 10)),
        content: decode::revision().into(),
        bytes,
        extra,
        origin,
        residency,
        slot,
    })
}

/// Whether the host texture `d` of the title's `desc` goes to the pool: a
/// texture the title fills itself (no target, depth or unordered access:
/// `texture::upload` writes every texel from title memory before a shader
/// reads it, so its pool memory needs no zeroing), with one sample, not a
/// planar video format, not one whose contents are not uploaded (its
/// committed memory stays zero, as before). Targets, depth, unordered
/// access and MSAA textures stay committed resources.
#[cfg(windows)]
fn pooled(desc: &ResourceDesc, d: &ResourceDesc) -> bool {
    const PLANAR: [u32; 8] = [103, 104, 105, 106, 110, 130, 131, 132];
    matches!(d.dimension, 2..=4)
        && d.flags & alias::WRITABLE == 0
        && d.sample_count <= 1
        && !PLANAR.contains(&d.format)
        && format::conversion(desc.format, d.format) != Some(None)
}

/// Counts a texture the pool did not place (committed instead), by why;
/// the first of each reason is logged.
#[cfg(windows)]
fn note_not_placed(desc: &ResourceDesc, why: &'static str) {
    static SEEN: std::sync::Mutex<Vec<&'static str>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&why) {
        seen.push(why);
        drop(seen);
        tracing::info!(
            ?desc,
            why,
            "a host texture not placed in the pool: committed (first of this reason)"
        );
    }
}

/// How long a committed texture's creation waits for its residency: below
/// VidMm's ceiling it takes well under a millisecond, past it the texture
/// is handed out not resident (the GPU gets it once it is).
#[cfg(windows)]
const COMMITTED_WAIT: std::time::Duration = std::time::Duration::from_millis(10);

/// A committed host texture of the title's `desc` (made as `d`, `hd`, in
/// `start`), the bytes it takes and its residency, asked for and not
/// waited for beyond [`COMMITTED_WAIT`]; targets and depth get a high
/// residency priority. `None`, logged and counted, if the driver refuses
/// it.
#[cfg(windows)]
fn committed(
    host: &crate::host::Host,
    desc: &ResourceDesc,
    d: &ResourceDesc,
    hd: &windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DESC,
    start: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES,
    clear: Option<[u32; 5]>,
) -> Option<(
    windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    u64,
    Arc<residency::Residency>,
)> {
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::core::Interface;
    const PC_FLAGS: u32 = 0xf;
    // The clear value names the format too: format, then color or depth
    // and stencil (5 words).
    let clear_value = clear.map(|mut value| {
        value[0] = pc_format(value[0]);
        value
    });
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    // SAFETY: a host description.
    let bytes = unsafe {
        host.device
            .GetResourceAllocationInfo(0, std::slice::from_ref(hd))
    }
    .SizeInBytes;
    // All ones: a description the device cannot size (it refuses it below).
    let bytes = if bytes == u64::MAX { 0 } else { bytes };
    let watch = CreationWatch::before(&host.device, d, bytes);
    let mut texture: Option<ID3D12Resource> = None;
    // SAFETY: the description has the host layout; the out pointer is
    // ours.
    let made = unsafe {
        host.device.CreateCommittedResource(
            &heap,
            residency::heap_flags(),
            hd,
            start,
            // The same layout: format, then color or depth and stencil.
            clear_value.as_ref().map(|v| std::ptr::from_ref(v).cast()),
            &mut texture,
        )
    };
    watch.after(&host.device, d, bytes, made.is_ok());
    if let Err(e) = made {
        tracing::warn!(error = %e, desc = ?d, "no host texture");
        let reason = format!("as {}: the driver refused it ({})", d.format, e.code());
        format::note_refused(format!(
            "format {} flags {:#x}: {reason}",
            desc.format,
            desc.flags & PC_FLAGS
        ));
        return None;
    }
    let texture = texture?;
    let pageable: Option<ID3D12Pageable> = texture.cast().ok();
    const TARGETS: u32 = 0x1 | 0x2;
    if d.flags & TARGETS != 0
        && let Ok(device1) = host.device.cast::<ID3D12Device1>()
    {
        // SAFETY: one live pageable, one priority.
        let set =
            unsafe { device1.SetResidencyPriority(1, &pageable, &D3D12_RESIDENCY_PRIORITY_HIGH) };
        if let Err(e) = set {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::info!(error = %e, "no residency priority for targets"));
        }
    }
    let residency = if residency::direct() {
        residency::Residency::resident()
    } else {
        let r = residency::Residency::ask(&host.device, vec![pageable]);
        if !r.wait(COMMITTED_WAIT) {
            static PENDING: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = PENDING.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if n <= 8 || n.is_power_of_two() {
                let (budget, usage) = crate::dxgi::video_memory(&host.device).unwrap_or_default();
                tracing::warn!(
                    n,
                    mib = bytes >> 20,
                    budget_mib = budget >> 20,
                    usage_mib = usage >> 20,
                    refused = r.refused(),
                    desc = ?d,
                    "a committed host texture handed out before it is resident: unused by the GPU until it is"
                );
            }
        }
        r
    };
    Some((texture, bytes, Arc::new(residency)))
}

/// What is logged around a host texture's creation: whether it is big or
/// over the video memory budget (then a line before the call), and when it
/// started (a slow one gets a line after).
#[cfg(windows)]
struct CreationWatch {
    started: std::time::Instant,
    told: bool,
}

#[cfg(windows)]
impl CreationWatch {
    /// From 64 MiB a creation is logged.
    const BIG: u64 = 64 << 20;
    /// A creation longer than this is logged after.
    const SLOW: std::time::Duration = std::time::Duration::from_millis(100);

    fn before(
        device: &windows::Win32::Graphics::Direct3D12::ID3D12Device,
        d: &ResourceDesc,
        bytes: u64,
    ) -> CreationWatch {
        let memory = crate::dxgi::video_memory(device);
        let over = memory.is_some_and(|(budget, usage)| usage.saturating_add(bytes) > budget);
        let told = bytes >= Self::BIG || over;
        if told {
            let (budget, usage) = memory.unwrap_or_default();
            tracing::info!(
                mib = bytes >> 20,
                budget_mib = budget >> 20,
                usage_mib = usage >> 20,
                over,
                desc = ?d,
                "making a host texture"
            );
        }
        if over {
            static OVER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = OVER.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if n.is_power_of_two() {
                tracing::warn!(
                    n,
                    "a host texture past the video memory budget: making it resident may wait"
                );
            }
        }
        CreationWatch {
            started: std::time::Instant::now(),
            told,
        }
    }

    fn after(
        self,
        device: &windows::Win32::Graphics::Direct3D12::ID3D12Device,
        d: &ResourceDesc,
        bytes: u64,
        made: bool,
    ) {
        let took = self.started.elapsed();
        if !self.told && took < Self::SLOW {
            return;
        }
        let (budget, usage) = crate::dxgi::video_memory(device).unwrap_or_default();
        let ms = took.as_secs_f64() * 1000.0;
        if took >= Self::SLOW {
            tracing::warn!(
                ms,
                mib = bytes >> 20,
                budget_mib = budget >> 20,
                usage_mib = usage >> 20,
                made,
                desc = ?d,
                "a host texture was slow to make"
            );
        } else {
            tracing::info!(
                ms,
                mib = bytes >> 20,
                usage_mib = usage >> 20,
                made,
                "host texture made"
            );
        }
    }
}

/// Counts a texture the host cannot create, by its title format, flags and
/// `why`; the first of each reason is logged with the description.
#[cfg(windows)]
fn refuse_texture(desc: &ResourceDesc, why: String) {
    const PC_FLAGS: u32 = 0xf;
    let reason = format!(
        "format {} flags {:#x}: {why}",
        desc.format,
        desc.flags & PC_FLAGS
    );
    if format::note_refused(reason.clone()) {
        tracing::warn!(
            ?desc,
            "no host texture: {reason} (refused before the driver; logged once)"
        );
    }
}

/// A texture's console memory and description; `None` for buffers.
pub(crate) fn texture_memory(this: This) -> Option<(*const u8, ResourceDesc)> {
    if this.is_null() {
        return None;
    }
    let r = resource(this);
    (r.desc.dimension != BUFFER).then(|| (r.data().cast_const(), r.desc))
}

/// Width, height and format of a single-sample 2D texture (`None` for buffers
/// or other textures), for [`crate::texback`].
#[cfg(windows)]
pub(crate) fn texture_kind(this: This) -> Option<(u64, u32, u32)> {
    if this.is_null() {
        return None;
    }
    let r = resource(this);
    const TEXTURE_2D: u32 = 3;
    (r.desc.dimension == TEXTURE_2D && r.desc.sample_count <= 1).then_some((
        r.desc.width,
        r.desc.height,
        r.desc.format,
    ))
}

/// Single-plane console layout, title memory address and byte size of single-sample
/// 2D texture `this`, for [`crate::texback`]; `None` for buffers, multisampled or
/// non-2D textures, multi-plane textures, or layouts without CPU tiling support.
#[cfg(windows)]
pub(crate) fn title_texture(this: This) -> Option<(xtp_formats::texture::Layout, usize, usize)> {
    if this.is_null() {
        return None;
    }
    let r = resource(this);
    const TEXTURE_2D: u32 = 3;
    if r.desc.dimension != TEXTURE_2D || r.desc.sample_count > 1 {
        return None;
    }
    let t = texture_layout(&r.desc)?;
    let [plane] = &t.planes[..] else {
        return None;
    };
    if plane.offset != 0 || !plane.layout.addressable() {
        return None;
    }
    let size = usize::try_from(t.size).ok()?;
    Some((plane.layout.clone(), r.data() as usize, size))
}

/// A buffer's host memory and size; `None` for textures.
pub(crate) fn buffer_bytes(this: This) -> Option<(*mut u8, u64)> {
    if this.is_null() {
        return None;
    }
    let r = resource(this);
    (r.desc.dimension == BUFFER).then(|| (r.data(), r.desc.width))
}

impl Backing {
    /// The memory's first byte.
    fn data(&self) -> *mut u8 {
        match *self {
            Backing::Committed(ref memory) => memory.ptr.as_ptr(),
            // SAFETY: placement checked the range lies inside the heap.
            Backing::Placed { heap, offset } => unsafe {
                (*heap.cast::<Heap>())
                    .memory
                    .ptr
                    .as_ptr()
                    .add(offset as usize)
            },
            Backing::Address(at) => at,
        }
    }

    /// Its address (GPU addresses are CPU addresses).
    #[cfg(windows)]
    fn address(&self) -> u64 {
        self.data() as u64
    }
}

impl Resource {
    fn data(&self) -> *mut u8 {
        self.backing.data()
    }
}

fn resource<'a>(this: This) -> &'a Resource {
    // SAFETY: called from resource methods only.
    unsafe { &*this.cast::<Resource>() }
}

#[cfg(windows)]
#[derive(Clone, Copy)]
enum TextureOrigin {
    Committed,
    HeapOffset,
    Address,
    PlacedView,
    External,
}

// The creation methods' own arguments, passed on, and how the resource is
// made (for the alias log).
#[allow(clippy::too_many_arguments)]
fn hand_out_resource(
    desc: ResourceDesc,
    heap: HeapProperties,
    heap_flags: u32,
    backing: Backing,
    state: u32,
    clear: *const c_void,
    iid: *const Iid,
    out: *mut This,
    how: &'static str,
) -> HRESULT {
    #[cfg(not(windows))]
    let _ = (state, clear, how);
    // A texture at the address of a live one the GPU writes shares its host
    // copy (`alias`), with the state it is in; else it gets its own.
    #[cfg(windows)]
    let address = backing.address();
    #[cfg(windows)]
    let shared = if desc.dimension == BUFFER {
        None
    } else {
        alias::shared(address, &desc, how)
    };
    #[cfg(windows)]
    let (host, shared) = match shared {
        Some(copy) => (Some(copy), true),
        None if desc.dimension == BUFFER => (None, false),
        None => {
            // The clear value, read now: format, then color or depth and
            // stencil (5 words).
            let clear = (!clear.is_null()).then(|| {
                // SAFETY: the title passes a whole clear value.
                unsafe { clear.cast::<[u32; 5]>().read_unaligned() }
            });
            let origin = match &backing {
                Backing::Committed(_) => TextureOrigin::Committed,
                Backing::Placed { .. } => TextureOrigin::HeapOffset,
                Backing::Address(_) if how == PLACED_VIEW => TextureOrigin::PlacedView,
                Backing::Address(_) => TextureOrigin::Address,
            };
            (
                host_texture(&desc, state, clear, origin).map(Arc::new),
                false,
            )
        }
    };
    let resource = Box::new(Resource {
        object: Object::new(&RESOURCE_FUNCTIONS),
        desc,
        heap,
        heap_flags,
        backing,
        #[cfg(windows)]
        host,
        #[cfg(windows)]
        shared,
        #[cfg(windows)]
        beside: std::sync::atomic::AtomicU64::new(0),
        #[cfg(windows)]
        twin: std::sync::OnceLock::new(),
    });
    #[cfg(windows)]
    {
        let at = std::ptr::from_ref::<Resource>(&resource) as usize;
        if desc.dimension == BUFFER {
            crate::shadow::register(at, resource.data() as u64, desc.width);
        } else {
            if how != PLACED_VIEW
                && !resource.shared
                && crate::texback::tracking()
                && alias::written_beside(address, &desc).is_none()
            {
                let len = footprint(&desc).map_or(1, |info| info.size as usize);
                crate::texback::replaced(address as usize, len);
            }
            if let Some(host) = &resource.host {
                alias::register(address, at, &desc, host);
            }
            // A texture the GPU does not write, made at the address by the
            // title: the memory is used anew, and views there that showed
            // a released target's copy are stale.
            if how != PLACED_VIEW && host_flags(desc.flags) & alias::WRITABLE == 0 {
                forget_shared_views(address);
            }
        }
    }
    object::hand_out(RESOURCE, resource, iid, &RESOURCE_IIDS, out)
}

/// Whether `this` is one of our resources: false for any other pointer,
/// readable or not (the planes of a present, read at a guessed stride).
#[cfg(windows)]
pub(crate) fn is_resource(this: This) -> bool {
    #[cfg(all(windows, feature = "diagnostics"))]
    if crate::record::is_proxy_of(this, RESOURCE_FUNCTIONS.0.as_ptr().cast()) {
        return true;
    }
    let mut table = [0u8; 8];
    !this.is_null()
        && crate::host::read_into(this.cast(), &mut table)
        && usize::from_le_bytes(table) == RESOURCE_FUNCTIONS.0.as_ptr() as usize
}

#[cfg(windows)]
impl Drop for Resource {
    fn drop(&mut self) {
        #[cfg(feature = "diagnostics")]
        if self
            .host
            .as_ref()
            .is_some_and(|h| std::sync::Arc::strong_count(h) > 1)
        {
            // Before lists held their textures, this freed a texture that
            // lists not yet run still named (a GPU page fault).
            static OUTLIVED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = OUTLIVED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if n.is_power_of_two() {
                tracing::info!(
                    n,
                    desc = ?self.desc,
                    "a title texture was released while a list not yet reset (or an alias, `alias`) names its host copy"
                );
            }
        }
        crate::texture::forget(std::ptr::from_ref::<Resource>(self) as usize);
        if self.desc.dimension == BUFFER {
            crate::shadow::unregister(
                std::ptr::from_ref::<Resource>(self) as usize,
                self.data() as u64,
            );
        } else {
            crate::texback::released(std::ptr::from_ref::<Resource>(self) as usize);
            alias::unregister(
                self.data() as u64,
                std::ptr::from_ref::<Resource>(self) as usize,
            );
        }
    }
}

pub(crate) extern "system" fn create_committed(
    _: This,
    heap: *const HeapProperties,
    heap_flags: u32,
    desc: *const ResourceDesc,
    state: u32,
    clear: *const c_void,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes both descriptions (null is tolerated).
    let (Some(&heap), Some(&desc)) = (unsafe { (heap.as_ref(), desc.as_ref()) }) else {
        return E_INVALIDARG;
    };
    let Some(info) = footprint(&desc) else {
        tracing::warn!(?desc, "no size for this resource");
        return E_INVALIDARG;
    };
    tracing::debug!(
        dimension = desc.dimension,
        size = info.size,
        heap = heap.kind,
        "committed resource"
    );
    let memory = match Memory::for_resource(info, heap.kind) {
        Ok(memory) => memory,
        Err(hr) => return hr,
    };
    hand_out_resource(
        desc,
        heap,
        heap_flags,
        Backing::Committed(memory),
        state,
        clear,
        iid,
        out,
        "committed",
    )
}

/// Device method 0x3f8, inferred from BTP 6's calls
/// (title+0x193e1b4: its back buffers, heap flag "allow display"): the
/// committed resource creation of the enhanced-barrier interfaces,
/// `(heap, heap flags, desc1, initial layout, clear value, protected
/// session, castable format count, castable formats, iid, out)`. The host
/// copy starts in the common state, as layouts are not tracked.
///
/// The castable formats are honoured where the host views a resource in
/// them without one: the host copy's own family (a typeless copy), or
/// within D3D12's casting rules ([`format::castable`]). The host has no
/// castable lists without `ID3D12Device10` and enhanced barriers, which
/// XTP does not use: each listed format is counted for the frame log,
/// whether the host can view the copy in it or not.
#[allow(clippy::too_many_arguments)] // the method's own arguments
pub(crate) extern "system" fn create_committed3(
    this: This,
    heap: *const HeapProperties,
    heap_flags: u32,
    desc: *const ResourceDesc,
    _layout: u32,
    clear: *const c_void,
    _protected: This,
    castable_count: u32,
    castable: *const u32,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    const COMMON: u32 = 0;
    let made = create_committed(this, heap, heap_flags, desc, COMMON, clear, iid, out);
    #[cfg(windows)]
    if made == S_OK && castable_count > 0 && !castable.is_null() && !out.is_null() {
        // SAFETY: the title passes `castable_count` formats; the resource
        // was just made there.
        let (formats, resource) = unsafe {
            (
                std::slice::from_raw_parts(castable, castable_count as usize),
                out.read(),
            )
        };
        note_castable(resource, formats);
    }
    #[cfg(not(windows))]
    let _ = (castable_count, castable);
    made
}

/// Counts the castable formats the title listed for resource `this`
/// (title formats): whether the host can view its host copy in each.
#[cfg(windows)]
fn note_castable(this: This, formats: &[u32]) {
    use format::Caps;
    let Some(host) = &resource(this).host else {
        return;
    };
    // SAFETY: a live resource.
    let own = unsafe { host.texture.GetDesc() }.Format.0 as u32;
    let typeless = family_members(own).is_some();
    for &title in formats {
        let view = pc_format(title);
        if view == own {
            continue;
        }
        let viewable = if typeless {
            family(view) == own
        } else {
            format::castable(own, view) && format::HostCaps.fully_typed_casting()
        };
        format::note_view_cast(own, view, format::ViewUse::Listed, viewable);
    }
}

/// Counts a render target view in host format `view` (none or 0: the
/// texture's own) of host texture `texture` in another format of its typed
/// family: whether D3D12 casts it, or does not allow the cast (the view is
/// made as the title asks either way).
#[cfg(windows)]
pub(crate) fn note_target_view(
    texture: &windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    view: Option<u32>,
) {
    use format::Caps;
    let Some(view) = view.filter(|&v| v != 0) else {
        return;
    };
    // SAFETY: a live resource.
    let own = unsafe { texture.GetDesc() }.Format.0 as u32;
    if own == view || family_members(own).is_some() {
        return;
    }
    let cast = format::castable(own, view) && format::HostCaps.fully_typed_casting();
    format::note_view_cast(own, view, format::ViewUse::Target, cast);
}

/// Registered page pools (64 KiB pages the title maps into tiled
/// resources by page index): first address and page count; a handle is an
/// index plus one, a removed pool leaves `None`.
static PAGE_POOLS: std::sync::Mutex<Vec<Option<(u64, u32)>>> = std::sync::Mutex::new(Vec::new());

fn page_pools() -> std::sync::MutexGuard<'static, Vec<Option<(u64, u32)>>> {
    PAGE_POOLS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Device method 0x250: `(address, page count, handle out)`.
pub(crate) extern "system" fn register_page_pool(
    _: This,
    address: u64,
    pages: u32,
    out: *mut usize,
) -> HRESULT {
    if out.is_null() || pages == 0 || !address.is_multiple_of(KIB_64) {
        return E_INVALIDARG;
    }
    tracing::debug!(address = format_args!("{address:#x}"), pages, "page pool");
    let mut pools = page_pools();
    pools.push(Some((address, pages)));
    // SAFETY: checked non-null; the title passes room for a handle.
    unsafe { out.write(pools.len()) };
    S_OK
}

/// Device method 0x258.
pub(crate) extern "system" fn unregister_page_pool(_: This, handle: usize) -> HRESULT {
    match page_pools().get_mut(handle.wrapping_sub(1)) {
        Some(pool @ Some(_)) => {
            *pool = None;
            S_OK
        }
        _ => E_INVALIDARG,
    }
}

/// Device method 0x428, one of the `ID3D12Device9` to `ID3D12Device11`
/// additions. From BTP 6's call: a resource at a virtual
/// address the title mapped itself, `(address, desc1, initial barrier
/// layout, clear value, iid, out)`, as `CreatePlacedResourceX` with
/// enhanced barriers. GPU addresses are CPU addresses, so the resource is
/// a view of that memory. `D3D12_RESOURCE_DESC1` starts like the plain
/// description; its sampler feedback region is not needed.
pub(crate) extern "system" fn create_at_address(
    _: This,
    address: u64,
    desc: *const ResourceDesc,
    layout: u32,
    clear: *const c_void,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(
        address = format_args!("{address:#x}"),
        dimension = desc.dimension,
        width = desc.width,
        layout,
        "resource at an address"
    );
    at_address(address, desc, clear, iid, out, "at an address")
}

/// A resource over memory at `address` (see [`create_at_address`]); `how`
/// says who asks, for the alias log.
fn at_address(
    address: u64,
    desc: ResourceDesc,
    clear: *const c_void,
    iid: *const Iid,
    out: *mut This,
    how: &'static str,
) -> HRESULT {
    if address == 0 {
        return E_INVALIDARG;
    }
    const COMMON: u32 = 0;
    const DEFAULT: HeapProperties = HeapProperties {
        kind: 1,
        cpu_pages: 0,
        pool: 0,
        creation_nodes: 1,
        visible_nodes: 1,
    };
    hand_out_resource(
        desc,
        DEFAULT,
        0,
        Backing::Address(address as *mut u8),
        COMMON,
        clear,
        iid,
        out,
        how,
    )
}

/// Textures the title placed in its own memory and views without a
/// resource object (device 0x230, BTP 6's streamed and
/// interface textures), by address and description (padding zeroed): a
/// resource at that address, made the first time a draw reads one, the
/// alias count (`alias::changes`) it was last matched at and the bytes it
/// covers. They are released when the runtime maps or unmaps memory they
/// overlap ([`memory_remapped`]), when a new one replaces them, and, when
/// they show a released target's copy, when the title makes another
/// texture at their address ([`forget_shared_views`]).
/// shortcut: memory the title rewrites itself, without remapping it or
/// making a resource there, keeps the textures of its earlier descriptions
/// (and a released target's copy) until then.
#[cfg(windows)]
static PLACED_TEXTURES: std::sync::Mutex<PlacedTextures> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());

/// Placed views' resources by address and description.
#[cfg(windows)]
type PlacedTextures = std::collections::BTreeMap<(u64, [u32; 14]), PlacedView>;

/// A placed view's resource, the alias count it was last matched at and
/// the bytes it covers. New entries can be invalidated while being made.
#[cfg(windows)]
#[derive(Debug, Clone, Copy)]
struct PlacedView {
    resource: Option<usize>,
    changes: u64,
    len: u64,
    /// Nonzero while a thread is making a copy outside the map lock.
    building: u64,
}

/// The most bytes a placed view's texture has covered: how far before a
/// remapped range one may start and still overlap it.
#[cfg(windows)]
static LONGEST_PLACED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// How [`texture_at`] names itself to [`hand_out_resource`] (and the alias
/// log).
#[cfg(windows)]
const PLACED_VIEW: &str = "placed view";

#[cfg(windows)]
static NEXT_PLACEMENT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
#[cfg(windows)]
static UNMAPPED_VIEWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(windows)]
static INVALIDATED_VIEWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Resources of placed views no longer in [`PLACED_TEXTURES`], with when
/// they left it: released [`GRACE`] later, as a caller of `texture_at` may
/// still be using one it was just given (lists keep only their address, and
/// their host copies).
#[cfg(windows)]
static RELEASED: std::sync::Mutex<std::collections::VecDeque<(std::time::Instant, usize)>> =
    std::sync::Mutex::new(std::collections::VecDeque::new());

#[cfg(windows)]
const GRACE: std::time::Duration = std::time::Duration::from_secs(2);

/// Releases `resources` (of placed views) once [`GRACE`] has passed, and
/// those waiting for longer.
#[cfg(windows)]
fn release_later(resources: Vec<usize>) {
    let mut waiting = RELEASED.lock().unwrap_or_else(|e| e.into_inner());
    let now = std::time::Instant::now();
    waiting.extend(resources.into_iter().map(|r| (now, r)));
    drop(waiting);
    release_waiting(now, GRACE);
}

/// Releases the resources that have waited `grace` or more.
#[cfg(windows)]
fn release_waiting(now: std::time::Instant, grace: std::time::Duration) {
    let mut waiting = RELEASED.lock().unwrap_or_else(|e| e.into_inner());
    let due = waiting
        .iter()
        .take_while(|(at, _)| now - *at >= grace)
        .count();
    let due: Vec<usize> = waiting.drain(..due).map(|(_, r)| r).collect();
    drop(waiting);
    for r in due {
        resource_release(r as This);
    }
}

/// Give back expired placed views even if streaming stops remapping memory.
#[cfg(windows)]
pub(crate) fn reap_placed_views() {
    release_waiting(std::time::Instant::now(), GRACE);
}

#[cfg(windows)]
pub(crate) fn log_placed_views() {
    use std::sync::atomic::Ordering::Relaxed;
    let cached = placed_textures().len();
    let retiring = RELEASED.lock().unwrap_or_else(|e| e.into_inner()).len();
    let (same_address, shared) = alias::counts();
    tracing::info!(
        cached,
        retiring,
        unmapped = UNMAPPED_VIEWS.load(Relaxed),
        invalidated = INVALIDATED_VIEWS.load(Relaxed),
        same_address,
        shared,
        "placed texture views"
    );
}

#[cfg(windows)]
fn placed_textures() -> std::sync::MutexGuard<'static, PlacedTextures> {
    PLACED_TEXTURES.lock().unwrap_or_else(|e| e.into_inner())
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: the
/// textures of placed views over any of it (starting there or before) no
/// longer show it, and are released (memory streamed in and out would
/// otherwise keep a host texture per page and description viewed for the
/// run).
#[cfg(windows)]
pub(crate) fn memory_remapped(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let (start, end) = (at as u64, (at as u64).saturating_add(len as u64));
    let gone: Vec<usize> = {
        let mut map = placed_textures();
        let from = start.saturating_sub(LONGEST_PLACED.load(std::sync::atomic::Ordering::Relaxed));
        let keys: Vec<_> = map
            .range((from, [0; 14])..(end, [0; 14]))
            .filter(|((address, _), view)| address.saturating_add(view.len.max(1)) > start)
            .map(|(k, _)| *k)
            .collect();
        keys.iter()
            .filter_map(|k| map.remove(k))
            .filter_map(|view| view.resource)
            .collect()
    };
    if !gone.is_empty() {
        INVALIDATED_VIEWS.fetch_add(gone.len() as u64, std::sync::atomic::Ordering::Relaxed);
        tracing::debug!(
            at = format_args!("{at:#x}"),
            len,
            textures = gone.len(),
            "placed views' textures released with their memory"
        );
    }
    release_later(gone);
}

/// The title made a texture the GPU does not write at `address`: placed
/// views there that show the copy of a target no longer there (`alias`)
/// are stale, and released. Without a new texture there, they keep it:
/// the title may read a target through a view after releasing its object
/// (title memory still holds what was drawn, on the console).
#[cfg(windows)]
fn forget_shared_views(address: u64) {
    let gone: Vec<usize> = {
        let mut map = placed_textures();
        let keys: Vec<_> = map
            .range((address, [0; 14])..=(address, [u32::MAX; 14]))
            .filter(|(_, view)| {
                view.resource.is_some_and(|made| {
                    let r = resource(made as This);
                    r.shared && alias::current(address, &r.desc).is_none()
                })
            })
            .map(|(k, _)| *k)
            .collect();
        keys.iter()
            .filter_map(|k| map.remove(k))
            .filter_map(|view| view.resource)
            .collect()
    };
    if !gone.is_empty() {
        tracing::debug!(
            address = format_args!("{address:#x}"),
            textures = gone.len(),
            "placed views of a released target dropped: a new texture at its address"
        );
        release_later(gone);
    }
}

/// The resource for a texture placed at `address` with the resource
/// description `desc` (as words), made on first use. When a texture the
/// GPU writes comes to that address after it (or another one replaces it),
/// the next use makes a new resource, which shares that texture's host
/// copy (`alias`), and the one it replaces is released ([`release_later`]).
/// The resource is made outside the map's lock: making a host texture can
/// take time, and every draw reading a placed view would wait behind it.
/// Only a resource in the map, looked at under its lock, is handed out:
/// one replaced or remapped meanwhile may be released [`GRACE`] later.
/// `None` while the host copy is not resident yet (the next use looks
/// again).
#[cfg(windows)]
pub(crate) fn texture_at(address: u64, desc: &[u32; 14]) -> Option<This> {
    texture_lookup(address, desc).ready()
}

/// What [`texture_lookup`] found for a placed view.
#[cfg(windows)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PlacedLookup {
    /// The resource made for that memory, its host copy resident.
    Ready(This),
    /// None yet: another thread is still making the first one.
    Building,
    /// None: no mapped page, a failed creation, or a host copy not
    /// resident yet.
    Absent,
}

#[cfg(windows)]
impl PlacedLookup {
    pub(crate) fn ready(self) -> Option<This> {
        match self {
            PlacedLookup::Ready(r) => Some(r),
            _ => None,
        }
    }
}

/// [`texture_at`], telling a view another thread is still making from one
/// there is none for.
#[cfg(windows)]
pub(crate) fn texture_lookup(address: u64, desc: &[u32; 14]) -> PlacedLookup {
    texture_lookup_with(address, desc, |d| {
        let mut out = std::ptr::null();
        let iid = &RESOURCE_IIDS[0];
        (at_address(address, d, std::ptr::null(), iid, &mut out, PLACED_VIEW) == S_OK)
            .then_some(out)
    })
}

#[cfg(all(test, windows))]
fn texture_at_with(
    address: u64,
    desc: &[u32; 14],
    make: impl FnOnce(ResourceDesc) -> Option<This>,
) -> Option<This> {
    texture_lookup_with(address, desc, make).ready()
}

#[cfg(windows)]
fn texture_lookup_with(
    address: u64,
    desc: &[u32; 14],
    make: impl FnOnce(ResourceDesc) -> Option<This>,
) -> PlacedLookup {
    use std::sync::atomic::Ordering::Relaxed;
    let found = |r: Option<This>| r.map_or(PlacedLookup::Absent, PlacedLookup::Ready);
    if address == 0 {
        return PlacedLookup::Absent;
    }
    let desc = canonical_resource_desc(*desc);
    // SAFETY: 56 bytes in the description's layout.
    let d = unsafe { desc.as_ptr().cast::<ResourceDesc>().read_unaligned() };
    if d.dimension == BUFFER {
        return PlacedLookup::Absent;
    }
    let key = (address, desc);
    let usable = |r: usize| {
        let r = r as This;
        ready_host(r).is_some().then_some(r)
    };
    let changes = alias::changes();
    let (ticket, len) = {
        let mut map = placed_textures();
        if let Some(view) = map.get_mut(&key) {
            if view.building != 0 {
                return view
                    .resource
                    .and_then(usable)
                    .map_or(PlacedLookup::Building, PlacedLookup::Ready);
            }
            if let Some(made) = view.resource {
                let stale = view.changes != changes
                    && alias::current(address, &d).is_some_and(|now| {
                        !resource(made as This)
                            .host
                            .as_ref()
                            .is_some_and(|c| Arc::ptr_eq(c, &now))
                    });
                if !stale {
                    view.changes = changes;
                    return found(usable(made));
                }
            }
        }
        // Publish the range before dropping the lock. A remap can now
        // cancel this creation even while the driver is allocating it.
        let len = footprint(&d).map_or(1, |info| info.size);
        LONGEST_PLACED.fetch_max(len, Relaxed);
        let ticket = NEXT_PLACEMENT.fetch_add(1, Relaxed);
        let view = map.entry(key).or_insert(PlacedView {
            resource: None,
            changes,
            len,
            building: 0,
        });
        view.building = ticket;
        (ticket, len)
    };
    // The indexed upload pass already defers views of unmapped pages.
    // Table draws and later descriptor checks also reach this path.
    // Keep GPU-writeable copies and aliases of live GPU sources eligible.
    let out = if host_flags(d.flags) & alias::WRITABLE != 0
        || alias::has_sources(address)
        || crate::texture::any_committed(address as *const u8, len as usize)
    {
        make(d)
    } else {
        UNMAPPED_VIEWS.fetch_add(1, Relaxed);
        None
    };
    let mut map = placed_textures();
    let Some(view) = map.get_mut(&key).filter(|v| v.building == ticket) else {
        let current = match map.get(&key) {
            Some(v) => match v.resource.and_then(usable) {
                Some(r) => PlacedLookup::Ready(r),
                None if v.building != 0 => PlacedLookup::Building,
                None => PlacedLookup::Absent,
            },
            None => PlacedLookup::Absent,
        };
        drop(map);
        if let Some(out) = out {
            resource_release(out);
        }
        return current;
    };
    view.building = 0;
    // Recorded even when the creation failed, as before a creation was made
    // outside the lock: a failed rebuild is not tried again on every lookup
    // (a slow creation on the title's thread), only at the next change.
    view.changes = changes;
    let replaced = out.and_then(|out| view.resource.replace(out as usize));
    let current = view.resource.and_then(usable);
    if view.resource.is_none() {
        map.remove(&key);
    }
    drop(map);
    if let Some(replaced) = replaced {
        release_later(vec![replaced]);
    }
    found(current)
}

/// The buffer a GPU-written texture's texels go through into a texture of
/// the same shape that reads them in another format ([`alias_copy`]), and
/// each level's and layer's layout in it, the written texture's first
/// (the same bytes: the texels are as wide). Made for a pair of host copies.
#[cfg(windows)]
struct Through {
    buffer: windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    from: std::sync::Weak<HostTexture>,
    to: std::sync::Weak<HostTexture>,
    parts: Vec<(
        windows::Win32::Graphics::Direct3D12::D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
        windows::Win32::Graphics::Direct3D12::D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    )>,
}

// SAFETY: D3D12 objects are free-threaded; lists only record copies with
// the buffer.
#[cfg(windows)]
unsafe impl Send for Through {}
// SAFETY: as above.
#[cfg(windows)]
unsafe impl Sync for Through {}

/// The most levels times layers [`alias_copy`] copies (a cube of nine
/// levels).
#[cfg(windows)]
const ALIAS_PARTS: u32 = 54;

/// The buffers of [`alias_copy`], by the reading texture's resource.
#[cfg(windows)]
type Throughs = std::collections::HashMap<usize, Arc<Through>>;

#[cfg(windows)]
static THROUGH: std::sync::Mutex<Option<Throughs>> = std::sync::Mutex::new(None);

/// The [`Through`] copying `from` into `to`, the host copy of resource
/// `key`: made the first time, or again for other copies. `None` if their
/// layouts differ (a stand-in format of another width) or there are too
/// many parts.
#[cfg(windows)]
fn through(
    host: &crate::host::Host,
    key: usize,
    from: &Arc<HostTexture>,
    to: &Arc<HostTexture>,
) -> Option<Arc<Through>> {
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    let mut map = THROUGH.lock().unwrap_or_else(|e| e.into_inner());
    let map = map.get_or_insert_with(Throughs::new);
    if let Some(t) = map.get(&key)
        && std::ptr::eq(t.from.as_ptr(), Arc::as_ptr(from))
        && std::ptr::eq(t.to.as_ptr(), Arc::as_ptr(to))
    {
        return Some(t.clone());
    }
    map.retain(|_, t| t.to.strong_count() > 0 && t.from.strong_count() > 0);
    // SAFETY: live textures.
    let (of_from, of_to) = unsafe { (from.texture.GetDesc(), to.texture.GetDesc()) };
    // Levels times layers (a volume's slices are one part).
    let parts = |d: &D3D12_RESOURCE_DESC| {
        let layers = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
            1
        } else {
            u32::from(d.DepthOrArraySize)
        };
        u32::from(d.MipLevels) * layers
    };
    let n = parts(&of_from);
    if n == 0 || n > ALIAS_PARTS || n != parts(&of_to) {
        return None;
    }
    let mut rows =
        [(); 2].map(|()| vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n as usize]);
    let mut bytes = [0u64; 2];
    // SAFETY: plain queries; the out pointers hold `n` layouts each.
    unsafe {
        for (i, d) in [of_from, of_to].iter().enumerate() {
            host.device.GetCopyableFootprints(
                d,
                0,
                n,
                0,
                Some(rows[i].as_mut_ptr()),
                None,
                None,
                Some(&mut bytes[i]),
            );
        }
    }
    let alike = rows[0].iter().zip(&rows[1]).all(|(a, b)| {
        let (fa, fb) = (a.Footprint, b.Footprint);
        a.Offset == b.Offset
            && (fa.Width, fa.Height, fa.Depth, fa.RowPitch)
                == (fb.Width, fb.Height, fb.Depth, fb.RowPitch)
    });
    if bytes[0] == 0 || bytes[0] != bytes[1] || !alike {
        return None;
    }
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes[0],
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation; the out pointer is ours.
    unsafe {
        host.device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut buffer,
            )
            .ok()?;
    }
    let buffer = buffer?;
    crate::host::name(&buffer, || {
        format!(
            "alias copy {}x{} f{} to f{}",
            of_from.Width, of_from.Height, of_from.Format.0, of_to.Format.0
        )
    });
    tracing::info!(
        width = of_from.Width,
        height = of_from.Height,
        parts = n,
        written_as = of_from.Format.0,
        read_as = of_to.Format.0,
        "a texture read over a GPU-written one in another format: copied from it on the GPU"
    );
    let [of_written, of_read] = rows;
    let through = Arc::new(Through {
        buffer,
        from: Arc::downgrade(from),
        to: Arc::downgrade(to),
        parts: of_written.into_iter().zip(of_read).collect(),
    });
    map.insert(key, through.clone());
    Some(through)
}

/// The program [`decode_now`] runs: R9G9B9E5 texels read as words from one
/// buffer, written as RGBA16F texels (alpha one) into another, each laid
/// out as a copy of one level or layer lays them (`p`: width, height,
/// depth slices, then for the source and the destination its offset, row
/// pitch and slice pitch, in bytes).
#[cfg(windows)]
const DECODE_SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=9, b0), SRV(t0), UAV(u0)"
cbuffer Part : register(b0) {
    uint width;
    uint height;
    uint depth;
    uint src_offset;
    uint src_pitch;
    uint src_slice;
    uint dst_offset;
    uint dst_pitch;
    uint dst_slice;
};
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= width || id.y >= height || id.z >= depth) return;
    uint v = src.Load(src_offset + id.z * src_slice + id.y * src_pitch + id.x * 4);
    // Nine-bit mantissas sharing a five-bit exponent biased by 15.
    float scale = exp2(float(v >> 27) - 24.0);
    float3 c = float3(v & 0x1ff, (v >> 9) & 0x1ff, (v >> 18) & 0x1ff) * scale;
    uint2 h = uint2(f32tof16(c.r) | f32tof16(c.g) << 16, f32tof16(c.b) | f32tof16(1.0) << 16);
    dst.Store2(dst_offset + id.z * dst_slice + id.y * dst_pitch + id.x * 8, h);
}
"#;

/// The decoding pass: its root signature and pipeline, made once.
#[cfg(windows)]
struct Decode {
    root: windows::Win32::Graphics::Direct3D12::ID3D12RootSignature,
    pso: windows::Win32::Graphics::Direct3D12::ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded; the pass is used under its lock.
#[cfg(windows)]
unsafe impl Send for Decode {}

#[cfg(windows)]
static DECODE: std::sync::Mutex<Option<Option<Decode>>> = std::sync::Mutex::new(None);

#[cfg(windows)]
fn make_decode(host: &crate::host::Host) -> windows::core::Result<Decode> {
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::core::s;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            DECODE_SOURCE.as_ptr().cast(),
            DECODE_SOURCE.len(),
            s!("rgb9e5"),
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
            tracing::warn!("R9G9B9E5 decoding pass: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is in the shader.
    unsafe {
        let bytes =
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes)?;
        let pso = host
            .device
            .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            })?;
        Ok(Decode { root, pso })
    }
}

/// A device-local buffer of `bytes`, in the common state, with unordered
/// access if `unordered`.
#[cfg(windows)]
fn device_buffer(
    host: &crate::host::Host,
    bytes: u64,
    unordered: bool,
) -> Option<windows::Win32::Graphics::Direct3D12::ID3D12Resource> {
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
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
        Flags: if unordered {
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
        } else {
            D3D12_RESOURCE_FLAG_NONE
        },
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation; the out pointer is ours.
    unsafe {
        host.device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut buffer,
            )
            .ok()?;
    }
    buffer
}

/// What [`decode_now`] keeps for a pair of host copies: the buffer the
/// written texture's words go to and the one the decoded texels come from,
/// and each level's and layer's layout in them (the written texture's
/// first).
#[cfg(windows)]
struct Decoded {
    words: windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    texels: windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    from: std::sync::Weak<HostTexture>,
    to: std::sync::Weak<HostTexture>,
    parts: Vec<(
        windows::Win32::Graphics::Direct3D12::D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
        windows::Win32::Graphics::Direct3D12::D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    )>,
}

// SAFETY: D3D12 objects are free-threaded; used under its map's lock.
#[cfg(windows)]
unsafe impl Send for Decoded {}

#[cfg(windows)]
static DECODED: std::sync::Mutex<Option<std::collections::HashMap<usize, Decoded>>> =
    std::sync::Mutex::new(None);

/// Whether a texture of title format `title` whose host copy is `to`
/// reads R9G9B9E5 through its RGBA16F stand-in, and `from` holds
/// four-byte texels: what [`decode_now`] turns one into the other.
#[cfg(windows)]
fn decodes(title: u32, from: &HostTexture, to: &HostTexture) -> bool {
    const R9G9B9E5: u32 = 67;
    const R16G16B16A16_FLOAT: u32 = 10;
    // SAFETY: live textures.
    let (f, t) = unsafe { (from.texture.GetDesc(), to.texture.GetDesc()) };
    title == R9G9B9E5
        && t.Format.0 as u32 == R16G16B16A16_FLOAT
        && element(f.Format.0 as u32) == Some((4, 1))
        && t.SampleDesc.Count == 1
}

/// Decodes the source words at the point where this list reads the view.
/// Earlier submissions determine whether the first read needs a decode.
#[cfg(windows)]
fn decode_now(
    list: &crate::host::HostList,
    host: &crate::host::Host,
    placed: This,
    source: This,
    from: &Arc<HostTexture>,
    to: &Arc<HostTexture>,
) -> bool {
    use std::sync::atomic::Ordering;
    use windows::Win32::Graphics::Direct3D12::*;
    if list
        .decode_uses
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .current(from, to)
    {
        return true;
    }
    let mut decoded = DECODED.lock().unwrap_or_else(|e| e.into_inner());
    let decoded = decoded.get_or_insert_with(std::collections::HashMap::new);
    let fits = |d: &Decoded| {
        std::ptr::eq(d.from.as_ptr(), Arc::as_ptr(from))
            && std::ptr::eq(d.to.as_ptr(), Arc::as_ptr(to))
    };
    if !decoded.get(&(placed as usize)).is_some_and(fits) {
        decoded.retain(|_, d| d.to.strong_count() > 0 && d.from.strong_count() > 0);
        let Some(made) = decoding(host, from, to) else {
            return false;
        };
        decoded.insert(placed as usize, made);
    }
    let Some(d) = decoded.get_mut(&(placed as usize)) else {
        return false;
    };
    let mut pass = DECODE.lock().unwrap_or_else(|e| e.into_inner());
    let Some(pass) = pass
        .get_or_insert_with(|| match make_decode(host) {
            Ok(p) => Some(p),
            Err(e) => {
                tracing::warn!(error = %e, "no R9G9B9E5 decoding pass: such aliases stay black");
                None
            }
        })
        .as_mut()
    else {
        return false;
    };
    const COPY_DEST: u32 = 0x400;
    const COPY_SOURCE: u32 = 0x800;
    let before = alias_source(list, source, COPY_SOURCE);
    crate::list::ensure_state(list, placed, COPY_DEST);
    let part = |texture: &ID3D12Resource, index: usize| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the location never releases.
        pResource: unsafe { std::mem::transmute_copy(texture) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: index as u32,
        },
    };
    let rows = |buffer: &ID3D12Resource, row: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT| {
        D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *row,
            },
        }
    };
    let state = |s: D3D12_RESOURCE_STATES| s;
    let (common, copy_dest, copy_source, read, unordered) = (
        state(D3D12_RESOURCE_STATE_COMMON),
        state(D3D12_RESOURCE_STATE_COPY_DEST),
        state(D3D12_RESOURCE_STATE_COPY_SOURCE),
        state(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
        state(D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
    );
    let t = crate::host::transition;
    let l = &list.list;
    let predicate = crate::texture::Predicate::take();
    if let Some(p) = &predicate {
        p.bind(list);
    }
    #[cfg(feature = "diagnostics")]
    list.note_op(|| format!("decode R9G9B9E5 from {source:?} into {placed:?}"));
    // SAFETY: our open list, pass and buffers (kept with the pair); live
    // textures the list holds through their states; every copy and store
    // inside the layouts the buffers were sized for.
    unsafe {
        l.ResourceBarrier(&[t(&d.words, common, copy_dest)]);
        for (index, (of_words, _)) in d.parts.iter().enumerate() {
            l.CopyTextureRegion(
                &rows(&d.words, of_words),
                0,
                0,
                0,
                &part(&from.texture, index),
                None,
            );
        }
        l.ResourceBarrier(&[
            t(&d.words, copy_dest, read),
            t(&d.texels, common, unordered),
        ]);
        l.SetComputeRootSignature(&pass.root);
        l.SetPipelineState(&pass.pso);
        l.SetComputeRootShaderResourceView(1, d.words.GetGPUVirtualAddress());
        l.SetComputeRootUnorderedAccessView(2, d.texels.GetGPUVirtualAddress());
        for (of_words, of_texels) in &d.parts {
            let (w, x) = (of_words.Footprint, of_texels.Footprint);
            let constants = [
                w.Width,
                w.Height,
                w.Depth,
                of_words.Offset as u32,
                w.RowPitch,
                w.RowPitch * w.Height,
                of_texels.Offset as u32,
                x.RowPitch,
                x.RowPitch * x.Height,
            ];
            l.SetComputeRoot32BitConstants(0, 9, constants.as_ptr().cast(), 0);
            l.Dispatch(w.Width.div_ceil(8), w.Height.div_ceil(8), w.Depth);
        }
        l.ResourceBarrier(&[
            t(&d.texels, unordered, copy_source),
            t(&d.words, read, common),
        ]);
        for (index, (_, of_texels)) in d.parts.iter().enumerate() {
            l.CopyTextureRegion(
                &part(&to.texture, index),
                0,
                0,
                0,
                &rows(&d.texels, of_texels),
                None,
            );
        }
        l.ResourceBarrier(&[t(&d.texels, copy_source, common)]);
    }
    if predicate.is_some() {
        crate::texture::Predicate::unbind(list);
    }
    list.keep(&d.words);
    list.keep(&d.texels);
    // The same command may read the source through an SRV or write its UAV.
    if let Some(before) = before {
        crate::list::ensure_state(list, source, before);
    }
    list.decode_pipeline.store(true, Ordering::Relaxed);
    #[cfg(feature = "diagnostics")]
    let count = crate::frameprobe::decoding(list, placed);
    decode::recorded(
        list,
        from,
        to,
        predicate,
        #[cfg(feature = "diagnostics")]
        count,
    );
    true
}

/// Compatibility entry point for callers of Host::submit.
/// Each reader's list already contains its decode commands.
#[cfg(windows)]
pub(crate) fn run_decodes(_: &crate::host::Host) {}

/// The buffers and layouts [`decode_now`] needs for `from` and `to`, if
/// they have the same shape (levels, layers, sizes).
#[cfg(windows)]
fn decoding(
    host: &crate::host::Host,
    from: &Arc<HostTexture>,
    to: &Arc<HostTexture>,
) -> Option<Decoded> {
    use windows::Win32::Graphics::Direct3D12::*;
    // SAFETY: live textures.
    let (of_from, of_to) = unsafe { (from.texture.GetDesc(), to.texture.GetDesc()) };
    let parts = |d: &D3D12_RESOURCE_DESC| {
        let layers = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
            1
        } else {
            u32::from(d.DepthOrArraySize)
        };
        u32::from(d.MipLevels) * layers
    };
    let n = parts(&of_from);
    if n == 0 || n > ALIAS_PARTS || n != parts(&of_to) {
        return None;
    }
    let mut rows =
        [(); 2].map(|()| vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n as usize]);
    let mut bytes = [0u64; 2];
    // SAFETY: plain queries; the out pointers hold `n` layouts each.
    unsafe {
        for (i, d) in [of_from, of_to].iter().enumerate() {
            host.device.GetCopyableFootprints(
                d,
                0,
                n,
                0,
                Some(rows[i].as_mut_ptr()),
                None,
                None,
                Some(&mut bytes[i]),
            );
        }
    }
    let alike = rows[0].iter().zip(&rows[1]).all(|(a, b)| {
        let (fa, fb) = (a.Footprint, b.Footprint);
        (fa.Width, fa.Height, fa.Depth) == (fb.Width, fb.Height, fb.Depth)
    });
    if bytes.contains(&0) || bytes.iter().any(|&b| b > u64::from(u32::MAX)) || !alike {
        return None;
    }
    let words = device_buffer(host, bytes[0], false)?;
    let texels = device_buffer(host, bytes[1], true)?;
    crate::host::name(&texels, || {
        format!("R9G9B9E5 decoded {}x{}", of_to.Width, of_to.Height)
    });
    tracing::info!(
        width = of_from.Width,
        height = of_from.Height,
        parts = n,
        written_as = of_from.Format.0,
        "a texture read as R9G9B9E5 over a GPU-written one: decoded from it on the GPU into its RGBA16F stand-in"
    );
    let [of_words, of_texels] = rows;
    Some(Decoded {
        words,
        texels,
        from: Arc::downgrade(from),
        to: Arc::downgrade(to),
        parts: of_words.into_iter().zip(of_texels).collect(),
    })
}

/// A title texture held alive by an owned reference ([`alias::written_beside`]);
/// the reference is released when this is dropped.
#[cfg(windows)]
pub(crate) struct Held(This);

#[cfg(windows)]
impl Held {
    /// Pins `this` if it still has a live reference, returning `None` if it is
    /// already being released. Must be called under the registry lock that
    /// prevents `this` from being freed while its reference count is checked.
    pub(super) fn pin(this: This) -> Option<Held> {
        object::add_ref_alive(this).then_some(Held(this))
    }

    /// The texture this handle keeps alive.
    pub(crate) fn get(&self) -> This {
        self.0
    }
}

#[cfg(windows)]
impl Drop for Held {
    fn drop(&mut self) {
        object::release_boxed::<Resource>(self.0);
    }
}

/// Transitions the source for copying and saves its previous state in case
/// the command already bound it as a render target or UAV.
#[cfg(windows)]
fn alias_source(list: &crate::host::HostList, source: This, state: u32) -> Option<u32> {
    use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES;
    list.transition(source, state).map(|(texture, before)| {
        let state = crate::host::legal_state(texture, state);
        let barrier = crate::host::transition(
            texture,
            D3D12_RESOURCE_STATES(before as i32),
            D3D12_RESOURCE_STATES(state as i32),
        );
        // SAFETY: an open list and a live texture.
        unsafe { list.list.ResourceBarrier(&[barrier]) };
        before
    })
}

/// `placed`, a title texture the GPU does not write, read by a shader in
/// `list`. If it lies over a GPU-written texture of the same shape whose
/// texels are as wide, in another format family (`alias::written_beside`:
/// light packed as R9G9B9E5 into R32_UINT by compute programs and read
/// through R9G9B9E5 views), that texture's texels are copied into its host
/// copy, through a buffer (the host copies between two formats in no other
/// way): the first time the list reads it, and again whenever the list used
/// the written texture otherwise since. Where the host keeps R9G9B9E5 as
/// RGBA16F, [`decode_now`] converts the texels in this list. Title memory does not have them: a
/// copy filled from it is black. Leaves `placed` in the copy destination
/// state and keeps its host copy from being filled from title memory again
/// (`texture::gpu_write`). True if `placed` is such a texture.
///
/// Ported from fxceless's VIctory fork (commit 4e17478: `alias_of` in
/// `resource.rs`, `texture::alias`), reworked onto XTP's alias registry.
#[cfg(windows)]
pub(crate) fn alias_copy(list: &crate::host::HostList, placed: This) -> bool {
    use windows::Win32::Graphics::Direct3D12::*;
    if placed.is_null() {
        return false;
    }
    let r = resource(placed);
    if r.desc.dimension == BUFFER || r.shared {
        return false;
    }
    // Every shader read of every texture comes here: the registry is
    // looked at again only once it changed.
    let changes = alias::changes();
    if r.beside.load(std::sync::atomic::Ordering::Relaxed) == changes {
        return false;
    }
    let Some((source, from)) = alias::written_beside(r.data() as u64, &r.desc) else {
        r.beside
            .store(changes, std::sync::atomic::Ordering::Relaxed);
        return false;
    };
    let (Some(to), Some(host)) = (ready_host(placed).cloned(), crate::host::host()) else {
        return false;
    };
    if !from.ready() {
        return false;
    }
    // The host has R9G9B9E5 as RGBA16F (`format`, the RTX 4080): texels
    // of another width, decoded rather than copied.
    if decodes(r.desc.format, &from, &to) {
        let decoded = decode_now(list, host, placed, source.get(), &from, &to);
        if decoded {
            crate::texture::gpu_write(list, placed, true);
        }
        return decoded;
    }
    let Some(through) = through(host, placed as usize, &from, &to) else {
        return false;
    };
    const COPY_DEST: u32 = 0x400;
    const COPY_SOURCE: u32 = 0x800;
    const TARGET_OR_UNORDERED: u32 = 0x4 | 0x8;
    let source_ptr = source.get();
    // Resources are 8-byte aligned: a key of the list's own for the copy.
    let first = list.first_use(placed as usize | 1);
    // The source's first use in the list records nothing (its prelude
    // brings it to the state); a use in another way since the last copy
    // shows as a change of state.
    let before = alias_source(list, source_ptr, COPY_SOURCE);
    crate::texture::gpu_write(list, placed, true);
    if !first && before.is_none() {
        return true;
    }
    crate::list::ensure_state(list, placed, COPY_DEST);
    #[cfg(feature = "diagnostics")]
    list.note_op(|| format!("alias copy of {source_ptr:?} into {placed:?}"));
    let part = |texture: &ID3D12Resource, index: usize| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the location never releases.
        pResource: unsafe { std::mem::transmute_copy(texture) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: index as u32,
        },
    };
    let rows = |row: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(&through.buffer) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: *row,
        },
    };
    let (common, read, written) = (
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    // SAFETY: an open list and live textures (the list holds both host
    // copies through their states); the buffer is as big as the
    // footprints and rests in the common state between copies.
    unsafe {
        let b = &through.buffer;
        list.list
            .ResourceBarrier(&[crate::host::transition(b, common, written)]);
        for (index, (of_written, _)) in through.parts.iter().enumerate() {
            let from = part(&from.texture, index);
            list.list
                .CopyTextureRegion(&rows(of_written), 0, 0, 0, &from, None);
        }
        list.list
            .ResourceBarrier(&[crate::host::transition(b, written, read)]);
        for (index, (_, of_read)) in through.parts.iter().enumerate() {
            let to = part(&to.texture, index);
            list.list
                .CopyTextureRegion(&to, 0, 0, 0, &rows(of_read), None);
        }
        list.list
            .ResourceBarrier(&[crate::host::transition(b, read, common)]);
    }
    list.keep(&through.buffer);
    // A draw that also writes the source (a view of it handled before this
    // one) needs it in that state again; the next read of the alias then
    // copies once more, as it must after a write.
    if let Some(before) = before
        && before & TARGET_OR_UNORDERED != 0
    {
        crate::list::ensure_state(list, source_ptr, before);
    }
    true
}

/// The colour twin of a depth texture the title also writes through
/// unordered access. The console allows both on one texture; the host
/// gives a depth texture no unordered access (`host_flags`), so shaders
/// write a colour texture of the same size and format family instead, and
/// one side is copied into the other when it is used after the other was
/// written (`HostList::twin_use`).
/// shortcut: whole textures are copied, not the mips and slices written.
/// A depth texture with a stencil plane gets a twin for each plane: BTP 6's
/// tiled lighting writes each pixel's material class into the depth plane
/// and masks into the stencil plane, then draws each class with the depth
/// test "equal". The stencil plane's twin is copied through a buffer; the
/// depth plane's is drawn into the plane by a pass of ours
/// (`HostList::pass_plane`). Unordered access views of a multisampled
/// texture stay null.
///
/// Ported from fxceless's VIctory fork (commit 578d459).
#[cfg(windows)]
pub(crate) struct Twin {
    /// The colour texture and the state it is in, as a host copy's.
    pub(crate) colour: Arc<HostTexture>,
    /// Which side lacks what the other was written last, after the lists
    /// submitted so far: bit 0 the depth texture, bit 1 the twin.
    pub(crate) behind: std::sync::atomic::AtomicU32,
    /// For a depth texture with a stencil plane, `colour` is the twin of
    /// that plane (8 bits a texel) and this the buffer the two are copied
    /// through: no colour format shares a family with a depth-stencil
    /// format, but the host copies between a buffer and one plane.
    pub(crate) through: Option<windows::Win32::Graphics::Direct3D12::ID3D12Resource>,
    /// For a depth texture with a stencil plane: the twin of its depth
    /// plane, 16 bits a texel for the console's 16-bit depth-stencil
    /// formats and a 32-bit float otherwise. No host colour format holds
    /// the bits of a 24-bit depth plane, so the two are not copied: a pass
    /// draws the twin into the plane.
    pub(crate) plane: Option<Arc<HostTexture>>,
}

// SAFETY: D3D12 objects are free-threaded.
#[cfg(windows)]
unsafe impl Send for Twin {}
// SAFETY: as above.
#[cfg(windows)]
unsafe impl Sync for Twin {}

/// A host texture of ours (not a title resource's copy) in `state`:
/// resident, unpooled, counted as alive while it lives.
#[cfg(windows)]
fn own_texture(
    texture: windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    state: u32,
) -> HostTexture {
    let origin = TextureOrigin::External;
    format::note_texture(true, 0, 0, false, origin);
    HostTexture {
        texture,
        state: state.into(),
        #[cfg(test)]
        decode_count: Default::default(),
        decode_tracked: false,
        content: decode::revision().into(),
        bytes: 0,
        extra: 0,
        origin,
        residency: Arc::new(residency::Residency::resident()),
        slot: None,
    }
}

/// Whether depth twins are made (`XTP_DEPTH_TWINS=0`: none, such views
/// stay null, as before; `stencil`: only for formats with a stencil plane;
/// `nodepth`: none for the depth planes of such formats).
#[cfg(windows)]
fn twins_asked() -> &'static str {
    static ASKED: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    ASKED.get_or_init(|| std::env::var("XTP_DEPTH_TWINS").unwrap_or_default())
}

/// The colour twin for a texture described by `desc` whose host copy is
/// `depth`, if the title asks for depth and unordered access both and the
/// host can copy between the two: the same size, mips and slices in the
/// depth format's typeless family (R16 for 16-bit depth, R32 for 32-bit),
/// or R8 for the stencil plane of a format that has one, with unordered
/// access only. A format with a stencil plane gets a twin for its depth
/// plane too, R16 for the console's 16-bit depth-stencil formats and R32
/// otherwise.
#[cfg(windows)]
fn twin_texture(
    desc: &ResourceDesc,
    depth: &windows::Win32::Graphics::Direct3D12::ID3D12Resource,
) -> Option<Arc<Twin>> {
    use windows::Win32::Graphics::Direct3D12::*;
    const DEPTH_AND_UNORDERED: u32 = 6;
    const UNORDERED: u32 = 4;
    const UNORDERED_ACCESS: u32 = 0x8;
    // R32 and R16 typeless: the families of the depth formats of one plane.
    const ONE_PLANE: [u32; 2] = [39, 53];
    // R32G8X24 and R24G8 typeless: depth with a stencil plane, whose twin
    // is R8 typeless.
    const WITH_STENCIL: [u32; 2] = [19, 44];
    const R8_TYPELESS: u32 = 60;
    if desc.dimension == BUFFER
        || desc.flags & DEPTH_AND_UNORDERED != DEPTH_AND_UNORDERED
        || twins_asked() == "0"
    {
        return None;
    }
    let typeless = family(pc_format(desc.format));
    let stencil = WITH_STENCIL.contains(&typeless);
    if !stencil && twins_asked() == "stencil" {
        return None;
    }
    if !(stencil || ONE_PLANE.contains(&typeless)) || desc.sample_count > 1 {
        return None;
    }
    let host = crate::host::host()?;
    let mut d = *desc;
    d.layout = 0;
    d.alignment = 0;
    d.flags = UNORDERED;
    d.format = if stencil { R8_TYPELESS } else { typeless };
    let make = |d: &ResourceDesc, what: &str| -> Option<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let mut texture: Option<ID3D12Resource> = None;
        // SAFETY: the description has the host layout, as in
        // `host_texture`; the out pointer is ours.
        let made = unsafe {
            host.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                std::ptr::from_ref(d).cast(),
                D3D12_RESOURCE_STATES(UNORDERED_ACCESS as i32),
                None,
                &mut texture,
            )
        };
        if let Err(e) = made {
            tracing::warn!(error = %e, desc = ?d, what, "no twin for a depth texture");
        }
        let texture = texture?;
        crate::host::name(&texture, || {
            format!("{what} {}x{} f{}", d.width, d.height, d.format)
        });
        Some(texture)
    };
    let texture = make(&d, "depth twin")?;
    // The buffer a stencil plane and its twin are copied through: as big
    // as the plane's mips and slices laid out for a copy (they follow the
    // depth plane's in the texture's subresources).
    let through = if stencil {
        // SAFETY: a live texture.
        let of = unsafe { depth.GetDesc() };
        let planes = u32::from(of.MipLevels) * u32::from(of.DepthOrArraySize);
        let mut bytes = 0;
        // SAFETY: a plain query; the out pointer is ours.
        unsafe {
            host.device.GetCopyableFootprints(
                &of,
                planes,
                planes,
                0,
                None,
                None,
                None,
                Some(&mut bytes),
            );
        }
        Some(device_buffer(host, bytes, false)?)
    } else {
        None
    };
    // The twin of the depth plane of a format with a stencil plane: R16
    // typeless (53) for the console's 16-bit depth-stencil formats (118 to
    // 120), whose shaders write 16-bit steps, R32 typeless (39) otherwise.
    let plane = if stencil && twins_asked() != "nodepth" {
        d.format = if (118..=120).contains(&desc.format) {
            53
        } else {
            39
        };
        make(&d, "depth plane twin").map(|t| Arc::new(own_texture(t, UNORDERED_ACCESS)))
    } else {
        None
    };
    tracing::debug!(desc = ?desc, stencil, plane = plane.is_some(), "a depth texture's colour twin");
    Some(Arc::new(Twin {
        colour: Arc::new(own_texture(texture, UNORDERED_ACCESS)),
        behind: std::sync::atomic::AtomicU32::new(0),
        through,
        plane,
    }))
}

/// The colour twin of title texture `this`, if it has one.
#[cfg(windows)]
pub(crate) fn twin(this: This) -> Option<Arc<Twin>> {
    if this.is_null() {
        return None;
    }
    let r = resource(this);
    if r.desc.dimension == BUFFER {
        return None;
    }
    r.twin.get().cloned().flatten()
}

/// The colour twin of title texture `this`, made now if it has none yet
/// and is a depth texture the title also asks unordered access for.
#[cfg(windows)]
fn twin_made(this: This) -> Option<Arc<Twin>> {
    if this.is_null() {
        return None;
    }
    let r = resource(this);
    if r.desc.dimension == BUFFER || r.shared {
        return None;
    }
    r.twin
        .get_or_init(|| {
            let host = r.host.as_ref()?;
            twin_texture(&r.desc, &host.texture)
        })
        .clone()
}

/// The texture and view for the title's unordered access view `desc` of
/// `this`, a depth texture with a colour twin: the twin's (or, for a view
/// of the depth plane of a texture with a stencil plane, that plane's
/// twin's), in the colour format of the twin's family for a depth or
/// typeless format (R16_UNORM for 16-bit depth, R32_FLOAT for 32-bit; the
/// bits are the same). Without a description: mip 0 of every slice. The
/// twin of a stencil plane (R8) stands for the views of that plane only,
/// which become R8_UINT views of its one plane. `None` for a texture
/// without a twin (`descriptor/host.rs` makes the view as before).
#[cfg(windows)]
pub(crate) fn twin_view(
    this: This,
    desc: Option<windows::Win32::Graphics::Direct3D12::D3D12_UNORDERED_ACCESS_VIEW_DESC>,
) -> Option<(
    &'static windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    windows::Win32::Graphics::Direct3D12::D3D12_UNORDERED_ACCESS_VIEW_DESC,
)> {
    use windows::Win32::Graphics::Direct3D12::*;
    let twin = twin_made(this)?;
    let view = |texture: &ID3D12Resource| {
        // SAFETY: a live resource.
        let d = unsafe { texture.GetDesc() };
        let mut desc = desc.unwrap_or_else(|| {
            let mut all = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
                ..Default::default()
            };
            if d.DepthOrArraySize > 1 {
                all.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
                all.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                    ArraySize: u32::from(d.DepthOrArraySize),
                    ..Default::default()
                };
            }
            all
        });
        // A stencil plane's twin (R8 typeless, 60): the title's view is of
        // the stencil plane if it names plane 1 or a stencil view format.
        if d.Format.0 == 60 {
            let mut plane = 0;
            match desc.ViewDimension {
                D3D12_UAV_DIMENSION_TEXTURE2D => {
                    // SAFETY: the member the dimension names.
                    let mut v = unsafe { desc.Anonymous.Texture2D };
                    plane = std::mem::take(&mut v.PlaneSlice);
                    desc.Anonymous.Texture2D = v;
                }
                D3D12_UAV_DIMENSION_TEXTURE2DARRAY => {
                    // SAFETY: the member the dimension names.
                    let mut v = unsafe { desc.Anonymous.Texture2DArray };
                    plane = std::mem::take(&mut v.PlaneSlice);
                    desc.Anonymous.Texture2DArray = v;
                }
                _ => {}
            }
            if plane != 1 && !matches!(desc.Format.0, 22 | 47) {
                return None;
            }
            desc.Format.0 = 62;
            return Some(desc);
        }
        // None, typeless and depth formats (R16: 0, 53, 55; R32: 0, 39,
        // 40), and those of a depth-stencil format's depth plane (19 to 21,
        // 44 to 46), for the twin of such a plane.
        desc.Format.0 = match (desc.Format.0, d.Format.0) {
            (0 | 53 | 55 | 19..=21 | 44..=46, 53) => 56,
            (0 | 39 | 40 | 19..=21 | 44..=46, 39) => 41,
            (other, _) => other,
        };
        Some(desc)
    };
    // SAFETY: the twin lives as long as the resource's, which the title
    // holds while it makes views of it.
    let colour: &'static ID3D12Resource = unsafe { &*std::ptr::from_ref(&twin.colour.texture) };
    if let Some(v) = view(colour) {
        return Some((colour, v));
    }
    let plane = twin.plane.as_ref()?;
    // SAFETY: as above.
    let plane: &'static ID3D12Resource = unsafe { &*std::ptr::from_ref(&plane.texture) };
    view(plane).map(|v| (plane, v))
}

pub(crate) extern "system" fn create_placed(
    _: This,
    heap: This,
    offset: u64,
    desc: *const ResourceDesc,
    state: u32,
    clear: *const c_void,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    if heap.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: the title passes one of our heaps.
    let h = unsafe { &*heap.cast::<Heap>() };
    let fits = footprint(&desc).is_some_and(|info| {
        offset.is_multiple_of(info.alignment)
            && offset
                .checked_add(info.size)
                .is_some_and(|end| end <= h.desc.size)
    });
    if !fits {
        tracing::warn!(
            ?desc,
            offset,
            heap = h.desc.size,
            "placed resource does not fit"
        );
        return E_INVALIDARG;
    }
    object::add_ref(heap);
    let backing = Backing::Placed { heap, offset };
    hand_out_resource(
        desc,
        h.desc.properties,
        h.desc.flags,
        backing,
        state,
        clear,
        iid,
        out,
        "placed in a heap",
    )
}

extern "system" fn resource_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        RESOURCE,
        iid,
        object::is(iid, &RESOURCE_IIDS).then_some(this),
        out,
    )
}
pub(crate) extern "system" fn resource_release(this: This) -> u32 {
    object::release_boxed_with::<Resource>(this, || crate::descriptor::forget(this as usize))
}
/// Memory is always mapped (subresources are not told apart yet); a null
/// `out` only asks for residency.
extern "system" fn map(this: This, _sub: u32, _read: *const c_void, out: *mut *mut u8) -> HRESULT {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a pointer.
        unsafe { out.write(resource(this).data()) };
    }
    S_OK
}
extern "system" fn unmap(_: This, _sub: u32, _written: *const c_void) {}
extern "system" fn resource_desc(this: This, out: *mut ResourceDesc) {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a description.
        unsafe { out.write(resource(this).desc) };
    }
}
/// Device method `GetCustomHeapProperties`, returning its struct through a
/// hidden first argument: what a standard heap type is as a custom heap.
/// Memory is unified and cache-coherent (as the architecture query says),
/// so every type is in pool L0, default heaps are not CPU-visible and the
/// others are write-back.
pub(crate) extern "system" fn custom_heap_properties(
    out: *mut HeapProperties,
    _: This,
    nodes: u32,
    kind: u32,
) -> *mut HeapProperties {
    const CUSTOM: u32 = 4;
    const NOT_AVAILABLE: u32 = 1;
    const WRITE_BACK: u32 = 3;
    const L0: u32 = 1;
    let nodes = nodes.max(1);
    let cpu_pages = match kind {
        1 => NOT_AVAILABLE,
        2 | 3 => WRITE_BACK,
        _ => 0,
    };
    let properties = HeapProperties {
        kind: CUSTOM,
        cpu_pages,
        pool: if cpu_pages == 0 { 0 } else { L0 },
        creation_nodes: nodes,
        visible_nodes: nodes,
    };
    // SAFETY: the title passes room for the result (null is tolerated).
    if let Some(out) = unsafe { out.as_mut() } {
        *out = properties;
    }
    out
}

extern "system" fn gpu_address(this: This) -> u64 {
    let r = resource(this);
    let at = r.data() as u64;
    // The console's GPU addresses fit in 40 bits; its ray-tracing builder
    // refuses others (diagnostics: which memory breaks that).
    #[cfg(feature = "diagnostics")]
    if at >> 40 != 0 {
        static ONCE: [std::sync::Once; 2] = [std::sync::Once::new(), std::sync::Once::new()];
        let backing = match r.backing {
            Backing::Committed(ref m) => format!("committed, attributes {:?}", m.attributes),
            Backing::Placed { offset, .. } => format!("placed in a heap at {offset:#x}"),
            Backing::Address(_) => "at a title address".into(),
        };
        ONCE[usize::from(r.desc.dimension == BUFFER)].call_once(|| {
            tracing::warn!(address = format_args!("{at:#x}"), backing, desc = ?r.desc, "a GPU address above 40 bits")
        });
    }
    at
}
extern "system" fn heap_properties(
    this: This,
    properties: *mut HeapProperties,
    flags: *mut u32,
) -> HRESULT {
    let r = resource(this);
    // SAFETY: either pointer may be null; otherwise the title passes room.
    unsafe {
        if let Some(p) = properties.as_mut() {
            *p = r.heap;
        }
        if let Some(f) = flags.as_mut() {
            *f = r.heap_flags;
        }
    }
    S_OK
}

/// IUnknown 3, object 4, device child 1, heap 1.
static HEAP_FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<HEAP, 24>());
    f[0] = heap_query as *const ();
    f[2] = heap_release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = heap_desc as *const (); // 0x40
    Table(f)
};

/// IUnknown 3, object 4, device child 1, resource 10 (7 as on PC, then 3
/// console ones).
static RESOURCE_FUNCTIONS: Table<32> = {
    let mut f = object::d3d12_object(object::missing_table::<RESOURCE, 32>());
    f[0] = resource_query as *const ();
    f[2] = resource_release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = map as *const (); // 0x40
    f[9] = unmap as *const (); // 0x48
    f[10] = resource_desc as *const (); // 0x50
    f[11] = gpu_address as *const (); // 0x58
    f[14] = heap_properties as *const (); // 0x70
    Table(f)
};

/// What tests of other modules need from title textures.
#[cfg(all(test, windows))]
pub(crate) mod tests_support {
    use super::*;
    use windows::Win32::Graphics::Direct3D12::*;

    /// Runs `f` while the placed view of `desc` at `address` looks as if
    /// another thread were still making its first copy: [`texture_at`]
    /// finds none there. The entry as it was comes back after.
    pub(crate) fn while_placed_building(address: u64, desc: &[u32; 14], f: impl FnOnce()) {
        let key = (address, canonical_resource_desc(*desc));
        let saved = placed_textures().remove(&key);
        placed_textures().insert(
            key,
            PlacedView {
                resource: None,
                changes: alias::changes(),
                len: 1,
                building: u64::MAX,
            },
        );
        f();
        let mut map = placed_textures();
        map.remove(&key);
        if let Some(saved) = saved {
            map.insert(key, saved);
        }
    }

    /// A `width` x `height` depth texture in `format` that the title also
    /// asks unordered access for (it gets a colour twin).
    pub(crate) fn depth_unordered(width: u64, height: u32, format: u32) -> ResourceDesc {
        ResourceDesc {
            flags: 6,
            ..target(width, height, format)
        }
    }

    /// A `width` x `height` render target in `format`.
    pub(crate) fn target(width: u64, height: u32, format: u32) -> ResourceDesc {
        ResourceDesc {
            dimension: 3,
            alignment: 0,
            width,
            height,
            depth_or_array_size: 1,
            mip_levels: 1,
            format,
            sample_count: 1,
            sample_quality: 0,
            layout: 0,
            flags: 1,
        }
    }

    /// A `width` x `height` texture in `format` that shaders only sample
    /// (the title fills it), in the layout library's pick.
    pub(crate) fn sampled(width: u64, height: u32, format: u32) -> ResourceDesc {
        ResourceDesc {
            flags: 0,
            ..target(width, height, format)
        }
    }

    /// [`sampled`] with linear layout (`layout = 1`), e.g. for movie planes.
    pub(crate) fn linear_sampled(width: u64, height: u32, format: u32) -> ResourceDesc {
        ResourceDesc {
            layout: 1,
            ..sampled(width, height, format)
        }
    }

    /// A title resource described by `desc` over memory at `address`.
    pub(crate) fn at_address(address: u64, desc: &ResourceDesc) -> This {
        let mut out = std::ptr::null();
        let made = create_at_address(
            std::ptr::null(),
            address,
            desc,
            0,
            std::ptr::null(),
            &RESOURCE_IIDS[0],
            &mut out,
        );
        assert_eq!(made, S_OK);
        out
    }

    /// A committed title resource described by `desc`.
    pub(crate) fn committed(desc: &ResourceDesc) -> This {
        let null = std::ptr::null();
        let heap = HeapProperties {
            kind: 1,
            cpu_pages: 0,
            pool: 0,
            creation_nodes: 0,
            visible_nodes: 0,
        };
        let mut out = null;
        let iid = &RESOURCE_IIDS[0];
        assert_eq!(
            create_committed(null, &heap, 0, desc, 0, null.cast(), iid, &mut out),
            S_OK
        );
        out
    }

    /// The bytes of subresource 0 of a 2D colour texture with rows at most
    /// 256 bytes wide. Rows are returned 256 bytes apart, in its own format.
    pub(crate) fn read_back(
        host: &crate::host::Host,
        texture: &ID3D12Resource,
        state: u32,
    ) -> Vec<u8> {
        // SAFETY: a live resource.
        let d = unsafe { texture.GetDesc() };
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        // SAFETY: room for the one footprint requested.
        unsafe {
            host.device
                .GetCopyableFootprints(&d, 0, 1, 0, Some(&mut footprint), None, None, None);
        }
        assert_eq!(footprint.Footprint.RowPitch, 256);
        let size = 256 * u64::from(d.Height);
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
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
        let mut readback: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a readback buffer.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .unwrap();
        }
        let readback = readback.unwrap();
        let list = host.new_list().unwrap();
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(&readback) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: footprint,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        let state = D3D12_RESOURCE_STATES(state as i32);
        let copy = D3D12_RESOURCE_STATE_COPY_SOURCE;
        // SAFETY: an open list; both locations name live resources.
        unsafe {
            if state != copy {
                list.list
                    .ResourceBarrier(&[crate::host::transition(texture, state, copy)]);
            }
            list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
            if state != copy {
                list.list
                    .ResourceBarrier(&[crate::host::transition(texture, copy, state)]);
            }
        }
        list.close();
        host.execute(&[&list]);
        host.drain();
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the copy ran (drained); the buffer is `size` bytes.
        unsafe {
            readback.Map(0, None, Some(&mut mapped)).unwrap();
            let bytes = std::slice::from_raw_parts(mapped.cast::<u8>(), size as usize).to_vec();
            readback.Unmap(0, None);
            bytes
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;

    type Release = extern "system" fn(This) -> u32;
    type Map = extern "system" fn(This, u32, *const c_void, *mut *mut u8) -> HRESULT;
    type Address = extern "system" fn(This) -> u64;

    const PROPERTIES: HeapProperties = HeapProperties {
        kind: 1,
        cpu_pages: 0,
        pool: 0,
        creation_nodes: 0,
        visible_nodes: 0,
    };

    fn buffer(width: u64) -> ResourceDesc {
        ResourceDesc {
            dimension: BUFFER,
            alignment: 0,
            width,
            height: 1,
            depth_or_array_size: 1,
            mip_levels: 1,
            format: 0,
            sample_count: 1,
            sample_quality: 0,
            layout: 1,
            flags: 0,
        }
    }

    fn texture(width: u64, height: u32, format: u32, mips: u16) -> ResourceDesc {
        ResourceDesc {
            dimension: 3,
            width,
            height,
            format,
            mip_levels: mips,
            layout: 0,
            ..buffer(0)
        }
    }

    fn heap(size: u64) -> This {
        let desc = HeapDesc {
            size,
            properties: PROPERTIES,
            alignment: 0,
            flags: 0,
        };
        let mut heap = std::ptr::null();
        assert_eq!(
            create_heap(std::ptr::null(), &desc, &HEAP_IIDS[0], &mut heap),
            S_OK
        );
        heap
    }

    #[test]
    fn heaps_are_zeroed_aligned_memory() {
        let heap = heap(1 << 20);
        // SAFETY: a heap we just made.
        let memory = unsafe { &(*heap.cast::<Heap>()).memory };
        assert_eq!(memory.ptr.as_ptr() as u64 % KIB_64, 0);
        let mut back = std::mem::MaybeUninit::<HeapDesc>::uninit();
        slot::<extern "system" fn(This, *mut HeapDesc)>(heap, 0x40)(heap, back.as_mut_ptr());
        // SAFETY: written above.
        assert_eq!(unsafe { back.assume_init() }.size, 1 << 20);
        assert_eq!(slot::<Release>(heap, 0x10)(heap), 0);

        let bad = HeapDesc {
            size: 1 << 20,
            properties: PROPERTIES,
            alignment: 3,
            flags: 0,
        };
        let mut out = std::ptr::null();
        assert_eq!(
            create_heap(std::ptr::null(), &bad, &HEAP_IIDS[0], &mut out),
            E_INVALIDARG
        );
    }

    #[test]
    fn device_setup_heap_properties_and_resources_at_addresses() {
        let device = crate::device::device();
        // GetCustomHeapProperties (0xd0): the struct comes back through the
        // hidden first argument, which is also returned.
        type Custom =
            extern "system" fn(*mut HeapProperties, This, u32, u32) -> *mut HeapProperties;
        let mut p = PROPERTIES;
        let back = slot::<Custom>(device, 0xd0)(&mut p, device, 1, 2);
        assert_eq!(back, &raw mut p);
        assert_eq!(
            (
                p.kind,
                p.cpu_pages,
                p.pool,
                p.creation_nodes,
                p.visible_nodes
            ),
            (4, 3, 1, 1, 1),
            "upload: custom, write-back, L0"
        );
        slot::<Custom>(device, 0xd0)(&mut p, device, 1, 1);
        assert_eq!(p.cpu_pages, 1, "default: not CPU-visible");

        // 0x428: a buffer over memory the title owns, as BTP 6 asks.
        type At = extern "system" fn(
            This,
            u64,
            *const ResourceDesc,
            u32,
            *const c_void,
            *const Iid,
            *mut This,
        ) -> HRESULT;
        let memory = vec![0u8; 0x1_0000];
        let address = memory.as_ptr() as u64;
        let mut r = std::ptr::null();
        let create = slot::<At>(device, 0x428);
        let desc = buffer(0x1_0000);
        let null = std::ptr::null();
        assert_eq!(
            create(
                device,
                address,
                &desc,
                u32::MAX,
                null,
                &RESOURCE_IIDS[0],
                &mut r
            ),
            0
        );
        assert_eq!(slot::<Address>(r, 0x58)(r), address);
        assert_eq!(slot::<Release>(r, 0x10)(r), 0);
        // The title's memory is still there after the release.
        assert_eq!(memory[0xffff], 0);
        assert_ne!(
            create(device, 0, &desc, u32::MAX, null, &RESOURCE_IIDS[0], &mut r),
            0
        );
    }

    #[test]
    fn sizes_cover_every_mip() {
        let size = |d: &ResourceDesc| footprint(d).map(|i| i.size);
        assert_eq!(size(&buffer(256)), Some(KIB_64));
        // Textures take their console layout and its alignment (4 KiB
        // standard swizzle when the description leaves it to the library):
        // 512x512 BC7 with 10 mips is 0x56000 bytes, 4 KiB aligned, as on
        // the console.
        let layout = |d: &ResourceDesc| texture_layout(d).unwrap().size;
        let rgba = texture(256, 256, 28, 0);
        assert_eq!(size(&rgba), Some(layout(&rgba).next_multiple_of(4096)));
        let bc7 = footprint(&texture(512, 512, 98, 10));
        let console = AllocationInfo {
            size: 0x56000,
            alignment: 0x1000,
        };
        assert_eq!(bc7, Some(console));
        assert!(layout(&rgba) >= 4 * (256 * 256 * 4) / 3, "every mip");
        // BC1 is 8 bytes per 4x4 block: 1024x1024, one mip = 512 KiB.
        assert_eq!(size(&texture(1024, 1024, 71, 1)), Some(512 << 10));
        assert_eq!(size(&texture(16, 16, 66, 1)), None, "1-bit format");
        // Linear layout (1) and an explicit mode (64 KiB display, 0x10a).
        let linear = ResourceDesc {
            layout: 1,
            ..texture(100, 60, 28, 1)
        };
        assert_eq!(
            texture_layout(&linear).unwrap().planes[0]
                .layout
                .desc
                .swizzle
                .0,
            0
        );
        let display = ResourceDesc {
            layout: 0x10a,
            ..texture(100, 60, 28, 1)
        };
        assert_eq!(
            texture_layout(&display).unwrap().planes[0]
                .layout
                .desc
                .swizzle
                .0,
            10
        );

        let mut info = AllocationInfo {
            size: 0,
            alignment: 0,
        };
        let descs = [buffer(100), texture(256, 256, 28, 0)];
        allocation_info(&mut info, std::ptr::null(), 0, 2, descs.as_ptr());
        assert_eq!(
            info,
            AllocationInfo {
                size: KIB_64 + size(&descs[1]).unwrap(),
                alignment: KIB_64
            }
        );
        allocation_info(&mut info, std::ptr::null(), 0, 1, &texture(1, 1, 999, 1));
        assert_eq!(info.size, u64::MAX);
    }

    #[test]
    fn upload_footprints_follow_the_pc_rules() {
        let null = std::ptr::null();
        let mut layouts = [NO_FOOTPRINT; 2];
        let (mut rows, mut sizes, mut total) = ([0; 2], [0; 2], 0);
        let mut footprints = |desc: &ResourceDesc, first, count| {
            let (l, r, s) = (layouts.as_mut_ptr(), rows.as_mut_ptr(), sizes.as_mut_ptr());
            copyable_footprints(null, desc, first, count, 0, l, r, s, &mut total);
            (layouts, rows, sizes, total)
        };
        // 100x60 RGBA8, 2 mips: 400-byte rows pitched to 512.
        let (l, r, s, t) = footprints(&texture(100, 60, 28, 2), 0, 2);
        assert_eq!((l[0].row_pitch, r[0], s[0]), (512, 60, 400));
        // Mip 1 (50x30, 200-byte rows) starts after 59 pitched rows and one
        // packed row, rounded up to 512.
        assert_eq!(l[1].offset, (512 * 59 + 400u64).next_multiple_of(512));
        assert_eq!((l[1].width, l[1].height, r[1]), (50, 30, 30));
        assert_eq!(t, l[1].offset + 256 * 29 + 200);
        // BC1: rows of 4x4 blocks, 8 bytes each.
        let (_, r, s, _) = footprints(&texture(64, 64, 71, 1), 0, 1);
        assert_eq!((r[0], s[0]), (16, 128));
        let (_, _, _, t) = footprints(&texture(64, 64, 71, 1), 1, 1);
        assert_eq!(t, u64::MAX, "no second subresource");
    }

    /// A 4x4 texture of `format` filled from a title buffer (rows 256 bytes
    /// apart, `fill`ed) through our list and queue, then read back from the
    /// host GPU: the host texture's format and its rows (256 bytes apart).
    /// `None` without a host GPU (the skip reported for `test`).
    #[cfg(windows)]
    fn through_a_list(test: &str, format: u32, fill: impl Fn(&mut [u8])) -> Option<(u32, Vec<u8>)> {
        use windows::Win32::Graphics::Direct3D12::*;
        let gpu = crate::testgpu::device(test)?;
        let host = gpu.host();
        let null = std::ptr::null();
        let iid = &RESOURCE_IIDS[0];
        const COPY_DEST: u32 = 0x400;
        let (mut image, mut source) = (null, null);
        let desc = texture(4, 4, format, 1);
        assert_eq!(
            create_committed(
                null,
                &PROPERTIES,
                0,
                &desc,
                COPY_DEST,
                null.cast(),
                iid,
                &mut image
            ),
            S_OK
        );
        assert_eq!(
            create_committed(
                null,
                &PROPERTIES,
                0,
                &buffer(1024),
                0,
                null.cast(),
                iid,
                &mut source
            ),
            S_OK
        );
        let (data, _) = buffer_bytes(source).unwrap();
        // SAFETY: the buffer holds 1024 bytes.
        fill(unsafe { std::slice::from_raw_parts_mut(data, 1024) });

        let queue_iid = guid(
            0x0ec870a6,
            0x5d7e,
            0x4c22,
            [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
        );
        let allocator_iid = guid(
            0x6102dee4,
            0xaf59,
            0x4b09,
            [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
        );
        let list_iid = guid(
            0x5b160d0f,
            0xac1b,
            0x4185,
            [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
        );
        let (mut queue, mut allocator, mut list) = (null, null, null);
        let queue_desc = [0u32; 4];
        assert_eq!(
            crate::queue::create(null, queue_desc.as_ptr().cast(), &queue_iid, &mut queue),
            S_OK
        );
        assert_eq!(
            crate::list::create_allocator(null, 0, &allocator_iid, &mut allocator),
            S_OK
        );
        assert_eq!(
            crate::list::create_list(null, 0, 0, allocator, null, &list_iid, &mut list),
            S_OK
        );

        #[repr(C)]
        struct Location {
            resource: This,
            kind: u32,
            footprint: PlacedFootprint,
        }
        let rows = PlacedFootprint {
            offset: 0,
            format,
            width: 4,
            height: 4,
            depth: 1,
            row_pitch: 256,
        };
        let to = Location {
            resource: image,
            kind: 0,
            footprint: PlacedFootprint { offset: 0, ..rows },
        };
        let from = Location {
            resource: source,
            kind: 1,
            footprint: rows,
        };
        type Copy =
            extern "system" fn(This, *const Location, u32, u32, u32, *const Location, *const u8);
        slot::<Copy>(list, 0x80)(list, &to, 0, 0, 0, &from, null.cast());
        slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list);
        slot::<extern "system" fn(This, u32, *const This)>(queue, 0x50)(queue, 1, &list);

        // Read the host texture back into a readback buffer.
        let readback_heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let readback_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 1024,
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
        let mut readback: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a readback buffer.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &readback_heap,
                    D3D12_HEAP_FLAG_NONE,
                    &readback_desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .unwrap();
        }
        let readback = readback.unwrap();
        let host_image = host_resource(image).unwrap();
        // SAFETY: a live resource.
        let host_format = unsafe { host_image.GetDesc() }.Format.0 as u32;
        let back = host.new_list().unwrap();
        let (_, before) = host_transition(image, 0x800).unwrap();
        // The upload list holds the texture until it is reset.
        assert!(std::sync::Arc::strong_count(&host_copy(image).unwrap()) > 2);
        use windows::core::Interface;
        let out_to = Location {
            resource: readback.as_raw().cast_const().cast(),
            kind: 1,
            footprint: PlacedFootprint {
                format: host_format,
                ..rows
            },
        };
        let out_from = Location {
            resource: host_image.as_raw().cast_const().cast(),
            kind: 0,
            footprint: to.footprint,
        };
        // SAFETY: an open list, live resources, locations in the host layout.
        unsafe {
            back.list.ResourceBarrier(&[crate::host::transition(
                host_image,
                D3D12_RESOURCE_STATES(before as i32),
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            back.list.CopyTextureRegion(
                (&raw const out_to).cast(),
                0,
                0,
                0,
                (&raw const out_from).cast(),
                None,
            );
        }
        back.close();
        host.execute(&[&back]);
        host.drain();
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the copy finished (drained); the buffer is 1024 bytes.
        let read = unsafe {
            readback
                .Map(
                    0,
                    Some(&D3D12_RANGE {
                        Begin: 0,
                        End: 1024,
                    }),
                    Some(&mut mapped),
                )
                .unwrap();
            std::slice::from_raw_parts(mapped.cast::<u8>(), 1024).to_vec()
        };
        Some((host_format, read))
    }

    /// A 4x4 texture filled from a title buffer through our list and queue,
    /// then read back from the host GPU.
    #[cfg(windows)]
    #[test]
    fn buffer_data_reaches_the_host_texture() {
        let pattern = |bytes: &mut [u8]| {
            for (i, b) in bytes.iter_mut().enumerate() {
                *b = (i % 251) as u8;
            }
        };
        let test = "resource::tests::buffer_data_reaches_the_host_texture";
        let Some((format, read)) = through_a_list(test, 28, pattern) else {
            return;
        };
        assert_eq!(format, 28);
        let mut bytes = vec![0; 1024];
        pattern(&mut bytes);
        for row in 0..4 {
            let at = row * 256;
            assert_eq!(read[at..at + 16], bytes[at..at + 16], "row {row}");
        }
    }

    /// B5G5R5A1 texels copied from a title buffer reach a host texture in
    /// its stand-in as the same values.
    #[cfg(windows)]
    #[test]
    fn buffer_data_reaches_a_stand_in_texture_converted() {
        // Per texel x of row y: red x*8, green y*8, blue 31, alpha x odd.
        let texel = |x: u16, y: u16| 31 | (y * 8) << 5 | (x * 8) << 10 | (x & 1) << 15;
        let fill = |bytes: &mut [u8]| {
            for y in 0..4u16 {
                for x in 0..4u16 {
                    let at = usize::from(y) * 256 + usize::from(x) * 2;
                    bytes[at..at + 2].copy_from_slice(&texel(x, y).to_le_bytes());
                }
            }
        };
        let test = "resource::tests::buffer_data_reaches_a_stand_in_texture_converted";
        let Some((format, read)) = through_a_list(test, 86, fill) else {
            return;
        };
        if format == 86 {
            // Not a missing GPU: nothing to convert on this one (WARP).
            eprintln!("SKIPPED {test}: the host has every use of B5G5R5A1");
            return;
        }
        assert_eq!(format, 28, "R8G8B8A8_UNORM stands in");
        let wide = |v: u16| ((u32::from(v) * 255 + 15) / 31) as u8;
        for y in 0..4u16 {
            for x in 0..4u16 {
                let at = usize::from(y) * 256 + usize::from(x) * 4;
                let alpha = if x & 1 == 1 { 255 } else { 0 };
                assert_eq!(
                    read[at..at + 4],
                    [wide(x * 8), wide(y * 8), 255, alpha],
                    "texel {x},{y}"
                );
            }
        }
    }

    /// A WARP device (the software adapter), for tests that need a device
    /// with known support; `None` (and a note) without one.
    #[cfg(windows)]
    fn warp(test: &str) -> Option<windows::Win32::Graphics::Direct3D12::ID3D12Device> {
        use windows::Win32::Graphics::Dxgi::*;
        // SAFETY: plain DXGI and D3D12 creation.
        let device = unsafe {
            CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
                .and_then(|f| f.EnumWarpAdapter::<IDXGIAdapter>())
                .and_then(|adapter| {
                    let mut device = None;
                    crate::agility::create_device(&adapter, &mut device)?;
                    device.ok_or_else(windows::core::Error::empty)
                })
        };
        match device {
            Ok(device) => Some(device),
            Err(e) => {
                crate::testgpu::skip(test, &format!("no WARP device ({e})"));
                None
            }
        }
    }

    /// Every title format with every kind of use: what [`host_desc`] lets
    /// through, `device` creates (no creation reaches the driver with what
    /// it does not support), and the first run's two refusals have a host
    /// texture.
    #[cfg(windows)]
    fn host_descriptions_are_ones(device: &windows::Win32::Graphics::Direct3D12::ID3D12Device) {
        use windows::Win32::Graphics::Direct3D12::*;
        let create = |d: &ResourceDesc| {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let mut texture: Option<ID3D12Resource> = None;
            // SAFETY: a description in the host's layout; the out pointer is
            // ours.
            unsafe {
                device.CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    std::ptr::from_ref(d).cast(),
                    D3D12_RESOURCE_STATES(crate::host::legal_for(d.flags, 0) as i32),
                    None,
                    &mut texture,
                )
            }
        };
        let (mut made, mut refused, mut stricter) = (0, 0, 0);
        for format in 1..=191 {
            for flags in [0, 1, 2, 4, 1 | 4, 2 | 8] {
                for samples in [1, 4] {
                    if samples > 1 && flags & 3 == 0 {
                        continue;
                    }
                    let desc = ResourceDesc {
                        flags,
                        sample_count: samples,
                        ..texture(64, 64, format, 1)
                    };
                    match host_desc(&desc, device) {
                        Ok(d) => {
                            if let Err(e) = create(&d) {
                                panic!("{desc:?} as {d:?} passed but the device refused it: {e}");
                            }
                            made += 1;
                        }
                        Err(why) => {
                            refused += 1;
                            // What the device would have taken anyway.
                            let mut raw = desc;
                            raw.layout = 0;
                            raw.format = pc_vertex_format(format);
                            if create(&raw).is_ok() {
                                stricter += 1;
                                eprintln!("stricter than the device: {desc:?}: {why}");
                            }
                        }
                    }
                }
            }
        }
        eprintln!(
            "{made} made, {refused} refused ({stricter} of them the device would have taken)"
        );
        assert!(made > 500, "most formats and uses are made");
        // The first run's: a B5G5R5A1 target that is also unordered access,
        // and a B4G4R4A4 target, both 2560x1440 (in R8G8B8A8_UNORM where
        // the device lacks a use of theirs, as an RTX 4080 does).
        for (format, flags) in [(86, 1 | 4), (115, 1)] {
            let desc = ResourceDesc {
                flags,
                ..texture(2560, 1440, format, 1)
            };
            let d = host_desc(&desc, device).unwrap();
            assert!([format, 28].contains(&d.format), "{format} as {}", d.format);
            create(&d).unwrap();
        }
    }

    #[cfg(windows)]
    #[test]
    fn host_descriptions_are_ones_warp_creates() {
        if let Some(device) = warp("resource::tests::host_descriptions_are_ones_warp_creates") {
            // Probes descriptions the device may refuse.
            crate::testgpu::expect_layer_errors(|| host_descriptions_are_ones(&device));
        }
    }

    /// As above on the host GPU (an RTX 4080 on the development PC).
    #[cfg(windows)]
    #[test]
    fn host_descriptions_are_ones_the_host_gpu_creates() {
        let test = "resource::tests::host_descriptions_are_ones_the_host_gpu_creates";
        if let Some(host) = crate::testgpu::device(test) {
            // Probes descriptions the device may refuse.
            crate::testgpu::expect_layer_errors(|| host_descriptions_are_ones(&host.device));
        }
    }
    #[test]
    fn committed_and_placed_resources_map_at_their_gpu_address() {
        let null = std::ptr::null();
        let mut committed = null;
        let hr = create_committed(
            null,
            &PROPERTIES,
            0,
            &buffer(256),
            0,
            null.cast(),
            &RESOURCE_IIDS[0],
            &mut committed,
        );
        assert_eq!(hr, S_OK);
        let mut data = std::ptr::null_mut();
        assert_eq!(
            slot::<Map>(committed, 0x40)(committed, 0, null.cast(), &mut data),
            S_OK
        );
        assert_eq!(slot::<Address>(committed, 0x58)(committed), data as u64);
        // SAFETY: the buffer is at least 256 zeroed bytes.
        assert_eq!(unsafe { *data.add(255) }, 0);
        assert_eq!(slot::<Release>(committed, 0x10)(committed), 0);

        let heap = heap(4 * KIB_64);
        let place = |offset, desc: &ResourceDesc, out: &mut This| {
            create_placed(
                null,
                heap,
                offset,
                desc,
                0,
                null.cast(),
                &RESOURCE_IIDS[0],
                out,
            )
        };
        let mut placed = null;
        assert_eq!(place(KIB_64, &buffer(KIB_64), &mut placed), S_OK);
        // SAFETY: a heap we made.
        let base = unsafe { (*heap.cast::<Heap>()).memory.ptr.as_ptr() } as u64;
        assert_eq!(slot::<Address>(placed, 0x58)(placed), base + KIB_64);
        let mut other = null;
        assert_eq!(
            place(4 * KIB_64, &buffer(1), &mut other),
            E_INVALIDARG,
            "past the end"
        );
        assert_eq!(
            place(100, &buffer(1), &mut other),
            E_INVALIDARG,
            "misaligned"
        );
        // The resource keeps the heap alive.
        assert_eq!(slot::<Release>(heap, 0x10)(heap), 1);
        assert_eq!(slot::<Release>(placed, 0x10)(placed), 0);
    }

    /// The whole view of a texture is D3D12's own: every mip (and slice),
    /// the default component mapping, the dimension of the texture.
    #[cfg(windows)]
    #[test]
    fn whole_views_are_d3d12s_default() {
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::*;
        let d = |dimension, arrays, samples| D3D12_RESOURCE_DESC {
            Dimension: dimension,
            Width: 8,
            Height: 8,
            DepthOrArraySize: arrays,
            MipLevels: 1,
            Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: samples,
                Quality: 0,
            },
            ..Default::default()
        };
        let plain = whole_view(&d(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 1, 1));
        assert_eq!(plain.ViewDimension, D3D12_SRV_DIMENSION_TEXTURE2D);
        assert_eq!(plain.Format, DXGI_FORMAT_R16G16B16A16_FLOAT);
        assert_eq!(plain.Shader4ComponentMapping, 0x1688);
        // SAFETY: the 2D member of a 2D view.
        assert_eq!(unsafe { plain.Anonymous.Texture2D.MipLevels }, u32::MAX);
        let array = whole_view(&d(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 6, 1));
        assert_eq!(array.ViewDimension, D3D12_SRV_DIMENSION_TEXTURE2DARRAY);
        // SAFETY: the array member of an array view.
        assert_eq!(unsafe { array.Anonymous.Texture2DArray.ArraySize }, 6);
        let ms = whole_view(&d(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 1, 4));
        assert_eq!(ms.ViewDimension, D3D12_SRV_DIMENSION_TEXTURE2DMS);
        let volume = whole_view(&d(D3D12_RESOURCE_DIMENSION_TEXTURE3D, 4, 1));
        assert_eq!(volume.ViewDimension, D3D12_SRV_DIMENSION_TEXTURE3D);
        let line = whole_view(&d(D3D12_RESOURCE_DIMENSION_TEXTURE1D, 2, 1));
        assert_eq!(line.ViewDimension, D3D12_SRV_DIMENSION_TEXTURE1DARRAY);
        // Read alpha as one: only the alpha selector changes.
        let one = format::mapping_alpha_one(plain.Shader4ComponentMapping);
        assert_eq!(one, 0x1688 & !(7 << 9) | 5 << 9);
    }

    /// Descriptions as words, with these padding words.
    fn words(desc: &ResourceDesc, pad: [u32; 2]) -> [u32; 14] {
        // SAFETY: a description is 56 bytes.
        let mut words: [u32; 14] = unsafe { std::mem::transmute_copy(desc) };
        (words[1], words[13]) = (pad[0], pad[1]);
        words
    }

    #[test]
    fn descriptions_compare_without_their_padding() {
        let desc = texture(16, 16, 28, 1);
        let (a, b) = (words(&desc, [1, 2]), words(&desc, [0xdead, 0xbeef]));
        assert_ne!(a, b);
        assert_eq!(canonical_resource_desc(a), canonical_resource_desc(b));
        let c = canonical_resource_desc(a);
        assert_eq!((c[1], c[13]), (0, 0));
        assert_eq!(c[0], 3, "the dimension stays");
        assert_eq!(c[2..13], a[2..13], "every field stays");
    }

    /// 64 KiB of memory at an address no live texture, alias source or
    /// placed view of an earlier test names (addresses come back after a
    /// release).
    #[cfg(windows)]
    fn fresh_memory(held: &mut Vec<Memory>) -> Memory {
        loop {
            let memory = Memory::zeroed(KIB_64, KIB_64).unwrap();
            let at = memory.address();
            let placed = PLACED_TEXTURES
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .keys()
                .any(|&(a, _)| a == at);
            if !placed && !alias::has_sources(at) {
                return memory;
            }
            held.push(memory);
        }
    }

    #[cfg(windows)]
    fn at(address: u64, desc: &ResourceDesc) -> This {
        let mut out = std::ptr::null();
        let iid = &RESOURCE_IIDS[0];
        let made = create_at_address(
            std::ptr::null(),
            address,
            desc,
            0,
            std::ptr::null(),
            iid,
            &mut out,
        );
        assert_eq!(made, S_OK);
        out
    }

    #[cfg(windows)]
    fn resident(r: This) {
        let copy = resource(r).host.as_ref().unwrap();
        assert!(copy.residency.wait(std::time::Duration::from_secs(5)));
    }

    /// A new view may still be waiting for residency. Wait for its fence before
    /// requesting the resident view from the same entry point.
    #[cfg(windows)]
    fn resident_view(address: u64, desc: &[u32; 14]) -> This {
        if let Some(view) = texture_at(address, desc) {
            return view;
        }
        let key = (address, canonical_resource_desc(*desc));
        let view = placed_textures()[&key].resource.expect("a published view") as This;
        resident(view);
        texture_at(address, desc).unwrap()
    }

    /// Drain only the deferred references for this test's memory.
    #[cfg(windows)]
    fn release_views_at(address: u64) {
        let mut due = Vec::new();
        RELEASED
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|&(_, raw)| {
                let r = raw as This;
                if resource(r).data() as u64 == address {
                    due.push(r);
                    false
                } else {
                    true
                }
            });
        for r in due {
            resource_release(r);
        }
    }

    #[cfg(windows)]
    fn same_copy(a: This, b: This) -> bool {
        Arc::ptr_eq(&host_copy(a).unwrap(), &host_copy(b).unwrap())
    }

    /// The 16x16 RGBA8 texels of a title texture's host copy, rows 256
    /// bytes apart.
    #[cfg(windows)]
    fn read_16x16(host: &crate::host::Host, resource: This) -> Vec<u8> {
        let copy = host_copy(resource).unwrap();
        let state = copy.state.load(std::sync::atomic::Ordering::Relaxed);
        tests_support::read_back(host, &copy.texture, state)
    }

    /// The dark world's first cause (VIctory #643): a render target drawn by
    /// the GPU, then read through a view of its memory (as BTP 6
    ///'s placed views are) or through another resource at its address.
    /// Both see what the GPU drew, not the zeros of title memory.
    #[cfg(windows)]
    #[test]
    fn views_of_a_drawn_target_read_what_the_gpu_drew() {
        let test = "resource::tests::views_of_a_drawn_target_read_what_the_gpu_drew";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let mut held = Vec::new();
        let memory = fresh_memory(&mut held);
        let address = memory.address();
        const RENDER_TARGET: u32 = 1;
        let target = at(
            address,
            &ResourceDesc {
                flags: RENDER_TARGET,
                ..texture(16, 16, 28, 1)
            },
        );
        // The title clears it through a render target view.
        let rtv = [0u64; 16];
        let null = std::ptr::null();
        crate::descriptor::create_rtv(null, target, null.cast(), rtv.as_ptr() as u64);
        let list = crate::testgpu::TitleList::new();
        type Clear = extern "system" fn(This, u64, *const [f32; 4], u32, *const u8);
        let color = [64.0f32 / 255.0, 128.0 / 255.0, 191.0 / 255.0, 1.0];
        slot::<Clear>(list.this(), 0x1a0)(list.this(), rtv.as_ptr() as u64, &color, 0, null.cast());
        list.run();

        // A placed view, its description padded with whatever the stack
        // held, twice: one resource, the target's host copy.
        let view = texture(16, 16, 28, 1);
        let placed = texture_at(address, &words(&view, [0xdead_beef, 7])).unwrap();
        let again = texture_at(address, &words(&view, [0xcafe_babe, 9])).unwrap();
        assert_eq!(placed, again, "padding makes no second texture");
        assert!(same_copy(placed, target));
        assert!(!crate::texture::title_written(
            host_resource(placed).unwrap()
        ));
        // A resource the title makes at that address shares it too.
        let other = at(address, &view);
        assert!(same_copy(other, target));

        let texels = read_16x16(host, placed);
        for y in 0..16 {
            for x in 0..16 {
                let at = y * 256 + x * 4;
                assert_eq!(texels[at..at + 4], [64, 128, 191, 255], "texel {x},{y}");
            }
        }
        for r in [other, target] {
            assert_eq!(slot::<Release>(r, 0x10)(r), 0);
        }
        assert!(!alias::has_sources(address));
    }

    /// What the host cannot view as asked keeps a copy of its own: another
    /// shape or family, writes the copy does not allow, a typed copy written
    /// in another format, shader reads of a depth target that denies them.
    /// A texture that only reads is no source: once the target is released,
    /// a later one at its address gets its own copy.
    #[cfg(windows)]
    #[test]
    fn aliases_share_only_what_the_host_can_view() {
        let test = "resource::tests::aliases_share_only_what_the_host_can_view";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let mut held = Vec::new();
        let memory = fresh_memory(&mut held);
        let address = memory.address();
        let view = texture(16, 16, 28, 1);
        let target = ResourceDesc { flags: 1, ..view };
        // A view first: no source, then the target cannot write its copy.
        let read = at(address, &view);
        let drawn = at(address, &target);
        assert!(!same_copy(read, drawn));
        let srgb = at(
            address,
            &ResourceDesc {
                format: 29,
                ..target
            },
        );
        assert!(
            !same_copy(drawn, srgb),
            "a typed copy drawn in another format"
        );
        let wide = at(address, &texture(32, 16, 28, 1));
        let half = at(address, &texture(16, 16, 10, 1));
        for other in [wide, half] {
            assert!(!same_copy(drawn, other), "another shape or family");
        }
        let unordered = at(address, &ResourceDesc { flags: 4, ..view });
        assert!(
            !same_copy(drawn, unordered),
            "writes the copy does not allow"
        );
        // An sRGB view reads the target's copy (in its format).
        let srgb_view = at(address, &texture(16, 16, 29, 1));
        assert!(same_copy(drawn, srgb_view));
        for r in [read, drawn, srgb, wide, half, unordered] {
            assert_eq!(slot::<Release>(r, 0x10)(r), 0);
        }
        // Only the view is left there: it is no source.
        assert!(!alias::has_sources(address));
        let later = at(address, &view);
        assert!(!same_copy(srgb_view, later));
        for r in [srgb_view, later] {
            assert_eq!(slot::<Release>(r, 0x10)(r), 0);
        }

        // Depth that shaders may not read.
        let depth = at(
            address,
            &ResourceDesc {
                flags: 2 | 8,
                ..texture(16, 16, 40, 1)
            },
        );
        let float = at(address, &texture(16, 16, 41, 1));
        assert!(!same_copy(depth, float));
        for r in [depth, float] {
            assert_eq!(slot::<Release>(r, 0x10)(r), 0);
        }
        assert!(!alias::has_sources(address));
    }

    /// The 32-bit texels of a title texture's host copy (at most 64 wide),
    /// in its own format, rows 64 texels apart.
    #[cfg(windows)]
    fn read_words(host: &crate::host::Host, resource: This) -> Vec<u32> {
        use windows::Win32::Graphics::Direct3D12::*;
        let copy = host_copy(resource).unwrap();
        let texture = &copy.texture;
        // SAFETY: a live resource.
        let d = unsafe { texture.GetDesc() };
        assert!(d.Width <= 64);
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        // SAFETY: a plain query for one subresource.
        unsafe {
            host.device
                .GetCopyableFootprints(&d, 0, 1, 0, Some(&mut footprint), None, None, None);
        }
        assert_eq!(footprint.Footprint.RowPitch, 256);
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let size = 256 * u64::from(d.Height);
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
        let mut readback: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a readback buffer.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .unwrap();
        }
        let readback = readback.unwrap();
        let list = host.new_list().unwrap();
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(&readback) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: footprint,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        let state = copy.state.load(std::sync::atomic::Ordering::Relaxed);
        let state = D3D12_RESOURCE_STATES(crate::host::legal_state(texture, state) as i32);
        let source = D3D12_RESOURCE_STATE_COPY_SOURCE;
        // SAFETY: an open list; both locations name live resources.
        unsafe {
            if state != source {
                list.list
                    .ResourceBarrier(&[crate::host::transition(texture, state, source)]);
            }
            list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
            if state != source {
                list.list
                    .ResourceBarrier(&[crate::host::transition(texture, source, state)]);
            }
        }
        list.close();
        host.execute(&[&list]);
        host.drain();
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the copy ran (drained); the buffer is `size` bytes.
        unsafe {
            readback.Map(0, None, Some(&mut mapped)).unwrap();
            let bytes = std::slice::from_raw_parts(mapped.cast::<u8>(), size as usize);
            let words = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&w| u32::from_le_bytes(w))
                .collect();
            readback.Unmap(0, None);
            words
        }
    }

    /// Light packed into an R32 target by the GPU, read through an
    /// R9G9B9E5 view placed at its address (fxceless's `alias_of`): the
    /// view's own copy holds title memory's zeros until a list reading it
    /// copies the target's texels into it; a second read in the list copies
    /// nothing more.
    #[cfg(windows)]
    #[test]
    fn a_view_in_another_format_reads_what_the_gpu_wrote() {
        let test = "resource::tests::a_view_in_another_format_reads_what_the_gpu_wrote";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let mut held = Vec::new();
        let memory = fresh_memory(&mut held);
        let address = memory.address();
        // R9G9B9E5 (0.5, 0.25, 0.125): mantissas 256, 128, 64, exponent 15.
        const PACKED: u32 = 256 | 128 << 9 | 64 << 18 | 15 << 27;
        const R32_FLOAT: u32 = 41;
        let target = at(
            address,
            &ResourceDesc {
                flags: 1,
                ..texture(16, 16, R32_FLOAT, 1)
            },
        );
        resident(target);
        let mut descriptor = [0u64; 16];
        let rtv = descriptor.as_mut_ptr() as u64;
        let null = std::ptr::null();
        crate::descriptor::create_rtv(null, target, null.cast(), rtv);
        let list = crate::testgpu::TitleList::new();
        type Clear = extern "system" fn(This, u64, *const [f32; 4], u32, *const u8);
        let packed = [f32::from_bits(PACKED), 0.0, 0.0, 0.0];
        let clear = slot::<Clear>(list.this(), 0x1a0);
        clear(list.this(), rtv, &packed, 0, null.cast());
        list.run();

        const R9G9B9E5: u32 = 67;
        let view = resident_view(address, &words(&texture(16, 16, R9G9B9E5, 1), [1, 2]));
        assert!(!same_copy(view, target), "two formats: two copies");
        let source = read_words(host, target);
        assert_eq!(source[0], PACKED, "packed word written to the R32 source");

        let reads = crate::testgpu::TitleList::new();
        assert!(alias_copy(reads.host(), view));
        assert!(
            alias_copy(reads.host(), view),
            "read again: nothing to copy"
        );
        reads.run();
        let texels = read_words(host, view);
        // SAFETY: the reader's command list retains the view's host copy.
        let format = unsafe { host_resource(view).unwrap().GetDesc() }.Format.0 as u32;
        for (x, y) in [(0, 0), (15, 15), (7, 9)] {
            let row = y * 64;
            match format {
                R9G9B9E5 => assert_eq!(texels[row + x], PACKED, "texel {x},{y}"),
                10 => assert_eq!(
                    texels[row + 2 * x..row + 2 * x + 2],
                    [0x3400_3800, 0x3c00_3000],
                    "RGBA16F texel {x},{y}"
                ),
                other => panic!("unexpected host format {other}"),
            }
        }
        // A texture of the target's own format over it is no such alias.
        let same = texture_at(address, &words(&texture(16, 16, R32_FLOAT, 1), [3, 4])).unwrap();
        assert!(!alias_copy(reads.host(), same));
        assert_eq!(slot::<Release>(target, 0x10)(target), 0);
    }

    /// The source a placed view aliases is pinned under the registry's lock so
    /// releasing the title's own reference between the lookup and the copy
    /// keeps the source alive until the alias copy finishes.
    #[cfg(windows)]
    #[test]
    fn an_alias_source_is_held_after_the_title_releases_it() {
        let test = "resource::tests::an_alias_source_is_held_after_the_title_releases_it";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let mut held = Vec::new();
        let memory = fresh_memory(&mut held);
        let address = memory.address();
        const R32_FLOAT: u32 = 41;
        const R9G9B9E5: u32 = 67;
        let source = at(
            address,
            &ResourceDesc {
                flags: 1,
                ..texture(4, 4, R32_FLOAT, 1)
            },
        );
        resident(source);
        let view = at(address, &texture(4, 4, R9G9B9E5, 1));
        assert!(alias::has_sources(address), "source registered at address");
        let (pinned, _host) =
            alias::written_beside(address, &resource(view).desc).expect("an alias source");
        assert_eq!(pinned.get(), source);
        // The title drops its own reference: the pin keeps the source alive
        // and registered so recording the alias copy still sees a live object.
        assert_eq!(resource_release(source), 1);
        assert!(
            host_resource(pinned.get()).is_some(),
            "a held source is still copyable"
        );
        assert!(
            alias::has_sources(address),
            "the held source is not unregistered yet"
        );
        drop(pinned);
        assert!(
            !alias::has_sources(address),
            "dropping the last reference unregisters and frees it"
        );
        assert_eq!(resource_release(view), 0);
    }

    /// Where the host keeps R9G9B9E5 as RGBA16F (the RTX 4080), light
    /// packed into an R32 target is decoded into the stand-in on the GPU:
    /// (0.5, 0.25, 0.125) and alpha one, as half floats. WARP keeps
    /// R9G9B9E5 itself, so the stand-in is made here as an RGBA16F texture.
    #[cfg(windows)]
    #[test]
    fn packed_light_is_decoded_into_an_rgba16f_stand_in() {
        use windows::Win32::Graphics::Dxgi::{
            CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIFactory4,
        };

        let test = "resource::tests::packed_light_is_decoded_into_an_rgba16f_stand_in";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        // SAFETY: plain DXGI queries for the adapter of this test's device.
        let adapter = unsafe {
            CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
                .unwrap()
                .EnumAdapterByLuid::<IDXGIAdapter1>(host.device.GetAdapterLuid())
                .unwrap()
                .GetDesc1()
                .unwrap()
        };
        let end = adapter
            .Description
            .iter()
            .position(|&c| c == 0)
            .unwrap_or(adapter.Description.len());
        let name = String::from_utf16_lossy(&adapter.Description[..end]);
        eprintln!(
            "{test}: selected DXGI adapter {name:?}, vendor {:#06x}, device {:#06x}, flags {:#x}",
            adapter.VendorId, adapter.DeviceId, adapter.Flags
        );
        let mut held = Vec::new();
        let written = fresh_memory(&mut held);
        const PACKED: u32 = 256 | 128 << 9 | 64 << 18 | 15 << 27;
        const R32_FLOAT: u32 = 41;
        let target = at(
            written.address(),
            &ResourceDesc {
                flags: 1,
                ..texture(16, 16, R32_FLOAT, 1)
            },
        );
        resident(target);
        let mut descriptor = [0u64; 16];
        let rtv = descriptor.as_mut_ptr() as u64;
        let null = std::ptr::null();
        crate::descriptor::create_rtv(null, target, null.cast(), rtv);
        let list = crate::testgpu::TitleList::new();
        type Clear = extern "system" fn(This, u64, *const [f32; 4], u32, *const u8);
        let packed = [f32::from_bits(PACKED), 0.0, 0.0, 0.0];
        let clear = slot::<Clear>(list.this(), 0x1a0);
        clear(list.this(), rtv, &packed, 0, null.cast());

        const RGBA16F: u32 = 10;
        let stand_in = at(written.address(), &texture(16, 16, 67, 1));
        // SAFETY: this test owns the view and has not used it yet. WARP normally
        // keeps format 67. Give it the same host copy used on RTX hardware.
        let view = unsafe { &mut *stand_in.cast_mut().cast::<Resource>() };
        view.host = host_texture(
            &texture(16, 16, RGBA16F, 1),
            0,
            None,
            TextureOrigin::Address,
        )
        .map(Arc::new);
        resident(stand_in);
        let to = host_copy(stand_in).unwrap();
        // SAFETY: the test retains the stand-in's live host copy.
        let format = unsafe { to.texture.GetDesc() }.Format.0 as u32;
        assert_eq!(format, RGBA16F, "forced stand-in host format");
        eprintln!("{test}: forced stand-in host format {format}");
        let elsewhere = fresh_memory(&mut held);
        let snapshot = at(elsewhere.address(), &texture(16, 16, RGBA16F, 1));
        resident(snapshot);
        const SHADER_RESOURCE: u32 = 0xc0;
        crate::list::ensure_state(list.host(), target, SHADER_RESOURCE);
        assert!(alias_copy(list.host(), stand_in));
        assert!(list.host().transition(target, SHADER_RESOURCE).is_none());
        assert!(alias_copy(list.host(), stand_in));
        // The GPU read between the two clears captures the first color.
        crate::list::ensure_state(list.host(), stand_in, 0x800);
        crate::list::ensure_state(list.host(), snapshot, 0x400);
        // SAFETY: the textures have matching formats and are in their copy states.
        unsafe {
            list.host()
                .list
                .CopyResource(host_resource(snapshot).unwrap(), &to.texture)
        };
        let second = [f32::from_bits(PACKED + (1 << 27)), 0.0, 0.0, 0.0];
        clear(list.this(), rtv, &second, 0, null.cast());
        assert!(alias_copy(list.host(), stand_in));
        // Record before submitting the list that fills the view, then replay.
        let again = crate::testgpu::TitleList::new();
        assert!(alias_copy(again.host(), stand_in));
        list.run();
        assert_eq!(
            to.decode_count.load(std::sync::atomic::Ordering::Relaxed),
            2
        );
        let words = read_words(host, snapshot);
        for (x, y) in [(0, 0), (15, 15), (7, 9)] {
            let at = y * 64 + 2 * x;
            // Halves 0.5 and 0.25, then 0.125 and 1.
            assert_eq!(
                [words[at], words[at + 1]],
                [0x3400_3800, 0x3c00_3000],
                "texel {x},{y}"
            );
        }
        let words = read_words(host, stand_in);
        assert_eq!([words[0], words[1]], [0x3800_3c00, 0x3c00_3400]);
        again.run();
        let expected = if crate::texture::predication() { 2 } else { 3 };
        assert_eq!(
            to.decode_count.load(std::sync::atomic::Ordering::Relaxed),
            expected
        );
        // A clear in another list precedes the same read on replay.
        let writer = crate::testgpu::TitleList::new();
        clear(writer.this(), rtv, &packed, 0, null.cast());
        writer.host().close();
        host.submit(&[writer.host(), again.host()]);
        host.drain();
        assert_eq!(
            to.decode_count.load(std::sync::atomic::Ordering::Relaxed),
            expected + 1
        );
        let words = read_words(host, stand_in);
        assert_eq!([words[0], words[1]], [0x3400_3800, 0x3c00_3000]);
        // Independent binary16 values: (1 + 1020/1024) * 2^15 = 65408
        // is 0x7bfc; 2^-24 is 0x0001; 2^-16 is 0x0100; 1 is 0x3c00.
        // Blue=256 keeps the minimum-red payload normal as R32_FLOAT,
        // so the clear does not flush its packed bits as a subnormal.
        for (case, packed, expected) in [
            ("maximum", 511 | 31 << 27, [0x0000_7bfc, 0x3c00_0000]),
            ("minimum", 1 | 256 << 18, [0x0000_0001, 0x3c00_0100]),
            ("zero", 0, [0x0000_0000, 0x3c00_0000]),
        ] {
            let list = crate::testgpu::TitleList::new();
            let color = [f32::from_bits(packed), 0.0, 0.0, 0.0];
            clear(list.this(), rtv, &color, 0, null.cast());
            assert!(alias_copy(list.host(), stand_in), "{case} decode");
            list.run();
            assert_eq!(read_words(host, target)[0], packed, "{case} source bits");
            let words = read_words(host, stand_in);
            eprintln!(
                "{test}: {case}: host format {format}, packed {packed:#010x}, expected {expected:08x?}, actual {:08x?}",
                [words[0], words[1]]
            );
            for (x, y) in [(0, 0), (15, 15), (7, 9)] {
                let at = y * 64 + 2 * x;
                assert_eq!([words[at], words[at + 1]], expected, "{case} texel {x},{y}");
            }
        }
        for r in [snapshot, stand_in, target] {
            assert_eq!(slot::<Release>(r, 0x10)(r), 0);
        }
    }

    /// A placed view made before the target it aliases is made again, on
    /// its next use, over the target's copy.
    #[cfg(windows)]
    #[test]
    fn a_placed_view_made_before_its_target_follows_it() {
        let test = "resource::tests::a_placed_view_made_before_its_target_follows_it";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let mut held = Vec::new();
        let memory = fresh_memory(&mut held);
        let address = memory.address();
        let view = words(&texture(16, 16, 28, 1), [3, 4]);
        let before = texture_at(address, &view).unwrap();
        assert!(crate::texture::title_written(
            host_resource(before).unwrap()
        ));
        // Taken now: once replaced, `before` may be released by any test
        // emptying the waiting list.
        let replaced = host_copy(before).unwrap();
        let target = at(
            address,
            &ResourceDesc {
                flags: 1,
                ..texture(16, 16, 28, 1)
            },
        );
        let after = texture_at(address, &view).unwrap();
        assert_ne!(after, before);
        assert!(same_copy(after, target));
        assert_eq!(
            texture_at(address, &words(&texture(16, 16, 28, 1), [5, 6])),
            Some(after)
        );
        assert_eq!(slot::<Release>(target, 0x10)(target), 0);
        // The view keeps the copy (title memory still holds what was
        // drawn, on the console); nothing makes it again.
        assert_eq!(texture_at(address, &view), Some(after));
        // The resource it replaced goes after its grace.
        memory_remapped(address as usize, KIB_64 as usize);
        release_views_at(address);
        assert_eq!(Arc::strong_count(&replaced), 1, "released");
    }

    /// The runtime remaps the memory of a placed view: its texture goes
    /// (after the grace), and the next view of it makes a new one.
    #[cfg(windows)]
    #[test]
    fn placed_views_go_with_their_memory() {
        let test = "resource::tests::placed_views_go_with_their_memory";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let mut held = Vec::new();
        let memory = fresh_memory(&mut held);
        let address = memory.address();
        let view = words(&texture(16, 16, 28, 1), [0, 0]);
        let first = texture_at(address, &view).unwrap();
        let copy = host_copy(first).unwrap();
        // Memory just past it is no matter.
        memory_remapped((address + KIB_64) as usize, KIB_64 as usize);
        assert_eq!(texture_at(address, &view), Some(first));
        memory_remapped(address as usize, 4096);
        let gone = PLACED_TEXTURES
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .keys()
            .all(|&(a, _)| a != address);
        assert!(gone);
        release_views_at(address);
        assert_eq!(Arc::strong_count(&copy), 1, "the resource let go of it");
        let again = texture_at(address, &view).unwrap();
        assert!(!Arc::ptr_eq(&host_copy(again).unwrap(), &copy));
        memory_remapped(address as usize, KIB_64 as usize);
        release_views_at(address);
    }

    #[cfg(windows)]
    #[test]
    fn a_read_only_placed_view_waits_for_its_first_mapped_page() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };

        let test = "resource::tests::a_read_only_placed_view_waits_for_its_first_mapped_page";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let desc = texture(256, 256, 28, 1);
        let len = footprint(&desc).unwrap().size as usize;
        let view = words(&desc, [0, 0]);
        // SAFETY: reserve an OS-selected range and commit one page inside it.
        let base = unsafe { VirtualAlloc(std::ptr::null(), len, MEM_RESERVE, PAGE_READWRITE) };
        assert!(!base.is_null() && len > 4096);
        let address = base as u64;
        assert!(texture_at_with(address, &view, |_| panic!("no host allocation")).is_none());
        assert!(!placed_textures().contains_key(&(address, view)));
        // SAFETY: the second page is within the reservation.
        let page = unsafe { VirtualAlloc(base.add(4096), 4096, MEM_COMMIT, PAGE_READWRITE) };
        assert!(!page.is_null());
        assert!(texture_at(address, &view).is_some());
        memory_remapped(address as usize, len);
        release_views_at(address);
        // SAFETY: release the original reservation once the view is gone.
        assert_ne!(unsafe { VirtualFree(base, 0, MEM_RELEASE) }, 0);
    }

    #[cfg(windows)]
    #[test]
    fn a_remap_cancels_a_placed_view_still_being_created() {
        let test = "resource::tests::a_remap_cancels_a_placed_view_still_being_created";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let mut held = Vec::new();
        let memory = fresh_memory(&mut held);
        let address = memory.address();
        let view = words(&texture(16, 16, 28, 1), [0, 0]);
        let mut old = None;
        let stale = texture_at_with(address, &view, |d| {
            assert!(texture_at_with(address, &view, |_| panic!("already building")).is_none());
            let mut out = std::ptr::null();
            let iid = &RESOURCE_IIDS[0];
            assert_eq!(
                at_address(address, d, std::ptr::null(), iid, &mut out, PLACED_VIEW),
                S_OK
            );
            old = resource(out).host.clone();
            memory_remapped(address as usize, KIB_64 as usize);
            Some(out)
        });
        assert!(stale.is_none());
        assert_eq!(
            Arc::strong_count(&old.unwrap()),
            1,
            "unpublished copy released"
        );
        assert!(texture_at(address, &view).is_some());
        memory_remapped(address as usize, KIB_64 as usize);
        release_views_at(address);
    }

    /// A remapped range that starts inside a placed view's texture takes it
    /// too (it shows that memory), not only those starting in the range.
    #[cfg(windows)]
    #[test]
    fn a_remapping_takes_the_views_it_overlaps() {
        let test = "resource::tests::a_remapping_takes_the_views_it_overlaps";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let mut held = Vec::new();
        let memory = fresh_memory(&mut held);
        let address = memory.address();
        // 64x64 RGBA8: 16 KiB.
        let desc = texture(64, 64, 28, 1);
        let len = footprint(&desc).unwrap().size;
        assert!(len >= 8192, "{len}");
        let view = words(&desc, [0, 0]);
        let first = texture_at(address, &view).unwrap();
        // Past its end: kept.
        memory_remapped((address + len) as usize, 4096);
        assert_eq!(texture_at(address, &view), Some(first));
        // Starting in its middle: taken.
        memory_remapped((address + len / 2) as usize, 4096);
        let gone = PLACED_TEXTURES
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .keys()
            .all(|&(a, _)| a != address);
        assert!(gone, "a texture over the remapped memory stays");
        release_views_at(address);
    }

    /// A view that shows a released target's copy keeps it until the title
    /// makes another texture at that address: then the view's next use
    /// makes a texture of its own (from title memory).
    #[cfg(windows)]
    #[test]
    fn a_released_targets_copy_goes_when_its_memory_is_used_again() {
        let test = "resource::tests::a_released_targets_copy_goes_when_its_memory_is_used_again";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let mut held = Vec::new();
        let memory = fresh_memory(&mut held);
        let address = memory.address();
        let plain = texture(16, 16, 28, 1);
        let target = at(address, &ResourceDesc { flags: 1, ..plain });
        let view = words(&plain, [0, 0]);
        let shown = resident_view(address, &view);
        assert!(same_copy(shown, target));
        let drawn = host_copy(target).unwrap();
        assert_eq!(slot::<Release>(target, 0x10)(target), 0);
        assert_eq!(
            texture_at(address, &view),
            Some(shown),
            "kept while nothing else is there"
        );
        // The title puts a texture of its own there.
        let streamed = at(address, &plain);
        release_views_at(address);
        // The allocator can reuse the released object's address. The host copy
        // retained in drawn identifies the contents being replaced.
        let now = resident_view(address, &view);
        assert!(!Arc::ptr_eq(&host_copy(now).unwrap(), &drawn));
        assert!(crate::texture::title_written(host_resource(now).unwrap()));
        assert_eq!(slot::<Release>(streamed, 0x10)(streamed), 0);
        memory_remapped(address as usize, KIB_64 as usize);
        release_views_at(address);
    }

    /// Textures the title fills itself are placed in the texture pool;
    /// targets, depth and unordered access stay committed, and every one
    /// is resident before it is handed to the GPU.
    #[cfg(windows)]
    #[test]
    fn title_filled_textures_go_to_the_pool_and_targets_do_not() {
        let test = "resource::tests::title_filled_textures_go_to_the_pool_and_targets_do_not";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let pooled = |flags: u32| {
            let r = tests_support::committed(&ResourceDesc {
                flags,
                ..texture(64, 64, 28, 1)
            });
            let host = resource(r).host.clone().expect("a host copy");
            let placed = host.slot.is_some();
            assert!(host.residency.wait(std::time::Duration::from_secs(5)));
            assert!(host_resource(r).is_some());
            assert_eq!(slot::<Release>(r, 0x10)(r), 0);
            placed
        };
        assert_eq!(pooled(0), pool::enabled(), "a texture the title fills");
        assert!(!pooled(1), "a render target");
        assert!(!pooled(4), "unordered access");
        // A stand-in's extra bytes are counted: R9G9B9E5 as RGBA16F doubles.
        let r = tests_support::committed(&texture(64, 64, 67, 1));
        let host = resource(r).host.clone().unwrap();
        if pc_format(67) == 10 {
            assert_eq!(
                host.extra * 2,
                host.bytes,
                "{} of {}",
                host.extra,
                host.bytes
            );
        }
        assert_eq!(slot::<Release>(r, 0x10)(r), 0);
    }
}
