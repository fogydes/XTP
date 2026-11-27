//! Command allocators and command lists.
//!
//! The title's inlined code writes packets into a list only while its put
//! pointer is below a limit, and calls the list's method otherwise. Our
//! lists keep the pointer and every limit at zero, so every command reaches
//! the table as a call.
//!
//! Each list records onto a host command list as its methods are called
//! (decision 0007); draw state lives in `crate::draw`. Methods not
//! written yet log once with their caller, so a run lists what to do next.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::mem::offset_of;

use crate::object::{
    self, ALLOCATOR, E_INVALIDARG, HRESULT, Iid, LIST, Object, S_OK, Table, This, guid,
};
use crate::queue::Written;
use crate::resource::PAGEABLE_IIDS;

const ALLOCATOR_IIDS: [Iid; 4] = [
    guid(
        0x6102dee4,
        0xaf59,
        0x4b09,
        [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

/// The list interfaces up to `ID3D12GraphicsCommandList6` (the table has
/// all their slots): those whose ids are the same as on PC, the console's
/// own List5 and List6 (BTP 6 asks for List6), then the
/// device child's.
const LIST_IIDS: [Iid; 10] = [
    guid(
        0x7116d91c,
        0xe7e4,
        0x47ce,
        [0xb8, 0xc6, 0xec, 0x81, 0x68, 0xf4, 0x37, 0xe5],
    ),
    guid(
        0x5b160d0f,
        0xac1b,
        0x4185,
        [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
    ),
    guid(
        0x553103fb,
        0x1fe7,
        0x4557,
        [0xbb, 0x38, 0x94, 0x6d, 0x7d, 0x0e, 0x7c, 0xa7],
    ),
    guid(
        0x38c3e585,
        0xff17,
        0x412c,
        [0x91, 0x50, 0x4f, 0xc6, 0xf9, 0xd7, 0x2a, 0x28],
    ),
    guid(
        0x6fda83a7,
        0xb84c,
        0x4e38,
        [0x9a, 0xc8, 0xc7, 0xbd, 0x22, 0x01, 0x6b, 0x3d],
    ),
    guid(
        0x8754318e,
        0xd3a9,
        0x4541,
        [0x98, 0xcf, 0x64, 0x5b, 0x50, 0xdc, 0x48, 0x74],
    ),
    guid(
        0x55050859,
        0x4024,
        0x474c,
        [0x98, 0xcf, 0x64, 0x5b, 0x51, 0xdc, 0x48, 0x74],
    ),
    guid(
        0xc3827890,
        0x4024,
        0x474c,
        [0x98, 0xcf, 0x64, 0x5b, 0x51, 0xdc, 0x48, 0x74],
    ),
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
struct Allocator {
    object: Object,
    kind: u32,
}

/// The layout the title's inlined code reads.
#[repr(C)]
struct List {
    object: Object,
    kind: u32,
    /// The put pointer, then the plain, draw, API and unbiased API limits:
    /// all zero, so every inline check fails.
    put: [u64; 5],
    reserved: [u32; 64],
    /// State the inline code keeps for building packets; unused by us.
    graphics_root_headers: [u32; 128],
    compute_root_headers: [u32; 128],
    pipeline_shadow: [u32; 8],
    /// Newer titles' lists are larger: BTP 6's state flush
    /// (title+0x32786) copies a pipeline's 32 bytes to +0x544..+0x564 and
    /// tests +0x548, so its layout runs past 0x560. Its size is unknown;
    /// this keeps our fields clear of whatever the title writes there.
    newer: [u32; 256],
    /// Ours, after the console layout: memory handed out by
    /// [`allocate_upload`], kept until the list is reset; the host list the
    /// calls record into (decision 0007), and the draw state the title set.
    uploads: std::sync::Mutex<Vec<Vec<u8>>>,
    #[cfg(all(windows, feature = "diagnostics"))]
    bounds_set: std::sync::atomic::AtomicBool,
    #[cfg(windows)]
    host: Option<crate::host::HostList>,
    #[cfg(windows)]
    state: std::sync::Mutex<crate::draw::State>,
    /// The pipeline the title set last (`SetPipelineState`, or the one its
    /// creation or reset passed): the program whose thread group
    /// `DispatchX` reads.
    #[cfg(windows)]
    pipeline: std::sync::atomic::AtomicUsize,
}
#[cfg(windows)]
impl Drop for List {
    /// The GPU may still read its upload memory, which goes before the host
    /// list (which waits for its submission too).
    fn drop(&mut self) {
        if let Some(host) = &self.host {
            host.settle();
        }
    }
}

const _: () = assert!(offset_of!(List, put) == 0x18);
const _: () = assert!(offset_of!(List, pipeline_shadow) + 32 == 0x560);

/// Runs `f` with the host list and draw state of a title list, if there
/// is a host GPU.
#[cfg(windows)]
fn with_state(this: This, f: impl FnOnce(&crate::host::HostList, &mut crate::draw::State)) {
    let Some(list) = host_list(this) else { return };
    // SAFETY: the title passes one of our lists.
    let state = unsafe { &(*this.cast::<List>()).state };
    f(list, &mut state.lock().unwrap_or_else(|e| e.into_inner()));
}

/// A host compute pass replaced the title's root signature and pipeline.
#[cfg(windows)]
pub(crate) fn forget_host_bindings(this: This) {
    with_state(this, |_, state| state.forget_host());
}

/// The host list behind a title list, if there is a host GPU.
#[cfg(windows)]
pub(crate) fn host_list<'a>(this: This) -> Option<&'a crate::host::HostList> {
    let list = query_list(this)?;
    // Une commande suivante (fermeture comprise) doit voir les résultats.
    list.query_resolves.flush(list);
    Some(list)
}

#[cfg(windows)]
fn query_list<'a>(this: This) -> Option<&'a crate::host::HostList> {
    if this.is_null() {
        return None;
    }
    // SAFETY: the title passes one of our lists.
    unsafe { (*this.cast::<List>()).host.as_ref() }
}

pub(crate) extern "system" fn create_allocator(
    _: This,
    kind: u32,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    let allocator = Box::new(Allocator {
        object: Object::new(&ALLOCATOR_FUNCTIONS),
        kind,
    });
    object::hand_out(ALLOCATOR, allocator, iid, &ALLOCATOR_IIDS, out)
}

/// Lists start open, as in D3D12.
pub(crate) extern "system" fn create_list(
    _: This,
    _nodes: u32,
    kind: u32,
    allocator: This,
    pipeline: This,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    new_list(kind, allocator, pipeline, iid, out)
}

/// The console's list creation: type, flags, node mask and barrier batch
/// size in a description.
#[repr(C)]
#[derive(Debug)]
pub(crate) struct ListDesc {
    kind: u32,
    flags: u32,
    nodes: u32,
    barrier_batch: u32,
}

pub(crate) extern "system" fn create_list_x(
    _: This,
    desc: *const ListDesc,
    allocator: This,
    pipeline: This,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(?desc, "console command list");
    new_list(desc.kind, allocator, pipeline, iid, out)
}

/// A list of `kind`, open, with `pipeline` (null: none) set as
/// `SetPipelineState` sets one: D3D12's initial pipeline state, which
/// draws and dispatches recorded before the title sets another use.
fn new_list(
    kind: u32,
    allocator: This,
    pipeline: This,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    if allocator.is_null() {
        return E_INVALIDARG;
    }
    tracing::debug!(kind, "command list");
    let list = Box::new(List {
        object: if crate::device::newer() {
            Object::new(&NEWER_LIST_FUNCTIONS)
        } else {
            Object::new(&LIST_FUNCTIONS)
        },
        kind,
        put: [0; 5],
        reserved: [0; 64],
        graphics_root_headers: [0; 128],
        compute_root_headers: [0; 128],
        pipeline_shadow: [0; 8],
        newer: [0; 256],
        uploads: Default::default(),
        #[cfg(all(windows, feature = "diagnostics"))]
        bounds_set: Default::default(),
        #[cfg(windows)]
        host: crate::host::host().and_then(|h| match h.new_list() {
            Ok(list) => Some(list),
            Err(e) => {
                tracing::warn!(error = %e, "no host command list");
                None
            }
        }),
        #[cfg(windows)]
        state: {
            let mut state = crate::draw::State::default();
            state.set_pipeline(pipeline);
            std::sync::Mutex::new(state)
        },
        #[cfg(windows)]
        pipeline: std::sync::atomic::AtomicUsize::new(pipeline as usize),
    });
    let made = object::hand_out(LIST, list, iid, &LIST_IIDS, out);
    if made == S_OK {
        crate::pipeline::bound(pipeline);
    }
    made
}

/// Keeps `pipeline` as the one the title set last on `this` (see
/// [`List::pipeline`]).
#[cfg(windows)]
fn keep_pipeline(this: This, pipeline: This) {
    if this.is_null() {
        return;
    }
    // SAFETY: the title passes one of our lists.
    let kept = unsafe { &(*this.cast::<List>()).pipeline };
    kept.store(pipeline as usize, std::sync::atomic::Ordering::Relaxed);
}

/// The pipeline the title set last on `this`.
#[cfg(windows)]
fn kept_pipeline(this: This) -> This {
    if this.is_null() {
        return std::ptr::null();
    }
    // SAFETY: the title passes one of our lists.
    let kept = unsafe { &(*this.cast::<List>()).pipeline };
    kept.load(std::sync::atomic::Ordering::Relaxed) as This
}

extern "system" fn allocator_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let found = object::is(iid, &ALLOCATOR_IIDS).then_some(this);
    object::answer(ALLOCATOR, iid, found, out)
}
extern "system" fn allocator_release(this: This) -> u32 {
    object::release_boxed::<Allocator>(this)
}
/// Nothing is recorded, so there is nothing to free.
extern "system" fn allocator_reset(_: This) -> HRESULT {
    S_OK
}

extern "system" fn list_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(LIST, iid, object::is(iid, &LIST_IIDS).then_some(this), out)
}
extern "system" fn list_release(this: This) -> u32 {
    object::release_boxed::<List>(this)
}
extern "system" fn list_type(this: This) -> u32 {
    // SAFETY: `this` is one of our lists.
    unsafe { (*this.cast::<List>()).kind }
}
extern "system" fn close(this: This) -> HRESULT {
    #[cfg(windows)]
    if let Some(host) = host_list(this) {
        host.close();
    }
    S_OK
}
extern "system" fn reset(this: This, allocator: This, pipeline: This) -> HRESULT {
    if allocator.is_null() {
        return E_INVALIDARG;
    }
    // A queue holding the list behind a fence runs it first.
    #[cfg(windows)]
    crate::queue::list_reset(this);
    // SAFETY: the title passes one of our lists.
    let uploads = unsafe { &(*this.cast::<List>()).uploads };
    let mut uploads = uploads.lock().unwrap_or_else(|e| e.into_inner());
    // Memory the list's running submission may still use goes once it has
    // finished (the host list hands over its own in `reset`).
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && list.running()
        && let Some(host) = crate::host::made()
    {
        let old = std::mem::take(&mut *uploads);
        host.after(move || drop(old));
    }
    uploads.clear();
    drop(uploads);
    #[cfg(all(windows, feature = "diagnostics"))]
    // SAFETY: le titre réinitialise une de nos listes.
    unsafe { &(*this.cast::<List>()).bounds_set }
        .store(false, std::sync::atomic::Ordering::Relaxed);
    #[cfg(windows)]
    keep_pipeline(this, pipeline);
    #[cfg(windows)]
    with_state(this, |host, state| {
        host.reset();
        state.reset(pipeline);
    });
    crate::pipeline::bound(pipeline);
    #[cfg(not(windows))]
    let _ = this;
    S_OK
}

/// A resource barrier as titles pass it: type, flags, then a resource and
/// either a subresource and two states (transition) or a second resource
/// (aliasing). The same layout as the host's.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Barrier {
    kind: u32,
    flags: u32,
    first: This,
    second: u64,
    last: u64,
}

/// The barrier on host resources, or `None` if it names a resource the host
/// GPU does not have (buffers so far).
#[cfg(windows)]
fn host_barrier(list: &crate::host::HostList, b: &Barrier) -> Option<Barrier> {
    use windows::core::Interface;
    let host = |r: This| crate::resource::host_resource(r).map(|h| h.as_raw().cast_const().cast());
    match b.kind {
        0 => {
            // A split barrier's first half: its second half transitions.
            const BEGIN_ONLY: u32 = 1;
            if b.flags & BEGIN_ONLY != 0 {
                return None;
            }
            // shortcut: whole resources, from the state the list tracked
            // (`HostList::transition`); subresources are not told apart.
            let wanted = crate::host::texture_state(b.last as u32);
            let (texture, before) = list.transition(b.first, wanted)?;
            Some(Barrier {
                kind: 0,
                flags: 0,
                first: texture.as_raw().cast_const().cast(),
                second: 0xffff_ffff | (u64::from(before) << 32),
                last: u64::from(crate::host::legal_state(texture, wanted)),
            })
        }
        1 => Some(Barrier {
            first: if b.first.is_null() {
                b.first
            } else {
                host(b.first)?
            },
            second: if b.second == 0 {
                0
            } else {
                host(b.second as This)? as u64
            },
            ..*b
        }),
        2 => Some(Barrier {
            first: if b.first.is_null() {
                b.first
            } else {
                host(b.first)?
            },
            ..*b
        }),
        _ => None,
    }
}

/// An unordered access barrier on every resource: earlier shader writes
/// finish before later work reads or writes.
const UAV_BARRIER: Barrier = Barrier {
    kind: 2,
    flags: 0,
    first: std::ptr::null(),
    second: 0,
    last: 0,
};

/// Returns the console's barrier "actions" (none).
extern "system" fn barrier(this: This, count: u32, barriers: *const Barrier) -> u32 {
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && !barriers.is_null()
    {
        // SAFETY: the title passes `count` barriers.
        let barriers = unsafe { std::slice::from_raw_parts(barriers, count as usize) };
        let mut host: Vec<Barrier> = barriers
            .iter()
            .filter_map(|b| host_barrier(list, b))
            .collect();
        // Buffers have host copies, not resources the barrier can name:
        // a barrier on one waits for every shader write before it.
        if barriers.iter().any(|b| {
            b.kind != 1 && !b.first.is_null() && crate::resource::buffer_bytes(b.first).is_some()
        }) {
            host.push(UAV_BARRIER);
        }
        if !host.is_empty() {
            use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_BARRIER;
            const _: () = assert!(size_of::<Barrier>() == size_of::<D3D12_RESOURCE_BARRIER>());
            // SAFETY: the same layout; borrowed resource pointers, never
            // released through this slice.
            let host = unsafe {
                std::slice::from_raw_parts(
                    host.as_ptr().cast::<D3D12_RESOURCE_BARRIER>(),
                    host.len(),
                )
            };
            // SAFETY: an open list and live resources.
            unsafe { list.list.ResourceBarrier(host) };
        }
    }
    0
}

/// Moves a title texture's host copy to `state` in the list, if needed.
#[cfg(windows)]
pub(crate) fn ensure_state(list: &crate::host::HostList, resource: This, state: u32) {
    use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES as State;
    if let Some((texture, before)) = list.transition(resource, state) {
        let state = crate::host::legal_state(texture, state);
        let barrier = crate::host::transition(texture, State(before as i32), State(state as i32));
        // SAFETY: an open list and a live texture.
        unsafe { list.list.ResourceBarrier(&[barrier]) };
    }
}

#[cfg(windows)]
const COPY_DEST: u32 = 0x400;
#[cfg(windows)]
const COPY_SOURCE: u32 = 0x800;

/// `len` bytes at `offset` of a title buffer, if they are inside it.
#[cfg(windows)]
fn buffer_range(buffer: This, offset: u64, len: u64) -> Option<*mut u8> {
    let (data, size) = crate::resource::buffer_bytes(buffer)?;
    let end = offset.checked_add(len)?;
    if end > size {
        tracing::warn!(offset, len, size, "copy outside its buffer");
        return None;
    }
    // SAFETY: inside the buffer, checked above.
    Some(unsafe { data.add(offset as usize) })
}

/// A copy of `len` bytes of title memory from `src` to `dst` in `list`: on
/// the GPU, in list order, when either side has a host copy (shaders may
/// write the source before it or read the destination after it); else of
/// title memory on the CPU when the list is submitted.
#[cfg(windows)]
fn copy_title(list: &crate::host::HostList, dst: *mut u8, src: *const u8, len: u64) {
    if !crate::shadow::copy(list, dst as u64, src as u64, len) {
        list.defer_copy(dst, src, len as usize);
    }
}

/// Buffer to buffer (see [`copy_title`]).
extern "system" fn copy_buffer_region(
    this: This,
    dst: This,
    dst_offset: u64,
    src: This,
    src_offset: u64,
    len: u64,
) {
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && let Some(to) = buffer_range(dst, dst_offset, len)
        && let Some(from) = buffer_range(src, src_offset, len)
    {
        copy_title(list, to, from, len);
    }
    #[cfg(not(windows))]
    let _ = (this, dst, dst_offset, src, src_offset, len);
}

extern "system" fn copy_resource(this: This, dst: This, src: This) {
    #[cfg(windows)]
    {
        let Some(list) = host_list(this) else {
            return;
        };
        if let (Some(to), Some(from)) = (
            crate::resource::host_resource(dst),
            crate::resource::host_resource(src),
        ) {
            if crate::texture::title_written(from) {
                crate::texture::upload(list, src, from);
            }
            list.twin_use(dst, false, true);
            list.twin_use(src, false, false);
            ensure_state(list, dst, COPY_DEST);
            ensure_state(list, src, COPY_SOURCE);
            crate::resource::decode::wrote(list, dst);
            crate::texture::gpu_write(list, dst, true);
            #[cfg(feature = "diagnostics")]
            list.note_op(|| format!("copy resource {src:?} to {dst:?}"));
            // SAFETY: an open list and live textures.
            unsafe { list.list.CopyResource(to, from) };
        } else if let (Some((to, a)), Some((from, b))) = (
            crate::resource::buffer_bytes(dst),
            crate::resource::buffer_bytes(src),
        ) {
            copy_title(list, to, from, a.min(b));
        }
    }
    #[cfg(not(windows))]
    let _ = (this, dst, src);
}

extern "system" fn copy_resource_x(this: This, dst: This, src: This, _flags: u32) {
    copy_resource(this, dst, src);
}

/// A copy location as titles pass it, laid out as the host's: a resource,
/// then a subresource index (type 0) or a placed footprint (type 1).
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CopyLocation {
    resource: This,
    kind: u32,
    footprint: crate::resource::PlacedFootprint,
}

/// Texture to texture on the GPU; buffer to texture on the GPU from the
/// buffer's host copy when it has one (work of the list before the copy
/// may write it: a read-back, a shader, a buffer copy), else through an
/// upload buffer filled from title memory at submission; texture to buffer
/// (a read-back) on the GPU into the buffer's host copy ([`read_back`]).
/// Footprints in the console's 16-bit depth with stencil are refused
/// ([`console_depth`]).
extern "system" fn copy_texture_region(
    this: This,
    dst: *const CopyLocation,
    x: u32,
    y: u32,
    z: u32,
    src: *const CopyLocation,
    source_box: *const [u32; 6],
) {
    #[cfg(windows)]
    {
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_BOX, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_TEXTURE_COPY_LOCATION, ID3D12Resource,
        };
        use windows::core::Interface;
        const FOOTPRINT: u32 = 1;
        // SAFETY: the title passes both locations (null is tolerated).
        let (Some(list), Some(&dst), Some(&src)) =
            (host_list(this), unsafe { dst.as_ref() }, unsafe {
                src.as_ref()
            })
        else {
            return;
        };
        let Some(to) = crate::resource::host_resource(dst.resource) else {
            read_back(list, dst, [x, y, z], src, source_box);
            return;
        };
        if src.kind == FOOTPRINT
            && (console_depth(src.footprint.format()) || console_depth(title_format(dst.resource)))
        {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(
                    format = src.footprint.format(),
                    texture = title_format(dst.resource),
                    "a buffer-to-texture copy in the console's 16-bit depth with stencil, \
                     which no host footprint describes: not done"
                )
            });
            return;
        }
        // A destination whose host format stands in for the title's.
        if src.kind == FOOTPRINT
            && crate::texture::copy_converted(
                list,
                dst.resource,
                dst.footprint.offset as u32,
                [x, y, z],
                src.resource,
                &src.footprint,
                source_box,
            )
        {
            crate::resource::decode::wrote(list, dst.resource);
            return;
        }
        let mut from = src;
        let _upload; // keeps the source alive while this call records it
        // A buffer read in COPY_SOURCE by the copy, back in COMMON after it.
        let mut reading: Option<ID3D12Resource> = None;
        #[cfg(feature = "diagnostics")]
        let mut read = None;
        if src.kind == FOOTPRINT {
            let (Some(host), Some(bytes)) = (crate::host::host(), src.footprint.bytes()) else {
                return;
            };
            let Some(data) = buffer_range(src.resource, src.footprint.offset, bytes) else {
                return;
            };
            let address = data as u64;
            if let Some(copy) = crate::shadow::existing(address, bytes) {
                // The host copy, as earlier work of the list leaves it: its
                // CPU memory has the bytes only once the list has run.
                let Some((buffer, offset)) = copy_source(list, host, &copy, address, bytes) else {
                    return;
                };
                from.resource = buffer.as_raw().cast_const().cast();
                from.footprint.offset = offset;
                #[cfg(feature = "diagnostics")]
                {
                    read = Some((copy.gpu(address), bytes));
                }
                reading = Some(buffer.clone());
                _upload = buffer;
            } else {
                let Some((buffer, staging)) = list.upload(host, bytes) else {
                    return;
                };
                // The list owns this mapped D3D12 upload buffer. Wine can
                // report its address as MEM_FREE, so use the host-upload path.
                list.defer_fill(staging, data, bytes as usize);
                from.resource = buffer.as_raw().cast_const().cast();
                from.footprint.offset = 0;
                #[cfg(feature = "diagnostics")]
                {
                    // SAFETY: a live buffer.
                    read = Some((unsafe { buffer.GetGPUVirtualAddress() }, bytes));
                }
                _upload = buffer;
            }
        } else {
            let Some(texture) = crate::resource::host_resource(src.resource) else {
                return;
            };
            if crate::texture::title_written(texture) {
                crate::texture::upload(list, src.resource, texture);
            }
            list.twin_use(src.resource, false, false);
            ensure_state(list, src.resource, COPY_SOURCE);
            from.resource = texture.as_raw().cast_const().cast();
        }
        list.twin_use(dst.resource, false, true);
        ensure_state(list, dst.resource, COPY_DEST);
        crate::resource::decode::wrote(list, dst.resource);
        crate::texture::gpu_write(list, dst.resource, false);
        let mut to_location = dst;
        to_location.resource = to.as_raw().cast_const().cast();
        let host_box = (!source_box.is_null()).then(|| source_box.cast::<D3D12_BOX>());
        #[cfg(feature = "diagnostics")]
        {
            let text = || {
                // SAFETY: a live texture; the box is the title's or null.
                let (d, b) = unsafe { (to.GetDesc(), host_box.map(|b| *b)) };
                format!(
                    "copy to {:?} {}x{} f{} sub/kind {} at {x},{y},{z} from {:?} kind {} footprint {:?} box {b:?}",
                    dst.resource,
                    d.Width,
                    d.Height,
                    d.Format.0,
                    dst.kind,
                    src.resource,
                    src.kind,
                    (src.kind == FOOTPRINT).then_some(src.footprint)
                )
            };
            match read {
                Some(range) => list.note_read(text, range),
                None => list.note_op(text),
            }
        }
        const _: () =
            assert!(size_of::<CopyLocation>() == size_of::<D3D12_TEXTURE_COPY_LOCATION>());
        // SAFETY: an open list; both locations in the host's layout, naming
        // live resources; the box is the title's (same layout).
        unsafe {
            list.list.CopyTextureRegion(
                (&raw const to_location).cast(),
                x,
                y,
                z,
                (&raw const from).cast(),
                host_box,
            );
            if let Some(buffer) = &reading {
                list.list.ResourceBarrier(&[crate::host::transition(
                    buffer,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                )]);
            }
        };
    }
    #[cfg(not(windows))]
    let _ = (this, dst, x, y, z, src, source_box);
}

/// The title format of texture `resource` (0 if it is none of ours).
#[cfg(windows)]
fn title_format(resource: This) -> u32 {
    crate::resource::texture_memory(resource).map_or(0, |(_, desc)| desc.format())
}

/// Whether `format` is the console's 16-bit depth with 8-bit stencil (118,
/// and 119 and 120 of its family), which no PC format is: the host makes
/// such textures 24-bit depth with stencil
/// ([`crate::resource::pc_vertex_format`]), two planes that a footprint in
/// the console's layout (3 bytes a texel) does not describe, and D3D12
/// knows no footprint format by the console's number. Copies between such
/// textures and buffers are refused (logged once) instead of being handed
/// to the host.
/// shortcut: converting the planes (depth to and from 24 bits, stencil
/// apart) would copy them; no title read seen needs it yet.
#[cfg(windows)]
fn console_depth(format: u32) -> bool {
    (118..=120).contains(&format)
}

/// `bytes` of title memory at `address`, held by host copy `copy`, as the
/// source of a texture copy at this place in `list`: the host copy is
/// brought up to date from title memory at submission, but where work of
/// the submission writes it ([`crate::shadow::Shadow::refresh`]), and read
/// on the GPU, so a read-back, a shader or a buffer copy before it in the
/// list is seen. Returns the buffer to read, in COPY_SOURCE until the copy
/// is recorded (the caller then puts it back in COMMON), and the
/// footprint's offset in it: the host copy itself when the bytes sit at a
/// placement D3D12 accepts (512 bytes), else a buffer of the list's own
/// they are copied into first.
#[cfg(windows)]
fn copy_source(
    list: &crate::host::HostList,
    host: &crate::host::Host,
    copy: &std::sync::Arc<crate::shadow::Shadow>,
    address: u64,
    bytes: u64,
) -> Option<(windows::Win32::Graphics::Direct3D12::ID3D12Resource, u64)> {
    use crate::host::{transition, uav_barrier};
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
    };
    const PLACEMENT: u64 = 512;
    let (common, source, dest) = (
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    let offset = copy.offset(address);
    let through = if offset.is_multiple_of(PLACEMENT) {
        None
    } else {
        let (through, _) = host.shared_buffer(bytes)?;
        list.keep(&through);
        Some(through)
    };
    copy.refresh(list, address, bytes);
    let b = &copy.buffer;
    // SAFETY: an open list and live buffers (the refresh keeps the host
    // copy until the list is reset, the list its own buffer); the range is
    // inside both.
    unsafe {
        match through {
            None => {
                list.list
                    .ResourceBarrier(&[uav_barrier(), transition(b, common, source)]);
                Some((b.clone(), offset))
            }
            Some(through) => {
                list.list.ResourceBarrier(&[
                    uav_barrier(),
                    transition(b, common, source),
                    transition(&through, common, dest),
                ]);
                list.list.CopyBufferRegion(&through, 0, b, offset, bytes);
                list.list.ResourceBarrier(&[
                    transition(b, source, common),
                    transition(&through, dest, source),
                ]);
                Some((through, 0))
            }
        }
    }
}

/// Texture to buffer (a read-back) into footprint `dst` of a title buffer:
/// copied on the GPU, in list order, into the buffer's host copy as the
/// footprint lays the texels out (rows at the title's pitch; the texture's
/// console tiling is its host copy's business, as for the other copies),
/// then into title memory once the list has run, as buffer copies go
/// ([`crate::shadow::copy`]): only the bytes the copy writes, row by row of
/// texels (of blocks, for block-compressed formats) in each depth slice
/// ([`written_spans`]). The bytes it leaves (between rows, outside its box)
/// are the title's, which its threads may write while the list runs: none
/// of them goes back. (Before, the footprint's whole range went back, as
/// title memory had it at submission, over what the title wrote since.)
/// shortcut: the host places a footprint 512 bytes apart in its buffer,
/// with rows a multiple of 256 bytes apart; BTP 6's read-back in VIctory
/// #1583's run (5x1 texels of format 2, rows 256 bytes apart) fits. Another
/// placement, a texture whose host format stands in for the title's (the
/// bytes would need converting back), the console's 16-bit depth with
/// stencil ([`console_depth`]) and a source that is no texture are logged
/// once and not copied. Each row narrower than its pitch is a range of its
/// own for the submission to write back (rows as wide make one range): a
/// read-back of a big region in narrow rows declares many.
#[cfg(windows)]
fn read_back(
    list: &crate::host::HostList,
    dst: CopyLocation,
    at: [u32; 3],
    src: CopyLocation,
    source_box: *const [u32; 6],
) {
    use std::sync::Once;
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_BOX, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST,
    };
    use windows::core::Interface;
    const FOOTPRINT: u32 = 1;
    let footprint = dst.footprint;
    // Also a texture destination whose host copy is not ready.
    if dst.kind != FOOTPRINT || src.kind == FOOTPRINT {
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            tracing::warn!(
                dst = dst.kind,
                src = src.kind,
                "a texture copy to a destination without a host copy that is no read-back: not done"
            )
        });
        return;
    }
    let Some(texture) = crate::resource::host_resource(src.resource) else {
        static ONCE: Once = Once::new();
        ONCE.call_once(|| tracing::warn!("a read-back of a texture without a host copy: not done"));
        return;
    };
    // SAFETY: a live texture.
    let desc = unsafe { texture.GetDesc() };
    let host_format = desc.Format.0 as u32;
    let format = crate::resource::texture_memory(src.resource)
        .map_or(footprint.format(), |(_, desc)| desc.format());
    if console_depth(format) || console_depth(footprint.format()) {
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            tracing::warn!(
                format,
                footprint = footprint.format(),
                "a read-back in the console's 16-bit depth with stencil, which no host \
                 footprint describes: not done"
            )
        });
        return;
    }
    if crate::resource::format::conversion(format, host_format).is_some() {
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            tracing::warn!(
                format,
                host_format,
                "a read-back of a texture whose host format stands in for the title's: not done"
            )
        });
        return;
    }
    let Some(bytes) = footprint.bytes() else {
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            tracing::warn!(
                format = footprint.format(),
                "a read-back in a format of unknown size: not done"
            )
        });
        return;
    };
    let Some(target) = buffer_range(dst.resource, footprint.offset, bytes) else {
        return;
    };
    let target = target as u64;
    let Some(copy) = crate::shadow::find(target, bytes) else {
        return;
    };
    let offset = copy.offset(target);
    let (_, _, _, pitch) = footprint.shape();
    if !offset.is_multiple_of(512) || !pitch.is_multiple_of(256) {
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            tracing::warn!(
                offset,
                pitch,
                "a read-back the host cannot place (footprints 512 bytes apart, rows 256): not done"
            )
        });
        return;
    }
    // The subresource index is the first word of a location of kind 0.
    let size = copy_size(&desc, src.footprint.offset as u32, source_box);
    let Some(spans) = written_spans(&footprint, at, size) else {
        return;
    };
    if spans.is_empty() {
        return;
    }
    if crate::texture::title_written(texture) {
        crate::texture::upload(list, src.resource, texture);
    }
    list.twin_use(src.resource, false, false);
    ensure_state(list, src.resource, COPY_SOURCE);
    // What the copy writes goes back, and only that: the rest of the
    // range is the title's.
    for &(from, len) in &spans {
        copy.gpu_copies(list, target + from, len);
    }
    let mut to = dst;
    to.resource = copy.buffer.as_raw().cast_const().cast();
    to.footprint.offset = offset;
    let mut from = src;
    from.resource = texture.as_raw().cast_const().cast();
    let host_box = (!source_box.is_null()).then(|| source_box.cast::<D3D12_BOX>());
    #[cfg(feature = "diagnostics")]
    list.note_op(|| {
        format!(
            "read back {:?} into {:?} {footprint:?} at {at:?}",
            src.resource, dst.resource
        )
    });
    let (common, dest) = (D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    // SAFETY: an open list; both locations in the host's layout, naming a
    // live texture and the host copy the list keeps (`gpu_copies`), the
    // footprint inside it; the box is the title's (same layout).
    unsafe {
        list.list.ResourceBarrier(&[
            crate::host::uav_barrier(),
            crate::host::transition(&copy.buffer, common, dest),
        ]);
        list.list.CopyTextureRegion(
            (&raw const to).cast(),
            at[0],
            at[1],
            at[2],
            (&raw const from).cast(),
            host_box,
        );
        list.list
            .ResourceBarrier(&[crate::host::transition(&copy.buffer, dest, common)]);
    }
}

/// The texels (width, height, depth) a texture copy takes: its box (left,
/// top, front, right, bottom, back), else all of subresource `sub` of a
/// texture described by `desc`.
#[cfg(windows)]
fn copy_size(
    desc: &windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DESC,
    sub: u32,
    source_box: *const [u32; 6],
) -> [u32; 3] {
    use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    // SAFETY: the title passes a box or null.
    if let Some(&[left, top, front, right, bottom, back]) = unsafe { source_box.as_ref() } {
        return [
            right.saturating_sub(left),
            bottom.saturating_sub(top),
            back.saturating_sub(front),
        ];
    }
    let mip = sub % u32::from(desc.MipLevels.max(1));
    let depth = if desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        (u32::from(desc.DepthOrArraySize) >> mip).max(1)
    } else {
        1
    };
    [
        u32::try_from((desc.Width >> mip).max(1)).unwrap_or(u32::MAX),
        (desc.Height >> mip).max(1),
        depth,
    ]
}

/// The bytes a texture copy of `size` texels (width, height, depth) to
/// texel `at` of `footprint` writes, as (offset from the footprint's start,
/// length): in each depth slice it reaches, each row of texels (of 4x4
/// blocks, for block-compressed formats) from the block holding `at`'s
/// column to the one holding the region's last, rows `pitch` bytes apart
/// and slices as many rows apart as the footprint has. Cut to the
/// footprint; ranges that touch are merged (rows as wide as the pitch make
/// one range). `None` for a format of unknown size.
#[cfg(any(windows, test))]
fn written_spans(
    footprint: &crate::resource::PlacedFootprint,
    at: [u32; 3],
    size: [u32; 3],
) -> Option<Vec<(u64, u64)>> {
    let (element, block) = crate::resource::element(footprint.format())?;
    let (width, height, depth, pitch) = footprint.shape();
    let [x, y, z] = at.map(u64::from);
    let [w, h, d] = size.map(u64::from);
    let (pitch, rows) = (u64::from(pitch), u64::from(height).div_ceil(block));
    let (left, right) = (
        x / block,
        (x + w)
            .div_ceil(block)
            .min(u64::from(width).div_ceil(block)),
    );
    let (top, bottom) = (y / block, (y + h).div_ceil(block).min(rows));
    let back = (z + d).min(u64::from(depth));
    let mut spans: Vec<(u64, u64)> = Vec::new();
    if left >= right {
        return Some(spans);
    }
    let len = (right - left) * element;
    for slice in z..back {
        for row in top..bottom {
            let start = (slice * rows + row) * pitch + left * element;
            match spans.last_mut() {
                Some((from, n)) if *from + *n == start => *n += len,
                _ => spans.push((start, len)),
            }
        }
    }
    Some(spans)
}

#[allow(clippy::too_many_arguments)] // the method's own arguments
extern "system" fn copy_texture_region_x(
    this: This,
    dst: *const CopyLocation,
    x: u32,
    y: u32,
    z: u32,
    src: *const CopyLocation,
    source_box: *const [u32; 6],
    _flags: u32,
) {
    copy_texture_region(this, dst, x, y, z, src, source_box);
}

extern "system" fn resolve(
    this: This,
    dst: This,
    dst_sub: u32,
    src: This,
    src_sub: u32,
    format: u32,
) {
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && let (Some(to), Some(from)) = (
            crate::resource::host_resource(dst),
            crate::resource::host_resource(src),
        )
    {
        use windows::Win32::Foundation::RECT;
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_RESOLVE_MODE_AVERAGE, ID3D12GraphicsCommandList1,
        };
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
        use windows::core::Interface;
        const RESOLVE_DEST: u32 = 0x1000;
        const RESOLVE_SOURCE: u32 = 0x2000;
        list.twin_use(dst, false, true);
        list.twin_use(src, false, false);
        ensure_state(list, dst, RESOLVE_DEST);
        ensure_state(list, src, RESOLVE_SOURCE);
        crate::resource::decode::wrote(list, dst);
        crate::texture::gpu_write(list, dst, false);
        #[cfg(feature = "diagnostics")]
        list.note_op(|| format!("resolve {src:?} to {dst:?} format {format}"));
        // SAFETY: an open list and live textures; the region is inside both.
        unsafe {
            let (a, b) = (to.GetDesc(), from.GetDesc());
            let format = DXGI_FORMAT(format as i32);
            if (a.Width, a.Height) == (b.Width, b.Height) {
                list.list
                    .ResolveSubresource(to, dst_sub, from, src_sub, format);
            } else if let Ok(list1) = list.list.cast::<ID3D12GraphicsCommandList1>() {
                // The console clips a larger source to the destination.
                let rect = RECT {
                    left: 0,
                    top: 0,
                    right: a.Width.min(b.Width) as i32,
                    bottom: a.Height.min(b.Height) as i32,
                };
                list1.ResolveSubresourceRegion(
                    to,
                    dst_sub,
                    0,
                    0,
                    from,
                    src_sub,
                    Some(&rect),
                    format,
                    D3D12_RESOLVE_MODE_AVERAGE,
                );
            }
        }
    }
    #[cfg(not(windows))]
    let _ = (this, dst, dst_sub, src, src_sub, format);
}

#[repr(C)]
pub(crate) struct Rect {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

extern "system" fn clear_rtv(
    this: This,
    handle: u64,
    color: *const [f32; 4],
    count: u32,
    rects: *const Rect,
) {
    #[cfg(windows)]
    {
        let (Some(list), Some(record)) = (host_list(this), crate::descriptor::rtv_record(handle))
        else {
            return;
        };
        let (Some(texture), Some(host)) = (
            crate::resource::host_resource(record.resource),
            crate::host::host(),
        ) else {
            return;
        };
        // SAFETY: the title passes four floats (null is tolerated).
        let Some(color) = (unsafe { color.as_ref() }) else {
            return;
        };
        let rects = (count > 0 && !rects.is_null()).then(|| {
            // SAFETY: the title passes `count` rectangles, laid out as the host's.
            unsafe { std::slice::from_raw_parts(rects.cast(), count as usize) }
        });
        const RENDER_TARGET: u32 = 0x4;
        let rtv = host.rtv(texture, record.desc().as_ref());
        // SAFETY: an open list, a fresh view of a live texture.
        unsafe {
            if let Some((_, before)) = list.transition(record.resource, RENDER_TARGET) {
                use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES as State;
                list.list.ResourceBarrier(&[crate::host::transition(
                    texture,
                    State(before as i32),
                    State(RENDER_TARGET as i32),
                )]);
            }
            crate::resource::decode::wrote(list, record.resource);
            list.list.ClearRenderTargetView(rtv, color, rects);
        }
    }
    #[cfg(not(windows))]
    let _ = (this, handle, color, count, rects);
}

/// `ClearDepthStencilView(handle, flags, depth, stencil, count, rects)`;
/// flags: 1 depth, 2 stencil.
extern "system" fn clear_dsv(
    this: This,
    handle: u64,
    flags: u32,
    depth: f32,
    stencil: u8,
    count: u32,
    rects: *const Rect,
) {
    #[cfg(windows)]
    {
        use windows::Win32::Graphics::Direct3D12::D3D12_CLEAR_FLAGS;
        let (Some(list), Some(record)) = (host_list(this), crate::descriptor::dsv_record(handle))
        else {
            return;
        };
        let (Some(texture), Some(host)) = (
            crate::resource::host_resource(record.resource),
            crate::host::host(),
        ) else {
            return;
        };
        let rects = (count > 0 && !rects.is_null()).then(|| {
            // SAFETY: the title passes `count` rectangles, laid out as the host's.
            unsafe { std::slice::from_raw_parts(rects.cast(), count as usize) }
        });
        const DEPTH_WRITE: u32 = 0x10;
        list.twin_use(record.resource, false, true);
        ensure_state(list, record.resource, DEPTH_WRITE);
        let dsv = host.dsv(texture, record.depth_desc().as_ref());
        // SAFETY: an open list, a fresh view of a live texture.
        unsafe {
            list.list.ClearDepthStencilView(
                dsv,
                D3D12_CLEAR_FLAGS(flags as i32 & 3),
                depth,
                stencil,
                rects,
            )
        };
    }
    #[cfg(not(windows))]
    let _ = (this, handle, flags, depth, stencil, count, rects);
}

/// Logs a warning the first time it is reached.
#[cfg(windows)]
macro_rules! warn_once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// What a UAV clear writes: the low bits of each value into each channel
/// (`Uint`), or the values in the view's format (`Float`).
#[derive(Clone, Copy, Debug)]
enum Clear {
    Uint([u32; 4]),
    Float([f32; 4]),
}

/// `ClearUnorderedAccessViewUint` (0x1a8): `(GPU handle, CPU handle,
/// resource, values, rect count, rects)`, see [`clear_uav`]. FidelityFX's
/// upscaler (FSR4) calls it on Series S.
extern "system" fn clear_uav_uint(
    this: This,
    gpu: u64,
    cpu: u64,
    resource: This,
    values: *const [u32; 4],
    count: u32,
    rects: *const Rect,
) {
    // SAFETY: the title passes four values (null is tolerated).
    if let Some(&values) = unsafe { values.as_ref() } {
        clear_uav(
            this,
            (gpu, cpu),
            resource,
            Clear::Uint(values),
            count,
            rects,
        );
    }
}

/// `ClearUnorderedAccessViewFloat` (0x1b0), as [`clear_uav_uint`].
extern "system" fn clear_uav_float(
    this: This,
    gpu: u64,
    cpu: u64,
    resource: This,
    values: *const [f32; 4],
    count: u32,
    rects: *const Rect,
) {
    // SAFETY: the title passes four values (null is tolerated).
    if let Some(&values) = unsafe { values.as_ref() } {
        clear_uav(
            this,
            (gpu, cpu),
            resource,
            Clear::Float(values),
            count,
            rects,
        );
    }
}

/// Whether the words of a buffer view describe a raw or a structured view,
/// which has no format to clear in floats: a stride (word 5) or the raw
/// flag (word 8).
#[cfg(windows)]
fn untyped_buffer_view(words: &[u32]) -> bool {
    const RAW: u32 = 1;
    words.get(5).is_some_and(|&stride| stride != 0) || words.get(8).is_some_and(|&f| f & RAW != 0)
}

/// The host format a UAV clear writes in: the host texture's (its view is
/// of the texture's family, whose formats share their channels), else a
/// typed buffer view's as the host view has it; 0 for a raw or structured
/// buffer view.
#[cfg(windows)]
fn clear_format(
    target: &windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    words: Option<&[u32]>,
) -> u32 {
    // SAFETY: a live resource.
    let own = unsafe { target.GetDesc() }.Format.0 as u32;
    if own != 0 {
        return own;
    }
    words
        .filter(|w| !untyped_buffer_view(w))
        .and_then(|w| w.first())
        .map_or(0, |&f| crate::resource::pc_format(f))
}

/// The UAV clears. Both handles are the console's: descriptor slots in
/// title memory, the first in the bound shader-visible heap, the second in
/// a CPU-only one. They hold our view records, not host descriptors: a host
/// driver handed them reads garbage (on AMD the process ended at the
/// world's start, as xcl #1291 found). The host gets descriptors of its own
/// for the view the slots hold: one in a dynamic slot of XTP's own
/// shader-visible heap, bound for the clear, and one in a CPU-only heap of
/// ours ([`clear_slots`]). The clear writes what the view names: the host
/// copy of a texture, or the host copy of a buffer's range, which goes back
/// to title memory after the submission as shader writes do. `resource`
/// is the view's (D3D12 asks for the same), so the view's is used.
fn clear_uav(
    this: This,
    (gpu, cpu): (u64, u64),
    resource: This,
    clear: Clear,
    count: u32,
    rects: *const Rect,
) {
    #[cfg(windows)]
    {
        use crate::descriptor::View;
        use windows::Win32::Foundation::RECT;
        const UNORDERED_ACCESS: u32 = 0x8;
        // The title writes the view at the CPU handle and copies it to the
        // GPU one: either holds its record.
        let Some(view) = crate::descriptor::view(cpu).or_else(|| crate::descriptor::view(gpu))
        else {
            warn_once!("a UAV clear of a slot without a view of ours: not done");
            return;
        };
        // Résoudre la texture une seule fois pour les deux descripteurs,
        // la transition d'état et le suivi de son écriture.
        let view = match view {
            View::Placed(true, ref resource, _) if resource[0] != 1 => {
                let Some(texture) = crate::descriptor::host::placed_texture(&view) else {
                    return;
                };
                texture
            }
            view => view,
        };
        let words: Option<&[u32]> = match &view {
            View::Unordered(_, _, desc) => desc.as_ref().map(|d| &d[..]),
            View::Placed(true, _, desc) => Some(&desc[..10]),
            _ => {
                warn_once!("a UAV clear of a view that is not an unordered access view: not done");
                return;
            }
        };
        let range = crate::descriptor::host::buffer_range(&view);
        if matches!(clear, Clear::Float(_))
            && range.is_some()
            && words.is_some_and(untyped_buffer_view)
        {
            warn_once!("a float clear of a raw or structured buffer view, which the host refuses");
            return;
        }
        with_state(this, |list, state| {
            let (Some(heap), Some(slots)) = (crate::descriptor::own_view_heap(), clear_slots())
            else {
                return;
            };
            let Some(slot) = heap.dynamic(1) else {
                warn_once!("a UAV clear without a dynamic slot of XTP's view heap: not done");
                return;
            };
            list.keep_dynamic(crate::descriptor::OWN_VIEW_HEAP, slot, 1);
            // Held until the clear is recorded: the host reads the CPU-only
            // descriptor then.
            let mut slots = slots.lock().unwrap_or_else(|e| e.into_inner());
            let at = slots.take();
            let Some(target) = crate::descriptor::host::make_view(&view, at) else {
                warn_once!(
                    ?view,
                    "a UAV clear of a view the host cannot make: not done"
                );
                return;
            };
            if !heap.make_at(slot, &view) {
                warn_once!(
                    ?view,
                    "a UAV clear of a view the host cannot place in its heap: not done"
                );
                return;
            }
            // Drivers differ on values wider than their channels: give them
            // the low bits only (format::uint_clear).
            let clear = match clear {
                Clear::Uint(v) => Clear::Uint(crate::resource::format::uint_clear(
                    clear_format(&target, words),
                    v,
                )),
                float => float,
            };
            let mut buffer = false;
            if let Some((address, len)) = range
                && len > 0
                && let Some(shadow) = crate::shadow::find(address, len)
            {
                // A clear of part of the range (rectangles) keeps the rest.
                shadow.refresh(list, address, len);
                shadow.gpu_writes(list, address, len);
                buffer = true;
            } else if let View::Unordered(texture, _, _) = view {
                let texture = texture as This;
                #[cfg(feature = "diagnostics")]
                crate::capture::note_target(texture);
                list.twin_use(texture, true, true);
                // Which twin the view clears, of a depth-stencil texture's two.
                list.twin_view_written(texture, &view);
                list.twins_written();
                ensure_state(list, texture, UNORDERED_ACCESS);
                list.texture_written(texture);
                crate::resource::decode::wrote(list, texture);
                crate::texture::gpu_write(list, texture, false);
            }
            let rects: &[RECT] = if count > 0 && !rects.is_null() {
                // SAFETY: the title passes `count` rectangles, laid out as the host's.
                unsafe { std::slice::from_raw_parts(rects.cast(), count as usize) }
            } else {
                &[]
            };
            #[cfg(feature = "diagnostics")]
            list.note_op(|| format!("clear UAV {clear:?} of {resource:?}"));
            #[cfg(not(feature = "diagnostics"))]
            let _ = resource;
            list.keep(&heap.heap);
            list.keep(&target);
            let gpu = heap.gpu_handle(slot);
            // SAFETY: an open list; our shader-visible heap, bound here,
            // holds the view at `gpu`, the CPU-only one at `at`, both of
            // the live `target`.
            unsafe {
                list.list.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
                match clear {
                    Clear::Uint(v) => list
                        .list
                        .ClearUnorderedAccessViewUint(gpu, at, &target, &v, rects),
                    Clear::Float(v) => list
                        .list
                        .ClearUnorderedAccessViewFloat(gpu, at, &target, &v, rects),
                }
                // Later shader access sees the clear.
                list.list.ResourceBarrier(&[crate::host::uav_barrier()]);
                if buffer {
                    // The clear promoted the buffer's host copy from COMMON
                    // to UNORDERED_ACCESS for the rest of the list: back to
                    // COMMON, the state the list's copies and writes into it
                    // name before theirs (`read_back`, `shadow::copy`,
                    // `shadow::write`). Before, their barriers named COMMON
                    // after a clear (the debug layer's message 527).
                    use windows::Win32::Graphics::Direct3D12::{
                        D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    };
                    list.list.ResourceBarrier(&[crate::host::transition(
                        &target,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COMMON,
                    )]);
                }
            }
            drop(slots);
            // The title's heaps go back on the list at its next draw.
            state.forget_host();
        });
    }
    #[cfg(not(windows))]
    let _ = (this, gpu, cpu, resource, clear, count, rects);
}

/// CPU-only view descriptors for the clears' second handle, taken in turn.
/// The host reads one while the clear is recorded (D3D12 asks for a
/// CPU-only heap so that drivers read it fast on the CPU), so one would do;
/// the ring keeps a driver that read it later right for its next clears.
#[cfg(windows)]
struct ClearSlots {
    /// Held: the descriptors live in it.
    _heap: windows::Win32::Graphics::Direct3D12::ID3D12DescriptorHeap,
    start: usize,
    step: usize,
    next: usize,
}

// SAFETY: D3D12 descriptor heaps are free-threaded; the rest is plain data
// behind the mutex.
#[cfg(windows)]
unsafe impl Send for ClearSlots {}

#[cfg(windows)]
const CLEAR_SLOTS: usize = 256;

#[cfg(windows)]
impl ClearSlots {
    fn take(&mut self) -> windows::Win32::Graphics::Direct3D12::D3D12_CPU_DESCRIPTOR_HANDLE {
        let at = self.start + self.next * self.step;
        self.next = (self.next + 1) % CLEAR_SLOTS;
        windows::Win32::Graphics::Direct3D12::D3D12_CPU_DESCRIPTOR_HANDLE { ptr: at }
    }
}

/// The clears' CPU-only descriptors, made on first use; `None` without a
/// host GPU.
#[cfg(windows)]
fn clear_slots() -> Option<&'static std::sync::Mutex<ClearSlots>> {
    use windows::Win32::Graphics::Direct3D12::*;
    static SLOTS: std::sync::OnceLock<Option<std::sync::Mutex<ClearSlots>>> =
        std::sync::OnceLock::new();
    SLOTS
        .get_or_init(|| {
            let host = crate::host::host()?;
            // SAFETY: plain heap creation and queries of a live heap.
            unsafe {
                let made = host.device.CreateDescriptorHeap::<ID3D12DescriptorHeap>(
                    &D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        NumDescriptors: CLEAR_SLOTS as u32,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                        NodeMask: 0,
                    },
                );
                let heap = match made {
                    Ok(heap) => heap,
                    Err(e) => {
                        tracing::warn!(error = %e, "no CPU-only heap for UAV clears");
                        return None;
                    }
                };
                let start = heap.GetCPUDescriptorHandleForHeapStart().ptr;
                let step = host
                    .device
                    .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
                    as usize;
                Some(std::sync::Mutex::new(ClearSlots {
                    _heap: heap,
                    start,
                    step,
                    next: 0,
                }))
            }
        })
        .as_ref()
}

/// A bilinear upscale, what `xtp_hle`'s stand-in for FidelityFX's upscaler
/// records for `ffxDispatch`: the `render` texels at the top left of
/// `color` stretched over the `size` pixels at the top left of `output`.
/// `jitter` is the dispatch's jitter offset, in `color`'s texels: the
/// camera offset the frame was rendered with, which the upscale takes back
/// (see [`upscale`]).
#[derive(Debug, Clone, Copy)]
pub struct Upscale {
    /// The title's command list.
    pub list: *const Object,
    /// The title's textures (`ID3D12Resource`).
    pub color: *const Object,
    pub output: *const Object,
    pub render: [u32; 2],
    pub size: [u32; 2],
    pub jitter: [f32; 2],
}

/// Records `u` in its list on the host GPU, a compute pass of ours: each
/// output pixel takes the input bilinearly at its own place in the input,
/// moved by the jitter as FidelityFX's upscalers place their input samples
/// (the texel at `i` holds the scene at `i + 0.5 - jitter`), and clamped
/// to the rendered region. `color` ends readable by compute shaders and
/// `output` in unordered access, the states the title gives FidelityFX.
/// False (nothing recorded) without a host GPU, without host copies of the
/// textures or views of them, or without descriptor slots.
pub fn upscale(u: &Upscale) -> bool {
    #[cfg(windows)]
    {
        let mut ran = false;
        with_state(u.list, |list, state| ran = upscale::record(list, state, u));
        ran
    }
    #[cfg(not(windows))]
    {
        let _ = u;
        false
    }
}

/// The bilinear upscale's shader and recording (see [`upscale`]).
#[cfg(windows)]
mod upscale {
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::core::s;

    use super::{This, Upscale, ensure_state};
    use crate::descriptor::View;
    use crate::host::HostList;

    const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=8, b0), DescriptorTable(SRV(t0)), " \
    "DescriptorTable(UAV(u0)), StaticSampler(s0, filter=FILTER_MIN_MAG_MIP_LINEAR, " \
    "addressU=TEXTURE_ADDRESS_CLAMP, addressV=TEXTURE_ADDRESS_CLAMP, " \
    "addressW=TEXTURE_ADDRESS_CLAMP)"
cbuffer Upscale : register(b0) {
    // Input texels per output pixel; the jitter, in input texels.
    float2 scale;
    float2 jitter;
    // The rendered region of the input, in texels.
    float2 render;
    // The output pixels to write.
    uint2 size;
};
Texture2D<float4> color : register(t0);
RWTexture2D<float4> output : register(u0);
SamplerState linear_clamp : register(s0);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= size)) return;
    // Where this pixel's centre is in the input, in texels: the texel at
    // i holds the scene at i + 0.5 - jitter.
    float2 at = (float2(id.xy) + 0.5) * scale + jitter;
    at = clamp(at, 0.5, render - 0.5);
    float w, h;
    color.GetDimensions(w, h);
    output[id.xy] = color.SampleLevel(linear_clamp, at / float2(w, h), 0);
}
"#;

    pub(super) struct Pass {
        root: ID3D12RootSignature,
        pso: ID3D12PipelineState,
    }

    // SAFETY: D3D12 objects are free-threaded.
    unsafe impl Send for Pass {}
    // SAFETY: as above.
    unsafe impl Sync for Pass {}

    pub(super) fn pass() -> Option<&'static Pass> {
        static PASS: std::sync::OnceLock<Option<Pass>> = std::sync::OnceLock::new();
        PASS.get_or_init(|| match make() {
            Ok(p) => Some(p),
            Err(e) => {
                tracing::warn!(error = %e, "no bilinear upscale pass: FidelityFX dispatches do nothing");
                None
            }
        })
        .as_ref()
    }

    fn make() -> windows::core::Result<Pass> {
        let device = &crate::host::host()
            .ok_or_else(windows::core::Error::empty)?
            .device;
        let mut code: Option<ID3DBlob> = None;
        let mut errors: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call; out pointers ours.
        let compiled = unsafe {
            D3DCompile(
                SOURCE.as_ptr().cast(),
                SOURCE.len(),
                s!("upscale"),
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
                tracing::warn!("bilinear upscale: {}", String::from_utf8_lossy(text));
            }
            return Err(e);
        }
        let code = code.ok_or_else(windows::core::Error::empty)?;
        // SAFETY: the blob's own bytes; the root signature is inside the shader.
        unsafe {
            let bytes = std::slice::from_raw_parts(
                code.GetBufferPointer().cast::<u8>(),
                code.GetBufferSize(),
            );
            let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
            let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            })?;
            Ok(Pass { root, pso })
        }
    }

    /// The constants the shader takes, in its order.
    pub(super) fn constants(u: &Upscale) -> [u32; 8] {
        let [rw, rh] = u.render.map(|v| v.max(1) as f32);
        let [ow, oh] = u.size.map(|v| v.max(1) as f32);
        [
            (rw / ow).to_bits(),
            (rh / oh).to_bits(),
            u.jitter[0].to_bits(),
            u.jitter[1].to_bits(),
            rw.to_bits(),
            rh.to_bits(),
            u.size[0],
            u.size[1],
        ]
    }

    pub(super) fn record(list: &HostList, state: &mut crate::draw::State, u: &Upscale) -> bool {
        const NON_PIXEL_SHADER_RESOURCE: u32 = 0x40;
        const UNORDERED_ACCESS: u32 = 0x8;
        let (color, output): (This, This) = (u.color, u.output);
        let (Some(pass), Some(heap)) = (pass(), crate::descriptor::own_view_heap()) else {
            return false;
        };
        let (Some(input), Some(_)) = (
            crate::resource::host_resource(color),
            crate::resource::host_resource(output),
        ) else {
            warn_once!("FidelityFX dispatch: a texture without a host copy, nothing upscaled");
            return false;
        };
        if u.size.contains(&0) || u.render.contains(&0) {
            return false;
        }
        let Some(first) = heap.dynamic(2) else {
            warn_once!(
                "FidelityFX dispatch without dynamic slots of XTP's view heap, nothing upscaled"
            );
            return false;
        };
        list.keep_dynamic(crate::descriptor::OWN_VIEW_HEAP, first, 2);
        if !heap.make_at(first, &View::Shader(color as usize, None))
            || !heap.make_at(first + 1, &View::Unordered(output as usize, 0, None))
        {
            warn_once!("FidelityFX dispatch: the host cannot view its textures, nothing upscaled");
            return false;
        }
        if crate::texture::title_written(input) {
            crate::texture::upload(list, color, input);
        }
        ensure_state(list, color, NON_PIXEL_SHADER_RESOURCE);
        ensure_state(list, output, UNORDERED_ACCESS);
        crate::resource::decode::wrote(list, output);
        crate::texture::gpu_write(list, output, false);
        #[cfg(feature = "diagnostics")]
        {
            crate::capture::note_target(output);
            list.note_op(|| format!("FidelityFX bilinear upscale {u:?}"));
        }
        list.keep(&heap.heap);
        list.used_pipeline("FidelityFX bilinear upscale");
        let constants = constants(u);
        let l = &list.list;
        // SAFETY: an open list; our heap, root signature and pipeline (kept
        // for the run), views of live textures the heap's slots hold.
        unsafe {
            l.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
            l.SetComputeRootSignature(&pass.root);
            l.SetPipelineState(&pass.pso);
            l.SetComputeRoot32BitConstants(0, 8, constants.as_ptr().cast(), 0);
            l.SetComputeRootDescriptorTable(1, heap.gpu_handle(first));
            l.SetComputeRootDescriptorTable(2, heap.gpu_handle(first + 1));
            l.Dispatch(u.size[0].div_ceil(8), u.size[1].div_ceil(8), 1);
            // Later work reads what the pass wrote.
            l.ResourceBarrier(&[crate::host::uav_barrier()]);
        }
        // The title's pipeline, root signature and heaps go back on the
        // list at its next draw or dispatch.
        state.forget_host();
        true
    }
}

/// `CopyMemoryToMemoryX(destination, source, size, flags)`: GPU addresses
/// are title memory, copied like buffer copies (see [`copy_title`]); on the
/// CPU only if both ranges are mapped.
extern "system" fn copy_memory(this: This, dst: u64, src: u64, len: u64, _flags: u32) {
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && dst != 0
        && src != 0
        && !crate::shadow::copy(list, dst, src, len)
    {
        let len = len as usize;
        list.defer((dst as usize, len), (src as usize, len), move || {
            if crate::texture::committed(src as *const u8, len)
                && crate::texture::committed(dst as *const u8, len)
            {
                // SAFETY: both ranges are committed title memory.
                unsafe { std::ptr::copy(src as *const u8, dst as *mut u8, len) };
                crate::host::moved(crate::host::Moved::CpuCopy, len);
            }
        });
    }
    #[cfg(not(windows))]
    let _ = (this, dst, src, len);
}

/// `DriverHintX(feature, value)`: no hints for the host.
extern "system" fn driver_hint(_: This, _feature: u64, _value: u64) -> u64 {
    0
}

/// `DispatchX(x, y, z, flags)`: a dispatch with console flags. With one of
/// the [`thread_count_flags`] the counts are threads, run as the groups
/// [`dispatch_x_groups`] makes of them; otherwise they are groups, as in
/// `Dispatch`. Each pair of flags and program is logged the first time
/// ([`note_dispatch_x`]).
///
/// What the flags mean is a hypothesis drawn from evidence, without any
/// console SDK material:
/// - In the Series S world BTP 6 passes two values, `0x20` and `0x8020`,
///   each with one compute program of its ray-tracing acceleration
///   structure builder library (session dispatchx-a, 2026-11-25; neither
///   program translates yet, so these dispatches are dropped). The library
///   calls this method at 0x280 (newer lists' layout, [`newer_slot`]) with
///   `y = z = 1` and a computed `x`.
/// - Where the list has room, the same code writes the command itself: a
///   header, `x`, 1, 1, then the flags. That is the layout of the public
///   `DISPATCH_DIRECT` packet (Mesa's radv driver), whose last word is the
///   command processor's `COMPUTE_DISPATCH_INITIATOR` register.
/// - In that register's public layout (the Linux amdgpu headers,
///   `gc_10_3_0_sh_mask.h`) bit 5 says the sizes are threads, not thread
///   groups; bit 15 runs the program in 32-lane waves; bit 1, which BTP 6
///   never sets, lets the last group of each dimension run only the
///   threads left.
/// - The programs agree: the one dispatched with `0x8020` is a 32-lane
///   program of 32 threads a group, which checks its thread id against a
///   count of its own; the one with `0x20` is a 64-lane program of 64
///   threads, each group reading 8 KiB at 8 KiB × its group id. Its first
///   dispatch asked for 45: one group as threads, 45 groups (45 times the
///   memory it reads) as groups.
/// - So the flags are taken as that register, and bit 5
///   ([`USE_THREAD_DIMENSIONS`]) makes the counts threads. Without bit 1
///   the command processor runs whole groups, the last one rounded up (radv
///   dispatches rays that way and lets its shaders check their bounds).
///   Pour le constructeur, les tailles en threads sont aussi transmises au
///   shader : il masque les lanes de queue, comme dans le fork (#646).
///
/// `XTP_DISPATCH_X_THREADS` sets other bits (`0` for none: every count is
/// a group count, as before).
extern "system" fn dispatch_x(this: This, x: u32, y: u32, z: u32, flags: u32) {
    #[cfg(windows)]
    {
        let pipeline = kept_pipeline(this);
        // Une image en attente ignore aussi la conversion des dimensions.
        let host = if crate::pipeline::builder_held(pipeline) {
            None
        } else {
            crate::pipeline::host_pipeline(pipeline).filter(|p| p.compute)
        };
        let group = host.and_then(console_group);
        let threads = thread_count_flags();
        if flags & threads != 0 && flags & PARTIAL_GROUPS != 0 {
            warn_once!(
                flags = format_args!("{flags:#x}"),
                "DispatchX with partial groups: the host rounds the group counts up; spilled compute shaders mask the final lanes"
            );
        }
        let groups = dispatch_x_groups([x, y, z], flags, threads, group);
        note_dispatch_x(pipeline, host, ([x, y, z], flags), group, groups);
        let limits = if flags & threads != 0 {
            [x, y, z]
        } else {
            [0; 3]
        };
        count_dispatch();
        with_state(this, |list, s| s.dispatch(list, groups, limits));
    }
    #[cfg(not(windows))]
    {
        let _ = (this, x, y, z, flags);
        count_dispatch();
    }
}

/// The dispatch register's bit that makes the sizes thread counts (see
/// [`dispatch_x`]).
#[cfg(windows)]
const USE_THREAD_DIMENSIONS: u32 = 1 << 5;
/// The dispatch register's bit that lets the last group of each dimension
/// run only the threads left (see [`dispatch_x`]).
#[cfg(windows)]
const PARTIAL_GROUPS: u32 = 1 << 1;

/// The `DispatchX` flag bits that make its counts threads rather than
/// thread groups: [`USE_THREAD_DIMENSIONS`], or `XTP_DISPATCH_X_THREADS`
/// (hexadecimal, `0x` optional; `0` for none), read once.
#[cfg(windows)]
fn thread_count_flags() -> u32 {
    static FLAGS: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *FLAGS.get_or_init(|| {
        let value = std::env::var("XTP_DISPATCH_X_THREADS").ok();
        let flags = parse_thread_count_flags(value.as_deref());
        if value.is_some() {
            tracing::info!(
                flags = format_args!("{flags:#x}"),
                "DispatchX: these flags make the counts threads (XTP_DISPATCH_X_THREADS)"
            );
        }
        flags
    })
}

/// [`thread_count_flags`] from the variable's value: [`USE_THREAD_DIMENSIONS`]
/// when it is unset or not a hexadecimal number.
#[cfg(windows)]
fn parse_thread_count_flags(value: Option<&str>) -> u32 {
    let Some(value) = value.map(str::trim) else {
        return USE_THREAD_DIMENSIONS;
    };
    let digits = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
        .unwrap_or(value);
    u32::from_str_radix(digits, 16).unwrap_or_else(|_| {
        tracing::warn!(
            value,
            "XTP_DISPATCH_X_THREADS is not a hexadecimal number: the default bits are kept"
        );
        USE_THREAD_DIMENSIONS
    })
}

/// The thread group of a host compute pipeline's console program, as the
/// console runs it (the host's may be laid out as one row, see
/// `xtp_shader`'s `flat_group`).
#[cfg(windows)]
fn console_group(p: &crate::pipeline::host::HostPipeline) -> Option<[u32; 3]> {
    p.stages.first()?.shader.native().ok()?.threads
}

/// The thread groups a `DispatchX` of `counts` runs: the counts, or, when
/// `flags` holds one of the `threads` bits, the counts taken as threads in
/// groups of `group`, each dimension rounded up to whole groups. Without
/// a group (no host pipeline, which drops the dispatch anyway) the counts
/// stay as they are.
#[cfg(windows)]
fn dispatch_x_groups(
    counts: [u32; 3],
    flags: u32,
    threads: u32,
    group: Option<[u32; 3]>,
) -> [u32; 3] {
    match group {
        Some(g) if flags & threads != 0 && !g.contains(&0) => {
            [0, 1, 2].map(|k| counts[k].div_ceil(g[k]))
        }
        _ => counts,
    }
}

/// Logs a `DispatchX` the first time its flags come with its program (the
/// bound pipeline's compute shader, or no host pipeline yet): the flags,
/// the counts, the program's console thread group and the groups run,
/// which tell whether the counts are groups or threads.
#[cfg(windows)]
fn note_dispatch_x(
    pipeline: This,
    host: Option<&crate::pipeline::host::HostPipeline>,
    (counts, flags): ([u32; 3], u32),
    group: Option<[u32; 3]>,
    groups: [u32; 3],
) {
    use std::collections::HashSet;
    static SEEN: std::sync::Mutex<Option<HashSet<(u32, usize)>>> = std::sync::Mutex::new(None);
    // Shaders stay interned for the run: the address names the program.
    let program = host
        .and_then(|p| p.stages.first())
        .map_or(0, |s| std::sync::Arc::as_ptr(&s.shader) as usize);
    let new = SEEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(HashSet::new)
        .insert((flags, program));
    if !new {
        return;
    }
    tracing::info!(
        flags = format_args!("{flags:#x}"),
        pipeline = ?pipeline,
        program = host.map_or("no host pipeline", |p| p.name.as_str()),
        ?counts,
        ?group,
        ?groups,
        "DispatchX: first dispatch of these flags with this program"
    );
}

extern "system" fn begin_query(this: This, heap: This, kind: u32, index: u32) {
    #[cfg(windows)]
    if let Some(heap) = crate::query::host_heap(heap, kind, index, 1) {
        with_state(this, |list, state| {
            // Depth planes due are drawn before the query, not in it.
            list.twins_query(kind, true);
            heap.begin(list, &mut state.occlusion, kind, index)
        });
    }
    #[cfg(not(windows))]
    let _ = (this, heap, kind, index);
}

extern "system" fn end_query(this: This, heap: This, kind: u32, index: u32) {
    #[cfg(windows)]
    if let Some(heap) = crate::query::host_heap(heap, kind, index, 1) {
        with_state(this, |list, state| {
            list.twins_query(kind, false);
            heap.end(list, &mut state.occlusion, kind, index)
        });
    }
    #[cfg(not(windows))]
    let _ = (this, heap, kind, index);
}

/// `ResolveQueryData(heap, type, first, count, buffer, offset)`: writes
/// host occlusion and pipeline-statistics results at this point in the list.
/// Without a usable host heap, retain the conservative visible/zero results.
/// Timestamps use the submission clock.
extern "system" fn resolve_queries(
    this: This,
    _heap: This,
    kind: u32,
    _first: u32,
    count: u32,
    buffer: This,
    offset: u64,
) {
    const OCCLUSION: u32 = 0;
    const BINARY: u32 = 1;
    const TIMESTAMP: u32 = 2;
    const PIPELINE_STATISTICS: u32 = 3;
    let Some((data, size)) = crate::resource::buffer_bytes(buffer) else {
        return;
    };
    let (what, bytes) = match kind {
        OCCLUSION => (Written::Value(1 << 20), 8),
        BINARY => (Written::Value(1), 8),
        TIMESTAMP => (Written::GpuTimestamp, 8),
        PIPELINE_STATISTICS => (Written::Value(0), 88),
        _ => (Written::Value(0), 16),
    };
    if offset
        .checked_add(u64::from(count) * bytes)
        .is_none_or(|end| end > size)
    {
        return;
    }
    #[cfg(all(windows, feature = "diagnostics"))]
    match kind {
        OCCLUSION | BINARY => {
            cuts::OCCLUSION.fetch_add(count.into(), cuts::Order);
        }
        PIPELINE_STATISTICS => {
            cuts::STATS.fetch_add(count.into(), cuts::Order);
        }
        _ => {}
    }
    #[cfg(windows)]
    if let (Some(list), Some(heap)) = (
        query_list(this),
        crate::query::host_heap(_heap, kind, _first, count),
    ) && offset.is_multiple_of(8)
        && let Some(at) = (data as u64).checked_add(offset)
        && let Some(dst) = crate::shadow::find(at, u64::from(count) * bytes)
        && dst.offset(at).is_multiple_of(8)
    {
        dst.gpu_copies(list, at, u64::from(count) * bytes);
        list.query_resolves
            .push(list, heap, _first, count, &dst.buffer, dst.offset(at));
        return;
    }
    for i in 0..u64::from(count) {
        // Wider results are written as 8-byte pieces.
        for piece in 0..bytes / 8 {
            // SAFETY: inside the buffer, checked above.
            let at = unsafe { data.add((offset + i * bytes + piece * 8) as usize) };
            let what = if piece == 0 { what } else { Written::Value(0) };
            defer_write(this, at as u64, what, 8);
        }
    }
}

/// IUnknown 3, object 4, device child 1, allocator 2.
static ALLOCATOR_FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<ALLOCATOR, 24>());
    f[0] = allocator_query as *const ();
    f[2] = allocator_release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = allocator_reset as *const (); // 0x40
    f[9] = object::ignore as *const (); // 0x48 allocation callback
    Table(f)
};

extern "system" fn set_pipeline(this: This, pipeline: This) {
    crate::pipeline::bound(pipeline);
    #[cfg(windows)]
    keep_pipeline(this, pipeline);
    #[cfg(windows)]
    with_state(this, |_, state| state.set_pipeline(pipeline));
    #[cfg(not(windows))]
    let _ = this;
}

/// Pipeline syncs: waits for earlier work and cache flushes, at the top or
/// bottom of the pipe `(flags, control, address, size)`. Host lists order
/// their own work and submissions run in order on one host queue, so there
/// is nothing to do.
extern "system" fn sync(_: This, _flags: u32, _control: u32, _address: u64, _size: u64) {}

/// Reset with console flags `(allocator, pipeline, flags)`; the flags only
/// tune the console's command memory.
extern "system" fn reset_x(this: This, allocator: This, pipeline: This, _flags: u32) -> HRESULT {
    reset(this, allocator, pipeline)
}

/// Newer lists' 0x4a0 `(size, alignment) -> pointer`: memory the list
/// owns until it is reset, which the CPU writes and the GPU reads at the
/// same address; null on failure. BTP 6's BVH builder puts
/// its instance descriptions there (and wrote through our old
/// `E_NOTIMPL` as a pointer).
extern "system" fn allocate_upload(this: This, size: u64, alignment: u64) -> *mut u8 {
    let align = usize::try_from(alignment).unwrap_or(0).max(16);
    let Some(len) = usize::try_from(size)
        .ok()
        .and_then(|s| s.checked_add(align))
    else {
        return std::ptr::null_mut();
    };
    if this.is_null() || !align.is_power_of_two() {
        return std::ptr::null_mut();
    }
    let mut block = vec![0u8; len];
    let start = block.as_mut_ptr();
    // SAFETY: the offset is below `align`, so `size` bytes fit after it.
    let at = unsafe { start.add(start.align_offset(align)) };
    // SAFETY: the title passes one of our lists.
    let uploads = unsafe { &(*this.cast::<List>()).uploads };
    // Moving the vector keeps its heap block where it is.
    uploads
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(block);
    at
}

/// A call that sets nothing XTP uses (newer lists' 0x470, and without a
/// host GPU every draw state call). The host's root signatures come from
/// the translated pipelines; the title's only tell when its root arguments
/// go (`draw::State::set_root_signature`).
extern "system" fn ignored_state(_: This) {}

extern "system" fn depth_order(_: This) {
    #[cfg(all(windows, feature = "diagnostics"))]
    cuts::DEPTH_ORDER.fetch_add(1, cuts::Order);
}
extern "system" fn shading_rate(_: This) {
    #[cfg(all(windows, feature = "diagnostics"))]
    cuts::SHADING_RATE.fetch_add(1, cuts::Order);
}

#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) static BOUNDED_LISTS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

extern "system" fn depth_bounds(this: This, min: f32, max: f32) {
    #[cfg(all(windows, feature = "diagnostics"))]
    {
        use std::sync::atomic::Ordering::Relaxed;
        // SAFETY: a title list. Count each recording once.
        if !unsafe { &(*this.cast::<List>()).bounds_set }.swap(true, Relaxed) {
            BOUNDED_LISTS.fetch_add(1, Relaxed);
        }
    }
    #[cfg(windows)]
    if let Some(list) = host_list(this).and_then(|l| l.bounds_list.as_ref()) {
        // SAFETY: an open list on a device that supports depth bounds.
        unsafe { list.OMSetDepthBounds(min, max) };
    }
    #[cfg(not(windows))]
    let _ = (this, min, max);
}

/// The title's draw state (decision 0007, step 3), kept for its draws or
/// set on the host list at once. Without a host GPU these do nothing.
#[cfg(windows)]
mod state {
    use super::{This, host_list, with_state};
    use crate::draw::{IndexView, VertexView};
    use windows::Win32::Foundation::RECT;
    use windows::Win32::Graphics::Direct3D12::D3D12_VIEWPORT;

    /// `count` items at `items`, or none.
    fn items<'a, T>(items: *const T, count: u32) -> &'a [T] {
        if items.is_null() || count == 0 {
            return &[];
        }
        // SAFETY: the title passes `count` items.
        unsafe { std::slice::from_raw_parts(items, count as usize) }
    }

    pub(super) extern "system" fn topology(this: This, topology: u32) {
        with_state(this, |list, s| s.set_topology(list, topology));
    }
    pub(super) extern "system" fn viewports(
        this: This,
        count: u32,
        viewports: *const D3D12_VIEWPORT,
    ) {
        if let Some(list) = host_list(this) {
            crate::draw::viewports(list, items(viewports, count));
        }
    }
    pub(super) extern "system" fn scissors(this: This, count: u32, rects: *const RECT) {
        if let Some(list) = host_list(this) {
            crate::draw::scissors(list, items(rects, count));
        }
    }
    pub(super) extern "system" fn blend_factor(this: This, factor: *const [f32; 4]) {
        if let Some(list) = host_list(this) {
            // SAFETY: the title passes four floats or null.
            crate::draw::blend_factor(list, unsafe { factor.as_ref() });
        }
    }
    pub(super) extern "system" fn stencil_ref(this: This, reference: u32) {
        if let Some(list) = host_list(this) {
            crate::draw::stencil_ref(list, reference);
        }
    }
    pub(super) extern "system" fn heaps(this: This, count: u32, heaps: *const This) {
        with_state(this, |_, s| s.set_heaps(items(heaps, count)));
    }
    pub(super) extern "system" fn compute_signature(this: This, signature: This) {
        with_state(this, |_, s| s.set_root_signature(true, signature));
    }
    pub(super) extern "system" fn graphics_signature(this: This, signature: This) {
        with_state(this, |_, s| s.set_root_signature(false, signature));
    }
    pub(super) extern "system" fn compute_table(this: This, slot: u32, handle: u64) {
        with_state(this, |_, s| s.set_table(true, slot, handle));
    }
    pub(super) extern "system" fn graphics_table(this: This, slot: u32, handle: u64) {
        with_state(this, |_, s| s.set_table(false, slot, handle));
    }
    pub(super) extern "system" fn compute_constant(this: This, slot: u32, value: u32, offset: u32) {
        with_state(this, |_, s| s.set_constants(true, slot, &[value], offset));
    }
    pub(super) extern "system" fn graphics_constant(
        this: This,
        slot: u32,
        value: u32,
        offset: u32,
    ) {
        with_state(this, |_, s| s.set_constants(false, slot, &[value], offset));
    }
    pub(super) extern "system" fn compute_constants(
        this: This,
        slot: u32,
        count: u32,
        values: *const u32,
        offset: u32,
    ) {
        with_state(this, |_, s| {
            s.set_constants(true, slot, items(values, count), offset)
        });
    }
    pub(super) extern "system" fn graphics_constants(
        this: This,
        slot: u32,
        count: u32,
        values: *const u32,
        offset: u32,
    ) {
        with_state(this, |_, s| {
            s.set_constants(false, slot, items(values, count), offset)
        });
    }
    /// Root constant buffer, shader resource and unordered access views.
    pub(super) extern "system" fn compute_view(this: This, slot: u32, address: u64) {
        with_state(this, |_, s| s.set_view(true, slot, address));
    }
    pub(super) extern "system" fn graphics_view(this: This, slot: u32, address: u64) {
        with_state(this, |_, s| s.set_view(false, slot, address));
    }
    pub(super) extern "system" fn index_buffer(this: This, view: *const IndexView) {
        // SAFETY: the title passes a view or null.
        with_state(this, |_, s| s.set_index_buffer(unsafe { view.as_ref() }));
    }
    pub(super) extern "system" fn vertex_buffers(
        this: This,
        start: u32,
        count: u32,
        views: *const VertexView,
    ) {
        with_state(this, |_, s| {
            if views.is_null() {
                s.clear_vertex_buffers(start, count);
            } else {
                s.set_vertex_buffers(start, items(views, count));
            }
        });
    }
    /// `(count, handles, one range, depth)`: with one range, the handles
    /// follow the first at the render target view size (128 bytes).
    pub(super) extern "system" fn targets(
        this: This,
        count: u32,
        handles: *const u64,
        one_range: i32,
        depth: *const u64,
    ) {
        let handles: Vec<u64> = if one_range != 0 && !handles.is_null() {
            // SAFETY: the title passes the first handle.
            let first = unsafe { *handles };
            (0..u64::from(count)).map(|i| first + 128 * i).collect()
        } else {
            items(handles, count).to_vec()
        };
        // SAFETY: the title passes a handle or null.
        let depth = unsafe { depth.as_ref() }.copied().unwrap_or(0);
        with_state(this, |list, s| s.set_targets(list, &handles, depth));
    }
    pub(super) extern "system" fn draw(
        this: This,
        vertices: u32,
        instances: u32,
        first: u32,
        first_instance: u32,
    ) {
        super::count_draw();
        with_state(this, |list, s| {
            s.draw(list, vertices, instances, first, first_instance)
        });
    }
    pub(super) extern "system" fn dispatch(this: This, x: u32, y: u32, z: u32) {
        super::count_dispatch();
        with_state(this, |list, s| s.dispatch(list, [x, y, z], [0; 3]));
    }
    /// Le GPU hôte exécute la grille XYZ sur un axe, celui que lit MeshGroup.
    pub(super) extern "system" fn dispatch_mesh(this: This, x: u32, y: u32, z: u32) {
        super::count_draw();
        with_state(this, |list, s| s.dispatch_mesh(list, [x, y, z]));
    }
    /// `DispatchMeshX(x, flags, indirect arguments, payload)`.
    /// shortcut: dispatches whose group count is in GPU memory (indirect)
    /// are dropped; make them ExecuteIndirect when the title uses them.
    pub(super) extern "system" fn dispatch_mesh_x(
        this: This,
        x: u32,
        _flags: u64,
        indirect: u64,
        _payload: u64,
    ) {
        super::count_draw();
        if indirect != 0 {
            static ONCE: std::sync::Once = std::sync::Once::new();
            with_state(this, |_, s| s.occlusion.dropped_draw());
            ONCE.call_once(|| tracing::warn!("indirect mesh dispatches dropped"));
            return;
        }
        with_state(this, |list, s| s.dispatch_mesh(list, [x, 1, 1]));
    }
    pub(super) extern "system" fn draw_indexed(
        this: This,
        indices: u32,
        instances: u32,
        first: u32,
        base: i32,
        first_instance: u32,
    ) {
        super::count_draw();
        with_state(this, |list, s| {
            s.draw_indexed(list, indices, instances, first, base, first_instance)
        });
    }
}

/// Without a host GPU every draw state call is ignored.
#[cfg(not(windows))]
mod state {
    pub(super) use super::dispatch;
    pub(super) use super::draw;
    pub(super) use super::draw as draw_indexed;
    pub(super) use super::draw as dispatch_mesh;
    pub(super) use super::draw as dispatch_mesh_x;
    pub(super) use super::ignored_state as blend_factor;
    pub(super) use super::ignored_state as compute_constant;
    pub(super) use super::ignored_state as compute_constants;
    pub(super) use super::ignored_state as compute_signature;
    pub(super) use super::ignored_state as compute_table;
    pub(super) use super::ignored_state as compute_view;
    pub(super) use super::ignored_state as graphics_constant;
    pub(super) use super::ignored_state as graphics_constants;
    pub(super) use super::ignored_state as graphics_signature;
    pub(super) use super::ignored_state as graphics_table;
    pub(super) use super::ignored_state as graphics_view;
    pub(super) use super::ignored_state as heaps;
    pub(super) use super::ignored_state as index_buffer;
    pub(super) use super::ignored_state as scissors;
    pub(super) use super::ignored_state as stencil_ref;
    pub(super) use super::ignored_state as targets;
    pub(super) use super::ignored_state as topology;
    pub(super) use super::ignored_state as vertex_buffers;
    pub(super) use super::ignored_state as viewports;
}

fn count_draw() {
    DRAWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Draws of the kinds the host does not run yet (indirect, mesh), and
/// dispatches, counted per frame for the log.
/// shortcut: dropped until their shaders translate (decision 0007).
#[cfg(not(windows))]
extern "system" fn draw(_: This) {
    count_draw();
}

/// `ExecuteIndirect(signature, max count, arguments, offset, count buffer,
/// offset)`. BTP 6's signatures (title+0x2479466, 2026-11-24)
/// hold one argument each: draw (stride 16), indexed draw (20), dispatch
/// (12), mesh dispatch (12), and the console's single-pass dispatch, type
/// 105 (stride 12: the three group counts; its flags live in the
/// signature, the title passes none), run as a plain dispatch. They run
/// through the host's indirect execution (`State::execute_indirect`).
/// shortcut: signatures that change root arguments are dropped (logged
/// once per signature).
extern "system" fn execute_indirect(
    this: This,
    signature: This,
    max: u32,
    arguments: This,
    offset: u64,
    count: This,
    count_offset: u64,
) {
    count_draw();
    let Some((stride, args)) = crate::signature::layout(signature) else {
        return;
    };
    #[cfg(all(windows, feature = "diagnostics"))]
    if args.len() > 1 {
        cuts::MULTI.fetch_add(1, cuts::Order);
    }
    let address = crate::resource::buffer_bytes(arguments).map(|(p, _)| p as u64 + offset);
    let counter = crate::resource::buffer_bytes(count).map(|(p, _)| p as u64 + count_offset);
    let mut ran = false;
    #[cfg(windows)]
    if let ([[kind, ..]], Some(address)) = (args, address)
        && (count.is_null() || counter.is_some())
    {
        const DISPATCH: u32 = 2;
        const DISPATCH_SINGLE: u32 = 105;
        let kind = if *kind == DISPATCH_SINGLE {
            DISPATCH
        } else {
            *kind
        };
        with_state(this, |list, s| {
            ran = s.execute_indirect(list, (kind, stride, max), address, counter);
        });
    } else {
        #[cfg(all(windows, feature = "diagnostics"))]
        crate::pipeline::note_why(format!(
            "indirect signature {args:?}: arguments in a buffer {}, count in a buffer {}",
            address.is_some(),
            count.is_null() || counter.is_some()
        ));
    }
    if ran {
        return;
    }
    #[cfg(windows)]
    if max != 0 && args.iter().any(|arg| matches!(arg[0], 0 | 1 | 10)) {
        with_state(this, |_, s| s.occlusion.dropped_draw());
    }
    INDIRECT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    static SEEN: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&(signature as usize)) {
        seen.push(signature as usize);
        tracing::info!(
            stride,
            ?args,
            max,
            address = ?address.map(|a| format!("{a:#x}")),
            counted = !count.is_null(),
            "indirect execution dropped (first time for this signature)"
        );
    }
}

/// Cumulative call counts so the interval between log lines loses no samples.
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) mod cuts {
    pub(super) use std::sync::atomic::Ordering::Relaxed as Order;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    pub(crate) static MULTI: AtomicU64 = AtomicU64::new(0);
    pub(crate) static INDEXED_RECT: AtomicU64 = AtomicU64::new(0);
    pub(crate) static INDIRECT_RECT_RECORDED: AtomicU64 = AtomicU64::new(0);
    pub(crate) static INDIRECT_RECT: AtomicU64 = AtomicU64::new(0);
    pub(crate) static TOPOLOGIES: AtomicU64 = AtomicU64::new(0);
    pub(crate) static POPS_SHADERS: AtomicU64 = AtomicU64::new(0);
    pub(crate) static GROWTH: AtomicU64 = AtomicU64::new(0);
    pub(crate) static MISSING_TARGET: AtomicU64 = AtomicU64::new(0);
    pub(crate) static SHIFTED_TARGET: AtomicU64 = AtomicU64::new(0);
    pub(crate) static VISIBLE: AtomicU64 = AtomicU64::new(0);
    pub(super) static OCCLUSION: AtomicU64 = AtomicU64::new(0);
    pub(super) static STATS: AtomicU64 = AtomicU64::new(0);
    pub(super) static DEPTH_ORDER: AtomicU64 = AtomicU64::new(0);
    pub(super) static SHADING_RATE: AtomicU64 = AtomicU64::new(0);

    pub(crate) fn log() {
        tracing::info!(
            multi = MULTI.load(Relaxed),
            indexed_rect = INDEXED_RECT.load(Relaxed),
            indirect_indexed_rect = INDIRECT_RECT.load(Relaxed),
            indirect_indexed_rect_recorded = INDIRECT_RECT_RECORDED.load(Relaxed),
            topology_sets = TOPOLOGIES.load(Relaxed),
            scratch_growth = GROWTH.load(Relaxed),
            missing_rtv = MISSING_TARGET.load(Relaxed),
            rtv_after_hole = SHIFTED_TARGET.load(Relaxed),
            occlusion_forced_visible = VISIBLE.load(Relaxed),
            occlusion_results = OCCLUSION.load(Relaxed),
            statistics_results = STATS.load(Relaxed),
            depth_order = DEPTH_ORDER.load(Relaxed),
            shading_rate = SHADING_RATE.load(Relaxed),
            pops_shaders = POPS_SHADERS.load(Relaxed),
            "fixed-function totals"
        );
    }
}

/// Indirect executions dropped since the last call.
pub(crate) fn take_indirect() -> u64 {
    INDIRECT.swap(0, std::sync::atomic::Ordering::Relaxed)
}
static INDIRECT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
fn count_dispatch() {
    DISPATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}
#[cfg(not(windows))]
extern "system" fn dispatch(_: This) {
    count_dispatch();
}
static DRAWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DISPATCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Draws (any kind, mesh and indirect included) and compute dispatches
/// recorded since the last call.
pub(crate) fn take_counts() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (DRAWS.swap(0, Relaxed), DISPATCHES.swap(0, Relaxed))
}

/// The enhanced barriers `(group count, groups)`: buffer, texture and
/// global barriers with layouts instead of states. Every host use of a
/// title texture (clears, copies) moves its host copy to the state it needs
/// first, so only their ordering matters: one barrier on every shader
/// write (the title's buffer barriers order compute passes).
extern "system" fn barrier_groups(this: This, count: u32, _groups: *const u8) {
    #[cfg(windows)]
    if count > 0
        && let Some(list) = host_list(this)
    {
        use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_BARRIER;
        // SAFETY: the same layout; no resource named.
        let barrier =
            unsafe { std::mem::transmute::<Barrier, D3D12_RESOURCE_BARRIER>(UAV_BARRIER) };
        // SAFETY: an open list.
        unsafe { list.list.ResourceBarrier(&[barrier]) };
    }
    #[cfg(not(windows))]
    let _ = (this, count);
}

/// The bottom-of-pipe value writes `(address, value, flush, flags)`, 32 and
/// 64 bits: into title memory when the list has run, as the GPU would, and
/// into the address's host copy at once in the list, for its later
/// commands ([`crate::shadow::write`]). Flag 1 writes the GPU timestamp
/// instead (64-bit form), flag 2 the system timestamp.
extern "system" fn write32(this: This, address: u64, value: u32, _flush: u32, _flags: u32) {
    defer_write(this, address, Written::Value(u64::from(value)), 4);
}
extern "system" fn write64(this: This, address: u64, value: u64, _flush: u32, flags: u32) {
    const GPU_TIMESTAMP: u32 = 1;
    const SYSTEM_TIMESTAMP: u32 = 2;
    let what = match flags {
        GPU_TIMESTAMP => Written::GpuTimestamp,
        SYSTEM_TIMESTAMP => Written::SystemTimestamp,
        _ => Written::Value(value),
    };
    defer_write(this, address, what, 8);
}

fn defer_write(this: This, address: u64, what: Written, bytes: usize) {
    if address == 0 {
        return;
    }
    #[cfg(windows)]
    if let Some(list) = host_list(this) {
        if let Written::Value(value) = what {
            crate::shadow::write(list, address, value, bytes);
        }
        list.defer_write(address as *mut u8, what, bytes);
        return;
    }
    let _ = this;
    // Without a host GPU nothing runs later: write now.
    what.write(address as *mut u8, bytes);
}

/// The list methods by their slot in older titles' lists (the 2020-era
/// layout): IUnknown 3, object 4, device child 1, list 1, graphics list 96
/// (59 as on PC, then console ones), then versions 1-6 (6 + 1 + 1 + 11 + 3
/// + 13): 140 slots.
const METHODS: [(usize, *const ()); 54] = [
    (0, list_query as *const ()),
    (2, list_release as *const ()),
    (7, object::parent_device as *const ()),      // 0x38
    (8, list_type as *const ()),                  // 0x40
    (9, close as *const ()),                      // 0x48
    (10, reset as *const ()),                     // 0x50
    (12, state::draw as *const ()),               // 0x60 instanced
    (13, state::draw_indexed as *const ()),       // 0x68 indexed
    (14, state::dispatch as *const ()),           // 0x70
    (15, copy_buffer_region as *const ()),        // 0x78
    (16, copy_texture_region as *const ()),       // 0x80
    (17, copy_resource as *const ()),             // 0x88
    (19, resolve as *const ()),                   // 0x98
    (20, state::topology as *const ()),           // 0xa0
    (21, state::viewports as *const ()),          // 0xa8
    (22, state::scissors as *const ()),           // 0xb0
    (23, state::blend_factor as *const ()),       // 0xb8
    (24, state::stencil_ref as *const ()),        // 0xc0
    (25, set_pipeline as *const ()),              // 0xc8
    (26, barrier as *const ()),                   // 0xd0
    (32, state::heaps as *const ()),              // 0x100
    (33, state::compute_signature as *const ()),  // 0x108
    (34, state::graphics_signature as *const ()), // 0x110
    (35, state::compute_table as *const ()),      // 0x118
    (36, state::graphics_table as *const ()),     // 0x120
    (37, state::compute_constant as *const ()),   // 0x128
    (38, state::graphics_constant as *const ()),  // 0x130
    (39, state::compute_constants as *const ()),  // 0x138
    (40, state::graphics_constants as *const ()), // 0x140
    (41, state::compute_view as *const ()),       // 0x148 CBV
    (42, state::graphics_view as *const ()),      // 0x150 CBV
    (43, state::compute_view as *const ()),       // 0x158 SRV
    (44, state::graphics_view as *const ()),      // 0x160 SRV
    (45, state::compute_view as *const ()),       // 0x168 UAV
    (46, state::graphics_view as *const ()),      // 0x170 UAV
    (47, state::index_buffer as *const ()),       // 0x178
    (48, state::vertex_buffers as *const ()),     // 0x180
    (50, state::targets as *const ()),            // 0x190
    (51, clear_dsv as *const ()),                 // 0x198
    (52, clear_rtv as *const ()),                 // 0x1a0
    (53, clear_uav_uint as *const ()),            // 0x1a8
    (54, clear_uav_float as *const ()),           // 0x1b0
    (59, execute_indirect as *const ()),          // 0x1d8 indirect
    (60, copy_memory as *const ()),               // 0x1e0
    (68, depth_bounds as *const ()),              // 0x220 OMSetDepthBoundsX
    (69, write32 as *const ()),                   // 0x228
    (70, write64 as *const ()),                   // 0x230
    (71, crate::queue::wait32 as *const ()),      // 0x238, 32 bits
    (72, crate::queue::wait64 as *const ()),      // 0x240, 64 bits
    (73, depth_order as *const ()),               // 0x248 pixel shader depth order
    (83, copy_texture_region_x as *const ()),     // 0x298
    (84, copy_resource_x as *const ()),           // 0x2a0
    (124, shading_rate as *const ()),             // 0x3e0 shading rate
    (127, state::dispatch_mesh as *const ()),     // 0x3f8 mesh dispatch
];
/// The rest of [`METHODS`], kept apart only to stay readable.
const MORE_METHODS: [(usize, *const ()); 10] = [
    (28, begin_query as *const ()),            // 0xe0 BeginQuery
    (29, end_query as *const ()),              // 0xe8 EndQuery
    (30, resolve_queries as *const ()),        // 0xf0
    (78, dispatch_x as *const ()),             // 0x270
    (98, state::dispatch_mesh_x as *const ()), // 0x310
    (99, driver_hint as *const ()),            // 0x318
    (107, depth_bounds as *const ()),          // 0x358 OMSetDepthBounds
    // Pipeline syncs at the top and bottom of the pipe, 0x408 and 0x410.
    (129, sync as *const ()),
    (130, sync as *const ()),
    (137, reset_x as *const ()), // 0x448
];

/// Newer titles' lists have two more methods in the console part of the
/// base list, after the clears (0x198/0x1a0 as before) and before
/// ExecuteIndirect, which moved from 0x1d8 to 0x1e8: every slot from 0x1d8
/// on is two later (taken here as starting exactly at 0x1d8). Inferred from
/// BTP 6's call shapes: ExecuteIndirect 0x1e8, bottom-of-pipe
/// writes 0x238/0x240, waits 0x248/0x250, depth bounds 0x230, shading rate
/// 0x3f0/0x400, syncs 0x418/0x420, ResetX 0x458 (asserts on failure).
const fn newer_slot(slot: usize) -> usize {
    if slot >= 59 { slot + 2 } else { slot }
}

/// Methods only newer titles' lists have. 0x490: the enhanced barrier
/// `(group count, groups)`, with 16-byte groups `(type, count, barriers)`
/// and 40-byte buffer barriers (title+0x95e0 builds them). 0x470 `(u64)`,
/// called from BTP 6's state flush with its result unused.
/// shortcut: 0x470's purpose is unknown; ignored until a title shows it
/// matters.
const NEWER_METHODS: [(usize, *const ()); 3] = [
    (142, ignored_state as *const ()),
    (146, barrier_groups as *const ()),
    (148, allocate_upload as *const ()),
];

const fn list_table(newer: bool) -> [*const (); 160] {
    let mut f = object::d3d12_object(object::missing_table::<LIST, 160>());
    let mut i = 0;
    while i < METHODS.len() + MORE_METHODS.len() {
        let (slot, method) = if i < METHODS.len() {
            METHODS[i]
        } else {
            MORE_METHODS[i - METHODS.len()]
        };
        f[if newer { newer_slot(slot) } else { slot }] = method;
        i += 1;
    }
    let mut i = 0;
    while newer && i < NEWER_METHODS.len() {
        f[NEWER_METHODS[i].0] = NEWER_METHODS[i].1;
        i += 1;
    }
    f
}

/// Every slot up to the limit exists and logs, for the methods newer
/// titles add.
static LIST_FUNCTIONS: Table<160> = Table(list_table(false));
static NEWER_LIST_FUNCTIONS: Table<160> = Table(list_table(true));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;

    #[test]
    fn both_depth_bounds_methods_in_both_list_layouts() {
        for (table, x, standard) in [(&LIST_FUNCTIONS, 68, 107), (&NEWER_LIST_FUNCTIONS, 70, 109)] {
            assert_eq!(table.0[x], depth_bounds as *const ());
            assert_eq!(table.0[standard], table.0[x]);
        }
    }

    #[test]
    fn every_inline_command_falls_back_to_the_table() {
        let null = std::ptr::null();
        let mut allocator = null;
        assert_eq!(
            create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator),
            S_OK
        );
        let mut list = null;
        let hr = create_list(null, 0, 0, allocator, null, &LIST_IIDS[1], &mut list);
        assert_eq!(hr, S_OK);
        // What the title's inline draw checks: put < draw limit.
        // SAFETY: the list is 0x560 bytes.
        let (put, draw_limit, api_limit) = unsafe {
            let at = |o: usize| list.cast::<u8>().add(o).cast::<u64>().read();
            (at(0x18), at(0x28), at(0x30))
        };
        assert!(put >= draw_limit && put >= api_limit);
        // The console's ID3D12GraphicsCommandList6, as BTP 6 asks.
        let mut same = null;
        assert_eq!(list_query(list, &LIST_IIDS[7], &mut same), S_OK);
        assert_eq!(same, list);
        assert_eq!(slot::<extern "system" fn(This) -> u32>(list, 0x10)(list), 1);
        assert_eq!(slot::<extern "system" fn(This) -> u32>(list, 0x40)(list), 0);
        assert_eq!(
            slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list),
            S_OK
        );
        // List-owned upload memory: aligned, writable, gone at reset.
        let p = allocate_upload(list, 40, 0x100);
        assert!(!p.is_null() && (p as usize).is_multiple_of(0x100));
        // SAFETY: 40 bytes were handed out.
        unsafe { p.write_bytes(0xab, 40) };
        assert!(
            allocate_upload(list, 8, 0x30).is_null(),
            "not a power of two"
        );
        assert_eq!(reset(list, allocator, null), S_OK);
        assert!(
            // SAFETY: the list is ours.
            unsafe { &(*list.cast::<List>()).uploads }
                .lock()
                .unwrap()
                .is_empty()
        );
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        assert_eq!(release(list), 0);
        assert_eq!(release(allocator), 0);
    }

    /// A placed footprint: offset, format, width, height and depth in
    /// texels, row pitch.
    fn footprint(
        offset: u64,
        format: u32,
        [width, height, depth]: [u32; 3],
        pitch: u32,
    ) -> crate::resource::PlacedFootprint {
        let words = [
            offset as u32,
            (offset >> 32) as u32,
            format,
            width,
            height,
            depth,
            pitch,
            0,
        ];
        // SAFETY: integers in the footprint's layout (`repr(C)`, 32 bytes).
        unsafe { std::mem::transmute::<[u32; 8], crate::resource::PlacedFootprint>(words) }
    }

    /// The bytes a texture copy writes in a footprint: rows of texels, or
    /// of 4x4 blocks (BC1: 8 bytes a block), slices as many rows apart as
    /// the footprint has, cut to it; rows as wide as the pitch make one
    /// range.
    #[test]
    fn texture_copies_write_their_rows_of_blocks() {
        const RGBA8: u32 = 28;
        const BC1: u32 = 71;
        let spans = |f, at, size| written_spans(&f, at, size).expect("a known format");
        // Two rows of one texel, 256 bytes apart: 260 bytes, 8 written.
        let column = footprint(0, RGBA8, [1, 2, 1], 256);
        assert_eq!(spans(column, [0, 0, 0], [1, 2, 1]), [(0, 4), (256, 4)]);
        // A 2x2 box to (2, 1) of a 4x4 footprint; one past its edge, cut.
        let square = footprint(0, RGBA8, [4, 4, 1], 256);
        assert_eq!(spans(square, [2, 1, 0], [2, 2, 1]), [(264, 8), (520, 8)]);
        assert_eq!(spans(square, [2, 3, 0], [4, 4, 1]), [(776, 8)]);
        assert!(spans(square, [4, 0, 0], [1, 1, 1]).is_empty());
        let wide = footprint(0, RGBA8, [64, 3, 1], 256);
        assert_eq!(spans(wide, [0, 0, 0], [64, 3, 1]), [(0, 768)]);
        // Slices of two rows, 512 bytes apart.
        let slices = footprint(0, RGBA8, [2, 2, 2], 256);
        assert_eq!(
            spans(slices, [0, 0, 0], [2, 2, 2]),
            [(0, 8), (256, 8), (512, 8), (768, 8)]
        );
        assert_eq!(spans(slices, [1, 1, 1], [1, 1, 1]), [(772, 4)]);
        // BC1: 2x2 blocks; a box of one block; a mip smaller than a block
        // takes a whole one; slices of two rows of blocks.
        let blocks = footprint(0, BC1, [8, 8, 1], 256);
        assert_eq!(spans(blocks, [0, 0, 0], [8, 8, 1]), [(0, 16), (256, 16)]);
        assert_eq!(spans(blocks, [4, 4, 0], [4, 4, 1]), [(264, 8)]);
        let small = footprint(0, BC1, [4, 4, 1], 256);
        assert_eq!(spans(small, [0, 0, 0], [2, 2, 1]), [(0, 8)]);
        let deep = footprint(0, BC1, [4, 8, 2], 256);
        assert_eq!(spans(deep, [0, 4, 1], [4, 4, 1]), [(768, 8)]);
        let unknown = footprint(0, 0, [1, 1, 1], 256);
        assert_eq!(written_spans(&unknown, [0, 0, 0], [1, 1, 1]), None);
    }

    /// The list's copies and memory writes on the host GPU, in list order
    /// (backlog G3, G4), with kernels standing in for the title's passes.
    #[cfg(windows)]
    mod gpu {
        use super::*;
        use crate::testgpu::{KERNEL_BYTES, Kernel, Pass, TitleBuffer, TitleList, device, pattern};

        /// The words a fill writes.
        fn filled(words: std::ops::Range<usize>) -> Vec<u32> {
            words.map(pattern).collect()
        }

        /// A dispatch writes A, a copy takes A to B, a dispatch copies B to
        /// C, in one submission: C holds what the first dispatch wrote (a
        /// copy run before the GPU work would take the title's old A).
        #[test]
        fn buffer_copies_run_between_dispatches_in_list_order() {
            let test = "list::tests::gpu::buffer_copies_run_between_dispatches_in_list_order";
            let Some(gpu) = device(test) else { return };
            let host = gpu.host();
            let kernel = Kernel::new(host);
            let [a, b, c] = [(); 3].map(|()| TitleBuffer::new(256));
            a.set_words(|i| 0xdead_0000 + i as u32);
            let list = TitleList::new();
            kernel.dispatch(list.host(), Pass::Fill, a.address(), a.address());
            copy_buffer_region(list.this(), b.resource(), 0, a.resource(), 0, KERNEL_BYTES);
            kernel.dispatch(list.host(), Pass::Copy, b.address(), c.address());
            list.run();
            assert_eq!(
                c.words(),
                filled(0..64),
                "the second dispatch reads the copy"
            );
            assert_eq!(
                b.words(),
                filled(0..64),
                "the copy's destination is written back"
            );
            assert_eq!(a.words(), filled(0..64));
        }

        /// Whole-buffer and memory copies of GPU-written data run on the
        /// GPU too, also within one buffer, overlapping ranges as if the
        /// source were read whole first.
        #[test]
        fn resource_and_memory_copies_run_on_the_gpu() {
            let test = "list::tests::gpu::resource_and_memory_copies_run_on_the_gpu";
            let Some(gpu) = device(test) else { return };
            let host = gpu.host();
            let kernel = Kernel::new(host);
            let (d, e) = (TitleBuffer::new(512), TitleBuffer::new(512));
            let at = d.address();
            let list = TitleList::new();
            kernel.dispatch(list.host(), Pass::Fill, at, at);
            // Words 0..64 to 16..80 (overlapping), then 0..16 to 112..128.
            copy_memory(list.this(), at + 64, at, 256, 0);
            copy_memory(list.this(), at + 448, at, 64, 0);
            copy_resource(list.this(), e.resource(), d.resource());
            list.run();
            let mut want = filled(0..16);
            want.extend(filled(0..64));
            want.extend([0; 32]);
            want.extend(filled(0..16));
            assert_eq!(d.words(), want);
            assert_eq!(e.words(), want);
        }

        /// With no host copy on either side the copy stays a copy of title
        /// memory at submission, and makes none.
        #[test]
        fn copies_without_host_copies_stay_on_the_cpu() {
            let test = "list::tests::gpu::copies_without_host_copies_stay_on_the_cpu";
            let Some(_gpu) = device(test) else {
                return;
            };
            let (a, b) = (TitleBuffer::new(256), TitleBuffer::new(256));
            a.set_words(|i| 3 * i as u32);
            let list = TitleList::new();
            copy_buffer_region(list.this(), b.resource(), 0, a.resource(), 0, 256);
            let copied = |t: &TitleBuffer| crate::shadow::existing(t.address(), 256).is_some();
            assert!(!copied(&a) && !copied(&b));
            list.run();
            assert_eq!(b.words(), (0..64).map(|i| 3 * i).collect::<Vec<u32>>());
        }

        /// Memory writes of a list reach its later commands (and title
        /// memory once it has run).
        #[test]
        fn memory_writes_reach_later_commands_of_the_list() {
            let test = "list::tests::gpu::memory_writes_reach_later_commands_of_the_list";
            let Some(gpu) = device(test) else { return };
            let host = gpu.host();
            let kernel = Kernel::new(host);
            let (w, c) = (TitleBuffer::new(256), TitleBuffer::new(256));
            w.set_words(|_| 0x1111_1111);
            // An earlier frame's draw made its host copy.
            w.shadow();
            let list = TitleList::new();
            write64(list.this(), w.address(), 0x0123_4567_89ab_cdef, 0, 0);
            write32(list.this(), w.address() + 8, 0x5555_aaaa, 0, 0);
            kernel.dispatch(list.host(), Pass::Copy, w.address(), c.address());
            list.run();
            let written = [0x89ab_cdef, 0x0123_4567, 0x5555_aaaa];
            let got = c.words();
            assert_eq!(got[..3], written, "the dispatch reads the written words");
            assert_eq!(got[3..], [0x1111_1111; 61]);
            assert_eq!(w.words()[..3], written);
        }

        /// A list created with a pipeline has it set, as `SetPipelineState`
        /// sets one, whichever creation method the title calls: its draw
        /// state and `DispatchX` use it until the title sets another, and
        /// its first bind is logged. A list created without one has none.
        /// Before, the creation's pipeline was dropped.
        #[test]
        fn lists_bind_the_initial_pipeline_on_creation() {
            use std::sync::Arc;
            use std::sync::atomic::{AtomicUsize, Ordering};
            use tracing::{Event, Metadata, Subscriber, span};

            /// Counts the pipelines' "first bind" lines.
            struct FirstBinds(Arc<AtomicUsize>);
            impl Subscriber for FirstBinds {
                fn enabled(&self, metadata: &Metadata<'_>) -> bool {
                    let fields = metadata.fields();
                    metadata.is_event()
                        && metadata.target() == "xtp_gpu::pipeline"
                        && ["pipeline", "frame", "hash"]
                            .iter()
                            .all(|f| fields.field(f).is_some())
                }
                fn new_span(&self, _: &span::Attributes<'_>) -> span::Id {
                    span::Id::from_u64(1)
                }
                fn record(&self, _: &span::Id, _: &span::Record<'_>) {}
                fn record_follows_from(&self, _: &span::Id, _: &span::Id) {}
                fn event(&self, event: &Event<'_>) {
                    if self.enabled(event.metadata()) {
                        self.0.fetch_add(1, Ordering::Relaxed);
                    }
                }
                fn enter(&self, _: &span::Id) {}
                fn exit(&self, _: &span::Id) {}
            }

            /// A compute pipeline description as the title passes one.
            #[repr(C)]
            struct ComputeDesc {
                root: This,
                code: *const u8,
                size: usize,
            }

            let test = "list::tests::gpu::lists_bind_the_initial_pipeline_on_creation";
            let Some(_gpu) = device(test) else { return };
            let null = std::ptr::null();
            let mut allocator = null;
            assert_eq!(
                create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator),
                S_OK
            );
            let desc = ListDesc {
                kind: 0,
                flags: 0,
                nodes: 0,
                barrier_batch: 0,
            };
            // Bytes of this test's own (shaders are shared by their bytes),
            // which nothing translates: the list only names the pipeline.
            let bytes = [0xa5, 0xb6, 0xc7, 0x46];
            let compute = ComputeDesc {
                root: null,
                code: bytes.as_ptr(),
                size: bytes.len(),
            };
            let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
            // What the list's draw state holds (an address no pipeline has
            // until it has been read).
            let drawn = |list: This| {
                let mut pipeline = std::ptr::without_provenance(1);
                with_state(list, |_, s| pipeline = s.pipeline());
                pipeline
            };
            let binds = Arc::new(AtomicUsize::new(0));
            let counting = tracing::Dispatch::new(FirstBinds(binds.clone()));
            tracing::dispatcher::with_default(&counting, || {
                for console in [false, true] {
                    let create = |pipeline, out| {
                        if console {
                            create_list_x(null, &desc, allocator, pipeline, &LIST_IIDS[1], out)
                        } else {
                            create_list(null, 0, 0, allocator, pipeline, &LIST_IIDS[1], out)
                        }
                    };
                    let mut pipeline = null;
                    assert_eq!(
                        crate::pipeline::create_compute(
                            null,
                            (&raw const compute).cast(),
                            &PAGEABLE_IIDS[0],
                            &mut pipeline,
                        ),
                        S_OK
                    );
                    let before = binds.load(Ordering::Relaxed);
                    let mut list = null;
                    assert_eq!(create(null, &mut list), S_OK);
                    assert_eq!((drawn(list), kept_pipeline(list)), (null, null));
                    assert_eq!(binds.load(Ordering::Relaxed), before);
                    assert_eq!(release(list), 0);

                    assert_eq!(create(pipeline, &mut list), S_OK);
                    assert_eq!(
                        drawn(list),
                        pipeline,
                        "the draw state's (console {console})"
                    );
                    assert_eq!(kept_pipeline(list), pipeline, "DispatchX's");
                    assert_eq!(binds.load(Ordering::Relaxed), before + 1, "its first bind");
                    assert_eq!(release(list), 0);
                    assert_eq!(release(pipeline), 0);
                }
            });
            assert_eq!(release(allocator), 0);
        }

        /// A copy location as the title passes one: a resource, the kind (0
        /// a subresource index, 1 a placed footprint), then the index or the
        /// footprint (offset, format, width, height, depth, row pitch).
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

        /// Texture copies into title buffers (read-backs) run on the GPU in
        /// list order, after the draws that wrote the texture, and reach
        /// title memory once the list has run, as the title's footprints
        /// lay them out: at their offset, rows at their pitch, a box at its
        /// place. The bytes the copies leave keep what the title wrote.
        /// Before, these copies were skipped ("texture to buffer copies not
        /// implemented").
        #[test]
        fn texture_read_backs_reach_title_buffers_as_their_footprints_lay_them_out() {
            let test = "list::tests::gpu::texture_read_backs_reach_title_buffers_as_their_footprints_lay_them_out";
            let Some(_gpu) = device(test) else { return };
            const TITLE: u32 = 0xdead_beef;
            let (outer, inner) = ([1, 2, 3, 4], [9, 8, 7, 6]);
            let t = texture(4, 4);
            let list = TitleList::new();
            fill(list.this(), t, (0, 0, 4, 4), outer);
            fill(list.this(), t, (1, 1, 3, 3), inner);
            let dst = TitleBuffer::new(4096);
            dst.set_words(|_| TITLE);
            let footprint = |offset| Location {
                resource: dst.resource(),
                kind: 1,
                offset,
                format: RGBA8,
                width: 4,
                height: 4,
                depth: 1,
                row_pitch: 256,
            };
            let source = Location {
                resource: t,
                kind: 0,
                offset: 0,
                format: 0,
                width: 0,
                height: 0,
                depth: 0,
                row_pitch: 0,
            };
            let copy = |to: &Location, [x, y, z]: [u32; 3], area: Option<&[u32; 6]>| {
                copy_texture_region(
                    list.this(),
                    std::ptr::from_ref(to).cast(),
                    x,
                    y,
                    z,
                    (&raw const source).cast(),
                    area.map_or(std::ptr::null(), std::ptr::from_ref),
                )
            };
            // All of it 512 bytes in; its middle 2x2 to (2, 1) of a
            // footprint 2048 bytes in (a box: left, top, front, right,
            // bottom, back).
            copy(&footprint(512), [0, 0, 0], None);
            copy(&footprint(2048), [2, 1, 0], Some(&[1, 1, 0, 3, 3, 1]));
            list.run();
            release(t);
            let mut want = vec![TITLE; 1024];
            for y in 0..4 {
                for x in 0..4 {
                    let middle = (1..3).contains(&x) && (1..3).contains(&y);
                    want[(512 + y * 256) / 4 + x] =
                        u32::from_le_bytes(if middle { inner } else { outer });
                }
            }
            for y in 1..3 {
                for x in 2..4 {
                    want[(2048 + y * 256) / 4 + x] = u32::from_le_bytes(inner);
                }
            }
            assert_eq!(dst.words(), want);
        }

        /// The location of subresource `index` of `texture`.
        fn subresource(texture: This, index: u32) -> Location {
            Location {
                resource: texture,
                kind: 0,
                offset: u64::from(index),
                format: 0,
                width: 0,
                height: 0,
                depth: 0,
                row_pitch: 0,
            }
        }

        /// A footprint `offset` bytes into `buffer`: `format`, width,
        /// height and depth in texels, rows `pitch` bytes apart.
        fn placed(
            buffer: &TitleBuffer,
            offset: u64,
            format: u32,
            [width, height, depth]: [u32; 3],
            pitch: u32,
        ) -> Location {
            Location {
                resource: buffer.resource(),
                kind: 1,
                offset,
                format,
                width,
                height,
                depth,
                row_pitch: pitch,
            }
        }

        /// `CopyTextureRegion` in `list` from `from` to texel `at` of `to`:
        /// `area` of `from` (a box: left, top, front, right, bottom, back),
        /// or all of it.
        fn copy_region(
            list: &TitleList,
            to: &Location,
            [x, y, z]: [u32; 3],
            from: &Location,
            area: Option<&[u32; 6]>,
        ) {
            copy_texture_region(
                list.this(),
                std::ptr::from_ref(to).cast(),
                x,
                y,
                z,
                std::ptr::from_ref(from).cast(),
                area.map_or(std::ptr::null(), std::ptr::from_ref),
            );
        }

        /// Writes `value` at title address `address` when `list`'s
        /// submission is prepared, after the steps recorded before (the
        /// host copies' refreshes from title memory among them) and before
        /// the GPU runs the list, so before anything goes back to title
        /// memory: as one of the title's threads writes while a list it
        /// submitted is in flight.
        fn title_write_meanwhile(list: &TitleList, address: u64, value: u32) {
            list.host().defer((address as usize, 4), (0, 0), move || {
                // SAFETY: title memory of a test buffer, alive until the
                // list has run.
                unsafe { (address as *mut u32).write_unaligned(value) }
            });
        }

        /// A read-back writes back only the bytes it copies: rows of
        /// texels, not the bytes between them nor beside a box, which the
        /// title may write while the list runs. Two rows of one texel 256
        /// bytes apart write bytes 0-3 and 256-259 of their footprint, a
        /// 2x2 box 8 bytes of two rows. Title writes between those rows,
        /// beside the box and above it, made once the submission took its
        /// bytes and before the write-back, stay. Before, the footprint's
        /// whole range went back as title memory was before those writes
        /// (Codex review of the #1583 port).
        #[test]
        fn read_backs_write_back_only_the_bytes_they_copy() {
            let test = "list::tests::gpu::read_backs_write_back_only_the_bytes_they_copy";
            let Some(_gpu) = device(test) else { return };
            const TITLE: u32 = 0xdead_beef;
            const LATER: u32 = 0x1a7e_0000;
            let (outer, inner) = ([1, 2, 3, 4], [9, 8, 7, 6]);
            let t = texture(4, 4);
            let list = TitleList::new();
            fill(list.this(), t, (0, 0, 4, 4), outer);
            fill(list.this(), t, (1, 1, 3, 3), inner);
            let dst = TitleBuffer::new(4096);
            dst.set_words(|_| TITLE);
            let source = subresource(t, 0);
            // Texels (1, 1) and (1, 2), 512 bytes in: 260 bytes.
            let column = placed(&dst, 512, RGBA8, [1, 2, 1], 256);
            copy_region(
                &list,
                &column,
                [0, 0, 0],
                &source,
                Some(&[1, 1, 0, 2, 3, 1]),
            );
            // The middle 2x2 to (1, 1) of a 4x4 footprint 2048 bytes in.
            let square = placed(&dst, 2048, RGBA8, [4, 4, 1], 256);
            copy_region(
                &list,
                &square,
                [1, 1, 0],
                &source,
                Some(&[1, 1, 0, 3, 3, 1]),
            );
            // Between the column's rows (twice), beside the box in its
            // first row (both sides), above it.
            let later = [512 + 64, 512 + 252, 2048 + 256, 2048 + 256 + 12, 2048 + 4];
            for (i, &byte) in later.iter().enumerate() {
                title_write_meanwhile(&list, dst.address() + byte, LATER + i as u32);
            }
            list.run();
            release(t);
            let inner = u32::from_le_bytes(inner);
            let mut want = vec![TITLE; 1024];
            want[512 / 4] = inner;
            want[(512 + 256) / 4] = inner;
            for y in 1..3 {
                for x in 1..3 {
                    want[(2048 + y * 256) / 4 + x] = inner;
                }
            }
            for (i, &byte) in later.iter().enumerate() {
                want[byte as usize / 4] = LATER + i as u32;
            }
            assert_eq!(dst.words(), want);
        }

        /// Texture A to buffer B, then B to texture C, in one list: C gets
        /// A's texels, which only B's host copy holds before the list has
        /// run (read on the GPU, in list order). So does a copy from bytes
        /// of B a shader wrote earlier in the list, at an offset the host
        /// places no footprint at (1280 bytes: not a multiple of 512),
        /// through a buffer of the list's own. Read back from C, in the
        /// same list. Before, B to C read B's title memory at submission:
        /// zeros (Codex review of the #1583 port).
        #[test]
        fn a_texture_copied_through_a_buffer_in_one_list_arrives() {
            let test = "list::tests::gpu::a_texture_copied_through_a_buffer_in_one_list_arrives";
            let Some(gpu) = device(test) else { return };
            let host = gpu.host();
            const RED: [u8; 4] = [255, 0, 0, 255];
            const OTHER: [u8; 4] = [1, 2, 3, 4];
            const WRITTEN: u64 = 1280;
            let kernel = Kernel::new(host);
            let (a, c) = (texture(4, 4), texture(64, 5));
            let (b, d) = (TitleBuffer::new(4096), TitleBuffer::new(4096));
            let list = TitleList::new();
            fill(list.this(), a, (0, 0, 4, 4), RED);
            fill(list.this(), c, (0, 0, 64, 5), OTHER);
            let square = placed(&b, 0, RGBA8, [4, 4, 1], 256);
            copy_region(&list, &square, [0, 0, 0], &subresource(a, 0), None);
            copy_region(&list, &subresource(c, 0), [0, 0, 0], &square, None);
            // 64 words of the fill pattern, one row of 64 texels into C's
            // last row.
            let at = b.address() + WRITTEN;
            kernel.dispatch(list.host(), Pass::Fill, at, at);
            let row = placed(&b, WRITTEN, RGBA8, [64, 1, 1], 256);
            copy_region(&list, &subresource(c, 0), [0, 4, 0], &row, None);
            let all = placed(&d, 0, RGBA8, [64, 5, 1], 256);
            copy_region(&list, &all, [0, 0, 0], &subresource(c, 0), None);
            list.run();
            for t in [a, c] {
                release(t);
            }
            let (red, other) = (u32::from_le_bytes(RED), u32::from_le_bytes(OTHER));
            let mut want = vec![0; 1024];
            for y in 0..4 {
                for x in 0..64 {
                    want[y * 64 + x] = if x < 4 { red } else { other };
                }
            }
            want[256..320].copy_from_slice(&filled(0..64));
            assert_eq!(d.words(), want, "C read back");
            let mut want = vec![0; 1024];
            for y in 0..4 {
                want[y * 64..y * 64 + 4].fill(red);
            }
            want[320..384].copy_from_slice(&filled(0..64));
            assert_eq!(b.words(), want, "B");
        }

        /// Read-backs of the second mip of the second slice (subresource 3
        /// of two mips and two slices), and of a box of the second mip of
        /// the first (subresource 1), in a block-compressed format (BC1:
        /// 4x4 blocks of 8 bytes, rows of blocks) and in RGBA8: each
        /// subresource's own texels arrive, rows at the footprint's pitch,
        /// and only those bytes go back (title writes beside them, made
        /// while the list is in flight, stay). The texels come from title
        /// buffers through buffer-to-texture copies.
        #[test]
        fn read_backs_of_mips_slices_and_blocks_copy_their_rows() {
            let test = "list::tests::gpu::read_backs_of_mips_slices_and_blocks_copy_their_rows";
            let Some(_gpu) = device(test) else { return };
            const BC1: u32 = 71;
            const TITLE: u32 = 0xdead_beef;
            const LATER: u32 = 0x1a7e_0000;
            // Format, mip 0's width and height, bytes of a block, its width.
            for (format, width, element, block) in [(RGBA8, 8u32, 4u32, 1u32), (BC1, 16, 8, 4)] {
                let t = title_texture(format, u64::from(width), width, [2, 2], 0);
                let (half, quarter) = (width / 2, width / 4);
                let s = TitleBuffer::new(4096);
                s.set_words(|i| 0x5100_0000 + i as u32);
                let d = TitleBuffer::new(4096);
                d.set_words(|_| TITLE);
                let list = TitleList::new();
                let mip = |b: &TitleBuffer, offset| placed(b, offset, format, [half, half, 1], 256);
                copy_region(&list, &subresource(t, 3), [0, 0, 0], &mip(&s, 0), None);
                copy_region(&list, &subresource(t, 1), [0, 0, 0], &mip(&s, 2048), None);
                copy_region(&list, &mip(&d, 512), [0, 0, 0], &subresource(t, 3), None);
                // The right half of the top half to the left of the bottom
                // half.
                let area = [quarter, 0, 0, half, quarter, 1];
                copy_region(
                    &list,
                    &mip(&d, 2048),
                    [0, quarter, 0],
                    &subresource(t, 1),
                    Some(&area),
                );
                // Blocks (texels for RGBA8) across and down mip 1, and the
                // box; the words of a row of each.
                let (rows, boxed) = ((half / block) as usize, (quarter / block) as usize);
                let words = |blocks: usize| blocks * element as usize / 4;
                let (row, box_row) = (words(rows), words(boxed));
                // Beside the first row, above the box, beside its first row.
                let later = [512 + 4 * row, 2048, 2048 + boxed * 256 + 4 * box_row];
                for (i, &byte) in later.iter().enumerate() {
                    title_write_meanwhile(&list, d.address() + byte as u64, LATER + i as u32);
                }
                list.run();
                release(t);
                let source = |byte: usize| 0x5100_0000 + byte as u32 / 4;
                let mut want = vec![TITLE; 1024];
                for r in 0..rows {
                    for k in 0..row {
                        want[(512 + r * 256) / 4 + k] = source(r * 256) + k as u32;
                    }
                }
                for j in 0..boxed {
                    for k in 0..box_row {
                        want[(2048 + (boxed + j) * 256) / 4 + k] =
                            source(2048 + j * 256 + 4 * box_row) + k as u32;
                    }
                }
                for (i, &byte) in later.iter().enumerate() {
                    want[byte / 4] = LATER + i as u32;
                }
                assert_eq!(d.words(), want, "format {format}");
            }
        }

        /// Copies between buffers and a texture in the console's 16-bit
        /// depth with stencil (format 118), which no host footprint
        /// describes, are refused: nothing reaches the host (which would
        /// get the console's format number in a footprint, an invalid
        /// copy), the list closes, and the buffer keeps its bytes. Before,
        /// both directions went to the host as they were (Codex review of
        /// the #1583 port).
        #[test]
        fn copies_in_the_consoles_depth_with_stencil_are_refused() {
            let test = "list::tests::gpu::copies_in_the_consoles_depth_with_stencil_are_refused";
            let Some(_gpu) = device(test) else { return };
            const D16S8: u32 = 118;
            const DEPTH_STENCIL: u32 = 2;
            const TITLE: u32 = 0xdead_beef;
            let t = title_texture(D16S8, 4, 4, [1, 1], DEPTH_STENCIL);
            assert!(
                crate::resource::host_resource(t).is_some(),
                "a host texture stands in for it"
            );
            let b = TitleBuffer::new(4096);
            b.set_words(|_| TITLE);
            let list = TitleList::new();
            for format in [D16S8, 45] {
                let footprint = placed(&b, 0, format, [4, 4, 1], 256);
                copy_region(&list, &footprint, [0, 0, 0], &subresource(t, 0), None);
            }
            copy_region(
                &list,
                &subresource(t, 0),
                [0, 0, 0],
                &placed(&b, 0, D16S8, [4, 4, 1], 256),
                None,
            );
            list.run();
            release(t);
            assert_eq!(b.words(), vec![TITLE; 1024]);
        }

        /// A read-back into a buffer the list wrote on the GPU just before
        /// (a UAV clear of it) and read (a dispatch, through a root view),
        /// in one list; then, in a second submission, another read-back
        /// into it. Under the D3D12 debug layer (`XTP_D3D12_DEBUG=1` with
        /// diagnostics, as CI's gpu-debug-layer job runs), the barriers
        /// taking its host copy to the copy's state and back must name the
        /// state it is in: before, the clear left it in UNORDERED_ACCESS
        /// (promoted from COMMON) and the read-back's barrier named COMMON
        /// (message 527, RESOURCE_BARRIER_BEFORE_AFTER_MISMATCH, on WARP).
        /// Title memory then holds the clear's words but where each
        /// read-back wrote, and so does what the dispatch read.
        /// shortcut: a vertex buffer or a root view is a GPU address, which
        /// the layer does not tie to a resource: only the clear's state is
        /// checked here.
        #[test]
        fn read_backs_into_a_buffer_the_gpu_wrote_and_read() {
            let test = "list::tests::gpu::read_backs_into_a_buffer_the_gpu_wrote_and_read";
            let Some(gpu) = device(test) else { return };
            let host = gpu.host();
            const R32_UINT: u32 = 42;
            const BUFFER: u32 = 1;
            const CLEARED: u32 = 0xc1ea_0000;
            const RED: [u8; 4] = [255, 0, 0, 255];
            let kernel = Kernel::new(host);
            let t = texture(4, 4);
            let (b, e) = (TitleBuffer::new(4096), TitleBuffer::new(256));
            b.set_words(|i| i as u32);
            let heap = crate::descriptor::TestHeap::views(1);
            let mut slot = [0u64; 4];
            let view = [R32_UINT, BUFFER, 0, 0, 1024, 0, 0, 0, 0, 0];
            let (gpu_handle, cpu) = console_handles(b.resource(), Some(&view), &mut slot, &heap);
            let list = TitleList::new();
            fill(list.this(), t, (0, 0, 4, 4), RED);
            let values = [CLEARED, 0, 0, 0];
            clear_uav_uint(
                list.this(),
                gpu_handle,
                cpu,
                b.resource(),
                &values,
                0,
                std::ptr::null(),
            );
            kernel.dispatch(list.host(), Pass::Copy, b.address(), e.address());
            copy_region(
                &list,
                &placed(&b, 512, RGBA8, [4, 4, 1], 256),
                [0, 0, 0],
                &subresource(t, 0),
                None,
            );
            list.run();
            let red = u32::from_le_bytes(RED);
            let mut want = vec![CLEARED; 1024];
            for y in 0..4 {
                want[(512 + y * 256) / 4..][..4].fill(red);
            }
            assert_eq!(b.words(), want, "after the first list");
            assert_eq!(e.words(), vec![CLEARED; 64], "what the dispatch read");
            let again = TitleList::new();
            copy_region(
                &again,
                &placed(&b, 2048, RGBA8, [1, 1, 1], 256),
                [0, 0, 0],
                &subresource(t, 0),
                Some(&[0, 0, 0, 1, 1, 1]),
            );
            again.run();
            release(t);
            want[2048 / 4] = red;
            assert_eq!(b.words(), want, "after the second list");
        }

        /// A title resource description (`D3D12_RESOURCE_DESC`), as the
        /// title passes it.
        #[repr(C)]
        struct TextureDesc {
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

        /// R8G8B8A8_UNORM, which every host keeps and reads back as bytes.
        const RGBA8: u32 = 28;
        const UNORDERED_ACCESS: u32 = 0x8;

        /// A committed `width` x `height` RGBA8 title texture, a render
        /// target with unordered access (as the FidelityFX output is).
        fn texture(width: u64, height: u32) -> This {
            title_texture(RGBA8, width, height, [1, 1], 1 | 4)
        }

        /// A committed 2D title texture: `format`, `width` x `height`,
        /// `[slices, mips]`, resource `flags` (1 render target, 2 depth
        /// stencil, 4 unordered access).
        fn title_texture(
            format: u32,
            width: u64,
            height: u32,
            [slices, mips]: [u16; 2],
            flags: u32,
        ) -> This {
            const DEFAULT_HEAP: [u32; 5] = [1, 0, 0, 0, 0];
            const RESOURCE_IID: Iid = guid(
                0x696442be,
                0xa72e,
                0x4059,
                [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
            );
            let desc = TextureDesc {
                dimension: 3,
                alignment: 0,
                width,
                height,
                depth_or_array_size: slices,
                mip_levels: mips,
                format,
                sample_count: 1,
                sample_quality: 0,
                layout: 0,
                flags,
            };
            let null = std::ptr::null();
            let mut out = null;
            let made = crate::resource::create_committed(
                null,
                DEFAULT_HEAP.as_ptr().cast(),
                0,
                (&raw const desc).cast(),
                0,
                null.cast(),
                &RESOURCE_IID,
                &mut out,
            );
            assert_eq!(made, S_OK, "a {width}x{height} title texture");
            out
        }

        fn release(object: This) {
            slot::<extern "system" fn(This) -> u32>(object, 0x10)(object);
        }

        /// Fills the rectangle `[x0, x1) x [y0, y1)` of `texture` with
        /// `rgba` (bytes) through a render target view, as the title clears.
        fn fill(list: This, texture: This, (x0, y0, x1, y1): (i32, i32, i32, i32), rgba: [u8; 4]) {
            let mut slot = [0u64; 16];
            let handle = slot.as_mut_ptr() as u64;
            crate::descriptor::create_rtv(std::ptr::null(), texture, std::ptr::null(), handle);
            let color = rgba.map(|c| f32::from(c) / 255.0);
            let rect = Rect {
                left: x0,
                top: y0,
                right: x1,
                bottom: y1,
            };
            clear_rtv(list, handle, &color, 1, &rect);
        }

        /// The red channel of row `y` of a read back texture (rows 256
        /// bytes apart).
        fn reds(texels: &[u8], y: usize, width: usize) -> Vec<u8> {
            (0..width).map(|x| texels[y * 256 + x * 4]).collect()
        }

        fn near(got: &[u8], want: &[u8]) -> bool {
            got.len() == want.len() && got.iter().zip(want).all(|(&g, &w)| g.abs_diff(w) <= 1)
        }

        /// The FidelityFX stand-in's bilinear upscale: an 8x2 input whose
        /// left 4x2 is rendered (red 0, 64, 128, 192 by column; 255 past
        /// the rendered region) goes to the 8x4 top left of a 16x6 output.
        /// Without jitter, output pixel x reads the input at (x + 0.5) / 2;
        /// with a jitter of a quarter input texel, a quarter further, as
        /// FidelityFX's convention has it (the texel at i holds the scene
        /// at i + 0.5 - jitter). Samples never reach past the rendered
        /// region, and the rest of the output is left as it was.
        #[test]
        fn the_fidelityfx_upscale_is_bilinear_jittered_and_stays_in_its_regions() {
            let test = "list::tests::gpu::the_fidelityfx_upscale_is_bilinear_jittered_and_stays_in_its_regions";
            let Some(gpu) = device(test) else { return };
            let host = gpu.host();
            let (color, plain, jittered) = (texture(8, 2), texture(16, 6), texture(16, 6));
            let list = TitleList::new();
            for (i, red) in [0u8, 64, 128, 192].into_iter().enumerate() {
                let x = i as i32;
                fill(list.this(), color, (x, 0, x + 1, 2), [red, 7, 9, 255]);
            }
            fill(list.this(), color, (4, 0, 8, 2), [255, 255, 255, 255]);
            for out in [plain, jittered] {
                fill(list.this(), out, (0, 0, 16, 6), [1, 2, 3, 4]);
            }
            for (out, jitter) in [(plain, [0.0, 0.0]), (jittered, [0.25, 0.0])] {
                let u = Upscale {
                    list: list.this(),
                    color,
                    output: out,
                    render: [4, 2],
                    size: [8, 4],
                    jitter,
                };
                assert!(upscale(&u), "the upscale is recorded");
            }
            list.run();
            let read = |t: This| {
                let texture = crate::resource::host_resource(t).expect("a host texture");
                crate::resource::tests_support::read_back(host, texture, UNORDERED_ACCESS)
            };
            let (a, b) = (read(plain), read(jittered));
            for t in [color, plain, jittered] {
                release(t);
            }
            for y in 0..4 {
                let row = reds(&a, y, 16);
                assert!(
                    near(&row[..8], &[0, 16, 48, 80, 112, 144, 176, 192]),
                    "row {y}: {row:?}"
                );
                assert_eq!(row[8..], [1; 8], "row {y} past the output size");
                let row = reds(&b, y, 16);
                assert!(
                    near(&row[..8], &[0, 32, 64, 96, 128, 160, 192, 192]),
                    "jittered row {y}: {row:?}"
                );
                // The other channels come along.
                assert!(near(&a[y * 256 + 4..y * 256 + 8], &[16, 7, 9, 255]));
            }
            for y in 4..6 {
                assert_eq!(reds(&a, y, 16), [1; 16], "row {y} below the output size");
            }
        }

        /// A view's record at `cpu` (memory of the title's, as a CPU-only
        /// heap is) copied to slot 0 of a shader-visible title heap, as the
        /// title prepares a UAV clear: the console handles the clear gets.
        fn console_handles(
            resource: This,
            desc: Option<&[u32; 10]>,
            cpu: &mut [u64; 4],
            heap: &crate::descriptor::TestHeap,
        ) -> (u64, u64) {
            let null: This = std::ptr::null();
            let cpu = cpu.as_mut_ptr() as u64;
            let desc = desc.map_or(std::ptr::null(), std::ptr::from_ref);
            crate::descriptor::create_uav(null, resource, null, desc, cpu);
            let gpu = crate::descriptor::heap(heap.0).start;
            const VIEWS: u32 = 0;
            crate::descriptor::copy_simple(null, 1, gpu, cpu, VIEWS);
            (gpu, cpu)
        }

        /// `ClearUnorderedAccessViewFloat` then `…Uint` on a texture, with
        /// the console's handles (slots of title memory, which a host
        /// driver cannot read): the float clear converts to the view's
        /// format, the integer one writes the values' low bits, and only
        /// inside its rectangle.
        #[test]
        fn uav_clears_of_a_texture_go_through_host_views() {
            let test = "list::tests::gpu::uav_clears_of_a_texture_go_through_host_views";
            let Some(gpu) = device(test) else { return };
            let host = gpu.host();
            let image = texture(4, 4);
            let heap = crate::descriptor::TestHeap::views(4);
            let mut cpu = [0u64; 4];
            let (gpu, cpu) = console_handles(image, None, &mut cpu, &heap);
            let list = TitleList::new();
            fill(list.this(), image, (0, 0, 4, 4), [9, 9, 9, 9]);
            let float = [0.5f32, 0.25, 0.0, 1.0];
            clear_uav_float(list.this(), gpu, cpu, image, &float, 0, std::ptr::null());
            let rect = Rect {
                left: 2,
                top: 0,
                right: 4,
                bottom: 2,
            };
            clear_uav_uint(list.this(), gpu, cpu, image, &[1, 2, 3, 0x104], 1, &rect);
            list.run();
            let texture = crate::resource::host_resource(image).expect("a host texture");
            let texels = crate::resource::tests_support::read_back(host, texture, UNORDERED_ACCESS);
            release(image);
            for y in 0..4 {
                for x in 0..4 {
                    let at = y * 256 + x * 4;
                    let want: &[u8] = if x >= 2 && y < 2 {
                        &[1, 2, 3, 4]
                    } else {
                        &[128, 64, 0, 255]
                    };
                    assert!(
                        near(&texels[at..at + 4], want),
                        "texel {x},{y}: {:?}",
                        &texels[at..at + 4]
                    );
                }
            }
        }

        #[test]
        fn a_placed_texture_clear_updates_the_same_host_copy_as_its_view() {
            let Some(gpu) = device("list::tests::gpu::placed_texture_clear") else {
                return;
            };
            let memory = crate::resource::Memory::zeroed(65536, 65536).unwrap();
            let address = memory.address();
            let desc = crate::resource::tests_support::target(4, 4, 28);
            // SAFETY: ResourceDesc contient les quatorze mots du descripteur console.
            let mut resource: [u32; 14] = unsafe { std::mem::transmute_copy(&desc) };
            resource[12] = 4;
            let mut view = [0; 16];
            view[..2].copy_from_slice(&[28, 4]);
            (view[10], view[11]) = (address as u32, (address >> 32) as u32);
            let heap = crate::descriptor::TestHeap::views(4);
            let handle = crate::descriptor::heap(heap.0).start;
            crate::descriptor::create_placed_uav(std::ptr::null(), &resource, &view, handle);
            let list = TitleList::new();
            clear_uav_uint(
                list.this(),
                handle,
                handle,
                std::ptr::null(),
                &[1, 2, 3, 4],
                0,
                std::ptr::null(),
            );
            list.run();
            let object = crate::resource::texture_at(address, &resource).unwrap();
            let texture = crate::resource::host_resource(object).unwrap();
            let texels =
                crate::resource::tests_support::read_back(gpu.host(), texture, UNORDERED_ACCESS);
            assert_eq!(&texels[..4], &[1, 2, 3, 4]);
            drop(heap);
            crate::resource::memory_remapped(address as usize, 65536);
        }

        /// A buffer view's integer clear writes its range of the host copy,
        /// which reaches title memory once the list has run; a float clear
        /// of a raw view is refused, as the host would.
        #[test]
        fn uav_clears_of_a_buffer_reach_title_memory() {
            let test = "list::tests::gpu::uav_clears_of_a_buffer_reach_title_memory";
            let Some(_gpu) = device(test) else {
                return;
            };
            const R32_UINT: u32 = 42;
            const R32_TYPELESS: u32 = 39;
            const BUFFER: u32 = 1;
            const RAW: u32 = 1;
            let buffer = TitleBuffer::new(256);
            buffer.set_words(|i| i as u32);
            // Format, dimension, first element (2 words), count, stride,
            // counter offset (2 words), flags: words 16 to 47.
            let typed = [R32_UINT, BUFFER, 16, 0, 32, 0, 0, 0, 0, 0];
            let raw = [R32_TYPELESS, BUFFER, 0, 0, 64, 0, 0, 0, RAW, 0];
            let heap = crate::descriptor::TestHeap::views(4);
            let list = TitleList::new();
            let mut slot = [0u64; 4];
            let (gpu, cpu) = console_handles(buffer.resource(), Some(&raw), &mut slot, &heap);
            clear_uav_float(
                list.this(),
                gpu,
                cpu,
                buffer.resource(),
                &[1.0; 4],
                0,
                std::ptr::null(),
            );
            let mut slot = [0u64; 4];
            let (gpu, cpu) = console_handles(buffer.resource(), Some(&typed), &mut slot, &heap);
            clear_uav_uint(
                list.this(),
                gpu,
                cpu,
                buffer.resource(),
                &[0xc1ea_0000, 0, 0, 0],
                0,
                std::ptr::null(),
            );
            list.run();
            let want: Vec<u32> = (0..64)
                .map(|i| {
                    if (16..48).contains(&i) {
                        0xc1ea_0000
                    } else {
                        i
                    }
                })
                .collect();
            assert_eq!(buffer.words(), want);
        }

        /// Clears of typed buffer views whose channels are narrower than
        /// the values. An integer clear keeps the low bits of each value:
        /// D3D's rule, which WARP follows; the L4's driver saturates what
        /// does not fit, so XTP passes it the low bits only
        /// (`format::uint_clear`, docs/analysis/hardware-tests.md). A float
        /// clear converts as D3D's float-to-normalized rules say (NaN gives
        /// 0, the rest is clamped, then rounded), on both.
        #[test]
        fn uav_clears_of_typed_buffers_write_the_documented_bits() {
            let test = "list::tests::gpu::uav_clears_of_typed_buffers_write_the_documented_bits";
            let Some(_gpu) = device(test) else {
                return;
            };
            const BUFFER: u32 = 1;
            // (view format, values, the word each of the 4 elements holds)
            let uint: [(u32, [u32; 4], u32); 7] = [
                // R8G8B8A8_UNORM
                (28, [1, 2, 3, 0x104], 0x0403_0201),
                // R8G8B8A8_UINT
                (30, [0x1ff, 0x80, 0x1234_5678, 7], 0x0778_80ff),
                // R8G8B8A8_SINT: -128, 383, 128, -1
                (32, [0xffff_ff80, 0x17f, 0x80, u32::MAX], 0xff80_7f80),
                // R16G16_UINT
                (36, [0x1_2345, 0xffff_0001, 0, 0], 0x0001_2345),
                // R16G16_SINT
                (38, [0x1_8000, 0xffff_fffe, 0, 0], 0xfffe_8000),
                // R10G10B10A2_UINT
                (25, [0x401, 0x3ff, 0x7ff, 5], 0x7fff_fc01),
                // R32_UINT
                (42, [0xc1ea_0000, 0, 0, 0], 0xc1ea_0000),
            ];
            let float: [(u32, [f32; 4], u32); 2] = [
                // R8G8B8A8_UNORM: 0, 255, 0, 63.75 rounded
                (28, [-0.5, 1.5, f32::NAN, 0.25], 0x4000_ff00),
                // R8G8B8A8_SNORM: -127, 127, 0, -31.75 rounded
                (31, [-2.0, 2.0, f32::NAN, -0.25], 0xe000_7f81),
            ];
            let buffer = TitleBuffer::new(256);
            let heap = crate::descriptor::TestHeap::views(4);
            let list = TitleList::new();
            // Case i clears elements 4i to 4i + 3, through a view of its own.
            let mut slots = vec![[0u64; 4]; uint.len() + float.len()];
            let mut cases = slots.iter_mut().enumerate();
            let mut handles = |format: u32| {
                let (i, slot) = cases.next().expect("a slot per case");
                let view = [format, BUFFER, 4 * i as u32, 0, 4, 0, 0, 0, 0, 0];
                console_handles(buffer.resource(), Some(&view), slot, &heap)
            };
            let null = std::ptr::null();
            for (format, values, _) in &uint {
                let (gpu, cpu) = handles(*format);
                clear_uav_uint(list.this(), gpu, cpu, buffer.resource(), values, 0, null);
            }
            for (format, values, _) in &float {
                let (gpu, cpu) = handles(*format);
                clear_uav_float(list.this(), gpu, cpu, buffer.resource(), values, 0, null);
            }
            list.run();
            let words = buffer.words();
            let cases = uint
                .iter()
                .map(|&(f, _, w)| (f, "uint", w))
                .chain(float.iter().map(|&(f, _, w)| (f, "float", w)));
            let wrong: Vec<String> = cases
                .enumerate()
                .filter_map(|(i, (format, kind, want))| {
                    let got = &words[4 * i..4 * i + 4];
                    (got != [want; 4])
                        .then(|| format!("format {format} ({kind}): {got:08x?}, want {want:08x}"))
                })
                .collect();
            assert!(wrong.is_empty(), "{}", wrong.join("; "));
        }
    }

    /// The thread-count bit, as BTP 6 sets it (see [`dispatch_x`]).
    #[cfg(windows)]
    const THREADS: u32 = USE_THREAD_DIMENSIONS;

    /// `DispatchX`'s counts are groups unless one of the switch's bits is
    /// in its flags; then they are threads, run as whole groups of the
    /// program, each dimension rounded up.
    #[cfg(windows)]
    #[test]
    fn dispatch_x_counts_are_groups_unless_a_flag_makes_them_threads() {
        let row = Some([64, 1, 1]);
        assert_eq!(dispatch_x_groups([5, 2, 1], 0, THREADS, row), [5, 2, 1]);
        assert_eq!(
            dispatch_x_groups([5, 2, 1], 0x8000, THREADS, row),
            [5, 2, 1],
            "32-lane waves only"
        );
        assert_eq!(
            dispatch_x_groups([45, 1, 1], 0x8020, THREADS, row),
            [1, 1, 1],
            "BTP 6's 32-lane form: one group where 45 ran before"
        );
        assert_eq!(
            dispatch_x_groups([45, 1, 1], THREADS | PARTIAL_GROUPS, THREADS, row),
            [1, 1, 1],
            "partial groups: still whole groups on the host"
        );
        assert_eq!(
            dispatch_x_groups([5, 2, 1], THREADS, 0, row),
            [5, 2, 1],
            "no bits switched on"
        );
        assert_eq!(
            dispatch_x_groups([1000, 1, 1], THREADS, THREADS, row),
            [16, 1, 1]
        );
        assert_eq!(
            dispatch_x_groups([1024, 1, 1], THREADS, THREADS, row),
            [16, 1, 1]
        );
        assert_eq!(
            dispatch_x_groups([1025, 1, 1], THREADS | 0x8000, THREADS, row),
            [17, 1, 1],
            "with another flag"
        );
        assert_eq!(
            dispatch_x_groups([20, 5, 3], THREADS, THREADS, Some([8, 4, 2])),
            [3, 2, 2],
            "each dimension by its own size"
        );
        assert_eq!(
            dispatch_x_groups([0, 7, 1], THREADS, THREADS, Some([8, 8, 1])),
            [0, 1, 1],
            "no threads, no groups"
        );
        assert_eq!(
            dispatch_x_groups([1024, 1, 1], THREADS, THREADS, Some([1024, 1, 1])),
            [1, 1, 1],
            "one group where 1,024 ran before"
        );
        assert_eq!(
            dispatch_x_groups([1000, 1, 1], THREADS, THREADS, None),
            [1000, 1, 1],
            "no program: as given (the dispatch is dropped)"
        );
        assert_eq!(
            dispatch_x_groups([1000, 1, 1], THREADS, THREADS, Some([0, 1, 1])),
            [1000, 1, 1],
            "an empty group: as given"
        );
    }

    /// The register's thread-count bit unless `XTP_DISPATCH_X_THREADS`
    /// gives others, in hexadecimal; `0` for none.
    #[cfg(windows)]
    #[test]
    fn thread_count_flags_are_read_in_hexadecimal() {
        assert_eq!(USE_THREAD_DIMENSIONS, 0x20);
        assert_eq!(parse_thread_count_flags(None), USE_THREAD_DIMENSIONS);
        assert_eq!(
            parse_thread_count_flags(Some("0")),
            0,
            "none: counts are groups"
        );
        assert_eq!(parse_thread_count_flags(Some("0x20")), 0x20);
        assert_eq!(parse_thread_count_flags(Some("0X1")), 1);
        assert_eq!(parse_thread_count_flags(Some(" 30 ")), 0x30);
        assert_eq!(
            parse_thread_count_flags(Some("threads")),
            USE_THREAD_DIMENSIONS,
            "not a number: the default"
        );
        assert_eq!(parse_thread_count_flags(Some("")), USE_THREAD_DIMENSIONS);
    }

    /// `DispatchX`'s two forms on the host GPU.
    #[cfg(windows)]
    mod dispatch_x_gpu {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::core::s;

        use super::THREADS;
        use crate::list::dispatch_x_groups;
        use crate::testgpu::{TitleBuffer, TitleList, device};

        /// A program of 8 × 4 threads that counts the threads it runs (word
        /// 0) and the extent they cover (words 1 to 3: the largest thread
        /// id + 1 in x, y and z).
        const COUNTING: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer seen : register(u0);
[RootSignature(RS)]
[numthreads(8, 4, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    seen.InterlockedAdd(0, 1);
    seen.InterlockedMax(4, id.x + 1);
    seen.InterlockedMax(8, id.y + 1);
    seen.InterlockedMax(12, id.z + 1);
}
"#;

        fn counting(host: &crate::host::Host) -> (ID3D12RootSignature, ID3D12PipelineState) {
            let mut code: Option<ID3DBlob> = None;
            let mut errors: Option<ID3DBlob> = None;
            // SAFETY: the source and names live through the call; out
            // pointers ours.
            let compiled = unsafe {
                D3DCompile(
                    COUNTING.as_ptr().cast(),
                    COUNTING.len(),
                    s!("counting"),
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
                panic!("the counting program did not compile ({e}): {text}");
            }
            let code = code.expect("compiled code");
            // SAFETY: the blob's own bytes; the root signature is inside it.
            unsafe {
                let bytes = std::slice::from_raw_parts(
                    code.GetBufferPointer().cast::<u8>(),
                    code.GetBufferSize(),
                );
                let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes).unwrap();
                let pso = host
                    .device
                    .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&root),
                        CS: D3D12_SHADER_BYTECODE {
                            pShaderBytecode: bytes.as_ptr().cast(),
                            BytecodeLength: bytes.len(),
                        },
                        ..Default::default()
                    })
                    .unwrap();
                (root, pso)
            }
        }

        /// The counting program dispatched with the groups `DispatchX`
        /// makes of each case's counts and flags. Counts in groups run
        /// those groups. Counts in threads run the whole groups that cover
        /// them: the host has no partial groups, so the last group of each
        /// dimension runs whole, and never 32 times the threads asked for.
        #[test]
        fn dispatch_x_runs_thread_counts_as_whole_groups() {
            let test = "list::tests::dispatch_x_gpu::dispatch_x_runs_thread_counts_as_whole_groups";
            let Some(gpu) = device(test) else { return };
            let host = gpu.host();
            let (root, pso) = counting(host);
            let group = Some([8, 4, 1]);
            // Counts, flags, then the threads run and the extent they cover.
            let cases: [([u32; 3], u32, u32, [u32; 3]); 6] = [
                ([3, 2, 1], 0, 192, [24, 8, 1]),
                ([2, 1, 1], 0x8000, 64, [16, 4, 1]),
                ([20, 5, 1], THREADS, 192, [24, 8, 1]),
                ([16, 8, 1], THREADS, 128, [16, 8, 1]),
                ([1, 1, 1], THREADS, 32, [8, 4, 1]),
                ([45, 1, 1], 0x8000 | THREADS, 192, [48, 4, 1]),
            ];
            let buffers: Vec<TitleBuffer> = cases.iter().map(|_| TitleBuffer::new(256)).collect();
            let list = TitleList::new();
            let l = list.host();
            l.keep(&root);
            l.keep(&pso);
            for (&(counts, flags, ..), buffer) in cases.iter().zip(&buffers) {
                let [x, y, z] = dispatch_x_groups(counts, flags, THREADS, group);
                let to = buffer.shadow();
                to.refresh(l, buffer.address(), 256);
                to.gpu_writes(l, buffer.address(), 256);
                // SAFETY: an open list, live objects (kept until the list is
                // reset) and a root view inside the host copy.
                unsafe {
                    l.list.SetComputeRootSignature(&root);
                    l.list.SetPipelineState(&pso);
                    l.list
                        .SetComputeRootUnorderedAccessView(0, to.gpu(buffer.address()));
                    l.list.Dispatch(x, y, z);
                }
            }
            list.run();
            for (case, buffer) in cases.iter().zip(&buffers) {
                let w = buffer.words();
                assert_eq!((w[0], [w[1], w[2], w[3]]), (case.2, case.3), "{case:?}");
            }
        }
    }
}
