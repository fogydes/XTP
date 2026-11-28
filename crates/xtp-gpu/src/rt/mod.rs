//! Host DXR builds recorded alongside the title's builder commands.

mod allocation;
mod dispatch;
pub(crate) use allocation::ResultBuffer;
pub(crate) use dispatch::dispatch_descriptor_tlas_in;
pub use dispatch::{
    IndirectTlas, dispatch_descriptor_tlas, dispatch_indirect_tlas, prepare_indirect_tlas,
};
pub(crate) mod compaction;
mod console;
mod indices;
mod instances;
#[cfg(feature = "diagnostics")]
pub(crate) mod probe;

use instances::BlasMap;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::Interface;

use crate::host::{Host, HostList, uav_barrier};
use crate::object::Object;

type Inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS;
type Build = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC;
type Geometry = D3D12_RAYTRACING_GEOMETRY_DESC;

fn device(host: &Host) -> Option<&'static ID3D12Device5> {
    static DEVICE: OnceLock<Option<ID3D12Device5>> = OnceLock::new();
    DEVICE
        .get_or_init(|| {
            let mut caps = D3D12_FEATURE_DATA_D3D12_OPTIONS5::default();
            // SAFETY: the output matches the feature and byte count.
            unsafe {
                host.device.CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS5,
                    (&raw mut caps).cast(),
                    size_of_val(&caps) as u32,
                )
            }
            .ok()?;
            if caps.RaytracingTier == D3D12_RAYTRACING_TIER_NOT_SUPPORTED {
                return None;
            }
            host.device.cast().ok()
        })
        .as_ref()
}

fn buffer(rec: &Recorder<'_>, bytes: u64, state: D3D12_RESOURCE_STATES) -> Option<ID3D12Resource> {
    let device = rec.device;
    let started = tracing::enabled!(tracing::Level::DEBUG).then(std::time::Instant::now);
    if let Some(buffer) = crate::host::buffers::rt_buffer(device, bytes, state) {
        // SAFETY: the recording activates the new placed buffer before its first use.
        // Earlier recordings have released the lease on any overlapping resource.
        unsafe {
            rec.commands.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Aliasing: std::mem::ManuallyDrop::new(D3D12_RESOURCE_ALIASING_BARRIER {
                        pResourceBefore: std::mem::ManuallyDrop::new(None),
                        pResourceAfter: std::mem::transmute_copy(&buffer),
                    }),
                },
                ..Default::default()
            }]);
        }
        rec.list.keep(&buffer);
        if let Some(started) = started {
            tracing::debug!(
                bytes,
                state = state.0,
                us = started.elapsed().as_micros() as u64,
                "host DXR buffer allocation"
            );
        }
        return Some(buffer);
    }
    None
}

/// What a host DXR build builds, for the device-removed report's markers:
/// a TLAS's instances, or a BLAS's geometries, triangles and largest
/// vertex count and stride.
///
/// # Safety
/// The active layout supplies NumDescs readable CPU descriptors or pointers to them.
#[cfg(feature = "diagnostics")]
unsafe fn describe(inputs: &Inputs) -> String {
    let flags = inputs.Flags.0;
    if inputs.Type == D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL {
        // SAFETY: TOP_LEVEL selects the instance address.
        let at = unsafe { inputs.Anonymous.InstanceDescs };
        return format!(
            "TLAS of {} instances at {at:#x}, flags {flags:#x}",
            inputs.NumDescs
        );
    }
    let (mut triangles, mut vertices, mut stride) = (0u64, 0u32, 0u64);
    for i in 0..inputs.NumDescs as usize {
        // SAFETY: the caller's descriptions, in their layout.
        let g = unsafe {
            match inputs.DescsLayout {
                D3D12_ELEMENTS_LAYOUT_ARRAY if !inputs.Anonymous.pGeometryDescs.is_null() => {
                    inputs.Anonymous.pGeometryDescs.add(i).as_ref()
                }
                D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS
                    if !inputs.Anonymous.ppGeometryDescs.is_null() =>
                {
                    (*inputs.Anonymous.ppGeometryDescs.add(i)).as_ref()
                }
                _ => None,
            }
        };
        let Some(g) = g.filter(|g| g.Type == D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES) else {
            continue;
        };
        // SAFETY: the type selects Triangles.
        let t = unsafe { &g.Anonymous.Triangles };
        let points = if t.IndexCount != 0 {
            t.IndexCount
        } else {
            t.VertexCount
        };
        triangles += u64::from(points / 3);
        vertices = vertices.max(t.VertexCount);
        stride = stride.max(t.VertexBuffer.StrideInBytes);
    }
    format!(
        "BLAS of {} geometries, {triangles} triangles, up to {vertices} vertices of stride          {stride}, flags {flags:#x}",
        inputs.NumDescs
    )
}

fn width(buffer: &ID3D12Resource) -> u64 {
    // SAFETY: the borrowed COM resource is live for this synchronous descriptor query.
    unsafe { buffer.GetDesc().Width }
}

struct Recorder<'a> {
    device: &'a ID3D12Device5,
    commands: ID3D12GraphicsCommandList4,
    list: &'a HostList,
}

/// Whether the host mirrors the title's acceleration structure builds,
/// copies and rehydrations. `XTP_RT_BUILDS=0` refuses them all, a
/// diagnostic for sessions that do not test ray tracing (one of these
/// builds hung the GPU in a DRED session). The empty TLAS and the trace
/// commands stay: a trace then binds the empty TLAS and every ray misses,
/// as on the console with an empty structure. Default: on.
fn builds_on() -> bool {
    #[cfg(test)]
    if tests::BUILDS_OFF.get() {
        return false;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = builds_asked(std::env::var("XTP_RT_BUILDS").ok().as_deref());
        if !on {
            tracing::warn!(
                "XTP_RT_BUILDS=0: the title's acceleration structure builds, copies and \
                 rehydrations are refused; traces bind an empty TLAS (every ray misses)"
            );
        }
        on
    })
}

/// Whether `XTP_RT_BUILDS`'s value leaves the title's builds on: anything
/// but `0`.
fn builds_asked(value: Option<&str>) -> bool {
    value.is_none_or(|v| v.trim() != "0")
}

impl<'a> Recorder<'a> {
    /// A recorder of host DXR work into `list`, whatever `XTP_RT_BUILDS`
    /// says: the empty TLAS and the trace commands are recorded either way
    /// ([`builds_on`] gates the title's builds, copies and rehydrations).
    fn new(list: &'a HostList) -> Option<Self> {
        Some(Self {
            device: device(crate::host::made()?)?,
            commands: list.list.cast().ok()?,
            list,
        })
    }

    /// A result of `bytes` for a structure at title address `at`: the
    /// result of a generation retired there that nothing holds any more
    /// ([`Registry::take_spare`]), else a new one.
    fn result(&self, at: Option<u64>, bytes: u64) -> Option<Arc<ResultBuffer>> {
        if let Some(spare) = at.and_then(|at| registry().take_spare(at, bytes)) {
            return Some(spare);
        }
        evicting(|| ResultBuffer::new(self, bytes))
    }

    /// Records the build of `inputs` into a result of its own, reusing one
    /// retired at title address `at` when it can.
    fn build(&self, inputs: &Inputs, at: Option<u64>, ordered: bool) -> Option<Structure> {
        let mut sizes = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
        // SAFETY: the caller owns all geometry descriptions for this call.
        unsafe {
            self.device
                .GetRaytracingAccelerationStructurePrebuildInfo(inputs, &mut sizes)
        };
        if sizes.ResultDataMaxSizeInBytes == 0 {
            return None;
        }
        let result = self.result(at, sizes.ResultDataMaxSizeInBytes)?;
        let mut scratch = self
            .list
            .rt_scratch
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let bytes = sizes.ScratchDataSizeInBytes;
        let mut scratch_address = 0;
        if bytes != 0 {
            if scratch.as_ref().is_none_or(|buffer| width(buffer) < bytes) {
                *scratch = Some(evicting(|| {
                    buffer(self, bytes, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
                })?);
            }
            let current = scratch.as_ref()?;
            self.list.keep(current);
            // SAFETY: current is a buffer in UAV state of at least the host prebuild scratch size.
            // The list kept this exact allocation even if its scratch slot is replaced later.
            scratch_address = unsafe { current.GetGPUVirtualAddress() };
        }
        let postbuild = inputs
            .Flags
            .contains(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION)
            .then(|| compaction::request(self.list))
            .flatten();
        self.list.keep_rt_result(&result);
        #[cfg(feature = "diagnostics")]
        self.list.note_op(|| {
            // SAFETY: the caller owns the geometry descriptions for this call.
            let what = unsafe { describe(inputs) };
            let at = at.map_or_else(|| "none".into(), |at| format!("{at:#x}"));
            format!(
                "host DXR build {what}, title address {at}, result {} at {:#x}, scratch {}",
                sizes.ResultDataMaxSizeInBytes,
                result.gpu(),
                sizes.ScratchDataSizeInBytes
            )
        });
        // SAFETY: all GPU ranges are owned by this recording. The global barriers order prior
        // geometry writers, scratch reuse on the host queue and subsequent reads of the result.
        unsafe {
            if !ordered {
                self.commands.ResourceBarrier(&[uav_barrier()]);
            }
            self.commands.BuildRaytracingAccelerationStructure(
                &Build {
                    Inputs: *inputs,
                    DestAccelerationStructureData: result.gpu(),
                    ScratchAccelerationStructureData: scratch_address,
                    SourceAccelerationStructureData: 0,
                },
                postbuild
                    .as_ref()
                    .map(|(desc, _)| std::slice::from_ref(desc)),
            );
            self.commands.ResourceBarrier(&[uav_barrier()]);
        }
        Some(Structure {
            producer: Producer::recorded(self.list),
            result,
            instances: None,
            compacted_size: postbuild.map(|(_, size)| size),
            holds: None,
        })
    }

    /// Records a copy of `source` into a result of its own, reusing one
    /// retired at title address `at` when it can.
    fn copy_structure(
        &self,
        source: &Structure,
        mode: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE,
        at: u64,
    ) -> Option<Structure> {
        let capacity = source.result.bytes;
        match mode {
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_CLONE => {}
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT
                if source.compacted_size.is_some() => {}
            _ => return None,
        }
        // A replay may rebuild different geometry into the same source.
        // Its allocation bounds every execution, unlike a prior fence's measurement.
        let bytes = capacity;
        let dest = self.result(Some(at), bytes)?;
        self.list.keep_rt_result(&dest);
        #[cfg(feature = "diagnostics")]
        self.list.note_op(|| {
            format!(
                "host DXR copy mode {} of {:#x} to {:#x}, title address {at:#x}, {bytes} bytes",
                mode.0,
                source.result.gpu(),
                dest.gpu()
            )
        });
        // SAFETY: independent AS resources in the required state, retained through execution.
        unsafe {
            self.commands.CopyRaytracingAccelerationStructure(
                dest.gpu(),
                source.result.gpu(),
                mode,
            );
            self.commands.ResourceBarrier(&[uav_barrier()]);
        }
        Some(Structure {
            producer: Producer::recorded(self.list),
            instances: source.instances.as_ref().map(|map| map.capture(self.list)),
            result: dest,
            ..source.clone()
        })
    }
}

struct Producer {
    recording: Arc<()>,
    submitted: AtomicBool,
}

impl Producer {
    fn recorded(list: &HostList) -> Arc<Self> {
        let producer = Arc::new(Self {
            recording: list.rt_recording(),
            submitted: AtomicBool::new(false),
        });
        let mark = producer.clone();
        // Host submissions are serialized. Deferred steps run in list order before ExecuteCommandLists.
        list.defer_host((0, 0), move || {
            mark.submitted.store(true, Ordering::Release)
        });
        producer
    }

    fn available(&self, recording: &Arc<()>) -> bool {
        Arc::ptr_eq(&self.recording, recording) || self.submitted.load(Ordering::Acquire)
    }
}

#[derive(Clone)]
struct Structure {
    producer: Arc<Producer>,
    result: Arc<ResultBuffer>,
    // Only TLAS generations refer to a snapshot of the BLAS address space.
    instances: Option<Arc<BlasMap>>,
    compacted_size: Option<Arc<AtomicU64>>,
    /// A TLAS's stand-in BLAS, which its instances of a BLAS the host does
    /// not have name ([`empty_tlas`]), or that of the empty TLAS it is.
    holds: Option<Arc<ResultBuffer>>,
}

impl Structure {
    fn retain(&self, list: &HostList) {
        list.keep_rt_result(&self.result);
        if let Some(blas) = &self.holds {
            list.keep_rt_result(blas);
        }
        if let Some(map) = &self.instances {
            map.capture(list);
        }
    }
}

/// The last structure actually reached in queue order at this title address.
type Slot = Arc<Mutex<Option<Structure>>>;
type WeakSlot = std::sync::Weak<Mutex<Option<Structure>>>;
type Candidates = Arc<BTreeMap<u64, Slot>>;

struct BuildSlot {
    known_extent: bool,
    generation: u64,
    slot: Slot,
    overwritten: Vec<Slot>,
}

impl BuildSlot {
    fn submit(&self, list: &HostList, built: Option<Structure>) {
        let slot = self.slot.clone();
        let overwritten = self.overwritten.clone();
        list.defer_host((0, 0), move || {
            for old in &overwritten {
                *old.lock().unwrap_or_else(|e| e.into_inner()) = None;
            }
            *slot.lock().unwrap_or_else(|e| e.into_inner()) = built.clone();
        });
    }
}

struct Destination {
    slot: Slot,
    end: u64,
    owner: Option<usize>,
    generation: u64,
}

/// The result of a generation retired at a title address, kept for a later
/// generation there, and the frame it was retired in.
struct Spare {
    result: Arc<ResultBuffer>,
    frame: u64,
}

/// Spares kept per title address: a structure rebuilt every frame retires
/// one generation per frame, each held by its slot until the next one is
/// submitted and by its recording until that has run and been reset.
const SPARES: usize = 4;
/// Frames a spare is kept: an address no longer rebuilt gives its spares
/// back.
const SPARE_FRAMES: u64 = 8;

#[derive(Default)]
struct Registry {
    structures: BTreeMap<u64, Structure>,
    // Includes builds still recording, so removal also cancels their publication.
    ranges: BTreeMap<u64, Destination>,
    waiting: BTreeMap<u64, (Option<usize>, WeakSlot)>,
    generation: u64,
    snapshot: Option<Candidates>,
    /// Results of retired generations by title address, oldest first.
    spares: BTreeMap<u64, Vec<Spare>>,
    /// The frame spares were last aged in.
    aged: u64,
}

fn registry() -> std::sync::MutexGuard<'static, Registry> {
    static TABLE: std::sync::LazyLock<Mutex<Registry>> = std::sync::LazyLock::new(Default::default);
    TABLE.lock().unwrap_or_else(|e| e.into_inner())
}

impl Registry {
    fn trace_slot(&mut self, address: u64) -> Slot {
        if let Some(range) = self.ranges.get(&address) {
            return range.slot.clone();
        }
        if let Some(slot) = self.waiting.get(&address).and_then(|(_, w)| w.upgrade()) {
            return slot;
        }
        self.waiting.retain(|_, (_, w)| w.strong_count() != 0);
        let slot = Slot::default();
        self.waiting.insert(
            address,
            (
                crate::shadow::buffer_owner(address, 1).map(|o| o.0),
                Arc::downgrade(&slot),
            ),
        );
        slot
    }

    fn retire(
        &mut self,
        start: u64,
        end: u64,
        owner: Option<Option<usize>>,
    ) -> (Option<Structure>, Vec<Slot>) {
        if end <= start {
            return (None, Vec::new());
        }
        let waiting: Vec<_> = self
            .waiting
            .range(start..end)
            .filter(|(_, (o, _))| owner.is_none_or(|owner| owner == *o))
            .map(|(&key, _)| key)
            .collect();
        for key in waiting {
            self.waiting.remove(&key);
        }
        let first = self
            .ranges
            .range(..start)
            .next_back()
            .filter(|(_, r)| r.end > start)
            .map_or(start, |(&key, _)| key);
        let keys: Vec<_> = self
            .ranges
            .range(first..end)
            .filter(|(_, r)| owner.is_none_or(|o| o == r.owner))
            .map(|(&key, _)| key)
            .collect();
        let mut previous = None;
        let mut slots = Vec::new();
        for key in keys {
            if let Some(range) = self.ranges.remove(&key) {
                slots.push(range.slot);
            }
            let old = self.structures.remove(&key);
            if old.as_ref().is_some_and(|s| s.instances.is_none()) {
                self.snapshot = None;
            }
            if key == start {
                previous = old;
            }
        }
        // Spares in memory the title released or now uses for another range.
        let spares: Vec<_> = self.spares.range(start..end).map(|(&key, _)| key).collect();
        for key in spares {
            self.spares.remove(&key);
        }
        (previous, slots)
    }

    /// Keeps `result`, of a generation retired at title address `at`, for a
    /// later one there.
    fn keep_spare(&mut self, at: u64, result: Arc<ResultBuffer>) {
        let spares = self.spares.entry(at).or_default();
        if spares.len() == SPARES {
            spares.remove(0);
        }
        spares.push(Spare {
            result,
            frame: self.aged,
        });
    }

    /// A result of `bytes` (at most twice as many) retired at title address
    /// `at` that nothing else holds: no slot, no TLAS's BLAS map and no
    /// recording, which holds it until it has run and been reset. Every GPU
    /// use of the old generation has then finished.
    fn take_spare(&mut self, at: u64, bytes: u64) -> Option<Arc<ResultBuffer>> {
        let spares = self.spares.get_mut(&at)?;
        let (i, _) = spares
            .iter()
            .enumerate()
            .filter(|(_, s)| {
                (bytes..=bytes.saturating_mul(2)).contains(&s.result.bytes)
                    && Arc::strong_count(&s.result) == 1
            })
            .min_by_key(|(_, s)| s.result.bytes)?;
        let spare = spares.remove(i);
        if spares.is_empty() {
            self.spares.remove(&at);
        }
        Some(spare.result)
    }

    /// Gives back the spares kept since more than [`SPARE_FRAMES`] before
    /// `frame`, once per frame.
    fn age_spares(&mut self, frame: u64) {
        if frame == self.aged {
            return;
        }
        self.aged = frame;
        self.spares.retain(|_, spares| {
            spares.retain(|s| frame.saturating_sub(s.frame) <= SPARE_FRAMES);
            !spares.is_empty()
        });
    }

    fn begin(&mut self, start: u64, bytes: Option<u64>) -> BuildSlot {
        let owner = crate::shadow::buffer_owner(start, bytes.unwrap_or(1));
        let end = match (bytes.filter(|&n| n > 0), owner) {
            (Some(n), _) => start.saturating_add(n),
            (None, Some((_, end))) => end,
            (None, None) => start.saturating_add(1),
        };
        let slot = self
            .ranges
            .get(&start)
            .map(|r| r.slot.clone())
            .or_else(|| self.waiting.remove(&start).and_then(|(_, w)| w.upgrade()))
            .unwrap_or_default();
        self.age_spares(crate::frame::current());
        // The spares at this address serve the new generation.
        let kept = self.spares.remove(&start);
        let (old, mut overwritten) = self.retire(start, end, None);
        if let Some(kept) = kept {
            self.spares.insert(start, kept);
        }
        // Its result serves a later generation, once its slot has the new
        // one and its recording has run.
        if let Some(old) = old {
            self.keep_spare(start, old.result);
        }
        overwritten.retain(|s| !Arc::ptr_eq(s, &slot));
        self.generation += 1;
        self.ranges.insert(
            start,
            Destination {
                slot: slot.clone(),
                end,
                owner: owner.map(|o| o.0),
                generation: self.generation,
            },
        );
        BuildSlot {
            known_extent: bytes
                .map_or(owner.is_some(), |n| n > 0 && start.checked_add(n).is_some()),
            generation: self.generation,
            slot,
            overwritten,
        }
    }

    fn finish(&mut self, start: u64, generation: u64, built: Option<Structure>) -> bool {
        if self
            .ranges
            .get(&start)
            .is_none_or(|r| r.generation != generation)
        {
            return false;
        }
        let Some(built) = built else {
            self.ranges.remove(&start);
            return false;
        };
        if built.instances.is_none() {
            self.snapshot = None;
        }
        self.structures.insert(start, built);
        true
    }

    fn blas_candidates(&mut self) -> Candidates {
        self.snapshot
            .get_or_insert_with(|| {
                Arc::new(
                    self.structures
                        .iter()
                        .filter(|(_, s)| s.instances.is_none())
                        .map(|(&addr, _)| (addr, self.ranges[&addr].slot.clone()))
                        .collect(),
                )
            })
            .clone()
    }
}

/// Retires the structures of title memory `[start, end)` that `owner`
/// released, and empties their slots: a TLAS recorded before, or a trace
/// of one, finds nothing there (its instances are inactive, its rays miss)
/// instead of a structure in released memory.
pub(crate) fn release_buffer(owner: Option<usize>, start: u64, end: u64) {
    let (previous, slots) = registry().retire(start, end, Some(owner));
    drop(previous);
    for slot in slots {
        let old = slot.lock().unwrap_or_else(|e| e.into_inner()).take();
        drop(old);
    }
}

/// `make`, tried again once the spares are given back ([`Registry::spares`])
/// if it fails: the RT budget is exhausted, and spares hold part of it.
fn evicting<T>(make: impl Fn() -> Option<T>) -> Option<T> {
    if let Some(made) = make() {
        return Some(made);
    }
    let spares = std::mem::take(&mut registry().spares);
    // The others are freed when what holds them goes.
    let free = spares
        .values()
        .flatten()
        .filter(|s| Arc::strong_count(&s.result) == 1)
        .count();
    drop(spares);
    if free == 0 {
        return None;
    }
    tracing::debug!(
        free,
        "host DXR allocation refused: spare results given back, tried again"
    );
    make()
}

/// Resolves and pins the TLAS and its possible BLAS references for this recording.
///
/// # Safety
/// `list` is a live title command list.
pub unsafe fn host_tlas(list: *const Object, title_tlas: u64) -> Option<u64> {
    host_tlas_in(crate::list::host_list(list)?, title_tlas)
}

/// [`host_tlas`] for the host recording itself.
pub(crate) fn host_tlas_in(list: &HostList, title_tlas: u64) -> Option<u64> {
    let entry = registry().structures.get(&title_tlas)?.clone();
    entry.instances.as_ref()?;
    if !entry.producer.available(&list.rt_recording()) {
        return None;
    }
    entry.retain(list);
    // SAFETY: the list owns this resource until the GPU has finished the recording.
    Some(entry.result.gpu())
}

/// A TLAS with nothing to hit, valid for this recording: every ray misses
/// it, as on the console with an empty structure. A trace binds it where no
/// TLAS is published: the program still runs and writes its misses (a null
/// root SRV instead removed the WARP device). Its one instance has mask 0,
/// which no ray's mask selects, over a real BLAS of one triangle: WARP
/// divided by zero building a TLAS of no instance, and faulted on an
/// instance of a null BLAS. One is built once and shared once its recording
/// has reached the queue; until then a recording builds its own here.
///
/// The TLAS and its BLAS, which the recording keeps: a title TLAS names
/// that BLAS, with mask 0, for an instance of a BLAS the host does not have
/// ([`instances::rewrite`]), and is that TLAS when it has no instance.
fn empty_tlas(rec: &Recorder<'_>) -> Option<(Arc<ResultBuffer>, Arc<ResultBuffer>)> {
    #[cfg(test)]
    if tests::EMPTY_FAILS.get() {
        return None;
    }
    /// The TLAS and the BLAS its instance names, which must outlive it.
    static EMPTY: Mutex<Option<(Structure, Arc<ResultBuffer>)>> = Mutex::new(None);
    let mut shared = EMPTY.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((tlas, blas)) = shared.as_ref()
        && tlas.producer.available(&rec.list.rt_recording())
    {
        rec.list.keep_rt_result(blas);
        rec.list.keep_rt_result(&tlas.result);
        return Some((tlas.result.clone(), blas.clone()));
    }
    let built = (|| {
        // Three vertices at the origin, then the instance record.
        let (vertices, data) = rec.list.scratch(36)?;
        let (instance, record) = rec.list.scratch(64)?;
        // SAFETY: 36 bytes of this recording's upload memory.
        unsafe { std::ptr::write_bytes(data, 0, 36) };
        let geometry = Geometry {
            Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
            Flags: D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
            Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                    Transform3x4: 0,
                    IndexFormat: DXGI_FORMAT_UNKNOWN,
                    VertexFormat: DXGI_FORMAT_R32G32B32_FLOAT,
                    IndexCount: 0,
                    VertexCount: 3,
                    IndexBuffer: 0,
                    VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                        StartAddress: vertices,
                        StrideInBytes: 12,
                    },
                },
            },
        };
        let blas = rec.build(
            &Inputs {
                Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
                Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_NONE,
                NumDescs: 1,
                DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
                Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                    pGeometryDescs: &geometry,
                },
            },
            None,
            false,
        )?;
        let address = blas.result.gpu();
        // An identity transform, ID and mask 0, the BLAS address last.
        let words: [u32; 16] = std::array::from_fn(|i| match i {
            0 | 5 | 10 => 1f32.to_bits(),
            14 => address as u32,
            15 => (address >> 32) as u32,
            _ => 0,
        });
        // SAFETY: 64 bytes of this recording's upload memory, 256-byte aligned.
        unsafe { record.cast::<[u32; 16]>().write(words) };
        // The BLAS build's trailing barrier orders it before this one.
        let tlas = rec.build(
            &Inputs {
                Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
                Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_NONE,
                NumDescs: 1,
                DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
                Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                    InstanceDescs: instance,
                },
            },
            None,
            true,
        )?;
        Some((tlas, blas.result))
    })();
    let Some((tlas, blas)) = built else {
        tracing::warn!("no empty TLAS: traces without a published TLAS are dropped");
        return None;
    };
    // Recordings that took an earlier one keep it; later ones wait for this
    // one's recording, which a recording that is never submitted cannot block.
    let result = (tlas.result.clone(), blas.clone());
    *shared = Some((tlas, blas));
    Some(result)
}

/// The GPU address of an empty TLAS ([`empty_tlas`]) for `list`, to bind
/// where a trace has no TLAS: every ray misses. Its build may be recorded
/// here, so call it before binding the trace's PSO and root arguments.
pub(crate) fn empty_tlas_in(list: &HostList) -> Option<u64> {
    Some(empty_tlas(&Recorder::new(list)?)?.0.gpu())
}

/// Compatibility name for the recording-scoped lookup.
pub use host_tlas as host_tlas_for_list;

/// Resolves one title descriptor slot and retains the TLAS for this dispatch.
/// Call separately for every slot, including distinct offsets in one descriptor table.
///
/// # Safety
/// `list` is a live title command list and `handle` is a readable title descriptor slot.
pub unsafe fn descriptor_tlas(list: *const Object, handle: u64) -> Option<u64> {
    descriptor_tlas_in(crate::list::host_list(list)?, handle)
}

/// [`descriptor_tlas`] for the host recording itself: the TLAS pinned for
/// `list`, once its producer is recorded there or submitted. `handle` is a
/// readable title descriptor slot.
pub(crate) fn descriptor_tlas_in(list: &HostList, handle: u64) -> Option<u64> {
    host_tlas_in(list, descriptor_address(handle)?)
}

fn descriptor_address(handle: u64) -> Option<u64> {
    use crate::descriptor::{View, host::buffer_range, view};
    let view = view(handle)?;
    let words = match &view {
        View::Shader(_, Some(words)) => &words[..],
        View::Placed(false, _, words) => &words[..],
        _ => return None,
    };
    Some(match D3D12_SRV_DIMENSION(words[1] as i32) {
        D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE => {
            u64::from(words[4]) | u64::from(words[5]) << 32
        }
        D3D12_SRV_DIMENSION_BUFFER => buffer_range(&view)?.0,
        _ => return None,
    })
}

/// What mirroring one title build did.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Mirrored {
    /// The structure is published at its title address.
    registered: bool,
    /// A build was recorded, and so its trailing UAV barrier, which orders
    /// the next build of a batch: a TLAS of no instance is the shared empty
    /// TLAS and records none.
    barrier: bool,
}

/// Mirrors a title build and invalidates an older destination on failure.
///
/// # Safety
/// `title_list` is a live title list and the active CPU arrays follow the builder ABI.
unsafe fn build_one(
    title_list: *const Object,
    desc: &Build,
    ordered: bool,
    title_bytes: Option<u64>,
) -> Mirrored {
    let dst = desc.DestAccelerationStructureData;
    if dst == 0 || !builds_on() {
        return Mirrored {
            registered: false,
            barrier: false,
        };
    }
    let generation = registry().begin(dst, title_bytes);
    let barrier = std::cell::Cell::new(false);
    let built = (|| {
        if !generation.known_extent {
            return None;
        }
        // The title supplies full geometry on refits. Rebuilding it also handles
        // changed index data without relying on DXR's fixed-topology update contract.
        let mut inputs = desc.Inputs;
        inputs.Flags &= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE
            | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION
            | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE
            | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD
            | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_MINIMIZE_MEMORY;
        let list = crate::list::host_list(title_list)?;
        let recorder = Recorder::new(list)?;
        let _timing =
            crate::timestamps::command(list, crate::timestamps::What::RaytracingBuild, None);
        match inputs.Type {
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL => {
                // SAFETY: the descriptors are borrowed from this title call only.
                let (mut geometries, jobs) = unsafe { triangles(list, &inputs) }?;
                if !jobs.is_empty() {
                    // Its own barriers order what came before the build.
                    indices::record(&recorder, title_list, &mut geometries, &jobs, dst)?;
                }
                let ordered = ordered || !jobs.is_empty();
                let inputs = Inputs {
                    DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
                    Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                        pGeometryDescs: geometries.as_ptr(),
                    },
                    ..inputs
                };
                let built = recorder.build(&inputs, Some(dst), ordered)?;
                barrier.set(true);
                Some(built)
            }
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL => {
                // The empty TLAS traces fall back to, made before they bind
                // their state, and its BLAS, which stands in for the BLAS of
                // an instance the host does not have. Without them, such an
                // instance would name a null BLAS and a TLAS of no instance
                // would be built as one, the two cases they stand in for
                // (WARP faults on both): the TLAS is refused.
                let Some((empty, stand_in)) = empty_tlas(&recorder) else {
                    tracing::warn!(
                        address = format_args!("{dst:#x}"),
                        "host DXR TLAS refused: no empty TLAS to stand in for its missing BLAS"
                    );
                    return None;
                };
                let (inputs, map) =
                    instances::rewrite(&recorder, title_list, &inputs, stand_in.gpu())?;
                if inputs.NumDescs == 0 {
                    // WARP divides by zero building a TLAS of no instance:
                    // the empty TLAS, which no ray hits either.
                    return Some(Structure {
                        producer: Producer::recorded(list),
                        result: empty,
                        instances: Some(map),
                        compacted_size: None,
                        holds: Some(stand_in),
                    });
                }
                // The rewrite's barriers order what came before this build.
                let built = recorder.build(&inputs, Some(dst), true)?;
                barrier.set(true);
                Some(Structure {
                    instances: Some(map),
                    holds: Some(stand_in),
                    ..built
                })
            }
            _ => None,
        }
    })();
    if built.is_none() {
        let top = desc.Inputs.Type == D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
        refused(if top { Refused::Tlas } else { Refused::Blas }, dst);
    }
    if let Some(list) = crate::list::host_list(title_list) {
        generation.submit(list, built.clone());
    }
    Mirrored {
        registered: registry().finish(dst, generation.generation, built),
        barrier: barrier.get(),
    }
}

/// What the host could not record at a title address.
#[derive(Clone, Copy)]
enum Refused {
    Blas,
    Tlas,
    Copy,
}

/// Logs a build or copy the host could not record, at powers of two of
/// each kind. Its title address then holds no structure in queue order:
/// an instance of a BLAS there is inactive, and a trace of a TLAS there
/// runs with a null one, where every ray misses.
fn refused(what: Refused, address: u64) {
    static COUNTS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
    let refused = COUNTS[what as usize].fetch_add(1, Ordering::Relaxed) + 1;
    if refused.is_power_of_two() {
        let what = match what {
            Refused::Blas => "BLAS build",
            Refused::Tlas => "TLAS build (its traces run with a null TLAS: every ray misses)",
            Refused::Copy => "structure copy",
        };
        tracing::warn!(
            address = format_args!("{address:#x}"),
            refused,
            "host DXR {what} refused: its title address holds no structure (the budget, input \
             and extent warnings before it say why)"
        );
    }
}

/// Rebuilds a rehydrated title entry and optionally samples its inputs on the GPU.
///
/// # Safety
/// The title list and CPU arrays obey the same contract as `build`.
pub unsafe fn rehydrate(list: *const Object, desc: &Build) -> bool {
    // SAFETY: the caller supplies a live list and descriptions.
    unsafe { rehydrate_sized(list, desc, None) }
}

/// Rehydrates a structure with the result size supplied by the title builder.
///
/// # Safety
/// The list and CPU arrays obey the same contract as `rehydrate`.
pub unsafe fn rehydrate_sized(list: *const Object, desc: &Build, title_bytes: Option<u64>) -> bool {
    // SAFETY: the caller supplies the same live inputs required by build.
    let recorded = unsafe { build_one(list, desc, false, title_bytes) }.registered;
    #[cfg(feature = "diagnostics")]
    if probe::enabled() {
        let sampled = recorded
            && crate::list::host_list(list).is_some_and(|list| {
                // SAFETY: the CPU descriptions remain borrowed for this entire rehydration call.
                unsafe { probe::record(list, desc).is_some() }
            });
        tracing::info!(recorded, sampled, "rehydrated geometry validation");
    }
    recorded
}

/// A title builder records into the address space shared by every builder on the host device.
#[derive(Default)]
pub struct Builder;

impl Builder {
    /// Mirrors a single title build.
    ///
    /// # Safety
    /// `list` is a live title command list and the CPU geometry arrays are valid for the call.
    pub unsafe fn build(&self, list: *const Object, desc: &Build) -> bool {
        // SAFETY: the caller supplies the list and geometry arrays.
        unsafe { build_one(list, desc, false, None) }.registered
    }

    /// Records a contiguous batch. The preceding build's final UAV barrier also
    /// orders scratch reuse and AS dependencies for the following entry; an
    /// entry that recorded no build (a TLAS of no instance, a refused one)
    /// leaves the next its own entry barrier.
    ///
    /// # Safety
    /// The list and descriptors obey the single-build contract. No external GPU commands
    /// may be recorded into the list while iterating the batch.
    pub unsafe fn build_batch<'a>(
        &self,
        list: *const Object,
        descs: impl Iterator<Item = &'a Build>,
    ) {
        // SAFETY: the caller supplies a contiguous recording with live descriptors.
        unsafe { self.build_batch_sized(list, descs.map(|d| (d, None))) }
    }

    /// Records a batch using the title builder's destination sizes.
    ///
    /// # Safety
    /// The list and descriptors obey the same contract as `build_batch`.
    pub unsafe fn build_batch_sized<'a>(
        &self,
        list: *const Object,
        descs: impl Iterator<Item = (&'a Build, Option<u64>)>,
    ) {
        let mut ordered = false;
        for (desc, title_bytes) in descs {
            // SAFETY: the batch supplies live inputs. Only a recorded build leaves a trailing barrier.
            ordered = unsafe { build_one(list, desc, ordered, title_bytes) }.barrier;
        }
    }

    /// Mirrors a clone or compact copy across title builders.
    ///
    /// # Safety
    /// `list` is a live title list.
    pub unsafe fn copy(&self, list: *const Object, dst: u64, src: u64, mode: i32) -> bool {
        if dst == 0 || !builds_on() {
            return false;
        }
        let mode = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE(mode);
        let (source, generation) = {
            let mut table = registry();
            let source = table.structures.get(&src).cloned();
            // A clone takes its source's title extent. A compacted copy is
            // smaller by the title builder's measure, which the call does
            // not give: the title packs the next one right after it. Such a
            // copy, or one of a source with no extent (its build refused),
            // claims its first byte only, which retires neither its packed
            // neighbours nor the rest of the buffer.
            let bytes = match mode {
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_CLONE => {
                    table.ranges.get(&src).map(|range| range.end - src)
                }
                _ => None,
            };
            let generation = table.begin(dst, Some(bytes.unwrap_or(1)));
            (source, generation)
        };
        let copied = (|| {
            if !generation.known_extent {
                return None;
            }
            let source = source?;
            let recorder = Recorder::new(crate::list::host_list(list)?)?;
            if !source.producer.available(&recorder.list.rt_recording()) {
                return None;
            }
            source.retain(recorder.list);
            recorder.copy_structure(&source, mode, dst)
        })();
        if copied.is_none() {
            refused(Refused::Copy, dst);
        }
        if let Some(list) = crate::list::host_list(list) {
            generation.submit(list, copied.clone());
        }
        registry().finish(dst, generation.generation, copied)
    }
}

fn input_range(
    list: &HostList,
    address: u64,
    bytes: u64,
    alignment: u64,
) -> Option<Arc<crate::shadow::Shadow>> {
    let end = address.checked_add(bytes)?;
    if address == 0 || !address.is_multiple_of(alignment) || bytes == 0 {
        return None;
    }
    let holders = crate::shadow::holders(address);
    if holders.is_empty() {
        // The streaming allocator also passes plain title memory. Reject gaps
        // before asking for the same loose shadow the title's GPU writers use.
        if !readable(address, end) {
            return None;
        }
    } else if !holders.iter().any(|&(base, len)| end <= base + len) {
        return None;
    }
    let shadow = crate::shadow::find(address, bytes)?;
    let gpu = shadow.gpu(address);
    if !gpu.is_multiple_of(alignment) {
        return None;
    }
    shadow.refresh(list, address, bytes);
    Some(shadow)
}

fn readable(mut address: u64, end: u64) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE,
        PAGE_EXECUTE_WRITECOPY, PAGE_GUARD, PAGE_READONLY, PAGE_READWRITE, PAGE_WRITECOPY,
        VirtualQuery,
    };
    while address < end {
        let mut info = MEMORY_BASIC_INFORMATION::default();
        // SAFETY: VirtualQuery inspects the address without dereferencing it and fills this local.
        let size = unsafe { VirtualQuery(address as _, &mut info, size_of_val(&info)) };
        if size == 0
            || info.State != MEM_COMMIT
            || info.Protect & PAGE_GUARD != 0
            || !matches!(
                info.Protect & 0xff,
                PAGE_READONLY
                    | PAGE_READWRITE
                    | PAGE_WRITECOPY
                    | PAGE_EXECUTE_READ
                    | PAGE_EXECUTE_READWRITE
                    | PAGE_EXECUTE_WRITECOPY
            )
        {
            return false;
        }
        let next = (info.BaseAddress as u64).saturating_add(info.RegionSize as u64);
        if next <= address {
            return false;
        }
        address = next;
    }
    true
}

/// Logs a title BLAS input refused before its build is recorded (counted;
/// at powers of two): the BLAS build is refused with it.
fn invalid<T>(geometry: usize, why: std::fmt::Arguments<'_>) -> Option<T> {
    static INVALID: AtomicU64 = AtomicU64::new(0);
    let invalid = INVALID.fetch_add(1, Ordering::Relaxed) + 1;
    if invalid.is_power_of_two() {
        tracing::warn!(
            geometry,
            invalid,
            "host DXR BLAS input refused before its build is recorded: {why}"
        );
    }
    None
}

/// Whether `shadow`'s buffer holds `bytes` from its host address `host`.
fn holds(shadow: &crate::shadow::Shadow, host: u64, bytes: u64) -> bool {
    // SAFETY: plain queries on a live buffer.
    let (base, width) = unsafe {
        (
            shadow.buffer.GetGPUVirtualAddress(),
            shadow.buffer.GetDesc().Width,
        )
    };
    host.checked_add(bytes)
        .is_some_and(|end| host >= base && end <= base + width)
}

/// Translates triangle descriptions for a host build. Each input of each
/// geometry (indices, vertices, transform) becomes a job of the pass that
/// makes it in the build's own buffer, where the build reads it
/// ([`indices`]); the pass sets the geometry's addresses when it is
/// recorded. The console builder's own formats are decoded there
/// ([`console`]): `R16_TYPELESS` indices are meshlet-packed, `R32G32_FLOAT`
/// vertices quantized, and the vertex stride is 15 bits and a flag.
///
/// Each input is checked first, and a BLAS with an invalid one is refused
/// with a log ([`invalid`]): a geometry that is not triangles or has an
/// unknown format, a vertex stride shorter than a vertex or past the
/// builder's 16 bits, a vertex or index count that is no whole number of
/// triangles, vertices (with their header), indices, packed indices'
/// parts or a transform outside title memory or misaligned, or more
/// geometries or triangles than DXR allows. Indices past the vertices,
/// decoded ones included, are checked on the GPU, in the version the build
/// reads ([`indices`]).
///
/// # Safety
/// The active layout supplies NumDescs readable CPU descriptors or pointers to them.
unsafe fn triangles(
    list: &HostList,
    inputs: &Inputs,
) -> Option<(Vec<Geometry>, Vec<indices::Job>)> {
    use indices::{Input, Job, Kind};
    let count = inputs.NumDescs;
    if count > D3D12_RAYTRACING_MAX_GEOMETRIES_PER_BOTTOM_LEVEL_ACCELERATION_STRUCTURE {
        return invalid(0, format_args!("{count} geometries"));
    }
    let mut geometries = Vec::new();
    let mut jobs = Vec::new();
    let mut total = 0u64;
    for i in 0..count as usize {
        // SAFETY: the caller provides NumDescs CPU descriptions in the selected layout.
        let g = unsafe {
            let p = match inputs.DescsLayout {
                D3D12_ELEMENTS_LAYOUT_ARRAY if !inputs.Anonymous.pGeometryDescs.is_null() => {
                    inputs.Anonymous.pGeometryDescs.add(i)
                }
                D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS
                    if !inputs.Anonymous.ppGeometryDescs.is_null() =>
                {
                    *inputs.Anonymous.ppGeometryDescs.add(i)
                }
                _ => return invalid(i, format_args!("no geometry descriptions")),
            };
            p.as_ref().copied()
        };
        let Some(mut g) = g else {
            return invalid(i, format_args!("a null geometry description"));
        };
        if g.Type != D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES {
            return invalid(
                i,
                format_args!("geometry type {} (triangles only)", g.Type.0),
            );
        }
        // SAFETY: the geometry type selects Triangles in our own copy.
        let t = unsafe { &mut g.Anonymous.Triangles };
        // The builder reads 15 bits of stride; bit 15 turns packed triangles.
        let raw = t.VertexBuffer.StrideInBytes;
        if raw > console::STRIDE | console::FLIP {
            return invalid(
                i,
                format_args!("vertex stride {raw:#x}, past the builder's 16 bits"),
            );
        }
        let (stride, flip) = (raw & console::STRIDE, raw & console::FLIP != 0);
        let quantized = t.VertexFormat == DXGI_FORMAT_R32G32_FLOAT;
        let (width, alignment) = match t.VertexFormat {
            DXGI_FORMAT_R32G32B32_FLOAT => (12, 4),
            // Quantized, read a word at a time.
            DXGI_FORMAT_R32G32_FLOAT => (console::QUANTIZED_VERTEX, 4),
            // DXR ignores A16; XYZ may be packed at a six-byte stride and
            // the final vertex need not have a fourth component.
            // https://learn.microsoft.com/windows/win32/api/d3d12/ns-d3d12-d3d12_raytracing_geometry_triangles_desc
            DXGI_FORMAT_R16G16B16A16_FLOAT | DXGI_FORMAT_R16G16B16A16_SNORM => (6, 2),
            DXGI_FORMAT_R16G16_FLOAT | DXGI_FORMAT_R16G16_SNORM => (4, 2),
            format => return invalid(i, format_args!("vertex format {}", format.0)),
        };
        if stride < width || !stride.is_multiple_of(alignment) {
            return invalid(
                i,
                format_args!("vertex stride {stride} for vertices of {width} bytes"),
            );
        }
        let vertices = t.VertexCount;
        let Some(bytes) = vertices
            .checked_sub(1)
            .and_then(|last| u64::from(last).checked_mul(stride)?.checked_add(width))
        else {
            return invalid(i, format_args!("no vertex"));
        };
        let start = t.VertexBuffer.StartAddress;
        if quantized {
            // From their header.
            let window = start.checked_sub(console::QUANTIZED_HEADER).and_then(|at| {
                let shadow = input_range(list, at, bytes + console::QUANTIZED_HEADER, 4)?;
                Some((shadow.gpu(at), shadow))
            });
            let Some((source, shadow)) = window else {
                return invalid(
                    i,
                    format_args!(
                        "quantized vertices {start:#x}+{bytes:#x} or their header outside title \
                         memory or misaligned"
                    ),
                );
            };
            jobs.push(Job {
                geometry: i,
                input: Input::Vertices {
                    format: DXGI_FORMAT_R32G32B32_FLOAT,
                    stride: 12,
                    skew: 0,
                },
                kind: Kind::Quantized {
                    stride: stride as u32,
                },
                shadow,
                source,
                count: vertices,
                vertices: 0,
            });
        } else {
            let Some(shadow) = input_range(list, start, bytes, alignment) else {
                return invalid(
                    i,
                    format_args!(
                        "vertices {start:#x}+{bytes:#x} outside title memory or misaligned"
                    ),
                );
            };
            // Copied a word at a time: 2-byte aligned ones start inside one.
            let host = shadow.gpu(start);
            let (source, skew) = (host & !3, host & 3);
            let words = (skew + bytes).div_ceil(4);
            let Some(words) = u32::try_from(words)
                .ok()
                .filter(|&w| holds(&shadow, source, 4 * u64::from(w)))
            else {
                return invalid(
                    i,
                    format_args!(
                        "vertices {start:#x}+{bytes:#x} end inside a word past their host copy"
                    ),
                );
            };
            jobs.push(Job {
                geometry: i,
                input: Input::Vertices {
                    format: t.VertexFormat,
                    stride,
                    skew: skew as u32,
                },
                kind: Kind::Words,
                shadow,
                source,
                count: words,
                vertices: 0,
            });
        }
        let index_size = match t.IndexFormat {
            DXGI_FORMAT_UNKNOWN if t.IndexCount == 0 && t.IndexBuffer == 0 => 0,
            // R16_TYPELESS: meshlet-packed (`console`).
            DXGI_FORMAT_R16_UINT | DXGI_FORMAT_R16_TYPELESS => 2,
            DXGI_FORMAT_R32_UINT => 4,
            format => {
                return invalid(
                    i,
                    format_args!("index format {} with {} indices", format.0, t.IndexCount),
                );
            }
        };
        let points = if index_size == 0 {
            vertices
        } else {
            t.IndexCount
        };
        if points == 0 || points % 3 != 0 {
            return invalid(
                i,
                format_args!("{points} vertices or indices: no whole triangles"),
            );
        }
        total += u64::from(points / 3);
        if index_size != 0 {
            let n = t.IndexCount;
            let (kind, shadow, source) = if t.IndexFormat == DXGI_FORMAT_R16_TYPELESS {
                // From the word its address is in.
                let at = t.IndexBuffer & !3;
                let window = (at != 0).then_some(at).and_then(|at| {
                    let bytes = console::packed_bytes(at, n / 3)?;
                    let shadow = input_range(list, at, bytes, 4)?;
                    let (host, limit) = (shadow.gpu(at), bytes.next_multiple_of(4));
                    let limit = u32::try_from(limit)
                        .ok()
                        .filter(|&l| holds(&shadow, host, l.into()))?;
                    Some((Kind::Packed { limit, flip }, shadow, host))
                });
                let Some(window) = window else {
                    return invalid(
                        i,
                        format_args!(
                            "{n} meshlet-packed indices at {at:#x}: parts outside title memory"
                        ),
                    );
                };
                window
            } else {
                let at = t.IndexBuffer;
                let bytes = u64::from(n) * index_size;
                let Some(shadow) = input_range(list, at, bytes, index_size) else {
                    return invalid(
                        i,
                        format_args!("{n} indices at {at:#x} outside title memory or misaligned"),
                    );
                };
                // The pass reads whole words: the last 16-bit index's must
                // lie in the host copy too.
                let host = shadow.gpu(at);
                let end = ((host & 3) + bytes).next_multiple_of(4);
                if !holds(&shadow, host & !3, end) {
                    return invalid(
                        i,
                        format_args!(
                            "{n} indices at {at:#x} end 2 bytes before their host copy's end"
                        ),
                    );
                }
                let kind = if index_size == 4 {
                    Kind::Wide
                } else {
                    Kind::Narrow
                };
                (kind, shadow, host)
            };
            jobs.push(Job {
                geometry: i,
                input: Input::Indices,
                kind,
                shadow,
                source,
                count: n,
                vertices,
            });
        }
        if t.Transform3x4 != 0 {
            let at = t.Transform3x4;
            let Some(shadow) = input_range(list, at, 48, 16) else {
                return invalid(
                    i,
                    format_args!("a transform at {at:#x} outside title memory or misaligned"),
                );
            };
            jobs.push(Job {
                geometry: i,
                input: Input::Transform,
                kind: Kind::Words,
                source: shadow.gpu(at),
                shadow,
                count: 12,
                vertices: 0,
            });
        }
        // Each set to what the pass makes when it is recorded.
        (t.VertexBuffer.StartAddress, t.IndexBuffer, t.Transform3x4) = (0, 0, 0);
        geometries.push(g);
    }
    if total > u64::from(D3D12_RAYTRACING_MAX_PRIMITIVES_PER_BOTTOM_LEVEL_ACCELERATION_STRUCTURE) {
        return invalid(0, format_args!("{total} triangles"));
    }
    Some((geometries, jobs))
}

#[cfg(test)]
mod tests;
