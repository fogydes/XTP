//! Indirect root SRVs let each recording supply its own TLAS at submission.
//! A missing TLAS binds an empty one, where every ray misses: the program
//! still runs and writes its output, as on the console with an empty
//! structure, instead of leaving what it wrote last frame.
use super::*;
use std::collections::VecDeque;
use windows::core::s;

/// Bytes of one direct trace command: the TLAS root SRV, then the groups.
const DIRECT_STRIDE: u32 = 20;
/// Bytes of one composed indirect trace command: the TLAS root SRV, the
/// groups, and four bytes that keep every SRV address on eight bytes.
const INDIRECT_STRIDE: u32 = 24;
/// Where the composed commands start; the command count leads.
const COMMANDS_AT: u64 = 8;
/// Indirect trace records one preparation composes at most.
const MAX_RECORDS: u32 = 1 << 16;
/// D3D12's limit on thread groups along each dimension.
const MAX_GROUPS: u32 = D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION;

pub(super) fn signature(
    root: &ID3D12RootSignature,
    param: u32,
    stride: u32,
) -> Option<ID3D12CommandSignature> {
    static CACHE: Mutex<Signatures> = Mutex::new(Signatures(VecDeque::new()));
    CACHE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(root, param, stride)
}

/// A retained root prevents its COM address from being reused for a different
/// signature. Lists retain the returned command signature independently, so
/// eviction cannot release an object still referenced by GPU work.
struct Signature {
    root: ID3D12RootSignature,
    param: u32,
    stride: u32,
    commands: ID3D12CommandSignature,
}

struct Signatures(VecDeque<Signature>);

impl Signatures {
    const LIMIT: usize = 64;

    fn get(
        &mut self,
        root: &ID3D12RootSignature,
        param: u32,
        stride: u32,
    ) -> Option<ID3D12CommandSignature> {
        if let Some(i) = self.0.iter().rposition(|s| {
            s.root.as_raw() == root.as_raw() && s.param == param && s.stride == stride
        }) {
            let hit = self.0.remove(i)?;
            let commands = hit.commands.clone();
            self.0.push_back(hit);
            return Some(commands);
        }
        // Create under the lock, so concurrent first uses make it only once.
        // A failed creation is not cached: a later request can retry it.
        let commands = make_signature(root, param, stride)?;
        if self.0.len() == Self::LIMIT {
            self.0.pop_front();
        }
        self.0.push_back(Signature {
            root: root.clone(),
            param,
            stride,
            commands: commands.clone(),
        });
        Some(commands)
    }
}

fn make_signature(
    root: &ID3D12RootSignature,
    param: u32,
    stride: u32,
) -> Option<ID3D12CommandSignature> {
    let mut srv = D3D12_INDIRECT_ARGUMENT_DESC {
        Type: D3D12_INDIRECT_ARGUMENT_TYPE_SHADER_RESOURCE_VIEW,
        ..Default::default()
    };
    srv.Anonymous.ShaderResourceView = D3D12_INDIRECT_ARGUMENT_DESC_0_3 {
        RootParameterIndex: param,
    };
    let args = [
        srv,
        D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
            ..Default::default()
        },
    ];
    let mut signature = None;
    // SAFETY: the descriptions match the packed upload record and live root signature.
    let made = unsafe {
        crate::host::made()?.device.CreateCommandSignature(
            &D3D12_COMMAND_SIGNATURE_DESC {
                ByteStride: stride,
                NumArgumentDescs: args.len() as u32,
                pArgumentDescs: args.as_ptr(),
                NodeMask: 0,
            },
            root,
            &mut signature,
        )
    };
    if let Err(error) = made {
        // A root parameter that is no root SRV (a one-view table) lands here.
        tracing::warn!(%error, param, "no trace command signature: the trace is dropped");
        return None;
    }
    signature
}

/// Counts a trace bound to the empty TLAS and logs it, at powers of two.
fn no_tlas(address: u64) {
    static EMPTY: AtomicU64 = AtomicU64::new(0);
    let traces = EMPTY.fetch_add(1, Ordering::Relaxed) + 1;
    if traces.is_power_of_two() {
        tracing::warn!(
            address = format_args!("{address:#x}"),
            traces,
            "a trace ran with an empty TLAS, every ray missing: no TLAS was published at its \
             title address in queue order (its build refused, released or not yet submitted)"
        );
    }
}

/// Counts a trace whose title descriptor names no TLAS the decoder can
/// read (empty, or not a raytracing or buffer SRV) and logs it, at powers
/// of two: it runs with the empty TLAS, every ray missing, and still
/// writes its output.
fn unreadable(handle: u64) {
    static UNREADABLE: AtomicU64 = AtomicU64::new(0);
    let traces = UNREADABLE.fetch_add(1, Ordering::Relaxed) + 1;
    if traces.is_power_of_two() {
        tracing::warn!(
            handle = format_args!("{handle:#x}"),
            traces,
            "a trace's TLAS descriptor names no TLAS: it runs with an empty TLAS, every ray              missing"
        );
    }
}

/// The title address of the TLAS the descriptor at `handle` names, or 0
/// (none: the empty TLAS) when it names none ([`unreadable`]).
fn trace_address(handle: u64) -> u64 {
    descriptor_address(handle).unwrap_or_else(|| {
        unreadable(handle);
        0
    })
}

/// Writes the GPU address of the TLAS published at title `address` in
/// queue order to the two words at `mapped` when the recording is
/// submitted, or that of an empty TLAS ([`empty_tlas`]), as for address 0.
/// The recording retains the TLAS it binds, that generation's BLAS and its
/// stand-in BLAS. `None` without an empty TLAS.
fn bind_at_submission(rec: &Recorder<'_>, address: u64, mapped: usize) -> Option<()> {
    let (empty, _) = empty_tlas(rec)?;
    let slot = if address == 0 {
        Slot::default()
    } else {
        registry().trace_slot(address)
    };
    let retained = Mutex::new(None);
    rec.list.defer_host((0, 0), move || {
        let tlas = slot.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let binding = tlas.and_then(|s| Some((s.result, s.instances?.current(), s.holds)));
        let gpu = binding
            .as_ref()
            .map_or_else(|| empty.gpu(), |(r, ..)| r.gpu());
        if binding.is_none() && address != 0 {
            no_tlas(address);
        }
        // SAFETY: two aligned words in the recording's private upload. Replay
        // waits for the preceding execution before rewriting the SRV.
        unsafe { (mapped as *mut [u32; 2]).write([gpu as u32, (gpu >> 32) as u32]) };
        *retained.lock().unwrap_or_else(|e| e.into_inner()) = binding;
    });
    Some(())
}

/// Dispatches a trace with this descriptor's TLAS at its place in submission
/// order, or an empty TLAS where none is published or the descriptor names
/// none: every ray misses, and the program still writes its output.
/// The caller sets the remaining root parameters and the compute PSO first.
/// The empty TLAS is built with the title's TLAS builds; failing that, here.
///
/// # Safety
/// The title list and descriptor slot are live. The currently bound compute root
/// signature is `root` and `param` is its acceleration-structure root SRV.
pub unsafe fn dispatch_descriptor_tlas(
    title_list: *const Object,
    handle: u64,
    root: &ID3D12RootSignature,
    param: u32,
    groups: [u32; 3],
) -> bool {
    let Some(list) = crate::list::host_list(title_list) else {
        return false;
    };
    dispatch_descriptor_tlas_in(list, handle, root, param, groups)
}

/// [`dispatch_descriptor_tlas`] for the host recording itself, which draws
/// have where the title list is not at hand. The currently bound compute
/// root signature is `root` and `param` is its acceleration-structure root
/// SRV.
pub(crate) fn dispatch_descriptor_tlas_in(
    list: &HostList,
    handle: u64,
    root: &ID3D12RootSignature,
    param: u32,
    groups: [u32; 3],
) -> bool {
    dispatch(list, trace_address(handle), root, param, groups).is_some()
}

pub(super) fn dispatch(
    list: &HostList,
    address: u64,
    root: &ID3D12RootSignature,
    param: u32,
    groups: [u32; 3],
) -> Option<()> {
    let signature = signature(root, param, DIRECT_STRIDE)?;
    let (args, mapped) = list.upload(crate::host::made()?, u64::from(DIRECT_STRIDE))?;
    // SAFETY: the record's group words, in the recording's private upload.
    unsafe { mapped.add(8).cast::<[u32; 3]>().write(groups) };
    bind_at_submission(&Recorder::new(list)?, address, mapped as usize)?;
    list.keep(&signature);
    // SAFETY: the retained upload holds one SRV and dispatch record.
    unsafe {
        list.list.ExecuteIndirect(&signature, 1, &args, 0, None, 0);
    }
    Some(())
}

/// The commands of a trace dispatch whose group counts the GPU wrote,
/// composed in the recording ([`prepare_indirect_tlas`]).
pub struct IndirectTlas {
    commands: ID3D12Resource,
    records: u32,
}

/// Composes, on the GPU at this place in the list, the commands of a trace
/// the title dispatches indirectly: `records` dispatch records of `stride`
/// bytes at title address `arguments`, and, when `count` is set, the
/// command count at that title address, as the title passes them to
/// `ExecuteIndirect`. Each command takes this descriptor's TLAS at its place
/// in submission order (an empty one when none: every ray misses) and the
/// title's groups as the GPU wrote them, clamped to D3D12's limit.
///
/// It records a compute pass of its own, which sets a compute root
/// signature and pipeline: call it before binding the trace's PSO and root
/// parameters, then [`dispatch_indirect_tlas`]. A caller that keeps the
/// root signature and pipeline it bound last (the draw state) forgets them
/// after this, so that it binds the trace's again (this cannot: that
/// caller holds the list's draw state). `None` if the arguments are outside
/// title memory or misaligned. A descriptor that names no TLAS gives the
/// empty one, as in [`dispatch_descriptor_tlas`].
///
/// # Safety
/// The title list and descriptor slot are live.
pub unsafe fn prepare_indirect_tlas(
    title_list: *const Object,
    handle: u64,
    (arguments, stride, records): (u64, u32, u32),
    count: Option<u64>,
) -> Option<IndirectTlas> {
    let recorder = Recorder::new(crate::list::host_list(title_list)?)?;
    compose(
        &recorder,
        trace_address(handle),
        (arguments, stride, records),
        count,
    )
}

/// Runs the trace commands [`prepare_indirect_tlas`] composed.
///
/// # Safety
/// The title list is the one the commands were prepared in. The currently
/// bound compute root signature is `root`, with every other root parameter
/// set, and `param` is its acceleration-structure root SRV.
pub unsafe fn dispatch_indirect_tlas(
    title_list: *const Object,
    prepared: &IndirectTlas,
    root: &ID3D12RootSignature,
    param: u32,
) -> bool {
    let Some(list) = crate::list::host_list(title_list) else {
        return false;
    };
    let Some(signature) = signature(root, param, INDIRECT_STRIDE) else {
        return false;
    };
    list.keep(&signature);
    // SAFETY: the list retains the composed commands, already in the
    // indirect-argument state, and their count word.
    unsafe {
        list.list.ExecuteIndirect(
            &signature,
            prepared.records,
            &prepared.commands,
            COMMANDS_AT,
            &prepared.commands,
            0,
        );
    }
    true
}

fn compose(
    rec: &Recorder<'_>,
    address: u64,
    (arguments, stride, records): (u64, u32, u32),
    count: Option<u64>,
) -> Option<IndirectTlas> {
    if records == 0 || records > MAX_RECORDS || stride < 12 || !stride.is_multiple_of(4) {
        return None;
    }
    let span = u64::from(stride) * u64::from(records - 1) + 12;
    let args = input_range(rec.list, arguments, span, 4)?;
    let counter = match count {
        Some(at) => Some((input_range(rec.list, at, 4, 4)?, at)),
        None => None,
    };
    let pass = compose_pass(rec.device)?;
    let (tlas, mapped) = rec.list.scratch(8)?;
    let bytes = COMMANDS_AT + u64::from(records) * u64::from(INDIRECT_STRIDE);
    let commands = buffer(rec, bytes, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)?;
    bind_at_submission(rec, address, mapped as usize)?;
    // Both title inputs go back to the common state once, even in one
    // buffer. They need no transition to the UAV state the pass reads them
    // in: a buffer in the common state is promoted to it, and one a title
    // shader wrote earlier in the submission already is (a barrier from the
    // common state would then name the wrong state).
    let mut inputs = vec![args.buffer.clone()];
    if let Some((shadow, _)) = &counter
        && shadow.buffer.as_raw() != args.buffer.as_raw()
    {
        inputs.push(shadow.buffer.clone());
    }
    let common = D3D12_RESOURCE_STATE_COMMON;
    let uav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    let before = [uav_barrier()];
    let mut after: Vec<_> = inputs
        .iter()
        .map(|b| crate::host::transition(b, uav, common))
        .collect();
    after.push(crate::host::transition(
        &commands,
        uav,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
    ));
    let gpu = args.gpu(arguments);
    let shape = [records, stride, u32::from(counter.is_some())];
    #[cfg(feature = "diagnostics")]
    rec.list.note_op(|| {
        format!(
            "host DXR trace commands: {records} records of {stride} bytes at title address              {arguments:#x}, count at {count:#x?}, TLAS at title address {address:#x}"
        )
    });
    // SAFETY: each root view covers the shader's checked range. The global
    // barrier orders the title's GPU writers of the arguments before the
    // pass; the recording retains its inputs and output.
    unsafe {
        let c = &rec.commands;
        c.ResourceBarrier(&before);
        c.SetComputeRootSignature(&pass.0);
        c.SetPipelineState(&pass.1);
        c.SetComputeRoot32BitConstants(0, 3, shape.as_ptr().cast(), 0);
        c.SetComputeRootShaderResourceView(1, tlas);
        c.SetComputeRootUnorderedAccessView(2, gpu);
        c.SetComputeRootUnorderedAccessView(3, counter.map_or(gpu, |(s, at)| s.gpu(at)));
        c.SetComputeRootUnorderedAccessView(4, commands.GetGPUVirtualAddress());
        c.Dispatch(records.div_ceil(64), 1, 1);
        c.ResourceBarrier(&after);
    }
    for input in &inputs {
        rec.list.keep(input);
    }
    Some(IndirectTlas { commands, records })
}

fn compose_pass(
    device: &ID3D12Device5,
) -> Option<&'static (ID3D12RootSignature, ID3D12PipelineState)> {
    static PASS: OnceLock<Option<(ID3D12RootSignature, ID3D12PipelineState)>> = OnceLock::new();
    PASS.get_or_init(|| {
        instances::compute_pass(
            device,
            include_bytes!("trace_args.hlsl"),
            s!("rt_trace_args"),
        )
    })
    .as_ref()
}

const _: () = assert!(MAX_GROUPS == 65535);

#[cfg(test)]
mod signature_tests {
    use super::*;

    #[test]
    #[ignore = "CPU microbenchmark on a hardware D3D12 device"]
    fn rt_signature_creation_timings() {
        let Some(gpu) = crate::testgpu::device("rt_signature_creation_timings") else {
            return;
        };
        let Some(device) = device(gpu.host()) else {
            crate::testgpu::skip("rt_signature_creation_timings", "no DXR device");
            return;
        };
        let (root, _) = instances::compute_pass(
            device,
            br#"[RootSignature("SRV(t0)")]
                [numthreads(1,1,1)] void main() {}"#,
            windows::core::s!("signature_timing_test"),
        )
        .unwrap();
        const CALLS: u32 = 2000;
        // Warm both paths; batches alternate their order to reduce drift.
        signature(&root, 0, 20).unwrap();
        make_signature(&root, 0, 20).unwrap();
        for batch in 0..7 {
            let mut times = [0u128; 2];
            for mode in [batch % 2, 1 - batch % 2] {
                let start = std::time::Instant::now();
                for _ in 0..CALLS {
                    let commands = if mode == 0 {
                        make_signature(&root, 0, 20)
                    } else {
                        signature(&root, 0, 20)
                    }
                    .unwrap();
                    std::hint::black_box(commands);
                }
                times[mode] = start.elapsed().as_nanos();
            }
            println!(
                "signature_timing batch={batch} calls={CALLS} uncached_ns={} cached_ns={}",
                times[0], times[1]
            );
        }
    }

    #[test]
    fn rt_signature_cache_is_bounded_and_keeps_recent_entries() {
        let Some(gpu) =
            crate::testgpu::device("rt_signature_cache_is_bounded_and_keeps_recent_entries")
        else {
            return;
        };
        let Some(device) = device(gpu.host()) else {
            crate::testgpu::skip(
                "rt_signature_cache_is_bounded_and_keeps_recent_entries",
                "no DXR device",
            );
            return;
        };
        let (root, _) = instances::compute_pass(
            device,
            br#"[RootSignature("SRV(t0)")]
                [numthreads(1,1,1)] void main() {}"#,
            windows::core::s!("signature_cache_test"),
        )
        .unwrap();
        let mut cache = Signatures(VecDeque::new());
        let first = cache.get(&root, 0, 20).unwrap();
        let evicted = cache.get(&root, 0, 24).unwrap();
        for i in 2..Signatures::LIMIT {
            cache.get(&root, 0, 20 + 4 * i as u32).unwrap();
        }
        assert_eq!(cache.get(&root, 0, 20).unwrap().as_raw(), first.as_raw());
        cache
            .get(&root, 0, 20 + 4 * Signatures::LIMIT as u32)
            .unwrap();
        assert_eq!(cache.0.len(), Signatures::LIMIT);
        assert!(
            !cache
                .0
                .iter()
                .any(|s| s.commands.as_raw() == evicted.as_raw())
        );
        assert_ne!(cache.get(&root, 0, 24).unwrap().as_raw(), evicted.as_raw());
        assert_eq!(cache.0.len(), Signatures::LIMIT);
        crate::testgpu::expect_layer_errors(|| {
            assert!(cache.get(&root, 1, 20).is_none());
            assert!(cache.get(&root, 1, 20).is_none());
        });
        assert_eq!(
            cache.0.len(),
            Signatures::LIMIT,
            "failures are not retained"
        );
    }
}
