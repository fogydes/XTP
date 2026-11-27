//! Keeps the host GPU's clocks up between submissions (`XTP_CARD_AWAKE=1`, off
//! by default).
//!
//! Between submissions in a frame the GPU can sit idle for a millisecond or two
//! many times while the CPU prepares the next submission. The driver can read
//! those gaps as a light load and lower GPU clocks, making the following
//! submission start slowly.
//!
//! When enabled, a worker thread runs one 64-thread compute group in a loop on
//! its own compute queue, taking a small slice of the GPU's wave slots so the
//! card stays active. The submit path only records a monotonic timestamp in an
//! atomic (no locks, blocking or allocation). Once no submission has gone out
//! for 250 ms (`DEFAULT_QUIET_MS`), the worker stops dispatching and sleeps
//! until submissions resume. When `XTP_CARD_AWAKE` is unset or not `1`, nothing
//! is created (no thread, no queue, no GPU resources).

use std::ffi::OsStr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use windows::Win32::Foundation::{
    CloseHandle, E_FAIL, HANDLE, WAIT_EVENT, WAIT_OBJECT_0, WAIT_TIMEOUT,
};
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::Win32::Graphics::Dxgi::DXGI_ERROR_DEVICE_REMOVED;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::{Interface, s};

/// One 64-thread compute group looping to keep the card active between
/// submissions (~2 ms per dispatch at [`DEFAULT_ITERATIONS`] on full clocks).
const SHADER: &str = r#"
#define RS "RootConstants(num32BitConstants=1, b0), UAV(u0)"
cbuffer Params : register(b0) { uint iterations; };
RWByteAddressBuffer sum : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint x = id.x;
    [loop] for (uint i = 0; i < iterations; i++) x = x * 1664525u + 1013904223u;
    sum.Store(id.x * 4u, x);
}
"#;

const DEFAULT_ITERATIONS: u32 = 400_000;
const DEFAULT_QUIET_MS: u64 = 250;
const IDLE_SLEEP: Duration = Duration::from_millis(2);
const WAIT_SLICE_MS: u32 = 1000;

#[derive(Clone, Copy)]
struct Config {
    quiet_ms: u64,
    iterations: u32,
}

impl Config {
    const DEFAULT: Self = Self {
        quiet_ms: DEFAULT_QUIET_MS,
        iterations: DEFAULT_ITERATIONS,
    };
}

/// Whether `XTP_CARD_AWAKE` enables the keep-awake loop (`1` only; off by default).
fn enabled_value(value: Option<&OsStr>) -> bool {
    value
        .and_then(OsStr::to_str)
        .is_some_and(|v| v.trim() == "1")
}

/// Reads `XTP_CARD_AWAKE` once (off by default; `1` enables it).
fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = enabled_value(std::env::var_os("XTP_CARD_AWAKE").as_deref());
        if on {
            tracing::info!("XTP_CARD_AWAKE: keeping the card awake between submissions");
        }
        on
    })
}

/// Lock-free state shared between the submit path and the keep-awake worker.
struct Shared {
    origin: Instant,
    /// Milliseconds since `origin` plus one when the last submission went out (0: none).
    last_ms: AtomicU64,
    stop: AtomicBool,
    #[cfg(test)]
    dispatches: AtomicU64,
    #[cfg(test)]
    quiet: AtomicBool,
}

impl Shared {
    fn new() -> Self {
        Self {
            origin: Instant::now(),
            last_ms: AtomicU64::new(0),
            stop: AtomicBool::new(false),
            #[cfg(test)]
            dispatches: AtomicU64::new(0),
            #[cfg(test)]
            quiet: AtomicBool::new(true),
        }
    }

    fn now_ms(&self) -> u64 {
        self.origin.elapsed().as_millis() as u64
    }

    fn record_ms(&self, ms: u64) {
        self.last_ms
            .fetch_max(ms.saturating_add(1), Ordering::Relaxed);
    }

    fn is_active_at(&self, now_ms: u64, quiet_ms: u64) -> bool {
        let last = self.last_ms.load(Ordering::Relaxed);
        last != 0 && now_ms.saturating_sub(last - 1) < quiet_ms
    }

    fn active(&self, quiet_ms: u64) -> bool {
        self.is_active_at(self.now_ms(), quiet_ms)
    }
}

/// `ID3D12Device` moved onto the keep-awake worker thread.
struct SendDevice(ID3D12Device);

// SAFETY: `ID3D12Device` methods are free-threaded.
unsafe impl Send for SendDevice {}

/// Handle to the keep-awake worker thread. Dropping it stops the worker, waits
/// for any in-flight compute dispatch to complete, and joins the thread.
pub(super) struct Awake {
    shared: Arc<Shared>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Awake {
    /// Starts the keep-awake worker for `device` if `XTP_CARD_AWAKE=1` is set.
    /// Returns `None` without creating a thread or GPU objects otherwise.
    pub(super) fn start_if_enabled(device: &ID3D12Device) -> Option<Self> {
        Self::start_if(enabled(), device, Config::DEFAULT)
    }

    fn start_if(on: bool, device: &ID3D12Device, config: Config) -> Option<Self> {
        if !on {
            return None;
        }
        let shared = Arc::new(Shared::new());
        let worker_shared = Arc::clone(&shared);
        let device = SendDevice(device.clone());
        let thread = match std::thread::Builder::new()
            .name("xtp-card-awake".into())
            .spawn(move || {
                let device = device;
                if let Err(e) = run(&device.0, &worker_shared, config) {
                    tracing::warn!(error = %e, "the card is not kept awake");
                }
            }) {
            Ok(thread) => thread,
            Err(e) => {
                tracing::warn!(error = %e, "could not start card keep-awake thread");
                return None;
            }
        };
        Some(Self {
            shared,
            thread: Some(thread),
        })
    }

    /// Notes that a submission was sent to the host queue.
    /// Lock-free and non-allocating: advances the monotonic timestamp only.
    pub(super) fn note_submission(&self) {
        self.shared.record_ms(self.shared.now_ms());
    }

    #[cfg(test)]
    fn dispatches(&self) -> u64 {
        self.shared.dispatches.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    fn is_quiet(&self) -> bool {
        self.shared.quiet.load(Ordering::Acquire)
    }
}

impl Drop for Awake {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            thread.thread().unpark();
            let _ = thread.join();
        }
    }
}

fn blob_bytes(blob: &ID3DBlob) -> &[u8] {
    // SAFETY: plain COM queries on a live `ID3DBlob`; only dereferenced when
    // both pointer and length are valid and non-empty.
    unsafe {
        let ptr = blob.GetBufferPointer().cast::<u8>();
        let len = blob.GetBufferSize();
        if ptr.is_null() || len == 0 {
            &[]
        } else {
            std::slice::from_raw_parts(ptr, len)
        }
    }
}

/// Owned Win32 auto-reset event handle closed on drop.
struct Event(HANDLE);

impl Event {
    fn new() -> windows::core::Result<Self> {
        // SAFETY: creates a private unnamed auto-reset event.
        unsafe { CreateEventW(None, false, false, None) }.map(Self)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: `self.0` is an owned event handle created by `CreateEventW`.
        let _ = unsafe { CloseHandle(self.0) };
    }
}

trait FenceOps {
    fn completed(&self) -> u64;
    fn arm(&self, target: u64) -> windows::core::Result<()>;
    fn wait_slice(&self, ms: u32) -> WAIT_EVENT;
    fn device_removed_reason(&self) -> windows::core::Result<()>;
}

struct D3d12FenceOps<'a> {
    device: &'a ID3D12Device,
    fence: &'a ID3D12Fence,
    event: HANDLE,
}

impl FenceOps for D3d12FenceOps<'_> {
    fn completed(&self) -> u64 {
        // SAFETY: plain query on a live D3D12 fence.
        unsafe { self.fence.GetCompletedValue() }
    }

    fn arm(&self, target: u64) -> windows::core::Result<()> {
        // SAFETY: `self.fence` and `self.event` are live; `Event` outlives `fence` in `run`.
        unsafe { self.fence.SetEventOnCompletion(target, self.event) }
    }

    fn wait_slice(&self, ms: u32) -> WAIT_EVENT {
        // SAFETY: `self.event` is a valid event handle owned by `Event`.
        unsafe { WaitForSingleObject(self.event, ms) }
    }

    fn device_removed_reason(&self) -> windows::core::Result<()> {
        // SAFETY: plain query on a live D3D12 device.
        unsafe { self.device.GetDeviceRemovedReason() }
    }
}

/// Tracks the highest fence value armed on one auto-reset event so timeouts
/// never stack duplicate `SetEventOnCompletion` registrations and shutdown
/// drains past any stale wake from an earlier fence target.
#[derive(Default)]
struct FenceWaiter {
    armed: Option<u64>,
}

impl FenceWaiter {
    fn device_removed_error(ops: &impl FenceOps) -> windows::core::Error {
        ops.device_removed_reason()
            .err()
            .unwrap_or_else(|| windows::core::Error::from_hresult(DXGI_ERROR_DEVICE_REMOVED))
    }

    /// Waits at most one slice for `target` without re-arming the same target
    /// across timeouts.
    fn wait_slice(&mut self, ops: &impl FenceOps, target: u64) -> windows::core::Result<()> {
        if self.armed != Some(target) {
            ops.arm(target)?;
            self.armed = Some(target);
        }
        match ops.wait_slice(WAIT_SLICE_MS) {
            WAIT_OBJECT_0 => Ok(()),
            WAIT_TIMEOUT => ops.device_removed_reason(),
            _ => {
                let err = windows::core::Error::from_thread();
                if err.code().is_ok() {
                    Err(windows::core::Error::from_hresult(E_FAIL))
                } else {
                    Err(err)
                }
            }
        }
    }

    /// Drains until the GPU reaches `target` or the device is removed, never
    /// returning early on a stale signal from an earlier fence value and never
    /// hanging indefinitely on device removal.
    fn drain(&mut self, ops: &impl FenceOps, target: u64) {
        if target == 0 {
            return;
        }
        loop {
            let completed = ops.completed();
            if completed >= target || completed == u64::MAX {
                break;
            }
            let armed = if self.armed == Some(target) {
                true
            } else if ops.arm(target).is_ok() {
                self.armed = Some(target);
                true
            } else {
                false
            };
            if armed {
                match ops.wait_slice(WAIT_SLICE_MS) {
                    WAIT_OBJECT_0 => continue,
                    WAIT_TIMEOUT => {
                        if ops.device_removed_reason().is_err() {
                            break;
                        }
                    }
                    _ => {
                        if ops.device_removed_reason().is_err() {
                            break;
                        }
                        std::thread::sleep(Duration::from_millis(1));
                    }
                }
            } else {
                if ops.device_removed_reason().is_err() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }
}

fn compile() -> windows::core::Result<ID3DBlob> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: `SHADER` and string literals live through the call; out pointers are ours.
    let compiled = unsafe {
        D3DCompile(
            SHADER.as_ptr().cast(),
            SHADER.len(),
            s!("xtp card awake"),
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
            let bytes = blob_bytes(&errors);
            if !bytes.is_empty() {
                tracing::warn!("card awake shader: {}", String::from_utf8_lossy(bytes));
            }
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    if blob_bytes(&code).is_empty() {
        return Err(windows::core::Error::empty());
    }
    Ok(code)
}

fn run(device: &ID3D12Device, shared: &Shared, config: Config) -> windows::core::Result<()> {
    let code = compile()?;
    let bytes = blob_bytes(&code);
    // Declared before `fence` (and all other D3D12 objects) so Rust's reverse
    // drop order releases `fence` before `CloseHandle` closes the event, even
    // if a wait aborts early on device removal.
    let event = Event::new()?;
    // SAFETY: plain D3D12 object creation; `code` lives through root signature
    // and PSO creation and carries the embedded root signature (`bytes` is
    // non-empty). Two closed lists are submitted in turn with at most two
    // dispatches in flight so a list is never reused while running. All GPU
    // objects outlive `waiter.drain`.
    unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso: ID3D12PipelineState =
            device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            })?;
        let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_COMPUTE,
            Priority: D3D12_COMMAND_QUEUE_PRIORITY_NORMAL.0,
            ..Default::default()
        })?;
        super::name(&queue, || "xtp card awake compute queue".into());
        let mut sum: Option<ID3D12Resource> = None;
        device.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            },
            D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 256,
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
            },
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut sum,
        )?;
        let sum = sum.ok_or_else(windows::core::Error::empty)?;
        super::name(&sum, || "xtp card awake buffer".into());
        let allocator: ID3D12CommandAllocator =
            device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE)?;
        let mut lists = Vec::with_capacity(2);
        for i in 0..2 {
            let list: ID3D12GraphicsCommandList =
                device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &allocator, &pso)?;
            super::name(&list, || format!("xtp card awake list {i}"));
            list.SetComputeRootSignature(&root);
            list.SetComputeRoot32BitConstant(0, config.iterations, 0);
            list.SetComputeRootUnorderedAccessView(1, sum.GetGPUVirtualAddress());
            list.Dispatch(1, 1, 1);
            list.Close()?;
            lists.push(Some(list.cast::<ID3D12CommandList>()?));
        }
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let ops = D3d12FenceOps {
            device,
            fence: &fence,
            event: event.0,
        };
        let mut waiter = FenceWaiter::default();
        let mut signalled = 0u64;
        let mut result = Ok(());
        while !shared.stop.load(Ordering::Acquire) && result.is_ok() {
            if !shared.active(config.quiet_ms) {
                #[cfg(test)]
                shared.quiet.store(true, Ordering::Release);
                std::thread::park_timeout(IDLE_SLEEP);
                continue;
            }
            #[cfg(test)]
            shared.quiet.store(false, Ordering::Release);
            let completed = ops.completed();
            if completed == u64::MAX {
                result = Err(FenceWaiter::device_removed_error(&ops));
                break;
            }
            // Keep at most two dispatches in flight (one per closed list).
            if signalled >= 2 && completed < signalled - 1 {
                result = waiter.wait_slice(&ops, signalled - 1);
                continue;
            }
            let Some(next) = signalled.checked_add(1).filter(|&n| n < u64::MAX) else {
                break;
            };
            queue.ExecuteCommandLists(&lists[(signalled % 2) as usize..][..1]);
            result = queue.Signal(&fence, next);
            if result.is_ok() {
                signalled = next;
                #[cfg(test)]
                shared.dispatches.fetch_add(1, Ordering::Relaxed);
            }
        }
        #[cfg(test)]
        shared.quiet.store(true, Ordering::Release);
        waiter.drain(&ops, signalled);
        result
    }
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};
    use std::collections::VecDeque;

    use windows::Win32::Foundation::WAIT_FAILED;

    use super::*;

    fn wait_until(timeout: Duration, mut step: impl FnMut(), check: impl Fn() -> bool) -> bool {
        let deadline = Instant::now() + timeout;
        while !check() && Instant::now() < deadline {
            step();
            std::thread::sleep(Duration::from_millis(20));
        }
        check()
    }

    struct FakeFence {
        completed: Cell<u64>,
        armed: RefCell<Vec<u64>>,
        waits: RefCell<VecDeque<(WAIT_EVENT, Option<u64>)>>,
        removed: RefCell<windows::core::Result<()>>,
        arm_fails: Cell<bool>,
    }

    impl FakeFence {
        fn new(completed: u64) -> Self {
            Self {
                completed: Cell::new(completed),
                armed: RefCell::new(Vec::new()),
                waits: RefCell::new(VecDeque::new()),
                removed: RefCell::new(Ok(())),
                arm_fails: Cell::new(false),
            }
        }

        fn push_wait(&self, result: WAIT_EVENT, advance_completed_to: Option<u64>) {
            self.waits
                .borrow_mut()
                .push_back((result, advance_completed_to));
        }
    }

    impl FenceOps for FakeFence {
        fn completed(&self) -> u64 {
            self.completed.get()
        }

        fn arm(&self, target: u64) -> windows::core::Result<()> {
            if self.arm_fails.get() {
                return Err(windows::core::Error::empty());
            }
            self.armed.borrow_mut().push(target);
            Ok(())
        }

        fn wait_slice(&self, _ms: u32) -> WAIT_EVENT {
            let (event, advance) = self
                .waits
                .borrow_mut()
                .pop_front()
                .expect("unexpected extra wait_slice call");
            if let Some(next) = advance {
                self.completed.set(next);
            }
            event
        }

        fn device_removed_reason(&self) -> windows::core::Result<()> {
            self.removed.borrow().clone()
        }
    }

    #[test]
    fn env_switch_is_off_unless_set_to_one() {
        for off in [
            None,
            Some(""),
            Some("0"),
            Some("false"),
            Some("true"),
            Some("2"),
        ] {
            assert!(!enabled_value(off.map(OsStr::new)), "{off:?}");
        }
        for on in ["1", " 1 "] {
            assert!(enabled_value(Some(OsStr::new(on))), "{on:?}");
        }
    }

    #[test]
    fn timestamps_are_monotonic_and_handle_window_boundaries() {
        let shared = Shared::new();
        assert!(!shared.is_active_at(0, 250));
        assert!(!shared.is_active_at(100, 0));

        // Recording at t=0 activates within [0, quiet_ms).
        shared.record_ms(0);
        assert!(shared.is_active_at(0, 250));
        assert!(shared.is_active_at(249, 250));
        assert!(!shared.is_active_at(250, 250));
        assert!(!shared.is_active_at(0, 0));

        // Out-of-order delayed record from an earlier thread cannot move last_ms backwards.
        shared.record_ms(400);
        shared.record_ms(100);
        assert!(shared.is_active_at(649, 250));
        assert!(!shared.is_active_at(650, 250));

        // Near u64::MAX does not overflow.
        shared.record_ms(u64::MAX);
        assert!(shared.is_active_at(u64::MAX, 250));
    }

    #[test]
    fn fence_waiter_does_not_rearm_on_timeout_and_drains_past_stale_signal() {
        let fence = FakeFence::new(0);
        let mut waiter = FenceWaiter::default();

        // Two timeouts waiting for target 1 must arm target 1 only once.
        fence.push_wait(WAIT_TIMEOUT, None);
        fence.push_wait(WAIT_TIMEOUT, None);
        waiter.wait_slice(&fence, 1).unwrap();
        waiter.wait_slice(&fence, 1).unwrap();
        assert_eq!(*fence.armed.borrow(), vec![1]);

        // Now shutdown drains target 2 while the earlier registration for 1 is
        // still pending: the first wake arrives when completed reaches 1 (stale),
        // so drain must keep waiting (without re-arming 2) until completed reaches 2.
        fence.push_wait(WAIT_OBJECT_0, Some(1));
        fence.push_wait(WAIT_OBJECT_0, Some(2));
        waiter.drain(&fence, 2);

        assert_eq!(fence.completed(), 2);
        assert_eq!(*fence.armed.borrow(), vec![1, 2]);
        assert!(fence.waits.borrow().is_empty());
    }

    #[test]
    fn fence_waiter_drain_exits_on_device_removal() {
        let fence = FakeFence::new(0);
        let mut waiter = FenceWaiter::default();
        fence.push_wait(WAIT_TIMEOUT, None);
        *fence.removed.borrow_mut() = Err(windows::core::Error::from_hresult(
            DXGI_ERROR_DEVICE_REMOVED,
        ));

        // Must exit after one slice rather than blocking indefinitely.
        waiter.drain(&fence, 2);
        assert_eq!(*fence.armed.borrow(), vec![2]);
        assert!(fence.waits.borrow().is_empty());

        // If SetEventOnCompletion fails while device is removed, drain also exits promptly.
        let failing = FakeFence::new(0);
        failing.arm_fails.set(true);
        *failing.removed.borrow_mut() = Err(windows::core::Error::from_hresult(
            DXGI_ERROR_DEVICE_REMOVED,
        ));
        let mut waiter = FenceWaiter::default();
        waiter.drain(&failing, 2);
        assert!(failing.armed.borrow().is_empty());
    }

    #[test]
    fn fence_waiter_reports_wait_failure_and_sentinel() {
        let fence = FakeFence::new(0);
        let mut waiter = FenceWaiter::default();
        fence.push_wait(WAIT_FAILED, None);
        assert!(waiter.wait_slice(&fence, 1).is_err());

        // u64::MAX sentinel still produces DXGI_ERROR_DEVICE_REMOVED even if
        // GetDeviceRemovedReason unexpectedly returns Ok(()).
        let err = FenceWaiter::device_removed_error(&fence);
        assert_eq!(err.code(), DXGI_ERROR_DEVICE_REMOVED);
    }

    #[test]
    fn does_nothing_when_off() {
        let Some(gpu) = crate::testgpu::device("host::awake::tests::does_nothing_when_off") else {
            return;
        };
        assert!(Awake::start_if(false, &gpu.device, Config::DEFAULT).is_none());
        if std::env::var_os("XTP_CARD_AWAKE").is_none() {
            assert!(gpu.host().awake.is_none());
        }
    }

    #[test]
    fn dispatches_while_submissions_arrive_then_goes_quiet() {
        let Some(gpu) = crate::testgpu::device(
            "host::awake::tests::dispatches_while_submissions_arrive_then_goes_quiet",
        ) else {
            return;
        };
        let config = Config {
            quiet_ms: 150,
            iterations: 2_000,
        };
        let awake = Awake::start_if(true, &gpu.device, config).expect("keep-awake thread starts");

        std::thread::sleep(Duration::from_millis(25));
        assert_eq!(awake.dispatches(), 0);
        assert!(awake.is_quiet());

        assert!(
            wait_until(
                Duration::from_secs(10),
                || awake.note_submission(),
                || awake.dispatches() >= 1
            ),
            "expected at least one keep-awake dispatch while submissions arrived"
        );
        for _ in 0..3 {
            awake.note_submission();
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(
            wait_until(Duration::from_secs(10), || {}, || awake.is_quiet()),
            "expected keep-awake worker to go quiet after quiet window"
        );

        let settled = awake.dispatches();
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(
            awake.dispatches(),
            settled,
            "expected no dispatches while quiet"
        );

        assert!(
            wait_until(
                Duration::from_secs(10),
                || awake.note_submission(),
                || awake.dispatches() > settled,
            ),
            "expected keep-awake worker to resume dispatching after a new submission"
        );
        drop(awake);
    }

    #[test]
    fn drop_while_active_drains_cleanly() {
        let Some(gpu) =
            crate::testgpu::device("host::awake::tests::drop_while_active_drains_cleanly")
        else {
            return;
        };
        let config = Config {
            quiet_ms: 500,
            iterations: 5_000,
        };
        let awake =
            Arc::new(Awake::start_if(true, &gpu.device, config).expect("keep-awake thread starts"));

        // Two threads concurrently note submissions so dispatches are actively
        // in flight when `Awake` is dropped.
        let stop_callers = Arc::new(AtomicBool::new(false));
        let mut callers = Vec::new();
        for _ in 0..2 {
            let awake = Arc::clone(&awake);
            let stop = Arc::clone(&stop_callers);
            callers.push(std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    awake.note_submission();
                    std::thread::sleep(Duration::from_millis(2));
                }
            }));
        }

        assert!(
            wait_until(Duration::from_secs(10), || {}, || awake.dispatches() >= 3),
            "expected active keep-awake dispatches before drop"
        );
        stop_callers.store(true, Ordering::Relaxed);
        for caller in callers {
            caller.join().unwrap();
        }

        // Drop while still within the quiet window (dispatches active/in-flight).
        let awake = Arc::try_unwrap(awake).ok().expect("unique awake owner");
        drop(awake);
    }
}
