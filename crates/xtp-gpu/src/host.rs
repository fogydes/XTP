//! The host GPU: one D3D12 device and direct queue (decision 0007), and a
//! window to present on.
//!
//! Submissions are not waited for (backlog G8, ported from
//! Marvisak/VIctory#595 by wbkepler): each signals the next value of the
//! host fence and joins the [`Timeline`], and is finished (what its GPU
//! copies and shaders wrote written back into title memory, then its
//! bottom-of-pipe writes and the title fence signals queued behind it) in
//! submission order once the GPU has run it: by our completion thread, or
//! by whichever thread first needs it (a later submission's step touching
//! memory it writes, a resubmission of one of its lists, a present).
//!
//! `XTP_SYNC_SUBMIT=1` brings back the old synchronous model, to bisect a
//! failure (gpu.md, risk 2): each submission is waited for and finished
//! before `execute` returns, and there is no completion thread.
//!
//! A present finishes every submission before it returns, unless one frame
//! is let run on while the title prepares the next (`XTP_FRAMES_IN_FLIGHT`,
//! [`frames`]).
//!
//! `XTP_CARD_AWAKE=1` ([`awake`]) runs a small compute loop on its own compute
//! queue between submissions so the driver does not lower GPU clocks during
//! short CPU gaps (off by default).

mod awake;
pub(crate) mod buffers;
mod frames;
pub(crate) mod preparation;
#[cfg(test)]
pub(crate) mod submissions;
mod timeline;
mod twins;
/// For tests elsewhere that hold submissions in flight.
#[cfg(test)]
pub(crate) use submissions::Gate;

use std::cell::Cell;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use preparation::Phase;
use timeline::{Timeline, Touches};

use windows::Win32::Foundation::{HANDLE, HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::{Interface, PCWSTR, w};

/// Resource states without the console's own bits (compression hints from
/// bit 25 up), which the host rejects.
pub(crate) fn pc_state(state: u32) -> u32 {
    state & 0x01ff_ffff
}

/// Milliseconds since the first call.
pub(crate) fn now_ms() -> u64 {
    static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
    EPOCH
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_millis() as u64
}

/// Available physical RAM on the host in bytes, from `GlobalMemoryStatusEx`.
pub(crate) fn available_physical_ram() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut status = MEMORYSTATUSEX {
        dwLength: size_of::<MEMORYSTATUSEX>() as u32,
        ..Default::default()
    };
    // SAFETY: a sized structure of ours.
    let ok = unsafe { GlobalMemoryStatusEx(&mut status) } != 0;
    ok.then_some(status.ullAvailPhys)
}

/// Idle timeout in milliseconds before trimming file-backed caches from the
/// working set, scaled down when free physical RAM is low.
pub(crate) fn ram_idle_ms(avail_phys: Option<u64>) -> u64 {
    const GIB: u64 = 1024 * 1024 * 1024;
    const NORMAL_MS: u64 = 10_000;
    const MODERATE_MS: u64 = 5_000;
    const HIGH_MS: u64 = 2_000;
    let Some(avail) = avail_phys else {
        return NORMAL_MS;
    };
    if avail < 2 * GIB {
        HIGH_MS
    } else if avail < 4 * GIB {
        MODERATE_MS
    } else {
        NORMAL_MS
    }
}

/// A console state as a host texture state: PC bits only, and none of the
/// buffer-only ones the console allows on textures (vertex, constant and
/// index buffer, stream out, indirect arguments, as in "generic read").
pub(crate) fn texture_state(state: u32) -> u32 {
    const BUFFER_ONLY: u32 = 0x1 | 0x2 | 0x100 | 0x200;
    pc_state(state) & !BUFFER_ONLY
}

/// Render target views live only until their command is recorded, so a
/// ring of host descriptors is enough.
const RTV_RING: u32 = 1024;

pub(crate) struct Host {
    /// Optional background compute loop (`XTP_CARD_AWAKE=1`) kept ahead of
    /// `device` so it shuts down and releases its GPU objects first on drop.
    awake: Option<awake::Awake>,
    pub(crate) device: ID3D12Device,
    pub(crate) depth_bounds: bool,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    /// The last value signalled on `fence`, behind the submission lock.
    submitted: Mutex<u64>,
    /// The value the last submission handed to the queue signals, stored
    /// before it is handed over ([`Host::executing`]).
    executing: AtomicU64,
    /// The value the last present's copy signals: the frame a flip wait
    /// and the next present wait for (0: none yet).
    presented: AtomicU64,
    /// Submissions not finished yet, oldest first.
    timeline: Timeline<Pending>,
    /// Whether the completion thread runs.
    finisher: std::sync::Once,
    /// When the GPU was last seen done with a submission (for the time
    /// the next one ran), behind the timeline's finishing.
    seen_done: Mutex<Option<std::time::Instant>>,
    rtvs: ID3D12DescriptorHeap,
    rtv_size: u32,
    /// A ring of depth views, used as `rtvs`.
    dsvs: ID3D12DescriptorHeap,
    dsv_size: u32,
    next_dsv: AtomicU32,
    /// With `XTP_D3D12_DEBUG` set (diagnostics builds): the debug
    /// layer's messages.
    messages: Option<ID3D12InfoQueue>,
    next_rtv: AtomicU32,
    presenter: Mutex<Option<Presenter>>,
}

// SAFETY: D3D12 devices, queues, fences and descriptor heaps are
// free-threaded; the timeline's lists are pinned while in it (`Pending`).
unsafe impl Send for Host {}
// SAFETY: as above.
unsafe impl Sync for Host {}

/// Time spent on submissions since the last frame-rate line, in
/// microseconds: the GPU running them (from each one's submission, or the
/// end of the one before, until it was seen done), the title's threads
/// waiting for them (for the GPU, or for finishing: [`Host::waiting`]),
/// and of that waiting, the part at frame ends (flip waits and presents:
/// [`Host::at_frame_end`]).
static SUBMISSION_US: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// The parts of [`SUBMISSION_US`].
const GPU_US: usize = 0;
const WAITED_US: usize = 1;
const FRAME_END_US: usize = 2;

/// Where the title's threads waited for submissions: the code that asked
/// for each wait (the caller of `Host::settle`, `Host::drain` or a step's
/// check, by `#[track_caller]`), with the microseconds and waits there.
#[cfg(any(test, feature = "diagnostics"))]
#[derive(Default)]
struct Waits(Vec<(&'static std::panic::Location<'static>, u64, u64)>);

#[cfg(any(test, feature = "diagnostics"))]
impl Waits {
    fn add(&mut self, at: &'static std::panic::Location<'static>, us: u64) {
        match self.0.iter_mut().find(|(place, ..)| *place == at) {
            Some((_, total, n)) => (*total, *n) = (*total + us, *n + 1),
            None => self.0.push((at, us, 1)),
        }
    }

    /// The `n` places waited at longest, over `frames` frames, as `file:line
    /// <ms> ms/frame (<waits>/frame)`; the counts start again.
    fn take(&mut self, frames: f64, n: usize) -> Vec<String> {
        let mut waits = std::mem::take(&mut self.0);
        waits.sort_by_key(|&(_, us, _)| std::cmp::Reverse(us));
        let frames = frames.max(1.0);
        waits
            .into_iter()
            .take(n)
            .map(|(at, us, count)| {
                let file = at.file().rsplit(['/', '\\']).next().unwrap_or(at.file());
                format!(
                    "{file}:{} {:.1} ms/frame ({:.1}/frame)",
                    at.line(),
                    us as f64 / 1e3 / frames,
                    count as f64 / frames
                )
            })
            .collect()
    }
}

/// The waits since the last frame-rate second (diagnostics).
#[cfg(feature = "diagnostics")]
static WAITS: Mutex<Waits> = Mutex::new(Waits(Vec::new()));

/// After submissions: GPU copy destinations and shader-written ranges
/// written back into title memory (microseconds, bytes written), since the
/// last frame-rate line.
static COPY_BACK: [std::sync::atomic::AtomicU64; 2] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// How long a submission took: on the GPU, and in all (preparing, GPU,
/// copying back, bottom-of-pipe writes).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct Ran {
    pub(crate) gpu: std::time::Duration,
    pub(crate) total: std::time::Duration,
}

impl std::ops::Add for Ran {
    type Output = Ran;

    fn add(self, other: Ran) -> Ran {
        Ran {
            gpu: self.gpu + other.gpu,
            total: self.total + other.total,
        }
    }
}

/// What the parts of a title submission run in parts ([`Host::run_as`])
/// took so far: their lists, and their times, each added once it ran.
type Parts = std::sync::Arc<Mutex<(usize, Ran)>>;

/// A title submission, or a part of one, as the frame-rate line counts it.
enum Title {
    /// A whole submission, or the last part of one with what the parts
    /// before it took: counted once it has run.
    Last(Option<Parts>),
    /// A part before the last: what it took goes to the last.
    Part(Parts),
}

impl Title {
    /// What the frame-rate line counts once this has run (`lists` lists,
    /// `ran`), if anything yet. Parts finish in order, the last after the
    /// others (`Host::run_as` waits for each before the next).
    fn counted(self, lists: usize, ran: Ran) -> Option<(usize, Ran)> {
        let add = |parts: &Parts| {
            let mut so_far = parts.lock().unwrap_or_else(|e| e.into_inner());
            *so_far = (so_far.0 + lists, so_far.1 + ran);
            *so_far
        };
        match self {
            Title::Last(None) => Some((lists, ran)),
            Title::Last(Some(parts)) => Some(add(&parts)),
            Title::Part(parts) => {
                add(&parts);
                None
            }
        }
    }
}

/// The title's submissions since the last frame-rate line (one per
/// `ExecuteCommandLists`, however many parts it ran in; ours, the present
/// copy, are not counted).
static TITLE: Mutex<Submissions> = Mutex::new(Submissions::new());

/// Title submissions over a frame-rate window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Submissions {
    count: u64,
    lists: u64,
    /// The longest one in all: microseconds in all and on the GPU, and its
    /// lists.
    longest_us: u64,
    longest_gpu_us: u64,
    longest_lists: u64,
}

/// [`Submissions`] per frame, as the frame-rate line shows them.
#[derive(Debug, Clone, Copy, PartialEq)]
struct PerFrame {
    submissions: f64,
    /// Lists per submission.
    lists: f64,
    longest_ms: f64,
    longest_gpu_ms: f64,
    longest_lists: u64,
}

impl Submissions {
    const fn new() -> Self {
        Submissions {
            count: 0,
            lists: 0,
            longest_us: 0,
            longest_gpu_us: 0,
            longest_lists: 0,
        }
    }

    fn add(&mut self, lists: usize, ran: Ran) {
        self.count += 1;
        self.lists += lists as u64;
        let us = ran.total.as_micros() as u64;
        if us >= self.longest_us {
            self.longest_us = us;
            self.longest_gpu_us = ran.gpu.as_micros() as u64;
            self.longest_lists = lists as u64;
        }
    }

    /// Over `frames` frames.
    fn per_frame(&self, frames: f64) -> PerFrame {
        PerFrame {
            submissions: self.count as f64 / frames.max(1.0),
            lists: self.lists as f64 / (self.count.max(1)) as f64,
            longest_ms: self.longest_us as f64 / 1e3,
            longest_gpu_ms: self.longest_gpu_us as f64 / 1e3,
            longest_lists: self.longest_lists,
        }
    }
}

/// Ranges `(of, start, len)` sorted by `key(of)` then start, those of one
/// key that overlap or touch merged into one.
pub(crate) fn merge_ranges<T>(
    mut ranges: Vec<(T, usize, usize)>,
    key: impl Fn(&T) -> usize,
) -> Vec<(T, usize, usize)> {
    ranges.sort_by_key(|(of, start, _)| (key(of), *start));
    let mut merged: Vec<(T, usize, usize)> = Vec::with_capacity(ranges.len());
    for (of, start, len) in ranges {
        match merged.last_mut() {
            Some((last, from, n)) if key(last) == key(&of) && start <= *from + *n => {
                *n = (*n).max(start + len - *from);
            }
            _ => merged.push((of, start, len)),
        }
    }
    merged
}

/// Of a frame of `frame_ms`, the time outside submissions (the title's own
/// work, recording and pipeline waits) when they took `inside_ms`.
fn outside(frame_ms: f64, inside_ms: f64) -> f64 {
    (frame_ms - inside_ms).max(0.0)
}

/// Of the GPU's time and the copying back, what the title's threads did
/// not wait for (it ran while they did other work): what is left of both
/// once `waited_ms` is taken off, never below 0. With every submission
/// waited for (`XTP_SYNC_SUBMIT`), about 0.
fn overlapped(gpu_ms: f64, back_ms: f64, waited_ms: f64) -> f64 {
    (gpu_ms + back_ms - waited_ms).max(0.0)
}

/// Whether `XTP_SYNC_SUBMIT`'s value asks for synchronous submission.
fn sync_asked(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_some_and(|v| !v.is_empty() && v != "0")
}

/// Whether submissions are waited for and finished in `execute`, as before
/// G8 (`XTP_SYNC_SUBMIT` set and not `0`): to bisect a failure the
/// asynchronous model may cause.
pub(crate) fn sync_submit() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = sync_asked(std::env::var_os("XTP_SYNC_SUBMIT").as_deref());
        if on {
            tracing::info!("XTP_SYNC_SUBMIT: every submission is waited for");
        }
        on
    })
}

/// How many of `runs` come before the first one named a second time (all
/// of them when none is): where a submission is split ([`Host::run_as`]).
/// Submissions name few lists; a search beats building a set.
fn first_repeat<T>(runs: &[&T]) -> usize {
    (1..runs.len())
        .find(|&i| runs[..i].iter().any(|r| std::ptr::eq(*r, runs[i])))
        .unwrap_or(runs.len())
}

/// Calls, waits before a repeated list, and microseconds spent in those waits.
#[cfg(feature = "diagnostics")]
static SPLITS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// What the runtime does when the host GPU fails for good: report it like a
/// crash and end the run. The console has no such state, and carrying on
/// draws nothing at seconds per frame.
static ON_FAILURE: OnceLock<fn(&str) -> !> = OnceLock::new();

/// Sets what a host GPU failure does (the runtime's crash report).
pub fn on_failure(report: fn(&str) -> !) {
    let _ = ON_FAILURE.set(report);
}

/// Ends the run on a host GPU failure: what failed, the removal reason and
/// the debug layer's first error, if any (`XTP_D3D12_DEBUG=1` in a
/// diagnostics build names the command).
pub(crate) fn fail(what: std::fmt::Arguments) -> ! {
    let mut text = what.to_string();
    if let Some(host) = host() {
        // SAFETY: a plain query.
        if let Err(e) = unsafe { host.device.GetDeviceRemovedReason() } {
            text.push_str(&format!("; device removed: {e}"));
        }
        match host.log_messages() {
            Some(first) => text.push_str(&format!("; debug layer: {first}")),
            None if host.messages.is_none() => {
                text.push_str("; a diagnostics build with XTP_D3D12_DEBUG=1 names the command")
            }
            None => {}
        }
    }
    tracing::error!("{text}");
    match ON_FAILURE.get() {
        Some(report) => report(&text),
        None => std::process::abort(),
    }
}

static HOST: OnceLock<Result<Host, String>> = OnceLock::new();

/// The host GPU if it was made (never makes it): for what has to wait for
/// or follow its submissions, of which there are none without it.
pub(crate) fn made() -> Option<&'static Host> {
    HOST.get()?.as_ref().ok()
}

/// The host GPU, created on first use; `None` without one (a run then
/// behaves as before: nothing executes).
pub(crate) fn host() -> Option<&'static Host> {
    created().as_ref().ok()
}

/// The host GPU, created on first use, or why there is none (tests print
/// it when they skip: `crate::testgpu`).
pub(crate) fn created() -> &'static Result<Host, String> {
    let made = HOST.get_or_init(|| {
        let made = software().and_then(|software| create(software).map_err(|e| e.to_string()));
        if let Err(e) = &made {
            tracing::warn!(error = %e, "no host D3D12 device; command lists will not run");
        }
        made
    });
    // In tests, the debug layer reports to `testgpu` from the device's first
    // use on, whichever way a test reached it.
    #[cfg(test)]
    if let Ok(host) = made {
        crate::testgpu::watch(host);
    }
    made
}

/// Whether the host device is made on the software adapter (WARP): in
/// diagnostics builds with `XTP_D3D12_WARP`, where a shader's wild access
/// is an access violation in this process (a crash report with the address
/// and instruction) instead of a GPU fault and driver reset.
#[cfg(not(test))]
fn software() -> Result<bool, String> {
    Ok(diagnostics_warp())
}

/// In tests, as `XTP_TEST_ADAPTER` says (`crate::testgpu`), else as in a
/// run.
#[cfg(test)]
fn software() -> Result<bool, String> {
    crate::testgpu::software().unwrap_or_else(|| Ok(diagnostics_warp()))
}

fn diagnostics_warp() -> bool {
    cfg!(feature = "diagnostics") && std::env::var_os("XTP_D3D12_WARP").is_some()
}

fn create(software: bool) -> windows::core::Result<Host> {
    // SAFETY: plain D3D12 object creation with valid descriptions.
    unsafe {
        // The debug layer only in builds with diagnostics.
        let debug = cfg!(feature = "diagnostics") && std::env::var_os("XTP_D3D12_DEBUG").is_some();
        if debug {
            let layer =
                match crate::agility::configuration_interface::<ID3D12Debug>(&CLSID_D3D12Debug) {
                    Ok(layer) => {
                        layer.EnableDebugLayer();
                        Some(layer)
                    }
                    Err(e) if crate::agility::is_loaded() => {
                        tracing::warn!(
                            error = %e,
                            "no D3D12 debug layer (XTP_D3D12_DEBUG requires \
                             d3d12SDKLayers.dll from Microsoft.Direct3D.D3D12 \
                             exactly 1.614.0 beside D3D12Core.dll)"
                        );
                        None
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "no D3D12 debug layer");
                        None
                    }
                };
            // GPU-based validation (`XTP_D3D12_GBV`): out-of-range
            // descriptor heap indices and the like become messages naming
            // the command instead of page faults. Without state tracking
            // (directly indexed textures are not transitioned).
            if std::env::var_os("XTP_D3D12_GBV").is_some()
                && let Some(l) = &layer
            {
                if let Ok(d1) = l.cast::<ID3D12Debug1>() {
                    d1.SetEnableGPUBasedValidation(true);
                }
                if let Ok(d2) = l.cast::<ID3D12Debug2>() {
                    d2.SetGPUBasedValidationFlags(
                        D3D12_GPU_BASED_VALIDATION_FLAGS_DISABLE_STATE_TRACKING,
                    );
                }
                tracing::info!("GPU-based validation on");
            }
        }
        // Diagnostics: device-removed data (the command that faulted, the
        // faulting address), read when the device is removed.
        if dred_on()
            && let Ok(dred) = crate::agility::configuration_interface::<
                ID3D12DeviceRemovedExtendedDataSettings,
            >(&CLSID_D3D12DeviceRemovedExtendedData)
        {
            dred.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            dred.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
        }
        // DRED 1.2: the markers' texts kept with their breadcrumbs.
        crate::dred::enable();
        let mut device: Option<ID3D12Device> = None;
        if software {
            let adapter = warp_adapter()?;
            crate::agility::create_device(&adapter, &mut device)?;
            tracing::info!("host device on the software adapter (WARP)");
        } else {
            let adapter = crate::dxgi::host_adapter();
            crate::agility::create_device(adapter.as_ref(), &mut device)?;
            crate::tdr::log();
        }
        let device = device.ok_or_else(windows::core::Error::empty)?;
        // Prime previously recorded static samplers before any root signature
        // with static samplers is created on this device.
        crate::pipeline::samplers::prime(&device);
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS2::default();
        let depth_bounds = device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS2,
                (&raw mut options).cast(),
                size_of_val(&options) as u32,
            )
            .is_ok()
            && options.DepthBoundsTestSupported.as_bool();
        let messages = debug
            .then(|| device.cast::<ID3D12InfoQueue>().ok())
            .flatten();
        let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            ..Default::default()
        })?;
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let rtvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let rtv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        let dsvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let dsv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
        // Heaps for buffers, made from now on by threads of their own.
        buffers::start(&device);
        let awake = awake::Awake::start_if_enabled(&device);
        tracing::info!(
            synchronous = sync_submit(),
            depth_bounds,
            "host D3D12 device ready"
        );
        Ok(Host {
            awake,
            device,
            depth_bounds,
            queue,
            fence,
            submitted: Mutex::new(0),
            executing: AtomicU64::new(0),
            presented: AtomicU64::new(0),
            timeline: Timeline::new(),
            finisher: std::sync::Once::new(),
            seen_done: Mutex::new(None),
            rtvs,
            rtv_size,
            dsvs,
            dsv_size,
            next_dsv: AtomicU32::new(0),
            messages,
            next_rtv: AtomicU32::new(0),
            presenter: Mutex::new(None),
        })
    }
}

/// DXGI may return hardware for `EnumWarpAdapter` (notably under Proton).
/// Reject it before creating a device in diagnostics' software-only mode.
fn warp_adapter() -> windows::core::Result<IDXGIAdapter> {
    // SAFETY: plain DXGI adapter queries.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let adapter: IDXGIAdapter = factory.EnumWarpAdapter()?;
        let desc = adapter.cast::<IDXGIAdapter1>()?.GetDesc1()?;
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
            tracing::warn!(
                vendor = format_args!("{:#x}", desc.VendorId),
                flags = format_args!("{:#x}", desc.Flags),
                "DXGI did not identify the WARP adapter as software; refusing device"
            );
            return Err(windows::core::Error::from_hresult(
                windows::Win32::Foundation::E_FAIL,
            ));
        }
        Ok(adapter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn submission_callbacks_run_after_the_heap_sync_join() {
        let Some(gpu) = crate::testgpu::device(
            "host::tests::submission_callbacks_run_after_the_heap_sync_join",
        ) else {
            return;
        };
        let list = gpu.host().new_list().unwrap();
        list.submitting(|| panic!("a list without callbacks does not need an early join"));

        let joined = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let observed = joined.clone();
        list.at_submit(move |_| assert!(observed.load(Ordering::Acquire)));
        list.submitting(|| joined.store(true, Ordering::Release));
        assert!(joined.load(Ordering::Acquire));
    }

    #[test]
    fn fills_merge_only_across_contiguous_bytes_of_one_upload() {
        let Some(gpu) =
            crate::testgpu::device("fills_merge_only_across_contiguous_bytes_of_one_upload")
        else {
            return;
        };
        let list = gpu.host().new_list().unwrap();
        let mut source = [0u8; 10];
        let mut upload = [0xffu8; 10];
        let src = source.as_ptr() as usize;
        let dst = upload.as_mut_ptr() as usize;
        list.defer_fill_at(dst, 0, src, 2);
        list.defer_fill_at(dst, 2, src + 2, 2);
        // Deux mappages qui se touchent, puis un trou dans la source.
        list.defer_fill_at(dst + 4, 0, src + 4, 1);
        list.defer_fill_at(dst + 4, 1, src + 6, 1);
        // Un trou dans la destination, puis une opération intercalée.
        list.defer_fill_at(dst + 4, 3, src + 7, 1);
        list.defer_host((0, 0), || {});
        list.defer_fill_at(dst + 4, 4, src + 8, 2);
        assert!(matches!(
            list.deferred().copies.as_slice(),
            [
                Step::Fill(_, _, 4),
                Step::Fill(_, _, 1),
                Step::Fill(_, _, 1),
                Step::Fill(_, _, 1),
                Step::Host(..),
                Step::Fill(_, _, 2),
            ]
        ));
        source.copy_from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let mut batch = Batch::default();
        list.run_deferred(&mut batch);
        assert_eq!(batch.work.fill_bytes, 9);
        assert_eq!(batch.work.merged_fills, 1);
        assert_eq!(batch.work.copy_bytes, 0);
        assert_eq!(batch.work.steps[preparation::Kind::Fill as usize], 5);
        assert_eq!(batch.work.steps[preparation::Kind::Host as usize], 1);
        assert_eq!(upload, [0, 1, 2, 3, 4, 6, 0xff, 7, 8, 9]);
    }

    #[test]
    fn refreshes_merge_overlaps_and_neighbours_until_a_write() {
        let Some(gpu) =
            crate::testgpu::device("refreshes_merge_overlaps_and_neighbours_until_a_write")
        else {
            return;
        };
        let list = gpu.host().new_list().unwrap();
        let buffer = crate::testgpu::TitleBuffer::new(256);
        let shadow = buffer.shadow();
        let base = buffer.address() as usize;
        for (offset, len) in [(8, 8), (4, 4), (12, 8), (10, 2), (24, 4), (8, 8)] {
            list.defer_refresh(&shadow, base + offset, len);
        }
        list.defer((base + 24, 4), (0, 0), || {});
        list.defer_refresh(&shadow, base + 24, 4);
        assert!(matches!(list.deferred().copies.as_slice(), [
            Step::Refresh(_, a, 16), Step::Refresh(_, b, 4),
            Step::Run(..), Step::Refresh(_, c, 4),
        ] if *a == base + 4 && *b == base + 24 && *c == *b));
    }

    #[test]
    fn batch_keeps_gaps_and_flushes_each_range_hit_by_a_copy() {
        let Some(_gpu) =
            crate::testgpu::device("batch_keeps_gaps_and_flushes_each_range_hit_by_a_copy")
        else {
            return;
        };
        let buffer = crate::testgpu::TitleBuffer::new(256);
        let base = buffer.address() as usize;
        let shadow = buffer.shadow();
        let mut batch = Batch::default();
        for offset in [24, 8, 12, 48, 0, 4] {
            batch.add(&shadow, base + offset, 4);
        }
        let ranges = |b: &Batch<'_>| {
            b.refreshes
                .iter()
                .map(|(_, a, z)| (a - base, z - base))
                .collect::<Vec<_>>()
        };
        assert_eq!(ranges(&batch), [(0, 16), (24, 28), (48, 52)]);
        batch.flush_before_copy(&shadow, base + 16, 8);
        assert_eq!(ranges(&batch).len(), 3);
        batch.flush_before_copy(&shadow, base + 15, 10);
        assert_eq!(ranges(&batch), [(48, 52)]);
    }

    #[test]
    fn wide_fills_finish_before_a_host_callback_and_keep_overlapping_fill_order() {
        let Some(gpu) = crate::testgpu::device("wide_fills_finish_before_a_host_callback") else {
            return;
        };
        let list = gpu.host().new_list().unwrap();
        let source = Box::new([0x11u8, 0x22, 0x33]);
        let mut upload = Box::new([0u8; 4]);
        let src = source.as_ptr();
        let dst = upload.as_mut_ptr();
        for i in 0..128 {
            let offset = i % 2;
            // SAFETY: each pointer is inside its own allocation.
            unsafe { list.defer_fill(dst.add(offset), src.add(offset), 1) };
        }
        let addr = dst as usize;
        list.defer_host((0, 0), move || {
            // SAFETY: the test owns this upload allocation through the callback.
            let filled = unsafe { std::slice::from_raw_parts(addr as *const u8, 4) };
            assert_eq!(filled, [0x11, 0x22, 0, 0]);
        });
        // SAFETY: both pointers are inside their allocations.
        unsafe { list.defer_fill(dst.add(1), src.add(2), 1) };
        list.run_deferred_with(&mut Batch::default(), true);
        assert_eq!(*upload, [0x11, 0x33, 0, 0]);

        let mut batch = Batch::default();
        let src = src as usize;
        batch
            .fills
            .extend((0..128).map(|i| (addr + i % 2, src + i % 3, 1)));
        assert!(!batch.fills_disjoint());
        batch.fill_with(true);
        // 126 is the last fill of byte 0 and 127 is the last of byte 1.
        assert_eq!(*upload, [0x11, 0x22, 0, 0]);
    }

    /// Concurrent shadow refreshes, batched fills, and parallel copy-back over
    /// 16 shadows on WARP produce the exact same bytes as the single-threaded
    /// path, preserving title writes (including three-way title-wins bytes).
    #[test]
    fn wide_refreshes_and_copy_back_over_16_shadows_match_single_threaded() {
        use crate::watch::PAGE;

        let test =
            "host::tests::wide_refreshes_and_copy_back_over_16_shadows_match_single_threaded";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let kernel = crate::testgpu::Kernel::new(host);

        const SHADOWS: usize = 16;
        const LEN: usize = 2 * PAGE;
        const FILLS: usize = 140;

        let initial = |shadow: usize, word: usize| {
            ((shadow as u32 + 1) << 24) ^ (word as u32).wrapping_mul(0x9e37_79b9)
        };
        let title_byte = |shadow, offset: usize| {
            initial(shadow, offset / 4).to_le_bytes()[offset % 4] ^ (0x50 + shadow) as u8
        };
        let run_case = |wide: bool| -> (Vec<Vec<u8>>, Vec<Vec<u8>>, Vec<u8>) {
            // 1. Batched upload fills above the parallel threshold (FEW = 96).
            let src_fill: Vec<u8> = (0..FILLS * 16)
                .map(|i| (i.wrapping_mul(37) + 11) as u8)
                .collect();
            let mut dst_fill = vec![0xeeu8; FILLS * 16];
            let mut fill_batch = Batch::default();
            for i in 0..FILLS {
                fill_batch.fills.push((
                    dst_fill.as_mut_ptr() as usize + i * 16,
                    src_fill.as_ptr() as usize + i * 16,
                    16,
                ));
            }
            fill_batch.fill_with(wide);
            assert_eq!(dst_fill, src_fill, "fills must match source (wide={wide})");

            // 2. 16 independent title buffers + shadows.
            let buffers: Vec<crate::testgpu::TitleBuffer> = (0..SHADOWS)
                .map(|_| crate::testgpu::TitleBuffer::new(LEN))
                .collect();
            let shadows: Vec<std::sync::Arc<crate::shadow::Shadow>> =
                buffers.iter().map(|b| b.shadow()).collect();

            for (s, buf) in buffers.iter().enumerate() {
                buf.set_words(|w| initial(s, w));
            }

            // Disjoint intervals per shadow, added interleaved across shadows so
            // Batch::flush_all must group them per shadow cleanly.
            let mut batch = Batch::default();
            for &(offset, len) in &[(PAGE + 0x200, 0x400), (0x100, 0x300), (0x800, 0x200)] {
                for (s, buf) in buffers.iter().enumerate() {
                    batch.add(&shadows[s], buf.address() as usize + offset, len);
                }
            }
            batch.flush_all(wide);
            for (s, buf) in buffers.iter().enumerate() {
                let base = buf.address() as usize;
                // SAFETY: `buf` owns `LEN` bytes of committed title memory.
                let title = unsafe { std::slice::from_raw_parts(base as *const u8, LEN) };
                for &(offset, len) in &[(0x100, 0x300), (0x800, 0x200), (PAGE + 0x200, 0x400)] {
                    assert_eq!(
                        shadows[s].host_bytes(base + offset, len),
                        title[offset..offset + len],
                        "disjoint refresh shadow {s} offset {offset:#x} (wide={wide})"
                    );
                }
            }

            // Refresh all 16 shadows whole, then take snapshots for one submission.
            for (s, buf) in buffers.iter().enumerate() {
                batch.add(&shadows[s], buf.address() as usize, LEN);
            }
            batch.flush_all(wide);

            let sub = crate::shadow::submission();
            for (s, buf) in buffers.iter().enumerate() {
                shadows[s].writing(sub, buf.address() as usize + 1, LEN - 3);
            }

            // Run a real WARP compute dispatch writing 256 bytes into both pages
            // of every shadow.
            let list = host.new_list().unwrap();
            for (s, buf) in buffers.iter().enumerate() {
                let base = buf.address();
                kernel.dispatch_fill_raw(&list, &shadows[s], base + 0x100);
                kernel.dispatch_fill_raw(&list, &shadows[s], base + PAGE as u64 + 0x200);
            }
            list.close();
            host.execute(&[&list]);
            host.drain();

            // Simulate concurrent title CPU writes while the GPU ran:
            // - overlapping with the GPU write at 0x108..0x118 (title must win)
            // - outside the GPU write at 0x400..0x420 and PAGE + 0x500..0x520
            for (s, buf) in buffers.iter().enumerate() {
                let base = buf.address() as usize;
                let tag = (0x50 + s) as u8;
                // SAFETY: inside `buf`'s committed allocation.
                unsafe {
                    for offset in 0x108..0x118 {
                        // Flip bits relative to the exact preimage so every
                        // byte here counts as a title write in the merge.
                        ((base + offset) as *mut u8).write(title_byte(s, offset));
                    }
                    std::ptr::write_bytes((base + 0x400) as *mut u8, tag ^ 0x2a, 32);
                    std::ptr::write_bytes((base + PAGE + 0x500) as *mut u8, tag ^ 0x55, 32);
                }
            }

            let back: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> = buffers
                .iter()
                .enumerate()
                .map(|(s, buf)| (shadows[s].clone(), buf.address() as usize + 1, LEN - 3))
                .collect();
            let written = crate::shadow::copy_back_all_with(sub, &back, wide);
            assert!(written > 0, "expected GPU bytes written back (wide={wide})");

            let mut titles = Vec::with_capacity(SHADOWS);
            let mut hosts = Vec::with_capacity(SHADOWS);
            for (s, buf) in buffers.iter().enumerate() {
                let base = buf.address() as usize;
                // SAFETY: inside `buf`'s committed allocation.
                let title = unsafe { std::slice::from_raw_parts(base as *const u8, LEN) }.to_vec();
                let host_mem = shadows[s].host_bytes(base, LEN);

                // Verify three-way title-wins and GPU-only bytes on every shadow.
                let tag = (0x50 + s) as u8;
                assert!(
                    (0x108..0x118).all(|i| title[i] == title_byte(s, i)),
                    "shadow {s}: overlapping title write must win (wide={wide})"
                );
                assert!(
                    (0x108..0x118).all(|i| host_mem[i] == title_byte(s, i)),
                    "shadow {s}: host copy must take winning title write (wide={wide})"
                );
                assert!(
                    title[0x400..0x420].iter().all(|&b| b == tag ^ 0x2a),
                    "shadow {s}: title write on page 0 must be preserved (wide={wide})"
                );
                assert!(
                    title[PAGE + 0x500..PAGE + 0x520]
                        .iter()
                        .all(|&b| b == tag ^ 0x55),
                    "shadow {s}: title write on page 1 must be preserved (wide={wide})"
                );
                // First GPU word at 0x100 (word 0 of pattern) and 0x118 (word 6).
                assert_eq!(
                    u32::from_le_bytes(title[0x100..0x104].try_into().unwrap()),
                    crate::testgpu::pattern(0),
                    "shadow {s}: GPU write at 0x100 must land (wide={wide})"
                );
                assert_eq!(
                    u32::from_le_bytes(title[0x118..0x11c].try_into().unwrap()),
                    crate::testgpu::pattern(6),
                    "shadow {s}: GPU write at 0x118 must land (wide={wide})"
                );
                assert_eq!(
                    u32::from_le_bytes(title[PAGE + 0x200..PAGE + 0x204].try_into().unwrap()),
                    crate::testgpu::pattern(0),
                    "shadow {s}: GPU write on page 1 must land (wide={wide})"
                );

                titles.push(title);
                hosts.push(host_mem);
            }
            (titles, hosts, dst_fill)
        };

        let (single_titles, single_hosts, single_fills) = run_case(false);
        let (wide_titles, wide_hosts, wide_fills) = run_case(true);
        assert_eq!(
            wide_fills, single_fills,
            "wide fills must match single-threaded"
        );
        for s in 0..SHADOWS {
            assert_eq!(
                wide_titles[s], single_titles[s],
                "shadow {s}: wide title memory must match single-threaded byte-for-byte"
            );
            assert_eq!(
                wide_hosts[s], single_hosts[s],
                "shadow {s}: wide host copy must match single-threaded byte-for-byte"
            );
        }

        // End-to-end submission through Host::execute / Host::complete over 16 shadows.
        let src_buffers: Vec<crate::testgpu::TitleBuffer> = (0..SHADOWS)
            .map(|_| crate::testgpu::TitleBuffer::new(4096))
            .collect();
        let dst_buffers: Vec<crate::testgpu::TitleBuffer> = (0..SHADOWS)
            .map(|_| crate::testgpu::TitleBuffer::new(4096))
            .collect();
        for (s, buf) in src_buffers.iter().enumerate() {
            buf.set_words(|w| ((s as u32 + 1) << 16) | (w as u32));
        }
        let list = crate::testgpu::TitleList::new();
        for (src, dst) in src_buffers.iter().zip(&dst_buffers) {
            kernel.dispatch(
                list.host(),
                crate::testgpu::Pass::Copy,
                src.address(),
                dst.address(),
            );
        }
        list.run();
        for (s, dst) in dst_buffers.iter().enumerate() {
            let expected: Vec<u32> = (0..64).map(|w| ((s as u32 + 1) << 16) | w).collect();
            assert_eq!(
                &dst.words()[..64],
                &expected[..],
                "end-to-end WARP copy over 16 shadows failed at shadow {s}"
            );
        }
    }

    #[cfg(feature = "diagnostics")]
    #[test]
    fn recorded_preserves_both_diagnostic_samples_without_deadlock() {
        let test = "host::tests::recorded_preserves_both_diagnostic_samples_without_deadlock";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let (done, result) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let list = host.new_list().unwrap();
            let snapshot = list.recorded();
            let deferred = list.deferred();
            done.send(
                std::sync::Arc::ptr_eq(&snapshot.dispatch_samples, &deferred.dispatch_samples)
                    && std::sync::Arc::ptr_eq(&snapshot.meshes, &deferred.meshes),
            )
            .unwrap();
        });
        // Un second verrouillage du même mutex bloquerait toute la suite de tests.
        assert!(
            result
                .recv_timeout(std::time::Duration::from_secs(10))
                .expect("recorded")
        );
        worker.join().unwrap();
    }

    /// A submission is split before each list it names again, by identity
    /// (equal values in two places are two lists).
    #[test]
    fn a_submission_is_split_before_a_list_named_again() {
        let (a, b, c, a_twin) = (1, 2, 3, 1);
        assert_eq!(first_repeat::<i32>(&[]), 0);
        assert_eq!(first_repeat(&[&a]), 1);
        assert_eq!(first_repeat(&[&a, &b, &c]), 3);
        assert_eq!(first_repeat(&[&a, &a]), 1);
        assert_eq!(first_repeat(&[&a, &b, &a]), 2);
        assert_eq!(first_repeat(&[&a, &b, &c, &b, &a]), 3);
        assert_eq!(first_repeat(&[&a, &a_twin]), 2);
    }

    /// Where a list stopped, the device-removed report names the commands
    /// between the barriers around it by their markers, and the noted
    /// commands after them: DRED's count may lag the GPU (xcl#1592).
    #[test]
    fn a_stopped_list_names_the_commands_around_and_after_it() {
        let (barrier, marker) = (
            D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER,
            D3D12_AUTO_BREADCRUMB_OP_SETMARKER,
        );
        let (dispatch, draw) = (
            D3D12_AUTO_BREADCRUMB_OP_DISPATCH,
            D3D12_AUTO_BREADCRUMB_OP_DRAWINSTANCED,
        );
        let history = [
            barrier, marker, dispatch, barrier, marker, dispatch, marker, draw,
        ];
        let markers = [
            (1, "first".to_owned()),
            (4, "second".to_owned()),
            (6, "third".to_owned()),
        ];
        let text = in_flight(&history, &markers, 2);
        assert!(
            text.starts_with(", commands 1..3 between its barriers:"),
            "{text}"
        );
        assert!(
            text.contains(&format!("; command 2 (op {}): first", dispatch.0)),
            "{text}"
        );
        let after = format!(
            "; after them (the count may lag the GPU): command 5 (op {}): second; command 7 \
             (op {}): third",
            dispatch.0, draw.0
        );
        assert!(text.ends_with(&after), "{text}");
    }

    #[test]
    fn warp_adapter_accepts_only_software() {
        // This probes DXGI without creating a D3D12 device or running a title.
        // SAFETY: plain DXGI adapter queries.
        let raw = unsafe {
            let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            else {
                return;
            };
            factory.EnumWarpAdapter::<IDXGIAdapter>()
        };
        let Ok(raw) = raw else { return };
        // SAFETY: EnumWarpAdapter returns a DXGI adapter, which implements
        // IDXGIAdapter1; `raw` owns that interface.
        let desc = unsafe { raw.cast::<IDXGIAdapter1>().unwrap().GetDesc1().unwrap() };
        eprintln!(
            "EnumWarpAdapter: vendor={:#x}, flags={:#x}",
            desc.VendorId, desc.Flags
        );
        assert_eq!(
            warp_adapter().is_ok(),
            desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
        );
    }

    #[test]
    fn adapter_ids_returns_vendor_and_device() {
        let Some(gpu) =
            crate::testgpu::device("host::tests::adapter_ids_returns_vendor_and_device")
        else {
            return;
        };
        let (vendor, device) = gpu.host().adapter_ids().expect("adapter IDs");
        assert!(vendor != 0 || device != 0);
    }

    #[test]
    fn the_frame_line_counts_title_submissions() {
        let ms = std::time::Duration::from_millis;
        let mut s = Submissions::new();
        let none = PerFrame {
            submissions: 0.0,
            lists: 0.0,
            longest_ms: 0.0,
            longest_gpu_ms: 0.0,
            longest_lists: 0,
        };
        assert_eq!(s.per_frame(2.0), none, "no submission: no division by zero");
        for (lists, gpu, total) in [(3, 10, 15), (5, 80, 120), (4, 30, 40), (0, 1, 2)] {
            s.add(
                lists,
                Ran {
                    gpu: ms(gpu),
                    total: ms(total),
                },
            );
        }
        // Four submissions over two frames, twelve lists; the longest is
        // the one longest in all.
        let want = PerFrame {
            submissions: 2.0,
            lists: 3.0,
            longest_ms: 120.0,
            longest_gpu_ms: 80.0,
            longest_lists: 5,
        };
        assert_eq!(s.per_frame(2.0), want);
        // Of a 500 ms frame with 300 ms in submissions, 200 outside them;
        // never below zero.
        assert_eq!(outside(500.0, 300.0), 200.0);
        assert_eq!(outside(100.0, 130.0), 0.0);
    }

    /// A title submission run in parts (a list named again) counts once on
    /// the frame-rate line, by its last part, with every part's lists and
    /// times; a whole one counts as itself.
    #[test]
    fn the_frame_line_counts_a_split_submission_once() {
        let ms = std::time::Duration::from_millis;
        let ran = |gpu, total| Ran {
            gpu: ms(gpu),
            total: ms(total),
        };
        let mut s = Submissions::new();
        let parts = Parts::default();
        // [a, b, a, c, a] runs as [a, b], [a, c], [a].
        assert_eq!(Title::Part(parts.clone()).counted(2, ran(10, 15)), None);
        assert_eq!(Title::Part(parts.clone()).counted(2, ran(20, 30)), None);
        let (lists, all) = Title::Last(Some(parts)).counted(1, ran(5, 8)).unwrap();
        s.add(lists, all);
        let whole = Title::Last(None).counted(3, ran(1, 2)).unwrap();
        s.add(whole.0, whole.1);
        let want = PerFrame {
            submissions: 2.0,
            lists: 4.0,
            longest_ms: 53.0,
            longest_gpu_ms: 35.0,
            longest_lists: 5,
        };
        assert_eq!(s.per_frame(1.0), want);
    }

    /// The GPU's time and the copying back the title's threads did not
    /// wait for: all of it waited for (synchronous submission) is no
    /// overlap; never below zero (the GPU's time is measured roughly).
    #[test]
    fn the_frame_line_shows_the_overlap() {
        assert_eq!(overlapped(200.0, 100.0, 300.0), 0.0);
        assert_eq!(overlapped(200.0, 100.0, 40.0), 260.0);
        assert_eq!(overlapped(10.0, 0.0, 15.0), 0.0);
    }

    /// Waits add up by the place that asked for them, longest first, per
    /// frame; taking them starts the counts again.
    #[test]
    fn waits_are_counted_by_where_they_were_asked() {
        #[track_caller]
        fn here() -> &'static std::panic::Location<'static> {
            std::panic::Location::caller()
        }
        let a = here();
        let b = here();
        let mut waits = Waits::default();
        waits.add(a, 1_000);
        waits.add(b, 6_000);
        waits.add(a, 3_000);
        let lines = waits.take(2.0, 8);
        let (la, lb) = (a.line(), b.line());
        assert_eq!(
            lines,
            [
                format!("host.rs:{lb} 3.0 ms/frame (0.5/frame)"),
                format!("host.rs:{la} 2.0 ms/frame (1.0/frame)"),
            ]
        );
        assert!(waits.take(1.0, 8).is_empty());
    }

    #[test]
    fn xtp_sync_submit_is_on_unless_empty_or_zero() {
        use std::ffi::OsStr;
        assert!(!sync_asked(None));
        assert!(!sync_asked(Some(OsStr::new(""))));
        assert!(!sync_asked(Some(OsStr::new("0"))));
        assert!(sync_asked(Some(OsStr::new("1"))));
        assert!(sync_asked(Some(OsStr::new("yes"))));
    }

    /// GPU copy destinations go back to title memory merged per host copy:
    /// overlapping and touching ranges as one, gaps left alone.
    #[test]
    fn copy_destinations_are_merged_per_host_copy() {
        let ranges = vec![
            (2u8, 100, 50),
            (1, 0, 10),
            (2, 150, 10),
            (1, 5, 3),
            (2, 0, 20),
            (1, 30, 5),
            (2, 140, 5),
        ];
        assert_eq!(
            merge_ranges(ranges, |k| usize::from(*k)),
            [(1, 0, 10), (1, 30, 5), (2, 0, 20), (2, 100, 60)]
        );
    }
}

/// A host command list and its allocator, always used together.
pub(crate) struct HostList {
    pub(crate) rt_scratch: Mutex<Option<ID3D12Resource>>,
    pub(crate) query_resolves: crate::query::Resolves,
    /// Swapped for a fresh one when the list is reset while it runs.
    allocator: Mutex<ID3D12CommandAllocator>,
    pub(crate) list: ID3D12GraphicsCommandList,
    pub(crate) bounds_list: Option<ID3D12GraphicsCommandList1>,
    /// Work for submission time, in recorded order.
    deferred: Mutex<Deferred>,
    /// Upload memory for data the list's draws make (expanded vertices),
    /// reused after a reset.
    scratch: Mutex<Scratch>,
    /// Host texture states the list assumes, by host copy: at its start and
    /// at its end. Lists are recorded on many threads and run in submission
    /// order, so the transitions from a texture's real state are made at
    /// submission, in `prelude`. Holding the copies keeps every texture the
    /// list names alive until it is reset, even if the title releases the
    /// resource first.
    states: Mutex<States>,
    /// The list that runs those transitions just before this one.
    prelude: Mutex<Prelude>,
    /// What the list does with depth textures that have a colour twin, by
    /// twin (`twins`).
    twins: Mutex<twins::TwinUses>,
    /// Whether `twins` may have something for the next draw or dispatch
    /// (`HostList::planes_now`, `HostList::twins_written`): read without
    /// its lock, so that a list with nothing due takes none per bind.
    twins_due: std::sync::atomic::AtomicBool,
    /// The pipelines the list binds, for reports of slow submissions:
    /// shared with its submissions ([`Recorded`]), a reset starts a new set.
    pipelines: Mutex<std::sync::Arc<std::collections::BTreeSet<String>>>,
    writer: AtomicU64,
    /// The work its indirect executions may still do (`indirect`): a
    /// shader-writable word, zeroed when the list is submitted.
    budget: Mutex<Option<(ID3D12Resource, usize)>>,
    pub(crate) decode_uses: Mutex<crate::resource::decode::Uses>,
    pub(crate) decode_pipeline: std::sync::atomic::AtomicBool,
    pub(crate) clamps: Mutex<crate::indirect::Runs>,
    /// The host fence value of its last submission (0: none): it is not
    /// dropped before that one has finished.
    last: AtomicU64,
    /// Up to which of its submissions the list's memory was handed to the
    /// timeline by a reset: its recording since uses none of theirs.
    fresh_after: AtomicU64,
    /// Whether its last submission was ours ([`Host::execute`]), not the
    /// title's: a reset then waits for it to finish.
    ours: std::sync::atomic::AtomicBool,
    /// The texts of the markers `note_op` set, in order (for device-removed
    /// data without marker texts), with the host GPU range a copy reads:
    /// shared like `pipelines`.
    #[cfg(feature = "diagnostics")]
    notes: Mutex<std::sync::Arc<Vec<Note>>>,
    /// With `XTP_GPU_TIMING`: a GPU timestamp before each noted command
    /// and at the end, and where they are read back.
    #[cfg(feature = "diagnostics")]
    stamps: OnceLock<Option<(ID3D12QueryHeap, ID3D12Resource)>>,
}

impl Drop for HostList {
    /// Its allocators and upload memory may not go while the GPU runs it,
    /// and a submission in flight still names it.
    fn drop(&mut self) {
        self.settle();
        crate::timestamps::forget(self);
    }
}

/// A closed prelude list and its allocator.
type Prelude = (ID3D12CommandAllocator, ID3D12GraphicsCommandList);

/// What a submission keeps of a list's recording for its reports (slow
/// submissions, device removal): a reset while it runs starts the list's
/// next recording afresh without taking them from it.
#[derive(Clone, Default)]
struct Recorded {
    rt_sizes: std::sync::Arc<Vec<crate::rt::compaction::CompactedSize>>,
    #[cfg(feature = "diagnostics")]
    rt_samples: std::sync::Arc<Vec<crate::rt::probe::Sample>>,
    #[cfg(feature = "diagnostics")]
    dispatch_samples: std::sync::Arc<Vec<crate::indirect::Sample>>,
    pipelines: std::sync::Arc<std::collections::BTreeSet<String>>,
    #[cfg(feature = "diagnostics")]
    notes: std::sync::Arc<Vec<Note>>,
    /// What its draws read and its dispatches wrote (`XTP_MESH_STATS`).
    #[cfg(feature = "diagnostics")]
    meshes: std::sync::Arc<Vec<crate::meshstat::Probe>>,
}

/// Timestamps per list (diagnostics); commands past them are not timed.
#[cfg(feature = "diagnostics")]
const STAMPS: u32 = 4096;

/// A marker's text and the host range its command reads (diagnostics).
#[cfg(feature = "diagnostics")]
type Note = (String, Option<(u64, u64)>);

type States =
    std::collections::HashMap<usize, (u32, u32, std::sync::Arc<crate::resource::HostTexture>)>;

/// Whether device-removed data is on (`XTP_D3D12_DRED`, diagnostics
/// builds).
pub(crate) fn dred_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        // `XTP_D3D12_DRED=0` (or empty) is off: not set alone.
        if !crate::dred::asked(std::env::var_os("XTP_D3D12_DRED").as_deref()) {
            return false;
        }
        cfg!(feature = "diagnostics") && std::env::var_os("XTP_D3D12_DRED").is_some()
    })
}

/// Whether each list of a submission runs and is timed on its own, slow
/// ones logged with their commands (`XTP_GPU_TIMING`, diagnostics
/// builds; the commands are noted with `XTP_D3D12_DRED` on).
#[cfg(feature = "diagnostics")]
fn timing() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("XTP_GPU_TIMING").is_some())
}

/// Names a host object for the device-removed report, which names the
/// allocations a page fault hit (only with device-removed data on).
pub(crate) fn name(object: &impl Interface, text: impl FnOnce() -> String) {
    if dred_on()
        && let Ok(object) = object.cast::<ID3D12Object>()
    {
        // SAFETY: a plain call on a live object.
        let _ = unsafe { object.SetName(&windows::core::HSTRING::from(text())) };
    }
}

/// The name a host list gets for the device-removed report ([`name`]).
fn list_name(list: &ID3D12GraphicsCommandList) -> String {
    format!("xtp host list {:#x}", list.as_raw() as usize)
}

/// Upload chunks handed out front to back: the chunks, the one in use and
/// the bytes used in it.
#[derive(Default)]
struct Scratch {
    chunks: Vec<(ID3D12Resource, usize, u64)>,
    current: usize,
    used: u64,
}

// SAFETY: the chunk pointers are persistent mappings of upload buffers.
unsafe impl Send for Scratch {}

const SCRATCH_CHUNK: u64 = 4 << 20;

/// What a list reset while it ran leaves to its running submission.
struct Retired {
    allocator: ID3D12CommandAllocator,
    prelude: Prelude,
    deferred: Deferred,
    scratch: Scratch,
    states: States,
    budget: Option<(ID3D12Resource, usize)>,
}

// SAFETY: D3D12 objects are free-threaded; the scratch and budget pointers
// are persistent mappings of their buffers, which go with them.
unsafe impl Send for Retired {}

/// Allocators, preludes and scratch chunks whose submissions finished,
/// for lists reset while they run (dozens per world frame).
static SPARE_ALLOCATORS: Mutex<Vec<ID3D12CommandAllocator>> = Mutex::new(Vec::new());
static SPARE_PRELUDES: Mutex<Vec<Prelude>> = Mutex::new(Vec::new());
static SPARE_SCRATCH: Mutex<Vec<(ID3D12Resource, usize, u64)>> = Mutex::new(Vec::new());
/// Spares kept of each kind at most.
const SPARES: usize = 256;

/// Recordings of lists ended (by a reset) since the last memory line, by the
/// scratch memory they used: up to 256 KiB, 1 MiB, 4 MiB, past 4 MiB; then
/// the chunks those lists held and the bytes they used.
static SCRATCH_USE: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
/// Scratch allocations larger than a chunk since the last memory line, each
/// in a buffer of its own: how many, and their bytes.
static SCRATCH_OWN: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Counts the scratch memory of a recording that ends ([`SCRATCH_USE`]).
fn note_scratch_use(s: &Scratch) {
    let used = if s.chunks.is_empty() {
        0
    } else {
        s.current as u64 * SCRATCH_CHUNK + s.used
    };
    let class = match used {
        0..=0x4_0000 => 0,
        0x4_0001..=0x10_0000 => 1,
        0x10_0001..=0x40_0000 => 2,
        _ => 3,
    };
    SCRATCH_USE[class].fetch_add(1, Ordering::Relaxed);
    SCRATCH_USE[4].fetch_add(s.chunks.len() as u64, Ordering::Relaxed);
    SCRATCH_USE[5].fetch_add(used, Ordering::Relaxed);
}

/// [`SCRATCH_USE`] since the last call, for the memory line.
fn scratch_use() -> String {
    let [small, mid, chunk, big, held, used] =
        SCRATCH_USE.each_ref().map(|n| n.swap(0, Ordering::Relaxed));
    let [own, own_bytes] = SCRATCH_OWN.each_ref().map(|n| n.swap(0, Ordering::Relaxed));
    let ended = small + mid + chunk + big;
    format!(
        "{ended} list recordings ended ({small} used up to 256 KiB of scratch, {mid} up to 1 MiB, \
         {chunk} up to 4 MiB, {big} more), holding {held} chunks for {} MiB used; \
         {own} allocations larger than a chunk in buffers of their own ({} KiB)",
        used >> 20,
        own_bytes >> 10
    )
}

/// The spare scratch chunks held now (for the memory line).
fn spare_scratch_chunks() -> usize {
    pool(&SPARE_SCRATCH).len()
}

fn pool<T>(spares: &'static Mutex<Vec<T>>) -> std::sync::MutexGuard<'static, Vec<T>> {
    spares.lock().unwrap_or_else(|e| e.into_inner())
}

impl Retired {
    /// Once the submission that used them has finished.
    fn retire(self) {
        // SAFETY: nothing running uses the allocators any more.
        unsafe {
            if self.allocator.Reset().is_ok() && pool(&SPARE_ALLOCATORS).len() < SPARES {
                pool(&SPARE_ALLOCATORS).push(self.allocator);
            }
            if self.prelude.0.Reset().is_ok() && pool(&SPARE_PRELUDES).len() < SPARES {
                pool(&SPARE_PRELUDES).push(self.prelude);
            }
        }
        let mut spare = pool(&SPARE_SCRATCH);
        let room = SPARES.saturating_sub(spare.len());
        spare.extend(self.scratch.chunks.into_iter().take(room));
        drop((self.deferred, self.states, self.budget));
    }
}

/// One piece of submission-time work, in recorded order.
enum Step {
    /// A copy into title memory (destination, source, bytes): the title's
    /// buffer copies where neither side has a host copy (the others run on
    /// the GPU), which later refreshes must see.
    Copy(usize, usize, usize),
    /// A copy of title memory into the list's own upload memory.
    Fill(usize, usize, usize),
    /// Brings a range (address, bytes) of a buffer's host copy up to date:
    /// once per submission, whichever lists ask, until a `Copy` or `Run`
    /// intervenes.
    Refresh(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Work that may write title memory: the range it writes, the range it
    /// reads (address, bytes).
    Run(Box<dyn Fn() + Send>, (usize, usize), (usize, usize)),
    /// Work that writes host memory only, and the title memory it reads.
    Host(Box<dyn Fn() + Send>, (usize, usize)),
    /// Shaders write a range (address, bytes) of a buffer's host copy.
    GpuWrite(std::sync::Arc<crate::shadow::Shadow>, usize, usize, u64),
    /// A buffer copy the list records on the GPU writes a range (address,
    /// bytes) of a buffer's host copy: all of it goes back to title memory
    /// once the list has run.
    GpuCopy(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// A memory write the list records on the GPU (bytes), for the
    /// frame-rate line.
    GpuWord(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Brings a texture's staging buffer up to date: once per submission,
    /// whichever lists sample it, until a `Copy` or `Run` intervenes.
    Texture(std::sync::Arc<crate::texture::Staging>),
    /// Reads a CPU-written linear movie plane into a per-list upload buffer.
    Plane((usize, usize), Box<dyn Fn() + Send>),
    /// Shaders of the list pick buffer views of a view heap's title slots
    /// when they run: the runs those views read are brought up to date, and
    /// (shaders writing through them) those they write declared written,
    /// once per submission whichever lists ask, until a `Copy` or `Run`
    /// intervenes (`crate::descriptor::host::PickedStep`).
    Picked(std::sync::Arc<crate::descriptor::host::PickedStep>),
}

/// Title memory moved by submissions since the last frame-rate line, as
/// (count, bytes): buffer copies made on the GPU between host copies, memory
/// writes made on the GPU inside lists, buffer copies made on the CPU.
#[derive(Clone, Copy)]
pub(crate) enum Moved {
    GpuCopy,
    ListWrite,
    CpuCopy,
}

static MOVED: [[std::sync::atomic::AtomicU64; 2]; 3] =
    [const { [const { std::sync::atomic::AtomicU64::new(0) }; 2] }; 3];

/// Counts title memory a submission moves (see [`Moved`]).
pub(crate) fn moved(what: Moved, bytes: usize) {
    let [count, total] = &MOVED[what as usize];
    count.fetch_add(1, Ordering::Relaxed);
    total.fetch_add(bytes as u64, Ordering::Relaxed);
}

/// Dynamic descriptor slots a recording leases (title heap, first,
/// count): given back when the recording goes, which for a list reset
/// while it runs is once that submission has finished.
struct DynamicLease(usize, u32, u32);

impl Drop for DynamicLease {
    fn drop(&mut self) {
        crate::descriptor::release_dynamic(self.0, self.1, self.2);
    }
}

/// Work a list runs as it closes ([`HostList::at_close`]).
type Closing = Box<dyn FnOnce(&HostList) + Send>;

/// Work a list runs at each of its submissions ([`HostList::at_submit`]).
type Submitting = std::sync::Arc<dyn Fn(&HostList) + Send + Sync>;

/// Latest write at each address, represented by the starts of runs with the
/// same serial. A zero serial is a gap. An overwrite preserves both tails.
#[derive(Default)]
struct WriteRanges(std::collections::BTreeMap<usize, u64>);

impl WriteRanges {
    fn at(&self, address: usize) -> u64 {
        self.0.range(..=address).next_back().map_or(0, |(_, &v)| v)
    }

    fn record(&mut self, from: usize, to: usize, serial: u64) {
        let tail = self.at(to);
        while let Some((&at, _)) = self.0.range(from..to).next() {
            self.0.remove(&at);
        }
        self.0.insert(from, serial);
        self.0.insert(to, tail);
    }

    fn latest(&self, from: usize, to: usize) -> u64 {
        self.0
            .range(from..to)
            .map(|(_, &v)| v)
            .max()
            .unwrap_or(0)
            .max(self.at(from))
    }
}

/// Per recording. Key zero covers submission-time writes to title memory.
/// GPU writes name the shadow they actually target.
#[derive(Default)]
struct BufferWrites {
    serial: u64,
    ranges: std::collections::HashMap<usize, WriteRanges>,
    unknown: u64,
    binding: bool,
    indexed_output: bool,
    shader_outputs: Vec<(usize, usize, usize)>,
}

impl BufferWrites {
    fn record(&mut self, copy: usize, address: usize, len: usize) {
        if len != 0 {
            self.serial += 1;
            self.ranges.entry(copy).or_default().record(
                address,
                address.saturating_add(len),
                self.serial,
            );
        }
    }

    fn latest(&self, copy: usize, address: usize, end: usize) -> u64 {
        self.ranges.get(&copy).map_or(0, |r| r.latest(address, end))
    }
}

/// What a list does on the CPU when it is submitted: copies between host
/// memory regions (title buffers, upload buffers and buffer copies), and
/// the upload buffers its GPU commands read, kept until the list is reset.
#[derive(Default)]
struct Deferred {
    rt_sizes: std::sync::Arc<Vec<crate::rt::compaction::CompactedSize>>,
    #[cfg(feature = "diagnostics")]
    rt_samples: std::sync::Arc<Vec<crate::rt::probe::Sample>>,
    #[cfg(feature = "diagnostics")]
    dispatch_samples: std::sync::Arc<Vec<crate::indirect::Sample>>,
    /// What the list's draws read and its dispatches wrote, in recorded
    /// order (`XTP_MESH_STATS`).
    #[cfg(feature = "diagnostics")]
    meshes: std::sync::Arc<Vec<crate::meshstat::Probe>>,
    /// Destination, source, bytes.
    copies: Vec<Step>,
    /// Fill steps saved at record time, counted each time this list runs.
    merged_fills: u64,
    uploads: Vec<ID3D12Resource>,
    /// Host objects the list's commands name (pipelines, root signatures,
    /// descriptor heaps): the title may release its objects before it
    /// submits the list.
    kept: Vec<windows::core::IUnknown>,
    rt_recording: Option<std::sync::Arc<()>>,
    rt_size_page: crate::rt::compaction::SizePage,
    rt_results: Vec<std::sync::Arc<crate::rt::ResultBuffer>>,
    /// Dynamic descriptor slots leased by this recording.
    dynamic: Vec<DynamicLease>,
    /// Buffer ranges refreshed since the last title copy (host copy,
    /// address, bytes): refreshing them again would find the same bytes.
    refreshed: std::collections::HashSet<(usize, usize, usize)>,
    /// Times `refreshed` was emptied ([`HostList::refresh_epoch`]).
    refresh_epoch: u64,
    /// Work the list runs as it closes ([`HostList::at_close`]).
    closing: Vec<Closing>,
    /// Work the list runs at each submission ([`HostList::at_submit`]).
    submitting: Vec<Submitting>,
    /// Textures whose contents the list brings up to date.
    textures: std::collections::HashSet<usize>,
    /// Title textures of wanted kinds ([`crate::texback`]) written by shaders
    /// in this list, copied to readback buffers when the list closes.
    back_wanted: Vec<crate::texback::Target>,
    /// Readback copies recorded at close, tiled into title memory once the
    /// submission completes.
    texture_backs: Vec<std::sync::Arc<crate::texback::Back>>,
    /// Bottom-of-pipe writes, done once the list has run: address, what,
    /// bytes.
    writes: Vec<(usize, crate::queue::Written, usize)>,
    /// The memory values the list waits for, in recorded order: its queue
    /// submits it once they are met (`queue::wait32`, `queue::wait64`).
    value_waits: Vec<crate::queue::ValueWait>,
    /// Whether a step zeroes the indirect-work budget.
    budget_zeroed: bool,
    indirect_writers: crate::indirect::Writers,
    buffer_writes: BufferWrites,
    /// Shadow and title range read by a recorded GPU copy into CBV memory.
    cbv_reads: std::collections::HashSet<(usize, usize, usize)>,
}

/// A list a submission in flight names.
#[derive(Clone, Copy)]
struct ListRef(*const HostList);

// SAFETY: host lists are used from any thread (their state is behind
// locks), and one is not dropped while a submission naming it is in flight
// (`HostList::last`); a reset hands the memory such a submission uses to
// the timeline instead of reusing it.
unsafe impl Send for ListRef {}

impl ListRef {
    fn get<'a>(self) -> &'a HostList {
        // SAFETY: as above, the list outlives the submission.
        unsafe { &*self.0 }
    }
}

/// What is left of a submission once the GPU has run it.
struct Pending {
    lists: Vec<ListRef>,
    /// Each list's recording as submitted.
    recorded: Vec<Recorded>,
    /// Its number in the host copies (`shadow::submission`).
    submission: u64,
    /// The destinations of its GPU buffer copies (host copy, address,
    /// bytes), merged per host copy.
    copied: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    /// The ranges its shaders write (host copy, address, bytes), merged
    /// per host copy.
    back: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    /// Compute-written texture readbacks to tile into title memory on completion.
    texture_backs: Vec<std::sync::Arc<crate::texback::Back>>,
    /// Its bottom-of-pipe writes, in list order: address, what, bytes.
    writes: Vec<(usize, crate::queue::Written, usize)>,
    submitted: std::time::Instant,
    /// The time it took to prepare.
    preparing: std::time::Duration,
    /// For a title submission or a part of one: how the frame-rate line
    /// counts it.
    title: Option<Title>,
}

impl Pending {
    /// A submission's, after its steps ran, with the title memory its GPU
    /// work writes (written back, written at the bottom of the pipe) and
    /// reads (host copies and textures brought up to date for it).
    fn of(runs: &[&HostList], submission: u64) -> (Pending, Touches) {
        type Ranges = Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>;
        let (mut back, mut copied): (Ranges, Ranges) = (Vec::new(), Vec::new());
        let mut texture_backs = Vec::new();
        let (mut writes, mut touches) = (Vec::new(), Touches::default());
        // One list's lock at a time: a submission may name a list twice.
        for list in runs {
            let deferred = list.deferred();
            for &(copy, from, to) in &deferred.cbv_reads {
                touches.cbv_reads.entry(copy).or_default().push((from, to));
            }
            for step in &deferred.copies {
                match step {
                    Step::GpuWrite(shadow, address, len, _) => {
                        back.push((shadow.clone(), *address, *len));
                    }
                    Step::GpuCopy(shadow, address, len) => {
                        copied.push((shadow.clone(), *address, *len));
                    }
                    Step::Refresh(_, address, len) => {
                        touches.reads.push((*address, address + len));
                    }
                    Step::Texture(staging) => {
                        touches.reads.push(staging.range());
                        touches
                            .staging
                            .insert(std::sync::Arc::as_ptr(staging) as usize);
                    }
                    Step::Plane((address, len), _) => {
                        touches.reads.push((*address, address.saturating_add(*len)));
                    }
                    // What the submission was given of the heap's listing,
                    // taken by its first step: the runs refreshed, and the
                    // runs its shaders write, which go back as `GpuWrite`s.
                    Step::Picked(step) => {
                        let (reads, writes) = step.touches(submission);
                        touches.reads.extend(reads);
                        back.extend(writes.into_iter().map(|(shadow, from, to)| {
                            (shadow, from as usize, (to - from) as usize)
                        }));
                    }
                    _ => {}
                }
            }
            for back in &deferred.texture_backs {
                texture_backs.push(back.clone());
                touches.labels.push(back.range());
            }
            for &(at, what, bytes) in &deferred.writes {
                writes.push((at, what, bytes));
                touches.labels.push((at, at + bytes));
            }
        }
        // GPU copy destinations are exactly what the GPU wrote, so all of
        // each goes back, merged per host copy (a world frame makes
        // thousands of small copies, often side by side: each write opens
        // the guard of its title memory).
        let copied = merge_ranges(copied, |s| std::sync::Arc::as_ptr(s) as usize);
        // Shader-written ranges merged per host copy too, those that overlap
        // or touch into one: each byte is compared and goes back once. The
        // world's passes declare the same big ranges dozens of times, and
        // overlapping views of its 174 MB heap; only identical ranges were
        // merged before, and a second range over bytes the first had put
        // back wrote older title bytes over newer ones.
        let back = merge_ranges(back, |s| std::sync::Arc::as_ptr(s) as usize);
        // What the GPU writes through host copies, each range once (later
        // steps look them up one by one).
        let through = |ranges: &Ranges| -> Vec<(usize, usize, usize)> {
            ranges
                .iter()
                .map(|(s, a, l)| (std::sync::Arc::as_ptr(s) as usize, *a, a + l))
                .collect()
        };
        (touches.shaders, touches.copies) = (through(&back), through(&copied));
        let pending = Pending {
            lists: runs.iter().map(|&l| ListRef(l)).collect(),
            recorded: runs.iter().map(|l| l.recorded()).collect(),
            submission,
            copied,
            back,
            texture_backs,
            writes,
            submitted: std::time::Instant::now(),
            preparing: std::time::Duration::ZERO,
            title: None,
        };
        (pending, touches)
    }
}

/// A thread's event for waiting on the host fence.
struct Event(HANDLE);

impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: our own handle, no longer waited on.
        let _ = unsafe { windows::Win32::Foundation::CloseHandle(self.0) };
    }
}

thread_local! {
    /// Null if it could not be made: the fence wait then blocks by itself.
    // SAFETY: plain event creation.
    static EVENT: Event = Event(unsafe { CreateEventW(None, false, false, None) }.unwrap_or_default());
    /// Whether this is the completion thread, whose waits are not the
    /// title's.
    static FINISHER: Cell<bool> = const { Cell::new(false) };
    /// How deep this thread is in [`Host::waiting`], and the microseconds
    /// it waited so far.
    static WAITING: Cell<(u32, u64)> = const { Cell::new((0, 0)) };
}

/// Microseconds this thread has waited for submissions so far.
fn waited_here() -> u64 {
    WAITING.get().1
}

impl Host {
    /// Whether the host runs mesh shaders at shader model 6.6 (translated
    /// mesh shaders need both); logged once.
    pub(crate) fn mesh_shaders(&self) -> bool {
        static MESH: OnceLock<bool> = OnceLock::new();
        *MESH.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
            let mut model = D3D12_FEATURE_DATA_SHADER_MODEL {
                HighestShaderModel: D3D_SHADER_MODEL_6_6,
            };
            // SAFETY: each query writes its own structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS7,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
                    && self
                        .device
                        .CheckFeatureSupport(
                            D3D12_FEATURE_SHADER_MODEL,
                            (&raw mut model).cast(),
                            size_of_val(&model) as u32,
                        )
                        .is_ok()
            };
            let mesh = ok
                && options.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED
                && model.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_6.0;
            tracing::info!(mesh, "host mesh shader support");
            mesh
        })
    }

    /// The adapter's PCI vendor and device IDs; `None` if the system does
    /// not name the adapter.
    // Ported from PastiiJay/klbf-lab commit f73233c8 (author aidzz / @BrokeAsh, GPL-3.0-or-later).
    pub(crate) fn adapter_ids(&self) -> Option<(u32, u32)> {
        use windows::Win32::Graphics::Dxgi::{
            CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIFactory4,
        };
        // SAFETY: plain DXGI queries for this device's adapter.
        let desc = unsafe {
            CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
                .ok()?
                .EnumAdapterByLuid::<IDXGIAdapter1>(self.device.GetAdapterLuid())
                .ok()?
                .GetDesc1()
                .ok()?
        };
        Some((desc.VendorId, desc.DeviceId))
    }

    /// A new list, open for recording.
    pub(crate) fn new_list(&self) -> windows::core::Result<HostList> {
        // SAFETY: plain object creation.
        unsafe {
            let allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None,
            )?;
            let prelude_allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let prelude: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &prelude_allocator,
                None,
            )?;
            prelude.Close()?;
            // Named for the device-removed report (with its data on), which
            // finds a list by name where DRED gives no pointer to it.
            name(&list, || list_name(&list));
            name(&prelude, || {
                format!("xtp host prelude {:#x}", prelude.as_raw() as usize)
            });
            Ok(HostList {
                rt_scratch: Mutex::default(),
                allocator: Mutex::new(allocator),
                bounds_list: if self.depth_bounds {
                    Some(list.cast()?)
                } else {
                    None
                },
                list,
                deferred: Mutex::default(),
                scratch: Mutex::default(),
                query_resolves: Default::default(),
                states: Mutex::default(),
                prelude: Mutex::new((prelude_allocator, prelude)),
                twins: Mutex::default(),
                twins_due: Default::default(),
                pipelines: Mutex::default(),
                writer: AtomicU64::new(0),
                budget: Mutex::default(),
                decode_uses: Mutex::default(),
                decode_pipeline: Default::default(),
                clamps: Mutex::default(),
                last: AtomicU64::new(0),
                fresh_after: AtomicU64::new(0),
                ours: std::sync::atomic::AtomicBool::new(false),
                #[cfg(feature = "diagnostics")]
                notes: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                stamps: OnceLock::new(),
            })
        }
    }

    /// A command allocator nothing runs from: a spare, or a new one.
    fn fresh_allocator(&self) -> Option<ID3D12CommandAllocator> {
        let spare = pool(&SPARE_ALLOCATORS).pop();
        spare.or_else(|| {
            // SAFETY: plain object creation.
            let made = unsafe {
                self.device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            };
            made.ok()
        })
    }

    /// A closed prelude list and its allocator, nothing running from them.
    fn fresh_prelude(&self) -> Option<Prelude> {
        let spare = pool(&SPARE_PRELUDES).pop();
        if spare.is_some() {
            return spare;
        }
        let allocator = self.fresh_allocator()?;
        // SAFETY: plain object creation.
        unsafe {
            let list: ID3D12GraphicsCommandList = self
                .device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                .ok()?;
            list.Close().ok()?;
            Some((allocator, list))
        }
    }

    /// Runs closed lists of ours (not counted on the frame-rate line). Their
    /// submission joins the timeline, finished once the GPU has run it.
    pub(crate) fn execute(&self, runs: &[&HostList]) {
        self.run(runs, false);
    }

    /// Runs a title submission (the lists of one `ExecuteCommandLists`),
    /// counted for the frame-rate line once it has finished.
    pub(crate) fn submit(&self, runs: &[&HostList]) {
        crate::resource::reap_placed_views();
        #[cfg(feature = "diagnostics")]
        crate::passdump::submitted(runs);
        #[cfg(feature = "diagnostics")]
        crate::frameprobe::submitted(runs);
        // Light the lists read through views in another format, decoded
        // first (`resource::alias_copy`).
        crate::resource::run_decodes(self);
        self.run(runs, true);
    }

    fn run(&self, runs: &[&HostList], title: bool) {
        self.run_as(runs, title, sync_submit());
    }

    /// As [`Host::run`], waited for and finished before returning with
    /// `sync`.
    fn run_as(&self, runs: &[&HostList], title: bool, sync: bool) {
        crate::pipeline::trim_library();
        #[cfg(feature = "diagnostics")]
        SPLITS[0].fetch_add(1, Ordering::Relaxed);
        if !sync {
            self.finisher.call_once(|| {
                let spawned = std::thread::Builder::new()
                    .name("xtp-gpu-finisher".into())
                    .spawn(|| made().into_iter().for_each(Host::finish_loop));
                if let Err(e) = spawned {
                    tracing::warn!(error = %e, "no completion thread; submissions finish when needed");
                }
            });
        }
        // A title may name a list more than once in a submission; D3D12 runs
        // a list again only once its previous execution has finished, in one
        // `ExecuteCommandLists` or not (debug layer message 553), and each
        // run of a list rewrites its prelude, upload memory and budget word.
        // Such a submission goes in parts, each up to the next list named
        // again, run once the part before has finished: the GPU runs the
        // same lists in the same order, and each run finds what the run
        // before wrote. The wait releases the host submission lock, but a
        // title queue retains its line lock and a held submission retains
        // release_held's global lock, delaying other queues. Counted once, by
        // its last part, with all its parts' lists and their times
        // preparing, on the GPU and copying back; the waits between them
        // count as waiting, as a caller's would.
        let mut rest = runs;
        let mut parts: Option<Parts> = None;
        loop {
            let (part, next) = rest.split_at(first_repeat(rest));
            let counted = title.then(|| match next {
                [] => Title::Last(parts.take()),
                _ => Title::Part(parts.get_or_insert_default().clone()),
            });
            self.run_part(part, counted, sync);
            let Some(again) = next.first() else {
                return;
            };
            #[cfg(feature = "diagnostics")]
            let waiting = {
                SPLITS[1].fetch_add(1, Ordering::Relaxed);
                std::time::Instant::now()
            };
            self.settle(again.last.load(Ordering::Acquire));
            #[cfg(feature = "diagnostics")]
            SPLITS[2].fetch_add(waiting.elapsed().as_micros() as u64, Ordering::Relaxed);
            rest = next;
        }
    }

    /// Runs `runs`, no list among them twice: one submission, a title's
    /// (or a part of one) with `title`, counted for the frame-rate line as
    /// it says.
    fn run_part(&self, runs: &[&HostList], title: Option<Title>, sync: bool) {
        let is_title = title.is_some();
        // One submission at a time from preparation to submission (to
        // copy-back with `XTP_SYNC_SUBMIT`): what a step waits for is
        // decided against the submissions before it.
        let mut submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        let started = std::time::Instant::now();
        let waited_before = waited_here();
        let recording = preparation::Recording::start(started, waited_before);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::Submissions, 1);
        // What has run is finished first. A list whose last submission is
        // still in flight with the memory it records into now (submitted
        // again without a reset) is waited for: its upload memory, budget
        // word and prelude are rewritten below.
        self.waiting(|| self.finish_reached());
        for list in runs {
            if list.running() {
                self.settle(list.last.load(Ordering::Acquire));
            }
        }
        // Submissions in flight at most (`frames::cap`).
        if self.timeline.len() >= frames::cap()
            && let Some((oldest, _)) = self.timeline.span()
        {
            self.settle(oldest);
        }
        let mut lists: Vec<Option<ID3D12CommandList>> = Vec::with_capacity(2 * runs.len());
        // Where each run's lists start in `lists`.
        let mut firsts = Vec::with_capacity(runs.len());
        let mut batch = Batch {
            host: Some(self),
            submission: crate::shadow::submission(),
            ..Batch::default()
        };
        // A full sweep also rewrites inputs and WriteBufferImmediate words.
        // Their readers and deferred title writes must finish before the
        // sweep, and this submission must take its snapshots afterwards.
        if crate::shadow::refresh_due(batch.submission) {
            self.drain();
            let _copy = Phase::Copies.enter();
            crate::shadow::refresh_everything(batch.submission);
        }
        if batch.submission.is_multiple_of(32) {
            crate::readmap::save_copy_back_map();
        }
        // In-flight lists may still read the shader-visible slots we are
        // about to rewrite. The submission lock excludes new executions
        // while the mirror runs. The disabled path does not drain.
        let mut heaps_synced =
            is_title.then(|| crate::descriptor::begin_sync_heaps(|| self.drain()));
        for list in runs {
            list.submitting(|| {
                if let Some(sync) = heaps_synced.take() {
                    sync.finish();
                }
            });
            list.run_deferred(&mut batch);
            firsts.push(lists.len());
            // Copies of textures the title wrote after the list read them,
            // found by its steps (`texture::late_copies`): before its
            // prelude, from the states the lists before leave.
            if let Some(late) = crate::texture::late_copies(list) {
                lists.push(Some(late));
            }
            if let Some(prelude) = list.prelude() {
                lists.push(Some(prelude));
            }
            lists.push(list.list.cast().ok());
        }
        batch.flush();
        // `XTP_MESH_STATS`: what the draws read, against title memory.
        #[cfg(feature = "diagnostics")]
        crate::meshstat::submitting(runs, batch.submission);
        // Copies of big title buffers in the card's memory
        // (`shadow::cards_around`): what their CPU side took goes up first,
        // and what the shaders wrote is found on the card after the lists,
        // in the same submission.
        let (cards_first, cards_last) = {
            let _copy = crate::shadow::cards().then(|| Phase::Copies.enter());
            crate::shadow::cards_around(batch.submission)
        };
        if let Some(first) = cards_first {
            lists.insert(0, Some(first));
            firsts.iter_mut().skip(1).for_each(|at| *at += 1);
        }
        lists.extend(cards_last.map(Some));
        let (mut pending, touches) = Pending::of(runs, batch.submission);
        // Wait for the background descriptor heap sync before sending the
        // command lists to the GPU (counted in preparing if it was not
        // finished yet).
        if let Some(sync) = heaps_synced {
            sync.finish();
        }
        // Preparing, without the waits for earlier submissions (counted as
        // waiting).
        let waited_now = waited_here();
        let waited = std::time::Duration::from_micros(waited_now - waited_before);
        let now = std::time::Instant::now();
        let preparing = (now - started).saturating_sub(waited);
        recording.finish(now, waited_now, preparing, &batch.work);
        pending.preparing = preparing;
        pending.title = title;
        // Diagnostics: each list on its own, timed and waited for
        // (`XTP_GPU_TIMING`).
        #[cfg(feature = "diagnostics")]
        let groups: Vec<&[Option<ID3D12CommandList>]> = if timing() {
            firsts
                .iter()
                .zip(firsts.iter().skip(1).chain([&lists.len()]))
                .map(|(&a, &b)| &lists[a..b])
                .collect()
        } else {
            vec![&lists[..]]
        };
        #[cfg(not(feature = "diagnostics"))]
        let groups = [&lists[..]];
        for (i, group) in groups.iter().enumerate() {
            *submitted += 1;
            // Before the GPU may start on it (`Host::executing`).
            self.executing.store(*submitted, Ordering::SeqCst);
            let started = std::time::Instant::now();
            // SAFETY: closed lists, not running (waited for above); our fence.
            let signalled = unsafe {
                self.queue.ExecuteCommandLists(group);
                self.queue.Signal(&self.fence, *submitted)
            };
            if let Some(awake) = &self.awake {
                awake.note_submission();
            }
            if let Err(e) = signalled {
                // Nothing would finish it.
                let recorded: Vec<Recorded> = runs.iter().map(|l| l.recorded()).collect();
                fail(format_args!(
                    "a host submission of {} lists was not signalled ({e}){}",
                    runs.len(),
                    self.removal_data(runs, &recorded)
                ));
            }
            #[cfg(feature = "diagnostics")]
            if groups.len() > 1 {
                self.waiting(|| self.wait_gpu(*submitted));
                if started.elapsed() > std::time::Duration::from_millis(50) {
                    // SAFETY: a plain query.
                    let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1);
                    let commands = runs[i]
                        .notes
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .len();
                    tracing::warn!(
                        ms = started.elapsed().as_millis(),
                        commands,
                        "slow host list"
                    );
                    for (ms, op) in runs[i].slowest(frequency, 6) {
                        tracing::warn!("slow list command, {ms:.1} ms: {op}");
                    }
                }
            }
            let _ = (i, started);
        }
        let _ = &firsts;
        for list in runs {
            list.last.store(*submitted, Ordering::Release);
            list.ours.store(!is_title, Ordering::Release);
        }
        // `XTP_PLANE_PROBE`: the probes of the lists' depth plane passes
        // are read once this submission has run.
        #[cfg(feature = "diagnostics")]
        runs.iter()
            .for_each(|list| list.twins_submitted(*submitted));
        self.timeline.push(*submitted, touches, pending);
        crate::timestamps::submitted(&self.queue, runs, *submitted, is_title);
        // `XTP_SYNC_SUBMIT`, and timed lists: finished before returning,
        // still one submission at a time.
        #[cfg(feature = "diagnostics")]
        let sync = sync || groups.len() > 1;
        if sync {
            self.settle(*submitted);
        }
    }

    /// The host fence value of the last submission handed to the queue,
    /// stored before the GPU may start on it: once [`Host::completed`]
    /// reaches what this returned, the GPU has run every submission that
    /// could have read host memory (descriptors, the stride table) before
    /// the call, and later ones read it as it is after the call. 0: none.
    pub(crate) fn executing(&self) -> u64 {
        self.executing.load(Ordering::SeqCst)
    }

    /// The host queue: reserved textures' tile mappings
    /// (`resource::pool::Tiles`) are queued on it, in order with the
    /// submissions.
    pub(crate) fn queue(&self) -> &ID3D12CommandQueue {
        &self.queue
    }

    /// The host fence value the GPU has reached (all ones once the device
    /// is removed).
    pub(crate) fn completed(&self) -> u64 {
        // SAFETY: a plain query.
        unsafe { self.fence.GetCompletedValue() }
    }

    /// Finishes, in order, the submissions the GPU has run (no waiting).
    fn finish_reached(&self) {
        // SAFETY: a plain query (all ones once the device is removed).
        let reached = unsafe { self.fence.GetCompletedValue() };
        let seen = std::time::Instant::now();
        self.timeline.finish(reached, |p| self.complete(p, seen));
        // Raw views of the stride tables that the fence has reached go
        // back to their pools now, however long their heap stays unused.
        crate::descriptor::host::fence_reached();
    }

    /// Finishes every submission up to the one that signals `value`,
    /// waiting for the GPU to run them.
    #[track_caller]
    pub(crate) fn settle(&self, value: u64) {
        if self
            .timeline
            .span()
            .is_some_and(|(oldest, _)| oldest <= value)
        {
            self.waiting(|| {
                self.wait_gpu(value);
                self.finish_reached();
            });
        }
    }

    /// Finishes every submission made so far.
    #[track_caller]
    pub(crate) fn drain(&self) {
        if let Some((_, newest)) = self.timeline.span() {
            self.settle(newest);
        }
    }

    /// Finishes the frame presented last, up to its present's copy: before
    /// a flip wait ends (a console flips a frame once the GPU is done with
    /// it), and before a present records its copy (the title is then never
    /// more than one frame ahead, even without flip waits). At once when
    /// that present finished every submission (frames in flight off).
    pub(crate) fn flip(&self) {
        let presented = self.presented.load(Ordering::Acquire);
        self.at_frame_end(|| self.settle(presented));
    }

    /// Once a present's copy is submitted (by `copy`): the frame it ends is
    /// the one the next flip and present wait for. Unless it may run on
    /// (`ahead`), every submission is finished now, as before frames in
    /// flight: the title may take a present for the GPU being done.
    fn presented(&self, copy: &HostList, ahead: bool) {
        self.presented
            .store(copy.last.load(Ordering::Acquire), Ordering::Release);
        if !ahead {
            self.at_frame_end(|| self.drain());
        }
    }

    /// Runs `f`, a wait at a frame end, adding what this thread waited in
    /// it to the frame-rate line's waiting at flips and presents.
    fn at_frame_end(&self, f: impl FnOnce()) {
        let before = waited_here();
        f();
        let waited = waited_here() - before;
        SUBMISSION_US[FRAME_END_US].fetch_add(waited, Ordering::Relaxed);
    }

    /// Before a submission's step reads title memory `[address, address +
    /// len)` (or, with `writing`, writes it): finishes the submissions in
    /// flight whose GPU work writes it (or reads it), so that the step sees
    /// title memory and host copies as if they had run when submitted.
    #[track_caller]
    fn settle_range(&self, address: usize, len: usize, writing: bool) {
        if let Some(value) = self
            .timeline
            .conflict(address, address.saturating_add(len), writing)
        {
            self.settle(value);
        }
    }

    /// Identity of the host copy an earlier submission is writing in a CBV's range.
    pub(crate) fn fill_writer(&self, address: u64, len: u64) -> Option<usize> {
        self.timeline
            .fill_writer(address as usize, address.checked_add(len)? as usize)
    }

    /// Runs `signal` once every submission made so far has finished: at
    /// once if none is in flight.
    pub(crate) fn after(&self, signal: impl FnOnce() + Send + 'static) {
        self.timeline.after(signal);
    }

    /// Runs `f`, counting its time as the title's threads waiting for
    /// submissions (for the GPU, or for their finishing): not on the
    /// completion thread, and once however deep the calls go. Diagnostics
    /// builds also count it by where the wait was asked ([`Waits`]).
    #[track_caller]
    fn waiting<T>(&self, f: impl FnOnce() -> T) -> T {
        if FINISHER.get() {
            return f();
        }
        #[cfg(feature = "diagnostics")]
        let at = std::panic::Location::caller();
        let (depth, so_far) = WAITING.get();
        WAITING.set((depth + 1, so_far));
        let started = std::time::Instant::now();
        let out = f();
        let (depth, so_far) = WAITING.get();
        let us = if depth == 1 {
            started.elapsed().as_micros() as u64
        } else {
            0
        };
        WAITING.set((depth - 1, so_far + us));
        SUBMISSION_US[WAITED_US].fetch_add(us, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        if us > 0 {
            WAITS.lock().unwrap_or_else(|e| e.into_inner()).add(at, us);
        }
        out
    }

    /// Blocks until the GPU has run what signals `value`; ends the run if
    /// the device is removed meanwhile. The wait goes in
    /// slices that ask about removal, which need not wake a fence's
    /// waiters, and a long one is logged (as xcl's fence waits do).
    fn wait_gpu(&self, value: u64) {
        /// Milliseconds between checks.
        const SLICE: u32 = 500;
        // SAFETY: plain calls on our fence and device and this thread's
        // event.
        unsafe {
            if self.fence.GetCompletedValue() >= value {
                return;
            }
            let started = std::time::Instant::now();
            let event = EVENT.with(|e| e.0);
            // Once: a wake for an earlier registration (left by a wait
            // that ended on removal) is told apart by the value.
            let registered = !event.is_invalid()
                && match self.fence.SetEventOnCompletion(value, event) {
                    Ok(()) => true,
                    Err(e) => {
                        tracing::warn!(error = %e, "host fence wait failed; polling");
                        false
                    }
                };
            let mut warned = 0;
            while self.fence.GetCompletedValue() < value {
                if registered {
                    WaitForSingleObject(event, SLICE);
                } else {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                if self.device.GetDeviceRemovedReason().is_err() {
                    self.removed(None);
                }
                // Every 5 s: Windows resets a GPU after about 2 s of one
                // submission, so this is work that never ends.
                let seconds = started.elapsed().as_secs();
                if seconds >= 5 * (warned + 1) {
                    warned = seconds / 5;
                    tracing::warn!(
                        value,
                        completed = self.fence.GetCompletedValue(),
                        seconds,
                        "still waiting for the host GPU"
                    );
                }
            }
        }
    }

    /// Ends the run on a removed device, with the device-removed data of
    /// every submission in flight (any of them may have faulted): the one
    /// being finished, if any, then the others. Once: a second thread that
    /// finds the device removed waits for the end.
    fn removed(&self, finishing: Option<&Pending>) -> ! {
        static REPORTING: Mutex<()> = Mutex::new(());
        let _once = REPORTING.lock().unwrap_or_else(|e| e.into_inner());
        let (mut runs, mut recorded): (Vec<&HostList>, Vec<Recorded>) = (Vec::new(), Vec::new());
        let mut add = |p: &Pending| {
            runs.extend(p.lists.iter().map(|l| l.get()));
            recorded.extend(p.recorded.iter().cloned());
        };
        finishing.into_iter().for_each(&mut add);
        let first = finishing.map_or(0, |p| p.lists.len());
        self.timeline.each_work(add);
        fail(format_args!(
            "host GPU device removed after a submission of {first} lists ({} lists in \
             flight){}{}",
            runs.len(),
            self.removal_data(&runs, &recorded),
            last_notes(&recorded)
        ));
    }

    /// The completion thread: finishes each submission as soon as the GPU
    /// has run it, so that title fences and bottom-of-pipe writes move on
    /// their own (the title waits for its events and reads its memory).
    ///
    /// A panic in it ends the run: the thread would be gone, and the title
    /// would wait for its fences for ever.
    fn finish_loop(&self) {
        crate::cores::keep_on_fast_cores("xtp-gpu-finisher");
        FINISHER.set(true);
        loop {
            let finished = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let value = self.timeline.next();
                self.wait_gpu(value);
                self.finish_reached();
            }));
            if let Err(panic) = finished {
                let what = panic
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_owned())
                    .or_else(|| panic.downcast_ref::<String>().cloned())
                    .unwrap_or_default();
                fail(format_args!(
                    "the host GPU's completion thread panicked ({what}): no submission would \
                     finish any more"
                ));
            }
        }
    }

    /// Once the GPU has run a submission (seen done at `seen`): what its
    /// GPU copies and shaders wrote goes back into title memory, then its
    /// bottom-of-pipe writes, which tell the title the GPU is done.
    fn complete(&self, p: Pending, seen: std::time::Instant) {
        #[cfg(test)]
        if submissions::PANIC_IN_COMPLETE.swap(false, Ordering::SeqCst) {
            panic!("a panic a test asked for");
        }
        // SAFETY: a plain query.
        if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
            self.removed(Some(&p));
        }
        for size in p.recorded.iter().flat_map(|r| r.rt_sizes.iter()) {
            size.complete();
        }
        // About how long it ran: from its submission or the end of the one
        // before, whichever is later, until it was seen done.
        let ran = {
            let mut last = self.seen_done.lock().unwrap_or_else(|e| e.into_inner());
            let from = last.map_or(p.submitted, |l| l.max(p.submitted));
            *last = Some(last.map_or(seen, |l| l.max(seen)));
            seen.saturating_duration_since(from)
        };
        SUBMISSION_US[GPU_US].fetch_add(ran.as_micros() as u64, Ordering::Relaxed);
        // The host resets a device whose submission runs for about 2 s.
        if ran > std::time::Duration::from_millis(250) {
            let pipelines: std::collections::BTreeSet<&String> =
                p.recorded.iter().flat_map(|r| r.pipelines.iter()).collect();
            tracing::warn!(
                ms = ran.as_millis(),
                lists = p.lists.len(),
                ?pipelines,
                "slow host submission"
            );
        }
        // What the GPU wrote goes back before the bottom-of-pipe writes,
        // which tell the title the GPU is done. GPU copies first: all of
        // each destination; shader-written ranges then add what differs
        // from their snapshot, each byte once per submission, however many
        // dispatches declared ranges over it (the world's passes declare the
        // same big ranges dozens of times).
        #[cfg(feature = "diagnostics")]
        for sample in p.recorded.iter().flat_map(|r| r.dispatch_samples.iter()) {
            sample.report();
        }
        #[cfg(feature = "diagnostics")]
        for sample in p.recorded.iter().flat_map(|r| r.rt_samples.iter()) {
            sample.report();
        }
        // `XTP_MESH_STATS`: what the draws read, as the GPU left it.
        #[cfg(feature = "diagnostics")]
        crate::meshstat::finished(p.submission, p.recorded.iter().map(|r| &r.meshes));
        let copying = std::time::Instant::now();
        for (shadow, address, len) in &p.copied {
            let written = shadow.write_back(p.submission, *address, *len);
            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
        }
        let written = crate::shadow::copy_back_all(p.submission, &p.back);
        COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
        for back in &p.texture_backs {
            let written = crate::texback::land(back);
            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
        }
        let back = copying.elapsed();
        COPY_BACK[0].fetch_add(back.as_micros() as u64, Ordering::Relaxed);
        for &(at, what, bytes) in &p.writes {
            what.write(at as *mut u8, bytes);
        }
        self.log_messages();
        let took = Ran {
            gpu: ran,
            total: p.preparing + ran + back,
        };
        if let Some((lists, ran)) = p.title.and_then(|t| t.counted(p.lists.len(), took)) {
            TITLE
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .add(lists, ran);
        }
    }

    /// With `XTP_D3D12_DRED`: the faulting address and the allocations
    /// it lies in (by the names we give them) or the title memory copy just
    /// below it; for each list that did not finish, the command it stopped
    /// at and what the draws and dispatches in flight there run (or, in
    /// lists that never started, the pipelines they bind).
    fn removal_data(&self, runs: &[&HostList], recorded: &[Recorded]) -> String {
        let Ok(dred) = self.device.cast::<ID3D12DeviceRemovedExtendedData1>() else {
            return String::new();
        };
        let mut text = String::new();
        // DRED 1.2: the device state, each stopped list's last finished and
        // first unfinished commands with their markers, the faulting
        // allocations' ranges. Our markers' texts by list, for a runtime
        // that keeps none with its breadcrumbs (NVIDIA's did not).
        #[cfg(feature = "diagnostics")]
        let notes: Vec<Vec<String>> = recorded
            .iter()
            .map(|r| r.notes.iter().map(|(note, _)| note.clone()).collect())
            .collect();
        #[cfg(not(feature = "diagnostics"))]
        let notes: Vec<Vec<String>> = Vec::new();
        text.push_str(&crate::dred::report(&self.device, runs, &notes));
        let mut fault_va = None;
        // SAFETY: the outputs are the runtime's, valid while the device
        // lives; nodes are walked as linked, histories have their counts.
        unsafe {
            if let Ok(fault) = dred.GetPageFaultAllocationOutput1() {
                let va = fault.PageFaultVA;
                if va != 0 {
                    fault_va = Some(va);
                    text.push_str(&format!("; page fault at {va:#x}"));
                } else if dred_on() {
                    // xcl#1592: a real page fault (at 0x700000000) came back
                    // as VA 0 on NVIDIA.
                    text.push_str(
                        "; page fault VA 0: none, or one the driver did not place (not a null \
                         address)",
                    );
                }
                for (what, head) in [
                    ("inside", fault.pHeadExistingAllocationNode),
                    ("inside freed", fault.pHeadRecentFreedAllocationNode),
                ] {
                    let mut names = Vec::new();
                    let mut node = head;
                    while let Some(n) = node.as_ref()
                        && names.len() < 8
                    {
                        let name = if n.ObjectNameW.is_null() {
                            "unnamed".to_owned()
                        } else {
                            n.ObjectNameW.to_string().unwrap_or_default()
                        };
                        names.push(format!("{name} (type {})", n.AllocationType.0));
                        node = n.pNext;
                    }
                    if !names.is_empty() {
                        text.push_str(&format!("; {what} {names:?}"));
                    }
                }
                if va != 0 {
                    text.push_str(&crate::shadow::near(va));
                    text.push_str(&crate::descriptor::host::near_heaps(va));
                }
            }
            // The plain output: the extended one (with marker texts) came
            // back without stopped lists on this host.
            let crumbs = match dred.GetAutoBreadcrumbsOutput() {
                Ok(crumbs) => crumbs,
                Err(e) => {
                    let why = if dred_on() {
                        ""
                    } else {
                        ": device-removed data is off (XTP_D3D12_DRED=1, diagnostics builds)"
                    };
                    text.push_str(&format!("; no breadcrumbs ({e}){why}"));
                    return text;
                }
            };
            // The run a node is about: by its list, or by the name we gave
            // that (NVIDIA's DRED gave a list's name and no pointer to it,
            // xcl#1592).
            let run_of = |p: &D3D12_AUTO_BREADCRUMB_NODE| {
                let ours = p.pCommandList.as_ref().map(|l| l.as_raw());
                let named = (!p.pCommandListDebugNameW.is_null())
                    .then(|| p.pCommandListDebugNameW.to_string().unwrap_or_default());
                runs.iter().position(|r| {
                    Some(r.list.as_raw()) == ours
                        || named.as_deref() == Some(list_name(&r.list).as_str())
                })
            };
            // Every list's progress (done/count), in the runtime's order.
            let mut progress = Vec::new();
            let mut node = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = node.as_ref() {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                let index = run_of(p);
                progress.push(format!("{done}/{} {index:?}", p.BreadcrumbCount));
                node = p.pNext;
            }
            text.push_str(&format!("; breadcrumbs (done/count, list) {progress:?}"));
            let mut plain = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = plain.as_ref() {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                if done < p.BreadcrumbCount && !p.pCommandHistory.is_null() {
                    let history =
                        std::slice::from_raw_parts(p.pCommandHistory, p.BreadcrumbCount as usize);
                    let index = run_of(p);
                    text.push_str(&format!(
                        "; list {index:?} stopped at command {done} of {} (op {})",
                        p.BreadcrumbCount, history[done as usize].0
                    ));
                    let run = index.and_then(|i| recorded.get(i));
                    // The k-th marker command is the k-th note.
                    #[cfg(feature = "diagnostics")]
                    let markers: Vec<(u32, String)> = run.map_or_else(Vec::new, |r| {
                        let notes = &r.notes;
                        history
                            .iter()
                            .enumerate()
                            .filter(|(_, op)| **op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
                            .zip(notes.iter())
                            .map(|((at, _), (note, range))| {
                                let place = fault_va.zip(*range).map(|(va, r)| fault_in(va, r));
                                (at as u32, format!("{note}{}", place.unwrap_or_default()))
                            })
                            .collect()
                    });
                    #[cfg(not(feature = "diagnostics"))]
                    let markers: Vec<(u32, String)> = Vec::new();
                    #[cfg(not(feature = "diagnostics"))]
                    let _ = fault_va;
                    if done > 0 {
                        text.push_str(&in_flight(history, &markers, done as usize));
                    } else if let Some(r) = run {
                        text.push_str(&format!(", its pipelines {:?}", r.pipelines));
                    }
                }
                plain = p.pNext;
            }
        }
        text
    }

    /// The debug layer's message queue (`XTP_D3D12_DEBUG=1` in a
    /// diagnostics build), for tests to fail on its errors.
    #[cfg(test)]
    pub(crate) fn message_queue(&self) -> Option<&ID3D12InfoQueue> {
        self.messages.as_ref()
    }

    /// Moves the debug layer's messages into our log; returns the first
    /// error among them.
    pub(crate) fn log_messages(&self) -> Option<String> {
        let Some(queue) = &self.messages else {
            return None;
        };
        let mut first_error = None;
        // SAFETY: each message is read into a buffer of the size the queue
        // reports, then the queue is emptied.
        unsafe {
            for i in 0..queue.GetNumStoredMessages() {
                let mut size = 0;
                if queue.GetMessage(i, None, &mut size).is_err() || size == 0 {
                    continue;
                }
                let mut buffer = vec![0u64; size.div_ceil(8)];
                let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if queue.GetMessage(i, Some(message), &mut size).is_ok() {
                    let m = &*message;
                    let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                    let text = String::from_utf8_lossy(text);
                    let text = text.trim_end_matches('\0');
                    tracing::warn!(id = m.ID.0, severity = m.Severity.0, "D3D12: {text}");
                    if m.Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 && first_error.is_none() {
                        first_error = Some(text.to_owned());
                    }
                }
            }
            queue.ClearStoredMessages();
        }
        first_error
    }

    /// A host RTV, valid until recorded. A null resource needs a typed description.
    pub(crate) fn rtv(
        &self,
        resource: impl windows::core::Param<ID3D12Resource>,
        desc: Option<&D3D12_RENDER_TARGET_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_rtv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.rtvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.rtv_size) as usize;
            self.device
                .CreateRenderTargetView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }

    /// A host depth view of `resource`, valid until it is recorded.
    pub(crate) fn dsv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_dsv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.dsvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.dsv_size) as usize;
            self.device
                .CreateDepthStencilView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }
}

impl HostList {
    /// The timestamp heap and its read-back buffer (diagnostics, timing on).
    #[cfg(feature = "diagnostics")]
    fn stamps(&self) -> Option<&(ID3D12QueryHeap, ID3D12Resource)> {
        if !timing() {
            return None;
        }
        self.stamps
            .get_or_init(|| {
                let device = &host()?.device;
                let mut heap: Option<ID3D12QueryHeap> = None;
                let mut back: Option<ID3D12Resource> = None;
                let readback = D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_READBACK,
                    ..Default::default()
                };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: u64::from(STAMPS) * 8,
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
                // SAFETY: plain object creation.
                unsafe {
                    device
                        .CreateQueryHeap(
                            &D3D12_QUERY_HEAP_DESC {
                                Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                                Count: STAMPS,
                                NodeMask: 0,
                            },
                            &mut heap,
                        )
                        .ok()?;
                    device
                        .CreateCommittedResource(
                            &readback,
                            D3D12_HEAP_FLAG_NONE,
                            &desc,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            None,
                            &mut back,
                        )
                        .ok()?;
                }
                Some((heap?, back?))
            })
            .as_ref()
    }

    /// The noted commands that took longest on the GPU, longest first, as
    /// (milliseconds, text) (diagnostics, timing on, after the list ran).
    #[cfg(feature = "diagnostics")]
    fn slowest(&self, frequency: u64, n: usize) -> Vec<(f64, String)> {
        let notes = self.notes.lock().unwrap_or_else(|e| e.into_inner());
        let (Some((_, back)), count) = (self.stamps(), notes.len().min(STAMPS as usize - 1)) else {
            return Vec::new();
        };
        let mut data = std::ptr::null_mut();
        let range = D3D12_RANGE {
            Begin: 0,
            End: (count + 1) * 8,
        };
        // SAFETY: a read-back buffer the finished list resolved into.
        let stamps: Vec<u64> = unsafe {
            if back.Map(0, Some(&range), Some(&mut data)).is_err() {
                return Vec::new();
            }
            let s = std::slice::from_raw_parts(data.cast::<u64>(), count + 1).to_vec();
            back.Unmap(0, None);
            s
        };
        let mut times: Vec<(f64, String)> = (0..count)
            .map(|i| {
                let ticks = stamps[i + 1].saturating_sub(stamps[i]);
                (
                    ticks as f64 * 1e3 / frequency.max(1) as f64,
                    notes[i].0.chars().take(300).collect(),
                )
            })
            .collect();
        times.sort_by(|a, b| b.0.total_cmp(&a.0));
        times.truncate(n);
        times
    }

    pub(crate) fn close(&self) {
        // What the recording asked to run as it closes ([`HostList::at_close`]).
        let closing = std::mem::take(&mut self.deferred().closing);
        for work in closing {
            work(self);
        }
        // Depth textures whose twins the list wrote last get what was
        // written (`twins`).
        self.twins_to_depth();
        self.textures_back();
        crate::timestamps::close(self);
        #[cfg(feature = "diagnostics")]
        crate::passdump::closed(self);
        #[cfg(feature = "diagnostics")]
        if let Some((heap, back)) = self.stamps() {
            let count =
                (self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32).min(STAMPS - 1);
            // SAFETY: an open list, our heap and read-back buffer.
            unsafe {
                self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, count);
                self.list
                    .ResolveQueryData(heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, count + 1, back, 0);
            }
        }
        // SAFETY: our list.
        if let Err(e) = unsafe { self.list.Close() } {
            fail(format_args!(
                "a host command list did not close ({e}): a command recorded into it was invalid"
            ));
        }
    }

    /// Waits for the list's last submission and finishes it (and those
    /// before it).
    #[track_caller]
    pub(crate) fn settle(&self) {
        let last = self.last.load(Ordering::Acquire);
        if last != 0
            && let Some(host) = made()
        {
            host.settle(last);
        }
    }

    /// Whether the list's last submission, with the memory the list
    /// records into now, has not finished yet.
    pub(crate) fn running(&self) -> bool {
        let last = self.last.load(Ordering::Acquire);
        last > self.fresh_after.load(Ordering::Acquire)
            && made().is_some_and(|h| h.timeline.span().is_some_and(|(oldest, _)| oldest <= last))
    }

    /// Opens the list for recording again. D3D12 lets a list be reset as
    /// soon as it is submitted (titles record the next frame into it while
    /// the GPU runs this one): what its running submission uses (allocator,
    /// prelude, upload and scratch memory, the objects, textures and
    /// dynamic descriptors it keeps, its budget word) then goes to the
    /// timeline, freed or reused once that submission has finished, and
    /// the list records into fresh ones.
    ///
    /// A list of ours (last run by [`Host::execute`]) waits for its
    /// submission instead. The title waits for its fences before it
    /// rewrites what its running lists read; our code rewrites its own as
    /// soon as it records again (the plane compositor's shader-visible
    /// views, `dxgi::planes`), which a present finishing every submission
    /// used to make safe, and one frame in flight does not.
    pub(crate) fn reset(&self) {
        if self.ours.load(Ordering::Acquire) && self.running() {
            self.settle();
        }
        note_scratch_use(&self.scratch.lock().unwrap_or_else(|e| e.into_inner()));
        let running = if self.running() { made() } else { None };
        let allocator = {
            let mut allocator = self.allocator.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(host) = running {
                let (Some(fresh), Some(prelude)) = (host.fresh_allocator(), host.fresh_prelude())
                else {
                    fail(format_args!(
                        "no fresh host command allocator for a running list"
                    ));
                };
                // In the order `prelude` and `scratch` take them.
                let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
                let mut old_prelude = self.prelude.lock().unwrap_or_else(|e| e.into_inner());
                let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
                let mut budget = self.budget.lock().unwrap_or_else(|e| e.into_inner());
                let retired = Retired {
                    allocator: std::mem::replace(&mut *allocator, fresh),
                    prelude: std::mem::replace(&mut *old_prelude, prelude),
                    deferred: std::mem::take(&mut *self.deferred()),
                    scratch: std::mem::take(&mut *scratch),
                    states: std::mem::take(&mut *states),
                    budget: budget.take(),
                };
                host.after(move || retired.retire());
                // Still named by that submission: dropping waits for it.
                self.fresh_after
                    .store(self.last.load(Ordering::Acquire), Ordering::Release);
            } else {
                *self.deferred() = Deferred::default();
                self.states
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clear();
                let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
                (scratch.current, scratch.used) = (0, 0);
                // SAFETY: nothing running uses the allocator.
                if let Err(e) = unsafe { allocator.Reset() } {
                    fail(format_args!("a host command allocator did not reset ({e})"));
                }
            }
            allocator.clone()
        };
        self.decode_uses
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
        self.decode_pipeline.store(false, Ordering::Relaxed);
        *self.clamps.lock().unwrap_or_else(|e| e.into_inner()) = Default::default();
        self.twins.lock().unwrap_or_else(|e| e.into_inner()).clear();
        // New ones: a submission in flight keeps its own (`Recorded`).
        *self.pipelines.lock().unwrap_or_else(|e| e.into_inner()) = Default::default();
        self.writer.store(0, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        {
            *self.notes.lock().unwrap_or_else(|e| e.into_inner()) = Default::default();
        }
        // SAFETY: a closed list (D3D12 allows resetting one that still
        // runs) and an allocator nothing running uses.
        if let Err(e) = unsafe { self.list.Reset(&allocator, None) } {
            fail(format_args!("a host command list did not reset ({e})"));
        }
        crate::timestamps::open(self);
    }

    /// Moves a title texture's host copy to `wanted` (a host state) in this
    /// list: the host texture and the state it leaves, if that is a change.
    /// The list's first use of a texture records nothing: its prelude brings
    /// the texture there at submission.
    pub(crate) fn transition(
        &self,
        resource: crate::object::This,
        wanted: u32,
    ) -> Option<(&'static ID3D12Resource, u32)> {
        let texture = crate::resource::host_resource(resource)?;
        let copy = crate::resource::host_copy(resource)?;
        let wanted = legal_state(texture, wanted);
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        match states.entry(std::sync::Arc::as_ptr(&copy) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert((wanted, wanted, copy));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let before = std::mem::replace(&mut o.get_mut().1, wanted);
                (before != wanted).then_some((texture, before))
            }
        }
    }

    /// At submission: the list that moves each texture this list uses from
    /// its real state to the one the list starts with; the textures' real
    /// states become those the list ends with.
    fn prelude(&self) -> Option<ID3D12CommandList> {
        // Depth textures with a colour twin: the side the list uses as it
        // finds it is first copied from the other if an earlier list left
        // it behind (`twins`); the copies' textures go through the copy
        // states.
        let syncs = self.twin_syncs();
        let states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        let (to_copy, mut barriers, copying) = twins::sync_states(&syncs, &states);
        barriers.extend(states.iter().filter_map(|(key, (first, last, copy))| {
            let real = copy.state.swap(*last, Ordering::Relaxed);
            let real = copying.get(key).copied().unwrap_or(real);
            (real != *first).then(|| {
                transition(
                    &copy.texture,
                    D3D12_RESOURCE_STATES(real as i32),
                    D3D12_RESOURCE_STATES(*first as i32),
                )
            })
        }));
        if barriers.is_empty() && syncs.is_empty() {
            return None;
        }
        let prelude = self.prelude.lock().unwrap_or_else(|e| e.into_inner());
        let (allocator, list) = &*prelude;
        // SAFETY: the prelude is not executing (`run` waits for the list's
        // last submission first, unless a reset gave it a fresh one); the
        // textures are live.
        let recorded = unsafe {
            allocator.Reset().and_then(|()| {
                list.Reset(allocator, None)?;
                if !to_copy.is_empty() {
                    list.ResourceBarrier(&to_copy);
                }
                for (depth, twin, colour) in &syncs {
                    twins::record_twin_copy(list, &depth.texture, twin, *colour);
                }
                if !barriers.is_empty() {
                    list.ResourceBarrier(&barriers);
                }
                list.Close()
            })
        };
        if let Err(e) = recorded {
            fail(format_args!("a host state prelude did not record ({e})"));
        }
        list.cast().ok()
    }

    /// Notes what the command about to be recorded does, as a marker
    /// command whose text the list keeps (diagnostics, with device-removed
    /// data on): the report pairs the breadcrumbs' marker commands with the
    /// texts in order and names the commands a fault stopped among.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_op(&self, what: impl FnOnce() -> String) {
        self.note(what, None);
    }

    /// As [`HostList::note_op`], for a copy that reads the host GPU range
    /// `(address, bytes)`: the report places the faulting address in it.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_read(&self, what: impl FnOnce() -> String, range: (u64, u64)) {
        self.note(what, Some(range));
    }

    #[cfg(feature = "diagnostics")]
    fn note(&self, what: impl FnOnce() -> String, range: Option<(u64, u64)>) {
        let what = crate::passdump::op(self, what);
        if dred_on() {
            let at = self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32;
            if let Some((heap, _)) = self.stamps()
                && at < STAMPS - 1
            {
                // SAFETY: an open list and our query heap.
                unsafe { self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, at) };
            }
            let what = what();
            let text: Vec<u16> = what.encode_utf16().chain([0]).collect();
            const UNICODE: u32 = 0;
            // SAFETY: an open list; the marker is copied when recorded.
            unsafe {
                self.list
                    .SetMarker(UNICODE, Some(text.as_ptr().cast()), 2 * text.len() as u32)
            };
            let mut notes = self.notes.lock().unwrap_or_else(|e| e.into_inner());
            std::sync::Arc::make_mut(&mut notes).push((what, range));
        }
    }

    /// Notes a pipeline the list binds.
    pub(crate) fn used_pipeline(&self, name: &str) {
        let mut pipelines = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
        if crate::readmap::enabled() {
            let digest = blake3::hash(name.as_bytes());
            self.writer.store(
                u64::from_le_bytes(digest.as_bytes()[..8].try_into().unwrap()),
                Ordering::Relaxed,
            );
        }
        if !pipelines.contains(name) {
            std::sync::Arc::make_mut(&mut pipelines).insert(name.to_owned());
        }
    }

    /// What a submission of the list keeps of its recording.
    fn recorded(&self) -> Recorded {
        #[cfg(feature = "diagnostics")]
        let (dispatch_samples, meshes) = {
            let deferred = self.deferred();
            (deferred.dispatch_samples.clone(), deferred.meshes.clone())
        };
        #[cfg(feature = "diagnostics")]
        let rt_samples = self.deferred().rt_samples.clone();
        let rt_sizes = self.deferred().rt_sizes.clone();
        Recorded {
            rt_sizes,
            #[cfg(feature = "diagnostics")]
            rt_samples,
            #[cfg(feature = "diagnostics")]
            dispatch_samples,
            pipelines: self
                .pipelines
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone(),
            #[cfg(feature = "diagnostics")]
            notes: self.notes.lock().unwrap_or_else(|e| e.into_inner()).clone(),
            #[cfg(feature = "diagnostics")]
            meshes,
        }
    }

    #[cfg(feature = "diagnostics")]
    pub(crate) fn rt_sample(&self, sample: crate::rt::probe::Sample) {
        std::sync::Arc::make_mut(&mut self.deferred().rt_samples).push(sample);
    }

    #[cfg(feature = "diagnostics")]
    pub(crate) fn dispatch_sample(&self, sample: crate::indirect::Sample) {
        std::sync::Arc::make_mut(&mut self.deferred().dispatch_samples).push(sample);
    }

    /// Notes what a draw or dispatch of the list read and wrote
    /// (`crate::meshstat`).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn mesh_probe(&self, probe: crate::meshstat::Probe) {
        std::sync::Arc::make_mut(&mut self.deferred().meshes).push(probe);
    }

    /// What the list's draws read and its dispatches wrote so far.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn mesh_probes(&self) -> std::sync::Arc<Vec<crate::meshstat::Probe>> {
        self.deferred().meshes.clone()
    }

    fn deferred(&self) -> std::sync::MutexGuard<'_, Deferred> {
        self.deferred.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Copies `len` bytes from `src` to `dst` when the list is submitted.
    pub(crate) fn defer_copy(&self, dst: *mut u8, src: *const u8, len: usize) {
        let mut deferred = self.deferred();
        deferred.buffer_writes.record(0, dst as usize, len);
        deferred
            .copies
            .push(Step::Copy(dst as usize, src as usize, len));
        deferred.refreshed.clear();
        deferred.refresh_epoch += 1;
    }

    /// Copies `len` bytes of title memory at `src` into the list's upload
    /// memory at `dst` when the list is submitted.
    pub(crate) fn defer_fill(&self, dst: *mut u8, src: *const u8, len: usize) {
        self.defer_fill_at(dst as usize, 0, src as usize, len);
    }

    /// `base` est le début du tampon mappé. Deux tampons distincts peuvent
    /// se suivre en mémoire sans constituer une seule destination valide.
    pub(crate) fn defer_fill_at(&self, base: usize, offset: usize, src: usize, len: usize) {
        let dst = base + offset;
        let mut deferred = self.deferred();
        if let Some(Step::Fill(prev_dst, prev_src, bytes)) = deferred.copies.last_mut()
            && *prev_dst >= base
            && prev_dst.checked_add(*bytes) == Some(dst)
            && prev_src.checked_add(*bytes) == Some(src)
        {
            *bytes += len;
            deferred.merged_fills += 1;
            return;
        }
        deferred.copies.push(Step::Fill(dst, src, len));
    }

    /// Counts the title copies after which [`HostList::defer_refresh`]
    /// refreshes a range again: a caller that refreshed what it reads at
    /// one count has nothing to refresh again while it holds.
    pub(crate) fn refresh_epoch(&self) -> u64 {
        self.deferred().refresh_epoch
    }

    /// Runs `work` as the list closes (before its commands end), in the
    /// order given; dropped with the recording if it never closes.
    pub(crate) fn at_close(&self, work: impl FnOnce(&HostList) + Send + 'static) {
        self.deferred().closing.push(Box::new(work));
    }

    /// Runs `work` at each submission of the list until it is reset, before
    /// its steps; what `work` defers runs after them, and at the
    /// submissions after this one too.
    pub(crate) fn at_submit(&self, work: impl Fn(&HostList) + Send + Sync + 'static) {
        self.deferred().submitting.push(std::sync::Arc::new(work));
    }

    /// Runs what the recording asked to run at each submission
    /// ([`HostList::at_submit`]), the list's lock let go meanwhile.
    fn submitting(&self, before: impl FnOnce()) {
        let work = self.deferred().submitting.clone();
        // Picked-view sweeps and their deferred refreshes need the listing
        // published by the mirror. The list's lock is free before waiting.
        if !work.is_empty() {
            before();
        }
        for work in work {
            work(self);
        }
    }

    /// Brings `[address, address + len)` of a buffer's host copy up to date
    /// when the list is submitted (once per range until a title copy
    /// intervenes), keeping the copy alive until the list is reset.
    pub(crate) fn defer_refresh(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(shadow) as usize, address, len);
        if deferred.refreshed.insert(key) {
            if let Some(Step::Refresh(previous, from, bytes)) = deferred.copies.last_mut()
                && std::sync::Arc::ptr_eq(previous, shadow)
                && address <= *from + *bytes
                && *from <= address + len
            {
                let end = (*from + *bytes).max(address + len);
                *from = (*from).min(address);
                *bytes = end - *from;
                return;
            }
            deferred
                .copies
                .push(Step::Refresh(shadow.clone(), address, len));
        }
    }

    /// Shaders of the list write `[address, address + len)` of a buffer's
    /// host copy: copied back into title memory once the list has run.
    pub(crate) fn defer_gpu_write(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let mut deferred = self.deferred();
        let writes = &mut deferred.buffer_writes;
        let copy = std::sync::Arc::as_ptr(shadow) as usize;
        if writes.binding {
            writes.shader_outputs.push((copy, address, len));
        } else {
            writes.record(copy, address, len);
        }
        deferred.indirect_writers.record(&shadow.buffer, true);
        deferred.copies.push(Step::GpuWrite(
            shadow.clone(),
            address,
            len,
            self.writer.load(Ordering::Relaxed),
        ));
    }

    /// A buffer copy the list records on the GPU writes `[address, address
    /// + len)` of a buffer's host copy: written into title memory, all of
    /// it, once the list has run (see [`crate::shadow::copy`]).
    pub(crate) fn defer_gpu_copy(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let mut deferred = self.deferred();
        deferred
            .buffer_writes
            .record(std::sync::Arc::as_ptr(shadow) as usize, address, len);
        deferred.indirect_writers.record(&shadow.buffer, false);
        deferred
            .copies
            .push(Step::GpuCopy(shadow.clone(), address, len));
    }

    /// Counts a memory write of `bytes` the list records on the GPU, once
    /// per submission (see [`crate::shadow::write`]).
    pub(crate) fn count_gpu_write(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        bytes: usize,
    ) {
        let mut deferred = self.deferred();
        deferred
            .buffer_writes
            .record(std::sync::Arc::as_ptr(shadow) as usize, address, bytes);
        deferred.indirect_writers.record(&shadow.buffer, false);
        deferred
            .copies
            .push(Step::GpuWord(shadow.clone(), address, bytes));
    }

    pub(crate) fn unknown_buffer_write(&self) {
        let mut deferred = self.deferred();
        deferred.indirect_writers.unknown();
        deferred.buffer_writes.indexed_output = true;
    }

    /// Shader UAVs are declared while binding, but they write after all the
    /// input copies of that command, even when both views alias.
    pub(crate) fn begin_shader_outputs(&self) {
        self.deferred().buffer_writes.binding = true;
    }

    pub(crate) fn end_shader_outputs(&self, bound: bool) {
        let mut deferred = self.deferred();
        let writes = &mut deferred.buffer_writes;
        writes.binding = false;
        while let Some((copy, address, len)) = writes.shader_outputs.pop() {
            if bound {
                writes.record(copy, address, len);
            }
        }
        if std::mem::take(&mut writes.indexed_output) && bound {
            // A shader indexing writable descriptors at run time can select
            // a descriptor installed after recording.
            writes.serial += 1;
            writes.unknown = writes.serial;
        }
    }

    pub(crate) fn cbv_read(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: u64,
        len: u64,
    ) {
        self.deferred().cbv_reads.insert((
            std::sync::Arc::as_ptr(shadow) as usize,
            address as usize,
            (address + len) as usize,
        ));
    }

    /// Latest submission-time title write and GPU write to this input.
    pub(crate) fn input_writes(&self, copy: Option<usize>, address: u64, len: u64) -> (u64, u64) {
        let deferred = self.deferred();
        let writes = &deferred.buffer_writes;
        let end = address.saturating_add(len) as usize;
        let from = address as usize;
        (
            writes.latest(0, from, end),
            copy.map_or(0, |c| writes.latest(c, from, end).max(writes.unknown)),
        )
    }

    pub(crate) fn indirect_writes(
        &self,
        args: &ID3D12Resource,
        count: Option<&ID3D12Resource>,
    ) -> crate::indirect::Writes {
        self.deferred().indirect_writers.before(args, count)
    }

    /// Notes a title texture written through unordered access; if its kind is
    /// configured for readback ([`crate::texback`]), the list copies it when
    /// closing.
    pub(crate) fn texture_written(&self, resource: crate::object::This) {
        if !crate::texback::wanted(resource) {
            return;
        }
        let mut deferred = self.deferred();
        deferred
            .back_wanted
            .retain(crate::texback::Target::is_valid);
        if !deferred
            .back_wanted
            .iter()
            .any(|target| target.owner() == resource as usize)
            && let Some(target) = crate::texback::Target::new(resource)
        {
            deferred.back_wanted.push(target);
        }
    }

    /// Copies wanted shader-written textures into readback buffers before the
    /// list closes ([`crate::texback::record`]).
    fn textures_back(&self) {
        if !crate::texback::tracking() {
            return;
        }
        const COPY_SOURCE: u32 = 0x800;
        let wanted = std::mem::take(&mut self.deferred().back_wanted);
        for target in wanted {
            let copy = target.copy.clone();
            if let Some(back) = crate::texback::record(&self.list, target, || {
                self.move_to(&copy, COPY_SOURCE);
            }) {
                self.deferred()
                    .texture_backs
                    .push(std::sync::Arc::new(back));
            }
        }
    }

    /// Stores `what` (`bytes` of it) at `dst` once the list has run.
    pub(crate) fn defer_write(&self, dst: *mut u8, what: crate::queue::Written, bytes: usize) {
        self.deferred().writes.push((dst as usize, what, bytes));
    }

    /// A memory value the list waits for (`queue::wait32`,
    /// `queue::wait64`), noted with where it was recorded: after work, and
    /// after how many of the list's own bottom-of-pipe writes.
    pub(crate) fn defer_value_wait(&self, mut wait: crate::queue::ValueWait) {
        let pipelines = !self
            .pipelines
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_empty();
        let states = !self
            .states
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_empty();
        let mut deferred = self.deferred();
        wait.writes_before = deferred.writes.len() as u32;
        wait.work_before =
            pipelines || states || !deferred.copies.is_empty() || wait.writes_before > 0;
        deferred.value_waits.push(wait);
    }

    /// The memory values the list waits for, in recorded order.
    pub(crate) fn value_waits(&self) -> Vec<crate::queue::ValueWait> {
        self.deferred().value_waits.clone()
    }

    /// The values the list's bottom-of-pipe writes store once it has run
    /// (timestamps left out): address, value, bytes.
    pub(crate) fn value_writes(&self) -> Vec<(u64, u64, u8)> {
        let deferred = self.deferred();
        let values = deferred
            .writes
            .iter()
            .filter_map(|&(at, what, bytes)| match what {
                crate::queue::Written::Value(v) => Some((at as u64, v, bytes as u8)),
                _ => None,
            });
        values.collect()
    }

    /// A mapped upload buffer of `size` bytes, alive until the list is
    /// reset.
    ///
    /// shortcut: one buffer per copy; pool them once uploads are frequent.
    #[track_caller]
    pub(crate) fn upload(&self, host: &Host, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let (buffer, data) = host.upload_buffer(size)?;
        name(&buffer, || format!("list upload {size:#x}"));
        self.deferred().uploads.push(buffer.clone());
        Some((buffer, data))
    }

    /// `len` bytes of upload memory (256-byte aligned) until the list is
    /// reset: the host GPU address and the CPU pointer.
    pub(crate) fn scratch(&self, len: u64) -> Option<(u64, *mut u8)> {
        let (gpu, base, offset) = self.scratch_region(len)?;
        Some((gpu, (base + offset) as *mut u8))
    }

    /// Réserve la destination et enregistre son remplissage à la soumission.
    pub(crate) fn scratch_fill(&self, src: u64, len: u64) -> Option<u64> {
        let (gpu, base, offset) = self.scratch_region(len)?;
        self.defer_fill_at(base, offset, src as usize, len as usize);
        Some(gpu)
    }

    /// Adresse GPU de la réservation, base CPU du tampon et décalage dans celui-ci.
    fn scratch_region(&self, len: u64) -> Option<(u64, usize, usize)> {
        let host = host()?;
        let len = len.max(1).next_multiple_of(256);
        if len > SCRATCH_CHUNK {
            SCRATCH_OWN[0].fetch_add(1, Ordering::Relaxed);
            SCRATCH_OWN[1].fetch_add(len, Ordering::Relaxed);
            let (buffer, data) = host.upload_buffer(len)?;
            name(&buffer, || format!("list scratch {len:#x}"));
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            self.deferred().uploads.push(buffer);
            return Some((gpu, data as usize, 0));
        }
        let mut s = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        if s.used + len > SCRATCH_CHUNK && s.current < s.chunks.len() {
            (s.current, s.used) = (s.current + 1, 0);
        }
        if s.current == s.chunks.len() {
            let spare = pool(&SPARE_SCRATCH).pop();
            let chunk = match spare {
                Some(chunk) => chunk,
                None => {
                    // Made without the lock held (it may wait for a heap);
                    // a list records on one thread at a time.
                    drop(s);
                    let (buffer, data) = host.upload_buffer(SCRATCH_CHUNK)?;
                    name(&buffer, || "list scratch".into());
                    // SAFETY: a live buffer.
                    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
                    s = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
                    (buffer, data as usize, gpu)
                }
            };
            s.chunks.push(chunk);
            s.current = s.chunks.len() - 1;
            s.used = 0;
        }
        let (_, data, gpu) = s.chunks[s.current];
        let at = s.used;
        s.used += len;
        Some((gpu + at, data, at as usize))
    }

    /// The host address of the list's indirect-work budget, zeroed at each
    /// submission of the list from the first use after a reset on.
    pub(crate) fn budget(&self) -> Option<ID3D12Resource> {
        let made = self
            .budget
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let (buffer, data) = match made {
            Some(made) => made,
            None => {
                // Made without the lock held (it may wait for a heap).
                let (b, d) = host()?.shared_buffer(256)?;
                let mut budget = self.budget.lock().unwrap_or_else(|e| e.into_inner());
                budget.get_or_insert((b, d as usize)).clone()
            }
        };
        let mut deferred = self.deferred();
        if !deferred.budget_zeroed {
            deferred.budget_zeroed = true;
            let at = data;
            // SAFETY: the list's own mapped word, written before it runs.
            deferred.copies.push(Step::Host(
                Box::new(move || unsafe { (at as *mut u32).write_volatile(0) }),
                (0, 0),
            ));
        }
        Some(buffer)
    }

    /// Keeps `object` alive until the list is reset.
    pub(crate) fn keep(&self, object: &impl Interface) {
        if let Ok(object) = object.cast::<windows::core::IUnknown>() {
            self.deferred().kept.push(object);
        }
    }

    pub(crate) fn rt_recording(&self) -> std::sync::Arc<()> {
        self.deferred()
            .rt_recording
            .get_or_insert_with(Default::default)
            .clone()
    }

    pub(crate) fn rt_compacted_size(&self) -> Option<(u64, crate::rt::compaction::CompactedSize)> {
        let mut deferred = self.deferred();
        let (address, slot) = deferred.rt_size_page.reserve()?;
        std::sync::Arc::make_mut(&mut deferred.rt_sizes).push(slot.clone());
        Some((address, slot))
    }

    pub(crate) fn keep_rt_result(&self, result: &std::sync::Arc<crate::rt::ResultBuffer>) {
        self.deferred().rt_results.push(result.clone());
    }

    /// Whether this is the first time since the reset that the list asks
    /// about `key` (a texture to bring up to date).
    pub(crate) fn first_use(&self, key: usize) -> bool {
        self.deferred().textures.insert(key)
    }

    /// Keeps dynamic descriptor slots `[first, first + n)` of the title
    /// heap at `heap` leased until the list is reset, or, if it is reset
    /// while it runs, until that submission has finished.
    pub(crate) fn keep_dynamic(&self, heap: usize, first: u32, n: u32) {
        self.deferred().dynamic.push(DynamicLease(heap, first, n));
    }

    /// Runs `work` when the list is submitted, in order with its copies.
    /// `work` may write title memory (refreshes after it copy again): the
    /// range `writes` (address, bytes); it reads `reads`.
    pub(crate) fn defer(
        &self,
        writes: (usize, usize),
        reads: (usize, usize),
        work: impl Fn() + Send + 'static,
    ) {
        let mut deferred = self.deferred();
        deferred.buffer_writes.record(0, writes.0, writes.1);
        deferred
            .copies
            .push(Step::Run(Box::new(work), writes, reads));
        deferred.refreshed.clear();
        deferred.refresh_epoch += 1;
    }

    /// As [`HostList::defer`], for work that writes only host memory (and
    /// reads the title's `reads`).
    pub(crate) fn defer_host(&self, reads: (usize, usize), work: impl Fn() + Send + 'static) {
        self.deferred()
            .copies
            .push(Step::Host(Box::new(work), reads));
    }

    /// Brings a texture's staging buffer up to date when the list is
    /// submitted, keeping it alive until the list is reset.
    pub(crate) fn defer_texture(&self, staging: std::sync::Arc<crate::texture::Staging>) {
        self.deferred().copies.push(Step::Texture(staging));
    }

    /// Refreshes a list-owned movie plane after settling its title memory range.
    pub(crate) fn defer_plane(&self, read: (usize, usize), fill: impl Fn() + Send + 'static) {
        self.deferred()
            .copies
            .push(Step::Plane(read, Box::new(fill)));
    }

    /// At each submission of the list, in order with its copies: what the
    /// buffer views of a view heap's slots that its shaders pick read is
    /// brought up to date, and what they write declared written
    /// (`crate::descriptor::host::PickedUse::defer`, once per refresh
    /// epoch).
    pub(crate) fn defer_picked(&self, step: std::sync::Arc<crate::descriptor::host::PickedStep>) {
        self.deferred().copies.push(Step::Picked(step));
    }

    /// The pipeline the list bound last, as its GPU writes name it
    /// (`crate::readmap`; 0 without it).
    pub(crate) fn writer(&self) -> u64 {
        self.writer.load(Ordering::Relaxed)
    }

    /// The steps the list runs at each submission (tests).
    #[cfg(test)]
    pub(crate) fn steps(&self) -> usize {
        self.deferred().copies.len()
    }

    /// Runs the list's submission-time steps; refreshes go into `batch`
    /// (the title's big buffers, a 174 MB one among them, are read by most
    /// lists of a submission, through many ranges).
    fn run_deferred(&self, batch: &mut Batch<'_>) {
        self.run_deferred_with(batch, crate::workers::wide());
    }

    fn run_deferred_with(&self, batch: &mut Batch<'_>, wide: bool) {
        #[cfg(feature = "diagnostics")]
        use crate::copystat::{C, add};
        use preparation::Kind;
        let _steps = Phase::Steps.enter();
        let deferred = self.deferred();
        batch.work.merged_fills += deferred.merged_fills;
        for step in &deferred.copies {
            if wide {
                // Deferred callbacks can observe upload memory or change
                // which texture should be refreshed. Keep runs within their
                // original position in this list, before its late copies.
                if !matches!(step, Step::Fill(..)) {
                    batch.fill_with(true);
                }
                if !matches!(step, Step::Texture(..)) {
                    batch.refresh_textures();
                }
            }
            let kind = match step {
                &Step::Copy(dst, src, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                    }
                    batch.settle(src, len, false);
                    batch.settle(dst, len, true);
                    batch.title_written_at(dst, len);
                    let _copy = Phase::Copies.enter();
                    copy_mapped(dst as *mut u8, src as *const u8, len, true);
                    moved(Moved::CpuCopy, len);
                    batch.work.copy_bytes += len as u64;
                    Kind::Copy
                }
                &Step::Fill(dst, src, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                    }
                    if wide {
                        if let Some(value) = batch.read_conflict(src, len) {
                            batch.fill_with(true);
                            batch
                                .host
                                .expect("conflict belongs to the host")
                                .settle(value);
                        }
                        batch.fills.push((dst, src, len));
                    } else {
                        batch.settle(src, len, false);
                        let _fill = Phase::Fills.enter();
                        // SAFETY: the list keeps the upload destination mapped
                        // and disjoint from title memory through submission.
                        unsafe { crate::shadow::fill_upload(dst as *mut u8, src, len) };
                    }
                    batch.work.fill_bytes += len as u64;
                    Kind::Fill
                }
                Step::Refresh(shadow, address, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Refreshes, 1);
                        add(C::RefreshBytes, *len as u64);
                    }
                    // Title memory an earlier submission's GPU work writes
                    // is current only once that one is written back, if the
                    // title wrote it since the copy took it.
                    batch.settle_refresh(shadow, *address, *len);
                    batch.add(shadow, *address, *len);
                    Kind::Refresh
                }
                Step::Run(work, _, _) | Step::Host(work, _) => {
                    let kind = match *step {
                        Step::Run(_, (to, written), (from, read)) => {
                            batch.settle(from, read, false);
                            batch.settle(to, written, true);
                            batch.title_written_at(to, written);
                            Kind::Run
                        }
                        Step::Host(_, (from, read)) => {
                            batch.settle(from, read, false);
                            Kind::Host
                        }
                        _ => unreachable!(),
                    };
                    #[cfg(feature = "diagnostics")]
                    let started = std::time::Instant::now();
                    work();
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Runs, 1);
                        add(C::RunUs, started.elapsed().as_micros() as u64);
                    }
                    kind
                }
                Step::GpuWrite(shadow, address, len, writer) => {
                    // The range's snapshot is taken from the host copy, which
                    // an earlier submission's GPU work may still be writing;
                    // the shaders start from the title's data.
                    batch.settle(*address, *len, false);
                    batch.flush();
                    let _snapshot = (!shadow.carded()).then(|| Phase::Snapshots.enter());
                    shadow.writing_by(batch.submission, *address, *len, *writer);
                    Kind::GpuWrite
                }
                // The copy reads host copies on the GPU, written back after.
                // Into a range an earlier submission's shaders write through
                // the same copy, it waits for that one: what the shaders
                // wrote is told from the copy's bytes, which it would change.
                Step::GpuCopy(shadow, address, len) => {
                    if shadow.carded() || crate::readmap::enabled() {
                        batch.settle(*address, *len, false);
                    } else {
                        batch.settle_shaders(shadow, *address, *len);
                    }
                    // Le bridage ou le titre peut avoir lu la destination avant la
                    // copie. Installer ses données initiales avant de la marquer occupée.
                    batch.flush_before_copy(shadow, *address, *len);
                    shadow.copying(batch.submission, *address, *len);
                    moved(Moved::GpuCopy, *len);
                    Kind::GpuCopy
                }
                Step::GpuWord(shadow, address, bytes) => {
                    if crate::readmap::enabled() || shadow.carded() {
                        batch.settle(*address, *bytes, false);
                        shadow.invalidate_mirror(*address, *bytes);
                    }
                    moved(Moved::ListWrite, *bytes);
                    Kind::GpuWord
                }
                Step::Texture(staging) => {
                    if batch.texture_due(staging) {
                        let (from, to) = staging.range();
                        if wide
                            && batch.read_conflict(from, to - from).is_none()
                            && !batch.has_staging_conflict(staging)
                        {
                            batch.texture_runs.push(staging.clone());
                        } else {
                            batch.refresh_textures();
                            batch.settle(from, to - from, false);
                            let _texture = Phase::Textures.enter();
                            batch.work.texture_bytes +=
                                staging.refresh(|| batch.settle_staging(staging)) as u64;
                        }
                    }
                    Kind::Texture
                }
                Step::Plane((from, len), fill) => {
                    batch.settle(*from, *len, false);
                    let _texture = Phase::Textures.enter();
                    fill();
                    batch.work.texture_bytes += *len as u64;
                    Kind::Texture
                }
                // Each run given as a `Refresh` of it, then each run written
                // as a `GpuWrite` of it: what the heap's listing gives this
                // submission (nothing if a list before gave it all, and
                // nothing has changed since).
                Step::Picked(step) => {
                    let (reads, writes) =
                        step.due_ranges(batch.submission, &batch.title_write_ranges);
                    for (shadow, from, to) in &reads {
                        let (address, len) = (*from as usize, (to - from) as usize);
                        #[cfg(feature = "diagnostics")]
                        {
                            add(C::Refreshes, 1);
                            add(C::RefreshBytes, len as u64);
                        }
                        batch.settle_refresh(shadow, address, len);
                        batch.add(shadow, address, len);
                    }
                    if !writes.is_empty() {
                        for (_, from, to) in &writes {
                            batch.settle(*from as usize, (to - from) as usize, false);
                        }
                        batch.flush();
                        for (shadow, from, to) in &writes {
                            let _snapshot = (!shadow.carded()).then(|| Phase::Snapshots.enter());
                            let (address, len) = (*from as usize, (to - from) as usize);
                            shadow.writing_by(batch.submission, address, len, step.writer());
                        }
                    }
                    Kind::Refresh
                }
            };
            batch.work.steps[kind as usize] += 1;
        }
        if wide {
            batch.fill_with(true);
            batch.refresh_textures();
        }
    }
}

/// A submission's pending refreshes, sorted by copy and address. Overlapping
/// and adjacent ranges merge. Done before a step that writes title memory or
/// gives a range to shaders, and before the lists run.
///
/// Also the textures brought up to date in the submission so far: most
/// lists of a world submission sample the same textures, and a texture in
/// physical-page memory is hashed whole each time it is checked. Checked
/// once, it is checked again only after a step that writes title memory.
#[derive(Default)]
struct Batch<'a> {
    work: preparation::Work,
    refreshes: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    /// Consecutive upload fills and scratch for checking their destinations.
    fills: Vec<(usize, usize, usize)>,
    fill_destinations: Vec<(usize, usize)>,
    /// Consecutive distinct staging buffers with no earlier in-flight users.
    texture_runs: Vec<std::sync::Arc<crate::texture::Staging>>,
    /// Staging buffers (by address) refreshed since the last title write.
    textures: std::collections::HashSet<usize>,
    /// The host, whose earlier submissions may still be in flight.
    host: Option<&'a Host>,
    /// The submission's number in the host copies (`shadow::submission`).
    submission: u64,
    /// Steps that wrote title memory so far ([`Batch::title_written`]):
    /// what was refreshed before one is refreshed again after it, and the
    /// title address ranges `[from, to)` written at each such step.
    title_writes: u64,
    title_write_ranges: Vec<(usize, usize)>,
}

impl Batch<'_> {
    /// Before a step reads (or, with `writing`, writes) title memory
    /// `[address, address + len)`: earlier submissions whose GPU work
    /// writes (or reads) it are finished first.
    #[track_caller]
    fn settle(&self, address: usize, len: usize, writing: bool) {
        if let Some(host) = self.host {
            host.settle_range(address, len, writing);
        }
    }

    /// Before a refresh of `[address, address + len)` of `shadow`: as
    /// [`Batch::settle`], but a submission whose shaders or copies write
    /// the range through the same host copy is waited for only if the
    /// title wrote a page of the overlap since the copy took it (the
    /// world's big heap is both refreshed and written on the GPU in most
    /// submissions).
    #[track_caller]
    fn settle_refresh(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let Some(host) = self.host else { return };
        let copy = std::sync::Arc::as_ptr(shadow) as usize;
        let to = address.saturating_add(len);
        if let Some(value) = host
            .timeline
            .refresh_conflict(copy, address, to, |a, b| shadow.stale(a, b))
        {
            host.settle(value);
        }
    }

    /// Before a GPU write to `[address, address + len)` of `shadow`: an
    /// earlier submission whose shaders write the range through it is
    /// finished first.
    #[track_caller]
    fn settle_shaders(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let Some(host) = self.host else { return };
        let copy = std::sync::Arc::as_ptr(shadow) as usize;
        let to = address.saturating_add(len);
        if let Some(value) = host.timeline.shader_conflict(copy, address, to) {
            host.settle(value);
        }
    }

    /// Before new bytes are detiled into a texture's staging buffer: an
    /// earlier submission whose GPU copy still reads it is finished first.
    #[track_caller]
    fn settle_staging(&self, staging: &crate::texture::Staging) {
        if let Some(host) = self.host
            && let Some(value) = host
                .timeline
                .staging_conflict(std::ptr::from_ref(staging) as usize)
        {
            host.settle(value);
        }
    }

    fn add(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>, address: usize, len: usize) {
        if len == 0 {
            return;
        }
        let end = address + len;
        let key = std::sync::Arc::as_ptr(shadow);
        let first = self
            .refreshes
            .partition_point(|(s, _, to)| (std::sync::Arc::as_ptr(s), *to) < (key, address));
        let pending = &self.refreshes[first..];
        let count = pending
            .partition_point(|(s, from, _)| (std::sync::Arc::as_ptr(s), *from) <= (key, end));
        if count == 0 {
            self.refreshes.insert(first, (shadow.clone(), address, end));
        } else {
            let to = end.max(pending[count - 1].2);
            let (_, from, last) = &mut self.refreshes[first];
            *from = address.min(*from);
            *last = to;
            self.refreshes.drain(first + 1..first + count);
        }
    }

    /// Whether `staging` is still to be refreshed in this submission (and
    /// notes it refreshed).
    fn texture_due<T>(&mut self, staging: &std::sync::Arc<T>) -> bool {
        self.textures
            .insert(std::sync::Arc::as_ptr(staging) as *const () as usize)
    }

    /// Whether an earlier in-flight submission still reads `staging`.
    fn has_staging_conflict(&self, staging: &crate::texture::Staging) -> bool {
        self.host.is_some_and(|host| {
            host.timeline
                .staging_conflict(std::ptr::from_ref(staging) as usize)
                .is_some()
        })
    }

    fn read_conflict(&self, address: usize, len: usize) -> Option<u64> {
        self.host?
            .timeline
            .conflict(address, address.saturating_add(len), false)
    }

    fn refresh_textures(&mut self) {
        if self.texture_runs.is_empty() {
            return;
        }
        let _texture = Phase::Textures.enter();
        let bytes = std::sync::atomic::AtomicUsize::new(0);
        crate::workers::each_wide(self.texture_runs.len(), &|i| {
            // Submission is serialized. The earlier staging users and
            // writers were ruled out before this run was queued.
            bytes.fetch_add(self.texture_runs[i].refresh(|| ()), Ordering::Relaxed);
        });
        self.texture_runs.clear();
        self.work.texture_bytes += bytes.into_inner() as u64;
    }

    /// Checks that no write in the batch can overlap another read or write.
    fn fills_disjoint(&mut self) -> bool {
        let destinations = &mut self.fill_destinations;
        destinations.clear();
        destinations.extend(
            self.fills
                .iter()
                .filter(|f| f.2 != 0)
                .map(|&(dst, _, len)| (dst, dst.saturating_add(len))),
        );
        destinations.sort_unstable();
        destinations.windows(2).all(|w| w[0].1 <= w[1].0)
            && self.fills.iter().all(|&(_, src, len)| {
                if len == 0 {
                    return true;
                }
                let i = destinations.partition_point(|&(_, end)| end <= src);
                destinations
                    .get(i)
                    .is_none_or(|&(dst, _)| src.saturating_add(len) <= dst)
            })
    }

    fn fill_with(&mut self, wide: bool) {
        const FEW: usize = 96;
        const PART: usize = 64;
        if self.fills.is_empty() {
            return;
        }
        let _fill = Phase::Fills.enter();
        let parallel = wide && self.fills.len() >= FEW && self.fills_disjoint();
        let fill = |&(dst, src, len): &(usize, usize, usize)| {
            // SAFETY: recording holds each mapped upload destination and
            // keeps it disjoint from its source. A parallel run also checked
            // destination ranges against one another and every source.
            unsafe { crate::shadow::fill_upload(dst as *mut u8, src, len) };
        };
        if parallel {
            let fills = &self.fills;
            crate::workers::each_wide(fills.len().div_ceil(PART), &|i| {
                let from = i * PART;
                fills[from..(from + PART).min(fills.len())]
                    .iter()
                    .for_each(fill);
            });
        } else {
            self.fills.iter().for_each(fill);
        }
        self.fills.clear();
    }

    /// Before a step that writes title memory: pending refreshes are done
    /// first, and textures are checked again after it.
    fn title_written(&mut self) {
        self.flush();
        self.textures.clear();
        self.title_writes += 1;
    }

    /// Like [`Batch::title_written`], recording `[address, address + len)`
    /// so later [`Step::Picked`] steps in the same submission re-refresh
    /// only picked runs touched by that write.
    fn title_written_at(&mut self, address: usize, len: usize) {
        self.title_written();
        self.title_write_ranges
            .push((address, address.saturating_add(len)));
    }

    fn flush_before_copy(
        &mut self,
        dst: &std::sync::Arc<crate::shadow::Shadow>,
        at: usize,
        len: usize,
    ) {
        let end = at.saturating_add(len);
        let key = std::sync::Arc::as_ptr(dst);
        let first = self
            .refreshes
            .partition_point(|(s, _, to)| (std::sync::Arc::as_ptr(s), *to) <= (key, at));
        let count = self.refreshes[first..]
            .partition_point(|(s, from, _)| (std::sync::Arc::as_ptr(s), *from) < (key, end));
        for (shadow, from, to) in self.refreshes.drain(first..first + count) {
            let _copy = Phase::Copies.enter();
            self.work.copy_bytes += Self::refresh_inner(&shadow, from, to) as u64;
        }
    }

    fn flush(&mut self) {
        self.flush_all(crate::workers::wide());
    }

    /// Flushes all pending disjoint shadow refreshes. When `wide` is enabled
    /// and multiple shadows are queued, ranges are grouped by shadow so no two
    /// worker threads ever contend on the same shadow's lock.
    fn flush_all(&mut self, wide: bool) {
        if self.refreshes.is_empty() {
            return;
        }
        let one_shadow = std::sync::Arc::ptr_eq(
            &self.refreshes.first().expect("nonempty").0,
            &self.refreshes.last().expect("nonempty").0,
        );
        if !wide || one_shadow {
            for (shadow, from, to) in self.refreshes.drain(..) {
                let _copy = Phase::Copies.enter();
                self.work.copy_bytes += Self::refresh_inner(&shadow, from, to) as u64;
            }
            return;
        }
        let _copy = Phase::Copies.enter();
        let refreshes = &self.refreshes;
        // `refreshes` is sorted by `(Arc::as_ptr(shadow), from)`, so each
        // contiguous run belongs to one shadow.
        let mut groups: Vec<std::ops::Range<usize>> = Vec::new();
        let mut start = 0;
        while start < refreshes.len() {
            let mut end = start + 1;
            while end < refreshes.len()
                && std::sync::Arc::ptr_eq(&refreshes[start].0, &refreshes[end].0)
            {
                end += 1;
            }
            groups.push(start..end);
            start = end;
        }
        let copied = std::sync::atomic::AtomicUsize::new(0);
        crate::workers::each_wide(groups.len(), &|g| {
            let mut mine = 0usize;
            for (shadow, from, to) in &refreshes[groups[g].clone()] {
                mine += Self::refresh_inner(shadow, *from, *to);
            }
            copied.fetch_add(mine, Ordering::Relaxed);
        });
        self.refreshes.clear();
        self.work.copy_bytes += copied.into_inner() as u64;
    }

    fn refresh_inner(shadow: &crate::shadow::Shadow, from: usize, to: usize) -> usize {
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let copied = shadow.update(from, to - from);
        #[cfg(feature = "diagnostics")]
        crate::copystat::range(from, to - from, started.elapsed().as_micros() as u64, 0);
        copied
    }
}

/// Copies title memory at submission. The ranges were checked against their
/// resources when recorded, but the title may since have unmapped pages of
/// them (tiled memory, page mappings), which read as zeros here; a title
/// destination (`title_dst`) that is not all mapped is left alone (but what
/// the kernel wrote of it before an unmapped page).
///
/// The source is read through the kernel (`ReadProcessMemory`), which
/// reports an unmapped page instead of faulting, and so does an unmapped
/// destination. Asking `VirtualQuery` first cost about 2 ms per copy in the
/// title's big pools (it walks the page tables to the region's end), which
/// made copies run at 100-200 MB/s; for title destinations it is asked only
/// when the kernel's copy fails (for a world frame's 6 000 small copies into
/// the heap view, asking first cost 70-185 ms, asking on failure 6-17 ms:
/// `watch::tests::guarded_copy_costs`).
pub(crate) fn copy_mapped(dst: *mut u8, src: *const u8, len: usize, title_dst: bool) {
    copy_mapped_inner(dst, src, len, title_dst, None);
}

/// Like [`copy_mapped`], returning the coalesced `(start, end)` byte ranges
/// relative to `src` that were read cleanly (excluding zero-filled holes).
pub(crate) fn copy_mapped_parts(
    dst: *mut u8,
    src: *const u8,
    len: usize,
    title_dst: bool,
) -> Vec<(usize, usize)> {
    let mut parts = Vec::new();
    copy_mapped_inner(dst, src, len, title_dst, Some(&mut parts));
    parts
}

fn push_mapped_part(parts: Option<&mut Vec<(usize, usize)>>, start: usize, len: usize) {
    if let Some(parts) = parts
        && len > 0
    {
        let end = start + len;
        match parts.last_mut() {
            Some(last) if last.1 == start => last.1 = end,
            _ => parts.push((start, end)),
        }
    }
}

fn copy_mapped_inner(
    dst: *mut u8,
    src: *const u8,
    len: usize,
    title_dst: bool,
    mut mapped: Option<&mut Vec<(usize, usize)>>,
) {
    #[cfg(feature = "diagnostics")]
    match crate::copystat::mode() {
        1 => {
            // SAFETY: timing only: the caller vouches for the source.
            unsafe { std::ptr::copy(src, dst, len) };
            push_mapped_part(mapped, 0, len);
            return;
        }
        2 => return,
        _ => {}
    }
    if len == 0 {
        return;
    }
    let _destination = (title_dst && crate::readmap::enabled())
        .then(|| crate::watch::kernel_writing_range(dst as usize, len, false));
    if title_dst && _destination.is_none() {
        // The kernel writes the destination: it must not be guarded.
        crate::watch::writing(dst as usize, len);
    }
    #[cfg(feature = "diagnostics")]
    let moved = std::time::Instant::now();
    let whole = read(src, dst, len);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(
        crate::copystat::C::MoveUs,
        moved.elapsed().as_micros() as u64,
    );
    if whole {
        push_mapped_part(mapped, 0, len);
        return;
    }
    if title_dst && !crate::texture::committed(dst, len) {
        return;
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::Fallbacks, 1);
    // Something is unmapped: page by page (64 KiB, then 4 KiB in a piece
    // that fails), zeros for what cannot be read.
    for_pieces(src as usize, len, 64 << 10, |at, n| {
        let offset = at - src as usize;
        let to = dst.wrapping_add(offset);
        if read(at as *const u8, to, n) {
            push_mapped_part(mapped.as_deref_mut(), offset, n);
        } else {
            for_pieces(at, n, 4 << 10, |at, n| {
                let offset = at - src as usize;
                let to = dst.wrapping_add(offset);
                if read(at as *const u8, to, n) {
                    push_mapped_part(mapped.as_deref_mut(), offset, n);
                } else {
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
                    // SAFETY: inside the destination, which is mapped.
                    unsafe { to.write_bytes(0, n) };
                }
            });
        }
    });
}

/// Reads title memory at `src` into `dst`; false if a page of it is not
/// readable.
pub(crate) fn read_into(src: *const u8, dst: &mut [u8]) -> bool {
    read(src, dst.as_mut_ptr(), dst.len())
}

/// Copies `len` bytes from `src` to `dst` unless a source page is not
/// readable.
fn read(src: *const u8, dst: *mut u8, len: usize) -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut done = 0;
    // SAFETY: the destination is mapped writable memory of `len` bytes; the
    // kernel checks the source.
    let copy = |done: &mut usize| unsafe {
        ReadProcessMemory(GetCurrentProcess(), src.cast(), dst.cast(), len, done) != 0
    };
    if copy(&mut done) {
        return true;
    }
    if !crate::readmap::enabled() {
        return false;
    }
    let Some(_pin) = crate::watch::pin_read(src as usize, len) else {
        return false;
    };
    copy(&mut done)
}

/// Copies host memory into title memory after a submission (what shaders
/// wrote), leaving pages that are not mapped alone.
pub(crate) fn write_mapped(dst: *mut u8, src: *const u8, len: usize) {
    use windows_sys::Win32::System::Diagnostics::Debug::WriteProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let _pin = crate::readmap::enabled()
        .then(|| crate::watch::kernel_writing_range(dst as usize, len, false));
    let write = |at: usize, n: usize| {
        let from = src.wrapping_add(at - dst as usize);
        let mut done = 0;
        // SAFETY: the source is our mapped host copy; the kernel checks the
        // destination.
        unsafe { WriteProcessMemory(GetCurrentProcess(), at as _, from.cast(), n, &mut done) != 0 }
    };
    if len == 0 || write(dst as usize, len) {
        return;
    }
    for_pieces(dst as usize, len, 64 << 10, |at, n| {
        if !write(at, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                write(at, n);
            });
        }
    });
}

/// For the frame-rate line: the video memory the adapter gives this
/// process and what it uses, local (the card) and non-local (system memory
/// the adapter maps), and the process's own memory and page faults since
/// the last line. It changes nothing.
fn log_memory(device: &ID3D12Device) {
    let mib = |b: u64| b >> 20;
    let (lb, lu) = crate::dxgi::video_memory(device).unwrap_or_default();
    let (nb, nu) = crate::dxgi::non_local_memory(device).unwrap_or_default();
    let (ws, private, faults) = process_memory().unwrap_or_default();
    static FAULTS: AtomicU64 = AtomicU64::new(0);
    let new_faults = faults.saturating_sub(FAULTS.swap(faults, Ordering::Relaxed));
    tracing::info!(
        "memory: card {} of {} MiB, system for the GPU {} of {} MiB; process working set {} MiB, \
         private {} MiB, {new_faults} page faults since the last line; snapshot mirrors {} MiB \
         ({} MiB of pages written); \
         {} spare list scratch chunks",
        mib(lu),
        mib(lb),
        mib(nu),
        mib(nb),
        mib(ws),
        mib(private),
        mib(crate::shadow::MIRRORS.load(Ordering::Relaxed)),
        mib(crate::shadow::MIRRORS_WRITTEN.load(Ordering::Relaxed)),
        spare_scratch_chunks(),
    );
    tracing::info!("list scratch: {}", scratch_use());
    tracing::info!("title memory windows: {}", crate::shadow::loose_use());
}

/// This process's working set, private bytes and page faults so far.
fn process_memory() -> Option<(u64, u64, u64)> {
    use windows_sys::Win32::System::ProcessStatus::{
        GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS_EX {
        cb: std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process; the structure is ours and sized as it says.
    let ok = unsafe {
        GetProcessMemoryInfo(
            GetCurrentProcess(),
            (&raw mut c).cast::<PROCESS_MEMORY_COUNTERS>(),
            c.cb,
        )
    };
    (ok != 0).then(|| {
        (
            c.WorkingSetSize as u64,
            c.PrivateUsage as u64,
            u64::from(c.PageFaultCount),
        )
    })
}

/// Calls `f` with the pieces of `[start, start + len)` cut at multiples of
/// `step`.
fn for_pieces(start: usize, len: usize, step: usize, mut f: impl FnMut(usize, usize)) {
    let end = start + len;
    let mut at = start;
    while at < end {
        let next = ((at / step + 1) * step).min(end);
        f(at, next - at);
        at = next;
    }
}

impl Host {
    /// A buffer in the upload heap, mapped for writing: placed in a heap
    /// reserved ahead ([`buffers`]), else committed.
    #[track_caller]
    pub(crate) fn upload_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        if let Some(placed) = buffers::buffer(buffers::Kind::Upload, size) {
            return Some(placed);
        }
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
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
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of an upload buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no upload buffer");
                None
            }
        }
    }

    /// A buffer of `size` bytes in the readback heap, in `COPY_DEST`.
    pub(crate) fn read_back_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
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
        // SAFETY: plain creation of a readback buffer.
        let made = unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut buffer,
            )
        };
        match made {
            Ok(()) => buffer,
            Err(e) => {
                tracing::warn!(error = %e, size, "no readback buffer");
                None
            }
        }
    }
}

/// The swap chain formats a presented texture can be copied into: its own
/// family's.
fn swap_format(format: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    Some(match format.0 {
        27..=32 => DXGI_FORMAT_R8G8B8A8_UNORM,
        87 | 90 | 91 => DXGI_FORMAT_B8G8R8A8_UNORM,
        23..=25 => DXGI_FORMAT_R10G10B10A2_UNORM,
        9..=14 => DXGI_FORMAT_R16G16B16A16_FLOAT,
        _ => return None,
    })
}

const BUFFERS: u32 = 2;

struct Presenter {
    swap_chain: IDXGISwapChain3,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
    copy: HostList,
    hwnd: HWND,
    meter: Meter,
    /// Tearing swap-chain and present state (latched off on refusal; swap-chain
    /// creation flags stay fixed for `ResizeBuffers`).
    present_mode: PresentMode,
    /// With frames in flight (diagnostics builds): the readback buffer of
    /// the last frame's black-frame probe, read once that frame finished.
    #[cfg(feature = "diagnostics")]
    probe: Option<ID3D12Resource>,
}

/// Swap-chain tearing capability and one-way present fallback state.
struct PresentMode {
    /// Whether the swap chain was created with `DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING`.
    /// Must remain unchanged across `ResizeBuffers`.
    swap_chain_tearing: bool,
    /// Whether `Present(0, ...)` passes `DXGI_PRESENT_ALLOW_TEARING`; latched
    /// off for the remainder of the run if a tearing present is refused.
    presents_tearing: bool,
    /// Whether a non-tearing present failure has already been logged.
    present_refused: bool,
}

impl PresentMode {
    fn new(tearing: bool) -> Self {
        Self {
            swap_chain_tearing: tearing,
            presents_tearing: tearing,
            present_refused: false,
        }
    }

    fn swap_chain_flags(&self) -> DXGI_SWAP_CHAIN_FLAG {
        if self.swap_chain_tearing {
            DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING
        } else {
            DXGI_SWAP_CHAIN_FLAG(0)
        }
    }

    fn present(
        &mut self,
        frame: u64,
        mut call: impl FnMut(DXGI_PRESENT) -> windows::core::HRESULT,
    ) -> windows::core::HRESULT {
        let flags = if self.presents_tearing {
            DXGI_PRESENT_ALLOW_TEARING
        } else {
            DXGI_PRESENT(0)
        };
        let shown = call(flags);
        if shown.is_err() && self.presents_tearing {
            tracing::warn!(
                code = %shown,
                frame,
                "tearing present refused; falling back to non-tearing presents for the rest of the run"
            );
            self.presents_tearing = false;
            let fallback = call(DXGI_PRESENT(0));
            if fallback.is_err() && !self.present_refused {
                self.present_refused = true;
                tracing::warn!(code = %fallback, frame, "present refused (logged once)");
            }
            return fallback;
        }
        if shown.is_err() && !self.present_refused {
            self.present_refused = true;
            tracing::warn!(code = %shown, frame, "present refused (logged once)");
        }
        shown
    }
}

fn tearing_enabled(env: Option<&str>) -> bool {
    env.is_none_or(|v| v.trim() != "0")
}

/// Whether DXGI allows tearing presents on `factory` (`DXGI_FEATURE_PRESENT_ALLOW_TEARING`).
fn query_tearing_support(factory: &IDXGIFactory2) -> bool {
    let Ok(factory) = factory.cast::<IDXGIFactory5>() else {
        return false;
    };
    let mut allowed = windows::core::BOOL(0);
    // SAFETY: `CheckFeatureSupport` writes a `BOOL`-sized value for
    // `DXGI_FEATURE_PRESENT_ALLOW_TEARING`.
    let asked = unsafe {
        factory.CheckFeatureSupport(
            DXGI_FEATURE_PRESENT_ALLOW_TEARING,
            (&raw mut allowed).cast(),
            size_of_val(&allowed) as u32,
        )
    };
    asked.is_ok() && allowed.as_bool()
}

/// Whether the swap chain should allow tearing presents (`XTP_TEARING=0`: off).
/// Without it, a focused independent-flip window with two buffers waits for the
/// next display refresh on each present, rounding frame times up to whole
/// refreshes. On variable-refresh displays nothing tears; on fixed-refresh
/// displays frames show as soon as presented.
fn tearing(factory: &IDXGIFactory2) -> bool {
    tearing_enabled(std::env::var("XTP_TEARING").ok().as_deref()) && query_tearing_support(factory)
}

/// How often the frame-rate line is logged: every 5 s, or every
/// `XTP_FRAME_LINE` seconds (1 to 60). Each line gives the last second or
/// so; with 1, every second of a short world is in the log (a Series S
/// world lasts a few seconds before the title's crash, task 36f1e12d).
fn line_every() -> std::time::Duration {
    static EVERY: std::sync::OnceLock<std::time::Duration> = std::sync::OnceLock::new();
    *EVERY.get_or_init(|| {
        let seconds = std::env::var("XTP_FRAME_LINE")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map_or(5, |s| s.clamp(1, 60));
        std::time::Duration::from_secs(seconds)
    })
}

/// The frame rate, shown in the window's caption every second and logged
/// every few seconds ([`line_every`]).
struct Meter {
    since: std::time::Instant,
    frames: u32,
    #[cfg(feature = "diagnostics")]
    copy_stats_frames: u64,
    drawn: u64,
    logged: std::time::Instant,
    /// The caption the window had, once read.
    caption: Option<Vec<u16>>,
    /// The last present, and the longest time between two this second.
    last: std::time::Instant,
    longest: std::time::Duration,
}

impl Meter {
    fn new() -> Self {
        let now = std::time::Instant::now();
        Meter {
            since: now,
            frames: 0,
            #[cfg(feature = "diagnostics")]
            copy_stats_frames: 0,
            drawn: crate::draw::drawn(),
            logged: now,
            caption: None,
            last: now,
            longest: std::time::Duration::ZERO,
        }
    }

    /// Counts a presented frame.
    fn frame(&mut self, hwnd: HWND) {
        // The frames' mode, and the A/B phase they ran in, before the clock
        // moves on for the next ones (`xtp_core::ab`).
        let ahead = if frames::ahead() { "on" } else { "off" };
        let ab = xtp_core::ab::present(self.since);
        #[cfg(feature = "diagnostics")]
        tracing::debug!(target: "xtp_gpu::frame", token = crate::frame::current(), "present");
        self.frames += 1;
        #[cfg(feature = "diagnostics")]
        {
            self.copy_stats_frames += 1;
        }
        let now = std::time::Instant::now();
        self.longest = self.longest.max(now - self.last);
        self.last = now;
        let elapsed = self.since.elapsed();
        if elapsed < std::time::Duration::from_secs(1) {
            return;
        }
        crate::pipeline::trim_library();
        let drawn = crate::draw::drawn();
        let frames = f64::from(self.frames);
        let fps = frames / elapsed.as_secs_f64();
        let ms = elapsed.as_secs_f64() * 1000.0 / frames;
        let draws = (drawn - self.drawn) as f64 / frames;
        (self.since, self.frames, self.drawn) = (std::time::Instant::now(), 0, drawn);
        // Of a frame, ms preparing submissions, the GPU running them, the
        // title's threads waiting for them, and of that the waits at flips
        // and presents.
        let [gpu, waited, at_ends] = [GPU_US, WAITED_US, FRAME_END_US]
            .map(|i| SUBMISSION_US[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames);
        let preparation = preparation::take();
        let per_frame_ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0 / frames;
        #[cfg(feature = "diagnostics")]
        let guard_wait_ms = per_frame_ms(crate::watch::take_guard_wait());
        let cpu = per_frame_ms(preparation.total);
        let [
            copy_ms,
            fill_ms,
            texture_ms,
            snapshot_ms,
            watch_ms,
            steps_ms,
            other_ms,
        ] = preparation.times.map(per_frame_ms);
        let work = preparation.work;
        let [copy_mb, fill_mb, texture_mb] = [work.copy_bytes, work.fill_bytes, work.texture_bytes]
            .map(|bytes| bytes as f64 / 1e6 / frames);
        let [back_ms, back_mb] =
            [0, 1].map(|i| COPY_BACK[i].swap(0, Ordering::Relaxed) as f64 / [1e3, 1e6][i] / frames);
        let dropped = crate::draw::take_dropped() as f64 / frames;
        let [cbv_copies, cbv_hits] = crate::draw::take_cbv_copies().map(|n| n as f64 / frames);
        let indirect = crate::list::take_indirect() as f64 / frames;
        let heap_sync = crate::descriptor::take_synced(frames);
        // The title's threads spend `cpu` and `waited` in submissions; the
        // GPU and the copying back run beside them for the rest.
        let outside_ms = outside(ms, cpu + waited);
        let overlap_ms = overlapped(gpu, back_ms, waited);
        let longest_frame = std::mem::take(&mut self.longest).as_secs_f64() * 1000.0;
        let title = std::mem::replace(
            &mut *TITLE.lock().unwrap_or_else(|e| e.into_inner()),
            Submissions::new(),
        )
        .per_frame(frames);
        let [
            (gpu_copies, gpu_copy_mb),
            (gpu_writes, _),
            (cpu_copies, cpu_copy_mb),
        ] = MOVED.each_ref().map(|[count, bytes]| {
            let [count, bytes] = [count, bytes].map(|c| c.swap(0, Ordering::Relaxed) as f64);
            (count / frames, bytes / 1e6 / frames)
        });
        // Over the same frames as the counts above.
        #[cfg(feature = "diagnostics")]
        let drops = crate::pipeline::take_drops();
        #[cfg(feature = "diagnostics")]
        let waits = WAITS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take(frames, 8);
        #[cfg(feature = "diagnostics")]
        let meshes = crate::meshstat::take();
        #[cfg(feature = "diagnostics")]
        twins::stale::frames(frames);
        if self.logged.elapsed() >= line_every() {
            self.logged = std::time::Instant::now();
            crate::loopstat::line();
            #[cfg(feature = "diagnostics")]
            crate::list::cuts::log();
            #[cfg(feature = "diagnostics")]
            tracing::info!(
                pipelines = crate::pipeline::host::BOUNDED_PIPELINES.load(Ordering::Relaxed),
                lists = crate::list::BOUNDED_LISTS.load(Ordering::Relaxed),
                "title depth bounds"
            );
            tracing::info!(
                "{ab}{fps:.1} fps, {ms:.1} ms/frame ({cpu:.3} preparing: copies {copy_ms:.3} ms \
                 {copy_mb:.1} MB, textures {texture_ms:.3} ms {texture_mb:.1} MB; {gpu:.1} on the \
                 GPU; {back_ms:.1} copying back {back_mb:.1} MB; {waited:.1} waiting for them \
                 ({at_ends:.1} at flips and presents), {overlap_ms:.1} overlapped; {outside_ms:.1} \
                 outside submissions; frames in flight {ahead}, the longest frame \
                 {longest_frame:.0} ms), \
                 {submissions:.1} submissions/frame of {lists:.1} lists, the \
                 longest {longest:.1} ms ({longest_gpu:.1} on the GPU, {longest_lists} lists), \
                 buffer copies {gpu_copies:.0} on the GPU {gpu_copy_mb:.1} MB and \
                 {cpu_copies:.0} on the CPU {cpu_copy_mb:.1} MB, {gpu_writes:.0} in-list writes \
                 on the GPU, {cbv_copies:.1} GPU CBV copies/frame ({cbv_hits:.1} cache hits/frame), \
                 {draws:.0} draws/frame, dropped {dropped:.0} without a host \
                 pipeline ({making} being made) and {indirect:.0} indirect",
                submissions = title.submissions,
                lists = title.lists,
                longest = title.longest_ms,
                longest_gpu = title.longest_gpu_ms,
                longest_lists = title.longest_lists,
                making = crate::pipeline::making()
            );
            tracing::info!("snapshots per frame: {snapshot_ms:.3} ms");
            #[cfg(feature = "diagnostics")]
            tracing::info!(
                calls = SPLITS[0].load(Ordering::Relaxed),
                waits = SPLITS[1].load(Ordering::Relaxed),
                wait_us = SPLITS[2].load(Ordering::Relaxed),
                "submission split totals"
            );
            tracing::info!("guards and write watch per frame: {watch_ms:.3} ms");
            #[cfg(feature = "diagnostics")]
            tracing::info!(
                "GUARDS contention outside preparation per frame: {guard_wait_ms:.3} ms"
            );
            let merged = work.merged_fills as f64 / frames;
            tracing::info!("fills per frame: {fill_ms:.3} ms, {fill_mb:.2} MB, {merged:.2} merged");
            let steps = work.steps.iter().sum::<u64>() as f64 / frames;
            let [
                copies,
                fills,
                refreshes,
                runs,
                hosts,
                writes,
                gpu_copies,
                words,
                textures,
            ] = work.steps.map(|count| count as f64 / frames);
            tracing::info!(
                "steps per frame: {steps:.1} ({copies:.1} copy, {fills:.1} fill, {refreshes:.1} refresh, \
                 {runs:.1} run, {hosts:.1} host, {writes:.1} GPU write, {gpu_copies:.1} GPU copy, \
                 {words:.1} GPU word, {textures:.1} texture), {steps_ms:.3} ms in remaining step work, \
                 {other_ms:.3} ms in other preparation"
            );
            heap_sync.log();
            crate::descriptor::host::log_picked_frame(frames);
            #[cfg(feature = "diagnostics")]
            crate::indirect::log_counts();
            #[cfg(feature = "diagnostics")]
            twins::stale::log();
            crate::device::log_frame();
            if let Some(pools) = buffers::summary() {
                tracing::info!("buffer heaps: {pools}");
            }
            if let Some(host) = host() {
                log_memory(&host.device);
            }
            #[cfg(feature = "diagnostics")]
            {
                crate::copystat::log(std::mem::take(&mut self.copy_stats_frames) as f64);
                if !waits.is_empty() {
                    tracing::info!(
                        "waiting for submissions, by where it was asked: {}",
                        waits.join("; ")
                    );
                }
                for (why, n) in drops.iter().take(40) {
                    tracing::info!(
                        "dropped {:.1}/frame without a host pipeline: {why}",
                        *n as f64 / frames
                    );
                }
                crate::meshstat::log(meshes, frames);
            }
        }
        if self.caption.is_none() {
            // The window's own text, read without a message.
            let mut buffer = vec![0u16; 512];
            // SAFETY: a window handle and a buffer of that length.
            let n = unsafe { InternalGetWindowText(hwnd, &mut buffer) };
            buffer.truncate(n.max(0) as usize);
            self.caption = Some(buffer);
        }
        let Some(caption) = &self.caption else { return };
        let text = format!(" | XTP {fps:.0} fps {ms:.1} ms/frame {draws:.0} draws/frame");
        let mut text: Vec<u16> = caption.iter().copied().chain(text.encode_utf16()).collect();
        text.push(0);
        // Setting the text waits for the window's thread (a title thread,
        // which may not pump for a while), so a thread of ours does it and
        // presenting never waits; it takes the newest text when free.
        static PENDING: Mutex<Option<(usize, Vec<u16>)>> = Mutex::new(None);
        static SETTER: std::sync::Once = std::sync::Once::new();
        *PENDING.lock().unwrap_or_else(|e| e.into_inner()) = Some((hwnd.0 as usize, text));
        SETTER.call_once(|| {
            let _ = std::thread::Builder::new()
                .name("xtp-caption".into())
                .spawn(|| {
                    loop {
                        let next = PENDING.lock().unwrap_or_else(|e| e.into_inner()).take();
                        match next {
                            // SAFETY: a window handle and a NUL-terminated text.
                            Some((hwnd, text)) => unsafe {
                                let _ = SetWindowTextW(HWND(hwnd as *mut _), PCWSTR(text.as_ptr()));
                            },
                            None => std::thread::sleep(std::time::Duration::from_millis(100)),
                        }
                    }
                });
        });
    }
}

impl Host {
    /// Shows `texture`, which is in `state` (a host state) and stays in it,
    /// as frame `frame` (saved if captures are on). `None` if nothing was
    /// shown; else, in diagnostics builds, whether the frame was black: with
    /// frames in flight, whether the frame before it was (this one's copy
    /// runs later; `XTP_FRAMES_IN_FLIGHT=0` reports each frame at its own
    /// present).
    ///
    /// The frame presented last is finished first ([`Host::flip`]): its
    /// copy list, back buffer and probe are free then. Then every
    /// submission is finished before returning, unless frames in flight
    /// let this one run on while the title prepares the next; a captured
    /// frame is finished, to be read back.
    pub(crate) fn present(&self, texture: &ID3D12Resource, state: u32, frame: u64) -> Option<bool> {
        // SAFETY: a live resource.
        let desc = unsafe { texture.GetDesc() };
        let Some(format) = swap_format(desc.Format) else {
            tracing::warn!(format = desc.Format.0, "cannot present this format");
            return None;
        };
        let (width, height) = (desc.Width as u32, desc.Height);
        let mut presenter = self.presenter.lock().unwrap_or_else(|e| e.into_inner());
        if presenter.is_none() {
            *presenter = match self.presenter(width, height, format) {
                Ok(p) => Some(p),
                Err(e) => {
                    tracing::warn!(error = %e, "no swap chain");
                    return None;
                }
            };
        }
        let p = presenter.as_mut()?;
        if (p.width, p.height, p.format) != (width, height, format) {
            // The back buffers may not be in use.
            self.drain();
            // SAFETY: no back buffer is held; every submission has finished.
            let resized = unsafe {
                p.swap_chain.ResizeBuffers(
                    BUFFERS,
                    width,
                    height,
                    format,
                    p.present_mode.swap_chain_flags(),
                )
            };
            match resized {
                Ok(()) => (p.width, p.height, p.format) = (width, height, format),
                Err(e) => {
                    tracing::warn!(error = %e, width, height, "swap chain not resized");
                    return None;
                }
            }
        }
        let state = D3D12_RESOURCE_STATES(state as i32);
        let ahead = frames::ahead();
        self.flip();
        // The frame before's probe: its copy has run.
        #[cfg(feature = "diagnostics")]
        let before = p.probe.take().map(|b| crate::capture::black(&b));
        // SAFETY: our swap chain, list and a live source texture.
        unsafe {
            let index = p.swap_chain.GetCurrentBackBufferIndex();
            let Ok(back) = p.swap_chain.GetBuffer::<ID3D12Resource>(index) else {
                return None;
            };
            p.copy.reset();
            let list = &p.copy.list;
            list.ResourceBarrier(&[
                transition(texture, state, D3D12_RESOURCE_STATE_COPY_SOURCE),
                transition(
                    &back,
                    D3D12_RESOURCE_STATE_PRESENT,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                ),
            ]);
            list.CopyResource(&back, texture);
            #[cfg(feature = "diagnostics")]
            let capture = crate::capture::wanted(frame)
                .then(|| crate::capture::read_back(&self.device, list, texture))
                .flatten();
            #[cfg(feature = "diagnostics")]
            let probe = crate::capture::probe(&self.device, list, texture);
            #[cfg(not(feature = "diagnostics"))]
            let _ = frame;
            list.ResourceBarrier(&[
                transition(texture, D3D12_RESOURCE_STATE_COPY_SOURCE, state),
                transition(
                    &back,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_PRESENT,
                ),
            ]);
            // XTP_SHADER_LOOP_COUNT: the frame's counters, copied after its lists.
            crate::loopstat::record(self, list);
            p.copy.close();
            self.execute(&[&p.copy]);
            // XTP_SHADER_LOOP_COUNT: the frame's stopped lanes, once its
            // submissions have finished.
            crate::loopstat::presented(self, frame);
            let _ = p
                .present_mode
                .present(frame, |flags| p.swap_chain.Present(0, flags));
            // A captured frame is finished, to be read back; its probe is
            // still read at the next present, as every probe with frames
            // in flight (each frame reported once, one present late).
            #[cfg(feature = "diagnostics")]
            let runs_on = ahead && capture.is_none();
            #[cfg(not(feature = "diagnostics"))]
            let runs_on = ahead;
            self.presented(&p.copy, runs_on);
            #[cfg(feature = "diagnostics")]
            if let Some((buffer, footprint)) = capture {
                crate::capture::save_read_back(frame, &buffer, &footprint);
            }
            p.meter.frame(p.hwnd);
            #[cfg(feature = "diagnostics")]
            return Some(if ahead {
                p.probe = probe;
                before.unwrap_or(false)
            } else {
                probe.is_some_and(|b| crate::capture::black(&b))
            });
            #[cfg(not(feature = "diagnostics"))]
            Some(false)
        }
    }

    fn presenter(
        &self,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
    ) -> windows::core::Result<Presenter> {
        let hwnd = match title_window() {
            Some(hwnd) => hwnd,
            None => window(width, height)?,
        };
        // SAFETY: plain object creation for our window.
        unsafe {
            let factory: IDXGIFactory2 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
            let present_mode = PresentMode::new(tearing(&factory));
            tracing::info!(
                tearing = present_mode.swap_chain_tearing,
                "swap chain: allow tearing presents when supported (XTP_TEARING=0: off)"
            );
            let desc = DXGI_SWAP_CHAIN_DESC1 {
                Width: width,
                Height: height,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                BufferCount: BUFFERS,
                SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                Flags: present_mode.swap_chain_flags().0 as u32,
                ..Default::default()
            };
            let swap_chain = factory
                .CreateSwapChainForHwnd(&self.queue, hwnd, &desc, None, None)?
                .cast::<IDXGISwapChain3>()?;
            let copy = self.new_list()?;
            copy.close();
            tracing::info!(width, height, format = format.0, "presenting to a window");
            Ok(Presenter {
                swap_chain,
                width,
                height,
                format,
                copy,
                hwnd,
                meter: Meter::new(),
                present_mode,
                #[cfg(feature = "diagnostics")]
                probe: None,
            })
        }
    }
}

/// A visible top-level window of this process: the one the title made, if
/// it made one.
fn title_window() -> Option<HWND> {
    extern "system" fn visit(hwnd: HWND, found: LPARAM) -> windows::core::BOOL {
        let mut process = 0;
        // SAFETY: a window handle from the enumeration; `found` is ours.
        unsafe {
            GetWindowThreadProcessId(hwnd, Some(&mut process));
            if process == windows::Win32::System::Threading::GetCurrentProcessId()
                && IsWindowVisible(hwnd).as_bool()
            {
                *(found.0 as *mut Option<HWND>) = Some(hwnd);
                return false.into();
            }
        }
        true.into()
    }
    let mut found: Option<HWND> = None;
    // SAFETY: the callback only writes `found`, which outlives the call.
    let _ = unsafe { EnumWindows(Some(visit), LPARAM(&raw mut found as isize)) };
    found
}

/// `state` without the bits the host texture's flags forbid: the console
/// lets the title name states its own resources allow (unordered access on
/// a depth target, whose host copy cannot have both).
pub(crate) fn legal_state(texture: &ID3D12Resource, state: u32) -> u32 {
    // SAFETY: a plain query.
    legal_for(unsafe { texture.GetDesc() }.Flags.0 as u32, state)
}

/// [`legal_state`] for a texture with resource flags `flags`.
pub(crate) fn legal_for(flags: u32, state: u32) -> u32 {
    // (flag, set or clear forbids, state bits): render target, unordered
    // access, depth, denied shader resource.
    const FORBIDDEN: [(u32, bool, u32); 4] = [
        (0x1, false, 0x4),
        (0x4, false, 0x8),
        (0x2, false, 0x10 | 0x20),
        (0x8, true, 0x40 | 0x80),
    ];
    FORBIDDEN
        .iter()
        .filter(|&&(flag, deny, _)| (flags & flag != 0) == deny)
        .fold(state, |s, &(_, _, bits)| s & !bits)
}

impl Host {
    /// A buffer in system memory that the CPU maps (write-back, so reading
    /// it is fast too) and shaders read and write: for host copies of title
    /// memory that shaders write, which upload memory cannot be. Starts in
    /// the common state (buffers move out of it implicitly), zeroed. Placed
    /// in a heap reserved ahead ([`buffers`]), else committed.
    #[track_caller]
    pub(crate) fn shared_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        if let Some(placed) = buffers::buffer(buffers::Kind::System, size) {
            return Some(placed);
        }
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
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
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a CPU-visible buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no shared buffer");
                None
            }
        }
    }
}

/// For adapters without device-removed data (the software one): the last
/// commands each list of the failed submission noted (diagnostics, with
/// `XTP_D3D12_DRED`).
fn last_notes(runs: &[Recorded]) -> String {
    #[cfg(feature = "diagnostics")]
    {
        let mut text = String::new();
        for (i, r) in runs.iter().enumerate() {
            let notes = &r.notes;
            for (n, _) in notes.iter().rev().take(6).rev() {
                text.push_str(&format!(
                    "; list {i} of {}, of its {}: {n}",
                    runs.len(),
                    notes.len()
                ));
            }
        }
        text
    }
    #[cfg(not(feature = "diagnostics"))]
    {
        let _ = runs;
        String::new()
    }
}

/// Where the faulting address `va` lies against a range `(start, bytes)` a
/// command reads (diagnostics).
#[cfg(feature = "diagnostics")]
fn fault_in(va: u64, (start, bytes): (u64, u64)) -> String {
    let end = start + bytes;
    if va < start {
        format!(" (fault {:#x} below its source)", start - va)
    } else if va < end {
        format!(" (fault at source +{:#x})", va - start)
    } else {
        format!(" (fault {:#x} past its source's end)", va - end)
    }
}

/// The commands that may have been running where a list stopped at
/// command `done`: those between the barriers around it (work between
/// barriers overlaps on the GPU), by kind, and what the markers just
/// before them say (`HostList::note_op`: diagnostics builds with
/// device-removed data on).
fn in_flight(
    history: &[D3D12_AUTO_BREADCRUMB_OP],
    markers: &[(u32, String)],
    done: usize,
) -> String {
    let barrier = |op: &D3D12_AUTO_BREADCRUMB_OP| *op == D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER;
    let from = history[..done]
        .iter()
        .rposition(barrier)
        .map_or(0, |b| b + 1);
    let to = history[done..]
        .iter()
        .position(barrier)
        .map_or(history.len(), |b| (done + b).max(done + 1));
    let mut text = format!(", commands {from}..{to} between its barriers:");
    let mut shown = 0;
    for i in from..to {
        let op = history[i];
        if op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER || shown == 12 {
            continue;
        }
        shown += 1;
        let marker = i
            .checked_sub(1)
            .filter(|&m| history[m] == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
            .and_then(|m| markers.iter().find(|(at, _)| *at as usize == m))
            .map_or("no marker", |(_, s)| s.as_str());
        text.push_str(&format!("; command {i} (op {}): {marker}", op.0));
    }
    // DRED's count may lag the GPU (xcl#1592: a fault in the second of two
    // dispatches left the first not done): the next noted commands too.
    let after: Vec<String> = (to..history.len().saturating_sub(1))
        .filter(|&m| history[m] == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
        .filter_map(|m| {
            let (_, note) = markers.iter().find(|(at, _)| *at as usize == m)?;
            Some(format!(
                "command {} (op {}): {note}",
                m + 1,
                history[m + 1].0
            ))
        })
        .take(3)
        .collect();
    if !after.is_empty() {
        text.push_str(&format!(
            "; after them (the count may lag the GPU): {}",
            after.join("; ")
        ));
    }
    text
}

/// An unordered access barrier on every resource: earlier shader writes
/// finish before later work reads or writes.
pub(crate) fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    }
}

pub(crate) fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// A window of the given client size, owned by its own message thread.
fn window(width: u32, height: u32) -> windows::core::Result<HWND> {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("xtp-window".into())
        .spawn(move || {
            let made = make_window(width, height);
            let ok = made.is_ok();
            let _ = sender.send(made.map(|h| h.0 as usize));
            if ok {
                pump();
            }
        })
        .map_err(|_| windows::core::Error::empty())?;
    let hwnd = receiver
        .recv()
        .map_err(|_| windows::core::Error::empty())??;
    Ok(HWND(hwnd as *mut _))
}

extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // SAFETY: the default handling for our window.
    unsafe { DefWindowProcW(hwnd, message, wparam, lparam) }
}

fn make_window(width: u32, height: u32) -> windows::core::Result<HWND> {
    const CLASS: PCWSTR = w!("XTP");
    // SAFETY: registers our class once and creates a window of it.
    unsafe {
        let instance = GetModuleHandleW(None)?;
        let class = WNDCLASSEXW {
            cbSize: size_of::<WNDCLASSEXW>() as u32,
            lpfnWndProc: Some(window_proc),
            hInstance: instance.into(),
            lpszClassName: CLASS,
            hCursor: LoadCursorW(None, IDC_ARROW)?,
            ..Default::default()
        };
        RegisterClassExW(&class);
        let mut rect = RECT {
            left: 0,
            top: 0,
            right: width as i32,
            bottom: height as i32,
        };
        AdjustWindowRect(&mut rect, WS_OVERLAPPEDWINDOW, false)?;
        CreateWindowExW(
            WINDOW_EX_STYLE(0),
            CLASS,
            w!("XTP"),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            rect.right - rect.left,
            rect.bottom - rect.top,
            None,
            None,
            Some(instance.into()),
            None,
        )
    }
}

fn pump() {
    let mut message = MSG::default();
    // SAFETY: the standard message loop of this thread's window.
    unsafe {
        while GetMessageW(&mut message, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
}

#[cfg(test)]
mod batch {
    // Batch::texture_due lives here; `mod tests` is already taken at the top of this file
    use super::Batch;
    use std::sync::Arc;

    #[test]
    fn a_texture_is_checked_once_per_submission() {
        let mut batch = Batch::default();
        let (a, b) = (Arc::new([0u8; 16]), Arc::new([0u8; 16]));
        // The first list sampling each texture checks it ...
        assert!(batch.texture_due(&a));
        assert!(batch.texture_due(&b));
        // ... later lists of the same submission do not.
        assert!(!batch.texture_due(&a));
        assert!(!batch.texture_due(&a.clone()));
        assert!(!batch.texture_due(&b));
    }

    #[test]
    fn title_writes_make_textures_due_again() {
        let mut batch = Batch::default();
        let a = Arc::new(1u32);
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A copy into title memory or work that may write it may have
        // changed the texture's memory.
        batch.title_written();
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A fresh submission starts over.
        let mut next = Batch::default();
        assert!(next.texture_due(&a));
    }

    /// The hashing a submission does for textures in physical-page memory,
    /// with every list checking the textures it samples (before) and with
    /// each checked once (after). Run with
    /// `cargo test --release -p xtp-gpu -- --ignored --nocapture texture_checks`.
    #[test]
    #[ignore = "benchmark"]
    fn texture_checks_per_submission() {
        const TEXTURES: usize = 48;
        const SIZE: usize = 4 << 20;
        const LISTS: usize = 12;
        let textures: Vec<Arc<Vec<u8>>> = (0..TEXTURES)
            .map(|t| Arc::new((0..SIZE).map(|i| (i * 31 + t) as u8).collect()))
            .collect();
        let submit = |dedupe: bool| {
            let mut batch = Batch::default();
            let mut hashed = 0usize;
            let started = std::time::Instant::now();
            for _ in 0..LISTS {
                for t in &textures {
                    if !dedupe || batch.texture_due(t) {
                        std::hint::black_box(blake3::hash(t));
                        hashed += 1;
                    }
                }
            }
            (started.elapsed(), hashed)
        };
        submit(false); // warm up
        let (before, n_before) = submit(false);
        let (after, n_after) = submit(true);
        assert_eq!((n_before, n_after), (TEXTURES * LISTS, TEXTURES));
        println!(
            "{LISTS} lists sampling {TEXTURES} textures of {} MiB: \
             before {n_before} hashes in {before:?}, after {n_after} in {after:?} ({:.1}x)",
            SIZE >> 20,
            before.as_secs_f64() / after.as_secs_f64()
        );
    }

    #[test]
    fn tearing_env_parses_opt_out() {
        assert!(super::tearing_enabled(None));
        assert!(super::tearing_enabled(Some("1")));
        assert!(super::tearing_enabled(Some("")));
        assert!(!super::tearing_enabled(Some("0")));
        assert!(!super::tearing_enabled(Some(" 0 ")));
    }

    #[test]
    fn tearing_present_falls_back_once_and_preserves_resize_flags() {
        use windows::Win32::Graphics::Dxgi::{
            DXGI_ERROR_INVALID_CALL, DXGI_PRESENT, DXGI_PRESENT_ALLOW_TEARING,
            DXGI_SWAP_CHAIN_FLAG, DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING,
        };
        use windows::core::HRESULT;

        let ok = HRESULT(0);
        let mut mode = super::PresentMode::new(true);
        assert_eq!(mode.swap_chain_flags(), DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING);

        let mut calls = Vec::new();
        assert_eq!(
            mode.present(1, |flags| {
                calls.push(flags);
                ok
            }),
            ok
        );
        assert_eq!(calls, [DXGI_PRESENT_ALLOW_TEARING]);

        // First refused tearing present retries immediately with 0 flags and latches off.
        calls.clear();
        assert_eq!(
            mode.present(2, |flags| {
                calls.push(flags);
                if flags == DXGI_PRESENT_ALLOW_TEARING {
                    DXGI_ERROR_INVALID_CALL
                } else {
                    ok
                }
            }),
            ok
        );
        assert_eq!(calls, [DXGI_PRESENT_ALLOW_TEARING, DXGI_PRESENT(0)]);
        assert!(!mode.presents_tearing);
        // ResizeBuffers must keep the original swap-chain creation flags.
        assert_eq!(mode.swap_chain_flags(), DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING);

        // Subsequent presents stay on non-tearing mode without retrying tearing.
        calls.clear();
        assert_eq!(
            mode.present(3, |flags| {
                calls.push(flags);
                ok
            }),
            ok
        );
        assert_eq!(calls, [DXGI_PRESENT(0)]);

        let disabled = super::PresentMode::new(false);
        assert_eq!(disabled.swap_chain_flags(), DXGI_SWAP_CHAIN_FLAG(0));
        assert!(!disabled.presents_tearing);
    }

    #[test]
    fn dxgi_factory_tearing_query_runs_cleanly() {
        use windows::Win32::Graphics::Dxgi::{
            CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIFactory2,
        };
        // SAFETY: plain DXGI factory creation.
        let factory: IDXGIFactory2 =
            unsafe { CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)) }.unwrap();
        let _ = super::query_tearing_support(&factory);
    }
}
