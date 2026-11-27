//! The host GPU in tests (backlog G1).
//!
//! A GPU test takes the host device from [`device`]. It is made on the
//! default (hardware) adapter, on the software one with
//! `XTP_TEST_ADAPTER=warp`, or not at all with `XTP_TEST_ADAPTER=none`.
//! Without a device the test passes after printing `SKIPPED <test>:
//! <reason>` on standard error (not captured, so a run lists its skips),
//! unless `XTP_REQUIRE_GPU` is set: then it fails (strict mode, for
//! machines that must run the GPU tests).
//!
//! The rest is what GPU tests share: title buffers, a title list and its
//! submission, and a compute kernel recorded into a host list as a title
//! dispatch would be.

use std::io::Write;
use std::sync::Arc;

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use crate::host::{Host, HostList};
use crate::object::{HRESULT, Iid, S_OK, This, guid, slot};
use crate::shadow::Shadow;

/// What `XTP_TEST_ADAPTER` asks for: `Some(Ok(true))` the software
/// adapter, `Some(Ok(false))` the default one, an error no device (the
/// reason tests print), `None` when it is not set (then as in a run).
pub(crate) fn software() -> Option<Result<bool, String>> {
    adapter(std::env::var("XTP_TEST_ADAPTER").ok().as_deref())
}

fn adapter(value: Option<&str>) -> Option<Result<bool, String>> {
    Some(match value? {
        "" | "hardware" => Ok(false),
        "warp" => Ok(true),
        "none" => Err("no device asked for (XTP_TEST_ADAPTER=none)".into()),
        other => Err(format!(
            "XTP_TEST_ADAPTER={other} is none of warp, hardware or none"
        )),
    })
}

/// Binds XTP's own view and sampler heaps on `list`: the translator's root
/// signatures index both heaps directly, so both must be bound before one is
/// set (debug layer message 1419).
pub(crate) fn bind_indexed_heaps(list: &HostList) {
    // SAFETY: an open list; XTP's own heaps live for the run.
    unsafe {
        list.list
            .SetDescriptorHeaps(&crate::descriptor::heaps_with_own_samplers(None));
    }
}

/// The host device for the test named `test`, or `None` once the skip is
/// reported (see the module). Keep it until the test ends: then the test
/// fails on the debug layer's errors (see [`Gpu`]).
pub(crate) fn device(test: &str) -> Option<Gpu> {
    match crate::host::created() {
        Ok(host) => {
            watch(host);
            Some(Gpu { host })
        }
        Err(reason) => {
            skip(test, reason);
            None
        }
    }
}

/// A test's host device. With the D3D12 debug layer on
/// (`XTP_D3D12_DEBUG=1` in a diagnostics build, as CI's `gpu-debug-layer`
/// job runs the GPU tests), dropping it fails the test with the errors the
/// layer raised on the test's thread (named after the test), and with any
/// it raised on other threads of ours meanwhile. When tests run one at a
/// time (`RUST_TEST_THREADS=1`), it also fails on errors left by another
/// test's thread, which then kept no guard.
pub(crate) struct Gpu {
    host: &'static Host,
}

impl Gpu {
    /// The device itself, as a plain reference (the guard stays with the
    /// test).
    pub(crate) fn host(&self) -> &'static Host {
        self.host
    }
}

thread_local! {
    /// Set while this thread hands the device invalid input on purpose.
    static EXPECTING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `f`, which hands the device invalid input on purpose: the errors
/// the debug layer raises on this thread meanwhile are expected, and not
/// counted against the test.
pub(crate) fn expect_layer_errors<T>(f: impl FnOnce() -> T) -> T {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            EXPECTING.with(|e| e.set(self.0));
        }
    }
    let _reset = Reset(EXPECTING.with(|e| e.replace(true)));
    f()
}

impl std::ops::Deref for Gpu {
    type Target = Host;
    fn deref(&self) -> &Host {
        self.host
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        let errors = layer_errors();
        if std::thread::panicking() {
            // Taken all the same, so they cannot fail the next test; shown,
            // since they may explain this one's failure.
            if !errors.is_empty() {
                let _ = writeln!(
                    std::io::stderr(),
                    "the D3D12 debug layer also reported {} error(s):\n{}",
                    errors.len(),
                    errors.join("\n")
                );
            }
            return;
        }
        if !errors.is_empty() {
            panic!(
                "the D3D12 debug layer reported {} error(s):\n{}",
                errors.len(),
                errors.join("\n")
            );
        }
    }
}

fn lock<T>(m: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn thread_name() -> String {
    std::thread::current()
        .name()
        .unwrap_or("unnamed")
        .to_owned()
}

/// An error the debug layer raised: the thread whose call reported it, the
/// message id and its text.
struct LayerError {
    thread: String,
    id: i32,
    text: String,
}

/// The debug layer's errors (`XTP_D3D12_DEBUG=1` in a diagnostics build).
static LAYER_ERRORS: std::sync::Mutex<Vec<LayerError>> = std::sync::Mutex::new(Vec::new());

/// Whether `thread` runs a test: the harness names a test's thread after
/// its path (`module::tests::name`); ours are named `xtp-…` or not at all.
fn is_test_thread(thread: &str) -> bool {
    thread.contains("::")
}

/// Whether tests run one at a time (`RUST_TEST_THREADS=1`): then errors of
/// another test's thread are that test's, left without a guard.
fn serial() -> bool {
    let args: Vec<String> = std::env::args().collect();
    std::env::var("RUST_TEST_THREADS").as_deref() == Ok("1")
        || args.iter().any(|a| a == "--test-threads=1")
        || args
            .windows(2)
            .any(|w| w[0] == "--test-threads" && w[1] == "1")
}

/// Takes the errors this thread raised, those raised on threads that are
/// no test's, and (run serially) those another test's thread left.
fn layer_errors() -> Vec<String> {
    let me = thread_name();
    let serial = serial();
    let mut errors = lock(&LAYER_ERRORS);
    let mut taken = Vec::new();
    errors.retain(|e| {
        if e.thread == me {
            taken.push(format!("{} (message id {})", e.text, e.id));
            false
        } else if !is_test_thread(&e.thread) {
            taken.push(format!(
                "{} (message id {}, on thread {})",
                e.text, e.id, e.thread
            ));
            false
        } else if serial {
            taken.push(format!(
                "{} (message id {}, left by test {}, which kept no device guard)",
                e.text, e.id, e.thread
            ));
            false
        } else {
            true
        }
    });
    taken
}

/// Has the debug layer report its errors to [`LAYER_ERRORS`] as they are
/// raised, on the thread that made the call, before the host moves them
/// into its log.
/// Whether it does: not when the layer was not asked for; an error when it
/// was but cannot report to us (reported once, and a failure in strict
/// mode, so a CI run cannot pass without the layer).
pub(crate) fn watch(host: &Host) -> bool {
    static WATCHING: std::sync::OnceLock<Result<bool, String>> = std::sync::OnceLock::new();
    let watching = WATCHING.get_or_init(|| {
        if std::env::var_os("XTP_D3D12_DEBUG").is_none() {
            return Ok(false);
        }
        if !cfg!(feature = "diagnostics") {
            return Err(
                "XTP_D3D12_DEBUG is set but this build has no diagnostics feature, which the \
                 layer needs (--features diagnostics)"
                    .into(),
            );
        }
        let Some(queue) = host.message_queue() else {
            return Err(
                "XTP_D3D12_DEBUG is set but the D3D12 debug layer is missing (install \
                        Graphics Tools)"
                    .into(),
            );
        };
        let queue = windows::core::Interface::cast::<ID3D12InfoQueue1>(queue).map_err(|e| {
            format!("the D3D12 debug layer has no message callback (ID3D12InfoQueue1): {e}")
        })?;
        let mut cookie = 0;
        // SAFETY: a static callback, registered once for the device's life.
        unsafe {
            queue.RegisterMessageCallback(
                Some(on_message),
                D3D12_MESSAGE_CALLBACK_FLAG_NONE,
                std::ptr::null_mut(),
                &mut cookie,
            )
        }
        .map_err(|e| format!("no D3D12 message callback: {e}"))?;
        Ok(true)
    });
    match watching {
        Ok(on) => *on,
        Err(why) => {
            let strict =
                std::env::var_os("XTP_REQUIRE_GPU").is_some_and(|v| !v.is_empty() && v != "0");
            assert!(!strict, "{why} (XTP_REQUIRE_GPU is set)");
            static TOLD: std::sync::Once = std::sync::Once::new();
            TOLD.call_once(|| {
                let _ = writeln!(std::io::stderr(), "{why}: its errors cannot fail the tests");
            });
            false
        }
    }
}

unsafe extern "system" fn on_message(
    _category: D3D12_MESSAGE_CATEGORY,
    severity: D3D12_MESSAGE_SEVERITY,
    id: D3D12_MESSAGE_ID,
    description: windows::core::PCSTR,
    _context: *mut std::ffi::c_void,
) {
    // A panic must not unwind into the layer (that aborts the run): one here
    // loses the message, which the host's log still has.
    let _ = std::panic::catch_unwind(|| {
        if severity.0 > D3D12_MESSAGE_SEVERITY_ERROR.0 || EXPECTING.with(|e| e.get()) {
            return;
        }
        // SAFETY: the layer passes a NUL-terminated description.
        let text = unsafe { description.to_string() }.unwrap_or_default();
        lock(&LAYER_ERRORS).push(LayerError {
            thread: thread_name(),
            id: id.0,
            text,
        });
    });
}

/// Reports that `test` did not run, for `reason`: on standard error, or as
/// the test's failure in strict mode.
pub(crate) fn skip(test: &str, reason: &str) {
    let strict = std::env::var_os("XTP_REQUIRE_GPU").is_some_and(|v| !v.is_empty() && v != "0");
    match skipped(test, reason, strict) {
        Ok(line) => {
            let _ = writeln!(std::io::stderr(), "{line}");
        }
        Err(failure) => panic!("{failure}"),
    }
}

/// The line a skip prints, or in strict mode the failure.
fn skipped(test: &str, reason: &str, strict: bool) -> Result<String, String> {
    if strict {
        Err(format!(
            "{test} needs a host GPU: {reason} (XTP_REQUIRE_GPU is set)"
        ))
    } else {
        Ok(format!("SKIPPED {test}: {reason}"))
    }
}

const RESOURCE_IID: Iid = guid(
    0x696442be,
    0xa72e,
    0x4059,
    [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
);
const ALLOCATOR_IID: Iid = guid(
    0x6102dee4,
    0xaf59,
    0x4b09,
    [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
);
const LIST_IID: Iid = guid(
    0x5b160d0f,
    0xac1b,
    0x4185,
    [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
);
const QUEUE_IID: Iid = guid(
    0x0ec870a6,
    0x5d7e,
    0x4c22,
    [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
);

fn release(object: This) {
    slot::<extern "system" fn(This) -> u32>(object, 0x10)(object);
}

/// A title buffer, created as the title creates one (a committed resource
/// in the default heap) and released when dropped.
pub(crate) struct TitleBuffer {
    resource: This,
    data: *mut u8,
    len: usize,
}

impl TitleBuffer {
    /// A zeroed buffer of `len` bytes.
    pub(crate) fn new(len: usize) -> TitleBuffer {
        /// The console's heap properties and resource description, as the
        /// title passes them.
        #[repr(C)]
        struct Desc {
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
        const _: () = assert!(size_of::<Desc>() == 56);
        const DEFAULT_HEAP: [u32; 5] = [1, 0, 0, 0, 0];
        let desc = Desc {
            dimension: 1,
            alignment: 0,
            width: len as u64,
            height: 1,
            depth_or_array_size: 1,
            mip_levels: 1,
            format: 0,
            sample_count: 1,
            sample_quality: 0,
            layout: 1,
            flags: 0,
        };
        let null = std::ptr::null();
        let mut resource = null;
        let made = crate::resource::create_committed(
            null,
            DEFAULT_HEAP.as_ptr().cast(),
            0,
            (&raw const desc).cast(),
            0,
            null.cast(),
            &RESOURCE_IID,
            &mut resource,
        );
        assert_eq!(made, S_OK, "a title buffer of {len} bytes");
        let (data, size) = crate::resource::buffer_bytes(resource).expect("a buffer");
        assert_eq!(size, len as u64);
        TitleBuffer {
            resource,
            data,
            len,
        }
    }

    /// The title resource.
    pub(crate) fn resource(&self) -> This {
        self.resource
    }

    /// Its title (and GPU) address.
    pub(crate) fn address(&self) -> u64 {
        self.data as u64
    }

    /// Its title memory as 32-bit words.
    pub(crate) fn words(&self) -> Vec<u32> {
        // SAFETY: the buffer's own memory, `len` bytes.
        let bytes = unsafe { std::slice::from_raw_parts(self.data, self.len) };
        let (words, _) = bytes.as_chunks::<4>();
        words.iter().map(|&w| u32::from_le_bytes(w)).collect()
    }

    /// Writes its title memory as the title's CPU would.
    pub(crate) fn set_words(&self, words: impl Fn(usize) -> u32) {
        for i in 0..self.len / 4 {
            // SAFETY: inside the buffer's own memory.
            unsafe { self.data.cast::<u32>().add(i).write_unaligned(words(i)) };
        }
    }

    /// Its host copy, made now if it is not yet: as if a draw had used it
    /// before (in an earlier frame).
    pub(crate) fn shadow(&self) -> Arc<Shadow> {
        crate::shadow::find(self.address(), self.len as u64).expect("a host copy")
    }
}

impl Drop for TitleBuffer {
    fn drop(&mut self) {
        release(self.resource);
    }
}

/// A title command list with its allocator, recording into a host list.
pub(crate) struct TitleList {
    list: This,
    allocator: This,
}

impl TitleList {
    pub(crate) fn new() -> TitleList {
        let null = std::ptr::null();
        let (mut allocator, mut list) = (null, null);
        assert_eq!(
            crate::list::create_allocator(null, 0, &ALLOCATOR_IID, &mut allocator),
            S_OK
        );
        assert_eq!(
            crate::list::create_list(null, 0, 0, allocator, null, &LIST_IID, &mut list),
            S_OK
        );
        TitleList { list, allocator }
    }

    /// The title list, for its methods.
    pub(crate) fn this(&self) -> This {
        self.list
    }

    /// Its host list (there is a device).
    pub(crate) fn host(&self) -> &HostList {
        crate::list::host_list(self.list).expect("a host list")
    }

    /// Closes the list and submits it on a title queue, as the title does;
    /// it has run when this returns.
    pub(crate) fn run(&self) {
        let null = std::ptr::null();
        let mut queue = null;
        let desc = [0u32; 4];
        assert_eq!(
            crate::queue::create(null, desc.as_ptr().cast(), &QUEUE_IID, &mut queue),
            S_OK
        );
        assert_eq!(
            slot::<extern "system" fn(This) -> HRESULT>(self.list, 0x48)(self.list),
            S_OK
        );
        slot::<extern "system" fn(This, u32, *const This)>(queue, 0x50)(queue, 1, &self.list);
        // Submissions are not waited for: finished here, written back.
        if let Some(host) = crate::host::made() {
            host.drain();
        }
        release(queue);
    }
}

impl Drop for TitleList {
    fn drop(&mut self) {
        release(self.list);
        release(self.allocator);
    }
}

/// Bytes a [`Kernel`] dispatch reads and writes: 64 words.
pub(crate) const KERNEL_BYTES: u64 = 256;

/// What a [`Kernel`] fill writes at word `i`.
pub(crate) fn pattern(i: usize) -> u32 {
    (i as u32).wrapping_mul(2_654_435_761).wrapping_add(12_345)
}

const KERNEL: &str = r#"
#define RS "RootConstants(num32BitConstants=1, b0), UAV(u0), UAV(u1)"
cbuffer Mode : register(b0) { uint mode; };
RWByteAddressBuffer src : register(u0);
RWByteAddressBuffer dst : register(u1);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint at = id.x * 4;
    uint word = src.Load(at);
    dst.Store(at, mode == 0 ? id.x * 2654435761u + 12345u : word);
}
"#;

/// A compute kernel standing in for a title pass: one group of 64 threads
/// that fills 64 words with [`pattern`] or copies 64 words.
pub(crate) struct Kernel {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

/// How a [`Kernel`] dispatch writes its destination.
#[derive(Clone, Copy)]
pub(crate) enum Pass {
    Fill,
    Copy,
}

impl Kernel {
    pub(crate) fn new(host: &Host) -> Kernel {
        let mut code: Option<ID3DBlob> = None;
        let mut errors: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call; out pointers
        // ours.
        let compiled = unsafe {
            D3DCompile(
                KERNEL.as_ptr().cast(),
                KERNEL.len(),
                s!("test kernel"),
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
            panic!("the test kernel did not compile ({e}): {text}");
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
            Kernel { root, pso }
        }
    }

    /// Records a dispatch into `list` as a title dispatch is recorded: the
    /// [`KERNEL_BYTES`] it reads at title address `src` and writes at `dst`
    /// brought up to date from title memory first, `dst` declared written
    /// by shaders (copied back after the submission), then a barrier as
    /// the title puts between dependent passes.
    pub(crate) fn dispatch(&self, list: &HostList, pass: Pass, src: u64, dst: u64) {
        let find = |at: u64| crate::shadow::find(at, KERNEL_BYTES).expect("a host copy");
        let (from, to) = (find(src), find(dst));
        from.refresh(list, src, KERNEL_BYTES);
        to.refresh(list, dst, KERNEL_BYTES);
        to.gpu_writes(list, dst, KERNEL_BYTES);
        list.keep(&self.root);
        list.keep(&self.pso);
        let mode = match pass {
            Pass::Fill => 0,
            Pass::Copy => 1,
        };
        // SAFETY: an open list, live objects (kept until the list is
        // reset) and root views inside the host copies.
        unsafe {
            list.list.SetComputeRootSignature(&self.root);
            list.list.SetPipelineState(&self.pso);
            list.list.SetComputeRoot32BitConstant(0, mode, 0);
            list.list
                .SetComputeRootUnorderedAccessView(1, from.gpu(src));
            list.list.SetComputeRootUnorderedAccessView(2, to.gpu(dst));
            list.list.Dispatch(1, 1, 1);
            list.list.ResourceBarrier(&[crate::host::uav_barrier()]);
        }
    }

    /// Records a [`Pass::Fill`] dispatch directly into `dst` at title address
    /// `at` without deferring refreshes or copy-backs, so tests can control
    /// snapshot, title writes, and copy-back explicitly.
    pub(crate) fn dispatch_fill_raw(&self, list: &HostList, dst: &Shadow, at: u64) {
        list.keep(&self.root);
        list.keep(&self.pso);
        // SAFETY: an open list, live pipeline/signature kept on the list, and
        // valid GPU virtual address inside `dst`.
        unsafe {
            list.list.SetComputeRootSignature(&self.root);
            list.list.SetPipelineState(&self.pso);
            list.list.SetComputeRoot32BitConstant(0, 0, 0);
            list.list.SetComputeRootUnorderedAccessView(1, dst.gpu(at));
            list.list.SetComputeRootUnorderedAccessView(2, dst.gpu(at));
            list.list.Dispatch(1, 1, 1);
            list.list.ResourceBarrier(&[crate::host::uav_barrier()]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_adapter_is_chosen_by_the_variable() {
        assert_eq!(adapter(None), None);
        assert_eq!(adapter(Some("warp")), Some(Ok(true)));
        assert_eq!(adapter(Some("hardware")), Some(Ok(false)));
        assert_eq!(adapter(Some("")), Some(Ok(false)));
        assert!(matches!(adapter(Some("none")), Some(Err(_))));
        let wrong = adapter(Some("wrap")).unwrap().unwrap_err();
        assert!(wrong.contains("wrap"), "{wrong}");
    }

    #[test]
    fn a_test_without_a_device_says_it_skipped_or_fails_when_strict() {
        assert_eq!(
            skipped("a::test", "no adapter", false),
            Ok("SKIPPED a::test: no adapter".to_owned())
        );
        let failure = skipped("a::test", "no adapter", true).unwrap_err();
        assert!(failure.contains("a::test") && failure.contains("no adapter"));
        assert!(failure.contains("XTP_REQUIRE_GPU"));
    }

    #[test]
    #[should_panic(expected = "XTP_REQUIRE_GPU")]
    fn a_strict_skip_fails_the_test() {
        if let Err(failure) = skipped("a::test", "no adapter", true) {
            panic!("{failure}");
        }
    }

    /// GPU tests keep their device guard until they end. No call of
    /// `device(..)` (or `isolated_device(..)`) is followed by `.is_none()`
    /// or bound to `_` (`let _ =`, `let Some(_) =`), which drops the guard at
    /// once and so escapes the debug layer's check; write
    /// `let Some(_gpu) = device(..) else { .. };`. Calls span lines as
    /// rustfmt wraps them; comment lines are skipped.
    #[test]
    fn gpu_tests_keep_their_device_guard() {
        fn sources(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    sources(&path, out);
                } else if path.extension().is_some_and(|e| e == "rs") {
                    out.push(path);
                }
            }
        }
        /// The calls of `code` that drop the guard at once, by line.
        fn dropped(code: &str) -> Vec<usize> {
            let needle = ["dev", "ice("].concat();
            let ident = |c: char| c.is_alphanumeric() || c == '_' || c == ':';
            let mut lines = Vec::new();
            for (at, _) in code.match_indices(&needle) {
                let before = &code[..at];
                let start = before.rfind(|c: char| !ident(c)).map_or(0, |p| p + 1);
                let callee = &code[start..at + needle.len() - 1];
                let ours = callee == "device"
                    || callee.ends_with("::device")
                    || callee.ends_with("isolated_device");
                let lead = before[..start].trim_end();
                if !ours || lead.ends_with("fn") {
                    continue;
                }
                let mut depth = 0;
                let mut end = None;
                for (i, c) in code[at + needle.len() - 1..].char_indices() {
                    match c {
                        '(' => depth += 1,
                        ')' => {
                            depth -= 1;
                            if depth == 0 {
                                end = Some(at + needle.len() - 1 + i + 1);
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                let Some(end) = end else { continue };
                let after = code[end..].trim_start();
                if after.starts_with(".is_none()")
                    || lead.ends_with("let _ =")
                    || lead.ends_with("let Some(_) =")
                {
                    lines.push(before.lines().count());
                }
            }
            lines
        }
        // The scan itself: what it must find, and what it must leave.
        let call = ["dev", "ice"].concat();
        let wrapped = format!("    if crate::testgpu::{call}(test)\n        .is_none()\n");
        assert_eq!(dropped(&wrapped).len(), 1, "{wrapped}");
        assert_eq!(dropped(&format!("let _ = {call}(test);")).len(), 1);
        assert_eq!(
            dropped(&format!("let Some(_) = {call}(test) else {{ return }};")).len(),
            1
        );
        assert!(dropped(&format!("let Some(_gpu) = {call}(test) else {{ return }};")).is_empty());
        assert!(dropped(&format!("let mut found = {call}();")).is_empty());

        let mut files = Vec::new();
        sources(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
            &mut files,
        );
        let mut found = Vec::new();
        for file in &files {
            let text = std::fs::read_to_string(file).unwrap();
            let code: Vec<&str> = text
                .lines()
                .map(|l| {
                    if l.trim_start().starts_with("//") {
                        ""
                    } else {
                        l
                    }
                })
                .collect();
            for n in dropped(&code.join("\n")) {
                found.push(format!("{}:{}", file.display(), n));
            }
        }
        assert!(files.len() > 50, "found {} sources", files.len());
        assert!(found.is_empty(), "keep the guard:\n{}", found.join("\n"));
    }

    /// With the debug layer on (CI's debug-layer step), a test that records
    /// a wrong barrier (the same state on both sides) fails when its device
    /// goes, with the layer's message.
    #[test]
    fn a_wrong_barrier_fails_the_test_with_the_layers_message() {
        let test = "testgpu::tests::a_wrong_barrier_fails_the_test_with_the_layers_message";
        let outcome = std::panic::catch_unwind(|| {
            let gpu = device(test)?;
            if !watch(gpu.host()) {
                return None;
            }
            let list = gpu.new_list().unwrap();
            let (buffer, _) = gpu.upload_buffer(256).unwrap();
            let state = D3D12_RESOURCE_STATE_COPY_DEST;
            let barrier = crate::host::transition(&buffer, state, state);
            // SAFETY: an open list and a live buffer; the barrier is wrong on
            // purpose, which the layer reports as it is recorded. The list is
            // never closed (that fails the run) nor submitted.
            unsafe { list.list.ResourceBarrier(&[barrier]) };
            Some(())
        });
        match outcome {
            Ok(None) => {
                let _ = writeln!(
                    std::io::stderr(),
                    "SKIPPED {test}: needs the debug layer (XTP_D3D12_DEBUG=1, diagnostics build)"
                );
            }
            Ok(Some(())) => panic!("the wrong barrier did not fail the test"),
            Err(failure) => {
                let text = failure
                    .downcast_ref::<String>()
                    .cloned()
                    .unwrap_or_default();
                assert!(
                    text.contains("the D3D12 debug layer reported")
                        && text.contains("ResourceBarrier"),
                    "{text}"
                );
            }
        }
    }

    /// The helper end to end: a kernel dispatched in a title list fills a
    /// title buffer through its host copy (copied back after the run).
    #[test]
    fn a_kernel_fills_a_title_buffer() {
        let Some(gpu) = device("testgpu::tests::a_kernel_fills_a_title_buffer") else {
            return;
        };
        let host = gpu.host();
        let kernel = Kernel::new(host);
        let (scratch, out) = (TitleBuffer::new(256), TitleBuffer::new(256));
        let list = TitleList::new();
        kernel.dispatch(list.host(), Pass::Fill, scratch.address(), out.address());
        list.run();
        let want: Vec<u32> = (0..64).map(pattern).collect();
        assert_eq!(out.words(), want);
    }
}
