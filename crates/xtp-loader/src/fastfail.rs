//! Failures that end the process without any exception filter running,
//! named in the crash log:
//!
//! - a stack-cookie check failure. MSVC's `__security_check_cookie` jumps
//!   to `__report_gsfailure`, which fast-fails (`int 29h`): no handler of
//!   ours sees it. Each image's `__report_gsfailure` is found through its
//!   cookie check ([`find_cookie_checks`]) and redirected to a report that
//!   names the function whose cookie was overwritten, then fast-fails the
//!   same way (`STATUS_STACK_BUFFER_OVERRUN`);
//! - a CRT invalid parameter. Titles' `api-ms-win-crt-*` imports bind to
//!   the host's UCRT, which with no handler fast-fails too. A process-wide
//!   handler names the caller; when it returns, the CRT goes on as it does
//!   after any handler (the function fails with `EINVAL`, and the
//!   `noreturn` variants still end the process);
//! - the process ending through `NtTerminateProcess` with a non-zero
//!   status (`TerminateProcess`, `ExitProcess`, the CRT's exits): ntdll's
//!   stub is redirected to a report of who asked, then does as asked.
//!
//! The redirections are 16 bytes written over the target's first
//! instructions (`redirect_code`); the target never runs again, except
//! `NtTerminateProcess`, whose overwritten instructions run from a copy.

use std::collections::BTreeSet;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};

use anyhow::{Context, Result, bail};
use windows_sys::Win32::Foundation::HANDLE;
use windows_sys::Win32::System::Diagnostics::Debug::{
    FlushInstructionCache, RtlCaptureContext, RtlLookupFunctionEntry,
};
use windows_sys::Win32::System::LibraryLoader::{
    GetModuleHandleW, GetProcAddress, LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExW,
};
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_READWRITE,
    VirtualAlloc, VirtualProtect,
};
use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetCurrentProcessId, GetProcessId};
use xtp_pe::ImageInfo;

use crate::crash::{Aligned16, unwind};
use crate::locate;

const SCN_MEM_EXECUTE: u32 = 0x2000_0000;
/// `__fastfail`'s code for a stack cookie check failure.
const FAST_FAIL_STACK_COOKIE_CHECK_FAILURE: u32 = 2;
/// Bytes a redirection writes ([`redirect_code`]).
const REDIRECT_LEN: usize = 16;

/// One `__security_check_cookie` found in code (addresses).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CookieCheck {
    pub(crate) check: usize,
    /// The jump to `__report_gsfailure`.
    pub(crate) failure_jump: usize,
    /// Where it jumps: `__report_gsfailure`.
    pub(crate) report: usize,
}

/// `rol rcx, 16; test cx, 0xffff`, the middle of a cookie check.
const CHECK_MIDDLE: [u8; 9] = [0x48, 0xc1, 0xc1, 0x10, 0x66, 0xf7, 0xc1, 0xff, 0xff];

/// The cookie checks in `code` (whose first byte is at address `at`), as
/// MSVC's x64 runtime builds `__security_check_cookie` (each `f2`, a `bnd`
/// prefix, optional):
///
/// ```text
///          cmp  rcx, [rip + __security_cookie]  ; 48 3b 0d rel32
///          jne  failure                         ; [f2] 75 rel8
///          rol  rcx, 16                         ; 48 c1 c1 10
///          test cx, 0xffff                      ; 66 f7 c1 ff ff
///          jne  restore                         ; [f2] 75 rel8
///          ret                                  ; [f2] c3
/// restore: ror  rcx, 16                         ; 48 c1 c9 10
/// failure: jmp  __report_gsfailure              ; [f2] e9 rel32 | [f2] eb rel8
/// ```
///
/// Found by its middle, then checked whole, both branch targets included.
pub(crate) fn find_cookie_checks(code: &[u8], at: usize) -> Vec<CookieCheck> {
    memchr::memmem::find_iter(code, &CHECK_MIDDLE)
        .filter_map(|middle| cookie_check_around(code, middle))
        .map(|(check, failure_jump, report)| CookieCheck {
            check: at + check,
            failure_jump: at + failure_jump,
            report: at.wrapping_add_signed(report as isize),
        })
        .collect()
}

/// The check whose middle is at `middle`: offsets of its `cmp`, of its
/// failure jump, and of that jump's target (which may be outside `code`).
fn cookie_check_around(code: &[u8], middle: usize) -> Option<(usize, usize, i64)> {
    let byte = |i: usize| code.get(i).copied();
    let rel8 = |i: usize| byte(i).map(|b| i64::from(b as i8));
    // An optional `bnd` prefix before `opcode` at `i`: the opcode's offset.
    let bnd = |i: usize, opcode: u8| match byte(i)? {
        0xf2 if byte(i + 1)? == opcode => Some(i + 1),
        b if b == opcode => Some(i),
        _ => None,
    };
    // jne failure, right before the middle.
    let first_jne = middle.checked_sub(2)?;
    if byte(first_jne)? != 0x75 {
        return None;
    }
    let first_start = match first_jne.checked_sub(1).and_then(byte) {
        Some(0xf2) => first_jne - 1,
        _ => first_jne,
    };
    let check = first_start.checked_sub(7)?;
    if code.get(check..check + 3)? != [0x48, 0x3b, 0x0d] {
        return None;
    }
    let failure = middle as i64 + rel8(first_jne + 1)?;
    // jne restore, ret, restore, failure.
    let second_jne = bnd(middle + CHECK_MIDDLE.len(), 0x75)?;
    let restore = (second_jne + 2) as i64 + rel8(second_jne + 1)?;
    let ret = bnd(second_jne + 2, 0xc3)?;
    if restore != ret as i64 + 1 || code.get(ret + 1..ret + 5)? != [0x48, 0xc1, 0xc9, 0x10] {
        return None;
    }
    let jump = ret + 5;
    if failure != jump as i64 {
        return None;
    }
    let opcode = match byte(jump)? {
        0xf2 => jump + 1,
        _ => jump,
    };
    let report = match byte(opcode)? {
        0xe9 => {
            let rel = i32::from_le_bytes(code.get(opcode + 1..opcode + 5)?.try_into().ok()?);
            (opcode + 5) as i64 + i64::from(rel)
        }
        0xeb => (opcode + 2) as i64 + rel8(opcode + 1)?,
        _ => return None,
    };
    Some((check, jump, report))
}

/// Which register a redirection puts the return address in (one the
/// target's own arguments leave free).
#[derive(Debug, Clone, Copy)]
enum Keep {
    Rdx,
    R8,
}

/// `mov <keep>, [rsp]; mov rax, to; jmp rax`: sends whoever runs it to
/// `to`, with the address its caller's call returns to in `keep`.
fn redirect_code(keep: Keep, to: usize) -> [u8; REDIRECT_LEN] {
    let mut code = [0u8; REDIRECT_LEN];
    code[..4].copy_from_slice(match keep {
        Keep::Rdx => &[0x48, 0x8b, 0x14, 0x24],
        Keep::R8 => &[0x4c, 0x8b, 0x04, 0x24],
    });
    code[4..6].copy_from_slice(&[0x48, 0xb8]);
    code[6..14].copy_from_slice(&(to as u64).to_le_bytes());
    code[14..].copy_from_slice(&[0xff, 0xe0]);
    code
}

/// Writes `bytes` over code at `at`.
///
/// # Safety
/// `at..at + bytes.len()` is mapped code no thread is running.
unsafe fn patch(at: usize, bytes: &[u8]) -> Result<()> {
    let mut old = 0;
    // SAFETY: mapped, per the caller.
    let ok = unsafe { VirtualProtect(at as _, bytes.len(), PAGE_EXECUTE_READWRITE, &mut old) };
    if ok == 0 {
        bail!(
            "VirtualProtect({at:#x}) failed: {}",
            std::io::Error::last_os_error()
        );
    }
    // SAFETY: writable now; nobody runs it, per the caller. Then the old
    // protection back, and our own instruction cache flushed.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), at as *mut u8, bytes.len());
        VirtualProtect(at as _, bytes.len(), old, &mut old);
        FlushInstructionCache(GetCurrentProcess(), at as _, bytes.len());
    }
    Ok(())
}

/// `code` in a page of its own, made executable. Leaked: it is called for
/// the life of the process.
fn executable(code: &[u8]) -> Result<usize> {
    // SAFETY: a fresh allocation; no existing memory is touched.
    let page = unsafe {
        VirtualAlloc(
            std::ptr::null(),
            code.len(),
            MEM_COMMIT | MEM_RESERVE,
            PAGE_READWRITE,
        )
    } as *mut u8;
    if page.is_null() {
        bail!("VirtualAlloc failed: {}", std::io::Error::last_os_error());
    }
    let mut old = 0;
    // SAFETY: our fresh page, large enough; then made read+execute.
    unsafe {
        std::ptr::copy_nonoverlapping(code.as_ptr(), page, code.len());
        if VirtualProtect(page as _, code.len(), PAGE_EXECUTE_READ, &mut old) == 0 {
            bail!("VirtualProtect failed: {}", std::io::Error::last_os_error());
        }
        FlushInstructionCache(GetCurrentProcess(), page as _, code.len());
    }
    Ok(page as usize)
}

/// The function containing `address`, by the unwind data registered for
/// its image: (start, length of its first part).
fn function_at(address: usize) -> Option<(usize, usize)> {
    let mut image = 0;
    // SAFETY: looks up a function table entry; reads nothing at `address`.
    let entry = unsafe { RtlLookupFunctionEntry(address as u64, &mut image, null_mut()) };
    if entry.is_null() {
        return None;
    }
    // SAFETY: a valid entry of `image`'s function table.
    let (begin, end) = unsafe { ((*entry).BeginAddress, (*entry).EndAddress) };
    let start = image as usize + begin as usize;
    Some((start, end.saturating_sub(begin) as usize))
}

/// Redirects every `__report_gsfailure` of the image at `base` to our
/// report. Only a target that is a whole function by the image's unwind
/// data, long enough for the redirection, is touched. Returns how many.
///
/// # Safety
/// `base` is the image `info` describes, mapped and not yet running.
pub(crate) unsafe fn guard_stack_cookies(base: usize, info: &ImageInfo) -> Result<usize> {
    let end = base + info.size_of_image as usize;
    let mut reports = BTreeSet::new();
    for section in info
        .sections
        .iter()
        .filter(|s| s.characteristics & SCN_MEM_EXECUTE != 0)
    {
        let start = base + section.rva as usize;
        let len = (section.virtual_size as usize).min(end.saturating_sub(start));
        // SAFETY: an executable section of the mapped image, so readable.
        let code = unsafe { std::slice::from_raw_parts(start as *const u8, len) };
        reports.extend(
            find_cookie_checks(code, start)
                .into_iter()
                .map(|c| c.report),
        );
    }
    let to = stack_cookie_failed as extern "system" fn(usize, usize) -> ! as usize;
    let mut guarded = 0;
    for report in reports {
        let whole = (base..end).contains(&report)
            && function_at(report).is_some_and(|(at, len)| at == report && len >= REDIRECT_LEN);
        if !whole {
            tracing::warn!(
                at = %locate(report),
                "a cookie check's report is not a function: left alone"
            );
            continue;
        }
        // SAFETY: a whole function of the image, not running yet.
        unsafe { patch(report, &redirect_code(Keep::Rdx, to)) }?;
        guarded += 1;
    }
    Ok(guarded)
}

/// The calling thread's frames, from its caller up, one line each.
#[inline(never)]
fn stack_lines() -> String {
    // SAFETY: a zeroed CONTEXT is valid; capture writes the current one.
    let frames = unsafe {
        let mut context = Aligned16(std::mem::zeroed());
        RtlCaptureContext(&mut context.0);
        unwind(context.0)
    };
    stack_text(&frames[1.min(frames.len())..])
}

fn stack_text(frames: &[u64]) -> String {
    frames
        .iter()
        .enumerate()
        .map(|(i, &pc)| format!("  #{i:<2} {}\n", locate(pc as usize)))
        .collect()
}

/// The calling thread's return addresses, from its caller up.
#[inline(never)]
fn frames() -> Vec<u64> {
    // SAFETY: as in `stack_lines`.
    let mut frames = unsafe {
        let mut context = Aligned16(std::mem::zeroed());
        RtlCaptureContext(&mut context.0);
        unwind(context.0)
    };
    frames.remove(0);
    frames
}

/// Whether `address` is in one of the DLLs `names` names.
pub(crate) fn in_module(address: usize, names: &[&str]) -> bool {
    locate(address)
        .split_once('+')
        .is_some_and(|(module, _)| names.iter().any(|n| module.eq_ignore_ascii_case(n)))
}

/// Ends the process at once, as `__fastfail(code)` does.
fn fast_fail(code: u32) -> ! {
    // SAFETY: `int 29h` ends the process with STATUS_STACK_BUFFER_OVERRUN
    // and `code`; it does not return.
    unsafe { std::arch::asm!("int 0x29", in("ecx") code, options(noreturn, nostack)) }
}

/// A redirected `__report_gsfailure`: `cookie` is the value that failed the
/// check, `caller` where the call to `__security_check_cookie` returns to,
/// in the function whose stack cookie was overwritten.
extern "system" fn stack_cookie_failed(cookie: usize, caller: usize) -> ! {
    let function = function_at(caller).map_or_else(|| "?".to_owned(), |(at, _)| locate(at));
    let text = format!(
        "XTP: stack cookie check failed in {} (function {function}): a buffer on its stack \
         overran (cookie {cookie:#x})\n{}",
        locate(caller),
        stack_lines()
    );
    crate::log_crash(&text);
    fast_fail(FAST_FAIL_STACK_COOKIE_CHECK_FAILURE)
}

type InvalidParameterHandler = unsafe extern "C" fn(*const u16, *const u16, *const u16, u32, usize);

/// The handler that was there before ours, to call after ours.
static PREVIOUS_HANDLER: AtomicUsize = AtomicUsize::new(0);
static INVALID_PARAMETERS: AtomicU32 = AtomicU32::new(0);
/// Invalid parameters reported in full; later ones are only counted.
const REPORTED_INVALID_PARAMETERS: u32 = 16;

fn wide(text: *const u16) -> String {
    if text.is_null() {
        return "?".into();
    }
    // SAFETY: the CRT passes NUL-terminated strings.
    let len = (0..)
        .take_while(|&i| unsafe { text.add(i).read() } != 0)
        .count();
    // SAFETY: `len` characters before the NUL.
    String::from_utf16_lossy(unsafe { std::slice::from_raw_parts(text, len) })
}

/// The UCRT's handler: names the code that called the CRT function which
/// refused its argument (the first frame after the CRT's own).
unsafe extern "C" fn invalid_parameter(
    expression: *const u16,
    function: *const u16,
    file: *const u16,
    line: u32,
    reserved: usize,
) {
    let n = INVALID_PARAMETERS.fetch_add(1, Ordering::Relaxed);
    if n < REPORTED_INVALID_PARAMETERS {
        let frames = frames();
        let caller = frames
            .iter()
            .map(|&pc| pc as usize)
            .skip_while(|&pc| !in_module(pc, &["ucrtbase.dll"]))
            .find(|&pc| !in_module(pc, &["ucrtbase.dll"]));
        let mut text = format!(
            "XTP: CRT invalid parameter, called from {}",
            caller.map_or_else(|| "?".to_owned(), locate)
        );
        // Only a debug CRT says what it checked.
        if !function.is_null() {
            let (e, f, w) = (wide(expression), wide(function), wide(file));
            text.push_str(&format!(" ({e} in {f}, {w}:{line})"));
        }
        if n + 1 == REPORTED_INVALID_PARAMETERS {
            text.push_str(" (further ones are not reported)");
        }
        text.push('\n');
        text.push_str(&stack_text(&frames));
        crate::log_crash(&text);
    }
    let previous = PREVIOUS_HANDLER.load(Ordering::Relaxed);
    if previous != 0 {
        // SAFETY: the handler the CRT had before ours, same signature.
        let previous: InvalidParameterHandler = unsafe { std::mem::transmute(previous) };
        // SAFETY: forwarding the CRT's own arguments.
        unsafe { previous(expression, function, file, line, reserved) };
    }
}

fn install_invalid_parameter_handler() -> Result<()> {
    let name: Vec<u16> = "ucrtbase.dll".encode_utf16().chain(Some(0)).collect();
    // SAFETY: a NUL-terminated name; loads (or finds) the host's UCRT.
    let ucrt = unsafe { LoadLibraryExW(name.as_ptr(), null_mut(), LOAD_LIBRARY_SEARCH_SYSTEM32) };
    if ucrt.is_null() {
        bail!("no ucrtbase.dll: {}", std::io::Error::last_os_error());
    }
    // SAFETY: a valid module and a NUL-terminated name.
    let set = unsafe { GetProcAddress(ucrt, c"_set_invalid_parameter_handler".as_ptr().cast()) }
        .context("ucrtbase.dll has no _set_invalid_parameter_handler")?;
    type Set = unsafe extern "C" fn(Option<InvalidParameterHandler>) -> usize;
    // SAFETY: the UCRT's documented signature.
    let set: Set = unsafe { std::mem::transmute(set) };
    // SAFETY: installs a handler that lives as long as the process.
    let previous = unsafe { set(Some(invalid_parameter)) };
    PREVIOUS_HANDLER.store(previous, Ordering::Relaxed);
    Ok(())
}

/// ntdll's `NtTerminateProcess` as Windows 10 and 11 build it, past its
/// system-call number (`mov r10, rcx; mov eax, <number>` before):
///
/// ```text
/// test byte [0x7ffe0308], 1   ; f6 04 25 08 03 fe 7f 01
/// jne  +3                     ; 75 03
/// syscall                     ; 0f 05
/// ret                         ; c3
/// int  0x2e                   ; cd 2e
/// ret                         ; c3
/// ```
///
/// The redirection takes its first 16 bytes; a copy of them runs, then
/// jumps back to the `jne`, so the system call itself is made from ntdll.
const SYSCALL_TAIL: [u8; 16] = [
    0xf6, 0x04, 0x25, 0x08, 0x03, 0xfe, 0x7f, 0x01, 0x75, 0x03, 0x0f, 0x05, 0xc3, 0xcd, 0x2e, 0xc3,
];

fn is_syscall_stub(code: &[u8; 24]) -> bool {
    code[..4] == [0x4c, 0x8b, 0xd1, 0xb8] && code[8..] == SYSCALL_TAIL
}

/// `NtTerminateProcess` as it was: its first 16 bytes, then a jump to the
/// rest of it at `at + 16`.
fn original_stub(stub: &[u8; 24], at: usize) -> Vec<u8> {
    let mut code = stub[..REDIRECT_LEN].to_vec();
    code.extend([0xff, 0x25, 0, 0, 0, 0]); // jmp [rip + 0]
    code.extend(((at + REDIRECT_LEN) as u64).to_le_bytes());
    code
}

static ORIGINAL_TERMINATE: AtomicUsize = AtomicUsize::new(0);
/// Set by the first call that ends this process: after it, other threads
/// may have ended holding the heap's lock, so nothing more is reported.
static ENDING: AtomicBool = AtomicBool::new(false);

fn this_process(process: HANDLE) -> bool {
    if process.is_null() || process as isize == -1 {
        return true;
    }
    // SAFETY: plain queries; a handle that is no process gives 0.
    unsafe { GetProcessId(process) == GetCurrentProcessId() }
}

/// ntdll's `NtTerminateProcess`, redirected; `caller` is where its call
/// returns to. A null `process` ends the other threads (`ExitProcess`'s
/// first step), the current process's handle ends the process.
unsafe extern "system" fn nt_terminate_process(process: HANDLE, status: i32, caller: usize) -> i32 {
    // The title's own exit (`threads`) ends threads itself before it gets
    // here, so it is not reported either.
    if this_process(process)
        && !ENDING.swap(true, Ordering::SeqCst)
        && status != 0
        && !crate::threads::exiting()
    {
        report_ending(status, caller);
    }
    type NtTerminateProcess = unsafe extern "system" fn(HANDLE, i32) -> i32;
    // SAFETY: set before the redirection was written; the original's code.
    let original: NtTerminateProcess =
        unsafe { std::mem::transmute(ORIGINAL_TERMINATE.load(Ordering::SeqCst)) };
    // SAFETY: the caller's own arguments.
    unsafe { original(process, status) }
}

/// Windows' own modules, between `NtTerminateProcess` and whoever wanted
/// the process to end.
const SYSTEM: [&str; 4] = [
    "ntdll.dll",
    "kernelbase.dll",
    "kernel32.dll",
    "ucrtbase.dll",
];

fn report_ending(status: i32, caller: usize) {
    let frames = frames();
    let asked_by = frames
        .iter()
        .map(|&pc| pc as usize)
        .skip_while(|&pc| pc != caller)
        .find(|&pc| !in_module(pc, &SYSTEM))
        .unwrap_or(caller);
    let text = format!(
        "XTP: process ending with status {:#010x}, asked by {} (through {})\n{}",
        status as u32,
        locate(asked_by),
        locate(caller),
        stack_text(&frames)
    );
    crate::log_crash(&text);
}

fn hook_nt_terminate_process() -> Result<()> {
    let name: Vec<u16> = "ntdll.dll".encode_utf16().chain(Some(0)).collect();
    // SAFETY: a NUL-terminated name; ntdll is always loaded.
    let ntdll = unsafe { GetModuleHandleW(name.as_ptr()) };
    // SAFETY: a valid module and a NUL-terminated name.
    let at = unsafe { GetProcAddress(ntdll, c"NtTerminateProcess".as_ptr().cast()) }
        .context("ntdll has no NtTerminateProcess")? as usize;
    // SAFETY: a system-call stub is 32 bytes of ntdll's code, readable.
    let stub = unsafe { (at as *const [u8; 24]).read_unaligned() };
    if !is_syscall_stub(&stub) {
        bail!("NtTerminateProcess is not the system-call stub we know: {stub:02x?}");
    }
    ORIGINAL_TERMINATE.store(executable(&original_stub(&stub, at))?, Ordering::SeqCst);
    let to = nt_terminate_process as unsafe extern "system" fn(HANDLE, i32, usize) -> i32;
    // SAFETY: ntdll's code; no title code runs yet, and our threads do not
    // end the process while this is written.
    unsafe { patch(at, &redirect_code(Keep::R8, to as usize)) }
}

/// Installs the process-wide parts (the invalid-parameter handler and the
/// `NtTerminateProcess` report), once. A part that cannot be installed is
/// logged and left out: the title still runs.
pub(crate) fn install() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if let Err(e) = install_invalid_parameter_handler() {
            tracing::warn!("CRT invalid parameters will not be reported: {e:#}");
        }
        if let Err(e) = hook_nt_terminate_process() {
            tracing::warn!("the process ending will not be reported: {e:#}");
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MSVC's cookie check, with or without `bnd` prefixes, at offset 0 of
    /// the result, followed by `__report_gsfailure` at `report` (an offset).
    fn cookie_check(bnd: bool, report: usize) -> Vec<u8> {
        let p = |code: &mut Vec<u8>| {
            if bnd {
                code.push(0xf2);
            }
        };
        let mut code = vec![0x48, 0x3b, 0x0d, 0x10, 0x20, 0x00, 0x00];
        p(&mut code);
        code.extend([0x75, 0x00]); // to `failure`, fixed below
        let first_rel = code.len() - 1;
        code.extend(CHECK_MIDDLE);
        p(&mut code);
        code.extend([0x75, 0x00]); // to `restore`
        let second_rel = code.len() - 1;
        p(&mut code);
        code.push(0xc3);
        let restore = code.len();
        code.extend([0x48, 0xc1, 0xc9, 0x10]);
        let failure = code.len();
        p(&mut code);
        code.push(0xe9);
        let rel = report as i64 - (code.len() + 4) as i64;
        code.extend((rel as i32).to_le_bytes());
        code[first_rel] = (failure - (first_rel + 1)) as u8;
        code[second_rel] = (restore - (second_rel + 1)) as u8;
        code
    }

    /// Bytes no one would mistake for code, the same every run.
    fn noise(len: usize, mut seed: u64) -> Vec<u8> {
        (0..len)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed as u8
            })
            .collect()
    }

    #[test]
    fn finds_the_cookie_check_with_and_without_bnd_prefixes() {
        // As MSVC emits them (bytes from the compiler's runtime: rel8s 0x12
        // and 0x02 with prefixes, 0x10 and 0x01 without).
        assert_eq!(&cookie_check(true, 0x100)[7..10], [0xf2, 0x75, 0x12]);
        assert_eq!(&cookie_check(false, 0x100)[7..9], [0x75, 0x10]);
        for bnd in [true, false] {
            let mut code = noise(0x40, 1);
            code.extend(cookie_check(bnd, 0x200 - 0x40));
            code.extend(noise(0x40, 2));
            let at = 0x1_4000_1000;
            let found = find_cookie_checks(&code, at);
            assert_eq!(found.len(), 1, "bnd {bnd}");
            let failure = if bnd { 0x40 + 28 } else { 0x40 + 25 };
            assert_eq!(found[0].check, at + 0x40);
            assert_eq!(found[0].failure_jump, at + failure);
            assert_eq!(found[0].report, at + 0x200, "bnd {bnd}");
        }
        // A short jump to the report, and a report before the check.
        let mut short = cookie_check(false, 0);
        short.truncate(25);
        short.extend([0xeb, 0x10]);
        assert_eq!(
            find_cookie_checks(&short, 0x1000)[0].report,
            0x1000 + 27 + 0x10
        );
        let before = find_cookie_checks(&cookie_check(true, 0), 0x5000);
        assert_eq!(before[0].report, 0x5000);
    }

    #[test]
    fn finds_nothing_in_random_or_almost_right_code() {
        assert!(find_cookie_checks(&noise(1 << 20, 3), 0).is_empty());
        // The middle alone, in noise.
        let mut code = noise(0x100, 4);
        code[0x80..0x89].copy_from_slice(&CHECK_MIDDLE);
        assert!(find_cookie_checks(&code, 0).is_empty());
        // Every byte of the check that is not an operand, changed in turn.
        for bnd in [true, false] {
            let check = cookie_check(bnd, 0x100);
            let jump = check.len() - 5;
            for i in (0..check.len()).filter(|&i| !(3..7).contains(&i) && i <= jump) {
                let mut broken = check.clone();
                broken[i] ^= 0xff;
                let found = find_cookie_checks(&broken, 0);
                assert!(found.is_empty(), "bnd {bnd}, byte {i}: {found:?}");
            }
        }
    }

    #[test]
    fn redirections_keep_the_return_address() {
        let code = redirect_code(Keep::Rdx, 0x1122_3344_5566_7788);
        assert_eq!(code[..6], [0x48, 0x8b, 0x14, 0x24, 0x48, 0xb8]);
        assert_eq!(code[6..14], 0x1122_3344_5566_7788_u64.to_le_bytes());
        assert_eq!(code[14..], [0xff, 0xe0]);
        assert_eq!(redirect_code(Keep::R8, 0)[..4], [0x4c, 0x8b, 0x04, 0x24]);
    }

    #[test]
    fn ntdll_system_calls_have_the_shape_the_redirection_needs() {
        let name: Vec<u16> = "ntdll.dll".encode_utf16().chain(Some(0)).collect();
        // SAFETY: ntdll is loaded; another system call than the one we
        // redirect, which other tests may have redirected already.
        let stub = unsafe {
            let ntdll = GetModuleHandleW(name.as_ptr());
            let at = GetProcAddress(ntdll, c"NtYieldExecution".as_ptr().cast()).unwrap();
            (at as *const [u8; 24]).read_unaligned()
        };
        assert!(is_syscall_stub(&stub), "{stub:02x?}");
        let mut other = stub;
        other[0] = 0xe9; // a jump some other hook wrote
        assert!(!is_syscall_stub(&other));
        let copy = original_stub(&stub, 0x7ff0_0000_1000);
        assert_eq!(copy[..16], stub[..16]);
        assert_eq!(copy[16..22], [0xff, 0x25, 0, 0, 0, 0]);
        assert_eq!(copy[22..], 0x7ff0_0000_1010_u64.to_le_bytes());
    }

    /// Runs `child` (a test of this module) in a child process with the
    /// crash log at a fresh path; returns its exit code, its output and the
    /// log.
    fn in_child(child: &str) -> (Option<i32>, String, String) {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("crashes.log");
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                &format!("fastfail::tests::{child}"),
                "--nocapture",
                // With one test thread the pretty formatter prefixes the
                // child's `expect` line with its test name. Keep it separate.
                "--quiet",
            ])
            .env("RUST_TEST_THREADS", "1")
            .env("XTP_FASTFAIL_LOG", &log)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
        let text = std::fs::read_to_string(&log).unwrap_or_default();
        (out.status.code(), stdout, text)
    }

    /// The line the child printed as `expect <location>`.
    fn expected(stdout: &str) -> &str {
        stdout
            .lines()
            .find_map(|l| l.strip_prefix("expect "))
            .unwrap_or_else(|| panic!("no expectation in {stdout:?}"))
    }

    /// The crash log the parent test chose, if this runs as its child.
    fn child_log() -> Option<std::path::PathBuf> {
        let log = std::env::var_os("XTP_FASTFAIL_LOG")?;
        crate::log_crashes_to(log.clone().into());
        Some(log.into())
    }

    #[test]
    fn cookie_failure_child() {
        if child_log().is_none() {
            return;
        }
        // A page with a cookie check at 0, `__report_gsfailure` at 0x100
        // (int3s: it must never run) and the cookie at 0x800.
        let mut code = vec![0xcc_u8; 0x1000];
        let check = cookie_check(true, 0x100);
        code[..check.len()].copy_from_slice(&check);
        let disp = (0x800 - 7) as u32; // rip-relative, from the cmp's end
        code[3..7].copy_from_slice(&disp.to_le_bytes());
        code[0x800..0x808].copy_from_slice(&0x2b99_2ddf_a232_u64.to_le_bytes());
        let page = executable(&code).unwrap();
        let found = find_cookie_checks(&code, page);
        assert_eq!(found[0].report, page + 0x100);
        let to = stack_cookie_failed as extern "system" fn(usize, usize) -> ! as usize;
        // SAFETY: our page, not running.
        unsafe { patch(page + 0x100, &redirect_code(Keep::Rdx, to)) }.unwrap();
        // SAFETY: skipped, then a check of a wrong cookie: it never returns.
        unsafe {
            let (_, back) = crate::test_call::call(page, [0xbad, 0, 0], false);
            println!("expect {}", locate(back));
            crate::test_call::call(page, [0xbad, 0, 0], true);
        }
        unreachable!("the check failed and ended the process");
    }

    #[test]
    fn a_failed_cookie_check_is_reported_then_fails_fast() {
        let (code, stdout, log) = in_child("cookie_failure_child");
        assert_eq!(
            code,
            Some(0xC000_0409_u32 as i32),
            "STATUS_STACK_BUFFER_OVERRUN"
        );
        let named = format!("XTP: stack cookie check failed in {}", expected(&stdout));
        assert!(log.contains(&named), "{named:?} in {log:?}");
        assert!(log.contains("cookie 0xbad"), "{log}");
    }

    #[test]
    fn invalid_parameter_child() {
        if child_log().is_none() {
            return;
        }
        install();
        let name: Vec<u16> = "ucrtbase.dll".encode_utf16().chain(Some(0)).collect();
        // SAFETY: the host UCRT and its `strcpy_s(dest, size, source)`,
        // given a null destination, which it refuses.
        unsafe {
            let ucrt = LoadLibraryExW(name.as_ptr(), null_mut(), LOAD_LIBRARY_SEARCH_SYSTEM32);
            let strcpy_s = GetProcAddress(ucrt, c"strcpy_s".as_ptr().cast()).unwrap() as usize;
            let source = c"x".as_ptr() as u64;
            let (_, back) = crate::test_call::call(strcpy_s, [0, 0, source], false);
            println!("expect {}", locate(back));
            let (result, _) = crate::test_call::call(strcpy_s, [0, 0, source], true);
            assert_eq!(result as u32, 22, "EINVAL, once the handler returned");
        }
    }

    #[test]
    fn a_crt_invalid_parameter_names_its_caller() {
        let (code, stdout, log) = in_child("invalid_parameter_child");
        assert_eq!(code, Some(0), "{stdout}");
        let named = format!(
            "XTP: CRT invalid parameter, called from {}",
            expected(&stdout)
        );
        assert!(log.contains(&named), "{named:?} in {log:?}");
    }

    #[test]
    fn terminate_child() {
        if child_log().is_none() {
            return;
        }
        crate::report_crashes();
        let name: Vec<u16> = "kernel32.dll".encode_utf16().chain(Some(0)).collect();
        // SAFETY: `TerminateProcess(GetCurrentProcess(), 3)`: never returns.
        unsafe {
            let kernel32 = GetModuleHandleW(name.as_ptr());
            let terminate =
                GetProcAddress(kernel32, c"TerminateProcess".as_ptr().cast()).unwrap() as usize;
            let (_, back) = crate::test_call::call(terminate, [u64::MAX, 3, 0], false);
            println!("expect {}", locate(back));
            crate::test_call::call(terminate, [u64::MAX, 3, 0], true);
        }
        unreachable!("TerminateProcess returned");
    }

    #[test]
    fn ending_the_process_with_a_status_names_who_asked() {
        let (code, stdout, log) = in_child("terminate_child");
        assert_eq!(code, Some(3), "{stdout}");
        let named = format!(
            "XTP: process ending with status 0x00000003, asked by {}",
            expected(&stdout)
        );
        assert!(log.contains(&named), "{named:?} in {log:?}");
        let through = log
            .to_ascii_lowercase()
            .contains("(through kernelbase.dll+");
        assert!(through, "{log}");
    }

    #[test]
    fn exit_child() {
        if child_log().is_none() {
            return;
        }
        crate::report_crashes();
        // `ExitProcess`, whose first step ends the other threads.
        std::process::exit(7);
    }

    #[test]
    fn exiting_with_a_status_is_reported_too() {
        let (code, stdout, log) = in_child("exit_child");
        assert_eq!(code, Some(7), "{stdout}");
        // Asked by Rust's `exit`, in this test's executable.
        let named = "XTP: process ending with status 0x00000007, asked by xtp_loader-";
        assert!(log.contains(named), "{named:?} in {log:?}");
    }
}
