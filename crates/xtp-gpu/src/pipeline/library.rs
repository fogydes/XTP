//! Compiled host pipelines kept on disk, and the title pipelines worth
//! starting early (decision 0007, "Pipeline cache").
//!
//! 1. Every host pipeline goes into the host's pipeline library, named by
//!    a hash of all that its host description holds (the translated
//!    shaders' code, which changes when a translation does, our root
//!    signature, the graphics state). The library is saved under the cache
//!    folder; the host refuses a library of another adapter or driver, and
//!    an entry whose description differs, so neither is ever used.
//! 2. The title pipelines a run drew or dispatched with are listed by a key
//!    known when the title creates them (their shader containers and
//!    graphics state); the next run hands those to the workers at
//!    creation, during loading.
//! 3. Beside them, the whole descriptions of the graphics and mesh
//!    pipelines the title creates ([`super::record`]), from which `xtp
//!    precook` makes them into the library: those of the whole game after
//!    one session, and again after a translator change.
//!
//! Loads and stores of different pipelines run side by side: the library
//! is thread-safe and synchronises itself, but for two loads of the same
//! pipeline at once, which "may modify the state of that pipeline within
//! the library in a non-thread-safe manner" (D3D12's remarks on
//! `CreatePipelineLibrary`): one pipeline's name is used by one thread at a
//! time ([`NAMES`]). Serialising takes the library alone, so that its size
//! cannot change between asking for it and writing. With every call
//! serialised, the workers made some 35 pipelines a second, most of them
//! loads (a Series S session of 2026-11-25, `20261125-022031`: 1,465
//! loaded and 607 made, some 400 ms each on 14 workers), and its world
//! dropped some 900 draws a frame "being made" for a minute.
//!
//! - A save counts as done once its file is written: the pipelines it
//!   holds stop counting as added then, not before, so that a save that
//!   fails is made again by the next one.
//! - Generations: a library cannot drop an entry, and every translator
//!   change adds a name for each pipeline whose code changed, beside the
//!   old one. `xtp precook` compacts oversized files ([`rebuild`]): the
//!   library starts empty, what it asks for is taken from the generation
//!   before when that has it (loaded there, stored here) and made
//!   otherwise, and the save replaces the file with the new generation
//!   alone. What only sessions asked for is made again when one asks.
//! - One library file per translator ([`current`]): sessions of branches
//!   whose translators differ alternate on the PC, and one file compacted
//!   to one translator made the other's pipelines again at every switch
//!   (some 90,000, 13 to 23 minutes). Each translation key has its own
//!   generation (`host-pipelines-<id>.bin` and its names file), and a
//!   process opens only its own: the file it reads into memory is no
//!   larger than before. A key with no generation yet starts from the
//!   most recently opened one, its file linked (or copied), so that a
//!   change that alters few programs still finds every other pipeline by
//!   name. [`GENERATIONS_FILE`] lists them, most recently opened first;
//!   past [`GENERATIONS_KEPT`] of them or [`GENERATIONS_BYTES`] of files,
//!   the least recently opened go (never the current one). A file linked
//!   from another counts its bytes again (the budget errs on the safe
//!   side). The one file of before generations (`host-pipelines.bin`)
//!   becomes the first generation made. The list also says what a
//!   generation holds besides its translator's pipelines (seeded from
//!   another, or compacted, and its size then: [`Lineage`]), from which
//!   `xtp precook` decides its next compaction. Every write of a library
//!   file replaces it by a rename, so a generation linked from another
//!   never changes the other's file. The list is changed by one thread at a
//!   time in a process; two processes of different translators on one
//!   cache at once could each let the other's generation go: sessions and
//!   precooks on a machine run one at a time.
//! - The names the library holds: D3D12 cannot list them, so every save
//!   writes them beside the file (`host-pipelines-<id>.names`, [`Place`]),
//!   with the file's size and BLAKE3, which the next process checks before
//!   it trusts them. A library read without them (an older file, or one saved by a process that died
//!   before writing them) learns them as they load.
//! - The precook does not load a pipeline the library is known to hold
//!   ([`without_loading`]): its work is to have each pipeline in the file,
//!   and a session loads it when it needs it. That round trip was the whole
//!   cost of an unchanged pipeline after a translator change.
//! - A name the library holds but whose entry does not load for its
//!   description (StorePipeline refuses a name it holds, so before this the
//!   pipeline was compiled again in every session) gets the pipeline under
//!   an alternate name ([`SLOTS`]), which the next session loads.
//! - Such a name is remembered as broken in the names file: the precook
//!   answers a pipeline by its names only when one of them is held and not
//!   known to fail, and otherwise makes it (under the next name), so that
//!   a pipeline a load refused is not counted ready again. What no process
//!   has loaded yet cannot be known to fail without a load: the session
//!   that first finds such an entry makes the pipeline beside it and says
//!   so in the names file, and the precooks after it count it no more.
//! - The library file is mapped read-only rather than read into a heap buffer
//!   ([`Blob`]): D3D12 reads the bytes for as long as the library lives, so
//!   mapping keeps the file out of private commit and lets Windows reclaim clean
//!   pages without pagefile I/O. The file handle stays open until the view is
//!   unmapped so replacing the file by a rename never zeroes unpaged pages of
//!   the old view; where a filesystem refuses to replace a mapped file, the save
//!   writes `<id>.next` and the next open adopts it ([`adopt_next`]). Once loads
//!   go idle, [`trim`] calls `VirtualUnlock` on the view to drop its pages from
//!   the working set. `XTP_LIBRARY_MAP=0` reads into a buffer instead;
//!   `XTP_LIBRARY_TRIM=0` skips working-set trimming.

use std::cell::Cell;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock, RwLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{HSTRING, Interface, PCWSTR};

static DIR: OnceLock<PathBuf> = OnceLock::new();

/// Keeps the pipeline library and the early list in `dir` from now on.
pub fn use_pipeline_cache(dir: PathBuf) {
    let _ = DIR.set(dir);
}

/// The folder the pipeline cache is kept in, once set.
pub(crate) fn dir() -> Option<&'static std::path::Path> {
    DIR.get().map(PathBuf::as_path)
}

/// Host pipelines created, loaded from the library, and title pipelines
/// started at creation, since the start.
pub(crate) static MADE: AtomicU64 = AtomicU64::new(0);
pub(crate) static LOADED: AtomicU64 = AtomicU64::new(0);
pub(crate) static EARLY: AtomicU64 = AtomicU64::new(0);
/// Host pipelines the precook found in the library's names and did not
/// load ([`without_loading`]), since the start.
pub(crate) static KNOWN: AtomicU64 = AtomicU64::new(0);
/// Microseconds pipelines waited for the library (their name, or a save),
/// and spent loading from it, since the start (the library's line).
static LOCK_US: AtomicU64 = AtomicU64::new(0);
static LOAD_US: AtomicU64 = AtomicU64::new(0);

/// Whether the library file is mapped read-only on open ([`Blob::Mapped`]).
/// On by default; `XTP_LIBRARY_MAP=0` reads the file into a heap buffer instead.
fn mapped() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("XTP_LIBRARY_MAP").is_none_or(|v| v != "0"))
}

/// A read-only view of a library file, unmapped on drop.
///
/// The file handle (`_file`) stays open until after `UnmapViewOfFile` runs in
/// `Drop::drop`: replacing a mapped file by a rename while its handle is closed
/// causes Windows to discard unpaged pages of the old file, which then read
/// back as zeros when faulted in later.
struct Mapped {
    at: windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS,
    len: usize,
    _file: std::fs::File,
}

// SAFETY: the read-only view and held file handle can be moved between threads.
unsafe impl Send for Mapped {}
// SAFETY: the mapping is read-only and never mutated through this handle.
unsafe impl Sync for Mapped {}

impl Mapped {
    /// Maps `file` (`len` bytes, non-zero) read-only and keeps `file` open
    /// until the view is unmapped.
    fn map(file: std::fs::File, len: usize) -> Option<Mapped> {
        use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_READ, MapViewOfFile, PAGE_READONLY,
        };
        if len == 0 || len > isize::MAX as usize {
            return None;
        }
        // SAFETY: an unnamed read-only mapping over a live file handle.
        let section = unsafe {
            CreateFileMappingW(
                file.as_raw_handle(),
                std::ptr::null(),
                PAGE_READONLY,
                0,
                0,
                std::ptr::null(),
            )
        };
        if section.is_null() {
            return None;
        }
        // SAFETY: CreateFileMappingW returned a valid handle owned by us.
        let section = unsafe { OwnedHandle::from_raw_handle(section) };
        // SAFETY: the section covers the whole file and permits read-only views of len bytes.
        let at = unsafe { MapViewOfFile(section.as_raw_handle(), FILE_MAP_READ, 0, 0, len) };
        (!at.Value.is_null()).then_some(Mapped {
            at,
            len,
            _file: file,
        })
    }

    /// Maps `path` read-only while holding its file handle open.
    fn open(path: &Path) -> Option<Mapped> {
        use std::os::windows::fs::OpenOptionsExt;
        // Share read and delete (for rename-over-mapped saves, hard links, and
        // concurrent readers) but not write, so nothing can mutate the mapped
        // bytes in place while the read-only view is live.
        const FILE_SHARE_READ: u32 = 1;
        const FILE_SHARE_DELETE: u32 = 4;
        let file = std::fs::OpenOptions::new()
            .read(true)
            .share_mode(FILE_SHARE_READ | FILE_SHARE_DELETE)
            .open(path)
            .ok()?;
        let len = usize::try_from(file.metadata().ok()?.len()).ok()?;
        Mapped::map(file, len)
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: the read-only view holds `len` readable bytes until dropped,
        // and the file was opened without `FILE_SHARE_WRITE` so no handle can
        // mutate the backing file in place while `_file` is held open.
        unsafe { std::slice::from_raw_parts(self.at.Value.cast(), self.len) }
    }
}

impl Drop for Mapped {
    fn drop(&mut self) {
        // SAFETY: this object owns the view and unmaps it once; _file drops afterwards.
        unsafe { windows_sys::Win32::System::Memory::UnmapViewOfFile(self.at) };
    }
}

/// The bytes a library is made from: the host reads them for as long as
/// the library lives, so they are kept beside it.
enum Blob {
    /// No file (an empty library).
    None,
    /// Bytes held in memory (`XTP_LIBRARY_MAP=0`, tests, or a file that could
    /// not be mapped).
    Held(Vec<u8>),
    /// The file itself, mapped read-only.
    Mapped(Mapped),
}

impl Blob {
    /// Opens the library file in `place`, adopting any pending `.next` file
    /// first. Maps it read-only unless `XTP_LIBRARY_MAP=0` or mapping fails.
    fn open(place: &Place) -> Blob {
        Blob::open_in(place, mapped())
    }

    fn open_in(place: &Place, map: bool) -> Blob {
        adopt_next(place);
        if map && let Some(file) = Mapped::open(&place.library) {
            return Blob::Mapped(file);
        }
        match std::fs::read(&place.library) {
            Ok(bytes) if !bytes.is_empty() => Blob::Held(bytes),
            _ => Blob::None,
        }
    }

    fn bytes(&self) -> &[u8] {
        match self {
            Blob::None => &[],
            Blob::Held(bytes) => bytes,
            Blob::Mapped(file) => file.bytes(),
        }
    }

    fn len(&self) -> usize {
        self.bytes().len()
    }

    fn is_empty(&self) -> bool {
        self.bytes().is_empty()
    }

    fn mapped_view(&self) -> Option<(usize, usize)> {
        match self {
            Blob::Mapped(file) => Some((file.at.Value as usize, file.len)),
            Blob::None | Blob::Held(_) => None,
        }
    }
}

/// State tracking idle working-set trimming of a process-lifetime mapped view.
struct TrimState {
    /// Mapped view base address (0 when unmapped) and length in bytes.
    view_at: AtomicUsize,
    view_len: AtomicUsize,
    /// Monotonic latest timestamp (`crate::host::now_ms`) when the mapped view
    /// was touched.
    loaded_at: AtomicU64,
    /// Monotonic touch counter incremented on every touch (starts at 1 on
    /// register) so trims cannot lose a re-arm or race on equal timestamps.
    touch_seq: AtomicU64,
    /// When `begin` last evaluated the idle threshold (rate-limited to 1 Hz).
    checked_at: AtomicU64,
    /// Packs `(trimmed_seq << 1) | running`, where bit 0 is 1 while a trim
    /// worker thread is in flight and the upper 63 bits hold the latest
    /// `touch_seq` for which a trim has been claimed.
    state: AtomicU64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TrimClaim {
    at: usize,
    len: usize,
    prev_state: u64,
}

impl TrimState {
    const RUNNING: u64 = 1;

    const fn new() -> Self {
        Self {
            view_at: AtomicUsize::new(0),
            view_len: AtomicUsize::new(0),
            loaded_at: AtomicU64::new(0),
            touch_seq: AtomicU64::new(0),
            checked_at: AtomicU64::new(0),
            state: AtomicU64::new(0),
        }
    }

    fn register(&self, at: usize, len: usize, now: u64) {
        self.view_len.store(len, Ordering::Relaxed);
        self.loaded_at.store(now, Ordering::Relaxed);
        self.touch_seq.store(1, Ordering::Relaxed);
        self.state.store(0, Ordering::Relaxed);
        self.view_at.store(at, Ordering::Release);
    }

    /// Notes that the mapped view at `at` was read, advancing the idle timer
    /// monotonically and re-arming trimming if `at` matches the registered view.
    fn note_touched(&self, at: usize, now: u64) {
        if at != 0 && self.view_at.load(Ordering::Acquire) == at {
            self.loaded_at.fetch_max(now, Ordering::Release);
            self.touch_seq.fetch_add(1, Ordering::Release);
        }
    }

    /// Claims a trim when the view is mapped, has been idle for at least
    /// `quiet_ms()`, has been touched since the last trim, and no trim thread
    /// is in flight. Rate-limited to once per second.
    fn begin(
        &self,
        now_fn: impl FnOnce() -> u64,
        quiet_ms: impl FnOnce() -> u64,
    ) -> Option<TrimClaim> {
        let at = self.view_at.load(Ordering::Acquire);
        if at == 0 {
            return None;
        }
        let state = self.state.load(Ordering::Acquire);
        let seq = self.touch_seq.load(Ordering::Acquire);
        if (state & Self::RUNNING) != 0 || (state >> 1) >= seq {
            return None;
        }
        let now = now_fn();
        let checked = self.checked_at.load(Ordering::Relaxed);
        if now < checked.saturating_add(1000)
            || self
                .checked_at
                .compare_exchange(checked, now, Ordering::Relaxed, Ordering::Relaxed)
                .is_err()
        {
            return None;
        }
        let seq = self.touch_seq.load(Ordering::Acquire);
        let loaded_at = self.loaded_at.load(Ordering::Acquire);
        if now < loaded_at.saturating_add(quiet_ms()) {
            return None;
        }
        let mut cur = self.state.load(Ordering::Acquire);
        loop {
            if (cur & Self::RUNNING) != 0 || (cur >> 1) >= seq {
                return None;
            }
            // Cancel if another thread touched the view while `quiet_ms` ran.
            if self.touch_seq.load(Ordering::Acquire) != seq {
                return None;
            }
            let next = (seq << 1) | Self::RUNNING;
            match self
                .state
                .compare_exchange_weak(cur, next, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => {
                    return Some(TrimClaim {
                        at,
                        len: self.view_len.load(Ordering::Relaxed),
                        prev_state: cur,
                    });
                }
                Err(actual) => cur = actual,
            }
        }
    }

    fn finish(&self) {
        self.state.fetch_and(!Self::RUNNING, Ordering::Release);
    }

    fn abort_spawn(&self, prev_state: u64) {
        // Exclusive holder of `RUNNING`: restoring `prev_state` clears
        // `RUNNING` and restores the previous `trimmed_seq` without racing
        // with `note_touched` (which only updates `loaded_at` and `touch_seq`).
        self.state.store(prev_state, Ordering::Release);
    }
}

static TRIM: TrimState = TrimState::new();

/// Records the static library's mapped view so [`trim`] can drop its pages
/// from the working set once loads go idle. Must only be called for the
/// process-lifetime library stored in [`library`].
fn register_mapped_view(lib: &Library) {
    if let Some((at, len)) = lib.mapped_view() {
        TRIM.register(at, len, crate::host::now_ms());
    }
}

/// Notes that a read touched `blob` (load, serialization, or hashing),
/// resetting the idle trim timer if it is the registered static view.
fn note_blob_touched(blob: &Blob) {
    if let Some((at, _)) = blob.mapped_view() {
        TRIM.note_touched(at, crate::host::now_ms());
    }
}

/// Drops the mapped library file's pages from the process working set once no
/// pipeline has been loaded for a few seconds (`crate::host::ram_idle_ms`).
///
/// Pages stay file-backed in Windows' standby list and fault back in if a
/// pipeline is loaded later. Rate-limited to once a second and run on a
/// short-lived background thread so callers never stall on page table walks.
/// `XTP_LIBRARY_TRIM=0` disables trimming.
pub(crate) fn trim() {
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("XTP_LIBRARY_TRIM").is_none_or(|v| v != "0")) {
        return;
    }
    let Some(claim) = TRIM.begin(crate::host::now_ms, || {
        crate::host::ram_idle_ms(crate::host::available_physical_ram())
    }) else {
        return;
    };
    struct RunningGuard;
    impl Drop for RunningGuard {
        fn drop(&mut self) {
            TRIM.finish();
        }
    }
    let (at, len) = (claim.at, claim.len);
    let work = move || {
        let _guard = RunningGuard;
        // SAFETY: the view belongs to the static pipeline library and stays
        // mapped for the rest of the process; unlocking unlocked pages is
        // documented to fail and drop them from the working set.
        unsafe { windows_sys::Win32::System::Memory::VirtualUnlock(at as *const _, len) };
        tracing::debug!(
            bytes = len,
            "host pipeline library view trimmed from working set"
        );
    };
    if std::thread::Builder::new()
        .name("xtp-library-trim".into())
        .spawn(work)
        .is_err()
    {
        TRIM.abort_spawn(claim.prev_state);
    }
}

struct Library {
    library: ID3D12PipelineLibrary1,
    /// The blob the library was made from: it must outlive the library.
    _blob: Blob,
    /// Changes since the last save, including replacing a generation.
    added: AtomicU64,
    /// While a new generation is made ([`rebuild`]): the one before, read
    /// only.
    previous: Option<Generation>,
    /// The names this library holds, as far as this process knows.
    names: RwLock<Names>,
    /// Names learned since the names file was last written.
    names_changed: AtomicBool,
    /// The library file this library holds the same pipelines as: the one
    /// it was read from (`_blob`, accepted: hashed when first needed) or
    /// the one last saved. `None` with `_blob` not the file's (empty, or
    /// rejected by the host).
    file: Mutex<Option<FileId>>,
    blob_is_file: bool,
}

/// A generation of the library read only, the blob it was made from, and
/// the names it holds if its names file was its own.
struct Generation {
    library: ID3D12PipelineLibrary1,
    _blob: Blob,
    names: Option<Names>,
}

impl Library {
    /// A library of `blob` (read from the file when `blob_is_file`), with
    /// the names known to be in it.
    fn new(
        library: ID3D12PipelineLibrary1,
        blob: Blob,
        names: Names,
        file: Option<FileId>,
        blob_is_file: bool,
        previous: Option<Generation>,
    ) -> Library {
        Library {
            library,
            _blob: blob,
            added: AtomicU64::new(0),
            previous,
            names: RwLock::new(names),
            names_changed: AtomicBool::new(false),
            file: Mutex::new(file),
            blob_is_file,
        }
    }

    fn names(&self) -> std::sync::RwLockReadGuard<'_, Names> {
        self.names.read().unwrap_or_else(|e| e.into_inner())
    }

    fn mapped_view(&self) -> Option<(usize, usize)> {
        self._blob
            .mapped_view()
            .or_else(|| self.previous.as_ref().and_then(|g| g._blob.mapped_view()))
    }

    /// `name` is held: stored by this process, refused by StorePipeline as
    /// one the library holds, or loaded.
    fn hold(&self, name: &str) {
        let mut names = self.names.write().unwrap_or_else(|e| e.into_inner());
        if names.held.insert(name.to_owned()) {
            self.names_changed.store(true, Ordering::Relaxed);
        }
    }

    /// A load of `name` for the description of that name loaded (`ok`), or
    /// failed: the name is held and mended, or marked broken if it is known
    /// to be held (a load also fails for a name the library does not hold,
    /// which a library read without its names may not know of).
    fn loaded(&self, name: &str, ok: bool) {
        let mut names = self.names.write().unwrap_or_else(|e| e.into_inner());
        let changed = if ok {
            let held = names.held.insert(name.to_owned());
            names.broken.remove(name) | held
        } else {
            names.held.contains(name) && names.broken.insert(name.to_owned())
        };
        if changed {
            self.names_changed.store(true, Ordering::Relaxed);
        }
    }

    /// Stores `pso` under the first of `slots` (its names, [`slot`]) that
    /// does not hold an entry this pipeline's description fails to load
    /// from: `failed` says the slots whose load failed here. A slot the
    /// library holds that was not tried holds this pipeline already (stored
    /// meanwhile by another thread: a name is its description's hash), and
    /// nothing more is stored; one that was tried and failed holds another
    /// entry, and the next slot is tried. A name StorePipeline refuses is
    /// held, by the same rules.
    fn store(
        &self,
        slots: &[(String, HSTRING)],
        failed: &[bool; SLOTS],
        pso: &ID3D12PipelineState,
    ) {
        for ((name, wide), &failed) in slots.iter().zip(failed) {
            if self.names().held.contains(name) {
                if failed {
                    continue;
                }
                return;
            }
            // SAFETY: a live pipeline; a name already stored is refused,
            // which leaves the library as it was.
            match unsafe { self.library.StorePipeline(PCWSTR(wide.as_ptr()), pso) } {
                Ok(()) => {
                    self.added.fetch_add(1, Ordering::Relaxed);
                    self.hold(name);
                    return;
                }
                Err(e) if e.code() == windows::Win32::Foundation::E_INVALIDARG => {
                    self.hold(name);
                    if !failed {
                        return;
                    }
                    // Held, and its load failed here: another entry.
                    self.loaded(name, false);
                }
                Err(e) => {
                    tracing::debug!(name, error = %e, "host pipeline not stored");
                    return;
                }
            }
        }
        tracing::debug!(
            name = slots[0].0,
            "host pipeline not stored: each of its names is held"
        );
    }
}

/// A library file: its size and BLAKE3, which say whether a names file is
/// that file's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FileId {
    bytes: u64,
    blake3: [u8; 32],
}

impl FileId {
    fn of(bytes: &[u8]) -> FileId {
        FileId {
            bytes: bytes.len() as u64,
            blake3: *blake3::hash(bytes).as_bytes(),
        }
    }
}

/// The names a library holds, as far as a process knows them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Names {
    held: HashSet<String>,
    /// Of `held`, the names whose entry did not load for the description
    /// of that name when a process last loaded it (another description's
    /// entry, or one the driver refuses).
    broken: HashSet<String>,
    /// `held` is every name the library holds: it started empty, or its
    /// names file said so. Otherwise a name not in it may still be there.
    complete: bool,
}

/// The names file beside the one library file of before generations
/// ([`LIBRARY_FILE`]; a generation's is [`Place::names`]), and the first
/// line of a names file. Version 2 marks broken names ([`Names::broken`]);
/// version 1, which did not, is read too.
const NAMES_FILE: &str = "host-pipelines.names";
const NAMES_FORMAT: &str = "xtp pipeline names 2";
const NAMES_FORMAT_1: &str = "xtp pipeline names 1";
/// After a broken name in the names file.
const BROKEN: &str = " broken";

impl Names {
    /// The names of a library that holds none.
    fn empty() -> Names {
        Names {
            complete: true,
            ..Names::default()
        }
    }

    /// The names of a library read without its names file: none known.
    fn unknown() -> Names {
        Names::default()
    }

    /// Whether the library holds some name of the pipeline named `name`
    /// that is not known to fail to load ([`Names::broken`]): what the
    /// precook answers without loading ([`without_loading`]).
    fn ready_any(&self, name: &str) -> bool {
        (0..SLOTS)
            .map(|k| slot(name, k))
            .any(|n| self.held.contains(&n) && !self.broken.contains(&n))
    }

    /// The names file of a library file `file` holding these names.
    fn text(&self, file: &FileId) -> String {
        let mut held: Vec<&str> = self.held.iter().map(String::as_str).collect();
        held.sort_unstable();
        let mut text = format!(
            "{NAMES_FORMAT}\nlibrary {} {} {}\n",
            file.bytes,
            blake3::Hash::from(file.blake3).to_hex(),
            if self.complete { "complete" } else { "partial" }
        );
        for name in held {
            text.push_str(name);
            if self.broken.contains(name) {
                text.push_str(BROKEN);
            }
            text.push('\n');
        }
        text
    }

    /// The names a names file reading `text` gives the library file of
    /// `blob`; `None` if it is another file's (or not a names file).
    fn read(text: &str, blob: &[u8]) -> Option<(Names, FileId)> {
        let mut lines = text.lines();
        let marks = match lines.next()? {
            NAMES_FORMAT => true,
            NAMES_FORMAT_1 => false,
            _ => return None,
        };
        let mut head = lines.next()?.strip_prefix("library ")?.split(' ');
        let bytes: u64 = head.next()?.parse().ok()?;
        let hash = head.next()?;
        let complete = match head.next()? {
            "complete" => true,
            "partial" => false,
            _ => return None,
        };
        if head.next().is_some() || bytes != blob.len() as u64 {
            return None;
        }
        // Hashed only once the size agrees: some 0.5 s per GiB.
        let file = FileId::of(blob);
        (blake3::Hash::from(file.blake3).to_hex().as_str() == hash).then_some(())?;
        let mut names = Names {
            complete,
            ..Names::default()
        };
        for line in lines {
            let (name, broken) = match line.strip_suffix(BROKEN) {
                Some(name) if marks => (name, true),
                _ => (line, false),
            };
            is_name(name).then_some(())?;
            if broken {
                names.broken.insert(name.to_owned());
            }
            names.held.insert(name.to_owned());
        }
        Some((names, file))
    }
}

/// Whether `text` is a pipeline's name in the library ([`name`]) or one of
/// its alternates ([`slot`]).
fn is_name(text: &str) -> bool {
    let (name, k) = match text.split_once('~') {
        Some((name, k)) => (
            name,
            k.parse::<usize>().ok().filter(|k| (1..SLOTS).contains(k)),
        ),
        None => (text, Some(0)),
    };
    k.is_some() && name.len() == 32 && name.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// The names a pipeline may be stored under: its own, then alternates for
/// when the entry of its own name does not load for its description.
const SLOTS: usize = 4;

/// The `k`th name of the pipeline named `name` (`0`: its own).
fn slot(name: &str, k: usize) -> String {
    if k == 0 {
        name.to_owned()
    } else {
        format!("{name}~{k}")
    }
}

thread_local! {
    /// Set by [`without_loading`] on the precook's threads.
    static SKIP: Cell<bool> = const { Cell::new(false) };
}

/// A pipeline of this process, answered for a pipeline the library holds
/// that [`without_loading`] does not load ([`found_in`] keeps the first it
/// makes or loads).
struct StandIn(ID3D12PipelineState);
// SAFETY: D3D12 objects are free-threaded, and this one is only held.
unsafe impl Send for StandIn {}
// SAFETY: as above.
unsafe impl Sync for StandIn {}
static STAND_IN: OnceLock<StandIn> = OnceLock::new();

/// Runs `make` (a host pipeline's creation, `HostPipeline::new`) with every
/// pipeline the library's names hold answered without loading it: the
/// pipeline `make` gets is then another one of this process (the first it
/// made or loaded), so only the precook calls this, as it keeps nothing it
/// makes. A pipeline the names do not hold is loaded or made as always.
/// The precook's work is to have each pipeline in the library file; a
/// session loads it, by its name, when it needs it.
pub(crate) fn without_loading<R>(make: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            SKIP.set(self.0);
        }
    }
    let _restore = Restore(SKIP.replace(true));
    make()
}

// SAFETY: D3D12 objects are free-threaded. The library synchronises its
// calls itself but for loads of one pipeline, which the name locks
// ([`NAMES`]) keep apart, and serialising is made with the library's write
// lock held, alone.
unsafe impl Send for Library {}
// SAFETY: as above.
unsafe impl Sync for Library {}

/// Locks by a pipeline's name: one thread at a time loads or stores a
/// pipeline of a name (one of these locks: a few names share one).
static NAMES: [Mutex<()>; 64] = [const { Mutex::new(()) }; 64];

/// The name lock of `name`.
fn name_lock(name: &str) -> &'static Mutex<()> {
    let h = name.bytes().fold(0usize, |h, b| {
        h.wrapping_mul(31).wrapping_add(usize::from(b))
    });
    &NAMES[h % NAMES.len()]
}

/// The one library file of before generations per translator, and its
/// names file: taken over by the first generation made ([`seed`]).
const LIBRARY_FILE: &str = "host-pipelines.bin";
const EARLY_FILE: &str = "early-pipelines.txt";

/// Where a generation's library file and its names file are.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Place {
    library: PathBuf,
    names: PathBuf,
}

impl Place {
    /// The files of generation `id` in `dir`.
    fn of(dir: &Path, id: &str) -> Place {
        Place {
            library: dir.join(format!("host-pipelines-{id}.bin")),
            names: dir.join(format!("host-pipelines-{id}.names")),
        }
    }

    /// The one library file of before generations ([`LIBRARY_FILE`]).
    fn legacy(dir: &Path) -> Place {
        Place {
            library: dir.join(LIBRARY_FILE),
            names: dir.join(NAMES_FILE),
        }
    }

    /// Fallback path for a saved library when renaming over a mapped
    /// [`Place::library`] fails; adopted on the next open ([`adopt_next`]).
    fn next(&self) -> PathBuf {
        self.library.with_extension("next")
    }
}

/// Renames a pending `.next` library file over `place.library` if present.
fn adopt_next(place: &Place) {
    let next = place.next();
    match std::fs::rename(&next, &place.library) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => tracing::warn!(
            error = %e,
            "host pipeline library saved last could not take the old one's place"
        ),
    }
}

/// The generations of the library, one per translation key, most recently
/// opened first: one line each, `<id> own`, `<id> compacted <bytes>` or
/// `<id> from <id>` ([`Lineage`]).
const GENERATIONS_FILE: &str = "generations.txt";
const GENERATIONS_FORMAT: &str = "xtp pipeline generations 1";
/// Generations kept at most, the current one included, and the bytes their
/// files take at most (the current one is kept whatever its size). A
/// generation linked from another counts that file's bytes again.
const GENERATIONS_KEPT: usize = 4;
const GENERATIONS_BYTES: u64 = 8 << 30;
/// What a generation is seeded from when it takes over the library file of
/// before generations ([`LIBRARY_FILE`]).
pub(crate) const LEGACY: &str = "legacy";

/// A generation of the library: its translation key's, and what it holds
/// besides that translator's pipelines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Lineage {
    /// The translation key's first 16 hexadecimal digits ([`generation_id`]).
    pub(crate) id: String,
    /// The generation its file started as (linked or copied: it holds that
    /// translator's pipelines too), [`LEGACY`] for the file of before, and
    /// not compacted since; `None` for one that started empty or was
    /// compacted.
    pub(crate) from: Option<String>,
    /// Its file's size just after its last compaction, while past the
    /// precook's bound (`precook::LIBRARY_REBUILD_BYTES`); `None` without
    /// one, or below the bound then.
    pub(crate) compacted: Option<u64>,
}

impl Lineage {
    /// A generation that starts empty.
    fn own(id: &str) -> Lineage {
        Lineage {
            id: id.to_owned(),
            from: None,
            compacted: None,
        }
    }

    fn line(&self) -> String {
        match (&self.from, self.compacted) {
            (Some(from), _) => format!("{} from {from}", self.id),
            (None, Some(bytes)) => format!("{} compacted {bytes}", self.id),
            (None, None) => format!("{} own", self.id),
        }
    }

    fn parse(line: &str) -> Option<Lineage> {
        let mut words = line.split(' ');
        let id = words.next().filter(|id| is_id(id))?;
        let mut lineage = Lineage::own(id);
        match (words.next()?, words.next()) {
            ("own", None) => {}
            ("compacted", Some(bytes)) => lineage.compacted = Some(bytes.parse().ok()?),
            ("from", Some(from)) if from == LEGACY || is_id(from) => {
                lineage.from = Some(from.to_owned());
            }
            _ => return None,
        }
        words.next().is_none().then_some(lineage)
    }
}

/// Whether `text` is a generation's id: 16 lower-case hexadecimal digits.
fn is_id(text: &str) -> bool {
    text.len() == 16 && text.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// The generation of this process's translation key: its translator, its
/// settings and the host's wave sizes (`shader_cache::host_key`). The host
/// must be made.
fn generation_id() -> String {
    crate::shader_cache::host_key()[..8]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The generations listed in `dir`, most recently opened first; none if
/// the list is missing or damaged (the files are then found by
/// [`settle`]).
fn read_generations(dir: &Path) -> Vec<Lineage> {
    let Ok(text) = std::fs::read_to_string(dir.join(GENERATIONS_FILE)) else {
        return Vec::new();
    };
    let mut lines = text.lines();
    if lines.next() != Some(GENERATIONS_FORMAT) {
        return Vec::new();
    }
    let mut list: Vec<Lineage> = Vec::new();
    for line in lines {
        match Lineage::parse(line) {
            Some(g) if !list.iter().any(|h| h.id == g.id) => list.push(g),
            Some(_) => {}
            None => return Vec::new(),
        }
    }
    list
}

fn write_generations(dir: &Path, list: &[Lineage]) -> std::io::Result<()> {
    let mut text = format!("{GENERATIONS_FORMAT}\n");
    for g in list {
        text.push_str(&g.line());
        text.push('\n');
    }
    write_atomically(&dir.join(GENERATIONS_FILE), text.as_bytes())
}

/// One process at a time changes the list of generations ([`settle`],
/// [`record_compaction`]); a precook and a session do not run together.
static GENERATIONS: Mutex<()> = Mutex::new(());

/// What [`settle`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Settled {
    place: Place,
    lineage: Lineage,
    /// The generations that went, most recently opened first.
    evicted: Vec<String>,
    /// It had no file: started now (seeded, or empty). Its file is not the
    /// one an earlier process of its key left, whatever its size.
    new: bool,
}

/// Makes generation `id` in `dir` the most recently opened, its file
/// seeded from the most recently opened other one if it has none
/// ([`seed`]), and lets the oldest go past `kept` generations or `budget`
/// bytes of files. Library files of generations the list does not hold
/// (written by a process whose list was not) count as the oldest.
fn settle(dir: &Path, id: &str, kept: usize, budget: u64) -> std::io::Result<Settled> {
    let _one = GENERATIONS.lock().unwrap_or_else(|e| e.into_inner());
    std::fs::create_dir_all(dir)?;
    let mut list = read_generations(dir);
    let mut found: Vec<String> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter_map(|n| {
            let id = n
                .strip_prefix("host-pipelines-")
                .and_then(|s| s.strip_suffix(".bin").or_else(|| s.strip_suffix(".next")))?;
            is_id(id).then(|| id.to_owned())
        })
        .filter(|id| !list.iter().any(|g| g.id == *id))
        .collect();
    found.sort_unstable();
    found.dedup();
    list.extend(found.iter().map(|id| Lineage::own(id)));
    let place = Place::of(dir, id);
    adopt_next(&place);
    let listed = list.iter().position(|g| g.id == id).map(|i| list.remove(i));
    let (lineage, new) = match listed {
        Some(lineage) if place.library.exists() => (lineage, false),
        // Listed, but its file is gone: seeded again.
        _ => (
            Lineage {
                from: seed(dir, &place, &list),
                ..Lineage::own(id)
            },
            true,
        ),
    };
    list.insert(0, lineage.clone());
    let evicted = evict(dir, &mut list, kept, budget);
    write_generations(dir, &list)?;
    Ok(Settled {
        place,
        lineage,
        evicted,
        new,
    })
}

/// Starts generation `place` as the most recently opened one of `list`
/// whose file is there (linked, else copied, with its names file), or as
/// the library file of before ([`LIBRARY_FILE`], moved). What it started
/// from: `None` if it starts empty.
fn seed(dir: &Path, place: &Place, list: &[Lineage]) -> Option<String> {
    let _ = std::fs::remove_file(&place.names);
    for g in list {
        let from = Place::of(dir, &g.id);
        adopt_next(&from);
        if !from.library.exists() {
            continue;
        }
        let linked = std::fs::hard_link(&from.library, &place.library)
            .or_else(|_| std::fs::copy(&from.library, &place.library).map(|_| ()));
        match linked {
            Ok(()) => {
                // The names file says the file's size and BLAKE3: the same
                // bytes, the same names.
                let _ = std::fs::copy(&from.names, &place.names);
                tracing::info!(
                    from = g.id,
                    "host pipeline library: a new generation for this translator, from the last one opened"
                );
                return Some(g.id.clone());
            }
            Err(e) => {
                tracing::warn!(from = g.id, error = %e, "host pipeline library: a generation not seeded");
                let _ = std::fs::remove_file(&place.library);
            }
        }
    }
    let legacy = Place::legacy(dir);
    adopt_next(&legacy);
    if legacy.library.exists() {
        match std::fs::rename(&legacy.library, &place.library) {
            Ok(()) => {
                let _ = std::fs::rename(&legacy.names, &place.names);
                tracing::info!(
                    "host pipeline library: the one file of before is this translator's generation now"
                );
                return Some(LEGACY.to_owned());
            }
            Err(e) => {
                tracing::warn!(error = %e, "host pipeline library: the file of before not taken over");
            }
        }
    }
    None
}

/// Keeps the first of `list` (the current generation) and those after it
/// while they are at most `kept` and their files at most `budget` bytes;
/// the others' files go, and their lines but for a file that could not be
/// removed. A listed generation without a file goes from the list. The
/// generations that went.
fn evict(dir: &Path, list: &mut Vec<Lineage>, kept: usize, budget: u64) -> Vec<String> {
    let mut evicted = Vec::new();
    let (mut count, mut total) = (0usize, 0u64);
    list.retain(|g| {
        let place = Place::of(dir, &g.id);
        let lib = std::fs::metadata(&place.library).ok().map(|m| m.len());
        let next = std::fs::metadata(place.next()).ok().map(|m| m.len());
        let bytes = match (lib, next) {
            (None, None) => None,
            (a, b) => Some(a.unwrap_or(0).saturating_add(b.unwrap_or(0))),
        };
        let first = count == 0;
        if first {
            count += 1;
            total = bytes.unwrap_or(0);
            return true;
        }
        let Some(bytes) = bytes else {
            return false;
        };
        if count < kept && total.saturating_add(bytes) <= budget {
            count += 1;
            total = total.saturating_add(bytes);
            return true;
        }
        let remove = |path: &Path| match std::fs::remove_file(path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e),
        };
        let removed = remove(&place.library).and_then(|()| remove(&place.next()));
        match removed {
            Ok(()) => {
                let _ = std::fs::remove_file(&place.names);
                tracing::info!(
                    generation = g.id,
                    bytes,
                    "host pipeline library: the least recently opened generation goes"
                );
                evicted.push(g.id.clone());
                false
            }
            Err(e) => {
                tracing::warn!(generation = g.id, error = %e, "host pipeline library: a generation could not go");
                true
            }
        }
    });
    evicted
}

/// Notes that generation `id` in `dir` was compacted into a file of `bytes`
/// ([`rebuild`]): it holds its translator's pipelines alone, and `bytes` is
/// its baseline if past `bound`.
fn record_compaction(dir: &Path, id: &str, bytes: u64, bound: u64) -> std::io::Result<()> {
    let _one = GENERATIONS.lock().unwrap_or_else(|e| e.into_inner());
    let mut list = read_generations(dir);
    let compacted = Lineage {
        compacted: (bytes > bound).then_some(bytes),
        ..Lineage::own(id)
    };
    match list.iter_mut().find(|g| g.id == id) {
        Some(g) => *g = compacted,
        None => list.insert(0, compacted),
    }
    write_generations(dir, &list)
}

/// This process's generation of the library: where its files are, and what
/// it held when settled ([`settle`], on first use: the host must be made).
/// `None` without a cache folder.
fn current() -> Option<&'static Settled> {
    static CURRENT: OnceLock<Option<Settled>> = OnceLock::new();
    CURRENT
        .get_or_init(|| {
            let dir = DIR.get()?;
            let id = generation_id();
            match settle(dir, &id, GENERATIONS_KEPT, GENERATIONS_BYTES) {
                Ok(s) => {
                    tracing::info!(
                        generation = s.lineage.id,
                        from = s.lineage.from.as_deref(),
                        new = s.new,
                        evicted = ?s.evicted,
                        "host pipeline library generation"
                    );
                    Some(s)
                }
                Err(e) => {
                    tracing::warn!(error = %e, "host pipeline library generations not listed");
                    // Its file, as it is: not known to be the one before.
                    Some(Settled {
                        place: Place::of(dir, &id),
                        lineage: Lineage::own(&id),
                        evicted: Vec::new(),
                        new: true,
                    })
                }
            }
        })
        .as_ref()
}

/// This process's generation of the library, as settled ([`current`]), and
/// whether it started in this process (`Settled::new`); `None` without a
/// cache folder.
pub(crate) fn generation() -> Option<(Lineage, bool)> {
    current().map(|s| (s.lineage.clone(), s.new))
}

/// Title pipelines listed for an early start, at most.
const EARLY_MAX: usize = 8192;

/// The library is opened on first use: once opened, [`rebuild`] comes too
/// late.
static OPENED: AtomicBool = AtomicBool::new(false);
static REBUILD: AtomicBool = AtomicBool::new(false);
/// The library was opened as a new generation ([`new_generation`]): what
/// it saves holds this translator's pipelines alone ([`record_compaction`]),
/// even when the file before was rejected or empty.
static REBUILT: AtomicBool = AtomicBool::new(false);

fn library() -> Option<&'static RwLock<Library>> {
    static LIBRARY: OnceLock<Option<RwLock<Library>>> = OnceLock::new();
    LIBRARY
        .get_or_init(|| {
            OPENED.store(true, Ordering::SeqCst);
            let device = crate::host::host()?.device.cast::<ID3D12Device1>().ok()?;
            // This translator's generation alone.
            let place = current().map(|s| &s.place);
            let blob = place.map_or(Blob::None, Blob::open);
            let names = place.and_then(|p| std::fs::read_to_string(&p.names).ok());
            let library = if REBUILD.load(Ordering::SeqCst) {
                let library = new_generation(&device, blob, names.as_deref());
                REBUILT.store(library.is_some(), Ordering::SeqCst);
                library
            } else {
                open(&device, blob, names.as_deref())
            };
            if let Some(lib) = &library {
                register_mapped_view(lib);
            } else {
                tracing::warn!("no host pipeline library: pipelines are not kept");
            }
            library.map(RwLock::new)
        })
        .as_ref()
}

/// Makes the library a new generation, the file's library its generation
/// before (see the module): asked before the library's first use (`xtp
/// precook`'s start), else too late, which it says.
pub(crate) fn rebuild() -> bool {
    REBUILD.store(true, Ordering::SeqCst);
    !OPENED.load(Ordering::SeqCst)
}

/// The size of `place`'s latest library file on disk (preferring a pending
/// `.next` file written when replacing a mapped `.bin` fell back).
fn file_bytes_at(place: &Place) -> Option<u64> {
    std::fs::metadata(place.next())
        .or_else(|_| std::fs::metadata(&place.library))
        .ok()
        .map(|m| m.len())
}

/// The size of this process's generation's library file, if there is one
/// (settled first: [`current`]).
pub(crate) fn file_bytes() -> Option<u64> {
    file_bytes_at(&current()?.place)
}

/// The host's library of `bytes`, which must outlive it.
fn make(device: &ID3D12Device1, bytes: &[u8]) -> windows::core::Result<ID3D12PipelineLibrary1> {
    // SAFETY: the callers keep the bytes beside the library.
    unsafe {
        device
            .CreatePipelineLibrary::<ID3D12PipelineLibrary>(bytes)?
            .cast()
    }
}

/// The library in `blob`, or an empty one if the host rejects the blob (an
/// adapter or driver change, damage), with the names `names` (the names
/// file's text) says it holds if they are that blob's.
fn open(device: &ID3D12Device1, blob: Blob, names: Option<&str>) -> Option<Library> {
    if !blob.is_empty() {
        match make(device, blob.bytes()) {
            Ok(library) => {
                let read = names.and_then(|text| Names::read(text, blob.bytes()));
                tracing::info!(
                    bytes = blob.len(),
                    mapped = matches!(blob, Blob::Mapped(_)),
                    names = read.as_ref().map(|(n, _)| n.held.len()),
                    "host pipeline library loaded"
                );
                let (names, file) = match read {
                    Some((names, file)) => (names, Some(file)),
                    None => (Names::unknown(), None),
                };
                return Some(Library::new(library, blob, names, file, true, None));
            }
            Err(e) => {
                tracing::info!(error = %e, "host pipeline library rejected; starting an empty one")
            }
        }
    } else {
        tracing::info!("no host pipeline library yet; starting an empty one");
    }
    // Release any rejected mapping before creating the empty library.
    drop(blob);
    let library = make(device, &[]).ok()?;
    Some(Library::new(
        library,
        Blob::None,
        Names::empty(),
        None,
        false,
        None,
    ))
}

/// An empty library, the one in `blob` its generation before (none if the
/// host rejects it), with the names `names` says it holds.
fn new_generation(device: &ID3D12Device1, blob: Blob, names: Option<&str>) -> Option<Library> {
    let previous = if blob.is_empty() {
        None
    } else {
        match make(device, blob.bytes()) {
            Ok(library) => Some(Generation {
                names: names
                    .and_then(|text| Names::read(text, blob.bytes()))
                    .map(|(n, _)| n),
                library,
                _blob: blob,
            }),
            Err(e) => {
                tracing::info!(error = %e, "host pipeline library rejected: the new generation starts from nothing");
                drop(blob);
                None
            }
        }
    };
    tracing::info!(
        before = previous.as_ref().map_or(0, |g| g._blob.len()),
        mapped = previous
            .as_ref()
            .is_some_and(|g| matches!(g._blob, Blob::Mapped(_))),
        "host pipeline library: a new generation, from the one before"
    );
    let library = make(device, &[]).ok()?;
    let library = Library::new(library, Blob::None, Names::empty(), None, false, previous);
    // Even a generation with no surviving pipelines replaces its file.
    library.added.store(1, Ordering::Relaxed);
    Some(library)
}

/// A pipeline's name in the library: the hash of what its host
/// description holds.
pub(crate) fn name(kind: &str, root: &[u8], code: &[&[u8]], state: &str) -> String {
    let mut h = blake3::Hasher::new();
    for part in [kind.as_bytes(), root]
        .into_iter()
        .chain(code.iter().copied())
        .chain([state.as_bytes()])
    {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part);
    }
    h.finalize().to_hex()[..32].to_owned()
}

/// The pipeline named `name`: loaded from the library by `load` if it is
/// there with the same description, else made by `create` and stored.
pub(crate) fn pipeline(
    name: &str,
    load: impl Fn(&ID3D12PipelineLibrary1, PCWSTR) -> windows::core::Result<ID3D12PipelineState>,
    create: impl FnOnce() -> windows::core::Result<ID3D12PipelineState>,
) -> windows::core::Result<ID3D12PipelineState> {
    let Some(library) = library() else {
        MADE.fetch_add(1, Ordering::Relaxed);
        return create();
    };
    found_in(library, name, load, create).map(|(pso, _)| pso)
}

/// Where a pipeline came from ([`found_in`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Source {
    Library,
    /// The generation before (stored into the library).
    Previous,
    Made,
    /// Held by the library and not loaded ([`without_loading`]): the
    /// pipeline answered is a stand-in.
    Known,
}

/// [`pipeline`] in `library`, and where it came from.
///
/// A pipeline is looked for under each of its names ([`slot`]) the library
/// is known to hold, and under its own name when the library's names are
/// not all known. Made, it is stored under the first name the library does
/// not hold: an entry of its name that does not load for this description
/// keeps its name, and the pipeline goes beside it, where the next session
/// finds it.
fn found_in(
    library: &RwLock<Library>,
    name: &str,
    load: impl Fn(&ID3D12PipelineLibrary1, PCWSTR) -> windows::core::Result<ID3D12PipelineState>,
    create: impl FnOnce() -> windows::core::Result<ID3D12PipelineState>,
) -> windows::core::Result<(ID3D12PipelineState, Source)> {
    if SKIP.get()
        && let Some(stand_in) = STAND_IN.get()
        && library
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .names()
            .ready_any(name)
    {
        KNOWN.fetch_add(1, Ordering::Relaxed);
        return Ok((stand_in.0.clone(), Source::Known));
    }
    let slots: Vec<(String, HSTRING)> = (0..SLOTS)
        .map(|k| {
            let name = slot(name, k);
            let wide = HSTRING::from(&name);
            (name, wide)
        })
        .collect();
    let since = |t: std::time::Instant| t.elapsed().as_micros() as u64;
    // The name's lock, then the library's (a save takes it alone).
    let take = || {
        let asked = std::time::Instant::now();
        let held = name_lock(name).lock().unwrap_or_else(|e| e.into_inner());
        let l = library.read().unwrap_or_else(|e| e.into_inner());
        LOCK_US.fetch_add(since(asked), Ordering::Relaxed);
        (held, l)
    };
    // The slots worth a load in a library of `names`: those it holds, or
    // all of them when `names` may miss some (a library read without its
    // names file: a load that finds no entry costs little).
    let tried = |names: Option<&Names>| -> Vec<usize> {
        (0..SLOTS)
            .filter(|&k| names.is_none_or(|n| !n.complete || n.held.contains(&slots[k].0)))
            .collect()
    };
    // The slots of this library whose load failed: their entries are other
    // descriptions', and a store goes past them ([`Library::store`]).
    let mut failed = [false; SLOTS];
    let loaded = {
        let (_name, l) = take();
        let started = std::time::Instant::now();
        let mut loaded = None;
        let ours = tried(Some(&l.names()));
        for k in &ours {
            match load(&l.library, PCWSTR(slots[*k].1.as_ptr())) {
                Ok(pso) => {
                    // Learned, when the names were not all known, and
                    // mended if a load failed before.
                    l.loaded(&slots[*k].0, true);
                    loaded = Some((pso, Source::Library));
                    break;
                }
                Err(_) => {
                    failed[*k] = true;
                    l.loaded(&slots[*k].0, false);
                }
            }
        }
        if !ours.is_empty() {
            note_blob_touched(&l._blob);
        }
        if loaded.is_none()
            && let Some(previous) = &l.previous
        {
            let prev = tried(previous.names.as_ref());
            for k in &prev {
                if let Ok(pso) = load(&previous.library, PCWSTR(slots[*k].1.as_ptr())) {
                    l.store(&slots, &failed, &pso);
                    loaded = Some((pso, Source::Previous));
                    break;
                }
            }
            if !prev.is_empty() {
                note_blob_touched(&previous._blob);
            }
        }
        LOAD_US.fetch_add(since(started), Ordering::Relaxed);
        loaded
    };
    if let Some(found) = loaded {
        LOADED.fetch_add(1, Ordering::Relaxed);
        keep_stand_in(&found.0);
        return Ok(found);
    }
    // Compiled outside the locks: compiles take up to seconds.
    let pso = create()?;
    MADE.fetch_add(1, Ordering::Relaxed);
    let (_name, l) = take();
    l.store(&slots, &failed, &pso);
    keep_stand_in(&pso);
    Ok((pso, Source::Made))
}

/// For tests: whether `pso` is the stand-in [`without_loading`] answers
/// for a pipeline the library's names hold.
#[cfg(test)]
pub(crate) fn is_stand_in(pso: &ID3D12PipelineState) -> bool {
    STAND_IN.get().is_some_and(|s| s.0 == *pso)
}

/// For tests: whether this process has a pipeline library (none: every
/// pipeline is made, and [`without_loading`] answers none by its name).
#[cfg(test)]
pub(crate) fn has_library() -> bool {
    library().is_some()
}

/// Keeps `pso` as the stand-in [`without_loading`] answers, if none is.
fn keep_stand_in(pso: &ID3D12PipelineState) {
    if STAND_IN.get().is_none() {
        let _ = STAND_IN.set(StandIn(pso.clone()));
    }
}

/// What a save did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Saved {
    /// Nothing to write (nothing added, the early list as it was), or not
    /// yet time to, or no library or cache folder.
    Nothing,
    Written,
    /// The reason; what was to be written still is.
    Failed(String),
}

/// Writes the library if pipelines were added since the last save, and the
/// early list with it or if it changed, once `SAVE_EVERY` has passed (or
/// `now`). The library is serialised only when it holds something new: it
/// takes the lock every load waits for, and the cache's library had grown
/// to 2.3 GB (2026-11-25), which a session wrote again every 20 s while it
/// drew new pipelines that the library already held.
///
/// One save at a time ([`write_library`]): a worker's gives way to one
/// under way, one asked `now` (at the end of a run or of a precook) waits
/// for it.
pub(crate) fn save(now: bool) -> Saved {
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    const SAVE_EVERY: std::time::Duration = std::time::Duration::from_secs(20);
    let (Some(dir), Some(Settled { place, lineage, .. })) = (DIR.get(), current()) else {
        return Saved::Nothing;
    };
    let Some(library) = library() else {
        return if REBUILD.load(Ordering::SeqCst) {
            Saved::Failed("the host could not create the replacement pipeline library".into())
        } else {
            Saved::Nothing
        };
    };
    {
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        if !now && last.is_some_and(|t| t.elapsed() < SAVE_EVERY) {
            return Saved::Nothing;
        }
        *last = Some(std::time::Instant::now());
    }
    let mut library_bytes = None;
    let saved = write_library(library, place, now, |bytes| {
        library_bytes = bytes;
        // Under the save's lock: whether the list changed since the last
        // save that wrote it.
        let used_changed = USED_CHANGED.swap(false, Ordering::Relaxed);
        if bytes.is_none() && !used_changed {
            return Ok(false);
        }
        let used: Vec<String> = USED
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .flatten()
            .cloned()
            .collect();
        if let Err(e) = write_atomically(&dir.join(EARLY_FILE), used.join("\n").as_bytes()) {
            if used_changed {
                USED_CHANGED.store(true, Ordering::Relaxed);
            }
            return Err(e);
        }
        let (created, unkept, distinct) = super::record::created_counts();
        tracing::info!(
            library = bytes.is_some(),
            bytes = bytes.unwrap_or(0),
            early = used.len(),
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            known = KNOWN.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            created,
            created_unkept = unkept,
            created_distinct = distinct,
            recorded = super::record::kept(),
            lock_wait_ms = LOCK_US.load(Ordering::Relaxed) / 1000,
            load_ms = LOAD_US.load(Ordering::Relaxed) / 1000,
            "host pipeline library saved"
        );
        Ok(true)
    });
    if let Saved::Failed(e) = &saved {
        tracing::warn!(error = %e, "host pipeline library not saved");
    }
    // A new generation written: the list says it holds its translator's
    // pipelines alone now, and its size (the last save's, a precook's
    // final one, is the one that stays).
    if let (Saved::Written, Some(bytes)) = (&saved, library_bytes)
        && REBUILT.load(Ordering::SeqCst)
        && let Err(e) = record_compaction(
            dir,
            &lineage.id,
            bytes as u64,
            super::precook::LIBRARY_REBUILD_BYTES,
        )
    {
        tracing::warn!(error = %e, "host pipeline library generations not listed");
    }
    saved
}

/// Writes `library` into `place` if pipelines were added to it, then runs
/// `then` with its size (`None`: none written), which says whether it
/// wrote something too. The pipelines the library holds stop counting as
/// added once both are done, not before.
///
/// One at a time, from what it serialises to that count: two at once would
/// both write what was added before them, then both stop counting it (the
/// count would wrap), and share the temporary files. With `wait`, it waits
/// for one under way; without, it gives way to it ([`Saved::Nothing`]).
fn write_library(
    library: &RwLock<Library>,
    place: &Place,
    wait: bool,
    then: impl FnOnce(Option<usize>) -> std::io::Result<bool>,
) -> Saved {
    static SAVING: Mutex<()> = Mutex::new(());
    let _saving = if wait {
        SAVING.lock().unwrap_or_else(|e| e.into_inner())
    } else {
        match SAVING.try_lock() {
            Ok(saving) => saving,
            Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => return Saved::Nothing,
        }
    };
    let path = &place.library;
    let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
    let serialised = (|| -> std::io::Result<_> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let (mapped, added, names) = {
            // Alone: no store between the size and the bytes, nor between
            // the bytes and the names they hold.
            let l = library.write().unwrap_or_else(|e| e.into_inner());
            let added = l.added.load(Ordering::Relaxed);
            if added == 0 {
                // The file holds what the library holds: the names learned
                // since (loads of a library read without them) are its.
                let names = l
                    .names_changed
                    .swap(false, Ordering::Relaxed)
                    .then(|| l.names().clone());
                drop(l);
                if let Some(names) = names {
                    write_names(library, place, &names, None);
                }
                return Ok(None);
            }
            l.names_changed.store(false, Ordering::Relaxed);
            let names = l.names().clone();
            let file = LibraryFile::serialize(&tmp, &l.library);
            note_blob_touched(&l._blob);
            (file?, added, names)
        };
        let size = mapped.size;
        mapped.flush()?;
        // SAFETY: the view holds `size` bytes the library wrote, and lives
        // until `mapped` is dropped below.
        let file =
            FileId::of(unsafe { std::slice::from_raw_parts(mapped.view.Value.cast(), size) });
        drop(mapped);
        replace_library(place, &tmp)?;
        write_names(library, place, &names, Some(file));
        Ok(Some((size, added)))
    })();
    let serialised = match serialised {
        Ok(result) => result,
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            return Saved::Failed(e.to_string());
        }
    };
    let written = then(serialised.map(|(size, _)| size));
    match (written, serialised) {
        (Err(e), _) => Saved::Failed(e.to_string()),
        (Ok(_), Some((_, added))) => {
            // Those stored meanwhile still count, for the next save.
            let l = library.read().unwrap_or_else(|e| e.into_inner());
            l.added.fetch_sub(added, Ordering::Relaxed);
            Saved::Written
        }
        (Ok(true), None) => Saved::Written,
        (Ok(false), None) => Saved::Nothing,
    }
}

/// Writes the names file of the library file in `place`: `names`, held by the
/// file just written (`file`, its identity), or by the file the library was
/// read from or last saved to (`None`). The names stay to write again if it
/// fails, or if that file is not known (a library that did not read back),
/// and a file whose names are not written is read without them: what the
/// names file says is always its library file's.
fn write_names(library: &RwLock<Library>, place: &Place, names: &Names, file: Option<FileId>) {
    let l = library.read().unwrap_or_else(|e| e.into_inner());
    let file = {
        let mut known = l.file.lock().unwrap_or_else(|e| e.into_inner());
        if file.is_some() {
            *known = file;
        } else if known.is_none() && l.blob_is_file {
            *known = Some(FileId::of(l._blob.bytes()));
            note_blob_touched(&l._blob);
        }
        *known
    };
    let written = file.map(|file| write_atomically(&place.names, names.text(&file).as_bytes()));
    match written {
        Some(Ok(())) => {}
        Some(Err(e)) => {
            tracing::warn!(error = %e, "host pipeline names not written");
            l.names_changed.store(true, Ordering::Relaxed);
        }
        None => l.names_changed.store(true, Ordering::Relaxed),
    }
}

/// D3D12 needs one contiguous output buffer. A writable view of the
/// temporary file avoids a second library-sized allocation in the heap.
struct LibraryFile {
    file: std::fs::File,
    view: windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS,
    size: usize,
}

impl LibraryFile {
    fn serialize(
        path: &std::path::Path,
        library: &ID3D12PipelineLibrary1,
    ) -> std::io::Result<Self> {
        use std::os::windows::{
            fs::OpenOptionsExt,
            io::{AsRawHandle, FromRawHandle, OwnedHandle},
        };
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, PAGE_READWRITE,
        };

        // SAFETY: a live library under its exclusive lock.
        let size = unsafe { library.GetSerializedSize() };
        if size > isize::MAX as usize {
            return Err(std::io::Error::other(
                "pipeline library is too large to map",
            ));
        }
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .share_mode(0)
            .open(path)?;
        file.set_len(size as u64)?;
        // SAFETY: an unnamed mapping of this file, extended to exactly size.
        let section = unsafe {
            CreateFileMappingW(
                file.as_raw_handle(),
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                0,
                std::ptr::null(),
            )
        };
        if section.is_null() {
            return Err(std::io::Error::last_os_error());
        }
        // SAFETY: the newly created handle is ours; the view retains its
        // section independently once this handle closes.
        let section = unsafe { OwnedHandle::from_raw_handle(section) };
        // SAFETY: the section covers the entire file and permits writes.
        let view = unsafe { MapViewOfFile(section.as_raw_handle(), FILE_MAP_WRITE, 0, 0, size) };
        if view.Value.is_null() {
            return Err(std::io::Error::last_os_error());
        }
        let mapped = Self { file, view, size };
        // SAFETY: the exclusively opened file has one writable view of size bytes.
        unsafe {
            let bytes = std::slice::from_raw_parts_mut(view.Value.cast(), size);
            library.Serialize(bytes).map_err(std::io::Error::other)?;
        }
        Ok(mapped)
    }

    fn flush(&self) -> std::io::Result<()> {
        use windows_sys::Win32::System::Memory::FlushViewOfFile;
        // SAFETY: this view and its file stay open through the flush.
        if unsafe { FlushViewOfFile(self.view.Value, self.size) } == 0 {
            return Err(std::io::Error::last_os_error());
        }
        self.file.sync_all()
    }
}

impl Drop for LibraryFile {
    fn drop(&mut self) {
        // SAFETY: this object owns the sole view and unmaps it exactly once.
        unsafe { windows_sys::Win32::System::Memory::UnmapViewOfFile(self.view) };
    }
}

/// Renames the written library `tmp` over `place.library`. Windows 11 allows
/// renaming over a mapped file, and our mapping keeps reading the old contents
/// while its file handle stays open ([`Mapped`]). Where a filesystem refuses
/// to replace a mapped file, the new library is left beside it as `.next` and
/// adopted on the next open ([`adopt_next`]). Returns whether the file was
/// replaced in place.
fn replace_library(place: &Place, tmp: &Path) -> std::io::Result<bool> {
    match std::fs::rename(tmp, &place.library) {
        Ok(()) => {
            if let Err(e) = std::fs::remove_file(place.next())
                && e.kind() != std::io::ErrorKind::NotFound
            {
                tracing::warn!(
                    next = %place.next().display(),
                    error = %e,
                    "stale host pipeline library .next file could not be removed"
                );
            }
            Ok(true)
        }
        Err(_) => {
            std::fs::rename(tmp, place.next())?;
            tracing::debug!(
                next = %place.next().display(),
                "host pipeline library left as .next for the next open"
            );
            Ok(false)
        }
    }
}

/// Writes `bytes` to a temporary file beside `path`, then renames it over
/// `path`: a reader never sees a part. One save at a time uses it
/// ([`save`]).
fn write_atomically(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
}

/// Title pipeline keys drawn or dispatched with (this run and the last).
static USED: Mutex<Option<HashSet<String>>> = Mutex::new(None);
static USED_CHANGED: AtomicBool = AtomicBool::new(false);

fn used() -> std::sync::MutexGuard<'static, Option<HashSet<String>>> {
    let mut used = USED.lock().unwrap_or_else(|e| e.into_inner());
    if used.is_none() {
        let listed = DIR
            .get()
            .and_then(|d| std::fs::read_to_string(d.join(EARLY_FILE)).ok())
            .unwrap_or_default();
        let set: HashSet<String> = listed
            .lines()
            .filter(|l| !l.is_empty())
            .map(str::to_owned)
            .collect();
        tracing::info!(pipelines = set.len(), "title pipelines to start early");
        *used = Some(set);
    }
    used
}

/// Whether the title pipeline with `key` was drawn with in an earlier run.
pub(crate) fn wanted_early(key: &str) -> bool {
    used().as_ref().is_some_and(|s| s.contains(key))
}

/// The title pipeline with `key` was drawn or dispatched with.
pub(crate) fn note_used(key: &str) {
    let mut used = used();
    let set = used.get_or_insert_with(HashSet::new);
    if set.len() < EARLY_MAX && set.insert(key.to_owned()) {
        USED_CHANGED.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A generation's id, for the tests that save into a folder.
    const ID: &str = "0123456789abcdef";

    /// Generation [`ID`]'s files in `dir`.
    fn place(dir: &Path) -> Place {
        Place::of(dir, ID)
    }

    #[test]
    fn names_change_with_what_goes_into_the_pipeline() {
        let a = name("graphics", b"root", &[b"vs", b"ps"], "state");
        assert_eq!(a, name("graphics", b"root", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps2"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps"], "state2"));
        assert_ne!(a, name("graphics", b"root2", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("mesh", b"root", &[b"vs", b"ps"], "state"));
        // Parts do not run into each other.
        assert_ne!(a, name("graphics", b"root", &[b"vsp", b"s"], "state"));
    }

    #[test]
    fn a_rejected_blob_gives_an_empty_library() {
        let test = "pipeline::library::tests::a_rejected_blob_gives_an_empty_library";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let device = gpu.device.cast::<ID3D12Device1>().expect("ID3D12Device1");
        // The layer reports the blob it cannot read; the library falls back.
        let library = crate::testgpu::expect_layer_errors(|| {
            open(
                &device,
                Blob::Held(b"not a pipeline library".repeat(16)),
                None,
            )
            .expect("an empty library")
        });
        // SAFETY: a plain query on a live library.
        assert!(unsafe { library.library.GetSerializedSize() } > 0);
        assert!(library._blob.is_empty());
    }

    /// Host pipelines of a few names, each asked for by several threads at
    /// once: every ask gets its pipeline (loads of one name kept apart,
    /// stores beside loads), most from the library, which still serialises
    /// after.
    #[test]
    fn pipelines_are_loaded_and_stored_side_by_side() {
        let test = "pipeline::library::tests::pipelines_are_loaded_and_stored_side_by_side";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        if library().is_none() {
            return;
        }
        let shaders: Vec<_> = [1, 2, 4, 8]
            .map(|n| {
                super::super::precook::tests::translated(
                    "compute",
                    xtp_shader::HwStage::Cs,
                    Some(&format!("[numthreads({n}, 1, 1)] void main() {{}}")),
                    windows::core::s!("cs_5_1"),
                    format!("library test: compute {n}").as_bytes(),
                )
            })
            .into();
        let loaded = LOADED.load(Ordering::Relaxed);
        std::thread::scope(|s| {
            for t in 0..8 {
                let shaders = &shaders;
                s.spawn(move || {
                    for k in 0..12 {
                        let shader = &shaders[(t + k) % shaders.len()];
                        let made = super::super::host::HostPipeline::new(
                            std::slice::from_ref(shader),
                            None,
                        );
                        assert!(made.is_some(), "a pipeline for {shader:?}");
                    }
                });
            }
        });
        // 96 asks of 4 pipelines: each compiled by at most the 8 threads.
        assert!(LOADED.load(Ordering::Relaxed) - loaded >= 96 - 4 * 8);
        let l = library()
            .unwrap()
            .write()
            .unwrap_or_else(|e| e.into_inner());
        // SAFETY: a plain query on a live library, alone.
        assert!(unsafe { l.library.GetSerializedSize() } > 0);
    }

    /// A compute kernel of `n` threads whose code holds its root signature.
    struct Kernel {
        code: Vec<u8>,
        root: ID3D12RootSignature,
    }

    fn kernel(device: &ID3D12Device, n: u32) -> Kernel {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        let source = format!("[RootSignature(\"\")] [numthreads({n}, 1, 1)] void main() {{}}");
        let mut blob = None;
        // SAFETY: the source lives through compilation and the blob owns
        // its bytes.
        let code = unsafe {
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                windows::core::PCSTR::null(),
                None,
                None,
                windows::core::s!("main"),
                windows::core::s!("cs_5_1"),
                0,
                0,
                &mut blob,
                None,
            )
            .unwrap();
            let blob = blob.unwrap();
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
                .to_vec()
        };
        // SAFETY: the code holds the root signature.
        let root = unsafe { device.CreateRootSignature(0, &code) }.unwrap();
        Kernel { code, root }
    }

    /// `k`'s pipeline named `name` in `library`, and where it came from.
    fn ask(library: &RwLock<Library>, device: &ID3D12Device, k: &Kernel, name: &str) -> Source {
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            // SAFETY: a copy of a live root signature, not released by the
            // description.
            pRootSignature: unsafe { std::mem::transmute_copy(&k.root) },
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: k.code.as_ptr().cast(),
                BytecodeLength: k.code.len(),
            },
            ..Default::default()
        };
        found_in(
            library,
            name,
            // SAFETY: the description points at live data.
            |l, name| unsafe { l.LoadComputePipeline(name, &desc) },
            // SAFETY: as above.
            || unsafe { device.CreateComputePipelineState(&desc) },
        )
        .unwrap()
        .1
    }

    /// A save that fails leaves the library's pipelines to save: the next
    /// save writes them, and only then do they stop counting as added.
    #[test]
    fn a_failed_save_is_made_again() {
        let test = "pipeline::library::tests::a_failed_save_is_made_again";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let device = host.device.cast::<ID3D12Device1>().unwrap();
        let library = RwLock::new(open(&device, Blob::None, None).unwrap());
        library.read().unwrap().added.store(2, Ordering::Relaxed);
        let dir = tempfile::tempdir().unwrap();
        // A file where the folder should be.
        let not_a_folder = dir.path().join("pipelines");
        std::fs::write(&not_a_folder, b"in the way").unwrap();
        assert!(matches!(
            write_library(&library, &place(&not_a_folder), true, |_| Ok(true)),
            Saved::Failed(_)
        ));
        assert_eq!(library.read().unwrap().added.load(Ordering::Relaxed), 2);
        // The early list not written: neither is the save.
        let good = dir.path().join("cache");
        assert!(matches!(
            write_library(&library, &place(&good), true, |_| Err(
                std::io::Error::other("full")
            )),
            Saved::Failed(_)
        ));
        assert_eq!(library.read().unwrap().added.load(Ordering::Relaxed), 2);
        let mut size = None;
        assert_eq!(
            write_library(&library, &place(&good), true, |bytes| {
                size = bytes;
                Ok(true)
            }),
            Saved::Written
        );
        assert_eq!(library.read().unwrap().added.load(Ordering::Relaxed), 0);
        let written = std::fs::metadata(place(&good).library).unwrap().len();
        assert_eq!(Some(written as usize), size);
        assert_eq!(
            write_library(&library, &place(&good), true, |_| Ok(false)),
            Saved::Nothing,
            "nothing new"
        );
    }

    /// Two saves at once: one asked to wait writes after the one under way,
    /// which counts what it wrote first, so that what was added stops
    /// counting once and the count does not wrap; one that does not wait
    /// gives way.
    #[test]
    fn saves_at_once_count_what_they_wrote_once() {
        let test = "pipeline::library::tests::saves_at_once_count_what_they_wrote_once";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let device = host.device.cast::<ID3D12Device1>().unwrap();
        let library = RwLock::new(open(&device, Blob::None, None).unwrap());
        library.read().unwrap().added.store(2, Ordering::Relaxed);
        let dir = tempfile::tempdir().unwrap();
        let (inside, go) = (std::sync::Barrier::new(2), std::sync::Barrier::new(2));
        std::thread::scope(|s| {
            let first = s.spawn(|| {
                write_library(&library, &place(dir.path()), true, |_| {
                    inside.wait();
                    go.wait();
                    Ok(true)
                })
            });
            inside.wait();
            // The first is between its write and its count.
            assert_eq!(
                write_library(&library, &place(dir.path()), false, |_| Ok(true)),
                Saved::Nothing,
                "gives way"
            );
            let second =
                s.spawn(|| write_library(&library, &place(dir.path()), true, |_| Ok(false)));
            go.wait();
            assert_eq!(first.join().unwrap(), Saved::Written);
            assert_eq!(
                second.join().unwrap(),
                Saved::Nothing,
                "nothing new by then"
            );
        });
        assert_eq!(library.read().unwrap().added.load(Ordering::Relaxed), 0);
    }

    /// A new generation takes what is asked for from the one before, and
    /// keeps only that: what was not asked for is made again later.
    #[test]
    fn a_new_generation_keeps_what_is_asked_for() {
        let test = "pipeline::library::tests::a_new_generation_keeps_what_is_asked_for";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let device = host.device.cast::<ID3D12Device1>().unwrap();
        let (a, b) = (kernel(&host.device, 3), kernel(&host.device, 5));
        let dir = tempfile::tempdir().unwrap();
        let saved = |library: &RwLock<Library>| {
            assert_eq!(
                write_library(library, &place(dir.path()), true, |_| Ok(true)),
                Saved::Written
            );
            std::fs::read(place(dir.path()).library).unwrap()
        };
        let old = RwLock::new(open(&device, Blob::None, None).unwrap());
        assert_eq!(ask(&old, &host.device, &a, "a"), Source::Made);
        assert_eq!(ask(&old, &host.device, &b, "b"), Source::Made);
        // Even a matching name must not return a pipeline whose
        // description has different bytecode.
        let mismatch = crate::testgpu::expect_layer_errors(|| ask(&old, &host.device, &b, "a"));
        assert_eq!(mismatch, Source::Made);
        assert_eq!(ask(&old, &host.device, &a, "a"), Source::Library);
        let new = RwLock::new(new_generation(&device, Blob::Held(saved(&old)), None).unwrap());
        assert_eq!(ask(&new, &host.device, &a, "a"), Source::Previous);
        assert_eq!(ask(&new, &host.device, &a, "a"), Source::Library);
        let next = RwLock::new(open(&device, Blob::Held(saved(&new)), None).unwrap());
        assert_eq!(ask(&next, &host.device, &a, "a"), Source::Library);
        assert_eq!(
            ask(&next, &host.device, &b, "b"),
            Source::Made,
            "not asked for"
        );
        let empty = RwLock::new(new_generation(&device, Blob::Held(saved(&next)), None).unwrap());
        let reopened = RwLock::new(open(&device, Blob::Held(saved(&empty)), None).unwrap());
        assert_eq!(
            ask(&reopened, &host.device, &a, "a"),
            Source::Made,
            "nothing survived"
        );
    }

    /// A names file is its library file's: read back with the names it was
    /// written with, and refused for another file (another size, or other
    /// bytes of the same size), another format or a line that is no name.
    #[test]
    fn names_are_read_back_for_their_own_file_only() {
        let blob = b"a serialised pipeline library".repeat(8);
        let a = "0123456789abcdef0123456789abcdef";
        let names = Names {
            held: [a.to_owned(), format!("{a}~1")].into(),
            broken: [a.to_owned()].into(),
            complete: true,
        };
        let file = FileId::of(&blob);
        let text = names.text(&file);
        assert_eq!(Names::read(&text, &blob), Some((names.clone(), file)));
        // A file of the first version marks none broken, and its marks are
        // no names.
        let first = text.replace(NAMES_FORMAT, NAMES_FORMAT_1);
        assert_eq!(Names::read(&first, &blob), None, "a mark in version 1");
        let first = first.replace(BROKEN, "");
        let unmarked = Names {
            broken: HashSet::new(),
            ..names.clone()
        };
        assert_eq!(Names::read(&first, &blob), Some((unmarked, file)));
        let partial = Names {
            complete: false,
            ..names.clone()
        };
        assert_eq!(
            Names::read(&partial.text(&file), &blob).map(|(n, _)| n),
            Some(partial)
        );
        let mut other = blob.clone();
        other[3] ^= 1;
        assert_eq!(Names::read(&text, &other), None, "other bytes");
        assert_eq!(Names::read(&text, &blob[1..]), None, "another size");
        assert_eq!(Names::read(&format!("{text}{a}~4\n"), &blob), None);
        assert_eq!(Names::read(&format!("{text}not a name\n"), &blob), None);
        assert_eq!(
            Names::read(&text.replace(NAMES_FORMAT, "xtp pipeline names 0"), &blob),
            None
        );
        assert_eq!(Names::read("", &blob), None);
        assert!(is_name(a) && is_name(&format!("{a}~3")));
        assert!(!is_name(&format!("{a}~0")) && !is_name(&a.to_uppercase()) && !is_name("a"));
        assert!(names.ready_any(a) && !names.ready_any(&a.replace('0', "f")));
        // Ready by a name held and not broken only.
        let only_broken = Names {
            held: [a.to_owned()].into(),
            ..names.clone()
        };
        assert!(!only_broken.ready_any(a), "its one name is broken");
        assert_eq!(
            Names::read(&format!("{text}{a}~2{BROKEN}{BROKEN}\n"), &blob),
            None
        );
    }

    /// An entry of a pipeline's name that does not load for its
    /// description (here: another kernel stored under that name) used to
    /// make the pipeline again each time, since StorePipeline refuses a
    /// name the library holds. The pipeline is stored under the next name
    /// instead, and found there from then on, after a save too; the names
    /// file says which names the saved library holds.
    #[test]
    fn a_name_that_does_not_load_gets_the_next_one() {
        let test = "pipeline::library::tests::a_name_that_does_not_load_gets_the_next_one";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let device = host.device.cast::<ID3D12Device1>().unwrap();
        let (a, b) = (kernel(&host.device, 3), kernel(&host.device, 5));
        let name = "00112233445566778899aabbccddeeff";
        let library = RwLock::new(open(&device, Blob::None, None).unwrap());
        assert_eq!(ask(&library, &host.device, &a, name), Source::Made);
        let other =
            || crate::testgpu::expect_layer_errors(|| ask(&library, &host.device, &b, name));
        assert_eq!(other(), Source::Made, "the entry of its name is another's");
        assert_eq!(other(), Source::Library, "found under the next name");
        assert_eq!(ask(&library, &host.device, &a, name), Source::Library);
        let held = library.read().unwrap().names().clone();
        assert_eq!(held.held, [name.to_owned(), slot(name, 1)].into());
        assert!(held.complete, "started empty");

        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            write_library(&library, &place(dir.path()), true, |_| Ok(true)),
            Saved::Written
        );
        let blob = std::fs::read(place(dir.path()).library).unwrap();
        let text = std::fs::read_to_string(place(dir.path()).names).unwrap();
        assert_eq!(Names::read(&text, &blob), Some((held, FileId::of(&blob))));
        let reopened = RwLock::new(open(&device, Blob::Held(blob), Some(&text)).unwrap());
        let other =
            || crate::testgpu::expect_layer_errors(|| ask(&reopened, &host.device, &b, name));
        assert_eq!(other(), Source::Library, "repaired for the next process");
        assert_eq!(
            reopened.read().unwrap().added.load(Ordering::Relaxed),
            0,
            "nothing made"
        );
    }

    /// The precook does not load a pipeline the library's names hold
    /// ([`without_loading`]); one they do not hold is made and stored as
    /// always. A library read without its names learns those that load,
    /// and a save writes the names learned.
    #[test]
    fn the_precook_does_not_load_what_the_names_hold() {
        let test = "pipeline::library::tests::the_precook_does_not_load_what_the_names_hold";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let device = host.device.cast::<ID3D12Device1>().unwrap();
        let (a, b) = (kernel(&host.device, 3), kernel(&host.device, 5));
        let (name_a, name_b) = (
            "a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0",
            "b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0",
        );
        let dir = tempfile::tempdir().unwrap();
        let library = RwLock::new(open(&device, Blob::None, None).unwrap());
        assert_eq!(ask(&library, &host.device, &a, name_a), Source::Made);
        assert_eq!(
            write_library(&library, &place(dir.path()), true, |_| Ok(true)),
            Saved::Written
        );
        let blob = std::fs::read(place(dir.path()).library).unwrap();
        let text = std::fs::read_to_string(place(dir.path()).names).unwrap();

        // Read with its names: held, so not loaded; another is made. (The
        // count is the process's: other tests' precooks, the replay's,
        // may add to it meanwhile. `Source::Known` says this ask's.)
        let known = KNOWN.load(Ordering::Relaxed);
        let named = RwLock::new(open(&device, Blob::Held(blob.clone()), Some(&text)).unwrap());
        assert_eq!(
            without_loading(|| ask(&named, &host.device, &a, name_a)),
            Source::Known
        );
        assert!(KNOWN.load(Ordering::Relaxed) > known, "counted");
        assert_eq!(
            without_loading(|| ask(&named, &host.device, &b, name_b)),
            Source::Made
        );
        assert_eq!(
            ask(&named, &host.device, &a, name_a),
            Source::Library,
            "a session loads it"
        );

        // Read without its names (an older file): loaded once, then known.
        let unnamed = RwLock::new(open(&device, Blob::Held(blob), None).unwrap());
        assert_eq!(
            without_loading(|| ask(&unnamed, &host.device, &a, name_a)),
            Source::Library
        );
        assert_eq!(
            without_loading(|| ask(&unnamed, &host.device, &a, name_a)),
            Source::Known
        );
        // Nothing added: the save writes the names learned alone, for the
        // file as it is.
        std::fs::remove_file(place(dir.path()).names).unwrap();
        assert_eq!(
            write_library(&unnamed, &place(dir.path()), true, |_| Ok(false)),
            Saved::Nothing
        );
        let blob = std::fs::read(place(dir.path()).library).unwrap();
        let text = std::fs::read_to_string(place(dir.path()).names).unwrap();
        let (learned, _) = Names::read(&text, &blob).expect("the file's names");
        assert_eq!(
            (learned.held, learned.complete),
            ([name_a.to_owned()].into(), false)
        );
    }

    /// The Codex review of 12556d6: the precook answered a pipeline by any
    /// name the library held, before the driver, even one whose entry does
    /// not load for its description. A load that fails now marks the name
    /// broken, in the names file too, and the precook does not answer by a
    /// broken name: it makes the pipeline under the next one, by which it
    /// answers from then on.
    #[test]
    fn the_precook_does_not_answer_by_a_broken_name() {
        let test = "pipeline::library::tests::the_precook_does_not_answer_by_a_broken_name";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let device = host.device.cast::<ID3D12Device1>().unwrap();
        let (a, b) = (kernel(&host.device, 3), kernel(&host.device, 7));
        let name = "c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0";
        let dir = tempfile::tempdir().unwrap();
        let saved = |library: &RwLock<Library>| {
            let wrote = write_library(library, &place(dir.path()), true, |_| Ok(true));
            assert!(!matches!(wrote, Saved::Failed(_)), "{wrote:?}");
            let blob = std::fs::read(place(dir.path()).library).unwrap();
            let text = std::fs::read_to_string(place(dir.path()).names).unwrap();
            (blob, text)
        };
        // The library's entry of the name is another description's.
        let library = RwLock::new(open(&device, Blob::None, None).unwrap());
        assert_eq!(ask(&library, &host.device, &a, name), Source::Made);
        let (blob, text) = saved(&library);

        // A session's load of `b` by that name fails, and its compile too
        // (nothing goes under the next name): the names file says broken.
        let session = RwLock::new(open(&device, Blob::Held(blob.clone()), Some(&text)).unwrap());
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            // SAFETY: a copy of a live root signature, not released by the
            // description.
            pRootSignature: unsafe { std::mem::transmute_copy(&b.root) },
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: b.code.as_ptr().cast(),
                BytecodeLength: b.code.len(),
            },
            ..Default::default()
        };
        let refused = crate::testgpu::expect_layer_errors(|| {
            found_in(
                &session,
                name,
                // SAFETY: the description points at live data.
                |l, name| unsafe { l.LoadComputePipeline(name, &desc) },
                || Err(windows::Win32::Foundation::E_FAIL.into()),
            )
        });
        assert!(refused.is_err());
        let (blob, text) = saved(&session);
        let (names, _) = Names::read(&text, &blob).expect("its names");
        assert_eq!(names.broken, [name.to_owned()].into(), "{text}");
        assert!(!names.ready_any(name));

        // The precook does not answer by it: it loads, then makes `b`,
        // under the next name.
        let precook = RwLock::new(open(&device, Blob::Held(blob), Some(&text)).unwrap());
        let made = crate::testgpu::expect_layer_errors(|| {
            without_loading(|| ask(&precook, &host.device, &b, name))
        });
        assert_eq!(made, Source::Made, "not answered by a broken name");
        assert_eq!(
            without_loading(|| ask(&precook, &host.device, &b, name)),
            Source::Known,
            "by the next name"
        );
        let (blob, text) = saved(&precook);
        let (names, _) = Names::read(&text, &blob).expect("its names");
        assert_eq!(
            (names.held, names.broken),
            (
                [name.to_owned(), slot(name, 1)].into(),
                [name.to_owned()].into()
            )
        );

        // A session after it loads `b` by the next name; `a` still loads
        // by its own, which mends it.
        let next = RwLock::new(open(&device, Blob::Held(blob), Some(&text)).unwrap());
        let found = crate::testgpu::expect_layer_errors(|| ask(&next, &host.device, &b, name));
        assert_eq!(found, Source::Library);
        assert_eq!(ask(&next, &host.device, &a, name), Source::Library);
        assert!(next.read().unwrap().names().broken.is_empty(), "mended");
    }

    const A: &str = "aaaaaaaaaaaaaaaa";
    const B: &str = "bbbbbbbbbbbbbbbb";
    const C: &str = "cccccccccccccccc";
    const D: &str = "dddddddddddddddd";
    const E: &str = "eeeeeeeeeeeeeeee";

    /// Saves `bytes` as `place`'s library file and `names` as its names,
    /// the way a save does: by a rename over the file (never through a
    /// link).
    fn save_as(place: &Place, bytes: &[u8], names: &str) {
        write_atomically(&place.library, bytes).unwrap();
        write_atomically(&place.names, names.as_bytes()).unwrap();
    }

    fn ids(list: &[Lineage]) -> Vec<&str> {
        list.iter().map(|g| g.id.as_str()).collect()
    }

    /// A generation's line in the list reads back as it was written; a line
    /// that is none is refused, and so is the whole list then.
    #[test]
    fn generation_lines_read_back() {
        let own = Lineage::own(ID);
        let from = Lineage {
            from: Some(B.into()),
            ..Lineage::own(ID)
        };
        let legacy = Lineage {
            from: Some(LEGACY.into()),
            ..Lineage::own(ID)
        };
        let compacted = Lineage {
            compacted: Some(3 << 30),
            ..Lineage::own(ID)
        };
        for g in [&own, &from, &legacy, &compacted] {
            assert_eq!(Lineage::parse(&g.line()).as_ref(), Some(g), "{}", g.line());
        }
        assert_eq!(own.line(), format!("{ID} own"));
        assert_eq!(from.line(), format!("{ID} from {B}"));
        assert_eq!(compacted.line(), format!("{ID} compacted {}", 3u64 << 30));
        for bad in [
            String::new(),
            ID.to_owned(),
            format!("{ID} own more"),
            format!("{} own", ID.to_uppercase()),
            format!("{ID}0 own"),
            format!("{ID} from elsewhere"),
            format!("{ID} from"),
            format!("{ID} compacted many"),
            format!("{ID} compacted"),
            format!("{ID} lost"),
        ] {
            assert_eq!(Lineage::parse(&bad), None, "{bad}");
        }
        let dir = tempfile::tempdir().unwrap();
        write_generations(dir.path(), &[compacted.clone(), Lineage::own(A)]).unwrap();
        assert_eq!(
            read_generations(dir.path()),
            [compacted.clone(), Lineage::own(A)]
        );
        let text = std::fs::read_to_string(dir.path().join(GENERATIONS_FILE)).unwrap();
        assert!(
            text.starts_with(&format!("{GENERATIONS_FORMAT}\n")),
            "{text}"
        );
        std::fs::write(
            dir.path().join(GENERATIONS_FILE),
            format!("{text}not a generation\n"),
        )
        .unwrap();
        assert_eq!(read_generations(dir.path()), [], "damaged");
        std::fs::write(
            dir.path().join(GENERATIONS_FILE),
            text.replace(GENERATIONS_FORMAT, "xtp pipeline generations 0"),
        )
        .unwrap();
        assert_eq!(read_generations(dir.path()), [], "another version");
    }

    /// A translation key with no generation starts from the last one opened:
    /// its file linked (or copied) and its names file copied, so that it
    /// finds every pipeline whose translations did not change by its name.
    /// Saving the new generation replaces its own file only.
    #[test]
    fn a_new_generation_starts_from_the_last_one_opened() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        let a = settle(d, A, GENERATIONS_KEPT, u64::MAX).unwrap();
        assert_eq!(a.lineage, Lineage::own(A), "nothing to start from");
        assert!(a.new);
        assert_eq!(a.place, Place::of(d, A));
        assert!(!a.place.library.exists());
        save_as(&a.place, b"A's pipelines", "A's names");

        let b = settle(d, B, GENERATIONS_KEPT, u64::MAX).unwrap();
        assert_eq!(b.lineage.from.as_deref(), Some(A));
        assert!(b.new);
        assert_eq!(std::fs::read(&b.place.library).unwrap(), b"A's pipelines");
        assert_eq!(
            std::fs::read_to_string(&b.place.names).unwrap(),
            "A's names"
        );
        assert_eq!(read_generations(d), [b.lineage.clone(), Lineage::own(A)]);

        save_as(&b.place, b"A's and B's pipelines", "B's names");
        assert_eq!(std::fs::read(&a.place.library).unwrap(), b"A's pipelines");
        assert_eq!(
            std::fs::read_to_string(&a.place.names).unwrap(),
            "A's names"
        );

        // A third starts from B, the last opened, not A.
        let c = settle(d, C, GENERATIONS_KEPT, u64::MAX).unwrap();
        assert_eq!(c.lineage.from.as_deref(), Some(B));
        assert_eq!(
            std::fs::read(&c.place.library).unwrap(),
            b"A's and B's pipelines"
        );
        assert_eq!(ids(&read_generations(d)), [C, B, A]);
        // Opened again, a generation keeps what it was.
        let again = settle(d, C, GENERATIONS_KEPT, u64::MAX).unwrap();
        assert_eq!(again.lineage, c.lineage);
        assert!(!again.new, "its file is the one it left");
        // Its file gone (let go by another process's list): seeded again,
        // and new, whatever its size (the precook's note does not hold it).
        std::fs::remove_file(&again.place.library).unwrap();
        let reseeded = settle(d, C, GENERATIONS_KEPT, u64::MAX).unwrap();
        assert_eq!(reseeded.lineage.from.as_deref(), Some(B));
        assert!(reseeded.new);
    }

    /// A, then B (whose translator differs: compacted to its own pipelines),
    /// then A again: A opens its own file as it left it, with its names, and
    /// makes nothing. Before generations, the one file held B's pipelines
    /// alone by then, and A's were all made again.
    #[test]
    fn a_translator_back_finds_its_own_generation() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        let a = settle(d, A, GENERATIONS_KEPT, u64::MAX).unwrap();
        save_as(&a.place, b"A's pipelines", "A's names");
        record_compaction(d, A, 13, 5).unwrap();
        let a_then = read_generations(d)[0].clone();
        assert_eq!(a_then.compacted, Some(13));

        let b = settle(d, B, GENERATIONS_KEPT, u64::MAX).unwrap();
        assert_eq!(b.lineage.from.as_deref(), Some(A));
        // B's precook compacts it: B's pipelines alone, below the bound.
        save_as(&b.place, b"B's", "B's names");
        record_compaction(d, B, 3, 5).unwrap();
        assert_eq!(read_generations(d), [Lineage::own(B), a_then.clone()]);

        let back = settle(d, A, GENERATIONS_KEPT, u64::MAX).unwrap();
        assert_eq!(back.lineage, a_then, "its own, as it was");
        assert!(!back.new);
        assert_eq!(
            std::fs::read(&back.place.library).unwrap(),
            b"A's pipelines"
        );
        assert_eq!(
            std::fs::read_to_string(&back.place.names).unwrap(),
            "A's names"
        );
        assert_eq!(back.evicted, Vec::<String>::new());
        assert_eq!(read_generations(d), [a_then, Lineage::own(B)]);
        assert_eq!(std::fs::read(&b.place.library).unwrap(), b"B's");
    }

    /// The one library file of before generations becomes the first
    /// generation made (moved, with its names); once there are generations,
    /// a new one starts from them, and the file of before is left alone.
    #[test]
    fn the_file_of_before_becomes_the_first_generation() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        let legacy = Place::legacy(d);
        save_as(&legacy, b"the one file", "its names");
        let a = settle(d, A, GENERATIONS_KEPT, u64::MAX).unwrap();
        assert_eq!(a.lineage.from.as_deref(), Some(LEGACY));
        assert_eq!(std::fs::read(&a.place.library).unwrap(), b"the one file");
        assert_eq!(
            std::fs::read_to_string(&a.place.names).unwrap(),
            "its names"
        );
        assert!(!legacy.library.exists() && !legacy.names.exists(), "moved");

        // An older build wrote the file of before again meanwhile.
        save_as(&legacy, b"an older build's", "its names");
        let b = settle(d, B, GENERATIONS_KEPT, u64::MAX).unwrap();
        assert_eq!(b.lineage.from.as_deref(), Some(A));
        assert_eq!(std::fs::read(&legacy.library).unwrap(), b"an older build's");
    }

    /// Past `kept` generations or `budget` bytes of files, the least
    /// recently opened go, files and names; the current one stays whatever
    /// its size. A listed generation whose file is gone leaves the list, and
    /// a file the list does not hold counts as the oldest.
    #[test]
    fn the_least_recently_opened_generations_go() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        let hundred = [7u8; 100];
        for id in [A, B, C, D] {
            let s = settle(d, id, 10, u64::MAX).unwrap();
            save_as(&s.place, &hundred, "names");
        }
        assert_eq!(ids(&read_generations(d)), [D, C, B, A]);

        // By count: E (from D) and two more.
        let e = settle(d, E, 3, u64::MAX).unwrap();
        assert_eq!(e.lineage.from.as_deref(), Some(D));
        assert_eq!(e.evicted, [B, A]);
        assert_eq!(ids(&read_generations(d)), [E, D, C]);
        for id in [A, B] {
            let gone = Place::of(d, id);
            assert!(!gone.library.exists() && !gone.names.exists(), "{id}");
        }
        assert!(Place::of(d, C).library.exists());

        // By bytes: 250 hold the current one and one more.
        let c = settle(d, C, 10, 250).unwrap();
        assert_eq!(c.evicted, [D]);
        assert_eq!(ids(&read_generations(d)), [C, E]);

        // The current one stays, past the budget alone.
        let e = settle(d, E, 10, 50).unwrap();
        assert_eq!(e.evicted, [C]);
        assert_eq!(ids(&read_generations(d)), [E]);
        assert!(Place::of(d, E).library.exists());

        // A file the list does not hold (written while the list was lost):
        // listed, as the oldest. A listed generation whose file is gone:
        // dropped.
        save_as(&Place::of(d, A), &hundred, "names");
        write_generations(d, &[Lineage::own(E), Lineage::own(B)]).unwrap();
        let e = settle(d, E, 10, u64::MAX).unwrap();
        assert_eq!(e.evicted, Vec::<String>::new());
        assert_eq!(ids(&read_generations(d)), [E, A]);
        // A damaged list: the files are listed again.
        std::fs::write(d.join(GENERATIONS_FILE), "damaged").unwrap();
        let a = settle(d, A, 10, u64::MAX).unwrap();
        assert_eq!(a.lineage, Lineage::own(A));
        assert_eq!(ids(&read_generations(d)), [A, E]);
    }

    /// A mapped library file reads back as written, survives replacement while
    /// mapped, and adopts `.next` on the next open if a rename over the live
    /// file is refused.
    #[test]
    fn the_library_file_is_mapped_and_replaced() {
        use std::os::windows::fs::OpenOptionsExt;

        let dir = tempfile::tempdir().unwrap();
        let p = place(dir.path());
        assert!(matches!(Blob::open(&p), Blob::None), "no file yet");
        std::fs::write(&p.library, b"").unwrap();
        assert!(matches!(Blob::open(&p), Blob::None), "an empty file");

        let old: Vec<u8> = (0..200_000u32).map(|i| i as u8).collect();
        std::fs::write(&p.library, &old).unwrap();
        let blob = Blob::open(&p);
        assert!(matches!(blob, Blob::Mapped(_)) || !mapped());
        assert_eq!(blob.bytes(), &old[..]);

        // While mapped, in-place write opens are rejected (protecting `&[u8]`
        // soundness), whereas hard-link seeding and concurrent reads succeed.
        if matches!(blob, Blob::Mapped(_)) {
            assert!(
                std::fs::OpenOptions::new()
                    .write(true)
                    .open(&p.library)
                    .is_err(),
                "mapped library must exclude FILE_SHARE_WRITE"
            );
            let linked = dir.path().join("seeded-hardlink.bin");
            std::fs::hard_link(&p.library, &linked).unwrap();
            assert_eq!(std::fs::read(&linked).unwrap(), old);
            std::fs::remove_file(&linked).unwrap();
        }

        // Replace while the old file is mapped.
        let tmp = p.library.with_extension("tmp");
        let new: Vec<u8> = (0..300_000u32).map(|i| (i * 7) as u8).collect();
        std::fs::write(&tmp, &new).unwrap();
        replace_library(&p, &tmp).unwrap();
        assert!(!tmp.exists());
        assert_eq!(blob.bytes(), &old[..]);
        drop(blob);

        assert_eq!(Blob::open(&p).bytes(), &new[..]);
        assert!(!p.next().exists());

        // When rename over the live file is refused (no FILE_SHARE_DELETE),
        // the save falls back to `.next` and the next open adopts it.
        const FILE_SHARE_READ: u32 = 1;
        let lock = std::fs::OpenOptions::new()
            .read(true)
            .share_mode(FILE_SHARE_READ)
            .open(&p.library)
            .unwrap();
        let newer: Vec<u8> = (0..150_000u32).map(|i| (i * 13) as u8).collect();
        std::fs::write(&tmp, &newer).unwrap();
        assert!(!replace_library(&p, &tmp).unwrap());
        assert!(p.next().exists());
        drop(lock);

        let adopted = Blob::open(&p);
        assert_eq!(adopted.bytes(), &newer[..]);
        assert!(!p.next().exists());

        // A later in-place replacement removes any stale `.next` file.
        let latest: Vec<u8> = (0..100_000u32).map(|i| (i * 3) as u8).collect();
        std::fs::write(p.next(), b"stale next").unwrap();
        std::fs::write(&tmp, &latest).unwrap();
        assert!(replace_library(&p, &tmp).unwrap());
        assert!(!p.next().exists());
        assert_eq!(Blob::open(&p).bytes(), &latest[..]);

        // Unmapped fallback (`XTP_LIBRARY_MAP=0` path) still adopts `.next`
        // and reads into `Blob::Held`.
        let held_next: Vec<u8> = (0..50_000u32).map(|i| (i * 5) as u8).collect();
        std::fs::write(p.next(), &held_next).unwrap();
        let held = Blob::open_in(&p, false);
        assert!(matches!(held, Blob::Held(_)));
        assert_eq!(held.bytes(), &held_next[..]);
        assert!(!p.next().exists());
    }

    #[test]
    fn next_files_integrate_with_generations_and_file_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();

        // Generation A only has `.next` (no `.bin`) and no generations.txt:
        // settle discovers it via `.next`, adopts it, and keeps it without re-seeding.
        let pa = Place::of(d, A);
        std::fs::write(pa.next(), b"generation A via next").unwrap();
        let sa = settle(d, A, GENERATIONS_KEPT, u64::MAX).unwrap();
        assert!(!sa.new);
        assert!(pa.library.exists());
        assert!(!pa.next().exists());
        assert_eq!(
            std::fs::read(&pa.library).unwrap(),
            b"generation A via next"
        );

        // `file_bytes_at` prefers `.next` when present.
        assert_eq!(
            file_bytes_at(&pa),
            Some(b"generation A via next".len() as u64)
        );
        std::fs::write(pa.next(), b"newer A").unwrap();
        assert_eq!(file_bytes_at(&pa), Some(b"newer A".len() as u64));

        // Seeding B adopts A's pending `.next` before linking/copying.
        let sb = settle(d, B, GENERATIONS_KEPT, u64::MAX).unwrap();
        assert_eq!(sb.lineage.from.as_deref(), Some(A));
        assert!(!pa.next().exists());
        assert_eq!(std::fs::read(&sb.place.library).unwrap(), b"newer A");

        // Eviction counts `.next` size toward budget and removes `.next`, `.bin`, `.names`.
        std::fs::write(pa.next(), [0x55u8; 200]).unwrap();
        std::fs::write(&pa.names, "names A").unwrap();
        let sc = settle(d, C, 10, 100).unwrap();
        assert!(sc.evicted.contains(&A.to_owned()));
        assert!(!pa.library.exists());
        assert!(!pa.next().exists());
        assert!(!pa.names.exists());
    }

    #[test]
    fn trim_state_arms_rate_limits_and_rearms_on_touch() {
        let state = TrimState::new();
        // Unmapped view returns None without querying clock or memory.
        assert_eq!(
            state.begin(|| panic!("clock should not be queried"), || 2_000),
            None
        );

        state.register(0x1000, 4096, 100);
        // Rate-limited until 1000 ms has elapsed from initial checked_at (0).
        assert_eq!(state.begin(|| 500, || 2_000), None);
        // Past 1000 ms, checked_at advances, but idle threshold (100 + 2000) not met.
        assert_eq!(state.begin(|| 1_500, || 2_000), None);
        // Rate-limited within 1000 ms of 1500.
        assert_eq!(state.begin(|| 2_499, || 2_000), None);
        // Past idle threshold: begins trim once.
        let claim = state.begin(|| 2_500, || 2_000).expect("first trim claim");
        assert_eq!((claim.at, claim.len), (0x1000, 4096));
        // Already running: returns None without querying clock, even if touched.
        assert_eq!(
            state.begin(
                || panic!("clock should not be queried while running"),
                || 2_000
            ),
            None
        );
        // A touch arriving while the trim thread is running does not allow a
        // concurrent trim, and re-arms a fresh trim once `finish()` runs.
        state.note_touched(0x1000, 3_000);
        assert_eq!(
            state.begin(|| panic!("no concurrent trim while running"), || 2_000),
            None
        );
        state.finish();
        // Still needs `quiet_ms` after the 3_000 touch.
        assert_eq!(state.begin(|| 4_500, || 2_000), None);
        let claim = state
            .begin(|| 5_500, || 2_000)
            .expect("re-armed after in-flight touch");
        assert_eq!((claim.at, claim.len), (0x1000, 4096));
        // After abort_spawn, previous state is restored and retry succeeds after 1 s.
        state.abort_spawn(claim.prev_state);
        let claim = state.begin(|| 6_500, || 2_000).expect("retry after abort");
        assert_eq!((claim.at, claim.len), (0x1000, 4096));
        state.finish();

        // Already trimmed for this touch_seq: returns None without querying clock.
        assert_eq!(
            state.begin(
                || panic!("clock should not be queried while trimmed"),
                || 2_000
            ),
            None
        );

        // Touching an unrelated view address does not re-arm.
        state.note_touched(0x2000, 7_000);
        assert_eq!(
            state.begin(|| panic!("unrelated touch must not re-arm"), || 2_000),
            None
        );

        // Stale out-of-order touch timestamp cannot move `loaded_at` backwards.
        state.note_touched(0x1000, 9_000);
        state.note_touched(0x1000, 4_000);
        assert_eq!(state.begin(|| 10_500, || 2_000), None);
        let claim = state
            .begin(|| 11_500, || 2_000)
            .expect("trim after monotonic loaded_at");
        assert_eq!((claim.at, claim.len), (0x1000, 4096));
        state.finish();

        // Touch racing with begin's quiet_ms check cancels trim and leaves it armed.
        state.note_touched(0x1000, 12_000);
        assert_eq!(
            state.begin(
                || 15_000,
                || {
                    state.note_touched(0x1000, 14_500);
                    2_000
                }
            ),
            None
        );
        assert_eq!(state.begin(|| 15_500, || 2_000), None);
        assert_eq!(state.begin(|| 16_499, || 2_000), None);
        let claim = state
            .begin(|| 17_499, || 2_000)
            .expect("trim after racing touch settles");
        assert_eq!((claim.at, claim.len), (0x1000, 4096));
        state.finish();
    }

    fn pattern(at: usize, out: &mut [u8]) {
        for (i, word) in out.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let v = ((at / 4 + i) as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(1);
            *word = v.to_le_bytes();
        }
    }

    /// Writes `len` bytes (whole MiB) of [`pattern`] past the system cache so
    /// no page starts resident. Returns false if the filesystem rejects it.
    fn write_uncached(path: &Path, len: usize) -> bool {
        use std::io::Write;
        use std::os::windows::fs::OpenOptionsExt;
        // FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH
        const UNCACHED: u32 = 0x2000_0000 | 0x8000_0000;
        const CHUNK: usize = 1 << 20;
        let Ok(mut file) = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .custom_flags(UNCACHED)
            .open(path)
        else {
            return false;
        };
        let mut store = vec![0u8; CHUNK + 4096];
        let start = store.as_ptr().align_offset(4096);
        for at in (0..len).step_by(CHUNK) {
            let chunk = &mut store[start..start + CHUNK];
            pattern(at, chunk);
            if file.write_all(chunk).is_err() {
                return false;
            }
        }
        true
    }

    /// Replacing a mapped library file and trimming its view with
    /// `VirtualUnlock` still reads back the original bytes when the file
    /// handle is held open.
    #[test]
    fn a_replaced_library_file_still_reads_from_the_disk() {
        const LEN: usize = 8 << 20;
        let dir = tempfile::tempdir().unwrap();
        let p = place(dir.path());
        if !write_uncached(&p.library, LEN) {
            return;
        }
        let mapped = Mapped::open(&p.library).expect("mapped library file");
        let tmp = p.library.with_extension("tmp");
        std::fs::write(&tmp, vec![0xEE; LEN]).unwrap();
        replace_library(&p, &tmp).unwrap();
        // SAFETY: `mapped` owns this view until it drops at the end of the test.
        unsafe {
            windows_sys::Win32::System::Memory::VirtualUnlock(
                mapped.at.Value as *const _,
                mapped.len,
            )
        };
        let mut expected = vec![0u8; 4096];
        for (page, bytes) in mapped.bytes().chunks(4096).enumerate() {
            pattern(page * 4096, &mut expected);
            assert_eq!(
                bytes,
                &expected[..],
                "page {page} of the replaced library no longer reads as the old file"
            );
        }
    }

    /// A library round-trips through save, read-only mapping, save over the
    /// mapped file, and `VirtualUnlock` trim on WARP.
    #[test]
    fn mapped_library_round_trips_on_warp() {
        let test = "pipeline::library::tests::mapped_library_round_trips_on_warp";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let device = host.device.cast::<ID3D12Device1>().unwrap();
        let (a, b) = (kernel(&host.device, 3), kernel(&host.device, 5));
        let (name_a, name_b) = (
            "11111111111111111111111111111111",
            "22222222222222222222222222222222",
        );
        let dir = tempfile::tempdir().unwrap();
        let p = place(dir.path());

        let initial = RwLock::new(open(&device, Blob::None, None).unwrap());
        assert_eq!(ask(&initial, &host.device, &a, name_a), Source::Made);
        assert_eq!(
            write_library(&initial, &p, true, |_| Ok(true)),
            Saved::Written
        );

        let blob = Blob::open(&p);
        assert!(matches!(blob, Blob::Mapped(_)) || !mapped());
        let text = std::fs::read_to_string(&p.names).unwrap();
        let reopened = RwLock::new(open(&device, blob, Some(&text)).unwrap());
        assert_eq!(ask(&reopened, &host.device, &a, name_a), Source::Library);

        // Save a second pipeline over the mapped file and trim the view.
        assert_eq!(ask(&reopened, &host.device, &b, name_b), Source::Made);
        assert_eq!(
            write_library(&reopened, &p, true, |_| Ok(true)),
            Saved::Written
        );
        if let Some((at, len)) = reopened.read().unwrap()._blob.mapped_view() {
            // SAFETY: `reopened` holds this mapped view alive until dropped below.
            unsafe {
                windows_sys::Win32::System::Memory::VirtualUnlock(at as *const _, len);
            }
        }
        assert_eq!(ask(&reopened, &host.device, &a, name_a), Source::Library);
        assert_eq!(ask(&reopened, &host.device, &b, name_b), Source::Library);
        drop(reopened);

        // Reopen from disk and verify both pipelines load from the mapped library.
        let blob = Blob::open(&p);
        assert!(matches!(blob, Blob::Mapped(_)) || !mapped());
        let text = std::fs::read_to_string(&p.names).unwrap();
        let final_lib = RwLock::new(open(&device, blob, Some(&text)).unwrap());
        assert_eq!(ask(&final_lib, &host.device, &a, name_a), Source::Library);
        assert_eq!(ask(&final_lib, &host.device, &b, name_b), Source::Library);
        assert_eq!(final_lib.read().unwrap().added.load(Ordering::Relaxed), 0);

        // Save while `.bin` refuses rename (`.next` fallback), then verify
        // `new_generation` adopts `.next`, maps `previous`, trims it, and
        // preserves `.names` matching on WARP.
        let c = kernel(&host.device, 7);
        let name_c = "33333333333333333333333333333333";
        assert_eq!(ask(&final_lib, &host.device, &c, name_c), Source::Made);
        {
            use std::os::windows::fs::OpenOptionsExt;
            const FILE_SHARE_READ: u32 = 1;
            let _lock = std::fs::OpenOptions::new()
                .read(true)
                .share_mode(FILE_SHARE_READ)
                .open(&p.library)
                .unwrap();
            assert_eq!(
                write_library(&final_lib, &p, true, |_| Ok(true)),
                Saved::Written
            );
            assert!(p.next().exists());
        }
        drop(final_lib);

        let blob = Blob::open(&p);
        assert!(!p.next().exists());
        assert!(matches!(blob, Blob::Mapped(_)) || !mapped());
        let text = std::fs::read_to_string(&p.names).unwrap();
        let rebuilt = RwLock::new(new_generation(&device, blob, Some(&text)).unwrap());
        if let Some((at, len)) = rebuilt.read().unwrap().mapped_view() {
            // SAFETY: `rebuilt` holds `previous._blob` alive until dropped below.
            unsafe {
                windows_sys::Win32::System::Memory::VirtualUnlock(at as *const _, len);
            }
        }
        assert_eq!(ask(&rebuilt, &host.device, &c, name_c), Source::Previous);
        assert_eq!(ask(&rebuilt, &host.device, &c, name_c), Source::Library);
        drop(rebuilt);
    }

    #[test]
    fn ram_idle_ms_scales_with_available_memory() {
        const GIB: u64 = 1024 * 1024 * 1024;
        assert_eq!(crate::host::ram_idle_ms(None), 10_000);
        assert_eq!(crate::host::ram_idle_ms(Some(0)), 2_000);
        assert_eq!(crate::host::ram_idle_ms(Some(GIB)), 2_000);
        assert_eq!(crate::host::ram_idle_ms(Some(2 * GIB - 1)), 2_000);
        assert_eq!(crate::host::ram_idle_ms(Some(2 * GIB)), 5_000);
        assert_eq!(crate::host::ram_idle_ms(Some(3 * GIB)), 5_000);
        assert_eq!(crate::host::ram_idle_ms(Some(4 * GIB - 1)), 5_000);
        assert_eq!(crate::host::ram_idle_ms(Some(4 * GIB)), 10_000);
        assert_eq!(crate::host::ram_idle_ms(Some(8 * GIB)), 10_000);
    }
}
