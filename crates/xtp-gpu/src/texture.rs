//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture.
//!
//! "Changed" means a page of the memory was written since the last detiling
//! ([`crate::watch`]), in write-watched memory. Physical-page views cannot
//! be watched: there the memory is guarded as buffer copies are
//! (`watch::writes` with `guarded`), and a content hash of the whole memory
//! tells a change only when a block of it was written since the last check
//! (or cannot be guarded); a clean check costs a walk of its blocks, not a
//! hash. Every writer of such memory opens the guards: the title's CPU by
//! faulting, our copies and the kernel's reads (`watch::writing`), and
//! DirectStorage's page destinations (xmem's report after its writes).
//! `XTP_TEXTURE_GUARDS=0` (or `off`, `false`) hashes at every check, as
//! before: a world frame checked the same textures about a hundred times
//! (`docs/analysis/gpu-frame.md`).
//!
//! A writer that opens no guard would leave such a texture stale for good,
//! later lists' copies predicated off: the title's own views of sections,
//! its memory calls that bypass xmem, GPU writes through UMA heaps
//! (`watch`'s shortcuts). A safety net covers them: at each check it hashes
//! guarded textures checked since it last hashed them, round-robin by
//! resource, within a budget a frame (`XTP_TEXTURE_NET`, 2 MiB by default,
//! taken from a texture a slice at a time; a budget a second where frames
//! do not advance), a lap every 2 s at most (a few textures are not hashed
//! again at every frame: 0.86 ms a frame of an L4 VM's CPU for six of them,
//! the whole budget, before). A texture whose memory no longer hashes as
//! what was detiled is hashed at its next check whatever its guards say,
//! refreshed if it changed, and logged once (with the count of such changes
//! so far). Its budget costs at most 0.65 ms a frame on a build-farm CPU
//! (`tests::net_costs`), 0.86 ms on the L4 VM's; not measured in the world
//! yet.
//!
//! A check hashes and detiles one snapshot of the memory: hashed in place
//! and detiled after the wait for an earlier copy, a title write during the
//! wait was detiled under the hash of the memory before it, which the
//! title writing those bytes back then matched.
//!
//! A texture whose host format stands in for the title's
//! (`resource::format`) has its texels converted on the way.
//!
//! Each list that samples such a texture records the copy of the staging
//! buffer into the host copy, under a predicate of its own
//! (`SetPredication`): at submission, once the staging buffer is up to
//! date, the copy is let run only if the host copy has not got these
//! bytes yet, so the first list after a change copies and the others do
//! not (a world frame samples the same textures in dozens of lists:
//! `docs/analysis/gpu-frame.md`). `XTP_TEXTURE_PREDICATION=0` runs every
//! list's copy, as before.
//!
//! A staging buffer goes once no list holds it (the lists that copied from
//! it ran and were reset) and no upload of it was recorded for a couple of
//! passes ([`release_staging`]); what was detiled and copied is kept, and
//! the next upload makes the buffer again, detiling and copying only what
//! changed (or for a host copy not known to hold it). Before, a staging
//! buffer lived as long as its texture, and the world's thousands filled
//! the staging limit of the textures picked by index for good. An upload
//! of a texture whose host copy holds what was detiled last, its memory
//! not written by what the write watch or the guards tell (or only a hash
//! tells), makes nothing again and records nothing in its list: the check
//! at the list's submission tells whether the title wrote the memory
//! since, until then, and if it did the buffer is made again, detiled, and
//! copied by a list of late copies that runs just before the list
//! ([`late_copies`]). No copy reads a buffer made again before something
//! is detiled into it: under a predicate, its decision at submission says
//! so; without one, the buffer is filled when the copy is recorded, or no
//! copy is recorded. Nothing is detiled from memory that is not mapped
//! (but the units of a reserved host copy whose memory is): the host copy
//! keeps its texels. The memory is read for a detiling into one room
//! shared by the checks, not into a copy of its own kept per texture.
//!
//! Textures read through descriptor tables are brought up to date by the
//! draws that read them. Directly indexed views are not known per draw:
//! the textures shaders pick from the view heap by an index they compute
//! are uploaded by a pass once a frame (`draw.rs` `heap_textures`, the
//! queue of `descriptor::host::TextureQueue`), the slots the title wrote
//! last first, and then checked for changes in turns at record time
//! ([`memory_change`]: the guards or the write watch, no hash), the changed
//! ones uploaded as a table read does (memory neither covers within a
//! budget of bytes a pass, as the check at submission hashes it). Ported
//! from fxceless's VIctory fork (commit 578d459), which uploads such a
//! texture once and again only when the title rewrites its slot.
//!
//! GPU writes into a texture the title fills (copies, resolves) and its
//! uploads follow submission order, not recording order (watch item 263,
//! [`gpu_write`]): from the submission of the first such write on, its
//! host copy keeps the host's contents, and an upload recorded in a list
//! submitted after it runs no copy, decided at that list's submission as
//! whether the host copy has the staging buffer's bytes is. A write into
//! part of the texture first uploads its title bytes, in the same list, so
//! the rest stays as the title filled it. VIctory #1934 (zondy1i) decides
//! both at submission too; this was written without its code.
//! shortcut: without a predicate (`XTP_TEXTURE_PREDICATION=0`, or none
//! left) an upload's copy runs whatever was submitted before it, as before
//! (from a staging buffer that holds the title's bytes: the module doc's
//! staging release).
//!
//! shortcut: a change detiles the whole texture, not the pages written.
//! Plane 0 is uploaded (every mip and array slice), depth and stencil
//! planes are not.
//!
//! shortcut: the safety net covers guarded textures only, not those in
//! write-watched memory.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use crate::host::{HostList, host};
use crate::object::This;
use crate::resource::format::Convert;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A texture's upload state: the staging buffer laid out as the host
/// texture's subresources, and what was last detiled. The buffer goes once
/// no list holds the state any more (its last copy ran) and is made again
/// for the next upload ([`release_staging`]); what was detiled and copied
/// is kept, so that an unchanged texture is neither detiled nor copied
/// again.
pub(crate) struct Staging {
    /// The staging buffer; `None` once released.
    upload: Mutex<Option<Upload>>,
    /// The buffer holds nothing detiled: made (again), nothing detiled
    /// into it since. No copy reads it then.
    blank: AtomicBool,
    /// The next check detiles the memory whatever its watch, guards or
    /// hash say: a blank buffer a copy is due from.
    force: AtomicBool,
    /// [`SWEEPS`] when its last upload was recorded.
    recorded: AtomicU64,
    /// The staging buffer's bytes.
    size: u64,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    layout: xtp_formats::texture::Layout,
    /// How texels convert into the host format, if it stands in for the
    /// title's.
    convert: Option<Convert>,
    /// Mips per array slice.
    mips: u32,
    memory: *const u8,
    last: Mutex<Last>,
    /// How many times the staging buffer was written (detiled).
    written: AtomicU64,
    /// What a host copy last got from it, as decided at submission: the
    /// count of writes then and the host copy (held weakly: its address is
    /// not reused meanwhile).
    copied: Mutex<Option<(u64, Weak<crate::resource::HostTexture>)>>,
    /// Checked through its guards since the safety net last hashed it.
    sampled: AtomicBool,
    /// The safety net found its memory differ from what was detiled: its
    /// next check hashes it whatever its guards say.
    suspect: AtomicBool,
    /// Changes its guards missed.
    missed: AtomicU64,
    /// A GPU write into its texture was submitted ([`host_owned`]): nothing
    /// is detiled into it any more.
    host_owned: AtomicBool,
    /// Its host copy's tiles, if it is a reserved texture of the pool
    /// (`resource::pool::Tiles`): the units its title memory fills are
    /// mapped as it is detiled ([`Staging::map_units`]).
    tiles: Option<Weak<crate::resource::pool::Tiles>>,
    /// The units of `tiles` its title memory filled when they were last
    /// mapped (by unit; empty: never mapped).
    wanted: Mutex<Vec<bool>>,
    /// Some of those are not mapped: left for want of room, or given back
    /// ([`evict_idle`]). Its host copy does not hold what was detiled then:
    /// its next upload detiles and copies again, mapping them.
    incomplete: AtomicBool,
    /// Its tiles were given back and not mapped again since.
    evicted: AtomicBool,
    /// [`SWEEPS`] and [`ASKS`] when a pass last asked about it
    /// ([`memory_change`]): a texture a shader may pick by index is asked
    /// about once a lap of the pass's checks.
    asked: AtomicU64,
    asked_at: AtomicU64,
    /// A pass looked at it at least once (its first look tells no lap).
    looked: AtomicBool,
    /// The bytes of its host copy (diagnostics: the census).
    #[cfg(feature = "diagnostics")]
    host_bytes: u64,
    /// When it was made, in [`BORN`]'s order (diagnostics: the census).
    #[cfg(feature = "diagnostics")]
    born: u64,
    /// Tells it apart in tests from a staging state of another test freed
    /// at the same address (whose copies and hashes the address would
    /// count).
    #[cfg(test)]
    id: u64,
    /// Its checks since the last summary (diagnostics).
    #[cfg(feature = "diagnostics")]
    seen: Seen,
}

/// A texture's checks since the last summary (diagnostics): all of them,
/// those again in a submission that had checked it, those its guards found
/// clean, the bytes hashed, the changes found.
#[cfg(feature = "diagnostics")]
#[derive(Default)]
struct Seen {
    checks: AtomicU64,
    again: AtomicU64,
    /// Found clean by the guards (no hash).
    clean: AtomicU64,
    hashed: AtomicU64,
    changed: AtomicU64,
}

/// What a texture's staging buffer holds: the hash of the memory detiled
/// (memory not watched), or the write-watch asking it was detiled after;
/// and the submission that checked it last (`timestamps::submissions`
/// then, diagnostics). The memory is read into the room the checks share
/// ([`SNAPSHOT`]): a room of its own, kept at the texture's size, held a
/// copy of every write-watched texture detiled so far, about 2 GiB for
/// four 512 MiB batches of them (Codex's review of c44a716, 3).
#[derive(Default)]
struct Last {
    hash: Option<blake3::Hash>,
    asked: u64,
    #[cfg(feature = "diagnostics")]
    checked: Option<u64>,
    /// The hash of the bytes detiled last (diagnostics: the census counts
    /// detilings again of the same bytes and of others).
    #[cfg(feature = "diagnostics")]
    detiled: Option<blake3::Hash>,
}

// SAFETY: the staging mapping is written only at submission, under
// `last`'s lock; the title memory is only read.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

/// Staging states by title resource, in the order the safety net walks.
static STAGING: Mutex<Option<BTreeMap<usize, Arc<Staging>>>> = Mutex::new(None);

/// The bytes of the staging buffers that live (upload memory): counted
/// when one is made, until it is destroyed ([`Upload`]'s drop), whether its
/// texture's state is in [`STAGING`] or only held by lists.
static STAGED: AtomicU64 = AtomicU64::new(0);

/// A texture's staging buffer and its mapping, counted in [`STAGED`] while
/// it lives.
struct Upload {
    buffer: ID3D12Resource,
    data: *mut u8,
    size: u64,
}

impl Drop for Upload {
    fn drop(&mut self) {
        STAGED.fetch_sub(self.size, Ordering::Relaxed);
    }
}

/// Calls of [`release_staging`] so far (a pass each): the clock of
/// [`Staging::recorded`]. Tests sweep at later times of it rather than
/// advance it: advanced by them, every other test's texture looked idle to
/// that test's own passes, which released its buffers inside their timed
/// windows.
static SWEEPS: AtomicU64 = AtomicU64::new(0);

/// Staging buffers released so far: count and bytes.
static RELEASED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// A pass's looks at served textures so far ([`memory_change`]): the clock
/// of [`Staging::asked_at`].
static ASKS: AtomicU64 = AtomicU64::new(0);

/// The most looks between two looks at one texture lately: a lap of the
/// pass's looks over the slots it serves (several slots may view one
/// texture, so the textures' count does not bound it). It fades by a
/// 128th a pass.
static LAP: AtomicU64 = AtomicU64::new(0);

/// Host copies whose tiles were given back so far ([`evict_idle`]): count
/// and bytes; and those mapped again after it (made again on demand).
static EVICTED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// The resource [`evict_idle`] looked at last.
static EVICT_CURSOR: Mutex<usize> = Mutex::new(0);

/// Passes without an upload recorded nor a look of a pass after which a
/// texture's tiles are given back ([`evict_idle`]), by default.
const EVICT_IDLE: u64 = 120;

/// [`EVICT_IDLE`] or `XTP_TEXTURE_EVICT` (passes; `0`, `off` or `false`:
/// never).
fn evict_after() -> Option<u64> {
    static PASSES: OnceLock<Option<u64>> = OnceLock::new();
    *PASSES.get_or_init(|| evict_after_of(env("XTP_TEXTURE_EVICT").as_deref()))
}

fn evict_after_of(value: Option<&str>) -> Option<u64> {
    match value.map(str::trim) {
        None => Some(EVICT_IDLE),
        Some(v) if off(v) => None,
        Some(v) => Some(v.parse::<u64>().unwrap_or_else(|_| {
            tracing::warn!(
                value = v,
                "XTP_TEXTURE_EVICT is not a number of passes: {EVICT_IDLE}"
            );
            EVICT_IDLE
        })),
    }
}

/// The resource [`release_staging`] looked at last: the next call goes on
/// after it.
static RELEASE_CURSOR: Mutex<usize> = Mutex::new(0);

/// Room for a check's snapshot of texture memory, kept at the largest
/// changed so far: the one room the checks read memory into, write-watched
/// or not ([`Last`]).
static SNAPSHOT: Mutex<Vec<u8>> = Mutex::new(Vec::new());

/// [`SNAPSHOT`]'s room, `len` bytes of it, out of the lock: a check that
/// finds it taken (another check at the same time) reads into a room of
/// its own, given back after ([`give_room`]).
fn take_room(len: usize) -> Vec<u8> {
    let mut room = std::mem::take(&mut *SNAPSHOT.lock().unwrap_or_else(|e| e.into_inner()));
    room.resize(len, 0);
    room
}

/// Gives a room [`take_room`] handed out back to [`SNAPSHOT`], which keeps
/// the larger of it and what it holds.
fn give_room(room: Vec<u8>) {
    let mut kept = SNAPSHOT.lock().unwrap_or_else(|e| e.into_inner());
    if room.capacity() > kept.capacity() {
        *kept = room;
    }
}

/// Textures whose host copy a GPU write (a copy or resolve into it) was
/// submitted for, as their lists' submissions are prepared ([`gpu_write`]):
/// their contents are on the host from then on, not in title memory, which
/// stays as it was.
static GPU_WRITTEN: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

/// Whether a GPU write into `resource`'s host copy was submitted (in the
/// submission being prepared, or before).
fn gpu_written(resource: usize) -> bool {
    GPU_WRITTEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|set| set.contains(&resource))
}

/// `list` writes `resource`'s host copy on the GPU next: a copy or resolve
/// into it, of every texel of every subresource if `whole`. From the
/// list's submission on, the host copy keeps the host's contents: title
/// memory is not uploaded into it any more (shortcut: even if the title
/// writes it later). Uploads recorded in lists submitted after that run no
/// copy, those submitted before run as usual: the order is the
/// submissions', not the recordings' (module doc). A write into part of a
/// texture the title fills keeps the rest as the title filled it: its title
/// bytes are uploaded first, in this list, unless the host copy has them
/// already or a GPU write was submitted before (both decided at
/// submission). Recorded where the list has the texture in the write's
/// state, before the write.
pub(crate) fn gpu_write(list: &HostList, resource: This, whole: bool) {
    let key = resource as usize;
    if gpu_written(key) {
        return;
    }
    if !whole
        && let Some(texture) = crate::resource::host_resource(resource)
        && title_written(texture)
    {
        load_before_write(list, resource, texture);
    }
    // A reserved host copy gets all its tiles before the write runs (the
    // title's memory has no say in it any more), and keeps them: its
    // contents are the host's, never given back.
    let tiles = crate::resource::host_resource(resource).and_then(crate::resource::pool::tiles_of);
    // After the upload's decision above, in the list's order.
    list.defer_host((0, 0), move || {
        if let (Some(tiles), Some(host)) = (&tiles, host()) {
            let all = vec![true; tiles.units().len()];
            if tiles.map(host, &all).short > 0 {
                once!("a GPU write into a reserved texture without room for all its tiles: part of it dropped");
            }
        }
        host_owned(key)
    });
}

/// Records the upload of `resource`'s title bytes into its host copy
/// `texture` before a GPU write into part of it, recorded next in the state
/// the list has the texture in now: the write waits for the upload, which
/// may write the same texels (through a transition out of the copy
/// destination state and back if the write is a copy too).
fn load_before_write(list: &HostList, resource: This, texture: &ID3D12Resource) {
    const COMMON: u32 = 0;
    const COPY_DEST: u32 = 0x400;
    // The write's state: the one the list moves the texture out of here.
    let state = match list.transition(resource, COPY_DEST) {
        Some((host, before)) => {
            let barrier = crate::host::transition(
                host,
                D3D12_RESOURCE_STATES(before as i32),
                D3D12_RESOURCE_STATES(COPY_DEST as i32),
            );
            // SAFETY: an open list and a live texture.
            unsafe { list.list.ResourceBarrier(&[barrier]) };
            before
        }
        None => COPY_DEST,
    };
    let uploaded = upload(list, resource, texture);
    if state != COPY_DEST {
        crate::list::ensure_state(list, resource, state);
    } else if uploaded {
        crate::list::ensure_state(list, resource, COMMON);
        crate::list::ensure_state(list, resource, COPY_DEST);
    }
}

/// At the submission of a GPU write into `resource`'s host copy: the host
/// has its contents from now on, and its staging buffer is not needed any
/// more (the lists holding it keep it until they are reset; its bytes leave
/// [`STAGED`] then).
fn host_owned(resource: usize) {
    GPU_WRITTEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(Default::default)
        .insert(resource);
    let gone = STAGING
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_mut()
        .and_then(|map| map.remove(&resource));
    if let Some(gone) = gone {
        gone.host_owned.store(true, Ordering::Relaxed);
    }
}

/// The title released a resource (its staging buffer goes once the lists
/// holding it are reset).
pub(crate) fn forget(resource: usize) {
    let gone = STAGING
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_mut()
        .and_then(|map| map.remove(&resource));
    drop(gone);
    let mut set = GPU_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(set) = set.as_mut() {
        set.remove(&resource);
    }
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    // SAFETY: a live resource.
    unsafe { texture.GetDesc() }.Flags.0 & TARGET_DEPTH_UNORDERED == 0
}

/// Records, once per list, the copy of `resource`'s staging buffer into
/// its host copy, and the detiling that refreshes the staging buffer at
/// submission if the memory changed. The copy runs only if the host copy
/// has not got the staging buffer's bytes yet and no GPU write into it was
/// submitted before (both decided at submission, see the module); without
/// a predicate to decide it, it is recorded only from a staging buffer
/// that holds a detiling. Leaves the texture in the copy destination state
/// if a copy was recorded. Whether one was.
pub(crate) fn upload(list: &HostList, resource: This, texture: &ID3D12Resource) -> bool {
    #[cfg(feature = "diagnostics")]
    summary_due();
    // shortcut: a texture the host GPU wrote keeps the host's contents even
    // if the title later writes its memory too.
    if gpu_written(resource as usize) || !list.first_use(resource as usize) {
        return false;
    }
    // The host copy `texture` is: what the copy writes (the list holds it
    // through its state below).
    let copy =
        crate::resource::host_copy(resource).filter(|c| c.texture.as_raw() == texture.as_raw());
    let target = copy.as_ref().map(Arc::downgrade);
    // Released, and its host copy has what is in the memory as far as can
    // be told now: nothing made again, nor recorded here; the check at
    // submission copies what the title writes until then. Before, a table
    // read made the buffer again whatever the room, only for the check at
    // submission to find nothing to detile (Codex's review of c44a716, 3;
    // of db37732, 1, for memory only a hash tells).
    if let Some(copy) = copy
        && let Some(state) = state_of(resource).filter(|s| s.late(target.as_ref()))
    {
        upload_late(list, resource, state, copy);
        return false;
    }
    // Made again if it was released: this list holds it from now on.
    let Ok(staging) = staging(resource, texture) else {
        return false;
    };
    if staging.movie_plane() {
        let (at, len) = staging.range();
        let len = len - at;
        let pitch = staging.footprints[0].Footprint.RowPitch as usize;
        let bytes = pitch * staging.rows[0] as usize;
        let Some((source, data)) = host().and_then(|h| list.upload(h, bytes as u64)) else {
            once!("no upload buffer for a CPU-written movie plane");
            return false;
        };
        let dst = data as usize;
        let predicate = target
            .as_ref()
            .filter(|_| may_predicate())
            .and_then(|_| Predicate::take());
        let plane = staging.clone();
        let fill = move |direct| fill_plane(&plane.layout, at, len, dst, bytes, pitch, direct);
        if predicate.is_none() && !fill(false) {
            return false;
        }
        record_predicated(
            list,
            resource,
            texture,
            &staging,
            &source,
            predicate.as_ref(),
        );
        let (bytes, key) = (bytes as u64, resource as usize);
        list.defer_plane((at, len), move || {
            let filled = fill(predicate.is_some());
            let runs = match &predicate {
                Some(p) => p.decide(filled && !gpu_written(key)),
                None => true,
            };
            COPIES[usize::from(runs)].fetch_add(1, Ordering::Relaxed);
            COPIED[usize::from(runs)].fetch_add(bytes, Ordering::Relaxed);
        });
        return true;
    }
    // Tiles its title memory fills are not mapped (left for want of room,
    // or given back) and a resident heap has room for them: the check at
    // submission detiles whatever the memory says, mapping them, and the
    // copy brings their texels. Without room, it waits (not detiled again
    // at every submission for nothing).
    if staging.wants_tiles() {
        staging.force.store(true, Ordering::Release);
    }
    // Without a predicate (none left, or predication off) the copy runs.
    let predicate = target
        .as_ref()
        .filter(|_| may_predicate())
        .and_then(|_| Predicate::take());
    // A buffer made again after a release (or new) holds nothing.
    if staging.blank.load(Ordering::Acquire) {
        match &predicate {
            // Its copy is decided at submission, not from a blank buffer:
            // an unchanged texture whose host copy has what was detiled
            // last needs neither a detiling nor a copy, any other gets one.
            Some(_) => {
                if !staging.holds(target.as_ref()) {
                    staging.force.store(true, Ordering::Release);
                }
            }
            // Its copy runs at submission whatever the buffer holds then:
            // the buffer is filled now, or no copy is recorded and the host
            // copy keeps its texels. Before, the check at submission could
            // leave it blank (the memory not mapped, or a GPU write into
            // the texture submitted meanwhile), and the copy replaced valid
            // texels with it (Codex's review of c44a716, 1).
            None => {
                if !staging.fill() {
                    return false;
                }
            }
        }
    }
    let Some(buffer) = staging.buffer() else {
        return false;
    };
    record_predicated(
        list,
        resource,
        texture,
        &staging,
        &buffer,
        predicate.as_ref(),
    );
    // Once per submission however many of its lists sample the texture.
    list.defer_texture(staging.clone());
    // Then, for this list: whether its copy runs. Not over what a GPU write
    // submitted before it put there.
    let (bytes, key) = (staging.size, resource as usize);
    list.defer_host((0, 0), move || {
        // Not from a buffer that holds nothing detiled (its memory was not
        // mapped at the check).
        let due = !gpu_written(key)
            && !staging.blank.load(Ordering::Acquire)
            && target.as_ref().is_none_or(|t| staging.copy_due(t));
        let runs = match &predicate {
            Some(p) => p.decide(due),
            None => true,
        };
        COPIES[usize::from(runs)].fetch_add(1, Ordering::Relaxed);
        COPIED[usize::from(runs)].fetch_add(bytes, Ordering::Relaxed);
        #[cfg(test)]
        DECIDED
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((staging.id, runs));
    });
    true
}

/// Copies all subresources of `staging` from `buffer` into `texture` on
/// `list`, guarded by `predicate` when present.
fn record_predicated(
    list: &HostList,
    resource: This,
    texture: &ID3D12Resource,
    staging: &Staging,
    buffer: &ID3D12Resource,
    predicate: Option<&Predicate>,
) {
    staging
        .recorded
        .store(SWEEPS.load(Ordering::Relaxed), Ordering::Relaxed);
    const COPY_DEST: u32 = 0x400;
    crate::list::ensure_state(list, resource, COPY_DEST);
    let timed = crate::timestamps::command(list, crate::timestamps::What::TextureUpload, None);
    if let Some(p) = predicate {
        // SAFETY: an open list; the predicate's buffer lives for the run.
        unsafe {
            list.list
                .SetPredication(&p.of.buffer, p.offset(), D3D12_PREDICATION_OP_EQUAL_ZERO)
        };
    }
    record_subresources(list, resource, texture, staging, buffer);
    if predicate.is_some() {
        // SAFETY: an open list.
        unsafe {
            list.list
                .SetPredication(None::<&ID3D12Resource>, 0, D3D12_PREDICATION_OP_EQUAL_ZERO)
        };
    }
    drop(timed);
}

/// Copies all subresources of `staging` from `buffer` into `texture` on `list`.
fn record_subresources(
    list: &HostList,
    resource: This,
    texture: &ID3D12Resource,
    staging: &Staging,
    buffer: &ID3D12Resource,
) {
    let _ = resource;
    for (i, footprint) in staging.footprints.iter().enumerate() {
        #[cfg(feature = "diagnostics")]
        {
            let f = footprint.Footprint;
            // SAFETY: live resources.
            let (gpu, d) = unsafe { (buffer.GetGPUVirtualAddress(), texture.GetDesc()) };
            let len = u64::from(f.RowPitch) * u64::from(staging.rows[i]) * u64::from(f.Depth);
            list.note_read(
                || {
                    format!(
                        "texture upload {resource:?} ({}x{}x{} f{} mips {}) sub {i} of {} from staging {gpu:#x}+{:#x}: offset {:#x} {}x{}x{} pitch {}",
                        d.Width,
                        d.Height,
                        d.DepthOrArraySize,
                        d.Format.0,
                        d.MipLevels,
                        staging.footprints.len(),
                        staging.size,
                        footprint.Offset,
                        f.Width,
                        f.Height,
                        f.Depth,
                        f.RowPitch
                    )
                },
                (gpu + footprint.Offset, len),
            );
        }
        copy_subresource(&list.list, texture, buffer, i, footprint);
    }
}

/// Copies a CPU-written linear movie plane from `[at, at + len)` into a
/// list's dedicated upload buffer `dst` (`bytes` bytes, host `pitch`).
// Ported from edsuuu/VI-BR-PORT commit fd36f4a4 (GPL-3.0-or-later).
fn fill_plane(
    layout: &xtp_formats::texture::Layout,
    at: usize,
    len: usize,
    dst: usize,
    bytes: usize,
    pitch: usize,
    direct: bool,
) -> bool {
    // SAFETY: the list owns this full upload buffer until its GPU work completes.
    let out = unsafe { std::slice::from_raw_parts_mut(dst as *mut u8, bytes) };
    let mip = &layout.mips[0];
    if direct
        && mip.offset == 0
        && mip.pitch as usize * layout.desc.element_bytes as usize == pitch
        && bytes <= len
        && crate::host::read_into(at as *const u8, out)
    {
        return true;
    }
    let mut read = take_room(len);
    if !crate::host::read_into(at as *const u8, &mut read) {
        give_room(read);
        return false;
    }
    let result = layout.detile(&read, 0, 0, out, pitch, bytes);
    if let Err(ref error) = result {
        once!("could not copy a linear movie plane into the list's upload buffer: {error:?}");
    }
    give_room(read);
    result.is_ok()
}

/// Records in `list` the copy of subresource `i` of a staging buffer
/// `buffer`, laid out at `footprint`, into `texture`.
fn copy_subresource(
    list: &ID3D12GraphicsCommandList,
    texture: &ID3D12Resource,
    buffer: &ID3D12Resource,
    i: usize,
    footprint: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
) {
    let to = D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the location never releases.
        pResource: unsafe { std::mem::transmute_copy(texture) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: i as u32,
        },
    };
    let from = D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(buffer) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: *footprint,
        },
    };
    // SAFETY: an open list; both locations name live resources (the
    // caller's list holds them until it is reset).
    unsafe { list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
}

/// [`upload`] of `resource`, whose staging buffer was released and whose
/// host copy `copy` has what was detiled last, the memory unchanged since
/// as far as can be told at record time ([`Staging::late`]): nothing is
/// made, nor recorded in `list`. The check at its submission tells whether
/// the title wrote the memory since, after the list was recorded too (and
/// hashes memory only a hash tells): if it did, the staging buffer is made
/// again and detiled, and copied into the host copy by the late copies
/// that run just before the list ([`late_copies`]). A second read in the
/// list is covered by this one. Before, the check was skipped, and a write
/// between the recording and the submission left the list's shaders the
/// old texels (Codex's review of db37732, 1).
fn upload_late(
    list: &HostList,
    resource: This,
    staging: Arc<Staging>,
    copy: Arc<crate::resource::HostTexture>,
) {
    // Read lately: its buffer, if made again, stays while it is.
    staging
        .recorded
        .store(SWEEPS.load(Ordering::Relaxed), Ordering::Relaxed);
    // The check at submission; the list holds the state, and the host copy
    // below, until it is reset.
    list.defer_texture(staging.clone());
    let (key, before) = (resource as usize, std::ptr::from_ref(list) as usize);
    list.defer_host((0, 0), move || {
        // Not over what a GPU write submitted before put there; not from a
        // buffer that holds no detiling (none made: nothing written); not
        // if the host copy has these bytes (copied by a list before).
        let target = Arc::downgrade(&copy);
        if gpu_written(key)
            || staging.buffer().is_none()
            || staging.blank.load(Ordering::Acquire)
            || !staging.copy_due(&target)
        {
            return;
        }
        if !record_late(before, &staging, &copy) {
            // Not copied after all: its next upload copies.
            *staging.copied.lock().unwrap_or_else(|e| e.into_inner()) = None;
            return;
        }
        COPIES[1].fetch_add(1, Ordering::Relaxed);
        COPIED[1].fetch_add(staging.size, Ordering::Relaxed);
        #[cfg(test)]
        DECIDED
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((staging.id, true));
    });
}

/// The late copies ([`upload_late`]): recorded at a submission into a list
/// of their own that runs just before the host list whose check found them
/// due, and before its prelude (`host.rs`), so that they move the texture
/// from the state the lists submitted before leave it in, and back. The
/// one being recorded, with the host list it runs before and the
/// submission it is recorded in (`timestamps::submissions` then); and those
/// handed to submissions, free once the GPU has run them.
struct LateLists {
    open: Option<(usize, u64, LateList)>,
    handed: Vec<LateList>,
}

/// A list of late copies and its allocator.
struct LateList {
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    /// The host fence value of the submission it was recorded for: free
    /// once the GPU has reached it.
    until: u64,
}

static LATE: Mutex<LateLists> = Mutex::new(LateLists {
    open: None,
    handed: Vec::new(),
});

impl LateList {
    /// A new one, open.
    fn new(host: &crate::host::Host) -> Option<LateList> {
        // SAFETY: plain object creation; the list starts open.
        unsafe {
            let allocator: ID3D12CommandAllocator = host
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .ok()?;
            let list: ID3D12GraphicsCommandList = host
                .device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                .ok()?;
            crate::host::name(&list, || "texture late copies".into());
            Some(LateList {
                allocator,
                list,
                until: 0,
            })
        }
    }
}

impl LateLists {
    /// The list the late copies for host list `before` are recorded into
    /// in the submission being prepared: the one open for it, or one
    /// opened (a free one, or a new one). `None` if none can be had.
    fn open_for(
        &mut self,
        host: &crate::host::Host,
        before: usize,
    ) -> Option<&ID3D12GraphicsCommandList> {
        let submission = crate::timestamps::submissions();
        if self
            .open
            .as_ref()
            .is_some_and(|(list, at, _)| *list != before || *at != submission)
            && let Some((_, _, stale)) = self.open.take()
        {
            // Open for a list whose submission did not take it: it never
            // ran, and is free.
            // SAFETY: an open list of ours.
            if unsafe { stale.list.Close() }.is_ok() {
                self.handed.push(LateList { until: 0, ..stale });
            }
        }
        if self.open.is_none() {
            let reached = host.completed();
            let late = match self.handed.iter().position(|l| l.until <= reached) {
                Some(at) => {
                    let late = self.handed.swap_remove(at);
                    // SAFETY: closed, and the GPU ran what it recorded.
                    unsafe {
                        late.allocator.Reset().ok()?;
                        late.list.Reset(&late.allocator, None).ok()?;
                    }
                    late
                }
                None => LateList::new(host)?,
            };
            // The value this submission signals (the one handed to the
            // queue last, plus one); a submission whose lists run one at
            // a time (diagnostics) finishes before the next is prepared.
            let until = host.executing() + 1;
            self.open = Some((before, submission, LateList { until, ..late }));
        }
        self.open.as_ref().map(|(_, _, late)| &late.list)
    }
}

/// Records, at the submission of host list `before`, the copy of
/// `staging`'s buffer into host copy `copy` among the late copies run just
/// before it ([`LateLists`]): from the state the texture is in then into
/// the copy destination state, and back. Whether it was recorded.
fn record_late(before: usize, staging: &Staging, copy: &crate::resource::HostTexture) -> bool {
    const COPY_DEST: u32 = 0x400;
    let (Some(host), Some(buffer)) = (host(), staging.buffer()) else {
        return false;
    };
    let mut late = LATE.lock().unwrap_or_else(|e| e.into_inner());
    let Some(list) = late.open_for(host, before) else {
        once!("no list for a texture's late copy: its next upload copies it");
        return false;
    };
    // The state the lists before leave it in (their preludes ran).
    let state = copy.state.load(Ordering::Relaxed);
    let [into, back] = [(state, COPY_DEST), (COPY_DEST, state)].map(|(from, to)| {
        crate::host::transition(
            &copy.texture,
            D3D12_RESOURCE_STATES(from as i32),
            D3D12_RESOURCE_STATES(to as i32),
        )
    });
    // SAFETY: an open list of ours; the texture and the buffer are held by
    // the host list's steps until it is reset, after its submission ran.
    unsafe {
        if state != COPY_DEST {
            list.ResourceBarrier(&[into]);
        }
        for (i, footprint) in staging.footprints.iter().enumerate() {
            copy_subresource(list, &copy.texture, &buffer, i, footprint);
        }
        if state != COPY_DEST {
            list.ResourceBarrier(&[back]);
        }
    }
    true
}

/// The late copies the steps of host list `list` recorded in this
/// submission ([`upload_late`]), closed: a list to run just before it and
/// its prelude. `None` if there are none. Called once the list's steps
/// ran.
pub(crate) fn late_copies(list: &HostList) -> Option<ID3D12CommandList> {
    let before = std::ptr::from_ref(list) as usize;
    let mut late = LATE.lock().unwrap_or_else(|e| e.into_inner());
    if late.open.as_ref().is_none_or(|(l, _, _)| *l != before) {
        return None;
    }
    let (_, _, made) = late.open.take()?;
    // SAFETY: the open list of late copies.
    let run = match unsafe { made.list.Close() } {
        Ok(()) => made.list.cast().ok(),
        Err(e) => {
            tracing::warn!(error = %e, "a list of late texture copies did not close: not run");
            None
        }
    };
    late.handed.push(made);
    run
}

/// What a look at a texture's memory before its upload is recorded tells
/// ([`memory_change`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Change {
    /// Not written since its staging buffer was detiled; or nothing to
    /// upload (the host GPU's own contents).
    No,
    /// Written since, suspected by the safety net, never detiled, or never
    /// staged (no staging buffer could be made for it then, or it took the
    /// place of the texture uploaded before): an upload is due.
    Yes,
    /// Memory neither the write watch nor the guards cover (or guards that
    /// are off or verified): only the check at submission, which hashes its
    /// bytes (these many), tells.
    Unknown(u64),
}

/// Whether the memory of `resource`, a texture a shader picks from the view
/// heap by index, changed since its staging buffer was last detiled: asked
/// at record time by the pass over such textures (`draw.rs`
/// `heap_textures`), so that the unchanged ones record no upload. The
/// write watch or the guards answer as they do for the check at
/// submission, without its hash; nothing is taken from them (the journal
/// keeps the writes for the check). A texture its guards find clean is the
/// safety net's to hash, as one a check found clean.
pub(crate) fn memory_change(resource: This) -> Change {
    let asks = ASKS.fetch_add(1, Ordering::Relaxed) + 1;
    if gpu_written(resource as usize) {
        return Change::No;
    }
    match state_of(resource) {
        Some(staging) => {
            // Asked about: a shader may pick it, its tiles stay.
            staging
                .asked
                .store(SWEEPS.load(Ordering::Relaxed), Ordering::Relaxed);
            let before = staging.asked_at.swap(asks, Ordering::Relaxed);
            if staging.looked.swap(true, Ordering::Relaxed) {
                LAP.fetch_max(asks.saturating_sub(before), Ordering::Relaxed);
            }
            // Tiles given back or left for want of room: made again on
            // demand, once a resident heap has room for them.
            if staging.wants_tiles() {
                return Change::Yes;
            }
            staging.change()
        }
        // Not "clean": nothing was ever uploaded from it.
        None => Change::Yes,
    }
}

/// `resource`'s staging state, if it has one (its buffer may be released).
fn state_of(resource: This) -> Option<Arc<Staging>> {
    let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    map.as_ref()
        .and_then(|map| map.get(&(resource as usize)).cloned())
}

/// Whether uploads may take a predicate: always, but on a test's thread
/// that runs as if there were none left ([`tests::WithoutPredicates`]).
fn may_predicate() -> bool {
    #[cfg(test)]
    if tests::WITHOUT_PREDICATES.get() {
        return false;
    }
    true
}

/// The bytes of `resource`'s staging buffer, if it has one now (tests).
#[cfg(test)]
pub(crate) fn staged_bytes(resource: This) -> Option<u64> {
    let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let staging = map.as_ref()?.get(&(resource as usize))?;
    let upload = staging.upload.lock().unwrap_or_else(|e| e.into_inner());
    upload.as_ref().map(|u| u.size)
}

/// What came of making a texture's staging buffer ([`stage`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Staged {
    /// It has one: these many bytes new (0: it had one already).
    Ready(u64),
    /// It would take these many bytes, more than the room given: none made.
    NoRoom(u64),
    /// None now (no host, no upload memory for it, or for a pass's refresh
    /// more room than a pass could give it now: [`Refreshes::stage`]):
    /// asked again later.
    NotNow,
    /// None ever: its contents cannot be uploaded (no console layout, an
    /// XOR mode, a stand-in format without a conversion).
    Never,
    /// None needed: a GPU write into its host copy was submitted, which
    /// keeps the host's contents.
    Host,
}

/// Makes the staging buffer of `resource` (host copy `texture`) for its
/// first upload if it has none, within `room` bytes; tells whether it has
/// one now. For the pass over textures shaders pick by index (`draw.rs`
/// `heap_textures`), which spends a budget on new ones: a texture that
/// already has one costs it nothing, and one that cannot have one yet is
/// looked at again later rather than taken for uploaded.
pub(crate) fn stage(resource: This, texture: &ID3D12Resource, room: u64) -> Staged {
    if gpu_written(resource as usize) {
        return Staged::Host;
    }
    match staging_within(resource, texture, room) {
        Ok((_, new)) => Staged::Ready(new),
        Err(why) => why,
    }
}

/// The staging buffers a pass's checks of served textures make again
/// (`draw.rs` `heap_textures`): those of textures uploaded before, whose
/// buffer was released ([`release_staging`]) and whose memory changed.
/// They take a share of the pass's budget of their own
/// ([`REFRESH_SHARE`]), which the first uploads checked before them cannot
/// spend (the first one of a pass fits whatever its bytes), and may pass
/// the staging limit by that share, which first uploads fill. Before, such
/// a texture took its bytes from what the first uploads left of the pass's
/// budget: none while the world loads, the shaders reading its old texels
/// all along (Codex's review of c44a716, 2). With its buffer, its refresh
/// costs nothing, as before buffers were released.
pub(crate) struct Refreshes {
    share: u64,
    made: u64,
}

/// [`Refreshes`]' share of a pass's budget: a quarter.
const REFRESH_SHARE: u64 = 4;

impl Refreshes {
    /// For a pass that may stage `budget` bytes.
    pub(crate) fn of(budget: u64) -> Refreshes {
        Refreshes {
            share: budget / REFRESH_SHARE,
            made: 0,
        }
    }

    /// The bytes made again so far.
    pub(crate) fn made(&self) -> u64 {
        self.made
    }

    /// Makes the staging buffer of served texture `resource` again if it
    /// was released, within the share left: what came of it (as [`stage`]
    /// tells). `None` if it was never staged: a first upload, which the
    /// pass stages within its own budget. Past the share left,
    /// [`Staged::NoRoom`]: the next pass's share takes it first. Past what
    /// a pass of its own could make with the staging buffers that live,
    /// [`Staged::NotNow`]: looked at again at its next turn, and the pass
    /// goes on with the textures after it ([`refused`]); so too with the
    /// host's commit too low for new staging buffers (`host::commit`).
    pub(crate) fn stage(&mut self, resource: This) -> Option<Staged> {
        if gpu_written(resource as usize) {
            return Some(Staged::Host);
        }
        let staging = state_of(resource)?;
        let (limit, staged) = (crate::descriptor::host::pass_staging(), staged_total());
        // None made with the host's commit low (`host::commit`).
        let admitted = crate::host::commit::admits_staging();
        let room = if admitted {
            refresh_room(self.share, self.made, limit, staged)
        } else {
            0
        };
        Some(match staging.make_upload(room) {
            Ok(made) => {
                self.made += made;
                Staged::Ready(made)
            }
            Err(Staged::NoRoom(_)) if !admitted => Staged::NotNow,
            Err(Staged::NoRoom(bytes)) => refused(bytes, self.share, self.made, limit, staged),
            Err(why) => why,
        })
    }
}

/// The bytes the next staging buffer [`Refreshes`] makes again may take,
/// `made` of its `share` made so far, `staged` bytes of staging buffers
/// living under the staging limit `limit`.
fn refresh_room(share: u64, made: u64, limit: u64, staged: u64) -> u64 {
    let pass = if made == 0 {
        u64::MAX
    } else {
        share.saturating_sub(made)
    };
    pass.min(limit.saturating_add(share).saturating_sub(staged))
}

/// What [`Refreshes`] tells of a staging buffer of `bytes` past the room it
/// had (`made` of its `share` made, `staged` bytes living under the limit
/// `limit`): [`Staged::NoRoom`] if the first of a pass would take it (the
/// share spent: the next pass starts with it); [`Staged::NotNow`] if not
/// even that one would. Before, both stopped the pass's checks and the next
/// pass started with it again: one texture past the staging limit and the
/// share (96 MiB, 64 and 16) left every changed texture after it stale,
/// one whose buffer lived included (Codex's review of db37732, 3).
fn refused(bytes: u64, share: u64, made: u64, limit: u64, staged: u64) -> Staged {
    if made > 0 && bytes <= refresh_room(share, 0, limit, staged) {
        Staged::NoRoom(bytes)
    } else {
        Staged::NotNow
    }
}

/// Releases `resource`'s staging buffer at once if no list holds it,
/// whatever uploads were recorded lately (tests elsewhere: other tests'
/// textures are left as they are); whether it went.
#[cfg(test)]
pub(crate) fn release_staging_now(resource: This) -> bool {
    let key = resource as usize;
    let now = SWEEPS.load(Ordering::Relaxed);
    release_staging_of(now, 0, usize::MAX, |r| r == key).0 == 1
}

/// Whether `resource` has a staging buffer over title memory at `address`:
/// a texture placed there that lives yet (its staging buffer goes when it
/// is released). Only the address is used: `resource` may be released.
pub(crate) fn staged_at(resource: usize, address: u64) -> bool {
    let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    map.as_ref()
        .and_then(|map| map.get(&resource))
        .is_some_and(|s| s.memory as u64 == address)
}

/// The bytes of every staging buffer that lives.
pub(crate) fn staged_total() -> u64 {
    STAGED.load(Ordering::Relaxed)
}

/// Staging buffers released by [`release_staging`] so far: count and bytes.
pub(crate) fn released_total() -> (u64, u64) {
    let [count, bytes] = RELEASED.each_ref().map(|a| a.load(Ordering::Relaxed));
    (count, bytes)
}

/// Releases the staging buffers that no list holds any more and whose
/// texture had no upload recorded in the last `idle` calls, looking at up
/// to `looks` textures from where the last call stopped. A list holds its
/// textures' staging states until it is reset, and one reset while it runs
/// until its submission has run: a state held by nothing but [`STAGING`]
/// is read by no submission, so its buffer can go (its last copy ran, the
/// fence passed). What was detiled and copied is kept: the next upload
/// makes the buffer again ([`staging_within`]), and detiles and copies
/// only if the memory changed or another host copy needs it. Returns the
/// buffers released and their bytes, which leave [`staged_total`]. Once a
/// pass over the textures picked by index (`descriptor::host`).
///
/// Ported from fxceless/victory-fork (578d459) by fxceless: its
/// `texture::let_go_idle` lets go of a staging state held by no list
/// (`Arc::strong_count`) and counts the bytes that live by their drop.
/// Changed here: only the buffer goes, not what was detiled and copied
/// (the fork's next upload detiles and copies again, and here a pass's
/// look at a served texture would find one without a state changed:
/// `memory_change`); idle is counted in calls, a pass each, not 10 s and
/// 4 frames (a world loading leaves gigabytes waiting for room); a call
/// looks at `looks` textures in turns, and the buffers are destroyed out
/// of the map's lock.
pub(crate) fn release_staging(idle: u64, looks: usize) -> (u64, u64) {
    #[cfg(test)]
    let alone = SWEPT_ALONE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    #[cfg(not(test))]
    let alone: [usize; 0] = [];
    let now = SWEEPS.fetch_add(1, Ordering::Relaxed) + 1;
    let released = release_staging_of(now, idle, looks, |r| !alone.contains(&r));
    let _ = LAP.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |lap| {
        Some(lap - lap / 128)
    });
    if let Some(after) = evict_after() {
        evict_idle(now, after, looks, |r| !alone.contains(&r));
    }
    #[cfg(feature = "diagnostics")]
    census::due();
    released
}

/// Gives back the tiles of host copies (reserved textures of the pool)
/// whose texture no list holds and that neither had an upload recorded nor
/// a pass's look ([`memory_change`]) in the last `idle` passes, nor in two
/// laps of the pass's looks ([`LAP`]: a texture a shader may pick by index
/// is looked at once a lap): the title's streaming left them, or they wait in a view
/// heap's slot no pass serves. Looks at up to `looks` textures from where
/// the last call stopped, those `of` takes (tests). The texture stays: its
/// views read zero until its next upload (a draw reading it through a
/// table, a pass serving its slot) maps its tiles again and copies its
/// texels in, from title memory. GPU-written textures (whose contents are
/// the host's) are not in the map. Returns the host copies and the bytes
/// of tiles given back.
fn evict_idle(now: u64, idle: u64, looks: usize, of: impl Fn(usize) -> bool) -> (u64, u64) {
    evict_idle_at(now, ASKS.load(Ordering::Relaxed), idle, looks, of)
}

/// [`evict_idle`] at time `now` of [`SWEEPS`] and `asks` of [`ASKS`]
/// (tests: later times than other tests' passes see).
fn evict_idle_at(
    now: u64,
    asks: u64,
    idle: u64,
    looks: usize,
    of: impl Fn(usize) -> bool,
) -> (u64, u64) {
    use std::ops::Bound::{Excluded, Unbounded};
    let Some(host) = host() else {
        return (0, 0);
    };
    let (due, lap) = {
        let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
        let Some(map) = map.as_ref() else {
            return (0, 0);
        };
        // Two laps of the looks: the longest seen lately between two looks
        // at a texture, and at least one a texture.
        let lap = 2 * LAP.load(Ordering::Relaxed).max(map.len() as u64);
        let mut cursor = EVICT_CURSOR.lock().unwrap_or_else(|e| e.into_inner());
        let after = map.range((Excluded(*cursor), Unbounded));
        let before = map.range(..=*cursor);
        let mut due = Vec::new();
        for (&resource, s) in after.chain(before).take(looks.min(map.len())) {
            *cursor = resource;
            // Under the map's lock: one held by the map alone is held by no
            // list (each takes it from the map).
            if Arc::strong_count(s) == 1
                && of(resource)
                && s.idle(now, idle, asks, lap)
                && s.tiles().is_some_and(|t| t.any())
            {
                due.push(s.clone());
            }
        }
        (due, lap)
    };
    let mut gone = (0, 0);
    for s in due {
        if let Some(bytes) = s.evict(host, now, idle, asks, lap) {
            gone.0 += 1;
            gone.1 += bytes;
        }
    }
    EVICTED[0].fetch_add(gone.0, Ordering::Relaxed);
    EVICTED[1].fetch_add(gone.1, Ordering::Relaxed);
    gone
}

/// Host copies whose tiles were given back so far: count and bytes, and
/// those mapped again after it.
#[cfg(any(test, feature = "diagnostics"))]
fn evicted_total() -> [u64; 3] {
    EVICTED.each_ref().map(|a| a.load(Ordering::Relaxed))
}

/// The textures tests here sweep themselves (`tests::SweptAlone`): the
/// passes of other tests, whose sweeps go over every texture, leave them
/// as they are, so that a test tells what its own sweeps release, and
/// when. Before, such a pass could release a test's buffer first (once
/// passes elsewhere had advanced [`SWEEPS`] past its idle time).
#[cfg(test)]
static SWEPT_ALONE: Mutex<Vec<usize>> = Mutex::new(Vec::new());

/// [`release_staging`] at time `now` of [`SWEEPS`], for the resources `of`
/// takes (tests: other tests' textures are in the same map).
fn release_staging_of(now: u64, idle: u64, looks: usize, of: impl Fn(usize) -> bool) -> (u64, u64) {
    use std::ops::Bound::{Excluded, Unbounded};
    let mut gone = Vec::new();
    {
        let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
        let Some(map) = map.as_ref() else {
            return (0, 0);
        };
        let mut cursor = RELEASE_CURSOR.lock().unwrap_or_else(|e| e.into_inner());
        let after = map.range((Excluded(*cursor), Unbounded));
        let before = map.range(..=*cursor);
        for (&resource, s) in after.chain(before).take(looks.min(map.len())) {
            *cursor = resource;
            // Under the map's lock: a state held by the map alone gets no
            // other holder meanwhile (every one is taken from the map).
            if Arc::strong_count(s) != 1
                || now.saturating_sub(s.recorded.load(Ordering::Relaxed)) < idle
                || !of(resource)
            {
                continue;
            }
            if let Some(upload) = s.upload.lock().unwrap_or_else(|e| e.into_inner()).take() {
                gone.push(upload);
            }
        }
    }
    let bytes: u64 = gone.iter().map(|u| u.size).sum();
    let count = gone.len() as u64;
    RELEASED[0].fetch_add(count, Ordering::Relaxed);
    RELEASED[1].fetch_add(bytes, Ordering::Relaxed);
    // Destroyed out of the map's lock: each one is a call into the runtime.
    drop(gone);
    (count, bytes)
}

/// The bytes of title memory a texture the title placed itself takes, by
/// its resource description; `None` without a console layout for it.
pub(crate) fn placed_len(resource: &[u32; 14]) -> Option<usize> {
    let words = crate::resource::canonical_resource_desc(*resource);
    // SAFETY: 56 bytes in the description's layout, as `texture_at` reads
    // them.
    let desc = unsafe {
        words
            .as_ptr()
            .cast::<crate::resource::ResourceDesc>()
            .read_unaligned()
    };
    Some(crate::resource::texture_layout(&desc)?.size as usize)
}

/// Whether any of `[at, at + len)` is committed memory (a texture streaming
/// in has some of its pages mapped before the others, not always its
/// first).
pub(crate) fn any_committed(at: *const u8, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut p = at as usize;
    let end = p.saturating_add(len.max(1));
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        if n == 0 {
            return false;
        }
        if info.State == MEM_COMMIT {
            return true;
        }
        p = info.BaseAddress as usize + info.RegionSize;
    }
    false
}

/// Each copy decided in tests: its staging state's [`Staging::id`] and
/// whether it ran.
#[cfg(test)]
static DECIDED: Mutex<Vec<(u64, bool)>> = Mutex::new(Vec::new());

/// Each hash a check took in tests: its staging state's [`Staging::id`].
#[cfg(test)]
static HASHED: Mutex<Vec<u64>> = Mutex::new(Vec::new());

/// The ids of staging states in tests.
#[cfg(test)]
static IDS: AtomicU64 = AtomicU64::new(0);

/// GPU copies of staging buffers recorded, by whether they ran: count and
/// bytes (staging sizes), since the last summary.
static COPIES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
static COPIED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// An environment variable's value, if set.
fn env(name: &str) -> Option<String> {
    std::env::var_os(name).map(|v| v.to_string_lossy().into_owned())
}

/// Whether `value` turns off a switch that is on by default: `0`, `off`
/// or `false`, in any case.
fn off(value: &str) -> bool {
    let v = value.trim();
    v == "0" || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false")
}

/// What `XTP_TEXTURE_GUARDS` asks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GuardMode {
    /// `0`, `off`, `false`: every check hashes.
    Off,
    /// Unset or anything else: a check hashes only after a write (or when
    /// the safety net suspects the texture).
    On,
    /// `verify`, in a diagnostics build: every check the guards find clean
    /// is hashed too, a change counted as missed (a writer that opens no
    /// guard). Other builds take it as on.
    Verify,
}

/// The mode `value` asks for, in a diagnostics build or not.
fn guard_mode_of(value: Option<&str>, diagnostics: bool) -> GuardMode {
    match value.map(str::trim) {
        Some(v) if off(v) => GuardMode::Off,
        Some(v) if diagnostics && v.eq_ignore_ascii_case("verify") => GuardMode::Verify,
        _ => GuardMode::On,
    }
}

fn guard_mode() -> GuardMode {
    static MODE: OnceLock<GuardMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        let value = env("XTP_TEXTURE_GUARDS");
        let diagnostics = cfg!(feature = "diagnostics");
        let mode = guard_mode_of(value.as_deref(), diagnostics);
        if !diagnostics && value.is_some_and(|v| v.trim().eq_ignore_ascii_case("verify")) {
            tracing::warn!(
                "XTP_TEXTURE_GUARDS=verify needs a diagnostics build: the guards are on, \
                 checked only by the safety net"
            );
        }
        mode
    })
}

/// Whether texture memory in views is guarded, a hash taken only after a
/// write (`XTP_TEXTURE_GUARDS`).
fn guards() -> bool {
    guard_mode() != GuardMode::Off
}

/// Whether every check the guards find clean is hashed too
/// (`XTP_TEXTURE_GUARDS=verify`, diagnostics).
fn verify_guards() -> bool {
    guard_mode() == GuardMode::Verify
}

/// Changes the guards missed, found by the safety net or by `verify`,
/// since the start.
static MISSED: AtomicU64 = AtomicU64::new(0);

/// Whether texture copies run under a predicate (`XTP_TEXTURE_PREDICATION`,
/// on unless `0`, `off` or `false`).
pub(crate) fn predication() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| env("XTP_TEXTURE_PREDICATION").is_none_or(|v| !off(&v)))
}

/// The safety net's budget in bytes a frame, as `value` (of
/// `XTP_TEXTURE_NET`) asks: MiB, 2 by default; `0`, `off` or `false` turn
/// the net off.
fn net_budget_of(value: Option<&str>) -> u64 {
    const DEFAULT_MIB: u64 = 2;
    let mib = match value.map(str::trim) {
        None => DEFAULT_MIB,
        Some(v) if off(v) => 0,
        Some(v) => v.parse::<u64>().unwrap_or_else(|_| {
            tracing::warn!(value = v, "XTP_TEXTURE_NET is not a number of MiB: 2 MiB");
            DEFAULT_MIB
        }),
    };
    mib.min(1 << 20) << 20
}

/// The safety net's budget a frame (`XTP_TEXTURE_NET`), 0 when it is off or
/// the guards are (every check then hashes).
fn net_budget() -> u64 {
    static BUDGET: OnceLock<u64> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let budget = net_budget_of(env("XTP_TEXTURE_NET").as_deref());
        if guards() { budget } else { 0 }
    })
}

/// What the safety net may still hash: a budget a frame, or a second where
/// frames do not advance (a title that does not wait for them).
#[derive(Debug, Default)]
struct Allowance {
    left: u64,
    frame: u64,
    given: Option<Instant>,
}

impl Allowance {
    /// A new budget once the frame changed or a second went by since the
    /// last one; what was left of that one is not carried over.
    fn renew(&mut self, budget: u64, frame: u64, now: Instant) {
        let due = self.given.is_none_or(|at| {
            frame != self.frame || now.saturating_duration_since(at) >= Duration::from_secs(1)
        });
        if due {
            *self = Allowance {
                left: budget,
                frame,
                given: Some(now),
            };
        }
    }
}

/// What the safety net reads and hashes at a time: a texture bigger than
/// what is left of a budget is hashed over several.
const NET_SLICE: usize = 256 << 10;

/// The shortest time from one lap of the safety net to the next: textures
/// that take less than its budget are not hashed again at every frame.
const NET_LAP: Duration = Duration::from_secs(2);

/// Whether the safety net may begin a lap at `now`, the last one begun at
/// `last`.
fn lap_due(last: Option<Instant>, now: Instant) -> bool {
    last.is_none_or(|at| now.saturating_duration_since(at) >= NET_LAP)
}

/// The safety net (module doc): its budget, where its walk is, the texture
/// it is hashing.
#[derive(Default)]
struct Net {
    allowance: Allowance,
    /// The resource it last looked at: the walk goes on after it.
    cursor: usize,
    /// When it last went round from the last resource to the first.
    lap: Option<Instant>,
    hashing: Option<Hashing>,
    /// Room for what it reads.
    read: Vec<u8>,
}

/// A texture the safety net is hashing: how far it got, the hash so far,
/// and its staging buffer's writes when it began.
struct Hashing {
    staging: Arc<Staging>,
    at: usize,
    hasher: blake3::Hasher,
    written: u64,
}

impl Hashing {
    /// The whole memory hashed: the texture is suspect if the hash is not
    /// that of what was detiled, unless it was detiled again meanwhile
    /// (from a snapshot newer than some of what was hashed here).
    fn finish(self) {
        let hash = self.hasher.finalize();
        let last = self.staging.last.lock().unwrap_or_else(|e| e.into_inner());
        if self.staging.written.load(Ordering::Acquire) == self.written && last.hash != Some(hash) {
            self.staging.suspect.store(true, Ordering::Relaxed);
        }
    }
}

static NET: Mutex<Option<Net>> = Mutex::new(None);

/// The safety net's bytes hashed, microseconds and laps since the last
/// summary (diagnostics).
static NET_SEEN: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Frames and time for the safety net's budget and laps: the title's
/// frames and the clock, and in tests what a test adds ([`NET_LATER`]).
fn net_clock() -> (u64, Instant) {
    let (frame, now) = (crate::frame::current(), Instant::now());
    #[cfg(test)]
    let (frame, now) = {
        let [frames, ms] = NET_LATER.each_ref().map(|a| a.load(Ordering::Relaxed));
        (frame + frames, now + Duration::from_millis(ms))
    };
    (frame, now)
}

/// Frames and milliseconds tests add for the safety net: the title's frame
/// counter is `frame`'s tests' to advance, and tests do not wait for laps.
#[cfg(test)]
static NET_LATER: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// The safety net's turn, at a check (module doc): hashes what its budget
/// allows of the guarded textures checked since it last hashed them, in
/// resource order from where it stopped, a lap every [`NET_LAP`] at most,
/// and marks suspect those whose memory no longer hashes as what was
/// detiled.
fn net_walk() {
    let budget = net_budget();
    if budget == 0 {
        return;
    }
    let started = Instant::now();
    let (frame, now) = net_clock();
    let mut net = NET.lock().unwrap_or_else(|e| e.into_inner());
    let Net {
        allowance,
        cursor,
        lap,
        hashing,
        read,
    } = net.get_or_insert_with(Net::default);
    allowance.renew(budget, frame, now);
    // Textures looked at in this walk: a lap at most.
    let mut looked = 0;
    let mut hashed = 0;
    while allowance.left > 0 {
        let Some(h) = hashing else {
            *hashing = next_sampled(cursor, &mut looked, lap, now);
            if hashing.is_none() {
                // Nothing left to hash, or the next lap is not due: the
                // rest waits for the next budget.
                allowance.left = 0;
            }
            continue;
        };
        let size = h.staging.layout.size as usize;
        if h.at < size {
            let n = (size - h.at).min(NET_SLICE).min(allowance.left as usize);
            read.resize(NET_SLICE, 0);
            // Through the kernel: the title may unmap the memory meanwhile.
            if !crate::host::read_into(h.staging.memory.wrapping_add(h.at), &mut read[..n]) {
                // Not all mapped: its check sees to it.
                *hashing = None;
                continue;
            }
            h.hasher.update(&read[..n]);
            h.at += n;
            allowance.left -= n as u64;
            hashed += n as u64;
        }
        if h.at >= size
            && let Some(done) = hashing.take()
        {
            done.finish();
        }
    }
    NET_SEEN[0].fetch_add(hashed, Ordering::Relaxed);
    NET_SEEN[1].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
}

/// The next texture after `cursor` checked through its guards since the
/// safety net last hashed it, going round to the first resource if a lap is
/// due (`lap`: when the last one began); `looked` counts the textures looked
/// at, a lap at most.
fn next_sampled(
    cursor: &mut usize,
    looked: &mut usize,
    lap: &mut Option<Instant>,
    now: Instant,
) -> Option<Hashing> {
    use std::ops::Bound::{Excluded, Unbounded};
    let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let map = map.as_ref()?;
    let after = map.range((Excluded(*cursor), Unbounded));
    for (&resource, s) in after.chain(map.range(..=*cursor)) {
        if *looked >= map.len() {
            return None;
        }
        if resource <= *cursor {
            // Round again.
            if !lap_due(*lap, now) {
                return None;
            }
            *lap = Some(now);
            NET_SEEN[2].fetch_add(1, Ordering::Relaxed);
        }
        *looked += 1;
        *cursor = resource;
        if s.sampled.swap(false, Ordering::Relaxed) {
            return Some(Hashing {
                staging: s.clone(),
                at: 0,
                hasher: blake3::Hasher::new(),
                written: s.written.load(Ordering::Acquire),
            });
        }
    }
    None
}

/// Predicates for texture copies: 8-byte words of one upload buffer (in a
/// read state, which predication's is part of), one per recorded copy.
struct Predicates {
    buffer: ID3D12Resource,
    data: *mut u64,
    free: Mutex<Vec<u32>>,
}

// SAFETY: D3D12 objects are free-threaded; each word of the persistent
// mapping is written only by the holder of its `Predicate`.
unsafe impl Send for Predicates {}
// SAFETY: as above.
unsafe impl Sync for Predicates {}

/// Predicate words: 64 Ki (512 KiB); a world frame records a few thousand
/// copies, held until their lists are reset.
const PREDICATES: u32 = 1 << 16;

static PREDICATE_WORDS: OnceLock<Option<Predicates>> = OnceLock::new();

/// One copy's predicate word: the copy runs unless it is 0. It goes back
/// when its holder goes (the list's submission step, dropped once the list
/// is reset and nothing running uses its recording).
pub(crate) struct Predicate {
    of: &'static Predicates,
    at: u32,
}

impl Predicate {
    /// A free word, set to run (so a copy whose step never ran would run);
    /// `None` without predication or once they are all taken (logged once).
    pub(crate) fn take() -> Option<Predicate> {
        let of = PREDICATE_WORDS
            .get_or_init(|| {
                if !predication() {
                    return None;
                }
                let (buffer, data) = host()?.upload_buffer(u64::from(PREDICATES) * 8)?;
                crate::host::name(&buffer, || "texture copy predicates".into());
                Some(Predicates {
                    buffer,
                    data: data.cast(),
                    free: Mutex::new((0..PREDICATES).rev().collect()),
                })
            })
            .as_ref()?;
        let Some(at) = of.free.lock().unwrap_or_else(|e| e.into_inner()).pop() else {
            once!("texture copy predicates ran out: copies run every time");
            return None;
        };
        let p = Predicate { of, at };
        p.set(true);
        Some(p)
    }

    pub(crate) fn bind(&self, list: &HostList) {
        // SAFETY: the predicate word stays allocated until this list finishes.
        unsafe {
            list.list.SetPredication(
                &self.of.buffer,
                self.offset(),
                D3D12_PREDICATION_OP_EQUAL_ZERO,
            )
        };
    }

    pub(crate) fn unbind(list: &HostList) {
        // SAFETY: the command list is open.
        unsafe {
            list.list
                .SetPredication(None::<&ID3D12Resource>, 0, D3D12_PREDICATION_OP_EQUAL_ZERO)
        };
    }

    fn offset(&self) -> u64 {
        u64::from(self.at) * 8
    }

    /// At the list's submission: its copy runs if `due`. A submission that
    /// names the list again runs it again only once this run has finished
    /// (`Host::run_as`), so no other run reads the word meanwhile. Returns
    /// whether the copy runs.
    pub(crate) fn decide(&self, due: bool) -> bool {
        self.set(due);
        due
    }

    /// Before the list runs: whether its copy runs.
    fn set(&self, runs: bool) {
        // SAFETY: this word of the mapped buffer is ours; the list that
        // reads it is not running (its submission is being prepared).
        unsafe {
            self.of
                .data
                .add(self.at as usize)
                .write_volatile(u64::from(runs))
        };
    }
}

impl Drop for Predicate {
    fn drop(&mut self) {
        self.of
            .free
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(self.at);
    }
}

/// A buffer-to-texture copy into subresource `sub` of `dst` at `at`, from
/// title buffer `src` laid out as `footprint` (title format), when `dst`'s
/// host format stands in for the footprint's: recorded here, the title
/// bytes converted into an upload buffer at submission. False when the
/// formats agree (the caller copies as it is).
#[allow(clippy::too_many_arguments)] // the copy's own arguments
pub(crate) fn copy_converted(
    list: &HostList,
    dst: This,
    sub: u32,
    at: [u32; 3],
    src: This,
    footprint: &crate::resource::PlacedFootprint,
    source_box: *const [u32; 6],
) -> bool {
    let Some(to) = crate::resource::host_resource(dst) else {
        return false;
    };
    // SAFETY: a live resource.
    let host_format = unsafe { to.GetDesc() }.Format;
    let Some(convert) =
        crate::resource::format::conversion(footprint.format(), host_format.0 as u32)
    else {
        return false;
    };
    let Some(convert) = convert else {
        once!(
            format = footprint.format(),
            "buffer-to-texture copy into a stand-in without a conversion dropped"
        );
        return true;
    };
    let (Some(host), Some(bytes), Some((data, size))) = (
        host(),
        footprint.bytes(),
        crate::resource::buffer_bytes(src),
    ) else {
        return true;
    };
    if footprint
        .offset
        .checked_add(bytes)
        .is_none_or(|end| end > size)
    {
        tracing::warn!(
            offset = footprint.offset,
            bytes,
            size,
            "copy outside its buffer"
        );
        return true;
    }
    let (width, height, depth, title_pitch) = footprint.shape();
    let (from, to_bytes) = convert.bytes();
    let lines = height as usize * depth as usize;
    let host_pitch = (width as usize * to_bytes).next_multiple_of(256);
    let host_bytes = host_pitch * lines.saturating_sub(1) + width as usize * to_bytes;
    let Some((buffer, mapped)) = list.upload(host, host_bytes as u64) else {
        return true;
    };
    // At submission, when the title has written the buffer.
    let (title, mapped) = (data as usize + footprint.offset as usize, mapped as usize);
    let read = title_pitch as usize * lines.saturating_sub(1) + width as usize * from;
    list.defer_host((title, read), move || {
        let mut texels = vec![0u8; width as usize * from];
        for line in 0..lines {
            let row = (title + line * title_pitch as usize) as *const u8;
            if !crate::host::read_into(row, &mut texels) {
                texels.fill(0);
            }
            // SAFETY: row `line` of the mapped upload buffer, `width`
            // texels of the host format.
            let out = unsafe {
                std::slice::from_raw_parts_mut(
                    (mapped + line * host_pitch) as *mut u8,
                    width as usize * to_bytes,
                )
            };
            convert.row(&texels, out);
        }
    });
    const COPY_DEST: u32 = 0x400;
    crate::list::ensure_state(list, dst, COPY_DEST);
    gpu_write(list, dst, false);
    let to_location = D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the location never releases.
        pResource: unsafe { std::mem::transmute_copy(to) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: sub,
        },
    };
    let from_location = D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(&buffer) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: 0,
                Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                    Format: host_format,
                    Width: width,
                    Height: height,
                    Depth: depth,
                    RowPitch: host_pitch as u32,
                },
            },
        },
    };
    let source_box = (!source_box.is_null()).then(|| source_box.cast::<D3D12_BOX>());
    // SAFETY: an open list; both locations name live resources (the upload
    // buffer lives until the list is reset); the box is the title's (the
    // host's layout, in texels of either format).
    unsafe {
        list.list.CopyTextureRegion(
            &to_location,
            at[0],
            at[1],
            at[2],
            &from_location,
            source_box,
        )
    };
    true
}

fn is_movie_plane(
    tiles_none: bool,
    convert_none: bool,
    layout: &xtp_formats::texture::Layout,
    footprints: &[D3D12_PLACED_SUBRESOURCE_FOOTPRINT],
) -> bool {
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8_UNORM;
    let desc = &layout.desc;
    tiles_none
        && convert_none
        && desc.dimension == xtp_formats::texture::Dimension::Tex2d
        && desc.swizzle == xtp_formats::texture::Swizzle::LINEAR
        && desc.depth == 1
        && desc.mips == 1
        && desc.samples == 1
        && footprints.len() == 1
        && footprints[0].Offset == 0
        && footprints[0].Footprint.Format == DXGI_FORMAT_R8_UNORM
}

impl Staging {
    /// Whether this texture is a CPU-written linear 2D single-mip `R8_UNORM`
    /// plane (such as a video decoder plane), uploaded into its own per-list
    /// upload buffer instead of the shared staging buffer.
    // Ported from edsuuu/VI-BR-PORT commit fd36f4a4 (GPL-3.0-or-later).
    fn movie_plane(&self) -> bool {
        is_movie_plane(
            self.tiles.is_none(),
            self.convert.is_none(),
            &self.layout,
            &self.footprints,
        )
    }

    /// The title memory the texture is read from: start and end.
    pub(crate) fn range(&self) -> (usize, usize) {
        let start = self.memory as usize;
        (start, start + self.layout.size as usize)
    }

    /// The staging buffer, if it has one now.
    fn buffer(&self) -> Option<ID3D12Resource> {
        let upload = self.upload.lock().unwrap_or_else(|e| e.into_inner());
        upload.as_ref().map(|u| u.buffer.clone())
    }

    /// The staging buffer's mapping, if it has one now.
    fn mapped(&self) -> Option<*mut u8> {
        let upload = self.upload.lock().unwrap_or_else(|e| e.into_inner());
        upload.as_ref().map(|u| u.data)
    }

    /// Its host copy's tiles, if it is a reserved texture that lives.
    fn tiles(&self) -> Option<Arc<crate::resource::pool::Tiles>> {
        self.tiles.as_ref()?.upgrade()
    }

    /// Whether it had neither an upload recorded nor a pass's look in the
    /// `idle` passes before `now`, nor a look in the `lap` looks before
    /// `asks`.
    fn idle(&self, now: u64, idle: u64, asks: u64, lap: u64) -> bool {
        let since = |at: &AtomicU64| now.saturating_sub(at.load(Ordering::Relaxed));
        since(&self.recorded) >= idle
            && since(&self.asked) >= idle
            && asks.saturating_sub(self.asked_at.load(Ordering::Relaxed)) >= lap
    }

    /// [`evict_idle`] for this texture, found idle and held by no list:
    /// gives its tiles back if it still is (under the lock its checks
    /// take, so that no mapping of a check is queued between). The bytes
    /// given back.
    fn evict(
        self: &Arc<Self>,
        host: &crate::host::Host,
        now: u64,
        idle: u64,
        asks: u64,
        lap: u64,
    ) -> Option<u64> {
        let _checks = self.last.lock().unwrap_or_else(|e| e.into_inner());
        // Held by the map and the sweep only: no list took it meanwhile.
        if Arc::strong_count(self) > 2 || !self.idle(now, idle, asks, lap) {
            return None;
        }
        let tiles = self.tiles()?;
        let bytes = tiles.mapped();
        if tiles.evict(host) == 0 {
            return None;
        }
        // Its host copy holds nothing now: its next upload detiles and
        // copies whatever the memory says, mapping the units its title
        // memory fills.
        *self.copied.lock().unwrap_or_else(|e| e.into_inner()) = None;
        self.incomplete.store(true, Ordering::Release);
        self.evicted.store(true, Ordering::Relaxed);
        self.force.store(true, Ordering::Release);
        Some(bytes)
    }

    /// Whether units of its tiles its title memory filled are not mapped
    /// (left for want of room, or given back) and a resident heap has room
    /// for them now: its next upload maps them.
    fn wants_tiles(&self) -> bool {
        self.incomplete.load(Ordering::Acquire) && self.room()
    }

    /// Whether a resident heap has room now for the units of its tiles its
    /// title memory filled that are not mapped.
    fn room(&self) -> bool {
        let Some(tiles) = self.tiles() else {
            return true;
        };
        let wanted = self.wanted.lock().unwrap_or_else(|e| e.into_inner());
        tiles.room_for(&wanted)
    }

    /// Before a detiling: maps the units of its host copy's tiles whose
    /// title memory is committed and unmaps the others (the title streamed
    /// them out, or never in), so that the copy that follows brings the
    /// texels of those it fills. Units no resident heap has room for are
    /// left: the texture is incomplete until they are mapped. Before,
    /// every host copy held memory for its whole mip chain: xcl #258 found
    /// 1.3 GB of title memory behind 22,000 host textures of 7.9 GB.
    fn map_units(&self) {
        let (Some(tiles), Some(host)) = (self.tiles(), host()) else {
            return;
        };
        let wanted = self.filled_units(&tiles);
        let done = tiles.map(host, &wanted);
        self.incomplete.store(done.short > 0, Ordering::Release);
        if done.added > 0 && self.evicted.swap(false, Ordering::Relaxed) {
            EVICTED[2].fetch_add(1, Ordering::Relaxed);
        }
        *self.wanted.lock().unwrap_or_else(|e| e.into_inner()) = wanted;
    }

    /// Which units of `tiles` its title memory fills now: those whose
    /// mips' title memory is all committed (in a 3D texture, all of them if
    /// the whole texture's is).
    fn filled_units(&self, tiles: &crate::resource::pool::Tiles) -> Vec<bool> {
        let parts = committed_parts(self.memory, self.layout.size as usize);
        self.units_in(tiles, &parts)
    }

    /// Which units of `tiles` are covered by the committed byte ranges `parts`.
    fn units_in(
        &self,
        tiles: &crate::resource::pool::Tiles,
        parts: &[(usize, usize)],
    ) -> Vec<bool> {
        let layout = &self.layout;
        let covered = |start: u64, end: u64| {
            parts
                .iter()
                .any(|&(a, b)| a as u64 <= start && end <= b as u64)
        };
        let all = covered(0, layout.size);
        let three_d = layout.desc.dimension == xtp_formats::texture::Dimension::Tex3d;
        tiles
            .units()
            .iter()
            .map(|u| {
                if all || three_d {
                    return all;
                }
                (u.first_mip..u.first_mip + u.mips).all(|mip| {
                    title_mip(layout, u.slice, mip).map_or(all, |(start, end)| covered(start, end))
                })
            })
            .collect()
    }

    /// Whether host copy `target` holds what was detiled last, as decided
    /// at the submissions so far (a copy into it from that detiling ran or
    /// was found not needed), and every unit of its tiles the title memory
    /// filled is mapped (or none of those left out can be now).
    fn holds(&self, target: Option<&Weak<crate::resource::HostTexture>>) -> bool {
        let Some(target) = target else {
            return false;
        };
        if self.wants_tiles() {
            return false;
        }
        let written = self.written.load(Ordering::Acquire);
        let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        copied
            .as_ref()
            .is_some_and(|(w, t)| *w == written && Weak::ptr_eq(t, target))
    }

    /// [`memory_change`] for this texture: whether its memory changed since
    /// it was last detiled, as the write watch or the guards tell.
    fn change(&self) -> Change {
        // Never detiled (its memory was not mapped at the checks so far).
        if self.suspect.load(Ordering::Relaxed) || self.written.load(Ordering::Acquire) == 0 {
            return Change::Yes;
        }
        let asked = self.last.lock().unwrap_or_else(|e| e.into_inner()).asked;
        let (start, end) = self.range();
        let since = || {
            if crate::watch::changed_since(start, end, asked) {
                Change::Yes
            } else {
                Change::No
            }
        };
        if crate::watch::writes(start, end, false).is_some() {
            return since();
        }
        if guards() && crate::watch::writes(start, end, true).is_some() {
            let change = since();
            if change == Change::Yes {
                return change;
            }
            if !verify_guards() {
                self.sampled.store(true, Ordering::Relaxed);
                return change;
            }
        }
        Change::Unknown(self.layout.size)
    }

    /// Whether its staging buffer was released and an upload into host copy
    /// `target` is left to the check at submission ([`upload_late`]): the
    /// host copy has what was detiled last, and the memory is not written
    /// since as the write watch or the guards tell, or only a hash tells
    /// (taken at submission, before anything is made again).
    fn late(&self, target: Option<&Weak<crate::resource::HostTexture>>) -> bool {
        let released = self
            .upload
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_none();
        released
            && !self.host_owned.load(Ordering::Relaxed)
            && self.holds(target)
            && self.change() != Change::Yes
    }

    /// Makes its staging buffer again if it was released, within `room`
    /// bytes: the bytes made (0 if it had one or uses per-list movie-plane
    /// upload buffers). The new one is blank.
    fn make_upload(&self, room: u64) -> Result<u64, Staged> {
        if self.movie_plane() {
            return Ok(0);
        }
        let mut upload = self.upload.lock().unwrap_or_else(|e| e.into_inner());
        if upload.is_some() {
            return Ok(0);
        }
        if self.size > room {
            return Err(Staged::NoRoom(self.size));
        }
        let host = host().ok_or(Staged::NotNow)?;
        let Some((buffer, data)) = host.upload_buffer(self.size) else {
            once!(
                size = self.size,
                "no upload memory for a texture's staging buffer made again: asked again at its next upload"
            );
            return Err(Staged::NotNow);
        };
        crate::host::name(&buffer, || {
            format!("staging of title memory {:?} (made again)", self.memory)
        });
        STAGED.fetch_add(self.size, Ordering::Relaxed);
        *upload = Some(Upload {
            buffer,
            data,
            size: self.size,
        });
        self.blank.store(true, Ordering::Release);
        Ok(self.size)
    }

    /// Detiles the texture memory into the staging buffer if it changed
    /// since the last time (and is mapped at all). `before_write` runs
    /// just before the staging buffer is overwritten: an earlier
    /// submission's GPU copy may still read it. The safety net takes its
    /// turn first (counted with the check).
    pub(crate) fn refresh(&self, before_write: impl FnOnce()) -> usize {
        net_walk();
        if self.host_owned.load(Ordering::Relaxed) {
            0
        } else {
            self.refresh_now(before_write)
        }
    }

    /// Returns the bytes of title memory read.
    fn refresh_now(&self, before_write: impl FnOnce()) -> usize {
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        // A blank buffer a copy is due from (`upload`): detiled whatever the
        // memory's checks say, as at its first check.
        let forced = self.force.swap(false, Ordering::AcqRel);
        self.check(&mut last, forced, before_write)
    }

    /// When a copy no predicate decides is recorded ([`upload`]): detiles
    /// the memory into a blank staging buffer, whatever its checks say, as
    /// its first check does. Whether the buffer holds a detiling now: not
    /// if the memory is not mapped, or a GPU write into the texture was
    /// submitted (its host copy keeps the host's contents). No copy reads a
    /// blank buffer (its decision at submission skips it, or none was
    /// recorded), so none is waited for; one that is not blank any more
    /// (detiled by a check meanwhile) is not written again.
    fn fill(&self) -> bool {
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        if !self.blank.load(Ordering::Acquire) {
            return true;
        }
        if self.host_owned.load(Ordering::Relaxed) {
            return false;
        }
        self.check(&mut last, true, || ());
        !self.blank.load(Ordering::Acquire)
    }

    /// [`Staging::refresh_now`] under the lock of `last`; `forced`: the
    /// memory is detiled whatever the write watch, the guards or the hash
    /// say, if it can be read. What tells a change is left as it was when
    /// nothing is detiled: the memory not mapped, or no staging buffer to be
    /// had. Before, a forced check forgot it, and the next one read the
    /// memory whatever it said.
    fn check(&self, last: &mut Last, forced: bool, before_write: impl FnOnce()) -> usize {
        let size = self.layout.size as usize;
        let start = self.memory as usize;
        #[cfg(feature = "diagnostics")]
        {
            let submission = crate::timestamps::submissions();
            self.seen.checks.fetch_add(1, Ordering::Relaxed);
            if last.checked.replace(submission) == Some(submission) {
                self.seen.again.fetch_add(1, Ordering::Relaxed);
            }
        }
        if let Some(now) = crate::watch::writes(start, start + size, false) {
            if !forced
                && last.asked != 0
                && !crate::watch::changed_since(start, start + size, last.asked)
            {
                return 0;
            }
            let Some(read) = self.snapshot(size) else {
                // Not mapped (streamed out): nothing to detile, the host
                // copy keeps its texels until a write tells new bytes.
                // Before, the write watch, which answers for a reservation
                // all decommitted (no page written), let it be read as
                // zeros, detiled and copied over valid texels (Codex's
                // review of db37732, 2).
                last.asked = now;
                return 0;
            };
            if !self.stage_again() {
                give_room(read);
                return 0;
            }
            last.asked = now;
            self.before_write(before_write);
            self.detile(&read);
            #[cfg(feature = "diagnostics")]
            census::detiled(&mut last.detiled, blake3::hash(&read));
            give_room(read);
            self.written.fetch_add(1, Ordering::Release);
            #[cfg(feature = "diagnostics")]
            {
                crate::copystat::add(crate::copystat::C::TexWatched, size as u64);
                self.seen.changed.fetch_add(1, Ordering::Relaxed);
            }
            return size;
        }
        // View memory: guarded, and hashed only after a write (or where a
        // block cannot be guarded, which every asking reports written).
        // Whether the guards reported a write (the memory most likely
        // changed), or found the memory clean though the safety net suspects
        // it or `verify` asks (a change is then one they missed).
        let (mut reported, mut unseen) = (false, false);
        if guards()
            && let Some(now) = crate::watch::writes(start, start + size, true)
        {
            let clean = !forced
                && last.asked != 0
                && !crate::watch::changed_since(start, start + size, last.asked);
            last.asked = now;
            // A write they report is hashed below anyway.
            let suspect = self.suspect.swap(false, Ordering::Relaxed);
            if clean {
                #[cfg(feature = "diagnostics")]
                self.seen.clean.fetch_add(1, Ordering::Relaxed);
                if !suspect && !verify_guards() {
                    // Answered by the guards: the safety net's to hash.
                    self.sampled.store(true, Ordering::Relaxed);
                    return 0;
                }
                unseen = true;
            } else {
                reported = true;
            }
            // Hashed below: nothing for the safety net until the next check.
            self.sampled.store(false, Ordering::Relaxed);
        }
        // A reserved host copy takes the units whose memory is committed
        // (`map_units`): a texture streaming in shows the mips it has. Any
        // other waits for all of it.
        let whole = committed(self.memory, size);
        if !whole && !(self.tiles.is_some() && any_committed(self.memory, size)) {
            #[cfg(feature = "diagnostics")]
            tracing::debug!(
                memory = ?self.memory,
                size,
                desc = ?self.layout.desc,
                "a sampled texture's memory is not committed"
            );
            return 0;
        }
        // Memory the guards report written most likely changed: its
        // snapshot is taken at once. Otherwise it is hashed in place first
        // (all of it committed), and copied only if that tells a change.
        if !forced && !reported && whole {
            // SAFETY: committed memory of the texture's layout size.
            let tiled = unsafe { std::slice::from_raw_parts(self.memory, size) };
            if last.hash == Some(self.hash(tiled)) {
                return size;
            }
        }
        // Hashed and detiled from one snapshot (module doc), taken through
        // the kernel, which tells pages unmapped since the check above. The
        // room is shared, and not held during the wait.
        let Some(snapshot) = self.snapshot(size) else {
            return 0;
        };
        let hash = self.hash(&snapshot);
        if (forced || last.hash != Some(hash)) && self.stage_again() {
            last.hash = Some(hash);
            if unseen {
                self.missed(size);
            }
            self.before_write(before_write);
            self.detile(&snapshot);
            self.written.fetch_add(1, Ordering::Release);
            #[cfg(feature = "diagnostics")]
            {
                crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
                self.seen.changed.fetch_add(1, Ordering::Relaxed);
                census::detiled(&mut last.detiled, hash);
            }
        }
        give_room(snapshot);
        size
    }

    /// The texture's memory, `size` bytes of it, read into the room the
    /// checks share ([`take_room`]) through the kernel, which tells pages
    /// that are not mapped: all of them mapped, or, for a reserved host copy,
    /// at least one unit of its tiles whose memory is all readable
    /// (`map_units` maps those units, the copy leaves the others; what is not
    /// mapped reads as zeros). `None` otherwise (including hole-only reads
    /// with no complete committed unit): nothing is detiled from it, so the
    /// host copy keeps its staged texels and digest.
    // Attribution: addresses sparse hole read digest preservation from
    // TaynanGT/Vice-Tay commit 26aeded1 (GPL-3.0-or-later) using committed
    // range/unit coverage rather than zero-content filtering.
    fn snapshot(&self, size: usize) -> Option<Vec<u8>> {
        let mut room = take_room(size);
        if crate::host::read_into(self.memory, &mut room) {
            return Some(room);
        }
        if let Some(tiles) = self.tiles()
            && any_committed(self.memory, size)
        {
            let parts = crate::host::copy_mapped_parts(room.as_mut_ptr(), self.memory, size, false);
            if !parts.is_empty() && self.units_in(&tiles, &parts).into_iter().any(|u| u) {
                return Some(room);
            }
        }
        give_room(room);
        None
    }

    /// Before a detiling: its staging buffer, made again if it was released
    /// (the check at submission of an upload left to it found the memory
    /// written: [`upload_late`]). False if none can be had now: the change
    /// waits for a later check.
    fn stage_again(&self) -> bool {
        self.make_upload(u64::MAX).is_ok()
    }

    /// Runs `before_write` (the wait for an earlier submission whose copy
    /// may read the staging buffer) unless the buffer is blank: no copy
    /// read it nor reads it (each needs a detiling first), and a buffer made
    /// again does not wait for the submissions that read the one before.
    fn before_write(&self, before_write: impl FnOnce()) {
        if !self.blank.load(Ordering::Acquire) {
            before_write();
        }
    }

    /// The hash of `bytes`, this texture's memory or its snapshot.
    fn hash(&self, bytes: &[u8]) -> blake3::Hash {
        #[cfg(test)]
        HASHED
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(self.id);
        #[cfg(feature = "diagnostics")]
        {
            crate::copystat::add(crate::copystat::C::TexHashed, bytes.len() as u64);
            self.seen
                .hashed
                .fetch_add(bytes.len() as u64, Ordering::Relaxed);
        }
        blake3::hash(bytes)
    }

    /// A change its guards missed, found by the safety net or by `verify`:
    /// counted, and logged the first time for this texture.
    fn missed(&self, size: usize) {
        let n = MISSED.fetch_add(1, Ordering::Relaxed) + 1;
        if self.missed.fetch_add(1, Ordering::Relaxed) == 0 {
            tracing::warn!(
                memory = ?self.memory,
                size,
                desc = ?self.layout.desc,
                missed = n,
                "a texture changed that its guards found clean (a writer that opens no guard): \
                 refreshed"
            );
        }
    }

    /// At submission, once the staging buffer is up to date: whether a copy
    /// into `target` has to run for it to hold the staging buffer's bytes.
    /// It is then taken to hold them (the copy runs in submission order,
    /// before any later list's).
    fn copy_due(&self, target: &Weak<crate::resource::HostTexture>) -> bool {
        let written = self.written.load(Ordering::Acquire);
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let due = !copied
            .as_ref()
            .is_some_and(|(w, t)| *w == written && Weak::ptr_eq(t, target));
        *copied = Some((written, target.clone()));
        due
    }

    /// Detiles `tiled` (the texture's memory) into the staging buffer, which
    /// is not blank from then on (a subresource that could not be detiled
    /// stays as it was, as before buffers were released).
    fn detile(&self, tiled: &[u8]) {
        // A list holding this state holds the buffer: there is one.
        let Some(data) = self.mapped() else {
            return;
        };
        self.map_units();
        match self.convert {
            Some(convert) => self.detile_converted(tiled, convert, data),
            None => self.detile_plain(tiled, data),
        }
        self.blank.store(false, Ordering::Release);
    }

    /// [`Staging::detile`] into `data`, the mapped staging buffer, in the
    /// host format.
    fn detile_plain(&self, tiled: &[u8], data: *mut u8) {
        #[cfg(feature = "diagnostics")]
        if tiled.iter().all(|&b| b == 0) {
            tracing::debug!(
                memory = ?self.memory,
                desc = ?self.layout.desc,
                "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
            );
        }
        for (i, (footprint, &rows)) in self.footprints.iter().zip(&self.rows).enumerate() {
            let (mip, slice) = (i as u32 % self.mips, i as u32 / self.mips);
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let len = pitch * rows as usize * f.Depth as usize;
            // SAFETY: the subresource's part of the mapped staging buffer.
            let linear =
                unsafe { std::slice::from_raw_parts_mut(data.add(footprint.Offset as usize), len) };
            if let Err(e) =
                self.layout
                    .detile(tiled, mip, slice, linear, pitch, pitch * rows as usize)
            {
                tracing::warn!(?e, mip, slice, "texture not detiled");
                return;
            }
        }
    }
}

impl Staging {
    /// As [`Staging::detile`], into a stand-in format: each subresource
    /// detiled in the title's format, then converted row by row into
    /// `data`, the mapped staging buffer.
    fn detile_converted(&self, tiled: &[u8], convert: Convert, data: *mut u8) {
        let (from, to) = convert.bytes();
        let mut title = Vec::new();
        for (i, (footprint, &rows)) in self.footprints.iter().zip(&self.rows).enumerate() {
            let (mip, slice) = (i as u32 % self.mips, i as u32 / self.mips);
            let f = footprint.Footprint;
            let width = f.Width as usize;
            let (pitch, lines) = (width * from, rows as usize * f.Depth as usize);
            title.clear();
            title.resize(pitch * lines, 0);
            if let Err(e) =
                self.layout
                    .detile(tiled, mip, slice, &mut title, pitch, pitch * rows as usize)
            {
                tracing::warn!(?e, mip, slice, "texture not detiled");
                return;
            }
            let host_pitch = f.RowPitch as usize;
            for (line, texels) in title.chunks_exact(pitch).enumerate() {
                // SAFETY: row `line` of the subresource's part of the mapped
                // staging buffer, `width` texels of the host format.
                let row = unsafe {
                    std::slice::from_raw_parts_mut(
                        data.add(footprint.Offset as usize + line * host_pitch),
                        width * to,
                    )
                };
                convert.row(texels, row);
            }
        }
    }
}

/// The committed parts of `[at, at + len)`, as offsets from `at` (start,
/// end), neighbours merged.
fn committed_parts(at: *const u8, len: usize) -> Vec<(usize, usize)> {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let base = at as usize;
    let end = base.saturating_add(len);
    let mut parts: Vec<(usize, usize)> = Vec::new();
    let mut p = base;
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        if n == 0 {
            break;
        }
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if next <= p {
            break;
        }
        if info.State == MEM_COMMIT {
            let (start, stop) = (p - base, next.min(end) - base);
            match parts.last_mut() {
                Some(last) if last.1 == start => last.1 = stop,
                _ => parts.push((start, stop)),
            }
        }
        p = next;
    }
    parts
}

/// Where mip `mip` of array slice `slice` lies in a texture's title memory
/// of layout `layout` (offsets, start and end): a mip in the mip tail is
/// the tail block's (at the start of the slice); a standard mip runs to
/// the next larger one (mips are placed from the smallest up). `None`
/// outside the layout.
fn title_mip(layout: &xtp_formats::texture::Layout, slice: u32, mip: u32) -> Option<(u64, u64)> {
    let mips = &layout.mips;
    let m = mip as usize;
    if m >= mips.len() || slice >= layout.slices.max(1) {
        return None;
    }
    let tail = (layout.first_tail_mip as usize).min(mips.len());
    let (start, end) = if m >= tail {
        let end = if tail > 0 {
            mips[tail - 1].offset
        } else {
            layout.slice_size
        };
        (0, end)
    } else {
        let end = if m == 0 {
            layout.slice_size
        } else {
            mips[m - 1].offset
        };
        (mips[m].offset, end)
    };
    let base = u64::from(slice) * layout.slice_size;
    (start < end && base + end <= layout.size).then_some((base + start, base + end))
}

/// Whether `[at, at + len)` is committed memory (streamed textures may be
/// unmapped until their pages arrive).
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut p = at as usize;
    let end = p + len;
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        if n == 0 || info.State != MEM_COMMIT {
            return false;
        }
        p = info.BaseAddress as usize + info.RegionSize;
    }
    true
}

/// The staging state of a title texture, made on first use; why there is
/// none if it cannot be made.
fn staging(resource: This, texture: &ID3D12Resource) -> Result<Arc<Staging>, Staged> {
    staging_within(resource, texture, u64::MAX).map(|(s, _)| s)
}

/// [`staging`], made only within `room` bytes ([`Staged::NoRoom`] past
/// them), with the bytes made (0 if it was there). A state whose staging
/// buffer was released ([`release_staging`]) gets one again, blank.
fn staging_within(
    resource: This,
    texture: &ID3D12Resource,
    room: u64,
) -> Result<(Arc<Staging>, u64), Staged> {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let map = map.get_or_insert_with(BTreeMap::new);
    if let Some(s) = map.get(&(resource as usize)) {
        let made = s.make_upload(room)?;
        return Ok((s.clone(), made));
    }
    let (memory, desc) = crate::resource::texture_memory(resource).ok_or(Staged::Never)?;
    let Some(layout) = crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())
    else {
        once!(
            ?desc,
            "texture contents without a console layout not uploaded"
        );
        return Err(Staged::Never);
    };
    if layout.desc.swizzle.xor() {
        once!(?desc, "texture contents in an XOR mode not uploaded");
        return Err(Staged::Never);
    }
    let host = host().ok_or(Staged::NotNow)?;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let convert = match crate::resource::format::conversion(desc.format(), d.Format.0 as u32) {
        Some(None) => {
            once!(
                ?desc,
                "texture contents in a format whose stand-in has no conversion not uploaded"
            );
            return Err(Staged::Never);
        }
        Some(convert) => convert,
        None => None,
    };
    let count = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(d.MipLevels)
    } else {
        u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize)
    };
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut total = 0;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            0,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    let tiles = crate::resource::pool::tiles_of(texture).map(|t| Arc::downgrade(&t));
    let (upload, made) = if is_movie_plane(tiles.is_none(), convert.is_none(), &layout, &footprints)
    {
        tracing::debug!(?desc, "movie plane uploaded via per-list upload buffers");
        (None, 0)
    } else {
        if total > room {
            return Err(Staged::NoRoom(total));
        }
        let Some((buffer, data)) = host.upload_buffer(total) else {
            once!(
                ?desc,
                total,
                "no upload memory for a texture's staging buffer: asked again at its next upload"
            );
            return Err(Staged::NotNow);
        };
        crate::host::name(&buffer, || {
            format!(
                "staging of {resource:?} {}x{}x{} f{}",
                d.Width, d.Height, d.DepthOrArraySize, d.Format.0
            )
        });
        tracing::debug!(?desc, total, "texture contents uploaded from title memory");
        // Under a target of its own, one line a texture: which title memory
        // gets uploaded at all, through tables or by index
        // (`XTP_LOG=info,xtp_gpu::texture_memory=debug`).
        tracing::debug!(
            target: "xtp_gpu::texture_memory",
            memory = ?memory,
            width = d.Width,
            height = d.Height,
            format = d.Format.0,
            bytes = layout.size,
            "a texture the title fills gets a staging buffer"
        );
        (
            Some(Upload {
                buffer,
                data,
                size: total,
            }),
            total,
        )
    };
    let s = Arc::new(Staging {
        upload: Mutex::new(upload),
        blank: AtomicBool::new(true),
        force: AtomicBool::new(false),
        recorded: AtomicU64::new(SWEEPS.load(Ordering::Relaxed)),
        size: total,
        footprints,
        rows,
        layout,
        convert,
        mips: u32::from(d.MipLevels),
        memory,
        last: Mutex::default(),
        written: AtomicU64::new(0),
        copied: Mutex::new(None),
        sampled: AtomicBool::new(false),
        suspect: AtomicBool::new(false),
        missed: AtomicU64::new(0),
        host_owned: AtomicBool::new(false),
        tiles,
        wanted: Mutex::new(Vec::new()),
        incomplete: AtomicBool::new(false),
        evicted: AtomicBool::new(false),
        asked: AtomicU64::new(SWEEPS.load(Ordering::Relaxed)),
        asked_at: AtomicU64::new(ASKS.load(Ordering::Relaxed)),
        looked: AtomicBool::new(false),
        #[cfg(feature = "diagnostics")]
        host_bytes: crate::resource::host_copy(resource).map_or(0, |h| h.bytes),
        #[cfg(feature = "diagnostics")]
        born: BORN.fetch_add(1, Ordering::Relaxed),
        #[cfg(test)]
        id: IDS.fetch_add(1, Ordering::Relaxed),
        #[cfg(feature = "diagnostics")]
        seen: Seen::default(),
    });
    if made > 0 {
        STAGED.fetch_add(made, Ordering::Relaxed);
    }
    map.insert(resource as usize, s.clone());
    Ok((s, made))
}

/// Every few seconds (diagnostics): per frame, the textures checked, how
/// often, the bytes hashed and the changes, the staging copies recorded and
/// those that ran, the safety net's work, and the textures that took most
/// hashing.
#[cfg(feature = "diagnostics")]
fn summary_due() {
    static NEXT: Mutex<Option<(Instant, u64)>> = Mutex::new(None);
    let (frames, seconds) = {
        let mut next = NEXT.lock().unwrap_or_else(|e| e.into_inner());
        let now = Instant::now();
        let frame = crate::frame::current();
        match *next {
            None => {
                *next = Some((now, frame));
                return;
            }
            Some((at, _)) if now.duration_since(at) < Duration::from_secs(5) => return,
            Some((at, from)) => {
                *next = Some((now, frame));
                (
                    frame.saturating_sub(from).max(1) as f64,
                    now.duration_since(at).as_secs_f64(),
                )
            }
        }
    };
    let take = |a: &AtomicU64| a.swap(0, Ordering::Relaxed) as f64 / frames;
    let mb = |bytes: f64| bytes / 1e6;
    /// One texture's line, per frame.
    struct Row {
        checks: f64,
        again: f64,
        clean: f64,
        hashed_mb: f64,
        changes: f64,
        staging: u64,
        memory: usize,
        desc: String,
    }
    let mut rows: Vec<Row> = {
        let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
        map.iter()
            .flat_map(|m| m.values())
            .map(|s| Row {
                checks: take(&s.seen.checks),
                again: take(&s.seen.again),
                clean: take(&s.seen.clean),
                hashed_mb: mb(take(&s.seen.hashed)),
                changes: take(&s.seen.changed),
                staging: s.size,
                memory: s.memory as usize,
                desc: format!(
                    "{}x{}x{}, {} mips, {}-byte elements of {} texels square, {:?}, {:.2} MB in \
                     title memory",
                    s.layout.desc.width,
                    s.layout.desc.height,
                    s.layout.desc.depth,
                    s.mips,
                    s.layout.desc.element_bytes,
                    s.layout.desc.block,
                    s.layout.desc.swizzle,
                    s.layout.size as f64 / 1e6
                ),
            })
            .filter(|r| r.checks > 0.0)
            .collect()
    };
    let [skipped, ran] = [0, 1].map(|i| (take(&COPIES[i]), mb(take(&COPIED[i]))));
    let missed = MISSED.load(Ordering::Relaxed);
    let sum = |f: fn(&Row) -> f64| -> f64 { rows.iter().map(f).sum() };
    tracing::info!(
        "texture checks per frame: {} textures checked {:.0} times ({:.0} again in a submission \
         that had checked them, {:.0} clean by their guards), {:.1} MB hashed, {:.1} changes; \
         staging copies {:.0} ran ({:.1} MB), {:.0} skipped ({:.1} MB)",
        rows.len(),
        sum(|r| r.checks),
        sum(|r| r.again),
        sum(|r| r.clean),
        sum(|r| r.hashed_mb),
        sum(|r| r.changes),
        ran.0,
        ran.1,
        skipped.0,
        skipped.1
    );
    let [net_mb, net_ms] = [mb(take(&NET_SEEN[0])), take(&NET_SEEN[1]) / 1e3];
    let laps = NET_SEEN[2].swap(0, Ordering::Relaxed);
    tracing::info!(
        "texture safety net per frame: {net_mb:.2} MB hashed in {net_ms:.2} ms (budget {:.1} MB); \
         {laps} laps of the textures in {seconds:.1} s; changes the guards missed so far \
         {missed}",
        mb(net_budget() as f64)
    );
    rows.sort_by(|a, b| {
        b.hashed_mb
            .total_cmp(&a.hashed_mb)
            .then(b.checks.total_cmp(&a.checks))
    });
    for r in rows.iter().take(12) {
        tracing::info!(
            "  texture at {:#x} ({}, staging {:.2} MB): {:.1} checks/frame ({:.1} again, {:.1} \
             clean), {:.1} MB hashed, {:.2} changes",
            r.memory,
            r.desc,
            r.staging as f64 / 1e6,
            r.checks,
            r.again,
            r.clean,
            r.hashed_mb,
            r.changes
        );
    }
}

/// Slots a pass over the textures shaders pick by index left for want of
/// video memory, and the bytes of their textures (`draw.rs`
/// `heap_textures`), since the last census.
static LEFT_FOR_ROOM: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// A pass left a slot whose texture of `bytes` the video memory left could
/// not take (for the census).
pub(crate) fn note_left_for_room(bytes: u64) {
    LEFT_FOR_ROOM[0].fetch_add(1, Ordering::Relaxed);
    LEFT_FOR_ROOM[1].fetch_add(bytes, Ordering::Relaxed);
}

/// The order staging states were made in (the census tells a texture
/// another one's memory was given to after it).
#[cfg(feature = "diagnostics")]
static BORN: AtomicU64 = AtomicU64::new(0);

/// Detilings of a texture after its first: of the same bytes as the one
/// before, of other bytes (Samuel's watch, item 402: fork #821 found
/// textures uploaded before their DirectStorage read ended, 2,186 of 2,671
/// uploads again with other bytes).
#[cfg(feature = "diagnostics")]
static REDETILED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Every 10 s (diagnostics), one line: the host copies of the textures the
/// title fills, by state (task cba81604, step 1), beside the frame log's
/// totals of host textures (those without a staging state were never
/// uploaded) and the pool's line.
#[cfg(feature = "diagnostics")]
mod census {
    use super::*;

    /// Passes within which a texture counts as read lately: an upload
    /// recorded (a draw through a table, a pass) or a pass's look.
    const LATELY: u64 = 64;

    /// The census's period.
    const EVERY: Duration = Duration::from_secs(10);

    /// A detiling of the bytes that hash `now`, after one of `before`.
    pub(super) fn detiled(before: &mut Option<blake3::Hash>, now: blake3::Hash) {
        if let Some(was) = before.replace(now) {
            REDETILED[usize::from(was != now)].fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Takes a census once [`EVERY`] (at a pass): the map's states, walked
    /// on a thread of its own (it queries their memory).
    pub(super) fn due() {
        static NEXT: Mutex<Option<Instant>> = Mutex::new(None);
        let now = Instant::now();
        {
            let mut next = NEXT.lock().unwrap_or_else(|e| e.into_inner());
            if next.is_some_and(|at| now < at) {
                return;
            }
            *next = Some(now + EVERY);
        }
        let states: Vec<Weak<Staging>> = {
            let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
            map.iter()
                .flat_map(|m| m.values())
                .map(Arc::downgrade)
                .collect()
        };
        let (sweeps, asks) = (SWEEPS.load(Ordering::Relaxed), ASKS.load(Ordering::Relaxed));
        let gpu_written = GPU_WRITTEN
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .map_or(0, |s| s.len());
        let spawned = std::thread::Builder::new()
            .name("xtp-texture-census".into())
            .spawn(move || take(&states, sweeps, asks, gpu_written));
        if let Err(e) = spawned {
            tracing::debug!(error = %e, "no texture census thread");
        }
    }

    /// Textures and their host copies' bytes.
    #[derive(Default, Clone, Copy)]
    struct Count {
        n: u64,
        bytes: u64,
    }

    impl Count {
        fn add(&mut self, bytes: u64) {
            self.n += 1;
            self.bytes += bytes;
        }
    }

    impl std::fmt::Display for Count {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{} ({} MiB)", self.n, self.bytes >> 20)
        }
    }

    /// One texture's part of the census.
    struct Seen {
        start: usize,
        end: usize,
        born: u64,
        bytes: u64,
    }

    fn take(states: &[Weak<Staging>], sweeps: u64, asks: u64, gpu_written: usize) {
        let started = Instant::now();
        let (mut all, mut lately, mut idle, mut never) = Default::default();
        let (mut gone, mut partly, mut tiled, mut incomplete) = Default::default();
        let (mut replaced, mut idle_lap): (Count, Count) = Default::default();
        let (mut uncommitted, mut mapped) = (0u64, 0u64);
        let lap = 2 * LAP.load(Ordering::Relaxed).max(states.len() as u64);
        let mut seen = Vec::with_capacity(states.len());
        for s in states.iter().filter_map(Weak::upgrade) {
            let s: Arc<Staging> = s;
            let bytes = s.host_bytes;
            Count::add(&mut all, bytes);
            let since = |at: &AtomicU64| sweeps.saturating_sub(at.load(Ordering::Relaxed));
            if since(&s.recorded).min(since(&s.asked)) < LATELY {
                Count::add(&mut lately, bytes);
            } else {
                Count::add(&mut idle, bytes);
                if asks.saturating_sub(s.asked_at.load(Ordering::Relaxed)) >= lap {
                    Count::add(&mut idle_lap, bytes);
                }
            }
            if s.written.load(Ordering::Acquire) == 0 {
                Count::add(&mut never, bytes);
            }
            let size = s.layout.size as usize;
            let parts = committed_parts(s.memory, size);
            let committed: usize = parts.iter().map(|&(a, b)| b - a).sum();
            if committed == 0 {
                Count::add(&mut gone, bytes);
            } else if committed < size {
                Count::add(&mut partly, bytes);
                uncommitted += uncommitted_bytes(&s, &parts);
            }
            if let Some(tiles) = s.tiles() {
                Count::add(&mut tiled, bytes);
                mapped += tiles.mapped();
            }
            if s.incomplete.load(Ordering::Relaxed) {
                Count::add(&mut incomplete, bytes);
            }
            seen.push(Seen {
                start: s.memory as usize,
                end: s.memory as usize + size,
                born: s.born,
                bytes,
            });
        }
        // Memory another texture made later also covers: the title gave it
        // to a new texture (or views it twice).
        seen.sort_by_key(|t| t.start);
        let mut older = vec![false; seen.len()];
        for i in 0..seen.len() {
            for j in i + 1..seen.len() {
                if seen[j].start >= seen[i].end {
                    break;
                }
                let first = if seen[i].born < seen[j].born { i } else { j };
                older[first] = true;
            }
        }
        for (t, _) in seen.iter().zip(&older).filter(|(_, o)| **o) {
            Count::add(&mut replaced, t.bytes);
        }
        let [evicted, evicted_bytes, again] = evicted_total();
        let [left, left_bytes] = LEFT_FOR_ROOM
            .each_ref()
            .map(|a| a.swap(0, Ordering::Relaxed));
        let [same, other] = REDETILED.each_ref().map(|a| a.load(Ordering::Relaxed));
        tracing::info!(
            "texture census: {all} textures with a staging state (host copies' bytes); read in the \
             last {LATELY} passes {lately}, not {idle} (of them not looked at by a pass for two \
             laps of its looks {idle_lap}); never given title bytes {never}; title memory gone \
             (none committed) {gone}; partly committed {partly}, {} MiB of their host copies \
             over uncommitted title memory; overlapped by a newer texture {replaced}; reserved \
             {tiled}, {} MiB of tiles mapped, incomplete {incomplete}; GPU-written {gpu_written}; \
             tiles given back so far {evicted} ({} MiB), mapped again {again}; detiled again \
             {same} times with the same bytes, {other} with other bytes; slots left for video \
             memory since the last census {left} ({} MiB); a lap of the passes' looks {} \
             looks; census {} ms",
            uncommitted >> 20,
            mapped >> 20,
            evicted_bytes >> 20,
            left_bytes >> 20,
            LAP.load(Ordering::Relaxed),
            started.elapsed().as_millis()
        );
    }

    /// The bytes of `s`'s host copy whose title memory is not all
    /// committed (`parts`), subresource by subresource (their share of the
    /// staging buffer's footprints).
    fn uncommitted_bytes(s: &Staging, parts: &[(usize, usize)]) -> u64 {
        let covered = |start: u64, end: u64| {
            parts
                .iter()
                .any(|&(a, b)| a as u64 <= start && end <= b as u64)
        };
        let mut out = 0u64;
        for (i, (footprint, &rows)) in s.footprints.iter().zip(&s.rows).enumerate() {
            let (mip, slice) = (i as u32 % s.mips, i as u32 / s.mips);
            let f = footprint.Footprint;
            let linear = u64::from(f.RowPitch) * u64::from(rows) * u64::from(f.Depth);
            if !title_mip(&s.layout, slice, mip).is_some_and(|(a, b)| covered(a, b)) {
                out += linear;
            }
        }
        // As a share of the host copy's bytes.
        out.saturating_mul(s.host_bytes) / s.size.max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::tests_support::{committed, read_back, sampled};

    const COPY_DEST: u32 = 0x400;

    /// The decisions made for `staging` so far, in order.
    fn decided(staging: &Arc<Staging>) -> Vec<bool> {
        decided_id(staging.id)
    }

    /// The decisions made so far for the staging state of id `key`.
    fn decided_id(key: u64) -> Vec<bool> {
        DECIDED
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|(k, _)| *k == key)
            .map(|&(_, ran)| ran)
            .collect()
    }

    /// Runs one host list per entry of `lists` in one submission, each
    /// sampling `resource` (its staging copy recorded as a draw does).
    fn sample_in(host: &crate::host::Host, resource: This, lists: usize) {
        let texture = crate::resource::host_resource(resource).expect("a host copy");
        let made: Vec<HostList> = (0..lists).map(|_| host.new_list().unwrap()).collect();
        for list in &made {
            upload(list, resource, texture);
            list.close();
        }
        let runs: Vec<&HostList> = made.iter().collect();
        host.execute(&runs);
        host.drain();
    }

    /// A texture the title fills is copied into its host copy by the first
    /// list that samples it, and not by the others, nor by later lists
    /// until its memory changes: the GPU skips their copies (a staging
    /// buffer written behind its back does not reach the host copy), and a
    /// change is copied by the next list.
    #[test]
    fn a_texture_is_copied_once_until_its_memory_changes() {
        let test = "texture::tests::a_texture_is_copied_once_until_its_memory_changes";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        if !predication() {
            return;
        }
        let resource = committed(&sampled(16, 16, 28));
        let started = std::time::Instant::now();
        while crate::resource::host_resource(resource).is_none() {
            assert!(
                started.elapsed().as_secs() < 5,
                "the host copy never became resident"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let texture = crate::resource::host_resource(resource).unwrap();
        let (memory, _) = crate::resource::texture_memory(resource).unwrap();
        let size = staging(resource, texture).unwrap().layout.size as usize;
        let fill = |byte: u8| {
            // SAFETY: the texture's own title memory, `size` bytes.
            unsafe { memory.cast_mut().write_bytes(byte, size) }
        };
        let texels = || -> Vec<u8> {
            let read = read_back(host, texture, COPY_DEST);
            (0..16)
                .flat_map(|y| read[y * 256..y * 256 + 64].to_vec())
                .collect()
        };
        let staging = staging(resource, texture).unwrap();

        // Two lists in one submission: the first copies, the second not.
        fill(0x11);
        sample_in(host, resource, 2);
        assert_eq!(decided(&staging), [true, false]);
        assert!(texels().iter().all(|&b| b == 0x11));

        // The memory did not change: the next list's copy is skipped on
        // the GPU, so what is in the staging buffer does not get through.
        // SAFETY: the mapped staging buffer, `size` bytes; nothing runs.
        unsafe {
            staging
                .mapped()
                .unwrap()
                .write_bytes(0x22, staging.size as usize)
        };
        sample_in(host, resource, 1);
        assert_eq!(decided(&staging), [true, false, false]);
        assert!(
            texels().iter().all(|&b| b == 0x11),
            "a skipped copy ran: {:x?}",
            &texels()[..16]
        );

        // A change is detiled and copied by the next list.
        fill(0x33);
        sample_in(host, resource, 2);
        assert_eq!(decided(&staging), [true, false, false, true, false]);
        assert!(texels().iter().all(|&b| b == 0x33));
        type Release = extern "system" fn(This) -> u32;
        assert_eq!(crate::object::slot::<Release>(resource, 0x10)(resource), 0);
    }

    /// A copy into another host copy is due even though the staging buffer
    /// did not change (the title's resource got a new host copy), and so
    /// is one into the first again after it.
    #[test]
    fn another_host_copy_is_copied_into() {
        let test = "texture::tests::another_host_copy_is_copied_into";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let [first, second] = [0, 1].map(|_| committed(&sampled(16, 16, 28)));
        let started = std::time::Instant::now();
        while [first, second]
            .iter()
            .any(|&r| crate::resource::host_copy(r).is_none())
        {
            assert!(
                started.elapsed().as_secs() < 5,
                "the host copies never became resident"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let [a, b] = [first, second].map(|r| crate::resource::host_copy(r).unwrap());
        let staging = staging(first, &a.texture).unwrap();
        let [a, b] = [&a, &b].map(Arc::downgrade);
        assert!(staging.copy_due(&a), "never copied");
        assert!(!staging.copy_due(&a), "copied, not changed");
        assert!(staging.copy_due(&b), "another host copy");
        assert!(staging.copy_due(&a), "the first one got the other's turn");
        staging.written.fetch_add(1, Ordering::Release);
        assert!(staging.copy_due(&a), "written again");
        assert!(!staging.copy_due(&a));
        type Release = extern "system" fn(This) -> u32;
        for r in [first, second] {
            assert_eq!(crate::object::slot::<Release>(r, 0x10)(r), 0);
        }
    }

    /// The times `staging` was hashed so far.
    fn hashes(staging: &Arc<Staging>) -> usize {
        let key = staging.id;
        HASHED
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|&&k| k == key)
            .count()
    }

    /// A texture in view memory (physical pages, which cannot be
    /// write-watched) is guarded: checks without a write in between take no
    /// hash and copy nothing; a write by the CPU (which faults), or one the
    /// writer reports (as xmem does after DirectStorage writes through a
    /// view of its own), is hashed, detiled and copied.
    #[test]
    fn a_texture_in_a_view_is_hashed_only_after_a_write() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, PAGE_READWRITE,
        };
        let test = "texture::tests::a_texture_in_a_view_is_hashed_only_after_a_write";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        if !guards() || !predication() {
            return;
        }
        // Guards are the process's: one test that guards views at a time
        // (the watch's own tests count what is guarded).
        let _serial = crate::watch::tests::serial();
        const LEN: usize = 64 << 10;
        // SAFETY: a fresh pagefile section mapped twice, read-write: the
        // title's view and a writer's own (as xmem's is).
        let (view, other) = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                LEN as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, LEN).Value as usize;
            let other = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, LEN).Value as usize;
            windows_sys::Win32::Foundation::CloseHandle(section);
            assert!(view != 0 && other != 0);
            (view, other)
        };
        let resource =
            crate::resource::tests_support::at_address(view as u64, &sampled(16, 16, 28));
        let started = std::time::Instant::now();
        while crate::resource::host_resource(resource).is_none() {
            assert!(
                started.elapsed().as_secs() < 5,
                "the host copy never became resident"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let texture = crate::resource::host_resource(resource).unwrap();
        let staging = staging(resource, texture).unwrap();
        let size = staging.layout.size as usize;
        assert!(size <= LEN);
        let texels = || -> Vec<u8> {
            let read = read_back(host, texture, COPY_DEST);
            (0..16)
                .flat_map(|y| read[y * 256..y * 256 + 64].to_vec())
                .collect()
        };
        // SAFETY: bytes of the title's view, as its CPU writes them (a
        // fault into the guards' handler where guarded).
        let fill = |at: usize, byte: u8| unsafe { (at as *mut u8).write_bytes(byte, size) };

        fill(view, 0x44);
        sample_in(host, resource, 1);
        assert_eq!((hashes(&staging), decided(&staging)), (1, vec![true]));
        assert!(texels().iter().all(|&b| b == 0x44));

        // Nothing written: no hash, no copy.
        sample_in(host, resource, 2);
        assert_eq!(
            (hashes(&staging), decided(&staging)),
            (1, vec![true, false, false])
        );

        // The CPU writes through the guarded view.
        fill(view, 0x55);
        sample_in(host, resource, 1);
        assert_eq!(hashes(&staging), 2);
        assert_eq!(decided(&staging).last(), Some(&true));
        assert!(texels().iter().all(|&b| b == 0x55));

        // A writer through its own view, which reports what it wrote.
        fill(other, 0x66);
        crate::watch::writing(view, size);
        sample_in(host, resource, 1);
        assert_eq!(hashes(&staging), 3);
        assert_eq!(decided(&staging).last(), Some(&true));
        assert!(texels().iter().all(|&b| b == 0x66));

        type Release = extern "system" fn(This) -> u32;
        assert_eq!(crate::object::slot::<Release>(resource, 0x10)(resource), 0);
        // Its guards go back (other tests count what is guarded); the views
        // stay mapped until the tests end: the watch's journal keeps the
        // pages these checks found written, which a test that allocates
        // fresh memory at the same address would read as its own.
        crate::watch::title_memory_remapped(view, LEN);
        let _ = other;
    }

    /// A list named twice in one submission runs twice, the second run
    /// once the first has finished (D3D12 message 553 otherwise, #45): the
    /// first copies, the second finds the texture current and does not.
    /// Submitted again in a later submission, nothing changed, it does not
    /// either. The layer raises nothing.
    #[test]
    fn a_list_named_twice_in_a_submission_copies_once() {
        let test = "texture::tests::a_list_named_twice_in_a_submission_copies_once";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        if !predication() {
            return;
        }
        let resource = committed(&sampled(16, 16, 28));
        let started = std::time::Instant::now();
        while crate::resource::host_resource(resource).is_none() {
            assert!(
                started.elapsed().as_secs() < 5,
                "the host copy never became resident"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let texture = crate::resource::host_resource(resource).unwrap();
        let (memory, _) = crate::resource::texture_memory(resource).unwrap();
        let staging = staging(resource, texture).unwrap();
        let size = staging.layout.size as usize;
        // SAFETY: the texture's own title memory, `size` bytes.
        unsafe { memory.cast_mut().write_bytes(0x11, size) };
        sample_in(host, resource, 1);
        // SAFETY: as above.
        unsafe { memory.cast_mut().write_bytes(0x77, size) };
        let list = host.new_list().unwrap();
        upload(&list, resource, texture);
        list.close();
        host.execute(&[&list, &list]);
        host.drain();
        assert_eq!(decided(&staging), [true, true, false]);
        let read = read_back(host, texture, COPY_DEST);
        let texels: Vec<u8> = (0..16)
            .flat_map(|y| read[y * 256..y * 256 + 64].to_vec())
            .collect();
        assert!(texels.iter().all(|&b| b == 0x77), "{:x?}", &texels[..16]);
        // The same recording in a submission of its own: nothing changed
        // since the last run, so it does not copy either.
        host.execute(&[&list]);
        host.drain();
        assert_eq!(decided(&staging), [true, true, false, false]);
        type Release = extern "system" fn(This) -> u32;
        assert_eq!(crate::object::slot::<Release>(resource, 0x10)(resource), 0);
    }

    #[test]
    fn switches_turn_off_with_0_off_or_false() {
        for v in ["0", "off", "OFF", "false", "False", " 0 "] {
            assert!(off(v), "{v:?}");
            assert_eq!(guard_mode_of(Some(v), true), GuardMode::Off, "{v:?}");
            assert_eq!(net_budget_of(Some(v)), 0, "{v:?}");
        }
        for v in ["1", "on", "yes", "", "verify"] {
            assert!(!off(v), "{v:?}");
        }
        assert_eq!(guard_mode_of(None, true), GuardMode::On);
        assert_eq!(guard_mode_of(Some("1"), false), GuardMode::On);
        // `verify` only where its counters are: on otherwise (and warned).
        assert_eq!(guard_mode_of(Some("verify"), true), GuardMode::Verify);
        assert_eq!(guard_mode_of(Some("Verify"), true), GuardMode::Verify);
        assert_eq!(guard_mode_of(Some("verify"), false), GuardMode::On);
        assert_eq!(net_budget_of(None), 2 << 20);
        assert_eq!(net_budget_of(Some("8")), 8 << 20);
        assert_eq!(net_budget_of(Some("lots")), 2 << 20);
    }

    /// The safety net's budget comes once a frame, whatever is spent of it,
    /// and once a second where frames do not advance; its laps come two
    /// seconds apart at least.
    #[test]
    fn the_net_is_given_its_budget_once_a_frame() {
        let t0 = Instant::now();
        let mut a = Allowance::default();
        a.renew(100, 7, t0);
        assert_eq!(a.left, 100);
        a.left = 30;
        // The same frame: what is left stays.
        a.renew(100, 7, t0 + Duration::from_millis(500));
        assert_eq!(a.left, 30);
        // The next frame: a budget, not what was left on top.
        a.renew(100, 8, t0 + Duration::from_millis(600));
        assert_eq!(a.left, 100);
        a.left = 0;
        // No frame for a second: a budget all the same.
        a.renew(100, 8, t0 + Duration::from_millis(1_500));
        assert_eq!(a.left, 0);
        a.renew(100, 8, t0 + Duration::from_millis(1_600));
        assert_eq!(a.left, 100);
        // Laps: the first at once, the next one not before `NET_LAP`.
        assert!(lap_due(None, t0));
        assert!(!lap_due(Some(t0), t0 + NET_LAP / 2));
        assert!(lap_due(Some(t0), t0 + NET_LAP));
    }

    /// The next frame for the safety net, a lap after the last.
    fn net_next_frame() {
        NET_LATER[0].fetch_add(1, Ordering::Relaxed);
        NET_LATER[1].fetch_add(NET_LAP.as_millis() as u64, Ordering::Relaxed);
    }

    /// Two read-write views of one fresh pagefile section of `len` bytes:
    /// the title's and a writer's own (as xmem's is).
    fn section_views(len: usize) -> (usize, usize) {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, PAGE_READWRITE,
        };
        // SAFETY: a fresh pagefile section mapped twice, read-write.
        unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                len as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, len).Value as usize;
            let other = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, len).Value as usize;
            windows_sys::Win32::Foundation::CloseHandle(section);
            assert!(view != 0 && other != 0);
            (view, other)
        }
    }

    /// What the safety net's work costs a frame, out of the game: its
    /// budget read through the kernel a slice at a time and hashed, over a
    /// 64 MiB view (more than the CPU's caches, as the title's texture pool
    /// is), against hashing it in place; and a check's snapshot (copied,
    /// then hashed) against the hash in place it replaces where the guards
    /// report a write. Best of three passes. Run with
    /// `cargo test --release -p xtp-gpu -- --ignored --nocapture net_costs`.
    #[test]
    #[ignore = "benchmark"]
    fn net_costs() {
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
        const LEN: usize = 64 << 20;
        let (view, other) = section_views(LEN);
        // SAFETY: the view, `LEN` bytes, ours alone.
        let memory = unsafe { std::slice::from_raw_parts_mut(view as *mut u8, LEN) };
        for (i, b) in memory.iter_mut().enumerate() {
            *b = (i.wrapping_mul(2_654_435_761) >> 13) as u8;
        }
        let budget = net_budget_of(None) as usize;
        let rounds = LEN / budget;
        // Milliseconds per budget's bytes, the best of three passes over
        // the view.
        let time = |f: &mut dyn FnMut(usize)| -> f64 {
            (0..3)
                .map(|_| {
                    let started = Instant::now();
                    for r in 0..rounds {
                        f(view + r * budget);
                    }
                    started.elapsed().as_secs_f64() * 1e3 / rounds as f64
                })
                .fold(f64::INFINITY, f64::min)
        };
        let mut read = vec![0u8; NET_SLICE];
        let net = time(&mut |at| {
            let mut hasher = blake3::Hasher::new();
            for off in (0..budget).step_by(NET_SLICE) {
                assert!(crate::host::read_into((at + off) as *const u8, &mut read));
                hasher.update(&read);
            }
            std::hint::black_box(hasher.finalize());
        });
        let in_place = time(&mut |at| {
            // SAFETY: inside the view.
            let bytes = unsafe { std::slice::from_raw_parts(at as *const u8, budget) };
            std::hint::black_box(blake3::hash(bytes));
        });
        let mut snapshot = vec![0u8; budget];
        let snap = time(&mut |at| {
            crate::host::copy_mapped(snapshot.as_mut_ptr(), at as *const u8, budget, false);
            std::hint::black_box(blake3::hash(&snapshot));
        });
        eprintln!(
            "texture safety net, per {} MiB (its budget a frame): {net:.3} ms read through the \
             kernel and hashed ({:.0} MB/s), {in_place:.3} ms hashed in place; a check's \
             snapshot, copied and hashed: {snap:.3} ms",
            budget >> 20,
            budget as f64 / 1e3 / net
        );
        for at in [view, other] {
            // SAFETY: views mapped above, not used after.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: at as *mut _,
                })
            };
        }
    }

    /// A title texture over the title's view `view`, once its host copy is
    /// resident.
    fn texture_in(view: usize) -> (This, &'static ID3D12Resource) {
        square_texture_in(view, 16)
    }

    /// As [`texture_in`], `width` texels square.
    fn square_texture_in(view: usize, width: u64) -> (This, &'static ID3D12Resource) {
        let resource = crate::resource::tests_support::at_address(
            view as u64,
            &sampled(width, width as u32, 28),
        );
        let started = std::time::Instant::now();
        loop {
            if let Some(texture) = crate::resource::host_resource(resource) {
                return (resource, texture);
            }
            assert!(
                started.elapsed().as_secs() < 5,
                "the host copy never became resident"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    /// The first rows of `texture` (16x16, 4 bytes a texel), as the host
    /// copy holds them.
    fn texels_of(host: &crate::host::Host, texture: &ID3D12Resource) -> Vec<u8> {
        let read = read_back(host, texture, COPY_DEST);
        (0..16)
            .flat_map(|y| read[y * 256..y * 256 + 64].to_vec())
            .collect()
    }

    /// A check hashes the memory it detiles: the title writing during the
    /// wait for an earlier copy, then writing back the bytes that were
    /// there, leaves the host copy with what the memory holds (the hash
    /// taken before the wait matched the bytes written back, and the
    /// staging buffer kept the write made during it).
    #[test]
    fn a_texture_is_hashed_as_it_is_detiled() {
        let test = "texture::tests::a_texture_is_hashed_as_it_is_detiled";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        if !guards() || !predication() {
            return;
        }
        let _serial = crate::watch::tests::serial();
        const LEN: usize = 64 << 10;
        let (view, other) = section_views(LEN);
        let (resource, texture) = texture_in(view);
        let staging = staging(resource, texture).unwrap();
        let size = staging.layout.size as usize;
        assert!(size <= LEN);
        // SAFETY: bytes of the title's view, as its CPU writes them (a
        // fault into the guards' handler where guarded).
        let fill = |byte: u8| unsafe { (view as *mut u8).write_bytes(byte, size) };

        fill(0x11);
        sample_in(host, resource, 1);
        assert!(texels_of(host, texture).iter().all(|&b| b == 0x11));

        // Written: the next check hashes and detiles, and while it waits
        // for an earlier copy the title writes again.
        fill(0x22);
        staging.refresh(|| fill(0x33));
        // Then writes back what the check took.
        fill(0x22);
        sample_in(host, resource, 1);
        let texels = texels_of(host, texture);
        assert!(texels.iter().all(|&b| b == 0x22), "{:x?}", &texels[..16]);

        type Release = extern "system" fn(This) -> u32;
        assert_eq!(crate::object::slot::<Release>(resource, 0x10)(resource), 0);
        // As in `a_texture_in_a_view_is_hashed_only_after_a_write`.
        crate::watch::title_memory_remapped(view, LEN);
        let _ = other;
    }

    /// A write no guard sees (through another view of the title's pages,
    /// which nothing guards nor reports) is found by the safety net within
    /// a frame's budget: the texture is refreshed at the check after its
    /// walk, the change counted and logged as missed, and found clean again
    /// after.
    #[test]
    fn a_write_no_guard_saw_is_found_by_the_safety_net() {
        let test = "texture::tests::a_write_no_guard_saw_is_found_by_the_safety_net";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        if !guards() || !predication() || net_budget() == 0 {
            return;
        }
        let _serial = crate::watch::tests::serial();
        const LEN: usize = 64 << 10;
        let (view, other) = section_views(LEN);
        let (resource, texture) = texture_in(view);
        let staging = staging(resource, texture).unwrap();
        let size = staging.layout.size as usize;
        // The budget covers it: what it hashes is the guarded textures
        // checked since, and the other tests' that guard textures take the
        // same lock as this one.
        assert!(size as u64 <= net_budget());
        // SAFETY: bytes of the title's view, or the other one, `size` of
        // them.
        let fill = |at: usize, byte: u8| unsafe { (at as *mut u8).write_bytes(byte, size) };

        fill(view, 0x44);
        sample_in(host, resource, 1);
        assert!(texels_of(host, texture).iter().all(|&b| b == 0x44));
        // Clean: no hash at the check.
        let hashed = hashes(&staging);
        sample_in(host, resource, 1);
        assert_eq!(hashes(&staging), hashed);

        // Through the other view, reported to no one.
        fill(other, 0x99);
        let missed = MISSED.load(Ordering::Relaxed);
        // Each check in a frame of its own a lap after the last, the net's
        // budget given anew. The walk at a check hashes the texture if a
        // check found it clean
        // since the net last hashed it: the last one did, unless another
        // test's walk (every check walks) hashed it before the write; then
        // this check finds it clean and the next walk hashes it. Changed,
        // it is refreshed by the check that walk comes before.
        let mut frames = 0;
        while !texels_of(host, texture).iter().all(|&b| b == 0x99) {
            frames += 1;
            assert!(frames <= 2, "not refreshed in {} frames", frames - 1);
            net_next_frame();
            sample_in(host, resource, 1);
        }
        assert_eq!(staging.missed.load(Ordering::Relaxed), 1);
        assert!(MISSED.load(Ordering::Relaxed) > missed);
        assert_eq!(decided(&staging).last(), Some(&true));

        // Hashed again by the net, it is not suspect: its checks stay clean.
        let hashed = hashes(&staging);
        for _ in 0..2 {
            net_next_frame();
            sample_in(host, resource, 1);
        }
        assert_eq!(hashes(&staging), hashed);
        assert_eq!(staging.missed.load(Ordering::Relaxed), 1);

        type Release = extern "system" fn(This) -> u32;
        assert_eq!(crate::object::slot::<Release>(resource, 0x10)(resource), 0);
        crate::watch::title_memory_remapped(view, LEN);
    }

    /// The look a pass over the textures shaders pick by index takes before
    /// it records an upload (`memory_change`): a texture without a staging
    /// buffer is due (nothing was uploaded from it: a staging buffer that
    /// could not be made is not "clean"); one never detiled is due; one in
    /// view memory its guards find clean is not, and is once the CPU wrote
    /// it, until a check has detiled the write (asking takes nothing from
    /// the check); one in memory neither guarded nor write-watched only the
    /// check's hash tells; one a GPU write was submitted into has nothing to
    /// upload, once that write's list is submitted and not before.
    #[test]
    fn a_look_before_an_upload_tells_written_memory() {
        let test = "texture::tests::a_look_before_an_upload_tells_written_memory";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        if !guards() || verify_guards() || !predication() {
            return;
        }
        let _serial = crate::watch::tests::serial();
        const LEN: usize = 64 << 10;
        let (view, other) = section_views(LEN);
        let (resource, texture) = texture_in(view);
        assert_eq!(memory_change(resource), Change::Yes, "never staged");
        let staging = staging(resource, texture).unwrap();
        assert_eq!(memory_change(resource), Change::Yes, "never detiled");
        let size = staging.layout.size as usize;
        // SAFETY: bytes of the title's view, as its CPU writes them (a
        // fault into the guards' handler).
        let fill = |byte: u8| unsafe { (view as *mut u8).write_bytes(byte, size) };
        fill(0x11);
        sample_in(host, resource, 1);
        assert_eq!(memory_change(resource), Change::No, "clean");
        fill(0x22);
        assert_eq!(memory_change(resource), Change::Yes, "written");
        assert_eq!(memory_change(resource), Change::Yes, "still, until a check");
        sample_in(host, resource, 1);
        assert!(texels_of(host, texture).iter().all(|&b| b == 0x22));
        assert_eq!(memory_change(resource), Change::No, "detiled");

        // A committed texture: memory of its own, which neither covers.
        let plain = committed(&sampled(16, 16, 28));
        let started = Instant::now();
        while crate::resource::host_resource(plain).is_none() {
            assert!(started.elapsed().as_secs() < 5, "never resident");
            std::thread::sleep(Duration::from_millis(1));
        }
        let own = super::staging(plain, crate::resource::host_resource(plain).unwrap()).unwrap();
        sample_in(host, plain, 1);
        assert_eq!(memory_change(plain), Change::Unknown(own.layout.size));
        // One the host copies into keeps the host's contents, from the
        // copy's submission on.
        let list = host.new_list().unwrap();
        gpu_write(&list, plain, true);
        list.close();
        assert_eq!(memory_change(plain), Change::Unknown(own.layout.size));
        host.execute(&[&list]);
        host.drain();
        assert_eq!(memory_change(plain), Change::No);
        assert_eq!(staged_bytes(plain), None, "its staging buffer goes");

        type Release = extern "system" fn(This) -> u32;
        for r in [resource, plain] {
            assert_eq!(crate::object::slot::<Release>(r, 0x10)(r), 0);
        }
        crate::watch::title_memory_remapped(view, LEN);
        let _ = other;
    }

    /// What the look before an upload costs (`memory_change`), out of the
    /// game: 1,024 textures in view memory, as the title's texture pool is
    /// (guarded), each 64 KiB from the next, staged and detiled once, then
    /// looked at with none written (best of three rounds), and after the
    /// CPU wrote each. Run with
    /// `cargo test --release -p xtp-gpu -- --ignored --nocapture memory_change_costs`.
    #[test]
    #[ignore = "benchmark"]
    fn memory_change_costs() {
        let test = "texture::tests::memory_change_costs";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        if !guards() || verify_guards() {
            return;
        }
        let _serial = crate::watch::tests::serial();
        const N: usize = 1024;
        const STEP: usize = 64 << 10;
        let (view, other) = section_views(N * STEP);
        let textures: Vec<(This, &ID3D12Resource)> =
            (0..N).map(|i| texture_in(view + i * STEP)).collect();
        // One list stages and uploads them all.
        let list = host.new_list().unwrap();
        for &(resource, texture) in &textures {
            upload(&list, resource, texture);
        }
        list.close();
        host.execute(&[&list]);
        host.drain();
        let round = || -> (f64, usize) {
            let started = Instant::now();
            let changed = textures
                .iter()
                .filter(|(r, _)| memory_change(*r) != Change::No)
                .count();
            (started.elapsed().as_secs_f64() * 1e6 / N as f64, changed)
        };
        let clean = (0..3).map(|_| round()).fold((f64::INFINITY, 0), |best, r| {
            (best.0.min(r.0), best.1.max(r.1))
        });
        for i in 0..N {
            // SAFETY: the first byte of each texture's memory, in the view.
            unsafe { ((view + i * STEP) as *mut u8).write_volatile(0x5a) };
        }
        let written = round();
        eprintln!(
            "the look before an upload, {N} guarded textures: {:.2} us each clean ({} not clean), \
             {:.2} us each written ({} found)",
            clean.0, clean.1, written.0, written.1
        );
        type Release = extern "system" fn(This) -> u32;
        for (r, _) in textures {
            crate::object::slot::<Release>(r, 0x10)(r);
        }
        crate::watch::title_memory_remapped(view, N * STEP);
        let _ = other;
    }

    /// A copy location as a title passes one (D3D12's layout): a resource,
    /// the kind (0 a subresource index, 1 a placed footprint), then the
    /// index or the footprint.
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

    /// A 2×2 RGBA8 block of `texel`s, in a title buffer (rows 256 bytes apart).
    fn block_of(texel: u32) -> crate::testgpu::TitleBuffer {
        let buffer = crate::testgpu::TitleBuffer::new(1 << 12);
        buffer.set_words(|i| if i % 64 < 2 && i < 128 { texel } else { 0 });
        buffer
    }

    /// Records, in `list` as the title does, the copy of `block` (a 2×2
    /// block in a title buffer) to texel `(x, y)` of subresource 0 of `dst`.
    fn copy_block(
        list: &crate::testgpu::TitleList,
        dst: This,
        (x, y): (u32, u32),
        block: &crate::testgpu::TitleBuffer,
    ) {
        let to = Location {
            resource: dst,
            kind: 0,
            offset: 0,
            format: 0,
            width: 0,
            height: 0,
            depth: 0,
            row_pitch: 0,
        };
        let from = Location {
            resource: block.resource(),
            kind: 1,
            offset: 0,
            format: 28,
            width: 2,
            height: 2,
            depth: 1,
            row_pitch: 256,
        };
        type Copy = extern "system" fn(
            This,
            *const Location,
            u32,
            u32,
            u32,
            *const Location,
            *const [u32; 6],
        );
        let this = list.this();
        crate::object::slot::<Copy>(this, 0x80)(this, &to, x, y, 0, &from, std::ptr::null());
    }

    /// The 16 texels of a 4×4 RGBA8 host copy, row by row.
    fn texels_4x4(host: &crate::host::Host, texture: &ID3D12Resource) -> Vec<u32> {
        let read = read_back(host, texture, COPY_DEST);
        (0..16)
            .map(|i| {
                let at = (i / 4) * 256 + (i % 4) * 4;
                u32::from_le_bytes(read[at..at + 4].try_into().unwrap())
            })
            .collect()
    }

    /// A fresh 4×4 RGBA8 texture the title fills with `background`, once
    /// its host copy is resident.
    fn texture_4x4(background: u8) -> (This, &'static ID3D12Resource) {
        let resource = committed(&sampled(4, 4, 28));
        let started = Instant::now();
        while crate::resource::host_resource(resource).is_none() {
            assert!(started.elapsed().as_secs() < 5, "never resident");
            std::thread::sleep(Duration::from_millis(1));
        }
        let (memory, desc) = crate::resource::texture_memory(resource).unwrap();
        let size = crate::resource::texture_layout(&desc).unwrap().size as usize;
        // SAFETY: the texture's own title memory, `size` bytes.
        unsafe { memory.cast_mut().write_bytes(background, size) };
        (resource, crate::resource::host_resource(resource).unwrap())
    }

    #[test]
    fn consecutive_texture_refreshes_stay_between_host_callbacks() {
        let Some(gpu) =
            crate::testgpu::device("consecutive_texture_refreshes_stay_between_host_callbacks")
        else {
            return;
        };
        let (a, a_texture) = texture_4x4(0x31);
        let (b, b_texture) = texture_4x4(0x42);
        let a_stage = staging(a, a_texture).unwrap();
        let b_stage = staging(b, b_texture).unwrap();
        let list = gpu.host().new_list().unwrap();
        let counts = [a_stage.clone(), b_stage.clone()];
        list.defer_host((0, 0), move || {
            for stage in &counts {
                assert_eq!(stage.written.load(Ordering::Acquire), 0);
            }
        });
        list.defer_texture(a_stage.clone());
        list.defer_texture(b_stage.clone());
        list.defer_host((0, 0), move || {
            for (stage, byte) in [(&a_stage, 0x31), (&b_stage, 0x42)] {
                assert_eq!(stage.written.load(Ordering::Acquire), 1);
                let upload = stage.upload.lock().unwrap_or_else(|e| e.into_inner());
                let upload = upload
                    .as_ref()
                    .expect("refresh detiled to an upload buffer");
                // SAFETY: the guard keeps the mapping alive. The first
                // 16 bytes are the first RGBA8 row, populated by the refresh.
                let row = unsafe { std::slice::from_raw_parts(upload.data, 16) };
                assert_eq!(row, [byte; 16]);
            }
        });
        list.close();
        gpu.host().execute(&[&list]);
        gpu.host().drain();
        release(a);
        release(b);
    }

    /// What native D3D12 holds after copies of 2×2 blocks of the given
    /// texels at the given places, in submission order, over `background`.
    fn expected(background: u32, copies: &[((u32, u32), u32)]) -> Vec<u32> {
        let mut texels = vec![background; 16];
        for &((x, y), texel) in copies {
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                texels[((y + dy) * 4 + x + dx) as usize] = texel;
            }
        }
        texels
    }

    /// A 2D RGBA8 texture of `slices` array slices and `mips` mips, `width`
    /// square, that shaders sample (the layout library's pick).
    fn sampled_chain(width: u64, slices: u16, mips: u16) -> crate::resource::ResourceDesc {
        let desc = sampled(width, width as u32, 28);
        // SAFETY: a description is 56 bytes, 14 words; word 7 holds the
        // array size (low half) and the mip count (high half).
        let mut words: [u32; 14] = unsafe { std::mem::transmute_copy(&desc) };
        words[7] = u32::from(slices) | u32::from(mips) << 16;
        // SAFETY: as above, the same layout back.
        unsafe { std::mem::transmute_copy(&words) }
    }

    /// Data that arrives after a texture's upload in a late mip only (the
    /// smallest, in the mip tail) or in another array slice only is found
    /// by the look before an upload and detiled into that subresource of
    /// the staging buffer. The cases of victory-fork #559 (NickJBc), whose
    /// fingerprint of three windows missed them; here the guards cover
    /// every page of the texture, so no window is needed.
    #[test]
    fn late_mips_and_other_slices_are_found_changed() {
        let test = "texture::tests::late_mips_and_other_slices_are_found_changed";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        if !guards() || verify_guards() || !predication() {
            return;
        }
        let _serial = crate::watch::tests::serial();
        const LEN: usize = 1 << 20;
        // A 256-wide chain of 9 mips (1x1 last), then an array of 6 slices.
        for (case, desc, (mip, slice)) in [
            ("the last mip", sampled_chain(256, 1, 9), (8, 0)),
            ("slice 3 of 6", sampled_chain(64, 6, 1), (0, 3)),
        ] {
            let (view, other) = section_views(LEN);
            let resource = crate::resource::tests_support::at_address(view as u64, &desc);
            let started = Instant::now();
            while crate::resource::host_resource(resource).is_none() {
                assert!(started.elapsed().as_secs() < 5, "never resident");
                std::thread::sleep(Duration::from_millis(1));
            }
            let texture = crate::resource::host_resource(resource).unwrap();
            let staging = staging(resource, texture).unwrap();
            let size = staging.layout.size as usize;
            assert!(size <= LEN, "{case}: {size}");
            // SAFETY: bytes of the title's view, as its CPU writes them.
            unsafe { (view as *mut u8).write_bytes(0x11, size) };
            sample_in(host, resource, 1);
            assert_eq!(memory_change(resource), Change::No, "{case}: clean");
            // The element (0, 0) of that mip and slice only.
            let at = staging.layout.offset(mip, 0, 0, slice).unwrap() as usize;
            assert!(at + 4 <= size, "{case}: {at}");
            // SAFETY: inside the texture's memory, in the view.
            unsafe { ((view + at) as *mut u32).write_unaligned(0x7777_7777) };
            assert_eq!(memory_change(resource), Change::Yes, "{case}: written");
            let before = staging.written.load(Ordering::Acquire);
            sample_in(host, resource, 1);
            assert!(
                staging.written.load(Ordering::Acquire) > before,
                "{case}: detiled"
            );
            // D3D12's subresource order: mips of slice 0, then of slice 1...
            let sub = (slice * staging.mips + mip) as usize;
            let offset = staging.footprints[sub].Offset as usize;
            // SAFETY: the mapped staging buffer; nothing writes it now.
            let texel = unsafe {
                staging
                    .mapped()
                    .unwrap()
                    .add(offset)
                    .cast::<u32>()
                    .read_unaligned()
            };
            assert_eq!(texel, 0x7777_7777, "{case}: subresource {sub}");
            assert_eq!(memory_change(resource), Change::No, "{case}: clean again");
            type Release = extern "system" fn(This) -> u32;
            assert_eq!(crate::object::slot::<Release>(resource, 0x10)(resource), 0);
            crate::watch::title_memory_remapped(view, LEN);
            let _ = other;
        }
    }

    /// Texture writes follow submission order, not recording order
    /// (watch.md item 263; reimplemented after andrewpozdnakov7-jpg's test
    /// on VIctory #1891). A 4×4 RGBA8 texture the title fills with a
    /// background; two lists copy overlapping 2×2 blocks, A at (0, 0) and
    /// B at (1, 1), recorded A, B or B, A and submitted A, B or B, A:
    /// native D3D12 keeps the background where neither copied and the
    /// block submitted last where both did, whatever the recording order.
    /// Then a list that samples the texture (its upload) and one that
    /// copies A into it, in the four orders: sampling writes nothing, so
    /// the texture ends as the background with A over it. Each case on a
    /// texture of its own; the cases that differ from native are listed
    /// together.
    ///
    /// Before `gpu_write` (on WARP, 2026-11-25), 4 of 4 two-copy cases lost
    /// the background (a partial copy did not first bring the title bytes)
    /// and 3 of 4 sample-and-copy cases differed (the copy counted from its
    /// recording: a sampling list recorded after it never uploaded the
    /// background, one recorded before it but submitted after overwrote
    /// the copy). The decisions at submission need predicates.
    #[test]
    fn texture_writes_follow_submission_order() {
        use crate::testgpu::TitleList;
        let test = "texture::tests::texture_writes_follow_submission_order";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        if !predication() {
            return;
        }
        let (background, a, b) = (0x1111_1111u32, 0xaaaa_aaaa, 0xbbbb_bbbb);
        let (block_a, block_b) = (block_of(a), block_of(b));
        let mut wrong = Vec::new();
        let release = |r: This| {
            type Release = extern "system" fn(This) -> u32;
            crate::object::slot::<Release>(r, 0x10)(r);
        };
        // Two copies.
        for (recorded_a_first, submitted_a_first) in
            [(true, true), (false, true), (true, false), (false, false)]
        {
            let (resource, texture) = texture_4x4(0x11);
            let (la, lb) = (TitleList::new(), TitleList::new());
            let mut record = [(&la, (0, 0), &block_a), (&lb, (1, 1), &block_b)];
            if !recorded_a_first {
                record.reverse();
            }
            for (list, at, block) in record {
                copy_block(list, resource, at, block);
            }
            let (first, second) = if submitted_a_first {
                (&la, &lb)
            } else {
                (&lb, &la)
            };
            first.run();
            second.run();
            let submitted = if submitted_a_first {
                [((0, 0), a), ((1, 1), b)]
            } else {
                [((1, 1), b), ((0, 0), a)]
            };
            let got = texels_4x4(host, texture);
            let want = expected(background, &submitted);
            if got != want {
                wrong.push(format!(
                    "copies recorded {} submitted {}: {} of 16 as native, got {got:08x?}",
                    if recorded_a_first { "A, B" } else { "B, A" },
                    if submitted_a_first { "A, B" } else { "B, A" },
                    got.iter().zip(&want).filter(|(g, w)| g == w).count()
                ));
            }
            release(resource);
        }
        // A sampling list and a copying one.
        for (recorded_s_first, submitted_s_first) in
            [(true, true), (false, true), (true, false), (false, false)]
        {
            let (resource, texture) = texture_4x4(0x11);
            let (ls, lc) = (TitleList::new(), TitleList::new());
            let sample = || upload(ls.host(), resource, texture);
            if recorded_s_first {
                sample();
                copy_block(&lc, resource, (0, 0), &block_a);
            } else {
                copy_block(&lc, resource, (0, 0), &block_a);
                sample();
            }
            let (first, second) = if submitted_s_first {
                (&ls, &lc)
            } else {
                (&lc, &ls)
            };
            first.run();
            second.run();
            let got = texels_4x4(host, texture);
            let want = expected(background, &[((0, 0), a)]);
            if got != want {
                wrong.push(format!(
                    "sample and copy recorded {} submitted {}: {} of 16 as native, got {got:08x?}",
                    if recorded_s_first { "S, C" } else { "C, S" },
                    if submitted_s_first { "S, C" } else { "C, S" },
                    got.iter().zip(&want).filter(|(g, w)| g == w).count()
                ));
            }
            release(resource);
        }
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }

    /// A pass's first upload makes a texture's staging buffer only within
    /// the room it has: past it none is made, the texture is not taken for
    /// uploaded (its look still finds it due), and it is made once there is
    /// room; one that has its staging buffer needs no room (a full staging
    /// memory leaves it served); one a GPU write was submitted into needs
    /// none at all.
    #[test]
    fn staging_buffers_are_made_within_their_room() {
        let test = "texture::tests::staging_buffers_are_made_within_their_room";
        let Some(host) = crate::testgpu::device(test) else {
            return;
        };
        let (resource, texture) = texture_4x4(0x11);
        let _alone = SweptAlone::new(resource);
        let Staged::NoRoom(bytes) = stage(resource, texture, 16) else {
            panic!("{:?}", stage(resource, texture, 16));
        };
        assert!(bytes > 16, "{bytes}");
        assert_eq!(staged_bytes(resource), None, "none made");
        assert_eq!(memory_change(resource), Change::Yes, "still due");
        assert_eq!(stage(resource, texture, bytes), Staged::Ready(bytes));
        assert_eq!(staged_bytes(resource), Some(bytes));
        assert_eq!(stage(resource, texture, 0), Staged::Ready(0), "had one");
        let list = host.new_list().unwrap();
        gpu_write(&list, resource, true);
        list.close();
        host.execute(&[&list]);
        host.drain();
        assert_eq!(stage(resource, texture, u64::MAX), Staged::Host);
        assert_eq!(staged_bytes(resource), None);
        type Release = extern "system" fn(This) -> u32;
        assert_eq!(crate::object::slot::<Release>(resource, 0x10)(resource), 0);
    }

    /// A staging buffer goes once no list holds its texture's state (the
    /// lists that copied from it ran and were reset: the fence passed) and
    /// no upload of it was recorded for `idle` sweeps, not before; its
    /// bytes are counted released. Its next upload, its memory only a hash
    /// tells, is left to the check at submission: unchanged memory makes
    /// nothing again, and is neither detiled nor copied (the host copy has
    /// its bytes); a change makes it again there, detiled and copied as
    /// before. A host copy not known to hold the bytes gets the buffer made
    /// again when the upload is recorded, and the bytes detiled and copied.
    /// Before, the upload made it again whatever the memory held (Codex's
    /// review of db37732, 1).
    #[test]
    fn staging_buffers_go_once_their_last_copy_ran() {
        let test = "texture::tests::staging_buffers_go_once_their_last_copy_ran";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        if !predication() {
            return;
        }
        const IDLE: u64 = 1000;
        let (resource, texture) = texture_4x4(0x11);
        let _alone = SweptAlone::new(resource);
        let key = resource as usize;
        // Other tests' textures are in the same map: only this one goes. A
        // sweep `later` passes on from now (without moving the passes'
        // clock, which other tests' passes read).
        let sweep_at = |later: u64| {
            let now = SWEEPS.load(Ordering::Relaxed) + later;
            release_staging_of(now, IDLE, usize::MAX, |r| r == key)
        };
        // The next pass's sweep; one past the idle time: the buffers and
        // bytes released.
        let sweep = || sweep_at(1);
        let sweep_out = || sweep_at(IDLE + 1);
        let state = || {
            let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
            map.as_ref().unwrap().get(&key).unwrap().clone()
        };
        let id = staging(resource, texture).unwrap().id;
        let bytes = staged_bytes(resource).expect("a staging buffer");
        let (memory, _) = crate::resource::texture_memory(resource).unwrap();
        let size = state().layout.size as usize;

        // Recorded and not run: the list holds it, however long it waits.
        let list = host.new_list().unwrap();
        upload(&list, resource, texture);
        list.close();
        assert_eq!(sweep_out(), (0, 0), "held by a list");
        assert_eq!(staged_bytes(resource), Some(bytes));
        host.execute(&[&list]);
        host.drain();
        drop(list);
        assert_eq!(decided_id(id), [true]);
        // Held by nothing once a list that sampled it lately ran and went,
        // its last upload recorded within `IDLE` sweeps: kept; then
        // released.
        sample_in(host, resource, 1);
        assert_eq!(decided_id(id), [true, false]);
        assert_eq!(sweep(), (0, 0), "recorded lately");
        assert_eq!(sweep_out(), (1, bytes), "released");
        assert_eq!(staged_bytes(resource), None);

        // Unchanged: nothing made again, neither detiled nor copied.
        let written = state().written.load(Ordering::Acquire);
        sample_in(host, resource, 1);
        assert_eq!(staged_bytes(resource), None, "not made again");
        assert_eq!(decided_id(id), [true, false]);
        assert_eq!(state().written.load(Ordering::Acquire), written);
        assert!(texels_4x4(host, texture).iter().all(|&t| t == 0x1111_1111));

        // Changed: made again at submission, detiled and copied.
        // SAFETY: the texture's own title memory, `size` bytes.
        unsafe { memory.cast_mut().write_bytes(0x33, size) };
        sample_in(host, resource, 1);
        assert_eq!(staged_bytes(resource), Some(bytes), "made again");
        assert_eq!(decided_id(id), [true, false, true]);
        assert!(texels_4x4(host, texture).iter().all(|&t| t == 0x3333_3333));

        // Released again; a host copy not known to hold the last detiling
        // (no copy decided into it) gets it detiled and copied.
        assert_eq!(sweep_out(), (1, bytes), "released again");
        *state().copied.lock().unwrap() = None;
        sample_in(host, resource, 1);
        assert_eq!(decided_id(id), [true, false, true, true]);
        assert!(!state().blank.load(Ordering::Acquire));
        assert!(texels_4x4(host, texture).iter().all(|&t| t == 0x3333_3333));
        type Release = extern "system" fn(This) -> u32;
        assert_eq!(crate::object::slot::<Release>(resource, 0x10)(resource), 0);
    }

    thread_local! {
        /// Uploads recorded on this thread take no predicate
        /// ([`WithoutPredicates`]).
        pub(super) static WITHOUT_PREDICATES: std::cell::Cell<bool> =
            const { std::cell::Cell::new(false) };
    }

    /// While it lives, uploads recorded on this thread take no predicate,
    /// as with `XTP_TEXTURE_PREDICATION=0` or once they are all taken.
    struct WithoutPredicates;

    impl WithoutPredicates {
        fn new() -> WithoutPredicates {
            WITHOUT_PREDICATES.set(true);
            WithoutPredicates
        }
    }

    impl Drop for WithoutPredicates {
        fn drop(&mut self) {
            WITHOUT_PREDICATES.set(false);
        }
    }

    /// While it lives, only the test that made it releases the staging
    /// buffer of its resource: the passes of other tests leave it as it is
    /// (`SWEPT_ALONE`).
    struct SweptAlone(usize);

    impl SweptAlone {
        fn new(resource: This) -> SweptAlone {
            let key = resource as usize;
            SWEPT_ALONE
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(key);
            SweptAlone(key)
        }
    }

    impl Drop for SweptAlone {
        fn drop(&mut self) {
            let mut alone = SWEPT_ALONE.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(at) = alone.iter().position(|&r| r == self.0) {
                alone.swap_remove(at);
            }
        }
    }

    /// `len` bytes of committed memory of our own: write-watched as the
    /// runtime reserves the title's if `watched`, plain otherwise (neither
    /// write-watched nor guarded: only a hash tells a change, and a check
    /// needs all of it committed).
    fn private_memory(len: usize, watched: bool) -> usize {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
        };
        const MEM_WRITE_WATCH: u32 = 0x0020_0000;
        let kind = MEM_RESERVE | MEM_COMMIT | if watched { MEM_WRITE_WATCH } else { 0 };
        // SAFETY: a fresh allocation of our own.
        let at = unsafe { VirtualAlloc(std::ptr::null(), len, kind, PAGE_READWRITE) } as usize;
        assert_ne!(at, 0, "memory of our own");
        at
    }

    /// Gives back memory [`private_memory`] made, once nothing is placed in
    /// it any more.
    fn free_memory(at: usize, len: usize) {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
        crate::resource::memory_remapped(at, len);
        // SAFETY: an allocation of ours; nothing is placed in it now.
        unsafe { VirtualFree(at as *mut _, 0, MEM_RELEASE) };
    }

    /// Releases `resource`'s staging buffer, unless a sweep of another
    /// test's did: it has none after.
    fn released(resource: This) {
        release_staging_now(resource);
        assert_eq!(staged_bytes(resource), None, "released");
    }

    /// Without a predicate (`XTP_TEXTURE_PREDICATION=0`, or none left), a
    /// copy recorded runs at submission whatever the staging buffer holds
    /// then: it is recorded only from one that holds a detiling. A buffer
    /// made again after its release is filled when the copy is recorded;
    /// if the memory is not mapped then (decommitted), no copy is recorded
    /// and the host copy keeps its texels. A GPU write into the texture
    /// submitted between the recording and the list's submission (whose
    /// check then detiles nothing) finds it filled: the copy brings the
    /// title's bytes, as without predicates before buffers were released.
    /// Before, both copies read the blank buffer made again (Codex's review
    /// of c44a716, 1). Each upload after a release here is into a host
    /// copy not known to hold the bytes: one that holds them leaves the
    /// read to the check at submission, which records no copy in the list
    /// (`a_write_after_a_list_closed_is_copied_before_it_runs`).
    #[test]
    fn copies_without_a_predicate_never_read_a_blank_buffer() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_DECOMMIT, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        let test = "texture::tests::copies_without_a_predicate_never_read_a_blank_buffer";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let _none = WithoutPredicates::new();
        const LEN: usize = 64 << 10;
        let at = private_memory(LEN, false);
        let (resource, texture) = texture_in(at);
        let _alone = SweptAlone::new(resource);
        let (id, size) = {
            let state = staging(resource, texture).unwrap();
            (state.id, state.layout.size as usize)
        };
        assert!(size <= LEN);
        // SAFETY: the texture's memory, ours, `size` bytes of it.
        let fill = |byte: u8| unsafe { (at as *mut u8).write_bytes(byte, size) };
        let all = |byte: u8| texels_of(host, texture).iter().all(|&b| b == byte);
        // Its host copy not known to hold what was detiled last.
        let unknown = || *state_of(resource).unwrap().copied.lock().unwrap() = None;

        // The first upload: filled when recorded, then copied.
        fill(0x11);
        sample_in(host, resource, 1);
        assert_eq!(decided_id(id), [true]);
        assert!(all(0x11));
        released(resource);

        // Its memory decommitted: the buffer made again cannot be filled,
        // and no copy is recorded.
        // SAFETY: pages of our allocation; nothing reads them meanwhile.
        assert_ne!(unsafe { VirtualFree(at as *mut _, LEN, MEM_DECOMMIT) }, 0);
        unknown();
        let list = host.new_list().unwrap();
        assert!(
            !upload(&list, resource, texture),
            "a copy from a blank buffer"
        );
        list.close();
        host.execute(&[&list]);
        host.drain();
        drop(list);
        assert_eq!(decided_id(id), [true]);
        assert!(all(0x11), "{:x?}", &texels_of(host, texture)[..16]);

        // Mapped again with other bytes: filled when recorded, then copied.
        // SAFETY: the pages decommitted above, committed again.
        let again = unsafe { VirtualAlloc(at as *const _, LEN, MEM_COMMIT, PAGE_READWRITE) };
        assert!(!again.is_null());
        fill(0x22);
        sample_in(host, resource, 1);
        assert_eq!(decided_id(id), [true, true]);
        assert!(all(0x22));

        // Released again, then written, and recorded before a GPU write
        // into the texture is submitted: the copy brings what was filled.
        released(resource);
        fill(0x33);
        unknown();
        let reader = host.new_list().unwrap();
        assert!(upload(&reader, resource, texture));
        reader.close();
        let writer = host.new_list().unwrap();
        gpu_write(&writer, resource, true);
        writer.close();
        host.execute(&[&writer]);
        host.execute(&[&reader]);
        host.drain();
        assert_eq!(decided_id(id), [true, true, true]);
        assert!(all(0x33), "{:x?}", &texels_of(host, texture)[..16]);
        drop((reader, writer));

        type Release = extern "system" fn(This) -> u32;
        assert_eq!(crate::object::slot::<Release>(resource, 0x10)(resource), 0);
        free_memory(at, LEN);
    }

    /// A texture in write-watched memory whose staging buffer was released:
    /// uploads while its host copy has what was detiled last and the memory
    /// is not written make nothing again and record no copy; once the
    /// memory is written, the next makes the buffer again, and the memory
    /// is read into the room the checks share, detiled and copied. Before,
    /// an upload made the buffer again whatever the memory, and each
    /// detiling of such a texture kept a copy of its memory of its own as
    /// long as it lived (Codex's review of c44a716, 3).
    #[test]
    fn write_watched_textures_are_made_again_only_once_written() {
        let test = "texture::tests::write_watched_textures_are_made_again_only_once_written";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        const LEN: usize = 64 << 10;
        let at = private_memory(LEN, true);
        let (resource, texture) = texture_in(at);
        let _alone = SweptAlone::new(resource);
        let (id, size) = {
            let state = staging(resource, texture).unwrap();
            (state.id, state.layout.size as usize)
        };
        // SAFETY: the texture's memory, ours, `size` bytes of it.
        let fill = |byte: u8| unsafe { (at as *mut u8).write_bytes(byte, size) };
        let all = |byte: u8| texels_of(host, texture).iter().all(|&b| b == byte);

        fill(0x11);
        sample_in(host, resource, 1);
        assert!(all(0x11));
        assert_eq!(memory_change(resource), Change::No, "write-watched, clean");
        released(resource);
        let decided = decided_id(id).len();
        sample_in(host, resource, 2);
        assert_eq!(staged_bytes(resource), None, "not made again");
        assert_eq!(decided_id(id).len(), decided, "no copy recorded");

        fill(0x44);
        assert_eq!(memory_change(resource), Change::Yes);
        sample_in(host, resource, 1);
        assert!(staged_bytes(resource).is_some(), "made again");
        assert_eq!(decided_id(id).last(), Some(&true));
        assert!(all(0x44), "{:x?}", &texels_of(host, texture)[..16]);
        assert_eq!(memory_change(resource), Change::No, "detiled");

        type Release = extern "system" fn(This) -> u32;
        assert_eq!(crate::object::slot::<Release>(resource, 0x10)(resource), 0);
        free_memory(at, LEN);
    }

    /// A texture whose staging buffer was released, its host copy holding
    /// what was detiled last and its memory not written since: a list that
    /// reads it (twice) records nothing for it; the title writes other bytes
    /// once the list is closed, then submits it: the check at submission
    /// finds the write, makes the buffer again, and copies the bytes before
    /// the list runs, from the state the lists before left the texture in
    /// and back to it; not over a GPU write into the texture submitted
    /// before the list. Not written, nothing is made again. In write-watched
    /// memory and in memory only a hash tells, with a predicate and
    /// without. Before, nothing was checked at submission, and the list's
    /// shaders read the old texels (Codex's review of db37732, 1).
    #[test]
    fn a_write_after_a_list_closed_is_copied_before_it_runs() {
        let test = "texture::tests::a_write_after_a_list_closed_is_copied_before_it_runs";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        const LEN: usize = 64 << 10;
        // Both shader-read states, as a draw's table leaves a texture.
        const READ: u32 = 0x40 | 0x80;
        for (watched, predicates) in [(true, true), (false, true), (true, false)] {
            let _none = (!predicates).then(WithoutPredicates::new);
            let at = private_memory(LEN, watched);
            let (resource, texture) = texture_in(at);
            let _alone = SweptAlone::new(resource);
            let (id, size) = {
                let state = staging(resource, texture).unwrap();
                (state.id, state.layout.size as usize)
            };
            // SAFETY: the texture's memory, ours, `size` bytes of it.
            let fill = |byte: u8| unsafe { (at as *mut u8).write_bytes(byte, size) };
            // Its first rows, read from the state the lists left it in.
            let texels = || {
                let copy = crate::resource::host_copy(resource).unwrap();
                let read = read_back(host, &copy.texture, copy.state.load(Ordering::Relaxed));
                (0..16)
                    .flat_map(|y| read[y * 256..y * 256 + 64].to_vec())
                    .collect::<Vec<u8>>()
            };
            let case = format!("write-watched {watched}, predicates {predicates}");

            fill(0x11);
            sample_in(host, resource, 1);
            // A draw leaves it in the shader-read states.
            let draw = host.new_list().unwrap();
            crate::list::ensure_state(&draw, resource, READ);
            draw.close();
            host.execute(&[&draw]);
            host.drain();
            drop(draw);
            assert!(texels().iter().all(|&b| b == 0x11), "{case}");
            released(resource);

            // Not written: nothing made again, nor copied.
            let decided = decided_id(id).len();
            sample_in(host, resource, 1);
            assert_eq!(staged_bytes(resource), None, "{case}: not made again");
            assert_eq!(decided_id(id).len(), decided, "{case}: nothing copied");

            // Written once the list closed: copied before it runs.
            let list = host.new_list().unwrap();
            assert!(
                !upload(&list, resource, texture),
                "{case}: nothing recorded"
            );
            assert!(!upload(&list, resource, texture), "{case}: read again");
            crate::list::ensure_state(&list, resource, READ);
            list.close();
            fill(0x22);
            host.execute(&[&list]);
            host.drain();
            drop(list);
            assert!(
                texels().iter().all(|&b| b == 0x22),
                "{case}: {:x?}",
                &texels()[..16]
            );
            assert_eq!(decided_id(id)[decided..], [true], "{case}: one copy");
            assert!(staged_bytes(resource).is_some(), "{case}: made again");
            assert_eq!(
                crate::resource::host_copy(resource)
                    .unwrap()
                    .state
                    .load(Ordering::Relaxed),
                READ,
                "{case}: left as the list leaves it"
            );

            // Released again, read, written once the list closed, but a GPU
            // write into the texture is submitted before the list: the
            // host's contents stay, nothing is copied over them.
            released(resource);
            let decided = decided_id(id).len();
            let reader = host.new_list().unwrap();
            assert!(!upload(&reader, resource, texture), "{case}");
            reader.close();
            fill(0x33);
            let writer = host.new_list().unwrap();
            gpu_write(&writer, resource, true);
            writer.close();
            host.execute(&[&writer]);
            host.execute(&[&reader]);
            host.drain();
            drop((reader, writer));
            assert_eq!(decided_id(id).len(), decided, "{case}: copied over");
            assert!(
                texels().iter().all(|&b| b == 0x22),
                "{case}: {:x?}",
                &texels()[..16]
            );
            release(resource);
            free_memory(at, LEN);
        }
        assert_eq!(host.log_messages(), None);
    }

    /// A texture in write-watched memory whose host copy holds its bytes,
    /// its staging buffer released, then its memory decommitted (the write
    /// watch still answers for the reservation, and finds no page written):
    /// a read left to the check at submission; a buffer made again by a
    /// pass's first upload (of another slot) and filled for a copy without
    /// a predicate; a copy forced at submission (its host copy not known to
    /// hold the bytes); part of the memory committed again and written:
    /// none takes anything from memory that is not mapped, and the host
    /// copy keeps its texels. All of it committed again and written, it is
    /// copied. Before, the second read the memory as zeros, detiled and
    /// copied them over the texels (Codex's review of db37732, 2).
    #[test]
    fn decommitted_write_watched_memory_never_reaches_the_host_copy() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_DECOMMIT, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        let test = "texture::tests::decommitted_write_watched_memory_never_reaches_the_host_copy";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        const LEN: usize = 64 << 10;
        const PAGE: usize = 4 << 10;
        let at = private_memory(LEN, true);
        // 16 KiB of texels: more than its first page.
        let (resource, texture) = square_texture_in(at, 64);
        let _alone = SweptAlone::new(resource);
        let (id, size) = {
            let state = staging(resource, texture).unwrap();
            (state.id, state.layout.size as usize)
        };
        assert!(size > PAGE && size <= LEN, "{size}");
        // SAFETY: the texture's memory, ours, `size` bytes of it, committed.
        let fill = |byte: u8, len: usize| unsafe { (at as *mut u8).write_bytes(byte, len) };
        let all = |byte: u8| texels_of(host, texture).iter().all(|&b| b == byte);
        let commit = |len: usize| {
            // SAFETY: pages of our reservation, committed again.
            let again = unsafe { VirtualAlloc(at as *const _, len, MEM_COMMIT, PAGE_READWRITE) };
            assert!(!again.is_null());
        };

        fill(0x11, size);
        sample_in(host, resource, 1);
        assert!(all(0x11));
        released(resource);
        // SAFETY: pages of our allocation; nothing reads them meanwhile.
        assert_ne!(unsafe { VirtualFree(at as *mut _, LEN, MEM_DECOMMIT) }, 0);
        let decided = decided_id(id).len();

        // Read by a list: left to the check at submission, which finds
        // nothing written.
        sample_in(host, resource, 1);
        assert!(all(0x11), "{:x?}", &texels_of(host, texture)[..16]);
        assert_eq!(staged_bytes(resource), None);

        // A pass's first upload of another slot of it: its buffer made
        // again within the room, then a copy without a predicate.
        let Staged::Ready(made) = stage(resource, texture, u64::MAX) else {
            panic!("{:?}", stage(resource, texture, u64::MAX));
        };
        assert!(made > 0, "made again");
        {
            let _none = WithoutPredicates::new();
            let list = host.new_list().unwrap();
            assert!(
                !upload(&list, resource, texture),
                "a copy from memory not mapped"
            );
            list.close();
            host.execute(&[&list]);
            host.drain();
        }
        assert!(all(0x11), "{:x?}", &texels_of(host, texture)[..16]);

        // A copy forced at submission, under a predicate: it does not run.
        *state_of(resource).unwrap().copied.lock().unwrap() = None;
        sample_in(host, resource, 1);
        assert!(all(0x11), "{:x?}", &texels_of(host, texture)[..16]);
        assert_eq!(decided_id(id)[decided..], [false]);

        // Its first page committed again and written, the rest not: a host
        // copy that is not reserved waits for all of it.
        commit(PAGE);
        fill(0x22, PAGE);
        assert_eq!(memory_change(resource), Change::Yes);
        sample_in(host, resource, 1);
        assert!(all(0x11), "{:x?}", &texels_of(host, texture)[..16]);
        assert_eq!(decided_id(id)[decided..], [false, false]);

        // All of it committed again and written: copied.
        commit(LEN);
        fill(0x33, size);
        sample_in(host, resource, 1);
        assert!(all(0x33), "{:x?}", &texels_of(host, texture)[..16]);
        assert_eq!(decided_id(id)[decided..], [false, false, true]);

        release(resource);
        free_memory(at, LEN);
        assert_eq!(host.log_messages(), None);
    }

    /// A pass's checks make the released staging buffers of served textures
    /// again from a share of the pass's budget of their own: the first
    /// whatever its bytes, the next ones within what is left of the share;
    /// one whose buffer lives costs nothing, one never staged is not theirs
    /// (a first upload), one a GPU write was submitted into needs none. The
    /// share may pass the staging limit by itself, not more. One past the
    /// share waits for the next pass's; one no pass could make now is left
    /// for its next turn, the textures after it going on.
    #[test]
    fn released_served_textures_are_made_again_from_a_share_of_their_own() {
        let test =
            "texture::tests::released_served_textures_are_made_again_from_a_share_of_their_own";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let [(a, ta), (b, tb), (never, _)] = [0x11, 0x22, 0x33].map(texture_4x4);
        let _alone = [a, b, never].map(SweptAlone::new);
        let bytes = staging(a, ta).unwrap().size;
        assert_eq!(staging(b, tb).unwrap().size, bytes);
        released(a);
        released(b);

        // A pass whose budget leaves one of them to the share.
        let mut refreshes = Refreshes::of(REFRESH_SHARE * bytes);
        assert_eq!(refreshes.stage(never), None, "a first upload");
        assert_eq!(refreshes.stage(a), Some(Staged::Ready(bytes)));
        assert_eq!(refreshes.stage(a), Some(Staged::Ready(0)), "it has one");
        assert_eq!(refreshes.stage(b), Some(Staged::NoRoom(bytes)), "past it");
        assert_eq!(refreshes.made(), bytes);
        // The first of a pass whatever its bytes.
        assert_eq!(Refreshes::of(0).stage(b), Some(Staged::Ready(bytes)));
        // One the host GPU writes.
        let list = host.new_list().unwrap();
        gpu_write(&list, a, true);
        list.close();
        host.execute(&[&list]);
        host.drain();
        assert_eq!(Refreshes::of(0).stage(a), Some(Staged::Host));

        // The room: the share, the first one past it, the staging limit
        // passed by the share at most.
        const MIB: u64 = 1 << 20;
        assert_eq!(refresh_room(16 * MIB, 0, 1024 * MIB, 0), 1040 * MIB);
        assert_eq!(refresh_room(16 * MIB, 4 * MIB, 1024 * MIB, 0), 12 * MIB);
        assert_eq!(refresh_room(16 * MIB, 20 * MIB, 1024 * MIB, 0), 0);
        assert_eq!(refresh_room(16 * MIB, 0, 1024 * MIB, 1030 * MIB), 10 * MIB);
        assert_eq!(refresh_room(16 * MIB, 0, 1024 * MIB, 2048 * MIB), 0);
        // Past it: the share spent, the first of the next pass; past what a
        // pass of its own could make, its next turn (the pass goes on).
        assert_eq!(
            refused(10 * MIB, 16 * MIB, 8 * MIB, 1024 * MIB, 0),
            Staged::NoRoom(10 * MIB)
        );
        assert_eq!(refused(96 * MIB, 16 * MIB, 0, 64 * MIB, 0), Staged::NotNow);
        assert_eq!(
            refused(96 * MIB, 16 * MIB, 8 * MIB, 64 * MIB, 0),
            Staged::NotNow
        );
        assert_eq!(
            refused(40 * MIB, 16 * MIB, 0, 1024 * MIB, 1000 * MIB),
            Staged::NotNow,
            "the staging buffers that live leave it no room"
        );

        type Release = extern "system" fn(This) -> u32;
        for r in [a, b, never] {
            assert_eq!(crate::object::slot::<Release>(r, 0x10)(r), 0);
        }
    }

    /// A title texture `desc` (at `address`, or committed) whose host copy
    /// is a reserved texture of the tests' pool, once it is ready (the
    /// pool's heaps resident), with its tiles; `None` (the resource
    /// released) on a device without reserved textures.
    fn reserved(
        desc: &crate::resource::ResourceDesc,
        address: Option<usize>,
    ) -> Option<(
        This,
        &'static ID3D12Resource,
        Arc<crate::resource::pool::Tiles>,
    )> {
        use crate::resource::pool::tests::{Pooled, pool_for_tests};
        let pool = pool_for_tests()?;
        assert!(pool.settle(Duration::from_secs(5)));
        assert!(pool.resident(Duration::from_secs(5)));
        let resource = {
            let _pooled = Pooled::new();
            match address {
                Some(at) => crate::resource::tests_support::at_address(at as u64, desc),
                None => committed(desc),
            }
        };
        let started = Instant::now();
        while crate::resource::host_resource(resource).is_none() {
            assert!(started.elapsed().as_secs() < 5, "never ready");
            std::thread::sleep(Duration::from_millis(1));
        }
        let texture = crate::resource::host_resource(resource).unwrap();
        match crate::resource::pool::tiles_of(texture) {
            Some(tiles) => Some((resource, texture, tiles)),
            None => {
                release(resource);
                None
            }
        }
    }

    fn release(resource: This) {
        type Release = extern "system" fn(This) -> u32;
        assert_eq!(crate::object::slot::<Release>(resource, 0x10)(resource), 0);
    }

    /// Whether mip `sub` of 256x256 RGBA8 `texture` (as the chains of these
    /// tests) holds `byte` in every texel.
    fn mip_is(host: &crate::host::Host, texture: &ID3D12Resource, sub: u32, byte: u8) -> bool {
        use crate::resource::pool::tests::{read_sub, rows_are};
        rows_are(&read_sub(host, texture, sub), (256 >> sub) * 4, byte)
    }

    /// Records in `list` a GPU copy of `byte`s over mip 0 of 256x256 RGBA8
    /// `resource`, as a title's copy into it is recorded (`list.rs`): its
    /// write first, then the copy.
    fn copy_over(host: &crate::host::Host, list: &HostList, resource: This, byte: u8) {
        let texture = crate::resource::host_resource(resource).unwrap();
        const PITCH: u32 = 256 * 4;
        let (buffer, data) = list.upload(host, u64::from(PITCH) * 256).unwrap();
        // SAFETY: the mapped upload buffer the list holds, its size.
        unsafe { data.write_bytes(byte, PITCH as usize * 256) };
        crate::list::ensure_state(list, resource, COPY_DEST);
        gpu_write(list, resource, true);
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM,
                        Width: 256,
                        Height: 256,
                        Depth: 1,
                        RowPitch: PITCH,
                    },
                },
            },
        };
        // SAFETY: an open list; both locations name live resources the
        // list holds.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }

    /// Evicts texture `key` if it is idle `passes` passes from now and
    /// `looks` looks after a pass last looked at it (later than other
    /// tests' passes see; their looks move the clock meanwhile).
    fn evict_later(key: usize, idle: u64, passes: u64, looks: u64) -> (u64, u64) {
        let now = SWEEPS.load(Ordering::Relaxed) + passes;
        let asked = state_of(key as This).map_or(0, |s| s.asked_at.load(Ordering::Relaxed));
        evict_idle_at(now, asked.saturating_add(looks), idle, usize::MAX, |r| {
            r == key
        })
    }

    /// A reserved host copy no list holds, without an upload recorded nor
    /// a pass's look for `idle` passes and two laps of looks, gives its
    /// tiles back: the texture stays and reads zero, and a pass's look
    /// finds it due; its next upload maps its tiles again and copies its
    /// texels back from title memory, unchanged. One a list holds, one
    /// uploaded lately or one a pass looked at within the lap keeps them.
    #[test]
    fn idle_reserved_textures_give_their_tiles_back_and_come_back_on_demand() {
        let test =
            "texture::tests::idle_reserved_textures_give_their_tiles_back_and_come_back_on_demand";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let Some((resource, texture, tiles)) = reserved(&sampled_chain(256, 1, 9), None) else {
            eprintln!("{test}: skipped, the device reserves no textures (tiled resources tier 2)");
            return;
        };
        let _alone = SweptAlone::new(resource);
        let key = resource as usize;
        let (memory, _) = crate::resource::texture_memory(resource).unwrap();
        let size = staging(resource, texture).unwrap().layout.size as usize;
        // SAFETY: the texture's own title memory, `size` bytes.
        unsafe { memory.cast_mut().write_bytes(0x11, size) };
        sample_in(host, resource, 1);
        assert!(mip_is(host, texture, 0, 0x11));
        let mapped = tiles.mapped();
        let all: u64 = tiles
            .units()
            .iter()
            .map(|u| u64::from(u.tiles) * crate::resource::pool::TILE)
            .sum();
        assert_eq!(mapped, all, "all its memory is committed");
        const IDLE: u64 = 1000;
        const LAPS: u64 = u64::MAX / 4;

        // Held by a list (an upload recorded, not run): kept.
        let list = host.new_list().unwrap();
        assert!(upload(&list, resource, texture));
        list.close();
        assert_eq!(evict_later(key, IDLE, IDLE + 1, LAPS), (0, 0), "held");
        host.execute(&[&list]);
        host.drain();
        drop(list);
        // Uploaded lately: kept.
        assert_eq!(evict_later(key, IDLE, 1, LAPS), (0, 0), "uploaded lately");
        // Looked at by a pass within the lap: kept.
        let _ = memory_change(resource);
        assert_eq!(evict_later(key, IDLE, IDLE + 1, 1), (0, 0), "looked at");
        // Idle: its tiles go, the texture stays and reads zero.
        assert_eq!(evict_later(key, IDLE, IDLE + 1, LAPS), (1, mapped));
        assert_eq!(tiles.mapped(), 0);
        assert!(mip_is(host, texture, 0, 0), "unmapped tiles read zero");
        assert_eq!(memory_change(resource), Change::Yes, "due again");
        assert_eq!(evict_later(key, IDLE, IDLE + 1, LAPS), (0, 0), "none left");

        // Made again on demand: mapped, and its texels copied back.
        let again = evicted_total()[2];
        sample_in(host, resource, 1);
        assert_eq!(tiles.mapped(), mapped);
        assert!(mip_is(host, texture, 0, 0x11), "copied back");
        assert!(evicted_total()[2] > again);
        assert!(
            !state_of(resource)
                .unwrap()
                .incomplete
                .load(Ordering::Acquire)
        );
        release(resource);
    }

    /// A reserved host copy the GPU writes gets all its tiles before the
    /// write runs and keeps them however idle: its contents are the host's
    /// (not in title memory), and a title write after it does not reach it.
    #[test]
    fn gpu_written_reserved_textures_keep_their_tiles() {
        let test = "texture::tests::gpu_written_reserved_textures_keep_their_tiles";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let Some((resource, texture, tiles)) = reserved(&sampled_chain(256, 1, 9), None) else {
            eprintln!("{test}: skipped, the device reserves no textures (tiled resources tier 2)");
            return;
        };
        let _alone = SweptAlone::new(resource);
        let key = resource as usize;
        let (memory, _) = crate::resource::texture_memory(resource).unwrap();
        let size = staging(resource, texture).unwrap().layout.size as usize;
        let fill = |byte: u8| {
            // SAFETY: the texture's own title memory, `size` bytes.
            unsafe { memory.cast_mut().write_bytes(byte, size) }
        };
        let all = vec![true; tiles.units().len()];

        // Given back while idle, then written by the GPU: every tile
        // mapped again before the copy runs, which lands.
        fill(0x11);
        sample_in(host, resource, 1);
        assert_eq!(evict_later(key, 1000, 1001, u64::MAX / 4).0, 1);
        assert!(!tiles.any());
        let list = host.new_list().unwrap();
        copy_over(host, &list, resource, 0x5a);
        list.close();
        host.execute(&[&list]);
        host.drain();
        drop(list);
        assert!(tiles.holds(&all), "all mapped for the write");
        assert!(mip_is(host, texture, 0, 0x5a), "the copy landed");

        // Kept, however idle.
        assert_eq!(
            evict_later(key, 1000, 1001, u64::MAX / 4),
            (0, 0),
            "GPU-written"
        );
        assert!(tiles.holds(&all));
        assert!(mip_is(host, texture, 0, 0x5a));
        // The title's later writes do not reach it.
        fill(0x33);
        sample_in(host, resource, 1);
        assert!(mip_is(host, texture, 0, 0x5a));
        release(resource);
    }

    /// A reserved host copy maps the units whose title memory is committed:
    /// mip 0 of a texture whose memory misses a page there stays unmapped
    /// (it reads zero) while the other mips get their texels; once the page
    /// is committed and written, mip 0 is mapped and copied too.
    #[test]
    fn reserved_textures_map_the_mips_their_title_memory_has() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_DECOMMIT, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        let test = "texture::tests::reserved_textures_map_the_mips_their_title_memory_has";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        const LEN: usize = 1 << 20;
        const PAGE: usize = 4096;
        let at = private_memory(LEN, false);
        let Some((resource, texture, tiles)) = reserved(&sampled_chain(256, 1, 9), Some(at)) else {
            eprintln!("{test}: skipped, the device reserves no textures (tiled resources tier 2)");
            free_memory(at, LEN);
            return;
        };
        let _alone = SweptAlone::new(resource);
        let state = staging(resource, texture).unwrap();
        let layout = state.layout.clone();
        let size = layout.size as usize;
        assert!(size <= LEN);
        // A page of mip 0's title memory, decommitted; the rest filled.
        let (start, end) = title_mip(&layout, 0, 0).expect("mip 0");
        let page = (start as usize).next_multiple_of(PAGE);
        assert!(page + PAGE <= end as usize, "a whole page in mip 0");
        // SAFETY: a page of our own allocation; nothing reads it meanwhile.
        let freed = unsafe { VirtualFree((at + page) as *mut _, PAGE, MEM_DECOMMIT) };
        assert_ne!(freed, 0);
        // SAFETY: the committed parts of our allocation.
        unsafe {
            (at as *mut u8).write_bytes(0x22, page);
            ((at + page + PAGE) as *mut u8).write_bytes(0x22, size - page - PAGE);
        }
        sample_in(host, resource, 1);
        let units = tiles.units().to_vec();
        let wanted = state.wanted.lock().unwrap().clone();
        assert_eq!(wanted.len(), units.len());
        for (u, &w) in units.iter().zip(&wanted) {
            assert_eq!(w, u.first_mip != 0, "{u:?}");
        }
        let bytes = |filter: &dyn Fn(&crate::resource::pool::Unit) -> bool| -> u64 {
            units
                .iter()
                .filter(|u| filter(u))
                .map(|u| u64::from(u.tiles) * crate::resource::pool::TILE)
                .sum()
        };
        assert_eq!(tiles.mapped(), bytes(&|u| u.first_mip != 0));
        assert!(mip_is(host, texture, 1, 0x22), "mip 1 has its texels");
        assert!(mip_is(host, texture, 0, 0), "mip 0 is not mapped");
        assert!(!state.incomplete.load(Ordering::Acquire), "none left out");

        // Committed and written: mapped and copied.
        // SAFETY: the page decommitted above, committed again.
        let again =
            unsafe { VirtualAlloc((at + page) as *const _, PAGE, MEM_COMMIT, PAGE_READWRITE) };
        assert!(!again.is_null());
        // SAFETY: the texture's memory, ours, all committed.
        unsafe { (at as *mut u8).write_bytes(0x22, size) };
        sample_in(host, resource, 1);
        assert_eq!(tiles.mapped(), bytes(&|_| true));
        assert!(mip_is(host, texture, 0, 0x22));
        drop(state);
        release(resource);
        free_memory(at, LEN);
    }

    /// Where mips lie in title memory: the standard ones from the largest
    /// at the end of the slice down, the mip tail's at its start; slices
    /// one after another; nothing outside the layout.
    #[test]
    fn title_mips_follow_the_layout() {
        let desc = sampled_chain(256, 2, 9);
        let layout = crate::resource::texture_layout(&desc).unwrap().planes[0]
            .layout
            .clone();
        let mut covered = Vec::new();
        for slice in 0..2 {
            for mip in 0..9 {
                let (a, b) = title_mip(&layout, slice, mip).unwrap();
                assert!(a < b && b <= layout.size, "{slice} {mip}");
                covered.push((a, b));
            }
            // Mip 0 ends its slice.
            assert_eq!(
                title_mip(&layout, slice, 0).unwrap().1,
                (u64::from(slice) + 1) * layout.slice_size
            );
        }
        covered.sort_unstable();
        covered.dedup();
        // The ranges of one slice tile it without overlap (tail mips
        // share the tail block).
        for w in covered.windows(2) {
            assert!(w[0].1 <= w[1].0, "{:?}", w);
        }
        assert_eq!(covered.last().unwrap().1, layout.size);
        assert_eq!(title_mip(&layout, 2, 0), None);
        assert_eq!(title_mip(&layout, 0, 9), None);
    }

    #[test]
    fn committed_parts_merge_neighbours() {
        use windows_sys::Win32::System::Memory::{MEM_DECOMMIT, VirtualFree};
        const PAGE: usize = 4096;
        let at = private_memory(8 * PAGE, false);
        assert_eq!(committed_parts(at as *const u8, 8 * PAGE), [(0, 8 * PAGE)]);
        // SAFETY: pages of our own allocation, unused.
        unsafe { VirtualFree((at + 3 * PAGE) as *mut _, 2 * PAGE, MEM_DECOMMIT) };
        assert_eq!(
            committed_parts(at as *const u8, 8 * PAGE),
            [(0, 3 * PAGE), (5 * PAGE, 8 * PAGE)]
        );
        assert_eq!(
            committed_parts((at + PAGE) as *const u8, 3 * PAGE),
            [(0, 2 * PAGE)]
        );
        free_memory(at, 8 * PAGE);
    }

    #[test]
    fn eviction_follows_its_switch() {
        assert_eq!(evict_after_of(None), Some(EVICT_IDLE));
        assert_eq!(evict_after_of(Some("300")), Some(300));
        assert_eq!(evict_after_of(Some("off")), None);
        assert_eq!(evict_after_of(Some("0")), None);
        assert_eq!(evict_after_of(Some("soon")), Some(EVICT_IDLE));
    }

    /// A list reset while its submission runs hands what it holds to the
    /// timeline until that submission has run: a sweep in between releases
    /// nothing, however idle the texture; one after it releases the buffer
    /// (Codex's review of c44a716: the test of a sweep between a list reset
    /// and its fence was missing).
    #[test]
    fn a_list_reset_in_flight_keeps_its_staging_buffer_until_its_submission_ran() {
        let test = "texture::tests::a_list_reset_in_flight_keeps_its_staging_buffer_until_its_submission_ran";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let (resource, texture) = texture_4x4(0x11);
        let _alone = SweptAlone::new(resource);
        let list = host.new_list().unwrap();
        assert!(upload(&list, resource, texture));
        list.close();
        let mut gate = crate::host::Gate::close(host);
        // As the title submits: the list may be reset while it runs.
        host.submit(&[&list]);
        assert!(list.running(), "held behind the gate");
        list.reset();
        assert!(
            !release_staging_now(resource),
            "held by the submission in flight"
        );
        assert!(staged_bytes(resource).is_some());
        gate.open();
        host.drain();
        released(resource);
        list.close();
        drop(list);
        assert!(texels_4x4(host, texture).iter().all(|&t| t == 0x1111_1111));
        type Release = extern "system" fn(This) -> u32;
        assert_eq!(crate::object::slot::<Release>(resource, 0x10)(resource), 0);
    }

    /// A hole-only sparse read (committed pages remain, but no complete unit is
    /// readable) returns `None` from `snapshot`, leaving both the staged
    /// texels and `last.hash` intact until a full read or complete committed
    /// units arrive.
    #[test]
    fn hole_only_sparse_read_preserves_staged_texels_and_digest() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_DECOMMIT, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        let test = "texture::tests::hole_only_sparse_read_preserves_staged_texels_and_digest";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        const LEN: usize = 1 << 20;
        const PAGE: usize = 4096;
        let at = private_memory(LEN, false);
        let Some((resource, texture, tiles)) = reserved(&sampled_chain(256, 1, 9), Some(at)) else {
            eprintln!("{test}: skipped, the device reserves no textures (tiled resources tier 2)");
            free_memory(at, LEN);
            return;
        };
        let _alone = SweptAlone::new(resource);
        let state = staging(resource, texture).unwrap();
        let size = state.layout.size as usize;
        assert!(size <= LEN && size > PAGE);

        // SAFETY: committed private test allocation.
        unsafe { (at as *mut u8).write_bytes(0x22, size) };
        sample_in(host, resource, 1);
        assert!(mip_is(host, texture, 0, 0x22));
        assert!(mip_is(host, texture, 1, 0x22));
        let staged_hash = state.last.lock().unwrap().hash;
        assert!(staged_hash.is_some());

        // Decommit everything except one tail page (zero-filled), leaving
        // `any_committed` true while no complete unit is covered.
        let hole_end = (size - PAGE) & !(PAGE - 1);
        assert!(hole_end > 0 && hole_end < size);
        // SAFETY: decommitting pages inside our private test allocation.
        let freed = unsafe { VirtualFree(at as *mut _, hole_end, MEM_DECOMMIT) };
        assert_ne!(freed, 0);
        // SAFETY: the last page remains committed.
        unsafe { ((at + hole_end) as *mut u8).write_bytes(0, size - hole_end) };
        assert!(any_committed(at as *const u8, size));
        assert!(!super::committed(at as *const u8, size));
        assert!(!state.filled_units(&tiles).into_iter().any(|u| u));
        assert!(state.snapshot(size).is_none());

        sample_in(host, resource, 1);
        assert_eq!(state.last.lock().unwrap().hash, staged_hash);
        assert!(mip_is(host, texture, 0, 0x22));
        assert!(mip_is(host, texture, 1, 0x22));

        // Recommit all pages with legitimate zeros: the full zero texture now
        // stages and updates the digest.
        // SAFETY: recommitting our private test allocation.
        let again = unsafe { VirtualAlloc(at as *const _, hole_end, MEM_COMMIT, PAGE_READWRITE) };
        assert!(!again.is_null());
        // SAFETY: all of `[at, at + size)` is committed.
        unsafe { (at as *mut u8).write_bytes(0, size) };
        sample_in(host, resource, 1);
        assert_ne!(state.last.lock().unwrap().hash, staged_hash);
        assert!(mip_is(host, texture, 0, 0));
        assert!(mip_is(host, texture, 1, 0));

        release(resource);
        free_memory(at, LEN);
    }

    /// A partially committed sparse texture whose committed units genuinely
    /// contain all-zero bytes still snapshots, reaches `map_units`, and stages
    /// those zeros into the committed units even while another unit has a hole
    /// (tested on both guarded/unwatched and write-watched title memory).
    #[test]
    fn sparse_zero_committed_units_stage_and_map_despite_unmapped_hole() {
        use windows_sys::Win32::System::Memory::{MEM_DECOMMIT, VirtualFree};
        let test =
            "texture::tests::sparse_zero_committed_units_stage_and_map_despite_unmapped_hole";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        const LEN: usize = 1 << 20;
        const PAGE: usize = 4096;
        for watched in [false, true] {
            let at = private_memory(LEN, watched);
            let Some((resource, texture, tiles)) = reserved(&sampled_chain(256, 1, 9), Some(at))
            else {
                eprintln!(
                    "{test}: skipped, the device reserves no textures (tiled resources tier 2)"
                );
                free_memory(at, LEN);
                return;
            };
            let _alone = SweptAlone::new(resource);
            let state = staging(resource, texture).unwrap();
            let layout = state.layout.clone();
            let size = layout.size as usize;
            assert!(size <= LEN);

            // Stage non-zero texels across the full texture first.
            // SAFETY: committed private test allocation.
            unsafe { (at as *mut u8).write_bytes(0x22, size) };
            sample_in(host, resource, 1);
            assert!(mip_is(host, texture, 0, 0x22));
            assert!(mip_is(host, texture, 1, 0x22));
            let initial_hash = state.last.lock().unwrap().hash;

            // Leave a hole in mip 0 while writing legitimate zeros into the
            // remaining committed units (mip 1..).
            let (start, end) = title_mip(&layout, 0, 0).expect("mip 0");
            let page = (start as usize).next_multiple_of(PAGE);
            assert!(page + PAGE <= end as usize);
            // SAFETY: decommitting one page of our private test allocation.
            let freed = unsafe { VirtualFree((at + page) as *mut _, PAGE, MEM_DECOMMIT) };
            assert_ne!(freed, 0);
            // SAFETY: writing the remaining committed pages of our allocation.
            unsafe {
                (at as *mut u8).write_bytes(0, page);
                ((at + page + PAGE) as *mut u8).write_bytes(0, size - page - PAGE);
            }

            sample_in(host, resource, 1);
            let units = tiles.units().to_vec();
            let wanted = state.wanted.lock().unwrap().clone();
            assert_eq!(wanted.len(), units.len());
            for (u, &w) in units.iter().zip(&wanted) {
                assert_eq!(w, u.first_mip != 0, "watched={watched}: {u:?}");
            }
            assert!(mip_is(host, texture, 1, 0), "watched={watched}");
            assert!(mip_is(host, texture, 0, 0), "watched={watched}");
            if !watched {
                assert_ne!(state.last.lock().unwrap().hash, initial_hash);
            }

            release(resource);
            free_memory(at, LEN);
        }
    }

    /// CPU-written linear `R8_UNORM` movie planes get their own per-list upload
    /// buffers, so overlapping flights never overwrite an earlier frame's upload
    /// or skip uploads when the title rewrites title memory between lists.
    #[test]
    fn movie_planes_use_distinct_per_list_upload_buffers() {
        let test = "texture::tests::movie_planes_use_distinct_per_list_upload_buffers";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        const COPY_SOURCE: u32 = 0x800;
        const DXGI_FORMAT_R8_UNORM: u32 = 61;

        for (width, height) in [(256u64, 4u32), (64, 4)] {
            let desc =
                crate::resource::tests_support::linear_sampled(width, height, DXGI_FORMAT_R8_UNORM);
            const LEN: usize = 64 << 10;
            let at = private_memory(LEN, false);
            let resource = crate::resource::tests_support::at_address(at as u64, &desc);
            let snap_res = committed(&desc);
            let started = std::time::Instant::now();
            while crate::resource::host_resource(resource).is_none()
                || crate::resource::host_resource(snap_res).is_none()
            {
                assert!(
                    started.elapsed().as_secs() < 5,
                    "the host copies never became resident"
                );
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            let texture = crate::resource::host_resource(resource).unwrap();
            let snap_tex = crate::resource::host_resource(snap_res).unwrap();
            let (memory, _) = crate::resource::texture_memory(resource).unwrap();
            let state = staging(resource, texture).unwrap();
            assert!(state.movie_plane());
            assert!(state.buffer().is_none());
            assert_eq!(staged_bytes(resource), None);
            let size = state.layout.size as usize;
            let fill = |byte: u8| {
                // SAFETY: the texture's own title memory, `size` bytes.
                unsafe { memory.cast_mut().write_bytes(byte, size) }
            };

            let list1 = host.new_list().unwrap();
            let snap = host.new_list().unwrap();
            let list2 = host.new_list().unwrap();

            let mut gate = crate::host::Gate::close(host);
            fill(0x11);
            assert!(upload(&list1, resource, texture));
            list1.close();
            host.execute(&[&list1]);

            crate::list::ensure_state(&snap, resource, COPY_SOURCE);
            crate::list::ensure_state(&snap, snap_res, COPY_DEST);
            // SAFETY: open list; both textures are live and transitioned.
            unsafe { snap.list.CopyResource(snap_tex, texture) };
            snap.close();
            host.execute(&[&snap]);

            fill(0x22);
            assert!(upload(&list2, resource, texture));
            list2.close();
            host.execute(&[&list2]);

            gate.open();
            host.drain();
            assert!(decided(&state).is_empty());

            let row_width = width as usize;
            let check = |tex: &ID3D12Resource, expected: u8| {
                let raw = read_back(host, tex, COPY_DEST);
                for y in 0..height as usize {
                    assert!(
                        raw[y * 256..y * 256 + row_width]
                            .iter()
                            .all(|&b| b == expected),
                        "row {y} mismatch for {width}x{height}: {:x?}",
                        &raw[y * 256..y * 256 + row_width.min(16)]
                    );
                }
            };
            check(snap_tex, 0x11);
            check(texture, 0x22);

            // Lose the CPU backing after recording. Both the predicated path
            // and the unpredicated list with a primed buffer retain frame two.
            for no_predicate in [false, true] {
                use windows_sys::Win32::System::Memory::{
                    MEM_COMMIT, MEM_DECOMMIT, PAGE_READWRITE, VirtualAlloc, VirtualFree,
                };
                let _mode = no_predicate.then(WithoutPredicates::new);
                let reader = host.new_list().unwrap();
                assert!(upload(&reader, resource, texture));
                reader.close();
                // SAFETY: these pages belong to the test and all earlier lists have drained.
                assert_ne!(unsafe { VirtualFree(at as *mut _, LEN, MEM_DECOMMIT) }, 0);
                host.execute(&[&reader]);
                host.drain();
                check(texture, 0x22);
                // SAFETY: recommit the same reserved pages before writing the next frame.
                let again =
                    unsafe { VirtualAlloc(at as *const _, LEN, MEM_COMMIT, PAGE_READWRITE) };
                assert!(!again.is_null());
                fill(0x22);
            }
            drop((list1, snap, list2));
            type Release = extern "system" fn(This) -> u32;
            assert_eq!(crate::object::slot::<Release>(resource, 0x10)(resource), 0);
            assert_eq!(crate::object::slot::<Release>(snap_res, 0x10)(snap_res), 0);
            free_memory(at, LEN);
        }
    }
}
