//! Textures at one address share the GPU's copy (after VIctory #643, by
//! nathanbzrr).
//!
//! On the console, resources at one address are one memory. Here a title
//! texture has a host copy, and what the GPU renders or writes into it stays
//! there: title memory keeps its old bytes. A texture made later at the same
//! address (a shader view of placed memory, `texture_at`; a resource placed
//! at the same heap offset or made at the same address) used to get a host
//! copy of its own, filled from title memory: zeros where the GPU drew. It now
//! takes the GPU-written copy when the host can view that copy as the title
//! asks:
//! - the same shape (dimension, size, array or depth, mips, samples) and
//!   format family;
//! - only the writes (render target, depth, unordered access) the copy
//!   allows, shader reads only if the copy allows them;
//! - a writable alias of a typed copy in the copy's own format.
//!
//! Only resources whose own description asks for GPU writes are sources:
//! once the last of them at an address is released, a texture made there
//! later (memory reused for streamed data, say) gets its own copy again.
//!
//! A texture the GPU does not write, made over a GPU-written one of the
//! same shape in another format family whose texels are as wide (the
//! title's compute programs pack light as R9G9B9E5 into R32_UINT textures
//! and read it through R9G9B9E5 views placed at their addresses), keeps a
//! copy of its own: the host cannot view one format as the other. It is
//! brought up to date from the written texture by a copy when a list reads
//! it ([`written_beside`], `resource::alias_copy`; ported from fxceless's
//! VIctory fork, `alias_of`, commit 4e17478).
//!
//! shortcut: aliases of another shape (a half-size alias, other levels) or
//! texel width still get their own copy, filled from title memory; that is
//! G19's shared-memory model, not a lookup by address.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use super::{HostTexture, ResourceDesc};

/// Render target, depth, unordered access.
pub(super) const WRITABLE: u32 = 0x7;
const DENY_SHADER_RESOURCE: u32 = 0x8;

/// What two descriptions must have in common to share a host copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Shape {
    dimension: u32,
    width: u64,
    height: u32,
    depth_or_array: u16,
    mips: u32,
    samples: u32,
    family: u32,
}

fn shape(desc: &ResourceDesc) -> Shape {
    Shape {
        dimension: desc.dimension,
        width: desc.width,
        height: desc.height,
        depth_or_array: desc.depth_or_array_size,
        mips: super::mip_count(desc),
        samples: desc.sample_count.max(1),
        // The title's format family: a stand-in is shared by title formats
        // of other families (R9G9B9E5 and R16G16B16A16_FLOAT both as
        // RGBA16F), whose texels the title reads differently.
        family: super::family(desc.format),
    }
}

/// A live title texture the GPU writes, at its address.
struct Source {
    resource: usize,
    shape: Shape,
    /// Bytes per texel of its title format (`None` for block formats).
    texel: Option<u64>,
    /// Its host copy's flags and format.
    flags: u32,
    format: u32,
    host: Weak<HostTexture>,
}

static SOURCES: Mutex<BTreeMap<u64, Vec<Source>>> = Mutex::new(BTreeMap::new());

/// Bumped whenever a source comes or goes: a lookup made before may now
/// find another copy.
static CHANGES: AtomicU64 = AtomicU64::new(1);

pub(super) fn changes() -> u64 {
    CHANGES.load(Ordering::Acquire)
}

fn sources() -> std::sync::MutexGuard<'static, BTreeMap<u64, Vec<Source>>> {
    SOURCES.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether the format is the typeless one of its family.
fn typeless(format: u32) -> bool {
    super::family(format) == format && super::family(format.wrapping_add(1)) == format
}

/// Why a texture described by `want` cannot share `source`'s host copy.
fn refusal(source: &Source, want: &ResourceDesc) -> Option<&'static str> {
    if source.shape != shape(want) {
        return Some("another shape or format family");
    }
    let flags = super::host_flags(want.flags);
    if flags & WRITABLE & !source.flags != 0 {
        return Some("writes the copy does not allow");
    }
    if flags & DENY_SHADER_RESOURCE == 0 && source.flags & DENY_SHADER_RESOURCE != 0 {
        return Some("shader reads the copy does not allow");
    }
    if flags & WRITABLE != 0 && !typeless(source.format) {
        let own = super::host_desc(want, &super::format::HostCaps)
            .map_or_else(|_| super::pc_format(want.format), |d| d.format);
        if own != source.format {
            return Some("a writable view of a typed copy in another format");
        }
    }
    None
}

/// Whether aliases share (diagnostics builds: `XTP_ALIAS_SHARING=0` turns
/// it off, to compare).
fn enabled() -> bool {
    #[cfg(feature = "diagnostics")]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("XTP_ALIAS_SHARING").map_or(true, |v| v != "0"))
    }
    #[cfg(not(feature = "diagnostics"))]
    true
}

/// The GPU-written host copy at `address` a texture described by `want`
/// can share, if there is one. `how` says how the texture is made, for the
/// log: the first aliases met are logged with what became of them.
pub(super) fn shared(address: u64, want: &ResourceDesc, how: &str) -> Option<Arc<HostTexture>> {
    let (found, why) = lookup(address, want)?;
    let on = enabled();
    note(address, want, how, found.as_ref().map(|f| f.1), why, on);
    found.filter(|_| on).map(|(host, _)| host)
}

/// As [`shared`], without the log.
pub(super) fn current(address: u64, want: &ResourceDesc) -> Option<Arc<HostTexture>> {
    let (found, _) = lookup(address, want)?;
    found.filter(|_| enabled()).map(|(host, _)| host)
}

/// A host copy to share and the source it comes from.
type Found = (Arc<HostTexture>, usize);

/// The copy `want` can share at `address` and its source, or why none
/// fits; `None` when no GPU-written texture lives there.
fn lookup(address: u64, want: &ResourceDesc) -> Option<(Option<Found>, &'static str)> {
    let mut map = sources();
    let list = map.get_mut(&address)?;
    list.retain(|s| s.host.strong_count() > 0);
    if list.is_empty() {
        map.remove(&address);
        return None;
    }
    let mut why = "released";
    for source in list.iter() {
        match refusal(source, want) {
            Some(reason) => why = reason,
            None => {
                if let Some(host) = source.host.upgrade() {
                    return Some((Some((host, source.resource)), ""));
                }
            }
        }
    }
    Some((None, why))
}

/// Bytes per texel of a title format, for formats of one texel a block.
fn texel(format: u32) -> Option<u64> {
    super::element(format).and_then(|(bytes, width)| (width == 1).then_some(bytes))
}

/// The GPU-written texture at `address` whose texels a texture described
/// by `want`, which the GPU does not write, reads in another format: one of
/// the same shape (dimension, size, array or depth, levels, one sample)
/// but another format family, texels as wide, and no depth. The title
/// resource is pinned under the registry's lock and returned held alongside
/// its host copy so releasing the title's reference before the alias copy is
/// recorded cannot free it; a source already being released is skipped.
/// After fxceless's `alias_of` (and fork PR #1336).
pub(super) fn written_beside(
    address: u64,
    want: &ResourceDesc,
) -> Option<(super::Held, Arc<HostTexture>)> {
    if super::host_flags(want.flags) & WRITABLE != 0 {
        return None;
    }
    let want_shape = shape(want);
    let want_texel = texel(want.format)?;
    const DEPTH: u32 = 0x2;
    let map = sources();
    map.get(&address)?.iter().find_map(|s| {
        let fits = s.shape.family != want_shape.family
            && Shape {
                family: want_shape.family,
                ..s.shape
            } == want_shape
            && want_shape.samples == 1
            && s.texel == Some(want_texel)
            && s.flags & DEPTH == 0;
        if !fits {
            return None;
        }
        // Upgrade `host` first so a `Held` guard is never dropped while
        // `SOURCES` is locked (the last release calls `unregister`).
        let host = s.host.upgrade()?;
        let held = super::Held::pin(s.resource as crate::object::This)?;
        Some((held, host))
    })
}

static SEEN: AtomicU64 = AtomicU64::new(0);
static SHARED: AtomicU64 = AtomicU64::new(0);

pub(super) fn counts() -> (u64, u64) {
    let shared = SHARED.load(Ordering::Relaxed);
    (SEEN.load(Ordering::Relaxed), shared)
}

/// Logs the first aliases met over a GPU-written texture (then every
/// power of two), shared or not and why.
fn note(address: u64, want: &ResourceDesc, how: &str, with: Option<usize>, why: &str, on: bool) {
    let n = SEEN.fetch_add(1, Ordering::Relaxed) + 1;
    if on && with.is_some() {
        SHARED.fetch_add(1, Ordering::Relaxed);
    }
    if n > 64 && !n.is_power_of_two() {
        return;
    }
    let want = format!(
        "dim {} {}x{}x{} mips {} f{} flags {:#x}",
        want.dimension,
        want.width,
        want.height,
        want.depth_or_array_size,
        want.mip_levels,
        want.format,
        want.flags & 0xf
    );
    match with {
        Some(source) if on => tracing::info!(
            n,
            address = format_args!("{address:#x}"),
            how,
            want,
            source = format_args!("{source:#x}"),
            "a texture over a GPU-written one shares its host copy"
        ),
        Some(source) => tracing::info!(
            n,
            address = format_args!("{address:#x}"),
            how,
            want,
            source = format_args!("{source:#x}"),
            "a texture over a GPU-written one would share its host copy (XTP_ALIAS_SHARING=0)"
        ),
        None => tracing::info!(
            n,
            address = format_args!("{address:#x}"),
            how,
            want,
            why,
            "a texture over a GPU-written one gets a copy of its own"
        ),
    }
}

/// Notes `resource`, a texture at `address` described by `desc` with the
/// host copy `host`, as a source if its description asks for GPU writes.
pub(super) fn register(
    address: u64,
    resource: usize,
    desc: &ResourceDesc,
    host: &Arc<HostTexture>,
) {
    if address == 0 || super::host_flags(desc.flags) & WRITABLE == 0 {
        return;
    }
    // SAFETY: a live resource.
    let d = unsafe { host.texture.GetDesc() };
    let source = Source {
        resource,
        shape: shape(desc),
        texel: texel(desc.format),
        flags: d.Flags.0 as u32,
        format: d.Format.0 as u32,
        host: Arc::downgrade(host),
    };
    sources().entry(address).or_default().push(source);
    CHANGES.fetch_add(1, Ordering::AcqRel);
}

/// The title released `resource`, a texture at `address`.
pub(super) fn unregister(address: u64, resource: usize) {
    let mut map = sources();
    let Some(list) = map.get_mut(&address) else {
        return;
    };
    let before = list.len();
    list.retain(|s| s.resource != resource);
    if list.len() != before {
        CHANGES.fetch_add(1, Ordering::AcqRel);
    }
    if list.is_empty() {
        map.remove(&address);
    }
}

pub(super) fn has_sources(address: u64) -> bool {
    sources().contains_key(&address)
}
