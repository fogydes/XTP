//! Translated shaders kept on disk (decision 0003): under the configured
//! cache folder, keyed by the container's content, by the translator that
//! made it (the fingerprint of `xtp-shader`'s sources, [`xtp_shader::SOURCES`],
//! and of this file, with every translation setting in the value the
//! translator uses: [`build`]), so a
//! changed translator never serves an old translation while a build that
//! changed only elsewhere keeps them, and by the wave sizes it was made for
//! (VIctory #223: a translation for a 32-lane host is not a 64-lane one's).
//! Entries go through `xtp-pipeline`'s content-addressed cache, which
//! writes them atomically; an entry that does not read back is translated
//! again.
//!
//! The entry is our own encoding of [`HostShader`]: a field added to it
//! must be added to `encode` and `decode` too (entries of older builds
//! are ignored anyway: this file is part of the key).
//!
//! A container this build cannot translate for a target is cached under the
//! same key as a translation. The disk cache verifies the payload's length
//! and digest first. The shader decoder then checks the refusal's marker,
//! text length, checksum and UTF-8. The translator's refusals
//! ([`xtp_shader::ShaderError`]: a malformed container, GFX10 code it
//! cannot decode or does not handle yet) depend only on the container and
//! on what `key()` hashes (the translator, every setting it runs with, the
//! target's wave sizes), so it would fail the same way again. Without that, every
//! run translated the same few hundred refused programs again (seconds each
//! for the large ones) while the title waited for them. Nothing transient
//! is kept: the host driver's compile, where out of memory and a removed
//! device happen, comes after this cache, and an entry that cannot be
//! written is answered from the translation without being kept.
//!
//! Across translator builds, nothing is served that this build's translator
//! did not make: an entry is found only under this build's key. What
//! carries over a translator change is downstream of the translation: a
//! host pipeline is named by the bytes of its translated shaders, root
//! signature and state (`pipeline::library`), so a translation whose bytes
//! did not change finds its pipelines again, and one that did gets new
//! ones. `xtp precook` translates the pack again after a change and
//! compares each answer with the entry the translator before kept
//! ([`compare_with`]): a count, for what the change altered, never an
//! answer. (A per-program fingerprint of the translator code a translation
//! ran is not one that can be proven: code changes a translation without
//! running, through constants, types, inlined generics and tables.)

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use xtp_pipeline::{Cache, Converter};
use xtp_shader::emit::{HostShader, StaticSampler, Target, UserData};
use xtp_shader::{HwStage, RootType};

static CACHE: OnceLock<Cache> = OnceLock::new();
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);

/// The disk the cache may take, oldest entries out first. One translator's
/// answers for the whole pack take 1.1 GiB (25,889 entries, `xtp precook`
/// on the PC, 2026-11-25): more than the former 1 GiB, which evicted what
/// the precook or the session before had made. Room for a few translators
/// (switches on and off) side by side.
const LIMIT: u64 = 8 << 30;

/// Keeps translated shaders in `dir` from now on.
pub fn use_shader_cache(dir: PathBuf) {
    match Cache::open_with_limit(dir, LIMIT) {
        Ok(cache) => {
            let _ = CACHE.set(cache);
        }
        Err(e) => tracing::warn!(error = %e, "no shader cache"),
    }
}

/// Lists the cache's folder again and writes its index anew from what is
/// there (`xtp_pipeline::Cache::reindex`), then trims it to its limit:
/// whether it could (false: a folder of the cache could not be listed, and
/// the index is left as it was, or the listing could not be written as the
/// index file: the log says which), and the entries and bytes it holds after;
/// `None` without a cache. The index only sizes the cache, and the sessions
/// between two precooks can leave it stale (a process stopped between an
/// entry's rename and its record, an index replaced under another
/// process's appends, files removed by hand): `xtp precook` sets it right
/// before each pass, while no title waits.
pub fn reindex() -> Option<(bool, usize, u64)> {
    let cache = CACHE.get()?;
    let listed = cache.reindex();
    let stats = cache.stats();
    Some((listed, stats.entries, stats.bytes))
}

/// Shaders found in the cache and translated since the start. Refusals,
/// fresh or kept, are neither: the frame line counts shaders.
pub(crate) fn counts() -> (u64, u64) {
    (HITS.load(Ordering::Relaxed), MISSES.load(Ordering::Relaxed))
}

/// Counts `result` in `counter` if it is a shader (not a refusal).
fn count<T, E>(counter: &AtomicU64, result: Result<T, E>) -> Result<T, E> {
    if result.is_ok() {
        counter.fetch_add(1, Ordering::Relaxed);
    }
    result
}

/// What identifies the translator a translation made now runs: the
/// fingerprint of `xtp-shader`'s sources ([`xtp_shader::SOURCES`]), this
/// file (the entries' encoding, [`ENCODING`]) and every setting the
/// translation runs with ([`xtp_shader::emit::Settings`]: the loop limits,
/// the loops' budget, the DXIL clean-up passes, the raw structured reads,
/// the loop-limit counter), in the values the translator uses. Not the
/// switches' spelling: `XTP_SHADER_LOOP_LIMIT` unset and set to 16384 give
/// the same loop limit, and once had the same key, but not the same
/// counted loops' limit (2^20 against 16384), so a session could get the
/// other setting's translation. Read at each call, as the translator reads
/// them: a switch of this thread alone (`with_loop_count`) has its own key.
/// Not the whole executable: before, the least change anywhere (the HLE, a
/// log line, another worker's branch) made every translation again, and
/// each worktree's sessions started cold.
fn build() -> [u8; 32] {
    static TRANSLATOR: OnceLock<[u8; 32]> = OnceLock::new();
    let translator = TRANSLATOR.get_or_init(|| {
        announce(&xtp_shader::emit::settings());
        translator_of(xtp_shader::SOURCES, ENCODING)
    });
    build_of(translator, &xtp_shader::emit::settings())
}

/// Says, once, which settings differ from the defaults.
fn announce(s: &xtp_shader::emit::Settings) {
    use xtp_shader::emit::{DEFAULT_LOOP_BUDGET, DEFAULT_LOOP_LIMIT};
    if s.loop_limit != DEFAULT_LOOP_LIMIT {
        let (limit, counted) = (s.loop_limit, s.counted_loop_limit);
        tracing::warn!(
            limit,
            counted,
            "XTP_SHADER_LOOP_LIMIT: translated loops stop at another limit"
        );
    } else if s.counted_loop_limit != xtp_shader::emit::DEFAULT_COUNTED_LOOP_LIMIT {
        let counted = s.counted_loop_limit;
        tracing::warn!(
            counted,
            "XTP_SHADER_LOOP_LIMIT: counted loops stop at the general limit"
        );
    }
    if s.loop_budget != DEFAULT_LOOP_BUDGET {
        let budget = s.loop_budget;
        tracing::warn!(
            budget,
            "XTP_SHADER_LOOP_BUDGET: back edges a lane may take in all loops together (0: no budget)"
        );
    }
    if s.dxil_opt != 0 {
        let passes = s.dxil_opt;
        tracing::info!(
            passes,
            "XTP_DXIL_OPT: clean-up passes run over translated shaders"
        );
    }
    if !s.raw_structured {
        tracing::info!("XTP_RAW_STRUCTURED=0: structured reads through typed views");
    }
    if s.loop_count {
        tracing::info!("XTP_SHADER_LOOP_COUNT: translated loops count the lanes they stop");
    }
    if !s.water_tile_slots {
        tracing::info!("XTP_WATER_TILE_SLOTS=0: wave32 per-lane tile-list rewrites disabled");
    }
}

/// This file: how entries are encoded and decoded, and which translation
/// each holds, so a change to it is a new key.
const ENCODING: &[u8] = include_bytes!("shader_cache.rs");

/// The translator's part of [`build`]'s key: its sources and the entries'
/// encoding.
fn translator_of(sources: &str, encoding: &[u8]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    for part in [sources.as_bytes(), encoding] {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part);
    }
    *h.finalize().as_bytes()
}

/// [`build`]'s key from its parts: the translator ([`translator_of`]) and
/// every setting, field by field (the pattern names them all, so a setting
/// added to [`xtp_shader::emit::Settings`] does not build until it is here).
fn build_of(translator: &[u8; 32], settings: &xtp_shader::emit::Settings) -> [u8; 32] {
    let xtp_shader::emit::Settings {
        loop_limit,
        counted_loop_limit,
        spin_limit,
        loop_budget,
        dxil_opt,
        raw_structured,
        loop_count,
        water_tile_slots,
    } = *settings;
    let mut h = blake3::Hasher::new();
    h.update(translator);
    for (name, value) in [
        ("loop limit", loop_limit),
        ("counted loop limit", counted_loop_limit),
        ("spin limit", spin_limit),
        ("loop budget", loop_budget),
        ("dxil opt", dxil_opt),
        ("raw structured", u32::from(raw_structured)),
        ("loop count", u32::from(loop_count)),
        ("water tile slots", u32::from(water_tile_slots)),
    ] {
        h.update(&(name.len() as u64).to_le_bytes());
        h.update(name.as_bytes());
        h.update(&value.to_le_bytes());
    }
    *h.finalize().as_bytes()
}

/// The key this run's translations are kept under: this build and the
/// host's target ([`crate::device::host_target`]). `xtp precook` notes it
/// beside a complete pass, to know a pass for this key was made.
#[cfg(windows)]
pub(crate) fn host_key() -> [u8; 32] {
    key(&build(), crate::device::host_target())
}

/// The translation of a console shader container for a host that runs
/// `target`'s wave sizes, from the cache when there is one.
pub(crate) fn host_shader(container: &[u8], target: Target) -> Result<HostShader, String> {
    host_shader_with(container, target, xtp_shader::emit::Options::default())
}

/// [`host_shader`] with per-shader translation `options`; non-default options
/// are cached under a separate key so default translations remain unaffected.
// Ported from PastiiJay/klbf-lab commit f73233c8 (author aidzz / @BrokeAsh, GPL-3.0-or-later).
pub(crate) fn host_shader_with(
    container: &[u8],
    target: Target,
    options: xtp_shader::emit::Options,
) -> Result<HostShader, String> {
    let translate = |c: &[u8]| {
        xtp_shader::emit::translate_for_with(c, target, options).map_err(|e| e.to_string())
    };
    let key = option_key(&key(&build(), target), options);
    match CACHE.get() {
        Some(cache) => kept_with_key(cache, &key, target, container, translate),
        None => translate(container),
    }
}

/// The translation of the export shader (ES) of a console vertex shader
/// container, the vertex shader of a pipeline with a geometry shader
/// (`xtp_shader::emit::translate_es_for_with`), from the cache when there is
/// one: under a key of its own, the container being the same.
pub(crate) fn host_es_shader_with(
    container: &[u8],
    target: Target,
    options: xtp_shader::emit::Options,
) -> Result<HostShader, String> {
    let translate = |c: &[u8]| {
        xtp_shader::emit::translate_es_for_with(c, target, options).map_err(|e| e.to_string())
    };
    match CACHE.get() {
        Some(cache) => {
            let base = key(&build(), target);
            let es_base = es_key(&base);
            let key = option_key(&es_base, options);
            let earlier = EARLIER
                .get()
                .filter(|(now, _)| *now == base && key == es_base)
                .map(|(_, e)| es_key(e));
            compared(cache, &key, earlier.as_ref(), container, translate)
        }
        None => translate(container),
    }
}

/// Folds non-default per-shader [`xtp_shader::emit::Options`] into `key` so
/// alternate translations never collide with default cache entries.
fn option_key(base: &[u8; 32], options: xtp_shader::emit::Options) -> [u8; 32] {
    if options == xtp_shader::emit::Options::default() {
        return *base;
    }
    let mut h = blake3::Hasher::new();
    h.update(base);
    h.update(b"self_sized_loads_via_memory");
    h.update(&[u8::from(options.self_sized_loads_via_memory)]);
    *h.finalize().as_bytes()
}

/// An ES translation's key ([`host_es_shader_with`]) from the container's
/// translation's ([`key`]).
fn es_key(key: &[u8; 32]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(key);
    h.update(b"export shader");
    *h.finalize().as_bytes()
}

/// `translate`'s translation of `container` for `target` from `cache`,
/// under this build's key for that target; made and stored on a miss.
/// The target that keys the entry is the one `translate` is for: the
/// caller passes both. An error of `translate` is kept too, as the
/// container's answer for this build and target: it must be one that would
/// come again (a refusal of the translator), never a passing condition.
#[cfg(test)]
pub(crate) fn kept(
    cache: &Cache,
    container: &[u8],
    target: Target,
    translate: impl Fn(&[u8]) -> Result<HostShader, String> + Send + Sync,
) -> Result<HostShader, String> {
    let key = key(&build(), target);
    kept_with_key(cache, &key, target, container, translate)
}

fn kept_with_key(
    cache: &Cache,
    key: &[u8; 32],
    target: Target,
    container: &[u8],
    translate: impl Fn(&[u8]) -> Result<HostShader, String> + Send + Sync,
) -> Result<HostShader, String> {
    let base = crate::shader_cache::key(&build(), target);
    let earlier = EARLIER
        .get()
        .filter(|(now, _)| *now == base && *key == base)
        .map(|&(_, e)| e);
    compared(cache, key, earlier.as_ref(), container, translate)
}

/// The key this build's translations are compared under, and the key of
/// the translator before ([`compare_with`]).
static EARLIER: OnceLock<([u8; 32], [u8; 32])> = OnceLock::new();

/// How this build's translations compare with the translator before's
/// ([`compare_with`]): the same answer (translation or refusal, byte for
/// byte), another one, or none kept to compare with.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Compared {
    pub same: u64,
    pub changed: u64,
    pub unknown: u64,
    /// The first containers whose answer changed: their BLAKE3's first 16
    /// hexadecimal digits (the pipeline log's `hash=`).
    pub examples: Vec<String>,
}

static COMPARED: Mutex<Compared> = Mutex::new(Compared {
    same: 0,
    changed: 0,
    unknown: 0,
    examples: Vec::new(),
});

/// Changed containers [`Compared`] names, at most.
const EXAMPLES: usize = 16;

/// From now on, each container this build translates for the host
/// ([`host_shader`], [`host_es_shader`]) is compared with the entry the
/// translator of key `earlier` (a host key, [`host_key`]'s, of another
/// build) kept for it, if the cache holds one. That entry is read, never
/// answered: the answer is this build's translation. False if it is too
/// late (a comparison is already set).
#[cfg(windows)]
pub(crate) fn compare_with(earlier: [u8; 32]) -> bool {
    EARLIER.set((host_key(), earlier)).is_ok()
}

/// The comparisons so far ([`compare_with`]).
pub(crate) fn compared_counts() -> Compared {
    COMPARED.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// [`cached_made`]'s answer, comparing a translation made now with the
/// entry the translator of key `earlier` kept for `container`.
fn compared(
    cache: &Cache,
    key: &[u8; 32],
    earlier: Option<&[u8; 32]>,
    container: &[u8],
    translate: impl Fn(&[u8]) -> Result<HostShader, String> + Send + Sync,
) -> Result<HostShader, String> {
    let (answer, made) = cached_made(cache, key, container, translate);
    if let Some(earlier) = earlier
        && made
    {
        let now = match &answer {
            Ok(shader) => encode(shader),
            Err(error) => encode_failure(error),
        };
        let kept = entry(cache, earlier, container);
        let mut c = COMPARED.lock().unwrap_or_else(|e| e.into_inner());
        match kept {
            None => c.unknown += 1,
            Some(kept) if *kept == *now => c.same += 1,
            Some(_) => {
                c.changed += 1;
                if c.examples.len() < EXAMPLES {
                    c.examples
                        .push(blake3::hash(container).to_hex()[..16].to_owned());
                }
            }
        }
    }
    answer
}

/// The entry kept under `key` for `container` (a translation or a refusal,
/// encoded), if the cache holds a sound one: read only, never made.
fn entry(cache: &Cache, key: &[u8; 32], container: &[u8]) -> Option<Arc<[u8]>> {
    /// [`Translate`]'s identity, with nothing to translate with.
    struct Peek;
    impl Converter for Peek {
        fn id(&self) -> &'static str {
            CONVERTER
        }
        fn version(&self) -> u32 {
            CONVERTER_VERSION
        }
        fn convert(&self, _: &[u8]) -> anyhow::Result<Vec<u8>> {
            anyhow::bail!("not kept")
        }
    }
    let input = [&key[..], container].concat();
    cache
        .get_or_convert_with(&Peek, &input, |bytes| {
            (decode(&bytes).is_some() || decode_failure(&bytes).is_some()).then_some(bytes)
        })
        .ok()
}

/// The part of an entry's key before the container: the build, and the
/// wave sizes translated for (fixed sizes, so no two pairs run together).
fn key(build: &[u8; 32], target: Target) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(build);
    h.update(&target.key());
    *h.finalize().as_bytes()
}

/// The cache entry's converter: the key ([`key`]) and the container in,
/// the encoded translation (or refusal) out. `ran` tells a miss from a hit.
struct Translate<F> {
    translate: F,
    ran: AtomicBool,
}

impl<F> Translate<F> {
    fn new(translate: F) -> Translate<F> {
        Translate {
            translate,
            ran: AtomicBool::new(false),
        }
    }
}

/// The converter's identity in the cache's keys ([`Translate`], and
/// [`entry`]'s reader).
const CONVERTER: &str = "shader.dxil";
/// The build fingerprint in the input does the versioning.
const CONVERTER_VERSION: u32 = 1;

impl<F: Fn(&[u8]) -> Result<HostShader, String> + Send + Sync> Converter for Translate<F> {
    fn id(&self) -> &'static str {
        CONVERTER
    }
    fn version(&self) -> u32 {
        CONVERTER_VERSION
    }
    fn convert(&self, input: &[u8]) -> anyhow::Result<Vec<u8>> {
        self.ran.store(true, Ordering::Relaxed);
        Ok(match (self.translate)(&input[32..]) {
            Ok(shader) => encode(&shader),
            Err(error) => encode_failure(&error),
        })
    }
}

/// [`cached_made`]'s answer alone (the tests' way in).
#[cfg(test)]
fn cached(
    cache: &Cache,
    key: &[u8; 32],
    container: &[u8],
    translate: impl Fn(&[u8]) -> Result<HostShader, String> + Send + Sync,
) -> Result<HostShader, String> {
    cached_made(cache, key, container, translate).0
}

/// `translate`'s answer for `container` from `cache` under `key`, made
/// and stored on a miss, and whether `translate` made it now (a miss whose
/// answer the cache gave back: not a failure of the cache itself, which is
/// no answer of this translator to compare).
fn cached_made(
    cache: &Cache,
    key: &[u8; 32],
    container: &[u8],
    translate: impl Fn(&[u8]) -> Result<HostShader, String> + Send + Sync,
) -> (Result<HostShader, String>, bool) {
    let input = [&key[..], container].concat();
    let converter = Translate::new(translate);
    let result = cache.get_or_convert_with(&converter, &input, |bytes| {
        decode(&bytes)
            .map(Ok)
            .or_else(|| decode_failure(&bytes).map(Err))
    });
    let made = converter.ran.load(Ordering::Relaxed);
    let result = match result {
        Ok(result) => result,
        Err(e) => return (Err(format!("{e:#}")), false),
    };
    let (counter, what) = if made {
        (&MISSES, "translated")
    } else {
        (&HITS, "from the cache")
    };
    tracing::debug!(ok = result.is_ok(), "shader {what}");
    (count(counter, result), made)
}

/// For tests: [`wave64_program`] with its lane reads.
#[cfg(test)]
pub(crate) fn wave64_in_half_program() -> (xtp_shader::Container<'static>, xtp_shader::NativeProgram)
{
    wave64_program(true)
}

/// For tests: a console wave64 compute program of 64 threads that stores
/// one dword per thread, dword 64 g + t for thread t of group g, through a
/// raw UAV whose table is user data 0, with an empty container. With
/// `in_half`, its lanes read only inside their 32-lane half (a `ds_swizzle`
/// xor 16, then a DPP row mirror) and thread t stores t ^ 31: translated
/// for a host without 64-lane waves it runs as two 32-lane waves and
/// declares `WaveSize(32)`, else `WaveSize(64)`. Without, thread t stores
/// t and the translation runs in waves of any size. The code is the wave64
/// port's own test program (`emit/split_tests.rs`): ported from
/// Marvisak/VIctory#233 (d1fecaa) by r4tedx.
#[cfg(test)]
pub(crate) fn wave64_program(
    in_half: bool,
) -> (xtp_shader::Container<'static>, xtp_shader::NativeProgram) {
    use xtp_shader::native::{ComputeIo, RootParam, StageIo};
    let threads = 64;
    let reads: &[u32] = if in_half {
        &[
            0xD8D4_401F,
            0x0100_0000, // ds_swizzle_b32 v1, v0 offset:swizzle(BITMASK_PERM, and 0x1f, xor 0x10)
            0x7E04_02FA,
            0xFF01_4001, // v_mov_b32_dpp v2, v1 row_mirror row_mask:0xf bank_mask:0xf
        ]
    } else {
        &[0x7E04_0300] // v_mov_b32 v2, v0
    };
    let words = [
        &[
            0xBE82_0300, // s_mov_b32 s2, s0
            0xBE83_0380, // s_mov_b32 s3, 0
            0xF408_0101,
            0xFA00_0000, // s_load_dwordx4 s[4:7], s[2:3], null
            0x9308_FF01,
            threads * 4, // s_mul_i32 s8, s1, threads * 4
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        ][..],
        reads,
        &[
            0x3406_0082, // v_lshlrev_b32 v3, 2, v0
            0x4A06_0608, // v_add_nc_u32 v3, s8, v3
            0xE070_1000,
            0x8001_0203, // buffer_store_dword v2, v3, s[4:7], 0 offen
            0xBF81_0000, // s_endpgm
        ],
    ]
    .concat();
    let native = xtp_shader::NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        float_mode: 0,
        vgprs: 8,
        // A UAV table in user SGPR 0; the group id in s1.
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: Vec::new(),
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1 | 1 << 7,
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let container = xtp_shader::Container {
        digest: [0; 16],
        version: (1, 0),
        parts: Vec::new(),
    };
    (container, native)
}

/// A translation's entry: this, then [`encode`]'s words. 2: with whether it
/// reads the runtime stride table (VIctory #193) and the buffers it reads
/// through slots picked at run time.
/// Version 3 ajoute la base XYZ des dispatches fractionnés.
const MAGIC: &[u8; 4] = b"VSH4";
/// A refusal's entry: this, then the text's length (little endian, 4 bytes),
/// then a checksum of the text (4 bytes), then the error's text (UTF-8). The
/// length tells a cut-short entry from a shorter error, and the checksum
/// tells a changed byte in an entry of the right length: either is damage,
/// and is translated again rather than served with a wrong reason.
const FAILED: &[u8; 4] = b"VSF2";
/// The bytes the payload's length and checksum take, after [`FAILED`].
const LENGTH: usize = 4;
const CHECK: usize = 4;

/// The first four bytes of the text's BLAKE3: enough to call a changed byte
/// damage (this is not against an adversary, only against a torn write).
fn checksum(text: &[u8]) -> [u8; 4] {
    let h = blake3::hash(text);
    h.as_bytes()[..CHECK].try_into().unwrap()
}

fn encode_failure(error: &str) -> Vec<u8> {
    let mut entry = Vec::with_capacity(FAILED.len() + LENGTH + CHECK + error.len());
    entry.extend_from_slice(FAILED);
    entry.extend_from_slice(&(error.len() as u32).to_le_bytes());
    entry.extend_from_slice(&checksum(error.as_bytes()));
    entry.extend_from_slice(error.as_bytes());
    entry
}

fn decode_failure(bytes: &[u8]) -> Option<String> {
    let rest = bytes.strip_prefix(FAILED.as_slice())?;
    let (head, text) = rest.split_at_checked(LENGTH + CHECK)?;
    let length = u32::from_le_bytes(head[..LENGTH].try_into().ok()?) as usize;
    if text.len() != length || checksum(text) != head[LENGTH..] {
        return None;
    }
    String::from_utf8(text.to_vec()).ok()
}
const STAGES: [HwStage; 8] = [
    HwStage::Ls,
    HwStage::Hs,
    HwStage::Es,
    HwStage::Gs,
    HwStage::Vs,
    HwStage::Ps,
    HwStage::Ms,
    HwStage::Cs,
];

fn root_code(r: RootType) -> u32 {
    match r {
        RootType::Constants => 0,
        RootType::Descriptor => 1,
        RootType::Table => 2,
        RootType::ImmediateSampler => 3,
        RootType::ResourceHeap => 4,
        RootType::SamplerHeap => 5,
        RootType::Other(x) => 6 + x,
    }
}

fn root_type(code: u32) -> RootType {
    match code {
        0 => RootType::Constants,
        1 => RootType::Descriptor,
        2 => RootType::Table,
        3 => RootType::ImmediateSampler,
        4 => RootType::ResourceHeap,
        5 => RootType::SamplerHeap,
        x => RootType::Other(x - 6),
    }
}

/// Little-endian words and length-prefixed lists.
fn encode(s: &HostShader) -> Vec<u8> {
    let mut out = MAGIC.to_vec();
    let word = |out: &mut Vec<u8>, v: u32| out.extend(v.to_le_bytes());
    let stage = STAGES.iter().position(|&x| x == s.stage).unwrap_or(0) as u32;
    word(&mut out, stage);
    word(&mut out, s.dxil.len() as u32);
    out.extend(&s.dxil);
    word(&mut out, s.user_data.len() as u32);
    for u in &s.user_data {
        for v in [u.index, u32::from(u.slot), root_code(u.root), u.dword] {
            word(&mut out, v);
        }
    }
    word(&mut out, s.samplers.len() as u32);
    for sampler in &s.samplers {
        for &v in sampler {
            word(&mut out, v);
        }
    }
    word(&mut out, s.static_samplers.len() as u32);
    for t in &s.static_samplers {
        let [u, v, w] = t.address;
        let words = [
            t.filter,
            u,
            v,
            w,
            t.mip_lod_bias.to_bits(),
            t.max_anisotropy,
            t.comparison,
            t.border,
            t.min_lod.to_bits(),
            t.max_lod.to_bits(),
        ];
        for v in words {
            word(&mut out, v);
        }
    }
    for slots in [&s.root_cbvs, &s.root_srvs, &s.root_uavs] {
        word(&mut out, slots.len() as u32);
        out.extend(slots);
    }
    word(&mut out, u32::from(s.stride_table));
    word(&mut out, u32::from(s.run_time_buffers));
    word(&mut out, s.group_base.unwrap_or(u32::MAX));
    out
}

fn decode(bytes: &[u8]) -> Option<HostShader> {
    struct Reader<'a>(&'a [u8]);
    impl<'a> Reader<'a> {
        fn take(&mut self, n: usize) -> Option<&'a [u8]> {
            let (head, rest) = self.0.split_at_checked(n)?;
            self.0 = rest;
            Some(head)
        }
        fn word(&mut self) -> Option<u32> {
            Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
        }
        fn list(&mut self) -> Option<Vec<u8>> {
            let n = self.word()? as usize;
            Some(self.take(n)?.to_vec())
        }
    }
    let mut r = Reader(bytes);
    if r.take(4)? != MAGIC {
        return None;
    }
    let stage = *STAGES.get(r.word()? as usize)?;
    let dxil = r.list()?;
    let n = r.word()? as usize;
    let mut user_data = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        user_data.push(UserData {
            index: r.word()?,
            slot: u8::try_from(r.word()?).ok()?,
            root: root_type(r.word()?),
            dword: r.word()?,
        });
    }
    let n = r.word()? as usize;
    let mut samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        samplers.push([r.word()?, r.word()?, r.word()?, r.word()?]);
    }
    let n = r.word()? as usize;
    let mut static_samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        static_samplers.push(StaticSampler {
            filter: r.word()?,
            address: [r.word()?, r.word()?, r.word()?],
            mip_lod_bias: f32::from_bits(r.word()?),
            max_anisotropy: r.word()?,
            comparison: r.word()?,
            border: r.word()?,
            min_lod: f32::from_bits(r.word()?),
            max_lod: f32::from_bits(r.word()?),
        });
    }
    let root_cbvs = r.list()?;
    let root_srvs = r.list()?;
    let root_uavs = r.list()?;
    let stride_table = match r.word()? {
        0 => false,
        1 => true,
        _ => return None,
    };
    let run_time_buffers = u8::try_from(r.word()?).ok()?;
    let group_base = match r.word()? {
        u32::MAX => None,
        n if n <= 60 && n.is_multiple_of(4) => Some(n),
        _ => return None,
    };
    r.0.is_empty().then_some(HostShader {
        group_base,
        stage,
        dxil,
        user_data,
        samplers,
        static_samplers,
        root_cbvs,
        root_srvs,
        root_uavs,
        stride_table,
        run_time_buffers,
    })
}

/// For tests: held by every test that goes through [`cached`] (here and in
/// `pipeline::host`), so that one reading the process-wide hit and miss
/// counters sees only its own.
#[cfg(test)]
pub(crate) fn serial() -> std::sync::MutexGuard<'static, ()> {
    static CACHE_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());
    CACHE_TESTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The frame line counts shaders ("from the cache and translated so
    /// far"): a refusal, fresh or served from its entry, is neither and
    /// moves no counter; a shader is counted translated once, then from
    /// the cache. Ported from Marvisak/VIctory#1094 (commit e090242) by
    /// adamdotpy.
    #[test]
    fn refusals_are_not_counted_as_translated() {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [9u8; 32];
        let container = b"yet another container that does not translate".as_slice();
        let fail = |_: &[u8]| Err::<HostShader, _>("a scalar memory read as data".to_owned());
        let error = Err("a scalar memory read as data".to_owned());

        let (h0, m0) = counts();
        assert_eq!(cached(&cache, &build, container, fail), error);
        assert_eq!(cached(&cache, &build, container, fail), error, "from disk");
        let (h1, m1) = counts();
        assert_eq!(
            (h1 - h0, m1 - m0),
            (0, 0),
            "a refusal, fresh or kept, is neither a hit nor a translation"
        );

        // Another build, so the kept refusal is not in the way.
        let ok = |_: &[u8]| Ok(shader());
        assert_eq!(cached(&cache, &[10u8; 32], container, ok), Ok(shader()));
        let (h2, m2) = counts();
        assert_eq!((h2 - h1, m2 - m1), (0, 1), "translated once");
        assert_eq!(cached(&cache, &[10u8; 32], container, ok), Ok(shader()));
        let (h3, m3) = counts();
        assert_eq!((h3 - h2, m3 - m2), (1, 0), "then from the cache");
    }

    fn shader() -> HostShader {
        HostShader {
            stage: HwStage::Ps,
            dxil: b"DXBC and the rest".to_vec(),
            user_data: vec![UserData {
                index: 8,
                slot: 1,
                root: RootType::Table,
                dword: 0,
            }],
            samplers: vec![[0x400, 0xc0_0000, 0x18f0_0000, 0]],
            static_samplers: vec![StaticSampler::from_words([
                0x400,
                0xc0_0000,
                0x18f0_0000,
                0,
            ])],
            root_cbvs: vec![2],
            root_srvs: vec![0, 3],
            root_uavs: vec![1],
            stride_table: true,
            run_time_buffers: 3,
            group_base: Some(12),
        }
    }

    #[test]
    fn translations_come_back_from_the_cache() {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let container = b"a console container".as_slice();
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 1, "the second came from disk");

        // Another build translates again.
        assert_eq!(cached(&cache, &[8; 32], container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 2);

        // A damaged entry is a miss, translated and stored again.
        for e in walk(dir.path()) {
            std::fs::write(e, b"VSH1 broken").unwrap();
        }
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 3);
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn failures_come_back_from_the_cache_too() {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let container = b"a container that does not translate".as_slice();
        let runs = AtomicU64::new(0);
        let fail = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Err::<HostShader, _>("a scalar memory read as data is not supported".to_owned())
        };
        let error = Err("a scalar memory read as data is not supported".to_owned());
        assert_eq!(cached(&cache, &build, container, fail), error);
        assert_eq!(cached(&cache, &build, container, fail), error);
        assert_eq!(runs.load(Ordering::Relaxed), 1, "the second came from disk");

        // Another build tries again, and may translate it.
        let ok = |_: &[u8]| Ok(shader());
        assert_eq!(cached(&cache, &[8; 32], container, ok), Ok(shader()));
        assert_eq!(cached(&cache, &build, container, fail), error);
        assert_eq!(runs.load(Ordering::Relaxed), 1);

        // The disk header is valid, but the refusal's own checksum is stale.
        let input = [&build[..], container].concat();
        let converter = Translate::new(fail);
        let entry = xtp_pipeline::CacheKey::new(&converter, &input);
        let mut damaged = encode_failure("unsupported operand");
        *damaged.last_mut().unwrap() ^= 1;
        let frame = [
            b"XTP\x01".as_slice(),
            &(damaged.len() as u64).to_le_bytes(),
            blake3::hash(&damaged).as_bytes(),
            &damaged,
        ]
        .concat();
        std::fs::write(cache.path_for(&entry), frame).unwrap();
        assert_eq!(
            cache.get_or_convert(&converter, &input).unwrap().as_ref(),
            damaged
        );
        assert_eq!(cached(&cache, &build, container, fail), error);
        assert_eq!(runs.load(Ordering::Relaxed), 2);
        assert_eq!(cached(&cache, &build, container, fail), error);
        assert_eq!(
            runs.load(Ordering::Relaxed),
            2,
            "the rebuilt refusal was cached"
        );
    }

    #[test]
    fn failure_entries_round_trip_and_are_no_shader() {
        let bytes = encode_failure("an address not from a root parameter");
        assert_eq!(
            decode_failure(&bytes).as_deref(),
            Some("an address not from a root parameter")
        );
        assert_eq!(decode(&bytes), None);
        assert_eq!(decode_failure(&encode(&shader())), None);
    }

    #[test]
    fn a_damaged_failure_entry_is_no_entry() {
        let full = encode_failure("a scalar memory read as data");
        assert_eq!(
            decode_failure(&full).as_deref(),
            Some("a scalar memory read as data")
        );
        // Truncated after the marker, inside and right after the header
        // (no text at all), and after a valid UTF-8 prefix: all damaged,
        // not an entry with a shorter error, so all are retried.
        for n in [
            FAILED.len(),
            FAILED.len() + 4,
            FAILED.len() + LENGTH + CHECK,
            full.len() - 1,
        ] {
            assert_eq!(decode_failure(&full[..n]), None, "cut to {n} bytes");
        }
        // A byte changed in the text with the length left intact: damage,
        // not an entry with a different (still valid) reason.
        let mut flipped = full.clone();
        let at = FAILED.len() + LENGTH + CHECK + 1;
        assert!(flipped[at].is_ascii());
        flipped[at] = if flipped[at] == b'a' { b'b' } else { b'a' };
        assert_eq!(decode_failure(&flipped), None, "same length, changed byte");
        // Text that is not UTF-8, and a marker followed by text that is not
        // one (a shader entry's magic, say) are no entry either.
        assert_eq!(
            decode_failure(&[FAILED.as_slice(), &[0xff, 0xfe]].concat()),
            None
        );
        assert_eq!(decode_failure(b"VSH1a scalar memory read as data"), None);
        // An entry of the previous layout (no checksum) is no entry: it is
        // translated again rather than read with its text's first bytes as
        // a checksum.
        let old = [b"VSF1".as_slice(), &full[FAILED.len()..]].concat();
        assert_eq!(decode_failure(&old), None);
    }

    /// A cache whose entry cannot be written (here: the entry's path is a
    /// directory, so the rename onto it fails) must not turn a refusal into
    /// an OS error: the reason is the answer, the cache is only a shortcut.
    /// Nor a translation (before the port it did too), and the answer is
    /// the translation already made, not a second one.
    #[test]
    fn a_cache_that_cannot_be_written_still_says_why() {
        let _serial = serial();
        use xtp_pipeline::CacheKey;
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let runs = AtomicU64::new(0);
        let fail = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Err::<HostShader, _>("a scalar memory read as data".to_owned())
        };
        let ok = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        // Block each entry's own path with a directory before the first call.
        let block = |container: &[u8]| {
            let input = [&build[..], container].concat();
            let key = CacheKey::new(&Translate::new(fail), &input);
            std::fs::create_dir_all(cache.path_for(&key)).unwrap();
        };
        let refused = b"a container that does not translate".as_slice();
        block(refused);
        assert_eq!(
            cached(&cache, &build, refused, fail),
            Err("a scalar memory read as data".to_owned()),
            "the reason, not the OS error"
        );
        assert_eq!(runs.load(Ordering::Relaxed), 1, "translated once");
        let translated = b"a container that translates".as_slice();
        block(translated);
        assert_eq!(cached(&cache, &build, translated, ok), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 2, "translated once");
    }

    /// The acceptance test of the port: a refusal is kept under the key a
    /// translation would have, the build and the target. A container is
    /// translated once per pair, then its refusal comes from the cache; a
    /// rebuilt translator or another target translates it again (and may
    /// take it: a 64-lane host runs what a 32-lane one refuses).
    #[test]
    fn refusals_are_kept_by_build_and_target() {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let container = b"a wave64 program that reads across its halves".as_slice();
        let (build, rebuilt) = ([7u8; 32], [8u8; 32]);
        let nvidia = Target {
            lanes_min: 32,
            lanes_max: 32,
            stride_table: false,
        };
        let runs = AtomicU64::new(0);
        // Taken for 64-lane hosts, and by the rebuilt translator for any.
        let get = |build: &[u8; 32], target: Target| {
            let translate = |_: &[u8]| {
                runs.fetch_add(1, Ordering::Relaxed);
                if target.runs(64) || *build == rebuilt {
                    Ok(shader())
                } else {
                    Err("it needs 64-lane waves".to_owned())
                }
            };
            cached(&cache, &key(build, target), container, translate)
        };
        let refused = Err("it needs 64-lane waves".to_owned());
        let runs = || runs.load(Ordering::Relaxed);
        assert_eq!(get(&build, nvidia), refused);
        assert_eq!(get(&build, nvidia), refused);
        assert_eq!(runs(), 1, "the second refusal came from the cache");

        assert_eq!(get(&build, Target::ANY), Ok(shader()), "another target");
        assert_eq!(runs(), 2);
        assert_eq!(get(&rebuilt, nvidia), Ok(shader()), "another build");
        assert_eq!(runs(), 3);

        // Each is its own entry, found again.
        assert_eq!(get(&build, nvidia), refused);
        assert_eq!(get(&build, Target::ANY), Ok(shader()));
        assert_eq!(get(&rebuilt, nvidia), Ok(shader()));
        assert_eq!(runs(), 3);
        assert_eq!(walk(dir.path()).len(), 3);
    }

    /// What `host_shader` keeps: the translator's own refusal, with its
    /// text, translated once.
    #[test]
    fn the_translators_refusal_is_kept_with_its_text() {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let runs = AtomicU64::new(0);
        let translate = |c: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            xtp_shader::emit::translate_for(c, Target::ANY).map_err(|e| e.to_string())
        };
        let container = b"not a shader container".as_slice();
        let refused = Err("not a DXBC container".to_owned());
        for _ in 0..2 {
            let got = cached(&cache, &key(&[7; 32], Target::ANY), container, translate);
            assert_eq!(got, refused);
        }
        assert_eq!(runs.load(Ordering::Relaxed), 1, "the second from the cache");
    }

    /// The cache's entry files: those in its shard folders (its root holds
    /// its index file beside them).
    fn walk(dir: &std::path::Path) -> Vec<PathBuf> {
        let mut files = Vec::new();
        for shard in std::fs::read_dir(dir).unwrap().flatten() {
            if shard.path().is_dir() {
                for e in std::fs::read_dir(shard.path()).unwrap().flatten() {
                    files.push(e.path());
                }
            }
        }
        files
    }

    /// Translations are kept by the wave sizes they were made for: one
    /// container's wave64 program, translated for an NVIDIA host (two
    /// 32-lane waves, `WaveSize(32)`) and for any host (`WaveSize(64)`),
    /// is two entries, each found again by its own target.
    #[cfg(windows)]
    #[test]
    fn translations_are_kept_by_target() {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let container = b"a console container".as_slice();
        let (c, native) = wave64_in_half_program();
        let nvidia = Target {
            lanes_min: 32,
            lanes_max: 32,
            stride_table: false,
        };
        let runs = AtomicU64::new(0);
        let get = |target: Target| {
            let translate = |_: &[u8]| {
                runs.fetch_add(1, Ordering::Relaxed);
                xtp_shader::emit::translate_program_for(&c, &native, target)
                    .map_err(|(_, e)| e.to_string())
            };
            let shader = cached(&cache, &key(&build, target), container, translate).unwrap();
            crate::pipeline::host::Needs::of(&shader.dxil)
                .unwrap()
                .lanes
        };
        assert_eq!(get(nvidia), (32, 32));
        assert_eq!(get(Target::ANY), (64, 64));
        assert_eq!(runs.load(Ordering::Relaxed), 2);
        assert_eq!(get(nvidia), (32, 32), "from its own entry");
        assert_eq!(get(Target::ANY), (64, 64));
        assert_eq!(runs.load(Ordering::Relaxed), 2);

        let rdna = Target {
            lanes_min: 32,
            lanes_max: 64,
            stride_table: false,
        };
        assert_ne!(key(&build, nvidia), key(&build, rdna));
        assert_ne!(key(&build, rdna), key(&build, Target::ANY));
        assert_ne!(key(&build, nvidia), key(&[8; 32], nvidia));
        assert_eq!(key(&build, nvidia), key(&build, nvidia));
    }

    /// The key is the translator's, not the executable's: this run's build
    /// key is made from the fingerprint of `xtp-shader`'s sources, this
    /// file and the settings alone, nothing else of the build. So two
    /// builds that differ only outside the translator share their entries
    /// (`apps/xtp/tests/precook.rs` runs two executables that do), and a
    /// change to a file of the translator, to the encoding or to a setting
    /// translates again.
    #[test]
    fn the_key_is_the_translators_not_the_executables() {
        let _serial = serial();
        let settings = xtp_shader::emit::settings();
        let translator = translator_of(xtp_shader::SOURCES, ENCODING);
        let ours = build_of(&translator, &settings);
        assert_eq!(build(), ours, "nothing but these parts");

        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let container = b"a console container".as_slice();
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        let get = |build: [u8; 32]| cached(&cache, &key(&build, Target::ANY), container, translate);
        let runs = || runs.load(Ordering::Relaxed);
        assert_eq!(get(ours), Ok(shader()));
        // Another build of the same translator: the key is made again from
        // the same parts, and finds the entry.
        let rebuilt = build_of(&translator_of(xtp_shader::SOURCES, ENCODING), &settings);
        assert_eq!(get(rebuilt), Ok(shader()));
        assert_eq!(runs(), 1, "shared by both builds");

        // A file of the translator changed: its fingerprint does
        // (`xtp_shader`'s fingerprint tests), and so does the key.
        let fingerprint = u128::from_str_radix(xtp_shader::SOURCES, 16).unwrap();
        let changed = format!("{:032x}", fingerprint ^ 1);
        let changed = build_of(&translator_of(&changed, ENCODING), &settings);
        assert_eq!(get(changed), Ok(shader()));
        assert_eq!(runs(), 2, "a changed translator translates again");
        let encoding = translator_of(xtp_shader::SOURCES, b"another encoding");
        assert_ne!(build_of(&encoding, &settings), ours);
    }

    /// A translation kept under the key `before` is not served under
    /// `after`: the container is translated again, and from then on each
    /// key finds its own entry.
    fn rejected_across(before: [u8; 32], after: [u8; 32]) {
        assert_ne!(before, after, "two keys");
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let container = b"a container translated under two settings".as_slice();
        let runs = AtomicU64::new(0);
        let get = |key: &[u8; 32], dxil: &[u8]| {
            let translate = |_: &[u8]| {
                runs.fetch_add(1, Ordering::Relaxed);
                Ok(HostShader {
                    dxil: dxil.to_vec(),
                    ..shader()
                })
            };
            cached(&cache, key, container, translate).map(|s| s.dxil)
        };
        let runs = || runs.load(Ordering::Relaxed);
        assert_eq!(get(&before, b"before"), Ok(b"before".to_vec()));
        assert_eq!(get(&before, b"unused"), Ok(b"before".to_vec()));
        assert_eq!(runs(), 1, "kept under its key");
        assert_eq!(
            get(&after, b"after"),
            Ok(b"after".to_vec()),
            "the other setting's entry is not reused"
        );
        assert_eq!(runs(), 2);
        assert_eq!(get(&before, b"unused"), Ok(b"before".to_vec()));
        assert_eq!(get(&after, b"unused"), Ok(b"after".to_vec()));
        assert_eq!(runs(), 2, "each finds its own");
    }

    /// [`rejected_across`] this build's settings and the same with `change`
    /// made, for any target.
    fn a_changed_setting_is_rejected(change: impl FnOnce(&mut xtp_shader::emit::Settings)) {
        let _serial = serial();
        let translator = translator_of(xtp_shader::SOURCES, ENCODING);
        let now = xtp_shader::emit::settings();
        let mut other = now;
        change(&mut other);
        assert_ne!(now, other, "a change");
        rejected_across(
            key(&build_of(&translator, &now), Target::ANY),
            key(&build_of(&translator, &other), Target::ANY),
        );
    }

    /// Another value whatever this run's is (Codex review of d286ffd, point
    /// 6: halving gave 1 again for `XTP_SHADER_LOOP_LIMIT=1`).
    #[test]
    fn the_loop_limit_is_in_the_key() {
        a_changed_setting_is_rejected(|s| s.loop_limit ^= 1);
    }

    #[test]
    fn the_counted_loop_limit_is_in_the_key() {
        a_changed_setting_is_rejected(|s| s.counted_loop_limit ^= 1);
    }

    #[test]
    fn the_spin_limit_is_in_the_key() {
        a_changed_setting_is_rejected(|s| s.spin_limit ^= 1);
    }

    #[test]
    fn the_loop_budget_is_in_the_key() {
        a_changed_setting_is_rejected(|s| s.loop_budget ^= 1);
    }

    #[test]
    fn the_dxil_clean_up_passes_are_in_the_key() {
        a_changed_setting_is_rejected(|s| s.dxil_opt ^= 1);
    }

    #[test]
    fn the_raw_structured_reads_are_in_the_key() {
        a_changed_setting_is_rejected(|s| s.raw_structured = !s.raw_structured);
    }

    #[test]
    fn the_loop_limit_counter_is_in_the_key() {
        a_changed_setting_is_rejected(|s| s.loop_count = !s.loop_count);
    }

    #[test]
    fn the_water_tile_slots_switch_is_in_the_key() {
        a_changed_setting_is_rejected(|s| s.water_tile_slots = !s.water_tile_slots);
    }

    #[test]
    fn self_sized_options_use_a_distinct_cache_key() {
        let _serial = serial();
        let translator = translator_of(xtp_shader::SOURCES, ENCODING);
        let base = key(
            &build_of(&translator, &xtp_shader::emit::settings()),
            Target::ANY,
        );
        assert_eq!(
            option_key(&base, xtp_shader::emit::Options::default()),
            base
        );
        let alt = option_key(
            &base,
            xtp_shader::emit::Options {
                self_sized_loads_via_memory: true,
            },
        );
        rejected_across(base, alt);
    }

    /// The Codex review of 12556d6: `XTP_SHADER_LOOP_LIMIT` unset and set
    /// to 16384 give the same loop limit, and had the same key, but the
    /// counted loops stop at 2^20 in one and 16384 in the other. Now two
    /// keys: a session of one does not get the other's translation.
    #[test]
    fn the_loop_limit_unset_and_set_to_its_default_are_two_keys() {
        use xtp_shader::emit::{
            DEFAULT_COUNTED_LOOP_LIMIT, DEFAULT_LOOP_LIMIT, DEFAULT_SPIN_LIMIT,
        };
        let _serial = serial();
        let now = xtp_shader::emit::settings();
        // What the translator reads from each (`xtp_shader::emit`'s
        // `loop_limit`, `counted_loop_limit` and `spin_limit`).
        let unset = xtp_shader::emit::Settings {
            loop_limit: DEFAULT_LOOP_LIMIT,
            counted_loop_limit: DEFAULT_COUNTED_LOOP_LIMIT,
            spin_limit: DEFAULT_SPIN_LIMIT.min(DEFAULT_LOOP_LIMIT),
            ..now
        };
        let explicit = xtp_shader::emit::Settings {
            loop_limit: 16384,
            counted_loop_limit: 16384,
            spin_limit: DEFAULT_SPIN_LIMIT.min(16384),
            ..now
        };
        assert_eq!(unset.loop_limit, explicit.loop_limit, "the review's case");
        if std::env::var_os("XTP_SHADER_LOOP_LIMIT").is_none() {
            assert_eq!(
                (now.loop_limit, now.counted_loop_limit, now.spin_limit),
                (unset.loop_limit, unset.counted_loop_limit, unset.spin_limit)
            );
        }
        let translator = translator_of(xtp_shader::SOURCES, ENCODING);
        rejected_across(
            key(&build_of(&translator, &unset), Target::ANY),
            key(&build_of(&translator, &explicit), Target::ANY),
        );
    }

    /// The key is read at each translation, as the translator reads its
    /// settings: a switch on for this thread alone (`with_loop_count`)
    /// keys what this thread translates, not what others do.
    #[test]
    fn a_switch_of_this_thread_keys_its_translations() {
        let _serial = serial();
        let translator = translator_of(xtp_shader::SOURCES, ENCODING);
        let counting = xtp_shader::emit::with_loop_count(build);
        let settings = xtp_shader::emit::with_loop_count(xtp_shader::emit::settings);
        assert!(settings.loop_count);
        assert_eq!(counting, build_of(&translator, &settings));
        if !xtp_shader::emit::loop_count() {
            assert_ne!(build(), counting, "off outside it");
        }
    }

    /// [`rejected_across`] two targets, under this build's settings.
    fn another_target_is_rejected(before: Target, after: Target) {
        let _serial = serial();
        let build = build();
        rejected_across(key(&build, before), key(&build, after));
    }

    /// The host's wave sizes key its translations: the fewest lanes, the
    /// most, and whether the host has the runtime stride table.
    #[test]
    fn the_fewest_host_lanes_are_in_the_key() {
        let rdna = Target {
            lanes_min: 32,
            lanes_max: 64,
            stride_table: false,
        };
        another_target_is_rejected(
            rdna,
            Target {
                lanes_min: 64,
                ..rdna
            },
        );
    }

    #[test]
    fn the_most_host_lanes_are_in_the_key() {
        let nvidia = Target {
            lanes_min: 32,
            lanes_max: 32,
            stride_table: false,
        };
        another_target_is_rejected(
            nvidia,
            Target {
                lanes_max: 64,
                ..nvidia
            },
        );
    }

    #[test]
    fn the_stride_table_is_in_the_key() {
        another_target_is_rejected(
            Target::ANY,
            Target {
                stride_table: true,
                ..Target::ANY
            },
        );
    }

    /// After a translator change, a translation made is compared with the
    /// entry the translator before kept for its container: the same bytes,
    /// other bytes (another translation, or a refusal lifted), or none
    /// kept. That entry is only read: the answer is the new translator's,
    /// the entry before stays as it was, and one served from the cache is
    /// not compared again.
    #[test]
    fn translations_are_compared_with_the_translator_befores() {
        let _serial = serial();
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let (before, now) = ([1u8; 32], [2u8; 32]);
        let other = HostShader {
            dxil: b"DXBC and something else".to_vec(),
            ..shader()
        };
        for container in [b"same".as_slice(), b"changed"] {
            assert_eq!(
                cached(&cache, &before, container, |_| Ok(shader())),
                Ok(shader())
            );
        }
        let refused = Err::<HostShader, _>("not supported".to_owned());
        assert_eq!(
            cached(&cache, &before, b"refused", |_| refused.clone()),
            refused
        );

        let start = compared_counts();
        let get = |container: &[u8], answer: &HostShader| {
            compared(&cache, &now, Some(&before), container, |_| {
                Ok(answer.clone())
            })
        };
        assert_eq!(get(b"same", &shader()), Ok(shader()));
        assert_eq!(get(b"changed", &other), Ok(other.clone()), "the answer now");
        assert_eq!(get(b"refused", &shader()), Ok(shader()), "a refusal lifted");
        assert_eq!(get(b"new", &shader()), Ok(shader()), "none kept before");
        assert_eq!(get(b"same", &other), Ok(shader()), "from the cache now");
        let end = compared_counts();
        assert_eq!(
            (
                end.same - start.same,
                end.changed - start.changed,
                end.unknown - start.unknown
            ),
            (1, 2, 1)
        );
        let hash = |c: &[u8]| blake3::hash(c).to_hex()[..16].to_owned();
        assert!(
            end.examples.len() == EXAMPLES
                || end.examples[start.examples.len()..] == [hash(b"changed"), hash(b"refused")],
            "{:?}",
            end.examples
        );
        assert_eq!(
            cached(&cache, &before, b"changed", |_| unreachable!()),
            Ok(shader()),
            "the entry before, as it was"
        );
        assert_eq!(entry(&cache, &[3; 32], b"same"), None, "read only");
        assert_eq!(walk(dir.path()).len(), 7, "nothing kept under the key read");
    }

    #[test]
    fn entries_round_trip_and_reject_damage() {
        let bytes = encode(&shader());
        assert_eq!(decode(&bytes), Some(shader()));
        assert_eq!(decode(&bytes[..bytes.len() - 1]), None);
        assert_eq!(decode(&[bytes.as_slice(), &[0]].concat()), None);
        assert_eq!(decode(b"VSH0"), None);
    }
}
