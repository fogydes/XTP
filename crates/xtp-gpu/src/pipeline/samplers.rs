//! Deterministic static sampler table order across runs.
//!
//! Some D3D12 drivers (notably NVIDIA) keep static samplers in an internal
//! hardware table, assigning each distinct sampler a slot when the first root
//! signature using it is created, and baking those slots into compiled
//! pipelines. Because worker threads reach root signatures in a different
//! order from run to run (and from `xtp precook`), pipelines with static
//! samplers miss the driver's on-disk cache and recompile at startup.
//!
//! Distinct static sampler states are stored in `static-samplers.txt` beside
//! the host pipeline library in order of first appearance. At device creation
//! ([`prime`]), each listed sampler gets a kept single-sampler root signature
//! in that exact order so its slot stays pinned. Unseen samplers ([`before`])
//! are created the same way and appended under an exclusive file lock.
//!
//! `XTP_SAMPLER_ORDER=0`: disable deterministic static sampler ordering.

use std::collections::HashSet;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::{Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant};

use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;

const FILE: &str = "static-samplers.txt";
/// Cap below NVIDIA's 2,032-entry hardware static sampler table limit.
const MOST: usize = 1024;

/// Sampler state words identifying a static sampler in the driver's table:
/// filter, U/V/W address modes, mip LOD bias, max anisotropy, comparison,
/// border color, min LOD, max LOD.
type Key = [u32; 10];

fn key(s: &D3D12_STATIC_SAMPLER_DESC) -> Key {
    [
        s.Filter.0 as u32,
        s.AddressU.0 as u32,
        s.AddressV.0 as u32,
        s.AddressW.0 as u32,
        s.MipLODBias.to_bits(),
        s.MaxAnisotropy,
        s.ComparisonFunc.0 as u32,
        s.BorderColor.0 as u32,
        s.MinLOD.to_bits(),
        s.MaxLOD.to_bits(),
    ]
}

fn description(k: &Key) -> D3D12_STATIC_SAMPLER_DESC {
    D3D12_STATIC_SAMPLER_DESC {
        Filter: D3D12_FILTER(k[0] as i32),
        AddressU: D3D12_TEXTURE_ADDRESS_MODE(k[1] as i32),
        AddressV: D3D12_TEXTURE_ADDRESS_MODE(k[2] as i32),
        AddressW: D3D12_TEXTURE_ADDRESS_MODE(k[3] as i32),
        MipLODBias: f32::from_bits(k[4]),
        MaxAnisotropy: k[5],
        ComparisonFunc: D3D12_COMPARISON_FUNC(k[6] as i32),
        BorderColor: D3D12_STATIC_BORDER_COLOR(k[7] as i32),
        MinLOD: f32::from_bits(k[8]),
        MaxLOD: f32::from_bits(k[9]),
        ShaderRegister: 0,
        RegisterSpace: 0,
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    }
}

fn line(k: &Key) -> String {
    let [a, b, c, d, e, f, g, h, i, j] = *k;
    format!("{a:x} {b:x} {c:x} {d:x} {e:x} {f:x} {g:x} {h:x} {i:x} {j:x}")
}

/// Parses one line of `static-samplers.txt`, rejecting truncated or extra-token lines.
fn parse(line: &str) -> Option<Key> {
    let mut k = [0u32; 10];
    let mut words = line.split_ascii_whitespace();
    for w in &mut k {
        *w = u32::from_str_radix(words.next()?, 16).ok()?;
    }
    words.next().is_none().then_some(k)
}

#[derive(Default)]
struct Listed {
    order: Vec<Key>,
    known: HashSet<Key>,
}

impl Listed {
    fn place(&mut self, k: Key) -> Option<usize> {
        if self.order.len() >= MOST || !self.known.insert(k) {
            return None;
        }
        self.order.push(k);
        Some(self.order.len() - 1)
    }
}

struct Kept(#[allow(dead_code)] ID3D12RootSignature);

// SAFETY: D3D12 root signatures are free-threaded; this handle is only kept alive.
unsafe impl Send for Kept {}

#[derive(Default)]
struct State {
    listed: Listed,
    kept: Vec<Kept>,
    overflow_logged: bool,
}

static SEEN: RwLock<Option<HashSet<Key>>> = RwLock::new(None);
static STATE: Mutex<Option<State>> = Mutex::new(None);

fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("XTP_SAMPLER_ORDER").is_none_or(|v| v != "0"))
}

/// Opens `static-samplers.txt` in `dir` exclusively so concurrent processes
/// serialize updates. Refuses paths inside any XTP repository.
fn open_alone(dir: &Path) -> Option<std::fs::File> {
    use std::os::windows::fs::OpenOptionsExt;
    let path = dir.join(FILE);
    if let Err(e) = xtp_core::ensure_outside_repo(&path) {
        tracing::warn!(error = %e, "refusing to store static sampler order inside the repository");
        return None;
    }
    if let Err(e) = std::fs::create_dir_all(dir) {
        tracing::warn!(error = %e, "could not create folder for static sampler order");
        return None;
    }
    let began = Instant::now();
    loop {
        let opened = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .share_mode(0)
            .open(&path);
        match opened {
            Ok(file) => return Some(file),
            Err(e)
                if matches!(e.raw_os_error(), Some(32 | 33))
                    && began.elapsed() < Duration::from_secs(10) =>
            {
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(e) => {
                tracing::warn!(error = %e, "static sampler list could not be opened");
                return None;
            }
        }
    }
}

/// Creates a root signature containing only the single static sampler `k`.
fn root_of(device: &ID3D12Device, k: &Key) -> windows::core::Result<ID3D12RootSignature> {
    let sampler = description(k);
    let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
        Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
        Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
            Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                NumStaticSamplers: 1,
                pStaticSamplers: &sampler,
                ..Default::default()
            },
        },
    };
    // SAFETY: `desc` points at `sampler`, which lives through serialization;
    // `bytes` borrows the serialized blob for `CreateRootSignature`.
    unsafe {
        let mut blob: Option<ID3DBlob> = None;
        D3D12SerializeVersionedRootSignature(&desc, &mut blob, None)?;
        let blob = blob.ok_or_else(windows::core::Error::empty)?;
        let bytes =
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
        device.CreateRootSignature(0, bytes)
    }
}

fn sync_in(device: &ID3D12Device, dir: Option<&Path>, mine: &[Key], state: &mut State) -> usize {
    let mut file = dir.and_then(open_alone);
    let mut listed = String::new();
    if let Some(file) = &mut file {
        let _ = file.read_to_string(&mut listed);
    }
    let mut to_create = Vec::new();
    for k in listed.lines().filter_map(parse) {
        if state.listed.place(k).is_some() {
            to_create.push(k);
        }
    }
    let mut added = String::new();
    for &k in mine {
        if state.listed.known.contains(&k) {
            continue;
        }
        if state.listed.place(k).is_some() {
            to_create.push(k);
            added.push_str(&line(&k));
            added.push('\n');
        } else {
            state.listed.known.insert(k);
            if !state.overflow_logged {
                state.overflow_logged = true;
                tracing::warn!(most = MOST, "static sampler order cap reached");
            }
        }
    }
    for k in &to_create {
        match root_of(device, k) {
            Ok(root) => state.kept.push(Kept(root)),
            Err(e) => {
                tracing::warn!(error = %e, sampler = %line(k), "static sampler root signature refused");
            }
        }
    }
    if let Some(file) = &mut file
        && !added.is_empty()
    {
        if !listed.is_empty() && !listed.ends_with('\n') {
            added.insert(0, '\n');
        }
        let wrote = file
            .seek(SeekFrom::End(0))
            .and_then(|_| file.write_all(added.as_bytes()));
        if let Err(e) = wrote {
            tracing::warn!(error = %e, "static sampler list was not written");
        }
    }
    state.listed.order.len()
}

fn sync(device: &ID3D12Device, mine: &[Key]) -> usize {
    let mut guard = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let state = guard.get_or_insert_with(State::default);
    let count = sync_in(device, super::library::dir(), mine, state);
    let mut seen = SEEN.write().unwrap_or_else(|e| e.into_inner());
    seen.get_or_insert_default()
        .extend(state.listed.known.iter().copied());
    count
}

/// Primes every previously recorded static sampler in saved order right after
/// device creation, before any root signature with static samplers is built.
pub(crate) fn prime(device: &ID3D12Device) {
    if !on() {
        return;
    }
    let began = Instant::now();
    let samplers = sync(device, &[]);
    tracing::info!(
        samplers,
        ms = began.elapsed().as_secs_f64() * 1e3,
        "static samplers primed in deterministic order (XTP_SAMPLER_ORDER=0: off)"
    );
}

/// Ensures every static sampler in `samplers` holds its deterministic slot
/// before a pipeline root signature referencing them is created.
pub(crate) fn before(device: &ID3D12Device, samplers: &[D3D12_STATIC_SAMPLER_DESC]) {
    if samplers.is_empty() || !on() {
        return;
    }
    let keys: Vec<Key> = samplers.iter().map(key).collect();
    {
        let seen = SEEN.read().unwrap_or_else(|e| e.into_inner());
        if seen
            .as_ref()
            .is_some_and(|s| keys.iter().all(|k| s.contains(k)))
        {
            return;
        }
    }
    sync(device, &keys);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sampler(filter: i32, register: u32) -> D3D12_STATIC_SAMPLER_DESC {
        D3D12_STATIC_SAMPLER_DESC {
            Filter: D3D12_FILTER(filter),
            AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressV: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
            AddressW: D3D12_TEXTURE_ADDRESS_MODE_MIRROR,
            MipLODBias: -0.75,
            MaxAnisotropy: 8,
            ComparisonFunc: D3D12_COMPARISON_FUNC_LESS_EQUAL,
            BorderColor: D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE,
            MinLOD: 0.5,
            MaxLOD: f32::MAX,
            ShaderRegister: register,
            RegisterSpace: 3,
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        }
    }

    #[test]
    fn sampler_round_trips_and_rejects_bad_lines() {
        let k = key(&sampler(0x15, 4));
        let whole = line(&k);
        assert_eq!(parse(&whole), Some(k));
        let back = description(&k);
        assert_eq!(key(&back), k);
        assert_eq!((back.MipLODBias, back.MaxLOD), (-0.75, f32::MAX));
        assert_eq!(key(&sampler(0x15, 9)), k);
        assert_ne!(parse(&whole[..whole.len() - 3]), Some(k));
        assert_eq!(parse("1 2 3"), None);
        assert_eq!(parse(&format!("{whole} 7")), None);
        assert_eq!(parse("zz 0 0 0 0 0 0 0 0 0"), None);
        assert_eq!(parse(""), None);
    }

    #[test]
    fn order_is_first_seen_and_capped_at_most() {
        let (a, b, c) = (
            key(&sampler(0, 0)),
            key(&sampler(0x15, 0)),
            key(&sampler(0x55, 0)),
        );
        let mut listed = Listed::default();
        assert_eq!(listed.place(a), Some(0));
        assert_eq!(listed.place(b), Some(1));
        assert_eq!(listed.place(a), None);
        assert_eq!(listed.place(c), Some(2));
        assert_eq!(listed.order, [a, b, c]);

        let mut capped = Listed::default();
        for n in 0..MOST as u32 + 5 {
            let mut s = sampler(0x15, 0);
            s.MinLOD = n as f32;
            capped.place(key(&s));
        }
        assert_eq!(capped.order.len(), MOST);
    }

    #[test]
    fn refuses_to_store_list_inside_repo() {
        let inside = Path::new(env!("CARGO_MANIFEST_DIR"));
        assert!(open_alone(inside).is_none());
        assert!(!inside.join(FILE).exists());
    }

    #[test]
    fn persists_and_primes_root_signatures_on_warp() {
        let Some(gpu) = crate::testgpu::device("samplers::persists_and_primes_root_signatures")
        else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let mut compare = sampler(0x95, 2);
        compare.ComparisonFunc = D3D12_COMPARISON_FUNC_GREATER;
        let (a, b, c) = (key(&sampler(0, 0)), key(&sampler(0x15, 1)), key(&compare));

        let mut run1 = State::default();
        assert_eq!(
            sync_in(&gpu.host().device, Some(dir.path()), &[a, b], &mut run1),
            2
        );
        assert_eq!(run1.kept.len(), 2);

        // Next run primes a, b in saved order even when meeting c, b, a first.
        let mut run2 = State::default();
        assert_eq!(
            sync_in(&gpu.host().device, Some(dir.path()), &[], &mut run2),
            2
        );
        assert_eq!(run2.listed.order, [a, b]);
        assert_eq!(
            sync_in(&gpu.host().device, Some(dir.path()), &[c, b, a], &mut run2),
            3
        );
        assert_eq!(run2.listed.order, [a, b, c]);
        assert_eq!(run2.kept.len(), 3);
    }
}
