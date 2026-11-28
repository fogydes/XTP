//! The host adapter the device is made on (docs/analysis/gpu.md, G22). By
//! default the first hardware adapter in DXGI's high-performance order that
//! makes a D3D12 device: a discrete GPU before an integrated one (an RTX
//! 4080 before a Ryzen's iGPU), where DXGI's own default is the adapter
//! driving the main display. `XTP_ADAPTER` picks another: its index in that
//! order, or a part of its name (case ignored). Every adapter is logged at
//! startup, then the one chosen and why.
//!
//! When `D3D12\D3D12Core.dll` (`Microsoft.Direct3D.D3D12`, **exactly
//! 1.614.0**) sits beside the executable, [`crate::agility`] loads it via
//! `ID3D12SDKConfiguration1::CreateDeviceFactory`; otherwise XTP falls back to
//! the Windows inbox D3D12 runtime. Translated shaders need shader model 6.6
//! and up (wave sizes, directly indexed heaps); the host's model is logged
//! with its wave lanes (`device::host_shader_model`).

use windows::Win32::Graphics::Direct3D12::ID3D12Device;
use windows::Win32::Graphics::Dxgi::*;
use windows::core::{IUnknown, Interface};

/// An adapter as the log and the choice see it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Adapter {
    pub(crate) name: String,
    pub(crate) vendor: u32,
    pub(crate) device: u32,
    pub(crate) dedicated_mib: u64,
    pub(crate) software: bool,
    /// Whether it makes a D3D12 device at feature level 12_0.
    pub(crate) usable: bool,
}

/// What `XTP_ADAPTER` asks for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Wanted {
    Index(usize),
    Name(String),
}

impl Wanted {
    pub(crate) fn parse(text: &str) -> Option<Wanted> {
        let text = text.trim();
        if text.is_empty() {
            return None;
        }
        Some(match text.parse() {
            Ok(index) => Wanted::Index(index),
            Err(_) => Wanted::Name(text.to_lowercase()),
        })
    }
}

/// The adapter to use among `adapters` (in preference order) and why:
/// the one `wanted` names if it is usable, else the first usable hardware
/// one; `None` if there is none (DXGI's default then).
pub(crate) fn pick(adapters: &[Adapter], wanted: Option<&Wanted>) -> Option<(usize, String)> {
    if let Some(wanted) = wanted {
        let found = match wanted {
            Wanted::Index(i) => adapters.get(*i).map(|a| (*i, a)),
            Wanted::Name(part) => adapters
                .iter()
                .enumerate()
                .find(|(_, a)| a.name.to_lowercase().contains(part.as_str())),
        };
        match found {
            Some((i, a)) if a.usable => return Some((i, format!("XTP_ADAPTER={wanted:?}"))),
            Some((i, _)) => tracing::warn!(
                index = i,
                "XTP_ADAPTER names an adapter without D3D12 feature level 12_0; using the default"
            ),
            None => tracing::warn!(?wanted, "XTP_ADAPTER names no adapter; using the default"),
        }
    }
    adapters
        .iter()
        .position(|a| a.usable && !a.software)
        .map(|i| (i, "the first high-performance hardware adapter".to_owned()))
}

/// The adapters in high-performance order (DXGI's enumeration order where
/// the system has no preference API), each with the interface to make a
/// device on.
fn adapters() -> windows::core::Result<Vec<(Adapter, IDXGIAdapter1)>> {
    // SAFETY: plain DXGI queries; every description is written by the call.
    unsafe {
        let factory: IDXGIFactory1 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let ordered = factory.cast::<IDXGIFactory6>().ok();
        let mut found = Vec::new();
        for i in 0.. {
            let adapter: windows::core::Result<IDXGIAdapter1> = match &ordered {
                Some(f) => f.EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE),
                None => factory.EnumAdapters1(i),
            };
            let Ok(adapter) = adapter else { break };
            let desc = adapter.GetDesc1()?;
            let len = desc.Description.iter().position(|&c| c == 0).unwrap_or(128);
            let usable = crate::agility::create_device(
                &adapter,
                std::ptr::null_mut::<Option<ID3D12Device>>(),
            )
            .is_ok();
            found.push((
                Adapter {
                    name: String::from_utf16_lossy(&desc.Description[..len]),
                    vendor: desc.VendorId,
                    device: desc.DeviceId,
                    dedicated_mib: desc.DedicatedVideoMemory as u64 >> 20,
                    software: desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
                    usable,
                },
                adapter,
            ));
        }
        Ok(found)
    }
}

/// The adapter to make the host device on (logged), or `None` for DXGI's
/// default.
pub(crate) fn host_adapter() -> Option<IUnknown> {
    let found = match adapters() {
        Ok(found) => found,
        Err(e) => {
            tracing::warn!(error = %e, "host adapters not listed; DXGI's default adapter");
            return None;
        }
    };
    for (i, (a, _)) in found.iter().enumerate() {
        tracing::info!(
            index = i,
            name = a.name,
            vendor = format_args!("{:#06x}", a.vendor),
            device = format_args!("{:#06x}", a.device),
            dedicated_mib = a.dedicated_mib,
            software = a.software,
            d3d12 = a.usable,
            "host adapter"
        );
    }
    let wanted = std::env::var("XTP_ADAPTER")
        .ok()
        .and_then(|t| Wanted::parse(&t));
    let list: Vec<Adapter> = found.iter().map(|(a, _)| a.clone()).collect();
    let Some((i, why)) = pick(&list, wanted.as_ref()) else {
        tracing::warn!("no hardware adapter with D3D12 feature level 12_0; DXGI's default adapter");
        return None;
    };
    let (a, adapter) = &found[i];
    tracing::info!(index = i, name = a.name, "host adapter chosen: {why}");
    adapter.cast().ok()
}

/// The local video memory `device`'s adapter gives this process, and how
/// much of it the process uses, in bytes: `(budget, usage)`. Past the
/// budget, making a resource resident can wait in the kernel for memory.
/// `None` if DXGI cannot tell.
pub(crate) fn video_memory(device: &ID3D12Device) -> Option<(u64, u64)> {
    video_memory_in(device, DXGI_MEMORY_SEGMENT_GROUP_LOCAL)
}

/// As [`video_memory`], for the non-local segment: the system memory the
/// adapter maps for this process (upload and readback heaps, custom heaps
/// in system memory, and card memory the driver moved out).
pub(crate) fn non_local_memory(device: &ID3D12Device) -> Option<(u64, u64)> {
    video_memory_in(device, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL)
}

/// [`video_memory`] for segment group `group`.
fn video_memory_in(device: &ID3D12Device, group: DXGI_MEMORY_SEGMENT_GROUP) -> Option<(u64, u64)> {
    /// The adapter, asked once.
    struct Found(Option<IDXGIAdapter3>);
    // SAFETY: DXGI adapters are free-threaded; this one is only queried.
    unsafe impl Send for Found {}
    // SAFETY: as above.
    unsafe impl Sync for Found {}
    static ADAPTER: std::sync::OnceLock<Found> = std::sync::OnceLock::new();
    let adapter = ADAPTER.get_or_init(|| {
        // SAFETY: plain DXGI queries.
        Found(unsafe {
            let luid = device.GetAdapterLuid();
            CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
                .and_then(|f| f.EnumAdapterByLuid::<IDXGIAdapter3>(luid))
                .ok()
        })
    });
    let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
    // SAFETY: the query writes its own structure.
    unsafe {
        adapter
            .0
            .as_ref()?
            .QueryVideoMemoryInfo(0, group, &mut info)
            .ok()?;
    }
    Some((info.Budget, info.CurrentUsage))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The host device's adapter reports a budget and this process's use.
    #[test]
    fn video_memory_has_a_budget() {
        let test = "dxgi::adapter::tests::video_memory_has_a_budget";
        let Some(gpu) = crate::testgpu::device(test) else {
            return;
        };
        let host = gpu.host();
        let (budget, usage) = video_memory(&host.device).expect("a budget");
        assert!(budget > 0, "{budget}");
        assert!(usage <= 1 << 40, "{usage}");
    }

    fn adapter(name: &str, software: bool, usable: bool) -> Adapter {
        Adapter {
            name: name.into(),
            vendor: 0,
            device: 0,
            dedicated_mib: 0,
            software,
            usable,
        }
    }

    /// The person's PC in high-performance order: the RTX first, then the
    /// iGPU, then WARP.
    fn pc() -> Vec<Adapter> {
        vec![
            adapter("NVIDIA GeForce RTX 4080 SUPER", false, true),
            adapter("AMD Radeon(TM) Graphics", false, true),
            adapter("Microsoft Basic Render Driver", true, true),
        ]
    }

    #[test]
    fn the_default_is_the_first_usable_hardware_adapter() {
        assert_eq!(pick(&pc(), None).map(|p| p.0), Some(0));
        let mut no_d3d12 = pc();
        no_d3d12[0].usable = false;
        assert_eq!(pick(&no_d3d12, None).map(|p| p.0), Some(1));
        let only_warp = vec![adapter("Microsoft Basic Render Driver", true, true)];
        assert_eq!(pick(&only_warp, None), None);
    }

    #[test]
    fn xtp_adapter_names_an_index_or_part_of_a_name() {
        let wanted = |t| Wanted::parse(t);
        assert_eq!(wanted(" 1 "), Some(Wanted::Index(1)));
        assert_eq!(wanted("Radeon"), Some(Wanted::Name("radeon".into())));
        assert_eq!(wanted(""), None);
        let pick = |t| pick(&pc(), wanted(t).as_ref()).map(|p| p.0);
        assert_eq!(pick("1"), Some(1));
        assert_eq!(pick("radeon"), Some(1));
        assert_eq!(pick("basic render"), Some(2), "WARP when asked by name");
        assert_eq!(pick("2"), Some(2));
        // Nothing by that name or index: the default.
        assert_eq!(pick("intel"), Some(0));
        assert_eq!(pick("7"), Some(0));
    }

    #[test]
    fn an_unusable_adapter_asked_for_falls_back() {
        let mut list = pc();
        list[1].usable = false;
        assert_eq!(pick(&list, Some(&Wanted::Index(1))).map(|p| p.0), Some(0));
    }

    /// This machine's adapters list, and the choice is one of them.
    #[test]
    fn adapters_list_on_this_machine() {
        let Ok(found) = adapters() else {
            eprintln!("SKIPPED adapters_list_on_this_machine: no DXGI factory");
            return;
        };
        for (a, _) in &found {
            eprintln!("{a:?}");
        }
        let list: Vec<Adapter> = found.into_iter().map(|(a, _)| a).collect();
        if let Some((i, _)) = pick(&list, None) {
            assert!(list[i].usable && !list[i].software);
        }
    }
}
