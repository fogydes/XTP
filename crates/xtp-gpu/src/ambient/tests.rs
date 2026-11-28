use super::*;
use crate::host::{Host, transition};
use crate::testgpu;
use windows::Win32::Foundation::RECT;

fn settings_with(values: &[(&str, &str)]) -> Settings {
    Settings::read(|name| {
        values
            .iter()
            .find(|(key, _)| *key == name)
            .map(|(_, value)| (*value).to_owned())
    })
}

#[test]
fn cut_and_floor_settings() {
    for text in [
        None,
        Some("0"),
        Some("-2"),
        Some("NaN"),
        Some("inf"),
        Some("bad"),
    ] {
        let settings = Settings::read(|_| text.map(str::to_owned));
        assert_eq!((settings.cut, settings.floor), (0.0, 0.0));
    }
    for ratio in ["0.5", "1.0"] {
        assert_eq!(settings_with(&[("XTP_AMBIENT_CUT", ratio)]).cut, 0.0);
    }
    let settings = settings_with(&[("XTP_AMBIENT_CUT", " 4 "), ("XTP_AMBIENT_FLOOR", "0.75")]);
    assert_eq!((settings.cut, settings.floor), (4.0_f32.ln(), 0.75));
    assert!(!settings.runs(Role::Resolve));
}

#[test]
fn settings_and_role_table() {
    assert_eq!(Mode::parse(""), Mode::Off);
    assert_eq!(Mode::parse("off"), Mode::Off);
    assert_eq!(Mode::parse("0"), Mode::Off);
    assert_eq!(Mode::parse("NaN"), Mode::Off);
    assert_eq!(Mode::parse("inf"), Mode::Off);
    assert_eq!(Mode::parse("0.8"), Mode::Estimate(0.8));
    assert_eq!(Mode::parse("constant: 0.3"), Mode::Constant(0.3));
    assert_eq!(Mode::parse("constant:-1"), Mode::Off);

    assert_eq!(positive("6"), Some(6.0));
    assert_eq!(positive(" 1.5 "), Some(1.5));
    assert_eq!(positive("0"), None);
    assert_eq!(positive("-2"), None);
    assert_eq!(positive("NaN"), None);
    assert!(SOURCE.contains("float lamps;"));
    assert!(SOURCE.contains("dot(c, LUMA) / bound"));

    let known = 0x0123_4567_89ab_cdef;
    let mut roles = HashMap::from([(known, Role::Resolve), (42, Role::Resolve)]);
    let mut invalid = Vec::new();
    override_roles(
        &mut roles,
        concat!(
            "radiance:0123456789ABCDEF, fedcba9876543210, 0x000000000000ff;",
            "air:0000000000000080;bad:0000000000000000;resolve:abc;invalid",
        ),
        |entry| invalid.push(entry.to_owned()),
    );
    assert_eq!(
        roles,
        HashMap::from([
            (known, Role::Radiance),
            (42, Role::Resolve),
            (0xfedc_ba98_7654_3210, Role::Radiance),
            (0x80, Role::Air),
        ])
    );
    assert_eq!(
        invalid,
        ["0x000000000000ff", "bad:0000000000000000", "abc", "invalid"]
    );
}

const WIDTH: u32 = 128;
const HEIGHT: u32 = 72;
const DEPTH: usize = 0;
const LIGHT: usize = 1;
const RADIANCE: usize = 2;
const AIR: usize = 3;

struct Scene {
    visible: ID3D12DescriptorHeap,
    cpu: ID3D12DescriptorHeap,
    increment: usize,
    depth: ID3D12Resource,
    light: ID3D12Resource,
    radiance: ID3D12Resource,
    air: ID3D12Resource,
}

fn committed(
    device: &ID3D12Device,
    desc: &D3D12_RESOURCE_DESC,
    heap: D3D12_HEAP_TYPE,
    state: D3D12_RESOURCE_STATES,
) -> ID3D12Resource {
    let mut resource = None;
    // SAFETY: a plain description, appropriate heap type and an output slot of ours.
    unsafe {
        device.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES {
                Type: heap,
                ..Default::default()
            },
            D3D12_HEAP_FLAG_NONE,
            desc,
            state,
            None,
            &mut resource,
        )
    }
    .unwrap();
    resource.unwrap()
}

impl Scene {
    fn new(device: &ID3D12Device) -> Self {
        let heap = |flags| {
            // SAFETY: plain descriptor heap creation.
            unsafe {
                device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 5,
                    Flags: flags,
                    NodeMask: 0,
                })
            }
            .unwrap()
        };
        let texture = |format, width, height, slices| {
            let dimension = if slices == 1 {
                D3D12_RESOURCE_DIMENSION_TEXTURE2D
            } else {
                D3D12_RESOURCE_DIMENSION_TEXTURE3D
            };
            committed(
                device,
                &D3D12_RESOURCE_DESC {
                    Dimension: dimension,
                    Width: width,
                    Height: height,
                    DepthOrArraySize: slices,
                    MipLevels: 1,
                    Format: format,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    ..Default::default()
                },
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )
        };
        let scene = Self {
            visible: heap(D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE),
            cpu: heap(D3D12_DESCRIPTOR_HEAP_FLAG_NONE),
            // SAFETY: a live device.
            increment: unsafe {
                device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
            } as usize,
            depth: texture(DXGI_FORMAT_R32_FLOAT, WIDTH.into(), HEIGHT, 1),
            light: texture(DXGI_FORMAT_R32_UINT, WIDTH.into(), HEIGHT, 1),
            radiance: texture(DXGI_FORMAT_R11G11B10_FLOAT, WIDTH.into(), HEIGHT, 1),
            air: texture(DXGI_FORMAT_R11G11B10_FLOAT, 8, 8, 2),
        };
        // SAFETY: whole-resource views of these single-mip textures, in the test heaps.
        unsafe {
            device.CreateShaderResourceView(
                &scene.depth,
                None,
                scene.cpu_at(&scene.visible, DEPTH),
            );
            for (index, texture) in [
                (LIGHT, &scene.light),
                (RADIANCE, &scene.radiance),
                (AIR, &scene.air),
                (4, &scene.depth),
            ] {
                for heap in [&scene.cpu, &scene.visible] {
                    device.CreateUnorderedAccessView(
                        texture,
                        None::<&ID3D12Resource>,
                        None,
                        scene.cpu_at(heap, index),
                    );
                }
            }
        }
        scene
    }

    fn cpu_at(&self, heap: &ID3D12DescriptorHeap, index: usize) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        // SAFETY: the heap holds all of these indices.
        let base = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: base.ptr + index * self.increment,
        }
    }

    fn gpu_at(&self, index: usize) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        // SAFETY: the shader-visible heap holds all of these indices.
        let base = unsafe { self.visible.GetGPUDescriptorHandleForHeapStart() };
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: base.ptr + (index * self.increment) as u64,
        }
    }

    fn prepare(&self, list: &HostList) {
        // SAFETY: an open command list, with the two views of the depth UAV
        // and its resource all live. The transition orders the clear before the reads.
        unsafe {
            list.list.SetDescriptorHeaps(&[Some(self.visible.clone())]);
            list.list.ClearUnorderedAccessViewFloat(
                self.gpu_at(4),
                self.cpu_at(&self.cpu, 4),
                &self.depth,
                &[0.05; 4],
                &[],
            );
            list.list.ResourceBarrier(&[transition(
                &self.depth,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            )]);
        }
    }

    fn bound(&self, role: Role) -> Bound {
        let view = |i, resource: &ID3D12Resource| Texture {
            handle: self.gpu_at(i),
            resource: resource.clone(),
        };
        let (image, size) = match role {
            Role::Resolve => (view(LIGHT, &self.light), [WIDTH, HEIGHT, 1]),
            Role::Radiance => (view(RADIANCE, &self.radiance), [WIDTH, HEIGHT, 1]),
            Role::Air => (view(AIR, &self.air), [8, 8, 2]),
        };
        Bound {
            role,
            image,
            size,
            depth: (role == Role::Resolve).then(|| view(DEPTH, &self.depth)),
        }
    }

    fn clear_light(&self, list: &HostList, bits: u32) {
        // SAFETY: the list has the heap set, and the light is a live UAV.
        unsafe {
            list.list.ClearUnorderedAccessViewUint(
                self.gpu_at(LIGHT),
                self.cpu_at(&self.cpu, LIGHT),
                &self.light,
                &[bits; 4],
                &[],
            );
        }
    }

    fn clear_radiance(&self, list: &HostList, value: f32, rects: &[RECT]) {
        // SAFETY: the heap is bound and radiance is a live UAV; order any preceding writes.
        unsafe {
            list.list.ResourceBarrier(&[uav_barrier()]);
            list.list.ClearUnorderedAccessViewFloat(
                self.gpu_at(RADIANCE),
                self.cpu_at(&self.cpu, RADIANCE),
                &self.radiance,
                &[value; 4],
                rects,
            );
        }
    }

    fn split_radiance(&self, list: &HostList) {
        self.clear_radiance(list, 2.0, &[]);
        self.clear_radiance(
            list,
            0.0,
            &[RECT {
                left: 0,
                top: 0,
                right: (WIDTH / 2) as i32,
                bottom: HEIGHT as i32,
            }],
        );
    }

    fn resolve_light(
        &self,
        gpu: &Host,
        list: &HostList,
        passes: &Passes,
        settings: Settings,
    ) -> Readback {
        self.clear_light(list, 0);
        assert!(passes.record(list, settings, &self.bound(Role::Resolve)));
        Readback::copy(gpu, list, &self.light)
    }
}

struct Readback {
    buffer: ID3D12Resource,
    pitch: usize,
    height: usize,
}

impl Readback {
    fn copy(host: &Host, list: &HostList, texture: &ID3D12Resource) -> Self {
        // SAFETY: a live 32-bit texture, in UAV state.
        let desc = unsafe { texture.GetDesc() };
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut bytes = 0;
        // SAFETY: the footprint and byte-count output each have room for one subresource.
        unsafe {
            host.device.GetCopyableFootprints(
                &desc,
                0,
                1,
                0,
                Some(&mut footprint),
                None,
                None,
                Some(&mut bytes),
            );
        }
        let buffer = committed(
            &host.device,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: bytes,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            },
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        let from = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(texture.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        let to = D3D12_TEXTURE_COPY_LOCATION {
            pResource: ManuallyDrop::new(Some(buffer.clone())),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: footprint,
            },
        };
        // SAFETY: both resources are live and the footprint fits the readback buffer.
        unsafe {
            list.list.ResourceBarrier(&[transition(
                texture,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
            list.list.ResourceBarrier(&[transition(
                texture,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
        }
        drop(ManuallyDrop::into_inner(from.pResource));
        drop(ManuallyDrop::into_inner(to.pResource));
        Self {
            buffer,
            pitch: footprint.Footprint.RowPitch as usize,
            height: desc.Height as usize,
        }
    }

    fn word(&self, x: usize, y: usize, z: usize) -> u32 {
        let at = (z * self.height + y) * self.pitch + 4 * x;
        let mut data = std::ptr::null_mut();
        // SAFETY: the submission is finished and the test's coordinates
        // address a copied texel. The CPU only reads the mapped range.
        unsafe {
            let range = D3D12_RANGE {
                Begin: at,
                End: at + 4,
            };
            self.buffer.Map(0, Some(&range), Some(&mut data)).unwrap();
            let bits = data.cast::<u8>().add(at).cast::<u32>().read_unaligned();
            self.buffer
                .Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
            bits
        }
    }
}

fn gray(bits: u32) -> f32 {
    (bits & 511) as f32 * 2.0f32.powi((bits >> 27) as i32 - 24)
}

#[test]
fn estimate_passes_use_radiance_and_bound_light() {
    let Some(gpu) = testgpu::device("ambient estimate") else {
        return;
    };
    let settings = Settings {
        mode: Mode::Estimate(0.8),
        air: 1.5,
        ..settings_with(&[])
    };
    let passes = Passes::make(&gpu.device, settings).unwrap();
    let scene = Scene::new(&gpu.device);
    let list = gpu.new_list().unwrap();
    scene.prepare(&list);
    let resolve = scene.bound(Role::Resolve);
    let keep = scene.bound(Role::Radiance);

    // Uniform RGB9E5 at 32. With no recorded cache it must survive the resolve.
    let bright = (21 << 27) | (256 << 18) | (256 << 9) | 256;
    scene.clear_light(&list, bright);
    assert!(passes.record(&list, settings, &resolve));
    let first = Readback::copy(&gpu, &list, &scene.light);

    scene.clear_radiance(&list, 2.0, &[]);
    assert!(passes.record(&list, settings, &keep));
    assert!(passes.record(&list, settings, &resolve));
    let bounded = Readback::copy(&gpu, &list, &scene.light);

    scene.clear_light(&list, 0);
    assert!(passes.record(&list, settings, &resolve));
    let uniform = Readback::copy(&gpu, &list, &scene.light);

    // The independent in-scattering bound uses the unscaled estimate.
    // SAFETY: the list has the heap set and the AIR slot is a live 3D UAV.
    unsafe {
        list.list.ClearUnorderedAccessViewFloat(
            scene.gpu_at(AIR),
            scene.cpu_at(&scene.cpu, AIR),
            &scene.air,
            &[8.0; 4],
            &[],
        );
    }
    assert!(passes.record(&list, settings, &scene.bound(Role::Air)));
    let air = Readback::copy(&gpu, &list, &scene.air);

    scene.split_radiance(&list);
    assert!(passes.record(&list, settings, &keep));
    scene.clear_light(&list, 0);
    assert!(passes.record(&list, settings, &resolve));
    let split = Readback::copy(&gpu, &list, &scene.light);
    list.close();
    gpu.execute(&[&list]);
    gpu.drain();

    assert_eq!(first.word(0, 0, 0), bright);
    let at = [(0, 0), (4, 36), (64, 36), (123, 36), (127, 71)];
    for (x, y) in at {
        let estimate = gray(uniform.word(x, y, 0));
        assert!(
            (estimate - 1.6).abs() < 0.02,
            "uniform estimate {estimate} at {x},{y}"
        );
        let capped = gray(bounded.word(x, y, 0));
        assert!(
            (capped - 6.4).abs() < 0.04,
            "capped light {capped} at {x},{y}"
        );
    }
    assert!(gray(split.word(4, 36, 0)) + 0.1 < gray(split.word(123, 36, 0)));
    // The R11_FLOAT red channel of 3.0 has exponent 16 and mantissa 32.
    assert_eq!(air.word(1, 1, 1) & 0x7ff, (16 << 6) | 32);
}

#[test]
fn off_leaves_output_and_constant_works_without_history() {
    let Some(gpu) = testgpu::device("ambient off and constant") else {
        return;
    };
    let scene = Scene::new(&gpu.device);
    let list = gpu.new_list().unwrap();
    scene.prepare(&list);
    scene.clear_light(&list, 123456);
    let settings = Settings {
        mode: Mode::Off,
        air: 1.5,
        ..settings_with(&[])
    };
    for role in [Role::Radiance, Role::Resolve, Role::Air] {
        assert!(!settings.record(&list, &scene.bound(role), || panic!(
            "pass initialization while off"
        )));
    }
    let light = Readback::copy(&gpu, &list, &scene.light);
    let constant = Settings {
        mode: Mode::Constant(0.3),
        air: 1.5,
        ..settings_with(&[])
    };
    assert!(!constant.runs(Role::Radiance));
    assert!(!constant.runs(Role::Air));
    let passes = Passes::make(&gpu.device, constant).unwrap();
    assert!(passes.record(&list, constant, &scene.bound(Role::Resolve)));
    let fixed = Readback::copy(&gpu, &list, &scene.light);
    list.close();
    gpu.execute(&[&list]);
    gpu.drain();
    assert_eq!(light.word(64, 36, 0), 123456);
    assert!((gray(fixed.word(64, 36, 0)) - 0.3).abs() < 0.002);
}

#[test]
fn estimate_skips_all_roles_without_r11_typed_uav_load() {
    let Some(gpu) = testgpu::device("ambient estimate without r11") else {
        return;
    };
    let estimate = Settings {
        mode: Mode::Estimate(0.8),
        air: 1.5,
        ..settings_with(&[])
    };
    let passes = Passes::make_with_r11(&gpu.device, estimate, false).unwrap();
    assert!(passes.keep.is_none());
    assert!(passes.air.is_none());

    let scene = Scene::new(&gpu.device);
    let list = gpu.new_list().unwrap();
    scene.prepare(&list);
    scene.clear_light(&list, 123456);

    for role in [Role::Radiance, Role::Resolve, Role::Air] {
        assert!(
            !passes.record(&list, estimate, &scene.bound(role)),
            "estimate role {role:?} must be skipped without R11 typed UAV load"
        );
    }
    let untouched = Readback::copy(&gpu, &list, &scene.light);

    // Constant mode does not rely on `keep` and still lifts light.
    let constant = Settings {
        mode: Mode::Constant(0.3),
        ..settings_with(&[])
    };
    assert!(passes.record(&list, constant, &scene.bound(Role::Resolve)));
    let lifted = Readback::copy(&gpu, &list, &scene.light);

    list.close();
    gpu.execute(&[&list]);
    gpu.drain();

    assert_eq!(untouched.word(64, 36, 0), 123456);
    assert!((gray(lifted.word(64, 36, 0)) - 0.3).abs() < 0.002);
}

#[test]
fn lamp_cap_limits_bright_radiance_relative_to_slow_mean() {
    let Some(gpu) = testgpu::device("ambient lamp cap") else {
        return;
    };
    let settings = Settings {
        mode: Mode::Estimate(0.8),
        lamps: 4.0,
        ..settings_with(&[])
    };
    let passes = Passes::make(&gpu.device, settings).unwrap();
    let scene = Scene::new(&gpu.device);
    let list = gpu.new_list().unwrap();
    scene.prepare(&list);
    let keep = scene.bound(Role::Radiance);
    let resolve = scene.bound(Role::Resolve);

    // Seed the slow frame mean at 0.5, then sample uniform 10.0 radiance.
    // Uncapped it would resolve near 8.0; with a 4x lamp cap (2.0) it stays well under 3.0.
    scene.clear_radiance(&list, 0.5, &[]);
    assert!(passes.record(&list, settings, &keep));
    scene.clear_radiance(&list, 10.0, &[]);
    assert!(passes.record(&list, settings, &keep));
    scene.clear_light(&list, 0);
    assert!(passes.record(&list, settings, &resolve));
    let capped = Readback::copy(&gpu, &list, &scene.light);
    list.close();
    gpu.execute(&[&list]);
    gpu.drain();

    let level = gray(capped.word(64, 36, 0));
    assert!(
        level < 3.0,
        "lamp cap did not bound bright radiance: {level}"
    );
}

#[test]
fn cut_resets_log_luminance_history_in_both_directions() {
    let Some(gpu) = testgpu::device("ambient cut") else {
        return;
    };
    let settings = settings_with(&[("XTP_AMBIENT", "1"), ("XTP_AMBIENT_CUT", "2")]);
    let passes = Passes::make(&gpu.device, settings).unwrap();
    let scene = Scene::new(&gpu.device);
    let list = gpu.new_list().unwrap();
    scene.prepare(&list);
    let keep = scene.bound(Role::Radiance);
    scene.split_radiance(&list);
    assert!(passes.record(&list, settings, &keep));
    let frames = [1.5, 2.0, 0.125].map(|value| {
        scene.clear_radiance(&list, value, &[]);
        assert!(passes.record(&list, settings, &keep));
        scene.resolve_light(&gpu, &list, &passes, settings)
    });
    list.close();
    gpu.execute(&[&list]);
    gpu.drain();

    // The split frame averages 1 arithmetically but under 0.05 geometrically.
    // Only the middle step follows slowly: 80% current + 20% slow RGB mean.
    for (frame, expected) in frames.iter().zip([1.5, 1.904, 0.125]) {
        let actual = gray(frame.word(64, 36, 0));
        assert!(
            (actual - expected).abs() < 0.005,
            "expected {expected}, got {actual}"
        );
    }
}

#[test]
fn floor_lifts_black_samples_relative_to_slow_luminance() {
    let Some(gpu) = testgpu::device("ambient floor") else {
        return;
    };
    let settings = settings_with(&[("XTP_AMBIENT", "0.5"), ("XTP_AMBIENT_FLOOR", "0.75")]);
    let passes = Passes::make(&gpu.device, settings).unwrap();
    let scene = Scene::new(&gpu.device);
    let list = gpu.new_list().unwrap();
    scene.prepare(&list);
    let keep = scene.bound(Role::Radiance);
    scene.clear_radiance(&list, 2.0, &[]);
    assert!(passes.record(&list, settings, &keep));
    let bright = scene.resolve_light(&gpu, &list, &passes, settings);
    scene.clear_radiance(&list, 0.0, &[]);
    assert!(passes.record(&list, settings, &keep));
    let baseline = scene.resolve_light(
        &gpu,
        &list,
        &passes,
        Settings {
            floor: 0.0,
            ..settings
        },
    );
    let lifted = scene.resolve_light(&gpu, &list, &passes, settings);
    list.close();
    gpu.execute(&[&list]);
    gpu.drain();

    let slow = (0.96 * 2.001_f32.ln() + 0.04 * 0.001_f32.ln()).exp() - 0.001;
    assert!((gray(bright.word(64, 36, 0)) - 1.0).abs() < 0.005);
    assert!((gray(baseline.word(64, 36, 0)) - 0.192).abs() < 0.005);
    assert!((gray(lifted.word(64, 36, 0)) - 0.75 * slow * 0.5).abs() < 0.005);
}
