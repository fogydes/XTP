use super::*;
use crate::testgpu::{Kernel, Pass, TitleBuffer, TitleList};

fn geometry(vertices: &TitleBuffer, indices: &TitleBuffer) -> Geometry {
    Geometry {
        Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
        Flags: D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
        Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
            Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                Transform3x4: 0,
                IndexFormat: DXGI_FORMAT_R16_UINT,
                VertexFormat: DXGI_FORMAT_R32G32B32_FLOAT,
                IndexCount: 3,
                VertexCount: 3,
                IndexBuffer: indices.address(),
                VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                    StartAddress: vertices.address(),
                    StrideInBytes: 12,
                },
            },
        },
    }
}

fn blas_input(geometry: &Geometry) -> Inputs {
    Inputs {
        Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
        Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
        NumDescs: 1,
        DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
        Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
            pGeometryDescs: geometry,
        },
    }
}

/// A title TLAS build at `dst` of the one instance at `instances`.
fn tlas_desc(dst: u64, instances: &TitleBuffer) -> Build {
    Build {
        DestAccelerationStructureData: dst,
        Inputs: Inputs {
            Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
            NumDescs: 1,
            Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 {
                InstanceDescs: instances.address(),
            },
            ..Default::default()
        },
        ..Default::default()
    }
}

/// The triangle (-1, -1), (0, 1), (1, -1) at z = 0: the ray of thread 0
/// of `ray_test.hlsl` hits it at two units, the ray of thread 1 misses.
const POSITIONS: [f32; 9] = [-1., -1., 0., 0., 1., 0., 1., -1., 0.];
/// What `ray_test.hlsl` writes when thread 0's ray hits at two units.
fn hit() -> [u32; 4] {
    [1, 2.0f32.to_bits(), 0, 0]
}
/// What it writes when both rays miss: an empty TLAS.
const MISSES: [u32; 4] = [0; 4];
/// What an output holds before any trace writes it.
const UNWRITTEN: [u32; 4] = [77; 4];

fn unwritten() -> TitleBuffer {
    let output = TitleBuffer::new(16);
    output.set_words(|_| 77);
    output
}

/// The host device for `test`, if it builds DXR structures; `rays` is
/// whether it also runs inline ray queries (DXR tier 1.1). A missing tier
/// is reported as a skip (a failure with `XTP_REQUIRE_GPU`), never silent.
fn dxr(test: &str) -> Option<(crate::testgpu::Gpu, bool)> {
    let gpu = crate::testgpu::device(test)?;
    let mut caps = D3D12_FEATURE_DATA_D3D12_OPTIONS5::default();
    // SAFETY: the output is the options structure selected by the feature identifier.
    let tier = unsafe {
        gpu.host().device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS5,
            (&raw mut caps).cast(),
            size_of_val(&caps) as u32,
        )
    }
    .map_or(D3D12_RAYTRACING_TIER_NOT_SUPPORTED, |()| {
        caps.RaytracingTier
    });
    // The DXR device this module records with.
    let builds = device(gpu.host());
    if tier == D3D12_RAYTRACING_TIER_NOT_SUPPORTED || builds.is_none() {
        crate::testgpu::skip(test, "the host device has no DXR");
        return None;
    }
    let rays = tier.0 >= D3D12_RAYTRACING_TIER_1_1.0;
    if !rays {
        crate::testgpu::skip(
            test,
            "the traces: inline ray queries require DXR tier 1.1 (the builds still run)",
        );
    }
    Some((gpu, rays))
}

/// A one-instance TLAS of the test triangle's BLAS, built and submitted.
struct World {
    tlas: TitleBuffer,
    blas: TitleBuffer,
    _inputs: [TitleBuffer; 3],
}

impl World {
    fn new() -> World {
        let (vertices, indices) = (TitleBuffer::new(36), TitleBuffer::new(8));
        vertices.set_words(|i| POSITIONS[i].to_bits());
        indices.set_words(|i| [1 << 16, 2][i]);
        let blas = TitleBuffer::new(1024);
        let tlas = TitleBuffer::new(1024);
        let instances = TitleBuffer::new(64);
        instances.set_words(|i| match i {
            0 | 5 | 10 => 1f32.to_bits(),
            12 => 0xff00_0000,
            14 => blas.address() as u32,
            15 => (blas.address() >> 32) as u32,
            _ => 0,
        });
        let producer = TitleList::new();
        let geometry = geometry(&vertices, &indices);
        // SAFETY: the title list and all CPU arrays are live for each call.
        unsafe {
            assert!(Builder.build(
                producer.this(),
                &Build {
                    DestAccelerationStructureData: blas.address(),
                    Inputs: blas_input(&geometry),
                    ..Default::default()
                }
            ));
            assert!(Builder.build(producer.this(), &tlas_desc(tlas.address(), &instances)));
        }
        producer.run();
        World {
            tlas,
            blas,
            _inputs: [vertices, indices, instances],
        }
    }
}

/// A title descriptor slot of a raytracing SRV naming title `address`.
struct Descriptor([u64; 4]);

impl Descriptor {
    fn new(address: u64) -> Box<Descriptor> {
        let mut slot = Box::new(Descriptor([0; 4]));
        let mut srv = [0u32; 10];
        srv[1] = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE.0 as u32;
        srv[4] = address as u32;
        srv[5] = (address >> 32) as u32;
        let handle = slot.0.as_mut_ptr() as u64;
        crate::descriptor::create_srv(std::ptr::null(), std::ptr::null(), &srv, handle);
        slot
    }

    fn handle(&self) -> u64 {
        self.0.as_ptr() as u64
    }
}

enum Scene {
    /// A TLAS host address, bound directly.
    Address(u64),
    /// A title descriptor slot, bound in submission order.
    Descriptor(u64),
    /// The same, dispatched with the host list alone, as draws do.
    HostDescriptor(u64),
    /// The same, with the root signature a host pipeline makes for a
    /// RayQuery splice: the TLAS at t240 among bounded root SRVs.
    Spliced(u64),
    /// A descriptor slot, with group counts at a title address the GPU
    /// wrote (and the command count at another, if set).
    Indirect(u64, u64, Option<u64>),
}

/// `ray_test.hlsl` compiled by the Windows SDK's DXC, with `defines`.
fn ray_program(defines: &[&str]) -> Vec<u8> {
    compiled("ray_test.hlsl", defines)
}

/// `file` of this folder compiled by the Windows SDK's DXC, with `defines`.
fn compiled(file: &str, defines: &[&str]) -> Vec<u8> {
    let dxc = std::env::var_os("XTP_PC_DXC")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            xtp_shader::emit::validator::Validator::default_path()
                .map(|p| p.with_file_name("dxc.exe"))
        })
        .expect("Windows SDK DXC for the inline ray test");
    let dir = tempfile::tempdir().unwrap();
    let binary = dir.path().join("ray.dxil");
    let compiled = std::process::Command::new(dxc)
        .args(["-T", "cs_6_5", "-E", "main", "-Fo"])
        .arg(&binary)
        .args(defines.iter().flat_map(|d| ["-D", d]))
        .arg(format!("{}/src/rt/{file}", env!("CARGO_MANIFEST_DIR")))
        .output()
        .unwrap();
    assert!(
        compiled.status.success(),
        "{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    std::fs::read(binary).unwrap()
}

/// The root signature of a spliced program reading the TLAS, with its
/// output at `u0`: the TLAS parameter is the host pipeline's own
/// ([`crate::pipeline::host::root_view`]). Its index and the signature.
fn spliced_root(host: &Host) -> (u32, ID3D12RootSignature) {
    use crate::pipeline::host::{bounded_root_srvs, root_srv_in_table, root_view};
    use xtp_shader::emit::{ROOT_CBV_SPACE, TLAS_REGISTER};
    // ray_test.hlsl names them: t240, space 1.
    assert_eq!((TLAS_REGISTER, ROOT_CBV_SPACE), (240, 1));
    assert!(bounded_root_srvs(), "XTP_ROOT_SRV_DIRECT is not set");
    assert!(root_srv_in_table(3) && !root_srv_in_table(TLAS_REGISTER));
    let mut ranges = Vec::new();
    let all = D3D12_SHADER_VISIBILITY_ALL;
    let other = root_view(D3D12_ROOT_PARAMETER_TYPE_SRV, 3, all, &mut ranges);
    assert_eq!(
        other.ParameterType,
        D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE
    );
    let tlas = root_view(
        D3D12_ROOT_PARAMETER_TYPE_SRV,
        TLAS_REGISTER,
        all,
        &mut ranges,
    );
    assert_eq!(tlas.ParameterType, D3D12_ROOT_PARAMETER_TYPE_SRV);
    let output = D3D12_ROOT_PARAMETER1 {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
        Anonymous: D3D12_ROOT_PARAMETER1_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR1::default(),
        },
        ShaderVisibility: all,
    };
    let (root, _) =
        crate::pipeline::host::root_signature(&host.device, &[tlas, output], &[], true).unwrap();
    (0, root)
}

fn trace(host: &Host, title_list: &TitleList, tlas: Scene, output: &TitleBuffer) {
    let spliced = matches!(tlas, Scene::Spliced(_));
    let code = ray_program(if spliced { &["SPLICED"] } else { &[] });
    trace_program(host, title_list, tlas, output, &code);
}

/// [`trace`] with the compiled ray program `code`, whose output is all of
/// `output`.
fn trace_program(
    host: &Host,
    title_list: &TitleList,
    tlas: Scene,
    output: &TitleBuffer,
    code: &[u8],
) {
    let list = title_list.host();
    let spliced = matches!(tlas, Scene::Spliced(_));
    let bytes = 4 * output.words().len() as u64;
    // The commands of an indirect trace come before its PSO and root views.
    let prepared = match tlas {
        // SAFETY: the test's live list and descriptor slot.
        Scene::Indirect(handle, arguments, count) => Some(
            unsafe { prepare_indirect_tlas(title_list.this(), handle, (arguments, 12, 1), count) }
                .expect("indirect trace commands"),
        ),
        _ => None,
    };
    let shadow = output.shadow();
    shadow.refresh(list, output.address(), bytes);
    shadow.gpu_writes(list, output.address(), bytes);
    // SAFETY: the root signature is embedded by DXC or made here. The list keeps
    // both objects and the output shadow, and the caller retains the TLAS through
    // completion.
    unsafe {
        let (param, root) = if spliced {
            crate::testgpu::bind_indexed_heaps(list);
            spliced_root(host)
        } else {
            (0, host.device.CreateRootSignature(0, code).unwrap())
        };
        let pso: ID3D12PipelineState = host
            .device
            .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: code.as_ptr().cast(),
                    BytecodeLength: code.len(),
                },
                ..Default::default()
            })
            .unwrap();
        list.keep(&root);
        list.keep(&pso);
        list.list.SetPipelineState(&pso);
        list.list.SetComputeRootSignature(&root);
        list.list
            .SetComputeRootUnorderedAccessView(1, shadow.gpu(output.address()));
        match tlas {
            Scene::Address(address) => {
                list.list.SetComputeRootShaderResourceView(0, address);
                list.list.Dispatch(1, 1, 1);
            }
            Scene::Descriptor(handle) | Scene::Spliced(handle) => assert!(
                dispatch_descriptor_tlas(title_list.this(), handle, &root, param, [1, 1, 1])
            ),
            Scene::HostDescriptor(handle) => assert!(dispatch_descriptor_tlas_in(
                list,
                handle,
                &root,
                param,
                [1, 1, 1]
            )),
            Scene::Indirect(..) => assert!(dispatch_indirect_tlas(
                title_list.this(),
                prepared.as_ref().unwrap(),
                &root,
                param
            )),
        }
        list.list.ResourceBarrier(&[uav_barrier()]);
    }
}

#[test]
fn rt_triangle_translation_requires_the_whole_input_range() {
    let Some(_gpu) =
        crate::testgpu::device("rt_triangle_translation_requires_the_whole_input_range")
    else {
        return;
    };
    let list = TitleList::new();
    let (vertices, indices) = (TitleBuffer::new(36), TitleBuffer::new(6));
    let mut geometry = geometry(&vertices, &indices);
    // SAFETY: the test owns the descriptor and all referenced buffers.
    unsafe {
        let (translated, jobs) = triangles(list.host(), &blas_input(&geometry)).unwrap();
        let t = translated[0].Anonymous.Triangles;
        // Set when the pass that makes them is recorded (`indices`).
        assert_eq!((t.VertexBuffer.StartAddress, t.IndexBuffer), (0, 0));
        let [made_vertices, made_indices] = &jobs[..] else {
            panic!("{} jobs", jobs.len());
        };
        assert_eq!(
            (
                made_vertices.input,
                made_vertices.kind,
                made_vertices.count,
                made_vertices.source
            ),
            (
                indices::Input::Vertices {
                    format: DXGI_FORMAT_R32G32B32_FLOAT,
                    stride: 12,
                    skew: 0
                },
                indices::Kind::Words,
                9,
                vertices.shadow().gpu(vertices.address())
            )
        );
        assert_eq!(
            (
                made_indices.input,
                made_indices.kind,
                made_indices.count,
                made_indices.vertices,
                made_indices.source
            ),
            (
                indices::Input::Indices,
                indices::Kind::Narrow,
                3,
                3,
                indices.shadow().gpu(indices.address())
            )
        );
        geometry.Anonymous.Triangles.IndexCount = 4;
        assert!(triangles(list.host(), &blas_input(&geometry)).is_none());
        geometry.Anonymous.Triangles.IndexCount = 3;
        geometry.Anonymous.Triangles.VertexBuffer.StartAddress += 4;
        assert!(triangles(list.host(), &blas_input(&geometry)).is_none());
    }
}

#[test]
fn rt_ray_hits_the_cloned_blas_at_two_units() {
    let test = "rt_ray_hits_the_cloned_blas_at_two_units";
    let Some((gpu, rays)) = dxr(test) else {
        return;
    };
    let host = gpu.host();
    let list = TitleList::new();
    let (vertices, indices) = (TitleBuffer::new(36), TitleBuffer::new(8));
    vertices.set_words(|i| POSITIONS[i].to_bits());
    indices.set_words(|i| [1 << 16, 2][i]);
    let output = TitleBuffer::new(16);
    output.set_words(|_| u32::MAX);
    let geometry = geometry(&vertices, &indices);
    let blas = TitleBuffer::new(1024);
    let cloning = TitleList::new();
    let compacted = TitleBuffer::new(1024);
    let cloned = TitleBuffer::new(1024);
    let first = TitleBuffer::new(1024);
    let second = TitleBuffer::new(1024);
    let instances = TitleBuffer::new(64);
    instances.set_words(|i| match i {
        0 | 5 | 10 => 1f32.to_bits(),
        12 => 0xff00_0000,
        _ => 0,
    });
    let address = TitleBuffer::new(8);
    address.set_words(|i| (cloned.address() >> (32 * i)) as u32);
    // This list writes the BLAS address on the GPU. The CPU instance still points to zero.
    instances.shadow();
    assert!(crate::shadow::copy(
        list.host(),
        instances.address() + 56,
        address.address(),
        8
    ));
    let future = TitleList::new();
    let too_early = TitleList::new();
    let delayed_hit = TitleBuffer::new(16);
    let missed = unwritten();
    let slot = Descriptor::new(first.address());
    if rays {
        trace(
            host,
            &future,
            Scene::Descriptor(slot.handle()),
            &delayed_hit,
        );
        trace(host, &too_early, Scene::Descriptor(slot.handle()), &missed);
        too_early.run();
        // Nothing published at its address in queue order: an empty TLAS, every ray missing.
        assert_eq!(missed.words(), MISSES);
    }

    let displaced = TitleBuffer::new(36);
    displaced.set_words(|i| (POSITIONS[i] + if i % 3 == 0 { 4. } else { 0. }).to_bits());
    let changed = self::geometry(&displaced, &indices);
    let after_refit = TitleBuffer::new(16);
    after_refit.set_words(|_| u32::MAX);
    // SAFETY: the title list and all CPU arrays are live for each call. The recording
    // retains every GPU input, including both generations after their title resources go.
    unsafe {
        // The TLAS is recorded before its BLAS address has a registry entry.
        assert!(Builder.build(list.this(), &tlas_desc(first.address(), &instances)));
        let producer = TitleList::new();
        let mut original = blas_input(&geometry);
        original.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION;
        assert!(Builder.build(
            producer.this(),
            &Build {
                DestAccelerationStructureData: blas.address(),
                Inputs: original,
                ..Default::default()
            }
        ));
        producer.run();
        let bytes = registry().structures[&blas.address()]
            .compacted_size
            .as_ref()
            .unwrap()
            .load(Ordering::Acquire);
        assert!(bytes > 0);
        let registry_source_capacity = registry().structures[&blas.address()].result.bytes;
        assert!(Builder.copy(
            cloning.this(),
            compacted.address(),
            blas.address(),
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT.0
        ));
        assert_eq!(
            registry().structures[&compacted.address()].result.bytes,
            registry_source_capacity
        );
        // The title packs the next compacted structure right after it.
        assert_eq!(
            registry().ranges[&compacted.address()].end,
            compacted.address() + 1,
            "a compacted copy claims its first byte"
        );
        assert!(Builder.copy(
            cloning.this(),
            cloned.address(),
            compacted.address(),
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_CLONE.0
        ));
        assert!(host_tlas(list.this(), cloned.address()).is_none());
        let mut updated = blas_input(&changed);
        updated.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE
            | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS(1 << 24);
        assert!(Builder.build(
            list.this(),
            &Build {
                DestAccelerationStructureData: cloned.address(),
                SourceAccelerationStructureData: cloned.address(),
                Inputs: updated,
                ..Default::default()
            }
        ));
        assert!(Builder.build(list.this(), &tlas_desc(second.address(), &instances)));
        assert!(host_tlas(cloning.this(), first.address()).is_none());
        let a = host_tlas_for_list(list.this(), first.address()).unwrap();
        let b = host_tlas_for_list(list.this(), second.address()).unwrap();
        assert_ne!(a, b);
        assert_eq!(a % 256, 0);
        assert_eq!(b % 256, 0);
        assert!(
            a.abs_diff(b) < 65536,
            "small TLAS results share one physical page"
        );
        assert_eq!(host_tlas(list.this(), first.address()), Some(a));
        assert_eq!(host_tlas(list.this(), second.address()), Some(b));
        let mut table = [0u64; 20];
        for (offset, address, expected) in [(0x60, first.address(), a), (0x80, second.address(), b)]
        {
            let slot = table.as_mut_ptr().cast::<u8>().add(offset) as u64;
            let mut desc = [0u32; 10];
            desc[1] = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE.0 as u32;
            desc[4] = address as u32;
            desc[5] = (address >> 32) as u32;
            crate::descriptor::create_srv(std::ptr::null(), std::ptr::null(), &desc, slot);
            assert_eq!(descriptor_tlas(list.this(), slot), Some(expected));
            // The variant on the host recording pins the same structure.
            assert_eq!(descriptor_tlas_in(list.host(), slot), Some(expected));
        }
        assert_eq!(
            descriptor_tlas(list.this(), table.as_ptr() as u64 + 0x60),
            Some(a)
        );
        assert_eq!(
            descriptor_tlas_in(cloning.host(), table.as_ptr() as u64 + 0x60),
            None,
            "its producer is not submitted"
        );
        let released = first.address();
        drop((blas, first, second));
        assert_eq!(host_tlas(list.this(), released), None);
        if rays {
            trace(host, &list, Scene::Address(a), &output);
            trace(host, &list, Scene::Address(b), &after_refit);
        }
    }
    // The TLAS was recorded before this producer was submitted.
    cloning.run();
    list.run();
    future.run();
    if rays {
        assert_eq!(delayed_hit.words(), hit());
        assert_eq!(output.words(), hit());
        assert_eq!(after_refit.words(), [0, 0, 1, 2.0f32.to_bits()]);
    }
}

#[test]
fn rt_spliced_trace_binds_t240_as_a_root_srv_with_bounded_root_srvs() {
    let test = "rt_spliced_trace_binds_t240_as_a_root_srv_with_bounded_root_srvs";
    let Some((gpu, rays)) = dxr(test) else {
        return;
    };
    let host = gpu.host();
    // The command signature takes the parameter on any DXR device.
    let (param, root) = spliced_root(host);
    assert!(dispatch::signature(&root, param, 20).is_some());
    if !rays {
        return;
    }
    let world = World::new();
    let slot = Descriptor::new(world.tlas.address());
    let list = TitleList::new();
    let out = unwritten();
    trace(host, &list, Scene::Spliced(slot.handle()), &out);
    list.run();
    assert_eq!(out.words(), hit());
}

#[test]
fn rt_indirect_trace_takes_the_groups_the_gpu_wrote() {
    let test = "rt_indirect_trace_takes_the_groups_the_gpu_wrote";
    let Some((gpu, rays)) = dxr(test) else {
        return;
    };
    let host = gpu.host();
    let world = World::new();
    let slot = Descriptor::new(world.tlas.address());
    let list = TitleList::new();
    // The title's CPU leaves zero groups; the GPU writes one.
    let (written, args) = (TitleBuffer::new(16), TitleBuffer::new(16));
    written.set_words(|i| u32::from(i < 3));
    args.shadow();
    assert!(crate::shadow::copy(
        list.host(),
        args.address(),
        written.address(),
        12
    ));
    let none = TitleBuffer::new(4);
    // SAFETY: the test's live list and descriptor slot.
    unsafe {
        // Arguments must be dword-aligned title memory.
        assert!(
            prepare_indirect_tlas(
                list.this(),
                slot.handle(),
                (args.address() + 2, 12, 1),
                None
            )
            .is_none()
        );
    }
    if !rays {
        return;
    }
    let (traced, uncounted) = (unwritten(), unwritten());
    trace(
        host,
        &list,
        Scene::Indirect(slot.handle(), args.address(), None),
        &traced,
    );
    // A command count of zero runs no command.
    trace(
        host,
        &list,
        Scene::Indirect(slot.handle(), args.address(), Some(none.address())),
        &uncounted,
    );
    list.run();
    assert_eq!(traced.words(), hit());
    assert_eq!(uncounted.words(), UNWRITTEN);
}

#[test]
fn rt_missing_tlas_writes_every_ray_as_a_miss() {
    let test = "rt_missing_tlas_writes_every_ray_as_a_miss";
    let Some((gpu, rays)) = dxr(test) else {
        return;
    };
    if !rays {
        return;
    }
    let host = gpu.host();
    let world = World::new();
    // A title address where no TLAS was ever published, and a BLAS address,
    // which is no TLAS either.
    let nothing = TitleBuffer::new(256);
    let blas = world.blas.address();
    let (direct, spliced, indirect, not_a_tlas) =
        (unwritten(), unwritten(), unwritten(), unwritten());
    let empty = Descriptor::new(nothing.address());
    let wrong = Descriptor::new(blas);
    // A descriptor the decoder reads as no view at all.
    let blank = Box::new(Descriptor([0; 4]));
    let (blank_direct, blank_indirect) = (unwritten(), unwritten());
    let args = TitleBuffer::new(12);
    args.set_words(|_| 1);
    let list = TitleList::new();
    trace(host, &list, Scene::Descriptor(empty.handle()), &direct);
    trace(host, &list, Scene::Spliced(empty.handle()), &spliced);
    trace(
        host,
        &list,
        Scene::Indirect(empty.handle(), args.address(), None),
        &indirect,
    );
    trace(host, &list, Scene::Descriptor(wrong.handle()), &not_a_tlas);
    trace(
        host,
        &list,
        Scene::Descriptor(blank.handle()),
        &blank_direct,
    );
    trace(
        host,
        &list,
        Scene::Indirect(blank.handle(), args.address(), None),
        &blank_indirect,
    );
    list.run();
    for out in [
        &direct,
        &spliced,
        &indirect,
        &not_a_tlas,
        &blank_direct,
        &blank_indirect,
    ] {
        assert_eq!(out.words(), MISSES, "the program ran and wrote its misses");
    }
}

/// Instance records naming each BLAS of `blas`, with an identity transform
/// and mask 0xff.
fn instances_of(blas: &[u64]) -> TitleBuffer {
    let records = TitleBuffer::new(64 * blas.len());
    records.set_words(|i| match i % 16 {
        0 | 5 | 10 => 1f32.to_bits(),
        12 => 0xff00_0000,
        14 => blas[i / 16] as u32,
        15 => (blas[i / 16] >> 32) as u32,
        _ => 0,
    });
    records
}

/// A title TLAS of no instance, or of an instance of a BLAS the host does
/// not have, makes every ray miss (WARP divided by zero building the first
/// and faulted on the null BLAS of the second); the other instances of a
/// TLAS keep their place and are hit.
#[test]
fn rt_tlas_of_no_instance_or_of_unknown_blas_misses() {
    let test = "rt_tlas_of_no_instance_or_of_unknown_blas_misses";
    let Some((gpu, rays)) = dxr(test) else {
        return;
    };
    let world = World::new();
    let nothing = TitleBuffer::new(1024);
    let unknown = instances_of(&[nothing.address()]);
    let mixed = instances_of(&[nothing.address(), world.blas.address()]);
    let tlas: [TitleBuffer; 3] = std::array::from_fn(|_| TitleBuffer::new(1024));
    let producer = TitleList::new();
    // SAFETY: the title list and the instance buffers are live.
    unsafe {
        let mut none = tlas_desc(tlas[0].address(), &unknown);
        none.Inputs.NumDescs = 0;
        assert!(Builder.build(producer.this(), &none));
        assert!(Builder.build(producer.this(), &tlas_desc(tlas[1].address(), &unknown)));
        let mut two = tlas_desc(tlas[2].address(), &mixed);
        two.Inputs.NumDescs = 2;
        assert!(Builder.build(producer.this(), &two));
    }
    producer.run();
    let held = |at: &TitleBuffer| registry().structures[&at.address()].holds.is_some();
    assert!(tlas.iter().all(held), "each TLAS keeps its stand-in BLAS");
    if !rays {
        return;
    }
    let slots = tlas.each_ref().map(|t| Descriptor::new(t.address()));
    let outs: [TitleBuffer; 3] = std::array::from_fn(|_| unwritten());
    let list = TitleList::new();
    for (slot, out) in slots.iter().zip(&outs) {
        trace(gpu.host(), &list, Scene::Descriptor(slot.handle()), out);
    }
    list.run();
    assert_eq!(outs[0].words(), MISSES, "no instance");
    assert_eq!(outs[1].words(), MISSES, "an unknown BLAS");
    assert_eq!(outs[2].words(), hit(), "the known instance after it");
}

#[test]
fn rt_translated_instance_moves_the_hit_to_its_translation() {
    let test = "rt_translated_instance_moves_the_hit_to_its_translation";
    let Some((gpu, rays)) = dxr(test) else {
        return;
    };
    let world = World::new();
    let instances = instances_of(&[world.blas.address()]);
    let mut words = instances.words();
    words[3] = 4f32.to_bits();
    instances.set_words(|i| words[i]);
    let tlas = TitleBuffer::new(1024);
    let build = TitleList::new();
    // SAFETY: the title list and instance buffer are live.
    unsafe { assert!(Builder.build(build.this(), &tlas_desc(tlas.address(), &instances))) };
    build.run();
    if !rays {
        return;
    }

    let slot = Descriptor::new(tlas.address());
    let out = unwritten();
    let list = TitleList::new();
    trace(gpu.host(), &list, Scene::Descriptor(slot.handle()), &out);
    list.run();
    assert_eq!(
        out.words(),
        [0, 0, 1, 2.0f32.to_bits()],
        "the instance misses at x=0 and hits at its translated x=4"
    );
}

/// A trace recorded before the title releases its TLAS's buffer, and
/// submitted after, misses: the release empties the slot the trace binds
/// at submission (the old TLAS stayed alive in it and was hit).
#[test]
fn rt_a_tlas_released_before_its_trace_is_submitted_misses() {
    let test = "rt_a_tlas_released_before_its_trace_is_submitted_misses";
    let Some((gpu, rays)) = dxr(test) else {
        return;
    };
    if !rays {
        return;
    }
    let World {
        tlas,
        blas,
        _inputs,
    } = World::new();
    let slot = Descriptor::new(tlas.address());
    let (released, kept) = (unwritten(), unwritten());
    let list = TitleList::new();
    trace(
        gpu.host(),
        &list,
        Scene::Descriptor(slot.handle()),
        &released,
    );
    drop(tlas);
    list.run();
    assert_eq!(released.words(), MISSES);
    // The same scene, not released, is hit.
    let world = World::new();
    let slot = Descriptor::new(world.tlas.address());
    let list = TitleList::new();
    trace(gpu.host(), &list, Scene::Descriptor(slot.handle()), &kept);
    list.run();
    assert_eq!(kept.words(), hit());
    drop(blas);
}

/// An indirect trace whose groups a title shader wrote through a UAV
/// earlier in its list: the arguments' buffer is then in the UAV state by
/// promotion, which the preparation must not name as common (the D3D12
/// debug layer, `XTP_D3D12_DEBUG=1`, fails the test on a wrong state).
#[test]
fn rt_indirect_trace_takes_groups_a_shader_wrote_in_its_list() {
    let test = "rt_indirect_trace_takes_groups_a_shader_wrote_in_its_list";
    let Some((gpu, rays)) = dxr(test) else {
        return;
    };
    if !rays {
        return;
    }
    let host = gpu.host();
    let world = World::new();
    let slot = Descriptor::new(world.tlas.address());
    let kernel = Kernel::new(host);
    let (written, args) = (TitleBuffer::new(256), TitleBuffer::new(256));
    written.set_words(|i| u32::from(i < 3));
    let list = TitleList::new();
    kernel.dispatch(list.host(), Pass::Copy, written.address(), args.address());
    let out = unwritten();
    trace(
        host,
        &list,
        Scene::Indirect(slot.handle(), args.address(), None),
        &out,
    );
    list.run();
    assert_eq!(out.words(), hit());
}

#[test]
fn rt_retirement_cancels_overlapping_publications_and_keeps_adjacent_ranges() {
    let mut table = Registry::default();
    let base = 0x4000;
    for offset in [0, 256, 512, 768] {
        table.begin(base + offset, Some(256));
    }
    let old = table.ranges[&(base + 256)].generation;
    let new = table.begin(base + 128, Some(384));
    assert!(!table.finish(base + 256, old, None));
    assert_eq!(
        table.ranges.keys().copied().collect::<Vec<_>>(),
        [base + 128, base + 512, base + 768]
    );
    table.retire(base + 180, base + 181, Some(None));
    assert!(!table.finish(base + 128, new.generation, None));
    assert_eq!(table.ranges.len(), 2);
}

#[test]
fn rt_owner_survives_more_than_64_intervening_buffers() {
    let base = 0x9876_0000_0000;
    crate::shadow::register(1, base, 1024);
    for i in 1..=100 {
        crate::shadow::register(i + 1, base + 2 * i as u64, 1);
    }
    assert_eq!(
        crate::shadow::buffer_owner(base + 255, 1),
        Some((1, base + 1024))
    );
    assert_eq!(
        crate::shadow::buffer_owner(base + 200, 1),
        Some((101, base + 201))
    );
    assert_eq!(
        crate::shadow::buffer_owner(base + 200, 16),
        Some((1, base + 1024))
    );
    registry().begin(base + 255, Some(128));
    crate::shadow::unregister(1, base);
    assert!(!registry().ranges.contains_key(&(base + 255)));
    for i in 1..=100 {
        crate::shadow::unregister(i + 1, base + 2 * i as u64);
    }
}

/// A spare serves a later generation at its title address once nothing
/// else holds it, if it fits without wasting over half of it; an address
/// keeps [`SPARES`] of them, for [`SPARE_FRAMES`] frames.
#[test]
fn rt_spares_serve_later_generations_once_free() {
    let test = "rt_spares_serve_later_generations_once_free";
    let Some(_gpu) = dxr(test) else {
        return;
    };
    let list = TitleList::new();
    let rec = Recorder::new(list.host()).expect("a DXR recorder");
    let mut table = Registry::default();
    let (free, held) = (
        ResultBuffer::new(&rec, 4096).unwrap(),
        ResultBuffer::new(&rec, 4096).unwrap(),
    );
    let gpu = free.gpu();
    // A recording or a slot still holds this one.
    let holder = held.clone();
    table.keep_spare(0x100, held);
    table.keep_spare(0x100, free);
    assert!(table.take_spare(0x200, 4096).is_none(), "another address");
    assert!(table.take_spare(0x100, 4097).is_none(), "too small");
    assert!(table.take_spare(0x100, 2047).is_none(), "over twice as big");
    let taken = table.take_spare(0x100, 2048).expect("the free one");
    assert_eq!(taken.gpu(), gpu);
    assert!(table.take_spare(0x100, 4096).is_none(), "the other is held");
    drop(holder);
    assert!(table.take_spare(0x100, 4096).is_some(), "free once let go");
    assert!(table.spares.is_empty());
    for _ in 0..SPARES + 2 {
        table.keep_spare(0x100, ResultBuffer::new(&rec, 256).unwrap());
    }
    assert_eq!(table.spares[&0x100].len(), SPARES);
    let frame = table.aged;
    table.age_spares(frame + SPARE_FRAMES);
    assert_eq!(table.spares[&0x100].len(), SPARES, "still recent");
    table.age_spares(frame + SPARE_FRAMES + 1);
    assert!(table.spares.is_empty(), "given back");
    // Memory the title releases or builds over drops its spares.
    table.keep_spare(0x300, ResultBuffer::new(&rec, 256).unwrap());
    table.retire(0x2c0, 0x340, None);
    assert!(table.spares.is_empty());
}

/// A TLAS rebuilt every frame at one title address alternates between two
/// results instead of allocating one a frame: the last generation stays in
/// its slot until the next is submitted, the one before is free again.
/// The reused result traces as the last build.
#[test]
fn rt_tlas_rebuilds_reuse_a_result_nothing_holds() {
    let test = "rt_tlas_rebuilds_reuse_a_result_nothing_holds";
    let Some((gpu, rays)) = dxr(test) else {
        return;
    };
    let world = World::new();
    let tlas = world.tlas.address();
    // A new allocation can get the same GPU address. A weak reference
    // identifies the result without keeping it in use.
    let generation = || Arc::downgrade(&registry().structures[&tlas].result);
    let rebuild = || {
        let list = TitleList::new();
        // SAFETY: the title list and the instance buffer are live.
        unsafe { assert!(Builder.build(list.this(), &tlas_desc(tlas, &world._inputs[2]))) };
        list.run();
        generation()
    };
    let first = generation();
    let second = rebuild();
    assert!(
        !first.ptr_eq(&second),
        "the first generation is still published"
    );

    // A watch test can reset its guards while the first result is a spare.
    drop(crate::watch::tests::serial());
    for expected in [&first, &second, &first] {
        assert!(rebuild().ptr_eq(expected), "reuse the retired result");
    }
    if rays {
        let slot = Descriptor::new(tlas);
        let list = TitleList::new();
        let out = unwritten();
        trace(gpu.host(), &list, Scene::Descriptor(slot.handle()), &out);
        list.run();
        assert_eq!(out.words(), hit());
    }
}

/// A compacted copy, or a copy of a structure whose build was refused (no
/// title extent), claims its first byte only: the structures the title
/// packed after it in the buffer stay.
#[test]
fn rt_copies_without_a_known_extent_claim_their_first_byte() {
    let test = "rt_copies_without_a_known_extent_claim_their_first_byte";
    let Some(_gpu) = crate::testgpu::device(test) else {
        return;
    };
    let packed = TitleBuffer::new(4096);
    let nothing = TitleBuffer::new(256);
    let base = packed.address();
    {
        let mut table = registry();
        table.begin(base + 1024, Some(512));
        table.begin(base + 2048, Some(512));
    }
    let list = TitleList::new();
    for (dst, mode) in [
        (
            base,
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT,
        ),
        (
            base + 512,
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_CLONE,
        ),
    ] {
        // SAFETY: a live title list; no structure at the source.
        assert!(!unsafe { Builder.copy(list.this(), dst, nothing.address(), mode.0) });
        let table = registry();
        assert!(
            table.ranges.contains_key(&(base + 1024)) && table.ranges.contains_key(&(base + 2048)),
            "the packed structures after {dst:#x} stay"
        );
    }
}

/// The title's release of a buffer empties the slots of its structures: a
/// TLAS or a trace recorded before finds nothing there.
#[test]
fn rt_released_buffers_empty_their_slots() {
    let test = "rt_released_buffers_empty_their_slots";
    let Some(_gpu) = dxr(test) else {
        return;
    };
    let World {
        tlas,
        blas,
        _inputs,
    } = World::new();
    let slots = {
        let table = registry();
        [&tlas, &blas].map(|b| table.ranges[&b.address()].slot.clone())
    };
    let published = |s: &Slot| s.lock().unwrap().is_some();
    assert!(slots.iter().all(published));
    drop((tlas, blas));
    assert!(!slots.iter().any(published), "emptied by the release");
}

/// A TLAS build takes the BLAS in the slot current as it reaches the
/// queue, and the one in the slot it recorded only where the current one
/// is empty.
#[test]
fn rt_current_slots_win_over_recorded_ones() {
    let test = "rt_current_slots_win_over_recorded_ones";
    let Some(_gpu) = dxr(test) else {
        return;
    };
    let world = World::new();
    let second = TitleBuffer::new(1024);
    let producer = TitleList::new();
    let geometry = geometry(&world._inputs[0], &world._inputs[1]);
    // SAFETY: the title list and the CPU arrays are live.
    unsafe {
        assert!(Builder.build(
            producer.this(),
            &Build {
                DestAccelerationStructureData: second.address(),
                Inputs: blas_input(&geometry),
                ..Default::default()
            }
        ));
    }
    producer.run();
    let (old, new) = {
        let table = registry();
        let at = |b: &TitleBuffer| table.structures[&b.address()].clone();
        (at(&world.blas), at(&second))
    };
    let at = 0x5157_0000_0000;
    let slot = |s: Option<&Structure>| -> Candidates {
        Arc::new(BTreeMap::from([(at, Arc::new(Mutex::new(s.cloned())))]))
    };
    let chosen = |recorded: &Candidates, current: &Candidates| {
        instances::UploadMap::resolve(recorded, current).map(|m| m.generations[0].1.gpu())
    };
    let recorded = slot(Some(&old));
    assert_eq!(chosen(&recorded, &slot(Some(&new))), Some(new.result.gpu()));
    assert_eq!(chosen(&recorded, &slot(None)), Some(old.result.gpu()));
    assert_eq!(
        chosen(&recorded, &Candidates::default()),
        Some(old.result.gpu())
    );
    assert_eq!(chosen(&slot(None), &slot(None)), None);
}

/// A BLAS input DXR cannot build from is refused before its build is
/// recorded: a count that is no whole number of triangles, a stride
/// shorter than a vertex.
#[test]
fn rt_invalid_blas_inputs_are_refused_before_their_build() {
    let test = "rt_invalid_blas_inputs_are_refused_before_their_build";
    let Some(_gpu) = crate::testgpu::device(test) else {
        return;
    };
    let list = TitleList::new();
    let (vertices, indices) = (TitleBuffer::new(36), TitleBuffer::new(8));
    vertices.set_words(|i| POSITIONS[i].to_bits());
    let translate = |g: &Geometry| {
        // SAFETY: the test owns the descriptor and the buffers it names.
        unsafe { triangles(list.host(), &blas_input(g)) }.is_some()
    };
    let valid = geometry(&vertices, &indices);
    indices.set_words(|i| [1 << 16, 2][i]);
    assert!(translate(&valid));
    // Index 3 of three vertices: checked on the GPU, as the build reads it
    // (`rt_indices_are_checked_as_the_build_reads_them`).
    indices.set_words(|i| [1 << 16, 3][i]);
    assert!(
        translate(&valid),
        "an index past the vertices, checked later"
    );
    indices.set_words(|i| [1 << 16, 2][i]);
    let mut two = valid;
    two.Anonymous.Triangles.IndexCount = 2;
    assert!(!translate(&two), "no whole triangle");
    let mut overlapping = valid;
    overlapping.Anonymous.Triangles.VertexBuffer.StrideInBytes = 4;
    assert!(!translate(&overlapping), "a stride shorter than a vertex");
    let mut past = valid;
    past.Anonymous.Triangles.VertexBuffer.StrideInBytes = 0x1_000c;
    assert!(!translate(&past), "a stride past the builder's 16 bits");
    let mut flagged = valid;
    flagged.Anonymous.Triangles.VertexBuffer.StrideInBytes = 0x800c;
    assert!(translate(&flagged), "bit 15 is a flag, the stride 12");
    let mut unindexed = valid;
    // SAFETY: the test's own triangle description.
    unsafe {
        let t = &mut unindexed.Anonymous.Triangles;
        (t.IndexFormat, t.IndexCount, t.IndexBuffer) = (DXGI_FORMAT_UNKNOWN, 0, 0);
    }
    assert!(translate(&unindexed), "three vertices without indices");
}

thread_local! {
    /// Set by a test to refuse the title's structures on its own thread,
    /// as `XTP_RT_BUILDS=0` does for a whole run ([`builds_on`]).
    pub(super) static BUILDS_OFF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The title's builds refused on this thread until dropped.
struct BuildsOff;

impl BuildsOff {
    fn new() -> BuildsOff {
        BUILDS_OFF.set(true);
        BuildsOff
    }
}

impl Drop for BuildsOff {
    fn drop(&mut self) {
        BUILDS_OFF.set(false);
    }
}

#[test]
fn rt_builds_switch_is_off_only_at_zero() {
    assert!(builds_asked(None));
    assert!(builds_asked(Some("1")));
    assert!(!builds_asked(Some("0")));
    assert!(!builds_asked(Some(" 0 ")));
}

/// With the title's structures off (`XTP_RT_BUILDS=0`), its BLAS and TLAS
/// builds, rehydrations and copies are refused, and the empty TLAS is
/// still made: a trace of the title's TLAS runs on it and every ray misses,
/// as on the console with an empty structure (under 4a0cdb3 the switch
/// dropped the empty TLAS too, and with it every trace).
#[test]
fn rt_builds_off_keeps_the_empty_tlas_and_traces_miss() {
    let test = "rt_builds_off_keeps_the_empty_tlas_and_traces_miss";
    let Some((gpu, rays)) = dxr(test) else {
        return;
    };
    let _off = BuildsOff::new();
    let (vertices, indices) = (TitleBuffer::new(36), TitleBuffer::new(8));
    vertices.set_words(|i| POSITIONS[i].to_bits());
    indices.set_words(|i| [1 << 16, 2][i]);
    let (blas, tlas, copy) = (
        TitleBuffer::new(1024),
        TitleBuffer::new(1024),
        TitleBuffer::new(1024),
    );
    let instances = instances_of(&[blas.address()]);
    let geometry = geometry(&vertices, &indices);
    let blas_build = Build {
        DestAccelerationStructureData: blas.address(),
        Inputs: blas_input(&geometry),
        ..Default::default()
    };
    let producer = TitleList::new();
    // SAFETY: the title list and every buffer the descriptions name are live.
    unsafe {
        assert!(!Builder.build(producer.this(), &blas_build), "a BLAS build");
        assert!(
            !Builder.build(producer.this(), &tlas_desc(tlas.address(), &instances)),
            "a TLAS build"
        );
        assert!(!rehydrate(producer.this(), &blas_build), "a rehydration");
        assert!(
            !Builder.copy(
                producer.this(),
                copy.address(),
                tlas.address(),
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_CLONE.0
            ),
            "a copy"
        );
    }
    producer.run();
    for at in [&blas, &tlas, &copy] {
        assert!(
            !registry().structures.contains_key(&at.address()),
            "no structure at a refused address"
        );
    }
    let list = TitleList::new();
    assert!(empty_tlas_in(list.host()).is_some(), "the empty TLAS");
    if !rays {
        list.run();
        return;
    }
    let slot = Descriptor::new(tlas.address());
    let out = unwritten();
    trace(gpu.host(), &list, Scene::Descriptor(slot.handle()), &out);
    list.run();
    assert_eq!(out.words(), MISSES, "the trace ran on the empty TLAS");
}

/// The known-bounds triangle (−5, 0, 1), (4, 0, 1), (4, 8, 9), moved `dz`
/// along z: its bounds are those corners.
fn bounds_vertices(dz: f32) -> [f32; 9] {
    [-5., 0., 1. + dz, 4., 0., 1. + dz, 4., 8., 9. + dz]
}

/// What `ray_bounds.hlsl` reads of that triangle: four hits at
/// t = 11 + y + dz (it lies in the plane z = 1 + y + dz), then four
/// misses just past its edges.
fn bounds_expected(dz: f32) -> [Option<f32>; 8] {
    let at = |y: f32| Some(11. + y + dz);
    [at(1.), at(7.), at(0.5), at(0.05), None, None, None, None]
}

/// Memory the runtime maps outside any title buffer object, as it maps the
/// title's physical pages: its host copies are loose 4 MiB windows
/// (`shadow::loose`). Never unmapped: copies made of it outlive the test.
fn plain_memory(len: usize) -> usize {
    use windows_sys::Win32::System::Memory::{
        CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, PAGE_READWRITE,
    };
    // SAFETY: a fresh section, mapped whole; the view keeps it.
    unsafe {
        let section = CreateFileMappingW(
            -1isize as _,
            std::ptr::null(),
            PAGE_READWRITE,
            0,
            len as u32,
            std::ptr::null(),
        );
        assert!(!section.is_null(), "a section");
        let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, len).Value as usize;
        windows_sys::Win32::Foundation::CloseHandle(section);
        assert_ne!(view, 0, "a view");
        view
    }
}

/// An upstream fork's fixture (#1816), reimplemented (watch item 497): a BLAS of known
/// bounds, (−5, 0, 1) to (4, 8, 9), whose vertices and indices are title
/// memory, read back by rays near each edge. Once in title buffers, once
/// in plain title memory, the vertices across a 4 MiB window boundary
/// (XTP's loose window copies); each built twice, the title's CPU moving
/// the vertices by 1 along z between the builds. A wrong address
/// translation gives misses or other distances; a stale copy gives the
/// first build's distances after the second.
#[test]
fn rt_blas_of_known_bounds_from_title_memory() {
    let test = "rt_blas_of_known_bounds_from_title_memory";
    let Some((gpu, rays)) = dxr(test) else {
        return;
    };
    if !rays {
        return;
    }
    let program = compiled("ray_bounds.hlsl", &[]);
    let (vertex_buffer, index_buffer) = (TitleBuffer::new(36), TitleBuffer::new(8));
    const WINDOW: usize = 4 << 20;
    let plain = plain_memory(2 * WINDOW);
    let boundary = (plain + WINDOW) & !(WINDOW - 1);
    let cases = [
        (
            "title buffers",
            vertex_buffer.address(),
            index_buffer.address(),
        ),
        (
            "plain title memory, across a window boundary",
            (boundary - 20) as u64,
            (boundary + 64) as u64,
        ),
    ];
    for (case, vertices, indices) in cases {
        // SAFETY: 8 bytes of the case's title memory, 2-byte aligned.
        unsafe { (indices as *mut [u16; 4]).write_unaligned([0, 1, 2, 0]) };
        let (blas, tlas) = (TitleBuffer::new(1024), TitleBuffer::new(1024));
        let instances = instances_of(&[blas.address()]);
        let slot = Descriptor::new(tlas.address());
        for dz in [0., 1.] {
            for (i, v) in bounds_vertices(dz).iter().enumerate() {
                // SAFETY: 36 bytes of the case's title memory, as its CPU writes them.
                unsafe { (vertices as *mut u32).add(i).write_unaligned(v.to_bits()) };
            }
            let geometry = Geometry {
                Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
                Flags: D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
                Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                    Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                        Transform3x4: 0,
                        IndexFormat: DXGI_FORMAT_R16_UINT,
                        VertexFormat: DXGI_FORMAT_R32G32B32_FLOAT,
                        IndexCount: 3,
                        VertexCount: 3,
                        IndexBuffer: indices,
                        VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                            StartAddress: vertices,
                            StrideInBytes: 12,
                        },
                    },
                },
            };
            let producer = TitleList::new();
            // SAFETY: the title list and every range the descriptions name are live.
            unsafe {
                assert!(
                    Builder.build(
                        producer.this(),
                        &Build {
                            DestAccelerationStructureData: blas.address(),
                            Inputs: blas_input(&geometry),
                            ..Default::default()
                        }
                    ),
                    "{case}: the BLAS build"
                );
                assert!(
                    Builder.build(producer.this(), &tlas_desc(tlas.address(), &instances)),
                    "{case}: the TLAS build"
                );
            }
            producer.run();
            let output = TitleBuffer::new(64);
            output.set_words(|_| 77);
            let list = TitleList::new();
            trace_program(
                gpu.host(),
                &list,
                Scene::Descriptor(slot.handle()),
                &output,
                &program,
            );
            list.run();
            let words = output.words();
            for (ray, want) in bounds_expected(dz).into_iter().enumerate() {
                let got = (words[2 * ray] == 1).then(|| f32::from_bits(words[2 * ray + 1]));
                match (got, want) {
                    (Some(got), Some(want)) => assert!(
                        (got - want).abs() < 1e-3,
                        "{case}, dz {dz}: ray {ray} hit at {got}, not {want}"
                    ),
                    (None, None) => {}
                    _ => panic!("{case}, dz {dz}: ray {ray} read {got:?}, not {want:?}"),
                }
            }
        }
    }
}

thread_local! {
    /// Set by a test to make the empty TLAS fail on its own thread
    /// ([`empty_tlas`]), as an exhausted RT budget would.
    pub(super) static EMPTY_FAILS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The empty TLAS failing on this thread until dropped.
struct EmptyFails;

impl EmptyFails {
    fn new() -> EmptyFails {
        EMPTY_FAILS.set(true);
        EmptyFails
    }
}

impl Drop for EmptyFails {
    fn drop(&mut self) {
        EMPTY_FAILS.set(false);
    }
}

/// A TLAS of no instance is the shared empty TLAS: it records no build and
/// so leaves no trailing barrier, and the next entry of a batch records its
/// own entry barrier (Codex review 11f27db1, P1-1: the next BLAS took it
/// for ordered and skipped its barrier). A recorded build leaves one.
#[test]
fn rt_a_tlas_of_no_instance_leaves_the_next_build_its_barrier() {
    let test = "rt_a_tlas_of_no_instance_leaves_the_next_build_its_barrier";
    let Some(_gpu) = dxr(test) else {
        return;
    };
    let (vertices, indices) = (TitleBuffer::new(36), TitleBuffer::new(8));
    vertices.set_words(|i| POSITIONS[i].to_bits());
    indices.set_words(|i| [1 << 16, 2][i]);
    let geometry = geometry(&vertices, &indices);
    let (blas, tlas, first) = (
        TitleBuffer::new(1024),
        TitleBuffer::new(1024),
        TitleBuffer::new(1024),
    );
    let instances = instances_of(&[blas.address()]);
    let no_instance = |at: &TitleBuffer| {
        let mut desc = tlas_desc(at.address(), &instances);
        desc.Inputs.NumDescs = 0;
        desc
    };
    // An empty TLAS made and submitted first, which the next takes.
    let producer = TitleList::new();
    // SAFETY: the title list and every buffer the descriptions name are live.
    unsafe { build_one(producer.this(), &no_instance(&first), false, None) };
    producer.run();
    let list = TitleList::new();
    // SAFETY: as above.
    unsafe {
        let empty = build_one(list.this(), &no_instance(&tlas), false, None);
        assert_eq!(
            empty,
            Mirrored {
                registered: true,
                barrier: false
            },
            "a TLAS of no instance"
        );
        let built = build_one(
            list.this(),
            &Build {
                DestAccelerationStructureData: blas.address(),
                Inputs: blas_input(&geometry),
                ..Default::default()
            },
            empty.barrier,
            None,
        );
        assert_eq!(
            built,
            Mirrored {
                registered: true,
                barrier: true
            },
            "a BLAS build"
        );
    }
    list.run();
}

/// A title TLAS is refused when the empty TLAS it would fall back to cannot
/// be made: of no instance it would be built as one, and an instance of a
/// BLAS the host does not have would name a null BLAS (Codex review
/// 11f27db1, P1-3).
#[test]
fn rt_a_tlas_without_its_stand_in_is_refused() {
    let test = "rt_a_tlas_without_its_stand_in_is_refused";
    let Some(_gpu) = dxr(test) else {
        return;
    };
    let _fails = EmptyFails::new();
    let nothing = TitleBuffer::new(1024);
    let unknown = instances_of(&[nothing.address()]);
    let tlas: [TitleBuffer; 2] = std::array::from_fn(|_| TitleBuffer::new(1024));
    let producer = TitleList::new();
    // SAFETY: the title list and the instance buffer are live.
    unsafe {
        let mut none = tlas_desc(tlas[0].address(), &unknown);
        none.Inputs.NumDescs = 0;
        assert!(
            !Builder.build(producer.this(), &none),
            "a TLAS of no instance"
        );
        assert!(
            !Builder.build(producer.this(), &tlas_desc(tlas[1].address(), &unknown)),
            "a TLAS of an unknown BLAS"
        );
    }
    producer.run();
    for t in &tlas {
        assert!(!registry().structures.contains_key(&t.address()));
    }
}

/// The indices a BLAS build reads are checked on the GPU, after every write
/// before the build in its submission (Codex review 11f27db1, P1-2): valid
/// in title memory but made invalid by a GPU copy before the build, the
/// build takes no triangle and every ray misses; invalid in title memory
/// but made valid by a GPU copy before the build, it is built and hit (the
/// check of title memory at recording refused it).
#[test]
fn rt_indices_are_checked_as_the_build_reads_them() {
    let test = "rt_indices_are_checked_as_the_build_reads_them";
    let Some((gpu, rays)) = dxr(test) else {
        return;
    };
    if !rays {
        return;
    }
    let vertices = TitleBuffer::new(36);
    vertices.set_words(|i| POSITIONS[i].to_bits());
    // 16-bit indices 0, 1, 2, and 0, 1, 3 (past the three vertices).
    let (valid, invalid) = ([1 << 16, 2], [1 << 16, 3]);
    let cases = [
        ("valid", valid, valid, true),
        ("made invalid by the GPU", valid, invalid, false),
        ("made valid by the GPU", invalid, valid, true),
    ];
    for (case, in_memory, written, hits) in cases {
        let indices = TitleBuffer::new(8);
        indices.set_words(|i| in_memory[i]);
        let patch = TitleBuffer::new(8);
        patch.set_words(|i| written[i]);
        indices.shadow();
        patch.shadow();
        let (blas, tlas) = (TitleBuffer::new(1024), TitleBuffer::new(1024));
        let instances = instances_of(&[blas.address()]);
        let geometry = geometry(&vertices, &indices);
        let list = TitleList::new();
        // The GPU writes the indices' host copy before the build.
        assert!(crate::shadow::copy(
            list.host(),
            indices.address(),
            patch.address(),
            8
        ));
        // SAFETY: the title list and every buffer the descriptions name are live.
        unsafe {
            assert!(
                Builder.build(
                    list.this(),
                    &Build {
                        DestAccelerationStructureData: blas.address(),
                        Inputs: blas_input(&geometry),
                        ..Default::default()
                    }
                ),
                "{case}: the BLAS build is recorded"
            );
            assert!(Builder.build(list.this(), &tlas_desc(tlas.address(), &instances)));
        }
        list.run();
        let slot = Descriptor::new(tlas.address());
        let out = unwritten();
        let tracing = TitleList::new();
        trace(gpu.host(), &tracing, Scene::Descriptor(slot.handle()), &out);
        tracing.run();
        assert_eq!(out.words(), if hits { hit() } else { MISSES }, "{case}");
    }
}

/// Three triangles side by side along x, each in a plane of its own, as
/// `ray_console.hlsl`'s rays read them: (−5, 0, 1), (4, 0, 1), (4, 8, 9) in
/// z = 1 + y; (15, 0, 2), (24, 0, 2), (24, 8, 6) in z = 2 + y / 2; (35, 0,
/// 3), (44, 0, 3), (44, 8, 19) in z = 3 + 2y. A wrong vertex moves a hit or
/// turns it into a miss.
const MESH: [[f32; 3]; 9] = [
    [-5., 0., 1.],
    [4., 0., 1.],
    [4., 8., 9.],
    [15., 0., 2.],
    [24., 0., 2.],
    [24., 8., 6.],
    [35., 0., 3.],
    [44., 0., 3.],
    [44., 8., 19.],
];

/// What each of `ray_console.hlsl`'s rays reads of [`MESH`]: its distance
/// (from z = −10, at its (x, y)) and the triangle hit, or a miss.
const MESH_READ: [Option<(f32, u32)>; 8] = [
    Some((12., 0)),
    Some((12.5, 1)),
    Some((15., 2)),
    Some((18., 0)),
    Some((27., 2)),
    None,
    None,
    None,
];

/// Every ray a miss.
const ALL_MISS: [Option<(f32, u32)>; 8] = [None; 8];

/// A title buffer of `len` bytes (a multiple of 4) holding `bytes` from
/// its byte `at`, zeros elsewhere.
fn title_bytes(len: usize, at: usize, bytes: &[u8]) -> TitleBuffer {
    let mut all = vec![0u8; len];
    all[at..at + bytes.len()].copy_from_slice(bytes);
    let buffer = TitleBuffer::new(len);
    buffer.set_words(|i| u32::from_le_bytes(all[4 * i..4 * i + 4].try_into().unwrap()));
    buffer
}

/// [`MESH`]'s vertices as `R32G32B32_FLOAT`, at the start of a title buffer.
fn mesh_floats() -> TitleBuffer {
    let bytes: Vec<u8> = MESH
        .iter()
        .flatten()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    title_bytes(bytes.len().next_multiple_of(16), 0, &bytes)
}

/// [`MESH`]'s indices as the console builder packs them: three meshlets,
/// which start at vertex entries 0, 5 and `third` (65,538 reaches past 16
/// bits, as bit 14 of the meshlet's first word); triangle 0 takes corners
/// (0, 1, 2) of meshlet 0, triangle 1 corners (2, 0, 1) of meshlet 1,
/// triangle 2 corners (1, 2, 0) of meshlet 2; one unused triangle word
/// after them, and bits above the meshlet count in the header. Every other
/// entry names no vertex (0xFFFF), and with `valid` false these too.
fn packed_mesh(third: u32, valid: bool) -> Vec<u8> {
    let at = third as usize;
    let mut entries = vec![0xffffu16; at + 3];
    if valid {
        entries[..3].copy_from_slice(&[0, 1, 2]);
        entries[5..8].copy_from_slice(&[4, 5, 3]);
        entries[at..].copy_from_slice(&[8, 6, 7]);
    }
    console::pack::indices(
        &[0, 5, third],
        &[(0, [0, 1, 2]), (1, [2, 0, 1]), (2, [1, 2, 0])],
        1,
        &entries,
        5,
    )
}

/// The origins and scales of quantized [`MESH`]es' headers: the first
/// coordinate (24 bits) reaches past 16 bits, the second (24) past its
/// first byte, the third (16) is small.
const ORIGIN: [f32; 3] = [-524_288., -32_768., -20.];
const SCALE: [f32; 3] = [0.5, 0.25, 1.];

/// [`MESH`]'s vertices quantized under axis order `axes`, `8 + pad` bytes
/// apart, their header at byte `at` of a title buffer: the buffer and the
/// first vertex's address.
fn quantized_mesh(axes: u32, pad: usize, at: usize) -> (TitleBuffer, u64) {
    let mut bytes = console::pack::header(ORIGIN, axes, SCALE);
    for &position in &MESH {
        let [a, b, c] = console::pack::quantize(position, axes, ORIGIN, SCALE);
        bytes.extend(console::pack::vertex(a, b, c, pad));
    }
    let buffer = title_bytes((at + bytes.len()).next_multiple_of(16), at, &bytes);
    let first = buffer.address() + at as u64 + console::QUANTIZED_HEADER;
    (buffer, first)
}

/// A geometry of [`MESH`]'s nine vertices at `vertex`, in `vertex_format`,
/// `stride` apart, and nine indices at `index`, in `index_format`.
fn mesh_geometry(
    (vertex, vertex_format, stride): (u64, DXGI_FORMAT, u64),
    (index, index_format): (u64, DXGI_FORMAT),
) -> Geometry {
    Geometry {
        Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
        Flags: D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE,
        Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
            Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                Transform3x4: 0,
                IndexFormat: index_format,
                VertexFormat: vertex_format,
                IndexCount: 9,
                VertexCount: 9,
                IndexBuffer: index,
                VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                    StartAddress: vertex,
                    StrideInBytes: stride,
                },
            },
        },
    }
}

/// How a test hands a BLAS to the host: the title's build or rehydration.
#[derive(Clone, Copy, Debug)]
enum Made {
    Built,
    Rehydrated,
}

/// A BLAS of `geometry` made as `made`, and a TLAS of it, recorded after
/// `before` in one list and submitted; then `ray_console.hlsl`'s rays
/// traced: what each read (hit, distance, front face, triangle).
fn console_rays(
    gpu: &crate::testgpu::Gpu,
    program: &[u8],
    geometry: &Geometry,
    made: Made,
    before: impl FnOnce(&TitleList),
) -> Vec<[u32; 4]> {
    let (blas, tlas) = (TitleBuffer::new(1024), TitleBuffer::new(1024));
    let instances = instances_of(&[blas.address()]);
    let list = TitleList::new();
    before(&list);
    let build = Build {
        DestAccelerationStructureData: blas.address(),
        Inputs: blas_input(geometry),
        ..Default::default()
    };
    // SAFETY: the title list and every range the descriptions name are live.
    unsafe {
        let recorded = match made {
            Made::Built => Builder.build(list.this(), &build),
            Made::Rehydrated => rehydrate(list.this(), &build),
        };
        assert!(recorded, "the BLAS ({made:?}) is recorded");
        assert!(Builder.build(list.this(), &tlas_desc(tlas.address(), &instances)));
    }
    list.run();
    let slot = Descriptor::new(tlas.address());
    let output = TitleBuffer::new(128);
    output.set_words(|_| 77);
    let tracing = TitleList::new();
    trace_program(
        gpu.host(),
        &tracing,
        Scene::Descriptor(slot.handle()),
        &output,
        program,
    );
    tracing.run();
    let words = output.words();
    words.chunks(4).map(|w| [w[0], w[1], w[2], w[3]]).collect()
}

/// Checks what the rays read against `want`: each hit at its distance
/// (within 1e-3) on its triangle, each miss a miss.
fn assert_rays(case: &str, read: &[[u32; 4]], want: &[Option<(f32, u32)>; 8]) {
    for (ray, (got, want)) in read.iter().zip(want).enumerate() {
        match want {
            Some((t, triangle)) => {
                assert_eq!(
                    got[0], 1,
                    "{case}: ray {ray} read {got:?}, wanted {t} on {triangle}"
                );
                let at = f32::from_bits(got[1]);
                assert!(
                    (at - t).abs() < 1e-3,
                    "{case}: ray {ray} hit at {at}, not {t}"
                );
                assert_eq!(got[3], *triangle, "{case}: ray {ray}'s triangle");
            }
            None => assert_eq!(got[0], 0, "{case}: ray {ray} read {got:?}, wanted a miss"),
        }
    }
}

/// For each ray that hit in `a`: whether its triangle faced it as in `b`.
fn same_faces(a: &[[u32; 4]], b: &[[u32; 4]]) -> Vec<bool> {
    (a.iter().zip(b))
        .filter(|(a, _)| a[0] == 1)
        .map(|(a, b)| a[2] == b[2])
        .collect()
}

/// The console builder's meshlet-packed indices (`R16_TYPELESS`, see
/// `console`): a BLAS of them builds the triangles their meshlets name,
/// hit where the triangles of the same vertices indexed plainly are, a
/// meshlet past entry 65,535 included; with the stride's bit 15, each
/// triangle is turned, as the plain indices with their first and third
/// swapped. Read as 16-bit indices (the path before), the same bytes name
/// vertices past the mesh's, and the build takes no triangle. The index
/// buffers start 4 or 6 bytes past a 16-byte boundary (Codex review
/// dcc6e70a, P1-2).
#[test]
fn rt_meshlet_packed_indices_build_the_triangles_they_name() {
    let test = "rt_meshlet_packed_indices_build_the_triangles_they_name";
    let Some((gpu, rays)) = dxr(test) else {
        return;
    };
    if !rays {
        return;
    }
    let program = compiled("ray_console.hlsl", &[]);
    let vertices = mesh_floats();
    let floats = (vertices.address(), DXGI_FORMAT_R32G32B32_FLOAT, 12);
    let turned_floats = (floats.0, floats.1, 12 | 0x8000);
    let packed = packed_mesh(65_538, true);
    let packed = title_bytes((4 + packed.len()).next_multiple_of(16), 4, &packed);
    let packed_at = packed.address() + 4;
    let plain = |order: [u16; 9]| {
        let bytes: Vec<u8> = order.iter().flat_map(|i| i.to_le_bytes()).collect();
        title_bytes(32, 6, &bytes)
    };
    let (ahead, turned) = (
        plain([0, 1, 2, 3, 4, 5, 6, 7, 8]),
        plain([2, 1, 0, 5, 4, 3, 8, 7, 6]),
    );
    let rays_of = |geometry: Geometry| console_rays(&gpu, &program, &geometry, Made::Built, |_| {});
    let reference = rays_of(mesh_geometry(
        floats,
        (ahead.address() + 6, DXGI_FORMAT_R16_UINT),
    ));
    assert_rays("plain indices", &reference, &MESH_READ);
    let decoded = rays_of(mesh_geometry(floats, (packed_at, DXGI_FORMAT_R16_TYPELESS)));
    assert_rays("meshlet-packed indices", &decoded, &MESH_READ);
    assert_eq!(decoded, reference, "the same hits, faces and triangles");
    let turned_reference = rays_of(mesh_geometry(
        floats,
        (turned.address() + 6, DXGI_FORMAT_R16_UINT),
    ));
    let turned_decoded = rays_of(mesh_geometry(
        turned_floats,
        (packed_at, DXGI_FORMAT_R16_TYPELESS),
    ));
    assert_rays("turned by bit 15", &turned_decoded, &MESH_READ);
    assert_eq!(turned_decoded, turned_reference, "as the swapped corners");
    assert!(
        same_faces(&decoded, &turned_decoded)
            .iter()
            .all(|same| !same),
        "bit 15 turns every triangle: {decoded:?} {turned_decoded:?}"
    );
    let before = rays_of(mesh_geometry(floats, (packed_at, DXGI_FORMAT_R16_UINT)));
    assert_rays("read as 16-bit indices", &before, &ALL_MISS);
}

/// The console builder's quantized vertices (`R32G32_FLOAT`, see
/// `console`): under each axis order, a BLAS of them builds [`MESH`]'s
/// triangles, hit at their distances, from the fixed-point coordinates,
/// the header's origins and scales (past 16 bits in the 24-bit ones) and
/// a 16-byte stride. Their headers start 0, 4 and 8 bytes past a 16-byte
/// boundary and their 16-bit indices 2 (Codex review dcc6e70a, P1-2).
#[test]
fn rt_quantized_vertices_build_their_positions() {
    let test = "rt_quantized_vertices_build_their_positions";
    let Some((gpu, rays)) = dxr(test) else {
        return;
    };
    if !rays {
        return;
    }
    let program = compiled("ray_console.hlsl", &[]);
    let order: Vec<u8> = (0..9u16).flat_map(|i| i.to_le_bytes()).collect();
    let indices = title_bytes(32, 2, &order);
    for (axes, at) in [(0, 0), (1, 4), (2, 8)] {
        let (_vertices, first) = quantized_mesh(axes, 8, at);
        let geometry = mesh_geometry(
            (first, DXGI_FORMAT_R32G32_FLOAT, 16),
            (indices.address() + 2, DXGI_FORMAT_R16_UINT),
        );
        let read = console_rays(&gpu, &program, &geometry, Made::Built, |_| {});
        assert_rays(&format!("axis order {axes}"), &read, &MESH_READ);
    }
}

/// A title-like geometry: quantized vertices 8 bytes apart and
/// meshlet-packed indices, the stride's bit 15 set or not, as the title's
/// (0x8008 and 8), built and rehydrated alike: [`MESH`]'s triangles, the
/// turned ones facing the other way.
#[test]
fn rt_title_like_geometry_builds_and_rehydrates_faithfully() {
    let test = "rt_title_like_geometry_builds_and_rehydrates_faithfully";
    let Some((gpu, rays)) = dxr(test) else {
        return;
    };
    if !rays {
        return;
    }
    let program = compiled("ray_console.hlsl", &[]);
    let (_vertices, first) = quantized_mesh(2, 0, 0);
    let packed = packed_mesh(13, true);
    let packed = title_bytes(packed.len().next_multiple_of(16), 0, &packed);
    for made in [Made::Built, Made::Rehydrated] {
        let geometry = |stride| {
            mesh_geometry(
                (first, DXGI_FORMAT_R32G32_FLOAT, stride),
                (packed.address(), DXGI_FORMAT_R16_TYPELESS),
            )
        };
        let ahead = console_rays(&gpu, &program, &geometry(8), made, |_| {});
        assert_rays(&format!("{made:?}, stride 8"), &ahead, &MESH_READ);
        let turned = console_rays(&gpu, &program, &geometry(0x8008), made, |_| {});
        assert_rays(&format!("{made:?}, stride 0x8008"), &turned, &MESH_READ);
        assert!(
            same_faces(&ahead, &turned).iter().all(|same| !same),
            "{made:?}: bit 15 turns every triangle"
        );
    }
}

/// Inputs a title shader wrote in the same submission, before the build,
/// are read as it left them (Codex review dcc6e70a, P1-1: the shader's
/// write promotes the title's copy to the unordered access state, which
/// the pass reads it in): meshlet-packed indices that name no vertex in
/// title memory but valid ones once a title pass has copied them over
/// build the mesh, and the reverse builds nothing.
#[test]
fn rt_inputs_a_title_shader_wrote_are_read_as_it_left_them() {
    let test = "rt_inputs_a_title_shader_wrote_are_read_as_it_left_them";
    let Some((gpu, rays)) = dxr(test) else {
        return;
    };
    if !rays {
        return;
    }
    let program = compiled("ray_console.hlsl", &[]);
    let kernel = Kernel::new(gpu.host());
    let vertices = mesh_floats();
    let len = crate::testgpu::KERNEL_BYTES as usize;
    let (valid, invalid) = (packed_mesh(10, true), packed_mesh(10, false));
    assert!(valid.len() <= len);
    let cases = [
        ("made valid by a title shader", &invalid, &valid, &MESH_READ),
        (
            "made invalid by a title shader",
            &valid,
            &invalid,
            &ALL_MISS,
        ),
    ];
    for (case, in_memory, written, want) in cases {
        let indices = title_bytes(len, 0, in_memory);
        let patch = title_bytes(len, 0, written);
        let geometry = mesh_geometry(
            (vertices.address(), DXGI_FORMAT_R32G32B32_FLOAT, 12),
            (indices.address(), DXGI_FORMAT_R16_TYPELESS),
        );
        let read = console_rays(&gpu, &program, &geometry, Made::Built, |list| {
            kernel.dispatch(list.host(), Pass::Copy, patch.address(), indices.address());
        });
        assert_rays(case, &read, want);
    }
}

/// The pass reads each window through a view whose base is 16-byte
/// aligned, the window's offset in it given apart (Codex review dcc6e70a,
/// P1-2).
#[test]
fn rt_input_views_are_16_byte_aligned() {
    for source in [0x7e57_0000u64, 0x7e57_0002, 0x7e57_0004, 0x7e57_000e] {
        let (base, offset) = indices::view(source);
        assert_eq!(base % 16, 0);
        assert_eq!(base + u64::from(offset), source);
        assert!(offset < 16);
    }
}

/// A trace dispatched with its host list alone, as draws dispatch it (task
/// ac895e39), takes the TLAS published at its place in submission order:
/// one that another list builds and submits after the trace is recorded,
/// and before it is submitted.
#[test]
fn rt_a_trace_by_its_host_list_takes_the_tlas_submitted_before_it() {
    let test = "rt_a_trace_by_its_host_list_takes_the_tlas_submitted_before_it";
    let Some((gpu, rays)) = dxr(test) else {
        return;
    };
    if !rays {
        return;
    }
    let (vertices, indices) = (TitleBuffer::new(36), TitleBuffer::new(8));
    vertices.set_words(|i| POSITIONS[i].to_bits());
    indices.set_words(|i| [1 << 16, 2][i]);
    let (blas, tlas) = (TitleBuffer::new(1024), TitleBuffer::new(1024));
    let instances = instances_of(&[blas.address()]);
    let slot = Descriptor::new(tlas.address());
    let out = unwritten();
    let tracing = TitleList::new();
    trace(
        gpu.host(),
        &tracing,
        Scene::HostDescriptor(slot.handle()),
        &out,
    );
    let producer = TitleList::new();
    let geometry = geometry(&vertices, &indices);
    // SAFETY: the title list and every buffer the descriptions name are live.
    unsafe {
        assert!(Builder.build(
            producer.this(),
            &Build {
                DestAccelerationStructureData: blas.address(),
                Inputs: blas_input(&geometry),
                ..Default::default()
            }
        ));
        assert!(Builder.build(producer.this(), &tlas_desc(tlas.address(), &instances)));
    }
    producer.run();
    tracing.run();
    assert_eq!(out.words(), hit(), "the TLAS submitted before the trace");
}
