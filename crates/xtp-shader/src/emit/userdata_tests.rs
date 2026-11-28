//! Synthetic tables of spilled parameters. Register rewrite cases:
//! Ported from Marvisak/VIctory#489 (head fe7210e) by louketa.

use super::{HostShader, translate_program};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{Container, ShaderError, ir, resource};

const PTR: [u32; 3] = [0xBE82_0300, 0x8703_FF01, 0xff]; // s2 = entry s0, s3 = entry s1 & 0xff
const END: u32 = 0xBF81_0000;

fn param(root: RootType, user_data: u8, sgprs: u32, slot: u8) -> RootParam {
    RootParam {
        root,
        user_data,
        sgprs,
        slot,
        table_offset_dwords: 0,
        base_register: 0,
    }
}

fn native(words: &[u32], layout: Vec<RootParam>) -> NativeProgram {
    NativeProgram {
        stage: HwStage::Cs,
        io: StageIo::Cs(ComputeIo {
            rsrc2: 4 << 1,
            threads: [1, 1, 1],
        }),
        wave64: false,
        float_mode: 0,
        vgprs: 8,
        used: vec![],
        layout,
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

fn track(n: &NativeProgram) -> Result<resource::Resources, ShaderError> {
    let p = ir::build(&n.code, n.wave64, n.vgprs)?;
    resource::track_native(&p, n, None)
}

fn overwrite_shader() -> HostShader {
    let words = [
        PTR.as_slice(),
        &[
            0xF410_0201,
            0xFA00_0010, // s_load_dwordx16 s[8:23], s[2:3], 0x10
            0xF404_0101,
            0xFA00_0000, // s_load_dwordx2 s[4:5], s[2:3], 0
            0x7E02_0210, // v1 = s16 (table dword 12)
            0x7E04_0205, // v2 = s5 (table dword 1)
            0xBE85_0380, // descriptor-table high half = 0
            0xF408_0202,
            0xFA00_0000, // load UAV descriptor through s[4:5]
            0xE070_0000,
            0x8002_0100, // store v1 at byte 0
            0xE070_0004,
            0x8002_0200, // store v2 at byte 4
            END,
        ],
    ]
    .concat();
    let mut n = native(
        &words,
        vec![
            param(RootType::Table, 16, 1, 4),
            param(RootType::Constants, 17, 19, 2),
        ],
    );
    n.used.push(UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 16,
        slot: 4,
    });
    let resources = track(&n).unwrap();
    assert_eq!(resources.spill_loads.len(), 2);
    assert_eq!(resources.scalar.len(), 1);
    assert_eq!(resources.scalar[0].1.param, 0);
    let bytes = crate::dxbc::build(&[]);
    translate_program(&Container::parse(&bytes).unwrap(), &n).unwrap()
}

#[test]
fn reloaded_words_keep_their_root_slots_after_register_overwrites() {
    let shader = overwrite_shader();
    let mut bindings: Vec<_> = shader.user_data.iter().map(|u| (u.slot, u.dword)).collect();
    bindings.sort_unstable();
    assert_eq!(bindings, [(2, 0), (2, 11), (4, 0)]);
    assert_eq!(shader.user_data_dwords(), 4);
}

#[cfg(windows)]
#[test]
fn dispatch_reads_spilled_constants_and_the_spilled_table() {
    use super::host::{Dispatch, Host, Root, View};
    let mut shader = overwrite_shader();
    if let Ok(mut validator) =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
    {
        validator.validate(&mut shader.dxil).unwrap();
    }
    let Some(host) = Host::test_device() else {
        return;
    };
    let (root, pso) = host.compute_pipeline(&shader).unwrap();
    let mut constants = vec![0; shader.user_data_dwords() as usize];
    for u in &shader.user_data {
        constants[u.index as usize] = match (u.slot, u.dword) {
            (2, 11) => 31,
            (2, 0) => 79,
            (4, 0) => 0,
            other => panic!("unexpected binding {other:?}"),
        };
    }
    let output = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(0, Root::Constants(constants))],
                groups: [1, 1, 1],
                samplers: &[],
                views: &[View::RawUav(2)],
            },
        )
        .unwrap();
    assert_eq!(output[0], [31, 79]);
}

#[test]
fn rejects_modified_pointers_dynamic_offsets_and_used_padding() {
    let layout = || vec![param(RootType::Constants, 16, 1, 0)];
    let load = [0xF404_0101, 0xFA00_0000]; // x2, second word is padding
    let words = [PTR.as_slice(), &load, &[END]].concat();
    let r = track(&native(&words, layout())).unwrap();
    assert_eq!(r.spill_loads.len(), 1);
    assert_eq!(r.params.len(), 1);
    for (k, word) in [
        (2, 0x7f),
        (0, 0xBE82_0304),
        (4, 0x0800_0000),
        (4, 0xFA00_0002),
    ] {
        let mut other = words.clone();
        other[k] = word;
        assert!(track(&native(&other, layout())).is_err(), "{word:x}");
    }
    let used = [PTR.as_slice(), &load, &[0xBE86_0305, END]].concat();
    assert!(track(&native(&used, layout())).is_err());
    // Entry s0:s1 are actual root arguments in this layout.
    let direct = vec![
        param(RootType::Descriptor, 0, 2, 1),
        param(RootType::Constants, 16, 1, 0),
    ];
    assert!(track(&native(&words, direct)).is_err());
}

/// Reload just the high word, then dereference that root descriptor. Its
/// halves must agree on the parameter even across distinct load sites.
#[test]
fn descriptor_halves_keep_their_identity_across_reloads() {
    let code = [
        PTR.as_slice(),
        &[
            0xF404_0101,
            0xFA00_0000, // s[4:5] = table dwords 0..1
            0xF400_0141,
            0xFA00_0004, // s5 = table dword 1 again
            0xF400_0182,
            0xFA00_0008, // s6 = *(s[4:5] + 8)
            END,
        ],
    ]
    .concat();
    let layout = || {
        vec![
            param(RootType::Descriptor, 16, 2, 3),
            param(RootType::Descriptor, 18, 2, 7),
        ]
    };
    let r = track(&native(&code, layout())).unwrap();
    assert_eq!(r.spill_loads.len(), 2);
    assert_eq!(r.scalar.len(), 1);
    assert_eq!((r.scalar[0].1.param, r.scalar[0].1.offset), (0, 8));
    let mut wrong = code;
    wrong[6] = 0xFA00_000C; // high word from the other descriptor
    assert!(track(&native(&wrong, layout())).is_err());
}

#[test]
fn overwriting_the_pointer_stops_recognition_at_the_next_load() {
    let code = [
        PTR.as_slice(),
        &[
            0xF404_0081,
            0xFA00_0000, // overwrite pointer s[2:3] with two constants
            0xF400_0101,
            0xFA00_0000, // dereference those constants
            END,
        ],
    ]
    .concat();
    let r = track(&native(&code, vec![param(RootType::Constants, 16, 2, 3)])).unwrap();
    assert_eq!(r.spill_loads.len(), 1);
    assert_eq!(r.constant_reads.values().next().unwrap().slot, 3);
    // Two constants from distinct parameters do not form a linked address.
    let result = track(&native(
        &code,
        vec![
            param(RootType::Constants, 16, 1, 3),
            param(RootType::Constants, 17, 1, 4),
        ],
    ));
    assert!(matches!(
        result,
        Err(ShaderError::Unsupported { offset: 20, .. })
    ));
}

#[test]
fn ambiguous_and_unmapped_first_spill_words_are_refused_even_when_dead() {
    let load = [0xF400_0101, 0xFA00_0004];
    let code = [PTR.as_slice(), &load, &[END]].concat();
    for layout in [
        vec![
            param(RootType::Constants, 16, 2, 0),
            param(RootType::Constants, 17, 1, 1),
        ],
        vec![param(RootType::Constants, 16, 1, 0)],
    ] {
        assert!(matches!(
            track(&native(&code, layout)),
            Err(ShaderError::Unsupported {
                offset: 12,
                what: "a spill-table dword without a unique root parameter"
            })
        ));
    }
}

#[test]
fn unused_entry_pointer_has_no_spill_bindings() {
    for code in [vec![END], vec![0xBE82_0300, END], vec![0xBE83_0301, END]] {
        let r = track(&native(&code, vec![param(RootType::Descriptor, 16, 2, 0)])).unwrap();
        assert!(r.spill_loads.is_empty() && r.params.is_empty());
    }
}

#[cfg(windows)]
#[test]
fn reads_a_packed_address_and_an_adjacent_pair_from_spilled_constants() {
    use super::host::{Dispatch, Host, Root, Stride, View};
    let mut words = [
        PTR.as_slice(),
        &[
            0xF404_0201,
            0xFA00_0000, // UAV in s[8:9].
            0xF404_0101,
            0xFA00_0008, // Constants in s[4:5].
            0xF400_0302,
            0xFA00_0008, // Read through the full pair.
            0xBE85_0380,
            0x8F84_8804,
            0x8804_8C04, // Address in units of 256 bytes + 12.
            0xF400_0342,
            0xFA00_0000,
            0xBE8A_03C1,
            0xBE8B_03FF,
            0x3101_4FAC,
            0x7E04_020C,
            0x7E06_020D,
            0xE070_0000,
            0x8002_0200,
            0xE070_0004,
            0x8002_0300,
            END,
        ],
    ]
    .concat();
    let program = |w: &[u32]| {
        let mut n = root_uav_program(w, [1; 3], false);
        n.layout.push(param(RootType::Constants, 18, 2, 6));
        n
    };
    let n = program(&words);
    let resources = track(&n).unwrap();
    let reads: Vec<_> = resources
        .constant_reads
        .values()
        .map(|r| (r.slot, r.dword, r.packed, r.offset))
        .collect();
    assert_eq!(reads, [(6, 0, false, 8), (6, 0, true, 12)]);
    words.insert(12, 0xBE85_0380); // Overwriting the high word of the offset breaks the proof.
    assert!(track(&program(&words)).is_err());
    let bytes = crate::dxbc::build(&[]);
    let cs = translate_program(&Container::parse(&bytes).unwrap(), &n).unwrap();
    let Some(host) = Host::test_device() else {
        return;
    };
    let (root, pso) = host.compute_pipeline(&cs).unwrap();
    let mut ud = vec![0; cs.user_data_dwords() as usize];
    for u in &cs.user_data {
        ud[u.index as usize] = match u.root {
            super::SCALAR_ADDRESS_SRV => 1,
            super::PACKED_ADDRESS_SRV => 2,
            _ => 0,
        };
    }
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![
                    (0, Root::Constants(ud)),
                    (
                        1,
                        Root::Table {
                            sampler: false,
                            index: 0,
                        },
                    ),
                ],
                groups: [1; 3],
                samplers: &[],
                views: &[
                    View::RawUav(2),
                    View::Buffer {
                        data: &[7, 5, 3, 1],
                        stride: Stride::Raw,
                    },
                    View::Buffer {
                        data: &[2, 4, 6, 8],
                        stride: Stride::Raw,
                    },
                ],
            },
        )
        .unwrap();
    assert_eq!(out[0], [3, 8]);
}

#[cfg(windows)]
#[test]
fn indexed_spilled_buffer_uses_the_matching_carry_and_gpu_offset() {
    use super::host::{Dispatch, Host, Root, Stride, View};
    let mut words = PTR.to_vec();
    words.extend([
        0xF404_0201,
        0xFA00_0000, // s[8:9]: output UAV
        0xF408_0101,
        0xFA00_0008, // s[4:7]: base low, high, index, spare
        0x8000_0604, // s0 = s4 + s6
        0x8201_8005, // s1 = s5 + carry of the low addition
        0x7E02_0200,
        0x7E04_0201, // v[1:2] = s[0:1]
        0xDC30_8000,
        0x037D_0001, // v3 = dword at this address
        0xBE8A_03C1,
        0xBE8B_03FF,
        0x3101_4FAC,
        0xE070_0000,
        0x8002_0300,
        END, // write v3 to the UAV
    ]);
    let program = |w: &[u32]| {
        let mut n = root_uav_program(w, [1; 3], false);
        n.layout.push(param(RootType::Constants, 18, 4, 6));
        n
    };
    let n = program(&words);
    assert!(
        track(&n)
            .unwrap()
            .constant_reads
            .values()
            .any(|r| r.index.is_some())
    );
    let mut wrong_high = words.clone();
    wrong_high[8] = 0x8201_8009; // UAV pair whose low word is missing from the sum
    assert!(track(&program(&wrong_high)).is_err());
    words.insert(8, 0xBF06_8004); // SCC is rewritten between the two additions
    assert!(track(&program(&words)).is_err());

    let bytes = crate::dxbc::build(&[]);
    let cs = translate_program(&Container::parse(&bytes).unwrap(), &n).unwrap();
    let Some(host) = Host::test_device() else {
        return;
    };
    let (root, pso) = host.compute_pipeline(&cs).unwrap();
    for (bias, index, expected) in [(0, 4, 0x77665544), (3, 2, 0x88776655), (4, u32::MAX, 0)] {
        let mut ud = vec![0; cs.user_data_dwords() as usize];
        for u in &cs.user_data {
            ud[u.index as usize] = match (u.root, u.dword) {
                (super::INDEXED_ADDRESS_SRV, 0) => 1 << 4 | bias,
                (RootType::Constants, 2) => index,
                _ => 0,
            };
        }
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![
                        (0, Root::Constants(ud)),
                        (
                            1,
                            Root::Table {
                                sampler: false,
                                index: 0,
                            },
                        ),
                    ],
                    groups: [1; 3],
                    samplers: &[],
                    views: &[
                        View::RawUav(1),
                        View::Buffer {
                            data: &[0x33221100, 0x77665544, 0xBBAA9988],
                            stride: Stride::Raw,
                        },
                    ],
                },
            )
            .unwrap();
        assert_eq!(out[0], [expected], "bias={bias}, index={index}");
    }
}

#[test]
fn phi_merges_reloads_only_for_the_same_spilled_descriptor() {
    let mut code = [
        PTR.as_slice(),
        &[
            0xBF06_8001,
            0xBF85_0003,
            0xF404_0101,
            0xFA00_0000,
            0xBF82_0002,
            0xF404_0101,
            0xFA00_0000,
            0xF400_0182,
            0xFA00_0008,
            END,
        ],
    ]
    .concat();
    let layout = || {
        vec![
            param(RootType::Descriptor, 16, 2, 3),
            param(RootType::Descriptor, 18, 2, 7),
        ]
    };
    let r = track(&native(&code, layout())).unwrap();
    assert_eq!(r.spill_loads.len(), 2);
    assert_ne!(r.spill_loads[0].0, r.spill_loads[1].0);
    assert_eq!((r.scalar[0].1.param, r.scalar[0].1.offset), (0, 8));
    code[9] = 0xFA00_0008;
    assert!(track(&native(&code, layout())).is_err());
}

#[cfg(windows)]
fn root_uav_program(words: &[u32], threads: [u32; 3], wave64: bool) -> NativeProgram {
    let mut n = native(words, vec![param(RootType::Descriptor, 16, 2, 3)]);
    n.wave64 = wave64;
    n.io = StageIo::Cs(ComputeIo {
        rsrc2: 4 << 1,
        threads,
    });
    n.used.push(UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Descriptor,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 16,
        slot: 3,
    });
    n
}

#[cfg(windows)]
#[test]
fn tbuffer_keeps_its_instruction_format_on_a_root_srv() {
    let words = [
        PTR.as_slice(),
        &[
            0xF404_0201,
            0xFA00_0000,
            0xBE8A_03C1,
            0xBE8B_0380,
            0xE828_1000,
            0x8002_0200,
            END, // The 8_UINT format is in the instruction.
        ],
    ]
    .concat();
    let mut n = root_uav_program(&words, [1; 3], false);
    n.used[0].kind = ResourceType::Srv;
    let bytes = crate::dxbc::build(&[]);
    let cs = translate_program(&Container::parse(&bytes).unwrap(), &n).unwrap();
    assert_eq!(cs.root_srvs, [3]);
}

#[cfg(windows)]
#[test]
fn phi_reads_the_spilled_word_only_on_the_edge_that_keeps_it() {
    use super::host::{Dispatch, Host, Root, View};
    let words = [
        PTR.as_slice(),
        &[
            0xF404_0201,
            0xFA00_0000,
            0xF404_0101,
            0xFA00_0008,
            0xBF06_8004,
            0xBF85_0001,
            0xBE85_0389,
            0xBE8A_03C1,
            0xBE8B_03FF,
            0x3101_4FAC,
            0x7E04_0205,
            0xE070_0000,
            0x8002_0200,
            END,
        ],
    ]
    .concat();
    let mut n = root_uav_program(&words, [1; 3], false);
    n.layout.push(param(RootType::Constants, 18, 2, 6));
    let bytes = crate::dxbc::build(&[]);
    let cs = translate_program(&Container::parse(&bytes).unwrap(), &n).unwrap();
    let Some(host) = Host::test_device() else {
        return;
    };
    let (root, pso) = host.compute_pipeline(&cs).unwrap();
    for (condition, expected) in [(0, 71), (1, 9)] {
        let mut ud = vec![0; cs.user_data_dwords() as usize];
        for u in cs
            .user_data
            .iter()
            .filter(|u| u.root == RootType::Constants)
        {
            ud[u.index as usize] = if u.dword == 0 { condition } else { 71 };
        }
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![
                        (0, Root::Constants(ud)),
                        (
                            1,
                            Root::Table {
                                sampler: false,
                                index: 0,
                            },
                        ),
                    ],
                    views: &[View::RawUav(1)],
                    samplers: &[],
                    groups: [1; 3],
                },
            )
            .unwrap();
        assert_eq!(out[0], [expected]);
    }
}

/// Ported from fxceless's VIctory fork #646 (head 170c24269) by mynzjerko.
/// A barrier separates the count of EXEC from the write to the root view.
#[cfg(windows)]
#[test]
fn partial_group_counts_lanes_and_preserves_the_end_of_the_root_uav() {
    use super::host::{Dispatch, Host, Root, View};
    let words = [
        PTR.as_slice(),
        &[
            0xF404_0201,
            0xFA00_0000, // s[8:9] receives the address of the UAV.
            0xBE8A_03C1,
            0xBE8B_03FF,
            0x3101_4FAC,
            0xBE86_107E, // s6 receives the population count of EXEC.
            0xBF8A_0000,
            0x3402_0082,
            0x7E04_0206,
            0xE070_1000,
            0x8002_0201,
            END,
        ],
    ]
    .concat();
    let n = root_uav_program(&words, [64, 1, 1], true);
    let bytes = crate::dxbc::build(&[]);
    let c = Container::parse(&bytes).unwrap();
    let Some(host) = Host::test_device() else {
        return;
    };
    for lanes in [32, 64] {
        let target = super::Target {
            lanes_min: lanes,
            lanes_max: lanes,
            stride_table: false,
        };
        let mut cs = super::translate_program_for(&c, &n, target).unwrap();
        assert_eq!(cs.root_uavs, [3]);
        if let Ok(mut validator) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            validator.validate(&mut cs.dxil).unwrap();
        }
        if !host.target().runs(lanes) {
            continue;
        }
        let (root, pso) = host.compute_pipeline(&cs).unwrap();
        for (limit, count) in [(45, 45), (1, 1), (0, 64)] {
            let mut ud = vec![0; cs.user_data_dwords() as usize];
            for word in &cs.user_data {
                ud[word.index as usize] = match word.root {
                    super::DISPATCH_THREADS => limit,
                    super::ROOT_UAV_BIAS => 12,
                    _ => panic!("unexpected binding {word:?}"),
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![
                            (0, Root::Constants(ud)),
                            (
                                1,
                                Root::Table {
                                    sampler: false,
                                    index: 0,
                                },
                            ),
                        ],
                        views: &[View::RawUav(68)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap();
            let mut expected = vec![0; 68];
            expected[3..3 + count].fill(count as u32);
            assert_eq!(out[0], expected, "wave {lanes}, limit {limit}");
        }
    }
}

#[cfg(windows)]
#[test]
fn root_halfword_crosses_a_dword_and_typed_access_preserves_its_value() {
    use super::host::{Dispatch, Host, Root, View};
    let words = [
        PTR.as_slice(),
        &[
            0xF404_0201,
            0xFA00_0000,
            0xBE8A_03C1,
            0xBE8B_03FF,
            0x3101_4FAC,
            0x7E04_02FF,
            0x12AB,
            0xE068_0003,
            0x8002_0200, // Half-word at byte 3.
            0xE028_0003,
            0x8002_0300,
            0xE010_000C,
            0x8002_0300, // Write and read in the R32_UINT format.
            0xE000_000C,
            0x8002_0400,
            0xE070_0010,
            0x8002_0400,
            0x7E08_0208,
            0x7E0A_0209,
            0xDC30_800C,
            0x067D_0004, // Same address through the VGPR pair.
            0xE070_0014,
            0x8002_0600,
            END,
        ],
    ]
    .concat();
    let n = root_uav_program(&words, [1; 3], false);
    let bytes = crate::dxbc::build(&[]);
    let cs = translate_program(&Container::parse(&bytes).unwrap(), &n).unwrap();
    let Some(host) = Host::test_device() else {
        return;
    };
    let (root, pso) = host.compute_pipeline(&cs).unwrap();
    let mut ud = vec![0; cs.user_data_dwords() as usize];
    let bias = cs
        .user_data
        .iter()
        .find(|u| u.root == super::ROOT_UAV_BIAS)
        .unwrap();
    ud[bias.index as usize] = 4;
    let out = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![
                    (0, Root::Constants(ud)),
                    (
                        1,
                        Root::Table {
                            sampler: false,
                            index: 0,
                        },
                    ),
                ],
                views: &[View::RawUav(8)],
                samplers: &[],
                groups: [1; 3],
            },
        )
        .unwrap();
    assert_eq!(out[0], [0, 0xAB00_0000, 0x12, 0, 0x12AB, 0x12AB, 0x12AB, 0]);
}

/// Global byte and half loads use their full byte address, including a half
/// at byte 3 of a dword. Signed loads extend the assembled field's sign bit.
#[cfg(windows)]
#[test]
fn global_subword_loads_preserve_both_bytes_and_signedness() {
    use super::host::{Dispatch, Host, Root, View};
    let Some(host) = Host::test_device() else {
        return;
    };
    let data = [0x80ff_7f01u32, 0x7fff_0081];
    let bytes: Vec<_> = data.iter().flat_map(|w| w.to_le_bytes()).collect();
    for (opcode, width, signed) in [(8, 1, false), (9, 1, true), (10, 2, false), (11, 2, true)] {
        for offset in (0..8u32).chain([14, u32::MAX]) {
            let overflow = offset == u32::MAX;
            let address = if overflow {
                [0x7E08_02FF, offset] // v4: scalar base's byte offset
            } else {
                [0x7E08_0208, 0x7E0A_0209] // v[4:5] = s[8:9]
            };
            let words = [
                PTR.as_slice(),
                &[
                    0xF404_0201,
                    0xFA00_0000, // s[8:9]: root UAV address
                    0xBE8A_03C1,
                    0xBE8B_03FF,
                    0x3101_4FAC,
                    0x7E04_02FF,
                    data[0],
                    0xE070_0000,
                    0x8002_0200, // store source dword 0
                    0x7E04_02FF,
                    data[1],
                    0xE070_0004,
                    0x8002_0200, // store source dword 1
                    0xBF8C_3F70, // s_waitcnt vmcnt(0)
                    address[0],
                    address[1],
                    0xDC00_8000 | (opcode << 18) | if overflow { 0 } else { offset },
                    if overflow { 0x0608_0004 } else { 0x067D_0004 }, // load into v6
                    0xBF8C_3F70,
                    0xE070_0008,
                    0x8002_0600, // store v6 at byte 8
                    END,
                ],
            ]
            .concat();
            let n = root_uav_program(&words, [1; 3], false);
            let container = crate::dxbc::build(&[]);
            let cs = translate_program(&Container::parse(&container).unwrap(), &n).unwrap();
            if let Ok(mut v) = super::validator::Validator::load(
                super::validator::Validator::default_path().as_deref(),
            ) {
                v.validate(&mut cs.dxil.clone()).unwrap();
            }
            let (root, pso) = host.compute_pipeline(&cs).unwrap();
            let at = offset as usize;
            let raw = if width == 1 {
                u32::from(bytes.get(at).copied().unwrap_or(0))
            } else {
                u32::from(u16::from_le_bytes([
                    bytes.get(at).copied().unwrap_or(0),
                    bytes.get(at + 1).copied().unwrap_or(0),
                ]))
            };
            let expected = match (width, signed) {
                (1, true) => raw as u8 as i8 as i32 as u32,
                (2, true) => raw as u16 as i16 as i32 as u32,
                _ => raw,
            };
            for bias in [0, 4] {
                if overflow && bias != 0 {
                    continue;
                }
                let mut ud = vec![0; cs.user_data_dwords() as usize];
                for u in &cs.user_data {
                    if u.root == super::ROOT_UAV_BIAS {
                        ud[u.index as usize] = bias;
                    }
                }
                let out = host
                    .dispatch(
                        &root,
                        &pso,
                        &Dispatch {
                            roots: vec![
                                (0, Root::Constants(ud)),
                                (
                                    1,
                                    Root::Table {
                                        sampler: false,
                                        index: 0,
                                    },
                                ),
                            ],
                            views: &[View::RawUav(4)],
                            samplers: &[],
                            groups: [1; 3],
                        },
                    )
                    .unwrap();
                let mut want = vec![0; 4];
                let first = bias as usize / 4;
                want[first..first + 3].copy_from_slice(&[data[0], data[1], expected]);
                assert_eq!(
                    out[0], want,
                    "opcode {opcode}, offset {offset}, bias {bias}"
                );
            }
        }
    }
}
