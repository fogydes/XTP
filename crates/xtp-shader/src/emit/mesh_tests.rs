//! Console wave64 mesh programs on hosts without 64-lane waves: their
//! first-lane reads, barriers and LDS (`lower/mesh.rs`, `lower/uniform.rs`,
//! `docs/analysis/mesh-wave64.md`). Hand-assembled GFX10 code, as in the
//! other tests of this crate: what each target's translation calls or
//! refuses, then one program drawn on the test adapter (WARP on this PC:
//! waves of 4 lanes, so a console wave is 16 host waves) and checked against
//! what the console's lanes give, in both console waves of its group.

use super::host::{Draw, Host};
use super::{HostShader, Target};
use crate::ShaderError;
use crate::emit::container::SigElement;
use crate::native::{MeshIo, NativeProgram, PsInput, StageIo, Usage, VsOutput};

const S_ENDPGM: u32 = 0xBF81_0000;
/// `exp prim, v9, off, off, off done`.
const EXP_PRIM: [u32; 2] = [0xF800_0941, 0x0000_0009];
/// `v_mbcnt_lo_u32_b32 v0, -1, 0`: the lane number (below 32).
const LANE: [u32; 2] = [0xD765_0000, 0x0001_00C1];
const V_MOV_V9_S5: u32 = 0x7E12_0205; // v_mov_b32 v9, s5
const SAVE_EXEC: u32 = 0xBE8A_047E; // s_mov_b64 s[10:11], exec
/// `ds_write_b32 v0, v1`: the program uses its LDS.
const DS_WRITE: [u32; 2] = [0xD834_0000, 0x0000_0100];
/// `ds_read_b32 v2, v0`.
const DS_READ: [u32; 2] = [0xD8D8_0000, 0x0200_0000];
const S_BARRIER: u32 = 0xBF8A_0000;
const RESTORE_EXEC: u32 = 0xBEFE_040A; // s_mov_b64 exec, s[10:11]

const NVIDIA: Target = Target {
    lanes_min: 32,
    lanes_max: 32,
    stride_table: false,
};
const RDNA: Target = Target {
    lanes_min: 32,
    lanes_max: 64,
    stride_table: false,
};
/// WARP on this PC: waves of 4 lanes.
const WARP: Target = Target {
    lanes_min: 4,
    lanes_max: 4,
    stride_table: false,
};

/// A console wave64 mesh program of `threads` lanes with `lds_bytes` of
/// LDS: `words`, then the primitive export of v9.
fn mesh(words: &[u32], threads: u32, lds_bytes: u32) -> NativeProgram {
    let mut code = words.to_vec();
    code.extend(EXP_PRIM);
    code.push(S_ENDPGM);
    NativeProgram {
        stage: crate::HwStage::Ms,
        wave64: true,
        float_mode: 0,
        vgprs: 24,
        layout: Vec::new(),
        used: Vec::new(),
        io: StageIo::Ms(MeshIo {
            outputs: Vec::new(),
            clip_cntl: 0,
            threads,
            max_vertices: 128,
            max_primitives: 128,
            topology: 2,
            lds_bytes,
        }),
        code: code.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

fn empty() -> crate::Container<'static> {
    crate::Container {
        digest: [0; 16],
        version: (1, 0),
        parts: Vec::new(),
    }
}

/// `native` translated for `target` (in a container of `parts`), and the
/// `dx.op` operations its module calls; checked by dxil.dll where it is.
fn translate(
    native: &NativeProgram,
    parts: &crate::Container,
    target: Target,
) -> Result<(HostShader, String), ShaderError> {
    let s = super::translate_program_for(parts, native, target).map_err(|(_, e)| e)?;
    let ops = super::last_module_summary();
    if let Ok(mut v) =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
    {
        v.validate(&mut s.dxil.clone())
            .unwrap_or_else(|e| panic!("validation: {e}\n{ops}"));
    }
    Ok((s, ops))
}

/// What the reasons of `lower/mesh.rs` say.
const FIRST_LANE: &str = "a first-lane read of a value lanes may hold differently";
const BARRIER: &str = "a group barrier lanes of a wave64 mesh group reach differently";
const LDS_IN_ONE_WAVE: &str = "LDS shared within one wave64 mesh wave";
const LDS_ORDER: &str =
    "LDS read after a write, or written after a read, with no group barrier between";

fn refusal(native: &NativeProgram, target: Target) -> Option<&'static str> {
    match translate(native, &empty(), target) {
        Ok(_) => None,
        Err(ShaderError::Unsupported { what, .. }) => Some(what),
        Err(e) => panic!("{e}"),
    }
}

/// A value the whole group holds is read as each lane's own, on any host:
/// no wave operation, no barrier.
#[test]
fn uniform_first_lane_reads_are_the_lanes_own() {
    let p = mesh(
        &[
            0xBE84_0387, // s_mov_b32 s4, 7
            0x7E02_0204, // v_mov_b32 v1, s4
            0x7E0A_0501, // v_readfirstlane_b32 s5, v1
            V_MOV_V9_S5,
        ],
        128,
        0,
    );
    for target in [NVIDIA, RDNA, Target::ANY] {
        let (_, ops) = translate(&p, &empty(), target).unwrap();
        assert!(!ops.contains("waveReadLaneFirst"), "{target:?}: {ops}");
        assert!(!ops.contains("barrier"), "{target:?}: {ops}");
    }
}

/// A lane's own value read from the first lane: on a host without 64-lane
/// waves, through group-shared memory between group barriers (EXEC full
/// and narrowed without a branch); with 64-lane waves, the host wave's
/// first lane as before.
#[test]
fn other_first_lane_reads_go_through_group_shared_memory() {
    let full_words = [
        LANE[0],
        LANE[1],
        0x7E0A_0500, // v_readfirstlane_b32 s5, v0
        V_MOV_V9_S5,
    ];
    let full = mesh(&full_words, 128, 0);
    let narrowed = mesh(
        &[
            LANE[0],
            LANE[1],
            SAVE_EXEC,
            0x7DA8_0094, // v_cmpx_gt_u32 exec, 20, v0
            0x7E0A_0500, // v_readfirstlane_b32 s5, v0
            RESTORE_EXEC,
            V_MOV_V9_S5,
        ],
        128,
        0,
    );
    for p in [&full, &narrowed] {
        let (_, ops) = translate(p, &empty(), NVIDIA).unwrap();
        assert!(ops.contains("barrier"), "{ops}");
        assert!(!ops.contains("waveReadLaneFirst"), "{ops}");
        let (_, ops) = translate(p, &empty(), RDNA).unwrap();
        assert!(ops.contains("waveReadLaneFirst"), "{ops}");
        assert!(!ops.contains("barrier"), "{ops}");
    }
    // Its slots fit next to LDS of the program's own, not in place of LDS
    // it never allocated.
    let with_lds = mesh(&[&DS_WRITE[..], &full_words].concat(), 128, 256);
    assert_eq!(refusal(&with_lds, NVIDIA), None);
}

/// Where lanes of the group branch apart, a first-lane read of a lane's
/// value has no form on a host without 64-lane waves (its exchange needs
/// every lane at a group barrier): refused with its reason; with 64-lane
/// waves it translates as before.
#[test]
fn first_lane_reads_where_lanes_branch_apart_are_refused() {
    let p = mesh(
        &[
            LANE[0],
            LANE[1],
            0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
            0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
            0xBF88_0002, // s_cbranch_execz +2
            0x7E0A_0500, // v_readfirstlane_b32 s5, v0
            V_MOV_V9_S5,
            0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
        ],
        128,
        0,
    );
    assert!(refusal(&p, NVIDIA).is_some_and(|w| w.starts_with(FIRST_LANE)));
    assert_eq!(refusal(&p, RDNA), None);
}

/// A program with LDS keeps its barriers: one in a block only some lanes
/// enter would hang a host wave that never gets there, and is refused on a
/// host without 64-lane waves; one where the lanes have met again is kept.
/// A group of one console wave that shares LDS is refused there (its
/// compiler left out the barriers between the halves).
#[test]
fn barriers_and_lds_of_wave64_mesh_groups() {
    let branch = [
        LANE[0],
        LANE[1],
        0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xBF88_0001, // s_cbranch_execz +1
    ];
    let inside = [&branch[..], &[0xBF8A_0000, 0x88FE_0A7E]].concat(); // s_barrier; s_or_b64 exec, exec, s[10:11]
    let after = [&branch[..], &[0x7E12_0280, 0x88FE_0A7E, 0xBF8A_0000]].concat(); // v_mov_b32 v9, 0; join; s_barrier
    let inside = mesh(&[&DS_WRITE[..], &inside].concat(), 128, 256);
    let after = mesh(&[&DS_WRITE[..], &after].concat(), 128, 256);
    assert!(refusal(&inside, NVIDIA).is_some_and(|w| w.starts_with(BARRIER)));
    assert_eq!(refusal(&inside, RDNA), None);
    let (_, ops) = translate(&after, &empty(), NVIDIA).unwrap();
    assert!(ops.contains("barrier"), "{ops}");
    let one_wave = mesh(&[DS_WRITE[0], DS_WRITE[1], 0x7E12_0280], 64, 256); // v_mov_b32 v9, 0
    assert!(refusal(&one_wave, NVIDIA).is_some_and(|w| w.starts_with(LDS_IN_ONE_WAVE)));
    assert_eq!(refusal(&one_wave, RDNA), None);
}

/// A group of one console wave spans several host waves as soon as it has
/// more lanes than the host's smallest wave: 32 lanes are one NVIDIA wave,
/// but eight of WARP's.
#[test]
fn lds_in_one_wave_counts_the_hosts_smallest_wave() {
    let p = mesh(&[DS_WRITE[0], DS_WRITE[1], 0x7E12_0280], 32, 256); // v_mov_b32 v9, 0
    assert_eq!(refusal(&p, NVIDIA), None);
    assert!(refusal(&p, WARP).is_some_and(|w| w.starts_with(LDS_IN_ONE_WAVE)));
}

/// One console wave's LDS accesses run in order, two host waves' do not:
/// in a group of several console waves, an LDS read after a write, or a
/// write after a read, needs a group barrier between them on a host
/// without 64-lane waves, else the program is refused. On the console's
/// paths: the next iteration of a loop, and the two sides of a branch
/// lanes take apart (which no lane runs both of, but the console wave
/// runs one after the other). Writes after writes, then a barrier, then
/// reads (BTP 6's five mesh programs with LDS) translate.
#[test]
fn lds_order_within_a_console_wave() {
    let p = |words: &[u32]| mesh(&[words, &[0x7E12_0280]].concat(), 128, 256); // v_mov_b32 v9, 0
    let ordered = p(&[&DS_WRITE[..], &DS_WRITE, &[S_BARRIER], &DS_READ].concat());
    let (_, ops) = translate(&ordered, &empty(), NVIDIA).unwrap();
    assert!(ops.contains("barrier"), "{ops}");
    let read_after_write = p(&[DS_WRITE, DS_READ].concat());
    let write_after_read = p(&[DS_READ, DS_WRITE].concat());
    let iteration = |barrier_after_read: bool| {
        let mut words = vec![0xBE84_0380]; // s_mov_b32 s4, 0
        words.extend(DS_WRITE); // loop:
        words.push(S_BARRIER);
        words.extend(DS_READ);
        if barrier_after_read {
            words.push(S_BARRIER);
        }
        words.extend([
            0x8104_8104, // s_add_i32 s4, s4, 1
            0xBF0A_8404, // s_cmp_lt_u32 s4, 4
            if barrier_after_read {
                0xBF85_FFF7 // s_cbranch_scc1 loop
            } else {
                0xBF85_FFF8 // s_cbranch_scc1 loop
            },
        ]);
        p(&words)
    };
    let sides = p(&[
        LANE[0],
        LANE[1],
        0x7D88_0094, // v_cmp_gt_u32 vcc, 20, v0
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xBF88_0002, // s_cbranch_execz else
        DS_WRITE[0],
        DS_WRITE[1],
        0x8AFE_7E0A, // else: s_andn2_b64 exec, s[10:11], exec
        0xBF88_0002, // s_cbranch_execz end
        DS_READ[0],
        DS_READ[1],
        0x88FE_0A7E, // end: s_or_b64 exec, exec, s[10:11]
    ]);
    let text = |p: &NativeProgram| -> Vec<String> {
        crate::gfx10::decode(&p.code)
            .unwrap()
            .iter()
            .map(|i| i.text(p.wave64))
            .collect()
    };
    for (p, want) in [
        (
            &ordered,
            &["ds_write_b32 v0, v1", "s_barrier", "ds_read_b32 v2, v0"][..],
        ),
        (&iteration(false), &["s_cbranch_scc1 label_0004"]),
        (&iteration(true), &["s_cbranch_scc1 label_0004"]),
        (
            &sides,
            &[
                "s_cbranch_execz label_001C",
                "s_andn2_b64 exec, s[10:11], exec",
                "s_cbranch_execz label_002C",
            ],
        ),
    ] {
        let t = text(p);
        for w in want {
            assert!(t.iter().any(|x| x == w), "{w}: {t:#?}");
        }
    }
    assert_eq!(refusal(&iteration(true), NVIDIA), None);
    for (p, what) in [
        (&read_after_write, "a read after a write"),
        (&write_after_read, "a write after a read"),
        (&iteration(false), "the next iteration's write"),
        (&sides, "the two sides of a lane branch"),
    ] {
        assert!(
            refusal(p, NVIDIA).is_some_and(|w| w.starts_with(LDS_ORDER)),
            "{what}: {:?}",
            refusal(p, NVIDIA)
        );
        assert!(
            refusal(p, WARP).is_some_and(|w| w.starts_with(LDS_ORDER)),
            "{what}"
        );
        assert_eq!(refusal(p, RDNA), None, "{what}");
    }
}

/// The console lane in v3 and thread in v0 for the strip draws.
const STRIP_THREAD: &[u32] = &[
    0xD765_0000,
    0x0001_00C1, // v_mbcnt_lo_u32_b32 v0, -1, 0
    0xD766_0000,
    0x0002_00C1, // v_mbcnt_hi_u32_b32 v0, -1, v0
    0x7E06_0300, // v_mov_b32 v3, v0 (the lane)
    0x9382_FF03,
    0x0004_0018, // s_bfe_u32 s2, s3, lit(0x00040018) (the wave)
    0xD746_0000,
    0x0401_0C02, // v_lshl_add_u32 v0, s2, 6, v0 (the thread)
];

/// The program [`wave64_mesh_reads_draw_the_console_waves_values`] draws.
/// Thread t of the group (console wave w = t / 64, lane l = t % 64)
/// exports vertex t at x = t / 2 of 63 columns, at the top when t is even
/// and the bottom when odd, and triangle (t, t + 1, t + 2) below 126: a
/// strip in which texel (k, row r) shows vertex 2k + r (the provoking
/// vertex). Its attribute is what five first-lane reads give, the
/// exchanged values made from the thread, so that each console wave's
/// first lane gives its own:
/// * 7 + 16 w: 7, a value of the group's (read as the lane's own), and w,
///   the console wave's index (the wave's), added on the scalar side;
/// * t + 100 from the console wave's first lane: 64 w + 100 (an exchange,
///   EXEC full);
/// * 2 t + 1, right after it: 128 w + 1 (a second exchange, back to back
///   with the first, through the same slots);
/// * 3 t + 7 with EXEC holding the lanes above 9 + 20 w only, lane 10 of
///   wave 0 and lane 30 of wave 1 first: 37 and 289 (an exchange, EXEC
///   narrowed differently in each console wave; VIctory issue #1132's
///   case, where the host wave's own first lane gave lanes 32-63 the wrong
///   value).
fn drawn_mesh() -> NativeProgram {
    let words = [
        0xBE84_0387, // s_mov_b32 s4, 7
        0x7E02_0204, // v_mov_b32 v1, s4
        0x7E0A_0501, // v_readfirstlane_b32 s5, v1
        0x7E02_0202, // v_mov_b32 v1, s2
        0x7E0C_0501, // v_readfirstlane_b32 s6, v1
        0x4A04_00FF,
        0x0000_0064, // v_add_nc_u32 v2, lit(100), v0
        0xD543_0004,
        0x0205_0500, // v_mad_u32_u24 v4, v0, 2, 1
        0x7E0E_0502, // v_readfirstlane_b32 s7, v2
        0x7E10_0504, // v_readfirstlane_b32 s8, v4
        0x930C_9402, // s_mul_i32 s12, s2, 20
        0x800C_890C, // s_add_u32 s12, s12, 9
        0xD543_0001,
        0x021D_0700, // v_mad_u32_u24 v1, v0, 3, 7
        SAVE_EXEC,
        0x7DA2_060C, // v_cmpx_lt_u32 exec, s12, v3
        0x7E12_0501, // v_readfirstlane_b32 s9, v1
        RESTORE_EXEC,
        0x8F0D_8406, // s_lshl_b32 s13, s6, 4
        0x800D_0D05, // s_add_u32 s13, s5, s13
        0x7E14_0C0D, // v_cvt_f32_u32 v10, s13
        0x7E16_0C07, // v_cvt_f32_u32 v11, s7
        0x7E18_0C08, // v_cvt_f32_u32 v12, s8
        0x7E1A_0C09, // v_cvt_f32_u32 v13, s9
    ];
    strip_mesh([STRIP_THREAD, &words].concat(), false)
}

/// Finish the strip with vertex positions and indices. A rotated triangle
/// has the same coverage with its last vertex moved to the provoking slot.
fn strip_mesh(mut words: Vec<u32>, rotate: bool) -> NativeProgram {
    words.extend([
        0x2C0A_0081, // v_lshrrev_b32 v5, 1, v0
        0x7E0A_0D05, // v_cvt_f32_u32 v5, v5
        0x100A_0AFF,
        0x3D02_0821, // v_mul_f32 v5, lit(2 / 63), v5
        0x060A_0AF3, // v_add_f32 v5, -1.0, v5
        0x360C_0081, // v_and_b32 v6, 1, v0
        0x7E0C_0D06, // v_cvt_f32_u32 v6, v6
        0x100C_0CF5, // v_mul_f32 v6, -2.0, v6
        0x060C_0CF2, // v_add_f32 v6, 1.0, v6
        0x7E0E_02F0, // v_mov_b32 v7, 0.5
        0x7E10_02F2, // v_mov_b32 v8, 1.0
        0xF800_08CF,
        0x0807_0605, // exp pos0, v5, v6, v7, v8 done
        0xF800_020F,
        0x0D0C_0B0A, // exp param0, v10, v11, v12, v13
        0x4A1C_0081, // v_add_nc_u32 v14, 1, v0
        0x4A1E_0082, // v_add_nc_u32 v15, 2, v0
    ]);
    if rotate {
        words.extend([
            0xD76F_0009,
            0x043D_1500, // v_lshl_or_b32 v9, v0, 10, v15
            0xD76F_0009,
            0x0425_290E, // v_lshl_or_b32 v9, v14, 20, v9
        ]);
    } else {
        words.extend([
            0xD76F_0009,
            0x0401_150E, // v_lshl_or_b32 v9, v14, 10, v0
            0xD76F_0009,
            0x0425_290F, // v_lshl_or_b32 v9, v15, 20, v9
        ]);
    }
    words.extend([
        0x7D82_00FF,
        0x0000_007D, // v_cmp_lt_u32 vcc, lit(125), v0
        0x7E20_7081, // v_bfrev_b32 v16, 1 (the null primitive)
        0x0212_2109, // v_cndmask_b32 v9, v9, v16, vcc
    ]);
    let mut p = mesh(&words, 128, 0);
    let StageIo::Ms(ms) = &mut p.io else {
        unreachable!()
    };
    ms.outputs = vec![VsOutput {
        usage: Usage::Generic,
        row: 1,
        param: 0,
        fp16: false,
        pack2: false,
    }];
    p
}

/// A pixel program that writes its attribute 0 as the console's flat
/// interpolation gives it (the provoking vertex's).
fn flat_pixel() -> NativeProgram {
    let words = [
        0xC802_0002, // v_interp_mov_f32 v0, p0, attr0.x
        0xC806_0102, // v_interp_mov_f32 v1, p0, attr0.y
        0xC80A_0202, // v_interp_mov_f32 v2, p0, attr0.z
        0xC80E_0302, // v_interp_mov_f32 v3, p0, attr0.w
        0xF800_180F,
        0x0302_0100, // exp mrt0, v0, v1, v2, v3 done vm
        S_ENDPGM,
    ];
    NativeProgram {
        stage: crate::HwStage::Ps,
        wave64: false,
        float_mode: 0,
        vgprs: 4,
        layout: Vec::new(),
        used: Vec::new(),
        io: StageIo::Ps {
            inputs: vec![PsInput {
                usage: Usage::Generic,
                row: 1,
                mask: 0b1111,
                attr: 0,
                flat: true,
                fp16: false,
                pack2: false,
                row2: 0,
            }],
            input_addr: 0,
            rsrc2: 0,
            db_shader_control: 0,
            patches: Vec::new(),
        },
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

fn element(name: &str, kind: u8, row: u32, usage: u8) -> SigElement {
    SigElement {
        name: name.into(),
        index: 0,
        kind,
        comp: 9,
        interp: 0,
        row: Some(row),
        col: 0,
        cols: 4,
        usage,
        half: false,
    }
}

/// The hand-assembled programs read as written (a wrong word would test
/// another program).
#[test]
fn the_programs_decode_as_written() {
    let text = |p: &NativeProgram| -> Vec<String> {
        crate::gfx10::decode(&p.code)
            .unwrap()
            .iter()
            .map(|i| i.text(p.wave64))
            .collect()
    };
    let ms = text(&drawn_mesh());
    for (at, want) in [
        (0, "v_mbcnt_lo_u32_b32 v0, -1, 0"),
        (3, "s_bfe_u32 s2, s3, lit(0x00040018)"),
        (4, "v_lshl_add_u32 v0, s2, 6, v0"),
        (10, "v_add_nc_u32 v2, lit(0x00000064), v0"),
        (11, "v_mad_u32_u24 v4, v0, 2, 1"),
        (12, "v_readfirstlane_b32 s7, v2"),
        (13, "v_readfirstlane_b32 s8, v4"),
        (14, "s_mul_i32 s12, s2, 20"),
        (15, "s_add_u32 s12, s12, 9"),
        (16, "v_mad_u32_u24 v1, v0, 3, 7"),
        (18, "v_cmpx_lt_u32 exec, s12, v3"),
        (19, "v_readfirstlane_b32 s9, v1"),
        (21, "s_lshl_b32 s13, s6, 4"),
        (22, "s_add_u32 s13, s5, s13"),
        (37, "exp pos0, v5, v6, v7, v8 done"),
        (38, "exp param0, v10, v11, v12, v13"),
        (41, "v_lshl_or_b32 v9, v14, 10, v0"),
        (43, "v_cmp_lt_u32 vcc, lit(0x0000007d), v0"),
        (45, "v_cndmask_b32 v9, v9, v16, vcc"),
        (46, "exp prim, v9, off, off, off done"),
    ] {
        assert_eq!(ms[at], want, "{ms:#?}");
    }
    let ps = text(&flat_pixel());
    assert_eq!(ps[3], "v_interp_mov_f32 v3, p0, attr0.w");
    assert_eq!(ps[4], "exp mrt0, v0, v1, v2, v3 done vm");
}

/// The mesh program `native`, whose one attribute is TEXCOORD in row 1,
/// drawn with [`flat_pixel`] into `size` texels on the test adapter (WARP
/// on this PC), translated for the adapter's waves: the target, the mesh
/// module's operations and the texels. `None` without an SM 6.6 device.
fn draw_flat(native: &NativeProgram, size: (u32, u32)) -> Option<(Target, String, Vec<[f32; 4]>)> {
    let host = Host::test_device()?;
    Some(draw_flat_on(&host, native, size))
}

pub(super) fn draw_flat_on(
    host: &Host,
    native: &NativeProgram,
    size: (u32, u32),
) -> (Target, String, Vec<[f32; 4]>) {
    let target = host.target();
    let (ms, ops) = translate_flat_mesh(native, target);
    let ps_sig = crate::dxbc::build(&[
        (
            crate::FourCC::ISG1,
            &super::container::signature(
                &[
                    element("SV_Position", 3, 0, 0),
                    element("TEXCOORD", 0, 1, 15),
                ],
                true,
            ),
        ),
        (
            crate::FourCC::OSG1,
            &super::container::signature(&[element("SV_Target", 16, 0, 15)], false),
        ),
    ]);
    let ps_c = crate::Container::parse(&ps_sig).unwrap();
    let (ps, _) = translate(&flat_pixel(), &ps_c, target).unwrap();
    let (root, pso) = host.mesh_pipeline(&ms, &ps).unwrap();
    let texels = host
        .draw(
            &root,
            &pso,
            &Draw {
                roots: Vec::new(),
                textures: &[],
                samplers: &[],
                vertices: &[],
                stride: 0,
                size,
                mesh_groups: 1,
            },
        )
        .unwrap();
    eprintln!("{target:?}: {} texels drawn", texels.len());
    (target, ops, texels)
}

pub(super) fn translate_flat_mesh(native: &NativeProgram, target: Target) -> (HostShader, String) {
    let sig = crate::dxbc::build(&[(
        crate::FourCC::OSG1,
        &super::container::signature(
            &[
                element("SV_Position", 3, 0, 15),
                element("TEXCOORD", 0, 1, 15),
            ],
            false,
        ),
    )]);
    translate(native, &crate::Container::parse(&sig).unwrap(), target).unwrap()
}

/// [`drawn_mesh`] drawn into 63 x 2 texels on the test adapter, for the
/// adapter's waves: every texel has the console waves' values.
#[test]
fn wave64_mesh_reads_draw_the_console_waves_values() {
    let Some((target, ops, texels)) = draw_flat(&drawn_mesh(), (63, 2)) else {
        return;
    };
    if !target.runs(64) {
        assert!(ops.contains("barrier"), "exchanges on {target:?}: {ops}");
    }
    let mut wrong = Vec::new();
    for (n, t) in texels.iter().enumerate() {
        let (k, row) = (n % 63, n / 63);
        let v = 2 * k + row;
        let w = (v / 64) as u32;
        let narrowed = [37, 289][w as usize];
        let want = [7 + 16 * w, 64 * w + 100, 128 * w + 1, narrowed].map(|x| x as f32);
        if *t != want {
            wrong.push(format!(
                "texel ({k}, {row}), vertex {v}: {t:?}, not {want:?}"
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {} texels wrong on {target:?}:\n{}",
        wrong.len(),
        texels.len(),
        wrong[..wrong.len().min(16)].join("\n")
    );
}

/// Each thread picks its own VGPR from 128 distinct values. Only the
/// threads whose attributes the draw samples enter the waterfall.
fn drawn_index_waterfall(rotate: bool) -> NativeProgram {
    let mut words = STRIP_THREAD.to_vec();
    for i in 0..128 {
        words.extend([
            0x7E00_02FF | ((32 + i) << 17), // v_mov_b32 v[32 + i], lit
            1000 + 3 * i,
        ]);
    }
    words.extend([
        if rotate { 0x7D86_00FF } else { 0x7D88_00FF },
        if rotate { 2 } else { 126 }, // vcc: thread >= 2 or thread < 126
        0xBE8A_246A,                  // s_and_saveexec_b64 s[10:11], vcc
        0xBF88_000A,                  // s_cbranch_execz to the outer restore
        0xBE8C_047E,                  // s_mov_b64 s[12:13], exec
        0x7E08_0500,                  // v_readfirstlane_b32 s4, v0
        0x7D84_0004,                  // v_cmp_eq_u32 vcc, s4, v0
        0xBEEA_246A,                  // s_and_saveexec_b64 vcc, vcc
        0xBEFC_0304,                  // s_mov_b32 m0, s4
        0x7E14_8720,                  // v_movrels_b32 v10, v32
        0x89FE_6A7E,                  // s_xor_b64 exec, exec, vcc
        0xBF89_FFF9,                  // s_cbranch_execnz to the read
        0xBEFE_040C,                  // s_mov_b64 exec, s[12:13]
        0x7E14_0D0A,                  // v_cvt_f32_u32 v10, v10
        RESTORE_EXEC,
    ]);
    let mut p = strip_mesh(words, rotate);
    p.vgprs = 160;
    p
}

#[test]
fn vgpr_index_waterfall_translates_inside_a_divergent_mesh_branch() {
    let p = drawn_index_waterfall(false);
    let (_, ops) = translate_flat_mesh(&p, NVIDIA);
    assert!(!ops.contains("waveReadLaneFirst"), "{ops}");
    assert!(!ops.contains("barrier"), "{ops}");
    let (_, ops) = translate_flat_mesh(&p, RDNA);
    assert!(ops.contains("waveReadLaneFirst"), "{ops}");
}

#[test]
fn vgpr_index_waterfall_draws_each_lanes_own_value_on_warp() {
    let Some(host) = Host::test_warp() else {
        return;
    };
    for rotate in [false, true] {
        let p = drawn_index_waterfall(rotate);
        let (target, ops, texels) = draw_flat_on(&host, &p, (63, 2));
        assert!(!target.runs(64), "{target:?}");
        assert!(!ops.contains("waveReadLaneFirst"), "{ops}");
        assert!(!ops.contains("barrier"), "{ops}");
        assert_eq!(texels.len(), 126);
        for (n, texel) in texels.into_iter().enumerate() {
            let thread = 2 * (n % 63) + n / 63 + 2 * usize::from(rotate);
            assert_eq!(
                texel,
                [(1000 + 3 * thread) as f32, 0.0, 0.0, 0.0],
                "thread {thread}"
            );
        }
    }
}

/// The loop limit's drain (`lower/flow.rs`: lanes the limit ends keep
/// passing group barriers) on mesh programs. Its counts of ended threads
/// are group-shared memory of their own, which the mesh PSV must count
/// with the program's LDS (dxil.dll checks it, [`translate`]): with
/// 64-lane waves the program below drains. A wave64 mesh program in halves
/// on a host without 64-lane waves does not: its first-lane exchanges add
/// barriers that no block counts (a debug assertion of the drain). The
/// program reads a lane's value from the first lane (an exchange on 32-lane
/// hosts), then runs the loop of [`lds_order_within_a_console_wave`]:
/// write, barrier, read, barrier.
#[test]
fn the_loop_limit_drain_of_mesh_programs() {
    let words = [
        LANE[0],
        LANE[1],
        0x7E0A_0500, // v_readfirstlane_b32 s5, v0
        0xBE84_0380, // s_mov_b32 s4, 0
        DS_WRITE[0], // loop: ds_write_b32 v0, v1
        DS_WRITE[1],
        S_BARRIER,
        DS_READ[0], // ds_read_b32 v2, v0
        DS_READ[1],
        S_BARRIER,
        0x8104_8104, // s_add_i32 s4, s4, 1
        0xBF0A_8404, // s_cmp_lt_u32 s4, 4
        0xBF85_FFF7, // s_cbranch_scc1 loop
        V_MOV_V9_S5,
    ];
    let p = mesh(&words, 128, 256);
    let text: Vec<String> = crate::gfx10::decode(&p.code)
        .unwrap()
        .iter()
        .map(|i| i.text(true))
        .collect();
    assert_eq!(text[9], "s_cbranch_scc1 label_0010", "{text:#?}");
    let (_, ops) = translate(&p, &empty(), NVIDIA).unwrap();
    assert!(!ops.contains("waveReadLaneFirst"), "an exchange: {ops}");
    let (_, ops) = translate(&p, &empty(), RDNA).unwrap();
    assert!(ops.contains("waveReadLaneFirst"), "{ops}");
}

/// Triangle (0, 1, 2), each corner's 9-bit index in its 10-bit field.
pub(super) const TRIANGLE: u32 = 1 << 10 | 2 << 20;
/// The corners' edge flags (bits 9, 19 and 29).
pub(super) const EDGE_FLAGS: u32 = 1 << 9 | 1 << 19 | 1 << 29;
/// The culling bit.
pub(super) const CULL: u32 = 1 << 31;

/// Ported from Marvisak/VIctory#1718 (head c0228d9) by tjthebigbearcat-hub:
/// a wave32 mesh program of 32 lanes, each exporting the words in `primitives` and,
/// as vertex `lane`, the position (lane & 1 ? 3 : -1, lane & 2 ? 3 : -1,
/// 0.5, 1), so that triangle (0, 1, 2) covers the whole target, with the
/// attribute (1, 0.5, 0.25, 1).
pub(super) fn triangle_mesh(primitives: &[u32]) -> NativeProgram {
    let mut words = vec![
        0xD765_0001,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
        0x3604_0281, // v_and_b32 v2, 1, v1
        0x3606_0282, // v_and_b32 v3, 2, v1
        0x7E04_0D02, // v_cvt_f32_u32 v2, v2
        0x7E06_0D03, // v_cvt_f32_u32 v3, v3
        0x1008_04F6, // v_mul_f32 v4, 4.0, v2
        0x0608_08F3, // v_add_f32 v4, -1.0, v4
        0x100A_06F4, // v_mul_f32 v5, 2.0, v3
        0x060A_0AF3, // v_add_f32 v5, -1.0, v5
        0x7E0C_02F0, // v_mov_b32 v6, 0.5
        0x7E0E_02F2, // v_mov_b32 v7, 1.0
        0x7E14_02F2, // v_mov_b32 v10, 1.0
        0x7E16_02F0, // v_mov_b32 v11, 0.5
        0x7E18_02FF,
        0x3E80_0000, // v_mov_b32 v12, lit(0.25)
        0x7E1A_02F2, // v_mov_b32 v13, 1.0
        0xF800_08CF,
        0x0706_0504, // exp pos0, v4, v5, v6, v7 done
        0xF800_020F,
        0x0D0C_0B0A, // exp param0, v10, v11, v12, v13
    ];
    for (i, &packed) in primitives.iter().enumerate() {
        let done = u32::from(i + 1 == primitives.len()) << 11;
        words.extend([0x7E12_02FF, packed, 0xF800_0141 | done, 9]);
    }
    words.push(S_ENDPGM);
    NativeProgram {
        stage: crate::HwStage::Ms,
        wave64: false,
        float_mode: 0,
        vgprs: 16,
        layout: Vec::new(),
        used: Vec::new(),
        io: StageIo::Ms(MeshIo {
            outputs: vec![VsOutput {
                usage: Usage::Generic,
                row: 1,
                param: 0,
                fp16: false,
                pack2: false,
            }],
            clip_cntl: 0,
            threads: 32,
            max_vertices: 32,
            max_primitives: 32,
            topology: 2,
            lds_bytes: 0,
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// Ported from Marvisak/VIctory#1718 (head c0228d9) by tjthebigbearcat-hub
/// (`edge_flagged_triangles_draw_as_unflagged_ones`): a primitive export's
/// corners are 9-bit indices, and the edge flag above each is no part of
/// it. On the test adapter, triangle (0, 1, 2) covers the target with its
/// corners' edge flags set as without them; with the culling bit, nothing
/// is lit. (A 10-bit mask read the flagged corners as vertices 512 to 514,
/// past the program's 32.) Each module is checked by dxil.dll where it is.
#[test]
fn edge_flagged_triangles_draw_as_unflagged_ones() {
    let text: Vec<String> = crate::gfx10::decode(&triangle_mesh(&[TRIANGLE]).code)
        .unwrap()
        .iter()
        .map(|i| i.text(false))
        .collect();
    for (at, want) in [
        (5, "v_mul_f32 v4, 4.0, v2"),
        (13, "v_mov_b32 v12, lit(0x3e800000)"),
        (15, "exp pos0, v4, v5, v6, v7 done"),
        (16, "exp param0, v10, v11, v12, v13"),
        (18, "exp prim, v9, off, off, off done"),
    ] {
        assert_eq!(text[at], want, "{text:#?}");
    }
    for (prim, culled) in [
        (TRIANGLE, false),
        (TRIANGLE | EDGE_FLAGS, false),
        (TRIANGLE | EDGE_FLAGS | CULL, true),
    ] {
        let Some((target, _, texels)) = draw_flat(&triangle_mesh(&[prim]), (16, 16)) else {
            return;
        };
        let lit = texels.iter().filter(|t| t[3] != 0.0).count();
        if culled {
            assert_eq!(
                lit, 0,
                "{target:?}, {prim:#x}: a culled triangle lit texels"
            );
            continue;
        }
        assert!(
            texels.iter().all(|t| *t == [1.0, 0.5, 0.25, 1.0]),
            "{target:?}, {prim:#x}: {lit} of {} texels lit",
            texels.len()
        );
    }
}
