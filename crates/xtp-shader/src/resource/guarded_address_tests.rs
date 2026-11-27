use crate::emit::{HostShader, translate_program};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{Container, ShaderError};

const VCC: u32 = 106;
const EXEC: u32 = 126;
const ZERO: u32 = 128;

/// A wave initializes the table pointer's high SGPR if it has any thread
/// with id below three. A second guard controls the access after the join.
fn translate(high: u32, guard: u32) -> Result<HostShader, ShaderError> {
    let words = [
        0xBE82_0300,         // s2 = table base from s0
        0x7D88_0083,         // vcc_lo = 3 > thread id
        0xBE8A_3C6A,         // save exec_lo to s10 and mask with vcc_lo
        0xBF88_0001,         // skip the assignment if exec is empty
        0xBE83_0300 | high,  // s_mov_b32 s3, high
        0xBEFE_030A,         // restore exec_lo from s10
        0xBE8A_3C00 | guard, // s_and_saveexec_b32 s10, guard
        0xBF88_0006,         // skip to end if exec is empty
        0xF408_0301,
        0xFA00_0000, // load the descriptor into s[12:15] via s[2:3]
        0x7E02_0293, // v1 = 19
        0x3404_0082, // v2 = thread id * 4
        0xE070_1000,
        0x8003_0102, // store v1 at byte v2 of the UAV
        0xBF81_0000, // s_endpgm
    ];
    let native = NativeProgram {
        stage: HwStage::Cs,
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1,
            threads: [4, 1, 1],
        }),
        wave64: false,
        float_mode: 0,
        vgprs: 8,
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }],
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let bytes = crate::dxbc::build(&[]);
    translate_program(&Container::parse(&bytes)?, &native).map_err(|(_, e)| e)
}

#[test]
fn traces_the_high_word_only_on_a_proven_wave_path() {
    translate(ZERO, VCC).unwrap();
    for (high, guard) in [(ZERO + 1, VCC), (ZERO, EXEC)] {
        assert!(matches!(
            translate(high, guard),
            Err(ShaderError::Unsupported {
                what: "an address with an untracked high half",
                ..
            })
        ));
    }
}

#[cfg(windows)]
#[test]
fn guarded_table_load_validates_and_writes_only_active_lanes_on_warp() {
    use crate::emit::host::{Dispatch, Host, Root, View};
    use crate::emit::validator::Validator;
    use std::io::{Write, stderr};

    let mut shader = translate(ZERO, VCC).unwrap();
    match Validator::load(Validator::default_path().as_deref()) {
        Ok(mut v) => v.validate(&mut shader.dxil).unwrap(),
        Err(e) => {
            let _ = writeln!(
                stderr().lock(),
                "SKIPPED guarded table load validation: {e}"
            );
        }
    }
    let host = match Host::new(true) {
        Ok(host) => host,
        Err(e) => {
            assert!(
                std::env::var("XTP_REQUIRE_GPU").as_deref() != Ok("1"),
                "{e}"
            );
            let _ = writeln!(
                stderr().lock(),
                "SKIPPED guarded table load WARP dispatch: {e}"
            );
            return;
        }
    };
    let (root, pso) = host.compute_pipeline(&shader).unwrap();
    let result = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(
                    0,
                    Root::Constants(vec![0; shader.user_data_dwords() as usize]),
                )],
                groups: [1, 1, 1],
                views: &[View::RawUav(4)],
                samplers: &[],
            },
        )
        .unwrap();
    assert_eq!(result[0], [19, 19, 19, 0]);
}

/// A wave loads the UAV buffer descriptor only in the live arm skipped by
/// `s_cbranch_execz`, leaving `s[12:15]` uninitialized on the skip edge. The
/// join block runs `buffer_store_dword` directly under the narrowed EXEC guard.
#[cfg(windows)]
#[test]
fn guarded_join_buffer_store_validates_and_writes_only_active_lanes_on_warp() {
    use crate::emit::host::{Dispatch, Host, Root, View};
    use crate::emit::validator::Validator;
    use std::io::{Write, stderr};

    let words = [
        0xBE82_0300u32, // s_mov_b32 s2, s0
        0xBE83_0380,    // s_mov_b32 s3, 0
        0x7E02_0293,    // v_mov_b32 v1, 19
        0x3404_0082,    // v_lshlrev_b32 v2, 2, v0
        0x7D88_0083,    // v_cmp_gt_u32 vcc_lo, 3, v0
        0x877E_6A7E,    // s_and_b32 exec_lo, exec_lo, vcc_lo
        0xBF88_0003,    // s_cbranch_execz -> join (skips 3 dwords)
        0xF408_0301,
        0xFA00_0000, // s_load_dwordx4 s[12:15], s[2:3], null
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE070_1000,
        0x8003_0102, // join: buffer_store_dword v1, v2, s[12:15], 0 offen
        0xBF81_0000, // s_endpgm
    ];
    let native = NativeProgram {
        stage: HwStage::Cs,
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1,
            threads: [4, 1, 1],
        }),
        wave64: false,
        float_mode: 0,
        vgprs: 8,
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }],
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let bytes = crate::dxbc::build(&[]);
    let mut shader = translate_program(&Container::parse(&bytes).unwrap(), &native).unwrap();
    match Validator::load(Validator::default_path().as_deref()) {
        Ok(mut v) => v.validate(&mut shader.dxil).unwrap(),
        Err(e) => {
            let _ = writeln!(
                stderr().lock(),
                "SKIPPED guarded join buffer store validation: {e}"
            );
        }
    }
    let host = match Host::new(true) {
        Ok(host) => host,
        Err(e) => {
            assert!(
                std::env::var("XTP_REQUIRE_GPU").as_deref() != Ok("1"),
                "{e}"
            );
            let _ = writeln!(
                stderr().lock(),
                "SKIPPED guarded join buffer store WARP dispatch: {e}"
            );
            return;
        }
    };
    let (root, pso) = host.compute_pipeline(&shader).unwrap();
    let result = host
        .dispatch(
            &root,
            &pso,
            &Dispatch {
                roots: vec![(
                    0,
                    Root::Constants(vec![0; shader.user_data_dwords() as usize]),
                )],
                groups: [1, 1, 1],
                views: &[View::RawUav(4)],
                samplers: &[],
            },
        )
        .unwrap();
    assert_eq!(result[0], [19, 19, 19, 0]);
}
