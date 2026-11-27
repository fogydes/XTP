# Wave64 mesh programs on 32-lane hosts

How XTP runs BTP 6's mesh programs on NVIDIA, whose waves have 32 lanes
when every one of these programs was written for the console's 64. What the
translator proves, what it exchanges through group-shared memory, and what
it still refuses, measured over the title's shader pack before and after.
Task mesh64 (0a31f0cb), on the person's PC (RTX 4080 SUPER,
Series S mode). The title is BTP 6. Logs and the pack stay under
`E:\xtp-data\` and the title folder, none in the repository.

## 1. Summary

- **Before:** the translator made 8,726 of the pack's 9,282 mesh programs
  for a 32-lane host, and the runtime refused every one of them (VIctory
  #578): the world lost the geometry
  these programs draw.
- **After:** 8,725 translate and run on NVIDIA by default. dxil.dll refuses
  none that it did not refuse before. 11 are refused, each for a precise
  reason (§5). `XTP_WAVE64_MESH=refuse` restores #578's refusal, for an A/B
  in game.
- The one wave-size hazard of the title's mesh code is `v_readfirstlane_b32`
  (§2):
  - 7,540 programs need no more than a proof that the value read is the
    same in every lane of the console wave, so each lane reads its own;
  - 1,185 read a value only the data makes uniform (the counts of the
    meshlet a group's first lane draws). They get the console's first lane
    through group-shared memory, between group barriers (§3.2).
- Tests: unit tests of the proof (`emit/lower/uniform.rs`), translation
  tests of every rule, and a program drawn on WARP whose 4-lane waves split
  each console wave into 16 (`emit/mesh_tests.rs`). The old lowering gets
  all 126 texels of that draw wrong; the new one, none.
- World sessions (§6): three on NVIDIA with the mesh programs running, and
  no device removal. In the third, 150 s in the world, the G-buffer holds
  the world's geometry. The presented image stays dark, for another reason
  than the mesh programs.
- **Independent review** (task a5ce0d66, 2026-11-24): no uniform value
  wrongly claimed, no exchange or barrier fault, but one unproven hazard
  and gaps in the rules and tests, now fixed:
  - LDS order within one console wave of a 128-lane group (§3.4): checked
    on the pack's five mesh programs with LDS, none affected; programs
    that would be are refused;
  - `vccz`, `execz`, `lds_direct`, EXEC named as an operand, and masks
    read as numbers are per lane, here and in the resources' tracer
    (§3.1);
  - one-wave LDS counted against the host's smallest wave, not 32;
  - tests that fail when their rule is broken: one per loop-exit rule, and
    a draw that tells the two console waves' exchanges apart (§4);
  - the mesh tests pass on the cloud L4 (NVIDIA, 32-lane waves) too.
  - The survey is unchanged (§5).

## 2. What depends on whole console waves in mesh code

XTP lowers mesh programs per lane. Their lanes are numbered from the flat
thread id, whatever the host's waves are: console lane = id % 64
(`v_mbcnt`, `lane_index`), console wave = id / 64 (the merged wave info in
`s3`). Mesh code reads no other lane except within quads (DPP quad
permutations, `ds_swizzle` in quad mode); every other cross-lane operation,
mask read as a number and whole-wave reduction is refused there, as before.
What remained, against the four hazards the task listed:

| Hazard | What XTP does now |
|---|---|
| 1. `v_readfirstlane_b32` of a value the console wave's lanes hold differently: lowered as the host's `WaveReadLaneFirst`, the upper 32-lane half read lane 32. | A value proven the same in every lane of the console wave is each lane's own (no wave operation). Any other one, where every lane of the group runs the block, goes through group-shared memory (§3.2). Elsewhere: refused. |
| 2. An `s_barrier` in a block one host wave skips: a hang. | The lowering keeps barriers only in programs with LDS (unchanged). Each one kept must be in a block every lane of the group runs as often as the others (§3.1); otherwise the program is refused. |
| 3. LDS one console wave shares across its two host halves. | A group of one console wave with LDS, larger than the host's smallest wave, is refused (no program of the pack). Groups of several console waves (all of BTP 6's: 128 lanes) order their LDS between waves with the barriers HLSL makes the program write; within one console wave, an LDS read after a write or a write after a read with no barrier between is refused (§3.4, review task a5ce0d66). None of the pack's five mesh programs with LDS has one. |
| 4. Output counts set once per group. | Already so: `setMeshOutputCounts` with the declared maxima, once, in the entry block (`mesh_prologue`). The console's own counts (`GS_ALLOC_REQ`) are not passed; the first-lane reads that only feed them are dead code here. |

## 3. Design

### 3.1 The proof (`emit/lower/uniform.rs`)

A forward analysis over the IR gives each value a level: the same in every
lane of the **group**, in every lane of a console **wave**, or per **lane**.
"The same" is about two lanes that run the same console instance of the
definition. It starts optimistic (everything the group's) and raises levels
until nothing changes.

- Inputs: user data and the group id (`v5`, read only through
  `v_readfirstlane`) are the group's; the merged wave info (`s3`) is the
  wave's; VGPRs the hardware does not load read as zero.
- Instructions: the highest level of their operands. A VGPR written under a
  guard (EXEC narrowed with no branch) adds the guard's level and the old
  value's; so does a partial write (accumulators, 16-bit halves, DPP rows,
  SDWA). A lane mask read lane by lane, as the select of `v_cndmask_b32`
  or the carry-in of a `*_co_ci_u32` form, counts by its lane bit, from
  structurisation's booleans: a select on `s_cselect_b64 …, exec, 0` of a
  scalar compare is the group's. Any other VALU read of a mask, or of EXEC
  named as an operand, reads it as a number: the lane's. So are `vccz`,
  `execz` and `lds_direct`, in any instruction, whatever VCC, EXEC and M0
  hold. Lane numbers, lane reads, LDS, atomics and the clock are the lane's.
- Memory: a load at the group's address is the group's, unless the
  program stores to memory or runs atomics on it. Two host waves then read
  at different times, and a store could fall between them; every load is
  then the lane's. No mesh program of the pack writes memory.
- Joins: a phi where two paths from the two sides of a branch first meet
  takes the branch's level (forward labelling from the branch's out-edges,
  as SSA phi placement does). Loop headers are not such joins, so a loop
  counter stays the group's inside its loop.
- Loops: a value the loop defines and a block after it reads takes the
  level of the loop's branches whose two sides meet again only after the
  loop (lanes leave at different iterations or by different exits: one
  iteration's post-dominators, with back edges and exits as sinks). Phis
  where several exit edges meet take it too, and so do header phis with
  several back edges, for branches that meet only at the next iteration.
- Control: a block's level is the highest level of the branches it is
  control dependent on, transitively (post-dominators of the whole graph).
  A group barrier needs the group's.

The resources' own uniformity (`resource/uniform.rs`, from b03d6089)
shares the operand helpers (`source_regs`, `lane_bit_operand`,
`is_exec`): `vccz`, `execz` and `lds_direct` are never uniform there
either, in a VALU or a scalar instruction.

What it assumes:
- the per-lane form itself: a lane in EXEC at a vector instruction has
  reached it, holding the console's registers;
- no other dispatch changing the memory it reads meanwhile;
- lanes that skip a block not reading the scalars it sets, as
  structurisation already assumes.

### 3.2 First-lane reads (`emit/lower/mesh.rs`, `valu.rs`)

On a host without 64-lane waves:

- **Proven** (the group's or the wave's): the lane's own value. Correct for
  any host wave, in any layout of the group's threads; no wave operation.
- A **proven VGPR-index waterfall** lets each lane use its own index when
  the first-lane read supplies M0 through local copies. The single-block
  loop must select precisely the lanes with the matching index and retry
  precisely the others. Only the indexed moves do per-lane work, and the
  scalar index and mask temporaries stay inside the loop. Every active
  lane then passes the compare and performs its move once, in one pass.
- **Otherwise, where the block's control level is the group's**: an
  exchange in two group-shared dwords per console wave (after the
  program's LDS; the container's group-shared size counts them).
  - With EXEC full in every lane that gets there, the console's first lane
    is lane 0 of each console wave. It stores the value, and after a group
    barrier every lane of the wave reads it. A second barrier comes before
    the slot can change.
  - With EXEC narrowed:
    1. lane 0 empties its wave's slot;
    2. barrier;
    3. the active lanes take the slot's minimum with their lane (an atomic);
    4. barrier;
    5. the lane found (lane 0 when none is active, as on the console)
       stores the value;
    6. barrier; every lane reads it; barrier.
  - The result is the console's in every lane, whichever host waves the
    lanes sit in (WARP's 4-lane waves included).
- **Otherwise**: refused ("a first-lane read of a value lanes may hold
  differently, where lanes of a wave64 mesh group branch differently, …").
  The refusal reaches the program only if an effect uses the value.

With 64-lane waves (AMD, or a target that holds 64), a proven read is the
lane's own and any other keeps `WaveReadLaneFirst`, as before.

### 3.3 The runtime (`xtp-gpu` `pipeline/host.rs`, `pipeline.rs`)

The translator decides. `Needs::mesh_reason` no longer refuses a wave64
mesh program translated for the host's own target. Two cases remain:

- `XTP_WAVE64_MESH=refuse`: #578's refusal, before translation as drops
  made it (`native_refusal`);
- a target with 64-lane waves forced (`XTP_HOST_LANES`) on a host without
  them, whose translations take the console waves to be whole.

The runtime's own scans (`mesh_first_lane_reads`, `mesh_lds_in_one_wave`)
are gone: the translator now does both, with the proof. The old opt-in
`XTP_WAVE64_MESH=allow` means nothing any more.

### 3.4 LDS order within a console wave (`emit/lower/mesh.rs`)

Added after an independent review of this landing (task a5ce0d66). On the
console one wave's `ds_*` instructions run in order, so its compiler may
let a lane read another lane's write of the same wave (after `s_waitcnt`)
with no `s_barrier` between. On NVIDIA console lanes 0-31 and 32-63 are
two warps (WARP: sixteen waves), which race there: silently wrong data.

- **The rule**, where a group spans several host waves (more lanes than
  the host's smallest wave): a program is refused ("LDS read after a
  write, or written after a read, with no group barrier between, …") when
  an LDS read may follow a write, or a write a read, with no group barrier
  the lowering keeps between them. The paths are the program's own edges,
  as the console's flow runs them: both sides of a branch lanes take
  apart, one after the other, and back edges into the next iteration.
  Atomics count as both. A barrier the lowering drops (the allocation
  request's) does not order anything.
- **Not checked: a write after a write.** Two lanes writing one dword with
  no barrier between race in the HLSL program itself, whatever the waves.
- **The pack**: the five mesh programs with LDS (01c33347724469a5,
  29fd13ee162eb134, b43838907d50bbb1, f973921e76cf2587, fe7e1ec48cffda56;
  128 lanes, 3,072 or 10,368 bytes of LDS) each write their LDS (4 or 14
  `ds_write*`), cross one `s_barrier` that every lane runs, then read it
  (11 or 41 `ds_read*`). They have no loop (no back edge), and no LDS
  access after the reads; their two other barriers come after the
  allocation request (`GS_ALLOC_REQ`), past every LDS access. No read
  follows a write without that barrier, no write follows a read: all five
  translate as before (§5). xcl #1252's sync
  before every LDS access of exactly five mesh programs adds, on these
  programs, nothing that their barrier does not already give.
- **Where it would be needed, a barrier is not inserted.** In a block every
  lane of the group runs (control level the group's), a group barrier
  before the read would order it. No program of the pack needs one, and
  the lowering of LDS accesses (`lds.rs`, `lower/mod.rs`) is outside this
  task: such programs are refused instead (follow-up below).

## 4. Tests

- `emit::lower::uniform::tests`: inputs and what they compute, selects on
  masks, joins and guards, loops lanes leave apart (inside: the group's;
  after: the lane's), loads of memory the program writes, barriers in
  blocks lanes enter apart, and VIctory issue #1498's program (by r4tedx,
  wave64 form: an exit mask counted after the loop is the lane's). The
  review found that this last test passes whether or not the loop-exit
  rules work (the rule for masks read as numbers decides it). Each
  loop-exit rule now has a test of its own, with a contrast case
  (task a5ce0d66):
  - a barrier inside a loop lanes leave at different iterations: the
    lane's (control dependence through the back edge);
  - a barrier after lanes end the program on their own (`s_endpgm` behind
    an EXEC branch): the lane's (ends lead to one exit node);
  - a loop header reached by two back edges from the two sides of a lane
    branch, each side setting a constant: the lane's (`returning`
    branches, `come_back`);
  - two exits, one on the lane and one on a scalar, meeting after the loop
    with a constant each, the meeting block a join only of the scalar
    branch: the lane's (`exit_joins_of`);
  - `vccz`, `execz`, `lds_direct` (VALU and scalar), and masks: by their
    lane bit as the select of `v_cndmask_b32` and the carry-in of
    `v_add_co_ci_u32`, the lane's as a number.
- `resource::uniform::tests`: `vccz`, `execz` and `lds_direct` are never
  uniform, with VCC and M0 holding constants.
- `emit::mesh_tests` (translation for 32-32, 32-64, 4-4 and any host, each
  module checked by dxil.dll where it is installed):
  - proven reads call no wave operation and no barrier;
  - other reads exchange (barriers, no `WaveReadLaneFirst`) on 32-lane
    hosts and keep `WaveReadLaneFirst` with 64-lane waves;
  - reads where lanes branch apart are refused on 32-lane hosts only;
  - barriers in a lane branch are refused, after the join kept;
  - one-wave LDS is refused, counted against the host's smallest wave (32
    lanes are one NVIDIA wave but eight of WARP's);
  - LDS order (§3.4): writes, a barrier, then reads translate; a read
    after a write, a write after a read, the next iteration's write after
    a read, and a write and a read on the two sides of a lane branch are
    refused on 32-lane hosts and WARP, translated with 64-lane waves.
- `emit::mesh_tests::wave64_mesh_reads_draw_the_console_waves_values`: a
  128-lane program draws a strip into 63 x 2 texels on the test adapter
  (WARP here, 4-lane waves), with in each vertex five first-lane reads,
  the exchanged values made from the thread t (not the lane), so that each
  console wave's first lane gives its own:
  - a value of the group's (7) and the console wave's index w, added;
  - t + 100 from the console wave's first lane (100 and 164);
  - right after it, 2 t + 1 (1 and 129): two full exchanges back to back,
    through the same slots;
  - 3 t + 7 with EXEC narrowed differently in each console wave (the lanes
    above 9 + 20 w: 37 and 289; VIctory issue #1132's case).

  All 126 texels are right. With the exchange turned off (mesh64's
  mutation, not committed), all 126 were wrong: the host wave's first lane
  gives 104, 108, …, and the narrowed read gives 7, lane 0's value outside
  EXEC. The first form of this test (lane + 100 and 3 lane + 7, the same
  value in both console waves, one exchange of each kind) could not tell
  the two console waves' slots apart; the review found it (task a5ce0d66).
  Two mutations, not committed, measured on WARP:

  | Mutation | first form | this form |
  |---|---|---|
  | both console waves use wave 0's slots | passes | 126 of 126 texels wrong |
  | no barrier after the exchange's reads | passes | 118 of 126 texels wrong |
- `xtp-gpu` `pipeline::host::tests`: the early refusal and the refusal after
  translation agree for real mesh translations (32-32, 4-4 and 32-64; they
  declare no wave size), for every switch and host; the refusal texts; a
  WARP pipeline is translated, not refused before, by default.
- **Each rule broken alone** (task a5ce0d66: a mutation of the code, not
  committed, and its test run on the build farm), every test named fails:

  | Rule broken | Test that fails |
  |---|---|
  | header phis with two back edges ignore `come_back` | `a_header_reached_by_two_back_edges` |
  | phis where exits meet ignore the loop's leaving level | `exits_that_meet_after_the_loop` |
  | a loop's blocks not control dependent on its own branch | `barriers_in_a_loop_lanes_leave_apart` |
  | ends not led to the exit node | `a_barrier_after_lanes_end_the_program` |
  | `vccz`, `execz`, `lds_direct` mapped to VCC, EXEC, M0 again | `special_operands_are_the_lanes`, `special_operands_are_never_uniform` |
  | every mask read by its lane bit again | `masks_count_by_their_lane_bit_only_as_selects_and_carries` |
  | the lane-bit rule for `v_cndmask_b32` only | the same |
  | one-wave LDS counted against 32 lanes | `lds_in_one_wave_counts_the_hosts_smallest_wave` |
  | LDS order over forward edges only | `lds_order_within_a_console_wave` |
  | LDS order without writes after reads | the same |
  | both console waves use wave 0's exchange slots | `wave64_mesh_reads_draw_the_console_waves_values` |
  | no barrier after an exchange's reads | the same |

## 5. Survey of the pack

Every MS program of `<title folder>\shaders.rpf`, translated for a 32-32 host and
checked by dxil.dll, with a tool outside the repository
(`E:\xtp-data\tmp\mesh64\measure`). Before: main at `c135f4c`, this
branch's base after its last merge.

| | before | after |
|---|---|---|
| MS programs | 9,282 | 9,282 |
| translated for 32-32 | 8,726 | 8,725 |
| … that the runtime lets run on NVIDIA | 0 | 8,725 |
| … reading their first lanes as their own (§3.1) | — | 7,540 |
| … with an exchange (§3.2) | — | 1,185 |
| refused for a first-lane read (this task) | — | 11 |
| refused for other reasons (unchanged causes) | 556 | 546 |
| translations dxil.dll refuses | 62 | the same 62 |

- The 11 refused: their reads are in loops that index VGPRs (`v_movrels`
  with the first lane's index in M0, a waterfall) inside branches lanes
  take apart. Each lane's own index would be right there (a waterfall that
  runs each lane once); proving that shape is left for later.
- One of them translated before. The other 10 failed for another reason
  before (a lane mask read as data, a scalar memory read as data, a
  structured buffer of unknown stride); they now stop at their first-lane
  read first.
- The 62 invalid translations are not this task's: 53 have more than 32 KiB
  of vertex and primitive outputs (VIctory #1385's case),
  9 have signature mismatches.
- The reasons the other 546 stop: a scalar memory read as data (383), a lane
  mask read as data (101), per-lane descriptors (34 + 12 + 6 + 3), a
  descriptor read only as data (5), a cross-lane operation (2).

**After the review's fixes** (task a5ce0d66): the same survey, for 32-32,
with a tool of its own outside the repository
(`E:\xtp-data\tmp\mesh-fix\measure`; the programs by hash). Before: main at
`9d4a630`, this task's base.

| | before | after |
|---|---|---|
| MS programs | 9,282 | 9,282 |
| translated for 32-32 (allowed on NVIDIA) | 8,725 | 8,725 |
| … with an exchange | 1,185 | 1,185 |
| … with LDS (§3.4) | 5 | 5 |
| refused for LDS order (§3.4) | — | 0 |
| refused (the 11 first-lane reads above included) | 557 | 557, the same counts by reason |

Every rule the review changed only raises a level or refuses more, so a
program can only lose a proof, never gain one; with every count the same,
no program changed. The one-wave LDS rule changes nothing for the pack
either on hosts of smaller waves: its five mesh programs with LDS have 128
lanes.

**VGPR-index waterfalls for issue #56** use the validated 32-32 survey
as the baseline.

| | Before (`e2298b0`) | After |
|---|---:|---|
| MS programs | 9,282 | Unmeasured |
| MS programs translated | 9,213 | Unmeasured |
| MS programs refused at a divergent first-lane read | 15 | Unmeasured |

Four of the 15 baseline refusals have the single-block shape now proven
by resource tracking (`45f081396c2c59ba`, `c89489afd89ee7f0`,
`0bd4d80ad23b106b`, `63083c777948927a`). Synthetic tests cover translation
and per-lane selection. A second survey is still needed for the counts,
validator verdicts and output hash comparison.

## 6. World sessions

Series S, `E:\tools\stall-probe\session.ps1`, autopilot from the golden
save, `XTP_SKIP_INTRO=1`, `XTP_WATCHDOG_HOOK=1`, `XTP_D3D12_DRED=1`,
captures at `E:\xtp-data\runs\frames-mesh*`. All three ran the same binary
(this branch at `f6af110`, mesh programs running) and none lost the device.

- **A** (19:13 UTC):
  - the world's gameplay reached at 44 s: 1.2 fps, 1,289 draws a frame,
    924 dropped;
  - 154 mesh translations (median 15 ms, at most 237 ms), 223 mesh
    pipelines made, none refused;
  - **no device removal**;
  - the title stopped 8 s later on an `int3` (title+0x23d8f03), reached
    from its progress watchdog's thread (title+0x2215eca), a firing path
    `XTP_WATCHDOG_HOOK` does not patch (its two sites are 0x2215c27 and
    0x2215d9b);
  - in the 7 s before, at once: main's new FSR4 path (every pass "falling
    back"), a 510 ms submission, and about 200 mesh pipelines made;
  - the captures stop at the opening credits, before the world.
- **B** (19:30 UTC, the autopilot ending the title 30 s into gameplay):
  - gameplay at 22 s, 237 mesh pipelines made, none refused;
  - no stop, so the stop of A does not come with every run of the mesh
    programs (the third watchdog path is task 54cb7617's);
  - the run ends at frame 598, as the world starts. Its first world frame
    (3,932 draws) is presented black, and the captured frames are the
    opening credits and their fade.
- **C** (19:51 UTC, 150 s in the world):
  - 25 frame-rate lines: 0.7 fps median (0.3 to 6.2), 2,525 draws a frame
    (up to 5,956), 985 dropped;
  - 523 mesh pipelines made, none refused. The mesh dispatches still
    dropped are of programs that do not translate for other reasons: a
    scalar memory read as data, a descriptor read from the first lane of
    computed VGPRs;
  - the render targets of frames 949, 979 and 1009 dumped
    (`XTP_DUMP_TARGETS=1280`). **Frame 1009's 1280 x 720 targets hold a
    street of the world in 3D**: fences, poles, parasols, facades, objects
    on the ground, whole and in place;
  - the presented frame shows the introduction's credits over a nearly
    black scene, and frames 949 and 979 are black: what darkens the image
    lies past the geometry (lighting, composition, upscaling, exposure),
    not in the mesh programs.

## 7. Upstream, reviewed

- **VIctory #940** (host-lane numbering of wave64 mesh, `ed9afc0`): not
  taken. It numbers mesh lanes by the host's lane so that host ballots and
  lane numbers agree. XTP reads no mask as a number in mesh code, and its
  console numbering from the flat id is right on any host wave; #940's
  would give lanes 32-63 the numbers 0-31.
- **VIctory issue #1132** (a lone `v_readfirstlane` under a partial EXEC):
  its case is a test here (§4). It also applies to XTP's compute path: a
  wave64 compute program whose only cross-lane operation is
  `v_readfirstlane` declares no wave size and keeps `WaveReadLaneFirst`
  (`Lower::new` does not count it among the reads that need the console
  flow). That is `lower/mod.rs`'s, not this task's: follow-up below.
- **VIctory #1133** (barrier rule): not taken. It keeps the memory fence
  and drops the group sync of barriers lanes reach apart. That trades a
  hang for silently unordered LDS. XTP refuses such programs instead, and
  the pack has none among those that keep barriers.
- **VIctory #1433** (console flow for groups of several console waves,
  barriers paired by count): the general form, for compute. Mesh programs
  here need no console flow: their only wave hazard is the first-lane read
  (§2), which the proof and the exchange cover.
- **five-star-recomp** (window analysis, supersteps): measured on the
  same pack, 6,217 of 8,765 mesh programs need no whole wave there. Its
  first reason against the rest is a first-lane read of a per-lane value
  "in an output or in GS_ALLOC_REQ" (1,338). XTP drops `GS_ALLOC_REQ`'s
  counts, and exchanges the reads that outputs need.
- **xcl #1193** (no exchange for a `v_readfirstlane` of a proven uniform
  value): the same idea as §3.1, for its own split form. xcl is
  GPL-2.0-only: ideas only, nothing taken.
- **Kyty's lane pairs** (AnyPS5's study: one host wave of 32 holds lanes i
  and i + 32): not needed. It would rewrite the mesh lowering, double the
  register pressure and the compile time, for 11 programs.
- **VIctory issue #1498** (a mask counted after a loop whose lanes leave
  apart): its program is a test of the proof. In mesh code such counts are
  refused as masks read as data, as before.
- **VIctory #1385** (mesh output counts from `GS_ALLOC_REQ`, under 32 KiB of
  outputs): not taken here. It concerns the 53 programs dxil.dll refuses
  for their output size, which are not this task's.

## 8. Known risks and follow-ups

- **Compute, #1132:** a lone `v_readfirstlane` of a per-lane value in a
  wave64 compute program on NVIDIA still reads its host wave's first lane.
  The proof could tell those that need the console flow; the routing is in
  `lower/mod.rs` (`Lower::new`).
- The **VGPR-index waterfall** proof for issue #56 covers a single block
  with local scalar temporaries and moves selected by index equality.
  The other 11 first-lane refusals in the `e2298b0` survey have different
  shapes (§5).
- **A lane stopped by the loop limit** (`LOOP_LIMIT`) leaves the program.
  If a group barrier follows, the other lanes wait for it: a hang where
  the console would only have been slow. The title's mesh programs that
  keep barriers are those with LDS (5); the exchanges sit where every lane
  runs the block. Not measured. Since then, main's loop limits drain such
  lanes (`lower/flow.rs`: they keep passing barriers), but not in a wave64
  mesh program in halves (task a5ce0d66): there the first-lane exchanges
  add barriers that the drain does not count, so this risk stands. The
  drain's counts of ended threads are group-shared memory that the mesh
  PSV now counts (`Shader::shared_extra`): without it, dxil.dll refused
  every mesh program with LDS and a barrier after a loop.
- **Cost:** each exchange is two or four group barriers. In the title the
  reads are once per group, outside loops; not measured on the 4080 yet.
- **LDS a mesh program allocates but never accesses** makes the
  container's group-shared size count bytes that no array holds, which
  dxil.dll refuses. No program of the pack does this. Exchanges leave
  such LDS out of the count.
- **A barrier before an unordered LDS access** (§3.4), in blocks every
  lane of the group runs, instead of refusing the program: a call before
  each LDS access the check flags, from `lower/lds.rs` or the instruction
  loop of `lower/mod.rs`. No program of the pack needs it.
- **Mesh waves of 32 on AMD** (review finding 6): the exchange and the
  checks apply where the target has no 64-lane wave
  (`Lower::mesh_halves`). A host whose range holds 64 but that runs mesh
  shaders in waves of 32 (an AMD driver may; `pipeline/host.rs` takes
  RDNA's mesh waves to be whole) would need them wherever `lanes_min` is
  below 64. That costs two or four group barriers per first-lane read
  that is not proven, on hosts that may not need them; measure it on AMD
  hardware first (none here).

## 9. Also in this task: two ports

The orchestrator added two ports to the review task (a5ce0d66). Each is
reviewed against the console's semantics, with credit, tests and a
mutation that each test catches.

### 9.1 Mesh primitive indices are 9 bits (VIctory #1718)

Ported from Marvisak/VIctory#1718 (head `c0228d9`) by
tjthebigbearcat-hub; watch item 177.

- **The format.** `exp prim` packs three corners of ten bits each: a 9-bit
  vertex index and, above it, the corner's edge flag (bits 9, 19 and 29).
  Bit 31 culls the primitive.
- **The fault.** `emit_indices` (`emit/lower/export.rs`) masked ten bits,
  so a flagged corner read as vertex 512 or more, past any mesh program's
  vertices: D3D12 drops or corrupts such a primitive. The mask is now
  `0x1FF`. The edge flags only matter to the console's edge-flag
  rasterisation, which the host has no use for.
- **The test** (`emit::mesh_tests::edge_flagged_triangles_draw_as_unflagged_ones`,
  the PR's, on XTP's draw helper): a 32-lane mesh program whose triangle
  (0, 1, 2) covers 16 x 16 texels. Plain and edge-flagged, all 256 texels
  are lit with the attribute; culled, none. dxil.dll takes each module.
  With the old 10-bit mask (a mutation, not committed), the edge-flagged
  triangle lights 0 of 256 texels on WARP, as the PR reports.
- **Not measured:** how many of BTP 6's triangles carry edge flags (the
  indices are computed at run time).

### 9.2 Tables in the code through joins and first lanes (fork pick 4)

Ported from fxceless's VIctory fork (commit `578d459`, its `code_join`,
`first_lane`, the `s_getpc_b64` lowering and two tests), with one change
from the review below. Resource tracking (`resource.rs`) refused two
shapes of address into a table the compiler appends to the code:

- **A join** ("an address from a phi", 8 compute programs:
  098fc28e44347a00, 21d27f785da6d2b1, 77a7af30c430cd16, 982f3cccd5c04294,
  9aa4a785e0800cda, b3e7781d53ff8326, d9978c76c746d809,
  f7c4f227a2a4c4b2): paths that each add an index to an `s_getpc_b64`
  (one table, or a table of their own) and meet before one load. BTP 6's
  fog reconstruction reads its sample offsets this way; the fork reports
  that, refused, "the lit frame was multiplied by a fog transmittance
  nothing had written". In the programs read, the joined addresses are
  VGPR pairs read by `global_load`, each lane at its own address, as on
  the console.
- **A first-lane read** ("an address read from the first lane of a
  per-lane value", 5ed77e9e8f084dea and 0a5d4f9047b3dce4, the
  average-luminance downsample that feeds the exposure): an address
  computed in VGPRs, read back with `v_readfirstlane_b32` before a scalar
  load. In 5ed77e9e8f084dea the index is a loop counter every lane steps
  alike, but the tracer's uniformity proof cannot show it: the loop is
  left on `vccz`, whose per-lane condition tests the lane's EXEC bit.

What XTP does now:

- the join, or the first-lane read, is the address's computed term, and
  counts the table's base itself: the emitter gives each `s_getpc_b64` its
  offset in the code (that of the next instruction, high dword 0;
  `emit/lower/salu.rs`), so the term is the byte in the code whichever
  table a path took (`Tracer::code_join`, `code_lane`, `counts_code`);
- each path of a join must add to exactly one `s_getpc_b64` and to no
  root parameter (`code_base`: the fork's looked for a table in any term);
  the high half must have the shape of a table's, joined or read from the
  first lane too (`code_high`; no lowering uses it).

**The review's change.** The fork took the first-lane read's *source* as
the address: each lane's own index. The console reads the table at the
first lane's index for every lane, so that is wrong wherever the index
differs between lanes. Here the computed term is the read itself, which
the emitter already lowers as the console's first lane, and it is taken
only where no guard narrows the read in the per-lane form: there the
host's first lane may be one the console's EXEC leaves out
(`a_first_lane_read_under_a_guard_of_a_per_lane_address_is_refused`).

**Shortcut** (`salu.rs`): a program that uses `s_getpc_b64` for anything
but a table address computes with the code offset where the console has
the address. A jump through it (`s_setpc_b64`, `s_swappc_b64`) is refused
as before.

Tests (`resource::tests`, `emit::lower::salu::tests`):

- the fork's two tests: two paths into one table give the join as the
  computed term at offset 0; the first-lane form gives the read itself
  (the fork's test asserted each lane's index);
- the fog's form, two tables, each with its own `s_getpc_b64`; with one
  path's address not in the code, the join is refused as before;
- the ten earlier tests of the uniformity rules (b03d6089) that expected
  "an address read from the first lane of a per-lane value" now see the
  address read from the first lane: a rule that wrongly proved the index
  uniform would make the index the computed term again;
- a dispatch on the test adapter (WARP here): lanes 0-15 read table 1 and
  lanes 16-31 table 2 through a join, every lane its own entry. With the
  counter's offset 4 bytes short (a mutation), every lane reads the dword
  before its own;
- a dispatch on 32-lane hardware (`--ignored`, the cloud L4): every lane
  computes its own entry's address and the scalar load reads the first
  lane's, so every lane stores entry 0 (the fork's form would store each
  lane's own); translation and validation for 32-32 and 32-64 run
  everywhere.

**Survey** (`xtp shader-survey --host-lanes 32-32`, run on the cloud L4
with the pack of its game folder; before: `7a3a318`, this task before both
ports; after: `e1b75d7`):

| | before | after |
|---|---|---|
| translations | 24,202 | 24,238 |
| compute translated (of 2,689) | 2,212 | 2,244 |
| pixel translated (of 12,101) | 11,083 | 11,087 |
| mesh translated (of 9,282) | 8,725 | 8,725 |
| refused: an address from a phi | 28 (compute) | 0 |
| refused: an address read from the first lane of a per-lane value | 5 (3 compute, 2 pixel) | 0 |
| refused at emit: `s_getpc_b64` | 3 (1 compute, 2 pixel) | 0 |

No other count moved: the 36 programs gained are those three rows, and no
program lost its translation. The session's census named 8 programs
for the join; the pack has 28.

**Follow-up: the three programs that stopped at `s_getpc_b64` in the
emitter** (187f053719d2206e and 17761234145ad256, pixel; b2711644ca5566cc,
compute) translate now that the counter has a value. Each takes it only
as a table address in the code (`s_add_u32` of a literal, `s_addc_u32` of
0, then an index added in VGPRs or SGPRs). The compute program also
parks those addresses in VGPR lanes (`v_writelane_b32`), which is
probably what reached an effect before. Not yet checked: that no address
it parks is read back as data, where the code offset stands for the
console's address (the shortcut in `salu.rs`).
