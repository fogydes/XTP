# Host texture VRAM: pooled heaps (task 450ec9ba)

Why the world stalled while loading, what XTP now does with host textures,
what it costs, and what is still to measure. Code:
`crates/xtp-gpu/src/resource/{pool,ranges,residency}.rs` and
`resource.rs` (`host_texture`, `committed`, `pooled`).

## 1. The problem

While the world loads, BTP 6 creates thousands of resources in
its memory (device method 0x428, `create_at_address`); each texture gets a
host copy (`host_texture`). Until this change every host copy was a
committed resource: its own implicit heap, its own residency operation.

stall's measurements (task 6bb65399, this PC; a D3D12 probe
without XTP, then sessions R and S in game):

- Past a global threshold (about 9.5–9.75 GB resident for all processes on
  the RTX 4080's 15.67 GB local segment), VidMm admits about one new
  resident allocation a second: 0.3–1.2 s for a 1.4 MB BC7 texture,
  0.5–0.74 s for a 64 MB buffer, 0.6–2.1 s (≈ 1.2 s) for a 256 MB buffer.
  The cost is per allocation, almost not per byte.
- The DXGI budget stays at 14.92 GB, so DENY_OVERBUDGET never fires there.
- CREATE_NOT_ZEROED changes nothing to it.
- In game: about 6,700 resources at an address before the stall, 3,086 live
  host textures for 8,141 MiB at the stall; past the threshold 2–4 textures
  come out every 5 s. A device restart did not change it (Samuel): it is this
  machine's behaviour, not a passing state. On the cloud L4 there is no
  ceiling (21 GB in 7.8 s).
- With `XTP_WATCHDOG_HOOK=1` the world is reached (intro-skip's session T2,
  main 5b51ddd): 160 s of loading, the world at +184 s, VidMm at
  9.5–9.7 GiB from +50 s, 2,431 creations during the loading. **The number
  to beat: 160 s.**

Per-texture residency gives about 0.9 MB/s past the threshold; heaps of
256 MB about 210 MB/s.

## 2. What XTP does now

### 2.1 Which textures go where

| host texture | where | why |
|---|---|---|
| no render target, depth or unordered access; 1 sample; not planar; contents uploaded | **pool** | the title fills it: `texture::upload` writes every mip and slice of plane 0 from title memory before a shader reads it, so pool memory needs no zeroing |
| render targets, depth targets | committed, residency priority HIGH | placed targets need a clear, discard or copy before first use (D3D12's rule for placed RT/DS), and their contents start at zero today |
| unordered access textures | committed | GPU-written: their contents start at zero today, which a reused pool range would not give |
| MSAA | committed | always targets on D3D12 (so no 4 MiB class) |
| planar video formats (NV12, P010, ...) | committed | only plane 0 is uploaded |
| stand-ins whose contents are not uploaded (116, 117) | committed | their committed memory stays zero, as before |
| more than a quarter heap (64 MiB) | committed | one heap each anyway |
| over the budget, or a placement the driver refuses | committed, logged | see 2.4 |

**The pool and its reserved textures are on by default on a host with less than 32 GiB of RAM (§3.5), off elsewhere (§3.3)**: `XTP_TEXTURE_POOL=1` and `XTP_TEXTURE_TILES=1` turn them on, `0` off; with the pool off, every host texture is committed, as before.

### 2.2 The pool (`resource/pool.rs`)

- Two classes by placement alignment: **4 KiB** (D3D12's small resource
  rule: no target, one sample, one slice, mip 0 in at most 16 standard 4 KiB
  tiles, granted by `GetResourceAllocationInfo`) and **64 KiB** (the rest).
  A committed texture always takes 64 KiB at least: a 16x16 texture is 1 KiB
  of texels in a 64 KiB allocation; in the pool it takes 4 KiB.
- Heaps: DEFAULT, `ALLOW_ONLY_NON_RT_DS_TEXTURES` (valid on resource heap
  tiers 1 and 2), `CREATE_NOT_RESIDENT | CREATE_NOT_ZEROED` (zeroed if the
  runtime refuses the flag). 256 MiB for the 64 KiB class, 32 MiB for the
  small one (`XTP_TEXTURE_HEAP_MIB`).
- Sub-allocation (`resource/ranges.rs`): best fit with merging of free
  neighbours. A buddy allocator rounds a 1.4 MB BC7 texture to 2 MiB (43 %
  lost); TLSF fits as well as best fit at these counts (hundreds of ranges a
  heap, a few hundred placements a second) and is more code.
- A thread (`xtp-texture-heaps`) keeps a **reserve** of free bytes in each
  class: one heap, 256 MiB and 32 MiB by default (`XTP_TEXTURE_RESERVE_MIB`;
  session A3 ran with 512 MiB, see 3.2: making a heap not resident takes
  0.1–1.9 ms, so one heap ahead is enough, and every MiB of reserve is
  resident memory the title does not use). Heaps
  still becoming resident count in it: the title's thread places textures
  in them and the GPU gets those once the heap is resident. Each heap is one
  residency request (`EnqueueMakeResident`, its own fence, within the
  budget); a texture is none.
- The title's thread only calls `CreatePlacedResource`. In the first 100 ms
  of a heap, a placement into it waits at most 10 ms for its residency (so
  the title can use a texture right after making it, as below the
  threshold); past the threshold that is at most about 100 ms a heap, never
  a texture. If the reserve is gone, the title's thread makes a heap itself
  (counted: "by the title's thread"), not resident, not waited for.
- A range goes back when the host copy is dropped. Lists hold host copies
  until their submission has run, so a range is never reused under the GPU.
  An empty heap is released unless the reserve needs it (hysteresis), and
  every empty heap goes when DXGI's budget-change notification shows usage
  past the budget.

### 2.3 Residency (`resource/residency.rs`)

- Every request has a fence of its own (review of dark-world's delivery:
  with one fence and values taken under a lock but enqueued outside it, a
  later request finishing first said an earlier texture was resident: a GPU
  page fault).
- Nothing is refused for good (review): a request refused (past the budget,
  or an error) is asked again a second later; meanwhile the texture is
  handed out but not given to the GPU. `host_resource`, `host_copy`,
  `host_transition` and `texture_at` answer as if it had no host copy until
  it is resident (`HostTexture::ready`).
- Committed textures wait at most 10 ms for their residency, then are
  handed out the same way.

### 2.4 The budget

- Before a heap, the pool checks `QueryVideoMemoryInfo`: usage plus heaps
  not yet resident plus the new heap must stay within the budget. Past it,
  no heap is made (logged once each time it starts, counted), the texture is
  made committed and asked to become resident with `DENY_OVERBUDGET`
  (refused, logged, retried every second). The process is never made
  resident past its budget.
- The budget-change notification
  (`RegisterVideoMemoryBudgetChangeNotificationEvent`) is followed by the
  thread: logged, and past the budget the empty heaps go.
- Render targets and depth targets get `D3D12_RESIDENCY_PRIORITY_HIGH`.

### 2.5 Counters (frame line, `format::report`)

`host textures: N alive, M MiB (P pooled, X MiB; C committed, Y MiB;
stand-in formats +S MiB); residency asked A times, refused R; texture
pool: H heaps (Q not resident yet), B MiB, U MiB placed in T textures; so
far ...`. Stand-ins (67 → RGBA16F, 85/86/115 → RGBA8, 116/117 → RGBA16F,
189 → RGBA16_SNORM, 190 → RG8) take twice the title's bytes: the `+S MiB`
is that half.

The frame report gives live count and MiB by the title creation path that
allocated each copy. The placed-view line gives current cached and retiring
views, cumulative deferrals for unmapped memory and remap invalidations,
and how many same-address lookups found a GPU-written source or shared it.
Descriptor and command-list references can outlive a cache entry.

## 3. Measurements

### 3.1 Micro-benchmark (WARP)

`resource::pool::tests::bench_5000_bc7_textures` (ignored; `cargo test -p
xtp-gpu --release -- --ignored --nocapture pool::tests::bench`): 5,000 BC7
1024² textures of 9 mips made three ways. Run on WARP, on a
build-farm VM (c2d-highcpu-16, no GPU), release build, tree e17e4f8 plus
this delivery:

| path | textures | µs/texture (mean) | p50 µs | p99 µs | max µs | ms until all resident | VRAM KiB/texture | private KiB/texture | residency requests |
|---|---|---|---|---|---|---|---|---|---|
| committed, resident at creation (before 2026-11-24) | 5000 | 51.2 | 49.5 | 87.2 | 144 | 256 | 1408 | 1414 | 5000 |
| committed, not resident + EnqueueMakeResident each, waited (dark-world) | 5000 | 61.7 | 53.9 | 226.5 | 304 | 309 | 1408 | 1366 | 5000 |
| pooled (256 MiB heaps, 512 MiB reserve) | 5000 | **7.9** | 5.8 | 49.3 | 340 | **39** | 1416 | 1421 | **27** |

D3D12 sizes the texture at 1,408 KiB, 64 KiB aligned. Pool: 27 heaps made
for the run, none by the creating thread; 576 MiB held before it (the
reserve); 4 empty heaps kept after (the reserve).

Reading:
- A creation costs 6 to 8 times less, and residency requests drop from one
  a texture to one a heap (185 times fewer). That is the cost the pool
  removes; on the 4080 past VidMm's ceiling each request costs about a
  second (stall), which WARP does not show.
- **No video memory is saved for these textures**: a 1024² BC7 texture is
  a multiple of 64 KiB either way (VRAM per texture 1,408 KiB committed,
  1,416 KiB pooled, the difference being the reserve heaps made during the
  run). The pool saves memory only on textures the small rule lets take
  4 KiB (a 16x16 texture: 4 KiB instead of 64 KiB); in A3, 2,374 pooled
  textures took 4,607 MiB, about 1.9 MiB each, so the world's textures are
  mostly not small ones.
- WARP's "video memory" is the process's own (private bytes follow it).

### 3.2 In game (Series X, this PC)

`E:\tools\stall-probe\session.ps1` (autopilot, golden data,
`XTP_WATCHDOG_HOOK=1`, under `measure.lock`, one session at a time); the
loading is timed from the intro cut (`XTP_SKIP_INTRO`, "intro movie cut")
to the autopilot's world condition ("line 17: met"). Logs in
`E:\xtp-data\cache\play\<UTC time>\run.log`, GPU and VidMm samples in
`E:\xtp-data\runs\stall-tex-pool-*`.

| session | binary, switch | world | loading | host textures at the world | residency requests | xtp dedicated VRAM, max |
|---|---|---|---|---|---|---|
| A (16:44Z) | f991283, pool | **not reached**: 0.2 fps from the first frames, the title stops at 32 textures, the autopilot times out in the front end | — | 32 (8 pooled) | 30 | 1.25 GB |
| B (16:51Z) | f991283, `XTP_TEXTURE_POOL=0`, intro played | +240.1 s | ≈ 223 s (no cut marker; intro ≈ 17 s) | 3,587, 8,884 MiB, all committed | 3,655 | 9.15 GB |
| A2 (17:02Z) | f991283, pool, stopped at 150 s by mistake | not reached in 150 s | — | 3,528 at +145 s, 8,861 MiB | — | — |
| A3 (17:07Z) | e17e4f8 (main merged), pool, 512 MiB reserve | +239.0 s | **230.4 s** | 3,561, 8,837 MiB: 2,374 pooled (4,607 MiB in 24 heaps of 5,472 MiB), 1,187 committed (4,230 MiB) | 1,219 | 9.90 GB (nvidia-smi 11.15 GB) |

What they show:
- The pool works in game: the world is reached, every texture the title
  fills goes to the pool (none refused, none placed by the title's thread
  after the first heap), and residency requests drop to a third (the
  committed textures and the heaps).
- **Loading is not faster.** In A3 as in B, submissions take 0.5 to 30 s
  on the GPU during the load (54 and 33 "still waiting for the host GPU"),
  and the frame time is mostly that. With main's code the per-texture wait
  had already left the creation call (T2: two slow creations); what is left
  is GPU-side, most likely VidMm paging the allocations still made one by
  one: the 1,187 committed textures (targets, depth, unordered access),
  `texture.rs`'s committed upload buffer per detiled texture, `shadow.rs`'s
  buffer per host copy.
- A's front-end stall did not come back in A2 (same binary, same switch) or
  A3. Its cause is not known: thread stacks were only taken in A2, where
  the title threads waited for the GPU (`queue::present`,
  `copy_page_mappings` → `Host::settle` → `wait_gpu`), as in a normal load.
- The 512 MiB reserve left 865 MiB of heap unused at the world (5,472 MiB
  for 4,607 placed), which is why the default is now one heap.

### 3.3 Decision: off by default

The person's decision (relayed by Samuel): XTP works in Series S
mode, where the world loads in about 10 s under VidMm's ceiling (series-s,
frame-prof's session C), and no more Series X session is run. B3 (the same
binary as A3, pool off) was therefore not run, and A3 is not shown to be at
least as good as committed textures: 230.4 s against about 223 s for B (an
earlier binary, the intro played, another moment), 0.75 GB more video memory
resident, and one unexplained front-end stall (A) with the pool on. So the
pool is **off by default** (`XTP_TEXTURE_POOL=1` turns it on). What stays on
for every texture: the residency path of 2.3 (one fence a request, no wait
beyond 10 ms on the title's thread, no texture refused for good) and the
review fixes of §4; session B ran on it.

To turn it on by default later, the Series S measure that decides it: a
session with and one without `XTP_TEXTURE_POOL=1`, same binary, loading
time, VidMm peak (the reserve must not push Series S past the ceiling), and
"handed out before their heap was resident" in the frame line (textures that
may have got null views).

### 3.4 Commit per driver

The probe creates 32 synthetic 4096 × 4096 RGBA8 DEFAULT textures (64 MiB
each, 2 GiB total on the adapters below). It takes a fresh baseline for
each allocation group and samples process `PrivateUsage` and
`PagefileUsage`, system `CommitTotal`, and DXGI local and non-local usage
500 ms after each step and 2.5 s after release. A queue fence completes
each batch of tile mappings or unmappings before the corresponding sample,
and the backing heap stays alive through the unmap sample.

Run the ignored test by itself on a quiet machine with 2 GiB of available
commit plus driver overhead. Use `hardware` for the default hardware
adapter and `warp` for the software adapter.

```powershell
$env:XTP_TEST_ADAPTER = 'hardware'  # or 'warp'
$env:XTP_BUFFER_POOL = '0'          # keep the test host's buffer heaps out of the samples
cargo test -p xtp-gpu --lib texture_commit_probe -- --ignored --nocapture
```

| Adapter | Driver (DXGI UMD) | Software | Windows build |
| --- | --- | --- | --- |
| NVIDIA GeForce RTX 5070 | 32.0.16.1664 | No | 26200.9974 (25H2) |
| Microsoft Basic Render Driver (WARP) | 10.0.26100.9818 | Yes | 26200.9974 (25H2) |

Each cell gives the deltas in MiB in this order: `PrivateUsage`,
`PagefileUsage`, `CommitTotal`, DXGI local, DXGI non-local.

| Case | RTX 5070 | WARP |
| --- | --- | --- |
| Committed with `CREATE_NOT_RESIDENT` | +2052.21, +2052.21, +2065.82, +0.00, +0.00 | +2052.14, +2052.14, +1947.29, +0.00, +0.00 |
| Committed after `MakeResident` | +2052.21, +2052.21, +2047.96, +2048.00, +0.00 | +2052.14, +2052.14, +1773.88, +2048.00, +0.00 |
| Committed after `Evict` | +2052.21, +2052.21, +2067.11, +0.00, +0.00 | +2052.14, +2052.14, +1787.19, +0.00, +0.00 |
| Released committed textures after `Evict` | +192.59, +192.59, +195.76, +0.00, +0.00 | +192.45, +192.45, +471.07, +0.00, +0.00 |
| Placed heap alone | +1859.63, +1859.63, +1866.71, +2048.00, +0.00 | +1859.64, +1859.64, +1924.69, +2048.00, +0.00 |
| Placed heap with 32 textures | +1859.84, +1859.84, +1872.93, +2048.00, +0.00 | +1859.70, +1859.70, +1968.35, +2048.00, +0.00 |
| Released placed textures and heap | -192.17, -192.17, +337.78, +0.00, +0.00 | -192.32, -192.32, +185.29, +0.00, +0.00 |
| Reserved, no tiles mapped | +0.00, +0.00, +137.07, +0.00, +0.00 | +4.52, +4.52, +585.59, +0.00, +0.00 |
| Reserved, all tiles mapped to one heap | +2052.10, +2052.10, +2193.82, +2048.00, +0.00 | +2056.70, +2056.70, +2752.62, +2048.00, +0.00 |
| Reserved after unmapping, heap still live | +2052.10, +2052.10, +2496.47, +2048.00, +0.00 | +2056.75, +2056.75, +2802.84, +2048.00, +0.00 |
| Released reserved textures and heap | +0.07, +0.07, +41.64, +0.00, +0.00 | +4.39, +4.39, +1048.02, +0.00, +0.00 |
| Committed, resident at creation | +2052.03, +2052.03, +1931.58, +2048.00, +0.00 | +2052.12, +2052.12, +2087.90, +2048.00, +0.00 |
| Released committed textures created resident | +192.41, +192.41, +316.20, +0.00, +0.00 | +192.50, +192.50, +431.05, +0.00, +0.00 |

Process commit stayed near 2 GiB through `MakeResident` and `Evict` on
both adapters. Unmapping the tiles left the backing heap charged until
release. Roughly 193 MiB of process commit remained after releasing the
first committed batch and carried into the placed-heap baseline, which
helps explain its lower process deltas. System `CommitTotal` includes
other processes, so it varied even after the resources were released.

This answers part of [#74](https://github.com/MiamiGame/XTP/issues/74).
Measurements on more drivers are wanted, especially AMD, Intel, older
NVIDIA drivers, and adapters with shared memory.

### 3.5 In game (Series S, 16 GiB of RAM): on by default below 32 GiB

The measure §3.3 asks for, on a host where the RAM is what runs out:
- RTX 5070, 12 GB, driver 32.0.16.1664; Ryzen 5 5600; 16 GiB of RAM;
  Windows 11 build 26200.
- Series S mode, the autopilot from the golden save to the world
  (`XTP_SKIP_INTRO=1`, `XTP_WATCHDOG_HOOK=1`), one session at a time.
- The same binary for every session; only `XTP_TEXTURE_POOL` and
  `XTP_TEXTURE_TILES` differ.
- A guard stops the title once the system's commit headroom falls below
  2 GB. On this host, commit runs out long before video memory does.

The world load runs from the first frame line with more than 500 host
textures alive to the autopilot's world condition ("line 17: met"). The
card and private figures are from the last frame line before control comes
back to the player.

| session | switches | world load | card (DXGI local) at the world | process private at the world | residency requests | handed out before their heap was resident | in the world before the guard |
|---|---|---|---|---|---|---|---|
| B1 | `XTP_TEXTURE_POOL=0`, `XTP_TEXTURE_TILES=0` | 85 s | 9,269 of 10,875 MiB | 18,716 MiB | 24,927 | — | 3 s |
| B2 | `XTP_TEXTURE_POOL=0`, `XTP_TEXTURE_TILES=0` | 70 s | 9,526 of 10,835 MiB | 19,073 MiB | 24,995 | — | 14 s |
| P | pool, `XTP_TEXTURE_TILES=0` | 203 s | 9,746 of 11,031 MiB | 18,370 MiB | 1,212 | 1 | 22 s |
| PT | pool and tiles (unset: this default) | 144 s | 5,266 of 11,176 MiB | 14,079 MiB | 1,216 | 0 | 192 s |

Seven earlier pool-and-tiles sessions loaded the world in 106 to 199 s.
They ran on binaries that differ only outside textures.

What they show:
- **Without the pool and its reserved textures, a 16 GiB host does not
  stay in the world.** Commit runs out 3 and 14 s after the title gets
  control (B1, B2).
- **The reserved textures are what saves the memory.** The pool alone (P)
  holds as much card and private memory as committed textures. With tiles
  (PT), 4.0 to 4.5 GB less video memory is in use and 4.3 to 5.0 GB less is
  private at the world than in B1, B2 and P. The world then runs for 192 s
  before the guard. The guard then
  stops it for other growth: the buffer copies of title memory and the
  snapshot mirrors (fogydes/XTP#45).
- **Loading is slower with the pool on this host:** 144 s against 70 and
  85 s, and 106 to 199 s over eight pool-and-tiles sessions. Residency
  requests drop from about 25,000 to 1,200.
- The card stays well below its budget (5.3 of 11.2 GB at the world, 7.2 GB
  at the peak), so the reserve does not push it past the ceiling. No
  texture was handed out before its heap was resident with tiles.

So `low_memory_host` (`resource/pool.rs`) turns both on when they are unset
on a host with less than 32 GiB of RAM. Elsewhere they stay off, as §3.3
decided.

## 4. Review items of dark-world's delivery (Samuel's relay)

1. Residency fences in order: one fence a request (2.3).
2. Use-after-free in `texture_at`: a resource is handed out only while it
   is in the map, looked at under its lock; a failed creation looks the key
   up again instead of using `before`, which a remapping may have released
   meanwhile. The returned pointer stays valid for `GRACE` (2 s): draw.rs
   uses it while it records the draw, so nothing changes there.
3. A texture not resident in time is no longer refused for good (2.3).
4. A stale target copy: views that show a released target's copy are
   dropped when the title makes another texture (one the GPU does not
   write) at that address. Narrower than "shared and no source left": that
   would drop the copy as soon as the target's object is released, while
   the title may still read what was drawn through a view (title memory
   holds it on the console).
5. `alias` compares the title format's family (a stand-in format is shared
   by title formats of other families); `memory_remapped` also releases
   placed views that start before the range and overlap it.

## 5. Critical review

What the brief asked and what came out of it:

- **The design rested on a cause that main had already moved.** The brief
  (from stall's first measurements) put the stall in each committed
  texture's synchronous `MakeResident`. By the time the pool ran in game,
  dark-world's NOT_RESIDENT path was on main and the creation call no longer
  waited (T2: two slow creations). The pool removes what was left of it from
  the title's thread, and cuts residency requests to a third, but the load is
  as slow: the remaining cost is on the GPU, in allocations the pool does not
  touch. I did not measure that before writing the pool; a session counting
  where the GPU time goes (`XTP_GPU_TIMING`) would have said it first.
- **The micro-benchmark measures WARP** (§3.1), whose "video memory" is the
  process's own and which has no VidMm ceiling: it shows the cost of the
  calls and the bytes per texture, not the stall. Only a session on the
  4080 shows the stall, and Series X sessions are over.
- **What the pool does not cover:** targets, depth and unordered access
  textures (1,187, 4.2 GiB in A3) stay committed, one residency request
  each. Placing them needs their first use initialised (a clear or discard,
  D3D12's rule for placed RT/DS) in a list, which is `list.rs`/`host.rs`
  work, and their contents to start at zero as today.
- **Below the threshold, nothing is gained or lost in time; the reserve
  costs memory.** With 512 MiB it left 865 MiB unused (now one heap).
- **The A stall is unexplained.** It happened once in three sessions with the
  pool on (A; A2 and A3 passed). A pool bug is possible; the default
  being off keeps it out of the way until a Series S session with stacks.

Known limits:

- A view made while its texture is not resident is a null view, and stays
  one until the title writes that descriptor again (descriptor/host.rs makes
  views when the title writes descriptors). Below the threshold this does
  not happen (the 10 ms wait); past it, textures placed in a heap still
  becoming resident are invisible until their descriptor is rewritten. A
  descriptor-side follow-up would remake such views once resident.
- Directly indexed views are not uploaded (texture.rs shortcut): in the
  pool their texels are what the range held before (another texture's)
  instead of zeros.
- texture.rs makes a committed upload buffer per texture it detiles (its
  staging buffer): one allocation and residency operation each, in system
  memory. Not measured against the threshold yet.
- The reserve keeps up to one heap per class unused (256 MiB + 32 MiB by default; 865 MiB in A3 with 512 MiB).
