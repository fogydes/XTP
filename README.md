# XTP

XTP is a compatibility layer for Xbox Series (GDK) titles on Windows. It
runs a title's unmodified x86-64 code in its own process, reimplements the
console system libraries, and translates D3D12.X commands and GFX10 shaders
to host D3D12 and DXIL at run time. The repository contains no game code or
data, no keys and no decryption, and XTP does not help anyone obtain title
files.

Nothing derived from a title (translated shaders, converted data) is
committed or distributed: it is made on first use and kept in the user's
cache folder. See [docs/architecture.md](docs/architecture.md).

## Status: experimental

No title is usable yet.

Tested on one machine: Windows 11, AMD RX 6700 XT.

> **Warning.** 3D rendering can currently hang a real GPU, and
> Windows then resets the display driver (every monitor goes black for a
> moment). To look further, use a diagnostics build with
> `XTP_D3D12_WARP=1`, which renders on the software adapter.

What each part does today and what it still needs:

| Subsystem | Today | Needs |
|-----------|-------|-------|
| Loader | Runs the title's executable and its DLLs in our process; crash reports name the faulting code and are kept in `crashes.log` | – |
| Runtime services | One signed-in offline user, licences, packages, saves that survive a kill at any step, text entry, message and error dialogs, text and SSML narration with installed Windows voices | Narration returns an empty WAVE for blank text or when Windows cannot produce audio. The dialogs and relaunch are untested live |
| Network | Offline by design: the title cannot reach the network | – |
| Files and streaming | Archives and streamed data read, zlib and BCPack decoded in parallel | Loading is limited by writes into title memory |
| Input | XInput pads with trigger rumble, keyboard, keyboard and mouse as a pad (keys by place, mouse as the right stick) | Trigger rumble and the mouse are untested in play |
| Audio | Menu, video and loading sound; XMA voices and Opus streams decode | Reverb is silent; Opus results unverified against the title |
| GPU: menus | Draws, compute and presentation on host D3D12; only changed title memory is copied | Frame pacing drifts on one screen in release builds |
| GPU: world | Mesh pipelines, shader-written buffers, descriptor lifetimes, indirect execution (arguments clamped), pipelines made by worker threads and kept on disk | Most of the scene is still missing (untranslated shaders drop draws); frames take seconds; real queries, depth bounds, HDR and scaling are missing |
| Shaders | Console GPU code recompiled to DXIL at first use, cached on disk. Menus: all. World: mesh 8,191 of 8,506, vertex/pixel 10,617 of 12,346, compute 1,643 of 2,539 | 64-bit image atomics, merged geometry stages, the ray-tracing builder's shaders; some compute programs need a GPU with 64-lane waves |
| Textures | Console tiled layouts and detiling for the modes the title uses | Render-target metadata planes |
| Series S mode (the default) | Smaller memory budgets; scenes load in seconds | 3D rendering, as above |

## Getting started

Requirements: Windows 10/11 x64, a CPU with AVX2, a Direct3D 12 GPU with
shader model 6.6 (and mesh shaders for the world), Rust (pinned in
`rust-toolchain.toml`).

Pixel shaders with wave operations take shader model 6.7
(`WaveOpsIncludeHelperLanes`, so that helper lanes take part in them as they
do on the console). A GPU without 6.7 refuses those pixel shaders, and their
draws are skipped.

The title also needs several GiB of available Windows commit at startup.
Check **Committed** under **Task Manager → Performance → Memory**. If
`crashes.log` reports Windows error 1455, close other applications or increase
the Windows paging file size, then retry.

```powershell
cargo xtask install-hooks                # policy and commit-message checks on every commit
copy xtp.example.toml xtp.toml   # then set your paths (outside the repo)
cargo test --workspace                   # needs no game data
cargo run -p xtp -- config           # check your paths
```

Run a title:

```powershell
cargo build --release -p xtp
target\release\xtp.exe run "<title folder>" --execute
```

With a folder, `xtp run` selects an existing executable named in
`MicrosoftGame.config`. If none matches, it looks for a single `.exe` in
the folder and reports why it skipped any named entries. When several match,
it lists them so you can pass one explicitly.

The window caption shows the framerate. Saves, settings and `crashes.log`
are kept under the data folder (default `%LOCALAPPDATA%\xtp\data`), derived
data under the cache folder.

Titles can read Windows caption colors, opacity, font, text size and high
contrast mode. Captions initially follow Windows ShowSounds, under Control
Panel > Ease of Access > Ease of Access Center > Use text or visual
alternatives for sounds. A title can turn captions on or off immediately
for its own session. XTP refreshes each host setting on demand at most
once per second. Without preferences, captions are off, with white text on
black, a transparent window, default font and edge, and 100% text size.
High contrast defaults to off. An active theme reports light or dark based
on its window background, or `Other` if Windows cannot supply that color.

Input: XInput pads; Xbox One and Series pads get their trigger motors too
(through Windows.Gaming.Input). The keyboard and mouse play as a pad while
the title's window has focus: a pad of their own while no pad is connected,
on top of the first pad otherwise. Keys go by place, not by label: the
letters below are a US keyboard's, and on AZERTY the key in W's place (Z)
pushes the stick forward just the same.

| Keys and mouse | Pad |
|------|-----|
| W A S D | left stick |
| mouse, I J K L | right stick |
| arrows | D-pad |
| Enter or Space, Backspace or R, X, Y | A, B, X, Y |
| Q / E, wheel down / up | left / right shoulder (a press a notch) |
| Z / C, right / left mouse button | left / right trigger |
| F, H or the middle mouse button | left stick click, right stick click |
| Esc, Tab | Menu, View |

The mouse plays while the cursor is held in the title's window, hidden:
from when the window is active until Alt+Tab or Esc (a menu's cursor is
yours); a click in the window takes it back, and is not a shot.
`[input]` in `xtp.toml` sets the mouse's sensitivity, curve, dead zone and
inversion, and other bindings (see `xtp.example.toml`), by control:
`a`, `b`, `x`, `y`, `menu`, `view`, `dpad_up` (`_down`, `_left`, `_right`),
`left_shoulder`, `right_shoulder`, `left_trigger`, `right_trigger`,
`left_stick_click`, `right_stick_click`, `left_stick_up` (`_down`, `_left`,
`_right`) and `right_stick_up` (likewise).

## Commands

```
xtp inspect <exe|dll> [--imports] [--exports] [--json]
xtp census [paths...] [--missing] [--strict] [--json]
xtp lib-exports <import.lib> [--list]
xtp shader-info [paths...] [--verbose] [--disasm]
xtp translate [paths...] [--validate] [--pipelines] [--list] [--jobs N]
xtp run <exe|folder> [--execute] [--unimplemented trap|zero] [--strict]
            [--console series-s|series-x] [-- <title arguments>]
xtp config
```

`xtp shader-info --disasm` decodes the GFX10 e64 f32 interpolation forms,
`image_gather4h`, the eight coarse gradient sample forms with `_g16`, and
`v_dot8_i32_i4`. The four non-comparison samples translate through
`SampleGrad`, which selects a level per lane. The comparison samples with
explicit gradients require SM 6.8. The interpolation, gather and signed dot
forms still need DXIL lowering.

`xtp inspect --json` emits a stable JSON object (`"format": "xtp-inspect 1"`)
with string fields `file`, `blake3`, `kind` (`"exe"` or `"dll"`), hex-string
fields `machine`, `image_base` and `entry_rva`, numeric fields `subsystem` and
`size_of_image`, and an `imports` array of `{dll, symbol, delay, resolution}`
objects (`delay` is a boolean; `resolution` is `"hle"`, `"host"`, `"missing"`,
with `"title"` reserved because static `inspect` does not map title-shipped
DLLs, so those report as `"missing"`). `--json` ignores `--imports` and
`--exports`.

`xtp census` classifies imports as HLE, host, title DLL, unverified,
table-stubbed, or missing. Each folder is scanned independently, without subfolders. A file
is scanned as one image with no title DLLs available. With no paths, census
uses the configured game and test-title folders. When a pair has different
outcomes across inputs, the totals count it once under the last applicable
class in that order. `--strict` disables host pass-through.

Matching title export forwarders are **unverified** in either mode: even a
target present in the folder does not prove Windows binding will succeed.
Census neither traverses nor loads those targets. Direct nonzero-RVA exports
remain title DLL imports; malformed or absent exports remain unresolved.
Unverified imports count toward totals, not resolved or missing coverage;
`--missing` lists them separately. Shared census expectations record them
without counting transitions to or from uncertainty as regressions or improvements.
Shared text keys quote names beginning with `#` or `"` to distinguish them
from ordinals. Regenerate older expectations containing those ambiguous names.

`xtp census --json` emits `"format": "xtp-census 1"` with numeric `images` and
`unreadable` fields, a `totals` object
(`{functions, hle, host, title, unverified, stubbed, missing}`), a `dlls` array
(`{dll, functions, hle, host, title, unverified, stubbed, missing, images}`), and
`missing`, `stubbed` and `unverified` arrays (each with `{dll, symbol, images}` rows).
Each array counts the images in its class. The `format` version number
increments on any breaking schema change; `unverified` is an additive field
and list in this unmerged format. Symbols remain strings; ordinal `#1` and
the literal name `"#1"` can appear as separate rows with the same display
string, while name hints do not create distinct functions.

The title sees an Xbox Series S unless `--console series-x` or
`XTP_CONSOLE=series-x` says otherwise: with Series S budgets the world
loads in seconds, where Series X's push the host GPU's resident memory
past the point at which Windows slows every new allocation, and loading
takes minutes (docs/analysis/series-s.md, stall.md). `cargo xtask play`
passes `XTP_CONSOLE` on as `--console`.

Logging: `XTP_LOG=xtp_loader=trace,info` (a release build logs
info and above).

## Debugging

A release build has no debugging aids. Build them in with

```powershell
cargo build --profile diag -p xtp --features diagnostics   # target\diag\xtp.exe
```

| Variable | Effect |
|----------|--------|
| `XTP_LOG=<target>=debug,...` | debug and trace output |
| `XTP_STACKS=20,60` | every thread's stack at those seconds |
| `XTP_WATCH=<rva>,...` | logs a title u32 when it changes |
| `XTP_KEYS=30:Enter,35:X` | presses keys at those seconds, focus or not (keys by place, named as in `[input.bindings]`) |
| `XTP_CRASH_SAVE=<rva>:<len>,...` | saves title memory to the cache folder on a crash |
| `XTP_CAPTURE=<dir>` | saves presented frames (powers of two, or `XTP_CAPTURE_FRAMES=<list>`) |
| `XTP_DUMP_TARGETS=1` | with the two above, saves every render target of those frames |
| `XTP_D3D12_DEBUG=1` | host debug layer |
| `XTP_D3D12_GBV=1` | with the above, GPU-based validation (bad descriptor use reported, not faulted; slow) |
| `XTP_D3D12_DRED=1` | on a GPU failure, the allocation hit and the commands in flight |
| `XTP_D3D12_DRED=contexts` | as the above, with DRED's breadcrumb contexts (marker texts kept by the runtime; slower, and empty on NVIDIA so far: docs/analysis/device-removed.md) |
| `XTP_RT_BUILDS=0` | (every build) the title's acceleration structure builds, copies and rehydrations are refused; traces bind an empty TLAS, where every ray misses |
| `XTP_SKIP_PIPELINES=<hash>,...` | omit draws and dispatches, including indirect ones, whose pipeline contains a listed 16-digit shader hash |
| `XTP_GPU_TIMING=1` | with the above, each list of a submission run and timed on its own; slow lists logged with their slowest commands |
| `XTP_D3D12_WARP=1` | host device on the software adapter (slow; cannot reset the driver) |
| `XTP_ADAPTER=<index>\|<name>` | (every build) host GPU to run on: its index in the startup log's `host adapter` lines, or part of its name (`XTP_ADAPTER=radeon`); by default the first high-performance hardware adapter, a discrete GPU before an integrated one |
| `XTP_PIPELINE_WAIT_MS=<graphics>,<compute>` | (every build) milliseconds per frame title threads may wait for pipelines being made before their draws and dispatches are dropped for that frame (default `2,8`) |
| `XTP_SHADER_LOOP_LIMIT=<n>` | (every build) translated shader loops stop after `n` iterations per lane instead of 16384 (2^14, under the host GPU watchdog); loops that only reread memory until it changes (spin waits) after 4096, or `n` if lower (own shader cache entries); a lower `n` shows whether loops hitting the limit make world frames slow, a higher one whether a wrong result comes from a legitimate loop the default cut short |
| `XTP_SHADER_LOOP_BUDGET=<n>` | (every build) a lane of a translated shader leaves, as at the loop limit, once it has gone back `n` times in all of its loops together instead of 1048576 (2^20; loops inside loops multiply their own limits); `0` turns the budget off (own shader cache entries) |
| `XTP_SHADER_LOOP_COUNT=1` | (every build) a lane a loop limit or the budget stops adds itself to its pipeline's counter (a root UAV, `u0` in space 2; own shader cache entries); with the frame line, a line gives the lanes stopped per frame and the pipelines that stopped the most, by their shaders' hashes, and `xtp_gpu::loopstat=debug` a line per frame with stopped lanes. Off, translations are byte for byte those without it |
| `XTP_DXIL_OPT=1` | (every build) clean-up passes over each translated shader before the driver gets it: dead code, constants and identities, one instruction for one value, what a branch taken says about its condition, one read for one place within a block; or only those named (`dead,fold,reuse,facts,loads`); a shader whose declared features they would change is kept as translated; off by default (own shader cache entries) |
| `XTP_DXIL_CENSUS=<file>` | (every build) appends one line per translated shader module to the file: its instructions by kind as translated and as written, and whether the clean-up passes were kept |
| `XTP_AUDIO_CAPTURE=<file.wav>` | records what the title plays (our process only) |
| `XTP_XMA_DUMP=<dir>` | saves the first 32 XMA voices' context and input |
| `XTP_APU_DUMP=<dir>` | saves the first 32 Opus streams' packets (up to 256 packets per stream) as `.opuscap` files outside the repository for reference comparison via `XTP_SAMPLE_OPUS` |
| `XTP_DSP_DUMP=<absolute dir>` | captures the first 32 DSP streams per process outside the repository. Each numbered stream directory holds `params.txt`, `ir.f32le` (omitted if unreadable or over 4 MiB), and the first four commands' buffers and gains as raw little-endian floats. On Windows, compare `.output-before.f32le` with `.output.f32le` to see how the command changed its destination. Disconnect waits for pending writes. |
| `XTP_WATCHDOG_HOOK=1` | every build: keeps the title's own watchdog from ending it after about 30 s without progress (slow world frames), and logs a warning (at most every 30 s) when it would have; a real deadlock then hangs for ever, with no `crashes.log` entry; known builds only, otherwise a warning and nothing patched |
| `XTP_TITLE_ASSERTS=continue` | every build: where the title's own assertion would end it because an entry of its archive store could not be made (`int3` at title+0x23d8f03; the entry's header was not read, docs/analysis/archive-null.md), it takes the title's own path for a null object instead, and logs how many times (at once, then at most every 10 s) and a census of that store; hides the failure, it does not fix it; known builds only, otherwise a warning and nothing patched |
| `XTP_XMEM_LEDGER=1` | every build: counts what the title asks of XMem by category (`XMemAlloc` by allocator id and memory type, committed resources included, the share our `XMemAllocDefault` serves, physical pages, virtual reserves, commits and frees, the budget reported) and logs each category that changed every 10 s: calls, failures, frees, live and peak bytes, bytes asked in all |
| `XTP_LOW_VA_RESERVE_MIB=<n>` | every Windows build: with the variable unset, a known build with a measured startup request gets a placeholder of that size below 4 GiB (185 MiB for the supported build). An explicit `n` from 1 to 4095 instead supplies up to `n` MiB in pieces when 32-bit placeholder allocations fail. The reserve commits no pages. Set `0` to disable it. |
| `XTP_PLAY_PRECOOK=1` | `cargo xtask play --measure`: runs `xtp precook` before the title, which a measured run skips by default (the title then translates and makes each pipeline at its first use; docs/analysis/precook.md §7) |
| `XTP_SKIP_INTRO=1` | every build: cuts the intro movie of the start-up front end (sent to its last frame when it starts), so the world starts loading about 13 s sooner (docs/analysis/intro-skip.md); known builds only, otherwise a warning and nothing changed |
| `XTP_SKIP_CPU_CHECK=1` | every build: skips the host CPU feature check that `xtp run --execute` performs before starting the title |
| `XTP_QUEUE_WAIT=0` | every build: a queue told to wait for a fence no signal has reached yet runs what it is given next at once, as before queues held it (to compare frame times); such waits are still counted on the `queue waits for fences so far` line |
| `XTP_VALUE_WAIT=0` | every build: a queue no longer holds a list that waits for a memory value (`Wait32BitValueX`, `Wait64BitValueX`) another queue or the CPU has not written yet; it runs at once, as before (docs/analysis/cross-queue-waits.md). The waits are still measured on the `list waits for memory values so far` line |
| `XTP_VALUE_WAIT_LOG=<n>` | every build: logs the first `n` (default 64) list waits for memory values one line each: at submission (queue, address, comparison, reference, how it was met), when a held list is released, and when one runs before it is met (always at least 16 of those) |
| `XTP_RT_STANDIN=1` | every build: the title's ray-tracing programs that do not translate yet (traced sun and local-light shadows, probe and irradiance-cache gathers, screen probes, reflections, particle collision) have their dropped dispatches' outputs cleared to "every ray missed" (shadow traces' textures lit; others as `XTP_RT_STANDIN_FILL=<program hash or role>[.<output>]=lit\|zero\|keep\|f:<v>\|u:<hex>;…` says); `ab[:<phase s>[:<settle s>]]` alternates off and on within one run; every build logs what each dropped trace would have written (`rt:` lines; docs/analysis/rt-standin.md) |
| `XTP_FRAMES_IN_FLIGHT=ab[:<phase s>[:<settle s>]]` | every build: one frame in flight off (phase A) and on (phase B) in turn within one run, phases of 20 s by default, the first 3 s of each not measured; each frame-rate line is tagged with its phase and `cargo xtask play` ends with the median per phase (docs/dev/START_HERE.md, "A/B a switch"); `1` on, `0` or unset off |
| `XTP_DIALOGS=auto` | every build: auto-dismisses `XGameUi` message and error dialogs without opening a window (message dialogs return `cancel_button`, error dialogs complete with `S_OK`; enabled automatically under `--autopilot`; does not affect text entry) |
| `XTP_SCRATCH_CHUNK_KIB=<kib>` | every Windows build: upload scratch chunk size per command list in KiB, rounded up to 64 KiB and capped at 4096 (4 MiB). When unset, `0`, or invalid, defaults to 256 KiB on hosts with less than 32 GiB of RAM and 4 MiB otherwise |

Every build appends crashes and host GPU failures to `crashes.log` in the
title's data folder. The loader also prints the first 16 title C++ throws
(decorated type name, readable MSVC `std::exception` message and title
`module+rva`) to stderr,
and maps legacy debugger thread names to Windows thread descriptions.

## Contributing

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) first: it
has the rules every change must follow (nothing from any game, no keys, no
console SDK material, no circumvention, GPL-3.0-or-later, no code
from the projects we only use as references), the commit and pull request
conventions and how review works. [docs/dev/START_HERE.md](docs/dev/START_HERE.md)
maps the code and lists good first issues. Before opening a pull request, run

```powershell
cargo xtask pr-check     # what CI checks, with a summary
```

No PC that runs the title? [docs/contributing-without-a-gaming-pc.md](docs/contributing-without-a-gaming-pc.md) lists work that needs no GPU.

Report vulnerabilities privately as [SECURITY.md](SECURITY.md) describes.
Everyone taking part follows the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).

XTP started from VIctory by Marvisak (GPL-3.0-or-later, commit `ffe310e`);
see [NOTICE.md](NOTICE.md) for the origin of the code and the projects used
as references only.
