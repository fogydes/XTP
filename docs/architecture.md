# Architecture

```
 user's files (outside repo)             XTP
 ───────────────────────────             ─────────────────────────────────────────
 game assets ──────────────────────────► xtp-pipeline + xtp-game
                                         (xtp-formats, xtp-shader)
                                                    │
                                                    ▼
                                         cache_dir: converted data
                                                    │
 game executable ─► xtp-loader ────► title code (native x86-64)
                    prepare, map, bind      │ imports
                                            ▼
                                         xtp-hle ── host Windows (shared APIs)
                                            │ D3D12.X
                                            ▼
                                         xtp-gpu ─► host D3D12
                                            │ shaders at pipeline creation
                                            ▼
                                         xtp-shader ◄─► cache_dir
```

Shaders are translated when the title creates a pipeline, from the bytes it
passes, wherever it loaded them from (executable, archive, memory).
Converting them ahead of time from files is only a warm-up.

## Crates

| Crate | Role |
|-------|------|
| `xtp-core` | config, asset-policy guard, logging |
| `xtp-pe` | read-only PE32+ and import-library analysis |
| `xtp-loader` | prepare, map, bind and start the title image |
| `xtp-hle` | console API implementations, host pass-through policy |
| `xtp-gpu` | D3D12.X objects, command lists and GPU memory → host D3D12 |
| `xtp-shader` | Xbox shader containers → host shaders |
| `xtp-formats` | the GPU's texture layouts (sizes, tiling, detiling), audio codecs; planned: BCPack |
| `xtp-pipeline` | converter trait, content-addressed cache |
| `xtp-game` | BTP 6-specific formats, build identities, hooks |
| `xtp-testkit` | dev-dependency only: visible test skips (`skip`, `XTP_REQUIRE`) |
| `apps/xtp` | command-line front end |
| `xtask` | asset-policy check, git hooks |
| `tests/fixtures/hle-probe` | a stand-in "title" (our own code) for loader tests |

Everything except `xtp-game` is platform-level and can be tested with
any Xbox Series title.

For the opt-in indirect-light stand-in, `xtp-game` supplies compute program
hashes and table bindings at startup. `xtp-gpu` matches those hashes when
it creates shaders and records its passes after the corresponding direct
dispatches, using the views already bound by the translated shader.
Radiance and estimate history follow execution order on the host queue.
`XTP_RT_STANDIN` can clear a dropped trace to misses upstream, while
`XTP_AMBIENT` works on the diffuse output of a resolve that ran.

Ray-tracing dispatches reuse up to 64 host command signatures, keyed by
the retained root signature, root SRV parameter and argument stride.
The least recently used entry is evicted when full. Recordings keep
their own COM references through completion, so eviction does not shorten
the lifetime of an in-flight command signature. Creation failures are
retried on later calls rather than cached.

The pipeline cache uses a 1 GiB payload budget by default. Pass a byte
budget to `Cache::open_with_limit`, or zero to disable disk access. When
space is needed, the cache evicts entries in write-time order and skips
active readers and files it cannot remove. Disk errors still let the caller
use a successful conversion. Each disk read verifies the payload length
and BLAKE3 checksum. Older entries without a header are rebuilt on demand.

Threads sharing a cache directory in one process share a budget, reader
pins and in-flight conversions. A converter or decoder must not look up its
own in-flight key, because it would wait for itself. Another process can
publish a valid entry that this process discovers on lookup. Each process
tracks its own pins and budget. `Cache::stats` reports indexed entries,
payload bytes and successful reuses; legacy files count at their full size.

## Import resolution

For every `dll!function` the title imports, in order:

1. our implementation in `xtp-hle::modules` (always wins);
2. the host's own function, if `HostPolicy` allows that DLL;
3. an export of a DLL the title ships beside its executable, mapped like
   the executable (only DLLs neither of the above serves);
4. a generated stub that names the import when called (`--unimplemented
   trap` breaks into the debugger and aborts; `zero` logs once and returns 0).

`LoadLibrary*` of a console-only DLL we implement (e.g. `xgameruntime.dll`)
returns a pseudo handle; `GetProcAddress` on it finds our functions.
`LoadLibrary*` of a title DLL returns its mapped image, mapping it first if
nothing imports it. By name, as by import, a title gets only host DLLs
`HostPolicy` passes through, and `GetProcAddress` on one returns our
override where import binding would.
`QueryApiImpl` hands out our runtime objects by provider and interface id
(`xtp-hle::modules::xgameruntime` lists the areas); online areas answer
as an offline console. Game saves are files under `XTP_DATA_DIR`.

Title storage is a drive on the console, and titles build paths on that.
`xtp-hle::modules::files` binds every path-taking title import
(kernel32 and the C/C++ runtime) and rewrites paths on a virtual drive to
the host folder behind it; persistent local storage is one such drive over
a folder under `XTP_DATA_DIR`. Other paths reach the host unchanged.

An exe's code reaches its static TLS as slot 0; mapped as a DLL it gets
another slot, so the loader redirects those loads (`xtp-loader::tls`).
`xtp run --execute` installs a crash reporter that names faults as
`module+rva` and unwinds the faulting thread's stack
(`xtp-loader::crash`). Handler installation through `report_crashes()` is
idempotent (`Once`). The same observer logs at most 16 first-chance C++
events, including rethrows (each verified title rethrow consumes one slot).
It accepts a title PC only when intervening frames belong to the exception
machinery: ntdll, kernelbase, ucrtbase, vcruntime140 or msvcp140 DLL families.
An unrelated host, HLE or unknown intermediary stops attribution even if a
title frame is deeper in the stack. A single nonblocking acquisition
snapshots title membership for the candidate PCs, without allocations under
the registry lock. A busy registry conservatively skips the event; poisoning
is recovered.
Metadata and strings are read fallibly with bounded lengths;
`std::exception` messages use the MSVC x64 non-virtual base layout without
calling object methods. Unsupported layouts retain a type diagnostic.
Unhandled C++ reports use the same fallible decoder, independently of the
first-chance budget. Debugger thread-name notifications set the current or
explicitly named host thread's description, restricted to this process.
Legacy extra parameters and reserved flags are tolerated; long readable
names are truncated. Both notifications continue normal exception dispatch.
Both observers return before any decoding or allocation once process exit
has begun, because terminated threads may still hold allocator or I/O locks.

`xtp census` reads the exports of eligible DLLs in each title folder
without mapping them or running title code. It retries unresolved
ordinals by name and shows which remaining imports have values in the
built-in stub table.
Matching title export forwarders are reported as unverified, regardless
of apparent target availability: census neither follows nor loads them and
does not claim native binding parity. They contribute to totals but not
resolved or missing counts.

## Shaders

`xtp_shader::ir::build` turns a native program into SSA, one
instruction per machine instruction. `xtp_shader::structure` then
removes EXEC: lane masks become per-lane booleans, wave-wide mask tests
become per-lane branches, and vector instructions keep a guard only where
a lane can reach them with its EXEC bit clear. `xtp_shader::resource`
traces each descriptor and scalar memory address to the root parameter it
comes from (table, heap, root descriptor or constant). A descriptor a
waterfall loop reads from the first lane (`v_readfirstlane_b32`) is traced
to the per-lane load or choice that gives each lane its own (a mask's
`v_cndmask_b32`, or an if/else whose arms each set it: traced on the paths
the lanes take, from structurisation), and accessed with a non-uniform heap
index in one pass. `xtp translate`
runs every shader it finds (containers, SGD5 files) through these and
lists what they cannot handle yet.
