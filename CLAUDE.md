# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`jmlx` is a pure, idiomatic Java 25 framework for Apple Silicon GPU tensor operations, wrapping Apple's
native MLX (`mlx-c`) with zero-copy FFM (Project Panama) bindings. The **v0.1 vertical slice** described
in `req/initial-plan.md` (native bootstrap, generated bindings, memory management, a handful of tensor
ops) is done, and two further phases are built on top of it: `req/phase3-plan.md` (broadcast-compatible
ops, relaxed matmul, `slice`, a batched `eval`) and `req/phase4-plan.md` (`se.alipsa.jmlx.nn` -- `Module`,
`Linear`, `QuantizedLinear`, normalization/activation layers, `MultiHeadAttention` with RoPE and a KV
cache, and reverse-mode autograd via `MLXGrad`/`ModuleGrad`), both delivered. `req/plans/phase5-m1-plan.md`
(`MLXIO` -- safetensors/GGUF checkpoint I/O) and `req/plans/phase5-m2-plan.md` (`jmlx-tokenizer` --
byte-level BPE tokenizer + `hfjinja` chat-template rendering) are also delivered. `req/project-outline.md`
describes the full multi-phase vision (autograd, `se.alipsa.jmlx.nn`, safetensors/tokenizers, model
loading); the rest of Phase 5 (model implementations, i.e. M3) is not yet implemented.

Requires macOS on Apple Silicon, macOS 26+, and a Java 25 toolchain.

## One-time native bootstrap

The native runtime (mlx-c compiled against a pinned `mlx-metal` wheel) is not checked in and must be
staged before anything native will run:

```sh
./scripts/bootstrap-native.sh
```

This is idempotent and downloads/verifies (trust-on-first-use, pinned SHA-256) a pinned `mlx-metal`
wheel and jextract build, clones `mlx-c` at a pinned commit, builds it against the wheel's MLX, and
stages the result as a **flat** directory at `native/install/lib/` (`libmlxc.dylib`, `libmlx.dylib`,
`libjaccl.dylib`, `mlx.metallib`). Flatness is a runtime invariant, not incidental: MLX finds
`mlx.metallib` and dyld resolves `@rpath` siblings by colocation with `libmlx.dylib`/`libmlxc.dylib`;
splitting into `lib/`/`share/` breaks both.

The Java bindings in `jmlx-ffi/src/main/generated/java` are committed jextract output, not generated
on the fly. Only regenerate them if the pinned mlx-c commit changes:

```sh
./scripts/regen-bindings.sh
```

After running it, `git diff --exit-code jmlx-ffi/src/main/generated/java` must be clean if nothing
should have changed — this is the bindings-drift check.

`scripts/checkDependencies.zsh` is a read-only report of available updates (Gradle plugins/deps, the
wrapper, the pinned `mlx-metal` version, and — since mlx-c versions independently of MLX — the mlx-c
tag that pairs with a newer wheel). `scripts/updateMlx.zsh <mlx-c-commit-sha>` repins `MLX_C_COMMIT` in
`bootstrap-native.sh` and re-runs bootstrap + regen for you; it takes a full 40-char SHA, not a short
one.

## Build, test, run

```sh
./gradlew build                # compiles jmlx-ffi, jmlx-core, jmlx-examples
./gradlew :jmlx-core:test       # memory lifecycle, numeric correctness, native error path
./gradlew test --tests "se.alipsa.jmlx.core.MLXArrayTest"   # a single test class
./gradlew :jmlx-examples:run    # runs HelloMLX end-to-end on real GPU hardware
```

Every module's native-dependent tests are **skipped, not failed**, when `native/install/lib/mlx.metallib`
is absent — see `@EnabledIfNativeAvailable` (a `jmlx-ffi` test fixture, shared via `testFixtures`,
delegating to `NativeLoader.ensureLoaded()` itself so the skip gate can never diverge from the real
loader logic). Don't add a separate existence check to decide whether native tests should run.

`jmlx-ffi` also has a `loaderGuardTest` task (wired into `check`) that exercises `NativeLoaderMissingMetallibTest`
in its own JVM against a disposable copy of the native dir with `mlx.metallib` excluded — it's excluded
from the regular `test` task because `NativeLoader.ensureLoaded()` caches its outcome, so it would race
every other native test over the real staging directory if run in the same JVM.

## Code style

Hand-written sources are Google Java Style, 2-space indent, 120-column width
(`config/spotless/eclipse-java-google-style-120col.xml`, `config/checkstyle/checkstyle.xml` — both
derived from Google's own upstream artifacts, with deviations documented in comments at the top of
each file). The generated jextract bindings under `jmlx-ffi/src/main/generated/java` are exempt from
both and must stay byte-identical to `scripts/regen-bindings.sh`'s output — never hand-edit them.

```sh
./gradlew spotlessCheck                                      # verify formatting
./gradlew spotlessApply                                       # reformat in place
./gradlew checkstyleMain checkstyleTest checkstyleTestFixtures # style/lint (part of build/check)
```

## Architecture

```
jmlx-examples    HelloMLX                          demo, end-to-end test
       |
jmlx-core        se.alipsa.jmlx.nn                 Module, Linear, QuantizedLinear,
                                                    RMSNorm/LayerNorm/SiLU/GELU/Embedding,
                                                    MultiHeadAttention, KVCache, ModuleGrad
                 se.alipsa.jmlx.core                MLX, MLXOps, MLXShape, MLXFast, MLXQuant,
                                                    MLXRandom, MLXGrad, MLXIO, MLXArray, DType,
                                                    MLXException
                 se.alipsa.jmlx.memory              MLXScope
       |
jmlx-ffi         se.alipsa.jmlx.ffi.*              committed jextract output
                 NativeLoader                      hand-written
       |
native/install/lib/  libmlxc.dylib                 built by scripts/bootstrap-native.sh
                     libmlx.dylib  libjaccl.dylib   staged from the mlx-metal wheel
                     mlx.metallib                   staged from the mlx-metal wheel

jmlx-tokenizer   se.alipsa.jmlx.tokenizer           HfTokenizer, ChatTemplateRenderer,
                                                    TokenizerJson/TokenizerJsonLoader, Vocabulary,
                                                    BpeMerger, ByteLevelCoding,
                                                    ByteLevelPreTokenizer, ByteLevelDecoder,
                                                    AddedTokenSplitter, TextNormalizer,
                                                    PostProcessorApplier, TokenizerException
                 (no "|" above: pure Java, no dependency on jmlx-ffi or native/install/lib)
```

Three modules, deliberately: the jextract output for `mlx/c/mlx.h` is a large generated blob. Isolating
it in `jmlx-ffi` means it compiles once and stays untouched by day-to-day iteration on `jmlx-core`,
keeping incremental builds fast and generated code out of review diffs. `jmlx-core` depends on
`jmlx-ffi` as `implementation`, not `api` — `MLXArray`/`MLX` wrap raw `MemorySegment` handles behind
plain Java types and never expose `jmlx-ffi` on their own public surface.

**A fourth module, `jmlx-tokenizer` (Phase 5 M2, `req/plans/phase5-m2-plan.md`), sits outside this
native chain entirely — the first pure-Java module in this codebase.** It has no dependency on
`jmlx-ffi` and never touches `native/install/lib`; its only dependencies are ordinary Maven artifacts
— Jackson (`tools.jackson.core:jackson-databind`, for parsing `tokenizer.json`) and `hfjinja`
(`se.alipsa:hfjinja`, for rendering HF `chat_template` Jinja strings). `HfTokenizer` is a from-scratch
Java port of the byte-level-BPE pipeline (vocabulary/merges, byte-level pre-tokenization and decoding,
normalization, added-token splitting, post-processing), not an FFM binding over `mlx-c` or any other
native tokenizer library — unlike `jmlx-core`, it does not call `NativeLoader.ensureLoaded()` and is
not part of the "Loading order matters" native-guard discussion below (which is specific to `MLX`,
`MLXScope`, `NativeOps`, `MLXGrad`, and `MLXIO`).

**Loading order matters.** jextract binds each downcall's method handle lazily, in a private
per-function holder class, the first time that function is called — and that first call fails unless
the dylib is already loaded by then. `MLX`, `MLXScope`, `NativeOps`, `MLXGrad`, and `MLXIO` each have a
static initializer that calls `NativeLoader.ensureLoaded()` for exactly this reason; any of the five can
be the first one touched, so each guards independently rather than relying on load order. `NativeOps`'s
guard covers `MLXOps`/`MLXShape`/`MLXFast`/`MLXQuant`/`MLXRandom` transitively, since every op in those
classes reaches native only through `NativeOps`'s own `checked`/`binaryOp`/`unaryOp`/`shapeOp`-family
helpers — none of those classes needs (or has) its own guard. `MLXGrad` and `MLXIO` both need their own
regardless: `MLXGrad` calls `mlx_h.*` directly at 17 call sites (`mlx_closure_*`/`mlx_value_and_grad` and
their frees), and `MLXIO` calls `mlx_h.*` directly for safetensors/GGUF I/O and their five
non-`mlx_array` struct types (`mlx_string`, `mlx_vector_string`, `mlx_map_string_to_array`,
`mlx_map_string_to_string`, `mlx_io_gguf`) — both bypass `NativeOps`'s helpers entirely, so neither can
rely on `NativeOps`'s guard the way the op facades do.

**`NativeLoader`** loads `libmlxc.dylib` via `System.load(absolutePath)`, not
`SymbolLookup.libraryLookup`: the bindings are generated *without* a `-l` flag, so their internal
`SYMBOL_LOOKUP` is `loaderLookup().or(defaultLookup())`, which only sees libraries registered with the
calling classloader — exactly what `System.load` does and `libraryLookup()` does not. It resolves the
native library directory from the `jmlx.library.path` system property first, then `JMLX_LIBRARY_PATH`.
The root `build.gradle` wires every `Test` task's `jmlx.library.path` to the real
`native/install/lib`; `jmlx-examples`'s `run` task does the same, but its `installDist`/`distZip` start
scripts deliberately do *not* get that property baked in (it's an absolute build-machine path) — a
distributed launcher elsewhere is expected to set `JMLX_LIBRARY_PATH` instead.

mlx-c's default error handler is `printf` + `exit(-1)` on failure, which would kill the JVM before any
status code is observable. `NativeLoader` installs a replacement handler (an FFM upcall stub, kept
alive in a process-lifetime `Arena` since mlx-c may call it any time) that just records the message on
a thread-local, since the mlx-c error convention fires synchronously on the calling thread.
`NativeOps.checked` (moved from `MLX.checked` in an earlier phase) clears that thread-local
immediately before every native call (not just after a failure) and wraps a
non-zero status into `MLXException`, attaching the recorded native message when present. A few mlx-c
entry points (e.g. `mlx_array_new_data`) have no status return at all and only signal failure via the
error handler plus a null `ctx` in the returned struct — those call sites check for that explicitly.

**Memory model:** `MLXScope` owns every native `mlx_array` handle allocated through it (it implements
`SegmentAllocator` so it can be passed directly to mlx-c's struct-returning constructors) and frees them
in reverse insertion order on `close()`. A `Cleaner` backstop exists in case a scope is never explicitly
closed, but the cleanup action is registered against a private `Holder` object, never the `MLXScope`
itself — capturing the scope in the cleanup action would make it permanently reachable and the action
would never run. `MLXScope` and every `MLXArray` allocated from it are confined to the thread that
created the scope (checked via `ensureOpen()`/thread-identity checks on every access, including
centrally in `MLXArray`); the sole exception is the JVM Cleaner thread, which only ever touches the
`Holder`, never the scope. An op's result is allocated into the innermost scope among every non-null
`MLXArray` operand it's given (`NativeOps.scopeOf`), not just its first — a rule that rejects two
operands only when their scopes are *unrelated* (siblings, or two independent roots), not merely
different: a related ancestor/descendant pair is legal and resolves to the descendant (e.g.
`Linear.forward`'s `add(y, bias)` across a model scope and a step scope, or
`MLXQuantTest.dequantizeWithBiasesInAChildScopeOfWAllocatesIntoTheChild`). `MLX.array(scope, ...)` and
a few other ops (creation ops, `MLXRandom.normal`/`uniform`, `MLXGrad.Fn#apply`) take the scope
explicitly since they have no operand to infer one from; `MLX.hoist`, `MLXShape.transpose(MLXArray,
MLXScope)`, and `MLXQuant.quantizedMatmul`'s explicit-target overload instead override `scopeOf`'s own
answer, letting a caller push a result toward an ancestor or descendant scope on purpose. See `MLX`'s
own class javadoc for the full, authoritative version of this rule.

**Lazy evaluation:** every op in `MLX` only builds mlx's lazy computation graph; nothing runs on
GPU/CPU until `MLX.eval(...)` (or the implicit eval inside `MLXArray.toFloatArray()`) forces it.
`toFloatArray()` also forces contiguity first (`mlx_contiguous`, row-major) before reading the raw data
pointer — a lazy op like `transpose` can otherwise yield a strided view, and reading raw data without
that step would return plausible-looking values in the wrong order instead of crashing.

Shape is plain `int[]`; there is no `Shape` type in this slice. `DType` covers `FLOAT32`, `INT32`,
`BOOL`, `UINT32` (the packed-weight dtype `QuantizedLinear` validates against), `FLOAT16` and
`BFLOAT16`. `defaultDevice()`/`defaultStream()` are resolved once from mlx-c's own defaults and cached
for the process lifetime — this slice doesn't expose device switching.

## Native version pinning

mlx-c is versioned independently of MLX itself (mlx-c's own `CMakeLists.txt` pins an exact MLX tag via
`GIT_TAG`), so a newer `mlx-metal` wheel does not imply any particular mlx-c commit works with it — the
two are only bumped together, via `scripts/checkDependencies.zsh` (which cross-references mlx-c's tags
against the wheel version) and `scripts/updateMlx.zsh`. Current pin: mlx-c `fba4470` (tracks `v0.6.0`)
against `mlx-metal==0.31.2` (`macosx_26_0_arm64`) — see Decision 9 in `req/initial-plan.md` for why the
latest 0.32.0 wheel was deliberately not used.

`scripts/jextract-overrides/mlx/c/half.h` is a trimmed override copied over the staged header before
running jextract: the real header's `__bf16` typedef (the raw C half-precision arithmetic type) is a
hard parse error for jextract on this target. This only strips that C-level typedef, not
bfloat16 support itself — `DType.BFLOAT16` (mapping to mlx-c's own `MLX_BFLOAT16` enum constant) is
fully supported by this slice; jextract only ever needed to parse past the C type, never bind it.
`regen-bindings.sh` runs jextract twice — an unfiltered
discovery pass, then a second pass restricted to symbols whose `--dump-includes` header path is under
`mlx/c/` — both to trim Darwin system-header spillover and because (independently) an unfiltered
whole-umbrella run silently drops `mlx_array_new`/`mlx_array_free`/`mlx_array_new_data`/
`mlx_array_new_data_managed` from the generated Java with no warning. See Decision 10 in
`req/initial-plan.md` for the full story if either script needs changing.

## Requirements docs

`req/initial-plan.md`, `req/phase3-plan.md`, and `req/phase4-plan.md` are living design documents, not
just history — they record the *why* behind non-obvious decisions (native library loading strategy,
jextract's two-pass generation, memory confinement rules, the `se.alipsa.jmlx.nn` module framework and
autograd design) that the code comments themselves point back to. `req/plans/phase4-m4-plan.md` is
`QuantizedLinear`'s own implementation plan, and `req/plans/phase5-m1-plan.md` is `MLXIO`'s, each
amended in place (not rewritten) as post-merge/post-implementation review found and fixed real gaps in
it. When touching `NativeLoader`, `MLXScope`, `MLX`, `NativeOps`, `MLXIO`, anything under
`se.alipsa.jmlx.nn`, or the bootstrap/regen scripts, check whether the relevant decision is already
recorded in one of these before re-deriving it. `req/project-outline.md` is the original,
broader multi-phase vision; where it conflicts with `req/initial-plan.md` on scope or naming, the
latter is authoritative for what's actually being built now.
