# jmlx v0.1 — Vertical Slice

## Context

`req/project-outline.md` describes `jmlx`: an idiomatic Java 25 framework wrapping Apple's native
`mlx-c` via Project Panama (FFM), targeting Apple Silicon GPU tensor ops and eventually LLM
inference. The outline lays out five phases ending in Llama/Qwen text generation — realistically
months of work.

**The repository contains no jmlx code, and the build is currently red.** Verified:

```
$ ./gradlew compileTestJava
> Task :lib:compileTestJava FAILED
  LibraryTest.java:11: error: cannot find symbol
          Library classUnderTest = new Library();
```

`lib/src/main/java/org/example/Library.java` has been deleted, but
`lib/src/test/java/org/example/LibraryTest.java` still references it. `lib/src/main/java/` now
contains only an empty `se/alipsa/jmlx/` directory — a statement of intended namespace. The build
otherwise remains the Gradle 9.7 `init` template, with unused `guava` and `commons-math3`
dependencies and JUnit Jupiter 6.0.1.

**Restoring a green build is step 1, not a side effect.** Every "still passes" claim in the
Verification section is meaningless until there is a passing baseline to regress against.

The build machine is capable but unprovisioned:

| Requirement | Status |
| --- | --- |
| JDK 25.0.3+11-LTS (FFM is final, no preview flags) | present |
| Apple Silicon arm64, macOS 26.6.1, Xcode 26.4 | present |
| `cmake` | **missing** |
| `jextract` | **missing** |
| `libmlx` / `libmlxc` / `mlx/c` headers / `mlx.metallib` | **missing** |

The risk this plan addresses: the outline commits to a large API surface (`jmlx.nn`, quantization,
safetensors, tokenizers) before anything has proven that Java can talk to MLX at all. Four things
are genuinely unvalidated — that jextract can consume `mlx/c/mlx.h` cleanly (specifically `half.h`'s
`__fp16` / `__bf16` and any `<complex.h>` use), that mlx-c's by-value struct convention survives the
FFM boundary, that `mlx.metallib` can be located at runtime, and that native array lifetimes can be
managed safely from the JVM.

**Intended outcome:** one demo that takes a `float[]`, builds arrays, runs `add` and `matmul` on the
Metal GPU, calls `eval()`, and reads results back to a `float[]` — with no leaks and no crashes.
That single path exercises every architectural layer in the outline. Once it is green, broadening
the op surface and adding `se.alipsa.jmlx.nn` become mechanical rather than speculative.

## Decisions taken

1. **Scope** — thin vertical slice: Phase 1 + Phase 2 + a minimal slice of Phase 3.
2. **Native bootstrap** — a pinned, checked-in shell script, run manually. Not a Gradle task.
3. **Bindings** — jextract output committed to the repo; contributors do not need jextract.
4. **Memory** — scope-primary with a `Cleaner` backstop.
5. **Modules** — three: `jmlx-ffi`, `jmlx-core`, `jmlx-examples`.
6. **Native acquisition** — fast path: take prebuilt binaries, C++ headers and the CMake package
   from the PyPI **`mlx-metal`** wheel, and cmake-build only `mlx-c` against them with
   `-DMLX_C_USE_SYSTEM_MLX=ON -DBUILD_SHARED_LIBS=ON`. Static linking is deferred to packaging.
7. **Namespace — `se.alipsa.jmlx.*`**, matching the existing empty source directory and the
   `Alipsa/jmlx` remote. Reverse-domain naming is required for Maven Central and gives collision-free
   JPMS module names. This supersedes `jmlx.core` / `jmlx.nn` in `req/project-outline.md`; updating
   that document is tracked as work in §8, not left implicit here.
8. **Generated-lookup composition** — jextract emits its own library lookup in a static
   initializer regardless of flags, which would bypass `NativeLoader`. How the two compose is
   decided by the §2 probe and **recorded as a new numbered entry in this Decisions list**, plus a
   comment in `scripts/regen-bindings.sh`, before any bindings are committed. Notably **do not**
   copy the `--use-system-load-library` flag from the vecxt template below: it emits
   `System.loadLibrary("mlxc")`, reintroducing both problems §5 exists to avoid.

## Research findings that shaped this plan

Verified during design; recorded because several contradict reasonable assumptions.

- **jextract for JDK 25 exists.** `25-jextract+2-4` (2025/11/25), including a `macos-aarch64`
  build, confirmed by direct HTTP 200 fetch of `jdk.java.net/jextract`. No fallback to hand-written
  `MethodHandle`s is needed. Some third-party mirrors still describe `25-jextract+1-1` as current;
  the bootstrap resolves the build from the page rather than hardcoding either string.
- **mlx-c does not build against a system MLX by default.** From its `CMakeLists.txt`:
  `option(MLX_C_USE_SYSTEM_MLX "Use system MLX" OFF)` (line 16) with an `else` branch that
  `FetchContent`s `ml-explore/mlx` at `GIT_TAG v0.31.2` and builds the entire C++ core — precisely
  the slow path Decision 6 exists to avoid. Setting it `ON` switches to `find_package(MLX REQUIRED)`.
  Separately, `option(BUILD_SHARED_LIBS ... OFF)` (line 14) means the default build produces a
  **static `libmlxc.a`**, not the `libmlxc.dylib` this plan assumes. Both flags are mandatory.
- **`find_package(MLX REQUIRED)` carries no version constraint**, so a mismatched MLX is accepted
  silently. This is not hypothetical: the current `mlx-metal` wheel is MLX **0.32.0** while mlx-c
  pins **v0.31.2**. The bootstrap must assert the pairing and fail loudly.
- **The dylibs are in the `mlx-metal` wheel, not the `mlx` wheel.** Verified by reading the wheels'
  zip central directories. `mlx-0.32.0-…macosx_26_0_arm64.whl` is **562 KB** and contains only
  Python plus `mlx/core.cpython-310-darwin.so` — no dylib, no metallib. It declares
  `mlx-metal==0.32.0; platform_system == "Darwin"`.
  `mlx_metal-0.32.0-py3-none-macosx_26_0_arm64.whl` (56.5 MB) contains everything needed:

  ```
  mlx/lib/libmlx.dylib   mlx/lib/libjaccl.dylib   mlx/lib/mlx.metallib
  mlx/include/mlx/ (233)  mlx/include/metal_cpp/ (139)  mlx/include/jaccl/ (10)
  mlx/share/cmake/MLX/{MLXConfig,MLXConfigVersion,MLXTargets,MLXTargets-release}.cmake
  mlx/lib/cmake/jaccl/jacclTargets{,-release}.cmake
  ```

  The C++ headers and CMake package config needed by `find_package(MLX)` are both present, which is
  what makes the fast path viable at all.
- **`mlx.metallib` is a separate, large, colocated file.** Sizes are *approximate and unconfirmed*:
  the measured `mlx-metal` **wheel** sizes are 40.82 MB (`macosx_15_0`) and 56.51 MB
  (`macosx_26_0`); the ~130 MB / ~162 MB figures circulating for the metallib itself are the
  uncompressed sizes and were gathered before it was established which wheel ships it. **Measure
  after extraction during §2 and replace these numbers**, since §3.7's size assertion depends on
  the real figure. MLX locates the file via `dladdr` relative to the image containing MLX.
  `mlx_metal_set_metallib_path` exists in C++ but is **not exposed in mlx-c**, so there is no API
  escape hatch and no environment variable. A missing metallib fails on the **first GPU op**, not
  at load — a late, confusing failure this plan must design against.
- **`libmlx.dylib` has its own `@rpath` sibling**, `libjaccl.dylib` (~930 KB). The canonical set to
  stage together is `libmlx.dylib` + `libmlxc.dylib` + `libjaccl.dylib` + `mlx.metallib`.
- **`SymbolLookup.libraryLookup(path, Arena.global())` is preferable to `System.load`.** It maps to
  `dlopen` while bypassing the classloader registry, so it is immune to
  `UnsatisfiedLinkError: … already loaded in another classloader`. `Arena.global()` specifically —
  a confined arena's close would `dlclose` MLX out from under the process.
- **Prior art is essentially nonexistent.** A GitHub code search for `"mlx_array" "MemorySegment"`
  returns zero results. The one real precedent is
  [Quafadas/vecxt](https://users.scala-lang.org/t/scala-apple-mlx/12032) (Scala 3 + jextract +
  mlx-c, abandoned Nov 2025), whose working invocation is a useful template:
  `jextract -t mlx --use-system-load-library -l mlxc --output <dir> --include-dir <mlx-c> mlx/c/mlx.h`.
  Note it targets the **umbrella header**, not a glob. Its author reported the generated code
  compiled in **Java** but not Scala, so the blockers he hit do not apply to us.
- **Deployment target silently changes behaviour.** MLX gates M5 "NAX" kernels on SDK and
  deployment target ≥ 26.2; when the gate fails the kernels are skipped **silently**. Not a v0.1
  concern, but it determines classifier strategy later.

## Architecture

```
jmlx-examples    HelloMLX                          demo, and an end-to-end test
       |
jmlx-core        MLX  MLXArray  MLXScope           hand-written idiomatic Java
                 DType  MLXException                se.alipsa.jmlx.{core,memory}
       |
jmlx-ffi         se.alipsa.jmlx.ffi.*              committed jextract output
                 NativeLoader                      hand-written
       |
native/install/lib/  libmlxc.dylib                 built by scripts/bootstrap-native.sh
                     libmlx.dylib  libjaccl.dylib   staged from the mlx-metal wheel
                     mlx.metallib                   staged from the mlx-metal wheel
```

**Runtime invariant, stated once and referenced everywhere:** all four artifacts live in **one flat
directory**. MLX finds `mlx.metallib` by colocation with its own image, and dyld resolves the
`@rpath` siblings the same way; splitting them into `lib/` and `share/` breaks both. Everywhere
this plan says "beside `libmlx.dylib`" or "beside `libmlxc.dylib`" it means this same invariant.

Note the wheel's own layout is *not* flat (`mlx/lib/`, `mlx/include/`, `mlx/share/cmake/`), and the
build needs it that way for `find_package(MLX)`. So the bootstrap keeps an unpacked wheel tree for
building and stages a separate flat directory for runtime.

Rationale for three modules: the jextract output for `mlx/c/mlx.h` is a large generated blob.
Isolating it in `jmlx-ffi` means it compiles once and is untouched by day-to-day iteration on
`jmlx-core`, keeping incremental builds fast and keeping generated code out of review diffs.

Shape is represented as plain `int[]`; no `Shape` type in v0.1.

## Out of scope for v0.1

`se.alipsa.jmlx.nn` (all layers), quantized modules, `MultiHeadAttention` / KV-cache, safetensors
parsing, tokenizers, autograd, the `jmlx-native-macos-arm64` classifier jar and its
extract-from-classpath loader, **distribution** signing and notarization, CI natives workflows,
dtypes beyond float32/int32, and any non-macOS target.

Note that ad-hoc re-signing after any post-link binary modification is **in** scope and mandatory
(§3) — it is a build-correctness step, not a distribution concern. Do not skip it citing this list.

Choosing the `macosx_26_0_arm64` wheel pins the minimum consumer macOS to 26.0. Acceptable for
v0.1, but it is the same axis as the NAX-gating note above and will drive classifier strategy when
packaging is taken up.

## Work breakdown

### 1. Restore a green baseline

- Delete `lib/src/test/java/org/example/LibraryTest.java` — its subject is already gone. This is
  what currently breaks `compileTestJava`.
- Confirm `./gradlew build` passes **before** making any other change, so later regressions are
  attributable.
- Remove `guava` and `commons-math3` from `gradle/libs.versions.toml` and the build file.
- **Fix a confirmed pre-existing bug:** `.gitignore:14`'s blanket `*.jar` rule shadows
  `gradle/wrapper/gradle-wrapper.jar` (`git check-ignore -v` confirms; only four files are tracked,
  so it has never been committed and `./gradlew` fails on a fresh clone). Add
  `!gradle/wrapper/gradle-wrapper.jar` and commit the jar. Commit the untracked `.gitattributes` in
  the same change — it already carries `*.jar binary`, which prevents the jar being mangled.
- Add `native/` to `.gitignore`.

### 2. De-risk probe — cheap answers before any build machinery

Done by hand in a scratch directory, on a green tree. Not self-contained, so do the prerequisites
first — they are needed here and reused by §3:

- Install `cmake` (Homebrew) and jextract for `macos-aarch64` from `https://jdk.java.net/jextract/`.
- Clone `ml-explore/mlx-c` and unpack the `mlx-metal` wheel into the scratch directory.
- **Measure the extracted `mlx.metallib`** and record the real size; the placeholder figures above
  are unconfirmed and §3.7 asserts against them.

Then five questions, each of which invalidates downstream work if answered badly:

1. **Does jextract digest the headers?** Run it over the umbrella header `mlx/c/mlx.h`. If it
   chokes on `half.h`'s `__fp16` / `__bf16` or on `<complex.h>`, the binding strategy changes (the
   module layout does not).
2. **What lookup does jextract emit, per flag?** Inspect the generated static initializer with and
   without `-l mlxc`. This settles Decision 8 — whether `NativeLoader` needs a post-generation shim,
   or bindings are generated without `-l` and given an injected lookup. Deciding after committing
   the generated blob is expensive.
3. **Is MLX's memory introspection in mlx-c's public headers?** `mlx_metal_set_metallib_path` is
   already known to be C++-only; the same partial-surface risk applies to the active/cached memory
   query. The leak test under "Testing approach" is the one test guarding this design's entire
   purpose — discovering at §6 that it cannot be written is expensive.
4. **What is mlx-c's error convention?** Status codes, or a default handler that aborts? Determines
   whether `check(...)` is viable at all (see §5). Also identify a **reachable native error path** —
   §7's Java-side shape validation will intercept the easy cases, so the error test in "Testing
   approach" needs a deliberate bypass to reach native at all. Design that now.
5. **Does mlx-c's source compile against the newer MLX headers?** This is an **API question, not an
   ABI one** — mlx-c is compiled from source against the wheel's C++ headers
   (`target_link_libraries(mlxc PRIVATE mlx)`, C++20), so nothing prebuilt is being linked. The
   question is simply whether 0.31.2-era mlx-c source compiles and links against MLX 0.32.0
   headers, and the answer is a twenty-minute experiment: try it. This reframing is what demotes
   the version mismatch from a bootstrap blocker to a probe question.

### 3. Native bootstrap

Assumes §2's prerequisites (cmake, jextract, mlx-c clone) are installed.

- **Resolve the jextract build string from the page at bootstrap time** rather than hardcoding it.
  (Confirmed present at time of writing: `25-jextract+2-4`, 2025/11/25.) Verifying the `.sha256`
  from the same origin as the download gives no independent integrity guarantee, so use
  **trust-on-first-use**: record the resolved hash in the repo and compare on every later run.
  Same treatment for the wheel.
- Write `scripts/bootstrap-native.sh`, idempotent, printing every resolved version:
  1. Download the **`mlx-metal`** wheel for `macosx_26_0_arm64` at a **pinned version, and record
     its sha256** — same trust boundary as jextract, same rigor. Not the `mlx` wheel, which ships no
     binaries. A wheel is a zip, so extract without pip or a virtualenv. Keep the unpacked tree
     (`mlx/lib`, `mlx/include`, `mlx/share/cmake`) intact for the build.
  2. Clone `ml-explore/mlx-c` at a **pinned commit SHA, not a tag** — tags are mutable upstream and
     this clone is the trust boundary for every binary in the build.
  3. **Assert the version pairing before building.** `find_package(MLX REQUIRED)` has no version
     constraint, so a mismatch is accepted silently. Compare the wheel's MLX version against the
     `GIT_TAG` in mlx-c's `CMakeLists.txt` and warn loudly on mismatch. As of writing these are
     **0.32.0 and v0.31.2** — the pairing must be chosen deliberately and recorded, informed by
     §2.5's compile experiment. Do not paper over it by deleting the assert.
  4. Configure mlx-c with, at minimum:
     ```
     -DMLX_C_USE_SYSTEM_MLX=ON           # else it FetchContents mlx v0.31.2 and builds the whole core
     -DBUILD_SHARED_LIBS=ON              # else the output is a static libmlxc.a, not a dylib
     -DMLX_C_BUILD_EXAMPLES=OFF          # defaults ON; wasted time and extra failure surface
     -DMLX_DIR=<wheel>/mlx/share/cmake/MLX
     -DCMAKE_PREFIX_PATH=<wheel>/mlx
     -DCMAKE_INSTALL_RPATH=@loader_path
     -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON # see below — without this the flag above does nothing useful
     ```
     `@loader_path` is what makes the flat runtime directory self-consistent. mlx-c sets no RPATH
     properties of its own, so injecting these is required, not redundant.
  5. **Stage via `cmake --install <build> --prefix <staging>`, not by copying out of the build
     tree.** `CMAKE_INSTALL_RPATH` is applied by CMake's *install* step; a binary sitting in the
     build tree still carries the build-tree rpath. Copy that one and `otool -L` will not resolve,
     putting you back to `install_name_tool`. `-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON` belts-and-braces
     this by making the build-tree binary carry the install rpath directly — set both.
     Then assemble the flat `native/install/lib/`: the installed `libmlxc.dylib` plus
     `libmlx.dylib`, `libjaccl.dylib` and `mlx.metallib` copied from `<wheel>/mlx/lib/`.
  6. Install mlx-c's `mlx/c/*.h` into `native/install/include/`.
  7. **Assert the result** rather than trusting it: `otool -L libmlxc.dylib` resolves every
     dependency against the flat directory, and `mlx.metallib` is present and of the size measured
     in §2 (a few KB means something went wrong).
  8. **Assert the fast path actually engaged:** `<build>/_deps/mlx-src` must **not** exist, and the
     configure log must report MLX found under the wheel prefix. If `MLX_C_USE_SYSTEM_MLX=ON` fails
     to take, everything still builds — you just get a silently different MLX.
- **If `install_name_tool` is used at any point, re-sign immediately** with
  `codesign --force --sign - <lib>`. Post-link modification invalidates the ad-hoc signature the
  linker applies automatically; the symptom is a `dlopen` failure or SIGKILL naming nothing
  relevant. Build time only, never runtime. The RPATH flags above should make this unnecessary —
  treat needing it as a signal that something else is wrong.
- Write `scripts/regen-bindings.sh`: run jextract **once, against the umbrella header**
  `native/install/include/mlx/c/mlx.h` (a `*.h` glob produces overlapping duplicate classes), output
  into `jmlx-ffi/src/main/generated/java`, package `se.alipsa.jmlx.ffi`. Pin the jextract version in
  a comment.
  - **Do not copy the vecxt template's `--use-system-load-library` flag.** It makes the generated
    code call `System.loadLibrary("mlxc")` from its own static initializer — reintroducing both the
    classloader-registry problem and the cause-loss problem that §5's `NativeLoader` exists to
    avoid, leaving that loader decorative. Apply the composition decided in §2.4 / Decision 8.
  - Expect Darwin system-header spillover. Filtering is a **two-pass process**, not a flag toggle:
    run jextract with `--dump-includes <file>`, edit the emitted argument file down, then re-run
    with `@<file>`.

### 4. Gradle restructure

Critical files: `settings.gradle`, `lib/build.gradle` (becomes `jmlx-core/build.gradle`),
`gradle/libs.versions.toml`.

- Replace `include('lib')` with the three modules; keep the Java 25 toolchain and JUnit Jupiter.
- Register `jmlx-ffi/src/main/generated/java` as a source directory.
- Apply the **`application` plugin** to `jmlx-examples` — required for `:jmlx-examples:run`.
- Native access flags on `test` and `run`, **and** `applicationDefaultJvmArgs` so generated start
  scripts carry them too. `--enable-native-access=ALL-UNNAMED` for now; the check applies to the
  *immediate caller's module*, so once `jmlx-ffi` is a named JPMS module this narrows to
  `--enable-native-access=se.alipsa.jmlx.ffi` and only the loader needs it. Funnelling every native
  call through one loader class now makes that transition free. Note the
  `Enable-Native-Access: ALL-UNNAMED` manifest attribute is **not** an alternative — it applies only
  to a jar launched via `java -jar`.
- **`org.gradle.configuration-cache=true` is enabled.** Do all native detection — filesystem probes,
  environment reads — at **execution time**, never at configuration time, or the config cache is
  poisoned.
- Inject the native directory into tests and `run` as an **absolute path** via a system property.
  `native/install/lib` is CWD-relative and resolves differently under `:jmlx-core:test`, an IDE
  run, and `:jmlx-examples:run`.

### 5. `jmlx-ffi` — raw bindings and loader

- Commit the jextract output.
- Hand-write `NativeLoader` exposing an **idempotent `ensureLoaded()`**, not a static initializer.
  A static initializer gives `ExceptionInInitializerError` on first touch and a bare
  `NoClassDefFoundError` with the cause discarded on every subsequent touch — precisely what a
  confused contributor hits on their second test run. Cache the outcome and rethrow the *original*
  cause.
- Resolve the **directory** in order: system property `jmlx.library.path`, environment variable,
  then the Gradle-injected absolute path. Load via
  `SymbolLookup.libraryLookup(dir.resolve("libmlxc.dylib"), Arena.global())`.
- **Validate before loading:** assert `mlx.metallib` sits beside the dylib, and fail fast naming
  `scripts/bootstrap-native.sh`. Without this the failure surfaces much later as an opaque error on
  the first GPU op and reads like a bug in the op.
- **Apply §2.4's finding on mlx-c's error convention.** The design below assumes every call returns
  a checkable status. If mlx-c's default handler aborts the process instead, `check(...)` never runs
  and the result is a JVM hard-kill with no Java stack; install a custom error handler at load time.
- **Gate:** a smoke downcall (device query) must pass before further work.

### 6. `jmlx-core` — memory and the array handle

`se.alipsa.jmlx.memory`:

- `MLXScope implements AutoCloseable`. Owns an ordered list of native handles; `close()` frees in
  reverse insertion order and is idempotent. A `Cleaner` action is registered as a backstop.

  `java.lang.foreign.Arena` cannot run custom cleanup actions, so the outline's
  `MLX.array(arena, ...)` signature cannot be implemented as written. `MLXScope` is our own type; it
  may hold an `Arena` internally for off-heap segments, but ownership of `mlx_array` handles is ours.

  **Capture rule — the failure mode of this entire pattern.** The `Cleaner` action must not capture
  `this`, nor anything with a reference path back to the `MLXScope`. If it does, the scope is never
  unreachable, the backstop never fires, and it fails *silently*. Put the handle list in a separate
  static-nested holder and register only that.

- **Thread-safety contract, stated explicitly:** `MLXScope` and `MLXArray` are **not** thread-safe
  and must be confined to one thread. The sole exception is the `Cleaner` thread invoking the
  backstop, which is why the backstop touches only the holder and never the scope.

`se.alipsa.jmlx.core`:

- `MLXArray` — native handle plus owning scope. `shape()`, `dtype()`, `ndim()`, `size()`,
  `toFloatArray()`, `close()`. Use after close throws `IllegalStateException` naming the array.
- **`toFloatArray()` must force a contiguous copy first.** MLX `transpose` is lazy and yields a
  strided view; reading the underlying data pointer row-major returns plausible-looking output in
  the wrong element order. This is a silent-wrong-answer bug, not a crash.
- `DType` — enum over mlx dtype constants. float32 and int32 only.
- `MLXException` — thrown by a central `check(...)` applied to every native call's status.

### 7. `jmlx-core` — operations and evaluation

- `MLX` static facade: `array(scope, float[], int[] shape)`, `add`, `subtract`, `multiply`,
  `divide`, `matmul`, `reshape`, `transpose`, `sum`, `exp`, `eval(MLXArray...)`, `defaultDevice()`,
  `defaultStream()`.
- Ops only build the graph. `eval()` is the sole explicit trigger; `toFloatArray()` evaluates
  implicitly — document this on the method.
- Validate shapes Java-side where cheap, so errors name the operands rather than surfacing as an
  opaque native status.

### 8. `jmlx-examples` and documentation

- `HelloMLX` — the outline's demo, reduced to the ops in the slice.
- Update `README.md` with bootstrap steps and the demo command.
- **Update `req/project-outline.md`** to `se.alipsa.jmlx.*` per Decision 7. Its architecture diagram
  and `JMLXDemo` code sample both use `jmlx.core` / `jmlx.nn` and are now wrong. While there, correct
  the sample's `MLX.array(arena, …)` signature, which cannot be implemented as written (see §6).

## Testing approach

TDD: each unit gets a failing test first.

The skip gate — `@EnabledIfNativeAvailable` — **must call `NativeLoader`'s own resolution and skip
on its failure**, not probe for a directory independently. Divergent logic produces false skips
(library present via system property, tests silently do not run — worse than failing) and false
failures (directory present, dylib wrong arch).

| Layer | What it proves |
| --- | --- |
| FFI smoke | A downcall reaches the dylib and returns a sane device. |
| Loader guard | With `mlx.metallib` absent, loading fails fast naming the bootstrap script. **Must be a forked JVM** — see below. |
| Native error path | A genuine native error surfaces as `MLXException`, **not** a process abort. Proves the §5 error-handler mitigation works; every other row here is a happy path. |
| Memory lifecycle | `close()` frees; double-close is a no-op; use-after-close throws; the `Cleaner` backstop fires for an escaped scope. |
| Numeric correctness | **Every** op in the facade — `add`, `subtract`, `multiply`, `divide`, `matmul`, `sum`, `exp` — against hand-computed values. `reshape` and `transpose` assert shape **and element values**, the latter being the only thing that catches the contiguity bug. |
| End-to-end | `HelloMLX` runs and produces expected output. |

**The loader-guard test needs its own JVM and its own directory.** `ensureLoaded()` is idempotent
and caches its outcome — correctly — so once any earlier test in the same JVM has loaded
successfully, removing `mlx.metallib` is unobservable and the result depends entirely on
class-loading order. Worse, `org.gradle.parallel=true` is set, so a test that moves a ~160 MB file
in the shared `native/install/lib/` races every other native test in the build. Give it a dedicated
Gradle task that forks a JVM with `jmlx.library.path` pointed at a **copy** of the directory with
the metallib removed. **Never mutate the real staging directory from a test.**

**The native-error test needs a reachable error path.** §7 validates shapes Java-side, which will
intercept the obvious provocations before they reach native. Design a deliberate bypass — an
internal entry point that skips validation — when writing §7, not when writing the test.

**Testing the `Cleaner` backstop.** "The action provably captures a holder, not the scope" is not
directly assertable — the capture rule is enforced *structurally*, by putting the handle list in a
static-nested holder, not by a test. What **is** observable is the consequence: hold a
`WeakReference` to the scope, drop the strong reference, and assert the referent is cleared while
its native handles remain live. A single `System.gc()` is only a hint, so drive it with a bounded
retry loop under `@Timeout` rather than a one-shot call, or the test will flake in CI.

**Leak test, specified concretely.** "Memory does not grow monotonically" is unfalsifiable — MLX
uses a caching allocator, so memory legitimately grows then plateaus. Instead: query MLX's own
**active** memory (not cached, not process RSS), run a warmup phase that is excluded from
measurement, then assert active memory after N iterations is within a fixed numeric threshold of
the post-warmup reading. This is the one test guarding the design's entire purpose, so it must be
able to fail. **Its feasibility is confirmed in §2.3** — if that query is not in mlx-c's public
headers, this test must be redesigned before §6 begins, not after.

## Verification

1. `./gradlew build` passes **before** any jmlx work begins (baseline restored).
2. `./scripts/bootstrap-native.sh` completes; `native/install/lib/` contains all four artifacts —
   `libmlxc.dylib`, `libmlx.dylib`, `libjaccl.dylib`, `mlx.metallib`. Check the metallib explicitly;
   it is the one most likely to be silently absent.
3. `otool -L native/install/lib/libmlxc.dylib` shows every dependency resolving **against the flat
   directory** — this is what `-DCMAKE_INSTALL_RPATH=@loader_path` buys, and it will not hold
   without it. `codesign -v` passes on each dylib.
4. `./scripts/regen-bindings.sh` completes; `jmlx-ffi/src/main/generated/java` is populated.
5. **Bindings-drift check.** Re-run `scripts/regen-bindings.sh` and assert
   `git diff --exit-code jmlx-ffi/src/main/generated/java` is clean. Committing generated output
   creates two independent sources of truth — the blob and the pinned headers — and this is the
   only thing standing between you and bindings that silently no longer match the library they
   call. Cheap enough to run in CI.
6. Clone → `bootstrap-native.sh` → `./gradlew build` passes. (`native/` is gitignored, so a clean
   clone never has the libraries; bootstrap is a required step, not an assumption.)
7. `./gradlew build` with `native/` absent — native tests report **skipped**, not failed.
8. `./gradlew :jmlx-examples:run` prints the expected shape and data.
9. **Run a real GPU kernel, not just a device query.** A device query succeeds even with the
   metallib missing; only a kernel dispatch proves the stack. Assert the default device is GPU
   *and* that a `matmul` returns correct values.
10. The forked loader-guard task passes: with `jmlx.library.path` pointed at a metallib-less copy,
    the loader fails fast naming the bootstrap script — proving the guard works rather than
    assuming it.
11. `req/project-outline.md` no longer references `jmlx.core` / `jmlx.nn`, and its code sample
    matches the API actually built (Decision 7, §8).
12. **`<build>/_deps/mlx-src` does not exist** and the configure log reports MLX found under the
    wheel prefix. This replaces an earlier "the bootstrap should feel fast" check — wall-clock is
    machine- and cache-dependent and the expected duration is still unmeasured, whereas the absence
    of a FetchContent source tree is a deterministic proof that `MLX_C_USE_SYSTEM_MLX=ON` took
    effect. If it silently did not, everything still builds against a different MLX.

## Open questions

The error-convention, memory-query and jextract-lookup questions are answered by the §2 de-risk
probe before implementation begins. The rest are flagged where they bite.

- **Is `mlx_array_free` safe to call from the `Cleaner` thread?** MLX may have thread affinity for
  stream or device operations. If not, the backstop must enqueue onto an owning thread instead.
  Determine before relying on the backstop.
- **Does mlx-c return status codes, or does its default error handler abort the process?** Decides
  whether `check(...)` is viable or a custom error handler must be installed at load time.
- **Is MLX's active/cached memory query exposed in mlx-c's public headers?** Decides whether the
  leak test is writable as specified.
- **Exact mlx-c contiguity semantics** — which call forces a contiguous copy, and whether
  `toFloatArray()` needs it unconditionally or only for non-contiguous views.
- **What lookup does jextract's generated static initializer emit per flag**, and therefore how
  `NativeLoader` composes with it (Decision 8).
- **Which mlx-c commit SHA pairs with which `mlx-metal` wheel version.** Known mismatched: the
  latest wheel is MLX 0.32.0, mlx-c pins v0.31.2. This is an **API question, not an ABI one** —
  mlx-c is built from source against the wheel's headers, so nothing prebuilt is linked and the
  answer is just "does it compile and link". Settled by the §2.5 experiment; options are to pin an
  older 0.31.2 wheel, use a newer mlx-c if one has bumped its tag, or confirm 0.32.0 compiles
  cleanly.
- **MLX build/bootstrap wall-clock time** — unmeasured; `cmake` is not yet installed.
