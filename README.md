# jmlx
A pure, idiomatic Java 25 framework for Apple Silicon GPU tensor operations and LLM inference—wrapping Apple's native MLX core with zero-copy FFM bindings.

This is the v0.1 vertical slice described in `req/initial-plan.md`: enough of the stack — native bootstrap, generated bindings, memory management, and a handful of tensor ops — to prove the whole pipeline works end to end on real Apple Silicon GPU hardware.

## Requirements

- macOS on Apple Silicon, macOS 26 or later.
- Java 25 (JDK 25 toolchain; Gradle's toolchain support will provision it if you don't already have one).
- `cmake`, `git`, `curl`, `unzip`, `otool`, `codesign`, `shasum`, `cc` on `PATH` (all standard on a normal macOS + Homebrew + Xcode Command Line Tools dev setup; `cc` is already implied by `cmake` building mlx-c's C++ source).

## One-time native bootstrap

The native runtime (mlx-c compiled against a pinned `mlx-metal` wheel) is not checked in. Build it once with:

```sh
./scripts/bootstrap-native.sh
```

This downloads and verifies (trust-on-first-use, pinned SHA-256) a pinned `mlx-metal` wheel and jextract build, clones `mlx-c` at a pinned commit, builds it against the wheel's MLX, and stages the result as a flat directory at `native/install/lib/` (`libmlxc.dylib`, `libmlx.dylib`, `libjaccl.dylib`, `mlx.metallib`). The script is idempotent — re-running it is safe and fast once everything is already staged.

The Java bindings in `jmlx-ffi/src/main/generated` are committed, not generated on the fly, but they were produced from the same headers by:

```sh
./scripts/regen-bindings.sh
```

Re-run this only if you change the pinned mlx-c commit; `git diff --exit-code jmlx-ffi/src/main/generated/java` should then be clean.

## Build, test, run

```sh
./gradlew build          # compiles jmlx-ffi, jmlx-core, jmlx-examples
./gradlew :jmlx-core:test  # memory lifecycle, numeric correctness, native error path -- against real hardware
./gradlew :jmlx-examples:run  # runs HelloMLX
```

`HelloMLX` builds two small matrices, adds and matrix-multiplies them, evaluates on the GPU, and prints the results:

```
a + b      = [2, 2] [6.0, 8.0, 10.0, 12.0]
a matmul b = [2, 2] [19.0, 22.0, 43.0, 50.0]
```

Every module's tests are skipped automatically (not failed) if `native/install/lib/mlx.metallib` isn't present — see `@EnabledIfNativeAvailable` in `jmlx-ffi`.

## Code style

Hand-written sources are Google Java Style with 2-space indentation and a 120-column width (`config/spotless/eclipse-java-google-style-120col.xml`, `config/checkstyle/checkstyle.xml` — both derived from Google's own upstream artifacts; see the comments in each for the exact, documented deviations). The generated jextract bindings under `jmlx-ffi/src/main/generated/java` are exempt from both, since they must stay byte-identical to `scripts/regen-bindings.sh`'s output.

```sh
./gradlew spotlessCheck   # verify formatting
./gradlew spotlessApply   # reformat in place
./gradlew checkstyleMain checkstyleTest checkstyleTestFixtures  # style/lint (part of `build`/`check`)
```

## Running a distributed build

`./gradlew :jmlx-examples:installDist` / `distZip` produce a standalone `jmlx-examples` launcher, but its start script does *not* embed this build machine's `native/install/lib` path — that path is only wired up for the `run` task's own convenience. To run the distributed launcher elsewhere, set `JMLX_LIBRARY_PATH` to wherever `bootstrap-native.sh` staged the native runtime on that machine:

```sh
JMLX_LIBRARY_PATH=/path/to/native/install/lib ./bin/jmlx-examples
```
