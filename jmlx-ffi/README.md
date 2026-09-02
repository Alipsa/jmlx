# jmlx-ffi

`jmlx-ffi` is the low-level Java Foreign Function & Memory (FFM) binding layer
for MLX C. It contains committed `jextract` output plus `NativeLoader`, which
loads the staged MLX runtime and installs an error handler that keeps native
failures from terminating the JVM.

Most applications should use `jmlx-core` rather than call these bindings
directly. `jmlx-core` wraps the generated FFM surface in scoped Java values and
is the supported high-level API. Use `jmlx-ffi` directly only when implementing
new core operations or working with an MLX C API that `jmlx-core` does not yet
expose.

## Requirements

This module targets the Java 25 FFM API and requires the MLX runtime on macOS
Apple Silicon. From a repository checkout, stage the native libraries first:

```sh
./scripts/bootstrap-native.sh
```

The script produces `native/install/lib`, containing `libmlxc.dylib`, `libmlx.dylib`,
`libjaccl.dylib`, and `mlx.metallib`. It requires the macOS/Xcode tooling and other prerequisites
listed in the script header.

## Use

Set the native-library directory before the first native call, then load it
explicitly. `ensureLoaded()` is idempotent and safe to call repeatedly.

```java
import se.alipsa.jmlx.ffi.NativeLoader;

System.setProperty("jmlx.library.path", "/path/to/native/install/lib");
NativeLoader.ensureLoaded();
```

You may use `JMLX_LIBRARY_PATH` instead of the system property. Generated
binding classes live in `se.alipsa.jmlx.ffi`; do not edit them by hand. Regenerate
them only with `scripts/regen-bindings.sh`.

## Build and verify

```sh
./gradlew :jmlx-ffi:check
```

The normal check includes a separate loader-guard test that verifies a missing
`mlx.metallib` fails clearly. The generated bindings are intentionally excluded
from formatting and Checkstyle; their reproducibility is guarded by the
regeneration workflow.
