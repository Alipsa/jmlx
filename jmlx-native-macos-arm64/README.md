# jmlx-native-macos-arm64

Native MLX and mlx-c runtime binaries for jmlx on macOS running on Apple Silicon. This artifact is
the optional runtime companion for `jmlx-ffi` and `jmlx-core`; it is not useful by itself and does
not provide a general MLX installation. Native inference is supported on Apple Silicon
macOS 27.2 or later with Java 25. Binaries retain a 26.2 deployment target and may load on
older systems; versions below 27.2 are unsupported. Hosted 26.x CI provides extra regression
coverage. Release acceptance uses physical 27.2.x hardware.

It packages `libmlxc.dylib`, `libmlx.dylib`, `libjaccl.dylib`, `mlx.metallib`, and pin metadata
produced by `scripts/bootstrap-native.sh`. `jmlx-ffi` automatically extracts those resources to a
per-pin directory under `~/Library/Application Support/se.alipsa.jmlx/native` when neither
`jmlx.library.path` nor `JMLX_LIBRARY_PATH` is configured.

## Use

Add the native artifact as a runtime dependency alongside the jmlx API module you use:

```groovy
dependencies {
  implementation("se.alipsa:jmlx-core:<version>")
  runtimeOnly("se.alipsa:jmlx-native-macos-arm64:<version>")
}
```

At the first native jmlx operation, the loader extracts the packaged runtime automatically. To use a
locally built runtime instead, set either `-Djmlx.library.path=/path/to/native/install/lib` or
`JMLX_LIBRARY_PATH=/path/to/native/install/lib`; an explicit path always takes precedence over the
packaged artifact.

## Cache maintenance

The loader retains each per-pin cache directory. Automatically deleting an older one could break a
running JVM that has loaded its dylibs but has not yet lazily opened `mlx.metallib`. To reclaim
space after upgrading, stop all jmlx JVMs and delete unneeded directories under
`~/Library/Application Support/se.alipsa.jmlx/native` (or the directory selected by
`-Djmlx.native.cache.path`).

The extraction lock waits up to five minutes by default. On slow or shared storage, configure a
longer wait with `-Djmlx.native.lock.timeout.seconds=<positive-seconds>`; values are capped at 365
days.

The artifact supports macOS/aarch64 only. It has the automatic module name
`se.alipsa.jmlx.nativelib.macosarm64`. `NativeArtifact.pin()` exposes the packaged `mlx-metal`
version and mlx-c commit for diagnostics without loading the native libraries.

## Building from this repository

Run `./scripts/bootstrap-native.sh` before building a distributable native jar. A normal unstaged
checkout still builds, but the local native jar intentionally has no bundled runtime and cannot be
published. The publish task refuses to run until the validated staged files and completion marker
are present.

The main, sources and Javadoc JARs carry `LICENSE`, `NOTICE`, `LICENSE-APACHE-2.0`,
`LICENSE-CC-BY-SA-4.0` and `THIRD-PARTY-NOTICES` under `META-INF`. Run
`./gradlew :jmlx-native-macos-arm64:verifyNativeReleasePayload` to verify their reviewed digests.
Before tagging a release, also run
`./gradlew :jmlx-native-macos-arm64:verifyNativeReleaseLicenses` after `git fetch origin`, with
`origin/main` and the linked notice commit available locally. This explicit check verifies scoped POM terms,
approval evidence and notice ancestry/freshness; see the
[release runbook](../tools/release-all/README.md) for the required history.
The [Phase 8.4 native license audit](../req/phase8-4-native-license-audit.md) records compiled-in
dependencies and unresolved provenance findings that must be resolved before release.

## License scope

The native distribution retains MLX's MIT grant for Apple/MLX contributions and separately identifies Norbert Juffa's covered expression in the reviewed Metal erf and Metal/CPU SIMD erfinv material under CC BY-SA 4.0. META-INF/THIRD-PARTY-NOTICES records the reviewed scope, revisions, modifications and redistribution basis; META-INF/LICENSE-CC-BY-SA-4.0 contains the complete CC license. No CC adapter's license is identified. Other bundled material retains its applicable licenses and notices. The owner-approved interpretation and its recorded counterargument are documented in the repository's owner license decision record.

See the [owner decision](../req/phase8-4-owner-license-decision.md) and
[approved notice](https://github.com/Alipsa/jmlx/blob/551eebeeb66a80ef3af467a6c1b4995c2a7f7034/jmlx-native-macos-arm64/THIRD-PARTY-NOTICES) for the reviewed basis and scope.
