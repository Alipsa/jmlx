# MLX Python oracle

This directory is the explicit, macOS/arm64-only differential oracle for Phase 6 fixtures. It is
never invoked by ordinary Java tests. `requirements.lock` pins the CPython 3.12 frontend and the
same `mlx-metal` backend wheel used by `scripts/bootstrap-native.sh`, including both wheel hashes.

On Apple Silicon:

```text
./tools/mlx-oracle/install.sh
./gradlew verifyMlxOracle verifyMlxOracleFixtures
```

The committed fixture explicitly selects the GPU and records its device, values, shapes, and dtypes
in canonical JSON. All float values are rounded to seven decimal places only after MLX evaluation so
diffs remain stable and reviewable.
Environment verification also compares the bootstrap-derived pins with the staged native runtime's
`native-pin.properties` whenever that completion marker is present.
`generateMlxOracleFixtures` is the only task allowed to rewrite expected JSON. Review its diff and
record provenance changes whenever the native pins change. An oracle setup/version failure is an
infrastructure failure; it is not evidence that Java output is incorrect.
