# MLX Python oracle

This directory is the explicit, macOS/arm64-only differential oracle for Phase 6 fixtures. It is
never invoked by ordinary Java tests. `requirements.lock` pins the CPython 3.12 frontend and the
same `mlx-metal` backend wheel used by `scripts/bootstrap-native.sh`, including both wheel hashes.

On Apple Silicon:

```text
./tools/mlx-oracle/install.sh
./gradlew verifyMlxOracle verifyMlxOracleFixtures
```

The committed fixtures explicitly select the GPU and record their device, values, shapes, and dtypes
in canonical JSON. Float values are serialized to seven decimal places after MLX evaluation to keep
the checked-in representation concise. Verification still compares the canonical JSON exactly; the
rounding is not a cross-device or cross-version numerical tolerance.
Environment verification also compares the bootstrap-derived pins with the staged native runtime's
`native-pin.properties` whenever that completion marker is present.
Each `*.input.json` maps to the same basename with `*.expected.json`; verification rejects missing
or orphan partners. `generateMlxOracleFixtures` is the only task allowed to rewrite expected JSON. Review its diff and
record provenance changes whenever the native pins change. An oracle setup/version failure is an
infrastructure failure; it is not evidence that Java output is incorrect.

After `scripts/updateMlx.zsh` changes the mlx-c pin, update `provenance.json`'s `mlxCCommit` before
running `generateMlxOracleFixtures`, because generation first verifies provenance. If the paired
`mlx` or `mlx-metal` distribution changes too, update its version, URL, and hash in both
`provenance.json` and `requirements.lock`, then rerun `install.sh` before fixture generation.
