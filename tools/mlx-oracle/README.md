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

Phase 7.1 uses the `phase7-1-core` dispatcher in `phase71.py` and the `phase7-1` provenance profile:
CPU and `MLX_ENABLE_TF32=0`. Legacy Phase 6 retains its GPU profile. All Gradle oracle tasks pin
precision as an environment variable and task input; the verifier evaluates a runtime operation on
every declared device. `platform.macOSMajorPolicy` is explicitly `minimum`: `macOSMajor` records
the minimum supported OS (26+), rather than the generating host version.

Nonfinite float serialization uses `NaN`, `Infinity` and `-Infinity` strings with `allow_nan=False`.
The shared core test-fixture reader decodes them; comparisons classify NaNs/infinity signs exactly.
QuickGELU's formula case follows HF's definition `x*sigmoid(1.702*x)` (the `quick_gelu` configuration
contract); it is not substituted with MLX's fast GELU approximation. Reference sources for every
other case are the installed pinned MLX core and nn implementations. Explicit input weights,
biases, statistics and options are committed beside expected values.

Only `./gradlew generateMlxOracleFixtures` rewrites canonical references. Review legacy outputs:
seven-decimal Phase 6 GPU regeneration currently drifts on the development macOS 27 host, so those
files remain unchanged and global verification fails rather than silently accepting drift. The
family selector leaves unrelated expected files untouched:

```sh
./gradlew generateMlxOracleFixtures -PmlxOracleFamily=phase7-1
./gradlew verifyMlxOracleFixtures -PmlxOracleFamily=phase7-1
```

The CPU family still requires byte-exact verification on CI before milestone acceptance. See
`req/plans/phase7-1-probe-findings.md` for contracts, pinned limitations and command results.
