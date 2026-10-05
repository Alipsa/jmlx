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
every selected device. Each profile declares `macOSMajor` and `macOSMajorPolicy`. Phase 6 GPU
fixtures require exactly macOS 26 for generation and verification; newer OS versions change their
rounded results. Phase 7.1 CPU fixtures allow macOS 26+ with a `minimum` policy. Unfiltered commands
automatically select compatible profiles and print which families were skipped. Newer hosts run
Phase 7.1 CPU fixtures without flags and leave Phase 6 GPU goldens unchanged. `-PmlxOracleFamily`
optionally selects a specific family; explicitly requesting Phase 6 on an incompatible host fails.
Direct single-fixture runner invocations also enforce the profile policy.

CI sets `-PmlxOracleRequireAllProfiles=true`: every recorded profile must be compatible, so an OS
image change fails instead of skipping Phase 6. This option cannot be combined with a family
selection. Direct Python tools expose the equivalent `--require-all-profiles` option.

Nonfinite float serialization uses `NaN`, `Infinity` and `-Infinity` strings with `allow_nan=False`.
The shared core test-fixture reader decodes them; comparisons classify NaNs/infinity signs exactly.
QuickGELU's formula case follows HF's definition `x*sigmoid(1.702*x)` (the `quick_gelu` configuration
contract); it is not substituted with MLX's fast GELU approximation. Reference sources for every
other case are the installed pinned MLX core and nn implementations. Explicit input weights,
biases, statistics and options are committed beside expected values.

Only `./gradlew generateMlxOracleFixtures` rewrites canonical references. Before host policies were
introduced, Phase 6 GPU outputs drifted at seven decimals on macOS 27. Legacy goldens were preserved.
Current unfiltered verification passes for CPU fixtures and reports Phase 6 as skipped; require-all
verification rejects this host before execution. The optional family selector leaves unrelated
expected files untouched:

```sh
./gradlew generateMlxOracleFixtures -PmlxOracleFamily=phase7-1
./gradlew verifyMlxOracleFixtures -PmlxOracleFamily=phase7-1
```

The CPU family still requires byte-exact verification on CI before milestone acceptance. See
`req/plans/phase7-1-probe-findings.md` for contracts, pinned limitations and command results.
