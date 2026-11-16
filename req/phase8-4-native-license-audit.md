# Phase 8.4 native transitive-license audit

Review date: 2026-11-16. Source baseline: PR #99 merge
`a97eb5b0c735cdc047a13c5539295cce64130eef`.

**The license release gate remains open.** This review found missing permissive notices and
adds them to the native main, sources and Javadoc JARs. Two provenance findings below require
resolution before release. Passing source/hash/packaging checks establishes consistency of
the reviewed material, not permission for the unresolved material.

The [machine-readable audit](phase8-4-native-license-audit.json) records the reviewed source
files, hashes of the current five-file native payload and open findings. The
[inventory](native-license-inventory.json) records 26 entries, immutable source URLs where
available, source SHA-256s, notice extraction ranges and packaged-text hashes.

## Scope and method

The payload is the SHA-256-pinned `mlx-metal` 0.32.4 macOS ARM64 wheel, plus locally compiled
mlx-c at `a26f2d584a14ec619ae8db9a09be03becd47598a`. Upstream MLX tag `v0.32.4`
resolves to `36969c094354086e15d2a327a9c50feb20b9cf11`. The local source copies are clean at
those revisions. Bootstrap uses the wheel's system MLX configuration; it does not compile a
second MLX. Native pins, generated bindings, binary deployment targets and native payload bytes
are unchanged by this audit.

The review followed root, CPU, Metal, I/O and distributed CMake target definitions; inspected
copyright/license and adaptation/reference comments throughout `mlx/` and mlx-c; inspected
fetched-dependency licenses; and compared that inventory with the packaged legal texts. It also
reviewed the wheel's installed metal-cpp license, build configuration and dylib linkage.
The linked source files and digests are retained in `sourceEvidence` in the JSON.

The wheel is a vendor binary. Release/tag pairing, installed headers and exported build
configuration identify the source to review; this is not a reproducible rebuild or an attested
complete build log for that wheel. When code is a compiled adaptation or an algorithm reference,
not a separately linked library, it is still considered. Where MLX cites an unversioned project,
the inventory's pinned upstream version is a license-evidence snapshot, not a claim that MLX
incorporated that exact release. Algorithm references receive conservative attribution; this
does not establish copied implementation code or settle the historical provenance.

## Component disposition

All confirmed texts below are carried in `META-INF/THIRD-PARTY-NOTICES`, except metal-cpp's
complete Apache text in `META-INF/LICENSE-APACHE-2.0`. The latter also supplies the complete
Apache terms for the attributed adaptations. The module's own MIT license and top-level
NOTICE are packaged alongside them in all three JARs.

| Component / adaptation | Evidence and disposition |
| --- | --- |
| MLX / JACCL | MLX MIT text; JACCL README explicitly uses MLX's license |
| mlx-c | MIT text from the pinned commit |
| metal-cpp 26 | Full Apache-2.0 text, byte-identical to the wheel's installed license |
| nlohmann/json 3.11.3 | MIT permission/disclaimer and all five distinct single-header copyright attributions; used by MLX and JACCL |
| fmt 12.1.0 | Complete MIT text and optional compiled-code exception; retain the text despite the exception |
| gguf-tools / FP16 | MIT texts for the pinned static GGUF implementation and FP16 provenance; FP16 also appears in MLX's type conversion header |
| pocketfft | Full embedded BSD-3-Clause notice, including the FFTW-derived DCT authors; source expressly identifies that code's BSD permission |
| V8 small_vector | Embedded BSD-3-Clause notice |
| ThreadPool | Embedded zlib notice, also checked against JACCL's separate modified copy |
| Norbert Juffa `expm1f` | Embedded BSD-2-Clause notice; confined to the material it identifies |
| NVIDIA / Filipe RNC Maia `cexpf` | Embedded Apache-2.0 attribution and complete Apache terms |
| NumPy / opt_einsum | **Added:** BSD-3-Clause and MIT texts for the einsum adaptation |
| supervised-lda NumpyFormat | **Added:** MIT text for the explicitly adapted NumPy loader |
| fastermath / Julien Pommier NEON math | **Added:** source-local BSD-3-Clause and zlib notices; Cephes origin remains a separate finding |
| PyTorch | **Added:** complete BSD-3-Clause license for CPU erf, Metal hypot and FP8 conversions; both copyright-year variants from the three cited commits are retained |
| TensorFlow | **Added:** Apache-2.0 copyright/header for the cumulative log-sum-exp gradient adaptation, plus complete Apache terms |
| JAX / Random123 | **Added:** JAX Apache attribution and Random123 BSD-3-Clause notice for Threefry references |
| VkFFT / CUB | **Added:** MIT and source-local BSD-3-Clause notices for Metal FFT and merge-sort references |
| Cephes | **Added:** original permission statement; remains `NATIVE-LICENSE-2` below |

The POM's MIT and Apache-2.0 entries describe the module and primary bundled work. They are
not an exhaustive transitive-license inventory; the complete notices travel in the artifacts.
Adaptations are identified as modifications rather than attributed as original jmlx code.

## Distribution boundary and exclusions

The [retained `otool -L` output](plans/evidence/native-license-audit-2026-11-16/dylib-dependencies.txt)
shows the sibling MLX/JACCL dylibs and Apple system frameworks/libraries: Metal, Foundation,
QuartzCore, Accelerate, libc++, libSystem and libobjc. Those system components are not copied
into the JAR. metal-cpp is header-only code compiled into the payload, so its notice is included.
Static/header-only dependencies above are included even though `otool` cannot list them.

The Python frontend, nanobind, Python standard library, upstream tests/benchmarks/examples and
their tool dependencies are not packaged. The backend wheel provides headers, dylibs and Metal
kernels; jmlx copies only the four native runtime files and its own pin file. The CUDA backend,
CUTLASS/cuDNN/cuFFT/NCCL implementations, Windows dlfcn/OpenBLAS paths and Linux BLAS/LAPACK
fallback are excluded by this macOS CPU/Metal build. CUB's algorithm attribution above concerns
Metal sort and is retained independently of the excluded CUDA backend.

MLX's MPI backend calls `dlopen("libmpi.dylib")`; JACCL similarly opens the OS RDMA library.
No OpenMPI or RDMA implementation library is bundled. A citation in the non-Accelerate LAPACK
workaround, JAX type-promotion documentation, mathematical derivations and the Hadamard matrix
reference are recorded as references, rather than evidence that those projects' libraries are
distributed. Optional functionality loading a user's additional libraries does not add those
libraries to the reviewed bundle.

## Open findings

### NATIVE-LICENSE-1: Juffa `erf` / `erfinv` provenance

The pinned Metal `erf.h` explicitly says its error function is based on
[njuffa's answer 35148199](https://stackoverflow.com/a/35148199). The inverse-error-function
coefficients and evaluation structure in that file and CPU `simd/math.h` also match
[njuffa's answer 49743348](https://stackoverflow.com/a/49743348); this second mapping is an
inference from the code comparison. The retrieved primary answer records both declare
`CC BY-SA 4.0`; their body hashes and edit timestamps are recorded in the audit JSON.

The existing BSD-2-Clause grant in `expm1f.h` is not evidence of an alternative grant for these
other functions. This review found no such grant. Adding a BSD notice for `expm1f` therefore
does not close this finding. Before release, retain an applicable alternative grant with an
exact code mapping, or review and fulfill the applicable attribution/share-alike obligations.
A separately reviewed upstream/pin or implementation change is another possible resolution.
This audit does not change the native code or claim those obligations have been satisfied.

### NATIVE-LICENSE-2: Cephes provenance chain

CPU SIMD `math.h` explicitly traces its exponential and sine/cosine implementations through
fastermath and NEON math to Cephes. Both immediate upstream notices are now retained.
The [original Cephes statement](https://www.netlib.org/cephes/readme) permits free use but has
no standard license identifier. Its exact text is now also packaged. This review did not
establish an additional grant for this exact provenance chain.

Document why that original permission covers this redistribution, or retain an applicable
grant for the derived material, before closing the gate. Another project's separate BSD
permission is not used as proof here. This is an unresolved review question, not a finding
that the code is prohibited or GPL-licensed.

## Verification and reproduction

The original 2026-11-16 verification passed 63 release selftests and the native Gradle checks.
Its digest-only cache checked the declared contents of 26 notice entries and 34 additional
source records, but fetched only 56 of the 57 distinct URL/digest pairs: two PyTorch sources
shared a cache entry. The review correction below supersedes that network-verification evidence.
The unprefixed retained logs remain historical records of that original implementation.

Use Python 3.11+ and the existing Java toolchains. Bootstrap first if the native files and
verified wheel are absent. From the repository root:

```sh
./gradlew :jmlx-native-macos-arm64:check \
  :jmlx-native-macos-arm64:verifyNativeReleasePayload \
  :jmlx-native-macos-arm64:generatePomFileForMavenPublication
bash tools/release-all/selftest.sh
python3 tools/release-all/audit_native_licenses.py \
  --wheel native/scratch/mlx_metal.whl \
  --cache build/releaseAll/native-license-audit-cache \
  --jar jmlx-native-macos-arm64/build/libs/jmlx-native-macos-arm64-0.1.0-SNAPSHOT.jar \
  --jar jmlx-native-macos-arm64/build/libs/jmlx-native-macos-arm64-0.1.0-SNAPSHOT-sources.jar \
  --jar jmlx-native-macos-arm64/build/libs/jmlx-native-macos-arm64-0.1.0-SNAPSHOT-javadoc.jar \
  --output build/releaseAll/native-license-verification.json
```

The script verifies the wheel digest, fetches/hash-checks the recorded primary sources on a
cache miss, checks every extracted notice against its source and the component's own section
of the packaged text, and checks all supplied JARs. Notice headings match inventory component
names exactly. The inventory independently requires all four legal files in every JAR.
Cache keys bind both the original URL and the declared digest, so two different URLs with
identical contents are fetched separately. Cached objects are rehashed before use, never
rewritten on a hit, and new objects are installed atomically after validation.
Notice files are decoded explicitly as UTF-8. Matching ignores line-end whitespace only.
Source/header excerpts keep their full permission conditions and
disclaimers; the JSON attribution record lists each distinct original copyright line.
With an empty cache the check also reproduces the recorded source hashes from the network.
Mutable Cephes/Stack Overflow sources fail the digest check if their reviewed content changes.
The Cephes readme has no commit-pinned URL: an ordinary upstream edit can cause this failure;
it is not by itself evidence of tampering. Review the new content before changing its recorded hash.

The default command verifies consistency and writes `releaseReady: false` with both finding
IDs. Add `--require-clear` to make those findings produce a nonzero exit, after writing the
report. Neither an empty cache nor successful JAR checks overrides that result. After a version
cut, use the actual candidate artifact filenames and rerun against the exact signed bundle;
these SNAPSHOT artifacts are development evidence only.

Small reports and logs are retained outside `build/` under
[`req/plans/evidence/native-license-audit-2026-11-16/`](plans/evidence/native-license-audit-2026-11-16/).
The wheel, source cache and large native artifacts remain ignored/reproducible inputs.
The JSON records their digests; `./gradlew clean` does not remove the committed reports.

## Review corrections (2026-11-16)

Review adds URL-bound atomic caching, explicit UTF-8 notice decoding, component section
matching, strict excerpt bounds, contextual Stack Exchange response errors and an independent
four-file legal inventory requirement. Companion source/Javadoc entries are now scanned for
the forbidden home/checkout prefixes by `release_inputs.py native --documentation-jar`, as
invoked by the explicit Gradle release check. Those path checks do not depend on file extensions.

New regressions exercise a non-UTF-8 subprocess, independent requests for equal-digest URLs,
mocked network/URL rewriting, atomic cache replacement/failure and read-only hits, JSON body
extraction/error responses, excerpt validation and section binding, inventory omission,
main-JAR missing notices and the companion-JAR CLI/path checks. Review evidence uses the
`review-` filename prefix and records its locale and fresh cache separately from the original run.
The two legal provenance findings remain unresolved.

The corrected run passed all 72 release selftests (64.873 seconds), including 15 focused
license tests, and the native Gradle checks (15 tasks: 6 executed, 9 up-to-date). Under
`LC_ALL=en_US.ISO8859-1`, `PYTHONUTF8=0` and `PYTHONCOERCECLOCALE=0`, Python measured its
preferred encoding as `ISO8859-1`; the real audit still verified all 26 entries and 34 source
records. Its initially empty cache gained 57 entries for 57 distinct URL/digest pairs, and
all three JARs passed. `--require-clear` again exited 1 for the two unresolved legal findings.
