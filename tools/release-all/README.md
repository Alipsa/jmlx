# Combined native release workflow

Phase 8.4 uses one combined deployment for the new coordinates selected by
[`released-8.4.json`](../release-smoke/released-8.4.json). Entries marked `central` are reused
and must never enter staging. The provisional tuple currently selects all nine modules;
see [the readiness audit](../../req/phase8-4-release-checklist.md).

Publication is a manual, explicitly authorized action after review of the exact signed ZIP
and final receipt. CI verifies; it never publishes. Do not run any per-module `release.sh`
for this tuple. The root script rejects an ordinary upload without a verified receipt.

## Prerequisites

Bash 4+, Groovy, Git, Python 3.11+, and explicit Java 25/21 installations selected by
`JMLX_RELEASE_JAVA_HOME` and `JMLX_RELEASE_JAVA21_HOME` (the latter may use a Java 21
`JAVA_HOME`). Non-resume release invocations launch Gradle with the selected Java 25 and
restrict toolchain paths to those two installations, disabling detection, downloads and
inherited `fromEnv` discovery. Bundle preparation requires macOS ARM64 and a complete
`native/install/lib` matching bootstrap pins, with validated dylib minimum OS/path hygiene.
The supported native floor is macOS 26.2; physical minimum-host kernel execution remains a
release gate. Ordinary contributor builds retain their unstaged-runtime skip behavior.

Signing and Portal credentials use the existing Gradle properties/environment contract.
Reports must contain no credentials. Receipts/logs live under ignored `build/releaseAll/`
or outside the checkout, since the source guard rejects untracked files too.

**External dependency:** the pinned nexus-release-plugin 2.5.1 always rebuilds before upload.
The tested upstream change is committed locally as `47f7a36` in `Alipsa/nexus-release-plugin`
(based on `d942747026bbe8bb22fdd38c6f768264d96284e0`). It adds `releaseAll.useExistingBundle`, preserving
source/version, saved ZIP/manifest receipt, availability and resume guards. A separate plugin
review/release and catalog bump are required. This repository refuses verified upload while
the pinned plugin lacks that property; no local substitution is allowed for a real release.

## Prepare, verify, then submit

1. Complete WP0 readiness and tuple review; rerun the pure-Java metadata audit after the plugin
   bump. Record the new plugin's historical 8.1 regressions on an unpushed throwaway commit.
2. Merge the reviewed release-version PR. Anchor an annotated immutable `jmlx-phase8.4-rc.N`
   tag at that SHA and work from a clean detached checkout. Never move the tag.
3. Run root checks, shared-build checks and `bash tools/release-all/selftest.sh`. Normal candidate
   smoke publishes only manifest-local entries and resolves reused entries solely from Central.
4. Run `./releaseAll.sh --dry-run`, inspect the exact selected set, then set
   `JMLX_RELEASE_CANDIDATE_TAG` and run `./releaseAll.sh --bundle-only`. Signing preflight is
   strict for bundling and warning-only for dry-run. No upload happens in either mode.
5. The wrapper reruns **only required test tasks** using task-level `--rerun`, disables the build
   cache, clears their result directories and collects supported Gradle task events via the
   explicit producer init script. The shared native suite inventory validates fresh JUnit XML,
   including both precision modes and loader/worker probes. The plan digest must stay unchanged.
6. The resulting immutable `preliminary-<invocation>.json` identifies ZIP/tag/SHA/manifest,
   native hashes, plugin receipt, exact Gradle argument vector and file digest, selected
   native/pure JDK vendor/full version/home, and this invocation's test evidence. Apple tool
   preflight runs before bundling and records either Xcode or selected CLT package metadata. Its
   printed SHA-256 is an input to every remote/local exact-bundle verification.
7. Use `bundle_smoke.py` for `macos-full`, `macos-pure` and local `linux-pure` reports. It calls
   the independent consumer in read-only bundle mode, records the environment and binds its
   log digest to the candidate. The Linux run uses a full scratch clone of the candidate tag,
   the `/projects` bind mount for ZIP/record transport, a fresh Linux Gradle home and an explicit
   `JMLX_LINUX_CONTAINER_IMAGE_ID`. It requires network access for the pinned wrapper/toolchains.
8. Upload **both ZIP and preliminary JSON** as assets of a temporary draft GitHub Release on
   the candidate tag, after authorization for that external write. Dispatch
   `release-candidate.yml` with both asset IDs/digests, Release ID and tag/peeled SHA. The transfer
   job has draft-access permissions and never executes the bundle. The dedicated physical
   minimum-host job has read-only repository access and checks tag object, HEAD, digests,
   native OS floor and actual packaged kernels. Provision its `jmlx-release-minimum` runner
   label before attempting this gate; a VM or moving hosted image cannot establish the floor.
9. Download minimum-host JSON/log evidence, then create the final receipt:

```sh
python3 tools/release-all/final_receipt.py --root "$PWD" \
  --bundle build/releaseAll/jmlx-mega-bundle.zip --preliminary <preliminary.json> \
  --report macos-full=<report.json> --report macos-pure=<report.json> \
  --report linux-pure=<report.json> --report minimum-host=<report.json> \
  --output build/releaseAll/verified.json
```

Each report's log and measured `.runtime.xml` must remain beside its JSON. The consumer JVM
itself writes feature/version/runtime build/vendor/home; its file and digest are verified,
independently of the Java launcher on PATH. Only a **final** receipt can be submitted:

```sh
./releaseAll.sh --publish-verified build/releaseAll/verified.json
./releaseAll.sh --resume <deployment-id>  # status-only recovery, no rebuild/re-upload
```

The receipt's bundle path must resolve to `build/releaseAll/jmlx-mega-bundle.zip`, whose digest
is independently compared with the preliminary record. The final receipt, native directory,
tag/SHA, plugin plan/receipt, JUnit/event evidence and report
logs are rechecked before upload. The plugin uploads the saved ZIP without `bundleAll`.
AUTOMATIC publication makes it public without a Portal inspection interval; authorization must
name the reviewed ZIP digest. If anything changes, prepare and verify a new candidate.

Run fresh-cache `central` smoke for both 8.4 profiles after Portal publication, including Java 21
on macOS and the local Ubuntu ARM64 container. Compare new downloaded artifacts with bundle
digests. Only then prepare development bumps and module tags. Preserve original tags for reused
modules. Never publish the temporary GitHub draft; delete it within 30 days of publication or
abandonment after archiving evidence, retaining the candidate tag.

## Guards and regression checks

`selftest.sh` runs stdlib unit tests plus real Gradle fixtures for exclusive repository routing
(POM and module metadata, conflicting Maven-local copies, missing sources, warm state) and
producer task outcomes (executed/up-to-date/cached/skipped, configuration-cache reuse).
`verify_native_tests.py` and CI read `native_suites.py`; no second suite list is maintained.
`release_inputs.py native <directory-or-jar> --bootstrap scripts/bootstrap-native.sh` checks
ARM64 Mach-O `LC_BUILD_VERSION`, native pins, personal path leaks and exact legal text digests.
Only the known pinned upstream MLX path is allowlisted; upstream binaries remain unchanged.
The native publication's unconditional unstaged guard also applies to combined staging.

The Portal upload documentation states a 1 GB archive limit; account-level publishing limits
must be checked against current Usage Center data rather than assuming a universal deployment
quota. See the dated readiness record for sources and pending account-specific verification.

Workflow run IDs and container image IDs are operator attestations, not authenticated remote
provenance. Review actual workflow runs and the declared host before publication. Both macOS
report kinds require ARM64. The minimum-host user must provide Python 3.11+; the workflow checks
it before downloading candidate inputs. CI runs checksum-pinned actionlint 1.7.12 for workflow
expression/context validation, in addition to the release regressions.
