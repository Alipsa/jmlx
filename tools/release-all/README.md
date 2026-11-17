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
The supported native floor is macOS 27.2 (any 27.2.x patch release); physical minimum-host
kernel execution remains a release gate. `MACOS_SUPPORTED_MINIMUM` in bootstrap supplies
the receipt `supportedMinimum` independently of the 26.2 binary deployment target. Hosted `macos-26`
checks provide additional coverage; they do not satisfy the physical 27.2.x release gate. Ordinary contributor builds retain their unstaged-runtime skip behavior.

Signing and Portal credentials use the existing Gradle properties/environment contract.
Reports must contain no credentials. Receipts/logs live under ignored `build/releaseAll/`
or outside the checkout, since the source guard rejects untracked files too.

The catalog pins nexus-release-plugin **2.6.0**, including the separately reviewed
[existing-bundle upload API](https://github.com/Alipsa/nexus-release-plugin/pull/15).
The wrapper obtains reviewed ZIP/manifest SHA-256 anchors only after full final-receipt
verification and passes both to the plugin. The root upload action verifies the receipt
again and compares both returned anchors with the plugin inputs before submission. The shell
rejects malformed verifier output before launching Gradle. See the
[historical 8.1 compatibility evidence](../../req/phase8-4-plugin-2.6.0-compatibility.md).
Resume trusts the manifest copy retained in the deployment journal; those digest inputs
are ignored for status-only recovery. Protect the journal against modification.

## Prepare, verify, then submit

1. Complete WP0 readiness and tuple review; rerun the pure-Java metadata audit after the plugin
   bump. Record the new plugin's historical 8.1 regressions on an unpushed throwaway commit.
2. Merge the reviewed release-version PR. Anchor an annotated immutable `jmlx-phase8.4-rc.N`
   tag at that SHA and work from a clean detached checkout. Never move the tag.
   Cut the candidate only after the minimum-host workflow-identity check
   (`verify_minimum_host_workflow`) is on `main` and included in the tagged commit:
   `bundle_smoke.py` runs from the candidate checkout.
   An older tag cannot produce the workflow block now required by final-receipt verification.
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
   `release-candidate.yml` from `main` with both asset IDs/digests, Release ID and tag/peeled SHA.
   The transfer job has draft-access permissions and never executes the bundle. The dedicated
   physical minimum-host job has read-only repository access and checks tag object, HEAD, digests,
   native OS floor and actual packaged kernels. Provision its `jmlx-release-minimum` runner
   label before attempting this gate; a VM or moving hosted image cannot establish the floor.
   Use the [dedicated-account setup procedure](../../req/phase8-4-minimum-host-setup.md);
   the existing `self-macos-xl` PR/Tier-B runner does not satisfy that isolation requirement.
   Revalidate registration, saved environment, runner/tool versions and queued label-matching
   jobs before starting the worker. Offline registrations can expire after more than 14 days.
9. Download minimum-host JSON/log evidence, including the actual run's **Set up job** log.
   Compare its `Current runner version` with
   the pre-dispatch measurement; resolve any update and record the actual version before
   accepting the report, as described in the dedicated-account procedure. Then create the
   final receipt:

```sh
python3 tools/release-all/final_receipt.py --root "$PWD" \
  --bundle build/releaseAll/jmlx-mega-bundle.zip --preliminary <preliminary.json> \
  --report macos-full=<report.json> --report macos-pure=<report.json> \
  --report linux-pure=<report.json> --report minimum-host=<report.json> \
  --output build/releaseAll/verified.json
```

Each report's log and measured `.runtime.xml` must remain beside its JSON. The consumer JVM
itself writes feature/version/runtime build/vendor/home; its file and digest are verified,
independently of the Java launcher on PATH. The minimum-host report also records the manual
workflow's main ref and SHA; review these against the actual GitHub run and reviewed workflow
commit. Job guards and recorded identity do not restrict other workflows from using its labels.
Only a **final** receipt can be submitted:

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

`compare_publications.py --central <flat-download-dir> --candidate <flat-rebuild-dir>
--output <report.json>` records raw/normalized hashes, `allEqual`, and JAR content/metadata
changes for the pure-Java reuse audit. `--require-equal` exits 1 on differences; `--publications <json>`
requires five exact filenames per module/version in that JSON object. Rebuild using the old
published versions in an isolated checkout;
never replace versions in the release worktree to simulate 8.1. The
[post-plugin audit](../../req/phase8-4-pure-java-audit.md) records toolchain identity,
normalization rules, current results and the required final-candidate rerun.

`selftest.sh` requires OpenSSL and the JDK `keytool` for its local TLS fixture. It runs
stdlib unit tests plus real Gradle fixtures for exclusive repository routing
(POM and module metadata, conflicting Maven-local copies, missing sources, warm state) and
producer task outcomes (executed/up-to-date/cached/skipped, configuration-cache reuse).
Its published-plugin HTTPS fixture checks the real root upload block and shell digest bridge
with dummy credentials and synthetic consumer evidence. Each run uses an empty Gradle home
and a private copy of the launch JDK default CAs extended with the fixture certificate, so
Plugin Portal resolution is tested cold. It does not establish physical-host or signed-candidate
compatibility.
`verify_native_tests.py` and CI read `native_suites.py`; no second suite list is maintained.
`release_inputs.py native <directory-or-jar> --bootstrap scripts/bootstrap-native.sh` checks
ARM64 Mach-O `LC_BUILD_VERSION`, native pins, personal path leaks and exact legal text digests.
Only the known pinned upstream MLX path is allowlisted; upstream binaries remain unchanged.
`--documentation-jar <jar>` also checks every reviewed notice in a companion artifact;
it scans companion entries for the supplied `--forbid` prefixes as well.
The main JAR's other entries receive the same scan; its five native payload entries retain
the separate binary/path verifier and pinned upstream MLX exception.
`:jmlx-native-macos-arm64:verifyNativeReleasePayload` checks main, sources and Javadoc jars.
The [native license audit](../../req/phase8-4-native-license-audit.md) records the source
inventory, reproduction command and unresolved provenance findings. Its
`audit_native_licenses.py` check supports `--require-clear`; source/notice consistency alone
does not close the license release gate.
The native publication's unconditional unstaged guard also applies to combined staging.

The Portal upload documentation states a 1 GB archive limit; account-level publishing limits
must be checked against current Usage Center data rather than assuming a universal deployment
quota. Follow the [capacity procedure](../../req/phase8-4-portal-capacity.md) to retain the
account snapshot and measure the exact signed bundle. Account verification remains pending.

Workflow run IDs and container image IDs are operator attestations, not authenticated remote
provenance. Review actual workflow runs and the declared host before publication. Both macOS
report kinds require ARM64. The minimum-host user must provide Python 3.11+; the workflow checks
it before downloading candidate inputs. CI runs checksum-pinned actionlint 1.7.12 for workflow
expression/context validation, in addition to the release regressions.
