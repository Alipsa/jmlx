# Phase 8.4 RC3 signed-candidate review

Prepared 2026-11-23. RC3 technical verification passed. This packet is ready for review of
the retained technical evidence; jinja human sign-off, current Portal capacity,
license-finding disposition, signed-candidate/delivery acceptance and publication
authorization remain pending. This packet is not eligible for publication acceptance. No
owner acceptance is asserted. RC2 remains historical and blocked by its lost inputs.

## Candidate identity and preservation

PR #120 merged as `33fd8d36aa7468a51f3e6ecc57bb581bf83731d6` after head CI passed. The
annotated immutable `jmlx-phase8.4-rc.3` tag has object ID
`146f7cf1b8b4557f20d732436dcdba6c45d20a55` and peels to that merge. The candidate checkout
remains detached and unchanged. This follow-up PR adds review evidence outside the tagged
candidate; it changes no candidate input or signed artifact.

- Signed ZIP SHA-256:
  `139bff18f638136a34a1e61de942322a1d5523a821deca8b282b05b9ca3a500c`
- Preliminary-record SHA-256:
  `929018af3b4e237576f48c1c75215bde6716f98efbfb98149deebde01fd42926`
- Final verified-receipt SHA-256:
  `db666ab8141dc08ca24e2e6384e0f714162d0acff9f6d12caaf4ce9b104120b6`

The [publication review manifest][packet] uses the owner-acceptance gate's schema. Root
`check` runs `verifyRc3ReviewEvidence`, enforcing recursive inventory, required sizes,
SHA-256 pins, the pinned review document and containment within this packet. The manifest is
excluded from its own inventory pins; the guard compares its digest with the living
checklist's reference. These checks enforce consistency. The reviewed Git commit and
separately authorized acceptance digest establish the later publication reference.

Candidate and audit checkouts survive under `~/release-candidates/jmlx-phase8.4-rc.3`
and `~/release-candidates/jmlx-phase8.4-rc.3-audit`. Complete copies, including ignored
native/build output, survive under
`~/release-candidates/jmlx-phase8.4-rc.3-backup-2026-11-23/`. The backup verification
checks all 170 receipt/window inputs in the completed copy, including all 104 original
JUnit XML reports. This is local preservation on the same Mac, outside Gradle clean and
temporary storage. It is not an independent-device backup. The post-window helper at
`minimum-host-window/admin-window.py` was edited after backup verification to remove a
trailing space. That edit changed one of the 170 archived input digests. The
[preservation repair][repair] records its restoration from the verified backup and a new
size/SHA-256 check of all 170 original inputs. The normalized backup snapshot and its
provenance now refer to the restored 6,219-byte original. Review-only helper revisions
are separate files; they do not replace preserved candidate evidence. Preserve original
absolute
paths: the verifier is not relocatable. Do not clean, rebuild or replace this candidate.

The [provenance index][provenance] records sizes and hashes of durable originals and public
copies. Incidental local paths and user/host labels are normalized in most text. The
final/preliminary receipts, four producer JSON inputs, consumer JSON/log/runtime sidecars
and timestamped account probes retain exact bytes for their cross-file pins, including
absolute paths. No signing key, Portal credential or runner authentication file is included.
The ZIP, native binaries, full checkouts and JUnit XML remain in durable local storage.

## Technical verification and toolchains

Root checks and native release-license verification passed at the tagged SHA, as did
separate buildSrc checks and all 122 release regressions. The pure-Java historical-version
audit executed all 27 publication tasks, checked compiler/Javadoc/daemon identities and
reproduced the 15 reviewed artifact comparisons against hash-verified Central downloads. The
historical comparison tools used BellSoft `21.0.12.1+1-LTS`. JARs remain identical; the
expected Gradle/Jackson publication metadata differences support the approved tuple.

Both exact-tag jinja `releaseVerification` passes and the nine-module dry-run passed.
Both complete reports/logs, dependency review, corpus report, POM and archive comparison
are retained separately. Their changelog input is
`5984b78b9933c167b9362b5465877a7f60a3b6a68a3a624bec0695d8b9af3a26`.
Both reports leave the human sign-off unchecked. The separate
[jinja sign-off record][jinja-signoff] explicitly remains pending for NOTICE, changelog,
POM, Javadoc, dependency report and corpus evidence.

Retained logging-only init scripts recorded the invoking and inner Gradle daemons as
BellSoft `25.0.5+11-LTS`. The verifier's selected launcher probe recorded Temurin
`21.0.13+9-LTS`; executed inner compiler/Javadoc tasks recorded that same Java 21 runtime.
Node was v26.7.0. This instrumentation changed no committed candidate input. The first
attempt stopped before verification because the hook read an unresolved executable; the
corrected hook uses Gradle's selected launcher. Its failed log is retained, followed by the
two successful passes. The first Linux launch failed before running its consumer because the
shell launcher was invoked directly; the Bash retry passed.

The [driver history][drivers] distinguishes executed sources from later revisions.
`verification/run-candidate.py` is the resume driver using `steps[2:]`; the initial
full-loop source that ran buildSrc and release-selftest was edited in place and was not
separately preserved. Its original driver log and both per-step invocation records are
retained. `verification/run-consumers.py` is the later Bash-launch revision and was not
executed end-to-end. The original direct-`linux-exec` source is not preserved; the exact
failed invocation and traceback are retained, and `resume-linux.py` is the successful retry
driver. These source-history limits do not change the measured task/report results.

Signed bundle preparation passed signing preflight and reran required tests and native
payload/license tasks with the build cache disabled. Its exact argv, task events, plugin
receipt, plan, native report and 104 JUnit digests remain pinned by the preliminary record.
This does not mean every archive/Javadoc task reran: native Javadoc was restored from cache
during root checks and reused during bundling, as discussed below.

| Exact signed-ZIP consumer | Measured runtime | Result |
| --- | --- | --- |
| macOS full | BellSoft 25.0.5+11-LTS | Passed |
| macOS pure Java | Temurin 21.0.13+9-LTS | Passed |
| Linux ARM64 pure Java | BellSoft 21.0.12.1+1-LTS | Passed |
| Physical minimum host, dedicated account | Temurin 25.0.5.1+2-LTS | Passed |

## Minimum-host window and final receipt

[Run 45206346138](https://github.com/Alipsa/jmlx/actions/runs/45206346138) ran the
reviewed workflow from `main` at the candidate SHA. Both transfer and minimum-host jobs
passed. Runner ID 23, job ID 137803124482 and the actual Set up job log establish runner
`2.338.2`, matching the pre-window measurement. The retained complete job log, setup
excerpt and report record physical ARM64 macOS 27.2.1 (26B131), Apple M5 Max, no
virtualization and successful exact-ZIP native kernel execution. This is the same Mac
as the local macOS runs, under a separate non-admin account; it is not independent
hardware coverage. Workflow IDs and container IDs remain operator attestations.

Pre-window, connected and post-window probes retain timestamped filenames, method records
with script/output hashes, and source-commit sidecars naming the candidate SHA. The account
measurements match across the window. Home mode `0700`, private-owner-path access checks,
saved `.env`/`.path`, runner identity and owner-managed Homebrew tools match the reviewed
baseline. Queue review found no active runs before startup. These checks retain their
parent-directory, owner-managed-toolchain and exposed-job limits.

At the retained post-window observation on 2026-11-23, SMB sharing was disabled; guest
read-write settings remained configured. Re-enabling the share would restore those settings.
This is a dated observation, not a continuing sharing guarantee. The runner was confirmed
offline/idle and its launchd service absent. The shutdown command succeeded, but its
immediate assertion observed `SIGTERMed` while launchd removed the service. The helper
believed to have run is retained as `window/admin-window-original.py`. Its raw digest was recorded only in
`backup-verification.json` at 15:36:06Z, after the stop observation at 15:34:19Z.
Window method records pin the probe script, not the helper; no window-time digest
establishes that these helper bytes ran. The backed-up post-window wait
revision is `window/admin-window-backup-snapshot.py`; it did not run in the window.
`window/admin-window.py` is a new review-only revision using explicit `SystemExit` safety
checks that remain active under `python -O`. It creates the probe exclusively without
following symlinks while the directory is root-owned, transfers file ownership by descriptor,
and transfers directory ownership last. It also waits for removal and has not been run
as root. Historical copies retain the staging race and have not been rewritten. The
archived start/stop/watcher wrappers are believed to have resolved their `admin-window.py`
basename to the original during the window, not to this new revision. Historical queue,
watcher and preservation helpers retain `assert` gates that optimization can disable;
the start wrapper invokes the queue helper without `-I`. These are historical records
and must not be reused as hardened drivers. The original
observation and later confirmation are retained. No candidate rerun was needed.

The final technical receipt was created and revalidated before the shutdown signal, then
revalidated again after the window closed. Revalidation must repeat immediately before any
separately authorized upload. Owner acceptance remains a separate gate.

## Signed native artifacts and remaining owner decisions

The [reproducible native comparison][comparison] verifies the exact PR #115 approval record
digest and compares all five native publications from this ZIP. Runtime, sources and Javadoc
JARs match the approved whole-file hashes; all five legal texts match in each JAR, with no
encrypted entries. Native POM and module metadata match the approved literal bytes. The RC2
generated-JavaScript and Javadoc checksum differences are absent in RC3. The native Javadoc
SHA-256 is `78c9ceb3bbae86b43ac96abd34f3cd05d80f3971574d02f69dd10573e7bf0728`.

The original approved Javadoc producer JVM remains unestablished. RC3 root checks restored
that exact archive's Javadoc output from cache; the current release JDK measurement is not
evidence of its cached producer. The archive comparison establishes exact equality; no
historical toolchain cause is inferred and historical producer provenance is unrecovered.
The pure-Java audit and exact-tag jinja builds have separate explicit producer records.

Reproduce the native comparison against the unchanged local ZIP with:

```sh
python3 req/plans/evidence/rc3-final-review-2026-11-23/compare-native.py \
  --bundle "$HOME/release-candidates/jmlx-phase8.4-rc.3/build/releaseAll/jmlx-mega-bundle.zip" \
  --approval req/plans/evidence/native-distribution-review-2026-11-22/owner-acceptance.json \
  --output /tmp/rc3-native-comparison.json
```

The proposed delivery remains one combined Central deployment of this saved ZIP, with the
original grants, attribution, modification indications, approved scope paragraph and full
legal texts/notices retained. No new recipient restriction or technical measure is proposed.
The owner's [recorded interpretation](phase8-4-owner-license-decision.md), adapter-license
counterargument and conditional PR #115 conclusion remain the basis for owner review.
Technical checks do not close `NATIVE-LICENSE-1` or `NATIVE-LICENSE-2`. Their disposition,
signed-candidate acceptance and actual delivery terms remain pending.

The ZIP measures 106,730,869 bytes, with 270 files totaling 107,448,758 bytes. Its CRCs,
unique filenames and conservative upload-size check pass. This is not a Portal account
capacity verdict. The [capacity status][capacity] requests current Usage Center scope,
plan/classification, monthly byte/file/release-event usage and limits, adjusted-limit expiry
and grace/enforcement banners. No current account snapshot has been supplied.

Temporary draft Release 455573517 retains only the exact ZIP and preliminary record;
downloaded asset bytes matched both expected digests. Keep it unpublished. Archive the
evidence and remove the draft/assets within 30 days of publication or abandonment,
preserving the immutable tag.

This PR may be merged as a technical evidence record, but acceptance must not name this
incomplete packet commit. Before acceptance, a superseding reviewed packet commit must
retain the actual jinja human sign-off, disposition of both license findings, current
Portal capacity snapshot and owner delivery/signed-candidate decisions. That update must
revise this review document, regenerate its pin and the publication manifest, and update
the checklist's manifest digest. Review and merge the superseding packet before issuing
acceptance. Actual acceptance must name its full commit and new manifest digest, this
ZIP and final receipt, and all explicit owner decisions under the
[acceptance contract](../tools/release-all/owner-acceptance.md).
Fetch the packet commit into the detached candidate repository without changing its
checkout. No acceptance record or publication authorization is created here. Fresh-cache
Central verification and module tags remain post-publication work.

[packet]: plans/evidence/rc3-final-review-2026-11-23/publication-review.json
[provenance]: plans/evidence/rc3-final-review-2026-11-23/evidence-provenance.json
[comparison]: plans/evidence/rc3-final-review-2026-11-23/verification/signed-native-comparison.json
[jinja-signoff]: plans/evidence/rc3-final-review-2026-11-23/jinja-human-signoff.json
[capacity]: plans/evidence/rc3-final-review-2026-11-23/portal-capacity-status.json
[drivers]: plans/evidence/rc3-final-review-2026-11-23/driver-history.json
[repair]: plans/evidence/rc3-final-review-2026-11-23/preservation-repair.json
