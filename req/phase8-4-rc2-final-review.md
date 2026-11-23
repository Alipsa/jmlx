# Phase 8.4 RC2 final candidate review

Prepared 2026-11-22 for Per Nyfelt; updated 2026-11-23. Historical technical verification
passed. RC2 is now blocked from acceptance and publication because its candidate checkout and
required producer evidence are missing. Owner acceptance, planned delivery terms and current
Portal capacity also remain pending. No Central publication is authorized. Earlier owner
decisions and evidence remain unchanged.

## Tagged candidate and retained evidence

PR #116 merged as `33676837f66993b5a809644dbbdc6e6550417334` after [CI run
45072595248](https://github.com/Alipsa/jmlx/actions/runs/45072595248) passed. The annotated
`jmlx-phase8.4-rc.2` tag has object ID `400b90e1a0b062a57b70a1007e68a4f6313f3976` and peels to
that merge. Preparation and verification used its clean detached checkout. This review is a
later documentation follow-up; it changes neither the candidate nor its artifacts. RC1 is
historical.

- Signed ZIP SHA-256:
  `0dbb1451f8015843dfd77aeff5ae5e7f1afdbb80bb01f489ee06dfd92f1ba35e`
- Preliminary-record SHA-256:
  `3a530d8386834abb20de23b96d4ec0a46e6cf62207952b193c2fc9d6e29689af`
- Final verified-receipt SHA-256:
  `5253382f3921c79f06b19746d479edf619e565a83d39b7678159549c0be84f55`

The [review manifest][review] pins the retained evidence. It includes exact consumer
JSON/log/runtime sidecars, preliminary and final receipts, native metadata, the signed-native
comparison, account probes, sharing snapshots and Portal ZIP inventory. Original local paths
remain in the receipts. Revalidation requires the original candidate checkout, native files,
task/JUnit evidence and ZIP; the committed copies alone are not a portable release.

On 2026-11-23, inspection confirmed that `/private/tmp/jmlx-phase8.4-rc.2` and its audit
checkout no longer exist. The retained ZIP survives, but the pinned plugin bundle receipt,
manifest, task-evidence JSON and Gradle argv JSON were not retained with it. Recovery also
requires all 104 digest-pinned JUnit XML reports under the candidate's module test-result paths
and the staged `native/install/lib` matching the preliminary record's complete native report.
The original JUnit XML bytes cannot be regenerated: timestamps and durations are part of their
digests. The main checkout's bundle receipt has different bytes; whether its native staging
matches has not been established. The preliminary record survives byte-for-byte under
`transport/` and in this packet, but must also be restored at its recorded original path. The
surviving signed ZIP must be restored from `transport/jmlx-mega-bundle.zip` to
`/private/tmp/jmlx-phase8.4-rc.2/build/releaseAll/jmlx-mega-bundle.zip`; the verifier requires
that exact candidate-root bundle path. Consumer reports/sidecars must also be restored to
their original absolute paths under the main checkout if those files are removed. Recovering
only the four producer JSON files is insufficient.

The saved final receipt cannot currently be revalidated. Do not accept or publish RC2 in this
state. Recover the original inputs from backup and pass the unchanged verifier, or cut and fully
verify a new candidate in a durable checkout. A relocatable verifier would require separate
review; it is not supplied here. Rebuilding missing inputs cannot restore this receipt's
provenance.

On 2026-11-23, the complete surviving `build/releaseAll/candidates/rc.2/` tree and all three
`window-2026-11-22.*` trees were copied to durable local storage outside the checkout:
`/Users/per/release-candidates/jmlx-phase8.4-rc.2-survivors-2026-11-23/`.
This includes the signed ZIP, transport preliminary record, four consumer reports and their
sidecars, prerequisite logs and window records. The copy preserves repository-relative paths.
The [backup record][backup] pins its local `backup-inventory.json`: all 3,438 file/symlink
entries were compared with the source before and after copying; the ZIP and four consumer
report hashes also match the receipt. `./gradlew clean` can delete the working copies under
`build/` but cannot delete this separate archive. This is a local preservation copy, not an
independent-device backup or recovery of the missing candidate inputs. Restore files to the
receipt's original absolute paths before attempting unchanged verification; the backup location
does not make the receipt relocatable.

The temporary draft Release contains only the verified ZIP and preliminary record; it stays
draft.

## Verification completed

Root checks, separate buildSrc checks, all 109 release regressions and smoke selftests passed.
Two consecutive jinja `releaseVerification` runs passed at the candidate SHA with pinned Node
v26.7.0. Both original logs are now retained as `jinja-verification-1.log` and
`jinja-verification-2.log` beside the report. The report's human sign-off remains unchecked;
owner review of NOTICE, changelog, POM, Javadoc, dependencies and corpus evidence is pending.
The historical-toolchain pure-Java audit reproduced all 15 reviewed comparison records against
hash-verified retained Central artifacts. The nine-module dry run selected the approved tuple.
Bundle preparation passed signing preflight and reran required tests and native payload/license
tasks with caching disabled.

| Exact signed-ZIP consumer | Measured runtime | Result |
| --- | --- | --- |
| macOS full | BellSoft 25.0.5+11-LTS | Passed |
| macOS pure Java | Temurin 21.0.13+9-LTS | Passed |
| Linux ARM64 pure Java | BellSoft 21.0.12.1+1-LTS | Passed |
| Physical minimum host, dedicated account | Temurin 25.0.5.1+2-LTS | Passed |

[Minimum-host run 45076371223](https://github.com/Alipsa/jmlx/actions/runs/45076371223) used
reviewed `main` at the candidate SHA, runner ID 23 and job ID 137374458588. Its report records
physical Apple M5 Max, ARM64 macOS 27.2.1 (26B131), no virtualization and successful exact-ZIP
consumer/kernel execution. This is the same physical Mac as the local macOS consumer runs, using
a separate account; it is not independent host evidence. The actual job setup log reports runner
2.338.2, matching the listener measurements before connection, after connection and after the
window. The full job log hash and retained setup excerpt support that comparison.

The pre-window, connected and post-window account measurement bytes match. Retained
`account-probe-provenance.json` maps each copy to its surviving window directory and
`source-commit.txt`. The connected observation also retains its original timestamped filename.
Pre-window output has no exact probe timestamp; do not infer one from file times. Private-home
access checks pass; their parent-directory and owner-managed-toolchain limits still apply. The
owner disabled the Public-folder SMB share before verification. At the last retained observation
(2026-11-22 22:18:35 UTC), the share was disabled, the worker was offline and idle, and launchd
had no loaded service. Guest read-write settings were still configured: re-enabling the share
would restore that exposure. These observations do not establish current state. The initial
shutdown check ran before termination completed and saw `SIGTERMed`; later inspection confirmed
removal. The original observation is retained locally. A helper revision waits for removal
instead of treating that immediate snapshot as a completed stop. No candidate rerun was
required.

## Signed native artifacts and metadata

The [signed-native comparison][comparison] checks ZIP members against the exact local artifacts
and literals accepted in PR #115. The runtime and sources JARs match those whole-JAR hashes. All
five legal texts match in all three JARs, and the native POM matches its approved bytes. The
scoped CC comment, immutable notice permalink and retained Cephes notices are preserved.

The native Javadoc JAR differs in entry content only in three generated JavaScript resources:
`script-files/script.js`, `script-files/search-page.js` and `script-files/search.js`. The
[retained diff][javadoc-diff] shows text/HTML assignment changes. No entry was added or removed,
and every other entry's content matches the approved archive. This is an archive comparison, not
a browser-behavior or security claim. The preliminary record identifies the RC2 release JDK as
BellSoft `25.0.5+11-LTS`. The approved PR #115 Javadoc producer JDK vendor/full runtime version
was not retained in the reviewed packet and remains unestablished; its post-merge log reused the
Javadoc task. A JDK patch difference is a hypothesis, not an established cause. Before
acceptance, recover the approved producer version and compare both actual Javadoc toolchains, or
review a replacement candidate with explicit producer records. The Java 21 pure-Java audit is
unrelated to this native JAR. The candidate native Javadoc SHA-256 is
`d832a704c6e97b89d12676e0ff7694fdf0da43f9c8788de637dfec96d784fcdf`.

The actual native module metadata differs from its approved literal only in the Javadoc
variant's size and four artifact checksums. Its candidate SHA-256 is
`28064ff4acb37e68373fec4f68653a03f07d67e7729bace2bbab5351f969d7ab`. The new size/checksums
describe the candidate Javadoc JAR. These exact differences need owner review; this record does
not silently extend the approval of the earlier module literal.

## Proposed delivery and remaining decisions

The proposed public delivery is one combined Central deployment of the unchanged signed ZIP,
with the original grants, full legal texts/notices and approved scope paragraph retained. The
draft release notes keep that paragraph unchanged. No new recipient restriction or technical
measure is proposed. The temporary GitHub draft is private transport only and must not be
published. Archive evidence and remove its assets/draft within 30 days of publication or
abandonment while preserving the immutable candidate tag.

The [owner's recorded interpretation](phase8-4-owner-license-decision.md) and counterargument
remain the basis for review. Technical checks do not resolve legal questions. The owner must
review this signed candidate, the documented metadata differences and proposed delivery under
that interpretation, then record the disposition of both license findings. Neither finding is
closed here. Applicable-law and compiled-boundary limits remain unchanged.

The ZIP measures 106,730,874 bytes and contains 270 files totaling 107,448,768 bytes, including
the signature/checksum sidecars. It passes the existing conservative upload-size check. The
[capacity procedure](phase8-4-portal-capacity.md) still requires current account usage, limits,
scope, plan and any enforcement or exemption conditions. The official documentation was
rechecked on 2026-11-22; the authenticated account snapshot is pending. One combined publish
operation is planned; projected file/byte usage must include other monthly activity.

After provenance recovery or replacement-candidate verification, owner review and capacity
confirmation, publication still requires separate explicit authorization naming the ZIP digest
above. Revalidate the saved receipt and unchanged ZIP immediately before using
`--publish-verified`. RC2 cannot pass this step with the currently retained inputs. Do not
rebuild or use a newly generated plan to substitute different bytes. Fresh-cache Central checks
remain after publication.

The existing uploader verifies technical receipts; it does not enforce RC2 owner acceptance.
Digest-specific human publication authorization remains mandatory. A digest-pinned final
acceptance gate is a separate implementation item before a replacement release workflow is
approved; the earlier `owner-acceptance.json` does not accept RC2 differences or jinja human
sign-off.

Original evidence retains local paths, host identity and the audit PATH. These are historical
raw records, not credentials. They are left byte-identical to preserve their pins; future public
summaries should normalize incidental local details and retain raw records privately.

Record any owner acceptance separately against the full reviewed commit and SHA-256 of this
Markdown file and the review manifest. Keep the approved snapshot unchanged afterward.

[review]: plans/evidence/rc2-final-review-2026-11-22/review.json
[comparison]: plans/evidence/rc2-final-review-2026-11-22/signed-native-comparison.json
[javadoc-diff]: plans/evidence/rc2-final-review-2026-11-22/javadoc-content.diff
[backup]: plans/evidence/rc2-final-review-2026-11-22/survivor-backup.json
