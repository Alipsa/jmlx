# Cephes notice implementation verification

2026-11-21. Owner decision merged in PR #109 as
`05703375d372cc0bf5696e4320c01b204dc45c2e`.

The only change to `jmlx-native-macos-arm64/THIRD-PARTY-NOTICES` is the Cephes section
heading. The original Netlib permission and all other notice bodies are byte-identical.
The inventory key and packaged-text hash follow the heading change. The audit's follow-up
entry links this evidence; its historical verification and artifact hashes are preserved.
The audit JSON's top-level `verification` and `artifacts` blocks describe the 2026-11-16
audit. Their superseded notice hash is not the current inventory hash; `verificationScope`
identifies that boundary beside the verification block.

[validation.json](validation.json) records the four local legal-text hashes, generated POM
hash and three rebuilt JAR identities, plus validation results. Paths identify local build
outputs, which are ignored and not committed. These are local producer artifacts, not a
signed candidate bundle or publication receipt. The POM is unchanged by this implementation;
its hash identifies the generated metadata reviewed alongside the notices.

The build base is `05703375d372cc0bf5696e4320c01b204dc45c2e`; native build inputs match
implementation commit `7892d186c881030150fefa622a230fd8314eead3`. Other PR changes record
audit/inventory metadata, documentation and evidence. Later review fixes do not alter native
build inputs. Compilation and Javadoc use Eclipse Temurin `25.0.4.1+1-LTS`; Gradle `9.9.0`
runs on BellSoft `21.0.12.1+1-LTS`. A forced rerun of the artifact tasks reproduced all
three recorded JAR hashes and the generated POM hash. The validation record's
`nativePayloadReference` points to the five prior-audit hashes used for the payload check.

[source-notice-report.json](source-notice-report.json) is the output of the existing
`tools/release-all/audit_native_licenses.py`, with the pinned wheel, URL-bound source cache
and all three rebuilt `0.1.0` JARs. Missing cache entries were fetched and hash-checked;
this was not an empty-cache replay. All 26 components and 34 additional source records
passed, together with packaged text checks for all three JARs.

Validation also ran native `check`, `verifyNativeReleasePayload` and
`generatePomFileForMavenPublication`, and the 17 tests in `test_native_licenses.py`.
A separate byte comparison confirmed the title is the only notice change. A direct
attribution check rejected the stale title, and all four legal texts in all three JARs
matched the inventory. The main JAR's native payload hashes still match the earlier audit.

Both findings remain open. Cephes awaits implementation review and final candidate
validation; Juffa awaits its separate licensing decision. RC1, its tag, ZIP and receipts
are unchanged. A changed release bundle requires a new candidate and verification.
