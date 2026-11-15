# Phase 8.4 plugin 2.6.0 integration

2026-11-15. This records development verification of the released plugin integration.
No jmlx release version, candidate tag, signed release bundle or Central deployment was created.

## Published dependency

[Upstream PR #15](https://github.com/Alipsa/nexus-release-plugin/pull/15) merged as
`cf1c7fd684bf09f93286c89e1ba8d024482488a9`. The jmlx catalog now pins
`se.alipsa.nexus-release-plugin` 2.6.0, resolved through the Gradle Plugin Portal.
The resolved implementation JAR has SHA-256:

```text
379e4b2266728deee1d347fbc6b04df9828a1e4ec23bf6ab53a6eb3a9ac6c5e7
```

The shell wrapper revalidates the final receipt and obtains the reviewed ZIP and manifest
hashes from its preliminary record through `--plugin-digests`. It passes those two anchors
to the plugin's `expectedBundleSha256` and `expectedManifestSha256` properties. The root
upload action revalidates the final receipt again before the plugin uploads its checked
snapshot. Digest extraction emits nothing if verification fails. Resume remains status-only;
it trusts the retained deployment journal, so protect that journal against modification.

## Historical 8.1 compatibility

A separate local clone starts at the published recovery base
`9bc85042df50ed3f979cb8d3e7c8767917c71483`. Its scratch branch contains only throwaway commit
`5f341f9a1d66910b5cb6a9a23581e75dd64abe9c`. That branch/commit is never pushed or tagged.
The checkout remains clean. Its complete source diff is:

```diff
-nexusRelease = { id = "se.alipsa.nexus-release-plugin", version = "2.5.1" }
+nexusRelease = { id = "se.alipsa.nexus-release-plugin", version = "2.6.0" }
```

The retained full Git diff has SHA-256
`71194fe4d9e954ad35bf77c004f46618c917b2c8f03c2143500c75b9cd8de255`.
The historical wrapper and source versions are unchanged, including Gradle 9.7.0.
Its separate Gradle home shares downloaded caches only; it contains no user credential
properties or inherited init scripts.

| Scenario | Result |
| --- | --- |
| `releaseAllPlan` against the real Central repository | Empty selection; all original pure-Java coordinates already released |
| Historical `releaseAll.sh --dry-run` against Central | Expected exit 1: its unchanged exact-three guard refuses the empty selection |
| Historical `releaseAll.sh --dry-run` against HTTPS absence fixtures | Exit 0; selects exactly jinja 0.6.1, tokenizer 0.1.1 and vision 0.1.1 |
| HTTPS fixture request journal | No upload requests; only repository/availability probes |

The eligible-three fixture uses a local TLS certificate/truststore, dummy Portal credentials,
404 repository responses and explicit `published:false` Portal responses. Fresh JVMs avoid
reusing trust material from an earlier fixture certificate. Missing signing keys are warnings
in dry-run mode; no signing, bundling or publication runs.

Detailed logs, both manifests, request journal, source diff, scratch identity and artifact
hashes are retained locally under ignored `build/releaseAll/plugin-2.6.0-evidence/`.

## Current integration checks

- Root `./gradlew check` and separate `./gradlew -p buildSrc check` pass with plugin 2.6.0.
  Ordinary incremental task reuse applies; this is not fresh release-invocation evidence.
- The shared native-suite verifier accepts 104 required JUnit files with no skipped suites.
- Release-smoke parser/preflight selftests pass; the full nine-module artifact consumer
  passes native extraction, scheduler/generation, adapter APIs and published graph checks.
  The separate fresh-cache Java 21 pure-Java artifact consumer also passes.
- The original local 23-test run passed using shared downloaded caches; it did not establish
  cold-cache TLS resolution. The 2026-11-16 correction below replaces that fixture setup.
  The suite includes the real root upload block and released plugin against a local HTTPS
  Portal. All nine synthetic producer coordinates enter the plan.
  The fixture confirms that ordinary uploads, source/manifest/ZIP drift and rejected test
  evidence make no upload request. Success uploads exactly the reviewed ZIP bytes without
  `bundleAll` and records the same hash in the deployment journal.
- The same fixture checks the actual shell wrapper's digest argument vector, including a
  receipt filename with spaces, and verifies that rejected evidence prevents Gradle invocation.
  Native/consumer evidence and shell preflight are synthetic in this integration fixture;
  the real final-receipt validator has separate evidence-integrity regression coverage.

## Cold-cache and receipt-binding correction (2026-11-16)

The first hosted CI run exposed a fixture error: its one-certificate truststore replaced
public CA roots and the developer-cache symlinks hid the resulting Portal resolution failure.
The fixture now starts with an entirely empty Gradle home, copies the selected launch JDK's
`cacerts` into a private truststore and adds only the fixture certificate to that copy. It
uses that same JDK for Gradle, checks that the original CA file is unchanged, and borrows no
artifact or JDK cache. Gradle output uses `--console=plain` for task-header assertions.

The root upload block has explicit extraction markers. Its upload action obtains the
reviewed hashes from the receipt verifier again and compares them with both plugin inputs,
including for direct Gradle invocations. The shell rejects malformed or extra verifier
output before launching Gradle. Regression cases cover both rejection paths and exact
successful upload. All 23 release selftests pass locally with the isolated cold-cache TLS
fixture. Hosted CI verification is pending this correction.

The recorded plugin JAR hash above identifies the resolved artifact; it is not currently
enforced by Gradle dependency verification. Verification metadata covering the release
plugin and its dependency graph remains a separate dependency-policy follow-up.

## Remaining release work

The external plugin prerequisite and historical compatibility checks are satisfied by this
change. The pure-Java artifact comparison must still run after this pin bump at the final
candidate SHA, and the provisional tuple still requires review. Physical minimum-host access,
legal/account capacity review, exact signed-bundle verification and explicit authorization to
publish that concrete bundle remain release gates. No development fixture replaces them.
