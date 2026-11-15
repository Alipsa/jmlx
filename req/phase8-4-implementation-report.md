# Phase 8.4 tooling implementation and evidence

2026-11-14. Local tooling verification complete for the checks listed below; release gates
remain open. No release version cut, candidate tag, signed release bundle, draft Release or
Central deployment has been created. See the [checklist](phase8-4-release-checklist.md).

## Implemented

- One strict 8.4 source/version manifest shared by both consumer profiles and the combined
  selection guard. Candidate preflight validates versions/overrides before probing Central
  and before cleanup/publication. Historical 8.1 Central smoke requires `--release 8.1`.
- Exact-version exclusive repositories for candidate-local versus reused-Central entries,
  identical for POM and module metadata. CI mode retains snapshot artifact smoke.
- Read-only bundle consumer input with preliminary-record/ZIP/manifest/tag object/HEAD checks,
  safe extraction and complete signed new-coordinate artifact presence checks.
- Explicit deployment target 26.2 and local source-prefix maps; clean mlx-c rebuild. Staged
  and packaged payload validation decodes ARM64 Mach-O `LC_BUILD_VERSION`, checks pins and
  path hygiene, and permits only the exact digest-scoped known upstream MLX path.
- Full Apache-2.0 and additional compiled-in dependency notices in native main, sources and
  Javadoc jars, with source/license inventory and exact packaged-text digest checks.
- Shared CI/local native suite inventory. Release bundle preparation clears required test
  outputs, disables cache and uses task-level reruns with a build-service completion listener;
  the preliminary record binds this invocation's JUnit/task outcomes, native bytes, toolchain,
  candidate identity and plugin receipt. It cannot authorize an upload.
- Final receipt creation/verification requires exact-bundle macOS full/pure, local Linux pure
  and physical minimum-host reports, their log digests and environment identities. Changed
  source, plan, runtime, bundle or evidence blocks submission.
- Manual minimum-host workflow transfers two digest-verified draft Release assets in a job
  with draft access; the separate native job has read-only repository permissions. It uses
  the exact annotated candidate tag, a fresh Gradle home and packaged Metal execution.
- Root upload entry rejects ordinary rebuilding uploads. The verified path requires the
  separately released plugin capability and revalidates the final receipt at upload time.

## Local checks

On the Apple Silicon macOS 27.2.1 host with rebuilt/staged native runtime:

- `./gradlew check`: passed both native precision modes and module verification.
- `python3 tools/release-all/verify_native_tests.py`: 104 required result files verified,
  none missing, skipped, empty or failing, including loader/extraction and worker probes.
- Separate `./gradlew -p buildSrc check`: passed.
- Native module check and opt-in packaged payload verifier: passed; mlx-c minos is now 26.2
  with no `/Users/` strings; upstream wheel bytes are unchanged.
- `bash tools/release-smoke/run.sh ci --profile all`: passed fresh-cache artifact checks,
  POM/module graph agreement over 45 runtime components, all nine documentation artifacts,
  native extraction, scheduler/generation and both adapters' public interfaces.
- `ci --profile pure-java`: passed Java 21/headless public APIs.
- Historical `central --profile pure-java --release 8.1`: passed the original
  0.6.1/0.1.1/0.1.1 Central tuple under a fresh consumer home.
- Release parser/selftests: passed manifest/preflight, binary metadata/path and ZIP/tag/digest
  integrity cases. Real Gradle fixtures verify repository isolation, cached/up-to-date/skipped
  task rejection and configuration-cache reuse; 12 release regression tests passed on both
  macOS and Ubuntu ARM64. A generated combined-staging Maven task using
  the actual native module script refuses an unstaged jar while ordinary local jar creation
  succeeds. This fixture establishes the unconditional task-type guard; it is not a signed
  end-to-end combined plugin deployment.
- Final-receipt regressions reject preliminary input, missing/changed logs, changed plans and
  cached tasks claiming success. Current plugin pin rejects existing-bundle upload during
  configuration, before any Portal request.

Ubuntu ARM64: in an isolated source copy, parser/release regressions and fresh-cache pure-Java
artifact smoke passed. The container lacked Python, so validation used a temporary standalone
CPython 3.12.12 runtime from Astral's 20251120 release, verified against SHA-256
`16e50e3facc006e918cf83604a0c9bb5d24e16d4164cd001e94d268f4f62259e`.
No system packages or container configuration were changed. This is development validation,
not candidate-tag/exact-signed-bundle evidence. No Windows behavior was changed or tested.

## Upstream dependency and remaining gates

The existing-bundle API change is committed locally as `47f7a36` in
`Alipsa/nexus-release-plugin`, based on `d942747026bbe8bb22fdd38c6f768264d96284e0`.
Its ModuleReleaseTest and CombinedReleaseDslTest suites passed in an isolated work copy,
including multipart ZIP byte identity, no rebuild and tamper rejection. Source publication
to the separate repository needs explicit approval after automatic review rejected the push.
No plugin release is claimed; the jmlx catalog remains at 2.5.1 and upload fails closed.

WP2 cannot close until that separate review/release, catalog bump and the scratch-commit 8.1
compatibility matrix pass. The physical minimum-host resource is not secured. The manual
workflow, exact signed-bundle consumers, TLS Portal integration, final receipt and Central
postpublication path still require their actual candidate execution after those prerequisites.
The proposed all-new tuple needs post-plugin audit/review; license inventory and current
account publishing capacity need release review. No earlier/local report substitutes for
these gates. Phase 8.4 is not complete.
