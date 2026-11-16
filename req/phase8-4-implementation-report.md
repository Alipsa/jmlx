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
  cached tasks claiming success. The 2.6.0 integration additionally verifies reviewed digest
  forwarding and exact uploaded bytes against a local HTTPS Portal fixture.

Ubuntu ARM64: in an isolated source copy, parser/release regressions and fresh-cache pure-Java
artifact smoke passed. The container lacked Python, so validation used a temporary standalone
CPython 3.12.12 runtime from Astral's 20251120 release, verified against SHA-256
`16e50e3facc006e918cf83604a0c9bb5d24e16d4164cd001e94d268f4f62259e`.
No system packages or container configuration were changed. This is development validation,
not candidate-tag/exact-signed-bundle evidence. No Windows behavior was changed or tested.

## Upstream dependency and remaining gates

The separately reviewed API is now published as nexus-release-plugin 2.6.0 and integrated
in the catalog. The wrapper passes both reviewed digest anchors after final-receipt
validation; the root action revalidates that receipt before plugin upload. The clean,
unpushed historical 8.1 scratch commit passes both published-empty and fixture-eligible
selection checks. The HTTPS integration fixture verifies byte-identical upload and failure
gates with synthetic producer/evidence inputs. See the
[compatibility report](phase8-4-plugin-2.6.0-compatibility.md) for exact identities and limits.

The physical minimum-host resource is not secured. Exact signed-bundle consumers, the manual
workflow, final receipt and Central postpublication path still require actual candidate
execution. No jmlx release has been performed.
The proposed all-new tuple needs post-plugin audit/review; license inventory and current
account publishing capacity need release review. No earlier/local report substitutes for
these gates. Phase 8.4 is not complete.

## Review corrections (2026-11-15)

The minimum-host workflow now creates its Gradle home through `GITHUB_ENV` at step scope;
checksum-pinned actionlint 1.7.12 rejects the former invalid job-level `runner` expression and
passes the corrected workflows. CI runs that context check. Transfer uses Python 3.12;
the physical host checks for provisioned Python 3.11+, and release entry points reject the
system Python 3.9 with a clear message before cleanup/building.

Final verification requires the receipt's canonical bundle path to be the plugin's configured
`build/releaseAll/jmlx-mega-bundle.zip` and independently checks its bytes. Both mocked-boundary
and real ZIP/tag-check fixtures reject a verified external copy with a replaced upload ZIP.
The consumers themselves emit actual JVM properties; reports retain/hash the XML and reject
invented JSON runtime properties. Java 21 public-API smoke passed on macOS and Ubuntu ARM64,
and full native smoke passed with a measured Java 25 runtime.

Bundling pins explicit Java 25/21 installations, disables discovery/downloads/inherited
`fromEnv`, and records the exact command vector plus each selected JDK's vendor/full version/
home. Root verification passed with only those explicit installations enabled. Tool preflight
now checks Apple developer tooling before bundling and supports CLT-only metadata; fixtures
cover both the CLT selection and xcodebuild failure fallback.

The exact digest-scoped upstream MLX path is removed before personal-prefix scanning; an
actual `--forbid /Users/runner` check now passes and the fixture still rejects other runner
paths. MIT is pinned alongside the other three notice files; NOTICE's wheel digest is checked
directly against bootstrap. Direct unverified `releaseAll` fails at graph readiness before
signing/bundling. Extraction failure cleanup and ARM64 checks for both macOS report kinds are
covered. Reports explicitly label workflow/container identity as operator attestations.

The expanded 21-test release suite passed on macOS and Ubuntu ARM64. Both the direct helper
and shell runner reject the actual system Python 3.9 without a traceback; the runner preserves
its pre-cleanup sentinel. The current workflows pass actionlint, while linting the previous
workflow confirms its forbidden job-level context is detected.

The later 2.6.0 integration closes the external plugin prerequisite; physical minimum-host
and signed-candidate execution gates remain open. No jmlx publication or tag was created.
