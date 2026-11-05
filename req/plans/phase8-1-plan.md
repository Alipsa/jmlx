# Phase 8.1 — First pure-Java releases

- **Status:** proposed implementation plan, 2026-11-03; revised 2026-11-04; no release performed.
  WP0 readiness recorded 2026-11-04 (gate table below); every gate in the table passed on
  2026-11-04 (the last open row, the 7.6 self-hosted qwen35 Tier-B run, passed in run
  41841745005). WP1–WP3 tooling merged as PR #80 (merge `dc3a61b`, 2026-11-04) under the
  owner authorization recorded below; the WP4 candidate cut and the WP5 publication remain
  outstanding.
- **Parent:** [Full roadmap §Phase 8](../full-roadmap.md#phase-8--first-releases-and-framework-adapters),
  milestone 8.1, and its release/versioning policy.
- **Prerequisite:** Phase 7 done under the [Phase 7 plan](phase7-plan.md): all milestones
  delivered or explicitly deferred with dates, and every pending acceptance gate passed or
  covered by a dated, recorded waiver. WP0 inventories the outstanding 7.1, 7.2, 7.3a, 7.3b,
  7.3c and 7.6 gates below and reconciles the incomplete gate lists in the parent documents.
  Milestone 7.6 was delivered in PR #79 (merge `e472c65`, 2026-11-03); merging it alone does
  not establish Phase 7 completion. Writing this plan does not start implementation or
  publication. The existing sequential gate remains in force for WP1–WP3 as well as release
  work; allowing tooling to start earlier requires an explicit, dated owner decision.

## 1. Goal, scope and exit gate

Release these independent coordinates to Maven Central and prove that a Java 21 application
can resolve and use them without MLX, FFM, native bootstrap, Python or Node:

| Coordinate | Release | Next development version | Published project dependency |
| --- | --- | --- | --- |
| `se.alipsa:jmlx-jinja` | `0.6.0` | `0.6.1-SNAPSHOT` | none |
| `se.alipsa:jmlx-tokenizer` | `0.1.0` | `0.1.1-SNAPSHOT` | `se.alipsa:jmlx-jinja:0.6.0` |
| `se.alipsa:jmlx-vision` | `0.1.0` | `0.1.1-SNAPSHOT` | none |

The proposed next versions are patch development lines; reassess before a later incompatible
change. Keep each module's own version declaration. The root and four native-chain modules
retain their current versions.

**Reuse at 8.4:** restore unchanged pure-Java modules to their published 8.1 versions before
POM generation and candidate verification; never republish those coordinates. Compare source,
resources, publication configuration and published dependency graphs against the 8.1 release
commit, ignoring only version/changelog bookkeeping. Changed modules receive new versions in
dependency order; a changed jinja dependency also requires a new tokenizer release. Keep the
resulting released tuple checked out through publication and Central verification.

The [roadmap's 8.4 amendment](../full-roadmap.md#phase-8--first-releases-and-framework-adapters)
requires candidate verification of reused coordinates exclusively from Central and new ones
exclusively from the smoke repository. The 8.4 sub-plan must design repository filters, a shared
source manifest and negative tests, explicitly covering shadowing copies in `~/.m2` as well as
the smoke repository and caches. Actual reused Central artifacts must pass full candidate
verification before any 8.4 publication. Carry this requirement and the reused/new version tuple
in the 8.1 runbook; implementation of mixed-source resolution belongs to 8.4.

The implementation deliverable is a partial-module release smoke plus a concrete release
runbook and release notes. The milestone exit gate additionally requires the three actual
publications, a successful fresh-cache Central smoke, and the subsequent development bumps.
Successful local publication alone is not completion.

Non-goals: native publication, MLX upgrades, binding regeneration, new tokenizer or processor
features, adapter modules (8.3), inference benchmarking (8.2), or changing release credentials,
signing policy or the byte-identical release scripts. No `se.alipsa:hfjinja` relocation POM is
published: callers must migrate the coordinate, packages and JPMS name explicitly. Preserve
the existing full native smoke.

## 2. Baseline findings and contract

Repository inspection establishes the following; these are code findings, not executed tests:

- `tools/release-smoke/run.sh` always publishes seven modules for `ci`/`candidate`, then runs
  the Java 25 `app` consumer. `central` skips local publication but uses the same full graph.
- `app/build.gradle` compiles against models, adds the native jar at runtime, resolves seven
  sources/Javadoc artifacts and hard-codes seven runtime jars. Its entry point checks native
  extraction and seeded batch inference. `pom-check` resolves the same graph through POMs.
- `repositories.gradle` already isolates `se.alipsa` to one repository, excludes that group
  from the fallback Central repository and strips snapshot suffixes in release mode. The
  runner uses a fresh consumer `GRADLE_USER_HOME`; these protections must remain. It does
  not clear the consumer's project-local `.gradle/` or subproject `build/` directories.
  Only `pom-check:writeRuntimeCoordinates` currently declares inputs; the app's
  `verifyPomMatchesModuleMetadata` reads the other project's build output.
- `app:verifyResolution` and `pom-check:writeRuntimeCoordinates` currently follow only
  `ResolvedDependencyResult` and silently skip `UnresolvedDependencyResult`. Missing
  components are caught indirectly by graph comparison; explicit failure is required.
- Jinja, tokenizer and vision target Java 21. Tokenizer and vision already enforce bytecode
  and published dependencies. Jinja has its own published-contract/release verification.
  Tokenizer exposes jinja through `api project(':jmlx-jinja')`; vision has no project dependency.
- Vision requires its pinned Pillow/PyTorch NOTICE in main, sources and Javadoc jars. The full
  smoke already checks the required source references, hashes and license texts.
- Every module's `release.sh` checks its independent non-snapshot version and runs build/release;
  root wiring gates snapshot dependencies, signing/credentials and checks. Jinja's heavier
  `releaseVerification` is deliberately separate and requires a clean release commit.
- CI has a Linux pure-Java job and a macOS native job; only the native job runs release smoke.

**Public Java API:** no additions or changes proposed. Consumers exercise existing
`Template.parse(...).render(...)`, `HfTokenizer.fromDirectory(...)`, encode/decode and chat
rendering, and `ImageDecoder`/`ImageTransforms` APIs. The jinja 0.6.0 release notes must still
enumerate changes since hfjinja 0.5.0, including coordinate, package and JPMS-name migration.

**Native declarations and probes:** none. No mlx-c symbol is used or pulled forward, and no
native ownership, callback, dtype, stream or lazy-evaluation contract changes. No inventory
regeneration is needed. Existing repository inventory checks continue to pass.

## 3. Smoke interface and isolation

Extend the existing runner with a named module-set profile:

```sh
./tools/release-smoke/run.sh ci --profile pure-java
./tools/release-smoke/run.sh candidate --profile pure-java
./tools/release-smoke/run.sh central --profile pure-java
```

`--profile all` remains the default, preserving the existing commands. Define profiles once
in the consumer configuration and ensure the runner's selected publishing tasks match them:

| Profile | Entry dependencies | Expected complete `se.alipsa` runtime graph | Runtime |
| --- | --- | --- | --- |
| `all` | models + runtime-only native jar | existing seven coordinates | Java 25, native |
| `pure-java` | tokenizer + vision | jinja, tokenizer, vision only | Java 21, headless |

These two explicit profiles satisfy the partial-module requirement. Arbitrary subsets,
including incomplete native combinations, are deferred. In `pure-java`, jinja must arrive
transitively through tokenizer rather than being added as a direct runtime dependency;
this proves the published dependency edge.

1. Parse all arguments before Gradle invocation, repository deletion or publication. Reject
   unknown modes/profiles, duplicate/conflicting arguments and unused trailing arguments.
   Keep `--record` for `ci --profile all` only; reject it for every pure-Java run and for
   candidate/Central before any side effect.
2. Pass a profile property to the independent consumer, and select only that profile's
   expected versions, artifacts and tasks. Existing environment version overrides remain
   supported. An unselected native version must not need to be a release or exist on Central.
   Any PR that changes a pure-Java producer version (release, recovery or development bump)
   must update the matching `tools/release-smoke/gradle.properties` default in the same change.
   This keeps both CI consumers' expected versions aligned with the locally published producers.
3. In candidate mode, preflight the actual selected producer versions and dependency versions
   before publishing anything: reject snapshots and mismatched effective consumer overrides.
   Removing `-SNAPSHOT` from a consumer default must never disguise snapshot producer output.
   Commit the released pure-Java tuple in `tools/release-smoke/released-8.1.properties`
   with the release PR. Pure candidate preflight must require all three file entries to
   equal the producer's own version, which must itself be a release version (a snapshot
   producer fails the preflight before any side effect), and must check consumer overrides
   against those file values. Central mode reads that tuple for `pure-java`, including
   after development bumps. Overrides may only supply selected modules absent from the file
   (for example, native modules in the full profile) or repeat a file value exactly; a
   disagreement is an error. An intentional tuple change requires a reviewed file update,
   including during recovery. Pure mode always requires the three committed entries.
   The tuple file is tightly formatted: `key=value` lines with no whitespace in the value;
   the shell preflight and the Gradle reader both reject any whitespace in a value, so the
   two parsers agree on what a valid entry is. Its line endings are pinned to LF in
   `.gitattributes`, and both readers reject a file containing carriage returns, so a
   `core.autocrlf=true` checkout cannot rewrite the file to CRLF and split the two parsers.
   Central has no fallback to checkout version defaults or stripped snapshot versions.
   Report the file values,
   overrides, effective versions and their sources. Fail clearly for any selected module
   without a released version. Reject snapshot overrides and any resolved snapshot in
   either release-mode graph.
4. In 8.1, `ci` and `candidate` publish selected modules to the disposable smoke repository only.
   `central` never publishes locally and resolves only Central. Never add `mavenLocal`,
   composite builds, project dependencies or checkout classes to the consumer classpath.
5. Keep the fresh consumer Gradle home, disabled build cache and exclusive repository filter.
   In candidate mode run the preflight immediately after successful argument parsing, before
   any other side effect. Then, before any Gradle invocation, clear the consumer's project
   state with `rm -rf "$smoke/.gradle" "$smoke/build" "$smoke"/*/build`.
   Keep cleanup confined to these consumer paths. A fresh home alone leaves that state intact.
   Share profile/version selection between metadata and POM-only consumers. Also declare
   profile, versions, released-tuple file and repository as verification task inputs; input
   tracking is a second defence against stale coordinate reports and artifact results.
6. Select projects in `settings.gradle` so the pure profile neither configures nor compiles the
   native `app`. Add a dedicated Java 21 `pure-java` consumer, retaining the full Java 25 app.
   Give POM-only resolution the same selected dependencies and a compatible toolchain.
   Have `run.sh` invoke exactly `:app:smoke` or `:pure-java:smoke` by full project path;
   do not add a root task named `smoke` or invoke the unqualified task selector.
7. Pure consumer execution uses `-Djava.awt.headless=true`, no native-access flag, no native
   cache/model assembly task and no native bootstrap. The checkout's Gradle build may still
   require its existing build JDK; Java 21 compatibility is proved by the consumer JVM and
   bytecode checks, not by asserting that Gradle itself must launch on Java 21.
   Update the runner's header and usage text to distinguish the cross-platform pure profile
   from the macOS ARM64/Java 25 full profile. The Linux Java job supplies both JDKs through
   inherited `GRADLE_OPTS`, including `org.gradle.java.installations.fromEnv` for
   `JAVA_HOME_21_X64,JAVA_HOME_25_X64`, with auto-download disabled. Preserve this inheritance
   in the isolated consumer invocation and document equivalent local JDK configuration in
   the README. Gradle still auto-detects JDKs in standard OS locations with a fresh home.
   A fresh home loses previously downloaded JDKs under the old home's `jdks` directory and
   any `org.gradle.java.installations.*` settings in its `gradle.properties`. Local runs may
   use OS auto-detection or the configured Foojay resolver to download a missing toolchain.
   GitHub's setup-java installations lie outside the auto-detected locations; that job uses
   the explicit `fromEnv` configuration above with downloads disabled.

## 4. Consumer assertions and fixtures

Factor profile-independent graph/artifact checks into smoke-local Gradle helpers, without
adding a production module or a dependency on the repository's `buildSrc`. Use resolved
component identities rather than filename counts alone.

For both profiles, verify exact selected versions, no unexpected `se.alipsa` components,
module artifacts rather than project/file substitutes, and identical complete runtime
coordinate sets under Gradle metadata and POM-only resolution. In release mode reject all
snapshot dependencies, including transitive third-party ones. Fail on unresolved components.
Resolve main, sources and Javadoc jars for each selected module, and assert required licenses.
Reuse the existing vision NOTICE checks for all three vision jars. Inspect the resolved main
jars: jinja's module descriptor must name `se.alipsa.jmlx.jinja`; tokenizer and vision must
carry `Automatic-Module-Name` values `se.alipsa.jmlx.tokenizer` and `se.alipsa.jmlx.vision`.
Check the published artifacts rather than the project jars. Native payload checks
apply when the native module is selected and remain mandatory for the full profile.

The pure consumer also checks loaded classes' code sources against the resolved jars and
rejects native-chain modules in its graph. Execute the following exact assertions, using
explicit checks that cannot be disabled by omitting `-ea`:

- **Jinja:** render a small message loop with a known literal expected output, including a
  non-ASCII character. This exercises jinja loaded transitively through the tokenizer edge.
- **Tokenizer:** load a small committed local tokenizer directory; assert hand-specified IDs,
  decode round-trip, exact rendered chat prompt and exact chat IDs. Exercise incremental
  decoding where a Unicode character spans byte tokens; combined output equals full decode.
  Use a fixed clock if the template observes time.
- **Vision:** reuse one resize case: select its PNG from
  `tools/image-oracle/fixtures/resize/`, its dimensions and resampling method from
  `tools/image-oracle/fixtures/resize.input.json`, and exact expected RGB bytes from
  `tools/image-oracle/fixtures/resize.expected.json`. The input-spec provenance key is
  `resize.input.json` in `tools/image-oracle/provenance.json`.
  Stage only the selected data as smoke fixtures;
  record the case name, original file SHA-256s, extracted-data digests and Pillow provenance
  in the smoke fixture document. The oracle provenance pins the input spec and PNGs, but
  does not pin `resize.expected.json`: record that file's own SHA-256 explicitly in the
  smoke provenance and verify it along with the extracted expected bytes. Its reference
  authority is the pinned Python `verifyImageOracleFixtures` run, not root `check`.
  A reviewed `generateImageOracleFixtures` run that changes this expected file requires
  updating the smoke copy and recorded digest in the same change, then rerunning both
  oracle verification and pure smoke. Map Pillow integer codes explicitly:
  `1` → `Resampling.LANCZOS`, `2` → `Resampling.BILINEAR`, `3` → `Resampling.BICUBIC`;
  reject other codes rather than treating integers as Java enum ordinals. These
  Pillow-recorded bytes are the resize reference, not hand-computed or Java-generated values.
  Separately normalize a simple input with analytically known float values. Assert decoded
  pixels, output shape/layout and resized values as well as dimensions. Keep ImageIO headless.

Add a `tools/release-smoke/fixtures/pure-java/` fixture set and provenance document. Prefer a
minimal hand-authored tokenizer plus the selected image-oracle case above, with extraction
instructions and digests; no Hub credentials, model downloads or remote code are required.
Existing fixtures may be reused only when their IDs/semantics are explicitly documented.
Do not copy an entire
checkpoint or use outputs generated by the candidate Java implementation as the reference.
These are hand-computed and pinned-oracle packaging/integration fixtures, not new Tier-B model
acceptance or a replacement for the existing tokenizer and image differential oracles.

## 5. Ordered work packages

### WP0 — Confirm readiness and record the release inputs

Reconcile this inventory against current run records; an implementation merge or local run
cannot close a hosted gate. For each row record a passing run/report URL and date, or an
owner-approved dated waiver with its scope and rationale. No waiver is granted by this plan.

| Pending gate | Source document | Required evidence | Pass or waiver |
| --- | --- | --- | --- |
| 7.1 legacy oracle drift | `phase7-1-plan.md` §7; `phase7-plan.md` §7.1 | Pinned Phase 6 oracle verification with unchanged serialized output/hashes, or a separately reviewed drift resolution | Pass. CI run [41724687057](https://github.com/Alipsa/jmlx/actions/runs/41724687057) (main at the PR #79 merge `e472c65`, 2026-11-03, green; hosted macOS 26 `native` job) — "Run root verification" executed `verifyMlxOracle verifyMlxOracleFixtures -PmlxOracleRequireAllProfiles=true` with the pinned oracle's serialized outputs unchanged |
| 7.1 cross-host CPU fixtures | `phase7-1-plan.md` §7; `full-roadmap.md` Phase 7 done | Canonical Phase 7.1 CPU fixture verification on development and CI hosts, including the accepted serialization policy | Pass. CI host: same run 41724687057 (hosted macOS 26). Development host: local `./gradlew verifyMlxOracle verifyMlxOracleFixtures` (phase7-1 profile, M5 Max, macOS 27) on 2026-11-04, green |
| 7.2 macOS 26 CI | `phase7-2-implementation-report.md`; `phase7-plan.md` §7.2 | Required native suites executed on macOS 26 | Pass. Run 41724687057 `native` job (hosted macOS 26): "Run native module checks" and "Assert required native suites executed" both green for the required 7.2 suites |
| 7.3a macOS 26 hosted acceptance | `phase7-3a-implementation-report.md`; `phase7-plan.md` §7.3a | Hosted WP6/WP7 suites executed on macOS 26 (Tier-B arm execution out of scope per the 2026-11-04 owner amendment) | Pass: run 41724687057 executed the required SmolVLM suites on hosted macOS 26. Tier-B arm: not executed and not executable on the free hosted runners by design (the smolvlm arm targets the self-hosted `self-macos-xl`; the float32 test footprint exceeds the 7 GB hosted tier and the larger hosted tier bills per minute) — recorded design decision in `tier-b.yml`, not a waiver. First self-hosted arm runs: 40403387275 (2026-10-26, branch `phase7-3b-tierb-self-hosted-runner`, green) and merged-head confirmation 40409087523 (`tools/tier-b/README.md`); re-confirmed in run 41352378744 (2026-11-01) and run 41841745005 (2026-11-04) |
| 7.3b macOS 26 hosted Tier-B | `phase7-plan.md` §7.3b; `AGENTS.md` Phase 7.3b additions | Required hosted macOS 26 suite execution (Tier-B arm execution out of scope per the 2026-11-04 owner amendment) | Pass: run 41724687057 executed the required Qwen3-VL suites on hosted macOS 26. Arm execution: self-hosted by design (11.16 GiB measured footprint peak); first self-hosted runs 40403387275 (2026-10-26) and merged-head confirmation 40409087523; hosted arm execution remains not executed and not feasible on the free hosted runners (recorded, not waived) |
| 7.3c macOS 26 hosted execution | `phase7-plan.md` §7.3c; `AGENTS.md` Phase 7.3c additions | Required macOS 26 hosted suite execution; local/self-hosted evidence does not substitute (Tier-B arm execution out of scope per the 2026-11-04 owner amendment) | Pass: run 41724687057 executed the required Phi-3.5-Vision suites on hosted macOS 26. The phi3v Tier-B arm is self-hosted by design (51.4 GB measured footprint peak); its first self-hosted run is the separate row below |
| 7.3c first self-hosted Tier-B run | `phase7-plan.md` §7.3c; `AGENTS.md` Phase 7.3c additions | First successful phi3v workflow run; pipeline acceptance only, not independent host/OS evidence | Pass. Run [41352378744](https://github.com/Alipsa/jmlx/actions/runs/41352378744) (workflow_dispatch, 2026-11-01, self-hosted `self-macos-xl` runner): the phi3v arm (revision `12b77fb`, task `phi3v-generation`, `TierBPhi3VTest`) succeeded. Pipeline acceptance only |
| 7.6 macOS 26 hosted CI | `full-roadmap.md` Phase 7 milestone table, 7.6 row | Required hosted native suites executed on macOS 26 | Pass. Run 41724687057 `native` job executed the required Qwen35 hybrid suites on hosted macOS 26 ("Assert required native suites executed" green) |
| 7.6 first self-hosted qwen35 Tier-B run | `full-roadmap.md` Phase 7 milestone table, 7.6 row | First successful qwen35 workflow arm | Pass. Run [41841745005](https://github.com/Alipsa/jmlx/actions/runs/41841745005) (workflow_dispatch, 2026-11-04, head `dc3a61b`, self-hosted `self-macos-xl` runner): the qwen35 arm (revision `2fc06364`, task `qwen35-generation`, `TierBQwen35Test`) succeeded in 7m23s — `max_abs_error=4.0340423583984375e-4` (ε = 0.001 / margin = 1e-4), native peak 5,118,123,572 B, exact greedy IDs and decoded output, `device=gpu` — bit-for-bit the manifest's local evidence. Pipeline acceptance only |

- Check the full Phase 7 exit gate and record links to passed runs or dated waivers. Update the
  stale 7.6 delivery-table row and §7.6 narrative in `phase7-plan.md` to PR #79 / `e472c65`,
  matching the roadmap while retaining any still-pending acceptance gates. Reconcile the
  7.6 sub-plan header and roadmap overview at the same time; do not infer hosted acceptance
  from the merge or local evidence. Reconcile the parent plan's gate list with this inventory,
  including 7.3c and 7.6; retain all unresolved acceptance requirements.
- Confirm the three target versions, root publishing guards, release-script identity and
  jinja's exact Node lock/checklist. Inventory the hfjinja-to-jinja migration changes.
- Record intended release notes/runbook locations, required reports and the version tuple.
  Inspect generated POMs, licenses and NOTICE before changing publication behavior.

**Gate:** readiness is explicit; missing acceptance evidence blocks implementation/publication
under the sequential roadmap rule, including tooling work in WP1–WP3. This plan itself may
be reviewed beforehand. Only an explicit, dated owner amendment may allow earlier tooling;
record that decision and its scope before starting, without implying publication readiness.

Recorded owner decision, 2026-11-04 (scope: WP1–WP3 tooling only): proceed with the profile
tooling, the Java 21 consumer and fixtures, the CI wiring and the release documents despite
the still-open 7.6 self-hosted qwen35 Tier-B row and the recorded not-executed hosted Tier-B
arm limitations above. This authorizes neither the WP4 candidate cut nor the WP5 publication;
the open rows remain release-cut blockers, and the hosted-arm limitations remain recorded
design decisions rather than waivers.

Recorded owner decision, 2026-11-04 (scope: Phase 7 done-gate interpretation): the done
list's "macOS 26 hosted runs for 7.2, 7.3a, 7.3b, 7.3c and 7.6" is the hosted macOS 26
suite execution — run 41724687057 (2026-11-03, hosted `native` job, green) executed the
required suites of all five milestones — read together with the done list's separate
first-self-hosted-run items where they exist. Hosted execution of the self-hosted Tier-B
arms (7.3a smolvlm, 7.3b qwen3vl, 7.3c phi3v, 7.6 qwen35) is out of scope for the done
gate: it is not executable on the free hosted runners by design (measured footprints vs
the 7 GB hosted tier) and remains a recorded design decision in `tier-b.yml` and the WP0
rows — not part of the gate's required evidence, neither a pass nor a waiver.

### WP1 — Implement profile selection and shared artifact verification

- Update `run.sh`, `settings.gradle`, `repositories.gradle`, both existing consumer build files
  and `gradle.properties`; add the smoke-local shared configuration as needed. Update the
  runner's platform/JDK header, usage text and fully qualified consumer task invocation.
- Implement parsing/preflight before destructive smoke-repository cleanup. Publish only selected
  producers, select expected graphs and resolve only selected documentation artifacts.
- Preserve default full-profile extraction, native payload, seeded inference and `[2]` cohort
  assertions; do not re-record the Mistral golden for this tooling change.

**Gate:** pure mode has exactly three jmlx runtime jars; the default still verifies all seven.

### WP2 — Add and exercise the Java 21 consumer

- Add `pure-java/build.gradle`, its smoke entry point and documented fixtures.
- Reuse artifact/NOTICE checks and implement the public-API assertions in §4.
- Add focused regression checks for profile parsing, version preflight, graph mismatch and
  task-input isolation, as described below. Avoid duplicating module-level unit suites.

**Gate:** `ci --profile pure-java` passes without staged native libraries; executed consumer
JVM is Java 21 and its dependency graph excludes ffi/core/models/native.

### WP3 — Wire CI and prepare the release documents

- Add `ci --profile pure-java` to `.github/workflows/ci.yml`'s Java job. Preserve the existing
  macOS full smoke and its native required-suite assertions. CI never publishes to Central.
  Verify that the consumer inherits the job's JDK discovery `GRADLE_OPTS` with both named
  JDK environment variables and auto-download disabled.
- Add `tools/release-smoke/README.md` documenting both profiles, modes, version overrides,
  producer-vs-consumer JDK requirements, isolated resolution, the released-tuple file and
  failure diagnosis. Update `AGENTS.md`'s Release smoke section and CI description for the
  two profiles and Linux pure smoke; retain the macOS full-smoke description.
- Add `req/phase8-1-release-checklist.md` with exact candidate/Central commands, dependency
  order, evidence locations, recovery steps and the §1 policy for restoring unchanged
  modules to released versions at 8.4, including its mandatory mixed-source candidate
  verification before publication. Preserve the dated roadmap 8.4 amendment linking this
  requirement to the future 8.4 sub-plan. Link to jinja's existing checklist rather than
  replacing its extra requirements.
- Prepare `req/phase8-1-release-notes.md` with provisional date and release identification;
  WP5's development-bump PR fills in the actual cut dates, per-module release SHAs, branches
  and tag mapping after publication. Do not try to embed a merge commit's own SHA in its
  committed notes. Explain the hfjinja migration, supported Java level, tokenizer/vision
  limitations, license/provenance and the three independent versions. State that no native
  runtime ships in this release and avoid adding inference-performance claims.

**Gate:** tooling PR is reviewable with pure and full smoke evidence. Changes go through a
feature branch and CI; merge only after explicit human review/approval, per `AGENTS.md`.

### WP4 — Cut and verify the release candidate

After the tooling PR is merged, prepare a release PR setting the three versions to 0.6.0,
0.1.0 and 0.1.0 and rolling jinja's Unreleased entries into its dated changelog. Update
smoke defaults/documented coordinates consistently, and commit the exact tuple in
`tools/release-smoke/released-8.1.properties`. Keep the native modules as snapshots.

Run selected module checks, the root guards and `buildSrc` checks when its logic changed.
From the clean release commit run the complete jinja checklist, including
`:jmlx-jinja:releaseVerification`, dependency-update review and report inspection; development
`-PreleaseVerificationAllowDirty` output is not release evidence. Then run:

```sh
./tools/release-smoke/run.sh candidate --profile pure-java
```

Review generated POMs, sources/Javadoc jars, licenses/NOTICE and the candidate report. Merge
the approved release PR with the required merge commit. Freeze merges to `main` from that
merge until WP5's development-bump PR merges; that PR must be the next merge after successful
Central verification. Before that success, only a reviewed recovery PR as defined in WP5 may
merge during the freeze. Keep the freeze during partial-publication recovery too. This prevents
unrelated changes from carrying versions already immutable on Central.
On clean `main` at the initial release merge commit,
unconditionally rerun jinja's complete release checklist verification (including
`releaseVerification`) and the pure candidate smoke, and inspect their reports. The merge
commit is the initial release SHA: absent recovery, all three `release.sh` runs must execute
from that same clean HEAD, with no intervening commits, checkout changes or version edits. Verify HEAD
before each run; if it changes, stop. Only WP5's recovery procedure may establish a different
release SHA for affected modules; do not resume publication from an unverified commit.
Record that exact SHA/branch in the release record; attach the final SHA to post-cut evidence
without editing the verified checkout. Publication is a later explicitly
authorized manual action; creating this plan authorizes neither publication nor PR merging.

### WP5 — Publish, verify Central and resume development

For the initial cut, execute each existing release script in this order at WP4's verified merge SHA,
retaining all three release versions through verification:

```sh
./jmlx-jinja/release.sh
./jmlx-tokenizer/release.sh
./jmlx-vision/release.sh
./tools/release-smoke/run.sh central --profile pure-java
```

Wait for jinja 0.6.0 to resolve from Central before releasing tokenizer so its dependency cannot
point to a failed deployment. Its generated POM names 0.6.0 regardless of propagation state.
The pinned nexus-release-plugin 2.2.0 submits uploads with `publishingType=AUTOMATIC`:
normal releases do not require clicking Release in the Central portal. Its `ReleaseTask`
accepts both `PUBLISHING` and `PUBLISHED` as success, so script success is not proof of
availability. This was checked in the cached 2.2.0 sources jar:
`CentralPortalClient.groovy:49` and `ReleaseTask.groovy:115–125`; preserve the source-jar
SHA-256 `6b2cbd42a469f024f1acca1e96e2b9e2c134224248e21db09f7490f4647314ed`
in the runbook evidence and recheck if the plugin changes. Record deployment IDs,
terminal portal state and actual Central availability for each module. Do not read
a successful upload as successful Central resolution. Use bounded retries for propagation
and preserve the failed resolution report if availability remains incomplete.

If publication partly succeeds, retain the released versions and resume only missing modules
after checking deployment state. Never overwrite an immutable released coordinate. A bad
published artifact needs a reviewed recovery PR with a patch version, updated
`released-8.1.properties` and release notes. This is the only PR other than the final
development-bump PR allowed to merge during the freeze. Include dependent modules whose
published dependency graph changes, assigning new versions to already-published coordinates.
Apply §1's comparison rule against the initial release SHA (or the recorded release SHA for
an already-patched module): a recovery PR must not change source, resources, publication
configuration or published dependency graphs of modules it does not re-version. Include the
effects of shared build conventions and dependency-version changes in that comparison; record
the review before merging the recovery PR.

Recovery candidate smoke locally rebuilds all three selected modules, including unchanged
coordinates already on Central. Those copies are not evidence for the immutable Central
artifacts; candidate success cannot establish the final mixed graph before publication.
The comparison guard limits this gap, and the post-publication fresh-cache Central smoke
remains the binding gate for the released tuple. Mixed-source candidate verification remains
8.4 work.

The recovery PR's merge commit defines a new release SHA for the affected modules. On that
clean SHA run the affected module checks, jinja's complete release checklist if jinja is
affected (including any jinja publication from this new SHA), and the pure candidate smoke;
inspect reports before publishing. Publish only
missing or replacement coordinates in dependency order, never successful coordinates again.
Record each module/version's own release SHA and branch, including superseded publications,
and rerun the complete fresh-cache Central smoke against the updated tuple. Keep the freeze
until recovery succeeds and the development-bump PR merges.
Do not bypass guards or start native publication to make the partial release pass.

After the complete fresh-cache Central smoke succeeds, record reports and published
coordinates. Create annotated per-module tags (`jmlx-jinja-v0.6.0`,
`jmlx-tokenizer-v0.1.0`, `jmlx-vision-v0.1.0` for the initial tuple) pointing to each module's
verified release SHA. After recovery, tag patched versions at their new verified SHAs.
Push tags as part of the authorized release procedure and record the tag-to-SHA mapping.
Never move an existing release tag; on partial publication tag only confirmed releases.
Then open a separate development-bump PR for the next development versions (advancing beyond
any recovery releases) and a fresh jinja Unreleased section. Reset the corresponding defaults
in `tools/release-smoke/gradle.properties` to those new development versions in that same PR,
so both Linux pure smoke and macOS full smoke compare against the versions the producers
publish in `ci` mode. Require both CI smokes to pass before merging and lifting the WP4 freeze.
That same PR must finalize
the release notes with per-module release SHAs/branches and tag mapping, update the roadmap's
8.1 status and `AGENTS.md`'s published-module statements, and add runbook evidence links and
the final released tuple. Distinguish the three released modules from the native chain.
Preserve the final released-tuple file unchanged during these development bumps so Central
smoke still selects the published versions.
The runbook must also carry the §1 8.4 restoration policy; snapshot bumps do not imply that
unchanged tokenizer/vision/jinja artifacts must be published again for models to release.

## 6. Verification matrix

| Check | Required evidence |
| --- | --- |
| Parser/preflight | Invalid mode/profile/arguments, forbidden recording and snapshot/mismatched candidate versions fail before smoke-repository cleanup or publishing; candidate rejects missing or producer-mismatched pure tuple entries; Central accepts overrides only for absent entries or exact file-value repeats, rejects conflicts and reports value sources; use disposable fixtures/stub commands for side-effect assertions |
| Pure CI smoke | Exactly three selected coordinates, Java 21 runtime through inherited CI JDK discovery with downloads disabled, public API outputs, jars/docs/licenses/NOTICE, metadata/POM agreement and fresh-cache resolution |
| Profile changes | Local Apple Silicon macOS run of pure/full in sequence with Java 21 and 25 available; clear project state each run and verify both graphs/artifact checks; this row is local evidence, not a new macOS CI pure-profile requirement |
| Central version selection | After snapshot bumps, pure Central mode still selects the committed 8.1 tuple; missing tuple entries and snapshot overrides fail clearly before resolution, with no snapshot-suffix fallback |
| Published JPMS names | Resolved jinja module descriptor and tokenizer/vision manifest names match §4 |
| Missing/wrong publication | Disposable repo missing one selected artifact or serving a wrong dependency version fails; no fallback to Central/local cache hides it |
| Vision fixture provenance | Input/PNG pins, explicit expected-file SHA-256 and extracted-byte digest match; resample codes use the documented mapping; reviewed oracle output changes update smoke data |
| Consumer isolation | No project/file substitution, no native graph or task execution for pure profile, headless image decode succeeds; fully qualified task executes the consumer once |
| Existing full smoke | All seven modules, fresh native extraction, unchanged seeded golden and scheduler cohort assertion still pass on Apple Silicon |
| Release candidate | Released producer versions and generated POMs; unconditional jinja verification and pure candidate rerun on clean main's merge SHA; initial release scripts use that same SHA; recovery records per-module SHAs and reruns affected checks, jinja verification if affected, and pure candidate smoke on each recovery SHA; native modules remain snapshots |
| Recovery comparison | Before merge, unchanged modules match their recorded release source/resources/publication configuration/dependency graphs, including shared build effects; local candidate rebuilds are not Central-artifact evidence; the final Central smoke is binding |
| Version changes | Release, recovery and development-bump PRs update pure-Java producer versions and matching smoke `gradle.properties` defaults together; Linux pure and macOS full CI smokes pass; release/recovery PRs update `released-8.1.properties`, while development bumps preserve the final released tuple |
| Central | Fresh Gradle home, selected released versions, no local jmlx source; pure API/artifact assertions pass after all three deployments |

Continue existing module checks and pinned oracle verification through CI. Python/Node remain
verification tools, never consumer prerequisites. Preserve committed oracle goldens/digests;
no re-generation is expected. No native leak/scope tests or performance benchmark is added:
there is no native implementation or inference-path change, and the full smoke retains its
existing lifecycle coverage. Cross-platform evidence comes from the Linux Java CI job and
the macOS native CI job; no Windows execution is required or claimed here. A local macOS
pure smoke alone does not establish cross-platform acceptance.

## 7. Completion and deferred work

The milestone is complete when all three release coordinates and their documentation artifacts
resolve from Central; the Java 21 pure smoke passes there and in CI; POM/metadata graphs,
licenses and vision NOTICE pass; each publication's release commit is recorded per module
with its required checks and pure candidate evidence, and jinja's full release checklist has
executed on each jinja publication's release commit; release notes/tags identify the per-module
cut SHAs/branches and migration breaks; and the development-bump PR carrying the final release
notes, tag mapping, roadmap/AGENTS status and runbook evidence links is merged. If recovery
was needed, these conditions apply to the final released tuple, with superseded publications
and their evidence retained in the release record.
The runbook explicitly requires 8.4 to restore unchanged modules to their published 8.1
versions before generating models' POM, reusing them without republishing; changed modules
and changed published dependency graphs receive new releases in dependency order.
The 8.4 handoff requires the mixed-source candidate extension in §1: actual reused Central
artifacts must pass full candidate verification with new smoke-repo artifacts before any
8.4 publication. Rebuilding reused coordinates locally alone cannot satisfy that gate.
Archive candidate, Central and jinja verification evidence with the release record without
committing credentials or local caches.

Deferred: arbitrary module subsets, the native/adapters release (8.4), performance baseline
(8.2), API expansion and any change to native pins. The smoke candidate preflight's dedicated
snapshot message: in the `all`-profile branch of `preflight_candidate` the version comparison
runs against the already-stripped expected value before the `*-SNAPSHOT` case, so a snapshot
producer is always rejected by the version-mismatch message and the case line cannot fire
(move the snapshot check above the comparison and have the selftest assert the message).

Bundle naming (implemented in the development-bump PR): the 8.1 production bundle was
submitted under the plugin's default name `bundle.zip`. The combined flow's bundle file is now
explicitly `jmlx-mega-bundle.zip` (the root `combinedRelease` block sets the plugin's
`bundleFile` property, a convention in nexus-release-plugin 2.5.1), because the Portal carries
the bundle's file name as the deployment name. Future combined releases upload under the new
name; the 8.1 record keeps the historical `bundle.zip` name.
Cross-platform claims stay limited to
executed CI evidence; this milestone adds no general MLX-parity claim.
