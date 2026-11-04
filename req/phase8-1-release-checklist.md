# Phase 8.1 release runbook: jinja 0.6.1, tokenizer 0.1.1, vision 0.1.1

This is the exact procedure for cutting, verifying, publishing and closing out the first
pure-Java releases. It is the runbook the [Phase 8.1 plan](plans/phase8-1-plan.md) WP4/WP5
refer to; it does not itself authorize publication. Publication is a later explicitly
authorized manual action.

## Version tuple

| Module | Release | Next development | Notes |
| --- | --- | --- | --- |
| `se.alipsa:jmlx-jinja` | `0.6.1` | `0.6.2-SNAPSHOT` | hfjinja migration line; no `se.alipsa:hfjinja` relocation POM |
| `se.alipsa:jmlx-tokenizer` | `0.1.1` | `0.1.2-SNAPSHOT` | published dependency `se.alipsa:jmlx-jinja:0.6.1` |
| `se.alipsa:jmlx-vision` | `0.1.1` | `0.1.2-SNAPSHOT` | no project dependency |

The initial `0.6.0` / `0.1.0` / `0.1.0` cut (release SHA `6097d8f`, published 2026-11-04) is
superseded: its bundles omitted the Gradle module metadata file (`.module`) because the
then-pinned nexus-release-plugin 2.2.0 bundles only JAR and POM files. The 2026-11-04 recovery
republished the three coordinates as `0.6.1` / `0.1.1` / `0.1.1` with the plugin pin at 2.4.1,
in **one combined `releaseAll` deployment** (see Publication below). The `0.6.0` / `0.1.0` /
`0.1.0` coordinates remain on Central as dead artifacts; never publish to them again and never
point a consumer at them.

The four native-chain modules (`jmlx-ffi`, `jmlx-core`, `jmlx-models`,
`jmlx-native-macos-arm64`) and the root keep their snapshot versions throughout; they are never
published in this milestone.

## Publication (one combined deployment)

`jmlx-tokenizer` depends on `se.alipsa:jmlx-jinja:0.6.1` via its published POM. In the combined
flow that edge is satisfied inside one bundle: the planner's fixed point selects tokenizer only
because jinja is selected in the same deployment, and Central resolves the edge once both
coordinates are live. The four native-chain SNAPSHOT modules are candidates in every plan and are
excluded — reported with their SNAPSHOT reasons — in every run.

Publication is one run of `./releaseAll.sh` from the verified release SHA on a clean tree
(`tools/release-all/README.md` documents the full task sequence):

1. `./releaseAll.sh --dry-run` — plan and report (expect 3 selected / 4 excluded), signing
   preflight in warning mode.
2. jinja's complete checklist from the same clean SHA (this is a separate manual gate:
   `releaseAll.sh`'s own per-module `check` does not include `:jmlx-jinja:releaseVerification`),
   plus the pure candidate smoke.
3. `./releaseAll.sh` — verification build, staging, bundle validation and the single upload.
   The `releaseAll` task polls to `PUBLISHED` plus repo1 HEAD 200 for every selected
   component's POM and main JAR before logging "published and available"; that is **not** the
   binding gate.
4. `./tools/release-smoke/run.sh central --profile pure-java` — the binding fresh-cache gate.

A script's success is not proof of availability: the pinned nexus-release-plugin 2.4.1 submits
`publishingType=AUTOMATIC` and accepts `PUBLISHING`/`PUBLISHED` alike at upload time — checked in
the 2.4.0 sources, `CentralPortalClient.groovy:64`, `ReleaseTask.groovy:133–143` (source-jar
SHA-256 `edd4e45ccb56ca667a6d05b8318d21b0434adb8b0501e9befb66d33f0d8b25f6`), and unchanged in
2.4.1, which adds only the optional static BOM/aggregate POM inputs (plugin repo commit
`63e46b9`, PR #10; recheck if the plugin changes). Since 2.3.0 the bundle is additionally
validated before upload and a POM that advertises Gradle metadata without a signed `.module`
entry is rejected (`BundleValidator.groovy:111`), so the 2.2.0 `.module`-omission shape cannot
recur silently.

Record: the single deployment ID, terminal Central portal state, actual Central availability, the
release SHA and branch, the bundle SHA-256 from `build/releaseAll/bundle-receipt.json`, and the
final tag mapping.

## WP4 — cut and verify the release candidate (recovery amendment)

The initial cut (`0.6.0` / `0.1.0` / `0.1.0`, release SHA `6097d8f`, published 2026-11-04) is
recorded in `req/phase8-1-release-notes.md`. The recovery release PR — the only release PR of
this milestone after the superseded cut — contains, in one commit:

- `jmlx-jinja/build.gradle` at `0.6.1`, `jmlx-tokenizer/build.gradle` and
  `jmlx-vision/build.gradle` at `0.1.1`, with changelog sections recording the supersession
  (jinja, tokenizer; vision has no changelog);
- the nexus-release-plugin pin at `2.4.1` (`gradle/libs.versions.toml`);
- the combined-release tooling: `releaseAll.sh`, `tools/release-all/`, the `settings.gradle`
  plugin-resolution gate, the root `build.gradle` `releaseAllMode` gate, and the committed
  sanitized `req/releaseAll-portal-fixture.json`;
- the three pure-Java `tools/release-smoke/gradle.properties` defaults and
  `released-8.1.properties` at `0.6.1` / `0.1.1` / `0.1.1`.

Verification before publication (all from the clean recovery merge commit):

1. The selected module checks and the root guards (`./gradlew check`); the combined flow's own
   verification build (Publication step 3) re-runs the module checks at publication time.
2. jinja's complete checklist — `jmlx-jinja/req/release-checklist.md`, including
   `:jmlx-jinja:releaseVerification`, the dependency-update review and report inspection. This
   is a separate manual gate: `releaseAll.sh`'s per-module `check` does not include
   `releaseVerification`. Development `-PreleaseVerificationAllowDirty` output is not release
   evidence.
3. The pure candidate smoke from the clean recovery merge commit:
   ```sh
   ./tools/release-smoke/run.sh candidate --profile pure-java
   ```
   The preflight must report all three tuple values against matching release producers. Review
   the generated POMs (tokenizer's POM must name `se.alipsa:jmlx-jinja:0.6.1`), the
   sources/Javadoc jars, licenses/NOTICE and the candidate report.
4. Merge the approved recovery PR with a required merge commit. The WP4 freeze rules from the
   original cut remain in force: only a reviewed recovery PR or the final development-bump PR
   may merge to `main` until the freeze lifts.
5. At the recovery merge commit (the release SHA for all three modules: the combined flow
   publishes all three from one HEAD with no per-module version edits — verify
   `git rev-parse HEAD` before the run and stop if it changes), unconditionally rerun jinja's
   complete checklist (including `releaseVerification`) and the pure candidate smoke, and
   inspect their reports.

## WP5 — publish, verify Central, resume development

1. Execute the Publication section above at the verified release SHA, retaining all three release
   versions through verification.
2. If the run fails **before** the upload is submitted: fix the cause and rerun `./releaseAll.sh`
   from the same SHA (the source guard makes a stale manifest impossible). If it fails **after**
   submission: `./releaseAll.sh --resume <deployment-id>` (the ID is journaled in
   `build/releaseAll/deployment.json` and visible on the Portal). **Never overwrite an immutable
   released coordinate** and never re-upload a bundle whose coordinates are already live.
3. Rerun `./tools/release-smoke/run.sh central --profile pure-java` (fresh cache). Preserve the
   failed-resolution report if availability remains incomplete after bounded retries; a
   successful upload is not a successful Central resolution.
4. Create annotated tags pointing at the verified release SHA:
   `jmlx-jinja-v0.6.1`, `jmlx-tokenizer-v0.1.1`, `jmlx-vision-v0.1.1`; push them as part of the
   authorized release procedure and record the tag-to-SHA mapping. Never move an existing release
   tag; on partial publication tag only confirmed releases.
5. Open the development-bump PR (the next merge after successful Central verification):
   advance the three modules beyond any recovery releases to `0.6.2-SNAPSHOT` / `0.1.2-SNAPSHOT`
   / `0.1.2-SNAPSHOT`, add a fresh jinja Unreleased section, reset the
   `tools/release-smoke/gradle.properties` defaults to those new development versions, finalize
   `req/phase8-1-release-notes.md` with the per-module release SHAs/branches and tag mapping,
   update `req/full-roadmap.md`'s 8.1 status and `AGENTS.md`'s published-module statements, and add
   the runbook evidence links and final released tuple. **Keep
   `released-8.1.properties` unchanged** so the Central smoke still selects the published tuple.
   Require both CI smokes (Linux pure, macOS full) to pass before merging and lifting the WP4
   freeze.

## Recovery (bad published artifact)

A recovery is a reviewed PR, the only PR other than the final development-bump PR allowed to
merge during the freeze:

- assign a new patch version to the affected coordinate(s), including dependent modules whose
  published dependency graph changes (already-published coordinates get new versions);
- update `released-8.1.properties` and `req/phase8-1-release-notes.md` in the same PR;
- comparison guard: unchanged modules must match their recorded release source, resources,
  publication configuration and published dependency graphs against the initial release SHA (or
  the recorded release SHA for an already-patched module), including the effects of shared build
  conventions and dependency-version changes; record the review before merging.
- on the recovery merge commit (the new release SHA for the affected modules), run the affected
  module checks, jinja's complete checklist if jinja is affected, and the pure candidate smoke;
  inspect reports before publishing.
- The recovery candidate smoke locally rebuilds all three selected modules, including unchanged
  coordinates already on Central. Those copies are **not** evidence for the immutable Central
  artifacts; candidate success cannot establish the final mixed graph before publication. The
  combined plan itself never re-selects a coordinate that is already published (fail-closed
  availability), so a recovery `./releaseAll.sh` uploads exactly the new versions in one
  deployment; never hand-publish a coordinate the plan excluded, and never publish a
  superseded coordinate again. Then rerun the complete fresh-cache Central smoke against the
  updated tuple. Keep the freeze until recovery succeeds and the development-bump PR merges.
- Do not bypass guards or start native publication to make a partial release pass.

## Restoring unchanged modules at 8.4 (plan §1 policy)

Before 8.4 generates its POMs, restore the unchanged pure-Java modules to their published 8.1
versions and **reuse them without republishing**: compare source, resources, publication
configuration and published dependency graphs against the 8.1 release commit, ignoring only
version/changelog bookkeeping. Changed modules receive new versions in dependency order (a
changed jinja dependency also requires a new tokenizer release), and the resulting released tuple
stays checked out through publication and Central verification. Snapshot bumps of the native chain
do not imply that unchanged tokenizer/vision/jinja artifacts must be published again for models to
release.

The [roadmap's dated 8.4 amendment](../full-roadmap.md#phase-8--first-releases-and-framework-adapters)
requires candidate verification of reused coordinates **exclusively from Central** and new ones
**exclusively from the smoke repository**, with a shared source manifest and repository filters
whose negative tests explicitly cover `~/.m2` shadowing copies as well as the smoke repository and
caches. Actual reused Central artifacts must pass full candidate verification **before any 8.4
publication**. Designing that mixed-source resolution belongs to the 8.4 sub-plan; this runbook
carries the requirement and the reused/new version tuple to it.

## Evidence locations

- Candidate: the run's log, `build/smoke-repo/` contents are disposable — preserve the report in
  the release record (do not commit caches or credentials).
- Combined release: `build/releaseAll/` (`manifest.json`, `report.txt`, `bundle.zip`,
  `bundle-receipt.json` with the bundle SHA-256, `deployment.json` with the deployment ID) and the
  `releaseAll` task's "published and available" log line; the `report.txt` selected/excluded
  listing is the per-run eligibility evidence.
- Central: the fresh-cache `run.sh central --profile pure-java` log, the deployment ID, terminal
  portal state and Central resolution evidence.
- jinja verification: `jmlx-jinja`'s `releaseVerification` outputs on the release commit.
- Tags: the tag-to-SHA mapping recorded in `req/phase8-1-release-notes.md`.
