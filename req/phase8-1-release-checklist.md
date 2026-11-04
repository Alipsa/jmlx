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
republished the three coordinates as `0.6.1` / `0.1.1` / `0.1.1` with the plugin pin at 2.4.0.
The `0.6.0` / `0.1.0` / `0.1.0` coordinates remain on Central as dead artifacts; never publish to
them again and never point a consumer at them.

The four native-chain modules (`jmlx-ffi`, `jmlx-core`, `jmlx-models`,
`jmlx-native-macos-arm64`) and the root keep their snapshot versions throughout; they are never
published in this milestone.

## Dependency order and publication order

`jmlx-tokenizer` depends on `se.alipsa:jmlx-jinja:0.6.1` via its published POM, so jinja must be
**resolved from Central** before tokenizer is released. `jmlx-vision` is independent. Publish in
this order, waiting for Central availability between steps (a script's success is not proof of
availability: the pinned nexus-release-plugin 2.4.0 submits `publishingType=AUTOMATIC` and its
`ReleaseTask` accepts `PUBLISHING` and `PUBLISHED` alike — checked in the 2.4.0 sources,
`CentralPortalClient.groovy:64`, `ReleaseTask.groovy:133–143`, source-jar SHA-256
`edd4e45ccb56ca667a6d05b8318d21b0434adb8b0501e9befb66d33f0d8b25f6`; recheck if the plugin
changes. Since 2.3.0 the `release` task additionally validates the bundle before upload and
rejects one whose POM advertises Gradle metadata without a signed `.module` entry
(`BundleValidator.groovy:111`), so the 2.2.0 `.module`-omission shape cannot recur silently):

1. `./jmlx-jinja/release.sh`
2. Wait until `se.alipsa:jmlx-jinja:0.6.1` **resolves from Central** (bounded retries; its generated
   POM names 0.6.1 regardless of propagation state).
3. `./jmlx-tokenizer/release.sh`
4. `./jmlx-vision/release.sh`
5. `./tools/release-smoke/run.sh central --profile pure-java` — the binding fresh-cache gate.

Record per module: the deployment ID, terminal Central portal state, actual Central availability,
the release SHA and branch (each module's own, not the PR's head), and the final tag mapping.

## WP4 — cut and verify the release candidate

1. After the tooling PR is merged, open the release PR from clean `main`:
   - set `jmlx-jinja/build.gradle` to `0.6.0`, `jmlx-tokenizer/build.gradle` and
     `jmlx-vision/build.gradle` to `0.1.0`;
   - roll jinja's Unreleased changelog entries into a dated `0.6.0` section;
   - set the three pure-Java `tools/release-smoke/gradle.properties` defaults to the release
     versions (the pure release mode ignores them — the tuple is the sole source — but `ci` mode
     must match what the producers publish, and at WP4 that is the release versions, not
     snapshots); the WP5 development bump resets them to the next `-SNAPSHOT` lines;
   - commit `tools/release-smoke/released-8.1.properties` with exactly `0.6.0` / `0.1.0` / `0.1.0`
     (it is already checked in with these values by the tooling PR; a release PR may keep it
     unchanged).
2. Run the selected module checks, the root guards (`./gradlew check`) and, when build-logic
   changed, `./gradlew -p buildSrc check`.
3. From the clean release commit run jinja's complete checklist —
   `jmlx-jinja/req/release-checklist.md`, including `:jmlx-jinja:releaseVerification`, the
   dependency-update review and report inspection. Development
   `-PreleaseVerificationAllowDirty` output is not release evidence.
4. Run the pure candidate smoke from the clean release commit:
   ```sh
   ./tools/release-smoke/run.sh candidate --profile pure-java
   ```
   The preflight must report all three tuple values against matching release producers. Review the
   generated POMs (tokenizer's POM must name `se.alipsa:jmlx-jinja:0.6.0`), the sources/Javadoc
   jars, licenses/NOTICE and the candidate report.
5. Merge the approved release PR with a required merge commit. **Freeze merges to `main` from that
   merge** until the development-bump PR merges; only a reviewed recovery PR (below) may merge
   during the freeze, including during partial-publication recovery. This prevents unrelated
   changes from carrying versions already immutable on Central.
6. On clean `main` at the initial release merge commit, unconditionally rerun jinja's complete
   checklist (including `releaseVerification`) and the pure candidate smoke, and inspect their
   reports. The merge commit is the initial release SHA: absent recovery, all three `release.sh`
   runs execute from that same clean HEAD with no intervening commits, checkout changes or
   version edits — verify `git rev-parse HEAD` before each run and stop if it changes.

## WP5 — publish, verify Central, resume development

1. Execute the publication order above at the verified release SHA, retaining all three release
   versions through verification.
2. If publication partly succeeds: retain the released versions, check each deployment's state,
   and resume only missing modules. **Never overwrite an immutable released coordinate.**
3. Rerun `./tools/release-smoke/run.sh central --profile pure-java` (fresh cache). Preserve the
   failed-resolution report if availability remains incomplete after bounded retries; a
   successful upload is not a successful Central resolution.
4. Create annotated tags pointing at each module's verified release SHA:
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
  artifacts; candidate success cannot establish the final mixed graph before publication. Publish
  only missing or replacement coordinates, in dependency order, never successful coordinates
  again, then rerun the complete fresh-cache Central smoke against the updated tuple. Keep the
  freeze until recovery succeeds and the development-bump PR merges.
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
- Central: the fresh-cache `run.sh central --profile pure-java` log and, per module, deployment
  ID, portal state and Central resolution evidence.
- jinja verification: `jmlx-jinja`'s `releaseVerification` outputs on each jinja publication's
  release commit.
- Tags: the tag-to-SHA mapping recorded in `req/phase8-1-release-notes.md`.
