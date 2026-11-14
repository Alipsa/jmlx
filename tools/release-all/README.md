# Combined release workflow (`releaseAll.sh`)

Publishes every **eligible, unreleased, non-SNAPSHOT** Gradle publication in this repository as
one Maven-layout ZIP and one automatic Maven Central Portal deployment. This is the Phase 8.1
publication mechanism: the three pure-Java modules (`jmlx-jinja`, `jmlx-tokenizer`,
`jmlx-vision`) go out together in a single deployment. The four native-chain modules
(`jmlx-ffi`, `jmlx-core`, `jmlx-models`, `jmlx-native-macos-arm64`) are named in the release
set but stay on their `-SNAPSHOT` versions until 8.4, so the planner excludes them (and reports
them as excluded) in every run.

The per-module `release.sh` scripts remain available for individual releases, but they are not
the Phase 8.1 path: Maven Central allows 10 deployments per month, and the combined flow spends
one slot for the whole tuple.

## How it works

`releaseAll.sh` drives the combined tasks of `se.alipsa.nexus-release-plugin` (pinned in
`gradle/libs.versions.toml`; 2.5.0 introduced the module-based combined release this workflow
uses, and the pin is at 2.5.1, which fixes the two 2.5.0 defects jmlx hit:
Alipsa/nexus-release-plugin#11, deterministic bundle entry times, and #12, plan/manifest
task wiring plus the up-front rejection of `releaseAllPlan` + `releaseAll` in one
invocation). The plugin derives the candidate set, the per-module staging publication, signing and
every output path from the root `build.gradle`'s `combinedRelease.modules` list; jmlx only sets
the module list, the committed Portal fixture and the report/resume overrides:

1. `releaseAllPlan` — builds the candidate set from the seven named modules' `maven`
   publications (their generated POMs, the published file lists including the `.module` a POM
   advertises via the `published-with-gradle-metadata` marker, and the project-dependency
   edges), checks each exact coordinate against Maven Central (repo1 POM probe, then
   authenticated Portal `/publisher/published` for the ambiguous case) and writes
   `build/releaseAll/manifest.json` plus a human-readable `report.txt`. SNAPSHOT versions and
   modules that (transitively) depend on unreleased coordinates are excluded with reasons; the
   plan requires a clean working tree and records HEAD.
2. `releaseAllSigningCheck` — verifies Gradle signing is configured for every selected module
   before the verification builds run (strict by default; warning mode only for `--dry-run`).
3. `bundleAll` — re-checks the source (HEAD, clean tree, generated-POM hashes, effective
   publication versions), then stages: for every selected module it signs the publication and
   runs `publish…ToCombinedReleaseStagingRepository` into `build/releaseAll/staging`, with the
   module's own `check` and the source guard wired in as dependencies, so the selected
   modules' tests run at publication time. The SNAPSHOT consumer `:jmlx-models` is not part of
   this flow; the runbook's root `./gradlew check` covers it. Finally the task validates every
   staged file (allow-list, signatures, checksums, POM metadata) and writes
   `build/releaseAll/jmlx-mega-bundle.zip` + `bundle-receipt.json` (the bundle SHA-256).
4. `releaseAll` — re-checks source and availability (a coordinate published since planning is
   refused), verifies the bundle and manifest hashes against the receipt, uploads the bundle
   once (`publishingType=AUTOMATIC`), journals to `build/releaseAll/deployment.json`, and polls
   to `PUBLISHED` plus repo1 HEAD 200 for every selected component's POM and main JAR before
   logging "published and available".

`releaseAllPrepareMaven` is registered by the plugin for consumers with static Maven
(BOM/aggregate) candidates; jmlx has none, so it never runs.

The Portal availability checks are **fail-closed**: an upload is refused unless the committed
sanitized fixture `req/releaseAll-portal-fixture.json` confirms that the Portal answers
`published:true` for a known-published `se.alipsa` coordinate and `published:false` for a
never-uploaded one. The fixture records no credential material. Re-capture it with
`groovy tools/release-all/capture-portal-fixture.groovy` (credentials from
`SONATYPE_USERNAME`/`SONATYPE_PASSWORD` or `~/.gradle/gradle.properties`) if the Portal API
behavior is ever in question; `releaseAll` itself does not refresh it.

## Usage

```sh
./releaseAll.sh --dry-run      # plan + report + signing preflight (warning mode); no upload
./releaseAll.sh --bundle-only  # … through bundleAll; writes jmlx-mega-bundle.zip; no upload
./releaseAll.sh                # full release, including the single upload
./releaseAll.sh --resume <deployment-id>   # poll an already-submitted deployment to PUBLISHED
```

Prerequisites (checked by `tools/release-all/preflight.sh`): Bash 4+ (`brew install bash` on
macOS), `java` 21+ on PATH (the Gradle daemon JVM; the workflow only executes the pure-Java
modules' tasks), `groovy`, `git`, a clean working tree at HEAD, and the `signing.keyId`,
`sonatypeUsername`, `sonatypePassword` Gradle properties (`~/.gradle/gradle.properties`).
Credentials resolve from the `sonatypeUsername`/`sonatypePassword` Gradle properties, falling
back to `SONATYPE_USERNAME`/`SONATYPE_PASSWORD`. The optional `releaseAllRepositoryUrl` and
`releaseAllPortalUrl` properties redirect the availability checks and the upload (testing
only).

## Files

| File | Role |
| --- | --- |
| `preflight.sh` | Tool preflight sourced by `releaseAll.sh`. |
| `capture-portal-fixture.groovy` | Captures `req/releaseAll-portal-fixture.json` from the live Portal (manual, credentialed). |

## Phase 8.3 adapter candidates

The root candidate list also names jmlx-spring-ai and jmlx-langchain4j.
They remain 0.1.0-SNAPSHOT and are excluded until 8.4. The existing 8.1
expected-selection guard in releaseAll.sh remains intact; 8.4 must update
that guard to its reviewed release tuple before publishing the native chain
and adapters. Both adapters depend on models and must release after models.
