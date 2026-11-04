# Combined release workflow (`releaseAll.sh`)

Publishes every **eligible, unreleased, non-SNAPSHOT** Gradle publication in this repository as
one Maven-layout ZIP and one automatic Maven Central Portal deployment. This is the Phase 8.1
publication mechanism: the three pure-Java modules (`jmlx-jinja`, `jmlx-tokenizer`,
`jmlx-vision`) go out together in a single deployment. The four native-chain modules
(`jmlx-ffi`, `jmlx-core`, `jmlx-models`, `jmlx-native-macos-arm64`) stay on their `-SNAPSHOT`
versions, so the planner excludes them (and reports them as excluded) in every run.

The per-module `release.sh` scripts remain available for individual releases, but they are not
the Phase 8.1 path: Maven Central allows 10 deployments per month, and the combined flow spends
one slot for the whole tuple.

## How it works

`releaseAll.sh` drives the six combined tasks of `se.alipsa.nexus-release-plugin`
(`gradle/libs.versions.toml`, pin `2.4.1`; the optional static POM inputs used by consumers with
BOM/aggregate components were added in 2.4.1, which is why jmlx can use the flow at all):

1. `releaseAllPlan` — discovers every subproject publication, generates each candidate POM,
   checks each exact coordinate against Maven Central (repo1 POM probe, then authenticated
   Portal `/publisher/published` for the ambiguous case) and writes
   `build/releaseAll/manifest.json` plus a human-readable `report.txt`. SNAPSHOT versions and
   modules that (transitively) depend on unreleased coordinates are excluded with reasons.
2. `releaseAllSigningCheck` — verifies Gradle signing is configured for every selected module
   before the hours-long verification build runs.
3. Per selected module: `check` + `verifyNoSnapshotDependencies` (run by `releaseAll.sh`).
4. `releaseAllStage` (with `-PreleaseAllPrepare`) — re-checks that HEAD and the working tree
   match the plan, then `publish…ToReleaseAllStagingRepository` writes each selected module's
   signed pom, `.module` and jars into `build/releaseAll/staging`.
5. `releaseAllPrepareMaven` — no static Maven components exist here, so Maven itself is never
   invoked; the task still runs the shared `staged-settings.groovy` helper (the plugin requires
   it as an input) and verifies staging against the manifest.
6. `bundleAll` — validates every staged file (allow-list, signatures, checksums, the POM
   `published-with-gradle-metadata` marker against the bundled `.module`) and writes
   `build/releaseAll/bundle.zip` + `bundle-receipt.json`.
7. `releaseAll` — re-checks source and availability, uploads the bundle once
   (`publishingType=AUTOMATIC`), journals to `build/releaseAll/deployment.json`, and polls to
   `PUBLISHED` plus repo1 HEAD 200 for every selected component's POM and main JAR.

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
./releaseAll.sh --bundle-only  # … through bundleAll; writes bundle.zip; no upload
./releaseAll.sh                # full release, including the single upload
./releaseAll.sh --resume <deployment-id>   # poll an already-submitted deployment to PUBLISHED
```

Prerequisites (checked by `tools/release-all/preflight.sh`): Bash 4+ (`brew install bash` on
macOS), `java` 21+ on PATH (the Gradle daemon JVM; the workflow only executes the pure-Java
modules' tasks), `mvn` 3.9.9+, `groovy`, `git`, a clean working tree at HEAD, and the
`signing.keyId`, `sonatypeUsername`, `sonatypePassword` Gradle properties
(`~/.gradle/gradle.properties`).

## Testing plugin changes locally

`releaseAll.sh` forwards `RELEASE_ALL_PLUGIN_DIR=<plugin checkout>` as
`-PreleaseAllPluginDir`, and `settings.gradle`'s `pluginManagement` block includes that
checkout as a composite build. In release mode (`-PreleaseAllMode`) the plugin otherwise
resolves **exclusively** from the Gradle Plugin Portal, so a locally published build can never
replace the released version in a production run.

Note: `gradle/libs.versions.toml` pins 2.4.1, so before 2.4.1 is on the Portal, **every**
Gradle invocation in this tree (including this workflow) needs the `RELEASE_ALL_PLUGIN_DIR`
override or it cannot resolve the plugin.

## Files

| File | Role |
| --- | --- |
| `release-all.gradle` | Applied by the root `build.gradle` under `-PreleaseAllMode`; builds the candidate JSON from the live publications, wires all six tasks, and registers the per-module `releaseAllStaging` repository. |
| `preflight.sh` | Tool and filesystem preflight sourced by `releaseAll.sh`. |
| `Paths.groovy` | Dependency-free safe path canonicalization/listing helper (no GNU utilities). |
| `staged-settings.groovy` | Copies `~/.m2/settings.xml` preserving its content and adds the disposable staging file remote (run by the plugin in every mode). |
| `verify-staged.sh` | Manifest-vs-staging file check; the plugin invokes it only for static Maven components, which jmlx has none of — it exists because the plugin declares it as a required input. |
| `capture-portal-fixture.groovy` | Captures `req/releaseAll-portal-fixture.json` from the live Portal (manual, credentialed). |
