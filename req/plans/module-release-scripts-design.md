# Independent module releases — design

Status: implemented. Date: 2026-09-01.

Scope: make `jmlx-jinja` and `jmlx-tokenizer` independently releasable to Maven Central, each
driven by its own `release.sh`. No other module is in scope.

## Why this is not just "copy hfjinja's release.sh"

The starting assumption was that hfjinja's 29-line `release.sh` would work as-is in `jmlx-jinja`.
It will not. PR #15 migrated the *verification* half of hfjinja's release machinery
(`releaseVerification`, `verifyPublicationMetadata`, `verifyModuleDescriptor`, the
`req/release-verification.json` contract) but left the *publishing* half behind. As of
`19e47c6`, `jmlx-jinja` cannot publish to Central at all:

| `release.sh` requires | State in `jmlx-jinja` |
|---|---|
| `./gradlew ... release` → a `release` task | Missing. Provided by `se.alipsa.nexus-release-plugin` in hfjinja; that plugin was not migrated. Only `releaseVerification` exists, which is a verification matrix, not a publish. |
| `signing` plugin + GPG config | Missing. hfjinja applies `id 'signing'`; `jmlx-jinja` applies only `maven-publish`. Central rejects unsigned artifacts. |
| `./gradlew` in the script's own directory | Missing. hfjinja's script does `cd "$SCRIPT_DIR"` then `./gradlew`; no module directory in this monorepo has a wrapper — it lives only at the repo root. |
| `tasks.named('release') { dependsOn check }` | Missing, along with the plugin defining `release`. |

`jmlx-tokenizer` additionally has no publishing configuration whatsoever, and no `version` line
of its own — it silently inherits the root's `0.5.0-SNAPSHOT`.

## Decisions

1. **Scope: `jmlx-jinja` + `jmlx-tokenizer` only.** `jmlx-core` and `jmlx-ffi` require macOS and
   a native bootstrap, and would need native-artifact packaging designed first. `jmlx-examples`
   is an `application` demo with no library surface.

2. **Dependency coupling: keep the project dependency, add a release-order guard.**
   `jmlx-tokenizer` keeps `api project(':jmlx-jinja')` so jinja changes stay live during local
   development. Gradle publishes that as the concrete coordinate
   `se.alipsa:jmlx-jinja:<jinja's current version>`, so releasing tokenizer while jinja is still
   a SNAPSHOT would emit a POM Central rejects. A guard makes that fail fast and early instead.
   Consequence, accepted deliberately: releases are ordered — jinja before tokenizer.

3. **Verification depth for `jmlx-tokenizer`: lightweight parity.** It gets a version line,
   publishing config, signing, a `release` task depending on `check`, `release.sh` with the
   SNAPSHOT guards, and one adapted POM dependency assertion. It does *not* get
   archive-reproducibility, a module-descriptor check, or the isolated clean-checkout matrix.
   Those exist in `jmlx-jinja` because it is a byte-exactness port of an upstream JS library
   whose fidelity has to be proven; `jmlx-tokenizer` carries no equivalent risk. It also has no
   `module-info.java`, so `verifyModuleDescriptor` has nothing to check.

4. **Shared wiring lives in the root `build.gradle`.** Applied only to subprojects that apply
   `maven-publish`. This follows the convention CLAUDE.md already states (root holds shared
   config; module scripts stay thin) and the `apply false` + `subprojects { apply plugin: }`
   pattern the root already uses for Spotless. Rejected: duplicating the block in both modules
   (two copies to keep in sync), and a `buildSrc` convention plugin (a whole build unit for ~25
   lines across two modules).

## Design

### Plugin and publishing wiring

Root `build.gradle` declares `id 'se.alipsa.nexus-release-plugin' version '2.2.0' apply false`.
A `subprojects` block applies `signing` and the nexus plugin only where `maven-publish` is
present, then configures, lifted from hfjinja's proven setup:

```groovy
signing {
    required = providers.gradleProperty('signing.keyId').present
    sign publishing.publications.maven
}

nexusReleasePlugin {
    userName = providers.gradleProperty('sonatypeUsername').orNull
    password = providers.gradleProperty('sonatypePassword').orNull
    mavenPublication = publishing.publications.maven
}

tasks.named('release') { dependsOn tasks.named('check') }
```

`signing.required` being conditional on `signing.keyId` is load-bearing: it keeps `check`, CI,
and `releaseVerification` keyless. Making signing unconditional would break all three.

`release` depending on `check` is also load-bearing. `release.sh` requests `build` and `release`
as independent tasks, and Gradle imposes no ordering between independently requested tasks — so
without this edge, publishing could begin before verification finished.

### `release.sh`

One byte-identical script per releasable module, at `jmlx-jinja/release.sh` and
`jmlx-tokenizer/release.sh`. It derives its module from its own directory name, so the two
copies are literally the same file:

```bash
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODULE=$(basename "$SCRIPT_DIR")
cd "$SCRIPT_DIR/.."                      # monorepo root — the wrapper lives here
./gradlew ":$MODULE:clean" ":$MODULE:build" ":$MODULE:release"
```

hfjinja's sdkman `source ~/.sdkman/bin/sdkman-init.sh` + `source jdk21` step is **dropped**. It
was appropriate for a standalone single-module project pinned to one JDK; here the two modules
both target Java 21 via their own toolchain overrides, and Gradle toolchains already resolve
the correct JDK per module regardless of the daemon's own JVM. Verified empirically: this repo
builds and tests clean today with no sdkman sourcing. Dropping it is also what keeps the two
scripts byte-identical.

Two guards before invoking Gradle:

- **Own-version assertion.** The module's `build.gradle` must declare its own `version =` line.
  Without this, a module silently rides the root's version and an "independent" release is not
  independent at all — exactly the accident this work exists to prevent.
- **SNAPSHOT assertion.** The effective version comes from `./gradlew -q ":$MODULE:properties"`
  rather than a `sed` over `build.gradle`. That is authoritative and not regex-fragile, and it
  reflects any override. Refuse to publish if it ends in `-SNAPSHOT`, as hfjinja does.

### `jmlx-tokenizer` publishing configuration

Adds `maven-publish`, `withSourcesJar()`, `withJavadocJar()`, its own
`version = '0.1.0-SNAPSHOT'` (it has never been published, so it starts a fresh line rather than
tracking jinja's 0.6.x), and a POM block mirroring `jmlx-jinja`'s: name, description, url, MIT
license, developer, and scm.

### The SNAPSHOT-dependency guard

A `verifyNoSnapshotDependencies` task parses the generated POM and fails if any `<version>`
element contains `-SNAPSHOT`. It reuses the POM-XML-parsing approach `jmlx-jinja`'s
`verifyPublicationMetadata` already established. Wired into `release` for every publishing
module.

This is what enforces the release ordering from decision 2: running `jmlx-tokenizer/release.sh`
while jinja is `0.6.0-SNAPSHOT` fails with a clear message rather than producing a POM Central
would reject on upload.

Additionally, `jmlx-tokenizer` gets an assertion that its published dependency set is exactly
`{tools.jackson.core:jackson-databind, se.alipsa:jmlx-jinja}`, so a stray `implementation`
dependency cannot leak into the POM unnoticed. `jmlx-jinja`'s own `verifyPublicationMetadata`
continues to assert *zero* dependencies and is left untouched — the two checks are deliberately
different, and conflating them would weaken jinja's stronger guarantee.

### Release workflow

Releasing `jmlx-jinja`:
1. Set `version = '0.6.0'` in `jmlx-jinja/build.gradle`.
2. Run `jmlx-jinja/release.sh`.
3. Bump to `0.6.1-SNAPSHOT`.

Releasing `jmlx-tokenizer` additionally requires that it resolves a *released* jinja, so jinja's
version in the repo must not be a SNAPSHOT at that moment. The guard enforces this.

## Decision 5: `jmlx-tokenizer` targets Java 21

`jmlx-jinja` deliberately overrides the toolchain to Java 21 to preserve hfjinja's compatibility
promise, and its `req/release-verification.json` pins `bytecodeMajor: 65`. `jmlx-tokenizer`
overrode nothing, so it inherited the root's Java **25**. Publishing it unchanged would have
shipped a Java-25-only artifact that consumers on 21–24 could not use — including anyone already
consuming `jmlx-jinja` at 21.

Nothing forces 25 on `jmlx-tokenizer`: it is pure Java with no native dependency, and the root's
25 exists for `jmlx-core`/`jmlx-ffi`, which need Panama FFM.

`jmlx-tokenizer` therefore adds a toolchain override to 21, matching `jmlx-jinja`.

Verified rather than assumed, by adding the override and building:

- `:jmlx-tokenizer:build` succeeds; `HfTokenizer.class` is bytecode major 65.
- 133 tests run, 0 failures, 0 skipped.
- `checkstyleMain`, `checkstyleTest`, and `spotlessCheck` all pass.
- No Java 22+ language or API feature is used anywhere in the module.
- No other module depends on `jmlx-tokenizer`, so lowering its target cannot affect the rest of
  the build.

## Testing

The publish path cannot be exercised end-to-end without real Central credentials, so testing
targets the guards, which is where the actual risk is:

- Generate each POM and assert `verifyNoSnapshotDependencies` fails on a SNAPSHOT version and
  passes on a release version.
- Assert `:jmlx-tokenizer:release` fails while `jmlx-jinja` is a SNAPSHOT.
- Assert the tokenizer dependency-set check fails when an extra dependency is added.
- Confirm `signing` stays inert without `signing.keyId`, so `check`, CI, and
  `releaseVerification` remain keyless.
- Confirm `./gradlew build` and `:jmlx-jinja:check` still pass, including the existing
  `verifyPublicationMetadata` and `verifyModuleDescriptor`.
- Assert `jmlx-tokenizer`'s compiled classes are bytecode major 65, the same way `jmlx-jinja`'s
  contract is checked, so the target cannot silently drift back to the inherited 25.

## Deliberately excluded

- **Automated version bumping.** hfjinja does not do it either; a human edits the version and
  the script only refuses SNAPSHOTs. Automating it invites accidental releases.
- **CI publishing.** Release stays a local, credentialed, manual action. The
  `jmlx-jinja CI` workflow continues to verify, not publish.
- **`jmlx-core` / `jmlx-ffi` / `jmlx-examples`.** See decision 1.
