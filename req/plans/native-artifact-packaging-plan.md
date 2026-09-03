# Native Artifact Packaging + Full Module Publishing

**Status: implemented.**

## Context

Phase 5 (`req/phase5-plan.md`: checkpoint I/O, tokenizer, reference models) is functionally
complete, but only `jmlx-jinja` and `jmlx-tokenizer` had a release path to Maven Central. CLAUDE.md
stated plainly that `jmlx-core`/`jmlx-ffi` "would need native-artifact packaging designed first"
before they could be published, and `jmlx-models` (also in the native chain) had the same gap.
`req/project-outline.md`'s own "Packaging & Distribution Strategy" section had named the shape as
aspirational since the very first plan (`req/initial-plan.md` explicitly listed "the
`jmlx-native-macos-arm64` classifier jar and its extract-from-classpath loader" as out of scope for
v0.1) but nothing had ever implemented it.

This plan does two things: (1) package the four MLX/mlx-c binaries
`scripts/bootstrap-native.sh` stages into `native/install/lib/` as a Maven-resolvable artifact with
a runtime classpath-extraction fallback in `NativeLoader`, and (2) wire `maven-publish` onto
`jmlx-ffi`, `jmlx-core`, and `jmlx-models`, following the exact pattern `jmlx-jinja`/`jmlx-tokenizer`
already established (root `build.gradle`'s reactive `plugins.withId('maven-publish')` wiring).
`jmlx-examples` (an `application` demo) stays unpublished, as it always was.

## Decisions taken

1. **New module, not a classifier jar.** `jmlx-native-macos-arm64` is a dedicated artifact (matching
   `req/project-outline.md`'s original naming) rather than a classifier on `jmlx-core`/`jmlx-ffi`.
   Simpler dependency resolution for a single platform; revisit only if/when a second platform
   (Linux, x86_64, other Apple Silicon) is actually built — no second build pipeline exists today.
2. **The bundled binaries, not a classifier jar's source.** The module's jar contains no compiled
   Java of its own — its only payload is `libmlxc.dylib`, `libmlx.dylib`, `libjaccl.dylib`,
   `mlx.metallib`, and `native-pin.properties` as classpath resources under
   `se/alipsa/jmlx/native/macos-aarch64/`. `withSourcesJar()`/`withJavadocJar()` are harmless no-ops
   given no Java source, kept only for artifact-set consistency with the other publishable modules.
3. **`native-pin.properties` embeds the real upstream pin inside the jar** (written by
   `scripts/bootstrap-native.sh` right after staging the flat directory: `mlxMetalVersion`,
   `mlxcCommit`), rather than encoding it in the Maven version string. Keeps the module's own SemVer
   independent of mlx-c's own independent versioning (already a distinct axis per CLAUDE.md's
   "Native version pinning" section) and gives `ClasspathNativeExtractor` a real cache key tied to
   the actual binaries.
4. **Independent SemVer for the new module** (`0.1.0-SNAPSHOT`), matching jinja/tokenizer's
   never-published-before precedent.
5. **`jmlx-core` does not depend on the native module.** A consumer opts in explicitly (documented
   in the published POM description, not forced by a Gradle dependency) — an existing
   `bootstrap-native.sh`-based dev flow isn't forced to also pull a ~180MB jar, and this keeps the
   decision of "system-installed vs. bundled native runtime" entirely with the consumer, matching
   `NativeLoader`'s existing `jmlx.library.path`/`JMLX_LIBRARY_PATH` precedent of never assuming one
   sourcing strategy.
6. **Extraction directory: `~/Library/Application Support/se.alipsa.jmlx/native/<hash>`**, overridable via a
   `jmlx.native.cache.path` system property (for sandboxed/read-only-home environments). The hash is
   SHA-256 of the packaged `native-pin.properties` bytes, so a JVM only pays the ~180MB extraction
   cost once per pin rather than on every process start, and a repin automatically gets a fresh
   cache entry rather than reusing a stale one.
7. **`jmlx-ffi`/`jmlx-core` get their own explicit `version = '0.5.0-SNAPSHOT'` line** (the same
   number they already carried via the root's `allprojects` default, made independent going
   forward); **`jmlx-models` starts a fresh `0.1.0-SNAPSHOT`** (never published, same reasoning
   tokenizer used for its own first version).
8. **`jmlx-core`/`jmlx-models` disable Javadoc's `missing` doclint category**
   (`-Xdoclint:all,-missing`) rather than retrofitting `@param`/`@return`/`@throws` tags onto an
   already prose-documented, pre-existing public surface once the root's reactive `-Werror` wiring
   applied to them for the first time (~100 and ~47 warnings respectively, all "no @param"/"no
   @return"/"no comment", none a genuine content problem). Writing filler tags for self-explanatory
   parameters (`MLXArray a`, `int axis`, `int[] shape`) would be pure noise, not documentation, and
   would violate this repo's own no-filler-comment convention. `-Werror` still gates every other
   Javadoc problem (broken `@link`, malformed HTML). `jmlx-ffi` took a different, narrower fix
   instead (see Task 7 below) since its problem was different in kind: one real missing `@return`
   in hand-written code, plus the committed jextract blob having no doc comments at all by design.

## Work breakdown

### 1. `jmlx-native-macos-arm64`: scaffold, stage, verify — **DONE**

`settings.gradle` gained the module; its `build.gradle` has a `stageNativeResources` task that
copies the 4 binaries from `native/install/lib/` into the module's resources, and a
`verifyPackagedNativeResources` task (wired into `check`) that re-opens the built jar as a zip and
confirms all 5 entries exist with `mlx.metallib` >= 100MB.

**Amendment found during implementation:** `stageNativeResources` was originally a `Copy` task.
Gradle's `Copy`/`Sync` task types are `@SkipWhenEmpty` on their source — when
`native/install/lib/` doesn't exist at all, Gradle marks the task `NO-SOURCE` and skips it *without
ever running `doFirst`/`doLast`*, so the intended "missing files" failure message never fired.
Fixed by making it a plain task with a manual `project.services.get(FileSystemOperations)` copy in
`doLast` (capturing the service at configuration time, not referencing `project` at execution time,
which is deprecated and configuration-cache-incompatible).

**Second amendment, more significant:** every other native-dependent test in this repo is
**skipped, not failed**, when `native/install/lib` is absent (`@EnabledIfNativeAvailable`,
CLAUDE.md's own stated invariant) — empirically verified before this work started
(`./gradlew build -x :jmlx-native-macos-arm64:build -x :jmlx-native-macos-arm64:check` with
`native/install/lib` renamed away: green except the pre-existing unrelated `jmlx-jinja:test`
failure). The first version of `stageNativeResources`/`verifyPackagedNativeResources` *failed
loudly* whenever `native/install/lib` was absent, which broke that invariant for the whole
aggregate `./gradlew build` — a contributor who never bootstrapped and doesn't care about native
functionality would now hit a hard failure on this one module. The final design leaves
`stageNativeResources` always actionable: an unstaged checkout removes any stale generated resources
and makes an empty local jar with a warning, while `verifyPackagedNativeResources` skips its content
assertion. This avoids `onlyIf` retaining stale outputs. A *partially* staged directory
(`mlx.metallib` present but something else missing — a broken or interrupted bootstrap run) still
fails loudly: that state is never "not bootstrapped yet," it's actually broken, and the "never
silently produce a corrupt jar" guarantee still applies once staging is attempted at all. Verified
all three states directly (unstaged → warning and BUILD SUCCESSFUL; partially staged → FAILED with
the exact missing-files message; fully staged → normal success), and re-verified the whole-repo build
tolerates a fresh, unbootstrapped checkout (see Task 7's matching amendment for
`jmlx-ffi:extractionFallbackTest`).

### 2. `native-pin.properties` — **DONE**

`scripts/bootstrap-native.sh` writes it into `native/install/lib/` only after the flat runtime passes
its dlopen, codesign, and metallib-size validation, from the script's own already-pinned
`MLX_METAL_VERSION`/`MLX_C_COMMIT` constants. Its presence is therefore the staging completion marker.

### 3. NOTICE/LICENSE — **DONE**

`jmlx-native-macos-arm64/NOTICE` attributes MLX (MIT, Apple Inc. 2023), mlx-c (MIT, ml-explore
2023), and metal-cpp (Apache License 2.0, Apple, vendored inside `libmlx.dylib` — confirmed
directly from the staged wheel's `mlx/include/metal_cpp/LICENSE.txt`; the wheel's own `METADATA`
only declares `License: MIT` and doesn't surface this vendored component). `LICENSE` is the repo's
own root MIT text. `jmlx-ffi/NOTICE` separately attributes the committed jextract bindings as
mechanically derived from mlx-c's public C headers (MIT) — a distinct attribution from the bundled
*binaries*, since `jmlx-ffi` ships no binaries of its own.

### 4. `maven-publish` wiring for the native module — **DONE**

Mirrors `jmlx-tokenizer/build.gradle`'s shape (POM with two `<license>` entries: MIT + Apache-2.0),
`Automatic-Module-Name: se.alipsa.jmlx.nativelib.macosarm64`, byte-identical `release.sh`.

### 5. `ClasspathNativeExtractor` + `NativeLoader` fallback — **DONE**

New package-private `jmlx-ffi/.../ClasspathNativeExtractor.java`. `NativeLoader.resolveLibraryDir()`
gained a third step between the `JMLX_LIBRARY_PATH` check and the final `throw`: if the platform is
`Mac OS X`/`aarch64`, try classpath extraction before giving up. The explicit-path override always
wins, so no existing test's behavior changed.

**Amendment found during implementation:** the initial atomic-rename fallback caught only
`FileAlreadyExistsException` to detect a lost extraction race. `ClasspathNativeExtractorTest`'s own
8-thread concurrency test failed immediately with a `FileSystemException` instead — renaming a
directory onto an already-populated destination directory is `ENOTEMPTY` under POSIX `rename()`
semantics, not `EEXIST`, and Java's `UnixFileSystemProvider` maps that to a different exception type.
Fixed by treating *any* failed rename as a possible lost race, verified by checking whether `target`
is a complete extraction afterward (rethrowing only if it genuinely isn't) — portable across
whatever exception shape a given filesystem happens to report, rather than pattern-matching specific
types.

### 6. Fixture-based + real end-to-end extraction tests — **DONE**

`ClasspathNativeExtractorTest` (tiny fake fixtures under a deliberately different resource root,
`native-fixture`, so they can never collide with the real ~180MB binaries) covers: not-on-classpath,
happy path, idempotency, an 8-thread concurrency race, cache-key sensitivity to different pin
content, and the pure `isSupportedPlatform(osName, osArch)` function. `jmlx-ffi:extractionFallbackTest`
(a forked `Test` task, same pattern as the existing `loaderGuardTest`) proves the real fallback
against the real, published-shape `jmlx-native-macos-arm64` jar on real hardware.

**Amendment found during implementation (the same class of bug as Task 1's, one level up):**
`extractionFallbackTest` initially had no gate at all, so on a genuinely fresh, unbootstrapped
checkout it would *fail* (not skip) once `jmlx-native-macos-arm64`'s jar was legitimately empty per
Task 1's fix — reintroducing the exact invariant violation Task 1 had just fixed, one dependency
level removed. Fixed with the identical `onlyIf` gate pattern, checked directly against
`native/install/lib/mlx.metallib`. Verified with a true fresh-clone simulation: `native/install/lib`
renamed away *and* every module's `build/` directory deleted (a prior successful build's stale
`build/generated/nativeResources/` output can otherwise mask this exact bug, since Gradle's
`onlyIf`-skip doesn't clean previously-produced outputs — this masked the bug on the first
verification attempt) — confirmed `SKIPPED` with a clear warning, `BUILD SUCCESSFUL`.

**Publication refinement:** the native module now has a small genuine Java metadata API
(`NativeArtifact`). Generated native resources are fed directly to `jar` rather than registered as a
source-set resource directory. Standard `withSourcesJar()`/`withJavadocJar()` consequently publish
honest, small companion artifacts containing that API and its documentation without duplicating the
native payload.

### 7. `jmlx-ffi`/`jmlx-core`/`jmlx-models` `maven-publish` wiring — **DONE**

Same shape as jinja/tokenizer. `jmlx-ffi` restricts both `checkstyleMain` (pre-existing) and (newly)
`javadoc`'s `source` to `src/main/java` only, excluding the committed jextract blob under
`src/main/generated/java` — the root's reactive `-Werror` wiring would otherwise demand doc comments
on a huge machine-generated surface that must stay byte-identical to `scripts/regen-bindings.sh`'s
output. This surfaced exactly one real, worth-fixing warning (a missing `@return` on
`NativeLoader.lastNativeError()`), fixed directly. `jmlx-core`/`jmlx-models` instead disabled
doclint's `missing` category repo-wide for their own `javadoc` tasks — see Decision 8 above for why
that's a different fix for a different-shaped problem. `jmlx-models` also got its own
`verifyPublishedDependencies` (mirroring `jmlx-tokenizer`'s, asserting the exact dependency set:
`jmlx-core`, `jmlx-tokenizer`, `jackson-databind`), since — unlike `jmlx-jinja`'s zero-dependency
guarantee — it legitimately has several.

### 8. CI — **DONE**

`.github/workflows/ci.yml`'s `native` job gained `:jmlx-native-macos-arm64:check`
appended to its existing check line. `bootstrap-native.sh` already runs earlier in that job, so
`native-pin.properties` exists before Gradle runs there.

### 9. Docs — **DONE**

`CLAUDE.md`'s "Releasing a module" section now names six publishable modules and both independent
release-order chains (jinja→tokenizer; ffi→core→models; the native module has no dependency
relationship to either and releases independently). Its Architecture section's diagram and
"only published modules" paragraph were updated to match.
`req/project-outline.md`'s Packaging & Distribution Strategy table (previously naming only 2 files
as "universal binaries" — stale even before this work, since it predated the real 4-file flat
layout) was corrected to the actual artifact shape.

## Verification (whole-repo, run repeatedly during implementation)

```sh
./gradlew build -x :jmlx-jinja:test                     # whole repo, 7 modules, green
./gradlew :jmlx-jinja:check :jmlx-tokenizer:check :jmlx-native-macos-arm64:check \
          :jmlx-ffi:check :jmlx-core:check :jmlx-models:check   # all 6 publishable modules together
for m in jmlx-jinja jmlx-tokenizer jmlx-native-macos-arm64 jmlx-ffi jmlx-core jmlx-models; do
  diff jmlx-jinja/release.sh "$m/release.sh" && echo "$m: identical"
done
for m in jmlx-native-macos-arm64 jmlx-ffi jmlx-core jmlx-models; do
  ./"$m"/release.sh; echo "$m exit=$?"                    # each refuses: still SNAPSHOT
done
```

Fresh-clone-without-bootstrap invariant, re-verified after every amendment above (rename
`native/install/lib` away, delete every module's `build/` directory, delete
`~/Library/Application Support/se.alipsa.jmlx`):

```sh
./gradlew build -x :jmlx-jinja:test    # still BUILD SUCCESSFUL; stale native resources removed with a warning
```

## Deliberately out of scope

- Actually running any `release.sh` against Central — publishing stays a separate, manual,
  credentialed action per existing convention.
- Multi-platform native artifacts (Linux, x86_64, other Apple Silicon) — no second build pipeline
  exists; the resource-path scheme (`macos-aarch64` as an explicit discriminator segment) leaves
  room for siblings if/when one is built.
- Checksum verification of extracted files against an embedded manifest, and cross-version extraction
  directory garbage collection for `~/Library/Application Support/se.alipsa.jmlx/native` — accepted as low-risk for v1
  (Central's own artifact integrity already covers the classpath contents; per-pin cache growth is
  bounded and small in practice).

## Post-PR review amendments

External review of PR #17 found three real gaps and several worthwhile hardening suggestions, all
verified against the actual code (not taken on faith) before fixing:

- **`jmlx-native-macos-arm64`'s publish path had no hard gate.** `stageNativeResources` and
  `verifyPackagedNativeResources` deliberately `onlyIf`-skip (not fail) on an unstaged checkout so
  `check`/`build` stay green — but `release` (`build.gradle:308`) `dependsOn check`, and a skipped
  task is a *passing* dependency. Reproduced directly: hiding `mlx.metallib` and running
  `publishToMavenLocal` completed successfully with an empty jar before this fix. `check`/`build`
  must stay lenient; publishing must not, since a bad Central release can never be taken back. Fixed
  with an unconditional `doFirst` gate on every `AbstractPublishToMaven` task (covers `release` and
  `publishToMavenLocal` alike), re-verified: fails loudly when unstaged, succeeds normally once
  binaries are staged again.
- **`ClasspathNativeExtractor`'s stale-target and cleanup-failure bugs.** A `target` directory that
  exists but is incomplete (interrupted extraction, a cache cleaner deleting one file, disk
  exhaustion) made `Files.move(..., ATOMIC_MOVE)` fail with a non-`EEXIST` `IOException`
  (`ENOTEMPTY`) forever after — `isComplete(target)` stayed false on every retry, so the failure was
  permanent and self-perpetuating rather than self-healing. Fixed by deleting an incomplete `target`
  and retrying the rename once. Separately, cleanup of the loser's temp copy (or of `tmp` on a
  genuine extraction failure) used the throwing `deleteRecursively`, so a failed *cleanup* could mask
  a successful extraction or the real underlying error; both call sites now use a best-effort
  `deleteQuietly`.
- **`stageNativeResources` re-copied the ~180MB staged directory on every single build** —
  `outputs.upToDateWhen { false }` was the (unnecessary) workaround for a missing directory breaking
  `inputs.dir`; a `FileTree` tolerates a missing root fine. Replaced with a real
  `inputs.files(fileTree(...)).skipWhenEmpty(false)` declaration; re-verified the task is now
  `UP-TO-DATE` on a second run with nothing changed.
- **Cache-key hardening**: the key hashed only `native-pin.properties`, so a contributor rebuilding
  the binaries locally at an unchanged pin (exactly the local `mavenLocal()`-publish workflow this
  repo's own tooling expects) would silently hit a stale cache entry. Mixed each binary's byte size
  (cheap: `URL#openConnection()` reports length without reading content) into the key — deliberately
  *not* a full content hash, since hashing ~180MB on every JVM start would defeat the point of the
  cache; a same-size local rebuild still collides, an accepted residual gap. Also: `isComplete` now
  requires `native-pin.properties` itself (previously ignored, despite defining the key), stale
  `.tmp-*` siblings older than an hour are swept best-effort on each extraction attempt, and the hex
  encoding uses `HexFormat` instead of a per-byte `String.format` loop.
- **Not changed**: a suggestion to hash all four binaries' full contents into the cache key was
  deliberately not adopted (see above — the cost is paid on every JVM start, not just on a rebuild).
  A question about Maven Central's artifact-size limits was researched, not coded: the Publisher
  Portal's per-upload bundle limit is 1GB; this module's payload (~182MB of binaries plus small
  sources/javadoc/POM) is well under that for a single release.

Re-verified after all of the above: `./gradlew build -x :jmlx-jinja:test` green; `:jmlx-ffi:test
--tests "*ClasspathNativeExtractorTest*"` and `:jmlx-ffi:extractionFallbackTest` (the real
~180MB-artifact end-to-end path) both green; the fresh-clone-without-bootstrap invariant re-checked
directly (hid `mlx.metallib`, ran `publishToMavenLocal`, confirmed the new hard failure, restored
the file, confirmed a normal publish succeeds and cleaned up the resulting `mavenLocal()` test
artifact).
