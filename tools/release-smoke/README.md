# Release smoke

An independent Gradle consumer that proves the published shape of the jmlx modules: it resolves
the selected coordinates **only** from a single source (a disposable local repository in
`ci`/`candidate` mode, Maven Central in `central` mode), runs the profile's smoke program against
the resolved jars, and verifies the complete runtime graph, POM-only agreement, documentation
artifacts, licenses, NOTICE provenance and JPMS names. It never uses `project(...)` dependencies,
`mavenLocal`, composite builds or checkout classes.

## Profiles

| Profile | Entry dependencies | Expected complete `se.alipsa` runtime graph | Runtime |
| --- | --- | --- | --- |
| `all` (default) | `jmlx-models` + runtime-only `jmlx-native-macos-arm64` | all seven coordinates | Java 25, native, macOS ARM64 |
| `pure-java` | `jmlx-tokenizer` + `jmlx-vision` (jinja arrives transitively through tokenizer's published `api` edge) | jinja, tokenizer, vision only | Java 21, headless, cross-platform |

`settings.gradle` selects the consumer projects per profile (`-PsmokeProfile`): the pure profile
neither configures nor compiles the native `app`, so a machine or CI job without a Java 25 toolchain
can still run it. `pom-check` (POM-only resolution) is shared and gets the same selected
dependencies and a compatible toolchain (21 pure / 25 full).

## Modes

```sh
./tools/release-smoke/run.sh ci                 # publish the selected modules' current versions to a
                                                # disposable repo (build/smoke-repo), then smoke them
./tools/release-smoke/run.sh candidate          # same, in release mode: the consumer expects released
                                                # (non-SNAPSHOT) versions; the candidate preflight checks the
                                                # actual producer versions before anything is published
./tools/release-smoke/run.sh central            # resolve everything from Maven Central (never publishes)
```

Every run uses a **fresh `GRADLE_USER_HOME`**, `--no-daemon --no-build-cache`, the checkout's
wrapper distribution and `env -u JMLX_LIBRARY_PATH`, so no cached, locally installed or
path-overridden jmlx artifact can be reused. `--record` (rewrite `goldens/mistral-sampled.properties`)
is `ci --profile all` only; it is rejected for every pure-Java run and for `candidate`/`central`.

`run.sh` parses all arguments before any side effect, clears the consumer's project state
(`.gradle`, `build`, each subproject's `build`) before any Gradle invocation, and invokes exactly
`:app:smoke` or `:pure-java:smoke` by full project path.

## Version selection

- `ci`: the `gradle.properties` defaults, overridable per module.
- `release` (candidate and central), profile `all`: the `gradle.properties` defaults with
  `-SNAPSHOT` stripped, overridable per module.
- `release`, profile `pure-java`: the committed **`released-8.1.properties`** tuple is the sole
  version source for both the local candidate run and the Central run. The `gradle.properties`
  defaults are ignored entirely (no snapshot-suffix fallback), an explicit override may only repeat
  a file value exactly, and a missing entry or a snapshot value fails before any resolution.
  Development snapshot bumps must not change this file; release and recovery PRs update it.

Version overrides are the `JMLX_<MODULE>_VERSION` environment variables (e.g.
`JMLX_TOKENIZER_VERSION=0.1.0`), forwarded to the consumer as `-P` properties. The version sources
are printed during configuration.

In `candidate` mode the preflight runs **before** the smoke repository is deleted or anything is
published: the pure profile requires all three tuple entries to equal the producers' own versions
(stripped of `-SNAPSHOT`) and overrides to repeat the file value; the full profile requires every
producer to equal the consumer's effective expected version (default or override, stripped) and to
not be a snapshot, so removing `-SNAPSHOT` from a consumer default never disguises snapshot
producer output. In this tooling PR `candidate --profile pure-java`'s preflight passes while the
producer versions are still `-SNAPSHOT` (the stripped values match the tuple); the consumer stage
then fails at resolution, until the release PR sets the producers to their release versions.

## JDK requirements

- **Producer (publish step):** the checkout's build JDK. The full profile's `candidate`/`ci`
  publish step also needs the staged native runtime for `jmlx-native-macos-arm64`
  (`scripts/bootstrap-native.sh`); the pure profile's publish step does not.
- **Consumer, profile `all`:** a Java 25 toolchain (plus the pinned wrapper distribution).
- **Consumer, profile `pure-java`:** a Java 21 toolchain only; the consumer JVM is asserted to be
  Java 21 and headless by the smoke program itself.

CI: the Linux Java job supplies both JDKs through the job's inherited `GRADLE_OPTS`
(`-Dorg.gradle.java.installations.fromEnv=JAVA_HOME_21_X64,JAVA_HOME_25_X64` with
`-Dorg.gradle.java.installations.auto-download=false`); the consumer invocation inherits them.
The macOS native job runs the full profile with its Java 25. Locally, Gradle auto-detects JDKs in
standard OS locations with a fresh home (or downloads a missing toolchain through the configured
Foojay resolver); GitHub's `setup-java` installations lie outside the auto-detected locations,
which is why the Linux job uses the explicit `fromEnv` configuration.

## What the smoke verifies

Both profiles: exact selected versions, no unexpected `se.alipsa` components, module artifacts
rather than project/file substitutes, the identical complete runtime coordinate set under Gradle
metadata and POM-only resolution, no snapshot dependency in either release-mode graph, main/
sources/Javadoc jars for every selected module, `META-INF/LICENSE` in every runtime jar, the
three vision jars' `META-INF/NOTICE` (pinned Pillow and PyTorch source references and license
texts), and the published JPMS names (jinja's module descriptor `se.alipsa.jmlx.jinja`,
tokenizer's and vision's `Automatic-Module-Name`).

- **`all` (`:app:smoke`):** the native payload in `jmlx-native-macos-arm64` (dylibs,
  `mlx.metallib`, pin file), fresh extraction from the packaged jar into a disposable cache, jar
  provenance for every probed class, the seeded two-request batched generation against the
  committed golden, and the `cohortSizes() == [2]` gate.
- **`pure-java` (`:pure-java:smoke`):** no native-chain class is even loadable; the written
  coordinate report holds exactly jinja/tokenizer/vision; every probed class comes from a resolved
  jar of the expected version. Then the public-API assertions: a jinja message loop with non-ASCII
  output; the hand-written byte-level BPE tokenizer's exact encode/decode facts, the exact rendered
  chat prompt (fixed clock) and its exact token ids, and incremental decoding where a Unicode
  character spans byte tokens; one committed image-oracle resize case decoded and resized
  byte-exactly against the Pillow-recorded bytes (Pillow codes 1/2/3 mapped explicitly, others
  rejected), with the fixture's digests re-verified against the pinned image-oracle sources; and an
  analytic `normalize` with exact float32 values.

## Fixtures

- `fixtures/`: the full profile's hand-written smoke tokenizer paired with the committed synthetic
  Mistral checkpoint (`fixtures/PROVENANCE.md`).
- `fixtures/pure-java/`: the pure profile's fixture set — one image-oracle resize case (PNG,
  case spec, expected bytes) plus the hand-written byte-level BPE tokenizer and its chat template.
  Every digest is recorded in `digests.txt`; the provenance, the cross-verification against the
  pinned HF tokenizers/transformers/Pillow references and the extraction instructions are in
  `fixtures/pure-java/PROVENANCE.md`. No Hub credentials, model downloads or remote code are
  required.
- `goldens/mistral-sampled.properties`: the full profile's seeded sampling golden
  (`--record` rewrites it, `ci --profile all` only).

## Regression tests

`selftest.sh` exercises `run.sh`'s argument parsing and candidate preflight in subshells against
disposable fixtures (the `JMLX_SMOKE_TUPLE_FILE`, `JMLX_SMOKE_PRODUCER_VERSIONS` and
`JMLX_SMOKE_REPO_DIR` test hooks), asserting exit codes and that a failing preflight leaves the
smoke repository untouched. It is a step of the Linux Java CI job.

## Failure diagnosis

- `no usable Gradle <version> distribution under <dists_dir>`: run `./gradlew --version` once
  from the checkout with its default `GRADLE_USER_HOME` so the wrapper downloads the distribution
  the consumer must use.
- `Could not find se.alipsa:<module>:<version>`: the selected source (disposable repo or Central)
  lacks that exact coordinate. In `ci`/`candidate` the publish step ran for the selected modules
  only — unselected modules deliberately may be missing; in `candidate --profile pure-java` with
  `-SNAPSHOT` producers this is expected until the release PR sets release versions; in `central`
  the module has not been published (yet) or the tuple names a version that was never released.
- `Release mode forbids SNAPSHOTs` / `... is a snapshot`: a selected producer (or a transitive
  dependency) is still a snapshot while the consumer is in release mode.
- `producer <module> version X != consumer expected Y` (preflight): the module's own `version =`
  line and the consumer's default/override/tuple disagree; fix the version pair in the same PR.
- `A project dependency leaked into the consumer graph`: the consumer build lost its isolation —
  check that no `project(...)` dependency or composite build was introduced.
- `Module-metadata and POM-only runtime classpaths differ`: a published POM and its Gradle
  metadata disagree on the runtime graph; inspect the module's `api`/`implementation`/`runtimeOnly`
  declaration.
- `consumer JVM must be Java 21` (pure profile): the Java 21 toolchain was not discovered — see
  the JDK requirements above.
