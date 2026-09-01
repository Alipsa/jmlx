# Independent Module Releases Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `jmlx-jinja` and `jmlx-tokenizer` independently releasable to Maven Central, each driven by its own `release.sh`.

**Architecture:** Publishing wiring (the `se.alipsa.nexus-release-plugin`, `signing`, `release` → `check` ordering, and a SNAPSHOT-dependency guard) lives in the root `build.gradle` and is applied reactively to any subproject that applies `maven-publish`. Each releasable module gets a byte-identical `release.sh` that derives its module name from its own directory, guards against publishing a SNAPSHOT, and delegates to the root Gradle wrapper.

**Tech Stack:** Gradle 9.x (Groovy DSL), `se.alipsa.nexus-release-plugin` 2.2.0, `maven-publish`, `signing`, Java 21 toolchain, Bash.

**Spec:** `req/plans/module-release-scripts-design.md`

## Global Constraints

- Both published modules target **Java 21** (bytecode major **65**). `jmlx-jinja` already overrides; `jmlx-tokenizer` gains the override in Task 1.
- Group is `se.alipsa` (inherited from root `allprojects`). Artifact ids are the module directory names.
- `jmlx-jinja` version is `0.6.0-SNAPSHOT`; `jmlx-tokenizer` becomes `0.1.0-SNAPSHOT`. The root stays `0.5.0-SNAPSHOT` for the unpublished modules.
- Signing must remain **inert when `signing.keyId` is absent**, so `check`, CI, and `releaseVerification` stay keyless.
- `jmlx-jinja`'s existing `verifyPublicationMetadata` (asserts **zero** published dependencies) must remain untouched and passing.
- The repo runs with `org.gradle.configuration-cache=false` (see `gradle.properties`); do not re-enable it.
- Style: Google Java Style via Spotless/Checkstyle for Java. Build scripts use 4-space indent, matching the existing root `build.gradle`.
- No automated version bumping and no CI publishing — release is a local, credentialed, manual action.

---

### Task 1: Target Java 21 in `jmlx-tokenizer`

**Files:**
- Modify: `jmlx-tokenizer/build.gradle`

**Interfaces:**
- Consumes: nothing.
- Produces: `jmlx-tokenizer` compiled at bytecode major 65. Later tasks publish these classes.

**Why:** `jmlx-tokenizer` overrides no toolchain, so it inherits the root's Java 25. Publishing it unchanged ships a Java-25-only artifact that consumers of `jmlx-jinja` (Java 21) could not use. Nothing requires 25 here — it exists for `jmlx-core`/`jmlx-ffi`, which need Panama FFM.

- [ ] **Step 1: Record the current (wrong) bytecode level**

```bash
./gradlew :jmlx-tokenizer:compileJava --rerun-tasks
od -An -tu1 -j6 -N2 jmlx-tokenizer/build/classes/java/main/se/alipsa/jmlx/tokenizer/HfTokenizer.class \
  | awk '{print "major=" $1*256+$2}'
```
Expected: `major=69` (Java 25). This is the failing state.

- [ ] **Step 2: Add the toolchain override**

In `jmlx-tokenizer/build.gradle`, insert between the `plugins` block and `dependencies`:

```groovy
// Pure Java with no native dependency, so it is not bound to the root's Java 25 toolchain
// (which exists for jmlx-core/jmlx-ffi's Panama FFM). Targeting 21 matches jmlx-jinja and
// keeps both published artifacts consumable from Java 21 onward.
java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(21)
    }
}
```

- [ ] **Step 3: Verify the bytecode level changed and nothing broke**

```bash
./gradlew :jmlx-tokenizer:build --rerun-tasks
od -An -tu1 -j6 -N2 jmlx-tokenizer/build/classes/java/main/se/alipsa/jmlx/tokenizer/HfTokenizer.class \
  | awk '{print "major=" $1*256+$2}'
grep -ho 'tests="[0-9]*"' jmlx-tokenizer/build/test-results/test/*.xml | sed 's/[^0-9]//g' \
  | awk '{s+=$1} END {print "tests:", s}'
grep -ho 'failures="[0-9]*"' jmlx-tokenizer/build/test-results/test/*.xml | sed 's/[^0-9]//g' \
  | awk '{s+=$1} END {print "failures:", s}'
```
Expected: `major=65`, `tests: 133`, `failures: 0`, BUILD SUCCESSFUL.

- [ ] **Step 4: Commit**

```bash
git add jmlx-tokenizer/build.gradle
git commit -m "Target Java 21 in jmlx-tokenizer to match jmlx-jinja"
```

---

### Task 2: Wire the release plugin and signing in the root build

**Files:**
- Modify: `build.gradle` (root) — `plugins` block at lines 9-11, and a new block appended after the existing `subprojects { ... }` block which ends at line 136

**Interfaces:**
- Consumes: Task 1's module (no direct coupling).
- Produces: a `release` task on every subproject applying `maven-publish`, ordered after `check`. Task 3 adds a dependency to this same `release` task; Task 5's `release.sh` invokes it.

**Why:** `jmlx-jinja` has `maven-publish` but no `signing` and no `release` task, so it cannot publish at all today. Applied reactively via `plugins.withId('maven-publish')` so a module opts in simply by applying `maven-publish`.

- [ ] **Step 1: Confirm no `release` task exists yet**

```bash
./gradlew :jmlx-jinja:tasks --all 2>/dev/null | grep -E "^release " || echo "NO release task"
```
Expected: `NO release task`. This is the failing state.

- [ ] **Step 2: Declare the plugin in the root `plugins` block**

Change the root `build.gradle` `plugins` block to:

```groovy
plugins {
    id 'com.diffplug.spotless' version '8.9.0' apply false
    id 'se.alipsa.nexus-release-plugin' version '2.2.0' apply false
}
```

- [ ] **Step 3: Add the publishing-module wiring**

Append to the root `build.gradle`, after the existing `subprojects { ... }` block:

```groovy
// Release wiring for publishable modules only. A module opts in by applying 'maven-publish';
// jmlx-core/jmlx-ffi/jmlx-examples do not, and are unaffected. Kept here rather than duplicated
// per module so there is one copy to keep correct -- the same reason Spotless is applied above.
subprojects {
    plugins.withId('maven-publish') {
        apply plugin: 'signing'
        apply plugin: 'se.alipsa.nexus-release-plugin'

        // The 'maven' publication is declared in the module's own build.gradle, which Gradle
        // evaluates after this block runs, so the publication cannot be referenced eagerly here.
        afterEvaluate {
            // Central rejects unsigned release bundles, but signing must stay inert without a
            // configured key so `check`, the jmlx-jinja CI workflow, and releaseVerification all
            // remain keyless. Making this unconditional would break all three.
            signing {
                required = providers.gradleProperty('signing.keyId').present
                sign publishing.publications.maven
            }

            nexusReleasePlugin {
                userName = providers.gradleProperty('sonatypeUsername').orNull
                password = providers.gradleProperty('sonatypePassword').orNull
                mavenPublication = publishing.publications.maven
            }
        }

        // release.sh requests `build` and `release` as independent tasks, and Gradle imposes no
        // ordering between independently requested tasks -- without this edge, publishing could
        // begin before verification finished.
        tasks.named('release') {
            dependsOn tasks.named('check')
        }
    }
}
```

**If `afterEvaluate` proves awkward** (for example the nexus extension is not resolvable at that
point), the documented fallback is to move only the `signing { }` and `nexusReleasePlugin { }`
blocks into each module's own `build.gradle` — where hfjinja has them, and where the publication
already exists — leaving plugin application and the `release`/`check` edge in the root. Record
the reason in a comment if you take the fallback.

- [ ] **Step 4: Verify the `release` task now exists and is ordered after `check`**

```bash
./gradlew :jmlx-jinja:tasks --all 2>/dev/null | grep -E "^release "
./gradlew :jmlx-jinja:release --dry-run 2>&1 | grep -E ":jmlx-jinja:(check|test|release)"
```
Expected: a `release` task is listed, and the dry run shows `:jmlx-jinja:check` scheduled before `:jmlx-jinja:release`.

- [ ] **Step 5: Verify signing stays inert without a key, and unpublished modules are untouched**

```bash
./gradlew :jmlx-jinja:check :jmlx-tokenizer:check
./gradlew :jmlx-core:tasks --all 2>/dev/null | grep -E "^release " || echo "jmlx-core has no release task (correct)"
```
Expected: both checks pass with no signing/GPG error, and `jmlx-core` has no `release` task.

- [ ] **Step 6: Commit**

```bash
git add build.gradle
git commit -m "Wire nexus-release-plugin and signing for publishable modules"
```

---

### Task 3: Add the SNAPSHOT-dependency guard

**Files:**
- Modify: `build.gradle` (root) — inside the `plugins.withId('maven-publish')` block added in Task 2

**Interfaces:**
- Consumes: Task 2's `plugins.withId('maven-publish')` block and its `release` task.
- Produces: task `verifyNoSnapshotDependencies` on every publishing module, wired into `release`.

**Why:** `jmlx-tokenizer` keeps `api project(':jmlx-jinja')`, which Gradle publishes as the concrete coordinate `se.alipsa:jmlx-jinja:<jinja's current version>`. Releasing tokenizer while jinja is a SNAPSHOT would emit a POM Central rejects on upload. This fails fast locally instead, and is what enforces jinja-before-tokenizer ordering.

- [ ] **Step 1: Add the guard task**

Inside the `plugins.withId('maven-publish') { ... }` block from Task 2, before the
`tasks.named('release')` block, add:

```groovy
        // Central rejects POMs referencing SNAPSHOT dependencies. jmlx-tokenizer depends on
        // jmlx-jinja via project(':jmlx-jinja'), which Gradle publishes as a concrete coordinate
        // carrying jinja's current version -- so this is what forces jinja to be released first.
        tasks.register('verifyNoSnapshotDependencies') {
            group = 'verification'
            description = 'Fails if the generated POM references any -SNAPSHOT dependency.'
            dependsOn tasks.named('generatePomFileForMavenPublication')
            def pomFile = layout.buildDirectory.file('publications/maven/pom-default.xml')
            inputs.file(pomFile)
            doLast {
                def pom = javax.xml.parsers.DocumentBuilderFactory.newInstance()
                        .newDocumentBuilder().parse(pomFile.get().asFile)
                def dependencyNodes = pom.getElementsByTagName('dependency')
                def offenders = []
                for (int i = 0; i < dependencyNodes.length; i++) {
                    def dependency = dependencyNodes.item(i)
                    def field = { String name ->
                        dependency.childNodes.find {
                            it.nodeType == org.w3c.dom.Node.ELEMENT_NODE && it.nodeName == name
                        }?.textContent
                    }
                    if (field('version')?.endsWith('-SNAPSHOT')) {
                        offenders << "${field('groupId')}:${field('artifactId')}:${field('version')}"
                    }
                }
                if (!offenders.isEmpty()) {
                    throw new GradleException(
                            "Cannot release ${project.name}: POM references SNAPSHOT dependencies: " +
                            "${offenders.join(', ')}. Release those modules first.")
                }
            }
        }
```

- [ ] **Step 2: Wire it into `release`**

Change the `tasks.named('release')` block from Task 2 to:

```groovy
        tasks.named('release') {
            dependsOn tasks.named('check'), tasks.named('verifyNoSnapshotDependencies')
        }
```

- [ ] **Step 3: Verify it passes for `jmlx-jinja` (which has zero dependencies)**

```bash
./gradlew :jmlx-jinja:verifyNoSnapshotDependencies
```
Expected: BUILD SUCCESSFUL — jinja publishes no dependencies, so there is nothing to offend.

- [ ] **Step 4: Verify it fires when a SNAPSHOT dependency is present**

This is the real test. It cannot run until Task 4 gives `jmlx-tokenizer` a publication, so
record it here and execute it as Task 4 Step 6. Do not skip it.

- [ ] **Step 5: Commit**

```bash
git add build.gradle
git commit -m "Add verifyNoSnapshotDependencies guard to the release path"
```

---

### Task 4: Add publishing configuration to `jmlx-tokenizer`

**Files:**
- Modify: `jmlx-tokenizer/build.gradle`

**Interfaces:**
- Consumes: Task 2's root wiring (signing, nexus plugin, `release`), Task 3's `verifyNoSnapshotDependencies`.
- Produces: a `maven` `MavenPublication` for `se.alipsa:jmlx-tokenizer:0.1.0-SNAPSHOT`, plus task `verifyPublishedDependencies`.

**Why:** `jmlx-tokenizer` has no publishing configuration at all and no version of its own — it silently inherits the root's `0.5.0-SNAPSHOT`. It starts a fresh `0.1.0` line because it has never been published; tracking jinja's `0.6.x` would falsely imply shared history.

- [ ] **Step 1: Confirm there is no publication yet**

```bash
./gradlew :jmlx-tokenizer:generatePomFileForMavenPublication 2>&1 | tail -3
```
Expected: failure — task not found. This is the failing state.

- [ ] **Step 2: Rewrite `jmlx-tokenizer/build.gradle`**

```groovy
plugins {
    id 'java-library'
    id 'maven-publish'
}

// Its own version line, deliberately independent of the root's 0.5.0-SNAPSHOT: an
// "independent release" that inherits the root version is not independent at all. Starts a
// fresh 0.1.0 line because jmlx-tokenizer has never been published.
version = '0.1.0-SNAPSHOT'

// Pure Java with no native dependency, so it is not bound to the root's Java 25 toolchain
// (which exists for jmlx-core/jmlx-ffi's Panama FFM). Targeting 21 matches jmlx-jinja and
// keeps both published artifacts consumable from Java 21 onward.
java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(21)
    }
    withSourcesJar()
    withJavadocJar()
}

dependencies {
    implementation libs.jackson.databind
    api project(':jmlx-jinja')
}

publishing {
    publications {
        maven(MavenPublication) {
            from components.java
            pom {
                name = 'jmlx-tokenizer'
                description = 'Pure-Java byte-level BPE tokenizer and Hugging Face chat-template rendering.'
                url = 'https://github.com/Alipsa/jmlx'
                licenses {
                    license {
                        name = 'MIT License'
                        url = 'https://opensource.org/license/mit'
                    }
                }
                developers {
                    developer {
                        id = 'pernyfelt'
                        name = 'Per Nyfelt'
                    }
                }
                scm {
                    connection = 'scm:git:https://github.com/Alipsa/jmlx.git'
                    developerConnection = 'scm:git:ssh://git@github.com:Alipsa/jmlx.git'
                    url = 'https://github.com/Alipsa/jmlx'
                }
            }
        }
    }
}

// jmlx-jinja's verifyPublicationMetadata asserts *zero* published dependencies; jmlx-tokenizer
// legitimately has two, so it asserts their exact identity instead. Deliberately different
// checks -- conflating them would weaken jinja's stronger guarantee.
tasks.register('verifyPublishedDependencies') {
    group = 'verification'
    description = 'Verifies the published POM declares exactly the expected dependencies.'
    dependsOn tasks.named('generatePomFileForMavenPublication')
    def pomFile = layout.buildDirectory.file('publications/maven/pom-default.xml')
    def expected = ['se.alipsa:jmlx-jinja', 'tools.jackson.core:jackson-databind']
    inputs.file(pomFile)
    inputs.property('expected', expected)
    doLast {
        def pom = javax.xml.parsers.DocumentBuilderFactory.newInstance()
                .newDocumentBuilder().parse(pomFile.get().asFile)
        def dependencyNodes = pom.getElementsByTagName('dependency')
        def actual = []
        for (int i = 0; i < dependencyNodes.length; i++) {
            def dependency = dependencyNodes.item(i)
            def field = { String name ->
                dependency.childNodes.find {
                    it.nodeType == org.w3c.dom.Node.ELEMENT_NODE && it.nodeName == name
                }?.textContent
            }
            actual << "${field('groupId')}:${field('artifactId')}".toString()
        }
        if (actual.toSorted() != expected.toSorted()) {
            throw new GradleException(
                    "Published dependencies changed. Expected ${expected.toSorted()}, " +
                    "found ${actual.toSorted()}.")
        }
    }
}

// Decision 5's compatibility promise is only worth anything if it cannot silently regress: if
// this override is ever removed, the module falls back to the root's Java 25 and the published
// artifact stops working for Java 21 consumers. jmlx-jinja pins the same invariant via its
// release-verification contract; jmlx-tokenizer has no module-info.java, so it asserts the
// bytecode level directly instead.
tasks.register('verifyBytecodeLevel') {
    group = 'verification'
    description = 'Verifies compiled classes are Java 21 (bytecode major 65).'
    dependsOn tasks.named('compileJava')
    def classesDir = layout.buildDirectory.dir('classes/java/main')
    inputs.dir(classesDir)
    doLast {
        def offenders = []
        classesDir.get().asFile.eachFileRecurse { file ->
            if (file.name.endsWith('.class')) {
                file.withDataInputStream { input ->
                    input.readInt()          // 0xCAFEBABE
                    input.readUnsignedShort() // minor
                    int major = input.readUnsignedShort()
                    if (major != 65) {
                        offenders << "${file.name}: major ${major}".toString()
                    }
                }
            }
        }
        if (!offenders.isEmpty()) {
            throw new GradleException(
                    "Expected Java 21 bytecode (major 65) but found: ${offenders.join(', ')}")
        }
    }
}

tasks.named('check') {
    dependsOn tasks.named('verifyPublishedDependencies'), tasks.named('verifyBytecodeLevel')
}
```

- [ ] **Step 3: Verify the POM is generated with the right coordinates and metadata**

```bash
./gradlew :jmlx-tokenizer:generatePomFileForMavenPublication
cat jmlx-tokenizer/build/publications/maven/pom-default.xml
```
Expected: `<groupId>se.alipsa</groupId>`, `<artifactId>jmlx-tokenizer</artifactId>`,
`<version>0.1.0-SNAPSHOT</version>`, non-empty `name`/`description`/`url`, an MIT `licenses`
block, a `developers` block, an `scm` block, and exactly two `<dependency>` entries —
`se.alipsa:jmlx-jinja` and `tools.jackson.core:jackson-databind`.

- [ ] **Step 4: Verify the dependency-set assertion passes**

```bash
./gradlew :jmlx-tokenizer:verifyPublishedDependencies
```
Expected: BUILD SUCCESSFUL.

- [ ] **Step 5: Verify the dependency-set assertion actually fires**

Temporarily add `implementation 'org.slf4j:slf4j-api:2.0.16'` to `jmlx-tokenizer/build.gradle`'s
`dependencies` block, then:

```bash
./gradlew :jmlx-tokenizer:verifyPublishedDependencies --rerun-tasks 2>&1 | grep -E "Published dependencies changed|BUILD"
```
Expected: FAIL with `Published dependencies changed.` **Remove the temporary dependency and
re-run to confirm it passes again before continuing.**

- [ ] **Step 6: Verify the SNAPSHOT guard from Task 3 actually fires**

`jmlx-jinja` is `0.6.0-SNAPSHOT`, so tokenizer's POM references a SNAPSHOT dependency:

```bash
./gradlew :jmlx-tokenizer:verifyNoSnapshotDependencies --rerun-tasks 2>&1 \
  | grep -E "SNAPSHOT dependencies|BUILD"
```
Expected: FAIL, naming `se.alipsa:jmlx-jinja:0.6.0-SNAPSHOT` and instructing that it be released
first. This is the release-ordering guarantee working as designed.

Now prove the other direction — that it *passes* once jinja is a release version:

```bash
cp jmlx-jinja/build.gradle /tmp/jinja.bak
sed -i '' "s/^version = '0.6.0-SNAPSHOT'$/version = '0.6.0'/" jmlx-jinja/build.gradle
./gradlew :jmlx-tokenizer:verifyNoSnapshotDependencies --rerun-tasks 2>&1 | grep -E "BUILD"
cp /tmp/jinja.bak jmlx-jinja/build.gradle && rm /tmp/jinja.bak
git diff --stat jmlx-jinja/build.gradle
```
Expected: BUILD SUCCESSFUL while jinja is `0.6.0`, and `git diff --stat` empty afterwards
confirming the file was restored. A guard that only ever fails is indistinguishable from a
broken guard, so both directions must be observed.

- [ ] **Step 7: Verify the bytecode assertion works in both directions**

```bash
./gradlew :jmlx-tokenizer:verifyBytecodeLevel
```
Expected: BUILD SUCCESSFUL (major 65 from Task 1).

Then confirm it catches a regression by temporarily deleting the `java { toolchain { ... } }`
block from `jmlx-tokenizer/build.gradle`:

```bash
cp jmlx-tokenizer/build.gradle /tmp/tok2.bak
python3 - <<'PY'
import pathlib, re
p = pathlib.Path('jmlx-tokenizer/build.gradle')
s = p.read_text()
s = re.sub(r"    toolchain \{\n        languageVersion = JavaLanguageVersion\.of\(21\)\n    \}\n", "", s)
p.write_text(s)
PY
./gradlew :jmlx-tokenizer:verifyBytecodeLevel --rerun-tasks 2>&1 | grep -E "major 69|BUILD"
cp /tmp/tok2.bak jmlx-tokenizer/build.gradle && rm /tmp/tok2.bak
git diff --stat jmlx-tokenizer/build.gradle
```
Expected: FAIL reporting `major 69`, then an empty `git diff --stat` after restoring.

- [ ] **Step 8: Verify the whole build still passes**

```bash
./gradlew build
```
Expected: BUILD SUCCESSFUL. Note `:jmlx-tokenizer:check` now includes
`verifyPublishedDependencies` but **not** `verifyNoSnapshotDependencies` (which is release-only,
and would fail the ordinary build while jinja is a SNAPSHOT).

- [ ] **Step 9: Commit**

```bash
git add jmlx-tokenizer/build.gradle
git commit -m "Publish jmlx-tokenizer as se.alipsa:jmlx-tokenizer:0.1.0-SNAPSHOT"
```

---

### Task 5: Add `release.sh` to both modules

**Files:**
- Create: `jmlx-jinja/release.sh`
- Create: `jmlx-tokenizer/release.sh`

**Interfaces:**
- Consumes: the `release` task from Task 2 and the guards from Tasks 3-4.
- Produces: the operator entry point for a release.

**Why:** hfjinja's `release.sh` cannot work here — it does `cd "$SCRIPT_DIR"` then `./gradlew`, but no module directory in this monorepo has a wrapper. Deriving the module from the directory name keeps the two copies byte-identical.

- [ ] **Step 1: Write the script**

Create `jmlx-jinja/release.sh` with exactly this content:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Byte-identical in every releasable module: the module name is derived from this script's own
# directory, so there is nothing module-specific to keep in sync. The Gradle wrapper lives only
# at the monorepo root, so we run from there with a qualified task path.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODULE=$(basename "$SCRIPT_DIR")
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

# An independent release requires the module to carry its own version. Without this it silently
# inherits the root version and every module would move in lockstep -- the exact accident this
# tooling exists to prevent.
if ! grep -qE "^version[[:space:]]*=[[:space:]]*'" "$MODULE/build.gradle"; then
  echo "$MODULE/build.gradle declares no version of its own; it would inherit the root version." >&2
  exit 1
fi

# Ask Gradle for the effective version rather than parsing build.gradle: authoritative, and not
# fragile to formatting.
VERSION=$(./gradlew -q ":$MODULE:properties" | sed -n 's/^version: //p' | head -n 1)
if [[ -z "$VERSION" ]]; then
  echo "Could not determine the version of $MODULE" >&2
  exit 1
fi
if [[ "$VERSION" == *-SNAPSHOT ]]; then
  echo "$MODULE $VERSION is a snapshot and cannot be published" >&2
  exit 1
fi

echo "Building and validating the release bundle for $MODULE $VERSION before publishing to Maven Central..."
./gradlew ":$MODULE:clean" ":$MODULE:build" ":$MODULE:release"
echo "$MODULE $VERSION published"
echo "see https://central.sonatype.com/publishing/deployments for more info"
```

- [ ] **Step 2: Copy it verbatim and make both executable**

```bash
cp jmlx-jinja/release.sh jmlx-tokenizer/release.sh
chmod +x jmlx-jinja/release.sh jmlx-tokenizer/release.sh
diff jmlx-jinja/release.sh jmlx-tokenizer/release.sh && echo "identical (correct)"
```
Expected: `identical (correct)`.

- [ ] **Step 3: Verify the SNAPSHOT guard stops both**

```bash
./jmlx-jinja/release.sh; echo "jinja exit=$?"
./jmlx-tokenizer/release.sh; echo "tokenizer exit=$?"
```
Expected: both print `... is a snapshot and cannot be published` and exit non-zero. **Nothing
should be published** — both modules are currently SNAPSHOTs.

- [ ] **Step 4: Verify the own-version guard fires**

```bash
cp jmlx-tokenizer/build.gradle /tmp/tok.bak
sed -i '' "/^version = '0.1.0-SNAPSHOT'$/d" jmlx-tokenizer/build.gradle
./jmlx-tokenizer/release.sh; echo "exit=$?"
cp /tmp/tok.bak jmlx-tokenizer/build.gradle && rm /tmp/tok.bak
```
Expected: `declares no version of its own` and a non-zero exit. Confirm the file is restored
(`git diff --stat jmlx-tokenizer/build.gradle` shows nothing) before continuing.

- [ ] **Step 5: Commit**

```bash
git add jmlx-jinja/release.sh jmlx-tokenizer/release.sh
git commit -m "Add per-module release.sh for independent releases"
```

---

### Task 6: Document the release process

**Files:**
- Modify: `CLAUDE.md`
- Modify: `req/plans/module-release-scripts-design.md` (status line only)

**Interfaces:**
- Consumes: everything above.
- Produces: no code.

**Why:** CLAUDE.md is the authoritative orientation document for this repo and currently says nothing about releasing. It also still describes `jmlx-tokenizer` without a version or publishing story.

- [ ] **Step 1: Add a release section to `CLAUDE.md`**

Add after the "Build, test, run" section:

```markdown
## Releasing a module

`jmlx-jinja` and `jmlx-tokenizer` are published to Maven Central independently of each other and
of the root project's version. Each has its own `release.sh`; the other three modules are not
published (`jmlx-core`/`jmlx-ffi` would need native-artifact packaging designed first, and
`jmlx-examples` is a demo).

```sh
./jmlx-jinja/release.sh        # publishes se.alipsa:jmlx-jinja
./jmlx-tokenizer/release.sh    # publishes se.alipsa:jmlx-tokenizer
```

Both scripts refuse to publish a `-SNAPSHOT`, and refuse a module that has no `version` line of
its own (which would mean it is silently riding the root version). To release: set the module's
version to a release value, run its `release.sh`, then bump it to the next `-SNAPSHOT`. Version
bumping is deliberately manual.

**Release order matters.** `jmlx-tokenizer` depends on `jmlx-jinja` via `api project(...)`, which
Gradle publishes as a concrete coordinate. `verifyNoSnapshotDependencies` fails the tokenizer
release while jinja is still a SNAPSHOT, so jinja must be released first.

Signing and Central credentials come from Gradle properties (`signing.keyId`,
`sonatypeUsername`, `sonatypePassword`), normally in `~/.gradle/gradle.properties`. Signing is
inert when `signing.keyId` is absent, so `check` and CI stay keyless. Releasing is a local,
credentialed, manual action — CI verifies but never publishes.
```

- [ ] **Step 2: Update the module descriptions in `CLAUDE.md`**

In the architecture section, immediately after the paragraph beginning
"**`jmlx-tokenizer` and `jmlx-jinja` sit outside the native chain entirely.**", add:

```markdown
Both are also the only **published** modules, and each carries its own version independent of the
root's `0.5.0-SNAPSHOT`: `jmlx-jinja` is `0.6.0-SNAPSHOT` (continuing the archived hfjinja
project's line) and `jmlx-tokenizer` is `0.1.0-SNAPSHOT`. Both override the toolchain to **Java
21** rather than inheriting the root's Java 25 — that 25 exists for `jmlx-core`/`jmlx-ffi`'s
Panama FFM, and neither pure-Java module needs it, so targeting 21 keeps the published artifacts
usable by Java 21 consumers. `jmlx-tokenizer`'s `verifyBytecodeLevel` task enforces this. See
"Releasing a module" above.
```

- [ ] **Step 3: Mark the design doc implemented**

In `req/plans/module-release-scripts-design.md`, change the status line to:

```markdown
Status: implemented. Date: 2026-09-01.
```

- [ ] **Step 4: Verify the docs match reality**

```bash
grep -n "release.sh" CLAUDE.md
./gradlew build
```
Expected: the release section is present, and BUILD SUCCESSFUL.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md req/plans/module-release-scripts-design.md
git commit -m "Document the per-module release process"
```

---

## Post-implementation verification

Run once at the end, before opening the PR:

```bash
./gradlew build                                   # whole repo green
./gradlew :jmlx-jinja:check :jmlx-tokenizer:check # both publishable modules
./gradlew :jmlx-jinja:verifyNoSnapshotDependencies        # passes: jinja has no deps
./gradlew :jmlx-tokenizer:verifyNoSnapshotDependencies || echo "correctly blocked"
diff jmlx-jinja/release.sh jmlx-tokenizer/release.sh      # identical
```

Confirm `jmlx-jinja`'s pre-existing `verifyPublicationMetadata` and `verifyModuleDescriptor`
still pass — this work must not weaken jinja's stronger zero-dependency guarantee.
