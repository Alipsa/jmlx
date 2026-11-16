# Phase 8.4 pure-Java artifact audit

2026-11-16. Audited source: `98190d44fd725b484e98e5e8b7b41efa925e0b23`
(PR #97 merge), against the published 8.1 baseline
`9bc85042df50ed3f979cb8d3e7c8767917c71483`. This is the post-plugin and
post-Jackson-update audit, not final candidate or signed-bundle evidence.
The [machine-readable report](phase8-4-pure-java-audit.json) records every artifact's raw
and normalized SHA-256, the exact build argument vector, JDK provenance and retained logs.

## Result and proposed tuple

| Module rebuilt at its old version | Main/sources/Javadoc JARs | POM | Gradle module metadata | Decision |
| --- | --- | --- | --- | --- |
| jinja 0.6.1 | All byte-identical to Central | Byte-identical | `createdBy.gradle.version`: 9.7.0 → 9.9.0 | New 0.6.2 |
| tokenizer 0.1.1 | All byte-identical to Central | Jackson 3.1.2 → 3.3.0 | Jackson and Gradle version differences | New 0.1.2 |
| vision 0.1.1 | All byte-identical to Central | Jackson 3.1.2 → 3.3.0 | Jackson and Gradle version differences | New 0.1.2 |

The accepted plan preserves publication metadata differences, including the Gradle producer
version. Equal JARs therefore do not permit reuse of these coordinates. The existing manifest's
three new pure-Java versions and `local` source assignments remain supported; its six first
native/framework releases are unchanged by this audit. When cutting release versions, tokenizer
must depend on the new jinja 0.6.2 coordinate. Its old 0.6.1 edge in the comparison is intentional:
all three old versions were restored together to isolate non-version changes.

There are no differences in the three `src/main` trees against 8.1. Their module build-file
changes are version/comment bookkeeping. The shared catalog changes Jackson's runtime edge;
Gradle changes the publication producer metadata. Other shared build changes concern native
modules, adapters, tests, formatting and release tooling. The observed artifacts contain no
additional pure-Java content changes. The relevant full source patch and POM/module diffs
are retained with the audit evidence.

## Toolchain and procedure

The retained 8.1 `jmlx-jinja/build/reports/release-verification.md` identifies BellSoft
21.0.12.1 and the baseline SHA. Its recorded main-JAR hash matches the fresh Central download.
Unlike that report, the published Javadoc banner establishes only feature version 21;
class-file major 65 likewise establishes the bytecode target, not compiler vendor or patch.
The historical report does not record a build number. The still-installed historical JDK
reports `21.0.12.1+1-LTS`.

The comparison uses a fresh download of BellSoft's matching macOS ARM64 JDK:

```text
https://download.bell-sw.com/java/21.0.12.1+1/bellsoft-jdk21.0.12.1+1-macos-aarch64.tar.gz
SHA-256: 0fffffac0fe6ebe7af583bdb15f8725e4873b874dd648cbf485e53fc2b4d9218
```

This is the locally computed digest of the archive downloaded from the vendor over HTTPS.
Compiler and Javadoc selections were recorded from Gradle in the same successful build;
both resolve to that extracted distribution and report BellSoft `21.0.12.1+1-LTS` for all
three modules. The normal checkout independently selects its installed BellSoft
`21.0.12.1+1-LTS` for these tools too. No compiler/Javadoc identity differences are normalized.

An isolated clone starts at the audited SHA. Only its three version declarations are changed
to 0.6.1/0.1.1/0.1.1; the main checkout retains its SNAPSHOT versions. The comparison runs
Gradle 9.9.0 with plugin 2.6.0, Jackson 3.3.0, a separate Gradle home containing no user
Gradle properties/init scripts, configuration/build caches disabled and task reruns. JAVA_HOME points
to the extracted comparison JDK; the original log does not independently record the daemon JVM.
Review reproductions record its java.home, vendor and runtime version explicitly (see below). For each module it runs `jar`, `sourcesJar`,
`javadocJar`, `generatePomFileForMavenPublication` and
`generateMetadataFileForMavenPublication`. All 27 selected tasks execute successfully.
No signing, Maven-local publication or Portal submission runs. The original invocation inherited
the caller environment apart from its recorded JAVA_HOME/PATH overrides; those other values
were not captured. The reproduction policy below does not describe that original environment.

All 15 historical artifacts are freshly downloaded from
`https://repo.maven.apache.org/maven2/se/alipsa/<module>/<version>/<filename>`.
The [committed evidence directory](plans/evidence/pure-java-audit-2026-11-16/) contains
the toolchain init scripts, version patch, normalized build command, all 15 explicit
copy-and-rename mappings, metadata diff, source diff, logs and comparison report. These files
survive `./gradlew clean`. Log paths are replaced with `$AUDIT`, `$REPO`, `$INSTALLED_JDKS`
and `$GRADLE_USER_HOME`; the historical candidate temporary path is `$HISTORICAL_CANDIDATE`.
The machine-readable report hashes these committed representations, rather than the original
machine-specific logs. Large downloaded artifacts and the JDK remain disposable build outputs.
`comparison-toolchains.log` is an earlier installed-JDK probe; the successful build's compiler
and Javadoc selections are in `build.log`.

To reproduce on macOS ARM64, download the BellSoft archive above. The recipe verifies its
recorded SHA-256 before extracting it into the workspace. Use a Gradle 9.9.0 executable and
an unused workspace outside `build/`. Here `python3` must provide `tarfile.data_filter`:
use Python 3.12+ or a patched Python 3.11.4+. The general release-tooling minimum remains
Python 3.11+; earlier 3.11 builds can use `--jdk` but cannot safely extract archives.
Stock macOS Python 3.9.6 does not meet the tooling minimum and lacks this extraction feature.

```sh
python3 req/plans/evidence/pure-java-audit-2026-11-16/reproduce.py \
  --workspace /tmp/jmlx-audit-reproduction \
  --jdk-archive /path/to/bellsoft-jdk21.0.12.1+1-macos-aarch64.tar.gz \
  --gradle /path/to/gradle-9.9.0/bin/gradle
```

Alternatively, `--jdk /path/to/existing/jdk` checks identity/version/path only; it cannot
establish that the installation came from the pinned archive. The selected provenance is
recorded in the workspace's `jdk-provenance.json`.

The recipe clones the recorded source SHA, applies only the committed old-version patch,
uses a fresh isolated Gradle home, reruns all publication tasks, copies each exact output
path recorded in `build-command.json`, downloads the 15 Central artifacts and compares them.
It refuses an existing workspace, checks the Gradle executable reports 9.9.0 and checks every
recorded daemon, compiler and Javadoc identity against the recorded BellSoft
`21.0.12.1+1-LTS` identities and selected JDK path. All Central downloads must match the
committed raw SHA-256 values. It then requires the complete comparison report (including every
artifact hash, change list and verdict) to equal the committed report, and exits nonzero on
any mismatch. The comparator runs with the recipe's own `sys.executable`.

The current reproduction build inherits only `HOME`, `PATH`, `TMPDIR` and `USER`, sets
`LC_ALL=C`, `LANG=C`, `LANGUAGE=C` and `TZ=UTC`, and points `JAVA_HOME` and the PATH prefix
to the selected JDK. Every other caller variable is dropped, including all
`ORG_GRADLE_PROJECT_*`, `ORG_GRADLE_*`, JVM options and other locale settings. The allow-list
and fixed values are recorded under `reproductionEnvironment` in `build-command.json` and
the audit JSON. The audit JSON's `build.environment` and `build-command.json`'s `environment` describe only the
original run's recorded JAVA_HOME/PATH overrides; accompanying notes distinguish these scopes.
The prior `portable-*` evidence used the earlier deny-list policy; the latest allow-list
and archive-verification evidence is recorded below. Git clone, checkout and patch steps use
the same allow-list plus `GIT_CONFIG_NOSYSTEM=1` and `GIT_CONFIG_GLOBAL=/dev/null`, excluding
caller `GIT_*` overrides and global/system config such as checkout filters or `core.autocrlf`.
The build does not inherit `HTTP_PROXY`, `HTTPS_PROXY` or `NO_PROXY`, so cold dependency
downloads require direct repository access.
The additional init script records the daemon JVM in `build.log`. Two `AUDIT_DAEMON` lines
are expected because the init script applies to the root build and `buildSrc`; both must
match. This recipe reproduces this dated development audit; a final-candidate audit requires
an explicitly updated SHA and reviewed version overrides.

For a standalone comparison of existing flat directories:

```sh
python3 tools/release-all/compare_publications.py \
  --central /tmp/jmlx-audit-reproduction/central \
  --candidate /tmp/jmlx-audit-reproduction/candidate \
  --publications req/plans/evidence/pure-java-audit-2026-11-16/publications.json \
  --output /tmp/jmlx-audit-reproduction/comparison.json
```

`--publications` requires the five exact filenames per module, including Javadoc. The report
contains top-level `allEqual` and per-artifact `equal` fields. Add `--require-equal` to fail
with exit 1 on differences, after writing the report. It is intentionally absent from this
audit's reproduction because five metadata files are expected to differ.

## Normalization and limits

JAR entries are sorted by name and serialized with fixed 1980-01-01 timestamps.
Canonical bytes use `ZIP_STORED`, preserving entry names and uncompressed bytes (including
directories), permissions/attributes, extra fields and entry/archive comments. No compression
runs during normalization, so normalized JAR hashes and the derived `.module` hashes do not
depend on the host's deflate implementation. The original compression method remains in the
metadata tuple compared for each entry, alongside attributes, extra fields, comments, flags
and creator/extractor versions. Thus a compression-method change still fails even if the
normalized hashes match. Use the verdict rather than normalized hashes alone.
Fields regenerated by Python's ZIP writer cannot hide a metadata change. `changedEntries`
includes content or metadata changes; archive-comment differences are reflected in the
artifact verdict and normalized hash.
Duplicate entry names fail. No Javadoc timestamps, banners, bytecode, API, resource, notice or
license content is rewritten. Only DOS entry timestamps are fixed; timestamps embedded in
extra fields remain compared verbatim. This is deliberately stricter than normalizing all
possible timestamp encodings; none of the current nine Central JARs has extra fields.
All nine JARs also match before any normalization.

POMs are compared byte-for-byte. For `.module`, the tool first verifies the original referenced
JAR sizes and all recorded checksums. It then substitutes only those size/checksum value tokens
with values recomputed from the canonical JAR bytes. Every other metadata byte is preserved,
including dependency versions, Gradle provenance and JSON formatting. Stale hashes fail instead
of being overwritten and accepted.

The original run passed 31 release selftests in 54.095 seconds. Review expanded the comparison
suite from seven to fifteen tests: ZIP metadata/comments, duplicate entries, unexpected URL
records, unsupported nested artifacts, malformed metadata, extra/missing artifacts, complete
five-file sets, CLI verdicts, and Javadoc changes through `compare()` are covered alongside
the existing dependency/provenance, formatting and stale-hash checks. Review validation and
fresh reproduction results are recorded below and in the machine-readable report.

Review validation on 2026-11-16 passed all 39 release selftests in 72.869 seconds, including
the cold-cache published-plugin TLS fixture, and `git diff --check`. The committed recipe was
run in a new temporary clone and Gradle home: all 27 tasks executed in 37 seconds, and all
15 artifact comparison records matched in a manual same-host check with that comparator.
Its deflated normalized hashes have since been replaced with stored-byte hashes in the
committed report. Nine raw JAR matches and five metadata differences remain. The recorded
`AUDIT_DAEMON` identifies BellSoft `21.0.12.1+1-LTS` at the extracted comparison JDK path.
The review build and selftest logs are committed alongside the original evidence.

The subsequent portability correction passed 46 release selftests in 75.423 seconds:
17 comparison tests and five reproduction-guard tests now cover compressor-free normalization
(including different deflate encodings), malformed publication-list shapes, wrong toolchains,
Gradle versions, Central downloads, altered results and environment controls. The strengthened
recipe ran successfully in another fresh clone and Gradle home, executing all 27 tasks in
35 seconds. It automatically verified Gradle 9.9.0, every daemon/compiler/Javadoc identity,
all 15 Central raw hashes and the entire committed stored-byte comparison report.
The nine identical JARs and five metadata differences remain. `portable-build.log`,
`portable-gradle-version.log` and `portable-selftest.log` retain this evidence; the JSON
hashes cover all committed evidence files. `git diff --check` passed.

The allow-list correction passed all 48 release selftests in 79.713 seconds. Seven
reproduction-guard tests now check the environment policy against both committed records,
including injected Gradle properties, and verify archive digest rejection, archive structure
and safe extraction. The test import temporarily disables bytecode writing; the evidence
directory has no `__pycache__` after the suite. A fresh archive-mode reproduction verified
SHA-256 `0fffffac0fe6ebe7af583bdb15f8725e4873b874dd648cbf485e53fc2b4d9218`, extracted
that JDK into its workspace and executed all 27 publication tasks in 40 seconds. Toolchain,
Central hash and complete result checks passed with the same nine raw JAR matches and five
metadata differences. The `allowlist-*` logs and JDK provenance record retain this run;
evidence digests were refreshed and `git diff --check` passed.

The Python/Git compatibility correction passed all 50 release selftests in 77.781 seconds.
Nine reproduction-guard tests now cover the missing-extraction-feature message and ignored
caller Git config/environment. Targeted tests also ran under stock macOS Python 3.9.6:
seven passed and the two archive tests skipped, without the former TypeError/AttributeError.
This targeted compatibility check does not lower the general tooling minimum. A fresh
archive-mode reproduction with isolated Git settings executed all 27 publication tasks in
40 seconds and passed every toolchain, Central hash and complete-report check.
`git-isolation-*` and `compatibility-*` retain these results. Evidence digests and
`git diff --check` passed.

The tuple still needs human review and minimum-host readiness before a candidate is frozen.
Repeat this comparison at the final candidate SHA, and rerun it after dependency, wrapper,
plugin or publication changes. This audit supplies neither physical macOS 26.2 execution nor
native-license/account-capacity review, final release verification or publication authorization.
