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
credentials/init scripts, configuration/build caches disabled and task reruns. Gradle itself
also launches on the extracted comparison JDK. For each module it runs `jar`, `sourcesJar`,
`javadocJar`, `generatePomFileForMavenPublication` and
`generateMetadataFileForMavenPublication`. All 27 selected tasks execute successfully.
No signing, Maven-local publication or Portal submission runs.

All 15 historical artifacts are freshly downloaded from
`https://repo.maven.apache.org/maven2/se/alipsa/<module>/<version>/<filename>`.
The rebuilt artifacts are copied into a separate flat directory using identical filenames.
Run the committed comparison tool from the repository root:

```sh
python3 tools/release-all/compare_publications.py \
  --central build/releaseAll/pure-java-audit-2026-11-16/central \
  --candidate build/releaseAll/pure-java-audit-2026-11-16/candidate \
  --output build/releaseAll/pure-java-audit-2026-11-16/comparison.json
```

The tool emits comparison facts; successful execution alone does not mean artifacts match.
Its `equal` fields determine the result. Reproduction requires the isolated rebuild and fresh
downloads above; `build-command.json` in the retained evidence records the exact invocation.

## Normalization and limits

JAR entries are sorted by name and serialized with fixed 1980-01-01 timestamps and stored
compression, preserving every entry name and uncompressed byte, including directory entries.
Duplicate entry names fail. No Javadoc timestamps, banners, bytecode, API, resource, notice or
license content is rewritten. All nine JARs also match before any normalization.

POMs are compared byte-for-byte. For `.module`, the tool first verifies the original referenced
JAR sizes and all recorded checksums. It then substitutes only those size/checksum value tokens
with values recomputed from the canonical JAR bytes. Every other metadata byte is preserved,
including dependency versions, Gradle provenance and JSON formatting. Stale hashes fail instead
of being overwritten and accepted.

Seven regression tests cover archive ordering/timestamps and derived metadata, dependency and
Gradle provenance changes, class/Javadoc changes, POM/module formatting, stale sizes/checksums
and missing artifacts. They run in the ordinary release selftest suite.
`bash tools/release-all/selftest.sh` passed all 31 tests locally in 54.095 seconds, including
the cold-cache published-plugin TLS fixture. `git diff --check` passed.

The tuple still needs human review and minimum-host readiness before a candidate is frozen.
Repeat this comparison at the final candidate SHA, and rerun it after dependency, wrapper,
plugin or publication changes. This audit supplies neither physical macOS 26.2 execution nor
native-license/account-capacity review, final release verification or publication authorization.
