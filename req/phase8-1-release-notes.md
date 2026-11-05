# Phase 8.1 release notes: jmlx-jinja 0.6.1, jmlx-tokenizer 0.1.1, jmlx-vision 0.1.1

**Status:** provisional. The initial `0.6.0` / `0.1.0` / `0.1.0` cut (release SHA `6097d8f`,
published 2026-11-04) is **superseded** (see Recovery below); the recovery re-release
`0.6.1` / `0.1.1` / `0.1.1` is cut and locally verified but not yet published. The
development-bump PR finalizes this document with the recovery release SHA, deployment record and
tag mapping after publication.

## What is released

The first pure-Java coordinates on Maven Central, independently versioned and published:

| Coordinate | Version | Java level | Published project dependency |
| --- | --- | --- | --- |
| `se.alipsa:jmlx-jinja` | `0.6.1` | 21 | none |
| `se.alipsa:jmlx-tokenizer` | `0.1.1` | 21 | `se.alipsa:jmlx-jinja:0.6.1` |
| `se.alipsa:jmlx-vision` | `0.1.1` | 21 | none |

All three are Java 21+: they require no MLX, no FFM, no native bootstrap and no macOS. A Java 21
consumer on any platform can resolve and use them; the pure-Java release smoke
(`tools/release-smoke/run.sh ci|candidate|central --profile pure-java`) enforces exactly that
consumer story, including a Java 21 headless consumer JVM.

**No native runtime ships in this release.** The four native-chain modules
(`jmlx-ffi`, `jmlx-core`, `jmlx-models`, `jmlx-native-macos-arm64`) are not published in this
milestone and remain `-SNAPSHOT`. There are no inference-performance claims here; performance
work is Phase 8.2.

## Recovery: superseding 0.6.0 / 0.1.0 / 0.1.0

The initial cut (release SHA `6097d8f`, 2026-11-04) published three complete coordinates whose
bundles **omitted the Gradle module metadata file** (`.module`): the then-pinned
nexus-release-plugin 2.2.0 bundles only JAR and POM files, while the generated POM advertises
Gradle metadata. Without the `.module`, a fresh-cache consumer's javadoc/sources resolution falls
back to POM classifiers and pulls in the dependencies' Javadoc jars, so the fresh-cache
`run.sh central --profile pure-java` gate failed against the published artifacts. The 2.2.0
omission is structural (a hard filter in the bundle task), not a misconfiguration.

Recovery decisions and facts:

- The three coordinates are re-released as `0.6.1` / `0.1.1` / `0.1.1` with **no functional
  changes** (jars are byte-identical to the initial cut; POMs differ only in the version
  coordinates; comparison guard recorded in the release record). Use `0.6.1` / `0.1.1` / `0.1.1`
  (or later); the `0.6.0` / `0.1.0` / `0.1.0` coordinates remain on Central as dead artifacts and
  are never published to again.
- nexus-release-plugin is pinned to 2.5.0, which provides the module-based combined release
  this workflow uses (the root `combinedRelease.modules` list; the plugin derives the candidate
  set, per-module staging, signing and output paths, and jmlx's earlier hand-rolled candidate
  walk is gone) and, since 2.3.0, validates every bundle before upload — the plan adds the
  `.module` to a candidate's file list whenever its POM advertises Gradle metadata, so a
  staging directory missing the signed `.module` fails before any upload; the 2.2.0 shape
  cannot recur silently.
- The recovery is published as **one combined `releaseAll` deployment** (the plugin's
  mega-bundle flow: one Maven-layout ZIP, one automatic Portal deployment) instead of three
  single deployments: Maven Central allows 10 deployments per month, three were already spent by
  the initial cut, and the combined flow spends one slot for the whole tuple. The tokenizer →
  jinja published-dependency edge is satisfied inside the bundle by the planner's fixed point.
  The combined tooling (`releaseAll.sh`, `tools/release-all/`, the committed sanitized Portal
  fixture) is part of the recovery PR; see `tools/release-all/README.md` and the runbook's
  Publication section.

## jmlx-jinja 0.6.1 (content: 0.6.0): the hfjinja migration

`jmlx-jinja` is the continuation of the archived standalone `hfjinja` project (the
`@huggingface/jinja` 0.5.9-compatible rendering engine), migrated into this repository and its
release line. Consumers migrating from `se.alipsa:hfjinja` 0.5.x must change:

- the Maven coordinate: `se.alipsa:hfjinja` → `se.alipsa:jmlx-jinja`;
- the Java packages: `se.alipsa.hfjinja` → `se.alipsa.jmlx.jinja`;
- the JPMS module name: `se.alipsa.hfjinja` → `se.alipsa.jmlx.jinja` (a real module with a
  `module-info.class`; the module descriptor is asserted by the release smoke against the
  published artifact).

No relocation POM for `se.alipsa:hfjinja` is published: the migration is explicit. The 0.6.0
changelog (in `jmlx-jinja/CHANGELOG.md` after the release PR rolls the Unreleased entries)
enumerates the changes since hfjinja 0.5.0, including the migration itself and the repository
integration (shared conventions, pinned Node/Python verification, the upstream corpus).

## jmlx-tokenizer 0.1.1 (content: 0.1.0)

The first published version of the Hugging Face tokenizer library (byte-level BPE, Metaspace BPE,
Unigram and WordPiece, chat-template rendering through `jmlx-jinja`, incremental generated-token
decoding). Its published dependency on `se.alipsa:jmlx-jinja:0.6.1` is the edge the pure smoke
exercises: the consumer depends on tokenizer and vision only, and jinja must arrive
transitively. Known limitations carried from the module's coverage reports (see
`req/phase6-2-tokenizer-components.md`): the supported component set is the pinned HF
`tokenizers` surface the module implements; components outside it fail with a clear error rather
than falling back.

## jmlx-vision 0.1.1 (content: 0.1.0)

The first published version of the pure-Java image library: PNG/JPEG decoding,
Pillow-exact bilinear/bicubic/LANCZOS resampling, normalization, and the SmolVLM/Idefics3 and
Qwen3-VL image processors. The resampling and alpha-composite kernels are a port of pinned
Pillow 12.3.0 sources, and the Phi-3-vision bicubic kernels are a port of pinned PyTorch sources;
`jmlx-vision/NOTICE` pins every ported source file by SHA-256 and carries the full MIT-CMU and
BSD 3-Clause license texts, and the same NOTICE is asserted in the published main, sources and
Javadoc jars.

## License and provenance

Every published jar carries `META-INF/LICENSE`. The vision jars additionally carry
`META-INF/NOTICE` with the ported-source provenance (SHA-256s and license texts) so a consumer
who downloads the sources or Javadoc jar can read the exact provenance. The pure-Java smoke
verifies all of this against the published artifacts, and `selftest.sh` plus the two CI smokes
(Linux pure, macOS full) keep the tooling itself regression-tested.

## Verification summary

- Pure candidate smoke from the clean release merge commit (`run.sh candidate --profile
  pure-java`), rerun unconditionally after the merge.
- jinja's complete release checklist, including `:jmlx-jinja:releaseVerification`, on the release
  commit.
- Comparison guard (recovery): main and sources jars byte-identical to the initial cut's
  published jars; javadoc jars differ only in the version string of the HTML titles; POMs differ
  only in the version coordinates (and tokenizer's jinja dependency version). The recovery
  bundle additionally carries the `.module` file (with signature and checksums) that the 2.2.0
  bundles omitted.
- Fresh-cache Central smoke (`run.sh central --profile pure-java`) after the combined deployment,
  with the deployment/portal/availability record (see `phase8-1-release-checklist.md`).
- Tags `jmlx-jinja-v0.6.1`, `jmlx-tokenizer-v0.1.1`, `jmlx-vision-v0.1.1` at the verified release
  SHA (mapping recorded below after publication).

## Release record (filled in by WP5)

### Initial cut (superseded)

- Cut date: 2026-11-04
- Release SHA / branch: `6097d8f` (merge of PR #82) on `main`
- Deployment IDs: `se.alipsa:jmlx-jinja:0.6.0` — `0a6be0f9-fa66-40e8-91c3-f814408d4f6c`;
  `se.alipsa:jmlx-tokenizer:0.1.0` — `7dff55e2-1fd1-4a85-b2b6-3752fc4fe688`;
  `se.alipsa:jmlx-vision:0.1.0` — `9be812c6-b0c9-4234-b055-84fcf1dfdf1d`
- Outcome: all three published and live on Central (jinja 19:48:47 UTC, tokenizer/vision
  20:05:31 UTC), but the bundles omitted the `.module` file (nexus-release-plugin 2.2.0); the
  fresh-cache Central smoke failed on javadoc/sources resolution. Coordinates superseded by the
  recovery below; never publish to them again.

### Recovery (0.6.1 / 0.1.1 / 0.1.1)

- Release SHA / branch: _pending (recovery PR merge commit)_
- Comparison guard (re-verified 2026-11-05 against the Central-published `0.6.0` / `0.1.0`
  artifacts downloaded from repo1): main and sources jars byte-identical (matching SHA-256),
  javadoc jars identical after normalizing the HTML version strings, POMs identical after
  normalizing the version coordinates (tokenizer's jinja edge `0.6.0` → `0.6.1` is the intended
  dependency change), and the superseded coordinates' `.module` files 404 on Central while the
  recovery bundle carries them signed.
- Deployment ID (single combined deployment): _pending_
- Bundle SHA-256 (`bundle-receipt.json`): the local `--bundle-only` run of the 2.5.0 tooling at
  branch tip `6b3557a` is `c26996e500a1706a2fc2885b2563faa35267d383a447d641db5175ec38c2637d`;
  the production run from the merge SHA re-records the binding value (an identical tree is
  expected to reproduce it byte-for-byte). The earlier
  `cf0eda90f3e3ba9bc6e800b3ff6a6bebe6c3a793bb46f4f9164778592c3997ea` was produced by the
  pre-2.5.0 tooling and is not the published bundle.
- Tag-to-SHA mapping: _pending_
- Central smoke report: _pending_
