# Phase 8.1 release notes: jmlx-jinja 0.6.0, jmlx-tokenizer 0.1.0, jmlx-vision 0.1.0

**Status:** provisional, prepared by the Phase 8.1 tooling PR on 2026-11-04. The release has not
been cut or published; the open Phase 7 acceptance gates block the release cut. WP5's
development-bump PR finalizes this document with the actual cut dates, per-module release SHAs,
branches and tag mapping after publication. No merge commit's own SHA is embedded here: each
module's release SHA is the clean commit its `release.sh` ran from, recorded per module.

## What is released

The first pure-Java coordinates on Maven Central, independently versioned and published:

| Coordinate | Version | Java level | Published project dependency |
| --- | --- | --- | --- |
| `se.alipsa:jmlx-jinja` | `0.6.0` | 21 | none |
| `se.alipsa:jmlx-tokenizer` | `0.1.0` | 21 | `se.alipsa:jmlx-jinja:0.6.0` |
| `se.alipsa:jmlx-vision` | `0.1.0` | 21 | none |

All three are Java 21+: they require no MLX, no FFM, no native bootstrap and no macOS. A Java 21
consumer on any platform can resolve and use them; the pure-Java release smoke
(`tools/release-smoke/run.sh ci|candidate|central --profile pure-java`) enforces exactly that
consumer story, including a Java 21 headless consumer JVM.

**No native runtime ships in this release.** The four native-chain modules
(`jmlx-ffi`, `jmlx-core`, `jmlx-models`, `jmlx-native-macos-arm64`) are not published in this
milestone and remain `-SNAPSHOT`. There are no inference-performance claims here; performance
work is Phase 8.2.

## jmlx-jinja 0.6.0: the hfjinja migration

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

## jmlx-tokenizer 0.1.0

The first published version of the Hugging Face tokenizer library (byte-level BPE, Metaspace BPE,
Unigram and WordPiece, chat-template rendering through `jmlx-jinja`, incremental generated-token
decoding). Its published dependency on `se.alipsa:jmlx-jinja:0.6.0` is the edge the pure smoke
exercises: the consumer depends on tokenizer and vision only, and jinja must arrive
transitively. Known limitations carried from the module's coverage reports (see
`req/phase6-2-tokenizer-components.md`): the supported component set is the pinned HF
`tokenizers` surface the module implements; components outside it fail with a clear error rather
than falling back.

## jmlx-vision 0.1.0

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
- jinja's complete release checklist, including `:jmlx-jinja:releaseVerification`, on each jinja
  publication's release commit.
- Fresh-cache Central smoke (`run.sh central --profile pure-java`) after all three deployments,
  with per-module deployment/portal/availability records (see
  `phase8-1-release-checklist.md`).
- Tags `jmlx-jinja-v0.6.0`, `jmlx-tokenizer-v0.1.0`, `jmlx-vision-v0.1.0` at the verified release
  SHAs (mapping recorded below after publication).

## Release record (filled in by WP5)

- Cut date: _pending_
- Initial release SHA / branch: _pending_
- Per-module release SHAs and branches: _pending_
- Tag-to-SHA mapping: _pending_
- Deployment IDs and terminal portal states: _pending_
- Central smoke report: _pending_
- Recovery events (if any): _none yet_
