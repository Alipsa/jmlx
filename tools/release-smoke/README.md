# Release smoke

Independent Gradle consumers resolve published Maven artifacts through both POM-only and
Gradle module metadata. They never use project dependencies, checkout classes, Maven local or
composite substitution. Every run clears consumer state and uses a fresh Gradle home with the
build cache disabled. Release commands need Python 3.11+ for shared manifest/evidence validation.

| Profile | Expected jmlx graph | Consumer |
| --- | --- | --- |
| `all` (default) | nine modules, including both framework adapters | Java 25, native macOS ARM64 |
| `pure-java` | jinja/tokenizer/vision, jinja transitive through tokenizer | Java 21, headless, cross-platform |

```sh
bash tools/release-smoke/run.sh ci --profile all
bash tools/release-smoke/run.sh ci --profile pure-java
bash tools/release-smoke/run.sh candidate --profile all
bash tools/release-smoke/run.sh candidate --profile pure-java
bash tools/release-smoke/run.sh central --profile all
bash tools/release-smoke/run.sh central --profile pure-java
bash tools/release-smoke/run.sh central --profile pure-java --release 8.1
```

`ci` publishes the selected checkout modules to disposable `build/smoke-repo`; its defaults
in `gradle.properties` follow producer versions. Existing `JMLX_<MODULE>_VERSION` overrides are
forwarded to Gradle. `--record` is permitted only for `ci --profile all` and rewrites the seeded
Mistral golden. Never record a new golden merely to accommodate native precision differences.

Release modes default to the committed `released-8.4.json`. Both profiles select their subset
of that same version/source manifest; snapshots and disagreeing overrides are rejected.
`candidate` checks actual producer versions and Central availability before cleanup/publication,
then publishes **only local entries**, resolving reused entries exclusively from real Central.
Exact-version exclusive repository filters are identical for POM and module-metadata consumers;
unknown jmlx coordinates cannot fall back to a generic repository. `central` publishes nothing
and resolves every selected coordinate from Central, preserving the manifest after development
bumps. The 8.4 tuple remains provisional until the release-readiness audit closes.

`--release 8.1` is historical Central verification of `released-8.1.properties`, accepted only
with `central --profile pure-java`. A missing/malformed entry or disagreeing override fails;
there is no SNAPSHOT-stripping fallback. Old 8.1 runbooks must use this explicit selection.

## Verify the signed bundle itself

After combined `--bundle-only` writes the preliminary identity record:

```sh
bash tools/release-smoke/run.sh candidate --profile all \
  --bundle <signed-bundle.zip> --bundle-record <preliminary.json> \
  --bundle-record-sha256 <expected-sha256>
```

The same inputs work with `pure-java`. This mode verifies record/ZIP/manifest digests, candidate
annotated tag object and HEAD, safe ZIP paths, exact new coordinates and signed artifact presence
before cleanup. It unpacks into a disposable Maven repository and **never publishes or invokes
producer tasks**. Wrapper distribution discovery still requires a full candidate checkout.
Reused entries resolve exclusively from actual Central. The bundle repository is removed on exit.
`tools/release-all/bundle_smoke.py` wraps this interface to create final-receipt-compatible
reports/logs with environment identity; see [the combined runbook](../release-all/README.md).

The locally signed ZIP and preliminary record reach the physical minimum-host workflow via
separate assets of a temporary draft Release, with both expected hashes passed independently.
The local Ubuntu ARM64 run uses a real scratch clone, `/projects` bind mount transport and
fresh Linux Gradle home. Both verify all input digests before extraction. No producer bootstrap
or model download is needed: inference uses the committed tiny Mistral/BERT fixtures.

## Assertions and fixtures

Both profiles check exact selected versions, no unresolved or SNAPSHOT release dependencies,
POM/metadata runtime-graph agreement, main/sources/Javadoc artifacts, licenses/NOTICE and JPMS
names. The native profile additionally checks the packaged payload/pins, disposable extraction,
seeded two-request shared cohort, public Spring AI/LangChain4j chat/stream/embedding behavior,
virtual threads, cancellation, stalled-demand bounds and memory release. Native main/sources/
Javadoc jars must carry the exact reviewed third-party notice texts.

The pure profile verifies Java 21/headless execution, public Jinja rendering, tokenizer
encode/decode/chat/incremental Unicode facts, Pillow-recorded resize bytes and analytic
normalization. It never configures the native app or requires a native runtime.

- `fixtures/PROVENANCE.md`: hand-written tokenizer paired with synthetic Mistral.
- `fixtures/bert/PROVENANCE.md`: sentence encoder tokenizer and committed BERT checkpoint.
- `fixtures/pure-java/PROVENANCE.md`: hand-computed tokenizer and digest-pinned image-oracle case.
- `goldens/mistral-sampled.properties`: seeded full-profile generation reference.

Run `bash tools/release-smoke/selftest.sh` for parser/preflight regressions and
`bash tools/release-all/selftest.sh` for manifest/native validation and real Gradle repository/
collector regressions. Portable CI runs both. Native CI separately asserts the required suites
using the shared release inventory; a skipped native suite is a failure there.

Toolchains: the producer uses the checkout's build JDK; pure consumers execute on Java 21 and
full consumers on Java 25. A fresh home does not inherit downloaded JDKs or user Gradle settings.
Retain `GRADLE_OPTS` toolchain discovery settings (CI uses `org.gradle.java.installations.fromEnv`
with auto-download disabled) or allow the configured Foojay resolver to download missing JDKs.
The wrapper distribution is located/downloaded before launching the isolated consumer.
