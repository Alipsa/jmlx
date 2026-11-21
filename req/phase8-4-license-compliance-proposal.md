# Phase 8.4 native-license compliance proposal

2026-11-21. Draft for owner review, based on PR #106 merge
`59cc6c1` and the [scoped review](phase8-4-scoped-license-review.md).
Status: proposed; no legal determination or permission grant recorded.
Both `NATIVE-LICENSE-1` and `NATIVE-LICENSE-2` remain open.

## Proposed disposition

Prepare a CC BY-SA 4.0 compliance implementation for the three identified Juffa
adaptations, conditional on the decisions below. For Cephes, seek a written assessment
of the retained original permission against the actual redistribution; otherwise require
an applicable grant or a traced licensed distribution. Notice additions alone close neither
finding. This proposal authorizes no publication and makes no changes to release artifacts.

The concrete [draft notice](plans/evidence/license-compliance-proposal-2026-11-21/notice-draft.txt)
is review material outside the native module. Its unresolved adapter-license paragraph must
be replaced with approved wording before packaging. No production task includes that file.

## Material and distribution to decide

The reference is the unchanged RC1 ZIP with SHA-256
`b4428fd0817273fd7387528c35519302c334aee444cbe6dcffe53d2a05f0ef2f`.
The candidate's recorded source and wheel association has the limits documented in the
scoped review; source hashes do not prove a reproducible wheel build.

| Adaptation | Reviewed answer revision | Proposed source-license route |
| --- | --- | --- |
| Metal `erf` | [35148199 r4][erf-r4], 2022-06-18 | CC BY-SA 4.0 review |
| Metal `erfinv` | [49743348 r3][erfinv-r3], 2021-06-10 | CC BY-SA 4.0 review |
| CPU SIMD `erfinv` | Same inverse answer revision | CC BY-SA 4.0 review |

The Metal functions are in `mlx/backend/metal/kernels/erf.h`; the CPU function is in
`mlx/backend/cpu/simd/math.h`. Their complete-file SHA-256 values are, respectively:

```text
31d88004390bbe08636c5f64a3cb3180ebc8d79a89267347de9f6c33aba0e9f9
5dbe79caa33a228d1354b2621876dc0fb3629dd6733c8ee4ebd0471790b0a94a
```

The [technical report](plans/evidence/scoped-license-review-2026-11-21/technical-findings.json)
binds answer revisions, bodies and code blocks to the prior audit. It distinguishes literal
sequence and multiset comparisons. These are comparison facts, not a copying history.
CPU `erf` has separate PyTorch provenance. The BSD notice for `expm1f` remains separate too.

The reviewer must assess the source material, MLX modifications and their compiled forms in
`libmlx.dylib` and `mlx.metallib`, including downstream use through jmlx. Include the native
main, sources and Javadoc JARs, POM, module metadata, README and release notes in that review.
Do not infer that splitting text into function-level notices establishes the legal boundary.

## Decisions required before implementation

Record each answer in the decision record below, with evidence and rationale.

1. **Juffa source/license mapping:** accept the three mappings or identify a different
   applicable grant. Confirm the creator identification, revision links and retained notices
   against the actual answer snapshots, including any supplied disclaimer or modification notice.
2. **Adaptation boundary and authority:** identify the material covered, the relevant rights
   holders and who can license each contribution. Address Apple's MLX modifications explicitly.
   jmlx cannot grant rights in someone else's contributions merely by adding a license label.
3. **Adapter-license identification:** identify the adapter(s), the license under which jmlx
   may redistribute their contributions, and the evidence supporting that identification.
   Explain its scope in source, compiled forms, other MLX code and the surrounding distribution,
   then approve metadata that accurately reflects it. jmlx's redistribution does not itself
   authorize choosing a license for Apple's or other adapters' contributions. If that basis
   cannot be established, seek clarification, an applicable grant or a reviewed replacement
   before changing release metadata. Any future adaptation by jmlx needs its own assessment.
4. **Downstream terms:** review the actual distribution terms and any technological measures
   against the approved treatment. Review all publication surfaces together, including licenses
   consumers see in the POM. Do not infer compliance from Maven availability or a notice test.
5. **Cephes permission scope:** explain whether the retained original permission covers the
   identified modifications and redistribution, including downstream recipients. If relying
   on another grant, identify its grantor, recipients, covered code and downstream applicability.

The [CC BY-SA 4.0 legal code][cc-legal], sections 1–3, governs attribution, changes, adapter
licensing and downstream conditions. The first entry in [Choosing a license][cc-authority]
cautions against applying licenses without authority; the [software FAQ][cc-software] explains
software-specific limitations. Section 1(b) defines the adapter's license in terms of rights
in the adapter's contributions. Section 2(a)(5)(B) addresses the offer recipients of adapted
material receive from the Licensor under the adapter's license; section 2(a)(5)(C) prohibits
downstream restrictions on the licensed material. Section 3(b)(3) binds the adapter applying
an adapter's license. The reviewer must distinguish those roles rather than treating jmlx as
the adapter of the existing MLX code. These sources frame the review questions; this proposal
does not decide the compiled adaptation boundary or establish an adapter's license.

For Cephes, retain the [Netlib permission][netlib] and the immediate fastermath/NEON notices.
The [2004 correspondence][cephes-email] and [reply][cephes-reply] need the scope assessment
in the [resolution review](phase8-4-native-license-resolution.md). Six selected coefficient
magnitudes per function match Netlib files labeled Release 2.2; fastermath refers to 2.8.
Neither observation identifies an applicable BSD grant for MLX's exact provenance chain.
Do not relabel the existing Cephes notice BSD-3-Clause on this evidence alone.

## Concrete implementation after approval

Use a separate implementation PR tied to the completed decision record and draft revision.

| File or surface | Proposed change |
| --- | --- |
| `jmlx-native-macos-arm64/THIRD-PARTY-NOTICES` | Add the approved Juffa section |
| `jmlx-native-macos-arm64/NOTICE` | Point to the approved scoped terms and provenance |
| Native module `LICENSE-CC-BY-SA-4.0` | Add the unmodified official license text |
| Native module `build.gradle` | Include that text in all three JARs; update scoped POM |
| `req/native-license-inventory.json` | Record upstream evidence and approved packaged hashes |
| `req/phase8-4-native-license-audit.json` | Record each resolution only after validation |
| Native README and 8.4 release notes | Explain approved scope and where terms are packaged |
| Release checks and independent smoke | Add the separate enforcement described below |

Only if the reviewer establishes this license's applicability, add a POM entry named
`Creative Commons Attribution-ShareAlike 4.0 International`, with URL
`https://creativecommons.org/licenses/by-sa/4.0/` and this **conditional draft comment**:

```text
Identifies the applicable license(s) for the reviewed Norbert Juffa-derived Metal erf
and Metal/CPU SIMD erfinv material, within the established source/compiled scope.
Scope and licensing basis: [immutable approved-notice URL]. Also packaged in
META-INF/THIRD-PARTY-NOTICES.
```

Replace that comment with the exact established scope and licensing basis. Establish the
permalink in this order:

1. Commit the approved notice first, then record that commit's full SHA. In a later commit,
   add the POM comment linking to that earlier SHA and the exact notice path. A commit cannot
   contain its own SHA. If the notice changes again, commit that change before updating its
   POM permalink in another commit.
2. Preserve the linked commit through a merge commit (`gh pr merge --merge`); do not squash
   or rebase the implementation branch. Before candidate cut, require the linked SHA to be
   an ancestor of reviewed `main`; after tagging, also require ancestry of the candidate.
   Merely being an accessible GitHub object is insufficient.
3. Read the linked notice offline with
   `git show <sha>:jmlx-native-macos-arm64/THIRD-PARTY-NOTICES`. Extract its Juffa section
   and compare it byte-for-byte with that section in each packaged native JAR using the new
   raw-byte extractor specified below. Fail on missing Git history, absent sections or any
   byte difference in explicit release verification. Also compare the local notice before
   packaging. A permalink is immutable but can become stale after later edits; repeat this
   comparison on each explicit pre-tag release verification and candidate check. Add
   regressions for stale notice text and a linked commit that is not an ancestor.

The URL must be a public repository permalink available independently of the JAR. Verify its
availability before candidate cut and retain it in release evidence; do not use mutable
`main` or a missing future tag. A POM reader must be able to inspect the scope without the
JAR. If another license applies to the adapter's contributions, reflect that distinction
instead of labeling those contributions BY-SA by assumption. The comment is not a decision.
Retain the existing MIT/Apache texts and other upstream notices, with scope descriptions
consistent with the approved treatment. Review both Gradle metadata and POM-only consumption.

Obtain the full CC text from the official legal-code page's plain-text link; record its URL,
retrieval date and SHA-256 in the implementation evidence. Include it as
`META-INF/LICENSE-CC-BY-SA-4.0` in main, sources and Javadoc JARs using the existing
`tasks.withType(Jar)` convention. Verify its contents, not only its filename.

For traceability, propose an explicitly identified provenance attachment containing the
approved source excerpts or headers, answer revision references, hashes and modification
mapping. Have its contents and license labeling reviewed before adding a classifier or
release asset. It is a traceability measure, not an asserted CC source-distribution mandate,
and it must not replace the native module's Java sources JAR or imply a wheel rebuild.

Cephes changes depend on the decision: preserve the existing original statement if that is
the accepted basis; add any applicable grant verbatim with its exact scope if another basis
is approved. Keep the immediate BSD/zlib notices in either case. A rationale alone must not
be transformed into a new grant or a guessed SPDX identifier.

### Required verifier extensions

These checks are proposed implementation work; the current schema cannot express the
jmlx-authored Juffa attribution. `audit_native_licenses.py:check_attribution` requires each
`noticeFragments`/`noticeLines` excerpt in both the pinned upstream bytes and the packaged
section. Creator labels, revision links and explanatory text written by jmlx are not upstream
excerpts. Do not put them in those fields or weaken the existing two-sided excerpt check.

Add a separate authored-attribution policy and check, with the following responsibilities:

- A new `req/native-authored-attributions.json` records the approved section title, required
  creator/user-ID and revision fields, source evidence references, license identification,
  approved scope text, decision reference and final section digest. This is a proposed new
  policy, not an existing inventory feature or a new upstream grant.
- Extend `audit_native_licenses.py` with a separate authored-attribution validator. Verify
  referenced upstream/revision evidence against its pinned digests, then check the authored
  requirements against the packaged section. Do not require authored phrases to occur in
  upstream headers or answer bodies. Keep upstream evidence validation and authored-text
  validation distinct, and fail if the required Juffa policy/section is absent.
- `notice_sections()` strips outer section whitespace and uses the first non-blank line
  as the component key. Use exactly this title in the draft policy and section:
  `Norbert Juffa erf / erfinv adaptations (source records: CC BY-SA 4.0)`.
  Any approved title change must update both together. Do not create a
  legacy excerpt-component entry for this authored section. If a component-key reference is
  added to the inventory, require the same exact title there too.
- Extend `release_inputs.py:native_notice_report` to require and check the approved authored
  policy against each native JAR's notice, independently of `packagedTexts` hashes. Add the
  corresponding requirement to the independent consumer checks. Hard-code the mandatory
  Juffa section title in a new `REQUIRED_NATIVE_AUTHORED_SECTIONS` constant next to
  `REQUIRED_NATIVE_LEGAL_TEXTS`, and independently in consumer smoke. Validate both policy
  coverage and actual sections against those constants. Removing the policy entry/file and
  section together must fail, even if someone also updates the enclosing notice digest.

Extend `release_inputs.py:REQUIRED_NATIVE_LEGAL_TEXTS` with `LICENSE-CC-BY-SA-4.0` and keep
`validate_native_legal_inventory` as a mandatory precondition to JAR validation. In
`tools/release-smoke/smoke-conventions.gradle:verifyPublishedArtifacts`, add an independent
required-text set check before iterating `inventory.packagedTexts`. It must also require the
CC text. Merely listing its digest in the inventory is insufficient: removing the file from
both inventory and all three JARs must fail both release verification and consumer smoke.

For native POM licenses, add a proposed `native_pom_report` helper in `release_inputs.py`.
Extend `check_bundle` to read the actual native `.pom` bytes out of the ZIP and pass them to
that helper; today it only requires POM/signature entries to exist. Assert the complete
approved license name/URL/scoped-comment tuples, including the immutable notice URL, against
expectations independent of the producer's `withXml` block. Do not use the POM's own entries
as the expected list.

Keep Python and Git-history checks in an explicit producer release task, proposed as
`verifyNativeReleaseLicenses`, depending on `generatePomFileForMavenPublication` and all
three native JAR tasks. It invokes the helper on the locally generated
`build/publications/maven/pom-default.xml` and checks the permalink against local and packaged
sections. Run it before tagging, and require it in bundle preparation alongside
`verifyNativeReleasePayload`; update the release task-execution inventory accordingly.
Do not wire it into `check` or `verifyPackagedNativeResources`. Ordinary contributor builds
must retain no Python or deep-history prerequisite, including an unstaged checkout.
The pre-tag task must not call `check_bundle`, whose tag/HEAD guard is candidate-only.

Only explicit release verification requires the linked commit and its ancestry locally.
Document fetching the required history before that task; fail with a clear diagnostic if it
is unavailable, without silently skipping or fetching. Ordinary PR CI remains compatible
with its current depth-1 merge checkout. Consumer smoke checks POM tuples and permalink
syntax/value without reading Git history. If a future CI job runs the history-dependent
release task, that job must explicitly use `actions/checkout` with `fetch-depth: 0`.

For permalink freshness, add a separate `raw_notice_sections(bytes)` extractor in
`release_inputs.py`. Recognize only complete separator lines containing the exact ASCII
notice separator followed by LF or CRLF. A section starts immediately after its separator
line and runs up to, but excludes, the next separator line; the final section runs to EOF.
Compare those untouched bytes, including leading blank lines, the title, spaces, blank lines
and original line endings. The file's trailing newline belongs to its final section when
present; adding or removing it is a byte difference. Before selecting sections, detect
separator lines accepted by the excerpt parser's lenient `[ \t]*` suffix rule but containing
trailing spaces/tabs. Reject them with an explicit malformed-separator diagnostic, including
the line number; do not treat them as ordinary section content or report a stale permalink.

For identification only, find the first non-blank line in each section: skip leading lines
containing only spaces/tabs after removing their line terminators, then remove only the title
line's terminator and require the exact title. For this identification pass, split physical
lines only on LF/CRLF and decode them as strict UTF-8; do not use `splitlines()`, which also
splits at form feeds and vertical tabs. Reject other Unicode whitespace before the title,
including form feed, vertical tab and U+00A0, with an unsupported-leading-whitespace
diagnostic. Thus broader `str.strip()` whitespace cannot silently produce a different title.
Keep skipped spaces/tabs-only lines in the compared bytes. Accept both a title immediately
after the separator and a title after blank lines, but do not treat those layouts as
byte-identical. Reject duplicate titles and retain all section bytes for comparison. The draft
now uses the existing notice file's layout, with a blank line between separator and title.

Do not reuse `audit_native_licenses.py:notice_sections()`, which strips/splits/rejoins text
for excerpt matching. Test both accepted title layouts, leading and internal blank-line
changes, trailing spaces, LF-versus-CRLF changes, changed content and missing sections.
Include a fixture with Juffa as the last section and no closing separator; verify extraction
through EOF and mismatches when the final newline is added or removed. Add fixtures with
separator suffix spaces/tabs and leading form-feed, vertical-tab and U+00A0 lines; require
the respective malformed-separator or unsupported-leading-whitespace error before any
freshness comparison, rather than a merged section, wrong title or stale-notice error.

For consumer verification, register `verifyNativePomLicenses` in the `pom-check` project
inside `smokeRegisterPomTask`. Resolve the selected native coordinate's published `@pom`
through its POM-only consumer repositories. Keep an independent Groovy XML/tuple comparator
in a new `tools/release-smoke/native-pom-licenses.gradle`; do not call the Python comparator
from consumer smoke. Make `:app:verifyPublishedArtifacts` depend on
`:pom-check:verifyNativePomLicenses`; `:app:smoke` already depends on published-artifact
verification. Skip registration/wiring for the pure-Java profile. Do not attach the license
check to `writeRuntimeCoordinates`, which remains a cached coordinate-report task.
The existing POM-versus-module graph check does not inspect license entries.

Store the approved full name/URL/comment tuples as a literal `REQUIRED_NATIVE_POM_LICENSES`
constant in `release_inputs.py` for producer/bundle verification. Store a separately reviewed
literal expected-tuple list in `native-pom-licenses.gradle` for the consumer. Both include the
approved permalink. Neither list is generated from the POM, the producer `withXml` block or
`req/native-authored-attributions.json`; a policy edit alone must not change expectations.
Update both lists only in a reviewed change tied to the completed decision record, including
any later notice/permalink revision. Producer and bundle checks intentionally share the
Python comparator; the consumer uses its own Groovy comparator and explicit expectations
so a Python implementation bug does not automatically pass consumer smoke.

Implement the Python comparison as a pure function over POM bytes. Add fixtures in
`tools/release-all/tests/test_native_licenses.py` covering missing, changed and duplicate
entries, scoped comments and permalink values. Test the function and explicit-release CLI
without a native runtime or candidate tag. Add a standalone Gradle fixture applying
`native-pom-licenses.gradle` to those POM fixtures; invoke it from the release regression
suite using `JMLX_RELEASE_TEST_GRADLE`, as with existing portable Gradle fixtures. This tests
both implementations on Linux-capable JVMs without MLX, Git history or model execution.
Do not claim that `tools/release-smoke/selftest.sh` executes these Gradle tasks; it remains
the parser/preflight harness. Use full native smoke on macOS for actual published POM-only
resolution and end-to-end task wiring.

## Validation and candidate replacement

The implementation PR should demonstrate all of the following before resolution:

- Upstream names, links, revision identities and modification descriptions match the approved
  record; required notices and license texts are retained in all three native JARs.
- Removing a Juffa attribution, approved scope text or license text causes a targeted
  authored-attribution or packaging check to fail, including when notice hashes are updated.
- Removing the CC text from both inventory and JARs fails the independent required-text checks.
  The new bundle and consumer POM checks reject absent/altered name, URL, scoped comment or
  approved-notice permalink, before and after tagging. Test section-title mismatches and
  simultaneous policy/section deletion, plus permalink ancestry and byte-level freshness.
- New inventory source bytes and packaged text digests are checked independently. Positive
  marker hits are expected in the new candidate; they neither fail it nor prove legal clearance.
- Root/shared-build checks, release regressions, native payload checks and both consumer
  profiles pass under the [release runbook](../tools/release-all/README.md).
- The completed decisions and artifact verification support closing each finding separately.
  A remaining finding keeps the release gate open even if the other is resolved.

Keep RC1, its tag, ZIP and receipts unchanged. A packaging/POM change requires a new reviewed
merge and immutable `jmlx-phase8.4-rc.2` (or the next unused higher candidate number), fresh
signatures, bundle identity and affected verification, including minimum-host exact-bundle
execution. Do not reuse RC1's receipt for changed bytes. Recheck changelog dates and final
candidate audit, then review Portal capacity and request publication approval against the
new verified receipt. Approval of this proposal is not Central publication authorization.

## Decision record for the reviewer

The [owner decision record](phase8-4-owner-license-decision.md) identifies Per Nyfelt as the
project reviewer, with Sweden as the confirmed review context. The owner accepted the Cephes
original-permission interpretation on 2026-11-21, subject to retained notices and artifact
validation. The Juffa decision remains pending. The owner record is the authoritative decision
template; the table below lists required coverage rather than a second template. Complete
the owner record in a reviewed PR; identify the exact proposal revision and final legal text
hashes. Implementation selection is not a
recorded legal determination. This workflow does not require outside legal approval. No
rights holder or outside reviewer has been contacted by this work.

| Field | Required record |
| --- | --- |
| Reviewer | Name, role, date, review context and applicable law (pending if undetermined) |
| Reviewed evidence | Candidate identity, proposal commit and scoped-review hashes |
| Juffa source mapping | Accept/reject/need evidence, with rationale and references |
| Adaptation scope | Exact source and compiled material, including surrounding-code treatment |
| Licensing authority | Rights holders and authority/basis for each contribution |
| Adapter-license basis | Identified license/grant, evidence, exact scope wording and rationale |
| Downstream terms | Terms/measures examined and reasoned compatibility conclusion |
| Cephes basis | Original permission interpretation or applicable grant and exact code mapping |
| Approved artifacts | Final notice/license/POM wording and reviewed hashes |
| Disposition per finding | Accept/reject/need evidence; conditions and responsible owner |
| Follow-up | Named implementation/review owner and any unresolved questions |

Official sources above were checked on 2026-11-21. CC and Netlib pages were readable through
the web tool; the Debian pages were unavailable there on this check. Their contents remain
the separately digest-pinned snapshots in the scoped review. No new raw-page hash is claimed.

[erf-r4]: https://stackoverflow.com/revisions/35148199/4
[erfinv-r3]: https://stackoverflow.com/revisions/49743348/3
[cc-legal]: https://creativecommons.org/licenses/by-sa/4.0/legalcode.en
[cc-software]: https://creativecommons.org/faq/#can-i-apply-a-creative-commons-license-to-software
[cc-authority]: https://creativecommons.org/faq/#choosing-a-license
[netlib]: https://netlib.org/cephes/readme
[cephes-email]: https://lists.debian.org/debian-legal/2004/12/msg00295.html
[cephes-reply]: https://lists.debian.org/debian-legal/2004/12/msg00297.html
