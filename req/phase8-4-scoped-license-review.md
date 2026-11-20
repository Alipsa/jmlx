# Phase 8.4 scoped native-license review

2026-11-21. Technical review of frozen `jmlx-phase8.4-rc.1`, based on reviewed main
`ddd2c33242502572aaa1a2a5e87a45f02815a615`. Both license findings remain open.
This memo supplies evidence, a proposed compliance route and explicit decision questions;
no owner/legal determination or new third-party permission has been obtained.

## Recommended disposition

Pursue a CC BY-SA 4.0 compliance review for the three Juffa adaptations, alongside a
permission-scope determination for the Cephes-derived CPU functions. An upstream reply is
one possible source of evidence; it is not the sole closure mechanism. The existing
[audit](phase8-4-native-license-audit.md) and
[resolution research](phase8-4-native-license-resolution.md) remain applicable.

RC1's current packaging does not implement the proposed Juffa compliance route. Package
changes would require a reviewed PR, a new immutable candidate, new signatures and repeated
affected verification. Preserve RC1 and its receipt as historical evidence.

## Exact distribution under review

| Identity | Value |
| --- | --- |
| Candidate annotated-tag object | `95d5dd75be5a205dfe3148c5c752c3696316a61a` |
| Signed ZIP SHA-256 | `b4428fd0817273fd7387528c35519302c334aee444cbe6dcffe53d2a05f0ef2f` |
| Preliminary-record SHA-256 | `981507b908b7f07d1da37a5418ed4d6158fc0ff58e7293a1aef2e7b56c7c9ed4` |
| Final receipt SHA-256 | `3ef7b8093dfa7046331052814988ae8ad33e60b069ffbdb00c83375009360f03` |
| MLX source reviewed | `36969c094354086e15d2a327a9c50feb20b9cf11` (`v0.32.4`) |
| Vendor wheel SHA-256 | `514df92d41a286895d2564b4b18f4da45096bd7a4a0c47e1042f345ad5a7f936` |

[Minimum-host run 44834519307][minimum-run]
passed against this ZIP. It establishes technical execution, not permission to distribute.
The two reviewed headers also hash-match those installed by the pinned wheel. The existing
audit's vendor-binary/source association remains a limitation: no reproducible wheel rebuild
or complete compilation record is claimed.

## Finding 1: Juffa revision and distribution scope

The complete revision lists contain four `erf` and three `erfinv` revisions. Applying the
previous audit's bounded decimal-literal comparison gives:

| Shipped adaptation | Matching answer revision | API license | Comparison |
| --- | --- | --- | --- |
| Metal `erf` | [35148199 r4][erf-r4] | 4.0 | Literal multiset subset |
| Metal `erfinv` | [49743348 r3][erfinv-r3] | 4.0 | Same literal sequence |
| CPU SIMD `erfinv` | [49743348 r3][erfinv-r3] | 4.0 | Same literal multiset |

The matching revisions are dated 2022-06-18 (`erf`) and 2021-06-10 (`erfinv`). CPU inverse
literals have a different order. The table reports the API's CC BY-SA version.
Every earlier revision fails the respective comparison. In particular, the original
CC BY-SA 3.0 revisions do not supply the matching coefficients. This strengthens the
case for reviewing 4.0 as the operative route; it does not prove the exact copying event,
legal adaptation boundary or absence of another grant. CPU `erf` remains outside this
finding; its PyTorch provenance was already handled separately.

[Stack Overflow's revision policy](https://stackoverflow.com/help/licensing) assigns license
versions by contribution date and exposes the applicable revision license. The
[CC BY-SA 4.0 terms](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en), sections
2–3, permit sharing and adaptation subject to attribution, retained notices, source links,
modification indications, applicable ShareAlike terms and downstream restrictions rules.
The reviewer must determine the adapted material and how those duties apply to the compiled
MLX/Metal distribution. MIT labels alone do not settle that boundary.

[Creative Commons' software FAQ][cc-software]
explains that CC licenses lack software-specific source-distribution provisions. Shipping
source can aid traceability but is not a substitute for satisfying the applicable terms.
Its [compatible-license list](https://creativecommons.org/compatible-licenses/) does not
list MIT as an adapter-license option for BY-SA 4.0. This memo neither relicenses all of
jmlx nor assumes that three function-level notices are sufficient for the compiled result.

### RC1 packaging gap and proposed changes

The native main, sources and Javadoc JARs all contain the reviewed third-party notice file.
The replay matches the packaged `expm1f`, fastermath, NEON and Netlib notice fragments to
their pinned sources, normalizing whitespace only. Every non-directory `META-INF` entry is
hashed and scanned with case-insensitive function and license markers, including `CC BY-SA`,
`CC-BY-SA` and the expanded Creative Commons Attribution-ShareAlike name. No `erf`/`erfinv`
function marker or CC ShareAlike marker appears in those entries. These are bounded marker
observations, not a semantic proof about every possible attribution spelling. None of
these JARs contains native `.h`, `.c`, `.cpp` or `.metal` source entries. The native POM's
license names, URLs and scoped comments are recorded in full: MIT for the jmlx artifact code
and Apache-2.0 for compiled metal-cpp. These observations are reproducible from the signed ZIP.

For a compliance implementation, prepare the following for reviewer approval:

- Identify Norbert Juffa, each answer/revision and the three exact adapted functions; retain
  other applicable copyrights and notices, including Apple's.
- Supply the applicable license reference and modification history: Metal/SIMD translation,
  vectorized evaluation and the later `expm1f` substitution, without implying Juffa endorsed MLX.
- State the approved scope and adapter-license treatment in the legal texts and publication
  metadata. Identify how that treatment covers the affected compiled material.
- Consider a separately identified source/provenance attachment with exact source hashes and
  changes. Do not present it as a general CC source-code mandate or proof of a reproducible wheel.
- Confirm the distribution imposes no incompatible downstream terms; review all three JARs,
  native payload, POM and user-facing license descriptions together.

## Finding 2: Cephes permission and exact source generation

The CPU header expressly cites fastermath for exponential approximation and NEON math for
trigonometry. Both immediate projects' notices and Netlib's permission are already packaged.
The immutable immediate-source snapshots remain evidence of their terms, not proof that MLX
copied exactly those revisions or that every underlying contribution is covered.

Fresh Netlib `single.tgz` inspection finds all six selected exponential coefficients in
`exp2f.c` and all six selected sine/cosine coefficients in `sinf.c`. Both files label themselves
**Cephes Release 2.2, June 1992**, with Moshier copyrights. The checks compare magnitudes,
not control flow, signs, full provenance or copyrightability. Fastermath describes its basis
as Cephes 2.8. Older file labels can persist inside later releases, so these observations
neither establish nor exclude coverage by a grant concerning Release 2.8.

[Netlib's original statement](https://netlib.org/cephes/readme) permits free use and disclaims
support/guarantees. It does not enumerate modification, redistribution or sublicensing rights.
The [2004 correspondence](https://lists.debian.org/debian-legal/2004/12/msg00295.html)
reproduces Moshier's distinction between his own distribution terms and permission for the
recipients' distribution, followed by a Release 2.8 example with a BSD placeholder. The
[reply](https://lists.debian.org/debian-legal/2004/12/msg00297.html) discusses obtaining code
actually distributed with the boilerplate. Neither record identifies MLX's precise source chain.

The reviewer should decide whether the original permission covers this redistribution,
with a written interpretation of modification and downstream scope, or identify an applicable
licensed distribution containing the derived material. Popularity, a different project's
permission, a version label or retaining notices alone does not establish that conclusion.
No exact licensed-distribution chain has been established by this review.

## Decision record to complete

An owner/legal reviewer should record their name, role, date, applicable jurisdiction and
reasoned answers to these questions. Counsel review remains appropriate for the legal
interpretations; the technical replay cannot supply those determinations.

- **Juffa:** accept the source/license mapping or identify a different applicable grant;
  define the adapted material and lawful adapter-license treatment through compiled distribution.
  Closure needs the review rationale, approved notices/metadata, downstream-terms review and
  validated changed artifacts.
- **Cephes:** explain why existing permission covers the actual material and redistribution,
  or identify an applicable grant/licensed distribution. Closure needs that scoped rationale
  or an exact grant/source mapping, with required notices.

If either determination cannot be made, pursue direct permission or a separately reviewed
replacement. The review decision must precede marking a finding resolved. Technical receipt
validation, upstream silence and the owner's willingness to accept uncertainty do not create
missing rights. No external reviewer has been engaged or contacted by this work.

## Evidence replay

The [machine-readable review](plans/evidence/scoped-license-review-2026-11-21/review.json)
records the candidate, source projections, hashes and limits. The
[technical findings](plans/evidence/scoped-license-review-2026-11-21/technical-findings.json)
retain facts rather than republishing answer bodies or correspondence. Legal pages were
reviewed through the web tool; direct snapshot requests returned HTTP 403. No raw HTML hash
is claimed for those pages, and they are not technical replay inputs.

With the unchanged RC1 ZIP available locally, run Python 3.11+:

```sh
python3 req/plans/evidence/scoped-license-review-2026-11-21/reproduce.py \
  --inputs build/releaseAll/scoped-license-review-2026-11-21/inputs --fetch-missing \
  --bundle build/releaseAll/candidates/rc.1/transport/jmlx-mega-bundle.zip \
  --output build/releaseAll/scoped-license-review-2026-11-21/replayed.json
```

Omit `--fetch-missing` for cached-input replay. The API projection rejects malformed/empty
responses, mixed post IDs and noncontiguous or duplicate revision numbers/GUIDs. It sorts the
validated complete snapshot by revision number and retains post ID, revision number/GUID/date,
license, body and owner user ID. It excludes quota and mutable user-profile metadata. Byte
inputs are hashed unchanged. The script reuses the earlier function extractor/decimal regex,
reads selected Cephes tar members in memory and inspects the three nested native JARs without
executing them. The newest answer code blocks must hash-match the earlier audit's comparison
inputs. CPU sequence equality is recorded separately from multiset equality. Fastermath's Cephes
2.8 reference and both CPU source citations are recorded; the two correspondence inputs pin
snapshots for manual interpretation only. Their permission meaning is not replayed.

A local `hashlib` SHA-256 verifies the prior recipe before executing those same bytes, with
no dependency bytecode written. The bundle is read once; those bytes are hashed and inspected.
The reviewed manifest is the trust anchor, not an authenticated publisher signature.
Input integrity and equality to the committed technical report are checked; this is evidence
reproduction, not automated legal clearance. Fresh-source changes require review of this dated
snapshot rather than silently updating the report. RC1's tag, ZIP and final receipt stay unchanged.

Focused regressions in `tools/release-all/tests/test_scoped_license_review.py` cover the
pre-execution hash guard, stable verified bytes, revision shape/completeness, marker variants,
all-META-INF scans, source notice and POM recording, answer-code linkage and CPU literal order.
They run through `bash tools/release-all/selftest.sh` alongside the existing regressions.


[minimum-run]: https://github.com/Alipsa/jmlx/actions/runs/44834519307
[erf-r4]: https://stackoverflow.com/revisions/35148199/4
[erfinv-r3]: https://stackoverflow.com/revisions/49743348/3
[cc-software]: https://creativecommons.org/faq/#can-i-apply-a-creative-commons-license-to-software
