# Phase 8.4 owner license decision record

Prepared 2026-11-21. Status: reviewer confirmed; substantive decisions pending.

| Field | Record |
| --- | --- |
| Reviewer | Per Nyfelt |
| Role | jmlx creator and maintainer; project owner approving release decisions |
| Review context | Sweden, confirmed by the owner on 2026-11-21 |
| Confirmation date | 2026-11-21; owner statements in the preparation conversation |
| Applicable law | Not determined; pending assessment of relevant rights and grants |
| Assessment | Owner judgment based on the evidence; reviewer is not a lawyer |
| Juffa disposition | Pending owner assessment and final wording |
| Cephes disposition | Pending owner assessment and rationale |
| Release gate | Open; neither finding has been resolved |

The owner makes jmlx's acceptance and release decisions. This workflow does not require an
external lawyer's approval; independent advice is optional. This record distinguishes an
owner's interpretation of existing permissions from a new permission granted by a rights
holder. Recording Sweden identifies the review context; it does not assert a Swedish-law
conclusion or establish that every relevant right is governed only by Swedish law.

The tables in this record are the authoritative decision template. The proposal lists the
required coverage; every pending field here must be completed or explicitly marked not
applicable with a reason. Owner statements above come from the preparation conversation,
not independently verified repository evidence.

## Evidence being assessed

The [implementation proposal](phase8-4-license-compliance-proposal.md) was reviewed and
merged in PR #107 as `7e826a25a54fadedf8a3ca2ed3e6d59d0ecd4555`. Its review approves the
proposal document; no substantive license determination was supplied with merge approval.

The [scoped review](phase8-4-scoped-license-review.md) records the bounded comparisons and
RC1 packaging. Its committed
[technical-findings.json][technical-report] has SHA-256:

```text
d37cba5d97188a2936abfb784603c5e19bbc94cc9c5e876b32ce09a20d3b1124
```

The committed [review.json][review-manifest], which pins that report hash, has SHA-256:

```text
b36e6b36b5a28730fb098ff8f5c205aa99efb57e3a473c2ee322117bfd4c5346
```

The reference RC1 candidate `jmlx-phase8.4-rc.1` signed ZIP remains:

```text
b4428fd0817273fd7387528c35519302c334aee444cbe6dcffe53d2a05f0ef2f
```

The [audit](phase8-4-native-license-audit.md) and
[resolution research](phase8-4-native-license-resolution.md) supply the provenance context.
No replacement candidate, new grant or changed notice is established by this record.

## Juffa: the decision to make

**Supported by the recorded evidence:** the bounded comparisons link Metal `erf` to answer
35148199 revision 4, and Metal/CPU SIMD `erfinv` to answer 49743348 revision 3. Those revision
records identify CC BY-SA 4.0. CPU `erf` is separate PyTorch-derived material. RC1 lacks the
proposed attribution and CC notice. The reviewed MLX headers retain Apple copyrights.

**Still to assess:** the scope and license basis for redistributing the adapted source and
compiled material, including Apple's/MLX contributors' modifications. The general MLX MIT
grant and Juffa's answer licenses are evidence to consider together; the review has not
established how they settle the adapter-license treatment for these exact adaptations.
The differing BSD forum examples do not establish an alternative grant for this code.

The [CC legal code][cc-legal], sections 1(b), 2(a)(5) and 3, distinguishes the adapter's
contributions, downstream offers/restrictions and attribution. The owner's decision should identify
the existing rights basis and its scope; jmlx's project approval does not itself license
Apple's or Juffa's contributions. This is the question for the owner's assessment, not an assertion
that a separate permission request is always required.

Choose a disposition and supply the reasoning:

- **Accept an existing licensing basis:** identify the license(s)/grant(s) relied on, why they
  cover these adaptations and their compiled distribution, and the scope of each. Specify
  how that scope treats surrounding MLX code. Then approve final attribution, license and
  POM wording plus the downstream-terms assessment before closing the finding.
- **Need more evidence:** identify the exact unresolved question and the evidence needed.
  The existing [MLX issue #4902][mlx-issue] is one route; additional advice is optional.
- **Replace the affected material:** choose a separately reviewed implementation/pin change
  and repeat the numerical, license and candidate verification required by the release plan.

| Owner decision field | Current value |
| --- | --- |
| Decision date and reviewed proposal commit | Pending |
| Technical source/revision mapping accepted, with rationale/references? | Pending |
| Source and compiled adaptation boundary | Pending |
| Rights holders and existing licensing basis | Pending |
| Reasoned adapter-license interpretation | Pending |
| Downstream terms examined and conclusion | Pending |
| Final notice/license/POM paths, exact wording and SHA-256 hashes | Pending; notice is a draft |
| Accept/reject/need evidence; conditions and responsible owner | Pending |
| Named implementation/review owner and unresolved questions | Pending |

## Cephes: the separate decision to make

**Supported by the recorded evidence:** MLX cites fastermath and NEON math for the relevant
CPU approximations. Their notices and the [original Netlib permission][netlib] are retained.
Netlib says the material may be used freely and disclaims support/guarantees. Selected
coefficients match Netlib files labeled Release 2.2; fastermath refers to 2.8.

**Still to assess:** whether that original permission covers the actual modifications,
redistribution and downstream use. The [2004 correspondence][cephes-email] and
[reply][cephes-reply] provide context; they do not by themselves trace MLX's exact chain to a
BSD-licensed distribution. Do not turn an interpretation of the original permission into a
claim that a new BSD grant was obtained.

Choose a disposition and supply the reasoning:

- **Accept the original permission as covering this use:** explain why its wording, source
  context and recorded provenance cover modification, distribution and downstream scope.
  Identify any conditions and the notices to retain. This would be the owner's documented
  interpretation of the existing permission, not a newly established grant.
- **Rely on another applicable grant:** identify its grantor, scope, covered material,
  downstream applicability and evidence linking it to these functions.
- **Need more evidence or replace:** state which route and what must be established next.

| Owner decision field | Current value |
| --- | --- |
| Decision date and reviewed proposal commit | Pending |
| Permission/grant relied on | Pending |
| Modification, redistribution and downstream rationale | Pending |
| Exact material/provenance covered | Pending |
| Required notices and conditions | Pending |
| Final notice/license/POM paths, exact wording and SHA-256 hashes | Pending |
| Accept/reject/need evidence; conditions and responsible owner | Pending |
| Named implementation/review owner and unresolved questions | Pending |

## What follows the owner's decisions

Record the owner's answers in a reviewed update, including the date, rationale, evidence and any
conditions. A statement accepting unspecified risk is not a substitute for the permission
interpretation being recorded. Implementation preparation can be authorized conditionally;
that does not itself close either finding.

Once the basis and scope are established, finalize the draft notice and implement the
proposal's packaging and independent enforcement checks. Resolve findings separately only
when the documented interpretation and changed-artifact validation support doing so.
Keep RC1 and its receipts unchanged; changed artifacts need a new candidate and verification.
Central publication still needs the owner's separate approval of the exact verified bundle.

[cc-legal]: https://creativecommons.org/licenses/by-sa/4.0/legalcode.en
[netlib]: https://netlib.org/cephes/readme
[technical-report]: plans/evidence/scoped-license-review-2026-11-21/technical-findings.json
[review-manifest]: plans/evidence/scoped-license-review-2026-11-21/review.json
[mlx-issue]: https://github.com/ml-explore/mlx/issues/4902
[cephes-email]: https://lists.debian.org/debian-legal/2004/12/msg00295.html
[cephes-reply]: https://lists.debian.org/debian-legal/2004/12/msg00297.html
