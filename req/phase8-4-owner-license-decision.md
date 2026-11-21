# Phase 8.4 owner license decision record

Prepared 2026-11-21. Status: Cephes interpretation and scope confirmed; Juffa decision pending.

| Field | Record |
| --- | --- |
| Reviewer | Per Nyfelt |
| Role | jmlx creator and maintainer; project owner approving release decisions |
| Review context | Sweden, confirmed by the owner on 2026-11-21 |
| Confirmation date | 2026-11-21; owner statements in the preparation conversation |
| Applicable law | Not determined; pending assessment of relevant rights and grants |
| Assessment | Owner judgment based on the evidence; reviewer is not a lawyer |
| Juffa disposition | Pending owner assessment and final wording |
| Cephes disposition | Interpretation and Netlib-material basis accepted; validation pending |
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

The [existing-license investigation](phase8-4-juffa-license-interpretation.md) examines
reliance on MIT for MLX contributions together with Juffa's direct CC grant. It records both
the distributor-role argument and the remaining adapter-license counterargument. Its proposed
rationale remains unapproved; the owner's request to investigate is not acceptance.

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

## Cephes: approved owner interpretation

On 2026-11-21, Per Nyfelt approved the following interpretation in the preparation
conversation. This is a substantive owner decision, separate from approval to merge PR #108.
The owner's approval and merge of [PR #109](https://github.com/Alipsa/jmlx/pull/109) is the
repository record approving this text.
The reviewed proposal and decision template are the versions on `main` at
`e639c3d52c805641d0fdabb182d07c14b5106bd5`.

**Approved rationale:** the owner interprets Stephen Moshier's original permission that the
material "may be used freely" as permitting the identified Cephes-derived modifications and
their redistribution through jmlx, including downstream use. The wording is broad but does
not expressly enumerate those activities; this conclusion is the owner's interpretation of
the existing permission. Approval is limited to the material identified in the scoped review
and requires retaining the original permission, applicable intermediary notices and the
MLX MIT notice for Apple/MLX contributions. It does not establish a new grant or relabel the
material as BSD-licensed.

The owner confirmed on 2026-11-21 that "what you see here" is applied to the matching Netlib
archive material as the permission-covered basis for this scoped decision, despite
fastermath's 2.8 label. The basis is the identified archive material in `./exp2f.c` and
`./sinf.c`, labeled Release 2.2 and pinned below, rather than an assumed permission covering
all Cephes 2.8 distributions. This is the owner's interpretation of the permission's reach
to the identified MLX-derived material; the selected coefficient matches do not establish
an exact provenance chain or complete implementation equivalence.

**Recorded scope and evidence:** MLX cites fastermath and NEON math for its CPU exponential
and sine/cosine approximations. The [scoped review](phase8-4-scoped-license-review.md) and
its [technical report][technical-report] identify the reviewed material. Selected coefficient
magnitudes match Netlib files labeled Release 2.2; fastermath refers to 2.8. These observations
do not establish an exact BSD-licensed provenance chain. The
[2004 correspondence][cephes-email] and [reply][cephes-reply] remain contextual evidence,
not the grant relied on by this decision.

The exact reviewed source scope is the Cephes-derived approximation material in
`mlx::core::simd::exp<T, N>` and `mlx::core::simd::sincos<Sine, T, N>` in
`mlx/backend/cpu/simd/math.h` at MLX commit
`36969c094354086e15d2a327a9c50feb20b9cf11`. The complete header SHA-256 is:

```text
5dbe79caa33a228d1354b2621876dc0fb3629dd6733c8ee4ebd0471790b0a94a
```

The scope includes the corresponding compiled code distributed in `libmlx.dylib` in the
reference RC1 candidate identified above, subject to the scoped review's source/wheel
association limits; this does not establish a reproducible binary build. The replay maps
`./exp2f.c` to MLX `exp` and `./sinf.c` to MLX `sincos`. The Netlib archive member hashes
recorded in the technical report are:

- `./exp2f.c` (Release 2.2), SHA-256:

  ```text
  b4f9d463126d5deada585824952ca4ff875334775ecfcdefa2a19a41d84c7ad2
  ```

- `./sinf.c` (Release 2.2), SHA-256:

  ```text
  84b3f281ce0de625b1238da576cf34a97b449924bf91c3423a548d99d82f2284
  ```

These are member hashes, not the archive hash. The comparisons cover six selected
coefficient magnitudes per function, not complete implementations or permission equivalence.

The rights holder identified for the original Cephes material is Stephen L. Moshier. The
[Netlib statement][netlib] identifies him and describes the material as copyrighted by its
author. The owner relies on that author's original permission for the identified underlying
material, while retaining the separate fastermath/NEON notices for intermediary contributions
and the MLX MIT notice for Apple/MLX modifications.
This decision does not extend an intermediary contributor's grant to Moshier's rights.

| Owner decision field | Current value |
| --- | --- |
| Decision date and reviewed proposal commit | 2026-11-21; full commit recorded above |
| Permission/grant relied on | Original Netlib permission; not a new or inferred BSD grant |
| Rights holder(s) and authority for the permission relied on | Moshier; author's statement above |
| Modification, redistribution and downstream rationale | Approved; see Netlib basis above |
| Exact material/provenance covered | Pinned header, functions, Netlib members and binary above |
| Required notices and conditions | Retain Netlib, fastermath/NEON and MLX MIT notices |
| Final notice/license/POM paths, exact wording and SHA-256 hashes | Hashes below; review pending |
| Accept/reject/need evidence; conditions and responsible owner | Conditional accept; Per Nyfelt |
| Named implementation/review owner and unresolved questions | Follow-up below |

**Implementation and local validation:** the Cephes section title is now
`Cephes math provenance (original upstream permission statement)`. Its permission text is
unchanged. The matching inventory component and packaged-text hash are updated; the audit's
Cephes follow-up entry links the owner decision and new verification. Historical audit
verification and artifact hashes remain unchanged.

The [local validation record][cephes-validation] pins the final local legal-text and generated
POM hashes and all three JAR hashes. The [source/notice report][cephes-source-report] checks
26 components and the legal texts in the main, sources and Javadoc JARs. Native Gradle checks,
explicit payload verification and 17 native-license tests passed. The native payload remains
byte-identical to the earlier audit. These are producer checks, not candidate verification.

**Follow-up:** Per Nyfelt owns review and acceptance. Implementation review remains pending.
Review the changed title, retained notices and recorded hashes, then validate the final new
candidate before closing the Cephes finding. Any newly identified material outside the
reviewed scope needs another assessment. Applicable law remains undetermined as recorded
above. `NATIVE-LICENSE-2` remains open for implementation review and final candidate
validation; `NATIVE-LICENSE-1` remains open pending the separate Juffa decision. Neither a
replacement candidate nor Central publication is approved by this decision.

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

[cephes-validation]: plans/evidence/cephes-notice-validation-2026-11-21/validation.json
[cephes-source-report]: plans/evidence/cephes-notice-validation-2026-11-21/source-notice-report.json
