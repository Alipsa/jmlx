# Phase 8.4 Juffa existing-license interpretation

2026-11-21. Research based on main `ea78fa6bb1a45fc87c4e7213bcf08f7bf92234ca`. Status:
proposed interpretation for owner review; no Juffa approval or finding closure.

## Assessment

There is a reasonably arguable redistribution basis using the two existing grants together:
MLX's MIT grant for rights its licensors hold, and Juffa's CC BY-SA 4.0 grant for his covered
material. That is a stronger argument than the earlier recommendation to wait for proof of the
upstream adapter-license treatment acknowledged. It requires jmlx's own CC compliance. The
evidence does not establish an MIT-only grant for Juffa's exact implementations.

This is an interpretation to assess, not a finding that MLX breached a license or that every
condition of the proposed route is met. The owner remains the decision-maker under the [owner
record](phase8-4-owner-license-decision.md). Sweden is the review context; applicable law
remains undetermined. No outside approval is required by this workflow.

## Rights and roles

The pinned [MLX MIT license][mit] expressly permits copying, modification, redistribution and
sublicensing, subject to retaining its copyright and permission notice. Its disclaimer also
disclaims warranties, expressly including non-infringement. It supports reliance on the rights
Apple/MLX can grant; it supplies neither a third-party ownership warranty nor an identified
separate grant from Juffa. The [initial source commit][initial] already contains both Metal
functions and an explicit Stack Overflow citation for `erf`. The [copyright commit][ack] adds
Apple headers and updates the MIT copyright line; its relevant patches contain no Juffa
permission record. These observations concern the inspected records, not private grants.

The [CC legal code][cc-legal] grants reproduction and sharing of adapted material in section
2(a)(1). Section 3(a) applies to sharing, including modified material. Section 3(b) addresses
adapted material the licensee produces. Sections 2(a)(5)(A)-(C) address downstream offers and
restrictions; section 6(a) terminates the noncomplying licensee's rights.

CC's [FAQ][cc-faq] says a breach terminates only the breaching licensee's license. It also
explains that the original license remains applicable after adaptation and recipients must
observe both the original and adapter's licenses. Those points support examining jmlx's own
permissions and conduct, rather than assuming an upstream compliance question automatically
extinguishes its rights. They do not establish that this particular adaptation is compliant.

The local [binary observation][binary-observation] confirms that `libmlx.dylib` and
`mlx.metallib` are byte-identical across the pinned wheel, staged runtime and rebuilt native
JAR. [Bootstrap](../scripts/bootstrap-native.sh) copies those vendor files; it builds mlx-c
separately. The observation covers the current producer JAR, not a newly signed candidate. It
strengthens the distributor-role argument for these two files. It does not establish a
reproducible MLX build, reconstruct every compiled function, or decide the legal boundary.

## The strongest interpretation and its counterargument

**Proposed inference:** jmlx can obtain permission for Juffa's covered contribution directly
under CC, while relying on MIT for Apple/MLX contributions, and redistribute the existing
vendor binaries with both sets of applicable terms preserved. It would not need to assert that
Apple relicensed Juffa's rights, or to choose a new license for Apple's contributions. An
upstream request can remain useful without being the exclusive route to a decision.

**Remaining counterargument:** treating MIT as the identified grant for Apple/MLX changes does
not demonstrate a conforming ShareAlike adapter's license. MIT is absent from CC's [approved
compatibility list][cc-compatible]. The direct-permissions argument must explain why the
combination suffices for downstream redistribution despite that unresolved upstream treatment.
An automatic downstream offer is not proof that MIT qualifies as a compatible adapter's
license. The distinction is material and should appear in an owner rationale.

My revised assessment is that requiring proof of upstream compliance as an absolute
prerequisite was too categorical. A reasoned owner decision can evaluate this direct-grants
interpretation. This investigation does not establish that the counterargument is legally
settled, or silently replace the still-pending decision in the owner record.

## Could the reused mathematics require no Juffa copyright permission?

There is a separate possible MIT-only theory: only unprotected mathematical ideas or facts
were reused, with MLX supplying its own protected expression. [Directive
2009/24/EC][directive], Article 1(2)-(3), distinguishes program expression and originality
from underlying ideas. In [SAS Institute, C-406/10][sas], paragraphs 39-46, the CJEU
distinguishes functionality from copied source/object expression. These authorities do not
decide this code's status or establish which law governs every distribution.

The [existing comparison](phase8-4-native-license-resolution.md) establishes coefficient
relationships, not copyrightability or independent authorship. The initial Metal code also
preserves evaluation structure, error comments and hexadecimal coefficient annotations. That
is reason to examine expression, rather than infer an exemption from numerical matches alone.
This review does not establish the MIT-only theory. Neither does it claim ownership of an
algorithm gives its author rights over every implementation.

## Other inspected routes

The [Stack Overflow licensing page][so-license] and the pinned revision evidence identify the
CC source-license route. The two retained BSD forum samples differ from the shipped versions.
A BSD grant permits modifications; differing coefficients alone do not prove that no
permissively licensed derivation is possible. The existing literal comparison also does not
establish a BSD derivation of the shipped expressive material. No new blanket grant was
identified in the inspected answer/profile pages or relevant GitHub patches. That bounded
search is not evidence that a private or uninspected public grant cannot exist.

The [upstream issue][mlx-issue] was still open with zero comments when checked on 2026-11-21.
No issue, comment, email or permission request was sent by this investigation.

## Proposed owner rationale for review

The following is a draft, not an approved determination:

> The owner would rely on Juffa's recorded CC BY-SA 4.0 grants for his covered material and
> MLX's MIT grant for Apple/MLX contributions. For the identified existing vendor binaries,
> jmlx would act as a distributor and meet its own applicable attribution, modification-notice
> and downstream conditions. This interpretation treats the two grants as the redistribution
> basis despite the unresolved upstream adapter-license treatment; it does not describe MIT as
> a CC-compatible adapter's license or claim an MIT-only grant for Juffa's contribution.
> Acceptance would depend on the scoped boundary and final notices, POM and distribution terms
> being approved and validated. New changes by jmlx would need a separate assessment.

Before accepting that rationale, the owner must decide whether it adequately addresses the
counterargument above. The material scope remains the three mappings in the [compliance
proposal](phase8-4-license-compliance-proposal.md): Metal `erf`, Metal `erfinv` and CPU SIMD
`erfinv`, including their compiled forms. Whether that boundary excludes other parts of a
combined binary is a legal judgment, not an implication of naming three functions. Final
approved wording/hashes and downstream-terms examination remain outstanding.

If accepted, finalize the conditional notice and its scoped license(s), implement the
proposal's enforcement, and validate a new candidate. Preserve the existing MIT notices,
retain Juffa attribution and CC terms where applicable, and avoid metadata suggesting those
rights are all MIT-only. If the owner cannot accept the interpretation, clarification or a
reviewed replacement remains available. Neither finding nor the publication gate changes here.

## Evidence limits

The historical snapshots and their digests are unchanged. Legal/FAQ pages were read through
the web tool on 2026-11-21; no raw-page hash or automatic legal replay is claimed. GitHub
file/history checks used the pinned source commit and the linked full commit IDs. The local
binary observation records independently inspectable hashes and exact input identities; it is
a local inspection, not a signed provenance attestation. No new runtime tests are needed for
this research-only change.

[mit]: https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/LICENSE
[initial]: https://github.com/ml-explore/mlx/commit/d1f86272a27b5a0729378ac75bb03727c7c0e553
[ack]: https://github.com/ml-explore/mlx/commit/46a39e5b1ff92d93833d9710ef64db207ce274ea
[cc-legal]: https://creativecommons.org/licenses/by-sa/4.0/legalcode.en
[cc-faq]: https://creativecommons.org/faq/
[cc-compatible]: https://creativecommons.org/compatible-licenses/
[directive]: https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32009L0024
[sas]: https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:62010CJ0406
[so-license]: https://stackoverflow.com/help/licensing
[mlx-issue]: https://github.com/ml-explore/mlx/issues/4902
[binary-observation]: plans/evidence/juffa-license-interpretation-2026-11-21/binary-observation.json
