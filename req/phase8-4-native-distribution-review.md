# Native distribution terms and final metadata review

2026-11-22. Prepared for Per Nyfelt after merging PR #114. Owner acceptance remains pending.
This review advances the remaining final-metadata and downstream-terms decisions from the
[owner record](phase8-4-owner-license-decision.md); it does not close either license finding.

## Exact material for the decision

The [review manifest][review] records the PR #114 merge, successful post-merge native payload
and license tasks, the immutable notice commit, and SHA-256/size of the local runtime, sources
and Javadoc JARs. It records the five legal-text hashes from each JAR and the exact POM/module
hashes, six actual POM license tuples, publication-source hashes and approval status.
The [task report][task-report] retains the successful six-license and three-JAR result.
These are local unsigned artifacts, not a new signed release candidate.

The retained literal [POM][pom], [module metadata][module] and [CC comment][comment] from the
implementation review remain the final metadata proposed for acceptance. The review manifest
records byte equality against those literals, rather than relying on the template approval.
The approved notice is linked from the POM to commit
`551eebeeb66a80ef3af467a6c1b4995c2a7f7034`. Its ancestry and raw section freshness now pass in
fetched origin/main and the merged checkout; the earlier pre-merge refusal is superseded.

## Actual surfaces and retained terms

The artifact's MIT LICENSE grants rights in jmlx native artifact code and retains its notice
and disclaimer conditions. NOTICE distinguishes that grant from upstream contributions and
points to the complete Apache and CC texts and THIRD-PARTY-NOTICES. The POM's MIT, Apache,
CC, BSD-2, BSD-3 and zlib entries identify their scoped material with illustrative examples.
References and attributions are described as retained notices, without asserting their
expression is present merely because an algorithm was referenced.

Juffa's creator/display name/profile, answer revisions, author revision notes, visible MLX
modifications, retained Apple notices and direct-CC/MIT basis remain in the authored section.
The section distinguishes the three reviewed functions and their corresponding compiled
forms from unrelated MLX material and Java code. It expressly identifies no CC adapter's
license and retains the unresolved section 2(a)(5)(B) counterargument. These observations
implement the owner's conditional interpretation, rather than creating a new grant.

The Cephes section retains Moshier's original permission and disclaimer under the accepted
scoped interpretation. Immediate fastermath/NEON and MLX notices remain. The differing BSD
forum samples do not grant rights for the exact reviewed Juffa versions, and future changes
by jmlx still require their own assessment.

The native README and draft release notes contain the exact previously approved publication
paragraph, beside links to the decision and immutable notice. The inspected Gradle module
metadata identifies the component, variants and artifact checksums; no additional recipient
license text was identified there. The main/sources/Javadoc META-INF entries and their hashes
are enumerated in the review manifest. No jmlx-added recipient restriction on licensed
copying, modification or redistribution was identified in these inspected surfaces.

No encrypted entries were observed in the three JARs. This is a bounded archive observation,
not a legal determination about every possible technological measure. The signatures used
by the later release flow provide integrity evidence; their presence alone would settle
neither additional-terms compliance nor the actual signed delivery path's treatment.

## Reasoning for the proposed owner conclusion

The [CC legal code][cc], sections 2(a)(5)(C) and 3, frames the additional-restrictions and
attribution questions. This review applies the owner's already recorded scope and
existing-grants interpretation; it does not independently resolve the adapter-license issue.

Within that interpretation, the inspected surfaces preserve the original grants and their
conditions, identify the material and changes, and distinguish each contributor's terms.
They do not replace Juffa's terms with jmlx's MIT grant or expand the CC label to unrelated
contributions. Retaining other upstream notices for their own material does not introduce a
new jmlx recipient contract for the scoped Juffa material. The no-additional-restrictions
statement in the authored notice matches the inspected LICENSE/NOTICE/POM/publication text.
Together, those observations support the conditional owner conclusion proposed below.
Unencrypted archives alone would not support it.

Acceptance remains limited to these exact local terms and metadata. Actual signed candidate
bytes, external publication/delivery terms and any changed technical measures must preserve
the same treatment and be reviewed before finding closure or publication. A changed surface
or scope requires another assessment. Existing source/wheel and compiled-boundary limits
remain; the review neither proves a vendor-wheel rebuild nor identifies binary offsets for
a legally separable work.

## Proposed decision for Per Nyfelt

The following is proposed text for the owner record, not an approval already received:

> I approve the final literal native POM, module metadata and CC comment identified by the
> review manifest's hashes, together with the retained Cephes implementation notices. Under
> my recorded conditional MIT/direct-CC interpretation and approved scope, I accept the
> reviewed downstream terms: the original grants, attribution and modification indications
> are retained, and the inspected local distribution surfaces add no jmlx restriction or
> technological measure against exercising the covered rights. I retain the recorded
> adapter-license counterargument. This conclusion is conditional on the signed candidate
> and its actual distribution preserving these terms and passing the required artifact,
> consumer and physical minimum-host validation. Both findings remain open until those
> conditions are met. This does not authorize Maven Central publication.

Approval would finish the proposed final-literal-metadata and local downstream-terms review.
It would permit moving to the next candidate-preparation steps under the release runbook;
it would not accept an unspecified future artifact. Reject or amend the statement if it
exceeds the owner's intended conclusion. Record any approval against the full reviewed
commit and the SHA-256 of this Markdown file and review manifest in the separate owner
record, without self-hashing either document.

Until that decision is received, the owner record and audit keep both findings open.
No replacement candidate, classifier, GitHub release asset or publication is created here.

[review]: plans/evidence/native-distribution-review-2026-11-22/review.json
[task-report]: plans/evidence/native-distribution-review-2026-11-22/postmerge-license-report.json
[pom]: plans/evidence/juffa-implementation-2026-11-22/native-pom.xml
[module]: plans/evidence/juffa-implementation-2026-11-22/native-module.json
[comment]: plans/evidence/juffa-implementation-2026-11-22/literal-cc-pom-comment.txt
[cc]: https://creativecommons.org/licenses/by-sa/4.0/legalcode.en
