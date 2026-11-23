# Final publication acceptance contract

This gate records an explicit owner decision after replacement-candidate review. It does
not infer consent from technical checks, a merged preparation PR, or PR #115's conditional
metadata acceptance. No actual acceptance is included with this implementation.

After creating and revalidating the final technical receipt, commit the replacement review
packet and its evidence in a follow-up PR based on the candidate commit or a descendant.
Keep the tagged candidate unchanged. The packet
uses this JSON structure (angle-bracket values below are placeholders):

```json
{
  "schema": "jmlx-publication-review-1",
  "candidate": "<copy the entire preliminary record's tag object>",
  "bundleSha256": "<signed ZIP SHA-256>",
  "verifiedReceiptSha256": "<final technical receipt SHA-256>",
  "acceptanceEligibility": {"eligible": true},
  "ownerDecisions": {
    "signedCandidateAndDifferences": "recorded",
    "deliveryTerms": "recorded",
    "licenseFindingDisposition": "recorded",
    "jinjaHumanSignOff": "recorded",
    "portalCapacityConfirmed": "recorded",
    "publicationAuthorized": "recorded"
  },
  "reviewDocument": {"path": "req/<replacement-review>.md", "sha256": "<SHA-256>"},
  "retainedFiles": [
    {"path": "req/plans/evidence/<replacement>/<file>", "sha256": "<SHA-256>"}
  ]
}
```

`candidate` must be an object copied verbatim, not the placeholder string above. Include
the final receipt, both jinja verification reports/logs, the jinja human sign-off and all
other reviewed evidence in `retainedFiles`. The review document must explain signed
artifact differences and both Javadoc toolchains, delivery terms, the disposition of the
license findings, and the current Portal capacity snapshot. Pins describe immutable Git
blobs at the packet commit, so subsequent changes to living release documents do not
invalidate them. Use repository-relative paths without `.` or `..` components. Ensure
the candidate repository has fetched the reviewed packet commit before verification.
The verifier requires that commit to descend from the candidate. It does not check a local
`origin/main` ref, which may be stale; the separately authorized acceptance digest attests
to the exact reviewed commit, and the operator must confirm its PR was reviewed and merged.

A technical packet may be committed while owner decisions, jinja sign-off or the current
Portal snapshot are explicitly pending. Such a packet cannot support acceptance. Before
acceptance, retain those completed records in a superseding reviewed packet commit,
update the review document and its digest, regenerate the manifest and update its
checklist digest. Review and merge that superseding commit, then cite its new commit and
manifest digest in the acceptance record. Setting acceptance decisions to `true` does
not replace the missing records. The verifier requires literal `true` in
`acceptanceEligibility.eligible` and all six required `ownerDecisions` entries to be
`"recorded"`; missing fields and any additional pending entries also fail closed.
Those packet fields must reflect completed retained records, not placeholders. The
verifier checks these gates, pins and owner attestations; it does not independently
determine whether the document's substantive review is complete.

After the owner reviews that packet, retain their explicit decision in a separate JSON
record. Its required fields are:

```json
{
  "schema": "jmlx-publication-acceptance-1",
  "reviewer": "<owner>",
  "acceptedAt": "<UTC timestamp>",
  "approvalStatement": "<owner's explicit authorization for this ZIP and receipt>",
  "candidate": "<copy the entire preliminary record's tag object>",
  "bundleSha256": "<signed ZIP SHA-256>",
  "verifiedReceiptSha256": "<final technical receipt SHA-256>",
  "reviewPacket": {
    "commit": "<full reviewed packet commit SHA>",
    "path": "req/plans/evidence/<replacement>/publication-review.json",
    "sha256": "<review manifest SHA-256>"
  },
  "decisions": {
    "signedCandidateAndDifferences": false,
    "deliveryTerms": false,
    "licenseFindingDisposition": false,
    "jinjaHumanSignOff": false,
    "portalCapacityConfirmed": false,
    "publicationAuthorized": false
  }
}
```

Both JSON records reject duplicate keys at every nesting level. `acceptedAt` must be an
ISO 8601 timestamp with a `T` separator and an explicit UTC offset (`Z` or `+00:00`).
The packet must retain a blob whose SHA-256 matches the final receipt being uploaded.
Jinja report/log completeness and human sign-off remain owner attestations.

All decisions must be literal `true` to upload. Set them only from actual owner approval;
the example deliberately grants none. Retain the acceptance outside the immutable
candidate source tree, in durable storage with the rest of the release packet. The owner
must authorize its exact SHA-256 separately; do not blindly compute a fresh digest from
whatever file is present at upload time.

```sh
./releaseAll.sh --publish-verified build/releaseAll/verified.json \
  --acceptance <durable-owner-acceptance.json> \
  --acceptance-sha256 <owner-authorized-acceptance-SHA-256>
```

The shell and Gradle upload action both revalidate technical evidence, acceptance bytes,
candidate/ZIP/receipt identity, and every pinned review blob before submission. Missing,
edited, historical or incomplete acceptance fails closed. Technical receipt creation and
`final_receipt.py --verify` remain available before acceptance; `--plugin-digests` requires
both acceptance arguments because it supplies upload anchors. Status-only `--resume`
retains its existing journal-based behavior and cannot upload a new bundle.

These records are operator attestations, not cryptographic authentication of a person or
a substitute for human review of the packet's contents. Protect the accepted digest and
the local release tooling as part of the manual, credentialed publication procedure.
