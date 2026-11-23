# Phase 8.4 replacement candidate preparation

Preparation began on 2026-11-23 after PR #117 merged. No RC3 tag, signed bundle,
receipt, owner acceptance or publication is asserted by this document. RC2 remains
blocked; its historical packet is unchanged.

The [recovery check](plans/evidence/rc3-preparation-2026-11-23/recovery-check.json)
compares the main checkout's complete native report with the retained RC2 preliminary
record: they match, including file digests, sizes, pins and deployment metadata.
None of the 104 pinned JUnit XML reports matches the main checkout's reports.
Time Machine has no configured destination and no backup volume is mounted. This
does not exclude an offline backup, but no local recovery source was found.

The surviving RC2 inputs remain preserved outside Gradle clean scope at the location
recorded in the [backup summary](plans/evidence/rc2-final-review-2026-11-22/survivor-backup.json).
Do not overwrite those files or move the RC2 tag. Matching native binaries alone cannot
restore the missing producer JSON and original JUnit reports.

The replacement keeps the approved nine-module tuple. Its five version-cut dates are
provisionally 2026-11-23 and must be reviewed against the actual cut date before tagging.
Replacement evidence must identify its own source commit and ZIP digest throughout.

Before cutting the replacement:

1. Implement and review the digest-pinned final owner-acceptance upload gate identified
   in the RC2 review. It must bind acceptance to the actual replacement bundle and
   reviewed packet, including the jinja human sign-off. Do not manufacture acceptance.
2. Merge the reviewed preparation and gate changes, then recheck the version-cut dates.
   Create an immutable annotated `jmlx-phase8.4-rc.3` tag at the reviewed source commit.
3. Use durable candidate and audit checkouts under `$HOME/release-candidates/`, with
   original absolute paths preserved. Follow the runbook's checks, dry-run and signed
   bundle preparation. Record both actual Javadoc producer toolchains explicitly.
4. Repeat the pure-Java audit, both jinja release-verification passes, all exact-bundle
   consumer runs and the minimum-host window for this candidate. Recheck sharing and
   runner isolation before the window; RC2 observations are historical.
5. Preserve the complete candidate and audit checkouts, including ignored output, and
   verify the backup against every receipt input. Exercise final-receipt revalidation
   before ending the window. Retain timestamps and source commits for each probe.

Owner signed-candidate/delivery review, jinja human sign-off and current Portal capacity
confirmation remain open. Publication requires separate explicit authorization of the
replacement ZIP digest after these gates pass. RC2 approval or evidence cannot substitute
for replacement-candidate evidence.
