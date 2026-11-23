# Phase 8.4 replacement candidate preparation

Preparation began on 2026-11-23 after PR #117 merged. No RC3 tag, signed bundle,
receipt, owner acceptance or publication is asserted by this document. RC2 remains
blocked; its historical packet is unchanged.

The [recovery check](plans/evidence/rc3-preparation-2026-11-23/recovery-check.json)
compares the main checkout's complete native report with the retained RC2 preliminary
record: they match, including file digests, sizes, pins and deployment metadata.
All 104 pinned JUnit paths exist; none matches the recorded per-path SHA-256.
Time Machine has no configured destination or local snapshots, and no backup volume is
mounted. APFS lists one OS update snapshot; it does not establish recovery of candidate
data. Offline backups were not searched. The retained probe records command results,
source identity, working-tree state and the exact forbidden-path arguments. The probe uses
its own repository's Python helper code and checks its reviewed SHA-256 before execution.
It also verifies the inspected preliminary record against RC2's reviewed SHA-256 before
parsing it or reading its JUnit paths. It never loads Python helper code from the inspected
tree. Git still runs on that tree to record HEAD and status, with filesystem monitoring
disabled and optional locks suppressed. Use a repository whose Git configuration and
attributes you trust: configured filters may execute during status.

Reproduce it from the repository root with:

```sh
python3 req/plans/evidence/rc3-preparation-2026-11-23/recovery-check.py --root .
```

This read-only probe reports current state; timestamps and outputs may differ later.
Run it from a checkout that retains the reviewed helper digest
`85b4f940f67faf916a9263a3920a69763837176af6685cb8c8db131e4ca4e849`.
Commits `6acbc4d` and `228e694` contain that helper and earlier versions of this probe;
the retained JSON's `sourceCommit` identifies the version used for the latest run.
A later helper edit intentionally stops the probe until its new pin is reviewed. Preserve
this source checkout to reproduce the retained run without changing the pin.

The surviving RC2 inputs remain preserved outside Gradle clean scope at the location
recorded in the [backup summary](plans/evidence/rc2-final-review-2026-11-22/survivor-backup.json).
Do not overwrite those files or move the RC2 tag. Matching native binaries alone cannot
restore the missing producer JSON and original JUnit reports.

The preparation checkout is `~/release-candidates/jmlx-phase8.4-rc.3-preparation` at
`05e926abfec80803001e62f0da2e049a32d5f423`, with a verified copy of native staging.
It retains the initial preparation commit, including its provisional date changes; it is
not the final candidate checkout and is not updated by subsequent PR fixes.

The replacement keeps the approved nine-module tuple. The final pre-tag PR aligns its five
version-cut dates to 2026-11-23. If the candidate is cut on a later date, align them again
and repeat both jinja verification passes after that change.
Replacement evidence must identify its own source commit and ZIP digest throughout.

Before cutting the replacement:

1. Completed: [PR #119](https://github.com/Alipsa/jmlx/pull/119) merged the
   [final owner-acceptance upload gate](../tools/release-all/owner-acceptance.md) as
   `cd4271e76ec7f768e468ce8649f0217be362a495`.
   It binds acceptance to the actual replacement bundle, final receipt and committed review
   packet, including explicit jinja human sign-off. Implementation does not grant acceptance.
2. Merge the reviewed preparation and gate changes, then align and review the cut dates
   in the final pre-tag PR. Repeat jinja verification after changing its policy inputs.
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

## Pre-tag preparation (2026-11-23)

[PR #119](https://github.com/Alipsa/jmlx/pull/119) merged the owner-acceptance upload gate
as `cd4271e76ec7f768e468ce8649f0217be362a495`. It requires an externally authorized
acceptance digest after the replacement review packet is committed; no acceptance has
been issued.

The pre-tag branch aligns the five changelog dates to 2026-11-23. Two consecutive jinja
verification passes and the nine-module release dry-run passed at clean source commit
`b4ee4d75836f3f5fe4445e72a690dae2fd1b7887`. Both reports, logs and supporting evidence
are retained in the [pre-tag packet](plans/evidence/rc3-pre-tag-2026-11-23/review.json).
Public copies normalize local paths; the packet records their hashes and the original
hashes. Originals survive outside Gradle clean scope at
`~/release-candidates/jmlx-phase8.4-rc.3-pre-tag-2026-11-23/`.
Jinja human sign-off remains pending.

Review and merge the pre-tag PR before creating RC3. If the actual candidate cut falls on
a later date, realign the five dates and repeat both passes after that policy input change.
The tagged candidate still needs its own verification, audit, signed bundle and consumer
evidence under the [runbook](../tools/release-all/README.md). These pre-tag results do
not close those gates or authorize publication.
