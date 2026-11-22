# Phase 8.4 native-license compliance implementation

2026-11-22. Scope and wording approved by Per Nyfelt in the implementation conversation.
The [owner record](../phase8-4-owner-license-decision.md) and
[acceptance record](evidence/juffa-implementation-2026-11-22/acceptance.json) pin the decision
and the exact packet at PR #113's merge. The historical packet remains unchanged.

## Implemented behavior

The native main, sources and Javadoc JARs retain the approved Juffa section, native NOTICE
and unmodified complete CC BY-SA 4.0 text. Existing upstream sections remain intact; the
new section's initial LF maintains the existing separator layout. Cephes retains its
approved original-permission treatment, alongside the immediate BSD and zlib notices.

The POM carries the approved six scoped entries and literal comments. Its CC comment links
notice commit `551eebeeb66a80ef3af467a6c1b4995c2a7f7034`, committed before the POM change.
The native README and draft release notes contain the exact approved publication paragraph,
with decision and notice links beside it. The audit describes the expanded scoped metadata.
Final literal metadata review remains pending; these changes do not resolve either finding.

`req/native-authored-attributions.json` separates jmlx-authored attribution requirements from
upstream excerpts. The audit verifies its pinned local approval evidence and upstream source
hashes, then checks the authored section. Existing two-sided upstream excerpt checks remain.
The producer and independent consumer both require the CC text and exact authored title,
including when someone deletes their inventory/policy entries and refreshes enclosing hashes.

`release_inputs.py` extracts raw sections without normalizing line endings or whitespace.
It rejects malformed separator suffixes, unsupported leading whitespace and duplicate titles.
Explicit release verification compares full license name/URL/comment tuples with independent
literal expectations and requires the notice commit in reviewed main and candidate history.
It compares the linked, local and packaged raw Juffa sections, including the last section's
final LF. Bundle validation checks the native POM and all three JARs from the actual ZIP.

The explicit pre-tag task is `:jmlx-native-macos-arm64:verifyNativeReleaseLicenses`; the
combined-release task inventory now requires it to execute alongside native payload checks.
Ordinary builds remain free of Python and deep-history requirements. An unstaged native
checkout still builds locally and still refuses publication. Consumer verification resolves
the POM through its POM-only repositories and uses separate Groovy comparators. The pure-Java
profile registers no native license task and needs no Git history.

## Verification and review evidence

The [validation record](evidence/juffa-implementation-2026-11-22/validation.json) identifies
commands, results, local unsigned JAR/POM/module hashes and verifier input hashes. The
retained literal POM, module metadata, CC comment and source/notice report are in the same
evidence directory. This is implementation evidence, not a signed candidate receipt.

Regressions cover raw LF/CRLF differences, section title/layout/EOF changes, malformed
separators, unsupported whitespace, duplicate titles, missing/changed/duplicate license
entries, altered scope/comments/permalinks, missing policy/text despite updated inventory,
and unavailable/non-ancestor notice history. The CLI history fixtures create synthetic Git
repositories without fetching the checkout's history or requiring a native runtime or tag.
The standalone Groovy fixtures independently reject the same POM mutations and legal-text,
section and policy deletion. Both actual consumer profiles are exercised separately.

## Remaining release work

Merge this implementation with a merge commit; do not squash or rebase the notice commit.
Sync reviewed main and run `verifyNativeReleaseLicenses` before tagging. Its real-checkout
pre-merge ancestry refusal is expected; positive execution is covered in an isolated history
fixture. Shallow contributor/PR builds must not run this explicit task.

Per Nyfelt must review the final literal metadata and the actual downstream distribution
terms and measures. Any resulting wording change needs another reviewed notice commit before
updating its permalink and independent expectations. Both `NATIVE-LICENSE-1` and
`NATIVE-LICENSE-2` stay open until their separate validation/decision conditions are met.

After review, follow the release runbook for a new immutable candidate, fresh signatures and
exact-bundle consumer and physical minimum-host verification. RC1, its tag, ZIP and receipts
are unchanged. This implementation does not create a candidate, provenance classifier or
release asset, and does not authorize Maven Central publication.
