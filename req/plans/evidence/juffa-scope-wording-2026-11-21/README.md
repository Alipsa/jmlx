# Juffa scope and wording candidates

2026-11-21. Prepared from main `332d8e1bee00140f7c2f6901ac93e75623d67d46`, the merge of
PR #112 recording the owner's conditional existing-grants interpretation.
Status: proposed scope and text; owner approval remains outstanding.

These files are outside the native module. No production task packages them. This packet
supersedes the earlier draft's unresolved-paragraph prompt for the next review; it does not
change the earlier draft, RC1, the inventory, packaged notices, POM or audit finding status.
The [owner record](../../../phase8-4-owner-license-decision.md) remains the decision authority.

## Proposed mapping acceptance and source/compiled scope

Accept the three mappings as the provenance basis, with the limits of the
[technical report](../scoped-license-review-2026-11-21/technical-findings.json):

- Metal `erf(float)` in `mlx/backend/metal/kernels/erf.h`: answer 35148199 revision 4.
- Metal `erfinv(float)` in that same header: answer 49743348 revision 3.
- `mlx::core::simd::erfinv<T, N>` in `mlx/backend/cpu/simd/math.h`: that inverse revision.

The exact MLX source commit is `36969c094354086e15d2a327a9c50feb20b9cf11`. The two complete
header hashes and answer revision identities are pinned in [review.json](review.json).
The manifest also records inspection of the complete hash-matched answer bodies: neither
supplied an additional attribution designation, copyright notice or warranty-disclaimer
notice. It records creator display metadata and both revision comments, summarized in the
candidate notice as the author's own revision notes. The visible downstream MLX modifications
are described separately. This bounded observation covers those two revisions, not all linked
material or the complete website, and leaves the owner review open.
These mappings are bounded comparison/provenance observations, not a reconstructed copying
history or a legal test of expression. The record does not convert coefficient matches into
proof of copyright infringement, ownership of mathematical algorithms, or complete equivalence.

The proposed legal boundary is Juffa's covered expression retained in those three functions
and its corresponding compiled forms in the identified vendor `libmlx.dylib` and
`mlx.metallib`, under the accepted direct-CC/MIT interpretation. Apple/MLX contributions
remain covered by their existing MIT grants. Other MLX functions, unrelated portions of the
combined binaries, and jmlx Java code retain their applicable terms; aggregation alone is
not the proposed reason for applying CC to them. This is a proposed owner judgment about
scope, not a boundary proved by function names, an inventory title or separable binary offsets.

CPU `erf` has separate PyTorch provenance. The BSD `expm1f` implementation remains separately
attributed. Cephes remains under its separately recorded original-permission interpretation.
The packet does not replace those notices or create new grants for any contributor.

The existing binary observation establishes unchanged vendor bytes and records the mlx-c
wrapper/import and Java-only sources-JAR checks. It does not exclude every possible inlined
implementation or establish a reproducible source-to-wheel build. The proposed scope must be
reassessed for changed pins, functions, compiled material or future jmlx adaptations.

The creator line retains the recorded display name `njuffa` and a direct profile URL.
Stack Overflow's
[attribution guidance](https://stackoverflow.blog/2009/06/25/attribution-required/)
(2009; checked 2026-11-22) asks for author names linked directly to their profiles.
Its [current licensing help](https://stackoverflow.com/help/licensing) points readers to the
terms and applicable license. These references support retaining the additional identity
information; they do not decide the outstanding scope or compliance review.

## Exact notice and license text for approval

[juffa-notice-candidate.txt](juffa-notice-candidate.txt) is the exact proposed new section for
`jmlx-native-macos-arm64/THIRD-PARTY-NOTICES`. It begins with one LF before the literal
separator, then a blank line before the exact required title. It includes the creator ID,
answer revisions, modification indications, Apple notices, licensing basis and scoped
downstream statement. Its final LF is part of the
candidate bytes. Append these bytes directly after the existing notice's final LF without
changing the existing bytes. This section is intended to be last; it runs to EOF without a closing
separator. The initial LF preserves the blank line before separators used by all existing
sections. Existing file bytes stay unchanged; that LF becomes the final byte of the preceding
PyTorch hypot section under the raw-section extraction rule.

This replaces the earlier unresolved paragraph with MIT for Apple/MLX contributions plus
the section 2(a)(5)(A) offer for Juffa's expression. It retains (B)'s unresolved counterargument
and identifies no CC adapter's license. It does not present MIT as one or assert an MIT-only
grant for Juffa's expression. The proposed scope still needs the owner's explicit acceptance.

[native-NOTICE-candidate.txt](native-NOTICE-candidate.txt) supplies the full proposed native
`NOTICE` bytes, including the scoped CC reference and paths to retained texts.
[LICENSE-CC-BY-SA-4.0](LICENSE-CC-BY-SA-4.0) is the unmodified complete official plain text,
retrieved from the legal-code page's plain-text link. Its URL, retrieval date and raw-byte
hash are in review.json. Retain the existing native `LICENSE`, Apache text and every other
upstream section. `.gitattributes` preserves this packet's raw bytes and pins the future
native license to LF while allowing its genuine upstream blank line at EOF.
The notice's no-additional-restrictions statement is a proposed undertaking;
it is not evidence that the eventual implementation or every distribution surface complies.

## POM wording and the deliberately unresolved permalink

[pom-licenses-candidate.json](pom-licenses-candidate.json) specifies the exact names, URLs
and comment wording for MIT, Apache-2.0, CC BY-SA 4.0, BSD-2-Clause, BSD-3-Clause and zlib.
MIT names mlx-c and other recorded MIT components; Apache names metal-cpp, cexpf and
TensorFlow adaptations and retains the JAX reference notice. The BSD and zlib entries
cover the corresponding retained notice sections. This also addresses existing metadata
gaps as part of the same proposed metadata review. Cephes retains its separately
recorded original-permission interpretation; these entries do not relabel it with a
standard license. Component examples introduced by "including" are illustrative; the
retained notices identify the complete recorded set. The opt_einsum, VkFFT, JAX,
Random123 and CUB references/attributions are described as retained notices, without
asserting that their expression is present in the binaries.

During implementation, update `req/phase8-4-native-license-audit.md` alongside the POM: its
current MIT/Apache-only description must describe the approved six-entry metadata and its
scoped, illustrative component examples. The audit JSON does not repeat that description;
review it for consistency and record finding resolutions only after validation. The
compliance proposal lists both audit files in the implementation follow-up.

The CC comment contains exactly one `{{APPROVED_NOTICE_PERMALINK}}` token. This file is a
comment template, not a generated POM or the final literal comment. Approval here would cover
the wording and substitution rule only. After approved notice bytes are committed to the
native module, substitute their full commit permalink in a later implementation commit.
The linked file must be `jmlx-native-macos-arm64/THIRD-PARTY-NOTICES`, preserve the notice
commit through a merge commit, and enforce ancestry plus raw-section freshness as specified
in the compliance proposal. Record and review the literal resolved comment and its hash
before final POM approval. A comment cannot link to its own commit.

The packet's JSON template hash is not a final POM hash. The native POM and Gradle metadata
must be generated and reviewed during implementation, including POM-only consumption and
consistent scoped descriptions on every publication surface.

## Proposed README and release-note wording

Use the following paragraph in the native README and Phase 8.4 release notes after approval:

> The native distribution retains MLX's MIT grant for Apple/MLX contributions and separately identifies Norbert Juffa's covered expression in the reviewed Metal erf and Metal/CPU SIMD erfinv material under CC BY-SA 4.0. META-INF/THIRD-PARTY-NOTICES records the reviewed scope, revisions, modifications and redistribution basis; META-INF/LICENSE-CC-BY-SA-4.0 contains the complete CC license. No CC adapter's license is identified. Other bundled material retains its applicable licenses and notices. The owner-approved interpretation and its recorded counterargument are documented in the repository's owner license decision record.

Its exact UTF-8 paragraph, without Markdown blockquote prefixes and with one final LF,
is retained as [publication-paragraph.txt](publication-paragraph.txt), with its hash in the
manifest. It is one unwrapped line to avoid hard-wrapped release-note rendering.
The native README and release notes will link the reviewed decision record and
notice; their final file hashes will be recorded after implementation. This does not approve
a new classifier or provenance attachment; that proposal remains a separate review item.

## Downstream terms and remaining work

Approval is requested for the proposed boundary, exact notice/license bytes and publication
paragraph, and the POM template plus substitution rule. The acceptance record should cite
review.json and the candidate hashes, and record the SHA-256 of both README.md and
review.json as read from the approved commit. Record those two digests in the separate
acceptance record so the boundary prose and manifest are bound without self-hashing.
Approving the existing licensing interpretation alone has not approved this scope or wording.
Review or merge of this preparation PR does not silently supply those substantive decisions.

The proposed distribution retains the original grants and makes no additional restriction
or effective-technological-measure claim against their exercise. The implementation must
check actual native LICENSE/NOTICE/third-party texts, POM and module metadata together, plus
README/release-note descriptions and the final signed distribution. Integrity signatures
and an unencrypted archive observation alone would not settle the legal terms question.
The owner's reasoned downstream compatibility conclusion is still pending that review.

After scope/wording approval, implement the approved text and the proposal's authored-notice,
mandatory-text/title, POM and permalink checks. Commit notice bytes before the POM permalink,
then review literal metadata and record final hashes. Build and validate a new immutable
candidate with the required release and consumer checks. Both findings and the release gate
remain open; RC1 and its receipts are unchanged. No publication is authorized by this packet.
