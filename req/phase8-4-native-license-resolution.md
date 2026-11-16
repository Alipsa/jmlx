# Phase 8.4 native license resolution review

Investigation date: 2026-11-16. Baseline: PR #100 merge
`ee807c99e89f06f63ef326e2bbb976b7cb247989`.

The next release step is an owner/legal decision on the two findings in the
[native audit](phase8-4-native-license-audit.md). This investigation found useful primary
sources, but did not establish an applicable additional grant for the shipped code.
Both findings and the license release gate remain open.

The recommended next action is to ask MLX maintainers for the relevant provenance and
permission records, using the [prepared request](plans/evidence/native-license-resolution-2026-11-16/upstream-request-draft.md)
after owner review. No request has been posted or sent. Native code, pins, bindings, notices,
artifact hashes and the publication flow are unchanged.

## Juffa implementations: the BSD samples differ

Juffa published explicit BSD-2-Clause code in his NVIDIA forum posts for
[`erff`](https://forums.developer.nvidia.com/t/optimized-version-of-single-precision-error-function-erff/40977/1)
and [`erfinvf`](https://forums.developer.nvidia.com/t/a-very-compact-implementation-of-the-inverse-error-function-erfinvf/61961/1).
The captured code samples retain their complete copyright, conditions and disclaimer under
the [evidence directory](plans/evidence/native-license-resolution-2026-11-16/).
Only trailing whitespace is removed and a final LF is retained; the manifest records the
raw extracted-code hash separately from each normalized retained sample's hash.

| Material | Shipped MLX version | BSD forum sample | Result |
| --- | --- | --- | --- |
| Metal `erf` | Threshold `0.927734375`; first polynomial coefficient `-1.72853470e-5` | Threshold `1.0`; first coefficient `2.58907676e-5` | Different sample |
| Metal/CPU `erfinv` | Natural logarithm; threshold magnitude `6.125`; tail first coefficient `3.03697567e-10` | CUDA approximate base-two logarithm; threshold `-8.8359375`; first coefficient `1.50166858e-11` | Different sample and evaluation |

The [comparison record](plans/evidence/native-license-resolution-2026-11-16/code-comparison.json)
captures signed decimal floating literals from the named functions and identifies every input
by SHA-256. The Metal inverse function's literal sequence matches
[Stack Overflow answer 49743348](https://stackoverflow.com/a/49743348); the CPU inverse
function has the same literal multiset with its threshold placed later in the function.
The Metal error function's literals are a multiset subset of
[answer 35148199](https://stackoverflow.com/a/35148199), whose alternative exponential branch
adds another literal. These are bounded lexical observations, not proofs of equivalent
program behavior or of copyright/license scope. The existing audit's source attribution
and matching coefficients still support treating these as provenance questions.

Neither captured BSD sample has the shipped function's literal sequence. Their common author,
similar algorithms and grants allowing modifications do not themselves establish the rights
under which MLX incorporated these particular Stack Overflow-derived versions. The two specific
answer comment lists were also fetched completely; no licensing/permission keyword was found
in those comments. This does not establish the absence of a grant elsewhere.

The pinned file's [history](https://api.github.com/repos/ml-explore/mlx/commits?path=mlx/backend/metal/kernels/erf.h&per_page=100&sha=36969c094354086e15d2a327a9c50feb20b9cf11)
has four commits. The latest is the
[`expm1f` substitution](https://github.com/ml-explore/mlx/commit/634b148dd4203f80b8cc8db3e33452a91785fad8)
from [PR #3025](https://github.com/ml-explore/mlx/pull/3025). That PR describes a numerical
change and had no issue-thread comments when checked; it does not provide an additional
license grant. A public history entry named `copyright + ack` is also not a substitute
for the underlying permission record. The research manifest records this bounded history check.

## Cephes: original correspondence needs a scope decision

The original [2004 correspondence](https://lists.debian.org/debian-legal/2004/12/msg00295.html)
contains Moshier's response to a request about distributing Cephes in particular packages.
He distinguishes the terms of his own distribution from permission for another distribution
and supplies example BSD boilerplate. This is stronger context than a third-party notice
claiming BSD, but it does not explicitly identify MLX, fastermath, NEON math or a blanket
grant to every downstream recipient of the original Netlib code.

The [direct reply](https://lists.debian.org/debian-legal/2004/12/msg00297.html), by Brian
Thomas Sniffen rather than Moshier, discusses using excerpts actually distributed with that
boilerplate and porting modifications if needed. That is a contemporary interpretation,
not a fresh grant from the copyright holder. It does not establish the path by which the
Cephes-derived material reached the pinned MLX SIMD implementation.

The existing Netlib permission statement and immediate BSD/zlib notices remain packaged.
The owner/legal reviewer must decide whether those records already cover this distribution,
or obtain an applicable grant or a documented licensed provenance chain. This investigation
does not label the material prohibited, GPL-only or automatically BSD-licensed.

## Resolution choices

| Route | Concrete evidence needed before closing the finding |
| --- | --- |
| Ask upstream for an applicable grant/provenance record (recommended) | Identify exact functions/files, original copyright holder, granted terms and downstream scope; retain the response or immutable source record and package its required notices |
| Review the existing permissions | For Juffa, determine the applicable license/version, adaptation boundary and obligations for compiled MLX and Metal code; for Cephes, document why the original permission or a traced licensed distribution covers the actual derived material |
| Separately review a native implementation or source change | Establish a licensed replacement and validate numerical behavior, fixtures, both precision modes, kernel execution and the complete rebuilt artifact; repeat the audit and candidate verification |

The Juffa answers' recorded license is CC BY-SA 4.0. Its official
[legal code](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en) distinguishes
attribution, modification notices, the adapter's license for adapted material and downstream
restrictions (sections 2 and 3). A link to that license alone does not establish that all
conditions for this compiled distribution have been met. The software-specific integration
and adaptation boundary require review; this document does not infer that all of jmlx becomes
CC-licensed. Creative Commons' [software FAQ](https://creativecommons.org/faq/#can-i-apply-a-creative-commons-license-to-software)
recommends software licenses for code and explains some differences from CC terms. Its
recommendation does not itself cancel the existing license or supply an alternative grant.

The first route preserves the reviewed native tuple and avoids choosing new licensing terms
or changing numerical implementations during release preparation. If the requested records
cannot be established, the owner can choose a reviewed compliance or replacement route.

## Evidence and validation

The [research manifest](plans/evidence/native-license-resolution-2026-11-16/research.json)
records source URLs, content projections and hashes. It identifies live pages as mutable;
their hashes capture what was reviewed, not an assurance of unchanged future responses.
Only the two explicitly BSD-licensed samples and derived comparison facts are retained here;
Stack Overflow answer bodies and correspondence HTML are referenced rather than recopied.
The original audit retains the Stack Overflow body hashes and licenses.

Comparison method: extract the named function using balanced braces from these particular
inputs, strip C/C++ comments, collect signed decimal floating literals, remove `f`/`F` suffixes
and normalize with Python's `decimal.Decimal`. The comparison does not parse arbitrary C++
and excludes integer/hexadecimal literals. Code input hashes and all seven extracted function
hashes are recorded, so readers can check the exact variants being compared. The two BSD
samples are reference evidence only and are not built or added to the native artifacts.

Validation checks source hashes against the existing audit, the five recorded comparison
results and every retained sample/report digest. The unchanged audit command still produces
the same result as PR #100's follow-up evidence, including `releaseReady: false` and both
finding IDs; `--require-clear` exits 1. No new runtime test or legal clearance is claimed.

Posting the prepared request is a separate third-party communication and requires the owner's
explicit instruction. The draft asks for provenance rather than asserting an upstream license
violation. A subsequent response must be reviewed against the actual code and bundle before
either finding is removed from the release gate.
