# Phase 8.4 native license resolution review

Investigation date: 2026-11-16. Baseline: PR #100 merge
`ee807c99e89f06f63ef326e2bbb976b7cb247989`.

The next release step is an owner/legal decision on the two findings in the
[native audit](phase8-4-native-license-audit.md). This investigation found useful primary
sources, but did not establish an applicable additional grant for the shipped code.
Both findings and the license release gate remain open.

The owner submitted [MLX issue #4902](https://github.com/ml-explore/mlx/issues/4902) on
2026-11-16, covering both provenance questions. It is open with no comments at this check;
the next action is to review any upstream response against the exact code and license terms.
The [prepared request](plans/evidence/native-license-resolution-2026-11-16/upstream-request-draft.md)
is retained as a historical draft; the owner posted an adapted version. Native code, pins,
bindings, notices, artifact hashes and the publication flow are unchanged.

## Juffa implementations: the BSD samples differ

CPU `erf` is outside this Juffa comparison: its SIMD implementation cites PyTorch's
Abramowitz–Stegun approximation and is already covered by the packaged PyTorch notice.
The Juffa scope is Metal `erf` and Metal/CPU `erfinv`, rather than every function in `simd/math.h`.

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

Comparison method: extract text starting at the function name (excluding the return type),
through its balanced closing brace, from these particular inputs. Strip C/C++ comments,
collect decimal floating literals, remove `f`/`F` suffixes and normalize with Python's
`decimal.Decimal`. An optional plus/minus sign is included only when directly attached to
the numeric token, as specified by `FLOAT_LITERAL_PATTERN` in the committed script.
A separated minus sign is not treated as part of the literal. The pattern requires a decimal
point, so exponent-only decimal floating forms such as `1e-5f` are also excluded; none occurs
in the seven recorded function inputs. The comparison does not parse arbitrary C++ and
excludes integer/hexadecimal literals. Code input hashes and all seven extracted function
hashes are recorded, so readers can check the exact variants being compared. The two BSD
samples are reference evidence only and are not built or added to the native artifacts.

The committed [comparison and projection script](plans/evidence/native-license-resolution-2026-11-16/compare_literals.py)
reproduces all seven function records, five checks and both comment-list hashes. From the
repository root with Python 3.11+:

```sh
python3 req/plans/evidence/native-license-resolution-2026-11-16/compare_literals.py \
  --inputs build/releaseAll/license-resolution-inputs --fetch-missing \
  --output build/releaseAll/license-resolution-reproduction.json
```

Release selftests import this dated recipe. Moving or archiving the evidence folder requires
retaining the recipe or updating the test's `RECIPE` path and recorded evidence/verifier paths
in the same change.

The script downloads missing inputs into a URL/digest-bound cache; omit `--fetch-missing`
to verify the cached responses offline. It checks the original audit's complete Stack Overflow
answer-body hashes before extracting their longest HTML-decoded code blocks, and validates
every function input/hash/literal record against `code-comparison.json`. The retained BSD
samples need no download. Changed projected content fails with a contextual error.

For each comment response, preserve the API `items` array order. Project each item to exactly
`{"comment_id": item["comment_id"], "creation_date": item["creation_date"],
"body": item["body"], "ownerUserId": item.get("owner", {}).get("user_id")}`.
`body` remains the raw HTML string; the owner field is flattened to `ownerUserId`, and a
missing owner/user ID becomes JSON null. Serialize the complete list as
`json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")`,
with no trailing newline. Both original hashes are preserved. The manifest records this exact
format and the script's digest. Partial comment pages are rejected rather than silently hashed.

The review rerun reproduced both original comment hashes and all five checks from fresh
downloads, then produced an identical report offline. Five focused regressions cover exact
serialization/order, incomplete responses, function/sign boundaries, HTML/body extraction and
URL-bound input integrity. All 79 release selftests passed (49.296 seconds); the research
manifest retains the verifier/test digests and reproduction result. No native runtime code,
notices, license findings or numerical implementations changed.

The limits/error-handling follow-up documents the decimal-point requirement and adds a CLI
regression for missing research source IDs and unexpected evidence value types. These failures
now report `license comparison: …` without a traceback. Cached-input replay preserves all
seven function records, five checks and both original comment hashes; the legal findings stay open.
All 80 release selftests passed (50.066 seconds), including six focused recipe tests.

Validation checks source hashes against the existing audit, the five recorded comparison
results and every retained sample/report digest. The unchanged audit command still produces
the same result as PR #100's follow-up evidence, including `releaseReady: false` and both
finding IDs; `--require-clear` exits 1. No new runtime test or legal clearance is claimed.

The owner posted the provenance request as `perNyfelt` at `2026-11-16T16:54:53Z`.
The assistant did not post an issue or comment. The request asks for provenance rather than
asserting an upstream license violation. A subsequent response must be reviewed against the
actual code and bundle before either finding is removed from the release gate.
