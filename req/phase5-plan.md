# jmlx Phase 5 — Model Loading & HuggingFace Interop

## Status — update this section as work lands

**Branch:** `phase5-plan`, off `main` at `f28a327` (PR #11, Phase 4 M4 `QuantizedLinear`, merged).
Phase 4 (`req/phase4-plan.md`) has every milestone M0a–M4 plus §9 documentation **Done** — but not
"fully Done": §10 (CI, self-hosted runner) is still **Not started**, per that
document's own Status table. (M0d is also `Done` only in a scoped-down sense — six generic op-body
helpers deliberately deferred past their original merge point, per that table's own M0d note; this
document's Research findings section below leans on that same precedent for M1's own C-string
helper.)

| Item | Status | Commit |
|---|---|---|
| M1 — Checkpoint I/O: `MLXIO`, safetensors + GGUF (§1) | Plan written, implementation not started — see `req/plans/phase5-m1-plan.md` | — |
| M2 — Tokenizer integration (§2) | Not started — needs its own research spike first | — |
| M3 — Reference models: `LlamaModel`, `QwenModel` (§3) | Not started | — |

## Context

`req/project-outline.md` names Phase 5 as "Load pre-trained safetensors and GGUF model files into
Java memory," with three listed deliverables: a safetensors parser, tokenizer integration, and
`LlamaModel`/`QwenModel` reference implementations demonstrating streaming text generation. Unlike
Phase 3 and Phase 4, which each built on ops the previous phase had already exposed, Phase 5 is the
first phase whose deliverables split cleanly into one item mlx-c already solves natively (checkpoint
I/O), one item mlx-c has no opinion on at all (tokenization), and one item that's pure composition
of everything already built (the reference models). That split drives this document's shape more
than any single design tension does — see Decisions below.

## Decisions taken

**D1 — Checkpoint I/O is a thin FFM facade, not a parser (supersedes `project-outline.md`'s
wording).** `req/project-outline.md`'s Phase 5 section describes "a pure Java `.safetensors` parser
converting raw memory segments directly into `MLXArray` tensors." That undersells what's already
available: mlx-c's committed, generated bindings
(`jmlx-ffi/src/main/generated/java/se/alipsa/jmlx/ffi/mlx_h.java`) expose full native load/save
support for **both** formats already —

- Safetensors: `mlx_load_safetensors`/`mlx_load_safetensors_reader` (path or arbitrary
  `mlx_io_reader`), `mlx_save_safetensors`/`mlx_save_safetensors_writer` — returning/taking an
  `mlx_map_string_to_array` (tensors) plus an `mlx_map_string_to_string` (string metadata) directly;
  `mlx_save_safetensors(const char *file, const mlx_map_string_to_array param, const
  mlx_map_string_to_string metadata)` needs no intermediate builder.
- GGUF: `mlx_load_gguf(mlx_io_gguf *gguf, const char *file, const mlx_stream s)` produces an opaque
  `mlx_io_gguf` handle read via `mlx_io_gguf_get_keys`/`get_array`/`get_metadata_array` (numeric --
  unified as `MLXArray`, consistent with mlx's own "everything is an array" convention)/
  `get_metadata_string`/`get_metadata_vector_string`. The write path is not symmetric with the read
  path: `mlx_save_gguf(const char *file, mlx_io_gguf gguf)` takes that same opaque handle, not raw
  tensor/metadata maps -- a caller must first build one via `mlx_io_gguf_new()`, then populate it
  entry-by-entry with `mlx_io_gguf_set_array`/`set_metadata_array`/`set_metadata_string`/
  `set_metadata_vector_string` (one call per tensor or metadata key), then free it with
  `mlx_io_gguf_free` after the save call, whether it succeeds or throws. `MLXIO.saveGguf`'s
  implementation is this builder loop, not a single pass-through call the way `saveSafetensors` is.
- Both load entry points take a trailing `const mlx_stream s`; neither save entry point does.
  `MLXIO`'s load methods pass `NativeOps.DEFAULT_STREAM` -- the field `MLXQuant`'s own hand-rolled
  bodies already reuse rather than re-resolving the stream per call, and the shape to match here
  since `MLXIO` calls `mlx_h` directly the same way. (`MLXArray` is the one class that calls
  `MLX.defaultStream()` directly, at its own three call sites -- not the facade-wide convention;
  `CLAUDE.md`'s Architecture section documents `defaultDevice()`/`defaultStream()` as resolved once
  and cached for the process lifetime.) No stream argument to plumb through on the save side.

So M1 is a facade over already-battle-tested native code — the same "native does the heavy lifting,
Java exposes it safely" shape every prior phase has taken (jextract bindings, `MLXOps`/`MLXShape`
wrapping mlx-c ops, `MLXGrad` wrapping `mlx_value_and_grad`). Writing a hand-rolled binary parser
would duplicate code this project already links against for no benefit. This reconciliation follows
the same pattern `req/project-outline.md`'s own Phase 3/4 sections already use to correct their own
stale wording in place rather than pretend the original phrasing was right.

**D2 — The new facade lives in `se.alipsa.jmlx.core` as `MLXIO`, not a new package or module.**
`MLXArray`'s only constructor, `MLXArray(MLXScope scope, MemorySegment handle)`, is
package-private, and `NativeOps` (`checked`, `scopeOf`, `cstr`) is a package-private `final` class —
both scoped to `se.alipsa.jmlx.core`. A class outside that package could not wrap a raw native
`mlx_array` handle into an `MLXArray` at all without opening new cross-package access on a type the
codebase has deliberately kept closed (`MLXArray`/`MLXScope` are both `public final`, no
subclassing). So `MLXIO` joins `MLXQuant`/`MLXRandom`/`MLXGrad` as another native-facing facade
class inside `core`, not a new `se.alipsa.jmlx.io` package — see §1 below for the exact API shape,
and `req/plans/phase5-m1-plan.md` for the full task-by-task implementation.

**D2a — `MLXIO` is the first facade in this codebase whose native surface returns handles that are not
`mlx_array` at all, and `MLXScope` cannot own any of them.** `mlx_io_gguf`, `mlx_map_string_to_array`,
`mlx_map_string_to_string`, `mlx_vector_string`, and `mlx_string` each come back from these load
calls with their own deallocator (`mlx_io_gguf_free`, `mlx_map_string_to_array_free`,
`mlx_map_string_to_string_free`, `mlx_vector_string_free`, `mlx_string_free`, plus the two
`*_iterator_free` variants for the maps) -- never `mlx_array_free`. `MLXScope.allocate`'s own javadoc
is explicit that it "assumes every segment handed out represents an `mlx_array` struct and will
later be passed to `mlx_array_free`. Passing a scope as an allocator for anything else corrupts the
handle list." So these five types must be allocated and freed through a confined `Arena`/direct
native calls, exactly like `MLXQuant`/`MLXGrad`'s own hand-rolled bodies already do for by-value
structs -- never through an `MLXScope`. The *individual* `mlx_array` tensor handles extracted out of
a loaded `mlx_map_string_to_array` are the opposite case: those genuinely are `mlx_array`s and do
need `MLXScope` registration, but there is no existing `MLXArray` operand to infer a target scope
from the way `scopeOf` does elsewhere in this codebase -- so `loadSafetensors`/`loadGguf` must each
take an explicit `MLXScope` parameter, the same shape `MLX.array(scope, ...)` already uses for
exactly this reason (`CLAUDE.md`'s memory-model rule). This is the most likely source of a real leak
or handle-list corruption in M1 if missed, and belongs in `req/plans/phase5-m1-plan.md`'s own Global
Constraints once that document exists.

**D2b — `mlx_io_gguf_new` is the specific call site this hazard becomes checkable at.** It binds as
`MemorySegment mlx_io_gguf_new(SegmentAllocator allocator)` -- jextract's generated signature takes
any `SegmentAllocator`, and `MLXScope` implements that interface (`CLAUDE.md`'s memory model:
"`MLXScope` ... implements `SegmentAllocator`"), so `saveGguf` passing a scope here compiles cleanly
and silently corrupts that scope's handle list -- exactly the failure mode `MLXScope.allocate`'s
javadoc warns about, with no compiler error to catch it. `MLX.newVectorArray` already has the
precedent to mirror: it declares its own allocator parameter as a plain `Arena`, not the broader
`SegmentAllocator` jextract would otherwise accept, specifically so this mistake is a compile error
rather than a runtime one -- "`allocator` MUST be a confined `Arena` (never an `MLXScope`)". Whatever
helper `saveGguf` builds its `mlx_io_gguf` through should take the same narrowed `Arena` parameter
type, not `SegmentAllocator`.

**D2c — Every one of these five types' free functions -- and both `*_iterator_free` variants --
returns a status `int`, same as `mlx_array_free(mlx_array arr)` itself** (the genuinely statusless
entry points are the `mlx_array` constructors, `mlx_array_new`/`mlx_array_new_data`, which "have no
status return at all and only signal failure via the error handler plus a null `ctx`" -- `CLAUDE.md`'s
error-handling paragraph, not its Memory model paragraph, notes this). Routing one of these seven
frees through `NativeOps.checked` from inside a `finally` block risks losing a real in-flight failure
during exception unwinding: `checked` throws on a non-zero status, and a fresh exception thrown from
a `finally` replaces whatever exception was already propagating instead of chaining it.

This codebase already has a precedent to follow rather than decide fresh: `MLXScope`'s own teardown
(`Holder.closeAll`'s reverse-order loop, and `freeOne`) calls `mlx_h.mlx_array_free` bare, outside
`NativeOps.checked`, discarding its status -- and does so unconditionally, not only when unwinding an
exception: neither call site sits inside a `try`/`catch` reacting to one, so the same discard applies
on an ordinary clean `close()` too. That covers more than the unwinding hazard above explains: a
cleanup failure on an otherwise-successful exit vanishes silently as well, not just one that would
otherwise mask a real error -- consistent with `MLXScope` treating a failed free as unactionable
information either way, not merely as the lesser of two evils during unwinding. Whether that discard
was a deliberate policy or merely incidental is not established by the code alone, so this is cited as
precedent to follow, not as a documented decision -- but absent a reason to diverge, `MLXIO`'s own
cleanup should swallow these seven frees' statuses the same way, on every exit path, not attach them
via `Throwable.addSuppressed`.

**D3 — Tokenizer integration (M2) is deliberately left unplanned here.** Unlike checkpoint I/O,
there is no existing binding to lean on: mlx-c has no tokenizer API at all. Whether the right move
is FFM-binding Hugging Face's `tokenizers` Rust crate's C API (mirroring how `mlx-c` itself was
bootstrapped from a pinned wheel — see `req/initial-plan.md`) or a pure-Java BPE/SentencePiece
implementation is a genuinely open architectural question this document does not resolve. M2 needs
its own research spike (probing whether `tokenizers` exposes a usable C ABI, checking license
compatibility, and prototyping load-time cost) before a plan for it can be written the way M1's plan
already can be. Do not start M2 implementation from this document — write `req/plans/phase5-m2-plan.md`
only after that spike.

**D4 — Reference models (M3) are pure composition, deferred until M1 and M2 both land.**
`LlamaModel`/`QwenModel` need nothing new at the tensor/module level: `se.alipsa.jmlx.nn` already
has `Linear`, `QuantizedLinear`, `RMSNorm`, `MultiHeadAttention`, `KVCache`, and RoPE (via
`MLXFast.rope`, per `req/project-outline.md`'s own Phase 4 reconciliation note, not
`req/phase4-plan.md` §7 -- §7 is only RoPE's signature spec, the reconciliation of RoPE shipping as
a `core` op rather than an `nn` module lives in the outline). What M3 adds is a checkpoint →
`Module` tree mapping (naming convention translation from HF/GGUF tensor names to this project's
`Module` field names) and an autoregressive generation loop built on `KVCache`. Both depend on M1's
`MLXIO` output shape and M2's tokenizer, so M3 cannot be scoped precisely until those land — this
document names it only to keep the phase's end state visible, not to plan its tasks.

## Research findings

**GGUF metadata is not uniformly stringly-typed, unlike safetensors.** Safetensors' native binding
returns exactly two maps: tensors (`mlx_map_string_to_array`) and metadata
(`mlx_map_string_to_string` — always strings, matching the safetensors spec's own JSON-header
design). GGUF's `mlx_io_gguf` instead exposes three separate metadata accessors —
`get_metadata_array` (numeric scalars/arrays, returned as `MLXArray`), `get_metadata_string`, and
`get_metadata_vector_string` (string arrays, e.g. tokenizer vocab-adjacent fields) — because GGUF's
own on-disk metadata format is a small typed-value union, not JSON. `MLXIO`'s GGUF result type must
mirror that three-way split rather than force everything through one `Map<String, String>` the way
safetensors' own metadata naturally fits.

**No C-string-to-`Java String` helper exists anywhere in `jmlx-core` today** (confirmed by
repository-wide grep for `getString`/`mlx_string`/`mlx_map_string`). `NativeOps.cstr` only goes
Java → C. Reading back C strings needs two distinct lifetimes, not one "mirror image of `cstr`"
helper:

- Borrowed `const char*`, needing only a copy into a Java `String`, no free call: both
  `mlx_map_string_to_array_iterator_next(const char **key, mlx_array *value, ...)`'s `key` (borrowed
  from the map's own storage -- `mlx_string_free` does not apply, and freeing it risks a double-free
  once the map itself is freed) and, identically, `mlx_map_string_to_string`'s own pair
  (`mlx_map_string_to_string_get(const char **value, ...)` and
  `mlx_map_string_to_string_iterator_next(const char **key, const char **value, ...)`, both
  `safetensors`' metadata path depends on) -- as well as `mlx_string_data(mlx_string str)`, whose own
  header comment states outright: "Returns a pointer to the string contents. The pointer is valid
  for the life duration of the string."
- Owned `mlx_string`, needing a copy followed by `mlx_string_free`: only
  `mlx_io_gguf_get_metadata_string(mlx_string *str, mlx_io_gguf io, const char *key)`'s out-param
  fits this shape among the accessors M1 needs.

**`mlx_vector_string_get(char **res, const mlx_vector_string vec, size_t idx)` is resolved as
borrowed, same bucket as the three accessors above -- confirmed from mlx-c's actual implementation,
not just its header.** The header carries no ownership comment and the non-`const` `char **` return
could plausibly have hinted at caller-owned, so this was flagged as a genuine blocker on `loadGguf`
(GGUF's sole key-enumeration path: `mlx_io_gguf_get_keys` returns an `mlx_vector_string`, and every
key must be read back out of it through this call -- there is no other route). Reading
`native/scratch/mlx-c/mlx/c/vector.cpp` (the pinned checkout at `fba4470b89073180056c9ea46c443051375f7399`,
matching `bootstrap-native.sh`'s own `MLX_C_COMMIT`) settles it directly: `*res =
mlx_vector_string_get_(vec).at(index).data();` -- a pointer into the vector's own internal
`std::string` storage, not a fresh allocation. No free call applies, and the same C-string-copy
helper built for the other three borrowed accessors covers this one too. This was resolved by
reading the real `.cpp` source rather than by writing a runtime probe -- stronger evidence than a
probe result, since it is the actual logic that will run, not an inferred behavior from one sample
call.

**GGUF/safetensors map iteration has a three-way status convention, not the usual 0-or-throw
`NativeOps.checked` shape.** `mlx_map_string_to_array_iterator_next`'s own implementation
(`native/scratch/mlx-c/mlx/c/map.cpp`) returns `0` on a successful read, `1` on a genuine error (goes
through `mlx_error`, same as everywhere else), and `2` when the iterator has already reached the
map's `end()` -- a normal loop-termination signal, not a failure. Routing this call through
`NativeOps.checked` unmodified would misread every ordinary end-of-map as an `MLXException`; the
helper that drives `loadSafetensors`'s tensor/metadata read loop needs its own three-way switch
instead of reusing `checked` as-is. Not yet confirmed whether `mlx_map_string_to_string_iterator_next`
shares the identical `0`/`1`/`2` convention -- likely, given the parallel design, but `req/plans/phase5-m1-plan.md`
should verify against its own `map.cpp` body before assuming it rather than citing this paragraph as
proof for both map types.

Per the M0d precedent in `req/phase4-plan.md` (add a generic helper only in the task that first needs
it, wired to a real consumer, never speculatively), the resulting helper(s) are added to `NativeOps`
in M1's own first task, not built ahead of time here.

## Out of scope for Phase 5 as currently planned

- Any tokenizer implementation or FFM binding (M2 is a research spike, not an implementation, until
  its own plan exists).
- `LlamaModel`/`QwenModel` and any other reference model (M3, deferred — see D4).
- Streaming/incremental text generation UX (console printing, sampling strategies beyond greedy
  argmax) — not named in `project-outline.md`'s Phase 5 deliverables and not needed to prove
  checkpoint loading works.
- Writing (`save`) support beyond exactly what M1's own round-trip tests need as a
  fixture-generation path. This is not conditional on whether the testing plan happens to exercise
  it: Testing approach (below) already commits to round-trip as the strategy for *both* formats, so
  `MLXIO.saveSafetensors` and `MLXIO.saveGguf` are both in scope for M1 -- see §1 above and D1's
  write-path note for `saveGguf`'s extra plumbing. What stays out of scope is any save
  capability beyond that fixture-generation need — this project has no fine-tuning/checkpoint-export
  use case yet to justify more (e.g. preserving a source checkpoint's exact tensor ordering or
  metadata round-trip fidelity beyond what the M1 tests themselves assert).

## Work breakdown

### 1. Checkpoint I/O — `MLXIO`, safetensors + GGUF (M1) — **PLAN WRITTEN, IMPLEMENTATION NOT STARTED**

Full task-by-task plan lives in `req/plans/phase5-m1-plan.md`. Summary: a new
`se.alipsa.jmlx.core.MLXIO` facade class (package-private-constructor constraints rule out a new
package — see D2) exposing `loadSafetensors`/`saveSafetensors`/`loadGguf`/`saveGguf`, following two
distinct precedents for two distinct properties rather than one class as a blanket model: `MLXGrad`'s
own `NativeLoader.ensureLoaded()` guard, since `MLXIO` calls `mlx_h` directly and cannot rely on
`NativeOps`'s transitive guard the way `MLXQuant` does (`MLXQuant` has no guard of its own precisely
because every entry point still funnels through `NativeOps.checked`/`scopeOf`/`cstr` first;
`MLXIO`'s hand-rolled bodies must not assume the same); and `MLXQuant`'s hand-rolled-body shape and
cached-stream reuse (D1 above) for everything else -- confined `Arena` for transient structs freed on
every exit path via `try/finally`, swallowing each free's own status per D2c above rather than
routing it through `NativeOps.checked`. `loadSafetensors`/`loadGguf` each take an explicit `MLXScope`
parameter (D2a above); `saveGguf` additionally builds and frees its own `mlx_io_gguf` via a confined
`Arena`-scoped helper, never an `MLXScope` (D2b above). Update CLAUDE.md's "Loading order matters"
paragraph and architecture-diagram class list to name `MLXIO` as the fifth native-loading-guard
class alongside `MLX`/`MLXScope`/`NativeOps`/`MLXGrad`.

### 2. Tokenizer integration (M2) — **NOT STARTED — blocked on a research spike, see D3**

Not planned in detail here. First step is the spike named in D3, written up as its own findings
section before `req/plans/phase5-m2-plan.md` exists.

### 3. Reference models — `LlamaModel`, `QwenModel` (M3) — **NOT STARTED — blocked on M1 and M2**

Not planned in detail here — see D4.

## Testing approach

Same conventions `req/plans/phase4-m1-plan.md`'s Global Constraint 7 already establishes project-wide:
`@EnabledIfNativeAvailable` on native-dependent test classes, `try (MLXScope scope = new MLXScope())`,
hand-computed goldens, element-value assertions rather than shape-only assertions. Round-trip alone
needs no external binary fixture files: `mlx_save_safetensors`/`mlx_save_gguf` exist precisely so
tests can round-trip (build small `MLXArray`s, save, reload in a fresh scope, assert tensor values
and metadata match) for both formats — the concrete test list is `req/plans/phase5-m1-plan.md`'s
Task 4.

Round-trip proves `MLXIO` is self-consistent (jmlx writes, jmlx reads back what it wrote); it does
not exercise the real-world variation Context's own quoted objective ("Load pre-trained safetensors
and GGUF model files") requires -- HF's vs. llama.cpp's differing metadata conventions, real
quantized tensor dtypes, or the tensor-naming schemes M3's checkpoint → `Module` mapping will
eventually depend on. `req/plans/phase5-m1-plan.md`'s Task 4 resolves this by choosing the second of
the two ways this document originally posed to close that gap: real-world validation is deferred to
M3 (where a real checkpoint is unavoidable anyway), not a checked-in fixture fragment added here --
M1's round-trip tests prove only the facade layer, not interop, and that is stated as a deliberate,
named scope boundary rather than left implicit.

## Open questions

- M2's core question (FFM-bind HF `tokenizers`'s C API vs. a pure-Java implementation) is
  unresolved — see D3.

No open question remains on the checkpoint-I/O (M1) side: `mlx_vector_string_get`'s ownership, the
last unresolved item blocking `loadGguf`'s design, is settled — see Research findings above.
