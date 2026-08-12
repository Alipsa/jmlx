# jmlx Phase 4 — `se.alipsa.jmlx.nn` Neural Network Modules

## Context

`req/project-outline.md:83-91` describes Phase 4 in four bullets: a `Module` base class, core layers
(`Linear`, `Embedding`, `RMSNorm`, `LayerNorm`, `RoPE`, `SiLU`, `GELU`), `QuantizedLinear`, and
`MultiHeadAttention` with KV caching. Phase 3 is delivered (`51fd5e5`); `MLX.java` is 607 lines with
20 public ops.

**`req/phase3-plan.md` addressed Java guards stricter than native. The risk this document addresses
is that the *memory model* — not the op surface — does not survive a transformer forward pass.** Two
shipped invariants break on the outline's own `JMLXDemo` sample (`project-outline.md:108-138`):

1. **`binaryOp` rejects operands from different scopes** (`MLX.java:247`). `Linear.forward(x)` is
   `matmul(x, W)` with `x` in a per-step scope and `W` in the model scope. That throws today.
2. **Every op result is registered with its scope for the scope's entire lifetime.** Phase 3 measured
   this and recorded it (`phase3-plan.md:652-660`): "everything this facade builds is retained until
   scope close regardless of `eval` strategy, by construction, at any chain depth." A 1000-token
   decode loop over 30 layers retains ~300k arrays. **This is not a leak — it is the design working
   as specified, and the specification is wrong for Phase 4.**

Both are public contracts. Changing them before layers ship is free; changing them after is breaking.
That, not "how to bind `mlx_fast_rope`", is what this document is for.

**A third constraint decides package layout before any layer is written.** `MLXArray`'s constructor,
`scope()` and `handle()` are package-private to `se.alipsa.jmlx.core` (`MLXArray.java:21`, `:26`,
`:31`). `se.alipsa.jmlx.nn` therefore cannot construct an array from a native result or read a
handle — it must be built *purely* on the public op surface, which is roughly 30 ops short. The
autograd upcall, which must build `MLXArray`s from raw `mlx_array` handles arriving from C++,
consequently **cannot live in `nn` at all**.

**Phase 4 needs no bindings regeneration.** All 618 mlx-c symbols are already bound, for the reason
given in `initial-plan.md` §5 — `regen-bindings.sh` derives its include list by grepping the
originating header path, so every mlx-c symbol is bound by construction. Two symbols a reader might
expect are absent because **they do not exist in this mlx-c at all**, so no regen would produce them:

* **`mlx_grad` does not exist.** `native/install/include/mlx/c/transforms.h` declares exactly seven
  functions; there is no gradient-only transform. Use `mlx_value_and_grad` and discard the value,
  which is what upstream Python does internally.
* **`mlx_fast_affine_quantize` does not exist.** Its role is filled by `mlx_quantize` with
  `mode="affine"`.

**Intended outcome:** a `Module`/layer surface whose memory ownership is expressible for a real
decode loop, with autograd reachable, built on an op surface expanded once rather than renegotiated
four times.

## Decisions taken

1. **`MLXScope` gains a parent link; op results allocate into the *innermost* scope among the
   operands**, requiring an ancestor relation. Unrelated scopes — siblings, two independent roots —
   are still rejected. See §2.
2. **`hoist`/`keep` are a copy, not a move** — `mlx_array_set` onto a fresh handle in the target
   scope. Simpler than free-list surgery inside `MLXScope`, and safe for the same reason cross-scope
   results are safe. See §2.
3. **Autograd is in scope**, as `se.alipsa.jmlx.core.MLXGrad` — *not* `nn` — built on
   `mlx_value_and_grad` over `mlx_closure`. The package is forced by `MLXArray`'s constructor
   visibility, not chosen. See §6.
4. **The `MLX` facade splits first, as pure motion**, before a single new op is added.
   `phase3-plan.md:850-852` pre-named the trigger; it is met. `MLX` does **not** remain a delegating
   facade — callers move. See §1.
5. **`Module` is an abstract class with explicit `param(name, value)` registration**, not records and
   not reflection. `Class.getDeclaredFields()` order is explicitly unspecified by the JLS, and an
   unspecified parameter order is a silent-wrong-gradient bug — the same severity class as
   `toFloatArray()`'s contiguity bug, not a style preference. See §5.
6. **`toFloatArray()` inserts an `astype(FLOAT32)`** for any inexact non-float32 dtype, rather than
   growing per-dtype readers. This is what keeps float16/bfloat16 working without touching the
   `half.h` override. See §4.
7. **`requireMatmulCompatible`'s dtype guard becomes "reject only when *both* operands are exact"**,
   in the same commit that adds `astype`. This discharges `phase3-plan.md:857-862`. See §4.
8. **Test goldens stay hand-computed** (Phase 3 Decision 5). Fixtures are sized so the arithmetic
   stays tractable; where a direct golden is impractical, assert a composition identity instead. See
   Testing approach.

## Research findings that shaped this plan

Read from mlx-c at `native/scratch/mlx-c` (`fba4470`), the staged headers under
`native/install/include/mlx/c/`, the wheel's C++ headers under `native/scratch/wheel/mlx/include/`,
and the committed jextract output. **Every citation below is verifiable from this repository** — the
wheel ships C++ *headers*, so unlike `phase3-plan.md` there is no need to pin upstream permalinks for
the load-bearing facts.

**Hoisting an array out of a scope that is about to close is safe, and this is provable rather than
plausible.** The fear is that freeing a child scope's handles would break a lazy graph whose
intermediates lived there. It does not:

* `native/scratch/wheel/mlx/include/mlx/array.h:206`, `:523` — an `ArrayDesc` owns its graph inputs
  **by value** (`std::vector<array> inputs`), and each `array` is a `std::shared_ptr<ArrayDesc>`. An
  output therefore holds a refcount on every input's descriptor.
* `native/scratch/mlx-c/mlx/c/private/array.h:49-53` — `mlx_array_free_` is
  `delete static_cast<mlx::core::array*>(d.ctx)`. Freeing a handle *decrements a refcount*; it cannot
  touch the graph.

Closing a child drops refcounts on descriptors the hoisted output still references. The graph
survives and a later `eval` recomputes correctly.

**The consequence must be stated explicitly, or the guard will be deleted for the wrong reason.**
The ancestor requirement in Decision 1 is an **ownership-comprehensibility invariant, not a
memory-safety one** — mlx's own refcounting already makes cross-scope results safe. A later reader
who discovers this and removes the check would be right by accident and would lose the only thing
that answers "which scope owns this output".

**`mlx_array_set` onto a null-`ctx` handle is the copy primitive `hoist` needs.**
`private/array.h:24-31`:

```cpp
inline mlx_array& mlx_array_set_(mlx_array& d, const mlx::core::array& s) {
  if (d.ctx) {
    *static_cast<mlx::core::array*>(d.ctx) = s;
  } else {
    d.ctx = new mlx::core::array(s);      // second wrapper, same ArrayDesc
  }
  return d;
}
```

So `hoist` yields an independent handle in the target scope sharing the source's descriptor. The
original stays owned by the child and dies with it. **Copy, not move** — the javadoc must say that
after the child closes, callers use the returned array and never the original.

**A non-zero return from a Java upcall is leak-free and surfaces as a normal `MLXException`.**
`native/scratch/mlx-c/mlx/c/closure.cpp:44-51`, quoted in full because the whole exception-safety
protocol in §6 rests on it:

```cpp
auto cpp_closure = [fun](const std::vector<mlx::core::array>& cpp_input) {
  auto input = mlx_vector_array_new_();
  mlx_vector_array_set_(input, cpp_input);
  auto res = mlx_vector_array_new_();
  auto status = fun(&res, input);          // <- the Java upcall
  mlx_vector_array_free(input);
  if (status) {
    mlx_vector_array_free(res);            // both vectors freed
    throw std::runtime_error("mlx_closure returned a non-zero value");
  }
  ...
```

Two facts follow. Returning `1` is a supported, non-leaking error path — which is what makes the
catch-and-return protocol in §6 viable rather than merely defensive. And the upcall must fill `res`
with **`mlx_vector_array_set_data`, not `mlx_vector_array_new_data`** — for a reason worth stating
precisely, because the obvious reason is false:

* **`res.ctx` is null**, not populated. `mlx_vector_array_new_()` is
  `return mlx_vector_array({nullptr});` (`private/vector.h:12-14`). An earlier draft of this document
  justified the choice by claiming a non-null ctx; a reader who checks would find null and "fix" the
  call back to `new_data`. That is `phase3-plan.md:826-830`'s failure mode #2 — a right conclusion
  carried by wrong evidence — so the wrong reason is recorded here rather than silently replaced.
* **The actual reason is arity.** `mlx_vector_array_new_data(const mlx_array* data, size_t size)`
  returns an `mlx_vector_array` **by value** (`native/install/include/mlx/c/vector.h:31`); it cannot
  write through `res` and would leave `*res` untouched. `mlx_vector_array_set_data` takes
  `mlx_vector_array* vec` (`vector.h:33`) and writes through it. Null ctx is handled: `set_` is the
  same `if (d.ctx) … else d.ctx = new …` shape as `mlx_array_set_` (`private/vector.h:26-36`).

**A `Throwable` escaping an FFM upcall stub terminates the JVM abruptly.** It is not catchable,
produces no `MLXException` and no Java stack trace. **It is reachable from ordinary user error**: a
shape mismatch inside a loss closure throws `IllegalArgumentException` from
`requireBroadcastCompatible` (`MLX.java:146`). This is the highest-severity hazard in Phase 4 and the
only one with no compile-time signal — `phase3-plan.md:838-843`'s failure mode #3 ("Java written but
never compiled or run") applies here at its worst, because even compiling is not sufficient.

**bfloat16 cannot be read element-wise from Java, and must not be "fixed".** Measured against the
committed bindings:

```
$ grep -c 'findOrThrow("mlx_array_data_bfloat16")' .../mlx_h.java    -> 0
$ grep -c 'findOrThrow("mlx_array_data_float16")'  .../mlx_h.java    -> 1
$ grep -c 'MLX_BFLOAT16'                            .../mlx_h.java    -> 4
```

The dtype *constant* is bound; the data accessor is not, because
`scripts/jextract-overrides/mlx/c/half.h` drops the `HAS_BFLOAT16` block. Per `initial-plan.md`
Decision 10 the `__bf16` typedef is a **whole-run fatal parse error producing zero output**, not a
per-symbol skip — so "just re-add it" is not a small change, it is a change that breaks the entire
bindings pipeline. Decision 6 makes it unnecessary.

**Nullable native parameters are by-value structs, not null pointers.** `mlx_fast_rms_norm`'s
`weight`, `mlx_fast_layer_norm`'s `weight` **and** `bias`, `mlx_fast_rope`'s `freqs`, SDPA's
`mask_arr` **and** `sinks`, `mlx_quantized_matmul`'s `biases`, and every `mlx_random_*` `key` are all
`/* may be null */` (`fast.h:93-99`, `:163-178`, `:189-198`; `random.h:36,43,51,58,64,71`;
`ops.h:809-819`). The parameter is passed **by value**, so the callee reads 8 bytes at the passed
address: the correct "null" is a zero-filled `mlx_array_` struct, **not** `MemorySegment.NULL`.
Passing `MemorySegment.NULL` is a segfault, not an exception.

That every RNG `key` is nullable is load-bearing for M1: weight initialization needs only a global
`mlx_random_seed`, with no Java-side key-splitting design.

**Three parameter shapes in `fast.h`/`ops.h` have no precedent anywhere in the existing facade**, and
all three appear in the same calls:

| Shape | Where | Handling |
| --- | --- | --- |
| `mlx_optional_int` / `_float` / `_dtype` **by value** | `quantize`/`dequantize`/`quantized_matmul` (`group_size`, `bits`, `dtype`), `rope` (`base`) | 8-byte struct `{value; has_value; pad(3)}` (`mlx_optional_int_.java:29-33`). **No constructor function exists** — allocate and set both fields |
| raw `const char*` | SDPA's `mask_mode` (`""`/`"causal"`/`"array"`), quantization's `mode` (`"affine"`) | **Not `mlx_string`**, which is bound and looks like the natural choice. The value sets are closed, so allocate once in the existing `FACADE_ARENA` (`MLX.java:42`), not per call |
| `mlx_vector_array*` **as the result** | `mlx_split`, `mlx_quantize` (returns `(w_q, scales, biases)`) | See `vectorOutOp` in §3 — the two-allocator hazard |

**`mlx_quantize`'s `w_q` output is `UINT32`-packed** (`ops.h:801-808`), which is why `DType` must
grow `UINT32` in §4 even though nothing will ever read it element-wise.

## Out of scope for Phase 4

* Safetensors/GGUF loading, tokenizers, `LlamaModel`/`QwenModel` — Phase 5. `Module.update(Map)` is
  the hook left for it.
* Optimizers beyond whatever one SGD step needs to prove `MLXGrad` end to end.
* Convolutions, dropout, batch norm, `Sequential` sugar. Add when a caller exists.
* **bfloat16 element-wise read-back from Java.** Blocked on the `half.h` override; Decision 6 removes
  the need. **Do not "fix" the override** — see Research findings.
* Device/stream selection. `defaultStream()` stays process-cached as in v0.1.
* Resolving `mlx_array_free` from the `Cleaner` thread. See Open questions — Phase 4 makes it worse,
  but does not change the mechanism.
* Splitting `nn` across sub-packages. One flat `se.alipsa.jmlx.nn` until it hurts.

## Work breakdown

Merge points are named **M0a**…**M4**. Each is independently green; M0a–M0d are one PR's worth of
infrastructure that adds no public layer at all.

### 1. Facade split — pure motion (M0a)

`MLX.java` is 607 lines; roughly 30 new ops plus six new helpers put it past 1,300 at this project's
javadoc density. `phase3-plan.md:850-852` named the split trigger in advance "so it happens
deliberately rather than at 700 lines under duress". It is met.

| Class (all `se.alipsa.jmlx.core`) | Contents |
| --- | --- |
| `MLX` | creation (`array`, `zeros`, `ones`, `full`, `arange`), `astype`, `eval`, `hoist`/`keep`, `defaultDevice`/`defaultStream`. Its class javadoc becomes the index pointing at the siblings |
| `MLXOps` | elementwise unary/binary, comparisons, `where`, reductions with axes+keepdims, `softmax`, `matmul`, `addmm`, `inner`, `outer` |
| `MLXShape` | `reshape`, `broadcastTo`, `squeeze`, `transpose`, `swapaxes`, `expandDims`, `flatten`, `slice`, `concatenate`, `split`, `stack`, `take`, `tril`/`triu` |
| `MLXFast` | `rmsNorm`, `layerNorm`, `rope`, `scaledDotProductAttention` — exactly the `fast.h` family, i.e. precisely the group carrying nullable-array + optional-struct + `const char*` params |
| `MLXQuant` | `quantize`, `dequantize`, `quantizedMatmul` |
| `MLXRandom` | `seed`, `normal`, `uniform`, `key`/`split` if a caller ever needs explicit keys. Weight init (§5); every native `key` param is nullable, so the seeded-global form needs no key plumbing |
| `MLXGrad` | §6 |
| `NativeOps` *(package-private)* | `DEFAULT_STREAM`, `checked`, `nativeFailure`, `scopeOf`, and every private op helper (`binaryOp`, `unaryOp`, `shapeOp` and the six added in §3) |

**`MLX` must not remain a delegating facade.** Delegation would double the javadoc surface — and in
this project **the javadoc is the evidence base**, as `requireMatmulCompatible`'s 18-line javadoc
(`MLX.java:152-169`) shows. Two copies means one goes stale, which is `phase3-plan.md:826-830`'s
failure mode #2 ("a decision recorded in prose but not in the block coded from") reproduced
structurally. It would also double the pre-1.0 public API that later needs deprecating, for no
external consumer, and hide which class owns a guard when the guards are the interesting part.

Cost: a breaking source change across seven in-repo files (`MLXNumericTest`, `MLXArrayTest`,
`MLXEvalTest`, `MLXGpuVerificationTest`, `MLXMemoryLeakTest`, `MLXNativeErrorTest`, `HelloMLX`).
Mechanical.

**Sequencing is what makes the cost near-zero: do the split first, as pure motion, before adding a
single new op.** Then `git diff -M --stat` on that commit shows renames only, and a reviewer can
verify "nothing changed but the address". Adding 30 ops and *then* splitting produces a diff nobody
can audit.

### 2. Child scopes — `MLXScope`, `MLXArray`, `binaryOp` (M0b)

```java
public MLXScope newChild();                                  // checkAccess() on parent first
public MLXScope parent();                                    // null for a root
public int depth();                                          // parent == null ? 0 : parent.depth + 1
public boolean isAncestorOf(MLXScope other);
public static MLXScope innermost(MLXScope x, MLXScope y);    // IllegalArgumentException if unrelated
```

`depth` makes the *pick* O(1); the ancestor *check* is an O(depth) walk, and depth is at most about
three in every realistic use (root → model → step). Do not cache ancestor sets — invalidation costs
more than the walk.

`newChild()` calls `checkAccess()` on the parent, so **a child always has the same owner thread as
its parent, by construction**. Ancestor-related scopes are therefore same-thread automatically —
which is exactly why the ancestor rule cannot be, and must not be described as, a confinement guard.

**The scope rule is a helper-level invariant over N operands, not a rewrite of `binaryOp`.** Every op
body — including the hand-rolled ones — resolves its target through one function:

```java
/** Innermost scope among all operands, skipping nulls (nullable native params). Touches every
 *  non-null operand's scope, so each one's checkAccess() runs.
 *  Precondition: at least one operand is non-null -- the caller always passes the primary
 *  input (x, q, w) which is never nullable in any mlx-c signature. Violating it throws
 *  IllegalArgumentException naming the op, NOT an empty-reduce ISE: the reachable case is
 *  scopeOf(weight, bias) inside a layerNorm body that forgot to pass x, and the message
 *  should say that rather than "empty".
 *  Throws IllegalArgumentException if any two operand scopes are unrelated. */
static MLXScope scopeOf(String op, MLXArray... operands);   // reduces through MLXScope.innermost
```

`binaryOp` becomes `MLXScope scope = scopeOf("add", a, b);`. It still rejects unrelated scopes — siblings,
and two independent roots — because that is the case where "which scope owns the output" has no
defensible answer. It stops rejecting parent/child pairs, which is the entire point.

**Stating the rule once, at the helper, is not a style preference — restating it per call site
silently drops the invariant on every op with more than two operands.** Verified operand counts:

| Op | Array operands | Nullable among them |
| --- | --- | --- |
| `mlx_fast_scaled_dot_product_attention` | 5 — `q`, `k`, `v`, `mask_arr`, `sinks` | `mask_arr`, `sinks` |
| `mlx_quantized_matmul` | 4 — `x`, `w`, `scales`, `biases` | `biases` |
| `mlx_gather_qmm` | 6 — `x`, `w`, `scales`, `biases`, `lhs_indices`, `rhs_indices` | **3** — `biases`, `lhs_indices`, `rhs_indices` (`ops.h:480-482`) |
| `mlx_dequantize` | 4 — `w`, `scales`, `biases`, `global_scale` | `biases`, `global_scale` |
| `mlx_fast_layer_norm` | 3 — `x`, `weight`, `bias` | `weight`, `bias` |
| `mlx_where`, `mlx_addmm` | 3 | none |
| `mlx_fast_rms_norm`, `mlx_fast_rope` | 2 — `x` + `weight` / `freqs` | `weight` / `freqs` |

The concrete failure this prevents is in the decode loop §7 targets: `k`/`v` live in a KV-cache scope
and `q` in the step scope, so an SDPA that allocates into `q.scope()` **never calls `checkAccess()`
on `k` or `v`** and silently drops the ownership invariant §2 exists to establish. `scopeOf("sdpa", q, k, v,
mask, sinks)` both picks the target and forces every operand's access check.

Nullable operands are skipped rather than rejected, since a null layer weight is legitimate.

**The comment at `MLX.java:242-246` is already stale today and must be rewritten in the same commit.**
It justifies the check by claiming "b's scope … is never touched at all", but `b.scope()` on the very
next line runs `MLXArray.ensureOpen()` → `MLXScope.checkAccess()` (`MLXArray.java:26-28`, `:122`) —
which `phase3-plan.md:361-366` added for exactly this reason and explicitly describes as demoting
this check to defence-in-depth. Leaving the comment would preserve a justification for a guard now
doing a different job, which is failure mode #2 verbatim.

`hoist` lives in `core`, because `MLXScope` sits in `se.alipsa.jmlx.memory` and cannot construct an
`MLXArray`:

```java
/** Lifts a into target, which must be a's own scope or an ancestor of it. Copy, not move:
 *  after the child closes, use the returned array and never {@code a}.
 *
 *  target == a.scope() is LEGAL and returns {@code a} itself -- ancestor-of is reflexive here.
 *  Identity, not a copy: when source and target are the same scope both handles would die
 *  together, so the copy buys nothing and an unconditional hoist(a, modelScope) in a loop
 *  would accumulate one fresh handle per iteration in the target -- the exact leak shape the
 *  fifth sub-hazard below is about. Consequence to document: in the reflexive case the result
 *  ALIASES the argument, so closing one closes the other. */
public static MLXArray hoist(MLXArray a, MLXScope target);

/** hoist(a, a.scope().parent()). IllegalStateException("MLXScope has no parent") for a root
 *  scope -- NOT a NullPointerException from an unchecked parent() deref. */
public static MLXArray keep(MLXArray a);
```

Implementation is `mlx_array_new(target)` then `mlx_array_set(lifted, a.handle())` — see Research
findings for why that is a second wrapper on the same descriptor.

**Five sub-hazards, each of which must land in this same commit:**

* **`Holder` gains `LinkedHashSet<Holder> children`, and `closeAll()` cascades children in reverse
  insertion order *before* freeing its own handles.** Mandatory, not tidy: the `Cleaner` backstop
  only ever sees a `Holder` (`MLXScope.java:46-51`), so a forgotten `child.close()` would otherwise
  leak silently. The capture rule (`initial-plan.md:420-423`) is preserved because `parent.holder`
  references `child.holder`, never the child `MLXScope` — so a child can still become unreachable and
  fire its own `Cleanable`; the parent's later cascade hits an already-closed `Holder` and returns
  early (`MLXScope.java:65-67`).
* **`child.close()` must call `parent.holder.removeChild(...)`**, or a hot per-step child loop under
  a long-lived parent accumulates dead `Holder` objects — a slow leak that no existing test sees.

  **Lock ordering, specified because both orders are reachable.** The cascade takes the parent
  holder's monitor and then each child's. If `removeChild` were called from *inside*
  `Holder.closeAll()`, `child.close()` would take child-then-parent. Both can run at once: the
  `Cleaner` thread can cascade a parent `Holder` (parent `MLXScope` unreachable) while the owner
  thread runs `child.close()` — the bullet above establishes that a child `MLXScope` can still be
  reachable when its parent is not. That is a classic ABBA deadlock. **`removeChild` is therefore
  called from `MLXScope.close()`, after `holder.closeAll()` has returned and released its monitor —
  never from inside a `Holder` method.** The cascade path does not call it at all: a cascading parent
  is discarding its whole child set anyway.
* **The cascade opens a new use-after-free of exactly the class `phase3-plan.md` §1 closed.**
  `Holder.closeAll()` cannot flip the child's `MLXScope.closed` field (that would be a capture), so
  after a parent cascade a child reads `closed == false` while its handles are already freed. Fix:
  `ensureOpen()` also consults the holder.

  ```java
  private void ensureOpen() {
    if (closed || holder.isClosed()) { throw new IllegalStateException("MLXScope is closed"); }
  }
  ```

  **`Holder.closed` must be changed from a plain field to `private volatile boolean closed`, and
  `isClosed()` must be an unsynchronized volatile read.** Today it is a plain `boolean` guarded
  entirely by `synchronized` methods (`MLXScope.java:52`, `:54`, `:64`, `:83`), which is correct for
  its current callers but wrong for this one: `MLXArray.ensureOpen()` runs on **every**
  `shape()`/`dtype()`/`handle()` access — several times per op — so a `synchronized boolean
  isClosed()` puts a monitor acquisition on the hottest path in the facade, while an unsynchronized
  plain read is a data race with the `Cleaner` thread that sets it. `volatile` is the only option
  that is both correct and free; the existing `synchronized` methods keep working unchanged, since
  `volatile` only strengthens their guarantees.

  This is a **mandatory test row**, not an optional one.
* **`MLXArray.scope()` becomes public** (package-private today, `MLXArray.java:26`). Creation ops have
  no operand to infer a scope from — `arange` for RoPE positions, `triu` for a causal mask, `zeros`
  for a KV cache — and must take one explicitly, exactly as `MLX.array` does (`MLX.java:69`). Using
  the *model* scope for them is a per-step leak whose symptom is OOM after N steps, not a failure.
  With a public `scope()`, `nn` writes `MLX.arange(x.scope(), ...)` and `forward(MLXArray x)` needs no
  scope parameter — which is what the outline's `JMLXDemo` sample already assumes. This is the same
  one-method-widening trade `phase3-plan.md:322-345` accepted for `checkAccess()`, and it costs
  Decision 6 of that document one more method. The alternative — threading `MLXScope` through every
  `forward` signature — pollutes `MultiHeadAttention` worst.
* **Single-operand ops have the same hazard as creation ops, and `scopeOf` cannot fix it.**
  `unaryOp`, `shapeOp` and the new `reduceOp`/`axisOp`/`axis2Op` have exactly one operand, so
  `scopeOf(op, a)` is `a.scope()` — correct by definition and useless as a guard. Any
  `astype(W)`, `transpose(W)`, `reshape(W)` or `swapaxes(W)` **inside `forward`** therefore allocates
  into the *model* scope: one retained array per parameter per step, with precisely the "OOM after N
  steps, not a failure" symptom named for creation ops above. `Linear.forward`'s `transpose(W)` is the first
  place this bites.

  The `scope()` javadoc ("`forward()` must never allocate here") is documentation, not enforcement.
  Two mitigations, both taken:

  1. **An explicit target overload on the unary helpers** — `MLXShape.transpose(a, MLXScope target)`
     and friends — so a layer that must reshape a weight per step can say where the result goes.
     Ergonomically ugly, which is the point: the ugly spelling is the one that is correct in a loop.
  2. ~~Cache the derived view in the constructor and recompute it from a change hook.~~
     **Withdrawn — see below. Mitigation 1 is the rule; `Linear` uses it.**

  **Why caching a derived weight view is withdrawn, recorded rather than deleted because it is the
  first thing an implementer will reach for again.** The idea was: `Linear` computes `W.T` once in
  its constructor, stores it in a plain field, and recomputes it from an overridable
  `onParametersUpdated()` when `update` replaces `W`. It survives §5 in isolation and **fails against
  §6's autograd rebinding, in two directions at once**:

  * **The restore leaks.** `ModuleGrad` rebinds parameters to the traced primals and restores them in
    a `finally` (§6). If the restore fires the recompute hook, `transpose(W)` resolves through
    `scopeOf` to `W.scope()` — the **model** scope — once per cached view per training step, in the
    scope that never closes. Thirty layers over a thousand steps is the same order as the ~300k
    figure Context ¶2 opens with. It leaks *because the restore is correct*.
  * **Suppressing the recompute dangles the field.** The obvious patch — skip the notify when the
    incoming values are reference-equal to the current binding — does not fire on the restore path at
    all, because the binding at that instant is the *traced* array, not the saved one. And if it did
    fire, the cached `W.T` would still be the view built during the traced rebind: a **step-scope**
    array, dangling the moment `apply` returns. Lazy invalidation fails identically — it puts the
    view in whichever scope the next `forward` ran in.

  **So `Linear` does not cache. `forward` computes `transpose(W, x.scope())` through mitigation 1's
  explicit-target overload**, landing the view in the step scope where it is freed with everything
  else. The cost is one lazy graph node per layer per forward — a strided view, not a copy, and
  exactly what upstream's `x @ self["weight"].T` does on every call.

  **Registering `W.T` as the parameter is also rejected, for an unrelated reason worth keeping.**
  `parameters()` is the surface `update(Map)` writes through and §9 names it as the safetensors entry
  point; registering `W.T` would desynchronize it from the checkpoint's on-disk `[out, in]` layout and
  force `update` to silently re-transpose — a per-layer convention with no signal.

  `onParametersUpdated()` survives, narrowed: **called by `update()` only, never by `rebind()`** (§6),
  for Phase 5 layers that genuinely need post-load work. Its javadoc must carry the warning above —
  a derived view cached across a trace is incompatible with `ModuleGrad` unless the layer reads it
  through an accessor rather than a field. See Open questions.

  Test coverage is `MLXMemoryLeakTest` with a child-scope loop; there is no compile-time signal.

### 3. Op helpers and creation ops (M0d)

Six new private helpers alongside `binaryOp`/`unaryOp`/`shapeOp`, so each shape exists once:

```java
reduceOp(name, a, int[] axes, boolean keepdims, ...)   // (res, a, const int*, size_t, bool, stream)
axisOp(name, a, int axis, ...)                          // expandDims, tril, triu, sumAxis, softmaxAxis
axis2Op(name, a, int i, int j, ...)                     // swapaxes, flatten, moveaxis
vectorInOp(name, MLXArray[] xs, int axis, ...)          // concatenate, stack
vectorOutOp(name, MLXScope target, ...)                 // split, quantize
nullableHandle(MLXArray a, SegmentAllocator tmp)        // zero-ctx STRUCT, never MemorySegment.NULL
optInt / optFloat / optDtype(SegmentAllocator tmp, T v) // by-value optional structs
cstr(String s)                                          // closed sets -> allocate once in FACADE_ARENA
```

**Every helper above, and every hand-rolled op body, resolves its target through `scopeOf(...)` from
§2 — passing *all* array operands including the nullable ones.** That is the single place the
ownership rule lives; a body that writes `a.scope()` directly is the defect §2's operand-count table
describes. Two exceptions take an explicit scope instead: the creation ops, which genuinely have no
operand, and `vectorOutOp`.

**`vectorOutOp`'s explicit target is a legibility choice, not a necessity — say so rather than
inventing a reason.** Both ops it serves *do* have array inputs (`mlx_split(res, a, …)`,
`mlx_quantize(res, w, …, global_scale, s)`), so `scopeOf("quantize", w, global_scale)` would resolve the target
fine. The parameter is kept because the helper's whole hazard is that its two allocators have
opposite correct answers on adjacent lines, and naming the scope in the signature is what makes that
contrast readable at the call site. An earlier draft justified it as "its inputs may be none", which
is false — recorded in Failure modes rather than quietly corrected.

**`vectorInOp` reduces its operands through `scopeOf`** — the ancestor rule generalized to N operands
— and builds its `mlx_vector_array` from a confined `Arena`, reusing a
`copyHandlesInto(handles, allocator)` factored out of `newVectorArray` (`MLX.java:523-549`) rather
than growing a second copy of the byte-for-byte struct-copy loop.

**`vectorOutOp` is the sharpest of the six: it uses two allocators with opposite correct answers on
adjacent lines.**

```java
try (Arena tmp = Arena.ofConfined()) {
  MemorySegment vec = mlx_h.mlx_vector_array_new(tmp);        // tmp -- NOT target
  try {
    checked(name, () -> op.apply(vec, ...));
    long n = mlx_h.mlx_vector_array_size(vec);
    for (int i = 0; i < n; i++) {
      MemorySegment h = mlx_h.mlx_array_new(target);          // target -- NOT tmp
      final long idx = i;
      checked(name, () -> mlx_h.mlx_vector_array_get(h, vec, idx));
      out[i] = new MLXArray(target, h);
    }
  } finally { mlx_h.mlx_vector_array_free(vec); }
}
```

Getting them backwards is `phase3-plan.md` Verification #12's wrong-type delete — applying
`delete static_cast<mlx::core::array*>(ctx)` to a `std::vector<mlx::core::array>*` — except now once
**per element**. Verification #7 extends that manual check to both new helpers, because no test can
catch it: it is UB, not a failed assertion.

Ops added at this merge point: `astype`, `array(scope, int[], int[])`, `zeros`, `ones`, `full`,
`arange`, `stopGradient`.

**`full` is not a pure creation op and must not be routed through `createShapeOp`.** Its fill value
is an `mlx_array` (`ops.h:437-443`), so `MLX.full(scope, shape, value, dtype)` has to construct a
scalar array first via `mlx_array_new_bool`/`_int`/`_float32` — all of which are **statusless and
return a null-ctx struct on failure** (`array.cpp:64-71`). That is the hazard class `MLX.array`
already handles explicitly (`MLX.java:79-90`), and `full`'s body must do the same: check
`mlx_array_.ctx(...).address() == 0` and free the scalar in a `finally`. `zeros`/`ones`/`arange` have
no such step.

### 4. `DType` and `astype` (M0c)

Add `BOOL`, `UINT32`, `FLOAT16`, `BFLOAT16`. The per-constant allowlist design at `DType.java:21-29`
already anticipated this and needs no structural change — only `fromNative`'s "only float32/int32"
message (`DType.java:37-38`) is falsified. `nativeValue()` stays package-private; `astype` is in
`core`.

* `BOOL` is **required, not optional** — the comparison ops `mlx_less`/`mlx_greater_equal`/`mlx_equal`
  produce it, so `dtype()` on a mask would otherwise throw from `fromNative`.
  **`mlx_triu` is not one of them:** `wheel/mlx/include/mlx/ops.h:129-130` is
  `MLX_API array triu(array x, int k = 0, StreamOrDevice s = {})` — dtype-*preserving*, not a
  predicate, so `triu` of a float32 array is float32. The boolean causal mask is
  **`triu(ones(shape, BOOL), 1)`**, which is where `BOOL` enters that path.

  **Use `ones`, not `full`.** `mlx_ones(res, shape, shape_num, dtype, s)` (`ops.h:718-723`) is a
  plain status-returning creation op. `mlx_full(res, shape, shape_num, const mlx_array vals, dtype,
  s)` (`ops.h:437-443`) takes its fill value **as an `mlx_array`**, so building a BOOL mask through
  `full` first needs `mlx_array_new_bool`, which is **statusless and signals failure only via a null
  ctx** (`array.cpp:64-71` returns `mlx_array_()` on exception) — the same hazard class `MLX.array`
  already handles specially at `MLX.java:79-90`. `ones` avoids that entirely for this path.
* `UINT32` is required because `mlx_quantize`'s `w_q` is uint32-packed. Read-back-only.
* `FLOAT16`/`BFLOAT16` are needed to load real weights in Phase 5, and are added now so `astype` and
  `matmul`'s guard are settled once rather than twice.

**`toFloatArray()` gains an `mlx_astype(FLOAT32)` ahead of `mlx_contiguous`** for any inexact
non-float32 dtype. **This is not a one-line insertion into the existing body.** `MLXArray.java:84-99`
allocates exactly one handle (`mlx_array_new(tmp)`) and frees it in exactly one `finally`, and the
comment at `:77-83` explains why that `finally` exists: `tmp` owns only the 8-byte struct, while the
ctx heap allocation is mlx-c's and is freed solely by `mlx_array_free`. The astype step needs **its
own handle and its own `finally`** — nested, inner-first — or the f16 path leaks one native array per
read, on both the success and failure paths. One extra lazy op on the
read-back path buys a single code path for f16/bf16/f32, no `half.h` change, and **no
`regen-bindings.sh` run** — so `phase3-plan.md` Verification #2's premise ("this phase required no
binding changes") survives into Phase 4. Exact dtypes stay rejected: silently `astype`-ing an
`INT32` array to float would hide the bug the check exists to surface. Add `toIntArray()` for
`INT32`; leave `UINT32` read-back-only, since packed weights are not meant to be inspected
element-wise from Java.

**`requireMatmulCompatible` is revised in this same commit**, discharging `phase3-plan.md:857-862`'s
explicit Phase 4 entry condition. `MLX.java:183-185` becomes:

```java
if (!a.dtype().isInexact() && !b.dtype().isInexact()) {
  throw new IllegalArgumentException("matmul: requires at least one inexact dtype, got "
      + a.dtype() + " and " + b.dtype());
}
```

and the javadoc paragraph at `MLX.java:159-168` — which states the divergence "is unreachable today"
— must be replaced, because `astype` makes that factually false in this very commit.

**The rejected alternative matters.** Implementing `DType.promote(a, b)` and testing
`promoted.isInexact()` would mirror native exactly *if the lattice were right*, but it reimplements
mlx's promotion table in Java with no native counterpart to defer to — a fresh instance of
`phase3-plan.md:821-826`'s failure mode #1. The both-exact rule needs no lattice and is a **provable
subset** of native's rejects, since `promote_types(exact, exact)` is always exact: it can
under-reject (only for pairs this facade cannot construct) but never over-reject. Over-rejecting is
the documented failure mode; under-rejecting surfaces as a clean `MLXException` carrying mlx's own
message.

### 5. `Module` and the simple layers (M1)

```java
public abstract class Module {
  protected Module(MLXScope scope);
  protected final MLXArray param(String name, MLXArray value);      // duplicate -> IllegalStateException
  protected final <M extends Module> M child(String name, M module);
  protected final MLXScope scope();          // WEIGHTS live here; forward() must never allocate here
  public final SequencedMap<String, MLXArray> parameters();         // depth-first, own params before
                                                                    // children, dotted paths, insertion-
                                                                    // ordered, immutable snapshot
  public final void update(Map<String, MLXArray> byPath);           // Phase 5 hook; see notify rule below
  public final void freeze();                                       // structural mutation throws after this
  /** Post-load hook for Phase 5 layers needing work after update(). Called by update() ONLY,
   *  never by rebind(). Caching a derived weight view here is incompatible with ModuleGrad
   *  tracing -- see the withdrawn mitigation in §2 before using it. */
  protected void onParametersUpdated() {}
}

/** Layers with a single-tensor forward. Deliberately NOT on Module: MHA's arity differs. */
public interface UnaryModule { MLXArray forward(MLXArray x); }
```

| Approach | Ordering guarantee | Phase 5 `update` | Grad re-scatter | Verdict |
| --- | --- | --- | --- | --- |
| abstract class + explicit `param(name, value)` | `LinkedHashMap` — specified insertion order | trivial, in place | index-stable | **chosen** |
| records / sealed interface | ctor components, ordered | requires rebuilding the whole tree per update | new `MLXArray` identities per step invalidate any captured order | rejected |
| reflection over fields | **none** — unspecified by the JLS | works | **silently wrong** | rejected |

**Weights live in the model scope; activations arrive in a descendant step scope; for any op that
takes an activation as an operand, `scopeOf` puts the result in the step scope automatically. That is
the payoff of §2, and why `forward` takes no scope parameter.**

**The automatic part stops at ops that touch only weights, and §2's final bullet is the governing
rule, not a footnote.** A single-operand op on a parameter — `astype(W)`, `transpose(W)` — allocates
into the model scope, and a creation op called with `scope()` does the same. Both leak once per step.
`Linear` must therefore compute its transposed view through the **explicit-target overload** —
`transpose(W, x.scope())`, landing it in the step scope — with `W` itself as the registered
parameter. It does **not** cache the view; §2 records why that mitigation was withdrawn. Layers
needing step-varying weight transforms use the explicit target overload. The `scope()`
javadoc says
`forward()` must never allocate here, but nothing enforces it — `MLXMemoryLeakTest` with a
child-scope loop is the only detection.

`update` must throw on an unknown path rather than ignore it — silently ignoring is how a Phase 5
weight-name typo becomes a garbage model with no error.

**`update`'s notification rule, spelled out because the obvious implementation is wrong.** `update`
is a `final` method on the *root*, but `parameters()` is depth-first with dotted paths, so a
safetensors load writes into nested `Linear`s several levels down. **Notifying only the receiver
leaves every nested layer's post-load state unrefreshed — exactly the bug the hook
was added to fix, surviving one level of nesting.** The contract is therefore:

> `update` performs **all** writes first, then walks the subtree and calls `onParametersUpdated()` on
> **every module whose own params were written**, in depth-first order.

All-writes-then-all-notifies, not interleaved: a layer whose derived state depends on two parameters
(or on a sibling's) would otherwise recompute against a half-updated tree. Modules with no written
params are skipped — the hook is not a generic "something changed somewhere" broadcast.

`Module` also gains **`rebind`**, used only by `ModuleGrad` (§6) and specified there:

```java
/** Value-only parameter replacement: same key set, new arrays. Legal after freeze(), which
 *  blocks STRUCTURAL mutation only. Does NOT call onParametersUpdated() -- see §6. */
public final void rebind(SequencedMap<String, MLXArray> values);
```

**`ModuleGrad`, the module-aware autograd wrapper, lands at M2 and is specified in §6** — next to
`MLXGrad.Fn`, whose contract it depends on, and in `nn` rather than `core` for the layering reason
given there. It is named here only so `Module`'s surface is complete.

Layers: `Linear`, `Embedding` (`take_axis` with INT32 indices), `RMSNorm`, `LayerNorm`, `SiLU`
(`x * sigmoid(x)`), `GELU` (exact: `0.5*x*(1+erf(x/√2))`; the tanh approximation is a second
constructor arg, not a second class).

**There is no `mlx_silu`, `mlx_gelu` or `mlx_relu` in the C API.** These are Python-level
compositions upstream too, so composing them in Java is expected rather than a binding gap.

Ops added: `swapaxes`, `take`/`takeAxis`, `sigmoid`, `erf`, `tanh`, `sqrt`, `rsqrt`, `square`,
`negative`, `power`, `maximum`, `mean`/`sum` with axes+keepdims, `MLXFast.rmsNorm` (nullable
`weight`) and `MLXFast.layerNorm` (**two** nullables), plus `MLXRandom` for weight init — the key is
nullable, so a global `mlx_random_seed` suffices and no Java-side key-splitting design is needed.

### 6. `MLXGrad` (M2)

Deliberately sequenced **after** M1, so there is a real `Module` tree and a real flatten order to test
against rather than a synthetic one.

```java
public final class MLXGrad {

  /** Primitive form: differentiate a function of a flat primal vector. */
  public static Fn valueAndGrad(Function<MLXArray[], MLXArray[]> body, int[] argnums);
  public static final class Fn implements AutoCloseable {
    /** target receives the values, the grads, and the arrays handed to body. */
    public Result apply(MLXScope target, MLXArray[] primals);
  }
  public record Result(MLXArray[] values, MLXArray[] grads) {}

}
```

**`MLXGrad` is primitive-only and must stay that way — a `Module`-aware overload here would invert
the layering this entire document is built on.** `MLXGrad` is in `core` (§1); `Module` is in `nn`
(§5). A `valueAndGrad(Module, ...)` in `core` makes `core` import `nn` while `nn` imports `core` — a
package cycle, and one that would block the `module-info.java` with `exports … to …` floated in the
last Open question. It also contradicts Context ¶3 directly, which states that `nn` builds on the
public op surface and that autograd lives in `core` *only* because the upcall needs `MLXArray`'s
package-private constructor.

**And the wrapper needs none of that access.** `Fn.apply(MLXScope, MLXArray[])`, `Result`,
`parameters()`, `freeze()` and `MLXScope` are all public; zipping grads back onto dotted paths is
plain Java. The module-aware form therefore lives in `nn` — see `ModuleGrad` below — and the
dependency arrow stays one-way.

**The target scope belongs on `apply`, not on the constructor, and the two placements are mutually
exclusive rather than a matter of taste.** Grads must land in the *step* scope (see the second hazard
below), and a step scope is created and closed once per iteration. Binding `target` at construction
forces a choice between two broken shapes: a `Fn` hoisted out of the loop has a `target` that is
closed by iteration two, and a `Fn` built inside the loop allocates and frees one confined `Arena`
and one FFM upcall stub **per step** — the exact per-shape stub churn this section spends a paragraph
avoiding when it rejects the `ERROR_HANDLER_ARENA` pattern. With the scope on `apply`, one `Fn`
spans the whole loop and each step's grads land in that step's scope.

**Package is forced, not chosen.** The upcall constructs `MLXArray`s from raw handles arriving from
C++, and that constructor is package-private to `se.alipsa.jmlx.core`. `nn` consumes `MLXGrad`; it
cannot contain it.

**Arena lifetime.** A per-`Fn` `Arena.ofConfined()`, closed in `Fn.close()` **after**
`mlx_closure_value_and_grad_free` and `mlx_closure_free`. Order is load-bearing: mlx's
`std::function` (`private/closure.h:16-21`) holds the raw stub pointer, so closing the arena first
leaves a dangling stub inside a live closure. **Do not copy `NativeLoader`'s process-lifetime
`ERROR_HANDLER_ARENA` pattern** (`NativeLoader.java:114-117`) — that arena is never closed because
mlx-c retains the error handler for the process lifetime, which is not true of a grad closure.
Copying it would leak one upcall stub per closure shape, forever.

`ofConfined` is correct **iff** mlx invokes the closure on the calling thread. It does
(`mlx_closure_value_and_grad_apply` → `mlx_closure_get_(cls)(...)`, `closure.cpp:96-108`,
synchronous), but Verification #0b asserts it rather than assuming it. `ofShared` is the one-word
fallback if it ever fails.

**`Result` is filled by two vector-array reads, not one.**
`mlx_closure_value_and_grad_apply(mlx_vector_array* res_0, mlx_vector_array* res_1, cls, input)`
(`native/install/include/mlx/c/closure.h:89-93`) has **two** out-params — `res_0` values, `res_1`
grads. Each unpacks with exactly the `vectorOutOp` shape from §3 and therefore carries the same
two-allocator hazard, twice. Verification #7's manual grep is understood to cover both.

**Round-trip inside the upcall.** The signature is
`int apply(MemorySegment out /* mlx_vector_array* */, MemorySegment in /* mlx_vector_array by value */)`.

* **`in`'s lifetime is the upcall only.** Never retain it, never store it in an `MLXArray`.
  `mlx_vector_array_get` copies each element into a fresh handle, which is what makes this safe.
* **Fill `out` with `mlx_vector_array_set_data`, never `new_data`** — because `new_data` returns a
  struct *by value* (`vector.h:31`) and would leave `*out` untouched, whereas `set_data` writes
  through the pointer (`vector.h:33`). **Not** because `out` is pre-populated: its ctx is null
  (`private/vector.h:12-14`). See Research findings for why the distinction is recorded rather than
  simplified. Reuse `copyHandlesInto` from §3.

**Exception safety — the three-step protocol, all steps required.**

1. The stub catches `Throwable`, stashes it on a `ThreadLocal`, returns `1`. Never let it escape.
2. mlx-c frees both vectors and throws `runtime_error` (`closure.cpp:48-51`), which propagates to
   `mlx_closure_value_and_grad_apply`'s catch → `mlx_error` → status 1 → `MLXException` via `checked`.
3. Java catches that `MLXException`, reads and clears the `ThreadLocal`, and — if non-null —
   **rethrows the original Java exception with the `MLXException` attached as suppressed**. Same shape
   as `attributeEvalFailure` (`MLX.java:559-576`).

Without step 3 the user sees a generic native message instead of their own `IllegalArgumentException`
— a diagnostics regression that reads as a native bug. **Clear the `ThreadLocal` before every
`apply`**, for exactly the reason `checked` clears `LAST_NATIVE_ERROR` *before* the call
(`MLX.java:594-596`): a stale value is otherwise misreported as the next failure's cause.

**`argnums`** is a plain `int*` + count. Guard Java-side — non-empty, strictly increasing, all in
`[0, primals.length)` — so the error names the operand (`initial-plan.md:446-448`).

**Only `values[0]` is differentiated, and it must be rank-0. Both belong in the javadoc and in a Java
guard.** `native/scratch/wheel/mlx/include/mlx/transforms.h:66-68` defines the general form as
returning "a pair of vectors of arrays one for the values and one for the gradients **wrt the first
value**", and the scalar convenience overload at `:102-112` is a wrapper that packs one array into a
one-element vector and returns `result.first[0]`. So a `body` returning `MLXArray[]` **silently
differentiates element 0 and ignores the rest**, and a non-rank-0 element 0 fails inside native with
an mlx-c string.

Guard it next to `argnums`, for the reason this document already gives one paragraph earlier: check
`values[0].ndim() == 0` on the way out of `body`, inside the upcall, and fail with a Java exception
naming the actual rank. **This is the most common user error in the whole API** — returning a
per-element loss instead of a reduced one — and it is the one place where deferring to native's
message costs the most.

#### `ModuleGrad` — the module-aware wrapper (`nn`, M2)

```java
package se.alipsa.jmlx.nn;

public final class ModuleGrad implements AutoCloseable {
  /** Freezes tree and captures parameters().keySet() -- the ORDER only, not the values.
   *  loss receives (params, inputs) and must return a rank-0 loss as element 0. */
  public static ModuleGrad of(Module tree, BiFunction<MLXArray[], MLXArray[], MLXArray[]> loss);

  /** Per-iteration: target is this step's scope, inputs is this batch. */
  public Result apply(MLXScope target, MLXArray[] inputs);

  public record Result(MLXArray value, SequencedMap<String, MLXArray> grads) {}

  @Override public void close();   // delegates to the inner MLXGrad.Fn
}
```

**The batch belongs on `apply`, for exactly the reason `target` does.** With only `target` there, a
batch could reach the body only by capture, so every batch would need a fresh `ModuleGrad` — each
owning a confined `Arena` and an FFM upcall stub. Same shape as Failure modes #5.

**Primal layout is specified, not implied: parameters first, inputs after,
`argnums = 0..paramCount-1`.** Differentiating w.r.t. the batch as well yields extra entries or a
shifted zip — silently wrong gradients with correct shapes.

#### Rebinding — the mechanism that makes the gradients mean anything

**Without it `ModuleGrad` computes gradients disconnected from the model, and the broken form is the
natural one to write.** `Linear.forward(x)` reads its own fields — the registered `W` and the
registered `W` (§5). The `MLXArray`s handed to `loss` as `params` are the ones `MLXGrad`
reconstructs from the primal vector *inside the upcall*: different objects. So the obvious loss —

```java
ModuleGrad.of(model, (params, inputs) -> new MLXArray[] { mse(model.forward(inputs[0]), inputs[1]) })
```

— traces a graph over `model`'s registered weights while `mlx_value_and_grad` differentiates w.r.t.
the primals, which are not inputs of that graph. No exception; meaningless numbers.

**Mechanism: `ModuleGrad` rebinds the tree to the traced primals around the `loss` call.** Inside the
closure body, before invoking `loss`:

```java
SequencedMap<String, MLXArray> saved = tree.parameters();   // re-read EVERY apply, see below
try {
  tree.rebind(zip(paths, tracedPrimals));
  return loss.apply(tracedPrimals, inputs);
} finally {
  tree.rebind(saved);            // MUST run, including on the exception path
}
```

Three things about this are load-bearing and none is optional:

* **`rebind` must NOT fire `onParametersUpdated()`, and this is what forced §2's cached-view
  mitigation to be withdrawn.** Firing it on the *restore* call would recompute every cached derived
  view against the model-scope arrays — one retained native array per view per training step, in the
  scope that never closes. Suppressing it by an identity check does not help: at restore time the
  module's current binding is the *traced* array, so `saved` is not reference-equal to it; and if the
  notify were suppressed anyway, the cached view would remain the one built during the traced rebind,
  a step-scope array dangling the moment `apply` returns. **Leak in one direction, use-after-free in
  the other.** The resolution is that no layer caches a derived weight view (§2), so `rebind` has
  nothing to notify and stays a pure value swap. `update()` still notifies, because its replacement
  is permanent rather than transient.

  **A layer that does cache one is therefore incompatible with `ModuleGrad` unless it reads the view
  through an accessor rather than a field.** Nothing enforces this; it is an Open question.
* **The restore must survive an exception thrown out of `loss`.** Traced primals are allocated in
  `apply`'s `target` — the *step* scope. If a rebind is left in place and the step scope closes,
  `model.forward` outside the traced region reads freed handles: the use-after-free class §2's
  cascade bullet exists to close, reached by a different route. The `finally` above is what closes
  it, and it must sit *inside* the upcall's `catch (Throwable)` so both protocols compose.
* **`rebind` is legal after `freeze()` because `freeze()` blocks structural mutation only** — the key
  set is unchanged, which is also what makes the restore exact.

**This mechanism is unverified against this mlx version and is a Verification #0 item, not a
citation.** Upstream `mlx.nn.value_and_grad` is understood to wrap the body with a
`model.update(params)` of the same shape, but **the `mlx-metal` wheel ships no `mlx/nn/` Python
package** (`native/scratch/wheel/mlx/` contains `utils.py` and `extension.py` only), so nothing in
this repository confirms it. Probe **0f** below settles it empirically instead of asserting it.

#### `apply` re-reads parameter *values* every call

`of()` captures the **key order only**. `apply` re-reads `tree.parameters()` in that order on every
call, and throws if the key set has drifted.

**Snapshotting values at `of()` — which an earlier draft's "captures … ONCE, here" invites — silently
differentiates against stale weights after the first optimizer step.** `freeze()` blocks structural
mutation, but replacing parameter *values* is exactly what `update(Map)` does, and it is both the
Phase 5 hook and the subject of the optimizer Open question. Correct shapes, no exception, model
trains to garbage. That is `Linear`'s stale-`W.T` bug one class over — and where that one got
`onParametersUpdated()`, this one has nothing but this rule.

**Lifetime:** `ModuleGrad` owns no native resources of its own; it holds one `MLXGrad.Fn` and its
`close()` delegates. The free-ordering rule above is stated once, for `Fn`, and applies transitively.

**Re-scatter hazard, named precisely.** Capture `parameters().keySet()` **once**, before building the
closure, and **never zip grads against a recomputed order**. The per-`apply` drift check above
*compares* against that captured order; it does not replace it. If anything mutates the module tree in between, a
recomputed order misassigns every gradient — no exception, just a model that trains to garbage.
Mitigation: `parameters()` returns an immutable snapshot, and capturing its key order once is on its
own sufficient.

**Freezing is a separate, explicit operation — `parameters()` must not mutate the module.** An
earlier draft had the read accessor set a `frozen` flag. That makes a getter permanently change
behaviour as a side effect, and it forecloses the legitimate case of inspecting a partially built
tree during construction or debugging. Instead: `Module.freeze()` is explicit, and **`nn.ModuleGrad.of`
(above, this section) — the only entry point that receives a tree — calls it**. Structural mutation after that point
throws; before it, `parameters()` is a plain read. Callers using the primitive `Fn` form freeze
explicitly or accept the risk, which is the correct division: the form that knows about modules is
the form that can enforce module invariants, and it is the form that lives in `nn`.

**Second hazard:** grads are allocated into `apply`'s `target`, which must be the **step** scope. If
it is the model scope, every step leaks one gradient tensor per parameter — which is why `target` is
a per-call argument rather than a `Fn` field.

### 7. RoPE, MultiHeadAttention, KV cache (M3)

`MLXFast.rope` — `mlx_optional_float base`, plain-`int` `offset` (the static KV-cache offset;
`mlx_fast_rope_dynamic` takes it as an array and is out of scope until a compiled decode loop exists),
nullable `freqs`.

`MLXFast.scaledDotProductAttention` — `const char* mask_mode` ∈ {`""`, `"causal"`, `"array"`}, plus
**two** nullable arrays (`mask_arr`, `sinks`; `sinks` is the easy one to miss).

Ops added: `concatenate` (KV append), `split` (fused QKV projection), `softmaxAxis` (the composed
cross-check path), `triu`/`tril` (causal mask), `where`, comparisons, `expandDims`, `flatten`.

### 8. `QuantizedLinear` (M4)

All three calls combine every awkward parameter shape at once; the exact signatures matter, so they
are tabulated rather than summarized (`native/install/include/mlx/c/ops.h:359-369`, `:801-808`,
`:809-819`):

| Method | `optInt` | `optDtype` | `cstr` | `nullableHandle` | Result |
| --- | --- | --- | --- | --- | --- |
| `quantize(w, gs, bits, mode)` | `group_size`, `bits` | — | `mode` | `global_scale` | **`vectorOutOp`** → `(w_q, scales, biases)` |
| `dequantize(w, scales, biases, …)` | `group_size`, `bits` | `dtype` | `mode` | `biases`, `global_scale` | single array |
| `quantizedMatmul(x, w, scales, biases, transpose, …)` | `group_size`, `bits` | — | `mode` | `biases` | single array |

`dequantize` is **not** "quantize plus `optDtype`" — it additionally has two nullable arrays. And
`quantizedMatmul`, though hand-rolled as the hot path rather than routed through a helper, still
needs `cstr`, both `optInt`s and `nullableHandle`; it is not a plain `binaryOp` with extra scalars.
All three resolve their target through `scopeOf(...)` over every array operand including the
nullables (§2's table).

### 9. Documentation

* `req/project-outline.md` Phase 4: mark delivered, with the same explicit reconciliation Phase 3
  used for "thread-safe" rather than a bare checkmark — here, that the outline's `nn` layer sits on a
  public op surface in `core`, and that autograd shipped although the outline lists no autograd
  deliverable under Phase 4.
* The outline's architecture diagram (`project-outline.md:13`) still advertises `Stream`, `Device`
  and `Autograd` as `core` types. `Autograd` becomes real as `MLXGrad`; `Stream`/`Device` still do
  not exist. Amend the line rather than leaving it aspirational.
* `MLX.java`'s class javadoc cites `initial-plan.md §7` and `phase3-plan.md`; add this document, and
  give each split class from §1 a javadoc pointing back at the index in `MLX`.
* `jmlx-examples/HelloMLX`: add a `Linear` forward pass inside a child scope, so the two behavioural
  changes in this phase (cross-scope ops, child-scope lifetime) are visible in the demo rather than
  only in tests.

## Testing approach

Style unchanged from `initial-plan.md` and `phase3-plan.md`: `try (MLXScope scope = new MLXScope())`,
hand-computed goldens, `EPS = 1e-5f` with a looser inline tolerance for float-heavy ops, and
**element values asserted, not just shapes**. Layer tests go in a new `se.alipsa.jmlx.nn` test
package; op tests extend `MLXNumericTest`.

| What | Why it earns its place |
| --- | --- |
| `innermost` returns the child for `(parent, child)` in **both** argument orders | argument-order-independent lifetime is the entire point of replacing "first operand's scope" |
| a binary op across two **unrelated roots** throws `IllegalArgumentException` | proves the relaxation did not become "anything goes" |
| **SDPA with `q` in a child scope and `k`/`v` in the parent allocates into the child; with `k`/`v` in an *unrelated* scope it throws** | the multi-operand rule from §2. A body that resolved its target from `q` alone passes the first half and silently passes the second too — so the rejection case is the one that discriminates |
| **`layerNorm(x, weight, bias)` with a null `weight` still picks the innermost of the non-null operands** | proves `scopeOf` skips nulls rather than NPEing or rejecting — a legitimate layer configuration |
| `keep(a)` on an array in a **root** scope throws `IllegalStateException`, not `NullPointerException` | an unchecked `parent()` deref is the obvious implementation and gives the wrong exception type |
| build in a child, `keep`, close the child, then `eval` + values vs. a hand-computed golden | asserts the `array.h`/`private/array.h` reasoning rather than inferring it |
| **parent cascade, then child access throws `IllegalStateException`** | the use-after-free the cascade *introduces*. Without `holder.isClosed()` in `ensureOpen()` this reads freed memory instead of throwing, and nothing else in the suite notices |
| `MLXMemoryLeakTest` with a per-iteration **child** scope, **including both a `Linear.forward` and a `ModuleGrad.apply` in the loop body** | the M0b/M1 test that matters: a missing cascade or missing `removeChild` shows as `activeMemoryBytes()` growth. It is also the **only** detection for both leaks §2 and §5 name — a creation op called with `scope()` instead of `x.scope()`, and a single-operand op on a weight inside `forward`. Neither has a compile-time signal, and the `ModuleGrad.apply` additionally covers the rebind/restore path §6 adds. The `Linear.forward` in the body is what makes the second leak reachable at all |
| `matmul(f32, i32)` **succeeds**; `matmul(i32, i32)` throws `IllegalArgumentException` | `phase3-plan.md:755` deferred this row as "unreachable today". `astype` makes it reachable; it stops being deferred in M0c |
| `toFloatArray()` on a float16 array returns correct values | Decision 6's astype path. The rejected alternative (per-dtype readers) is untestable for bf16 at all |
| `parameters()` order is insertion order across nested modules | the silent-wrong-gradient bug that reflection-based registration would cause |
| `Module.update` with an unknown path throws | silently ignoring it is how a Phase 5 weight-name typo becomes a garbage model |
| **`Linear.parameters()` returns `W` with the checkpoint's `[out, in]` shape, not `W.T`** | pins the layout contract Phase 5 loads against. A `W.T` registration passes every forward-pass test and only fails when real weights land |
| **`update` on a parent module holding a **nested** `Linear`, then `forward`, uses the new weights** | that `update` walks the subtree rather than notifying only its receiver. Trivially true for a non-caching `Linear`, which is the point — it pins the traversal contract before a Phase 5 layer relies on it. **The `Linear` must be nested, not the receiver** |
| `hoist(a, a.scope())` returns `a` itself, and a loop doing it 1000× shows no memory growth | the reflexive case. A copy-returning implementation passes an equality-of-values assertion and leaks — so assert identity, and pair it with the leak loop |
| `MLXGrad.Fn` reused across two step scopes, each `apply` landing grads in its own | proves `target` moved to `apply`; a construction-bound target throws on the second iteration |
| **one `ModuleGrad` reused across two iterations with *different* `inputs`, grads differing accordingly** | proves the batch moved to `apply` too. A capture-based implementation returns iteration 1's grads twice — same shapes, no exception |
| **`ModuleGrad` grads are w.r.t. parameters only: perturbing an input changes the loss but produces no grad entry for it** | the `argnums = 0..paramCount-1` convention. Differentiating w.r.t. the batch as well yields extra entries or a shifted zip — silently wrong gradients with correct shapes |
| **`update` the model, then `apply`: grads reflect the *new* weights** | the stale-values capture. `of()` must take the key order only, not a values snapshot. The different-inputs row above **passes against the broken version**, so this row is the only one that discriminates |
| **grads through a `Module` whose `forward` reads its own fields match hand-computed values, and do *not* match when rebinding is disabled** | §6's rebinding mechanism. The negative half is what proves rebinding is load-bearing rather than incidental |
| **after `apply` returns and the step scope closes, `forward` still works — including when `loss` threw** | the restore path. Without the `finally`, the model's parameters point into a closed scope and this reads freed memory rather than throwing |
| `body` returning a rank-1 element 0 throws a Java exception naming the rank | `transforms.h:66-68` differentiates `values[0]` only and requires it rank-0. The most common user error in the API; without the guard it surfaces as an mlx-c string |
| grad of `sum(x*x)` at `[1,2,3]` is `[2,4,6]` | the whole closure round-trip in one assertion |
| **a Java exception thrown inside a grad closure surfaces as that exception, JVM alive** | asserts step 3 of §6's protocol was implemented rather than dropped. Assert the exception **type** — without step 3 an `MLXException` is thrown too, so a test asserting only "does not succeed" would pass against the broken version |
| SDPA cross-checked against the composed `softmax(qk/√d)v` path, same golden | one golden validates two independent code paths; catches a wrong `scale` or a transposed head axis, which shape assertions cannot |
| `dequantize(quantize(w)) ≈ w` within quantization error | the only tractable `QuantizedLinear` assertion; a direct golden would encode mlx's packing layout |

**Named caveat.** The layer tests use deliberately tiny fixtures (1 batch, 2 heads, dim 4, 3 tokens)
so the goldens stay hand-computable per Decision 8. That means they prove *the right symbol is called
with the right arguments*, not that attention is numerically robust at model scale. Do not add a
large-fixture test to compensate — it would have no computable golden and would degenerate into
asserting shapes.

## Verification

0. **Before writing any of §2 or §6: one scratch class, compiled *and run* against the real
   bindings.** Ordered first deliberately. `phase3-plan.md:838-843` records three defects that
   reached committed drafts because Java was written but never compiled, and 0b below cannot be
   settled by reading at all.
   * **0a** — hoist past a child close: build `y = multiply(exp(x), x)` in a child, `keep(y)`, close
     the child, `eval(y)` + `toFloatArray(y)` vs. a hand-computed golden; `mlx_get_active_memory()`
     back to baseline after the *parent* closes.
   * **0b** — three stubs. One returning `1` → expect `MLXException`, **JVM survives**. One
     *throwing* → expect the **JVM to die** (forked JVM; `jmlx-ffi`'s `loaderGuardTest` task is the
     template). One asserting `Thread.currentThread()` equals the caller, which is what licenses
     `Arena.ofConfined`. The JVM-dies half must be *observed*, not reasoned about: it is what makes
     §6's `catch (Throwable)` load-bearing rather than defensive.
   * **0c** — `mlx_value_and_grad` on `f(x) = sum(x*x)`, `x = [1,2,3]` → `[2,4,6]`. Exercises the
     full round-trip including `mlx_vector_array_set_data` on the `out` param.
   * **0d** — `mlx_quantize` on a `[1,64]` float32, `group_size=32`, `bits=4` → 3-element vector,
     `w_q.dtype() == UINT32`. Four unknowns in one call: by-value `optInt`, `const char*` mode,
     nullable `global_scale`, and `vectorOutOp`'s two allocators.
   * **0e** — `mlx_fast_rope` with `base` present and absent, confirming the `mlx_optional_float`
     encoding and the plain-`int` `offset`.
   * **0f** — **the rebinding mechanism, which no citation in this repo can settle.** Build a
     two-parameter toy `Module` whose `forward` reads its own fields; differentiate a rank-0 loss
     through `ModuleGrad`; assert the grads match hand-computed values. Then assert the *negative*:
     with rebinding disabled, the same call does **not** produce those values — because if it does,
     rebinding is not the mechanism and §6's design is wrong. Also assert that after `apply` returns
     **and the step scope closes**, `model.forward` still works — the restore path — and that it
     still works when `loss` throws.

     The negative half is the point. A probe that only shows "grads look right with rebinding on"
     cannot distinguish a correct mechanism from one where mlx happens to trace through the
     registered arrays anyway.
1. `./gradlew build` — Spotless, Checkstyle, and the forked `loaderGuardTest` included.
2. `scripts/regen-bindings.sh` then `git diff --exit-code jmlx-ffi/src/main/generated/java` — must be
   clean. **Proves this document's load-bearing premise:** Phase 4 required no binding changes,
   including for bfloat16.
3. **M0a is pure motion — checked as "no method body changed", not as "renames only".**
   `git diff -M` reports a rename only when a file is deleted and a similar one added, and `MLX.java`
   *survives* M0a (§1 keeps creation, `astype`, `eval`, `hoist`, `defaultDevice`/`defaultStream` in
   it). Git therefore sees `MLX.java` modified plus six new files, plus `MLXArray.java`
   (`MLX.checked` → `NativeOps.checked`) and the eight caller files §1 already costs out — so a
   renames-only assertion **fails against a correct commit** and would get deleted as broken. Use
   `git diff -C -C --find-copies-harder --stat` to surface the copy detection, then confirm by eye
   over `git show <sha>` that no method body changed — only package/class/import lines and
   visibility. Recorded this way because the naive form was in an earlier draft of this document.
4. `./gradlew :jmlx-core:test` — all suites, including the new `MLXScopeTest` cascade rows.
5. `./gradlew :jmlx-core:test --tests '*MLXMemoryLeakTest*'` — no growth with a child scope created
   and closed per iteration.
6. **`grep -rn 'mlx_array_new(' jmlx-core/src/main` — every call site's allocator is one of three
   allowed forms**, and the third is not optional:
   * `scopeOf(...)`'s result — every op body;
   * an explicit `MLXScope` parameter — creation ops, `hoist`, `vectorOutOp`;
   * **a confined `Arena`, where the handle is freed by a local `finally` rather than owned by a
     scope** — the read-back path. This is `MLXArray.java:84` today (`mlx_array_new(tmp)`, with the
     comment at `:77-83` explaining why), and §4 adds a second such site for the astype step.

   **The third case exists because omitting it makes this item fail against a correct M0c**, on two
   sites that are already correct on `main` — the same defect Verification #3 was rewritten for, and
   the same allowed/forbidden split #7 draws for `mlx_vector_array_new`. State the two greps in that
   shape: *allocator is a scope ⇒ the handle outlives the call; allocator is a confined `Arena` ⇒ a
   local `finally` frees it.* Anything else is the bug.

   So stated, this subsumes the weaker `grep -rn 'scope != .*\.scope()'` (which only proves one
   spelling of the old same-scope rejection is gone) and catches **one** defect §2 names: a
   multi-operand op resolving its target from one operand instead of all of them.

   **It does not catch the single-operand-op-inside-`forward` leak, and must not claim to.** A
   correct `unaryOp` body allocates via `scopeOf(op, a)` — allowed form #1 — whether `a` is an
   activation or a weight; the defect lives at the **`nn` call site** (`transpose(W)` inside
   `forward`), which this grep never looks at. §2's fifth sub-hazard, §5, and the Testing table all
   say `MLXMemoryLeakTest` with a child-scope loop is the *only* detection for it. An over-claim here
   would contradict three sections and is the same shape as the two entries Failure modes records.
6a. **`grep -rn 'se\.alipsa\.jmlx\.nn' jmlx-core/src/main/java/se/alipsa/jmlx/{core,memory}` — expect
   zero matches.** The one-way dependency Context ¶3 asserts and §6 restates. `nn` → `core` is the
   only legal direction; a single import the other way is the cycle that blocks the
   `module-info.java` in the last Open question. Cheap, and the round-4 defect it catches was
   introduced by a fix, not by the original design.
7. `grep -rn 'mlx_vector_array_new' jmlx-core/src/main` — every call site's allocator confirmed **by
   eye** to be a confined `Arena`, never an `MLXScope`. Extends `phase3-plan.md` Verification #12 to
   `vectorInOp` and `vectorOutOp`. Worth one manual look because no test can catch it: it is UB, not
   a failed assertion.
8. `./gradlew :jmlx-core:test --tests '*MLXGpuVerificationTest*'` — kernels still dispatch to GPU.
9. `./gradlew :jmlx-examples:run` — HelloMLX prints correct values for the `Linear` forward pass in a
   child scope.
10. `req/project-outline.md` Phase 4 marked delivered, with the autograd and package-layout
    reconciliations of §9 present rather than a bare status line.

## Failure modes this document kept producing

Six review rounds raised twenty-three correctness defects and seventeen further suggestions against
this document. **Several are `phase3-plan.md`'s own named failure modes, re-made** — and three
recurred *inside the fix for the previous instance*, which is the argument for keeping this section
alive rather than treating it as history.

**The recurrences are the most useful entries here.** Round 2 found that:

* **Verification #6, written in round 1 to replace a weak grep, had the exact defect Verification #3
  had just been rewritten for**: it fails against a correct implementation. It required every
  `mlx_array_new(` allocator to come from `scopeOf` or a scope parameter — which reports
  `MLXArray.java:84` (a correct confined-`Arena` site on `main` today) as a violation, plus the second
  such site §4 adds. **Writing a check and immediately writing another check of the same kind is not
  the same as applying the lesson**; the fix has to be run against a hypothetical *correct* tree, one
  item at a time.
* **`vectorOutOp`'s explicit-target justification was invented in the same round that added the
  "right conclusion, wrong evidence" entry below.** The claim "its inputs may be none" is false —
  both `mlx_split` and `mlx_quantize` take array inputs. The conclusion (keep the parameter) stands
  on legibility grounds. This is failure mode #1 re-made one section away from where it was recorded.

Round 3 found a third, in the very item round 2 had just rewritten:

* **Verification #6's rewrite over-claimed its own coverage.** Round 2 fixed it to allow three
  allocator forms, then closed by asserting it catches "a single-operand op inside `forward`
  allocating into the model scope" — which it cannot, because a correct `unaryOp` uses `scopeOf` and
  the defect lives at the `nn` call site the grep never reads. The claim contradicted three other
  sections that all name `MLXMemoryLeakTest` as the only detection. **A verification item is not
  finished when it stops producing false positives; it is finished when its stated coverage matches
  what it mechanically inspects.**

Round 4 found a fourth, in the overload round 3 had added to fix the previous one:

* **The `Module`-aware `valueAndGrad` — added in round 3 to give `freeze()` a caller — inverted the
  package layering the document's own Context ¶3 establishes.** Putting it on `core.MLXGrad` made
  `core` import `nn` while `nn` imports `core`: a cycle, and one that would block the
  `module-info.java` in the last Open question. **A fix that reaches for the nearest class is how a
  layering rule stated in the Context gets violated by §6 without anyone editing the Context.** The
  test is mechanical and was not applied: for every new signature, name the package of every type in
  it and check the arrows. Relocating to `nn.ModuleGrad` costs nothing, because the wrapper needs no
  package-private access at all.
* And in the same overload, **the per-iteration batch had no path to the body** — reproducing round
  2's finding #5 (`Fn.target` bound at construction) one overload over, with the same consequence:
  one confined `Arena` and one upcall stub per batch. **When a finding is "per-iteration data must go
  on the per-iteration method", check every method that will be called per iteration, not just the
  one the finding named.**

Round 5 found the deepest one, and it was latent from the moment `ModuleGrad` was first sketched:

* **`ModuleGrad` as specified computed gradients disconnected from the model.** `Linear.forward`
  reads its own fields; the `params` handed to `loss` are the arrays `MLXGrad` reconstructs inside
  the upcall. Different objects — so the natural loss traces over the registered weights while mlx
  differentiates w.r.t. the primals, and the grads are meaningless. **No signal: correct shapes, no
  exception.** The fix (rebind around the body, restore in a `finally`) only works *because* of the
  round-3 `onParametersUpdated()` fix, since `Linear`'s cached `W.T` would otherwise keep the trace
  pointed at the original array. **Two fixes from different rounds that are each incomplete without
  the other is a sign the interaction was never modelled** — and neither round noticed.

  Two smaller ones came with it, both from not reading the contract of the thing being wrapped:
  `values[0]` is the only differentiated output and must be rank-0 (`transforms.h:66-68`), which was
  nowhere in the doc; and `of()` capturing parameter *values* rather than key order would
  differentiate against stale weights after the first optimizer step — `Linear`'s stale-view bug one
  class over, in a document that had already recorded that bug once.

Round 6 found that round 5's own fix collided with round 3's, and that the collision was the third
sign of the same thing:

* **The rebinding mechanism (round 5) and the cached-view mitigation (round 3) could not both
  stand.** Firing the change hook on rebind's *restore* leaks one array per view per step into the
  model scope; suppressing it dangles the view into a closed step scope. Round 5 asserted the first
  half as a *requirement* ("`rebind` must fire `onParametersUpdated()`") without tracing the restore
  call. **Three consecutive rounds now — round 3 vs. round 5, round 5's own two halves, and this —
  where two locally-correct specifications compose into a defect.** The resolution was to withdraw
  the earlier mitigation entirely rather than patch the interaction, which is usually the signal that
  the mitigation was solving the wrong problem: `Linear` simply uses the explicit-target overload
  §2 already specified, at a cost of one lazy view per forward.

Round 3's other four findings share one shape — **a change made in one place, not propagated to the
places that quote it.** `scopeOf` gained an `op` parameter that neither call example used; `freeze()`
was given a caller that a later signature change had removed; `onParametersUpdated()` was specified
without saying which modules receive it, so the obvious implementation reintroduced the bug it was
added to fix one level of nesting down; "Four sub-hazards" kept its count after a fifth was added.
None is subtle. All four are failure mode #2 again, and all four would have been caught by
**re-reading this document's own code blocks after each edit, not only the prose around them** — the
blocks are what a reader copies from.

1. **A right conclusion carried by wrong evidence** — `phase3-plan.md:826-830`'s failure mode #2.
   The `set_data`-vs-`new_data` recommendation was correct, but justified by a claim that `res.ctx` is
   non-null. It is null (`private/vector.h:12-14`). A reader following this document's own
   "the javadoc is the evidence base" methodology would have checked, found null, and "fixed" the call
   back to the broken form. **The wrong reason is now recorded next to the right one** rather than
   quietly replaced, in Research findings.
2. **A guard stated at one call site instead of at the helper** — the shape of
   `phase3-plan.md:821-826`'s failure mode #1. §2 originally rewrote `binaryOp` alone, leaving the
   five-operand SDPA, four-operand `quantizedMatmul` and three-operand `layerNorm` with no rule at
   all. Fixed by making `scopeOf(String, MLXArray...)` the invariant. **The lesson generalizes: when a rule is
   introduced by editing one function, count the other functions that need it before writing it down.**
3. **A verification step that cannot pass.** Verification #3 asserted `git diff -M` shows renames
   only, which is impossible for a split where the original file survives. A verification that fails
   against a *correct* commit gets deleted as broken, taking the real check with it. **Every
   Verification item must be run mentally against a hypothetical correct implementation before it
   ships**, not only against a broken one.
4. **Three claims asserted from memory rather than checked**, all falsified by one grep each:
   `mlx_triu` producing `BOOL` (it is dtype-preserving, `wheel/.../ops.h:130`);
   `mlx_closure_value_and_grad_apply` having one out-param (it has two, `closure.h:89-93`);
   `dequantize` differing from `quantize` only by `optDtype` (it has two extra nullable arrays,
   `ops.h:359-369`). Each was a header read away. This is failure mode #3 in its cheapest form.
5. **An API whose two stated constraints could not both hold.** `MLXGrad.Fn` bound `target` at
   construction while requiring it to be the per-iteration step scope — so a hoisted `Fn` had a
   closed target by iteration two, and a per-step `Fn` churned exactly the upcall stubs §6 rejects the
   `ERROR_HANDLER_ARENA` pattern to avoid. Neither half was wrong in isolation. **Check lifetime
   constraints against each other, not only against the native API.**
6. **A fix that broke a hook two sections away.** "Store `W.T` as `Linear`'s parameter" solved the
   §5 leak and silently changed what `parameters()` means — the surface Phase 5's safetensors loader
   writes through. Optimizing a hot path is where a documented contract elsewhere gets violated
   without anyone editing that contract.
7. **A concurrency claim with no memory model.** `Holder.isClosed()` was specified without saying
   whether it is synchronized — on a field read that `MLXArray.ensureOpen()` performs several times
   per op, and which the `Cleaner` thread writes. Unsynchronized is a data race; synchronized is a
   monitor on the hot path. **`volatile` was the only correct answer and had to be stated, not left
   to the implementer.** Same for the ABBA ordering between the cascade and `removeChild`.

## Open questions

* **Is `mlx_array_free` safe from the `Cleaner` thread?** Open since `initial-plan.md:554-557`,
  restated at `phase3-plan.md:846-849`, still commented at `MLXScope.java:69-74`. **Phase 4 makes it
  worse, not merely more frequent:** a child `MLXScope` can now be collected while its parent is
  alive, so the backstop can fire on a live scope tree rather than only at shutdown. The fallback —
  enqueue onto the owning thread — is unchanged. Decide whether §2 is the commit that finally probes
  it.
* **Does mlx retain the `value_and_grad` closure past `apply`?** Determines whether `Fn.close()` may
  close the stub arena at all, or whether `Fn` must become process-lifetime. Probe: apply twice,
  close, apply again → expect a clean `MLXException`, not a crash.
* **Optimizer step interaction with `hoist`.** Updated weights are produced in the *step* scope and
  must be hoisted into the model scope, replacing the old ones — so `Module.update` must free the
  previous handles or the model scope grows by one full parameter set per step. Named now as a
  Phase 5 entry condition rather than discovered then.
* **Does `MLXFast` warrant its own package?** It is the only group carrying nullable-array,
  optional-struct and `const char*` params simultaneously. Kept in `core` for now; revisit if Phase 5
  pulls in more of `fast.h`.
* **`MLXArray.scope()` becoming public is a permanent widening for an internal need**, the same trade
  `phase3-plan.md:322-345` made for `checkAccess()`. Two such methods is the point at which a
  `module-info.java` with `exports ... to ...` starts to look cheaper than the alternative. Revisit if
  a third appears.
