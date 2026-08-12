# jmlx Phase 3 — Tensor Operations & Lazy Evaluation Engine

## Context

`req/project-outline.md` describes Phase 3 in four bullets. `req/initial-plan.md` covered Phases 1+2
plus a thin op slice and is merged (`dd4fddf`). This document covers the rest of Phase 3.

**It addresses a different kind of risk from `req/initial-plan.md`, and is long for a different
reason.** `initial-plan.md` ran 568 lines because v0.1's unknowns were *native*: whether jextract
could consume `mlx/c/mlx.h` at all, whether mlx-c's by-value structs survive the FFM boundary,
whether `mlx.metallib` could be found at runtime. Length there bought **de-risking** — probes and
fallbacks for things that might not work.

This document is longer still, and none of it is de-risking. It is **evidence**: quoted native source
at a pinned SHA, measured behaviour, and exact generated signatures. That is the whole cost centre —
two C++ quotations and §4's pseudocode account for roughly a hundred lines between them. An earlier
draft of this paragraph claimed to be "an order of magnitude shorter"; it was, at 380 lines, and
seven review rounds of adding evidence made it false. The claim is dropped rather than defended.

What the original framing was *for* still holds and is worth stating directly: **the op-adding
pattern from `initial-plan.md` §7 is not restated here.** It works, it is established, and repeating
it eleven times would add length without adding evidence.

Phase 3 has no native unknowns:

**Every symbol Phase 3 needs is already in the committed jextract output.** Verified against
`jmlx-ffi/src/main/generated/java/se/alipsa/jmlx/ffi/mlx_h.java`:

```
$ for fn in mlx_log mlx_sin mlx_cos mlx_broadcast_to mlx_squeeze mlx_squeeze_axes \
            mlx_transpose_axes mlx_slice mlx_inner mlx_outer mlx_eval \
            mlx_vector_array_new_data mlx_vector_array_free; do
    printf '%-26s %s\n' "$fn" "$(grep -c "findOrThrow(\"$fn\")" .../mlx_h.java)"; done
mlx_log                    1
mlx_sin                    1
mlx_cos                    1
mlx_broadcast_to           1
mlx_squeeze                1
mlx_squeeze_axes           1
mlx_transpose_axes         1
mlx_slice                  1
mlx_inner                  1
mlx_outer                  1
mlx_eval                   1
mlx_vector_array_new_data  1
mlx_vector_array_free      1
```

This follows from `initial-plan.md` §5: `scripts/regen-bindings.sh` derives its include list by
`grep`-ing the originating header path (`*/mlx/c/*`) out of a `--dump-includes` pass, rather than
authoring it. Every mlx-c symbol is therefore already bound. **No `regen-bindings.sh` run, no
include-list edit, no new `-l` flag.** Verification #2 asserts this.

Two exceptions worth knowing before starting:

* **`mlx_dot` does not exist.** mlx-c offers `mlx_inner`, `mlx_outer` and `mlx_tensordot`. NumPy's
  `dot` has rank-dependent semantics (vector dot for 1-D, matmul for 2-D, tensor contraction above);
  inventing a `MLX.dot` that reproduces them would be a Java-side reimplementation with no native
  counterpart to defer to. The outline's "vector dot products, outer products" is satisfied by
  exposing mlx-c's own names — 1-D `inner` *is* the vector dot product.
* **`mlx_eval` takes an `mlx_vector_array`** (`native/install/include/mlx/c/transforms.h:43`), not a
  single `mlx_array`. That is a container type this codebase has never touched. See §4.

**The risk this plan actually addresses is that two Java-side guards are stricter than MLX itself,
and Phase 4 cannot be built on them.** `MLX.requireSameShape` (`MLX.java:130`, applied to
`add`/`subtract`/`multiply`/`divide`) rejects broadcastable operands, and `matmul` (`MLX.java:123`) is
rank-2-only. Consequences:

* Phase 3 as written would ship `broadcast_to` as an explicit op *next to* elementwise ops that reject
  broadcastable operands — internally incoherent.
* Phase 4's `Linear.forward` is `x @ W.T + b`, with `b` shaped `[out]` against a `[batch, out]`
  result. That `add` throws today.
* Phase 4's `MultiHeadAttention` needs rank-3/4 batched matmul. That throws today.

These are public method contracts. Changing them after the ops ship is a breaking change; changing
them now is free. **That, not "how to bind `sin`", is what this document is for.** The op-adding
pattern established in `initial-plan.md` §7 works and is not restated here.

**Intended outcome:** the outline's Phase 3 op surface, with operand semantics that match MLX rather
than a stricter subset of it, so Phase 4 is additive rather than a renegotiation.

## Decisions taken

1. **Broadcast-compatible guards replace `requireSameShape`.** The check stays Java-side rather than
   being delegated to native, so errors name the operands and their shapes. See §2.
2. **`matmul` relaxes to rank ≥ 1**, with mlx's own rank-1 promotion and batch-dim broadcasting.
   Rank-0 is still rejected, because mlx rejects it. Its dtype guard is `DType.isInexact()` on both
   operands — mirroring native's rule, not the narrower `== FLOAT32`.
3. **`eval` becomes a single `mlx_eval` over an `mlx_vector_array`**, replacing the per-array
   `mlx_array_eval` loop. This is what the outline's "`eval(MLXArray... arrays)` dispatcher" means.
   See §4, including the peak-memory tradeoff this incurs.
4. **Thread confinement is asserted centrally in `MLXArray.ensureOpen()`**, closing a pre-existing
   hole rather than only the one §4 would otherwise open. See §1.
5. **Test goldens stay hand-computed**, with `java.lang.Math.*` for transcendentals. No generated
   fixtures and no Python dependency in the test story: the claim under test is "the binding calls the
   right symbol with the right arguments", not "MLX's libm is accurate".
6. **`MLX` remains the sole public facade.** Two new *private* helpers (`unaryOp`, `shapeOp`) mirror
   the existing `binaryOp`, so the pattern exists once rather than fourteen times. The public surface
   stays flat.
7. **No scalar overloads.** `MLX.array(scope, new float[] {2f}, new int[] {})` already builds a rank-0
   array, and with Decision 1 in place `multiply(a, thatScalar)` broadcasts. Revisit in Phase 4 if the
   ergonomics bite.
8. **`slice` gets two Java guards: length agreement with `a.ndim()`, and a zero-stride rejection.**
   Negative indices and negative strides are *supported by native* and must not be rejected; a zero
   stride is *undefined behaviour* in native and must be. See §3.

## Research findings that shaped this plan

Read from mlx-c at `native/scratch/mlx-c` (`fba4470`) and from mlx C++ upstream `v0.31.2` — the wheel
version pinned by `initial-plan.md` Decision 9.

**The mlx-c citations below are verifiable from this repository; the mlx C++ ones are not.** The
wheel ships headers and a binary only, with no vendored `.cpp`. Since the C++ citations are the
evidentiary base for every guard in §2 — broadcast rules, matmul promotion order, `inner`'s rank-0
short-circuit, `outer`'s cast — they are pinned to the immutable tag SHA rather than the tag name,
and the four load-bearing functions are quoted in full below so a reader can check the reasoning
without network access.

Permalink base (`v0.31.2` = `68cf2fddd8de5edd8ab3d926391772b2e2cedad8`):

```
https://github.com/ml-explore/mlx/blob/68cf2fddd8de5edd8ab3d926391772b2e2cedad8/mlx/<file>#L<line>
```

Where this document reports *measured* behaviour rather than quoted source (the zero-stride table in
§3), it was produced against the matching Python package in a throwaway venv, which ships the
extension module the `mlx-metal` wheel does not:

```
python3 -m venv /tmp/mlxtest && /tmp/mlxtest/bin/pip install 'mlx==0.31.2'
```

**MLX's broadcasting is a literal NumPy implementation.** `broadcast_shapes`, upstream
`mlx/utils.cpp:136-167`, right-aligns the shapes and folds:

```cpp
// upstream mlx/utils.cpp:136-167 @ 68cf2fd, quoted in full
Shape broadcast_shapes(const Shape& s1, const Shape& s2) {
  // Use the same broadcasting rules as numpy
  // https://numpy.org/doc/1.20/user/theory.broadcasting.html
  int ndim1 = s1.size();
  int ndim2 = s2.size();
  int ndim = std::max(ndim1, ndim2);
  int diff = std::abs(ndim1 - ndim2);
  const auto& big   = ndim1 > ndim2 ? s1 : s2;
  const auto& small = ndim1 > ndim2 ? s2 : s1;
  Shape out_shape(ndim);
  for (int i = ndim - 1; i >= diff; --i) {
    auto a = big[i];
    auto b = small[i - diff];
    if (b == a) {
      out_shape[i] = a;
    } else if (a == 1 || b == 1) {
      out_shape[i] = a * b;      // 0 if a or b is 0 otherwise max(a, b)
    } else {
      throw std::invalid_argument("[broadcast_shapes] Shapes … cannot be broadcast.");
    }
  }
  for (int i = diff - 1; i >= 0; --i) out_shape[i] = big[i];
  return out_shape;
}
```

Two details are load-bearing and easy to get wrong:

* The per-dimension predicate is `d1 == d2 || d1 == 1 || d2 == 1`. **Writing it as
  `d1 <= 1 || d2 <= 1` accepts `0` against `3`, which native rejects** — a false-accept, surfacing as
  `MLXException` from native instead of `IllegalArgumentException` from Java.
* The result dimension is `a * b`, **not** `max(a, b)`, so `1` against `0` is `0`. This only matters
  if the facade ever predicts a result shape rather than reading it back from native. It currently
  never does; do not start.
* Rank-0 against anything is always compatible — the compare loop simply does not execute.

`add`/`subtract`/`multiply`/`divide` all route through this same code, via
`broadcast_arrays({astype(a, T), astype(b, T)})` (upstream `ops.cpp:2836-2843`).

**Three uses need three separate checks. Sharing one helper produces two false-accepts.** This is the
single most likely implementation error in §2:

| Op | Actual native rule | Evidence |
| --- | --- | --- |
| `add`/`subtract`/`multiply`/`divide` | full NumPy broadcast | upstream `ops.cpp:2836-2843` |
| `inner` | last dims **exactly equal**, not broadcast-compatible. Short-circuits to `multiply` when either operand is rank-0, so the check must be skipped in that case | upstream `ops.cpp:5453-5463` |
| `broadcast_to` | **directional**: requires `broadcast_shapes(a.shape, target) == target`. `broadcast_to(a[3], [1])` is "compatible" but native throws | upstream `ops.cpp:1601-1613` |

```cpp
// upstream mlx/ops.cpp:5453-5463 @ 68cf2fd — inner's two rules, quoted
array inner(const array& a, const array& b, StreamOrDevice s) {
  if (a.ndim() == 0 || b.ndim() == 0) {
    return multiply(a, b, s);                       // rank-0 short-circuit
  }
  if (a.shape(-1) != b.shape(-1)) {                 // EXACT, not broadcast
    throw std::invalid_argument("[inner] a and b must have the same last dimension.");
  }
  return tensordot(a, b, {-1}, {-1}, s);
}
```

**`matmul`'s real rules** (upstream `ops.cpp:3192-3267`), in evaluation order: rank-0 on either side
throws; rank-1 `a` becomes `expand_dims(a, 0)` and rank-1 `b` becomes `expand_dims(b, 1)`; the
inner-dim check runs on the **promoted** shapes; the promoted dtype must be inexact; then batch dims
broadcast over `shape[:-2]` (upstream `primitives.cpp:948-964`). **Implementation consequence: promote
in Java first, then compare inner dims.** Written against raw shapes, `b.shape(-2)` on a rank-1 `b`
indexes out of bounds.

**`matmul` additionally rejects exact dtypes, and the guard for this must mirror that rule rather
than narrow it.** Native's test is `issubdtype(out_type, inexact)` on the *promoted* type — upstream
`ops.cpp:3222-3230`, i.e. float16, bfloat16, float32 and complex64 all pass.

A `dtype() == FLOAT32` guard would be **strictly narrower than native, which is the exact defect this
document exists to remove.** It would have to be relaxed the moment float16 lands in Phase 4's
quantized layers — the same renegotiation Decisions 1 and 2 are spending Phase 3 to avoid. Instead:

* Add `DType.isInexact()` as a predicate on the enum, **as an allowlist field, not
  `return this != INT32;`**. A denylist is the same false-accept shape as the `d1 <= 1 || d2 <= 1`
  trap above: it is correct for today's two constants and silently wrong the moment `INT8`, `UINT32`
  or `BOOL` is added, each of which would report inexact and be accepted by `matmul`. The allowlist
  form is the same size and cannot rot:

  ```java
  FLOAT32(mlx_h.MLX_FLOAT32(), true),
  INT32(mlx_h.MLX_INT32(), false);

  private final boolean inexact;

  public boolean isInexact() {
    return inexact;
  }
  ```

  Named after native's own predicate, `issubdtype(out_type, inexact)`.
* **Check both operands**, not just the receiver. `matmul` has two.

This guard is *currently unreachable*: nothing in the facade constructs an INT32 array, and
`DType.fromNative` rejects every other dtype outright. It is specified anyway because it mirrors
native exactly, and it becomes reachable the moment an `astype` or `int[]` factory lands.

For the record, the absence of a dtype guard on `add`/`multiply`/`inner`/`outer` is **not an
inconsistency**: those ops call `promote_types` and accept int32 happily (upstream `ops.cpp:2836-2843`).
There is no native dtype rule there to mirror. Each guard mirrors its own op's actual rule; that is
the principle, not "every op gets a dtype check".

**`outer` truncates silently — on `a` only.** Quoted in full so the asymmetry of the guard is
evidently deliberate rather than an oversight:

```cpp
// upstream mlx/ops.cpp:5448-5451 @ 68cf2fd
array outer(const array& a, const array& b, StreamOrDevice s) {
  return multiply(
      reshape(a, {static_cast<int>(a.size()), 1}, s), flatten(b, s), s);
}
```

`a` goes through `static_cast<int>(a.size())`: above 2^31 elements the `size_t → int` cast wraps with
**no native error to catch**, so Java must guard `a.size() <= Integer.MAX_VALUE` itself. **`b` goes
through `flatten`, which needs no `int` cast and is therefore safe** — hence the guard is correctly
one-sided.

This guard is **not in the Testing table and cannot be**: triggering it needs a float32 array above
2^31 elements, i.e. 8 GB. It is asserted by inspection only.

**`mlx_vector_array` ownership — the highest-risk question in this plan, and the answer is that it is
safe.** The fear was that `mlx_vector_array_free` would free the contained arrays, double-freeing
handles that `MLXScope` owns. It does not. Chain of evidence:

* `native/scratch/mlx-c/mlx/c/vector.cpp:41-54` does
  `mlx_vector_array_get_(vec).push_back(mlx_array_get_(data[i]))`.
* `mlx_array_get_` returns `mlx::core::array&` (`native/scratch/mlx-c/mlx/c/private/array.h:42-47`),
  so `push_back` **copy-constructs** into the vector.
* That copy constructor is `= default` over a single `std::shared_ptr<ArrayDesc>`
  (`native/scratch/wheel/mlx/include/mlx/array.h:84`, `:523`) — a refcount increment, not a buffer
  copy.
* `mlx_vector_array_free` → `delete static_cast<std::vector<mlx::core::array>*>(d.ctx)`
  (`vector.cpp:31-39`, `private/vector.h:56-60`). It destroys the vector and its element copies
  (refcount decrements) and **never calls `mlx_array_free`**. The scope's handles are separate
  `new mlx::core::array` heap objects (`private/array.h:16-18`).

No aliasing, no double-free. **The `finally` is still load-bearing, for the opposite reason:** not
freeing the vector leaks the `std::vector` *and* pins N `ArrayDesc` refcounts, keeping device buffers
alive past scope close — visible as growth in `mlx_get_active_memory`.

**The vector must come from a confined `Arena`, never from the scope — and this one is heap corruption
rather than a leak.** `MLXScope.allocate` registers every segment it hands out for `mlx_array_free`
(`MLXScope.java:54-62`, `:76-78`), as its class javadoc warns (`MLXScope.java:21-26`). Routing a
vector handle through it would later apply
`delete static_cast<mlx::core::array*>(ctx)` (`private/array.h:49-53`) to a
`std::vector<mlx::core::array>*` — a wrong-type delete — while the vector itself is never properly
freed.

**The thread-confinement hole is pre-existing, and Decision 3 cannot close it incidentally.**
`MLXArray.handle()` calls `ensureOpen()`, which checks `closed` and nothing else
(`MLXArray.java:116-120`). In `binaryOp`, `mlx_array_new(scope)` (`MLX.java:158`) *incidentally*
routes through `MLXScope.allocate` → `checkThread()` (`MLXScope.java:100`), so `a`'s scope is guarded
and the same-scope check at `MLX.java:155-157` extends that guard to `b`. **Today's `eval`
(`MLX.java:214-218`) touches no scope at all**, so `MLX.eval(a)` from a foreign thread already
bypasses the confinement contract entirely, for every array. Decision 3 does not fix this, because
Decision 3 forbids the one thing that would have triggered `checkThread()` by accident.

## Out of scope for Phase 3

* `sqrt`, `negative`, `abs`, `tensordot`, `slice_update`, `async_eval`. All already bound, all four
  lines each. Add when a caller exists.
* Scalar operand overloads (Decision 7).
* Zero-stride `slice` — rejected Java-side rather than supported, because native's behaviour for it
  is undefined (Decision 8, §3). Negative indices and negative strides are *not* excluded; they work.
* dtypes beyond FLOAT32. `initial-plan.md`'s v0.1 constraint stands; `DType.INT32` remains
  read-back-only.
* Splitting `MLX` into multiple facade classes. See Open questions.
* Resolving whether `mlx_array_free` is safe from the `Cleaner` thread. See Open questions.

## Work breakdown

### 1. Confinement — `MLXScope` and `MLXArray`

Do this first: §3 and §4 both depend on it, and it changes the exception thrown on misuse everywhere.

* `MLXScope`: add `public void checkAccess()` exposing the existing private `checkThread()` +
  `ensureOpen()` pair. `MLXArray` is in `se.alipsa.jmlx.core`, `MLXScope` in `se.alipsa.jmlx.memory`,
  and both `owner` and `closed` are private, so this cannot be package-private.

  **Naming the tradeoff, because it cuts against Decision 6.** There is no `module-info.java` anywhere
  in the tree (verified: `find . -name module-info.java -not -path '*/build/*'` returns nothing), so
  `public` here is *permanent public API added purely to serve a cross-package internal need*, in a
  document that elsewhere insists the public surface stays flat. Three options, in order of
  preference:

  1. **`public void checkAccess()`** on `MLXScope`, javadoc'd as internal, delegating to the existing
     private `checkThread()` + `ensureOpen()`. Widens the public surface by one method.
  2. Add `module-info.java` with `exports se.alipsa.jmlx.memory to se.alipsa.jmlx.core;` — the
     properly-scoped answer, but drags the whole project into the module system for one method, and
     the `--enable-native-access` wiring would need revisiting.
  3. `MLXArray` holds its own `Thread owner`, captured in its constructor. No new `MLXScope` API.

  **Take option 1, and accept that it costs one method against Decision 6.** Option 3 looks like the
  minimal answer and is not: `checkAccess()` is `checkThread()` **plus** `ensureOpen()`, and option 3
  can only reproduce the thread half. `MLXScope.closed` is private and cross-package, so an array
  holding its own `Thread` cannot see whether its scope is still open.

  **That half is load-bearing, and dropping it would open a use-after-free next to the hole this
  section closes.** `MLXScope.close()` sets `closed = true` and calls `holder.closeAll()`, which frees
  the handles directly and never touches any `MLXArray` (`MLXScope.java:129-134`, `:64-81`) — so every
  `MLXArray.closed` flag stays `false` after its scope closes. Under option 1, `array.shape()` at that
  point throws `IllegalStateException("MLXScope is closed")`. Under option 3, `ensureOpen()` passes and
  `mlx_h.mlx_array_shape(handle)` reads a freed handle: undefined behaviour, not an exception.
  **No existing test covers this path**, so nothing would catch the regression.

  Option 2 remains the right cleanup if the project ever modularizes; revisit then.
* `MLXArray.ensureOpen()`: call `scope.checkAccess()` after the array's own `closed` check. **One
  added call per handle read closes both the foreign-thread hole and the closed-scope hole for
  `eval`, `exp`, `sum`, `transpose`, `toFloatArray` and all eleven new ops at once**, and demotes
  `binaryOp`'s same-scope check from sole guard to defence-in-depth. The cost is a field read and a
  reference comparison, negligible against a downcall.

  Add a test for the scope-closed path too, since none exists: an `MLXArray` whose scope has been
  closed must throw `IllegalStateException` from `shape()`, not read freed memory.

Two things this must not break, both already correct by construction — assert them in review rather
than assuming:

* `MLXArray.close()` does not call `ensureOpen()`, so the deliberate free-before-`closed = true`
  ordering (`MLXArray.java:104-114`) still holds. A foreign-thread `close()` still throws from
  `scope.free()` and still leaves `closed` false.
* The `Cleaner` backstop calls `mlx_h.mlx_array_free` directly from `Holder.closeAll()`
  (`MLXScope.java:76-78`), bypassing `MLXArray` entirely. Adding a thread assertion to
  `MLXArray.ensureOpen()` therefore cannot deadlock or disable the backstop.

### 2. Guards — `MLX.java`

Three **separate** private helpers. Sharing one is the false-accept trap documented above.

* `requireBroadcastCompatible(a, b, op)` — replaces `requireSameShape`, which has no callers outside
  `MLX.java` (`:97`, `:103`, `:109`, `:115`, `:130`), so the swap is contained. Predicate:
  `d1 == d2 || d1 == 1 || d2 == 1`, right-aligned, shorter rank padded on the left.
* `requireMatmulCompatible(a, b)` — reject rank-0; promote rank-1; compare inner dims on the
  **promoted** shapes; batch-broadcast `shape[:-2]`; require `dtype().isInexact()` on **both**
  operands (not `== FLOAT32` — see the dtype discussion in Research findings).
* `requireBroadcastableTo(a, targetShape)` — directional, **plus a non-negative check on every
  element of `targetShape`**. This is the only guard whose input is a user-supplied `int[]` rather
  than shapes read back from native, and without the check it has a false-accept of exactly the class
  §2 exists to prevent: `broadcastTo(a[1], new int[] {-1})` satisfies `d1 == d2 || d1 == 1 || d2 == 1`
  via `d1 == 1`, and then `out_shape[i] = a * b` gives `1 * -1 == -1`, which equals the target — so
  the directional check passes too, and the failure surfaces as `MLXException` from native. That
  breaks the contract the Testing table asserts ("proves the Java guard fires before native").

**Javadocs falsified by these decisions — six, not four.** Update all of:

| Location | Current text | Falsified by |
| --- | --- | --- |
| `MLX.java:95`, `:101`, `:107`, `:113` | "…of two same-shaped arrays" | Decision 1 — these can now return a shape matching neither operand |
| `MLX.java:119` | "Matrix product of two **rank-2** arrays" | Decision 2 — rank ≥ 1 |
| `MLX.java:203` | "Reverses every axis; **there is no partial-permutation overload in this slice**" | §3 adds `transpose(a, int[] axes)` |
| `MLX.java:211-212` | "mirrors mlx-c's `mlx_array_eval`" | Decision 3 — it now mirrors `mlx_eval` over a vector |

**`MLXNativeErrorTest` needs no fixture change, and this is worth a comment in the test.** Its
`[3]`/`[2]` pair is still broadcast-*incompatible* under NumPy rules, so
`"[broadcast_shapes] Shapes (3) and (2) cannot be broadcast."` still surfaces and the existing
substring assertion holds. Two facts to pin in a comment, because the broadcast-aware guard invites
exactly the wrong inference:

* **`addUnchecked` is not dead code, and becoming an exact mirror of native is precisely why.** It is
  the only remaining route to the native error handler, and that test is the sole coverage of the
  `exit(-1)` mitigation (`NativeLoader.java:119-128`).
* **The fixture shapes must stay broadcast-INcompatible.** A future edit "modernizing" them to
  something compatible (e.g. `[2,2]` against `[2]`) silently converts the test into a happy path, and
  `assertThrows` fails. The pair recorded as empirically verified at `initial-plan.md:191`
  (`(2,2)`/`(3)`) also still works and additionally exercises the rank-mismatch branch.

### 3. Ops — `MLX.java`

Add two private helpers mirroring `binaryOp` (`MLX.java:148-166`), then **refactor the existing
`exp`/`sum`/`transpose` onto them**, so the pattern exists once rather than fourteen times:

* `unaryOp(name, a, UnaryOp op)` — for the `(res, a, stream)` signature.
* `shapeOp(name, a, int[] param, ShapeOp op)` — for `(res, a, const int*, size_t, stream)`, wrapping
  the confined-`Arena` + `allocateFrom(JAVA_INT, …)` sequence that `reshape` (`MLX.java:184-201`)
  currently inlines.

| Method | Native | Notes |
| --- | --- | --- |
| `log`, `sin`, `cos` | `mlx_log` / `_sin` / `_cos` | `unaryOp` |
| `inner(a, b)` | `mlx_inner` | plain binary signature → `binaryOp`; exact-last-dim guard, skipped for rank-0. The outline's "vector dot product" |
| `outer(a, b)` | `mlx_outer` | `binaryOp`; guard `a.size() <= Integer.MAX_VALUE` — no native error exists to catch |
| `broadcastTo(a, int[] shape)` | `mlx_broadcast_to` | `shapeOp` + directional guard |
| `squeeze(a)` | `mlx_squeeze` | all size-1 axes |
| `squeeze(a, int[] axes)` | `mlx_squeeze_axes` | `shapeOp` |
| `transpose(a, int[] axes)` | `mlx_transpose_axes` | `shapeOp`. Beyond the outline's "transposed matrices", but Phase 4's attention head reshuffle needs it and it is one method plus one test |
| `slice(a, start, stop)` | `mlx_slice` | strides default to all-1s. Guards below |
| `slice(a, start, stop, strides)` | `mlx_slice` | three `const int*` params, so it does not fit `shapeOp`; own confined-`Arena` body |

**`slice`'s guards, specified rather than left implicit.** It takes three user-supplied `int[]`s plus
their independent `size_t` counts (`mlx_h.java:33416`), so the guards need stating — but the answer is
"two", and the reasoning for why it is not more is the useful part.

Quoted in full, **without elisions** — the line that decides the zero-stride case below is one an
abbreviated quote naturally drops.

```cpp
// upstream mlx/ops.cpp:656-696 @ 68cf2fd — normalize_slice, which slice() delegates to
for (int i = 0; i < shape.size(); ++i) {
  // Following numpy docs
  //  Negative i and j are interpreted as n + i and n + j where n is
  //  the number of elements in the corresponding dimension. Negative
  //  k makes stepping go towards smaller indices
  auto n = shape[i];
  auto s = start[i];
  s = s < 0 ? s + n : s;                          // negative START is normalized
  auto e = stop[i];
  e = e < 0 ? e + n : e;                          // negative STOP is normalized

  // Note: -ve strides require start >= stop
  if (strides[i] < 0) {
    has_neg_strides = true;                       // negative STRIDE is the reverse form
    auto st = std::min(s, n - 1);                 // clamped, not thrown
    auto ed = e > -1 ? e : -1;
    start[i] = st;
    ed = ed > st ? st : ed;
    auto str = -strides[i];
    out_shape[i] = (start[i] - ed + str - 1) / str;
  } else {
    auto st = std::max(static_cast<ShapeElem>(0), std::min(s, n));   // clamped, not thrown
    auto ed = std::max(static_cast<ShapeElem>(0), std::min(e, n));
    start[i] = st;
    ed = ed < st ? st : ed;
    out_shape[i] = (ed - start[i] + strides[i] - 1) / strides[i];    // <-- zero stride divides by 0
  }
  // Simplify the stride if it's unused
  if (out_shape[i] == 1) {
    strides[i] = 1;
  }
}
```

So the guard list is two items:

* **`start.length == stop.length == a.ndim()`**, plus `strides.length == a.ndim()` on the 4-arg
  overload. Reject otherwise with `IllegalArgumentException` naming the lengths and `a.ndim()`.
  This **mirrors** native rather than narrowing it — upstream `ops.cpp:757-763` raises
  `"[slice] Invalid number of indices or strides for array with dimension N."` on the same
  condition. It is worth doing Java-side only because the Java error can name which of the three
  arrays disagreed.

* **Reject `strides[i] == 0`** — on the 4-arg overload only. The 3-arg overload synthesizes all-1s
  and cannot reach it, so a check there would be dead code. Note the branch structure above: `0` is
  not `< 0`, so a zero stride takes the `else` branch and reaches
  `out_shape[i] = (ed - start[i] + strides[i] - 1) / strides[i]` with a zero divisor.
  **That is division by zero — C++ undefined behaviour — reachable from a public API method.**
  Measured on mlx 0.31.2, Apple Silicon, to establish what it actually does rather than assume:

  | | zero stride |
  | --- | --- |
  | mlx 0.31.2 aarch64 (`a[::0]`, `a[::0,:]`, `a[1:3:0,:]`) | silently returns shape `(0,)` / `(0,4)`; no error, no crash |
  | NumPy | `ValueError: slice step cannot be zero` |

  It does **not** crash, because aarch64's `sdiv` returns 0 for a zero divisor instead of trapping,
  so `out_shape[i]` becomes `0`. That is a property of the current architecture and codegen, not a
  guarantee the language makes: the expression is UB, and it degrades to a *silently wrong empty
  result* where NumPy raises.

  **That measurement went through mlx's Python binding, and the conclusion is about the C API jmlx
  calls — so close the gap rather than infer across it.** It holds because `mlx_slice` forwards its
  three arrays *verbatim* with no normalization of its own
  (`native/scratch/mlx-c/mlx/c/ops.cpp:3125-3143`):

  ```cpp
  mlx::core::slice(
      mlx_array_get_(a),
      mlx::core::Shape(start,   start   + start_num),
      mlx::core::Shape(stop,    stop    + stop_num),
      mlx::core::Shape(strides, strides + strides_num),
      mlx_stream_get_(s));
  ```

  Both bindings therefore converge on the same `normalize_slice` quoted above. The observed shape of
  `0` is itself corroboration that this code path ran: it is exactly what `sdiv`-returns-zero predicts
  from the quoted division.

  **Why this is not the "stricter than native" defect.** That principle governs native behaviour that
  is *defined* — rejecting a negative stride was wrong precisely because native specifies what a
  negative stride means. Native specifies nothing here. Converting UB into a named
  `IllegalArgumentException` is not narrowing the contract; it is the only way to give the method one.
  Phrase the message after NumPy's ("slice step cannot be zero") rather than as a jmlx restriction.

And explicitly **not**:

* Do **not** reject negative `start`/`stop` — native normalizes them NumPy-style.
* Do **not** reject negative `strides` — that is native's reverse-a-dimension form.
* Do **not** bounds-check `stop[i] <= a.shape()[i]` — native clamps rather than throwing.

### 4. `eval` — `MLX.java`

```
public static void eval(MLXArray... arrays)
  int n = arrays.length;
  if (n == 0) return;

  // Pass 1 CAPTURES; it does not merely touch. handle() is thread-checked per §1,
  // so this both validates every array and yields the segments the copy loop uses --
  // the "cannot throw mid-build" invariant then holds by construction rather than
  // by reading each handle a second time below.
  MemorySegment[] handles = new MemorySegment[n];
  for (int i = 0; i < n; i++) handles[i] = arrays[i].handle();

  try (Arena tmp = Arena.ofConfined()) {
    MemorySegment buf = mlx_array_.allocateArray(n, tmp);
    for i: MemorySegment.copy(handles[i], 0L, buf, i * mlx_array_.sizeof(), mlx_array_.sizeof());
    NativeLoader.clearLastNativeError();
    // NOTE the allocator argument. This is the call the confined-Arena paragraph
    // below is about: passing `scope` here instead of `tmp` is the wrong-type
    // delete. The generated signature is
    //   mlx_vector_array_new_data(SegmentAllocator allocator, MemorySegment data, long size)
    // -- mlx_h.java:5471.
    MemorySegment vec = mlx_h.mlx_vector_array_new_data(tmp, buf, n);
    if (mlx_vector_array_.ctx(vec).address() == 0) throw nativeFailure("mlx_vector_array_new_data");
    try {
      checked(() -> mlx_h.mlx_eval(vec));
    } catch (MLXException e) {
      // Restores the per-array attribution the joint eval loses. MUST sit here,
      // inside the try-with-resources and BEFORE the finally frees vec: it
      // re-evaluates the INDIVIDUAL handles via mlx_array_eval, not the vector.
      throw attributeEvalFailure(e, handles);
    } finally {
      mlx_h.mlx_vector_array_free(vec);
    }
  }

// Re-runs the per-array loop to name the offender; rethrows the original
// unchanged if it cannot reproduce the failure, so this never masks the error
// it is trying to describe.
// REQUIRES a new MLXException(String, Throwable) ctor -- see below.
private static MLXException attributeEvalFailure(MLXException original, MemorySegment[] handles)
  for (int i = 0; i < handles.length; i++) {
    // `h`, not handles[i], inside the lambda: checked() takes an IntSupplier
    // (MLX.java:227), and the loop counter is mutated each iteration, so capturing
    // it directly is "local variables referenced from a lambda expression must
    // be final or effectively final". `i` in the catch body is fine -- string
    // concatenation is not a capture.
    final MemorySegment h = handles[i];
    try { checked(() -> mlx_h.mlx_array_eval(h)); }
    catch (MLXException perArray) { return new MLXException("eval: array[" + i + "] failed", perArray); }
  }
  return original;
```

**`MLXException` needs a second constructor.** It currently declares only `MLXException(String)`
(`MLXException.java:7-9`), so the block above does not compile as written. Add the conventional
cause-carrying form:

```java
/** Creates an exception carrying {@code cause}'s mlx-c failure message as context. */
public MLXException(String message, Throwable cause) {
  super(message, cause);
}
```

Chaining rather than string-concatenating the inner message keeps the per-array exception's stack
trace, which is the part that identifies where the re-run failed.

* `mlx_array_` is `structLayout(C_POINTER)` (`mlx_array_.java:28-30`), and `C_POINTER` resolves to
  `canonicalLayouts().get("void*")` (`mlx_h$shared.java:27-28`), so `sizeof()` is 8 on
  macOS/aarch64 and `sequenceLayout` adds no padding (element size equals alignment). **Use a raw
  `MemorySegment.copy` of `sizeof()` bytes rather than a `ctx` get/set round-trip** — a byte-for-byte
  struct copy stays correct if `mlx_array_` ever gains a field.
* `mlx_vector_array_new_data` is statusless and returns a null-ctx struct on failure
  (`vector.cpp:41-54`), exactly the hazard class `MLX.array` already handles at `MLX.java:79-90`.
  Hence the explicit `mlx_vector_array_.ctx(…).address() == 0` check
  (`mlx_vector_array_.java:69`) and the `clearLastNativeError()` **before** the call, for the reason
  given in `MLX.checked`'s javadoc.
* **`size == 0` returns a non-null ctx** — `vector.cpp:45` heap-allocates an empty `std::vector` — so
  the ctx check will not misfire on an empty varargs call. The early return exists only to skip work.
* **No same-scope check, unlike `binaryOp`.** `eval` allocates no result, so there is no "which scope
  owns the output" question to settle, and evaluating arrays from two scopes on one thread is
  legitimate. §1's per-array thread assertion is the correct guard here, and a same-scope check would
  wrongly reject valid code.
* Factor the buffer construction into a private helper. `mlx_async_eval`
  (`transforms.h:31`) takes the same `mlx_vector_array` and would reuse it verbatim.

**This is a tradeoff, not a pure improvement, and should be documented as one.** One `mlx_eval` is one
tape build plus one synchronization instead of N plus N (upstream `transforms.cpp:330-345`). But it
schedules all N graphs before waiting, so all N sets of intermediates are live simultaneously —
**peak memory can go up** relative to the loop, which let each graph's temporaries be released before
the next began.

**Measure this rather than asserting it.** It is the risk that actually bites in Phase 4, where a
transformer forward pass evals many arrays at once, and Decision 3 is hard to reverse once callers
depend on the joint-eval semantics. Record one number — peak bytes for the loop versus the vector
over the same N-array workload, via `mlx_get_peak_memory` / `mlx_reset_peak_memory`
(`native/install/include/mlx/c/memory.h:35-36`) — and put it in this document. Verification #9. An
informal one-off measurement is enough; the point is that the tradeoff is quantified rather than
hypothetical.

**Measured (Task 4, mlx 0.31.2, Apple Silicon):** workload = N independent `matmul` results (each
built from two freshly-materialized `[M, M]` float32 inputs via `MLX.array`, i.e. `2N` materialized
operands plus `N` lazy matmul outputs), evaluated either via `N` sequential `mlx_array_eval` calls
(the old per-array loop, called directly, bypassing `MLX.eval`) or via one `mlx_eval` over an
`mlx_vector_array` of all `N` results (the new joint form) — `mlx_reset_peak_memory` before each phase,
`mlx_get_peak_memory` after. Two scales tried, 3 trials each, fresh `MLXScope` per phase so the other
phase's arrays are not still resident: `N=8, M=2048` (384 MiB total workload) and `N=32, M=1024` (also
384 MiB total). **Result: identical peak in every trial at both scales — 402,653,184 bytes
(exactly `3 × N × M² × 4`, i.e. exactly the sum of the `2N` materialized operands plus the `N`
evaluated results, with zero measurable overhead attributable to scheduling) for both the loop and the
vector; delta = 0 bytes in all 6 trials.** No difference in either direction was observed for this
workload on this backend — reported honestly rather than assumed, per the instruction above not to
assert the tradeoff without measuring it. This does not falsify the tradeoff in general (a workload
whose ops carry substantial backend-internal scratch beyond their own output buffer, which `matmul` at
this size apparently does not on Metal, is where it would be expected to show), but it is the actual
number this measurement produced, not a hypothetical one.

**Error attribution regresses, and has a free mitigation — take it.** One `mlx_eval` is
all-or-nothing with no index in the message, where the loop could name which array failed. On the
*error path only*, catch the `MLXException` and re-run the per-array `mlx_array_eval` loop to
identify the offending array, then rethrow with its index. This costs nothing on the happy path,
needs no extra native surface, and restores the diagnostic the loop gave for free. If the re-run
fails to reproduce the error, rethrow the original rather than masking it.

**This is in the pseudocode above, not just here.** Two things about it are non-obvious enough to be
worth re-stating: the `catch` must sit inside the try-with-resources and *before* the `finally` frees
`vec`, and the re-run evaluates the individual handles via `mlx_array_eval` — not the vector, which
is exactly what failed.

**Two questions this mitigation raises, both measured rather than assumed** (mlx 0.31.2, Apple
Silicon; venv recipe in Research findings):

*Is an eval-time failure even constructible, or does mlx validate everything at op-build time?* It
validates **shapes** eagerly — `a + b` on `[3]` and `[2]` throws from the `mlx_add` downcall, which is
why `MLXNativeErrorTest` works at all and why every guard in §2 can be a Java mirror. So a shape error
cannot exercise this path. **Allocation-class errors are genuinely deferred**, and are deterministic
and CI-safe because Metal rejects against its device limit without attempting the allocation:

**Build the oversized array with `outer`, not with `MLX.array`.** The obvious Python recipe —
`mx.zeros((2**20, 2**20))` — has no Java counterpart: `MLX.array` enforces
`product(shape) == data.length` (`MLX.java:68-75`), so that shape would need a `float[]` of 2^40
elements, and Java arrays are `int`-indexed with a cap of `Integer.MAX_VALUE` — 512× too small. The
call would throw `IllegalArgumentException` from the Java length check, a different exception from a
different layer than the test needs. Phase 3 also adds no `zeros`/`ones`.

`outer` does it from operands a `float[]` can hold, and stays under its own
`a.size() <= Integer.MAX_VALUE` guard (2^20 ≪ 2^31−1):

```java
float[] v = new float[1 << 20];               // 4 MB -- trivially allocatable
MLXArray a = MLX.array(scope, v, new int[] {1 << 20});
MLXArray bad = MLX.outer(a, a);               // builds shape [2^20, 2^20]; 4 TiB backing
MLX.eval(bad);
// MLXException: [metal::malloc] Attempting to allocate 4398046511104 bytes
//               which is greater than the maximum...
```

Measured end to end in the three-array form the test actually needs — `eval(good, bad, good)` throws
in 0.001 s, the per-array re-run gives `ok / THREW / ok` in under 1 ms each, and both good arrays
remain readable afterwards. The Testing-table row for this mitigation is therefore implementable; it
is **not** in the "asserted by inspection" bucket that `outer`'s overflow guard and `matmul`'s
`isInexact()` guard sit in.

*Is re-entering eval on a graph that just failed safe, or does it hang?* The Named caveat below notes
that mlx's status flags turn a second eval into a `wait()`, which raises the possibility of waiting on
an event that will never signal. Measured with the three-array fixture above: the joint eval throws in
single-digit milliseconds, then the per-array re-run returns `ok / THREW / ok`, each in under a
millisecond. **The failing array re-throws immediately rather than waiting, and the joint failure
leaves its siblings still readable.** No hang.

(Timings here and above are order-of-magnitude, not benchmarks — the point is "milliseconds, and it
returns" rather than any particular number. Independent runs have landed anywhere in 1–50 ms for the
joint eval depending on the fixture used.)

### 5. Documentation

* `req/project-outline.md` Phase 3: mark delivered; record that `mlx_dot` does not exist and that
  `inner`/`outer` cover the outline's dot/outer deliverable.

  **Reconcile the "thread-safe" wording rather than silently marking it done.**
  `project-outline.md:68` says "**Thread-safe** `eval(MLXArray... arrays)` dispatchers". Decision 4
  plus §1 deliver something different: a *thread-confined* `eval` that throws on foreign-thread
  access. That is a defensible reinterpretation — `MLXScope` is confined by construction
  (`initial-plan.md` §6), so a genuinely thread-safe `eval` would contradict the memory model the
  project already committed to — but it is a substitution, not the literal deliverable. Amend that
  outline line to say "thread-confined, enforced" so a later reader does not score it as unmet.
* `MLX.java`'s class javadoc cites `req/initial-plan.md §7`; add this document.
* `jmlx-examples/HelloMLX`: add one broadcast `add` and one multi-array `eval`, so the two behavioural
  changes in this phase are visible in the demo rather than only in tests.

## Testing approach

Style is unchanged from `initial-plan.md`: `try (MLXScope scope = new MLXScope())`, hand-computed
goldens, `EPS = 1e-5f` with a looser inline `1e-3f` for float-heavy ops, and **element values
asserted, not just shapes**. New cases go in `MLXNumericTest` unless noted.

| What | Why it earns its place |
| --- | --- |
| `log`/`sin`/`cos` vs `Math.*`, `1e-5f` | proves the right symbol is called; not a libm accuracy test |
| `add([2,3],[3])`, `add([2,3],[1,3])`, `add([2,2], rank0)` | the headline contract change. Assert result shape **and** values |
| `add([2,3],[4])` throws `IllegalArgumentException` | proves the Java guard fires before native, not after |
| `matmul` all three rank-1 forms; batched `[2,3,4]×[4,5]→[2,3,5]`; rank-0 rejected | the promotion path is where an off-by-one in axis indexing hides. **No row for the `isInexact()` guard: it is unreachable today** — nothing in the facade constructs an INT32 array — so it is asserted by inspection, like `outer`'s overflow guard |
| `inner` 1-D equals hand-computed dot; `outer` shape and values | |
| `broadcastTo` success **and** `[3]→[1]` rejected with `IllegalArgumentException` | proves the directional check exists rather than the symmetric one |
| `broadcastTo(a[1], new int[] {-1})` rejected with `IllegalArgumentException` | the negative-dimension false-accept in §2. Without the non-negative check this throws `MLXException` from native instead — assert the exception **type**, since both throw |
| `slice` with `start.length != a.ndim()` rejected with `IllegalArgumentException` | first of §3's two guards |
| `slice` with a **negative** `start`, and with a **negative stride** (reversing a dimension), both succeed and return correct values | **these must not throw.** Native supports both; a guard rejecting them is the failure mode Decision 8 exists to prevent, and this row is what catches it |
| `slice` with `strides[i] == 0` throws `IllegalArgumentException` | the UB guard. Assert the exception **type and message** — without the guard this does not throw at all, it silently returns a zero-sized dimension, so a test asserting only "does not succeed" would pass against the broken version |
| `squeeze` both overloads; `transpose(a, axes)` with a non-reversing permutation | |
| `slice` contiguous **and strided** | strided is the one that matters: it yields a non-contiguous view, exercising `toFloatArray()`'s `mlx_contiguous` path the way `transposeReordersElementsNotJustShape` does |
| `eval()` zero-arg no-op | proves the null-ctx check does not misfire on the non-null empty vector |
| `eval(a, b, c)` all three correct, **including arrays from two scopes on one thread** | the case a same-scope check would wrongly reject |
| `eval` twice is idempotent | hits the `none unscheduled` fast path |
| `eval(good, bad, good)` with `bad = MLX.outer(a, a)` for `a` of 2^20 elements: the `MLXException` message names **index 1** | §4's attribution mitigation, and the only check that the `catch` was implemented rather than dropped. **Use an allocation-class error, not a shape error** — shapes are validated at op-build time, so `MLXNativeErrorTest`'s fixture throws long before `eval` and cannot exercise this path. **And build it with `outer`, not `MLX.array`** — see §4; the direct route needs a `float[]` 512× larger than the JVM allows. Measured: joint eval throws in ~1 ms, re-run attributes in under 1 ms each |
| `MLXMemoryLeakTest` with a multi-array `eval` in the loop | **the test that matters for §4**: a missing `mlx_vector_array_free` shows up as `activeMemoryBytes()` growth, and a double-free aborts the JVM, which is itself the assertion |
| `MLXScopeTest`: foreign-thread array access throws; owning-thread `close()` afterwards still works | §1 |
| **`MLXScopeTest`: an `MLXArray` whose *scope* has been closed throws `IllegalStateException` from `shape()`** | **the test that discriminates §1's option 1 from option 3.** `MLXScope.close()` frees handles via `holder.closeAll()` and never touches any `MLXArray`, so `MLXArray.closed` is still `false` at this point — without `checkAccess()`'s `ensureOpen()` half this reads a freed handle instead of throwing. A future "simplification" to an array-local `Thread owner` would drop exactly this check, and nothing else in the suite would notice |

**Named caveat.** No test can prove `mlx_eval`-over-a-vector differs from the loop. Results are
numerically identical and neither recomputes (`array::eval()` is literally `eval({*this})`, upstream
`array.cpp:154-161`; the status flags make a second eval a `wait()`). The difference is one tape build
instead of N, which is only observable as timing. **Do not add a timing assertion to CI.** The tests
above cover the correctness and lifetime risks that the rewrite actually introduces.

## Verification

0. **Before writing any of §4: paste its pseudocode block into a scratch class and compile it against
   the real generated bindings.** It references five real signatures
   (`mlx_vector_array_new_data`, `mlx_array_.allocateArray`, `mlx_array_.sizeof`,
   `mlx_vector_array_.ctx`, and `checked`'s `IntSupplier`), and nothing in a Markdown file checks any
   of them. Three defects reached committed drafts of that block; two were plain compile errors that
   `javac` would have rejected in under a second, and re-reading caught neither. This step costs
   minutes and is ordered first deliberately.
1. `./gradlew build` — Spotless, Checkstyle, and the forked `loaderGuardTest` included.
2. `scripts/regen-bindings.sh` then `git diff --exit-code jmlx-ffi/src/main/generated/java` — must be
   clean. **This proves the load-bearing premise of this document**: Phase 3 required no binding
   changes.
3. `./gradlew :jmlx-core:test --tests '*MLXNumericTest*'` — all new ops green.
4. `./gradlew :jmlx-core:test --tests '*MLXMemoryLeakTest*'` — no active-memory growth across the
   measured iterations, with a multi-array `eval` in the loop body.
5. `./gradlew :jmlx-core:test --tests '*MLXNativeErrorTest*'` — passes with **unmodified fixtures**,
   confirming §2's guard did not convert it into a happy path.
6. `./gradlew :jmlx-core:test --tests '*MLXScopeTest*'` — **both** foreign-thread access and
   closed-scope access throw. The second is the use-after-free guard from §1; a run that only proves
   the first has not exercised the half of `checkAccess()` that option 3 would have dropped.
7. `./gradlew :jmlx-core:test --tests '*MLXGpuVerificationTest*'` — kernels still dispatch to GPU.
8. `./gradlew :jmlx-examples:run` — HelloMLX prints correct values for the broadcast `add` and the
   multi-array `eval`.
9. **Peak-memory number recorded in §4** (loop vs. vector over the same N-array workload, via
   `mlx_get_peak_memory`). The tradeoff Decision 3 accepts must be quantified before it ships, not
   asserted. Fails if §4 still reads "can go up" with no measurement.
10. `grep -rn requireSameShape jmlx-core/src` — expect zero matches.
11. `grep -rn 'Thread.currentThread()' jmlx-core/src/main/java/se/alipsa/jmlx/core | wc -l` — **expect
    0**. The baseline in `jmlx-core/src/main` as a whole is already 2, both pre-existing in
    `MLXScope` (`:95` capturing `owner`, `:122` in `checkThread`), so an unscoped grep fails against a
    correct implementation. Scoping it to the `core` package asserts the actual invariant: thread
    checking lives in `se.alipsa.jmlx.memory` and is reached through `ensureOpen()`, and no op grew
    its own copy.
12. `grep -rn 'mlx_vector_array_new_data' jmlx-core/src/main` — expect exactly one call site, and
    confirm by eye that its first argument is the confined `Arena`, never an `MLXScope`. This is the
    heap-corruption case in §4; it is worth one manual look because no test can catch it — passing a
    scope there is UB, not a failed assertion.

## Failure modes this document kept producing

Eight review rounds raised 14 correctness defects and 22 further suggestions against this document.
The defects fall into three clusters, recorded as what to watch for in Phase 4 rather than as
history — the warning against re-making each one already lives where it would be re-made.

1. **A Java guard narrower than native.** Twice, *by this document*: `matmul`'s
   `dtype() == FLOAT32`, and `slice` rejecting negative indices and strides. (The two guards in
   shipped v0.1 code — `requireSameShape` and `matmul` rank-2, still at `MLX.java:130` and `:123` —
   are this document's **premise**, not its output; see Context.) The `slice` one is the instructive
   case: it was introduced *by the fix for the previous instance*, in the same commit that corrected
   `matmul`'s dtype guard. The resolving principle is in §3 — mirror native where native is defined,
   guard where it is undefined, and say which case applies.
2. **A decision recorded in prose but not in the block coded from.** Three times: the `isInexact()`
   switch reached Decision 2 but not §2; §4's attribution mitigation had no `catch` in its
   pseudocode; §1's scope-closed test was mandated in a sentence and absent from the Testing table.
   Inline comments carrying the reasoning — as at the allocator argument in §4 — are the mitigation.
3. **Java written but never compiled or run.** Three times, all in §4's block: a dropped
   `SegmentAllocator`, a lambda capturing a mutated loop counter, and a fixture recipe needing a
   2^40-element `float[]`. **Re-reading caught none of them** — they were found by checking
   signatures and by execution. Two of the three are plain compile errors; hence Verification #0.
   (A fourth, `new MLXException(msg, cause)` against a class declaring only `(String)`, was caught
   while writing and never reached a commit — by checking the class instead of assuming it, which is
   the same habit.)

## Open questions

* **Is `mlx_array_free` safe to call from the `Cleaner` thread rather than the scope's owning
  thread?** Carried over unresolved from `initial-plan.md`, still commented at `MLXScope.java:69-74`.
  Phase 3 multiplies array churn but does not change the mechanism, so it is out of scope here. The
  fallback — enqueue onto the owning thread — is unchanged.
* `MLX.java` lands around 400 lines after §3. Acceptable now. Naming the split trigger
  (`MLXOps`/`MLXShape`/`MLXLinalg` behind a delegating `MLX`) means it happens deliberately rather
  than at 700 lines under duress.
* Native wart, not fixable from Java: if `mlx_array_get_` throws on a null-ctx handle,
  `vector.cpp:50-53` returns a fresh null-ctx struct **without freeing the partially built vector** —
  mlx-c leaks it. `MLXArray.ensureOpen()` already prevents passing one, so this is recorded rather
  than worked around.
