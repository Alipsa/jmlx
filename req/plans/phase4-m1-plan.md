# jmlx Phase 4 — M1 implementation plan (`Module` and the simple layers)

**Spec:** `req/phase4-plan.md` — specifically §2 (child scopes / `scopeOf` / explicit-target overloads),
§3 (op-helper shapes, deferred generic helpers), §4 (`DType`/`astype`), §5 (`Module` and the simple
layers), Research findings (nullable-by-value-struct params, statusless scalar constructors), and
Testing approach. This plan is the spec's argument for M1 only — M2 (`MLXGrad`/`ModuleGrad`), M3
(RoPE/attention/KV cache) and M4 (`QuantizedLinear`) are explicitly out of scope for this branch. Where
this plan gives an exact signature or value, that value is authoritative for the task; where it defers
to the spec's prose, the spec is authoritative.

Branch: `worktree-phase4-m1`, off `main` at `ae4636c` (PR #5 merged). Native bootstrap has been run in
this worktree (`native/install/lib/mlx.metallib` exists) — `@EnabledIfNativeAvailable` tests run for
real, not skip.

## Global Constraints

Apply these to every task below; do not restate them per task, but every task's diff is checked
against them.

1. **New ops go in the class §1 of the spec names for them, never in `MLX.java`.** `MLX.java` is
   closed to new ops since M0a (creation/`astype`/`eval`/`hoist`/`keep`/device·stream only). This
   plan's op additions go in `MLXOps`, `MLXShape`, `MLXFast`, or `MLXRandom` as stated per task.
2. **Every op body with 2+ `MLXArray` operands (nullable ones included) resolves its target scope via
   `NativeOps.scopeOf(opName, operands...)`, passing every array operand.** Never read one operand's
   `.scope()` directly when there is more than one operand. A single-operand op may use `a.scope()`
   directly (that is what `scopeOf` reduces to anyway) — this is what the existing `unaryOp`/`shapeOp`
   already do; keep that pattern for new single-operand ops.
3. **Every native call goes through `NativeOps.checked(...)`** (the two-arg form, naming the op),
   matching every existing op body. Never call an `mlx_h` function outside `checked`.
4. **Nullable native array parameters are zero-ctx `mlx_array_` structs passed by value, never
   `MemorySegment.NULL`.** Task 4 adds the one helper this requires (`NativeOps.nullableHandle`); every
   later task that needs a nullable array parameter uses it, never invents its own null-struct
   construction.
5. **Do not add a generic op-body helper without a real consumer landing in the same task.** This is
   the M0d precedent (`req/phase4-plan.md` Status section): `reduceOp`, `axis2Op`, `nullableHandle`, and
   the explicit-target `unaryOp` overload are each added in the task that first needs them, wired
   directly to that task's real op — never speculatively.
6. **Style:** run `./gradlew spotlessApply` before finishing a task, then `./gradlew build` (Spotless
   check + Checkstyle + full test suite, including native tests) must succeed. Report actual test
   counts from the build output (e.g. `BUILD SUCCESSFUL`, with the per-module test task list), not just
   "tests pass."
7. **Testing style, matching every existing test file exactly:** `@EnabledIfNativeAvailable` on the
   test class; `try (MLXScope scope = new MLXScope())`; hand-computed goldens; assert element values via
   `assertArrayEquals(expected, actual.toFloatArray(), EPS)`, not just shapes. `EPS = 1e-5f` for
   ops whose golden is an exact rational/integer result; use a looser inline tolerance (e.g. `1e-3f`) only
   where the golden itself involves an irrational function (`sigmoid`/`tanh`/`erf`/`sqrt`-of-a-sum), and
   say in a comment why that op needs it.
8. **New op-level tests for `MLXOps`/`MLXShape` additions go into the existing
   `jmlx-core/src/test/java/se/alipsa/jmlx/core/MLXNumericTest.java`**, matching how `reshape`/
   `transpose`/`slice`/`broadcastTo`/`squeeze` are already tested there rather than in per-op files.
   `MLXFast`/`MLXRandom` are new classes with no existing test home — new test classes
   `MLXFastTest.java`/`MLXRandomTest.java` alongside it are correct for those.
9. **`se.alipsa.jmlx.nn` is a new package inside the `jmlx-core` Gradle module**
   (`jmlx-core/src/main/java/se/alipsa/jmlx/nn/`, tests under
   `jmlx-core/src/test/java/se/alipsa/jmlx/nn/`) — not a new Gradle module. It depends on
   `se.alipsa.jmlx.core` and `se.alipsa.jmlx.memory`; the reverse must never happen
   (`grep -rn 'se\.alipsa\.jmlx\.nn' jmlx-core/src/main/java/se/alipsa/jmlx/{core,memory}` must stay
   empty — spec Verification #6a).

## Task 1: `Module` — the abstract base class (`se.alipsa.jmlx.nn`)

Create `jmlx-core/src/main/java/se/alipsa/jmlx/nn/Module.java` and
`jmlx-core/src/main/java/se/alipsa/jmlx/nn/UnaryModule.java`. This task touches no native ops and has
no dependency on Tasks 2-4; it can be built and reviewed standalone.

```java
package se.alipsa.jmlx.nn;

public interface UnaryModule {
  MLXArray forward(MLXArray x);   // import se.alipsa.jmlx.core.MLXArray
}
```

```java
package se.alipsa.jmlx.nn;

public abstract class Module {
  protected Module(MLXScope scope);                                   // import se.alipsa.jmlx.memory.MLXScope
  protected final MLXArray param(String name, MLXArray value);        // register; returns value
  protected final MLXArray param(String name);                        // look up the CURRENT bound value
  protected final <M extends Module> M child(String name, M module);  // register a submodule; returns module
  protected final MLXScope scope();
  public final SequencedMap<String, MLXArray> parameters();
  public final void update(Map<String, MLXArray> byPath);
  public final void freeze();
  protected void onParametersUpdated() {}
  public final void rebind(SequencedMap<String, MLXArray> values);
}
```

**The two `param` overloads exist for a reason that is not in the spec's code block and must be
followed exactly — this is this plan's ruling, made because M2 (`ModuleGrad`, out of scope for this
branch but the reason this shape exists) only works if it is followed now.** §6 of the spec describes
`ModuleGrad` rebinding a module tree's parameters to traced primals around a loss call, and states
"`Linear.forward(x)` reads its own fields — the registered `W`." If a layer's constructor caches the
`MLXArray` returned by `param(name, value)` into a private field and `forward()` reads that field, a
later `rebind()` call updates `Module`'s internal map but the layer's cached field still points at the
*original* array — `rebind` becomes a no-op from the layer's point of view, and grads computed through
it later (M2) would be silently disconnected from the model, which is exactly the "no signal: correct
shapes, no exception" failure §6's own Failure-modes entry (round 5) describes. **The fix, required in
this task and consumed in Task 5: `param(String name, MLXArray value)` registers and returns the value
(constructors may use the return value only to validate shape, e.g. `int[] shape = param("weight",
w).shape();`, never to cache it for `forward()`), and every layer built in Task 5 calls `param(String
name)` fresh inside `forward()`** to read whatever is currently bound. A plain scalar field (a `float
eps`, a `boolean hasBias`) is not an `MLXArray` and has no rebind hazard — those may be cached normally.

Field layout backing the above: a `LinkedHashMap<String, MLXArray> params` (own parameters, local
names, insertion order) and a `LinkedHashMap<String, Module> children` (submodules, local names,
insertion order), both private, plus `private boolean frozen`.

`param(String name, MLXArray value)` (setter): throws `IllegalStateException` naming `name` if `frozen`
is true, or if `name` is already a key in `params`. Otherwise `params.put(name, value)`, return `value`.

`param(String name)` (getter): throws `IllegalStateException` naming `name` if `name` is not a key in
`params`. Otherwise return `params.get(name)`.

`child(String name, M module)`: throws `IllegalStateException` naming `name` if `frozen` is true, or if
`name` is already a key in `children`. Otherwise `children.put(name, module)`, return `module`.

`scope()`: returns the `MLXScope` passed to the constructor.

`parameters()`: depth-first, own params before children (matching `req/phase4-plan.md:632-634`'s
"depth-first, own params before children, dotted paths, insertion-ordered, immutable snapshot"). Build a
`LinkedHashMap<String, MLXArray>` via a private recursive `collectParameters(String prefix,
SequencedMap<String, MLXArray> out)`: for each entry in `params` (insertion order), `out.put(prefix +
key, value)`; then for each entry in `children` (insertion order), recurse into
`child.collectParameters(prefix + childName + ".", out)`. The public method wraps the result in
`Collections.unmodifiableSequencedMap(...)`.

`update(Map<String, MLXArray> byPath)`: for each entry, resolve the dotted path to the owning `Module`
and write the array into that module's *local* `params` map (see `writeByPath` below), collecting the
set of touched modules; then walk the tree depth-first (this module first, then `children` in
insertion order, recursively) and call `onParametersUpdated()` on exactly the modules in the touched
set — **all writes complete before any notify runs** (spec §5: "All-writes-then-all-notifies, not
interleaved"). An unresolvable path throws `IllegalArgumentException` naming the full path — do not
silently ignore it.

Concretely, resolve+write via a private recursive helper that returns the *owning `Module`* (not the
array), so callers can build the touched set:

```java
private Module resolveAndWrite(String path, MLXArray value) {
  int dot = path.indexOf('.');
  if (dot < 0) {
    if (!params.containsKey(path)) {
      throw new IllegalArgumentException("unknown parameter path \"" + path + "\"");
    }
    params.put(path, value);
    return this;
  }
  String childName = path.substring(0, dot);
  Module target = children.get(childName);
  if (target == null) {
    throw new IllegalArgumentException("unknown parameter path \"" + path + "\"");
  }
  return target.resolveAndWrite(path.substring(dot + 1), value);
}
```

`update` calls `resolveAndWrite` per entry into a `Set<Module> touched = new HashSet<>()`, then a
private `notifyDepthFirst(Set<Module> touched)` (this module first if `touched.contains(this)`, then
each child in insertion order, recursively) drives the notification pass.

`freeze()`: sets `this.frozen = true`, then calls `freeze()` on every module in `children.values()`
(insertion order) — freezing must cascade, or a nested submodule registered before its ancestor's
`freeze()` call could still accept a new `param`/`child` after the ancestor believes the tree is frozen.

`rebind(SequencedMap<String, MLXArray> values)`: for each entry, call `resolveAndWrite(key, value)` and
discard the returned owner — **never calls `onParametersUpdated()`, on any module, for any reason**
(spec §6: "rebind must NOT fire `onParametersUpdated()`"). Legal after `freeze()` (`resolveAndWrite`
never touches `frozen`). An unresolvable path throws `IllegalArgumentException`, same as `update`.

`onParametersUpdated()`: `protected`, empty body, overridable. Called only from `update`'s
`notifyDepthFirst`, never from `rebind`.

### Tests — `jmlx-core/src/test/java/se/alipsa/jmlx/nn/ModuleTest.java`

`Module` has no ops of its own, so tests use `MLX.array(scope, ...)` (existing, `se.alipsa.jmlx.core`)
to build tiny param arrays, plus two minimal test-only `Module` subclasses defined in the test file
itself (do not reach for `Linear` — it does not exist until Task 5):

```java
private static final class Leaf extends Module {
  boolean notified;
  Leaf(MLXScope scope, MLXArray w) { super(scope); param("w", w); }
  @Override protected void onParametersUpdated() { notified = true; }
}
private static final class Branch extends Module {
  Branch(MLXScope scope, Module child) { super(scope); child("leaf", child); }
}
```

Required cases:
- `param(name, value)` twice with the same name throws `IllegalStateException`.
- `child(name, module)` twice with the same name throws `IllegalStateException`.
- `param`/`child` after `freeze()` throws `IllegalStateException`, both on the frozen module directly
  and on a child registered *before* the ancestor's `freeze()` call (proves the cascade: freeze the
  `Branch`, then try `child("other", ...)` on the already-registered `Leaf` instance — it must throw
  too).
- `parameters()` on a `Branch` wrapping a `Leaf` named `"leaf"` with a param named `"w"` returns exactly
  one entry keyed `"leaf.w"`.
- `parameters()` order is insertion order: a module with params `"a"`, `"b"` then a child `"c"` — the
  returned map's key order is `["a", "b", "c.<child's own params in their insertion order>"]`.
- `update(Map.of("leaf.w", newArray))` on a `Branch` wrapping that `Leaf`: after the call, the `Leaf`
  instance's `notified` field is `true` and the `Branch`'s own `onParametersUpdated()` was never called
  (give `Branch` the same `notified` field/override and assert it stays `false` — the branch itself
  wrote nothing).
- `update` with an unknown path throws `IllegalArgumentException` naming the path.
- `rebind` after `freeze()` succeeds (no exception) and does **not** flip `Leaf`'s `notified` flag.
- `rebind` with an unknown path throws `IllegalArgumentException`.

## Task 2: Elementwise/reduction ops (`MLXOps`)

File: `jmlx-core/src/main/java/se/alipsa/jmlx/core/MLXOps.java`. Add, in this exact native mapping
(`mlx_h` signatures confirmed against `native/install/include/mlx/c/ops.h` and the generated
`mlx_h.java` in this repo):

```java
public static MLXArray sigmoid(MLXArray a) { return NativeOps.unaryOp("sigmoid", a, mlx_h::mlx_sigmoid); }
public static MLXArray erf(MLXArray a)     { return NativeOps.unaryOp("erf", a, mlx_h::mlx_erf); }
public static MLXArray tanh(MLXArray a)    { return NativeOps.unaryOp("tanh", a, mlx_h::mlx_tanh); }
public static MLXArray sqrt(MLXArray a)    { return NativeOps.unaryOp("sqrt", a, mlx_h::mlx_sqrt); }
public static MLXArray rsqrt(MLXArray a)   { return NativeOps.unaryOp("rsqrt", a, mlx_h::mlx_rsqrt); }
public static MLXArray square(MLXArray a)  { return NativeOps.unaryOp("square", a, mlx_h::mlx_square); }
public static MLXArray negative(MLXArray a){ return NativeOps.unaryOp("negative", a, mlx_h::mlx_negative); }
```

`maximum`/`power` are binary and broadcasting, exactly like the existing `add`/`subtract`/`multiply`/
`divide` bodies just above them in this file — same shape, same Java-side guard:

```java
public static MLXArray maximum(MLXArray a, MLXArray b) {
  requireBroadcastCompatible(a, b, "maximum");
  return NativeOps.binaryOp("maximum", a, b, mlx_h::mlx_maximum);
}

public static MLXArray power(MLXArray a, MLXArray b) {
  requireBroadcastCompatible(a, b, "power");
  return NativeOps.binaryOp("power", a, b, mlx_h::mlx_power);
}
```

(`requireBroadcastCompatible` already exists in this file, private, used by `add`/`subtract`/
`multiply`/`divide` — reuse it verbatim, do not duplicate its body.)

`sum`/`mean` with axes+keepdims need a new shape neither `unaryOp` nor `shapeOp` has: `(res, a, const
int* axes, size_t axes_num, bool keepdims, stream)` — this is exactly `reduceOp`, the generic helper
`req/phase4-plan.md` §3 named and deferred pending a real consumer. Add it to
`jmlx-core/src/main/java/se/alipsa/jmlx/core/NativeOps.java`, next to `shapeOp` (same shape plus one
`boolean`):

```java
static MLXArray reduceOp(String opName, MLXArray a, int[] axes, boolean keepdims, ReduceOp op) {
  MLXScope scope = a.scope();
  try (Arena tmp = Arena.ofConfined()) {
    MemorySegment nativeAxes = tmp.allocateFrom(ValueLayout.JAVA_INT, axes);
    MemorySegment res = mlx_h.mlx_array_new(scope);
    checked(opName, () -> op.apply(res, a.handle(), nativeAxes, axes.length, keepdims, DEFAULT_STREAM));
    return new MLXArray(scope, res);
  }
}

@FunctionalInterface
interface ReduceOp {
  int apply(MemorySegment res, MemorySegment a, MemorySegment axes, long axesNum, boolean keepdims, MemorySegment stream);
}
```

Then in `MLXOps` (existing no-arg `sum(MLXArray a)` stays exactly as is — do not touch it or its
javadoc; these are new overloads):

```java
public static MLXArray sum(MLXArray a, int[] axes, boolean keepdims) {
  return NativeOps.reduceOp("sum", a, axes, keepdims, mlx_h::mlx_sum_axes);
}

public static MLXArray mean(MLXArray a, int[] axes, boolean keepdims) {
  return NativeOps.reduceOp("mean", a, axes, keepdims, mlx_h::mlx_mean_axes);
}
```

### Tests — append to `jmlx-core/src/test/java/se/alipsa/jmlx/core/MLXNumericTest.java`

Exact-value goldens (`EPS = 1e-5f`, the class's existing constant):
- `sqrt([4,9,16])` = `[2,3,4]`.
- `rsqrt([4,16])` = `[0.5,0.25]`.
- `square([2,-3,4])` = `[4,9,16]`.
- `negative([2,-3,0])` = `[-2,3,0]`.
- `maximum([1,5,3],[4,2,3])` = `[4,5,3]`.
- `power([2,3,4],[3,2,1])` = `[8,9,4]`.
- `sum(a=[[1,2,3],[4,5,6]] shape[2,3], axes=[1], keepdims=false)` = `[6,15]` shape `[2]`.
- `sum(a, axes=[0], keepdims=true)` = `[[5,7,9]]` shape `[1,3]`.
- `mean(a, axes=[1], keepdims=false)` = `[2,5]` shape `[2]`.
- `mean(a, axes=[0], keepdims=true)` = `[[2.5,3.5,4.5]]` shape `[1,3]`.

Looser-tolerance goldens (irrational; use `1e-4f` inline, with a one-line comment saying why):
- `sigmoid([0,1,-1])` = `[0.5, 0.7310586, 0.2689414]`.
- `tanh([0,1,-1])` = `[0.0, 0.7615942, -0.7615942]`.
- `erf([0,1])` = `[0.0, 0.8427008]`.

Also add one Java-side-guard test: `maximum`/`power` with incompatible shapes throws
`IllegalArgumentException` (mirror `add`'s existing incompatible-shape test). Do **not** re-derive the
parent/child cross-scope test for `maximum`/`power` — that behavior is already proven once for
`binaryOp` generically via the existing `add` test of that shape; repeating it per new binary op adds
no coverage.

## Task 3: `swapaxes`/`take`/`takeAxis`, and the explicit-target `transpose` overload (`MLXShape`)

File: `jmlx-core/src/main/java/se/alipsa/jmlx/core/MLXShape.java`.

**`swapaxes`** needs a new shape: `(res, a, int axis1, int axis2, stream)` — two plain ints, no array
param, one operand. This is `axis2Op`, §3's second deferred generic helper. Add to `NativeOps.java`,
next to `shapeOp`:

```java
static MLXArray axis2Op(String opName, MLXArray a, int axis1, int axis2, Axis2Op op) {
  MLXScope scope = a.scope();
  MemorySegment res = mlx_h.mlx_array_new(scope);
  checked(opName, () -> op.apply(res, a.handle(), axis1, axis2, DEFAULT_STREAM));
  return new MLXArray(scope, res);
}

@FunctionalInterface
interface Axis2Op {
  int apply(MemorySegment res, MemorySegment a, int axis1, int axis2, MemorySegment stream);
}
```

```java
public static MLXArray swapaxes(MLXArray a, int axis1, int axis2) {
  return NativeOps.axis2Op("swapaxes", a, axis1, axis2, mlx_h::mlx_swapaxes);
}
```

**`take`** fits the existing `BinaryOp` shape `(res, a, b, stream)` exactly — no new helper:

```java
public static MLXArray take(MLXArray a, MLXArray indices) {
  return NativeOps.binaryOp("take", a, indices, mlx_h::mlx_take);
}
```

Upstream semantics (confirmed against `native/scratch/wheel/mlx/include/mlx/ops.h:1077-1078`, "Take
array entries at the given indices treating the array as flattened"): `a` is flattened before indexing,
regardless of its own rank. Say this in the javadoc — a reader expecting `a`'s own shape to matter here
would be wrong.

**`takeAxis`** has no existing helper shape (two array operands *and* an int, none of the six generic
helpers cover this): hand-roll it, same style as `MLXOps.sum()`'s existing hand-rolled body just above
`inner` in that file. Resolve its scope via `NativeOps.scopeOf`, since it has two operands:

```java
public static MLXArray takeAxis(MLXArray a, MLXArray indices, int axis) {
  MLXScope scope = NativeOps.scopeOf("takeAxis", a, indices);
  MemorySegment res = mlx_h.mlx_array_new(scope);
  NativeOps.checked("takeAxis",
      () -> mlx_h.mlx_take_axis(res, a.handle(), indices.handle(), axis, NativeOps.DEFAULT_STREAM));
  return new MLXArray(scope, res);
}
```

Upstream semantics (confirmed against `ops.h:1072-1074`, "Take array slices at the given indices of the
specified axis"): result shape is `a.shape()[:axis] + indices.shape() + a.shape()[axis+1:]`. This is
the exact op `Embedding` (Task 5) needs: `weight` shape `[numEmbeddings, dim]`, `indices` any shape,
`axis = 0`.

**The explicit-target `transpose` overload** — required by `req/phase4-plan.md` §2 mitigation 1 for
`Linear` (Task 5), independent of whether `Linear` lands in this same task. Add an explicit-target
overload of `NativeOps.unaryOp` (the existing single-arg-target one becomes a one-line delegation to
it — do not duplicate its body):

```java
static MLXArray unaryOp(String opName, MLXArray a, MLXScope target, UnaryOp op) {
  MemorySegment res = mlx_h.mlx_array_new(target);
  checked(opName, () -> op.apply(res, a.handle(), DEFAULT_STREAM));
  return new MLXArray(target, res);
}

static MLXArray unaryOp(String opName, MLXArray a, UnaryOp op) {
  return unaryOp(opName, a, a.scope(), op);
}
```

Then in `MLXShape`, alongside the existing `transpose(MLXArray a)` / `transpose(MLXArray a, int[]
axes)`:

```java
/** Reverses every axis, allocating the result into {@code target} instead of {@code a.scope()}. See
 *  req/phase4-plan.md §2 mitigation 1: lets a weight-derived view computed inside {@code forward()}
 *  land in the caller's (step) scope rather than leaking into {@code a}'s own (model) scope once per
 *  call. */
public static MLXArray transpose(MLXArray a, MLXScope target) {
  return NativeOps.unaryOp("transpose", a, target, mlx_h::mlx_transpose);
}
```

### Tests — append to `MLXNumericTest.java`

- `swapaxes([[1,2,3],[4,5,6]] shape[2,3], 0, 1)` = `[[1,4],[2,5],[3,6]]` shape `[3,2]`. `EPS=1e-5f`.
- `take([[1,2],[3,4]] shape[2,2], indices=INT32[0,3,1])` = `[1,4,2]` shape `[3]`. Build indices via
  `MLX.array(scope, new int[]{0,3,1}, new int[]{3})` (existing `int[]` overload — produces INT32).
- `takeAxis(table=[[1,2],[3,4],[5,6]] shape[3,2], indices=INT32[2,0], axis=0)` = `[[5,6],[1,2]]` shape
  `[2,2]`.
- `transpose(a, target)` where `target` is a **parent** scope of `a.scope()`: result's `.scope()` is
  `target`, not `a.scope()` — mirrors the existing `addAcrossParentAndChildScopeAllocatesIntoTheChildRegardlessOfOperandOrder`
  pattern one call away: build `a` in a child scope, call `transpose(a, parentScope)`, `assertSame(parentScope,
  result.scope())`, then close the child scope and confirm the result is still readable (`toFloatArray()`
  succeeds and matches the transposed golden) — this is what actually proves the view escaped the child
  before it closed, not just that the call didn't throw.

## Task 4: `MLXFast.rmsNorm`/`layerNorm`, `MLXRandom.seed`/`normal`/`uniform`

**`nullableHandle`** — the third deferred generic helper, needed by every op in this task. Add to
`NativeOps.java`. A "null" `mlx_array` for a by-value nullable parameter is a zero-`ctx` struct, never
`MemorySegment.NULL` (spec Research findings: "the correct null is a zero-filled `mlx_array_` struct...
Passing `MemorySegment.NULL` is a segfault, not an exception"). `SegmentAllocator.allocate` zero-fills
by default, so allocating one fresh struct *is* the null value:

```java
static MemorySegment nullableHandle(MLXArray a, SegmentAllocator tmp) {
  return a == null ? tmp.allocate(mlx_array_.layout()) : a.handle();
}
```

Add `import java.lang.foreign.SegmentAllocator;` and `import se.alipsa.jmlx.ffi.mlx_array_;` to
`NativeOps.java`.

### `MLXFast` (`jmlx-core/src/main/java/se/alipsa/jmlx/core/MLXFast.java`)

Confirmed native signatures (`native/install/include/mlx/c/fast.h:93-99,163-168`):
`mlx_fast_rms_norm(res, x, weight /*nullable*/, eps, s)`;
`mlx_fast_layer_norm(res, x, weight /*nullable*/, bias /*nullable*/, eps, s)`.

```java
public static MLXArray rmsNorm(MLXArray x, MLXArray weight, float eps) {
  MLXScope scope = NativeOps.scopeOf("rmsNorm", x, weight);
  try (Arena tmp = Arena.ofConfined()) {
    MemorySegment weightHandle = NativeOps.nullableHandle(weight, tmp);
    MemorySegment res = mlx_h.mlx_array_new(scope);
    NativeOps.checked("rmsNorm",
        () -> mlx_h.mlx_fast_rms_norm(res, x.handle(), weightHandle, eps, NativeOps.DEFAULT_STREAM));
    return new MLXArray(scope, res);
  }
}

public static MLXArray layerNorm(MLXArray x, MLXArray weight, MLXArray bias, float eps) {
  MLXScope scope = NativeOps.scopeOf("layerNorm", x, weight, bias);
  try (Arena tmp = Arena.ofConfined()) {
    MemorySegment weightHandle = NativeOps.nullableHandle(weight, tmp);
    MemorySegment biasHandle = NativeOps.nullableHandle(bias, tmp);
    MemorySegment res = mlx_h.mlx_array_new(scope);
    NativeOps.checked("layerNorm", () -> mlx_h.mlx_fast_layer_norm(res, x.handle(), weightHandle,
        biasHandle, eps, NativeOps.DEFAULT_STREAM));
    return new MLXArray(scope, res);
  }
}
```

`weight`/`bias` being Java `null` is a **legitimate call** (unweighted/unbiased norm) — do not add a
null-check that throws; `nullableHandle`/`scopeOf` both already treat `null` as "absent," not an error.

### `MLXRandom` (`jmlx-core/src/main/java/se/alipsa/jmlx/core/MLXRandom.java`)

Confirmed native signatures (`native/install/include/mlx/c/random.h:101-109,130,150-158` and the
generated `mlx_h.java` in this repo): `mlx_random_seed(uint64_t seed)` — no result, no stream, global
RNG state; `mlx_random_normal(res, shape, shape_num, dtype, loc, scale, key /*nullable*/, s)` — `loc`/
`scale` are plain `float`, not arrays; `mlx_random_uniform(res, low, high, shape, shape_num, dtype, key
/*nullable*/, s)` — **`low`/`high` are `mlx_array`, by value, not nullable and not `float`** — read the
signature carefully, this is the one surprise in this task.

```java
public static void seed(long seed) {
  NativeOps.checked("seed", () -> mlx_h.mlx_random_seed(seed));
}

public static MLXArray normal(MLXScope scope, int[] shape, DType dtype, float loc, float scale) {
  try (Arena tmp = Arena.ofConfined()) {
    MemorySegment nativeShape = tmp.allocateFrom(ValueLayout.JAVA_INT, shape);
    MemorySegment key = NativeOps.nullableHandle(null, tmp);
    MemorySegment res = mlx_h.mlx_array_new(scope);
    NativeOps.checked("normal", () -> mlx_h.mlx_random_normal(res, nativeShape, shape.length,
        dtype.nativeValue(), loc, scale, key, NativeOps.DEFAULT_STREAM));
    return new MLXArray(scope, res);
  }
}
```

`uniform`'s `low`/`high` being real `mlx_array` params (not `float`) means the Java-facing `float low,
float high` must each become a throwaway scalar array first — the exact same statusless-constructor
hazard `MLX.full` already handles (`mlx_array_new_float32` signals failure only via the error handler
plus a null-`ctx` return, never a status). **Do not duplicate that four-line check twice inline** (once
for `low`, once for `high`); factor a tiny private helper local to this method or file:

```java
private static MemorySegment float32Scalar(String opName, float value, Arena tmp) {
  NativeLoader.clearLastNativeError();
  MemorySegment scalar = mlx_h.mlx_array_new_float32(tmp, value);
  if (mlx_array_.ctx(scalar).address() == 0) {
    throw NativeOps.nativeFailure(opName + ": mlx_array_new_float32");
  }
  return scalar;
}

public static MLXArray uniform(MLXScope scope, int[] shape, DType dtype, float low, float high) {
  try (Arena tmp = Arena.ofConfined()) {
    MemorySegment lowScalar = float32Scalar("uniform", low, tmp);
    MemorySegment highScalar = float32Scalar("uniform", high, tmp);
    try {
      MemorySegment nativeShape = tmp.allocateFrom(ValueLayout.JAVA_INT, shape);
      MemorySegment key = NativeOps.nullableHandle(null, tmp);
      MemorySegment res = mlx_h.mlx_array_new(scope);
      NativeOps.checked("uniform", () -> mlx_h.mlx_random_uniform(res, lowScalar, highScalar, nativeShape,
          shape.length, dtype.nativeValue(), key, NativeOps.DEFAULT_STREAM));
      return new MLXArray(scope, res);
    } finally {
      mlx_h.mlx_array_free(lowScalar);
      mlx_h.mlx_array_free(highScalar);
    }
  }
}
```

(**Corrected from an earlier draft of this plan, which incorrectly asserted `lowScalar`/`highScalar`
need no explicit free.** `tmp` only owns the 8-byte `mlx_array_` struct each scalar constructor writes
into; the heap-allocated `mlx::core::array` the struct's `ctx` points at is owned by mlx-c and freed
solely by `mlx_array_free` — the exact invariant `MLXArray.toFloatArray()`'s javadoc documents and the
exact reason `MLX.full` already wraps its own scalar construction in a `finally`. `mlx_random_uniform`
takes `low`/`high` as `const mlx_array`, i.e. it borrows them rather than adopting them, so both
scalars must be freed in a `finally` here too, mirroring `full`'s pattern rather than diverging from
it.)

This plan deliberately does **not** touch `MLX.full`'s existing scalar-construction code to share this
new helper — `full` is already shipped and twice-reviewed (PR #5); leave it as is. The duplication this
avoids is only between `low` and `high` within this one new method.

Add imports to `MLXRandom.java`: `java.lang.foreign.Arena`, `java.lang.foreign.MemorySegment`,
`java.lang.foreign.ValueLayout`, `se.alipsa.jmlx.ffi.NativeLoader`, `se.alipsa.jmlx.ffi.mlx_array_`,
`se.alipsa.jmlx.ffi.mlx_h`, `se.alipsa.jmlx.memory.MLXScope`.

### Tests

New file `jmlx-core/src/test/java/se/alipsa/jmlx/core/MLXFastTest.java` (`@EnabledIfNativeAvailable`,
same style as `MLXNumericTest`):

- `rmsNorm(x=[1,2,3,4], weight=null, eps=1e-5f)` ≈ `x * (1/sqrt(7.5))` = `[0.365148, 0.730296, 1.095444,
  1.460592]`. (`mean(x^2) = 30/4 = 7.5`; `eps` shifts the 6th decimal, negligible against the tolerance
  below.) Tolerance `1e-3f`, with a comment explaining why (irrational `sqrt`, and the golden ignores
  `eps`'s sub-1e-5 perturbation).
- Same input with `weight=[2,2,2,2]`: golden is exactly double the above, same tolerance.
- `layerNorm(x=[1,2,3,4], weight=null, bias=null, eps=1e-5f)`: `mean=2.5`, `var=1.25` (exact — no
  irrational input), `std=sqrt(1.25)≈1.118034`; golden `(x-mean)/std` = `[-1.341641, -0.447214,
  0.447214, 1.341641]`. Tolerance `1e-3f` (the `sqrt` is irrational even though `var` itself is exact).
- Same input with `weight=[2,2,2,2]`, `bias=[1,1,1,1]`: golden is `2*v+1` per element of the above =
  `[-1.683282, 0.105573, 1.894427, 3.683282]`, same tolerance.
- `rmsNorm`/`layerNorm` with `x` in a child scope and `weight` (non-null) in the parent: result's
  `.scope()` is the child (proves `scopeOf` is wired for the nullable-operand case, mirroring the
  spec's Testing-approach row "`layerNorm(x, weight, bias)` with a null `weight` still picks the
  innermost of the non-null operands" — cover both: one case with `weight` null and `bias` non-null in
  a different scope from `x`, one with both non-null).

New file `jmlx-core/src/test/java/se/alipsa/jmlx/core/MLXRandomTest.java`:

- `normal`/`uniform` produce the requested `shape` and `dtype` (e.g. `shape=[100]`, `DType.FLOAT32`).
- **Determinism, the only meaningful assertion for a random op without hand-computing the RNG
  algorithm:** `MLXRandom.seed(42)` then `normal(...)` twice in two independent scopes produces
  bit-identical `toFloatArray()` results; `MLXRandom.seed(42)` then `MLXRandom.seed(7)` then `seed(42)`
  again reproduces the first call's values exactly. This is the standard way to test a seeded RNG
  without asserting specific values that would pin this repo to mlx's exact algorithm.
- `uniform(scope, shape, DType.FLOAT32, low=-1f, high=1f)` after a fixed seed: every returned element is
  `>= -1f && < 1f` (a property assertion, not a golden — this is the one case in this task where
  "assert values, not just shapes" is not achievable, since the values are the RNG's, not a formula's;
  say so in a one-line comment so a future reader does not read the absence of a golden as an oversight).

## Task 5: The layers (`se.alipsa.jmlx.nn`)

Depends on Tasks 1-4 (`Module`, the op additions, `nullableHandle`). Package:
`jmlx-core/src/main/java/se/alipsa/jmlx/nn/`. Every layer implements `UnaryModule` except `Embedding`,
whose `forward` is still single-`MLXArray`-in/single-`MLXArray`-out and therefore also implements
`UnaryModule` — there is no arity mismatch here (MHA, out of scope, is the one that would not fit).

**Load-bearing rule from Task 1, restated because violating it here is the one mistake that would pass
every test in this task and only break in M2 (not built yet, so nothing in this branch would catch it):
every layer's `forward()` reads its `MLXArray` parameters via `param(String name)` (the getter), never
via a field cached from the constructor's `param(name, value)` return.** Plain scalar fields (`eps`,
`hasBias`) are fine to cache.

### `Linear`

```java
package se.alipsa.jmlx.nn;

public final class Linear extends Module implements UnaryModule {
  private final boolean hasBias;

  public Linear(MLXScope scope, MLXArray weight, MLXArray bias) {
    super(scope);
    if (weight.ndim() != 2) {
      throw new IllegalArgumentException("Linear: weight must be rank 2 [out, in], got shape "
          + java.util.Arrays.toString(weight.shape()));
    }
    if (bias != null && (bias.ndim() != 1 || bias.shape()[0] != weight.shape()[0])) {
      throw new IllegalArgumentException("Linear: bias must be rank 1 with length weight.shape()[0]="
          + weight.shape()[0] + ", got shape " + java.util.Arrays.toString(bias.shape()));
    }
    param("weight", weight);
    hasBias = bias != null;
    if (hasBias) {
      param("bias", bias);
    }
  }

  @Override
  public MLXArray forward(MLXArray x) {
    MLXArray weightT = MLXShape.transpose(param("weight"), x.scope());
    MLXArray y = MLXOps.matmul(x, weightT);
    return hasBias ? MLXOps.add(y, param("bias")) : y;
  }
}
```

`weight` is registered and stored in the checkpoint's `[out, in]` layout — never transposed before
registration; `forward` transposes a fresh view per call via the Task 3 explicit-target overload,
landing it in `x.scope()` (the step scope), not `weight`'s own (model) scope. This is
`req/phase4-plan.md` §2's withdrawn-cache mitigation record — do not reintroduce a cached `W.T` field.

### `Embedding`

```java
public final class Embedding extends Module implements UnaryModule {
  public Embedding(MLXScope scope, MLXArray weight) {
    super(scope);
    if (weight.ndim() != 2) {
      throw new IllegalArgumentException("Embedding: weight must be rank 2 [numEmbeddings, dim], got shape "
          + java.util.Arrays.toString(weight.shape()));
    }
    param("weight", weight);
  }

  @Override
  public MLXArray forward(MLXArray indices) {
    return MLXShape.takeAxis(param("weight"), indices, 0);
  }
}
```

### `RMSNorm`

```java
public final class RMSNorm extends Module implements UnaryModule {
  private final float eps;

  public RMSNorm(MLXScope scope, MLXArray weight, float eps) {
    super(scope);
    param("weight", weight);
    this.eps = eps;
  }

  @Override
  public MLXArray forward(MLXArray x) {
    return MLXFast.rmsNorm(x, param("weight"), eps);
  }
}
```

(`weight` is required, not nullable, for this layer — matching how every real `RMSNorm` usage trains
one. `MLXFast.rmsNorm` itself still accepts a null `weight`, unused by this layer, for a caller that
wants the bare op.)

### `LayerNorm`

```java
public final class LayerNorm extends Module implements UnaryModule {
  private final float eps;

  public LayerNorm(MLXScope scope, MLXArray weight, MLXArray bias, float eps) {
    super(scope);
    param("weight", weight);
    param("bias", bias);
    this.eps = eps;
  }

  @Override
  public MLXArray forward(MLXArray x) {
    return MLXFast.layerNorm(x, param("weight"), param("bias"), eps);
  }
}
```

(Both `weight` and `bias` required for this layer, same reasoning as `RMSNorm`. A caller needing
`affine=false` calls `MLXFast.layerNorm(x, null, null, eps)` directly — that is what the bare op is for.)

### `SiLU`

```java
public final class SiLU extends Module implements UnaryModule {
  public SiLU(MLXScope scope) {
    super(scope);
  }

  @Override
  public MLXArray forward(MLXArray x) {
    return MLXOps.multiply(x, MLXOps.sigmoid(x));
  }
}
```

No parameters — `scope` is accepted only to satisfy `Module`'s constructor contract; nothing is
registered.

### `GELU` (exact form only — the tanh approximation is out of scope for this task)

```java
public final class GELU extends Module implements UnaryModule {
  private static final float SQRT2 = 1.4142135f;

  public GELU(MLXScope scope) {
    super(scope);
  }

  @Override
  public MLXArray forward(MLXArray x) {
    MLXScope s = x.scope();   // scalar constants below go here, NOT this.scope() -- §2's fifth
                               // sub-hazard: a creation op called with the model scope inside
                               // forward() leaks once per call.
    MLXArray sqrt2 = MLX.full(s, new int[0], SQRT2, DType.FLOAT32);
    MLXArray half = MLX.full(s, new int[0], 0.5f, DType.FLOAT32);
    MLXArray one = MLX.full(s, new int[0], 1f, DType.FLOAT32);
    MLXArray erfTerm = MLXOps.erf(MLXOps.divide(x, sqrt2));
    return MLXOps.multiply(MLXOps.multiply(half, x), MLXOps.add(one, erfTerm));
  }
}
```

`new int[0]` is a rank-0 (scalar) shape — confirm `MLX.full` accepts it (it takes a general `int[]
shape` with no rank restriction; a rank-0 scalar broadcasts against any shape in the elementwise ops
that consume it, same as every other broadcasting op in this facade).

### Tests — `jmlx-core/src/test/java/se/alipsa/jmlx/nn/`, one file per layer (or combine `SiLU`+`GELU`
into one `ActivationsTest.java` — either is fine, match whichever reads better once written)

**`LinearTest`:**
- `weight=[[1,0,1],[0,1,1]]` shape `[2,3]`, `bias=[10,20]`, `x=[[1,2,3]]` shape `[1,3]` →
  `forward(x)` = `[[14,25]]` shape `[1,2]`. `EPS=1e-5f` (exact integers).
- **`linear.parameters().get("weight")` has shape `[2,3]`** (the checkpoint layout) **and its values
  equal the original `weight` array, not its transpose** — this is the spec's named regression test
  ("`Linear.parameters()` returns `W` with the checkpoint's `[out, in]` shape, not `W.T`"): a `W.T`
  registration would pass the `forward()` golden above and only fail here.
- Constructing with a rank-1 `weight`, or a `bias` of the wrong length, throws `IllegalArgumentException`.
- **`update` on a parent module holding a *nested* `Linear`, then `forward`, uses the new weights.**
  Build a trivial wrapper `Module` (test-local, like Task 1's `Branch`) whose only child is a `Linear`
  named e.g. `"proj"`; call `wrapper.update(Map.of("proj.weight", newWeight))`; call
  `linear.forward(x)` again and confirm the result reflects `newWeight`, not the original. **The
  `Linear` must be the nested module, not the direct receiver of `update`** — the spec calls this out
  explicitly because a `Linear` that IS the receiver would pass even if the depth-first tree walk in
  `Module.update` were broken.
- **Memory:** a per-iteration child-scope loop (same shape as `MLXMemoryLeakTest`'s existing
  `activeMemoryDoesNotGrowWithPerIterationChildScopeUnderALongLivedParent`, imported/copied into this
  test file with `NativeMemoryProbe.activeMemoryBytes()`): `Linear` constructed once with its `weight`
  in a long-lived parent (model) scope; each iteration opens a child (step) scope, builds a fresh
  activation in it, calls `linear.forward(activation)`, `MLX.eval(...)` the result, closes the child.
  Assert active memory does not grow beyond the same generous fixed threshold the existing leak test
  uses. This is the one test in this task with no compile-time signal for the bug it catches (a
  `transpose(weight)` accidentally allocating into `weight.scope()` instead of `x.scope()`).

**`EmbeddingTest`:** `weight=[[1,2],[3,4],[5,6]]` shape `[3,2]`, `indices=INT32[2,0]` →
`forward(indices)` = `[[5,6],[1,2]]` shape `[2,2]`. `EPS=1e-5f`.

**`RMSNormTest`:** reuse Task 4's `MLXFastTest` goldens through the layer:
`weight=[2,2,2,2]`, `eps=1e-5f`, `x=[1,2,3,4]` → `forward(x)` ≈ `[0.730296, 1.460592, 2.190888,
2.921184]`, tolerance `1e-3f` (same irrational-`sqrt` reasoning as Task 4).

**`LayerNormTest`:** `weight=[2,2,2,2]`, `bias=[1,1,1,1]`, `eps=1e-5f`, `x=[1,2,3,4]` → `forward(x)` ≈
`[-1.683282, 0.105573, 1.894427, 3.683282]`, tolerance `1e-3f`.

**`SiLU`/`ActivationsTest`:** `forward([0,1,-1])` ≈ `[0.0, 0.7310586, -0.2689414]`, tolerance `1e-4f`
(irrational `sigmoid`).

**`GELU`/`ActivationsTest`:** `forward([0,1,-1])` ≈ `[0.0, 0.8413447, -0.1586553]`, tolerance `1e-4f`
(irrational `erf`). These are the standard-normal-CDF identity values
(`GELU(x) = x * Φ(x)`, `Φ(1) = 0.8413447`, `Φ(-1) = 0.1586553`) — cross-check against that identity in a
comment if useful, but assert against the plain `0.5*x*(1+erf(x/sqrt2))` formula's numeric result, not
the `Φ` restatement, since that is what the implementation actually computes.

## Verification (run at the end of every task, not only once at the end of the branch)

1. `./gradlew spotlessApply` then `./gradlew build` — must be `BUILD SUCCESSFUL`, including
   `jmlx-ffi:loaderGuardTest` and every module's `test` task. Paste the actual task list / test counts
   in the task report, not a paraphrase.
2. `grep -rn 'se\.alipsa\.jmlx\.nn' jmlx-core/src/main/java/se/alipsa/jmlx/{core,memory}` — must be
   empty after every task (only matters once Task 5 exists, but cheap to run from Task 1 onward).
3. `grep -rn 'mlx_array_new(' jmlx-core/src/main` — every call site's allocator must be one of: a
   `scopeOf(...)` result, an explicit `MLXScope` parameter (creation ops, the new explicit-target
   `transpose`), or a confined `Arena` freed by a local `finally` (the read-back path, and the
   `nullableHandle` allocation in Task 4, which is a zero-filled struct with no native `ctx` heap
   allocation behind it, so it is confined-`Arena`-scoped and never individually freed because the
   whole arena closes at the end of the `try` -- unlike `float32Scalar`'s result, which DOES have a
   `ctx` heap allocation and must be freed via `mlx_array_free` in a `finally`, exactly like `full`'s
   scalar).
4. `./gradlew :jmlx-core:test --tests '*MLXMemoryLeakTest*'` and, after Task 5,
   `--tests '*LinearTest*'` — no active-memory growth beyond the existing threshold.
