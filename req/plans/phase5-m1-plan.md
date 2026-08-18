# jmlx Phase 5 — M1 implementation plan (`MLXIO`, safetensors + GGUF)

Concrete task-by-task plan for `req/phase5-plan.md`'s M1: a new `se.alipsa.jmlx.core.MLXIO` facade
exposing `loadSafetensors`/`saveSafetensors`/`loadGguf`/`saveGguf`. Written after `req/phase5-plan.md`
itself settled every design question this plan depends on (D1 native surface, D2/D2a/D2b/D2c
handle-ownership rules) across six review rounds -- this document does not re-derive those decisions,
it only cites and applies them. Read `req/phase5-plan.md` first if the *why* behind any constraint
below is unclear.

## Findings from this plan's pre-work

**All five non-`mlx_array` struct types this facade touches are the same one-word `{ void* ctx; }`
shape `mlx_array_` itself is** (confirmed by reading `native/install/include/mlx/c/{string,vector,map,
io_types}.h` directly, not inferred from usage): `mlx_string_`, `mlx_vector_string_`,
`mlx_map_string_to_array_`, `mlx_map_string_to_string_`, and `mlx_io_gguf_` (`io_types.h:101`) each
declare exactly one `void* ctx` field, nothing else. This is why every read/write pattern already
established for `mlx_array` in `MLXQuant`/`MLX` (pre-allocate a struct-shaped out-param, pass it to a
native call that writes into it, treat the resulting `MemorySegment` as "the handle") carries over
unchanged to these five types -- there is no new struct layout to reason about, only a new
*deallocator* per type (D2a) and a new *allocator hazard* per type (D2b, below).

**D2b's `SegmentAllocator` trap is not unique to `mlx_io_gguf_new` -- it is every by-value-returning
constructor these five types have**, confirmed against the generated bindings
(`jmlx-ffi/src/main/generated/java/se/alipsa/jmlx/ffi/mlx_h.java`): `mlx_io_gguf_new(SegmentAllocator
allocator)`, `mlx_map_string_to_array_new(SegmentAllocator allocator)`,
`mlx_map_string_to_array_iterator_new(SegmentAllocator allocator, MemorySegment map)`,
`mlx_map_string_to_string_new(SegmentAllocator allocator)`,
`mlx_map_string_to_string_iterator_new(SegmentAllocator allocator, MemorySegment map)`,
`mlx_vector_string_new(SegmentAllocator allocator)`, and `mlx_string_new(SegmentAllocator allocator)`
all take the same jextract-added `allocator` parameter for their by-value struct return (the actual C
signatures take no such parameter at all -- e.g. `mlx_io_gguf mlx_io_gguf_new(void)` in
`io_types.h:104` -- jextract adds it purely to have somewhere to place the returned struct's bytes on
the Java side). Every one of these seven calls this plan makes must pass a confined `Arena`, never an
`MLXScope`, mirroring `MLX.newVectorArray`'s narrowed-`Arena`-parameter precedent (D2b already names
this pattern; this just confirms the full call list it applies to).

**`NativeOps.cstr` is the wrong tool for every string this facade passes to native, and the codebase
already has a documented reason why, not just this plan's own reasoning.** `cstr`'s own javadoc: "an
intern-cache contract sized for a bounded, `private static final`-held set of literals" -- and
`MLXQuant.checkMode`'s javadoc (guarding `quantize`/`dequantize`'s `mode` parameter before it reaches
`cstr`) states the failure mode explicitly: "a raw pass-through of this caller-supplied parameter
would let a typo'd or dynamically-built string grow `FACADE_ARENA` by one segment per distinct value
ever seen." File paths (`loadSafetensors`/`loadGguf`/`saveSafetensors`/`saveGguf`'s `file` parameter,
bound as plain `const char* file` per `io.h:37,44,51,57` -- not `mlx_string`) and every tensor/metadata
key this facade writes (`mlx_io_gguf_set_array`'s `key`, `mlx_map_string_to_array_insert`'s `key`, one
call per tensor in a checkpoint that may have hundreds) are exactly the unbounded, per-call,
caller-supplied strings `cstr`'s contract excludes -- reusing it here would leak one segment per
distinct path/key for the process lifetime (`FACADE_ARENA` is `Arena.ofShared()`, never closed).
`MLXIO` must allocate every such string directly off the same confined `Arena` already open for that
call (`tmp.allocateFrom(s)`), the same way `MLXQuant`'s own hand-rolled bodies allocate their other
transient out-params -- never through `cstr`, and never through a new cache of its own (there is no
bounded literal set here to cache).

**Reading a string back the other direction -- borrowed `const char*`/`char*` into a Java
`String`** -- has no existing helper (`req/phase5-plan.md`'s Research findings already established
this via repo-wide grep) and needs the standard FFM idiom for a native-owned, length-unknown,
NUL-terminated pointer: `ptr.reinterpret(Long.MAX_VALUE).getString(0)`. `reinterpret` is required
before `getString` because a raw pointer value read out of a struct field (e.g. `keySlot.get(ADDRESS,
0)`) or filled in as an out-param comes back as a zero-length segment with no declared bounds; without
it, `getString` throws `IndexOutOfBoundsException` rather than reading anything. Every accessor this
plan uses this for -- `mlx_string_data`, `mlx_vector_string_get`, `mlx_map_string_to_array_iterator_next`'s
`key`, `mlx_map_string_to_string_get`/`iterator_next` -- returns exactly this shape (all confirmed
borrowed against the actual mlx-c `.cpp` sources, per `req/phase5-plan.md`'s Research findings; this
plan does not re-verify what that document already settled).

**Map iteration is a three-way status, not `NativeOps.checked`'s usual 0-or-throw.** Read directly from
`native/scratch/mlx-c/mlx/c/map.cpp` (the pinned checkout, `fba4470b89073180056c9ea46c443051375f7399`):
`mlx_map_string_to_array_iterator_next` returns `0` on a successful read, `1` on a genuine error (via
`mlx_error`), and `2` once the iterator has already reached the map's `end()` -- a normal
loop-termination signal. `req/phase5-plan.md`'s Research findings flagged this and left open whether
`mlx_map_string_to_string_iterator_next` shares the identical convention; reading
`map.cpp`'s `mlx_map_string_to_string_iterator_next` body directly (same file, the string-to-string
half) confirms it does -- identical `0`/`1`/`2` structure, same `.end()` check. Both loops in this
plan use a small dedicated helper, not `NativeOps.checked`, for exactly this reason (Task 1 below).

**Every one of this plan's seven non-`mlx_array` free functions returns `int`, confirmed against the
generated bindings** (not just the header, since a status-returning C signature can still bind as
`void` if jextract's return-type inference goes wrong): `mlx_io_gguf_free`,
`mlx_map_string_to_array_free`, `mlx_map_string_to_array_iterator_free`,
`mlx_map_string_to_string_free`, `mlx_map_string_to_string_iterator_free`, `mlx_vector_string_free`,
`mlx_string_free` are all `public static int ...(MemorySegment ...)`. Per D2c, every one of these is
swallowed (bare call, discarded return value, no `NativeOps.checked` wrapper) on every exit path,
matching `MLXScope.Holder.closeAll`/`freeOne`'s own unconditional-discard precedent.

## Global Constraints

Numbered as its own list per this codebase's planning-doc convention (`req/plans/phase4-m1-plan.md`
Global Constraint 7 is the test-shape constraint this plan's own Testing section below applies
verbatim; constraints below are the ones specific to this plan, in addition to that project-wide one).

1. **`MLXIO` gets its own `NativeLoader.ensureLoaded()` static guard**, mirroring `MLXGrad`, not
   `MLXQuant` -- `MLXIO` calls `mlx_h.*` directly at every call site (D1: no `NativeOps` op-facade
   helper exists for any of these native functions), so it cannot rely on `NativeOps`'s guard the way
   `MLXQuant` does (`req/phase5-plan.md` §1 already names this split explicitly).
2. **Every one of the five non-`mlx_array` handle types (`mlx_io_gguf`, `mlx_map_string_to_array`,
   `mlx_map_string_to_string`, `mlx_vector_string`, `mlx_string`) is allocated via a confined
   `Arena.ofConfined()` inside a `try`-with-resources, never via an `MLXScope`** (D2a/D2b) -- and freed
   inside that same `try`'s body (not relying on `Arena.close()` to release the native mlx-c object;
   the `Arena` only owns the Java-side memory holding the struct's `ctx` field, not whatever
   native-side resource `ctx` points at).
3. **`loadSafetensors`/`loadGguf` each take an explicit `MLXScope target` parameter** (D2a) -- there is
   no existing `MLXArray` operand to infer a scope from the way `NativeOps.scopeOf` does elsewhere in
   this codebase, the same reason `MLX.array(scope, ...)` takes one explicitly.
4. **Every `mlx_array` extracted out of a loaded map/GGUF handle is pre-allocated via
   `mlx_h.mlx_array_new(target)` before being handed to the read call as an out-param**, exactly
   `MLXQuant.dequantize`'s own `MemorySegment res = mlx_h.mlx_array_new(target); ...; return new
   MLXArray(target, res);` pattern -- no new allocation idiom invented for this plan.
5. **File paths and every write-side key string are allocated per-call from the same confined `Arena`
   already open for that call, never via `NativeOps.cstr`** (Findings above) -- `tmp.allocateFrom(s)`,
   discarded when the `Arena` closes.
6. **Every one of the seven non-`mlx_array` frees (D2c) is swallowed bare, on every exit path,
   `try`/`finally`** -- never routed through `NativeOps.checked`.
7. **Map-iterator loops use a dedicated three-way status helper, not `NativeOps.checked`** (Findings
   above: status `2` is loop termination, not failure).
8. **`saveGguf` builds its own `mlx_io_gguf` via the confined-`Arena` constructor (D2b), populates it
   entry-by-entry (`mlx_io_gguf_set_array`/`set_metadata_*`), then frees it in a `finally` regardless of
   whether `mlx_save_gguf` itself succeeds** (D1's asymmetric-write-path note).

## Native surface this plan uses

| Function | Signature (generated binding) | Used by |
|---|---|---|
| `mlx_load_safetensors` | `int(MemorySegment res_0, MemorySegment res_1, MemorySegment file, MemorySegment s)` | `loadSafetensors` |
| `mlx_save_safetensors` | `int(MemorySegment file, MemorySegment param, MemorySegment metadata)` | `saveSafetensors` |
| `mlx_load_gguf` | `int(MemorySegment gguf, MemorySegment file, MemorySegment s)` | `loadGguf` |
| `mlx_save_gguf` | `int(MemorySegment file, MemorySegment gguf)` | `saveGguf` |
| `mlx_io_gguf_new` | `MemorySegment(SegmentAllocator allocator)` | `saveGguf` (confined `Arena` only, D2b) |
| `mlx_io_gguf_free` | `int(MemorySegment io)` | `loadGguf`, `saveGguf` |
| `mlx_io_gguf_get_keys` | `int(MemorySegment keys, MemorySegment io)` | `loadGguf` |
| `mlx_io_gguf_get_array` | `int(MemorySegment arr, MemorySegment io, MemorySegment key)` | `loadGguf` |
| `mlx_io_gguf_get_metadata_array` | `int(MemorySegment arr, MemorySegment io, MemorySegment key)` | `loadGguf` |
| `mlx_io_gguf_get_metadata_string` | `int(MemorySegment str, MemorySegment io, MemorySegment key)` | `loadGguf` |
| `mlx_io_gguf_get_metadata_vector_string` | `int(MemorySegment vstr, MemorySegment io, MemorySegment key)` | `loadGguf` |
| `mlx_io_gguf_has_metadata_array/_string/_vector_string` | `int(MemorySegment flag, MemorySegment io, MemorySegment key)` | `loadGguf` (dispatch which accessor a key needs) |
| `mlx_io_gguf_set_array` | `int(MemorySegment io, MemorySegment key, MemorySegment arr)` | `saveGguf` |
| `mlx_io_gguf_set_metadata_array/_string/_vector_string` | `int(MemorySegment io, MemorySegment key, MemorySegment v)` | `saveGguf` |
| `mlx_map_string_to_array_new` | `MemorySegment(SegmentAllocator allocator)` | `saveSafetensors`, `saveGguf` fixtures |
| `mlx_map_string_to_array_free` | `int(MemorySegment map)` | all four methods |
| `mlx_map_string_to_array_insert` | `int(MemorySegment map, MemorySegment key, MemorySegment value)` | `saveSafetensors` |
| `mlx_map_string_to_array_iterator_new` | `MemorySegment(SegmentAllocator allocator, MemorySegment map)` | `loadSafetensors` |
| `mlx_map_string_to_array_iterator_free` | `int(MemorySegment it)` | `loadSafetensors` |
| `mlx_map_string_to_array_iterator_next` | `int(MemorySegment key, MemorySegment value, MemorySegment it)` | `loadSafetensors` |
| `mlx_map_string_to_string_*` | same five shapes as above, `MemorySegment`/`String` swapped for values | safetensors metadata |
| `mlx_vector_string_new` | `MemorySegment(SegmentAllocator allocator)` | `saveGguf` (vector-string metadata) |
| `mlx_vector_string_free` | `int(MemorySegment vec)` | `loadGguf`, `saveGguf` |
| `mlx_vector_string_size` | `long(MemorySegment vec)` | `loadGguf` |
| `mlx_vector_string_get` | `int(MemorySegment res, MemorySegment vec, long idx)` | `loadGguf` |
| `mlx_vector_string_append_value` | `int(MemorySegment vec, MemorySegment val)` | `saveGguf` |
| `mlx_string_data` | `MemorySegment(MemorySegment str)` | `loadGguf` (metadata string value) |
| `mlx_string_free` | `int(MemorySegment str)` | `loadGguf` |
| `mlx_array_new` | `MemorySegment(SegmentAllocator allocator)` | every tensor readback (Global Constraint 4) |

## Task 1: `NativeOps` additions (`se.alipsa.jmlx.core`)

Two small, narrowly-scoped helpers, added here because this is the first task that needs them (Findings
above, same M0d discipline `req/phase4-plan.md` already established and `req/phase5-plan.md`'s Research
findings already cite).

### Step 1a: `NativeOps.readCString`

```java
/**
 * Reads a borrowed, NUL-terminated {@code const char*}/{@code char*} into a Java {@code String}.
 * {@code ptr} must outlive this call but is never freed by it -- every accessor this facade uses this
 * for ({@code mlx_string_data}, {@code mlx_vector_string_get}, the two map iterator/get families)
 * returns a pointer into storage some other handle already owns (req/phase5-plan.md's Research
 * findings). {@code reinterpret} is required before {@code getString}: a raw pointer value read out of
 * a struct field or written into an out-param slot comes back as a zero-length segment with no
 * declared bounds, and {@code getString} on one throws {@code IndexOutOfBoundsException} rather than
 * reading anything.
 */
static String readCString(MemorySegment ptr) {
  return ptr.reinterpret(Long.MAX_VALUE).getString(0);
}
```

### Step 1b: `NativeOps.mapIteratorNext` (three-way status, package-private)

Not a native call itself -- a thin status-classifying wrapper so `MLXIO`'s two map-read loops (Task 2,
Task 3) do not each hand-roll the same `0`/`1`/`2` switch. Takes the raw `IntSupplier` the same way
`checked` does, but returns a boolean ("has another entry") instead of throwing on success:

```java
/**
 * Classifies one of the two {@code mlx_map_string_to_*_iterator_next} calls' three-way status
 * (Findings above, confirmed against {@code map.cpp} directly): {@code 0} means the out-params were
 * written and there is a current entry, {@code 2} means the iterator had already reached the map's
 * end (ordinary loop termination, not failure), and anything else is a genuine error routed through
 * the same {@link MLXException} path {@link #checked} uses.
 */
static boolean mapIteratorNext(String opName, IntSupplier nativeCall) {
  NativeLoader.clearLastNativeError();
  int status = nativeCall.getAsInt();
  if (status == 0) {
    return true;
  }
  if (status == 2) {
    return false;
  }
  throw nativeFailure(opName + ": mlx-c call failed with status " + status);
}
```

## Task 2: `MLXIO` — safetensors (`se.alipsa.jmlx.core`)

```java
package se.alipsa.jmlx.core;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import se.alipsa.jmlx.ffi.NativeLoader;
import se.alipsa.jmlx.ffi.mlx_h;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Checkpoint I/O: a thin facade over mlx-c's own safetensors/GGUF load and save, not a hand-rolled
 * parser (req/phase5-plan.md D1). Joins {@link MLXQuant}/{@link MLXRandom}/{@link MLXGrad} as another
 * native-facing facade inside {@code core} (D2) -- see {@link MLX}'s javadoc for the full sibling
 * index, updated by this plan's own documentation task to include this class.
 */
public final class MLXIO {

  private MLXIO() {}

  static {
    NativeLoader.ensureLoaded();
  }

  /** The two maps {@code mlx_load_safetensors}/{@code mlx_save_safetensors} exchange directly. */
  public record SafetensorsResult(Map<String, MLXArray> tensors, Map<String, String> metadata) {}

  public static SafetensorsResult loadSafetensors(MLXScope target, String file) {
    Objects.requireNonNull(target, "loadSafetensors: target must not be null");
    Objects.requireNonNull(file, "loadSafetensors: file must not be null");
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment tensorMap = tmp.allocate(ValueLayout.ADDRESS);
      MemorySegment metaMap = tmp.allocate(ValueLayout.ADDRESS);
      MemorySegment filePath = tmp.allocateFrom(file);
      try {
        NativeOps.checked(
            "loadSafetensors",
            () ->
                mlx_h.mlx_load_safetensors(
                    tensorMap, metaMap, filePath, NativeOps.DEFAULT_STREAM));
        Map<String, MLXArray> tensors = readArrayMap(tensorMap, target, tmp);
        Map<String, String> metadata = readStringMap(metaMap, tmp);
        return new SafetensorsResult(tensors, metadata);
      } finally {
        mlx_h.mlx_map_string_to_array_free(tensorMap);
        mlx_h.mlx_map_string_to_string_free(metaMap);
      }
    }
  }

  public static void saveSafetensors(
      String file, Map<String, MLXArray> tensors, Map<String, String> metadata) {
    Objects.requireNonNull(file, "saveSafetensors: file must not be null");
    Objects.requireNonNull(tensors, "saveSafetensors: tensors must not be null");
    Objects.requireNonNull(metadata, "saveSafetensors: metadata must not be null");
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment tensorMap = mlx_h.mlx_map_string_to_array_new(tmp);
      MemorySegment metaMap = mlx_h.mlx_map_string_to_string_new(tmp);
      try {
        for (Map.Entry<String, MLXArray> e : tensors.entrySet()) {
          MemorySegment key = tmp.allocateFrom(e.getKey());
          NativeOps.checked(
              "saveSafetensors.insert",
              () -> mlx_h.mlx_map_string_to_array_insert(tensorMap, key, e.getValue().handle()));
        }
        for (Map.Entry<String, String> e : metadata.entrySet()) {
          MemorySegment key = tmp.allocateFrom(e.getKey());
          MemorySegment value = tmp.allocateFrom(e.getValue());
          NativeOps.checked(
              "saveSafetensors.insertMetadata",
              () -> mlx_h.mlx_map_string_to_string_insert(metaMap, key, value));
        }
        MemorySegment filePath = tmp.allocateFrom(file);
        NativeOps.checked(
            "saveSafetensors", () -> mlx_h.mlx_save_safetensors(filePath, tensorMap, metaMap));
      } finally {
        mlx_h.mlx_map_string_to_array_free(tensorMap);
        mlx_h.mlx_map_string_to_string_free(metaMap);
      }
    }
  }

  /** Drains an {@code mlx_map_string_to_array} into a Java map, freeing only the iterator it opens. */
  private static Map<String, MLXArray> readArrayMap(
      MemorySegment map, MLXScope target, Arena tmp) {
    Map<String, MLXArray> result = new LinkedHashMap<>();
    MemorySegment it = mlx_h.mlx_map_string_to_array_iterator_new(tmp, map);
    try {
      MemorySegment keySlot = tmp.allocate(ValueLayout.ADDRESS);
      while (NativeOps.mapIteratorNext(
          "loadSafetensors.next",
          () -> {
            MemorySegment valueSlot = mlx_h.mlx_array_new(target);
            int status = mlx_h.mlx_map_string_to_array_iterator_next(keySlot, valueSlot, it);
            if (status == 0) {
              String key = NativeOps.readCString(keySlot.get(ValueLayout.ADDRESS, 0));
              result.put(key, new MLXArray(target, valueSlot));
            }
            return status;
          })) {
        // body runs inside the supplier above; loop only drives iteration
      }
      return result;
    } finally {
      mlx_h.mlx_map_string_to_array_iterator_free(it);
    }
  }

  /** Drains an {@code mlx_map_string_to_string} into a Java map, mirroring {@link #readArrayMap}. */
  private static Map<String, String> readStringMap(MemorySegment map, Arena tmp) {
    Map<String, String> result = new LinkedHashMap<>();
    MemorySegment it = mlx_h.mlx_map_string_to_string_iterator_new(tmp, map);
    try {
      MemorySegment keySlot = tmp.allocate(ValueLayout.ADDRESS);
      MemorySegment valueSlot = tmp.allocate(ValueLayout.ADDRESS);
      while (NativeOps.mapIteratorNext(
          "loadSafetensors.nextMetadata",
          () -> {
            int status = mlx_h.mlx_map_string_to_string_iterator_next(keySlot, valueSlot, it);
            if (status == 0) {
              String key = NativeOps.readCString(keySlot.get(ValueLayout.ADDRESS, 0));
              String value = NativeOps.readCString(valueSlot.get(ValueLayout.ADDRESS, 0));
              result.put(key, value);
            }
            return status;
          })) {
        // body runs inside the supplier above
      }
      return result;
    } finally {
      mlx_h.mlx_map_string_to_string_iterator_free(it);
    }
  }
}
```

**Note on the loop shape above:** `mapIteratorNext`'s `IntSupplier` lambda both drives the native call
*and* records the entry into `result` as a side effect, since the only way to know a read succeeded is
the same status value the loop condition consumes -- there is no clean way to separate "did the read
succeed" from "what did it read" across the three-way status without either a mutable holder (used
here, via the enclosing `result`/local variables) or a small record return type. If review finds the
side-effecting lambda unclear, the alternative is a tiny private `enum ReadStatus { OK, DONE }`-plus-value
record returned from a non-lambda helper method instead -- functionally identical, more verbose. Left as
a call to make once real review feedback exists, not speculatively.

## Task 3: `MLXIO` — GGUF (continues the same class)

```java
  /** GGUF's three-way metadata split (Research findings, req/phase5-plan.md): unlike safetensors'
   * single string-keyed metadata map, GGUF's on-disk format is a small typed-value union. */
  public record GgufResult(
      Map<String, MLXArray> tensors,
      Map<String, MLXArray> metadataArrays,
      Map<String, String> metadataStrings,
      Map<String, java.util.List<String>> metadataVectorStrings) {}

  public static GgufResult loadGguf(MLXScope target, String file) {
    Objects.requireNonNull(target, "loadGguf: target must not be null");
    Objects.requireNonNull(file, "loadGguf: file must not be null");
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment io = tmp.allocate(ValueLayout.ADDRESS);
      MemorySegment filePath = tmp.allocateFrom(file);
      try {
        NativeOps.checked(
            "loadGguf", () -> mlx_h.mlx_load_gguf(io, filePath, NativeOps.DEFAULT_STREAM));

        MemorySegment keys = mlx_h.mlx_vector_string_new(tmp);
        try {
          NativeOps.checked("loadGguf.getKeys", () -> mlx_h.mlx_io_gguf_get_keys(keys, io));
          long n = mlx_h.mlx_vector_string_size(keys);
          Map<String, MLXArray> tensors = new LinkedHashMap<>();
          Map<String, MLXArray> metaArrays = new LinkedHashMap<>();
          Map<String, String> metaStrings = new LinkedHashMap<>();
          Map<String, java.util.List<String>> metaVectorStrings = new LinkedHashMap<>();
          MemorySegment keySlot = tmp.allocate(ValueLayout.ADDRESS);
          MemorySegment flagSlot = tmp.allocate(ValueLayout.JAVA_BOOLEAN);
          for (long i = 0; i < n; i++) {
            NativeOps.checked("loadGguf.getKey", () -> mlx_h.mlx_vector_string_get(keySlot, keys, i));
            String key = NativeOps.readCString(keySlot.get(ValueLayout.ADDRESS, 0));
            MemorySegment keyC = tmp.allocateFrom(key);
            readGgufEntry(io, key, keyC, flagSlot, tmp, target, tensors, metaArrays, metaStrings,
                metaVectorStrings);
          }
          return new GgufResult(tensors, metaArrays, metaStrings, metaVectorStrings);
        } finally {
          mlx_h.mlx_vector_string_free(keys);
        }
      } finally {
        mlx_h.mlx_io_gguf_free(io);
      }
    }
  }

  /**
   * One key's worth of {@code loadGguf} dispatch: {@code mlx_io_gguf_get_keys} does not say which
   * of {@code get_array}/{@code get_metadata_array}/{@code get_metadata_string}/
   * {@code get_metadata_vector_string} a given key belongs under, so each key is probed with the
   * three {@code has_metadata_*} predicates first; a key that matches none of them is a tensor,
   * read via {@code get_array}. Order matters only in that a key cannot legitimately match more
   * than one bucket -- not verified here, since mlx-c's own GGUF writer is assumed to produce
   * well-formed output (this is a read path, not a validator).
   */
  private static void readGgufEntry(
      MemorySegment io,
      String key,
      MemorySegment keyC,
      MemorySegment flagSlot,
      Arena tmp,
      MLXScope target,
      Map<String, MLXArray> tensors,
      Map<String, MLXArray> metaArrays,
      Map<String, String> metaStrings,
      Map<String, java.util.List<String>> metaVectorStrings) {
    NativeOps.checked(
        "loadGguf.hasMetadataArray",
        () -> mlx_h.mlx_io_gguf_has_metadata_array(flagSlot, io, keyC));
    if (flagSlot.get(ValueLayout.JAVA_BOOLEAN, 0)) {
      MemorySegment arr = mlx_h.mlx_array_new(target);
      NativeOps.checked(
          "loadGguf.getMetadataArray", () -> mlx_h.mlx_io_gguf_get_metadata_array(arr, io, keyC));
      metaArrays.put(key, new MLXArray(target, arr));
      return;
    }
    NativeOps.checked(
        "loadGguf.hasMetadataString",
        () -> mlx_h.mlx_io_gguf_has_metadata_string(flagSlot, io, keyC));
    if (flagSlot.get(ValueLayout.JAVA_BOOLEAN, 0)) {
      MemorySegment strSlot = tmp.allocate(ValueLayout.ADDRESS);
      NativeOps.checked(
          "loadGguf.getMetadataString",
          () -> mlx_h.mlx_io_gguf_get_metadata_string(strSlot, io, keyC));
      MemorySegment str = strSlot.get(ValueLayout.ADDRESS, 0);
      try {
        metaStrings.put(key, NativeOps.readCString(mlx_h.mlx_string_data(str)));
      } finally {
        mlx_h.mlx_string_free(str);
      }
      return;
    }
    NativeOps.checked(
        "loadGguf.hasMetadataVectorString",
        () -> mlx_h.mlx_io_gguf_has_metadata_vector_string(flagSlot, io, keyC));
    if (flagSlot.get(ValueLayout.JAVA_BOOLEAN, 0)) {
      MemorySegment vstrSlot = tmp.allocate(ValueLayout.ADDRESS);
      NativeOps.checked(
          "loadGguf.getMetadataVectorString",
          () -> mlx_h.mlx_io_gguf_get_metadata_vector_string(vstrSlot, io, keyC));
      MemorySegment vstr = vstrSlot.get(ValueLayout.ADDRESS, 0);
      try {
        long vn = mlx_h.mlx_vector_string_size(vstr);
        java.util.List<String> values = new java.util.ArrayList<>();
        MemorySegment itemSlot = tmp.allocate(ValueLayout.ADDRESS);
        for (long i = 0; i < vn; i++) {
          long idx = i;
          NativeOps.checked(
              "loadGguf.getVectorStringItem",
              () -> mlx_h.mlx_vector_string_get(itemSlot, vstr, idx));
          values.add(NativeOps.readCString(itemSlot.get(ValueLayout.ADDRESS, 0)));
        }
        metaVectorStrings.put(key, values);
      } finally {
        mlx_h.mlx_vector_string_free(vstr);
      }
      return;
    }
    MemorySegment arr = mlx_h.mlx_array_new(target);
    NativeOps.checked("loadGguf.getArray", () -> mlx_h.mlx_io_gguf_get_array(arr, io, keyC));
    tensors.put(key, new MLXArray(target, arr));
  }

  public static void saveGguf(
      String file,
      Map<String, MLXArray> tensors,
      Map<String, MLXArray> metadataArrays,
      Map<String, String> metadataStrings,
      Map<String, java.util.List<String>> metadataVectorStrings) {
    Objects.requireNonNull(file, "saveGguf: file must not be null");
    Objects.requireNonNull(tensors, "saveGguf: tensors must not be null");
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment io = mlx_h.mlx_io_gguf_new(tmp);
      try {
        for (Map.Entry<String, MLXArray> e : tensors.entrySet()) {
          MemorySegment key = tmp.allocateFrom(e.getKey());
          NativeOps.checked(
              "saveGguf.setArray", () -> mlx_h.mlx_io_gguf_set_array(io, key, e.getValue().handle()));
        }
        for (Map.Entry<String, MLXArray> e : metadataArrays.entrySet()) {
          MemorySegment key = tmp.allocateFrom(e.getKey());
          NativeOps.checked(
              "saveGguf.setMetadataArray",
              () -> mlx_h.mlx_io_gguf_set_metadata_array(io, key, e.getValue().handle()));
        }
        for (Map.Entry<String, String> e : metadataStrings.entrySet()) {
          MemorySegment key = tmp.allocateFrom(e.getKey());
          MemorySegment mstr = mlx_h.mlx_string_new(tmp);
          NativeOps.checked(
              "saveGguf.setStringValue",
              () -> mlx_h.mlx_string_set(mstr, tmp.allocateFrom(e.getValue())));
          NativeOps.checked(
              "saveGguf.setMetadataString", () -> mlx_h.mlx_io_gguf_set_metadata_string(io, key, mstr));
        }
        for (Map.Entry<String, java.util.List<String>> e : metadataVectorStrings.entrySet()) {
          MemorySegment key = tmp.allocateFrom(e.getKey());
          MemorySegment mvstr = mlx_h.mlx_vector_string_new(tmp);
          for (String value : e.getValue()) {
            MemorySegment v = tmp.allocateFrom(value);
            NativeOps.checked(
                "saveGguf.appendVectorStringValue",
                () -> mlx_h.mlx_vector_string_append_value(mvstr, v));
          }
          NativeOps.checked(
              "saveGguf.setMetadataVectorString",
              () -> mlx_h.mlx_io_gguf_set_metadata_vector_string(io, key, mvstr));
        }
        MemorySegment filePath = tmp.allocateFrom(file);
        NativeOps.checked("saveGguf", () -> mlx_h.mlx_save_gguf(filePath, io));
      } finally {
        mlx_h.mlx_io_gguf_free(io);
      }
    }
  }
```

**`mlx_string_set`'s exact signature needs confirming against the generated binding before this task
starts** (`mlx_string_set(MemorySegment str, MemorySegment src)` per the header, but jextract's
`src` parameter type -- another `mlx_string`, or a raw `const char*` -- was not confirmed against
`mlx_h.java` while writing this plan; `string.h`'s own `mlx_string_new_data(SegmentAllocator, const
char* str)` may be the more direct route to build an `mlx_string` from a Java string in one call
instead of `mlx_string_new` + `mlx_string_set`. Task 3's implementer should re-check
`jmlx-ffi/src/main/generated/java/se/alipsa/jmlx/ffi/mlx_h.java`'s exact `mlx_string_new_data`/
`mlx_string_set` signatures before writing this method's body and simplify accordingly -- the code
above is illustrative of the shape (build-then-populate through the same confined `Arena`), not a
guarantee that `mlx_string_set` is the right call.

## Task 4: `MLXIOTest.java` (new file, `jmlx-core/src/test/java/se/alipsa/jmlx/core/`)

Per Global Constraint 7 (`req/plans/phase4-m1-plan.md`): `@EnabledIfNativeAvailable`,
`try (MLXScope scope = new MLXScope())`, hand-computed goldens, element-value assertions.

| Test | Asserts |
|---|---|
| `saveThenLoadSafetensorsRoundTripsTensorValues` | Build 2-3 small `MLXArray`s with known values, `saveSafetensors` to a `@TempDir` file, `loadSafetensors` into a fresh scope, assert every tensor's `toFloatArray()` matches and every key is present. |
| `saveThenLoadSafetensorsRoundTripsMetadata` | Non-empty string metadata map round-trips exactly (key set and values). |
| `loadSafetensorsEmptyMetadataMapIsEmptyNotNull` | A tensor-only save (no metadata) loads back an empty `Map`, not a `null` or a map containing a stray empty-string key (guards against the iterator loop mishandling zero entries). |
| `saveThenLoadGgufRoundTripsTensorsAndAllThreeMetadataKinds` | One tensor, one numeric metadata array, one string metadata value, one string-list metadata value -- all four round-trip and land in the correct one of `GgufResult`'s four maps, not misclassified into a neighboring bucket. |
| `loadGgufMetadataVectorStringPreservesOrder` | A metadata string-list with 3+ distinct values round-trips in the same order (guards against `mlx_vector_string_append_value`/read-back order not matching insertion order). |
| `loadSafetensorsUnknownFileThrowsMLXException` | A nonexistent path throws `MLXException`, not an unchecked native crash or a silent empty result. |
| `loadSafetensorsAllocatesIntoTheGivenScope` | Tensors loaded into a child scope are unusable once only the child (not an ancestor) is closed -- confirms `target` is actually honored, not silently defaulted to some other scope. |

**Real-checkpoint fixture gap (Testing approach in `req/phase5-plan.md`):** none of the tests above
touch an externally-authored safetensors/GGUF file -- every fixture is written by `MLXIO` itself, so
this suite proves internal round-trip consistency only. Per that document's own resolution, real-world
interop (HF's vs. llama.cpp's differing metadata conventions, real quantized dtypes) is explicitly
deferred to M3, where a real checkpoint is unavoidable anyway -- this is not a gap this plan's own test
task should try to close.

## Task 5: Documentation

- `CLAUDE.md`'s "Loading order matters" paragraph: add `MLXIO` as the fifth class with its own
  `NativeLoader.ensureLoaded()` guard, alongside `MLX`/`MLXScope`/`NativeOps`/`MLXGrad`, and explain why
  (calls `mlx_h` directly, same reasoning already given for `MLXGrad`).
- `CLAUDE.md`'s Architecture section class list: add `MLXIO` to the `se.alipsa.jmlx.core` row.
- `MLX`'s own class javadoc (the "This class does not delegate to them" paragraph listing
  `MLXOps`/`MLXShape`/`MLXFast`/`MLXQuant`/`MLXRandom`/`MLXGrad`): add `MLXIO` to that list so the
  sibling index stays complete.
- `req/phase5-plan.md`'s Status table: flip M1's row to `Done` (or `In progress`, if this plan is
  implemented across more than one sitting) once this plan's tasks land, same convention
  `req/phase4-plan.md`'s own Status table already uses.

## Task 6: full verification pass

- `./gradlew :jmlx-core:test --tests "se.alipsa.jmlx.core.MLXIOTest"` — new suite passes on real
  hardware (native tests skip, not fail, if `mlx.metallib` is absent — do not treat a skip as a pass
  without also running it once against real native binaries).
- `./gradlew build` — confirms `jmlx-ffi`/`jmlx-core`/`jmlx-examples` all still compile.
- `./gradlew spotlessCheck checkstyleMain checkstyleTest checkstyleTestFixtures` — formatting/style,
  same gate every prior milestone has run.
- `git diff --exit-code jmlx-ffi/src/main/generated/java` — this plan adds no new native surface (every
  function it uses already exists in the pinned mlx-c commit's generated bindings), so this must stay
  clean; a diff here would mean something accidentally triggered a regen.

## Deliberately not covered by this plan

- Tokenizer integration (M2) and reference models (M3) — both explicitly out of scope for M1
  (`req/phase5-plan.md` D3/D4).
- `mlx_load_safetensors_reader`/`mlx_save_safetensors_writer`/custom `mlx_io_reader`/`mlx_io_writer`
  vtables — this plan only binds the path-based load/save entry points; a caller-supplied stream
  reader/writer has no named consumer yet (same YAGNI reasoning `req/phase5-plan.md`'s Out-of-scope
  section already applies to save-beyond-round-trip).
- Any general-purpose safetensors/GGUF *validation* (malformed-file diagnostics beyond whatever
  `MLXException` message mlx-c's own error handler produces) — this facade surfaces native errors, it
  does not add its own layer of file-format checking.
- A real-checkpoint fixture file (Task 4's own note above) — left to M3 per `req/phase5-plan.md`'s
  Testing approach resolution.
