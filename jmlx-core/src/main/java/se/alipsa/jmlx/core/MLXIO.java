package se.alipsa.jmlx.core;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import se.alipsa.jmlx.ffi.NativeLoader;
import se.alipsa.jmlx.ffi.mlx_h;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Checkpoint I/O: a thin facade over mlx-c's own safetensors/GGUF load and save, not a hand-rolled
 * parser (req/phase5-plan.md D1). Joins {@link MLXQuant}/{@link MLXRandom}/{@link MLXGrad} as
 * another native-facing facade inside {@code core} (D2) -- see {@link MLX}'s javadoc for the full
 * sibling index.
 */
public final class MLXIO {

  private MLXIO() {}

  static {
    NativeLoader.ensureLoaded();
  }

  /** The two maps {@code mlx_load_safetensors}/{@code mlx_save_safetensors} exchange directly. */
  public record SafetensorsResult(Map<String, MLXArray> tensors, Map<String, String> metadata) {}

  /**
   * Loads every tensor and metadata entry from a safetensors file, allocated into {@code target}.
   */
  public static SafetensorsResult loadSafetensors(MLXScope target, String file) {
    Objects.requireNonNull(target, "loadSafetensors: target must not be null");
    Objects.requireNonNull(file, "loadSafetensors: file must not be null");
    try (Arena tmp = Arena.ofConfined()) {
      // Built via the proper constructor, not a raw allocated slot -- mlx_load_safetensors writes
      // through mlx_map_string_to_array_set_/mlx_map_string_to_string_set_, both of which branch on
      // whether ctx is already non-null (req/plans/phase5-m1-plan.md's Findings); this couples the
      // call to mlx-c's own public constructors rather than to how those private helpers happen to
      // treat a zeroed slot today.
      MemorySegment tensorMap = mlx_h.mlx_map_string_to_array_new(tmp);
      MemorySegment metaMap = mlx_h.mlx_map_string_to_string_new(tmp);
      MemorySegment filePath = tmp.allocateFrom(file);
      try {
        NativeOps.checked(
            "loadSafetensors",
            () ->
                mlx_h.mlx_load_safetensors(
                    tensorMap, metaMap, filePath, NativeOps.DEFAULT_CPU_STREAM));
        Map<String, MLXArray> tensors = readArrayMap(tensorMap, target, tmp);
        Map<String, String> metadata = readStringMap(metaMap, tmp);
        return new SafetensorsResult(tensors, metadata);
      } finally {
        mlx_h.mlx_map_string_to_array_free(tensorMap);
        mlx_h.mlx_map_string_to_string_free(metaMap);
      }
    }
  }

  /** Writes every tensor and metadata entry to a safetensors file at {@code file}. */
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

  /**
   * Drains an {@code mlx_map_string_to_array} into a Java map. {@code valueScratch} is read into
   * via a scratch {@code mlx_array} (built with {@code tmp}, never {@code target}) reused across
   * every iteration -- built once, before the loop, rather than fresh per call, precisely so the
   * loop's terminating {@code iterator_next} call (status {@code 2}, returns before ever touching
   * {@code *value}) never wastes a {@code target}-tracked allocation nobody will use. Only once an
   * iteration's status confirms a real entry does a fresh {@code target}-tracked array get built
   * and the scratch's current contents copied into it via the public {@code mlx_array_set} --
   * {@code valueScratch} itself is freed once the whole loop ends, since it is never {@code
   * MLXScope}-tracked and nothing else will release whatever it last held.
   */
  private static Map<String, MLXArray> readArrayMap(MemorySegment map, MLXScope target, Arena tmp) {
    Map<String, MLXArray> result = new LinkedHashMap<>();
    MemorySegment it = mlx_h.mlx_map_string_to_array_iterator_new(tmp, map);
    MemorySegment valueScratch = mlx_h.mlx_array_new(tmp);
    try {
      MemorySegment keySlot = tmp.allocate(ValueLayout.ADDRESS);
      while (NativeOps.mapIteratorNext(
          "loadSafetensors.next",
          () -> mlx_h.mlx_map_string_to_array_iterator_next(keySlot, valueScratch, it))) {
        String key = NativeOps.readNativeString(keySlot.get(ValueLayout.ADDRESS, 0));
        MemorySegment arr = mlx_h.mlx_array_new(target);
        NativeOps.checked(
            "loadSafetensors.next.copy", () -> mlx_h.mlx_array_set(arr, valueScratch));
        result.put(key, new MLXArray(target, arr));
      }
      return result;
    } finally {
      mlx_h.mlx_array_free(valueScratch);
      mlx_h.mlx_map_string_to_array_iterator_free(it);
    }
  }

  /**
   * Drains an {@code mlx_map_string_to_string} into a Java map. No scratch-array dance needed here
   * (unlike {@link #readArrayMap}): both {@code key} and {@code value} are plain {@code const
   * char**} out-params, not struct out-params with their own null-ctx hazard, so there is nothing
   * to allocate-then-copy -- {@code NativeOps.readNativeString} reads both directly once the status
   * confirms a real entry.
   */
  private static Map<String, String> readStringMap(MemorySegment map, Arena tmp) {
    Map<String, String> result = new LinkedHashMap<>();
    MemorySegment it = mlx_h.mlx_map_string_to_string_iterator_new(tmp, map);
    try {
      MemorySegment keySlot = tmp.allocate(ValueLayout.ADDRESS);
      MemorySegment valueSlot = tmp.allocate(ValueLayout.ADDRESS);
      while (NativeOps.mapIteratorNext(
          "loadSafetensors.nextMetadata",
          () -> mlx_h.mlx_map_string_to_string_iterator_next(keySlot, valueSlot, it))) {
        String key = NativeOps.readNativeString(keySlot.get(ValueLayout.ADDRESS, 0));
        String value = NativeOps.readNativeString(valueSlot.get(ValueLayout.ADDRESS, 0));
        result.put(key, value);
      }
      return result;
    } finally {
      mlx_h.mlx_map_string_to_string_iterator_free(it);
    }
  }

  /**
   * GGUF's tensor and metadata keys occupy two separate namespaces read through separate accessors,
   * not one key space dispatched by probing -- an assumption this facade's first draft got wrong,
   * amended in req/plans/phase5-m1-plan.md once testing against the real runtime surfaced it.
   * {@code mlx_io_gguf_get_keys} enumerates only the tensor map (confirmed against {@code
   * mlx_io_gguf_get_keys}'s own body in {@code io_types.cpp}, which iterates {@code
   * mlx_io_gguf_get_(io).first} specifically); there is no equivalent enumerator for metadata.
   * GGUF's on-disk metadata keys follow the format's own well-known vocabulary ({@code
   * general.name}, {@code tokenizer.ggml.tokens}, etc.) that a caller -- typically a model loader
   * that already knows which keys its architecture defines -- supplies by name, mirroring how
   * {@code mlx_c/examples/example-gguf.c} itself only ever probes {@code has_metadata_*} against a
   * key it already has in hand. An empty set for a metadata kind the caller doesn't need is the
   * normal case, not a workaround.
   */
  public record GgufResult(
      Map<String, MLXArray> tensors,
      Map<String, MLXArray> metadataArrays,
      Map<String, String> metadataStrings,
      Map<String, List<String>> metadataVectorStrings) {}

  /**
   * Loads every tensor from a GGUF file, allocated into {@code target}, plus whichever of the
   * caller-named {@code metadata*Keys} the file actually has -- see {@link GgufResult}'s own
   * javadoc for why metadata keys must be named rather than discovered.
   */
  public static GgufResult loadGguf(
      MLXScope target,
      String file,
      Set<String> metadataArrayKeys,
      Set<String> metadataStringKeys,
      Set<String> metadataVectorStringKeys) {
    Objects.requireNonNull(target, "loadGguf: target must not be null");
    Objects.requireNonNull(file, "loadGguf: file must not be null");
    Objects.requireNonNull(metadataArrayKeys, "loadGguf: metadataArrayKeys must not be null");
    Objects.requireNonNull(metadataStringKeys, "loadGguf: metadataStringKeys must not be null");
    Objects.requireNonNull(
        metadataVectorStringKeys, "loadGguf: metadataVectorStringKeys must not be null");
    try (Arena tmp = Arena.ofConfined()) {
      // mlx_h.mlx_io_gguf_new(tmp), not a raw allocated slot -- mlx_load_gguf writes through
      // mlx_io_gguf_set_, which unconditionally `delete`s whatever ctx already holds before
      // replacing it (req/plans/phase5-m1-plan.md's Findings).
      MemorySegment io = mlx_h.mlx_io_gguf_new(tmp);
      MemorySegment filePath = tmp.allocateFrom(file);
      try {
        NativeOps.checked(
            "loadGguf", () -> mlx_h.mlx_load_gguf(io, filePath, NativeOps.DEFAULT_CPU_STREAM));
        Map<String, MLXArray> tensors = readGgufTensors(io, target, tmp);
        Map<String, MLXArray> metaArrays =
            readGgufMetadataArrays(io, target, tmp, metadataArrayKeys);
        Map<String, String> metaStrings = readGgufMetadataStrings(io, tmp, metadataStringKeys);
        Map<String, List<String>> metaVectorStrings =
            readGgufMetadataVectorStrings(io, tmp, metadataVectorStringKeys);
        return new GgufResult(tensors, metaArrays, metaStrings, metaVectorStrings);
      } finally {
        mlx_h.mlx_io_gguf_free(io);
      }
    }
  }

  private static Map<String, MLXArray> readGgufTensors(
      MemorySegment io, MLXScope target, Arena tmp) {
    Map<String, MLXArray> tensors = new LinkedHashMap<>();
    MemorySegment keys = mlx_h.mlx_vector_string_new(tmp);
    try {
      NativeOps.checked("loadGguf.getKeys", () -> mlx_h.mlx_io_gguf_get_keys(keys, io));
      long n = mlx_h.mlx_vector_string_size(keys);
      MemorySegment keySlot = tmp.allocate(ValueLayout.ADDRESS);
      for (long i = 0; i < n; i++) {
        long idx = i; // i itself is mutated by the loop, not effectively final -- cannot be
        // captured by the lambda below without this copy.
        NativeOps.checked("loadGguf.getKey", () -> mlx_h.mlx_vector_string_get(keySlot, keys, idx));
        String key = NativeOps.readNativeString(keySlot.get(ValueLayout.ADDRESS, 0));
        MemorySegment keyC = tmp.allocateFrom(key);
        MemorySegment arr = mlx_h.mlx_array_new(target);
        NativeOps.checked("loadGguf.getArray", () -> mlx_h.mlx_io_gguf_get_array(arr, io, keyC));
        tensors.put(key, new MLXArray(target, arr));
      }
      return tensors;
    } finally {
      mlx_h.mlx_vector_string_free(keys);
    }
  }

  private static Map<String, MLXArray> readGgufMetadataArrays(
      MemorySegment io, MLXScope target, Arena tmp, Set<String> keys) {
    Map<String, MLXArray> result = new LinkedHashMap<>();
    MemorySegment flagSlot = tmp.allocate(ValueLayout.JAVA_BOOLEAN);
    for (String key : keys) {
      MemorySegment keyC = tmp.allocateFrom(key);
      NativeOps.hasMetadataProbe(
          "loadGguf.hasMetadataArray",
          () -> mlx_h.mlx_io_gguf_has_metadata_array(flagSlot, io, keyC));
      if (!flagSlot.get(ValueLayout.JAVA_BOOLEAN, 0)) {
        continue;
      }
      MemorySegment arr = mlx_h.mlx_array_new(target);
      NativeOps.checked(
          "loadGguf.getMetadataArray", () -> mlx_h.mlx_io_gguf_get_metadata_array(arr, io, keyC));
      result.put(key, new MLXArray(target, arr));
    }
    return result;
  }

  private static Map<String, String> readGgufMetadataStrings(
      MemorySegment io, Arena tmp, Set<String> keys) {
    Map<String, String> result = new LinkedHashMap<>();
    MemorySegment flagSlot = tmp.allocate(ValueLayout.JAVA_BOOLEAN);
    for (String key : keys) {
      MemorySegment keyC = tmp.allocateFrom(key);
      NativeOps.hasMetadataProbe(
          "loadGguf.hasMetadataString",
          () -> mlx_h.mlx_io_gguf_has_metadata_string(flagSlot, io, keyC));
      if (!flagSlot.get(ValueLayout.JAVA_BOOLEAN, 0)) {
        continue;
      }
      // mlx_h.mlx_string_new(tmp), not a raw allocated slot -- get_metadata_string writes through
      // mlx_string_set_, same null-ctx reasoning as mlx_load_gguf's io above. strSlot IS the
      // mlx_string handle once populated: mlx_string_data/mlx_string_free take mlx_string BY VALUE,
      // so jextract binds them to take this same struct-shaped segment directly -- passing
      // strSlot.get(ADDRESS, 0) here would dereference one level too far.
      MemorySegment strSlot = mlx_h.mlx_string_new(tmp);
      try {
        NativeOps.checked(
            "loadGguf.getMetadataString",
            () -> mlx_h.mlx_io_gguf_get_metadata_string(strSlot, io, keyC));
        result.put(key, NativeOps.readNativeString(mlx_h.mlx_string_data(strSlot)));
      } finally {
        mlx_h.mlx_string_free(strSlot);
      }
    }
    return result;
  }

  private static Map<String, List<String>> readGgufMetadataVectorStrings(
      MemorySegment io, Arena tmp, Set<String> keys) {
    Map<String, List<String>> result = new LinkedHashMap<>();
    MemorySegment flagSlot = tmp.allocate(ValueLayout.JAVA_BOOLEAN);
    for (String key : keys) {
      MemorySegment keyC = tmp.allocateFrom(key);
      NativeOps.hasMetadataProbe(
          "loadGguf.hasMetadataVectorString",
          () -> mlx_h.mlx_io_gguf_has_metadata_vector_string(flagSlot, io, keyC));
      if (!flagSlot.get(ValueLayout.JAVA_BOOLEAN, 0)) {
        continue;
      }
      // Same reasoning as strSlot above: mlx_h.mlx_vector_string_new(tmp), and vstrSlot is passed
      // directly to mlx_vector_string_size/get/free (all by-value) -- never dereferenced through
      // .get(ADDRESS, 0) first.
      MemorySegment vstrSlot = mlx_h.mlx_vector_string_new(tmp);
      try {
        NativeOps.checked(
            "loadGguf.getMetadataVectorString",
            () -> mlx_h.mlx_io_gguf_get_metadata_vector_string(vstrSlot, io, keyC));
        long vn = mlx_h.mlx_vector_string_size(vstrSlot);
        List<String> values = new ArrayList<>();
        MemorySegment itemSlot = tmp.allocate(ValueLayout.ADDRESS);
        for (long i = 0; i < vn; i++) {
          long idx = i; // same effectively-final requirement as loadGguf's own key loop above.
          NativeOps.checked(
              "loadGguf.getVectorStringItem",
              () -> mlx_h.mlx_vector_string_get(itemSlot, vstrSlot, idx));
          values.add(NativeOps.readNativeString(itemSlot.get(ValueLayout.ADDRESS, 0)));
        }
        result.put(key, values);
      } finally {
        mlx_h.mlx_vector_string_free(vstrSlot);
      }
    }
    return result;
  }

  /** Writes every tensor and all three kinds of metadata to a GGUF file at {@code file}. */
  public static void saveGguf(
      String file,
      Map<String, MLXArray> tensors,
      Map<String, MLXArray> metadataArrays,
      Map<String, String> metadataStrings,
      Map<String, List<String>> metadataVectorStrings) {
    Objects.requireNonNull(file, "saveGguf: file must not be null");
    Objects.requireNonNull(tensors, "saveGguf: tensors must not be null");
    Objects.requireNonNull(metadataArrays, "saveGguf: metadataArrays must not be null");
    Objects.requireNonNull(metadataStrings, "saveGguf: metadataStrings must not be null");
    Objects.requireNonNull(
        metadataVectorStrings, "saveGguf: metadataVectorStrings must not be null");
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment io = mlx_h.mlx_io_gguf_new(tmp);
      try {
        for (Map.Entry<String, MLXArray> e : tensors.entrySet()) {
          MemorySegment key = tmp.allocateFrom(e.getKey());
          NativeOps.checked(
              "saveGguf.setArray",
              () -> mlx_h.mlx_io_gguf_set_array(io, key, e.getValue().handle()));
        }
        for (Map.Entry<String, MLXArray> e : metadataArrays.entrySet()) {
          MemorySegment key = tmp.allocateFrom(e.getKey());
          NativeOps.checked(
              "saveGguf.setMetadataArray",
              () -> mlx_h.mlx_io_gguf_set_metadata_array(io, key, e.getValue().handle()));
        }
        for (Map.Entry<String, String> e : metadataStrings.entrySet()) {
          // mlx_io_gguf_set_metadata_string(mlx_io_gguf, const char* key, const char* mstr) takes
          // mstr as a plain C string, not an mlx_string -- no mlx_string_new/_set round-trip needed
          // at all, unlike set_metadata_vector_string below, whose mvstr genuinely is an
          // mlx_vector_string by value.
          MemorySegment key = tmp.allocateFrom(e.getKey());
          MemorySegment value = tmp.allocateFrom(e.getValue());
          NativeOps.checked(
              "saveGguf.setMetadataString",
              () -> mlx_h.mlx_io_gguf_set_metadata_string(io, key, value));
        }
        for (Map.Entry<String, List<String>> e : metadataVectorStrings.entrySet()) {
          // mlx_io_gguf_set_metadata_vector_string copies mvstr's contents into the map rather than
          // adopting it -- unlike set_array/set_metadata_array's MLXArray.handle() inputs, which
          // the
          // caller's own MLXScope already owns, mvstr is allocated fresh right here and nothing
          // else
          // will ever free it; the try/finally is load-bearing, not just hygiene, since both
          // checked
          // calls below can throw before setMetadataVectorString's own call runs.
          MemorySegment key = tmp.allocateFrom(e.getKey());
          MemorySegment mvstr = mlx_h.mlx_vector_string_new(tmp);
          try {
            for (String value : e.getValue()) {
              MemorySegment v = tmp.allocateFrom(value);
              NativeOps.checked(
                  "saveGguf.appendVectorStringValue",
                  () -> mlx_h.mlx_vector_string_append_value(mvstr, v));
            }
            NativeOps.checked(
                "saveGguf.setMetadataVectorString",
                () -> mlx_h.mlx_io_gguf_set_metadata_vector_string(io, key, mvstr));
          } finally {
            mlx_h.mlx_vector_string_free(mvstr);
          }
        }
        MemorySegment filePath = tmp.allocateFrom(file);
        NativeOps.checked("saveGguf", () -> mlx_h.mlx_save_gguf(filePath, io));
      } finally {
        mlx_h.mlx_io_gguf_free(io);
      }
    }
  }
}
