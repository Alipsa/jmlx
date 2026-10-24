package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.KVCache;

/**
 * Strict full-float32 text-only equivalence: the composite's multi-axis-prefill logits must match a
 * bare Qwen3-family decoder built from the same re-keyed text weights with the family's plain base
 * RoPE. A text-only Qwen3-VL prompt plans the same coordinate on all three axes, so the request's
 * {@code MultiAxisRope} rotation and the family's {@code RopeSpec.Base} rotation describe one and
 * the same angle table (the tiny checkpoint's BF16 storage is up-cast to FLOAT32 on both sides
 * before any module is built).
 */
@Tag("full-float32")
@EnabledIfNativeAvailable
class Qwen3VlTextOnlyEquivalenceTest {

  private static final int[] IDS = {1, 2, 3, 14, 5, 7};

  /** The Qwen3-family decoder the reference side runs, built from re-keyed text weights. */
  private static final class ReferenceDecoder extends DecoderModel {
    ReferenceDecoder(
        MLXScope scope, ArchitectureDescriptor descriptor, Map<String, MLXArray> weights) {
      super(scope, descriptor, weights);
    }
  }

  private static MLXArray upcast(MLXArray value) {
    return value.dtype() == DType.FLOAT32 ? value : MLX.astype(value, DType.FLOAT32);
  }

  @Test
  void textOnlyPrefillMatchesTheBareQwen3Decoder(@TempDir Path dir) throws Exception {
    TinyCheckpoints.randomQwen3Vl(dir, 2024L);
    try (MLXScope root = new MLXScope()) {
      Qwen3VlModel model = Qwen3VlModel.load(root, dir);
      float[] composite;
      try (MLXScope generation = root.newChild()) {
        composite = model.textOnlyPrefillLogitsForTest(generation, IDS);
      }
      assertEquals(21, composite.length);
      Qwen3VlDescriptor d = model.descriptorForTest();
      try (MLXScope reference = root.newChild()) {
        // A fresh, independent load of the same checkpoint: the text half, re-keyed from
        // model.language_model.* to model.* and up-cast, feeds the family decoder.
        Map<String, MLXArray> loaded = CheckpointLoader.load(reference, dir, d.tensorPlan());
        Map<String, MLXArray> text = new LinkedHashMap<>();
        for (Map.Entry<String, MLXArray> entry : loaded.entrySet()) {
          String key = entry.getKey();
          if (!key.startsWith("model.language_model.") && !key.equals("lm_head.weight")) {
            continue; // the visual half
          }
          String textKey =
              key.startsWith("model.language_model.")
                  ? "model." + key.substring("model.language_model.".length())
                  : key;
          text.put(textKey, upcast(entry.getValue()));
        }
        ReferenceDecoder decoder = new ReferenceDecoder(reference, d.text(), text);
        List<KVCache> caches = new ArrayList<>();
        for (int i = 0; i < d.text().dimensions().numHiddenLayers(); i++) {
          caches.add(new KVCache(reference));
        }
        MLXArray ids = MLX.array(reference, IDS, new int[] {1, IDS.length});
        MLXArray logits = decoder.stepLogits(ids, caches, new int[] {IDS.length}, null);
        float[] plain = logits.toFloatArray();
        assertEquals(21, plain.length);
        float max = 0f;
        for (int i = 0; i < composite.length; i++) {
          max = Math.max(max, Math.abs(composite[i] - plain[i]));
        }
        final float maxDiff = max;
        assertArrayEquals(plain, composite, 0.0f, () -> "max |diff| = " + maxDiff);
      }
    }
  }
}
