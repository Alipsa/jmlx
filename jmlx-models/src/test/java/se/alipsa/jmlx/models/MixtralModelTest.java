package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.KVCache;

/** Mixtral checkpoint coverage over a committed Hugging Face reference. */
@Tag("full-float32")
class MixtralModelTest {
  @Test
  @EnabledIfNativeAvailable
  void prefillAndDecodeMatchHuggingFace() throws Exception {
    FamilyReferenceLogits.assertMatches("mixtral", MixtralModel.class);
  }

  /**
   * The Hugging Face golden's 6-token prompt gives 12 expert slots, below {@code
   * SwitchGlu.SORT_THRESHOLD}, so it never reaches the sorted path. A 40-token prefill gives 80
   * slots and does; feeding the same tokens one at a time gives 2 slots per step and stays on the
   * unsorted path. Causal attention makes the two logits equal, so any disagreement is a
   * sorted-path bug at model level.
   */
  @Test
  @EnabledIfNativeAvailable
  void sortedPrefillMatchesTokenByTokenDecode() throws Exception {
    int tokens = 40;
    int[] ids = new int[tokens];
    for (int i = 0; i < tokens; i++) {
      ids[i] = (7 * i + 3) % 120 + 1;
    }
    Path checkpoint =
        Path.of(
            System.getProperty("jmlx.repository.root"),
            "tools",
            "hf-reference",
            "goldens",
            "checkpoints",
            "mixtral");
    try (MLXScope modelScope = new MLXScope();
        MLXScope inference = modelScope.newChild()) {
      MixtralModel model = (MixtralModel) TextGenerationModels.load(modelScope, checkpoint);
      int layers = model.config().numHiddenLayers();
      List<KVCache> prefillCaches = new ArrayList<>();
      List<KVCache> stepCaches = new ArrayList<>();
      for (int i = 0; i < layers; i++) {
        prefillCaches.add(new KVCache(inference));
        stepCaches.add(new KVCache(inference));
      }
      float[] prefill;
      try (MLXScope activation = inference.newChild()) {
        prefill =
            model
                .forward(MLX.array(activation, ids, new int[] {1, tokens}), prefillCaches)
                .toFloatArray();
      }
      int vocab = prefill.length / tokens;
      for (int t = 0; t < tokens; t++) {
        try (MLXScope activation = inference.newChild()) {
          MLXArray logits =
              model.forward(
                  MLX.array(activation, new int[] {ids[t]}, new int[] {1, 1}), stepCaches);
          float[] expected = new float[vocab];
          System.arraycopy(prefill, t * vocab, expected, 0, vocab);
          assertArrayEquals(expected, logits.toFloatArray(), 1e-4f, "position " + t);
        }
      }
      assertEquals(tokens, prefillCaches.getFirst().offset());
      assertEquals(tokens, stepCaches.getFirst().offset());
    }
  }
}
