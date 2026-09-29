package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.KVCache;
import se.alipsa.jmlx.tokenizer.HfTokenizer;

/** Opt-in checkpoint check; exact IDs are asserted only for a recorded device/runtime pin. */
class TierBSmokeTest {
  private static final String PROMPT = "Hello, my name is";

  @Test
  void loadsRendersAndGenerates() throws Exception {
    String directory = System.getProperty("jmlx.tier.b.model.dir", "");
    assumeTrue(!directory.isBlank(), "Set JMLX_TIER_B_MODEL_DIR to run Tier B");
    Path modelDirectory = Path.of(directory);
    HfTokenizer tokenizer = HfTokenizer.fromDirectory(modelDirectory);
    String rendered =
        tokenizer.renderChat(
            List.of(java.util.Map.of("role", "user", "content", PROMPT)),
            se.alipsa.jmlx.tokenizer.ChatTemplateOptions.defaults(false));
    assertFalse(rendered.isBlank());
    List<Integer> promptIds = tokenizer.encode(rendered, false);
    assertFalse(promptIds.isEmpty());
    int[] ids = promptIds.stream().mapToInt(Integer::intValue).toArray();

    try (MLXScope scope = new MLXScope();
        MLXScope activation = scope.newChild()) {
      TextGenerationModel model = TextGenerationModels.load(scope, modelDirectory);
      String expectedType = System.getProperty("jmlx.tier.b.expected.model.type", "");
      if (!expectedType.isBlank()) {
        assertEquals(expectedType, model.metadata().modelType());
      }
      assertTrue(model.metadata().vocabSize() >= tokenizer.vocabSize());
      DecoderModel decoder = (DecoderModel) model;
      List<KVCache> caches = new ArrayList<>();
      for (int i = 0; i < model.metadata().numHiddenLayers(); i++) {
        caches.add(new KVCache(activation));
      }
      MLXArray logits =
          decoder.forward(MLX.array(activation, ids, new int[] {1, ids.length}), caches);
      for (float value : logits.toFloatArray()) {
        assertFalse(Float.isNaN(value), "NaN logit");
      }
      GenerationResult result =
          model.generate(
              GenerationRequest.text(
                  tokenizer,
                  PROMPT,
                  PromptSpecialTokens.ADD,
                  GenerationConfig.greedyDefaults(16, Set.of()),
                  CancellationToken.NONE),
              ignored -> {});
      assertEquals(16, result.generatedTokenIds().size());
      assertFalse(result.generatedText().isBlank());
      assertFalse(tokenizer.decode(result.generatedTokenIds(), false).isBlank());
      System.out.println(
          "Tier B observed: model_type="
              + model.metadata().modelType()
              + ", os="
              + System.getProperty("os.name")
              + " "
              + System.getProperty("os.version")
              + ", arch="
              + System.getProperty("os.arch")
              + ", java="
              + Runtime.version()
              + ", generated_ids="
              + result.generatedTokenIds());

      String pin = System.getProperty("jmlx.tier.b.pin", "");
      String recordedPin = System.getProperty("jmlx.tier.b.recorded.pin", "");
      String expectedTokens = System.getProperty("jmlx.tier.b.expected.tokens", "");
      if (!recordedPin.isBlank() && recordedPin.equals(pin) && !expectedTokens.isBlank()) {
        List<Integer> expected =
            Arrays.stream(expectedTokens.split(",")).map(Integer::parseInt).toList();
        assertEquals(expected, result.generatedTokenIds());
      } else {
        System.err.println(
            "Tier B structural pass only: device/macOS/MLX pin differs or exact IDs are"
                + " unrecorded");
      }
    }
  }
}
