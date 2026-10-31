package se.alipsa.jmlx.models;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.KVCache;
import se.alipsa.jmlx.tokenizer.HfTokenizer;
import se.alipsa.jmlx.vision.ImageDecoder;
import se.alipsa.jmlx.vision.RgbImage;

/**
 * Shared fixtures and requests for the WP7 lifecycle tests: the WP6 golden checkpoint's three
 * fixture images, its tiny tokenizer, and the mixed request set (text-only plus one- and two-image
 * prompts whose marker counts match their images). Every image request's prompt carries the markers
 * in order, since the composite model replaces exactly the marker rows with the projected rows.
 */
final class Phi3VTestSupport {
  private Phi3VTestSupport() {}

  static Path directory() {
    return Phi3VGoldenTest.directory();
  }

  static HfTokenizer tokenizer() throws Exception {
    return HfTokenizer.fromDirectory(directory());
  }

  static RgbImage wide() throws Exception {
    return ImageDecoder.decode(directory().resolve("images/wide-672x336.png"));
  }

  static RgbImage square() throws Exception {
    return ImageDecoder.decode(directory().resolve("images/square-336x336.png"));
  }

  static RgbImage tall() throws Exception {
    return ImageDecoder.decode(directory().resolve("images/tall-336x672.png"));
  }

  /** One tokenizer-backed request (the composite model requires retained prompt text). */
  static GenerationRequest request(
      HfTokenizer tokenizer, String prompt, List<RgbImage> images, int maxNewTokens) {
    return GenerationRequest.text(
            tokenizer,
            prompt,
            PromptSpecialTokens.ADD,
            GenerationConfig.greedyDefaults(maxNewTokens, Set.of()),
            CancellationToken.NONE)
        .withImages(images);
  }

  /**
   * The five mixed requests at the given budget: text-only (60), wide (465), tall (481), square
   * (768) and two-image wide-then-square (1229) expanded prompts. All budgets are small enough that
   * no request's generation crosses the 470 boundary (the wide request ends at 469).
   */
  static List<GenerationRequest> requests(int maxNewTokens) throws Exception {
    HfTokenizer tokenizer = tokenizer();
    List<RgbImage> none = List.of();
    return List.of(
        request(tokenizer, "toast ".repeat(10), none, maxNewTokens),
        request(tokenizer, "c d <|image_1|> e s", List.of(wide()), maxNewTokens),
        request(tokenizer, "toast <|image_1|>toast ", List.of(tall()), maxNewTokens),
        request(tokenizer, "toast <|image_1|>toast", List.of(square()), maxNewTokens),
        request(
            tokenizer,
            "toast <|image_1|>toast <|image_2|>toast",
            List.of(wide(), square()),
            maxNewTokens));
  }

  /**
   * One caller-owned {@link KVCache} per decoder layer, as the production step loop builds them.
   */
  static List<KVCache> caches(Phi3VModel model, MLXScope generation) {
    List<KVCache> caches = new ArrayList<>();
    for (int i = 0; i < model.metadata().numHiddenLayers(); i++) {
      caches.add(new KVCache(generation));
    }
    return caches;
  }

  /**
   * The request's retained projection rows in bytes: one FLOAT32 row of decoder hidden width per
   * sentinel destination, held in the request's generation scope for the whole generation.
   */
  static long projectedRowBytes(Phi3VPromptPlan plan, int hiddenSize) {
    return (long) plan.totalSentinelRows() * hiddenSize * Float.BYTES;
  }
}
