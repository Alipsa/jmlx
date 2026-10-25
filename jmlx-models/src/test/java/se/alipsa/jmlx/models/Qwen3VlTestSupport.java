package se.alipsa.jmlx.models;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.KVCache;
import se.alipsa.jmlx.vision.ImageDecoder;
import se.alipsa.jmlx.vision.RgbImage;

/** Shared requests and uncached reference computation for WP7 lifecycle tests. */
final class Qwen3VlTestSupport {
  private Qwen3VlTestSupport() {}

  static List<GenerationRequest> requests(int count) throws Exception {
    var directory = Qwen3VlModelFloatGoldenTest.directory().resolve("images");
    RgbImage a = ImageDecoder.decode(directory.resolve("a.png"));
    RgbImage b = ImageDecoder.decode(directory.resolve("b.png"));
    return List.of(
        request(new int[] {12, 10, 13, 3, 4, 5, 6, 7}, List.of(a), count),
        request(new int[] {12, 10, 13, 12, 10, 13, 3, 19, 7}, List.of(a, b), count),
        request(new int[] {3, 12, 10, 13, 4, 5, 12, 10, 13, 6, 7}, List.of(b, a), count),
        request(new int[] {3, 4, 5, 6, 7}, List.of(), count));
  }

  static GenerationRequest request(int[] ids, List<RgbImage> images, int count) {
    return new GenerationRequest(
            ids, GenerationConfig.greedyDefaults(count, Set.of()), CancellationToken.NONE)
        .withImages(images);
  }

  static List<KVCache> caches(Qwen3VlModel model, MLXScope generation) {
    List<KVCache> caches = new ArrayList<>();
    for (int i = 0; i < model.metadata().numHiddenLayers(); i++) {
      caches.add(new KVCache(generation));
    }
    return caches;
  }

  static MLXArray decode(
      Qwen3VlModel model,
      MLXScope activation,
      Qwen3VlPromptPlan plan,
      List<KVCache> caches,
      int token) {
    int position = plan.ropePositionAt(caches.getFirst().nextPosition());
    return model
        .decoderForTest()
        .decode(
            MLX.array(activation, new int[] {token}, new int[] {1, 1}),
            caches,
            model
                .descriptorForTest()
                .multiAxisRope()
                .prepare(activation, 1, 1, new int[][] {{position}, {position}, {position}}),
            null);
  }

  /**
   * Recomputes every image and the whole history with fresh caches. Generated markers never pass
   * through expansion: only original prompt destinations receive visual replacement/DeepStack.
   */
  static float[] fullLogits(
      Qwen3VlModel model, MLXScope scope, GenerationRequest request, int[] history) {
    var plan = model.plan(request.promptTokenIds(), request.images());
    int promptLength = plan.expandedLength();
    int length = promptLength + history.length;
    int[] ids = Arrays.copyOf(plan.expandedIds(), length);
    System.arraycopy(history, 0, ids, promptLength, history.length);
    int[][] positions = {
      Arrays.copyOf(plan.temporalPositions(), length),
      Arrays.copyOf(plan.heightPositions(), length),
      Arrays.copyOf(plan.widthPositions(), length)
    };
    for (int i = promptLength; i < length; i++) {
      for (int[] axis : positions) {
        axis[i] = plan.ropePositionAt(i);
      }
    }
    var features = model.features(CancellationToken.NONE, request.images(), scope);
    int rows = plan.featureDestinations().length;
    int hidden = model.decoderForTest().config().hiddenSize();
    var logits =
        model
            .decoderForTest()
            .stepLogits(
                MLX.array(scope, ids, new int[] {1, length}),
                caches(model, scope),
                new int[] {length},
                features == null
                    ? null
                    : embedded ->
                        VisualFeatureInjection.replaceImageRows(
                            embedded,
                            MLXShape.reshape(features.merged(), new int[] {1, rows, hidden}),
                            plan.featureDestinations()),
                model.descriptorForTest().multiAxisRope().prepare(scope, 1, length, positions),
                features == null
                    ? null
                    : (layer, states) ->
                        layer >= features.deepstackLevels().size()
                            ? states
                            : VisualFeatureInjection.addMaskedFeatures(
                                scope,
                                states,
                                MLXShape.reshape(
                                    features.deepstackLevels().get(layer),
                                    new int[] {1, rows, hidden}),
                                plan.featureDestinations()));
    return logits.toFloatArray();
  }
}
