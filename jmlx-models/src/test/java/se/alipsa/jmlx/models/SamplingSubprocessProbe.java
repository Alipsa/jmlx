package se.alipsa.jmlx.models;

import java.nio.file.Path;
import java.util.OptionalLong;
import java.util.Set;
import java.util.stream.IntStream;
import se.alipsa.jmlx.memory.MLXScope;

/** Fresh-JVM entry point used only by {@link LlamaModelTest}. */
final class SamplingSubprocessProbe {
  private SamplingSubprocessProbe() {}

  public static void main(String[] args) throws Exception {
    Path modelDirectory = Path.of(args[0]);
    long seed = Long.parseLong(args[1]);
    int maxNewTokens = Integer.parseInt(args[2]);
    try (MLXScope scope = new MLXScope()) {
      LlamaModel model = LlamaModel.load(scope, modelDirectory);
      GenerationConfig policy =
          new GenerationConfig(
              maxNewTokens, OptionalLong.of(seed), 1, 0, 1, 0, 1, 0, 0, Set.of(), Set.of(), true);
      GenerationResult result =
          model.generate(
              new GenerationRequest(new int[] {1}, policy, CancellationToken.NONE), ignored -> {});
      String canonical =
          IntStream.range(0, result.generatedTokenIds().size())
              .mapToObj(
                  index ->
                      result.generatedTokenIds().get(index)
                          + ","
                          + result.logProbabilities().get(index))
              .reduce((left, right) -> left + ";" + right)
              .orElse("");
      System.out.println(canonical);
    }
  }
}
