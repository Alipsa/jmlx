package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.memory.MLXScope;
import tools.jackson.databind.ObjectMapper;

/** Opt-in counter for actual public generation: lazy graph, all-roots boundary and readback. */
@Tag("qwen35-counter")
class Qwen35DecodeCounterTest {
  @Test
  void fullHybridDecodeHasNoInternalEvalAndOneAllRootsBoundary() throws Exception {
    final List<Map<String, Object>> samples = new ArrayList<>();
    final AtomicBoolean decoding = new AtomicBoolean();
    final AtomicBoolean readback = new AtomicBoolean();
    try (MLXScope scope = new MLXScope()) {
      Qwen35Model model =
          Qwen35Model.load(scope, Path.of("..", "tools", "qwen35-reference", "fixtures", "hf"));
      model.observerForTest(
          stage -> {
            if (stage.equals("decode")) {
              decoding.set(true);
              agent("start");
            }
          });
      model.evaluatorForTest(
          roots -> {
            if (!decoding.get()) {
              MLX.eval(roots);
              return;
            }
            Map<String, Long> internal = stop();
            assertEquals(0L, internal.getOrDefault("mlx_eval", 0L));
            assertEquals(0L, internal.getOrDefault("mlx_array_eval", 0L));
            agent("start");
            MLX.eval(roots);
            Map<String, Long> boundary = stop();
            assertEquals(1L, boundary.getOrDefault("mlx_eval", 0L));
            Map<String, Object> sample = new LinkedHashMap<>();
            sample.put("lazy_decode_and_sampling", internal);
            sample.put("explicit_all_roots_boundary", boundary);
            samples.add(sample);
            agent("start");
            readback.set(true);
          });
      var request =
          new GenerationRequest(
              new int[] {1, 3, 4, 5},
              GenerationConfig.greedyDefaults(3, Set.of()),
              CancellationToken.NONE);
      model.generate(
          request,
          event -> {
            if (event.tokenId() != null && readback.getAndSet(false)) {
              Map<String, Long> tail = stop();
              assertEquals(0L, tail.getOrDefault("mlx_eval", 0L));
              samples.getLast().put("already_evaluated_readback_and_commit", tail);
            }
          });
    } finally {
      System.setProperty("jmlx.binding.recording", "false");
    }
    assertEquals(2, samples.size());
    Map<String, Object> record =
        Map.of(
            "kind",
            "actual generated native API downcalls; timings invalid; not GPU dispatch counts",
            "fixture",
            "HF tiny: three linear layers, one full-attention layer, FLOAT32",
            "mlx_enable_tf32",
            System.getenv("MLX_ENABLE_TF32"),
            "decode_samples",
            samples);
    Path output = Path.of(System.getProperty("jmlx.qwen35.counter.output"));
    Files.createDirectories(output.getParent());
    new ObjectMapper().writerWithDefaultPrettyPrinter().writeValue(output.toFile(), record);
  }

  private static Object agent(String method) {
    try {
      return Class.forName("se.alipsa.jmlx.nn.BindingCounterAgent").getMethod(method).invoke(null);
    } catch (ReflectiveOperationException error) {
      throw new IllegalStateException(
          "Run the qwen35DecodeCounter task with its existing counter agent", error);
    }
  }

  private static Map<String, Long> stop() {
    Map<String, Long> result = new TreeMap<>();
    ((Map<?, ?>) agent("stop"))
        .forEach((key, value) -> result.put(key.toString(), ((Number) value).longValue()));
    return result;
  }
}
