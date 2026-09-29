package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
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

    try (RssSampler rss = new RssSampler();
        MLXScope scope = new MLXScope();
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
      String pin = runtimePin();
      System.out.println(
          "Tier B observed: model_type="
              + model.metadata().modelType()
              + ", pin="
              + pin
              + ", java="
              + Runtime.version()
              + ", peak_test_jvm_rss_kib="
              + rss.peakKiB()
              + ", generated_ids="
              + result.generatedTokenIds());

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

  private static String runtimePin() throws Exception {
    Path file =
        Path.of(
            System.getProperty("jmlx.repository.root"), "native/install/lib/native-pin.properties");
    Properties nativePin = new Properties();
    try (var input = Files.newInputStream(file)) {
      nativePin.load(input);
    }
    Process device =
        new ProcessBuilder("/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string").start();
    String hardware = new String(device.getInputStream().readAllBytes()).trim();
    if (device.waitFor() != 0 || hardware.isBlank()) {
      throw new IllegalStateException("cannot identify Tier B Apple Silicon device");
    }
    return hardware
        + "|macOS="
        + System.getProperty("os.version")
        + "|arch="
        + System.getProperty("os.arch")
        + "|mlx-metal="
        + nativePin.getProperty("mlxMetalVersion")
        + "|mlx-c="
        + nativePin.getProperty("mlxcCommit")
        + "|batch=1";
  }

  private static final class RssSampler implements AutoCloseable {
    private final AtomicLong peakKiB = new AtomicLong();
    private final Thread worker = Thread.ofVirtual().start(this::sample);

    private void sample() {
      while (!Thread.currentThread().isInterrupted()) {
        try {
          Process process =
              new ProcessBuilder(
                      "/bin/ps", "-o", "rss=", "-p", Long.toString(ProcessHandle.current().pid()))
                  .start();
          String output = new String(process.getInputStream().readAllBytes()).trim();
          if (process.waitFor() == 0 && !output.isBlank()) {
            peakKiB.accumulateAndGet(Long.parseLong(output), Math::max);
          }
          Thread.sleep(100);
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
        } catch (Exception e) {
          throw new IllegalStateException("cannot sample Tier B JVM resident memory", e);
        }
      }
    }

    private long peakKiB() {
      long value = peakKiB.get();
      if (value == 0) {
        throw new IllegalStateException("Tier B JVM resident memory was not measured");
      }
      return value;
    }

    @Override
    public void close() throws InterruptedException {
      worker.interrupt();
      worker.join();
    }
  }
}
