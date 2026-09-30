package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.KVCache;
import se.alipsa.jmlx.tokenizer.ChatTemplateOptions;
import se.alipsa.jmlx.tokenizer.HfTokenizer;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

/** Opt-in checkpoint check; exact IDs are asserted only for a recorded device/runtime pin. */
class TierBSmokeTest {
  private static final String PROMPT = "Hello, my name is";

  @Test
  void loadsRendersAndGenerates() throws Exception {
    String directory = System.getProperty("jmlx.tier.b.model.dir", "");
    assumeTrue(!directory.isBlank(), "Set JMLX_TIER_B_MODEL_DIR to run Tier B");
    Path modelDirectory = Path.of(directory);
    String manifestPath = System.getProperty("jmlx.tier.b.manifest", "");
    JsonNode manifest =
        manifestPath.isBlank() ? null : new ObjectMapper().readTree(Path.of(manifestPath).toFile());
    HfTokenizer tokenizer = HfTokenizer.fromDirectory(modelDirectory);
    List<Map<String, Object>> messages = List.of(Map.of("role", "user", "content", PROMPT));
    ChatTemplateOptions chatOptions = ChatTemplateOptions.defaults(false);
    String rendered = tokenizer.renderChat(messages, chatOptions);
    assertFalse(rendered.isBlank());
    List<Integer> promptIds = tokenizer.encode(rendered, false);
    assertFalse(promptIds.isEmpty());
    if (manifest != null) {
      assertEquals(manifest.path("expected_chat_text").asString(), rendered);
      assertEquals(ids(manifest.path("expected_chat_ids")), promptIds);
      assertEquals(ids(manifest.path("expected_prompt_ids")), tokenizer.encode(PROMPT, false));
    }
    int[] ids = promptIds.stream().mapToInt(Integer::intValue).toArray();
    GenerationRequest request =
        GenerationRequest.chat(
            tokenizer,
            messages,
            chatOptions,
            GenerationConfig.greedyDefaults(16, Set.of()),
            CancellationToken.NONE);
    assertArrayEquals(ids, request.promptTokenIds());

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
      GenerationResult result = model.generate(request, ignored -> {});
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

      String recordedPin = manifest == null ? "" : manifest.path("recorded_pin").asString("");
      JsonNode expectedTokens = manifest == null ? null : manifest.path("expected_token_ids");
      if (!recordedPin.isBlank()
          && recordedPin.equals(pin)
          && expectedTokens != null
          && expectedTokens.isArray()) {
        assertEquals(ids(expectedTokens), result.generatedTokenIds());
      } else {
        System.err.println(
            "Tier B structural pass only: device/macOS/MLX pin differs or exact IDs are"
                + " unrecorded");
      }
    }
  }

  private static List<Integer> ids(JsonNode values) {
    assertTrue(values.isArray(), "expected ID list must be an array");
    List<Integer> result = new ArrayList<>();
    for (JsonNode value : values) {
      result.add(value.asInt());
    }
    return result;
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
