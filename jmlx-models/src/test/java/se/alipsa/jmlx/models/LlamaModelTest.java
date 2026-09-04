package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXIO;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.ffi.NativeMemoryProbe;
import se.alipsa.jmlx.memory.MLXScope;

@EnabledIfNativeAvailable
class LlamaModelTest {
  @Test
  void legacyAndCommonGenerationAgree(@TempDir Path dir) throws Exception {
    writeTinyLlamaCheckpoint(dir);
    try (MLXScope modelScope = new MLXScope()) {
      LlamaModel model = LlamaModel.load(modelScope, dir);
      assertEquals(List.of(1, 0, 0), model.generate(new int[] {1}, 2, Set.of()));
      assertThrows(IllegalArgumentException.class, () -> model.generate(null, 2, Set.of()));
      TextGenerationModel common = TextGenerationModels.load(modelScope, dir);
      assertEquals(4, common.metadata().vocabSize());
      List<GenerationEvent> events = new ArrayList<>();
      GenerationResult result =
          common.generate(
              new GenerationRequest(
                  new int[] {1},
                  GenerationConfig.greedyDefaults(2, Set.of()),
                  CancellationToken.NONE),
              events::add);
      assertEquals(List.of(1), result.promptTokenIds());
      assertEquals(List.of(0, 0), result.generatedTokenIds());
      assertEquals(FinishReason.MAX_TOKENS, result.finishReason());
      assertEquals(
          List.of(0, 0), events.subList(0, 2).stream().map(GenerationEvent::tokenId).toList());
      assertNull(events.getLast().tokenId());
      assertEquals(FinishReason.MAX_TOKENS, events.getLast().finishReason());
    }
  }

  @Test
  void reportsEosStopAndMaxTokenTermination(@TempDir Path dir) throws Exception {
    writeTinyLlamaCheckpoint(dir);
    try (MLXScope modelScope = new MLXScope()) {
      LlamaModel model = LlamaModel.load(modelScope, dir);
      List<GenerationEvent> eosEvents = new ArrayList<>();
      GenerationResult eos =
          model.generate(
              new GenerationRequest(
                  new int[] {1},
                  GenerationConfig.greedyDefaults(2, Set.of(0)),
                  CancellationToken.NONE),
              eosEvents::add);
      assertEquals(FinishReason.EOS, eos.finishReason());
      assertEquals(List.of(0), eos.generatedTokenIds());
      assertEquals(
          eos.generatedTokenIds(),
          eosEvents.stream()
              .map(GenerationEvent::tokenId)
              .filter(java.util.Objects::nonNull)
              .toList());

      List<GenerationEvent> stopEvents = new ArrayList<>();
      GenerationResult stopped =
          model.generate(
              new GenerationRequest(
                  new int[] {1},
                  GenerationConfig.greedyDefaults(2, Set.of(), Set.of(0)),
                  CancellationToken.NONE),
              stopEvents::add);
      assertEquals(FinishReason.STOP_TOKEN, stopped.finishReason());
      assertEquals(List.of(), stopped.generatedTokenIds());
      assertEquals(1, stopEvents.size());
      assertEquals(FinishReason.STOP_TOKEN, stopEvents.getFirst().finishReason());

      List<GenerationEvent> bothTerminalEvents = new ArrayList<>();
      assertEquals(
          FinishReason.EOS,
          model
              .generate(
                  new GenerationRequest(
                      new int[] {1},
                      GenerationConfig.greedyDefaults(2, Set.of(0), Set.of(0)),
                      CancellationToken.NONE),
                  bothTerminalEvents::add)
              .finishReason());
      assertEquals(FinishReason.EOS, bothTerminalEvents.getFirst().finishReason());
    }
  }

  @Test
  void cancellationIsPolledByTheGenerationThread(@TempDir Path dir) throws Exception {
    writeTinyLlamaCheckpoint(dir);
    try (MLXScope modelScope = new MLXScope()) {
      LlamaModel model = LlamaModel.load(modelScope, dir);
      AtomicBoolean cancelled = new AtomicBoolean();
      AtomicReference<Thread> pollingThread = new AtomicReference<>();
      AtomicReference<Thread> generationThread = new AtomicReference<>();
      GenerationResult cancelledResult =
          model.generate(
              new GenerationRequest(
                  new int[] {1},
                  GenerationConfig.greedyDefaults(2, Set.of()),
                  () -> {
                    pollingThread.set(Thread.currentThread());
                    return cancelled.get();
                  }),
              event -> {
                generationThread.set(Thread.currentThread());
                Thread canceller =
                    Thread.ofPlatform()
                        .start(
                            () -> {
                              cancelled.set(true);
                            });
                try {
                  canceller.join();
                } catch (InterruptedException e) {
                  Thread.currentThread().interrupt();
                  throw new AssertionError(e);
                }
              });
      assertEquals(FinishReason.CANCELLED, cancelledResult.finishReason());
      assertEquals(List.of(0), cancelledResult.generatedTokenIds());
      assertEquals(Thread.currentThread(), pollingThread.get());
      assertEquals(Thread.currentThread(), generationThread.get());
    }
  }

  @Test
  void cancellationBeforePrefillEmitsOnlyATerminalEvent(@TempDir Path dir) throws Exception {
    writeTinyLlamaCheckpoint(dir);
    try (MLXScope modelScope = new MLXScope()) {
      LlamaModel model = LlamaModel.load(modelScope, dir);
      List<GenerationEvent> events = new ArrayList<>();

      GenerationResult result =
          model.generate(
              new GenerationRequest(
                  new int[] {1}, GenerationConfig.greedyDefaults(2, Set.of()), () -> true),
              events::add);

      assertEquals(FinishReason.CANCELLED, result.finishReason());
      assertEquals(List.of(), result.generatedTokenIds());
      assertEquals(List.of(GenerationEvent.finished(FinishReason.CANCELLED)), events);
    }
  }

  @Test
  void rejectsNonGreedyPolicy(@TempDir Path dir) throws Exception {
    writeTinyLlamaCheckpoint(dir);
    try (MLXScope modelScope = new MLXScope()) {
      LlamaModel model = LlamaModel.load(modelScope, dir);
      assertThrows(
          UnsupportedOperationException.class,
          () ->
              model.generate(
                  new GenerationRequest(
                      new int[] {1},
                      new GenerationConfig(
                          1,
                          java.util.OptionalLong.empty(),
                          1,
                          0,
                          1,
                          0,
                          1,
                          0,
                          0,
                          Set.of(),
                          Set.of(),
                          false),
                      CancellationToken.NONE),
                  ignored -> {}));
    }
  }

  @Test
  void listenerFailuresHaveDefinedCompletionBehavior(@TempDir Path dir) throws Exception {
    writeTinyLlamaCheckpoint(dir);
    try (MLXScope modelScope = new MLXScope()) {
      LlamaModel model = LlamaModel.load(modelScope, dir);
      List<GenerationEvent> failedListenerEvents = new ArrayList<>();
      GenerationAbortedException listenerFailure =
          assertThrows(
              GenerationAbortedException.class,
              () ->
                  model.generate(
                      new GenerationRequest(
                          new int[] {1},
                          GenerationConfig.greedyDefaults(2, Set.of()),
                          CancellationToken.NONE),
                      event -> {
                        failedListenerEvents.add(event);
                        throw new IllegalStateException("listener failed");
                      }));
      assertEquals(1, failedListenerEvents.size());
      assertEquals(0, failedListenerEvents.getFirst().tokenId());
      assertEquals(List.of(1), listenerFailure.promptTokenIds());
      assertEquals(List.of(0), listenerFailure.generatedTokenIds());
      GenerationResult terminalListenerFailure =
          model.generate(
              new GenerationRequest(
                  new int[] {1},
                  GenerationConfig.greedyDefaults(0, Set.of()),
                  CancellationToken.NONE),
              event -> {
                throw new IllegalStateException("terminal listener failed");
              });
      assertEquals(FinishReason.MAX_TOKENS, terminalListenerFailure.finishReason());
    }
  }

  @Test
  void closesGenerationScopesOnEveryTerminalPath(@TempDir Path dir) throws Exception {
    writeTinyLlamaCheckpoint(dir);
    try (MLXScope modelScope = new MLXScope()) {
      LlamaModel model = LlamaModel.load(modelScope, dir);
      exerciseTerminalPaths(model);
      long baseline = NativeMemoryProbe.activeMemoryBytes();
      // MLX's active-memory counter excludes cached allocator blocks; the 4 KiB allowance absorbs
      // small runner-side bookkeeping while remaining far below one leaked decoder activation.
      for (int i = 0; i < 20; i++) {
        exerciseTerminalPaths(model);
      }
      long after = NativeMemoryProbe.activeMemoryBytes();
      assertTrue(
          after <= baseline + 4096,
          "generation scopes leaked native memory (baseline="
              + baseline
              + ", after="
              + after
              + ")");
    }
  }

  private static void exerciseTerminalPaths(LlamaModel model) {
    model.generate(new int[] {1}, 2, Set.of());
    model.generate(new int[] {1}, 2, Set.of(0));
    model.generate(
        new GenerationRequest(
            new int[] {1},
            GenerationConfig.greedyDefaults(2, Set.of(), Set.of(0)),
            CancellationToken.NONE),
        ignored -> {});
    model.generate(
        new GenerationRequest(
            new int[] {1}, GenerationConfig.greedyDefaults(2, Set.of()), () -> true),
        ignored -> {});
    assertThrows(
        GenerationAbortedException.class,
        () ->
            model.generate(
                new GenerationRequest(
                    new int[] {1},
                    GenerationConfig.greedyDefaults(2, Set.of()),
                    CancellationToken.NONE),
                event -> {
                  throw new IllegalStateException("listener failed");
                }));
    AtomicBoolean cancelled = new AtomicBoolean();
    model.generate(
        new GenerationRequest(
            new int[] {1}, GenerationConfig.greedyDefaults(2, Set.of()), cancelled::get),
        event -> cancelled.set(true));
  }

  private static void writeTinyLlamaCheckpoint(Path dir) throws Exception {
    Files.writeString(
        dir.resolve("config.json"),
        """
        {"model_type":"llama","vocab_size":4,"hidden_size":4,"intermediate_size":8,
         "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":2,
         "rms_norm_eps":0.000001,"rope_theta":10000,"tie_word_embeddings":true}
        """);
    try (MLXScope saveScope = new MLXScope()) {
      Map<String, MLXArray> tensors = tinyCheckpoint(saveScope);
      tensors.remove("lm_head.weight");
      MLXIO.saveSafetensors(dir.resolve("model.safetensors").toString(), tensors, Map.of());
    }
  }

  @Test
  void rejectsQwenBeforeAttemptingCheckpointLoad(@TempDir Path dir) throws Exception {
    Files.writeString(
        dir.resolve("config.json"),
        """
        {"model_type":"qwen2","vocab_size":4,"hidden_size":4,"intermediate_size":8,
         "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1}
        """);

    try (MLXScope scope = new MLXScope()) {
      IllegalArgumentException e =
          assertThrows(IllegalArgumentException.class, () -> LlamaModel.load(scope, dir));
      assertTrue(e.getMessage().contains("expected model_type llama"), e.getMessage());
    }
  }

  @Test
  void throwsWhenConfigRequiresAttentionBiasButCheckpointHasNone(@TempDir Path dir)
      throws Exception {
    Files.writeString(
        dir.resolve("config.json"),
        """
        {"model_type":"llama","vocab_size":4,"hidden_size":4,"intermediate_size":8,
         "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":2,
         "rms_norm_eps":0.000001,"rope_theta":10000,"tie_word_embeddings":true,
         "attention_bias":true}
        """);
    try (MLXScope saveScope = new MLXScope()) {
      Map<String, MLXArray> tensors = tinyCheckpoint(saveScope);
      tensors.remove("lm_head.weight");
      MLXIO.saveSafetensors(dir.resolve("model.safetensors").toString(), tensors, Map.of());
    }
    try (MLXScope modelScope = new MLXScope()) {
      IllegalArgumentException e =
          assertThrows(IllegalArgumentException.class, () -> LlamaModel.load(modelScope, dir));
      assertTrue(e.getMessage().contains("self_attn.q_proj.bias"), e.getMessage());
    }
  }

  private static Map<String, MLXArray> tinyCheckpoint(MLXScope scope) {
    Map<String, MLXArray> tensors = new LinkedHashMap<>();
    tensors.put("model.embed_tokens.weight", zeros(scope, 4, 4));
    tensors.put("model.norm.weight", zeros(scope, 4));
    tensors.put("lm_head.weight", zeros(scope, 4, 4));
    String p = "model.layers.0.";
    tensors.put(p + "input_layernorm.weight", zeros(scope, 4));
    tensors.put(p + "post_attention_layernorm.weight", zeros(scope, 4));
    // No bias tensors: attention_bias:true in config.json requires all four, so loading must
    // fail before any of these weights matter.
    tensors.put(p + "self_attn.q_proj.weight", zeros(scope, 4, 4));
    tensors.put(p + "self_attn.k_proj.weight", zeros(scope, 4, 4));
    tensors.put(p + "self_attn.v_proj.weight", zeros(scope, 4, 4));
    tensors.put(p + "self_attn.o_proj.weight", zeros(scope, 4, 4));
    tensors.put(p + "mlp.gate_proj.weight", zeros(scope, 8, 4));
    tensors.put(p + "mlp.up_proj.weight", zeros(scope, 8, 4));
    tensors.put(p + "mlp.down_proj.weight", zeros(scope, 4, 8));
    return tensors;
  }

  private static MLXArray zeros(MLXScope scope, int... shape) {
    return MLX.zeros(scope, shape, DType.FLOAT32);
  }
}
