package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXIO;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import tools.jackson.databind.ObjectMapper;

/** Header validation guards decoder assembly before native safetensors loading. */
class DecoderAssemblerTest {

  private static final Path MIXTRAL =
      Path.of(
          System.getProperty("jmlx.repository.root"),
          "tools",
          "hf-reference",
          "goldens",
          "checkpoints",
          "mixtral");

  private static Map<String, MLXArray> mixtralTensors(MLXScope scope) {
    return new java.util.LinkedHashMap<>(
        MLXIO.loadSafetensors(scope, MIXTRAL.resolve("model.safetensors").toString()).tensors());
  }

  private static ArchitectureDescriptor mixtralDescriptor() throws IOException {
    return ArchitectureMappings.parse(
        new ObjectMapper().readTree(MIXTRAL.resolve("config.json").toFile()));
  }

  @Test
  @EnabledIfNativeAvailable
  void expertSourceTensorsAreClosedAfterStacking() throws Exception {
    try (MLXScope scope = new MLXScope()) {
      Map<String, MLXArray> tensors = mixtralTensors(scope);
      MLXArray expert = tensors.get("model.layers.0.block_sparse_moe.experts.0.w1.weight");
      MLXArray router = tensors.get("model.layers.0.block_sparse_moe.gate.weight");
      DecoderAssembler.assemble(scope, mixtralDescriptor(), tensors);
      assertThrows(IllegalStateException.class, expert::shape);
      router.shape(); // non-expert tensors stay open: they are the live parameters
    }
  }

  @Test
  @EnabledIfNativeAvailable
  void mismatchedExpertShapeNamesTheTensorKey() throws Exception {
    String key = "model.layers.1.block_sparse_moe.experts.2.w2.weight";
    try (MLXScope scope = new MLXScope()) {
      Map<String, MLXArray> tensors = mixtralTensors(scope);
      tensors.put(key, MLX.zeros(scope, new int[] {3, 3}, DType.FLOAT32));
      String message =
          assertThrows(
                  IllegalArgumentException.class,
                  () -> DecoderAssembler.assemble(scope, mixtralDescriptor(), tensors))
              .getMessage();
      assertTrue(message.contains(key), message);
    }
  }

  @Test
  @EnabledIfNativeAvailable
  void explicitOutputHeadWinsOverTieWordEmbeddings(@TempDir Path directory) throws Exception {
    TinyCheckpoints.randomLlama(directory, 42, 2, false, true);
    ArchitectureDescriptor descriptor =
        ArchitectureMappings.parse(
            new ObjectMapper().readTree(directory.resolve("config.json").toFile()));
    try (MLXScope scope = new MLXScope()) {
      Map<String, MLXArray> tensors =
          new java.util.LinkedHashMap<>(
              MLXIO
                  .loadSafetensors(scope, directory.resolve("model.safetensors").toString())
                  .tensors());
      tensors.put("lm_head.weight", MLX.zeros(scope, new int[] {128, 64}, DType.FLOAT32));
      assertNotNull(DecoderAssembler.assemble(scope, descriptor, tensors).lmHead());
    }
  }

  @Test
  @EnabledIfNativeAvailable
  void tiedEmbeddingWithoutOutputHeadHasNoLmHead(@TempDir Path directory) throws Exception {
    TinyCheckpoints.randomLlama(directory, 42, 2, false, true);
    ArchitectureDescriptor descriptor =
        ArchitectureMappings.parse(
            new ObjectMapper().readTree(directory.resolve("config.json").toFile()));
    try (MLXScope scope = new MLXScope()) {
      Map<String, MLXArray> tensors =
          MLXIO.loadSafetensors(scope, directory.resolve("model.safetensors").toString()).tensors();
      assertNull(DecoderAssembler.assemble(scope, descriptor, tensors).lmHead());
    }
  }

  @Test
  void missingMlpTensorStopsBeforeNativeLoad(@TempDir Path directory) throws IOException {
    TensorPlan plan = ArchitectureMappings.tensorPlan(TestDescriptors.llama(1, false));
    String missing = "model.layers.0.mlp.down_proj.weight";
    Set<String> available = new java.util.HashSet<>(plan.required());
    available.remove(missing);
    StringBuilder header = new StringBuilder("{");
    boolean first = true;
    for (String name : available) {
      if (!first) {
        header.append(',');
      }
      first = false;
      header
          .append('"')
          .append(name)
          .append("\":{\"dtype\":\"F32\",\"shape\":[0],\"data_offsets\":[0,0]}");
    }
    header.append('}');
    byte[] bytes = header.toString().getBytes(StandardCharsets.UTF_8);
    ByteBuffer data = ByteBuffer.allocate(8 + bytes.length).order(ByteOrder.LITTLE_ENDIAN);
    Files.write(
        directory.resolve("model.safetensors"), data.putLong(bytes.length).put(bytes).array());
    AtomicInteger loads = new AtomicInteger();
    String message =
        assertThrows(
                IllegalArgumentException.class,
                () ->
                    CheckpointLoader.load(
                        null,
                        directory,
                        plan,
                        (scope, shard) -> {
                          loads.incrementAndGet();
                          throw new AssertionError("native load must not run");
                        }))
            .getMessage();
    assertTrue(message.contains(missing), message);
    assertEquals(0, loads.get());
  }
}
