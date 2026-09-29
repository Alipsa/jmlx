package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Header validation guards decoder assembly before native safetensors loading. */
class DecoderAssemblerTest {

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
