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
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class CheckpointIndexTest {
  @TempDir Path directory;

  @Test
  void missingIndexedTensorFailsBeforeLoader() throws IOException {
    shard("model-1.safetensors", "a");
    index(Map.of("a", "model-1.safetensors", "missing.key", "model-1.safetensors"));
    assertPreflightFails("missing.key");
  }

  @Test
  void unindexedShardTensorFailsBeforeLoader() throws IOException {
    shard("model-1.safetensors", "a", "extra.key");
    index(Map.of("a", "model-1.safetensors"));
    assertPreflightFails("extra.key");
  }

  @Test
  void wrongShardFailsBeforeLoader() throws IOException {
    shard("model-1.safetensors", "a");
    shard("model-2.safetensors", "b");
    index(Map.of("a", "model-2.safetensors", "b", "model-1.safetensors"));
    assertPreflightFails("a");
  }

  @Test
  void metadataIsNotIndexedTensor() throws IOException {
    shard("model-1.safetensors", "a");
    index(Map.of("a", "model-1.safetensors"));
    assertEquals(Set.of("a"), CheckpointLoader.preflight(directory, plan("a")).tensorNames());
  }

  @Test
  void consolidatedIsIgnoredBesideModelShard() throws IOException {
    shard("consolidated.safetensors", "old");
    shard("model-1.safetensors", "a");
    var checked = CheckpointLoader.preflight(directory, plan("a"));
    assertEquals(Set.of("a"), checked.tensorNames());
    assertEquals(1, checked.shards().size());
  }

  @Test
  void consolidatedAloneIsRead() throws IOException {
    shard("consolidated.safetensors", "a");
    assertEquals(Set.of("a"), CheckpointLoader.preflight(directory, plan("a")).tensorNames());
  }

  @Test
  void invalidPlanFailsBeforeLoader() throws IOException {
    shard("model-1.safetensors", "a");
    AtomicInteger calls = new AtomicInteger();
    String message =
        assertThrows(
                IllegalArgumentException.class,
                () ->
                    CheckpointLoader.load(
                        null,
                        directory,
                        plan("absent"),
                        (scope, file) -> {
                          calls.incrementAndGet();
                          throw new AssertionError("loader called");
                        }))
            .getMessage();
    assertTrue(message.contains("absent"));
    assertEquals(0, calls.get());
  }

  private void assertPreflightFails(String key) {
    AtomicInteger calls = new AtomicInteger();
    String message =
        assertThrows(
                IllegalArgumentException.class,
                () ->
                    CheckpointLoader.load(
                        null,
                        directory,
                        plan("a", "b", "extra.key"),
                        (scope, file) -> {
                          calls.incrementAndGet();
                          throw new AssertionError("loader called");
                        }))
            .getMessage();
    assertTrue(message.contains(key), message);
    assertEquals(0, calls.get());
  }

  private static TensorPlan plan(String... keys) {
    return new TensorPlan(Set.of(keys), Set.of(), Set.of(), Set.of());
  }

  private void shard(String name, String... keys) throws IOException {
    Map<String, String> entries = new LinkedHashMap<>();
    entries.put("__metadata__", "{\"format\":\"pt\"}");
    for (String key : keys) {
      entries.put(key, "{\"dtype\":\"F32\",\"shape\":[0],\"data_offsets\":[0,0]}");
    }
    String header =
        entries.entrySet().stream()
            .map(e -> "\"" + e.getKey() + "\":" + e.getValue())
            .collect(java.util.stream.Collectors.joining(",", "{", "}"));
    byte[] bytes = header.getBytes(StandardCharsets.UTF_8);
    ByteBuffer buffer = ByteBuffer.allocate(8 + bytes.length).order(ByteOrder.LITTLE_ENDIAN);
    buffer.putLong(bytes.length).put(bytes);
    Files.write(directory.resolve(name), buffer.array());
  }

  private void index(Map<String, String> weights) throws IOException {
    String mapping =
        weights.entrySet().stream()
            .map(e -> "\"" + e.getKey() + "\":\"" + e.getValue() + "\"")
            .collect(java.util.stream.Collectors.joining(",", "{", "}"));
    Files.writeString(
        directory.resolve("model.safetensors.index.json"), "{\"weight_map\":" + mapping + "}");
  }
}
