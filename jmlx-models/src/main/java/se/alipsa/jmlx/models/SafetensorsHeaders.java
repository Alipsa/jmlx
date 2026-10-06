package se.alipsa.jmlx.models;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.channels.ReadableByteChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import tools.jackson.core.JacksonException;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

/** Reads safetensors names without loading tensor data or initializing the native runtime. */
final class SafetensorsHeaders {
  private static final long MAX_HEADER_BYTES = 100_000_000;
  private static final ObjectMapper MAPPER = new ObjectMapper();

  private SafetensorsHeaders() {}

  static Map<Path, Set<String>> tensorNames(List<Path> files) throws IOException {
    Map<Path, Set<String>> names = new LinkedHashMap<>();
    for (Path file : files) {
      try (FileChannel channel = FileChannel.open(file, StandardOpenOption.READ)) {
        ByteBuffer length = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        readFully(channel, length, "invalid safetensors header in " + file);
        length.flip();
        long count = length.getLong();
        if (count < 0 || count > MAX_HEADER_BYTES || count > channel.size() - Long.BYTES) {
          throw new IllegalArgumentException("invalid safetensors header length in " + file);
        }
        ByteBuffer bytes = ByteBuffer.allocate((int) count);
        readFully(channel, bytes, "truncated safetensors header in " + file);
        JsonNode root;
        try {
          root = MAPPER.readTree(new String(bytes.array(), StandardCharsets.UTF_8));
        } catch (JacksonException e) {
          throw new IOException("invalid safetensors header JSON in " + file, e);
        }
        if (root == null || !root.isObject()) {
          throw new IllegalArgumentException("invalid safetensors header JSON in " + file);
        }
        names.put(
            file,
            root.properties().stream()
                .map(Map.Entry::getKey)
                .filter(k -> !"__metadata__".equals(k))
                .collect(java.util.stream.Collectors.toSet()));
      }
    }
    return names;
  }

  /** Validates serialized non-parameter BERT index buffers before any native loading. */
  static void validateBertBuffers(List<Path> files, String prefix, int positions)
      throws IOException {
    for (Path file : files) {
      try (FileChannel channel = FileChannel.open(file, StandardOpenOption.READ)) {
        ByteBuffer length = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
        readFully(channel, length, "truncated safetensors length");
        length.flip();
        long headerSize = length.getLong();
        if (headerSize < 0 || headerSize > MAX_HEADER_BYTES || headerSize > channel.size() - 8) {
          throw new IllegalArgumentException("invalid safetensors header size");
        }
        ByteBuffer bytes = ByteBuffer.allocate((int) headerSize);
        readFully(channel, bytes, "truncated safetensors header");
        JsonNode root;
        try {
          root = MAPPER.readTree(new String(bytes.array(), StandardCharsets.UTF_8));
        } catch (JacksonException e) {
          throw new IOException("invalid safetensors header JSON in " + file, e);
        }
        for (String name : List.of("position_ids", "token_type_ids")) {
          JsonNode buffer = root.get(prefix + "embeddings." + name);
          if (buffer == null) {
            continue;
          }
          String dtype = buffer.path("dtype").asString();
          int width = dtype.equals("I64") ? 8 : dtype.equals("I32") ? 4 : 0;
          JsonNode shape = buffer.path("shape");
          JsonNode offsets = buffer.path("data_offsets");
          if (width == 0
              || !shape.isArray()
              || shape.size() != 2
              || shape.get(0).asLong() != 1
              || shape.get(1).asLong() != positions
              || !offsets.isArray()
              || offsets.size() != 2
              || !offsets.get(0).isIntegralNumber()
              || !offsets.get(1).isIntegralNumber()) {
            throw new IllegalArgumentException("invalid BERT buffer dtype/shape: " + name);
          }
          long start = offsets.get(0).asLong();
          long end = offsets.get(1).asLong();
          long count = (long) positions * width;
          if (start < 0
              || end < start
              || end - start != count
              || end > channel.size() - 8 - headerSize
              || count > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("invalid BERT buffer offsets: " + name);
          }
          channel.position(8 + headerSize + start);
          ByteBuffer data = ByteBuffer.allocate((int) count).order(ByteOrder.LITTLE_ENDIAN);
          readFully(channel, data, "truncated BERT buffer");
          data.flip();
          for (int i = 0; i < positions; i++) {
            long value = width == 8 ? data.getLong() : data.getInt();
            if (value != (name.equals("position_ids") ? i : 0)) {
              throw new IllegalArgumentException("invalid BERT buffer value: " + name);
            }
          }
        }
      }
    }
  }

  static void readFully(ReadableByteChannel channel, ByteBuffer target, String eofMessage)
      throws IOException {
    while (target.hasRemaining()) {
      if (channel.read(target) < 0) {
        throw new IllegalArgumentException(eofMessage);
      }
    }
  }
}
