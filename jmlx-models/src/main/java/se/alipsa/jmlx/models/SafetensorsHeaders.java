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
        JsonNode root = MAPPER.readTree(new String(bytes.array(), StandardCharsets.UTF_8));
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

  static void readFully(ReadableByteChannel channel, ByteBuffer target, String eofMessage)
      throws IOException {
    while (target.hasRemaining()) {
      if (channel.read(target) < 0) {
        throw new IllegalArgumentException(eofMessage);
      }
    }
  }
}
