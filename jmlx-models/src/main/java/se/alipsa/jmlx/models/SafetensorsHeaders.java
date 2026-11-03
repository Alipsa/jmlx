package se.alipsa.jmlx.models;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.channels.ReadableByteChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Comparator;
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

  /** One immutable, validated on-disk tensor descriptor, with no native initialization. */
  record TensorInfo(Path shard, String dtype, List<Integer> shape, long start, long end) {
    TensorInfo {
      shape = List.copyOf(shape);
    }
  }

  /** Complete shape/type/offset inventory; rejects duplicate names and invalid byte ranges. */
  static Map<String, TensorInfo> inventory(List<Path> files) throws IOException {
    Map<String, TensorInfo> result = new LinkedHashMap<>();
    for (Path file : files) {
      JsonNode root = header(file);
      long payload;
      try (FileChannel channel = FileChannel.open(file, StandardOpenOption.READ)) {
        ByteBuffer length = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
        readFully(channel, length, "truncated safetensors header length");
        length.flip();
        payload = channel.size() - 8 - length.getLong();
      }
      List<TensorInfo> spans = new ArrayList<>();
      for (var entry : root.properties()) {
        String key = entry.getKey();
        if (key.equals("__metadata__")) {
          continue;
        }
        JsonNode value = entry.getValue();
        JsonNode type = value.get("dtype");
        JsonNode dimensions = value.get("shape");
        JsonNode offsets = value.get("data_offsets");
        if (!value.isObject()
            || type == null
            || !type.isString()
            || dimensions == null
            || !dimensions.isArray()
            || offsets == null
            || !offsets.isArray()
            || offsets.size() != 2) {
          throw new IllegalArgumentException("invalid tensor header: " + key);
        }
        int bytes =
            switch (type.asString()) {
              case "BOOL", "I8", "U8" -> 1;
              case "F16", "BF16", "I16", "U16" -> 2;
              case "F32", "I32", "U32" -> 4;
              case "F64", "I64", "U64" -> 8;
              default -> throw new IllegalArgumentException("unsupported header dtype: " + key);
            };
        List<Integer> shape = new ArrayList<>();
        long size = bytes;
        try {
          for (JsonNode dimension : dimensions) {
            if (!dimension.isIntegralNumber()
                || !dimension.canConvertToInt()
                || dimension.asInt() < 0) {
              throw new IllegalArgumentException("invalid tensor shape: " + key);
            }
            shape.add(dimension.asInt());
            size = Math.multiplyExact(size, dimension.asInt());
          }
        } catch (ArithmeticException e) {
          throw new IllegalArgumentException("tensor byte count overflows: " + key, e);
        }
        if (!offsets.get(0).isIntegralNumber()
            || !offsets.get(0).canConvertToLong()
            || !offsets.get(1).isIntegralNumber()
            || !offsets.get(1).canConvertToLong()) {
          throw new IllegalArgumentException("invalid tensor offsets: " + key);
        }
        long start = offsets.get(0).asLong();
        long end = offsets.get(1).asLong();
        if (start < 0 || end < start || end > payload || end - start != size) {
          throw new IllegalArgumentException("invalid tensor byte range: " + key);
        }
        TensorInfo info = new TensorInfo(file, type.asString(), shape, start, end);
        if (result.putIfAbsent(key, info) != null) {
          throw new IllegalArgumentException("duplicate tensor in checkpoint shards: " + key);
        }
        spans.add(info);
      }
      spans.sort(Comparator.comparingLong(TensorInfo::start).thenComparingLong(TensorInfo::end));
      long cursor = 0;
      for (TensorInfo span : spans) {
        if (span.start() != cursor) {
          throw new IllegalArgumentException("overlapping or gapped tensor byte ranges in " + file);
        }
        cursor = span.end();
      }
      if (cursor != payload) {
        throw new IllegalArgumentException("unclaimed tensor payload in " + file);
      }
    }
    return Map.copyOf(result);
  }

  /** Compares already validated tied storage without allocating a native parameter. */
  static boolean sameData(TensorInfo left, TensorInfo right) throws IOException {
    if (!left.dtype().equals(right.dtype())
        || !left.shape().equals(right.shape())
        || left.end() - left.start() != right.end() - right.start()) {
      return false;
    }
    try (FileChannel a = FileChannel.open(left.shard(), StandardOpenOption.READ);
        FileChannel b = FileChannel.open(right.shard(), StandardOpenOption.READ)) {
      a.position(payloadStart(a) + left.start());
      b.position(payloadStart(b) + right.start());
      ByteBuffer x = ByteBuffer.allocate(1024 * 1024);
      ByteBuffer y = ByteBuffer.allocate(1024 * 1024);
      long remaining = left.end() - left.start();
      while (remaining > 0) {
        int count = (int) Math.min(remaining, x.capacity());
        x.clear().limit(count);
        y.clear().limit(count);
        readFully(a, x, "truncated tied tensor");
        readFully(b, y, "truncated tied tensor");
        x.flip();
        y.flip();
        if (!x.equals(y)) {
          return false;
        }
        remaining -= count;
      }
      return true;
    }
  }

  private static long payloadStart(FileChannel channel) throws IOException {
    ByteBuffer bytes = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
    readFully(channel, bytes, "truncated safetensors length");
    bytes.flip();
    long length = bytes.getLong();
    if (length < 0 || length > MAX_HEADER_BYTES || length > channel.size() - 8) {
      throw new IllegalArgumentException("invalid tied tensor shard header length");
    }
    return 8 + length;
  }

  static Map<Path, Set<String>> tensorNames(List<Path> files) throws IOException {
    Map<Path, Set<String>> names = new LinkedHashMap<>();
    for (Path file : files) {
      names.put(
          file,
          header(file).properties().stream()
              .map(Map.Entry::getKey)
              .filter(k -> !"__metadata__".equals(k))
              .collect(java.util.stream.Collectors.toSet()));
    }
    return names;
  }

  /**
   * The merged {@code __metadata__} strings of every shard header, without loading tensor data or
   * initializing the native runtime. Shards without a {@code __metadata__} block contribute
   * nothing; a block that is not an object or that holds a non-string value is rejected by name and
   * file; a key repeated across shards must agree in value, or the collision is rejected naming
   * both files.
   *
   * @throws IOException if a header cannot be read or its JSON is malformed
   * @throws IllegalArgumentException if a header, a {@code __metadata__} block or a value is
   *     invalid, or a key's value disagrees across shards
   */
  static Map<String, String> metadata(List<Path> files) throws IOException {
    Map<String, String> merged = new LinkedHashMap<>();
    Map<String, Path> sources = new LinkedHashMap<>();
    for (Path file : files) {
      JsonNode block = header(file).get("__metadata__");
      if (block == null || block.isNull()) {
        continue;
      }
      if (!block.isObject()) {
        throw new IllegalArgumentException("invalid __metadata__ block in " + file);
      }
      for (var entry : block.properties()) {
        JsonNode value = entry.getValue();
        if (!value.isString()) {
          throw new IllegalArgumentException(
              "__metadata__ key '" + entry.getKey() + "' in " + file + " must be a string");
        }
        Path source = sources.get(entry.getKey());
        if (source == null) {
          merged.put(entry.getKey(), value.asString());
          sources.put(entry.getKey(), file);
        } else if (!merged.get(entry.getKey()).equals(value.asString())) {
          throw new IllegalArgumentException(
              "__metadata__ key '"
                  + entry.getKey()
                  + "' differs between "
                  + source
                  + " and "
                  + file);
        }
      }
    }
    return merged;
  }

  /** Reads and validates one shard's header JSON object. */
  private static JsonNode header(Path file) throws IOException {
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
      return root;
    }
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
