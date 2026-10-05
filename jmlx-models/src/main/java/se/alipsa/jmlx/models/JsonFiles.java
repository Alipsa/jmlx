package se.alipsa.jmlx.models;

import java.io.IOException;
import java.nio.file.Path;
import tools.jackson.core.JacksonException;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

/**
 * Reads checkpoint JSON files, wrapping Jackson 3's unchecked parse and I/O failures in the checked
 * {@link IOException} this package's loaders contract on.
 */
final class JsonFiles {

  private static final ObjectMapper MAPPER = new ObjectMapper();

  private JsonFiles() {}

  static JsonNode read(Path file) throws IOException {
    try {
      return MAPPER.readTree(file.toFile());
    } catch (JacksonException e) {
      throw new IOException("failed to read " + file.toAbsolutePath().normalize(), e);
    }
  }
}
