package se.alipsa.jmlx.vision;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.Base64;
import java.util.Objects;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

/** Shared access to the committed image-oracle fixtures and the pinned test configuration. */
final class OracleFixtures {
  private static final ObjectMapper MAPPER = new ObjectMapper();

  private OracleFixtures() {}

  /** The {@code tools/image-oracle/fixtures} directory of the repository checkout. */
  static Path fixtures() {
    String root =
        Objects.requireNonNull(
            System.getProperty("jmlx.repository.root"), "jmlx.repository.root must be set");
    return Path.of(root, "tools", "image-oracle", "fixtures");
  }

  /** Reads a committed fixture file as a JSON tree, relative to the fixture root. */
  static JsonNode read(String relative) throws IOException {
    return MAPPER.readTree(fixtures().resolve(relative).toFile());
  }

  /** The pinned {@code preprocessor_config.json} test resource as text. */
  static String configText() throws IOException {
    try (InputStream in =
        Objects.requireNonNull(
            OracleFixtures.class.getResourceAsStream(
                "/se/alipsa/jmlx/vision/preprocessor_config.json"))) {
      return new String(in.readAllBytes(), StandardCharsets.UTF_8);
    }
  }

  static byte[] base64(String encoded) {
    return Base64.getDecoder().decode(encoded);
  }

  static String sha256Hex(byte[] bytes) {
    try {
      byte[] digest = MessageDigest.getInstance("SHA-256").digest(bytes);
      StringBuilder sb = new StringBuilder(digest.length * 2);
      for (byte b : digest) {
        sb.append(String.format("%02x", b & 0xFF));
      }
      return sb.toString();
    } catch (NoSuchAlgorithmException e) {
      throw new IllegalStateException("SHA-256 is unavailable", e);
    }
  }

  /** Decodes the oracle's big-endian IEEE-754 float32 bits ({@code float32_be_base64}). */
  static float[] float32(byte[] bigEndian) {
    float[] out = new float[bigEndian.length / 4];
    for (int i = 0; i < out.length; i++) {
      int b = i * 4;
      out[i] =
          Float.intBitsToFloat(
              ((bigEndian[b] & 0xFF) << 24)
                  | ((bigEndian[b + 1] & 0xFF) << 16)
                  | ((bigEndian[b + 2] & 0xFF) << 8)
                  | (bigEndian[b + 3] & 0xFF));
    }
    return out;
  }

  /** The row-major {@code (y0..y1) x (x0..x1)} pixel slice of an interleaved RGB image. */
  static byte[] crop(RgbImage image, int x0, int y0, int x1, int y1) {
    int width = x1 - x0;
    int height = y1 - y0;
    byte[] patch = new byte[width * height * 3];
    byte[] pixels = image.pixels();
    for (int y = 0; y < height; y++) {
      System.arraycopy(
          pixels, (y0 + y) * image.width() * 3 + x0 * 3, patch, y * width * 3, width * 3);
    }
    return patch;
  }

  static JsonNode findCase(JsonNode cases, String name) {
    for (int i = 0; i < cases.size(); i++) {
      if (name.equals(cases.get(i).required("name").asString())) {
        return cases.get(i);
      }
    }
    throw new IllegalStateException("no case named " + name);
  }

  static int[] ints(JsonNode array) {
    int[] out = new int[array.size()];
    for (int i = 0; i < out.length; i++) {
      out[i] = array.get(i).intValue();
    }
    return out;
  }

  /** A row-major mask of {@code count} valid pixels, all ones, as the oracle records it. */
  static int[] maskAllOnes(int count) {
    int[] mask = new int[count];
    Arrays.fill(mask, 1);
    return mask;
  }
}
