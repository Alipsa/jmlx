package se.alipsa.jmlx.buildsrc;

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Duration;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.Map;
import java.util.Set;

/** Downloads immutable public Hub artifacts with bounded streaming and SHA-256 verification. */
public final class TierBArtifactDownloader {
  private static final URI HUB = URI.create("https://huggingface.co/");

  private TierBArtifactDownloader() {}

  interface Transport {
    Response request(String relativePath, String method) throws IOException, InterruptedException;
  }

  record Response(int status, String contentLength, InputStream body) implements AutoCloseable {
    @Override
    public void close() throws IOException {
      body.close();
    }
  }

  static HttpClient client() {
    return HttpClient.newBuilder()
        .followRedirects(HttpClient.Redirect.NORMAL)
        .connectTimeout(Duration.ofSeconds(60))
        .build();
  }

  static Transport transport(URI baseUri) {
    HttpClient client = client();
    return (relative, method) -> {
      HttpRequest request =
          HttpRequest.newBuilder(baseUri.resolve(relative))
              .timeout(Duration.ofMinutes(5))
              .method(method, HttpRequest.BodyPublishers.noBody())
              .build();
      HttpResponse<InputStream> response =
          client.send(request, HttpResponse.BodyHandlers.ofInputStream());
      return new Response(
          response.statusCode(),
          response.headers().firstValue("Content-Length").orElse(null),
          response.body());
    };
  }

  /** Downloads the files in a manifest; existing files are verified on every invocation. */
  public static void download(Path manifest, Path target) throws IOException, InterruptedException {
    download(manifest, target, transport(HUB));
  }

  static void download(Path manifest, Path target, Transport transport)
      throws IOException, InterruptedException {
    Map<?, ?> root = object(new JsonParser(Files.readString(manifest)).parse());
    String repository = text(root.get("repository"));
    safeName(repository);
    String revision = text(root.get("revision"));
    if (!revision.matches("[0-9a-f]{40}")) {
      throw new IllegalArgumentException("revision must be a full immutable commit SHA");
    }
    final long totalCap = cap(root.get("max_total_bytes"));
    Map<?, ?> files = object(root.get("files"));
    Set<String> names = new HashSet<>();
    for (Object key : files.keySet()) {
      String name = text(key);
      safeName(name);
      if (!names.add(name)) {
        throw new IllegalArgumentException("duplicate artifact path: " + name);
      }
      Map<?, ?> spec = object(files.get(name));
      cap(spec.get("max_bytes"));
      if (!text(spec.get("sha256")).matches("[0-9a-f]{64}")) {
        throw new IllegalArgumentException("invalid SHA-256 for " + name);
      }
    }
    // Prevent a manifest from using another artifact's partial-file name.
    for (String name : names) {
      if (names.contains(name + ".part")) {
        throw new IllegalArgumentException("conflicting partial-file path: " + name);
      }
    }
    Files.createDirectories(target);
    Path realRoot = target.toRealPath();
    long total = 0;
    for (String name : names.stream().sorted().toList()) {
      Map<?, ?> spec = object(files.get(name));
      long fileCap = cap(spec.get("max_bytes"));
      String hash = text(spec.get("sha256"));
      Path destination = contained(realRoot, name);
      Path partial = contained(realRoot, name + ".part");
      String relative = repository + "/resolve/" + revision + "/" + name;
      long expected;
      try (Response response = transport.request(relative, "HEAD")) {
        expected = length(response, fileCap, totalCap - total);
      }
      if (Files.isRegularFile(destination, LinkOption.NOFOLLOW_LINKS)) {
        long size = Files.size(destination);
        if (size <= fileCap && size <= totalCap - total && hash.equals(sha256(destination))) {
          total += size;
          continue;
        }
      }
      long count = 0;
      MessageDigest digest = digest();
      try {
        // Recheck containment immediately before opening either filesystem output.
        destination = contained(realRoot, name);
        partial = contained(realRoot, name + ".part");
        try (Response response = transport.request(relative, "GET")) {
          long getSize = length(response, fileCap, totalCap - total);
          if (getSize != expected) {
            throw new IOException(name + ": HEAD/GET Content-Length mismatch");
          }
          try (var output = Files.newOutputStream(partial)) {
            byte[] buffer = new byte[1024 * 1024];
            int read;
            while ((read = response.body().read(buffer)) != -1) {
              if (read > fileCap - count || read > totalCap - total - count) {
                throw new IOException(name + ": streamed size cap exceeded");
              }
              count += read;
              digest.update(buffer, 0, read);
              output.write(buffer, 0, read);
            }
          }
          if (count != getSize) {
            throw new IOException(name + ": response length mismatch");
          }
        }
        if (!hash.equals(HexFormat.of().formatHex(digest.digest()))) {
          throw new IOException(name + ": SHA-256 mismatch");
        }
        Files.move(
            partial,
            destination,
            StandardCopyOption.ATOMIC_MOVE,
            StandardCopyOption.REPLACE_EXISTING);
        total += count;
        System.out.println(name + ": " + count + " bytes, SHA-256 verified");
      } finally {
        Files.deleteIfExists(partial);
      }
    }
    System.out.println("artifact total: " + total + " bytes");
  }

  private static long length(Response response, long fileCap, long remaining) throws IOException {
    if (response.status() < 200 || response.status() >= 300) {
      throw new IOException("non-success final HTTP status: " + response.status());
    }
    try {
      long length = Long.parseLong(response.contentLength());
      if (length < 0 || length > fileCap || length > remaining) {
        throw new IOException("response exceeds size cap");
      }
      return length;
    } catch (NumberFormatException e) {
      throw new IOException("missing or invalid final Content-Length", e);
    }
  }

  private static Path contained(Path root, String name) throws IOException {
    Path current = root;
    String[] components = name.split("/");
    for (int i = 0; i < components.length; i++) {
      current = current.resolve(components[i]);
      if (Files.isSymbolicLink(current)) {
        throw new IOException("symlink artifact path: " + name);
      }
      if (i < components.length - 1) {
        Files.createDirectories(current);
      }
      if (Files.exists(current) && !current.toRealPath().startsWith(root)) {
        throw new IOException("artifact path escapes target: " + name);
      }
    }
    return current;
  }

  private static void safeName(String name) {
    if (name.isEmpty()
        || name.contains("\\")
        || name.contains(":")
        || name.contains("%")
        || name.contains("?")
        || name.contains("#")) {
      throw new IllegalArgumentException("invalid artifact path: " + name);
    }
    for (String component : name.split("/", -1)) {
      if (component.isEmpty()
          || component.equals(".")
          || component.equals("..")
          || !component.matches("[A-Za-z0-9_.-]+")) {
        throw new IllegalArgumentException("invalid artifact path: " + name);
      }
    }
  }

  private static Map<?, ?> object(Object value) {
    if (value instanceof Map<?, ?> map) {
      return map;
    }
    throw new IllegalArgumentException("expected JSON object");
  }

  private static String text(Object value) {
    if (value instanceof String text) {
      return text;
    }
    throw new IllegalArgumentException("expected JSON string");
  }

  private static long cap(Object value) {
    if (value instanceof Long number && number > 0) {
      return number;
    }
    throw new IllegalArgumentException("byte cap must be a positive signed-long JSON integer");
  }

  private static MessageDigest digest() {
    try {
      return MessageDigest.getInstance("SHA-256");
    } catch (NoSuchAlgorithmException e) {
      throw new AssertionError(e);
    }
  }

  private static String sha256(Path path) throws IOException {
    MessageDigest digest = digest();
    try (var stream = Files.newInputStream(path)) {
      byte[] buffer = new byte[1024 * 1024];
      int read;
      while ((read = stream.read(buffer)) != -1) {
        digest.update(buffer, 0, read);
      }
    }
    return HexFormat.of().formatHex(digest.digest());
  }
}
