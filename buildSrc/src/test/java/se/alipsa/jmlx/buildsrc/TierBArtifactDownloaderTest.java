package se.alipsa.jmlx.buildsrc;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.sun.net.httpserver.HttpServer;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.http.HttpClient;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class TierBArtifactDownloaderTest {
  @TempDir Path temporary;
  private static final byte[] CONTENT = "fixture".getBytes(java.nio.charset.StandardCharsets.UTF_8);

  private Path manifest(String name, long cap, long total, String hash) throws Exception {
    Path path = temporary.resolve("manifest.json");
    Files.writeString(
        path,
        "{\"repository\":\"test/model\",\"revision\":\""
            + "a".repeat(40)
            + "\",\"max_total_bytes\":"
            + total
            + ",\"files\":{\""
            + name
            + "\":{\"max_bytes\":"
            + cap
            + ",\"sha256\":\""
            + hash
            + "\"}}}");
    return path;
  }

  private String hash() throws Exception {
    return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(CONTENT));
  }

  private TierBArtifactDownloader.Response response(int status, String length, byte[] body) {
    return new TierBArtifactDownloader.Response(status, length, new ByteArrayInputStream(body));
  }

  @Test
  void downloadsNestedAndVerifiesCachedFiles() throws Exception {
    Path manifest = manifest("1_Pooling/config.json", 10, 10, hash());
    Path target = temporary.resolve("model");
    AtomicInteger gets = new AtomicInteger();
    TierBArtifactDownloader.Transport transport =
        (path, method) -> {
          if (method.equals("GET")) {
            gets.incrementAndGet();
          }
          return response(200, "7", CONTENT);
        };
    TierBArtifactDownloader.download(manifest, target, transport);
    assertEquals("fixture", Files.readString(target.resolve("1_Pooling/config.json")));
    TierBArtifactDownloader.download(manifest, target, transport);
    assertEquals(1, gets.get());
    Files.writeString(target.resolve("1_Pooling/config.json"), "changed");
    TierBArtifactDownloader.download(manifest, target, transport);
    assertEquals(2, gets.get());
  }

  @Test
  void rejectsTraversalAndSymlinkEscapes() throws Exception {
    Path target = temporary.resolve("model");
    for (String name : new String[] {"../outside", "/absolute", "x//y", "x/./y", "x/../y"}) {
      Path manifest = manifest(name, 10, 10, hash());
      assertThrows(
          IllegalArgumentException.class,
          () ->
              TierBArtifactDownloader.download(
                  manifest, target, (p, m) -> response(200, "7", CONTENT)));
    }
    Files.createDirectories(target);
    Files.createSymbolicLink(target.resolve("link"), temporary);
    Path manifest = manifest("link/outside", 10, 10, hash());
    assertThrows(
        IOException.class,
        () ->
            TierBArtifactDownloader.download(
                manifest, target, (p, m) -> response(200, "7", CONTENT)));
  }

  @Test
  void rejectsFinalStatusesMissingLengthsAndStreamedCaps() throws Exception {
    Path target = temporary.resolve("model");
    Path manifest = manifest("config.json", 6, 6, hash());
    for (int status : new int[] {302, 404, 500}) {
      assertThrows(
          IOException.class,
          () ->
              TierBArtifactDownloader.download(
                  manifest, target, (p, m) -> response(status, "6", CONTENT)));
    }
    for (String length : new String[] {null, "bad", "-1", "7"}) {
      assertThrows(
          IOException.class,
          () ->
              TierBArtifactDownloader.download(
                  manifest, target, (p, m) -> response(200, length, CONTENT)));
    }
    assertThrows(
        IOException.class,
        () ->
            TierBArtifactDownloader.download(
                manifest, target, (p, m) -> response(200, "6", CONTENT)));
    assertFalse(Files.exists(target.resolve("config.json.part")));
    assertFalse(Files.exists(target.resolve("config.json")));
  }

  @Test
  void rejectsHashAndHeadGetMismatchAndCleansPartialFiles() throws Exception {
    Path target = temporary.resolve("model");
    Path manifest = manifest("config.json", 10, 10, "0".repeat(64));
    assertThrows(
        IOException.class,
        () ->
            TierBArtifactDownloader.download(
                manifest, target, (p, m) -> response(200, "7", CONTENT)));
    assertFalse(Files.exists(target.resolve("config.json.part")));
    assertThrows(
        IOException.class,
        () ->
            TierBArtifactDownloader.download(
                manifest, target, (p, m) -> response(200, m.equals("HEAD") ? "7" : "8", CONTENT)));
    assertFalse(Files.exists(target.resolve("config.json.part")));
  }

  @Test
  void aggregateCapIncludesPreviouslyVerifiedFiles() throws Exception {
    Path manifest = manifest("a.json", 10, 11, hash());
    String source = Files.readString(manifest);
    Files.writeString(
        manifest,
        source.replace(
            "\"a.json\":",
            "\"b.json\":{\"max_bytes\":10,\"sha256\":\"" + hash() + "\"},\"a.json\":"));
    Path target = temporary.resolve("model");
    assertThrows(
        IOException.class,
        () ->
            TierBArtifactDownloader.download(
                manifest, target, (path, method) -> response(200, "7", CONTENT)));
    assertEquals("fixture", Files.readString(target.resolve("a.json")));
    assertFalse(Files.exists(target.resolve("b.json")));
    assertFalse(Files.exists(target.resolve("b.json.part")));
  }

  @Test
  void productionTransportFollowsCrossHostRedirectChain() throws Exception {
    assertEquals(HttpClient.Redirect.NORMAL, TierBArtifactDownloader.client().followRedirects());
    HttpServer first = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
    HttpServer second = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
    try {
      second.createContext(
          "/final",
          exchange -> {
            exchange.getResponseHeaders().set("Content-Length", "7");
            exchange.sendResponseHeaders(200, exchange.getRequestMethod().equals("HEAD") ? -1 : 7);
            if (!exchange.getRequestMethod().equals("HEAD")) {
              exchange.getResponseBody().write(CONTENT);
            }
            exchange.close();
          });
      first.createContext(
          "/",
          exchange -> {
            exchange
                .getResponseHeaders()
                .set("Location", "http://localhost:" + second.getAddress().getPort() + "/final");
            exchange.sendResponseHeaders(302, -1);
            exchange.close();
          });
      first.start();
      second.start();
      Path manifest = manifest("nested/config.json", 10, 10, hash());
      Path target = temporary.resolve("model");
      TierBArtifactDownloader.download(
          manifest,
          target,
          TierBArtifactDownloader.transport(
              URI.create("http://127.0.0.1:" + first.getAddress().getPort() + "/")));
      assertEquals("fixture", Files.readString(target.resolve("nested/config.json")));
    } finally {
      first.stop(0);
      second.stop(0);
    }
  }
}
