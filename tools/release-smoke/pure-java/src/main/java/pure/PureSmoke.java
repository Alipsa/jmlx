package pure;

import java.awt.GraphicsEnvironment;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.CodeSource;
import java.security.MessageDigest;
import java.time.Clock;
import java.time.Instant;
import java.time.ZoneOffset;
import java.util.Arrays;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.regex.Pattern;
import se.alipsa.jmlx.jinja.RenderOptions;
import se.alipsa.jmlx.jinja.Template;
import se.alipsa.jmlx.tokenizer.ChatTemplateOptions;
import se.alipsa.jmlx.tokenizer.HfTokenizer;
import se.alipsa.jmlx.tokenizer.IncrementalTokenDecoder;
import se.alipsa.jmlx.vision.ImageDecoder;
import se.alipsa.jmlx.vision.ImageTensor;
import se.alipsa.jmlx.vision.ImageTransforms;
import se.alipsa.jmlx.vision.Layout;
import se.alipsa.jmlx.vision.Resampling;
import se.alipsa.jmlx.vision.RgbImage;

/**
 * Pure-Java release smoke: runs against the published jinja/tokenizer/vision jars only, with no
 * native runtime, no model and no GPU. The consumer JVM must be Java 21 and headless; the
 * dependency graph must be exactly jinja (transitively through tokenizer), tokenizer and vision.
 *
 * <p>The fixture facts are pinned in {@code tools/release-smoke/fixtures/pure-java/PROVENANCE.md}
 * and were cross-verified against the pinned HF tokenizers, transformers and Pillow references.
 */
public final class PureSmoke {

  /** Maven Central's timestamped snapshot jar name, normalized back to the plain -SNAPSHOT name. */
  private static final Pattern SNAPSHOT_STAMP = Pattern.compile("-\\d{8}\\.\\d{6}-\\d+");

  private static final int[] PROMPT_IDS = {
    12, 13, 12, 14, 15, 16, 17, 8, 18, 15, 8, 18, 10, 19, 15, 20, 31, 21, 12, 15, 10, 17, 8, 3, 4,
    5, 32, 7, 8, 9, 7, 33, 11, 31, 22, 12, 12, 19, 12, 14, 22, 23, 14, 17, 31, 22, 14, 8, 24, 25,
    24, 26, 27, 28, 28, 27, 25, 29, 30, 28, 24, 17, 25, 25, 31
  };

  public static void main(String[] args) throws Exception {
    Path checkout = Path.of(requireProperty("smoke.checkout.root")).toAbsolutePath().normalize();
    Path fixtures = Path.of(requireProperty("smoke.fixtures"));
    Path coordinates = Path.of(requireProperty("smoke.runtime.coordinates"));
    String mode = requireProperty("smoke.mode");
    Map<String, String> expected = expectedVersions();

    checkEnvironment(coordinates, expected, mode);
    checkJarProvenance(checkout, expected);
    checkJinja();
    checkTokenizer(fixtures);
    checkVision(checkout, fixtures);

    System.out.println("Pure smoke passed: " + expected + " (mode " + mode + ")");
  }

  /**
   * The consumer JVM is Java 21 and headless, the written coordinate report holds exactly the
   * three pure modules (no native-chain component), and in release mode nothing is a snapshot.
   */
  private static void checkEnvironment(Path coordinates, Map<String, String> expected, String mode)
      throws IOException {
    check(Runtime.version().feature() == 21, "consumer JVM must be Java 21, is " + Runtime.version());
    check(GraphicsEnvironment.isHeadless(), "consumer must run headless");
    Set<String> modules = new TreeSet<>();
    for (String raw : Files.readAllLines(coordinates)) {
      String line = raw.trim();
      if (line.isEmpty()) {
        continue;
      }
      String[] parts = line.split(":");
      check(parts.length == 3, "bad coordinate line: " + line);
      if ("se.alipsa".equals(parts[0])) {
        check(expected.containsKey(parts[1]), "unexpected se.alipsa component in the pure graph: " + line);
        modules.add(parts[1]);
      }
      if ("release".equals(mode)) {
        check(!line.endsWith("-SNAPSHOT"), "release mode forbids snapshots: " + line);
      }
    }
    check(
        modules.equals(expected.keySet()),
        "pure graph modules " + modules + " != expected " + expected.keySet());
    // The native chain must not even be loadable: no ffi, core, models or native jar on the classpath.
    for (String nativeClass : List.of("se.alipsa.jmlx.ffi.NativeLoader", "se.alipsa.jmlx.core.MLX")) {
      try {
        Class.forName(nativeClass);
        check(false, "native-chain class is loadable on the pure consumer: " + nativeClass);
      } catch (ClassNotFoundException absent) {
        // expected: the native chain is not in the pure graph
      }
    }
  }

  /**
   * Every probed class comes from a resolved jar of the expected version: not from a checkout build
   * directory, and not from {@code mavenLocal}. In ci/candidate modes the disposable smoke
   * repository inside the checkout is the published location for the local run, and Gradle resolves
   * release artifacts from a local file repository in place (no dependency-cache copy), so a jar
   * under it is the expected code source, not a working-tree leak.
   */
  private static void checkJarProvenance(Path checkout, Map<String, String> expected) throws Exception {
    Path smokeRepo = smokeRepoRoot();
    List<Class<?>> probes = List.of(Template.class, HfTokenizer.class, ImageDecoder.class);
    for (Class<?> probe : probes) {
      CodeSource source = probe.getProtectionDomain().getCodeSource();
      check(source != null, probe + " has no code source");
      Path jar = Path.of(URI.create(source.getLocation().toString())).toAbsolutePath();
      check(jar.toString().endsWith(".jar"), probe + " is not loaded from a jar: " + jar);
      Path normalized = jar.normalize();
      // The smoke's own tree and its disposable repository live inside the checkout, so exclude
      // them; any other path under the checkout (build/libs, build/classes) is a leak.
      check(
          !normalized.startsWith(checkout)
              || normalized.startsWith(checkout.resolve("tools/release-smoke"))
              || (smokeRepo != null && normalized.startsWith(smokeRepo)),
          probe + " came from the checkout: " + jar);
      check(!jar.toString().contains("/.m2/"), probe + " came from mavenLocal: " + jar);
      String artifact = jar.getFileName().toString();
      String module =
          expected.keySet().stream().filter(artifact::startsWith).findFirst().orElse(null);
      check(module != null, "unexpected jar " + artifact);
      String version = SNAPSHOT_STAMP.matcher(artifact).replaceAll("-SNAPSHOT");
      check(
          version.equals(module + "-" + expected.get(module) + ".jar"),
          artifact + " is not the expected version " + expected.get(module));
    }
  }

  /** A small message loop with a known literal output, including non-ASCII, through the jinja edge. */
  private static void checkJinja() {
    Template template = Template.parse("{% for word in words %}{{ word }}-{% endfor %}é");
    String rendered = template.render(Map.of("words", List.of("héllo", "world")));
    check("héllo-world-é".equals(rendered), "jinja render: " + rendered);
  }

  /**
   * The hand-written byte-level BPE tokenizer: exact encode/decode facts, the exact rendered chat
   * prompt (fixed clock) and its exact token ids, and incremental decoding where {@code é} spans two
   * byte tokens (jmlx holds back the incomplete UTF-8 tail until the second byte token arrives).
   */
  private static void checkTokenizer(Path fixtures) {
    HfTokenizer tokenizer = HfTokenizer.fromDirectory(fixtures.resolve("tokenizer"));

    List<Integer> base = tokenizer.encode("héllo world", false);
    check(
        base.equals(List.of(3, 4, 5, 32, 7, 8, 9, 7, 33, 11)),
        "encode(héllo world): " + base);
    check("héllo world".equals(tokenizer.decode(base, false)), "decode round-trip");
    check("héllo world".equals(tokenizer.decode(base, true)), "skip-special round-trip");

    List<Map<String, Object>> messages =
        List.of(
            Map.of("role", "system", "content", "be brief"),
            Map.of("role", "user", "content", "héllo world"));
    RenderOptions clock =
        RenderOptions.builder()
            .clock(Clock.fixed(Instant.parse("2026-11-04T12:00:00Z"), ZoneOffset.UTC))
            .zoneId(ZoneOffset.UTC)
            .build();
    ChatTemplateOptions options = new ChatTemplateOptions("", true, Map.of(), clock);
    String prompt = tokenizer.renderChat(messages, options);
    String expectedPrompt = "system: be brief\nuser: héllo world\nassistant:\nat 2026-11-04T12:00\n";
    check(expectedPrompt.equals(prompt), "rendered prompt: " + escape(prompt));

    List<Integer> promptIds = tokenizer.encode(prompt, false);
    check(
        promptIds.equals(ids(PROMPT_IDS)),
        "prompt ids (" + promptIds.size() + "): " + promptIds);

    IncrementalTokenDecoder decoder = tokenizer.newIncrementalDecoder(false);
    String[] expectedDeltas = {"h", "", "é", "ll", "o", " ", "w", "o", "rl", "d", ""};
    StringBuilder combined = new StringBuilder();
    int i = 0;
    for (int id : base) {
      String delta = decoder.append(id);
      check(expectedDeltas[i].equals(delta), "delta " + i + " for id " + id + ": " + escape(delta));
      combined.append(delta);
      i++;
    }
    check(decoder.finish().isEmpty(), "finish must be empty");
    check("héllo world".equals(combined.toString()), "combined deltas != full decode");

    check("<s></s><unk>".equals(tokenizer.decode(List.of(0, 1, 2), false)), "special decode");
    check(tokenizer.decode(List.of(0, 1, 2), true).isEmpty(), "skip-special decode");
  }

  /**
   * The committed image-oracle resize case: the staged spec must match the oracle's input spec, the
   * pinned sources must match the recorded digests, and the decoded+resized pixels must equal the
   * Pillow-recorded expected bytes. The Pillow resample code maps explicitly (1 LANCZOS, 2 BILINEAR,
   * 3 BICUBIC); any other code is rejected, never treated as a Java enum ordinal. A separate
   * analytic normalization covers the rescale/normalize arithmetic.
   */
  private static void checkVision(Path checkout, Path fixtures) throws Exception {
    Map<String, Object> staged =
        Json.parseObject(Files.readString(fixtures.resolve("resize-case.json")));
    String name = Json.str(staged.get("name"), "case name");
    String image = Json.str(staged.get("image"), "case image");
    int width = Json.num(staged.get("width"), "case width");
    int height = Json.num(staged.get("height"), "case height");
    int resample = Json.num(staged.get("resample"), "case resample");
    check("small-96x64-oddup-101x73-bilinear".equals(name), "case name: " + name);
    check("resize/small-96x64.png".equals(image), "case image: " + image);
    check(width == 101 && height == 73, "case dims " + width + "x" + height);

    // The staged spec must be byte-for-byte the oracle's own case from resize.input.json.
    Path oracleFixtures = checkout.resolve("tools/image-oracle/fixtures");
    Map<String, Object> input =
        Json.parseObject(Files.readString(oracleFixtures.resolve("resize.input.json")));
    Map<String, Object> oracleCase = Json.findCase(input, "cases", name);
    check(image.equals(Json.str(oracleCase.get("image"), "oracle image")), "staged image field");
    check(width == Json.num(oracleCase.get("width"), "oracle width"), "staged width field");
    check(height == Json.num(oracleCase.get("height"), "oracle height"), "staged height field");
    check(resample == Json.num(oracleCase.get("resample"), "oracle resample"), "staged resample field");

    // The expected pixels come from the oracle's expected file, not from this implementation.
    Map<String, Object> expectedFile =
        Json.parseObject(Files.readString(oracleFixtures.resolve("resize.expected.json")));
    Map<String, Object> expectedCase = Json.findCase(expectedFile, "cases", name);
    byte[] expectedPixels =
        Base64.getDecoder().decode(Json.str(expectedCase.get("pixelsBase64"), "pixelsBase64"));

    // The image-oracle provenance pins the input spec and the PNG by SHA-256.
    Map<String, Object> provenance =
        Json.parseObject(Files.readString(checkout.resolve("tools/image-oracle/provenance.json")));
    Map<String, Object> pins = Json.object(provenance.get("fixtureSources"), "fixtureSources");
    check(
        sha256(oracleFixtures.resolve("resize.input.json"))
            .equals(Json.str(pins.get("resize.input.json"), "pin for resize.input.json")),
        "input spec no longer matches its provenance pin");
    check(
        sha256(oracleFixtures.resolve(image))
            .equals(Json.str(pins.get(image), "pin for " + image)),
        "oracle PNG no longer matches its provenance pin");

    // The fixture's recorded digests: staged-file entries resolve against the fixtures directory;
    // source-of-truth entries start with ../ segments and resolve against the checkout's tools/
    // directory (their leading ../ segments are walked off, since the staged fixtures live at a
    // different depth than the committed fixture directory).
    for (Map.Entry<String, String> entry :
        parseDigests(fixtures.resolve("digests.txt")).entrySet()) {
      Path source =
          entry.getKey().startsWith("../")
              ? resolveAgainstTools(checkout, entry.getKey())
              : fixtures.resolve(entry.getKey());
      check(Files.isRegularFile(source), "no fixture source at " + source);
      check(
          sha256(source).equals(entry.getValue()),
          "digest mismatch for " + entry.getKey() + " (recorded " + entry.getValue() + ")");
    }

    // Decode + resize must be byte-exact against the Pillow-recorded bytes.
    Path png = fixtures.resolve("resize-small-96x64.png");
    RgbImage source = ImageDecoder.decode(png);
    check(
        source.width() == 96 && source.height() == 64,
        "decoded dims " + source.width() + "x" + source.height());
    Resampling kernel =
        switch (resample) {
          case 1 -> Resampling.LANCZOS;
          case 2 -> Resampling.BILINEAR;
          case 3 -> Resampling.BICUBIC;
          default ->
              throw new IllegalStateException(
                  "resample code " + resample + " is not a documented Pillow code");
        };
    RgbImage resized = ImageTransforms.resize(source, width, height, kernel);
    check(resized.width() == width && resized.height() == height, "resized dims");
    checkBytes(expectedPixels, resized.pixels(), "resized pixels");
    checkBytes(
        expectedPixels,
        Files.readAllBytes(fixtures.resolve("resize-expected.bin")),
        "staged expected bytes");

    // Analytic normalization: scale 0.5, mean 0.5, std 1.0 over a 2x1 image [0..5].
    RgbImage tiny = new RgbImage(2, 1, new byte[] {0, 1, 2, 3, 4, 5});
    ImageTensor tensor =
        ImageTransforms.normalize(
            tiny, 0.5, new float[] {0.5f, 0.5f, 0.5f}, new float[] {1f, 1f, 1f});
    float[] want = {-0.5f, 0.0f, 0.5f, 1.0f, 1.5f, 2.0f};
    float[] got = tensor.values();
    check(got.length == want.length, "tensor length " + got.length);
    for (int k = 0; k < want.length; k++) {
      check(got[k] == want[k], "tensor[" + k + "] = " + got[k] + ", want " + want[k]);
    }
    check(Arrays.equals(tensor.shape(), new int[] {1, 1, 2, 3}), "shape " + Arrays.toString(tensor.shape()));
    check(tensor.layout() == Layout.NHWC, "layout " + tensor.layout());
  }

  private static Map<String, String> expectedVersions() {
    Map<String, String> out = new LinkedHashMap<>();
    for (String pair : requireProperty("smoke.expected.versions").split(",")) {
      String[] kv = pair.split("=");
      out.put(kv[0], kv[1]);
    }
    return out;
  }

  /** {@code <sha256>  <path>} lines; {@code #} lines are comments. */
  private static Map<String, String> parseDigests(Path digestsFile) throws IOException {
    Map<String, String> out = new LinkedHashMap<>();
    for (String raw : Files.readAllLines(digestsFile)) {
      String line = raw.trim();
      if (line.isEmpty() || line.startsWith("#")) {
        continue;
      }
      int sep = line.indexOf("  ");
      if (sep < 0) {
        throw new IllegalStateException("bad digest line: " + line);
      }
      out.put(line.substring(sep + 2).trim(), line.substring(0, sep).trim());
    }
    check(!out.isEmpty(), "no digests recorded in " + digestsFile);
    return out;
  }

  /** A digest path's leading {@code ../} segments walk up to the checkout's tools/ directory. */
  private static Path resolveAgainstTools(Path checkout, String relative) {
    String rest = relative;
    while (rest.startsWith("../")) {
      rest = rest.substring(3);
    }
    if (rest.isEmpty() || rest.startsWith("..")) {
      throw new IllegalStateException("digest path escapes the tools/ directory: " + relative);
    }
    return checkout.resolve("tools").resolve(rest).normalize();
  }

  private static String sha256(Path path) throws IOException {
    final MessageDigest digest;
    try {
      digest = MessageDigest.getInstance("SHA-256");
    } catch (java.security.NoSuchAlgorithmException impossible) {
      throw new IllegalStateException("SHA-256 is missing from the JDK", impossible);
    }
    byte[] chunk = new byte[8192];
    try (InputStream in = Files.newInputStream(path)) {
      int n;
      while ((n = in.read(chunk)) > 0) {
        digest.update(chunk, 0, n);
      }
    }
    StringBuilder out = new StringBuilder(64);
    for (byte b : digest.digest()) {
      out.append(String.format("%02x", b));
    }
    return out.toString();
  }

  private static void checkBytes(byte[] want, byte[] got, String what) {
    check(want.length == got.length, what + " length " + got.length + ", want " + want.length);
    for (int i = 0; i < want.length; i++) {
      if (want[i] != got[i]) {
        check(false, what + " differ at byte " + i + ": got " + (got[i] & 0xFF) + ", want " + (want[i] & 0xFF));
      }
    }
  }

  /** The disposable local smoke repository this run resolved from, or null for a remote source. */
  private static Path smokeRepoRoot() {
    String url = System.getProperty("smoke.repo.url");
    if (url == null || !url.startsWith("file://")) {
      return null;
    }
    return Path.of(URI.create(url)).toAbsolutePath().normalize();
  }

  private static String requireProperty(String name) {
    String value = System.getProperty(name);
    if (value == null || value.isEmpty()) {
      throw new IllegalStateException("missing required property " + name);
    }
    return value;
  }

  private static void check(boolean condition, String message) {
    if (!condition) {
      throw new AssertionError(message);
    }
  }

  private static String escape(String s) {
    return "\"" + s.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r") + "\"";
  }

  private static List<Integer> ids(int[] values) {
    List<Integer> out = new java.util.ArrayList<>(values.length);
    for (int v : values) {
      out.add(v);
    }
    return out;
  }

  /**
   * Minimal JSON reader for the flat fixture files: objects, arrays, strings, integers, booleans
   * and null. Jackson is a runtime-only dependency of the tokenizer (its {@code implementation}
   * edge), so it is deliberately not used here; the smoke must not gain a compile dependency on it.
   */
  private static final class Json {

    private final String src;
    private int pos;

    private Json(String src) {
      this.src = src;
    }

    @SuppressWarnings("unchecked")
    static Map<String, Object> parseObject(String text) {
      Json parser = new Json(text);
      parser.ws();
      Object value = parser.value();
      parser.ws();
      if (parser.pos != parser.src.length()) {
        throw new IllegalArgumentException("trailing JSON at " + parser.pos);
      }
      return object(value, "root");
    }

    static Map<String, Object> findCase(Map<String, Object> root, String key, String name) {
      for (Object entry : array(root.get(key), key)) {
        Map<String, Object> caseMap = object(entry, key + " entry");
        if (name.equals(caseMap.get("name"))) {
          return caseMap;
        }
      }
      throw new IllegalStateException("no case named " + name + " in " + key);
    }

    @SuppressWarnings("unchecked")
    static Map<String, Object> object(Object value, String what) {
      if (!(value instanceof Map)) {
        throw new IllegalArgumentException(what + " is not an object");
      }
      return (Map<String, Object>) value;
    }

    @SuppressWarnings("unchecked")
    static List<Object> array(Object value, String what) {
      if (!(value instanceof List)) {
        throw new IllegalArgumentException(what + " is not an array");
      }
      return (List<Object>) value;
    }

    static String str(Object value, String what) {
      if (!(value instanceof String)) {
        throw new IllegalArgumentException(what + " is not a string");
      }
      return (String) value;
    }

    static int num(Object value, String what) {
      if (!(value instanceof Long)) {
        throw new IllegalArgumentException(what + " is not an integer");
      }
      return ((Long) value).intValue();
    }

    private Object value() {
      ws();
      return switch (peek()) {
        case '{' -> objectValue();
        case '[' -> arrayValue();
        case '"' -> stringValue();
        case 't', 'f' -> boolValue();
        case 'n' -> nullValue();
        default -> numberValue();
      };
    }

    private Map<String, Object> objectValue() {
      Map<String, Object> out = new LinkedHashMap<>();
      expect('{');
      ws();
      if (peek() == '}') {
        pos++;
        return out;
      }
      while (true) {
        ws();
        String key = stringValue();
        ws();
        expect(':');
        out.put(key, value());
        ws();
        char c = next();
        if (c == ',') {
          continue;
        }
        if (c == '}') {
          return out;
        }
        throw new IllegalArgumentException("expected , or } at " + (pos - 1) + ", got " + c);
      }
    }

    private List<Object> arrayValue() {
      List<Object> out = new java.util.ArrayList<>();
      expect('[');
      ws();
      if (peek() == ']') {
        pos++;
        return out;
      }
      while (true) {
        out.add(value());
        ws();
        char c = next();
        if (c == ',') {
          continue;
        }
        if (c == ']') {
          return out;
        }
        throw new IllegalArgumentException("expected , or ] at " + (pos - 1) + ", got " + c);
      }
    }

    private String stringValue() {
      expect('"');
      StringBuilder out = new StringBuilder();
      while (true) {
        char c = next();
        if (c == '"') {
          return out.toString();
        }
        if (c != '\\') {
          out.append(c);
          continue;
        }
        char escape = next();
        switch (escape) {
          case '"' -> out.append('"');
          case '\\' -> out.append('\\');
          case '/' -> out.append('/');
          case 'b' -> out.append('\b');
          case 'f' -> out.append('\f');
          case 'n' -> out.append('\n');
          case 'r' -> out.append('\r');
          case 't' -> out.append('\t');
          case 'u' -> {
            out.append((char) Integer.parseInt(src.substring(pos, pos + 4), 16));
            pos += 4;
          }
          default -> throw new IllegalArgumentException("bad escape \\" + escape);
        }
      }
    }

    private Object numberValue() {
      int start = pos;
      while (pos < src.length() && "+-0123456789.eE".indexOf(src.charAt(pos)) >= 0) {
        pos++;
      }
      String text = src.substring(start, pos);
      if (text.isEmpty() || text.contains(".") || text.contains("e") || text.contains("E")) {
        throw new IllegalArgumentException("only integers are supported by the smoke JSON reader: " + text);
      }
      return Long.parseLong(text);
    }

    private Object boolValue() {
      if (src.startsWith("true", pos)) {
        pos += 4;
        return Boolean.TRUE;
      }
      if (src.startsWith("false", pos)) {
        pos += 5;
        return Boolean.FALSE;
      }
      throw new IllegalArgumentException("bad literal at " + pos);
    }

    private Object nullValue() {
      if (src.startsWith("null", pos)) {
        pos += 4;
        return null;
      }
      throw new IllegalArgumentException("bad literal at " + pos);
    }

    private void ws() {
      while (pos < src.length() && Character.isWhitespace(src.charAt(pos))) {
        pos++;
      }
    }

    private char peek() {
      if (pos >= src.length()) {
        throw new IllegalArgumentException("unexpected end of JSON");
      }
      return src.charAt(pos);
    }

    private char next() {
      char c = peek();
      pos++;
      return c;
    }

    private void expect(char c) {
      char actual = next();
      if (actual != c) {
        throw new IllegalArgumentException("expected " + c + " at " + (pos - 1) + ", got " + actual);
      }
    }
  }
}
