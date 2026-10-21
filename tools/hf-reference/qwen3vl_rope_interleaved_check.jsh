// Phase 7.3b WP4 Java twin of qwen3vl_rope_interleaved_check.py: reads the
// pinned-reference float32 case table printed by that probe and checks
// jmlx's frozen se.alipsa.jmlx.nn.MultiAxisRope component against it with a
// strict 1e-5 max absolute error, plus rotate self-consistency (two rotates
// of the same input must be bit-identical). It is never invoked by Gradle
// and writes no golden; the pinned transformers output is authoritative, so
// a divergence means the Java component is fixed, not the probe.
//
// Invocation (from the repository root; the jars must be built first):
//
//   ./gradlew :jmlx-core:jar :jmlx-ffi:jar
//   tools/hf-reference/.venv/bin/python \
//     tools/hf-reference/qwen3vl_rope_interleaved_check.py \
//     > build/qwen3vl-rope-reference.txt
//   JMLX_LIBRARY_PATH=$PWD/native/install/lib \
//   JMLX_ROPE_REFERENCE=$PWD/build/qwen3vl-rope-reference.txt \
//     "$JDK25_HOME/bin/jshell" --execution local -s -J--enable-native-access=ALL-UNNAMED \
//     --class-path jmlx-core/build/libs/jmlx-core-0.5.0-SNAPSHOT.jar:jmlx-ffi/build/libs/jmlx-ffi-0.5.0-SNAPSHOT.jar \
//     tools/hf-reference/qwen3vl_rope_interleaved_check.jsh
//
// $JDK25_HOME is the Java 25 toolchain home the build uses (on this machine:
// ~/.gradle/jdks/eclipse_adoptium-25-aarch64-os_x.2/jdk-25.0.4.1+1/Contents/Home).
// JMLX_LIBRARY_PATH is the staged native library directory that NativeLoader
// resolves after the jmlx.library.path property. jshell has no argv, so the
// table path comes from the JMLX_ROPE_REFERENCE environment variable.
// --execution local runs the snippets inside the jshell process, so the
// System.exit verdict is the probe's exit code (the default child-engine
// mode swallows it); -s suppresses jshell's prompts.
//
// Exit codes: 0 all cases within 1e-5; 1 any case over the threshold or a
// rotate self-consistency failure; 2 missing JMLX_ROPE_REFERENCE or an
// unrecognized table; 3 any other failure.

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.MultiAxisRope;
import se.alipsa.jmlx.nn.MultiAxisRotation;

final class Cursor {
  final List<String> tokens;
  int pos;

  Cursor(List<String> tokens) {
    this.tokens = tokens;
  }

  String next() {
    if (pos >= tokens.size()) {
      throw new IllegalStateException("reference table ended unexpectedly");
    }
    return tokens.get(pos++);
  }

  int nextInt() {
    return Integer.parseInt(next());
  }

  int[] nextInts(int count) {
    int[] out = new int[count];
    for (int i = 0; i < count; i++) {
      out[i] = nextInt();
    }
    return out;
  }

  float[] nextFloats(int count) {
    float[] out = new float[count];
    for (int i = 0; i < count; i++) {
      out[i] = (float) Double.parseDouble(next());
    }
    return out;
  }

  void expect(String key) {
    String got = next();
    if (!got.equals(key)) {
      throw new IllegalStateException("expected '" + key + "' but found '" + got + "'");
    }
  }

  static double maxAbs(float[] actual, float[] expected) {
    double max = 0.0;
    for (int i = 0; i < actual.length; i++) {
      max = Math.max(max, Math.abs((double) actual[i] - (double) expected[i]));
    }
    return max;
  }
}

String refPath = System.getenv("JMLX_ROPE_REFERENCE");
if (refPath == null) {
  System.err.println(
      "set JMLX_ROPE_REFERENCE to the table printed by qwen3vl_rope_interleaved_check.py");
  System.exit(2);
}

double threshold = 1e-5;
int caseCount = 0;
int failureCount = 0;
double worstError = 0.0;
String worstCase = "";
try {
  List<String> tokens = new ArrayList<>();
  for (String line : Files.readAllLines(Path.of(refPath))) {
    for (String part : line.split("\\s+")) {
      if (!part.isEmpty()) {
        tokens.add(part);
      }
    }
  }
  Cursor cursor = new Cursor(tokens);
  if (!"#".equals(cursor.next())
      || !"qwen3vl-rope-interleaved-reference".equals(cursor.next())
      || !"v1".equals(cursor.next())) {
    throw new IllegalStateException("unrecognized reference table schema");
  }
  for (String key : new String[] {"transformers", "torch", "python", "platform"}) {
    cursor.expect(key);
    cursor.next();
  }
  while (cursor.pos < cursor.tokens.size()) {
    cursor.expect("case");
    String name = cursor.next();
    cursor.expect("theta");
    int theta = cursor.nextInt();
    cursor.expect("head_dim");
    int headDim = cursor.nextInt();
    cursor.expect("sections");
    int[] sections = cursor.nextInts(3);
    cursor.expect("batch");
    int batch = cursor.nextInt();
    cursor.expect("heads");
    int heads = cursor.nextInt();
    cursor.expect("sequence");
    int sequence = cursor.nextInt();
    cursor.expect("positions_t");
    int[] t = cursor.nextInts(batch * sequence);
    cursor.expect("positions_h");
    int[] h = cursor.nextInts(batch * sequence);
    cursor.expect("positions_w");
    int[] w = cursor.nextInts(batch * sequence);
    int count = batch * heads * sequence * headDim;
    cursor.expect("x");
    float[] x = cursor.nextFloats(count);
    cursor.expect("q");
    float[] q = cursor.nextFloats(count);
    cursor.expect("k");
    float[] k = cursor.nextFloats(count);

    float[] y;
    boolean selfConsistent;
    try (MLXScope scope = new MLXScope()) {
      MultiAxisRope rope = new MultiAxisRope(theta, sections);
      MultiAxisRotation rotation = rope.prepare(scope, batch, sequence, new int[][] {t, h, w});
      MLXArray xArray = MLX.array(scope, x, new int[] {batch, heads, sequence, headDim});
      y = rotation.rotate(xArray).toFloatArray();
      float[] yRepeat = rotation.rotate(xArray).toFloatArray();
      selfConsistent = Arrays.equals(y, yRepeat);
    }
    double maxQ = Cursor.maxAbs(y, q);
    double maxK = Cursor.maxAbs(y, k);
    boolean pass = selfConsistent && maxQ <= threshold && maxK <= threshold;
    caseCount++;
    if (Math.max(maxQ, maxK) > worstError) {
      worstError = Math.max(maxQ, maxK);
      worstCase = name;
    }
    if (!pass) {
      failureCount++;
    }
    System.out.printf(
        "case %s: max_abs_error_q=%.3e max_abs_error_k=%.3e self_consistent=%b %s%n",
        name, maxQ, maxK, selfConsistent, pass ? "OK" : "MISMATCH");
  }
  if (caseCount == 0) {
    throw new IllegalStateException("no cases found in the reference table");
  }
  if (failureCount > 0) {
    System.out.println(
        failureCount + "/" + caseCount + " cases failed; worst " + worstError + " in " + worstCase);
    System.exit(1);
  }
  System.out.println("all " + caseCount + " cases match MultiAxisRope within " + threshold);
  System.exit(0);
} catch (Exception e) {
  System.err.println("qwen3vl_rope_interleaved_check.jsh failed: " + e);
  System.exit(3);
}
