package se.alipsa.jmlx.buildsrc;

import com.sun.source.tree.CompilationUnitTree;
import com.sun.source.tree.IdentifierTree;
import com.sun.source.tree.ImportTree;
import com.sun.source.tree.MemberSelectTree;
import com.sun.source.tree.Tree;
import com.sun.source.util.JavacTask;
import com.sun.source.util.SourcePositions;
import com.sun.source.util.TreePathScanner;
import com.sun.source.util.Trees;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;
import javax.tools.JavaCompiler;
import javax.tools.StandardJavaFileManager;
import javax.tools.ToolProvider;

/**
 * Verifies that handwritten Java uses of generated MLX bindings have an explicit inventory record.
 */
public final class MlxApiCallSites {

  private static final String FFI_PACKAGE = "se.alipsa.jmlx.ffi.";
  private static final String GENERATED_DIRECTORY = "jmlx-ffi/src/main/generated/java";

  private MlxApiCallSites() {}

  /** Fails with every unrecorded generated-binding use in handwritten Java source. */
  public static void verify(Path repositoryRoot) throws IOException {
    Set<String> mapped = MlxApiInventory.explicitMappings(repositoryRoot);
    List<Violation> violations = new ArrayList<>();
    for (Path source : handwrittenJavaSources(repositoryRoot)) {
      violations.addAll(scan(repositoryRoot, source, mapped));
    }
    if (!violations.isEmpty()) {
      violations.sort(Comparator.comparing(Violation::path).thenComparingLong(Violation::line));
      StringBuilder message = new StringBuilder("Unrecorded generated MLX binding use(s):\n");
      for (Violation violation : violations) {
        message
            .append("- ")
            .append(violation.path())
            .append(":")
            .append(violation.line())
            .append(" ")
            .append(violation.identity())
            .append("; add an explicit req/mlx-api-inventory-overrides.json record\n");
      }
      throw new IllegalStateException(message.toString());
    }
  }

  private static List<Path> handwrittenJavaSources(Path repositoryRoot) throws IOException {
    try (Stream<Path> paths = Files.walk(repositoryRoot)) {
      return paths
          .filter(path -> path.toString().endsWith(".java"))
          .filter(path -> !repositoryRoot.relativize(path).startsWith(GENERATED_DIRECTORY))
          .filter(path -> !path.toString().contains("/build/"))
          .sorted()
          .toList();
    }
  }

  private static List<Violation> scan(Path repositoryRoot, Path source, Set<String> mapped)
      throws IOException {
    JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
    if (compiler == null) {
      throw new IllegalStateException("verifyMlxApiCallSites requires a JDK compiler");
    }
    try (StandardJavaFileManager files = compiler.getStandardFileManager(null, null, null)) {
      JavacTask task =
          (JavacTask)
              compiler.getTask(
                  null, files, null, List.of("-proc:none"), null, files.getJavaFileObjects(source));
      Iterable<? extends CompilationUnitTree> units = task.parse();
      SourcePositions positions = Trees.instance(task).getSourcePositions();
      List<Violation> violations = new ArrayList<>();
      for (CompilationUnitTree unit : units) {
        new Scanner(repositoryRoot, unit, positions, mapped, violations).scan(unit, null);
      }
      return violations;
    }
  }

  private record Violation(String path, long line, String identity) {}

  private static final class Scanner extends TreePathScanner<Void, Void> {
    private final Path repositoryRoot;
    private final CompilationUnitTree unit;
    private final SourcePositions positions;
    private final Set<String> mapped;
    private final List<Violation> violations;
    private final Map<String, String> imports = new HashMap<>();
    private final Set<String> seen = new HashSet<>();

    Scanner(
        Path repositoryRoot,
        CompilationUnitTree unit,
        SourcePositions positions,
        Set<String> mapped,
        List<Violation> violations) {
      this.repositoryRoot = repositoryRoot;
      this.unit = unit;
      this.positions = positions;
      this.mapped = mapped;
      this.violations = violations;
      readImports(unit.getImports());
    }

    @Override
    public Void visitMemberSelect(MemberSelectTree tree, Void unused) {
      String owner = tree.getExpression().toString();
      String member = tree.getIdentifier().toString();
      if (owner.equals("mlx_h") || owner.endsWith(".mlx_h")) {
        record("mlx_h." + member, tree, member.startsWith("mlx_") || member.startsWith("MLX_"));
      } else if (owner.startsWith(FFI_PACKAGE) && isGeneratedType(member)) {
        record(member, tree, true);
      }
      return super.visitMemberSelect(tree, unused);
    }

    @Override
    public Void visitIdentifier(IdentifierTree tree, Void unused) {
      String identity = imports.get(tree.getName().toString());
      if (identity != null) {
        record(identity, tree, true);
      }
      return super.visitIdentifier(tree, unused);
    }

    private void readImports(List<? extends ImportTree> importTrees) {
      for (ImportTree importTree : importTrees) {
        String imported = importTree.getQualifiedIdentifier().toString();
        if (!imported.startsWith(FFI_PACKAGE)) {
          continue;
        }
        String suffix = imported.substring(FFI_PACKAGE.length());
        if (importTree.isStatic()) {
          if (suffix.startsWith("mlx_h.mlx_") || suffix.startsWith("mlx_h.MLX_")) {
            record("mlx_h." + suffix.substring("mlx_h.".length()), importTree, true);
          }
        } else if (isGeneratedType(suffix)) {
          imports.put(suffix, suffix);
        }
      }
    }

    private static boolean isGeneratedType(String name) {
      return name.endsWith("_")
          || name.contains("$fun")
          || name.contains("$dtor")
          || name.equals("mlx_error_handler_func");
    }

    private void record(String identity, Tree tree, boolean generatedIdentity) {
      if (!generatedIdentity || mapped.contains(identity)) {
        return;
      }
      long position = positions.getStartPosition(unit, tree);
      long line = unit.getLineMap().getLineNumber(position);
      String key = line + ":" + identity;
      if (seen.add(key)) {
        violations.add(
            new Violation(
                repositoryRoot.relativize(Path.of(unit.getSourceFile().toUri())).toString(),
                line,
                identity));
      }
    }
  }
}
