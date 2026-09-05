package se.alipsa.jmlx.buildsrc;

import com.sun.source.tree.CompilationUnitTree;
import com.sun.source.tree.IdentifierTree;
import com.sun.source.tree.ImportTree;
import com.sun.source.tree.MemberReferenceTree;
import com.sun.source.tree.MemberSelectTree;
import com.sun.source.tree.Tree;
import com.sun.source.util.JavacTask;
import com.sun.source.util.SourcePositions;
import com.sun.source.util.TreePath;
import com.sun.source.util.TreePathScanner;
import com.sun.source.util.Trees;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.tools.Diagnostic;
import javax.tools.DiagnosticCollector;
import javax.tools.JavaCompiler;
import javax.tools.JavaFileObject;
import javax.tools.StandardJavaFileManager;
import javax.tools.ToolProvider;

/**
 * Verifies that handwritten Java uses of generated MLX bindings have an explicit inventory record.
 */
public final class MlxApiCallSites {

  private static final String FFI_PACKAGE = "se.alipsa.jmlx.ffi.";
  private static final String FFI_PACKAGE_NAME = "se.alipsa.jmlx.ffi";
  private static final String SHARED_INFRASTRUCTURE_TYPE = "mlx_h$shared";
  private static final Set<String> MLX_H_INFRASTRUCTURE =
      Set.of(
          "align",
          "LIBRARY_ARENA",
          "SYMBOL_LOOKUP",
          "traceDowncall",
          "TRACE_DOWNCALLS",
          "upcallHandle");

  private MlxApiCallSites() {}

  /**
   * Entry point for the toolchain task declared in the root build; the required parser feature
   * level is the second argument.
   */
  public static void main(String[] arguments) {
    try {
      if (arguments.length != 2) {
        throw new IllegalArgumentException(
            "Usage: MlxApiCallSites <repository-root> <java-feature>");
      }
      verify(Path.of(arguments[0]), Integer.parseInt(arguments[1]));
    } catch (IllegalArgumentException | IllegalStateException reported) {
      System.err.println(
          reported.getMessage() == null ? reported.getClass().getName() : reported.getMessage());
      System.exit(1);
    } catch (IOException | RuntimeException unexpected) {
      unexpected.printStackTrace();
      System.exit(1);
    }
  }

  /**
   * Verifies handwritten Java without a minimum parser-JDK check; intended for direct unit tests.
   */
  public static void verify(Path repositoryRoot) throws IOException {
    verify(repositoryRoot, 0);
  }

  static void verify(Path repositoryRoot, int requiredJavaFeature) throws IOException {
    if (Runtime.version().feature() < requiredJavaFeature) {
      throw new IllegalStateException(
          "verifyMlxApiCallSites requires JDK >= "
              + requiredJavaFeature
              + " to parse this project's Java sources; running JDK is "
              + Runtime.version().feature());
    }
    RepositorySources repositorySources = RepositorySources.discover(repositoryRoot);
    MlxApiInventory.GuardData guardData =
        MlxApiInventory.guardData(repositoryRoot, repositorySources);
    List<Violation> violations = new ArrayList<>();
    Set<String> observed = new HashSet<>();
    violations.addAll(
        scan(
            repositoryRoot,
            repositorySources.handwrittenJavaSources(),
            guardData.mappingSources(),
            guardData.generatedTypes(),
            observed));
    for (String mapping : guardData.observedUseRequired()) {
      if (!observed.contains(mapping)) {
        violations.add(
            new Violation(
                "req/mlx-api-inventory-overrides.json",
                1,
                mapping,
                "remove the stale inventory mapping or add its handwritten use"));
      }
    }
    if (!violations.isEmpty()) {
      violations.sort(Comparator.comparing(Violation::path).thenComparingLong(Violation::line));
      StringBuilder message = new StringBuilder("MLX binding inventory guard failures:\n");
      for (Violation violation : violations) {
        message
            .append("- ")
            .append(violation.path())
            .append(":")
            .append(violation.line())
            .append(" ")
            .append(violation.identity())
            .append("; ")
            .append(violation.remedy())
            .append("\n");
      }
      throw new IllegalStateException(message.toString());
    }
  }

  private static List<Violation> scan(
      Path repositoryRoot,
      List<Path> sources,
      Map<String, Set<String>> mappingSources,
      Set<String> generatedTypes,
      Set<String> observed)
      throws IOException {
    if (sources.isEmpty()) {
      return List.of();
    }
    JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
    if (compiler == null) {
      throw new IllegalStateException("verifyMlxApiCallSites requires a JDK compiler");
    }
    DiagnosticCollector<JavaFileObject> diagnostics = new DiagnosticCollector<>();
    try (StandardJavaFileManager files = compiler.getStandardFileManager(diagnostics, null, null)) {
      JavacTask task =
          (JavacTask)
              compiler.getTask(
                  null,
                  files,
                  diagnostics,
                  List.of("-proc:none"),
                  null,
                  files.getJavaFileObjectsFromPaths(sources));
      Iterable<? extends CompilationUnitTree> units = task.parse();
      List<Diagnostic<? extends JavaFileObject>> errors =
          diagnostics.getDiagnostics().stream()
              .filter(diagnostic -> diagnostic.getKind() == Diagnostic.Kind.ERROR)
              .toList();
      if (!errors.isEmpty()) {
        StringBuilder message = new StringBuilder("Unable to parse handwritten Java source:\n");
        for (Diagnostic<? extends JavaFileObject> error : errors) {
          message
              .append("- ")
              .append(error.getSource() == null ? "<unknown>" : error.getSource().getName())
              .append(":" + error.getLineNumber() + " ")
              .append(error.getMessage(null))
              .append("\n");
        }
        throw new IllegalStateException(message.toString());
      }
      SourcePositions positions = Trees.instance(task).getSourcePositions();
      List<Violation> violations = new ArrayList<>();
      for (CompilationUnitTree unit : units) {
        new Scanner(
                repositoryRoot,
                unit,
                positions,
                mappingSources,
                generatedTypes,
                violations,
                observed)
            .scan(unit, null);
      }
      return violations;
    }
  }

  private record Violation(String path, long line, String identity, String remedy) {}

  private static final class Scanner extends TreePathScanner<Void, Void> {
    private final CompilationUnitTree unit;
    private final SourcePositions positions;
    private final Map<String, Set<String>> mappingSources;
    private final Set<String> generatedTypes;
    private final List<Violation> violations;
    private final Set<String> observed;
    private final String sourcePath;
    private final boolean inFfiPackage;
    private final Map<String, String> imports = new HashMap<>();
    private final Set<String> seen = new HashSet<>();

    Scanner(
        Path repositoryRoot,
        CompilationUnitTree unit,
        SourcePositions positions,
        Map<String, Set<String>> mappingSources,
        Set<String> generatedTypes,
        List<Violation> violations,
        Set<String> observed) {
      this.unit = unit;
      this.positions = positions;
      this.mappingSources = mappingSources;
      this.generatedTypes = generatedTypes;
      this.violations = violations;
      this.observed = observed;
      this.sourcePath = repositoryRoot.relativize(Path.of(unit.getSourceFile().toUri())).toString();
      this.inFfiPackage =
          unit.getPackageName() != null
              && unit.getPackageName().toString().equals(FFI_PACKAGE_NAME);
      readImports(unit.getImports());
    }

    @Override
    public Void visitMemberSelect(MemberSelectTree tree, Void unused) {
      String owner = tree.getExpression().toString();
      String member = tree.getIdentifier().toString();
      recordMember(owner, member, tree);
      return super.visitMemberSelect(tree, unused);
    }

    @Override
    public Void visitMemberReference(MemberReferenceTree tree, Void unused) {
      recordMember(tree.getQualifierExpression().toString(), tree.getName().toString(), tree);
      return super.visitMemberReference(tree, unused);
    }

    @Override
    public Void visitIdentifier(IdentifierTree tree, Void unused) {
      String name = tree.getName().toString();
      String identity = imports.get(name);
      if (identity == null && inFfiPackage && generatedTypes.contains(name)) {
        identity = name;
      }
      if (identity != null && !isAllowedSharedInfrastructureQualifier(identity, tree)) {
        record(identity, tree);
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
        if (suffix.equals("*") || suffix.endsWith(".*")) {
          rejectStarImport(importTree);
          continue;
        }
        if (importTree.isStatic()) {
          if (suffix.startsWith("mlx_h.")) {
            String member = suffix.substring("mlx_h.".length());
            if ((member.startsWith("mlx_")
                    || member.startsWith("_mlx_")
                    || member.startsWith("MLX_"))
                && !isNamedInfrastructure(member)) {
              record("mlx_h." + member, importTree);
            }
          } else if (suffix.startsWith(SHARED_INFRASTRUCTURE_TYPE + ".")) {
            String member = suffix.substring((SHARED_INFRASTRUCTURE_TYPE + ".").length());
            if (!member.startsWith("C_")) {
              record(SHARED_INFRASTRUCTURE_TYPE, importTree);
            }
          }
        } else if (generatedTypes.contains(suffix)) {
          imports.put(suffix, suffix);
        }
      }
    }

    private static boolean isNamedInfrastructure(String member) {
      return member.startsWith("C_")
          || MLX_H_INFRASTRUCTURE.contains(member)
          || member.matches("_?mlx_[A-Za-z0-9_]+\\$(handle|descriptor|address)");
    }

    private boolean isAllowedSharedInfrastructureQualifier(String identity, IdentifierTree tree) {
      if (!identity.equals(SHARED_INFRASTRUCTURE_TYPE)) {
        return false;
      }
      TreePath path = getCurrentPath();
      if (path == null || path.getParentPath() == null) {
        return false;
      }
      Tree parent = path.getParentPath().getLeaf();
      return parent instanceof MemberSelectTree memberSelect
          && memberSelect.getExpression() == tree
          && memberSelect.getIdentifier().toString().startsWith("C_");
    }

    private void recordMember(String owner, String member, Tree tree) {
      if (owner.equals("mlx_h") || owner.endsWith(".mlx_h")) {
        if (!isNamedInfrastructure(member)) {
          record("mlx_h." + member, tree);
        }
        return;
      }
      String type =
          owner.startsWith(FFI_PACKAGE) ? owner.substring(owner.lastIndexOf('.') + 1) : owner;
      if (generatedTypes.contains(type)
          && (owner.startsWith(FFI_PACKAGE) || imports.containsKey(type))) {
        if (!type.equals(SHARED_INFRASTRUCTURE_TYPE) || !member.startsWith("C_")) {
          record(type, tree);
        }
      }
    }

    private void record(String identity, Tree tree) {
      observed.add(identity);
      Set<String> sources = mappingSources.get(identity);
      if (sources != null && (sources.isEmpty() || sources.contains(sourcePath))) {
        return;
      }
      addViolation(identity, tree);
    }

    private void rejectStarImport(ImportTree tree) {
      addViolation(
          FFI_PACKAGE_NAME + ".* (star imports are not allowed)",
          tree,
          "replace the star import with explicit generated binding imports");
    }

    private void addViolation(String identity, Tree tree) {
      addViolation(identity, tree, "add an explicit req/mlx-api-inventory-overrides.json record");
    }

    private void addViolation(String identity, Tree tree, String remedy) {
      long position = positions.getStartPosition(unit, tree);
      long line = unit.getLineMap().getLineNumber(position);
      String key = line + ":" + identity;
      if (seen.add(key)) {
        violations.add(new Violation(sourcePath, line, identity, remedy));
      }
    }
  }
}
