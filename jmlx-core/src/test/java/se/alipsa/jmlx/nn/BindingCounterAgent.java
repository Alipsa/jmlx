package se.alipsa.jmlx.nn;

import java.lang.classfile.ClassFile;
import java.lang.classfile.CodeElement;
import java.lang.classfile.CodeTransform;
import java.lang.classfile.MethodModel;
import java.lang.classfile.MethodTransform;
import java.lang.classfile.instruction.InvokeInstruction;
import java.lang.constant.ClassDesc;
import java.lang.constant.ConstantDescs;
import java.lang.constant.MethodTypeDesc;
import java.lang.instrument.ClassFileTransformer;
import java.lang.instrument.Instrumentation;
import java.security.ProtectionDomain;
import java.util.Map;
import java.util.TreeMap;

/** Opt-in benchmark agent: counts actual generated MethodHandle downcalls in a separate JVM run. */
public final class BindingCounterAgent {
  private static final String PREFIX = "jmlx.qwen35.binding.count.";
  private static final String INSTALLED = "jmlx.qwen35.binding.installed";
  private static final String RECORDING = "jmlx.qwen35.binding.recording";
  private static final String FAILURE = "jmlx.qwen35.binding.failure";

  private BindingCounterAgent() {}

  /** Installs entry instrumentation without editing any generated source or reference math. */
  public static void premain(String arguments, Instrumentation instrumentation) {
    System.setProperty(INSTALLED, "true");
    instrumentation.addTransformer(
        new ClassFileTransformer() {
          @Override
          public byte[] transform(
              ClassLoader loader,
              String name,
              Class<?> type,
              ProtectionDomain domain,
              byte[] bytes) {
            if (name == null || !name.startsWith("se/alipsa/jmlx/ffi/mlx_h")) {
              return null;
            }
            try {
              ClassFile classFile = ClassFile.of();
              return classFile.transformClass(
                  classFile.parse(bytes),
                  (builder, element) -> {
                    if (element instanceof MethodModel method && isDowncall(method)) {
                      builder.transformMethod(
                          method,
                          MethodTransform.transformingCode(
                              CodeTransform.ofStateful(
                                  () ->
                                      new CodeTransform() {
                                        private boolean first = true;

                                        @Override
                                        public void accept(
                                            java.lang.classfile.CodeBuilder code,
                                            CodeElement instruction) {
                                          if (first) {
                                            code.ldc(method.methodName().stringValue())
                                                .invokestatic(
                                                    ClassDesc.of(
                                                        BindingCounterAgent.class.getName()),
                                                    "hit",
                                                    MethodTypeDesc.of(
                                                        ConstantDescs.CD_void,
                                                        ConstantDescs.CD_String));
                                            first = false;
                                          }
                                          code.with(instruction);
                                        }
                                      })));
                    } else {
                      builder.with(element);
                    }
                  });
            } catch (RuntimeException | Error error) {
              System.setProperty(FAILURE, error.toString());
              return null;
            }
          }
        });
  }

  private static boolean isDowncall(MethodModel method) {
    return method.methodName().stringValue().matches("mlx_[a-z0-9_]+")
        && method
            .code()
            .map(
                code ->
                    code.elementStream()
                        .anyMatch(
                            element ->
                                element instanceof InvokeInstruction invoke
                                    && invoke.name().equalsString("invokeExact")))
            .orElse(false);
  }

  /** Called only by instrumented generated downcall bodies. */
  public static void hit(String name) {
    if (Boolean.getBoolean(RECORDING)) {
      synchronized (System.getProperties()) {
        String key = PREFIX + name;
        long count = Long.parseLong(System.getProperty(key, "0"));
        System.setProperty(key, Long.toString(count + 1));
      }
    }
  }

  /** Whether this JVM was explicitly launched with this benchmark agent. */
  public static boolean isInstalled() {
    return Boolean.getBoolean(INSTALLED);
  }

  /** Begins one counted sample; callers never use that sample for timing comparisons. */
  public static void start() {
    if (!isInstalled() || System.getProperty(FAILURE) != null) {
      throw new IllegalStateException(
          "Binding-count agent was not installed successfully: " + System.getProperty(FAILURE));
    }
    // String properties share the counter across Gradle's isolated test/agent classloaders.
    System.getProperties().stringPropertyNames().stream()
        .filter(key -> key.startsWith(PREFIX))
        .forEach(System::clearProperty);
    System.setProperty(RECORDING, "true");
  }

  /** Ends a sample and returns attempted native downcall counts by exact binding name. */
  public static Map<String, Long> stop() {
    System.setProperty(RECORDING, "false");
    if (System.getProperty(FAILURE) != null) {
      throw new IllegalStateException(
          "Binding instrumentation failed: " + System.getProperty(FAILURE));
    }
    Map<String, Long> result = new TreeMap<>();
    System.getProperties().stringPropertyNames().stream()
        .filter(key -> key.startsWith(PREFIX))
        .forEach(
            key ->
                result.put(
                    key.substring(PREFIX.length()), Long.parseLong(System.getProperty(key))));
    return result;
  }
}
