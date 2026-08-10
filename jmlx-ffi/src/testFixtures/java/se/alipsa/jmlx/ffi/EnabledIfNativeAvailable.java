package se.alipsa.jmlx.ffi;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import org.junit.jupiter.api.extension.ConditionEvaluationResult;
import org.junit.jupiter.api.extension.ExecutionCondition;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.extension.ExtensionContext;

/**
 * Skips a test unless {@link NativeLoader#ensureLoaded()} succeeds.
 *
 * <p>
 * Delegates to {@code NativeLoader}'s own resolution rather than probing for a directory independently
 * (req/initial-plan.md, Testing approach): divergent logic between the skip gate and the loader produces false skips
 * (library present via system property, tests silently do not run) and false failures (directory present, dylib wrong
 * arch).
 */
@Retention(RetentionPolicy.RUNTIME)
@Target({ElementType.TYPE, ElementType.METHOD})
@ExtendWith(EnabledIfNativeAvailable.Condition.class)
public @interface EnabledIfNativeAvailable {

  /** Backs {@link EnabledIfNativeAvailable} via JUnit's {@link ExtendWith} mechanism. */
  final class Condition implements ExecutionCondition {
    @Override
    public ConditionEvaluationResult evaluateExecutionCondition(ExtensionContext context) {
      try {
        NativeLoader.ensureLoaded();
        return ConditionEvaluationResult.enabled("jmlx native library loaded");
      } catch (RuntimeException e) {
        return ConditionEvaluationResult.disabled("jmlx native library not available: " + e.getMessage());
      }
    }
  }
}
