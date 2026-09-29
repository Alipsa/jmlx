package se.alipsa.jmlx.models;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;

/** Expected Hugging Face tensor names for one decoder architecture. */
public record TensorPlan(
    Set<String> required, Set<String> optional, Set<String> forbidden, Set<Pattern> ignored) {
  /** Copies the expected tensor-name sets into immutable sets. */
  public TensorPlan {
    required = Set.copyOf(required);
    optional = Set.copyOf(optional);
    forbidden = Set.copyOf(forbidden);
    ignored = Set.copyOf(ignored);
  }

  /** Reports all missing, forbidden and unexpected tensor names together. */
  public void validate(Set<String> available) {
    List<String> problems = new ArrayList<>();
    required.stream()
        .filter(k -> !available.contains(k))
        .sorted()
        .forEach(k -> problems.add("missing tensor '" + k + "'"));
    available.stream()
        .filter(forbidden::contains)
        .sorted()
        .forEach(k -> problems.add("forbidden tensor '" + k + "' (capability: bias disabled)"));
    available.stream()
        .filter(k -> !required.contains(k) && !optional.contains(k) && !forbidden.contains(k))
        .filter(k -> ignored.stream().noneMatch(p -> p.matcher(k).matches()))
        .sorted()
        .forEach(k -> problems.add("unexpected tensor '" + k + "'"));
    if (!problems.isEmpty()) {
      throw new IllegalArgumentException("checkpoint tensor plan: " + String.join("; ", problems));
    }
  }
}
