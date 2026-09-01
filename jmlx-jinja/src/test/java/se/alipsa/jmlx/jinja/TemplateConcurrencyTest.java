package se.alipsa.jmlx.jinja;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.jupiter.api.Timeout;
import se.alipsa.jmlx.jinja.internal.ast.Statement;

@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class TemplateConcurrencyTest {
  private static final int WORKERS = 16;
  private static final int ROUNDS = 8;
  // One isolated render currently charges 98-120 cumulative characters for the ids used here;
  // 256 permits two renders but exposes a shared budget across this suite's constrained calls.
  private static final int CONSTRAINED_MAX_OUTPUT_LENGTH = 256;
  private static final String SOURCE =
      "{% macro wrap() %}[{{ caller() }}]{% endmacro %}"
          + "{% set ns = namespace(seen=false) %}"
          + "{% set state.seen = id %}"
          + "{% set ns.seen = id %}"
          + "{% if use_host %}{{ format_tool(id) }}{% endif %}"
          + "{% filter upper %}{% call wrap() %}{{ id }}:"
          + "{% if fail %}{{ raise_exception(id) }}{% endif %}"
          + "{% for value in values %}{{ loop.index }}={{ value }};{% endfor %}"
          + ":{{ state.seen }}:{{ ns.seen }}{% endcall %}{% endfilter %}";

  @Test
  // This must run first when executing this class alone: its constrained priming render must be
  // the first budget constructed to make a hypothetical shared RenderBudget observable. In the
  // full test JVM earlier renders may prime such a defect with the default limit instead.
  @Order(1)
  @Timeout(value = 30, unit = TimeUnit.SECONDS, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
  void oneParsedTemplateRendersIndependentlyAcrossThreads() throws Exception {
    var template = Template.parse(SOURCE);
    var hostCalls = new AtomicInteger();
    HostFunction formatTool =
        arguments -> {
          hostCalls.incrementAndGet();
          return "host-" + arguments.getFirst();
        };
    var options = RenderOptions.builder().hostFunction("format_tool", formatTool).build();
    var constrainedOptions =
        RenderOptions.builder()
            .maxOutputLength(CONSTRAINED_MAX_OUTPUT_LENGTH)
            .hostFunction("format_tool", formatTool)
            .build();
    renderAndAssert(template, constrainedOptions, constrainedOptions, WORKERS + 1, 0, false);
    var ready = new CountDownLatch(WORKERS);
    var start = new CountDownLatch(1);
    var executor = Executors.newFixedThreadPool(WORKERS, daemonThreadFactory());
    var futures = new ArrayList<Future<?>>();
    try {
      for (var worker = 0; worker < WORKERS; worker++) {
        var workerId = worker;
        futures.add(
            executor.submit(
                () -> {
                  ready.countDown();
                  try {
                    if (!start.await(10, TimeUnit.SECONDS)) {
                      throw new AssertionError(
                          "worker " + workerId + " did not receive start signal");
                    }
                  } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                    throw new AssertionError(
                        "worker " + workerId + " was interrupted", interrupted);
                  }
                  for (var round = 0; round < ROUNDS; round++) {
                    try {
                      renderAndAssert(template, options, constrainedOptions, workerId, round, true);
                    } catch (AssertionError | RuntimeException failure) {
                      throw new AssertionError(
                          "worker " + workerId + ", round " + round + " failed", failure);
                    }
                  }
                }));
      }
      assertTrue(ready.await(10, TimeUnit.SECONDS), "not every worker reached the start barrier");
      start.countDown();
      for (var future : futures) {
        future.get(20, TimeUnit.SECONDS);
      }
      assertEquals(expectedHostCalls(), hostCalls.get());
    } finally {
      start.countDown();
      shutdown(executor);
    }
  }

  @Test
  @Order(2)
  void templateStoresOnlyItsFinalParsedProgram() throws Exception {
    final var template = Template.parse("{{ value }}");
    var fields =
        Arrays.stream(Template.class.getDeclaredFields())
            .filter(field -> !field.isSynthetic())
            .toList();
    assertEquals(1, fields.size());
    var program = fields.getFirst();
    assertEquals("program", program.getName());
    assertEquals(Statement.Program.class, program.getType());
    assertTrue(Modifier.isFinal(program.getModifiers()));
    program.setAccessible(true);
    var parsedProgram = program.get(template);
    for (var value = 0; value < 10; value++) {
      assertEquals(Integer.toString(value), template.render(Map.of("value", value)));
    }
    assertSame(parsedProgram, program.get(template));
    assertNotNull(parsedProgram);
  }

  @Test
  void oneParsedTemplateFormatsIndependentlyAcrossThreads() throws Exception {
    var template = Template.parse("{% if a %}{{ x }}{% endif %}");
    var expected = "{%- if a -%}\n  {{- x -}}\n{%- endif -%}";
    var executor = Executors.newFixedThreadPool(WORKERS, daemonThreadFactory());
    try {
      var futures = new ArrayList<Future<?>>();
      for (var worker = 0; worker < WORKERS; worker++) {
        futures.add(executor.submit(() -> assertEquals(expected, template.format(2))));
      }
      for (var future : futures) {
        future.get(10, TimeUnit.SECONDS);
      }
    } finally {
      shutdown(executor);
    }
  }

  private static void renderAndAssert(
      Template template,
      RenderOptions options,
      RenderOptions constrainedOptions,
      int worker,
      int round,
      boolean allowFailure) {
    var id = "W" + worker + "R" + round;
    var overload = (worker + round) % 4;
    final var explicitOptions = overload == 1 || overload == 3;
    final var fail = allowFailure && (worker + round) % 5 == 0;
    var state = new LinkedHashMap<String, Object>();
    state.put("seen", "caller-value");
    state.put("nested", new ArrayList<>(List.of("first", "second")));
    final var expectedState = copyState(state);
    var context = new LinkedHashMap<String, Object>();
    context.put("id", id);
    context.put("state", state);
    context.put("values", List.of("a", "b"));
    context.put("use_host", explicitOptions);
    context.put("fail", fail);

    if (fail) {
      var error =
          captureFailure(
              template, context, optionsFor(options, constrainedOptions, worker, round), overload);
      assertEquals(ErrorCategory.EXPLICIT_RAISE, error.category());
      assertTrue(error.getMessage().contains(id));
    } else {
      var expected =
          (explicitOptions ? "host-" + id : "") + "[" + id + ":1=A;2=B;:" + id + ":" + id + "]";
      String actual;
      switch (overload) {
        case 0 -> actual = template.render(context);
        case 1 ->
            actual =
                template.render(context, optionsFor(options, constrainedOptions, worker, round));
        case 2 -> {
          var output = new StringBuilder();
          template.render(context, output);
          actual = output.toString();
        }
        case 3 -> {
          var output = new StringBuilder();
          template.render(context, output, optionsFor(options, constrainedOptions, worker, round));
          actual = output.toString();
        }
        default -> throw new AssertionError("unreachable overload selection");
      }
      assertEquals(expected, actual);
    }
    assertEquals(expectedState, state);
  }

  private static RenderOptions optionsFor(
      RenderOptions options, RenderOptions constrainedOptions, int worker, int round) {
    return (worker + round) % 3 == 0 ? constrainedOptions : options;
  }

  private static int expectedHostCalls() {
    var calls =
        1; // The constrained explicit-options priming render above invokes format_tool once.
    for (var worker = 0; worker < WORKERS; worker++) {
      for (var round = 0; round < ROUNDS; round++) {
        var overload = (worker + round) % 4;
        if (overload == 1 || overload == 3) {
          calls++;
        }
      }
    }
    return calls;
  }

  private static TemplateRenderException captureFailure(
      Template template, Map<String, Object> context, RenderOptions options, int overload) {
    return switch (overload) {
      case 0 -> capture(() -> template.render(context));
      case 1 -> capture(() -> template.render(context, options));
      case 2 ->
          capture(
              () -> {
                var output = new StringBuilder();
                template.render(context, output);
              });
      case 3 ->
          capture(
              () -> {
                var output = new StringBuilder();
                template.render(context, output, options);
              });
      default -> throw new AssertionError("unreachable overload selection");
    };
  }

  private static TemplateRenderException capture(RenderCall call) {
    try {
      call.run();
    } catch (TemplateRenderException error) {
      return error;
    }
    throw new AssertionError("expected template render failure");
  }

  private static Map<String, Object> copyState(Map<String, Object> state) {
    var copy = new LinkedHashMap<String, Object>();
    copy.put("seen", state.get("seen"));
    copy.put("nested", new ArrayList<>((List<?>) state.get("nested")));
    return copy;
  }

  private static ThreadFactory daemonThreadFactory() {
    var sequence = new AtomicLong();
    return task -> {
      var thread = new Thread(task, "jmlx-jinja-concurrency-" + sequence.incrementAndGet());
      thread.setDaemon(true);
      return thread;
    };
  }

  private static void shutdown(ExecutorService executor) throws InterruptedException {
    executor.shutdown();
    if (!executor.awaitTermination(10, TimeUnit.SECONDS)) {
      executor.shutdownNow();
      assertTrue(executor.awaitTermination(10, TimeUnit.SECONDS), "worker executor did not stop");
    }
  }

  @FunctionalInterface
  private interface RenderCall {
    void run();
  }
}
