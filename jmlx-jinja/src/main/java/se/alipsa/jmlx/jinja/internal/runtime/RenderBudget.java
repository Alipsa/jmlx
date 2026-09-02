package se.alipsa.jmlx.jinja.internal.runtime;

import se.alipsa.jmlx.jinja.ErrorCategory;
import se.alipsa.jmlx.jinja.RenderOptions;
import se.alipsa.jmlx.jinja.SourceLocation;
import se.alipsa.jmlx.jinja.TemplateRenderException;

final class RenderBudget {
  private final RenderOptions options;
  private long steps;
  private long iterations;
  private long output;
  private long rangeElements;
  private int macroDepth;

  RenderBudget(RenderOptions options) {
    this.options = options;
  }

  void chargeStep(SourceLocation location) {
    if (++steps > options.maxSteps()) {
      fail("Maximum render steps exceeded", location);
    }
  }

  void chargeLoopIteration(SourceLocation location) {
    if (++iterations > options.maxLoopIterations()) {
      fail("Maximum loop iterations exceeded", location);
    }
  }

  void chargeOutput(int length, SourceLocation location) {
    if ((output += length) > options.maxOutputLength()) {
      fail("Maximum output length exceeded", location);
    }
  }

  void chargeRangeElement(SourceLocation location) {
    if (++rangeElements > options.maxLoopIterations()) {
      fail("Maximum loop iterations exceeded", location);
    }
  }

  int remainingOutputLength() {
    return (int) Math.max(0, options.maxOutputLength() - output);
  }

  // Unlike the counters above (monotonic totals), macro depth must go up on entry and back down
  // on exit. Checking before incrementing (rather than incrementing then checking, as the other
  // charge*() methods do) means the throwing path never touches macroDepth at all, so a caller
  // that fails to pair this with exitMacro() in a finally cannot leak a level — the invariant is
  // structural, not caller-enforced.
  void enterMacro(SourceLocation location) {
    if (macroDepth >= options.maxMacroDepth()) {
      fail("Maximum macro call depth exceeded", location);
    }
    macroDepth++;
  }

  void exitMacro() {
    macroDepth--;
  }

  private static void fail(String message, SourceLocation location) {
    throw new TemplateRenderException(message, ErrorCategory.RESOURCE_LIMIT, location);
  }
}
