package se.alipsa.jmlx.models;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.function.Consumer;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.tokenizer.HfTokenizer;
import se.alipsa.jmlx.tokenizer.IncrementalTokenDecoder;
import se.alipsa.jmlx.tokenizer.TokenizerException;

/** Shared synchronous sampling/event loop; architecture state and execution stay with adapters. */
final class GenerationLoop {
  /** Internal cooperative cancellation during preparation; no partial step may be committed. */
  static final class Cancelled extends RuntimeException {}

  // Preserve the existing terminal-listener logging category across extraction.
  private static final System.Logger LOGGER = System.getLogger(DecoderModel.class.getName());

  interface State extends AutoCloseable {
    MLXArray firstStep(MLXScope activation);

    MLXArray decodeStep(MLXScope activation, int token);

    MLXArray[] retainedArrays();

    void onStepFailure();

    default void commitStep() {}

    @Override
    default void close() {}
  }

  @FunctionalInterface
  interface Factory {
    State create(MLXScope generation);
  }

  private GenerationLoop() {}

  static GenerationResult run(
      MLXScope owner,
      int vocabulary,
      StepBoundaryEvaluator evaluator,
      GenerationRequest request,
      GenerationConfig policy,
      int[] prompt,
      int[] frequencySeed,
      int promptPositions,
      Consumer<GenerationEvent> listener,
      Factory factory) {
    HfTokenizer tokenizer = request.tokenizer();
    List<Integer> generated = new ArrayList<>();
    List<Double> logProbabilities = new ArrayList<>();
    StringBuilder generatedText = tokenizer == null ? null : new StringBuilder();
    IncrementalTokenDecoder decoder = null;
    LinkedHashMap<Integer, Integer> frequencies = PenaltyInputs.frequencies(frequencySeed);
    FinishReason reason = FinishReason.MAX_TOKENS;
    if (!request.cancellationToken().isCancelled() && policy.maxNewTokens() > 0) {
      decoder = tokenizer == null ? null : tokenizer.newIncrementalDecoder(true);
      try (MLXScope generation = owner.newChild();
          SamplingPipeline sampler =
              new SamplingPipeline(generation, policy, vocabulary, evaluator);
          State state = factory.create(generation)) {
        int[] input = prompt;
        for (int step = 0; step < policy.maxNewTokens(); step++) {
          if (request.cancellationToken().isCancelled()) {
            reason = FinishReason.CANCELLED;
            break;
          }
          try (MLXScope activation = generation.newChild()) {
            SamplingPipeline.Selection selection;
            try {
              MLXArray lastLogits =
                  step == 0 ? state.firstStep(activation) : state.decodeStep(activation, input[0]);
              selection =
                  sampler.select(
                      lastLogits,
                      PenaltyInputs.from(frequencies, vocabulary),
                      step,
                      state.retainedArrays());
              state.commitStep();
            } catch (Cancelled cancelled) {
              state.onStepFailure();
              reason = FinishReason.CANCELLED;
              break;
            } catch (RuntimeException | Error failure) {
              state.onStepFailure();
              throw failure;
            }
            int next = selection.tokenId();
            boolean eos = policy.eosTokenIds().contains(next);
            if (!eos && policy.stopTokenIds().contains(next)) {
              reason = FinishReason.STOP_TOKEN;
              break;
            }
            generated.add(next);
            frequencies.merge(next, 1, Integer::sum);
            if (selection.logProbability() != null) {
              logProbabilities.add(selection.logProbability());
            }
            String textDelta = null;
            if (decoder != null) {
              try {
                textDelta = decoder.append(next);
                generatedText.append(textDelta);
              } catch (TokenizerException e) {
                throw new GenerationAbortedException(
                    toList(prompt), generated, "output decoder", next, e);
              }
            }
            try {
              listener.accept(tokenEvent(next, textDelta, selection.logProbability()));
            } catch (RuntimeException e) {
              throw new GenerationAbortedException(toList(prompt), generated, e);
            }
            if (eos) {
              reason = FinishReason.EOS;
              break;
            }
            input = new int[] {next};
          }
        }
      }
    } else if (request.cancellationToken().isCancelled()) {
      reason = FinishReason.CANCELLED;
    }
    String terminalDelta = null;
    if (tokenizer != null) {
      terminalDelta = "";
      if (decoder != null) {
        try {
          terminalDelta = decoder.finish();
        } catch (TokenizerException e) {
          throw new GenerationAbortedException(
              toList(prompt), generated, "output decoder finish", null, e);
        }
      }
      generatedText.append(terminalDelta);
    }
    GenerationResult result =
        new GenerationResult(
            toList(prompt),
            generated,
            reason,
            logProbabilities,
            generatedText == null ? null : generatedText.toString(),
            promptPositions);
    try {
      listener.accept(
          terminalDelta == null
              ? GenerationEvent.finished(reason)
              : GenerationEvent.finished(reason, terminalDelta));
    } catch (RuntimeException e) {
      LOGGER.log(System.Logger.Level.WARNING, "generation terminal listener failed", e);
    }
    return result;
  }

  static GenerationEvent tokenEvent(int tokenId, String textDelta, Double logProbability) {
    if (textDelta == null) {
      return logProbability == null
          ? GenerationEvent.token(tokenId)
          : GenerationEvent.token(tokenId, logProbability);
    }
    return logProbability == null
        ? GenerationEvent.token(tokenId, textDelta)
        : GenerationEvent.token(tokenId, textDelta, logProbability);
  }

  private static List<Integer> toList(int[] ids) {
    return Arrays.stream(ids).boxed().toList();
  }
}
