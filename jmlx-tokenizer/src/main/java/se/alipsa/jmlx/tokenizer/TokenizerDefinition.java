package se.alipsa.jmlx.tokenizer;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import tools.jackson.databind.JsonNode;

/** Immutable internal tokenizer configuration used by the component runtime. */
record TokenizerDefinition(
    JsonNode normalizer,
    JsonNode preTokenizer,
    List<PostProcessorStep> postProcessor,
    Model model,
    JsonNode decoder,
    List<AddedToken> addedTokens,
    EncodingOptions configuredDefaults) {

  TokenizerDefinition {
    normalizer = normalizer == null ? null : normalizer.deepCopy();
    preTokenizer = preTokenizer == null ? null : preTokenizer.deepCopy();
    postProcessor = List.copyOf(postProcessor);
    model = Objects.requireNonNull(model, "model");
    decoder = Objects.requireNonNull(decoder, "decoder").deepCopy();
    addedTokens = List.copyOf(addedTokens);
    configuredDefaults = Objects.requireNonNull(configuredDefaults, "configuredDefaults");
  }

  sealed interface Model permits Bpe, Unigram, WordPiece {
    Map<String, Integer> vocab();
  }

  record Bpe(
      Map<String, Integer> vocab,
      Map<String, Integer> mergeRanks,
      String unknownToken,
      boolean fuseUnknown,
      boolean byteFallback,
      String continuingSubwordPrefix,
      String endOfWordSuffix,
      boolean ignoreMerges)
      implements Model {
    Bpe {
      vocab = Map.copyOf(vocab);
      mergeRanks = Map.copyOf(mergeRanks);
      continuingSubwordPrefix = Objects.requireNonNull(continuingSubwordPrefix);
      endOfWordSuffix = Objects.requireNonNull(endOfWordSuffix);
    }
  }

  record Unigram(
      Map<String, Integer> vocab,
      List<String> tokens,
      List<Double> scores,
      int unknownId,
      boolean byteFallback)
      implements Model {
    Unigram {
      vocab = Map.copyOf(vocab);
      tokens = List.copyOf(tokens);
      scores = List.copyOf(scores);
      if (tokens.size() != scores.size()) {
        throw new IllegalArgumentException("Unigram tokens and scores must align");
      }
    }
  }

  record WordPiece(
      Map<String, Integer> vocab,
      String unknownToken,
      String continuingSubwordPrefix,
      int maxInputCharsPerWord)
      implements Model {
    WordPiece {
      vocab = Map.copyOf(vocab);
      unknownToken = Objects.requireNonNull(unknownToken);
      continuingSubwordPrefix = Objects.requireNonNull(continuingSubwordPrefix);
    }
  }
}
