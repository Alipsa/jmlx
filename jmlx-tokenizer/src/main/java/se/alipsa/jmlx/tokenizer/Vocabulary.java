package se.alipsa.jmlx.tokenizer;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Token string ↔ id lookups, merging a BPE model's own vocab with the file's {@code added_tokens}.
 */
public final class Vocabulary {

  private final Map<String, Integer> tokenToId;
  private final Map<Integer, String> idToToken;
  private final Set<Integer> specialIds;

  /**
   * Merges {@code modelVocab} with {@code addedTokens}, the latter taking precedence on id
   * collision.
   */
  public Vocabulary(Map<String, Integer> modelVocab, List<AddedToken> addedTokens) {
    Objects.requireNonNull(modelVocab, "Vocabulary: modelVocab must not be null");
    Objects.requireNonNull(addedTokens, "Vocabulary: addedTokens must not be null");
    this.tokenToId = new HashMap<>(modelVocab);
    this.idToToken = new HashMap<>();
    modelVocab.forEach((token, id) -> idToToken.put(id, token));
    this.specialIds = new HashSet<>();
    for (AddedToken t : addedTokens) {
      tokenToId.put(t.content(), t.id());
      idToToken.put(t.id(), t.content());
      if (t.special()) {
        specialIds.add(t.id());
      }
    }
  }

  /** Looks up a token string's id, throwing if it is not in the vocabulary. */
  public int idOf(String token) {
    Objects.requireNonNull(token, "Vocabulary.idOf: token must not be null");
    Integer id = tokenToId.get(token);
    if (id == null) {
      throw new TokenizerException(
          "Vocabulary.idOf: no vocabulary entry for token '" + token + "'");
    }
    return id;
  }

  /** Looks up an id's token string, throwing if it is not in the vocabulary. */
  public String tokenOf(int id) {
    String token = idToToken.get(id);
    if (token == null) {
      throw new TokenizerException("Vocabulary.tokenOf: no vocabulary entry for id " + id);
    }
    return token;
  }

  /** Whether {@code id} is one of the file's special (not just added) tokens. */
  public boolean isSpecial(int id) {
    return specialIds.contains(id);
  }
}
