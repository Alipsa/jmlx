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
  private final int maxKnownId;

  /**
   * Merges {@code modelVocab} with {@code addedTokens}: an added token always takes precedence, for
   * both its own token string and its own id. Any {@code modelVocab} entry sharing either one with
   * an added token is removed from the *other* map too, so {@code idOf} and {@code tokenOf} stay
   * exact mutual inverses even across a collision. {@code specialIds} tracks whichever token
   * currently occupies an id, not every added token that ever claimed it: an added token's own
   * {@code special} flag always wins for its id, including clearing a stale {@code true} left by an
   * earlier added token that collided on the same id.
   */
  public Vocabulary(Map<String, Integer> modelVocab, List<AddedToken> addedTokens) {
    Objects.requireNonNull(modelVocab, "Vocabulary: modelVocab must not be null");
    Objects.requireNonNull(addedTokens, "Vocabulary: addedTokens must not be null");
    this.tokenToId = new HashMap<>(modelVocab);
    this.idToToken = new HashMap<>();
    modelVocab.forEach((token, id) -> idToToken.put(id, token));
    this.specialIds = new HashSet<>();
    for (AddedToken t : addedTokens) {
      Integer previousIdForToken = tokenToId.get(t.content());
      if (previousIdForToken != null && !previousIdForToken.equals(t.id())) {
        idToToken.remove(previousIdForToken);
        // previousIdForToken no longer has any token mapped to it -- specialIds must be pruned
        // here too, not just on the id-collision path below, or isSpecial(previousIdForToken)
        // stays true even though hasId(previousIdForToken) is now false (PR #14 review round 3,
        // finding 5).
        specialIds.remove(previousIdForToken);
      }
      String previousTokenForId = idToToken.get(t.id());
      if (previousTokenForId != null && !previousTokenForId.equals(t.content())) {
        tokenToId.remove(previousTokenForId);
      }
      tokenToId.put(t.content(), t.id());
      idToToken.put(t.id(), t.content());
      if (t.special()) {
        specialIds.add(t.id());
      } else {
        specialIds.remove(t.id());
      }
    }
    // Computed from the final idToToken, not tracked incrementally across modelVocab/addedTokens
    // as ids are added: an id later vacated by a collision (above) must not still count towards
    // the max, or decode's skip-vs-throw split (see maxKnownId's own javadoc) treats a
    // deliberately-unmapped id as an in-range hole instead of the above-vocab case it actually is
    // (PR #14 review round 4, finding 6).
    int max = -1;
    for (int id : idToToken.keySet()) {
      max = Math.max(max, id);
    }
    this.maxKnownId = max;
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

  /** Whether {@code id} has a vocabulary entry (see {@link HfTokenizer#decode}). */
  public boolean hasId(int id) {
    return idToToken.containsKey(id);
  }

  /**
   * Whether {@code token} has a vocabulary entry (see {@link HfTokenizer}'s TemplateProcessing
   * conflict check, which needs the mirror of {@link #hasId}/{@link #tokenOf}: a template id can be
   * free while the token *string* it names is already claimed by a different id).
   */
  public boolean hasToken(String token) {
    return tokenToId.containsKey(token);
  }

  /**
   * The largest id assigned by either {@code modelVocab} or {@code addedTokens}. {@code
   * HfTokenizer#decode} uses this to tell a legitimate above-vocab id (e.g. a sampled logit outside
   * a checkpoint's trained vocab -- see {@link HfTokenizer#decode}) apart from an in-range hole,
   * which is always a bug and should not be silently skipped.
   */
  public int maxKnownId() {
    return maxKnownId;
  }
}
