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
   *
   * <p>{@code modelVocab} itself is required to have no two different token strings sharing one id
   * -- verified directly: without this check, {@code new Vocabulary(Map.of("a", 5, "b", 5),
   * List.of())} let {@code idOf("a")} and {@code idOf("b")} both resolve to {@code 5} while {@code
   * tokenOf(5)} returned only whichever token {@code modelVocab.forEach} happened to iterate last,
   * silently breaking the mutual-inverse guarantee this javadoc's first paragraph promises, with
   * the winner depending on {@code HashMap} iteration order. {@link TokenizerJsonLoader#load}'s own
   * loader-level check (Task 4's own amendment) already rejects this for a real {@code
   * tokenizer.json} file with a better, file-specific message; this constructor's own check is a
   * second, defense-in-depth line for {@code Vocabulary}'s broader public API -- callers exist,
   * e.g. every test in {@code VocabularyTest}, that construct it directly from an arbitrary map,
   * not through the loader (PR #14 review round 8, finding 5, following {@code SpecialTokenInfo}'s
   * round-3 precedent of validating in the type itself, not just at one call site).
   */
  public Vocabulary(Map<String, Integer> modelVocab, List<AddedToken> addedTokens) {
    Objects.requireNonNull(modelVocab, "Vocabulary: modelVocab must not be null");
    Objects.requireNonNull(addedTokens, "Vocabulary: addedTokens must not be null");
    this.tokenToId = new HashMap<>(modelVocab);
    this.idToToken = new HashMap<>();
    modelVocab.forEach(
        (token, id) -> {
          // No != token.equals(...) guard needed: modelVocab's keys are unique (it's a Map), so
          // the only way idToToken.put(id, token) can return a non-null existingToken is when a
          // DIFFERENT token key already claimed this id -- existingToken == token is structurally
          // impossible here, unlike the analogous list-based checks elsewhere in this codebase
          // (e.g. HfTokenizer#requireInternallyConsistentTemplateTokens), where the same (id,
          // content) pair really can appear twice and must not be treated as a contradiction (PR
          // #14 review round 9, finding 6, removing the dead clause this check was given when it
          // was first written).
          String existingToken = idToToken.put(id, token);
          if (existingToken != null) {
            throw new TokenizerException(
                "Vocabulary: modelVocab has id "
                    + id
                    + " for both '"
                    + existingToken
                    + "' and '"
                    + token
                    + "'");
          }
        });
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
   * The largest id with a vocabulary entry *after* collision cleanup -- not simply the largest id
   * ever assigned while constructing this {@link Vocabulary}: an id vacated by a later {@code
   * addedTokens} collision (see the constructor's javadoc) no longer counts even though it was
   * briefly assigned during construction, since it is no longer a real vocabulary entry (PR #14
   * review round 5, finding 7, correcting this javadoc after round 4, finding 6, changed the
   * computation itself but left this description of it stale). {@code HfTokenizer#decode} uses this
   * to tell a legitimate above-vocab id (e.g. a sampled logit outside a checkpoint's trained vocab
   * -- see {@link HfTokenizer#decode}) apart from an in-range hole, which is always a bug and
   * should not be silently skipped.
   */
  public int maxKnownId() {
    return maxKnownId;
  }

  /** Number of ids in the contiguous vocabulary range needed for a model output head. */
  public int vocabSize() {
    return maxKnownId + 1;
  }
}
