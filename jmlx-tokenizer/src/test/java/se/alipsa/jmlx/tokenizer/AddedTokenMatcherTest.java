package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import org.junit.jupiter.api.Test;

/** {@link AddedTokenMatcher} single-word rejection and longest-match selection among inputs. */
class AddedTokenMatcherTest {

  @Test
  void singleWordRejectionDoesNotFallBackToShorterAddedToken() {
    List<AddedToken> tokens =
        List.of(
            new AddedToken(10, "ab", true, false, false, false, false),
            new AddedToken(11, "a", false, false, false, false, false));
    AddedTokenMatcher matcher = new AddedTokenMatcher(tokens, false, null);
    List<AddedTokenMatcher.Segment> segments = matcher.split(AlignedText.original("xabc"));
    assertEquals(1, segments.size());
    assertEquals("xabc", segments.get(0).text().text());
    assertEquals(null, segments.get(0).token());
  }

  @Test
  void singleWordRejectionSkipsTheWholeRejectedSpan() {
    List<AddedToken> tokens =
        List.of(
            new AddedToken(10, "ab", true, false, false, false, false),
            new AddedToken(11, "b", false, false, false, false, false));
    AddedTokenMatcher matcher = new AddedTokenMatcher(tokens, false, null);
    List<AddedTokenMatcher.Segment> segments = matcher.split(AlignedText.original("xabc"));
    assertEquals(1, segments.size());
    assertEquals(null, segments.get(0).token());
  }

  @Test
  void addedTokenMatcherPrefersLongestAmongSharedFirstCodePoint() {
    List<AddedToken> tokens =
        List.of(new AddedToken(1, "<a>", true), new AddedToken(2, "<a><b>", true));
    AddedTokenMatcher matcher = new AddedTokenMatcher(tokens, false, null);
    List<AddedTokenMatcher.Segment> segments = matcher.split(AlignedText.original("x<a><b><a>"));
    assertEquals(
        List.of("x", "<a><b>", "<a>"), segments.stream().map(s -> s.text().text()).toList());
    assertEquals(2, segments.get(1).token().id());
  }
}
