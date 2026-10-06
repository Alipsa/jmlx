package se.alipsa.jmlx.tokenizer;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** SentencePiece's serialized Darts trie and UTF-8 replacement table. */
final class PrecompiledNormalizer {
  private static final Pattern GRAPHEME = Pattern.compile("\\X");
  private final int[] trie;
  private final byte[] replacements;

  PrecompiledNormalizer(String encoded) {
    try {
      byte[] bytes = Base64.getDecoder().decode(encoded);
      if (bytes.length < 8) {
        throw malformed();
      }
      ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
      long size = Integer.toUnsignedLong(buffer.getInt());
      if (size == 0 || size % 4 != 0 || size > bytes.length - 5) {
        throw malformed();
      }
      trie = new int[(int) size / 4];
      for (int i = 0; i < trie.length; i++) {
        trie[i] = buffer.getInt();
      }
      replacements = new byte[buffer.remaining()];
      buffer.get(replacements);
      utf8(replacements, 0, replacements.length);
      if (replacements[replacements.length - 1] != 0 || offset(trie[0]) >= trie.length) {
        throw malformed();
      }
    } catch (IllegalArgumentException | CharacterCodingException e) {
      throw new TokenizerException("Precompiled: malformed precompiled_charsmap", e);
    }
  }

  AlignedText apply(AlignedText input) {
    List<Change> output = new ArrayList<>();
    Matcher graphemes = GRAPHEME.matcher(input.text());
    int index = 0;
    while (graphemes.find()) {
      String grapheme = graphemes.group();
      int count = grapheme.codePointCount(0, grapheme.length());
      List<AlignedText.Unit> units = input.units().subList(index, index + count);
      String replacement =
          grapheme.getBytes(StandardCharsets.UTF_8).length < 6 ? transform(grapheme) : null;
      if (replacement != null) {
        replace(output, units, replacement);
      } else {
        for (AlignedText.Unit unit : units) {
          replacement = transform(unit.value());
          if (replacement == null) {
            output.add(new Change(unit.value(), 0));
          } else {
            replace(output, List.of(unit), replacement);
          }
        }
      }
      index += count;
    }
    List<AlignedText.Unit> aligned = new ArrayList<>();
    int cursor = 0;
    for (Change change : output) {
      int source = change.delta() > 0 ? cursor - 1 : cursor;
      if (source < 0 || source >= input.units().size()) {
        throw new TokenizerException("Precompiled: invalid normalization alignment");
      }
      AlignedText.Unit unit = input.units().get(source);
      aligned.add(new AlignedText.Unit(change.value(), unit.startByte(), unit.endByte()));
      if (change.delta() <= 0) {
        cursor += 1 - change.delta();
      }
    }
    return new AlignedText(aligned);
  }

  private record Change(String value, int delta) {}

  private static void replace(
      List<Change> output, List<AlignedText.Unit> input, String replacement) {
    int[] scalars = replacement.codePoints().toArray();
    int difference = scalars.length - input.size();
    for (int i = 0; i < scalars.length; i++) {
      output.add(
          new Change(
              new String(Character.toChars(scalars[i])),
              difference > 0 && i >= scalars.length - difference ? 1 : 0));
    }
    if (difference < 0 && !output.isEmpty()) {
      Change last = output.removeLast();
      output.add(new Change(last.value(), last.delta() + difference));
    }
  }

  private String transform(String input) {
    long position = offset(trie[0]);
    for (byte value : input.getBytes(StandardCharsets.UTF_8)) {
      int label = Byte.toUnsignedInt(value);
      if (label == 0) {
        break;
      }
      position ^= label;
      int unit = unit(position);
      if ((unit & 0x800000ffL) != label) {
        return null;
      }
      position ^= offset(unit);
      if ((unit & 0x100) != 0) {
        int start = unit(position) & 0x7fffffff;
        if (start >= replacements.length) {
          throw malformed();
        }
        int end = start;
        while (end < replacements.length && replacements[end] != 0) {
          end++;
        }
        if (end == replacements.length) {
          throw malformed();
        }
        try {
          return utf8(replacements, start, end - start);
        } catch (CharacterCodingException e) {
          throw new TokenizerException("Precompiled: invalid replacement UTF-8", e);
        }
      }
    }
    return null;
  }

  private int unit(long position) {
    if (position < 0 || position >= trie.length) {
      throw malformed();
    }
    return trie[(int) position];
  }

  private static long offset(int unit) {
    return (Integer.toUnsignedLong(unit) >>> 10) << ((unit & 0x200) >>> 6);
  }

  private static String utf8(byte[] bytes, int offset, int length) throws CharacterCodingException {
    return StandardCharsets.UTF_8
        .newDecoder()
        .onMalformedInput(CodingErrorAction.REPORT)
        .onUnmappableCharacter(CodingErrorAction.REPORT)
        .decode(ByteBuffer.wrap(bytes, offset, length))
        .toString();
  }

  private static TokenizerException malformed() {
    return new TokenizerException("Precompiled: malformed charsmap trie or replacement table");
  }
}
