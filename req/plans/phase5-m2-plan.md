# jmlx Phase 5 — M2 implementation plan (`jmlx-tokenizer`, byte-level BPE + chat templates)

Concrete task-by-task plan for `req/phase5-plan.md`'s M2. The architecture question that document's
own D3 left open is now resolved by explicit decision, not desk research: **pure-Java port**, as a new
`jmlx-tokenizer` Gradle module (sibling to `jmlx-ffi`/`jmlx-core`/`jmlx-examples`, not a package inside
`jmlx-core`) — chosen over a standalone repo because it has no external reuse target yet and over the
FFM/`tokenizers-cpp` path because it avoids the Rust toolchain this machine doesn't have. Chat-template
rendering uses `se.alipsa:hfjinja:0.5.0` (Maven Central), the user's own Java port of
`@huggingface/jinja` — not the hand-format-per-model or GraalJS options D3's amendment also
considered.

**Scope is deliberately narrower than "port swift-transformers."** `req/phase5-plan.md`'s own
"ship exactly what's needed" precedent applies: M3's actual named targets are Llama-3-Instruct and
Qwen2.5-Instruct, and both declare `"tokenizer_class": "PreTrainedTokenizerFast"` backed by a
byte-level BPE `model.type: "BPE"` in their `tokenizer.json` — not `LlamaTokenizer`'s legacy
SentencePiece/Metaspace path. This plan ports only what byte-level BPE needs: no
WordPiece/Unigram/SentencePiece, no `BertTokenizer`, no general normalizer plugin chain (just
`None`/`NFC`, the only two either target model declares), no `Trie`-based added-token matching
(confirmed unused by this path — a single combined regex handles it), and no sentence-pair
(`pair`-template) post-processing (chat/completion use is single-sequence only).

## Findings from this plan's pre-work

**Both target models' real `tokenizer.json`/`tokenizer_config.json` were fetched directly and
inspected** (`Qwen/Qwen2.5-0.5B-Instruct`, and `NousResearch/Meta-Llama-3-8B-Instruct` — an ungated
re-upload of the same gated `meta-llama/Meta-Llama-3-8B-Instruct` weights/tokenizer, used here only
because `meta-llama`'s own repo requires accepting a license gate the fetching environment can't do
non-interactively; the tokenizer files themselves are redistributed unchanged). Findings below are
against these real files, not reconstructions:

- **Pretokenizer regex, byte-exact from the JSON itself** (not `llama.cpp`'s rewritten-for-a-
  lookahead-free-engine form, which is a red herring for a Java port — `java.util.regex.Pattern`
  supports `(?i:...)` and lookahead natively):
  - QWEN2.5: `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
  - LLAMA3: same, with `\p{N}` replaced by `\p{N}{1,3}`.
  - Both wrapped in a `pre_tokenizer.Sequence` as `[{type: "Split", pattern: {Regex: "..."},
    behavior: "Isolated", invert: false}, {type: "ByteLevel", add_prefix_space: false, use_regex:
    false, trim_offsets: <false for Qwen, true for Llama-3, irrelevant here — offsets aren't tracked>}]`.
    `behavior: "Isolated"` + `invert: false` on a pattern that's exhaustive over all Unicode text means
    this reduces to a plain `Matcher.find()` loop collecting every match in order — no need to port
    `tokenizers`' general `Split`/`Invert`/`Removed`/`MergedWithPrevious` behavior enum, only the one
    case these two files actually use. The following `ByteLevel` stage has `use_regex: false`
    (splitting already happened), so it does nothing but byte-encode each already-split chunk.
- **Normalizer:** Qwen2.5 declares `{"type": "NFC"}`; Llama-3 declares `null` (none). Both real,
  no other normalizer type appears in either file — confirms the `None`/`NFC`-only scope above.
- **Post-processor — the two real shapes to support, nothing else:**
  - Qwen2.5: `{"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex":
    false}` — a bare `ByteLevel` post-processor. In the real `tokenizers` Rust implementation this step
    exists purely to fix up byte-offset↔character-offset mappings; since this port does not track
    offsets (an explicit, named scope boundary — jmlx's use case is encode/decode, not
    offset-aware tooling), a bare `ByteLevel` post-processor step is a no-op on the token list itself.
    This matches Qwen's own `tokenizer_config.json` (`bos_token: null`, `add_bos_token: false`) — Qwen
    prepends nothing; `<|im_start|>`/`<|im_end|>` are literal added-token text the chat template emits.
  - Llama-3: `{"type": "Sequence", "processors": [{"type": "ByteLevel", ...}, {"type":
    "TemplateProcessing", "single": [{"SpecialToken": {"id": "<|begin_of_text|>", "type_id": 0}},
    {"Sequence": {"id": "A", "type_id": 0}}], "special_tokens": {"<|begin_of_text|>": {"id":
    "<|begin_of_text|>", "ids": [128000], "tokens": ["<|begin_of_text|>"]}}}]}` — the `ByteLevel` step
    is again a no-op per above; `TemplateProcessing`'s `single` template is what prepends
    `<|begin_of_text|>` as BOS when `addSpecialTokens` is true. Only the `single` template is needed
    (no `pair`).
- **`model.ignore_merges: true` on Llama-3 (absent/false on Qwen2.5) is a real, load-bearing BPE
  algorithm variant, confirmed against `tokenizers/src/models/bpe/model.rs` (`v0.21.2`) directly, not
  swift-transformers (which doesn't implement it — a real gap in the porting source, not a Java-port
  simplification):** when true, `tokenize_with_cache` checks whether the *entire* pre-token string is
  already a single vocab entry, and if so returns it as one token, skipping the merge loop entirely —
  before the merge cache is even consulted. When false (Qwen2.5's case), this shortcut is skipped and
  the normal rank-ordered merge loop always runs. Both models have `dropout: null`, `fuse_unk: false`,
  `byte_fallback: false`, `continuing_subword_prefix`/`end_of_word_suffix` empty/absent — the
  hexadecimal-byte-fallback path (`BPETokenizer.swift`'s `hexaEncode`) is therefore dead code for both
  real targets and is out of scope (an out-of-vocab merge result should never occur given
  `byte_fallback: false` and that every raw byte already has its own vocab entry by construction of
  byte-level BPE training; Task 6 below fails loudly with `TokenizerException` if it ever does, rather
  than silently guessing).
- **`merges` is a JSON array of single space-separated strings** (e.g. `"Ġ Ġ"`), not the older
  two-element-array-per-merge format — confirmed by direct inspection of both files. Array index
  order is priority rank (lower index merges first).

  **Amendment (PR #14 review, round 2): this bullet has the history backwards.** The
  two-element-array form (`["Ġ", "Ġ"]`) is not older — it is the *newer* serialization, introduced
  in HF `tokenizers` PR #909 (released in `tokenizers` v0.20.0) specifically to make each merge's
  two components unambiguous without a separator convention. The single space-separated string is
  the older format this plan's two fixture files happened to use, and remains valid: `tokenizers`
  still round-trips both. `TokenizerJsonLoader.parseModel` (Task 4, below) accepts either shape.
- **`added_tokens`: zero entries on either model set `lstrip`/`rstrip`/`single_word`/`normalized`**
  (checked all 22 Qwen2.5 entries and all 256 Llama-3 entries programmatically, not sampled) — the
  whitespace-eating variant of added-token matching `Tokenizer.swift` supports is dead code for both
  real targets and is out of scope; added-token matching is exact-literal-string matching only.
- **`Trie.swift` is unused by this path.** The actual `PreTrainedTokenizer.tokenize()` path in
  `swift-transformers` builds one combined `NSRegularExpression` from all `added_tokens[].content`
  (escaped, longest-first) and splits on that directly — not trie search. This plan does the Java
  equivalent (`Pattern.quote` per token, `|`-joined, longest-first) rather than porting `Trie.swift`.
- **Real chat templates, fetched verbatim from each model's own `tokenizer_config.json`** (not a
  community-reformatted mirror): both captured below in Task 14's fixtures. Qwen2.5's chat_template
  is ChatML-style with tool-calling support (uses Jinja's `tojson` filter); Llama-3's is a much
  shorter single-loop template with no tool-calling. `bos_token`/`eos_token`: Qwen2.5 has
  `bos_token: null`, `eos_token: "<|im_end|>"`; Llama-3 has `bos_token: "<|begin_of_text|>"`,
  `eos_token: "<|eot_id|>"`.
- **`hfjinja`'s real public API** (`se.alipsa:hfjinja:0.5.0`, JPMS module `se.alipsa.hfjinja`, zero
  runtime dependencies, Java 21+): a single class `se.alipsa.hfjinja.Template` —
  `Template.parse(String source)` returns a `Template`; `.render(Map<String,?> context)` returns the
  rendered `String`. Exceptions `TemplateSyntaxException` (parse) / `TemplateRenderException` (render)
  both extend `HfJinjaException`, both unchecked. Built-in globals already include `raise_exception`,
  `range`, `namespace`, `tojson`-style JSON serialization support for chat-template use — no host
  functions need registering for either target model's template.
- **Snapshot builds may resolve from `mavenLocal()` before `mavenCentral()` to support locally
  published snapshot artifacts during development; release builds resolve only from Maven Central.**
  This environment cannot reach Maven Central, so its snapshot development build uses locally
  available dependencies. [Amended (PR #14 review, rounds 1 and 3): the "cannot reach Maven
  Central" premise in the previous sentence is mistaken -- see Task 1's Step 1c amendment below for
  the corrected diagnosis and the `build.gradle` shape actually shipped.] `se.alipsa:hfjinja:0.5.0`
  is present locally (the user built and installed it there directly).
  `tools.jackson.core:jackson-databind` is present only up to **`3.1.2`** —
  `3.2.2` (Maven Central's actual latest at spike time) is *not* present locally and would fail to
  resolve in this environment, so this plan pins Jackson to **`3.1.2`**, confirmed as a complete
  local artifact (`.jar`/`.pom`/sources, no partial/failed-download markers). A machine with real
  Maven Central access could use a newer Jackson 3.x release instead; this plan pins to what this
  build can actually resolve today.
- **Jackson 3.1.2** (`tools.jackson.core:jackson-databind:3.1.2`) is the JSON parser: no JSON library
  exists anywhere in this repo today.
  Jackson 3 renamed several `JsonNode` accessors from its 2.x line (`asText()` → `asString()`,
  `textValue()` → `stringValue()`, `isTextual()` → `isString()`) and made its base exception
  (`tools.jackson.core.JacksonException`, replacing `JsonProcessingException`) unchecked
  (`extends RuntimeException`, not `IOException`) — code below uses the 3.x names; double-check
  against the installed jar's javadoc during implementation in case a minor accessor name differs.
  A full `ObjectMapper.readTree(...)` parse (not a manual streaming walk) is used even though
  `tokenizer.json` is large (Qwen2.5: 7.0 MB / 151,643 vocab entries; Llama-3: 9.1 MB / 128,000 vocab
  entries, both measured directly) — a sub-second, one-time parse cost that's negligible next to this
  framework's own multi-GB safetensors/GGUF loading, so a manual streaming parser would be premature
  optimization for no measured benefit.
- **jmlx conventions this plan follows:** `TokenizerException extends RuntimeException` with the
  same two-constructor shape as `se.alipsa.jmlx.core.MLXException` (message-only, message+cause) —
  a new type, not a reuse of `MLXException`, since that type's own Javadoc scopes it specifically to
  mlx-c native failures. `Objects.requireNonNull(x, "method: x must not be null")` on every public
  parameter. JUnit Jupiter 6.0.1, plain `org.junit.jupiter.api.Assertions.*` (no AssertJ/Hamcrest),
  matching `MLXIOTest`. **Unlike every existing `jmlx-core` test class, none of this module's tests
  need `@EnabledIfNativeAvailable`** — a pure-Java tokenizer has no native dependency at all, so its
  suite runs (and gates CI) even on a machine with no `mlx.metallib` staged, which existing native
  tests cannot do. This is a real, worth-noting usability improvement, not an oversight.

## Global Constraints

1. **No SentencePiece, WordPiece, Unigram, `BertTokenizer`, general normalizer chain, `Trie`-based
   added-token matching, offset tracking, or sentence-pair post-processing.** Every one of these is
   confirmed dead code for Llama-3-Instruct and Qwen2.5-Instruct specifically (Findings above) and is
   out of scope for this plan, not merely deferred silently.
2. **Every public method parameter is null-checked via `Objects.requireNonNull` with a
   `"ClassName.methodName: paramName must not be null"`-style message**, matching existing
   `jmlx-core` convention.
3. **All native-format field names (`ignore_merges`, `add_prefix_space`, `byte_fallback`, etc.) are
   preserved verbatim as JSON keys mapped by the Jackson loader (Task 4), even though the Java fields
   they populate use `lowerCamelCase`** — matching how `tokenizer.json`/`tokenizer_config.json` are
   actually shaped, so a future engineer comparing this code against a real file isn't translating
   names in their head.
4. **`BpeMerger`'s merge loop is the plain "repeatedly find the lowest-rank adjacent pair, merge,
   repeat" reference algorithm** (the same algorithm HF's own reference Python BPE implementations
   use), not `BPETokenizer.swift`'s doubly-linked-list-plus-min-heap optimization. Pre-token chunks
   are short (a handful of characters; the pretokenizer regex guarantees this), so the reference
   algorithm's worse asymptotic complexity is not a real performance concern here — the swap
   trades a real implementation-complexity cost for a performance win this workload doesn't need,
   which is a case the codebase's own YAGNI convention already covers elsewhere in this doc.
5. **Every JSON-shape assumption below (regex strings, post-processor shapes, `ignore_merges`
   semantics, absence of `lstrip`/`rstrip` on added tokens) is stated as a target-model-specific fact,
   not a general `tokenizers`-library guarantee** — a future third model with a different
   `tokenizer.json` shape may need this plan revisited, not silently misinterpreted by code that
   assumes today's two shapes are the only ones that exist.
6. **`ChatTemplateRenderer`'s context map takes an open `Map<String, Object> extraContext` merged in
   underneath the fixed `messages`/`add_generation_prompt`/`bos_token`/`eos_token` keys**, so
   tool-calling variables (`tools`, etc.) Qwen2.5's own chat_template already references can be added
   by a caller later (M3's own job) without this class changing shape.

## Third-party dependencies this plan adds

| Dependency | Coordinates | Used by |
|---|---|---|
| Jackson databind | `tools.jackson.core:jackson-databind:3.1.2` (pulls `jackson-core` and `com.fasterxml.jackson.core:jackson-annotations` transitively — the annotations artifact keeps its 2.x coordinates even under Jackson 3; pinned to `3.1.2`, not Central's newer `3.2.2`, because only `3.1.2` is present in this environment's `~/.m2/repository` — see Findings) | Task 4 (`TokenizerJson`) |
| hfjinja | `se.alipsa:hfjinja:0.5.0` (present in `~/.m2/repository`; resolved via `mavenLocal()`, added to the root `build.gradle` by Task 1) | Task 13 (`ChatTemplateRenderer`) |

## File Structure

New Gradle module `jmlx-tokenizer`, package `se.alipsa.jmlx.tokenizer`:

```
jmlx-tokenizer/
  build.gradle
  src/main/java/se/alipsa/jmlx/tokenizer/
    TokenizerException.java
    ByteLevelCoding.java
    AddedToken.java                  (record)
    BpeModelConfig.java              (record)
    PreTokenizerConfig.java          (record)
    NormalizerKind.java              (enum)
    PostProcessorStep.java           (sealed interface)
    ByteLevelStep.java               (record, implements PostProcessorStep)
    TemplateProcessingStep.java      (record, implements PostProcessorStep)
    TemplateItem.java                (sealed interface)
    SpecialTokenItem.java            (record, implements TemplateItem)
    SequenceItem.java                (record, implements TemplateItem)
    SpecialTokenInfo.java            (record)
    TokenizerJson.java               (record + Jackson loader)
    Vocabulary.java
    BpeMerger.java
    ByteLevelPreTokenizer.java
    TextNormalizer.java
    AddedTokenSplitter.java
    PostProcessorApplier.java
    ByteLevelDecoder.java
    HfTokenizer.java
    ChatTemplateRenderer.java
  src/test/java/se/alipsa/jmlx/tokenizer/
    ByteLevelCodingTest.java
    TokenizerJsonTest.java
    BpeMergerTest.java
    ByteLevelPreTokenizerTest.java
    AddedTokenSplitterTest.java
    PostProcessorApplierTest.java
    ByteLevelDecoderTest.java
    HfTokenizerTest.java
    ChatTemplateRendererTest.java
  src/test/resources/se/alipsa/jmlx/tokenizer/
    qwen-style.tokenizer.json
    llama3-style.tokenizer.json
    qwen2.5-instruct-chat-template.jinja
    llama3-instruct-chat-template.jinja
```

## Task 1: Module scaffolding

**Step 1a — `settings.gradle`:** add `jmlx-tokenizer` to the `include(...)` call:

```groovy
include('jmlx-ffi', 'jmlx-core', 'jmlx-tokenizer', 'jmlx-examples')
```

**Step 1b — `gradle/libs.versions.toml`:** add version and library entries alongside the existing
`junit-jupiter` ones:

```toml
[versions]
junit-jupiter = "6.0.1"
jackson = "3.1.2"
hfjinja = "0.5.0"

[libraries]
junit-jupiter-api = { module = "org.junit.jupiter:junit-jupiter-api", version.ref = "junit-jupiter" }
jackson-databind = { module = "tools.jackson.core:jackson-databind", version.ref = "jackson" }
hfjinja = { module = "se.alipsa:hfjinja", version.ref = "hfjinja" }
```

**Step 1c — root `build.gradle`: add `mavenLocal()`.** This build environment cannot reach
`mavenCentral()`; both new dependencies must resolve from `~/.m2/repository` instead (Findings
above). Add `mavenLocal()` *before* `mavenCentral()` in the existing `allprojects` block so it's
checked first:

```groovy
allprojects {
    repositories {
        mavenLocal()
        mavenCentral()
    }
}
```

`mavenCentral()` stays in the list (not removed) so a build environment that *does* have real
network access still works unchanged — this only adds a local-first lookup, it doesn't take
anything away.

**Amendment (PR #14 review, rounds 1 and 3): the "this build environment cannot reach
`mavenCentral()`" premise above was mistaken, and the shipped `build.gradle` does not match the
snippet above.** Live dependency resolution against a stock `mavenCentral()`-only repository list
(`./gradlew :jmlx-tokenizer:dependencies --configuration compileClasspath --refresh-dependencies`,
and a direct `curl` against `repo1.maven.org`) confirms both `se.alipsa:hfjinja:0.5.0` and
`tools.jackson.core:jackson-databind:3.1.2` resolve from Maven Central alone -- this environment was
never actually unable to reach it. The root `build.gradle` instead gates `mavenLocal()` on `jmlx`'s
own root project version (`group = 'se.alipsa'`, `version = '0.5.0-SNAPSHOT'`, both set in an
`allprojects {}` block) being a `-SNAPSHOT` build, an ordinary and unrelated local-development
convenience (per this repo's own `~/.claude/CLAUDE.md` policy: "If a project is a SNAPSHOT version,
`mavenLocal()` should be enabled — always"), not a network workaround:

```groovy
allprojects {
    repositories {
        mavenCentral()
        if (rootProject.version.toString().endsWith('-SNAPSHOT')) {
            mavenLocal()
        }
    }
}
```

`mavenCentral()` is checked first here (not `mavenLocal()`), so a release build -- or any build
whose `jmlx` version isn't a `-SNAPSHOT` -- resolves every dependency from Central alone and can
never have a stale same-GAV artifact in `~/.m2/repository` silently win with no diagnostic.

**Step 1d — `jmlx-tokenizer/build.gradle`:** new file, mirroring `jmlx-core/build.gradle`'s
plugin/toolchain/spotless/checkstyle blocks but with this module's own dependencies:

```groovy
plugins {
    id 'java-library'
}

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(25)
    }
}

dependencies {
    implementation libs.jackson.databind
    implementation libs.hfjinja
    testImplementation libs.junit.jupiter.api
    testRuntimeOnly 'org.junit.jupiter:junit-jupiter-engine'
}

test {
    useJUnitPlatform()
}
```

(Copy the actual `spotless`/`checkstyle` configuration blocks from `jmlx-core/build.gradle` verbatim
rather than retyping them here — they must stay byte-identical across modules.)

**Step 1e — verify:** `./gradlew :jmlx-tokenizer:build` succeeds with zero source files (an empty
module compiles), confirming both `mavenLocal()` resolution and the module wiring work together.
Commit before writing any class.

```bash
git add settings.gradle gradle/libs.versions.toml build.gradle jmlx-tokenizer/build.gradle
git commit -m "Scaffold jmlx-tokenizer module"
```

## Task 2: `TokenizerException`

```java
package se.alipsa.jmlx.tokenizer;

/** Thrown when a tokenizer.json file, chat template, or token stream cannot be processed. */
public final class TokenizerException extends RuntimeException {

  /** Creates an exception carrying a description of the tokenizer failure. */
  public TokenizerException(String message) {
    super(message);
  }

  /** Creates an exception carrying {@code cause}'s failure as context. */
  public TokenizerException(String message, Throwable cause) {
    super(message, cause);
  }
}
```

No test file needed for this class alone (`RuntimeException` subclasses are exercised by the classes
that throw them, per every other exception class in this codebase).

## Task 3: `ByteLevelCoding`

The standard GPT-2 byte↔Unicode table (`ByteEncoder.swift`'s own source, static data, portable
verbatim; independently re-derivable from the well-known `bytes_to_unicode()` reference algorithm).

```java
package se.alipsa.jmlx.tokenizer;

import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

/**
 * The GPT-2 byte-level encoding: maps each of the 256 possible byte values to a printable Unicode
 * character, so that BPE vocab/merges (which operate on printable text) can represent arbitrary
 * bytes, including whitespace and control characters, without ambiguity.
 */
public final class ByteLevelCoding {

  private static final int[] BYTE_TO_CODE_POINT = new int[256];
  private static final Map<Integer, Integer> CODE_POINT_TO_BYTE = new HashMap<>();

  static {
    boolean[] isPrintable = new boolean[256];
    for (int b = '!'; b <= '~'; b++) {
      isPrintable[b] = true;
    }
    for (int b = 0xA1; b <= 0xAC; b++) {
      isPrintable[b] = true;
    }
    for (int b = 0xAE; b <= 0xFF; b++) {
      isPrintable[b] = true;
    }
    int nextExtraCodePoint = 256;
    for (int b = 0; b < 256; b++) {
      int codePoint = isPrintable[b] ? b : nextExtraCodePoint++;
      BYTE_TO_CODE_POINT[b] = codePoint;
      CODE_POINT_TO_BYTE.put(codePoint, b);
    }
  }

  private ByteLevelCoding() {}

  /** Encodes raw UTF-8 bytes as a byte-level string: one Unicode character per input byte. */
  public static String encode(byte[] utf8Bytes) {
    Objects.requireNonNull(utf8Bytes, "ByteLevelCoding.encode: utf8Bytes must not be null");
    StringBuilder sb = new StringBuilder(utf8Bytes.length);
    for (byte b : utf8Bytes) {
      sb.appendCodePoint(BYTE_TO_CODE_POINT[b & 0xFF]);
    }
    return sb.toString();
  }

  /** Encodes a plain-text string (its UTF-8 bytes) as a byte-level string. */
  public static String encode(String text) {
    Objects.requireNonNull(text, "ByteLevelCoding.encode: text must not be null");
    return encode(text.getBytes(StandardCharsets.UTF_8));
  }

  /**
   * Decodes a byte-level string back to raw bytes. Does not decode as UTF-8 itself: a multi-byte
   * UTF-8 character can be split across separate BPE tokens, so callers must concatenate the raw
   * bytes of every consecutive byte-level token before UTF-8-decoding the combined buffer
   * (see {@link ByteLevelDecoder}).
   */
  public static byte[] decodeToBytes(String byteLevelText) {
    Objects.requireNonNull(byteLevelText, "ByteLevelCoding.decodeToBytes: byteLevelText must not be null");
    byte[] out = new byte[byteLevelText.codePointCount(0, byteLevelText.length())];
    int i = 0;
    int index = 0;
    while (index < byteLevelText.length()) {
      int codePoint = byteLevelText.codePointAt(index);
      Integer b = CODE_POINT_TO_BYTE.get(codePoint);
      if (b == null) {
        throw new TokenizerException(
            "ByteLevelCoding.decodeToBytes: code point " + codePoint + " is not a valid byte-level character");
      }
      out[i++] = b.byteValue();
      index += Character.charCount(codePoint);
    }
    return out;
  }
}
```

(Add `import java.util.Objects;` alongside the others.)

**`ByteLevelCodingTest.java`:**

```java
package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Test;

class ByteLevelCodingTest {

  @Test
  void spaceEncodesToGSubscriptCharacter() {
    // The well-known GPT-2 encoding: byte 0x20 (space) maps to U+0120 ('Ġ').
    assertEquals("Ġ", ByteLevelCoding.encode(" "));
  }

  @Test
  void asciiLettersEncodeUnchanged() {
    assertEquals("low", ByteLevelCoding.encode("low"));
  }

  @Test
  void everyByteValueRoundTrips() {
    byte[] allBytes = new byte[256];
    for (int b = 0; b < 256; b++) {
      allBytes[b] = (byte) b;
    }
    String encoded = ByteLevelCoding.encode(allBytes);
    assertArrayEquals(allBytes, ByteLevelCoding.decodeToBytes(encoded));
  }

  @Test
  void multiByteUtf8RoundTrips() {
    byte[] utf8 = "héllo 中文".getBytes(StandardCharsets.UTF_8);
    String encoded = ByteLevelCoding.encode(utf8);
    assertArrayEquals(utf8, ByteLevelCoding.decodeToBytes(encoded));
  }
}
```

## Task 4: `TokenizerJson` — model records + Jackson loader

**Step 4a — the model records:**

```java
package se.alipsa.jmlx.tokenizer;

/** One entry from {@code tokenizer.json}'s {@code added_tokens} array. */
public record AddedToken(int id, String content, boolean special) {}
```

```java
package se.alipsa.jmlx.tokenizer;

import java.util.Map;

/** {@code tokenizer.json}'s {@code model} object, scoped to the byte-level-BPE fields this port uses. */
public record BpeModelConfig(Map<String, Integer> vocab, Map<String, Integer> mergeRank, boolean ignoreMerges) {}
```

```java
package se.alipsa.jmlx.tokenizer;

import java.util.regex.Pattern;

/** {@code tokenizer.json}'s {@code pre_tokenizer}, scoped to the {@code Split}+{@code ByteLevel} shape both target models use. */
public record PreTokenizerConfig(Pattern splitPattern, boolean addPrefixSpace) {}
```

```java
package se.alipsa.jmlx.tokenizer;

/** {@code tokenizer.json}'s {@code normalizer}: {@code null} (NONE) or {@code {"type": "NFC"}}. */
public enum NormalizerKind {
  NONE,
  NFC
}
```

```java
package se.alipsa.jmlx.tokenizer;

/** One step of a (possibly {@code Sequence}-wrapped) {@code post_processor}. */
public sealed interface PostProcessorStep permits ByteLevelStep, TemplateProcessingStep {}
```

```java
package se.alipsa.jmlx.tokenizer;

/** The {@code ByteLevel} post-processor step: a no-op on the token list (this port does not track offsets). */
public record ByteLevelStep() implements PostProcessorStep {}
```

```java
package se.alipsa.jmlx.tokenizer;

import java.util.List;
import java.util.Map;

/** The {@code TemplateProcessing} post-processor step's {@code single} template. */
public record TemplateProcessingStep(List<TemplateItem> single, Map<String, SpecialTokenInfo> specialTokens)
    implements PostProcessorStep {}
```

```java
package se.alipsa.jmlx.tokenizer;

/** One item of a {@code TemplateProcessing} template: either a literal special token or the real sequence. */
public sealed interface TemplateItem permits SpecialTokenItem, SequenceItem {}
```

```java
package se.alipsa.jmlx.tokenizer;

/** A {@code {"SpecialToken": {"id": "..."}}} template item, referencing a key into {@code specialTokens}. */
public record SpecialTokenItem(String id) implements TemplateItem {}
```

```java
package se.alipsa.jmlx.tokenizer;

/** A {@code {"Sequence": {"id": "A"}}} template item: splice in the real encoded token sequence here. */
public record SequenceItem() implements TemplateItem {}
```

```java
package se.alipsa.jmlx.tokenizer;

import java.util.List;

/** One entry of a {@code TemplateProcessing} step's {@code special_tokens} map. */
public record SpecialTokenInfo(String id, List<Integer> ids, List<String> tokens) {}
```

```java
package se.alipsa.jmlx.tokenizer;

import java.util.List;

/** The parsed, byte-level-BPE-scoped contents of a {@code tokenizer.json} file. */
public record TokenizerJson(
    NormalizerKind normalizer,
    PreTokenizerConfig preTokenizer,
    List<PostProcessorStep> postProcessor,
    BpeModelConfig model,
    List<AddedToken> addedTokens) {}
```

**Step 4b — the loader** (`TokenizerJsonLoader`, a separate class from the `TokenizerJson` record
itself, matching this codebase's own preference for small single-responsibility files):

```java
package se.alipsa.jmlx.tokenizer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Pattern;
import tools.jackson.core.JacksonException;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

/** Parses a {@code tokenizer.json} file into a {@link TokenizerJson}. */
public final class TokenizerJsonLoader {

  private static final ObjectMapper MAPPER = new ObjectMapper();

  private TokenizerJsonLoader() {}

  /** Loads and parses {@code path} as a byte-level-BPE {@code tokenizer.json}. */
  public static TokenizerJson load(Path path) {
    Objects.requireNonNull(path, "TokenizerJsonLoader.load: path must not be null");
    JsonNode root;
    try {
      root = MAPPER.readTree(Files.newInputStream(path));
    } catch (IOException | JacksonException e) {
      throw new TokenizerException("TokenizerJsonLoader.load: failed to parse " + path, e);
    }
    return new TokenizerJson(
        parseNormalizer(root.path("normalizer")),
        parsePreTokenizer(root.get("pre_tokenizer")),
        parsePostProcessor(root.path("post_processor")),
        parseModel(root.get("model")),
        parseAddedTokens(root.path("added_tokens")));
  }

  private static NormalizerKind parseNormalizer(JsonNode node) {
    if (node.isNull() || node.isMissingNode()) {
      return NormalizerKind.NONE;
    }
    String type = node.path("type").asString("");
    if ("NFC".equals(type)) {
      return NormalizerKind.NFC;
    }
    throw new TokenizerException("TokenizerJsonLoader: unsupported normalizer type '" + type + "'");
  }

  private static PreTokenizerConfig parsePreTokenizer(JsonNode node) {
    if (node == null || !"Sequence".equals(node.path("type").asString(""))) {
      throw new TokenizerException("TokenizerJsonLoader: expected pre_tokenizer.type == 'Sequence'");
    }
    String regex = null;
    boolean addPrefixSpace = false;
    for (JsonNode step : node.get("pretokenizers")) {
      String stepType = step.path("type").asString("");
      if ("Split".equals(stepType)) {
        regex = step.path("pattern").path("Regex").asString(null);
      } else if ("ByteLevel".equals(stepType)) {
        addPrefixSpace = step.path("add_prefix_space").asBoolean(false);
      }
    }
    if (regex == null) {
      throw new TokenizerException("TokenizerJsonLoader: pre_tokenizer.pretokenizers has no Split step");
    }
    return new PreTokenizerConfig(Pattern.compile(regex, Pattern.UNICODE_CHARACTER_CLASS), addPrefixSpace);
  }

  private static List<PostProcessorStep> parsePostProcessor(JsonNode node) {
    List<PostProcessorStep> steps = new ArrayList<>();
    if (node.isNull() || node.isMissingNode()) {
      return steps;
    }
    String type = node.path("type").asString("");
    if ("Sequence".equals(type)) {
      for (JsonNode step : node.get("processors")) {
        steps.add(parsePostProcessorStep(step));
      }
    } else {
      steps.add(parsePostProcessorStep(node));
    }
    return steps;
  }

  private static PostProcessorStep parsePostProcessorStep(JsonNode node) {
    String type = node.path("type").asString("");
    if ("ByteLevel".equals(type)) {
      return new ByteLevelStep();
    }
    if ("TemplateProcessing".equals(type)) {
      List<TemplateItem> single = new ArrayList<>();
      for (JsonNode item : node.get("single")) {
        if (item.has("SpecialToken")) {
          single.add(new SpecialTokenItem(item.path("SpecialToken").path("id").asString()));
        } else if (item.has("Sequence")) {
          single.add(new SequenceItem());
        } else {
          throw new TokenizerException("TokenizerJsonLoader: unrecognized TemplateProcessing item " + item);
        }
      }
      Map<String, SpecialTokenInfo> specialTokens = new LinkedHashMap<>();
      for (Map.Entry<String, JsonNode> entry : node.path("special_tokens").properties()) {
        JsonNode v = entry.getValue();
        List<Integer> ids = new ArrayList<>();
        v.path("ids").forEach(idNode -> ids.add(idNode.asInt()));
        List<String> tokens = new ArrayList<>();
        v.path("tokens").forEach(tokenNode -> tokens.add(tokenNode.asString()));
        specialTokens.put(entry.getKey(), new SpecialTokenInfo(v.path("id").asString(), ids, tokens));
      }
      return new TemplateProcessingStep(single, specialTokens);
    }
    throw new TokenizerException("TokenizerJsonLoader: unsupported post_processor step type '" + type + "'");
  }

  private static BpeModelConfig parseModel(JsonNode node) {
    if (!"BPE".equals(node.path("type").asString(""))) {
      throw new TokenizerException("TokenizerJsonLoader: expected model.type == 'BPE'");
    }
    Map<String, Integer> vocab = new HashMap<>();
    for (Map.Entry<String, JsonNode> entry : node.path("vocab").properties()) {
      vocab.put(entry.getKey(), entry.getValue().asInt());
    }
    Map<String, Integer> mergeRank = new HashMap<>();
    int rank = 0;
    for (JsonNode merge : node.path("merges")) {
      mergeRank.put(merge.asString(), rank++);
    }
    boolean ignoreMerges = node.path("ignore_merges").asBoolean(false);
    return new BpeModelConfig(vocab, mergeRank, ignoreMerges);
  }

  private static List<AddedToken> parseAddedTokens(JsonNode node) {
    List<AddedToken> tokens = new ArrayList<>();
    for (JsonNode entry : node) {
      tokens.add(new AddedToken(
          entry.path("id").asInt(), entry.path("content").asString(), entry.path("special").asBoolean(false)));
    }
    return tokens;
  }
}
```

**Amendment (PR #14 review, rounds 1-2): the listing above predates six fixes; left as originally
written, per this repo's convention of amending rather than rewriting a merged plan wholesale.**

- `load`'s `try`/`catch` originally wrapped only `MAPPER.readTree(...)`; every `parse*` call below
  it ran outside the `try` block, so a Jackson-internal exception thrown from deep inside one of
  them (e.g. calling `asString()` on a non-textual node) escaped `HfTokenizer.fromFile` unwrapped,
  with no file path attached. The `try` block now wraps the whole method body.
- `parseModel`'s merge-parsing loop originally called `merge.asString()` unconditionally, assuming
  every element of `merges` is a single space-separated string. It now accepts both that format and
  the two-element-array format (see the amendment two bullets above), and NPE-guards the array
  case: `merge.get(1)` is `null` for a malformed 1-element array, and calling `.asString()` on that
  null used to throw a raw, uncaught `NullPointerException` — now checked and rethrown as
  `TokenizerException`.
- That same loop originally used `mergeRank.put(...)`, so a duplicate pair (reachable once both
  merge serializations can appear in one array) let whichever occurrence came *last* win, silently
  downgrading an earlier pair's priority. `tokenizers` itself keeps the *first* occurrence; the loop
  now uses `mergeRank.putIfAbsent(...)` to match. (Re-joining an array pair with a plain `" "`
  separator is safe either way: byte `0x20` is never a byte-level-printable-range byte — the
  printable range starts at `!` — so a literal space can never occur inside a byte-level symbol,
  and can't collide with a merge's own delimiter.)
- An empty `mergeRank` (e.g. `"merges": []`) originally parsed successfully into a `BpeModelConfig`
  that `BpeMerger` would then silently degrade against — see `BpeMergerTest`'s own
  `withoutIgnoreMergesAndNoMergeRulesEachByteStaysItsOwnSymbol` case, which documents exactly that
  degradation. `parseModel` now throws if `mergeRank` ends up empty.
- `parsePreTokenizer` originally collapsed the ordered `pretokenizers` array into an unordered
  `(regex, addPrefixSpace)` pair by scanning for a `"Split"`-typed step and a `"ByteLevel"`-typed
  step independently, ignoring `behavior`, `invert`, and `use_regex` entirely. A file with
  `invert: true` (which flips which spans count as matches), a non-`"Isolated"` `behavior`, a
  `use_regex: true` `ByteLevel` step, more than two steps, or the two steps in reversed order loaded
  without complaint and then tokenized differently from HF with no diagnostic — exactly the
  silent-divergence failure mode Global Constraint 5 warns about. `parsePreTokenizer` now requires
  `pretokenizers` to be exactly `[Split, ByteLevel]` in that order, with `Split.behavior ==
  "Isolated"`, `Split.invert == false`, and `ByteLevel.use_regex == false`, throwing
  `TokenizerException` otherwise.
- `TemplateProcessing`'s `special_tokens.*.ids`/`.tokens` arrays were parsed with no length check.
  `PostProcessorApplier` (Task 10, amended below) pairs them positionally; an entry with mismatched
  lengths let extra `ids` silently drop or, in the worst case (populated `ids` with an *empty*
  `tokens`), made the whole special token vanish from a template's output with no error.
  `parsePostProcessorStep` now throws if a `special_tokens` entry's `ids` and `tokens` lengths
  differ.

**`TokenizerJsonTest.java`** (against the synthetic fixtures from Task 14 — written here but only
runnable once Task 14 lands; note the dependency in the commit message):

```java
package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import org.junit.jupiter.api.Test;

class TokenizerJsonTest {

  private Path fixture(String name) {
    return Path.of("src/test/resources/se/alipsa/jmlx/tokenizer/" + name);
  }

  @Test
  void qwenStyleFixtureParsesNfcNormalizerAndByteLevelPostProcessor() {
    TokenizerJson json = TokenizerJsonLoader.load(fixture("qwen-style.tokenizer.json"));
    assertEquals(NormalizerKind.NFC, json.normalizer());
    assertFalse(json.model().ignoreMerges());
    assertEquals(1, json.postProcessor().size());
    assertTrue(json.postProcessor().get(0) instanceof ByteLevelStep);
  }

  @Test
  void llama3StyleFixtureParsesIgnoreMergesAndTemplateProcessing() {
    TokenizerJson json = TokenizerJsonLoader.load(fixture("llama3-style.tokenizer.json"));
    assertEquals(NormalizerKind.NONE, json.normalizer());
    assertTrue(json.model().ignoreMerges());
    assertEquals(2, json.postProcessor().size());
    assertTrue(json.postProcessor().get(1) instanceof TemplateProcessingStep);
  }
}
```

## Task 5: `Vocabulary`

```java
package se.alipsa.jmlx.tokenizer;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/** Token string ↔ id lookups, merging a BPE model's own vocab with the file's {@code added_tokens}. */
public final class Vocabulary {

  private final Map<String, Integer> tokenToId;
  private final Map<Integer, String> idToToken;
  private final Set<Integer> specialIds;

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
      throw new TokenizerException("Vocabulary.idOf: no vocabulary entry for token '" + token + "'");
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

  /** Whether {@code token} exists in the vocabulary at all (used by {@code ignore_merges}). */
  public boolean contains(String token) {
    Objects.requireNonNull(token, "Vocabulary.contains: token must not be null");
    return tokenToId.containsKey(token);
  }
}
```

**Amendment (PR #14 review, round 2): the constructor above under-specifies collision handling, and
this plan's "no standalone test file" call below turned out wrong.** Two real gaps, both fixed:

- The constructor's `tokenToId.put`/`idToToken.put` pair only ever *adds* an added token's own
  entries; it never removes whatever the colliding id or token string previously pointed to. If an
  added token's id collides with a *different* model-vocab token, `tokenToId` ends up with two
  string keys mapping to the same id (the original plus the added one) while `idToToken` keeps only
  the added one — so `idOf(originalToken)` still returns the id, but `tokenOf(that id)` no longer
  returns `originalToken`. The symmetric case (an added token's *string* colliding with a different
  model-vocab id) breaks the same invariant the other direction. The constructor now removes the
  stale reverse-mapping on both an id collision and a token-string collision, so `idOf`/`tokenOf`
  stay exact mutual inverses across any collision, not just consistent for the non-colliding case
  this plan's snippet implicitly assumed.
- `specialIds.add(t.id())` is also one-directional: if a *later* added token collides on the same id
  as an *earlier* one and is not itself `special`, the id's entry in `specialIds` is never removed,
  so `isSpecial` keeps reporting `true` for a token that is not special. The loop now does
  `specialIds.add(t.id())`/`specialIds.remove(t.id())` based on the *current* occupant's `special`
  flag, so `specialIds` always reflects whichever token currently owns the id.
- A `maxKnownId()` accessor was added (the largest id assigned by either `modelVocab` or
  `addedTokens`), for `HfTokenizer#decode` (Task 12, amended below) to distinguish a legitimately
  above-vocab id from an in-range hole.
- This plan's "no standalone test file" call turned out wrong once the collision-handling behavior
  above needed exercising directly: `VocabularyTest.java` now covers both collision directions, the
  `specialIds` pruning fix, and `maxKnownId()`, independent of whatever `HfTokenizerTest` happens to
  reach through a real tokenizer.json fixture.

## Task 6: `BpeMerger`

```java
package se.alipsa.jmlx.tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** The byte-level BPE merge algorithm: repeatedly applies the lowest-rank adjacent-pair merge. */
public final class BpeMerger {

  private final BpeModelConfig model;

  public BpeMerger(BpeModelConfig model) {
    this.model = Objects.requireNonNull(model, "BpeMerger: model must not be null");
  }

  /** Merges one byte-level-encoded pre-token chunk into its final BPE symbol sequence. */
  public List<String> merge(String byteLevelWord) {
    Objects.requireNonNull(byteLevelWord, "BpeMerger.merge: byteLevelWord must not be null");
    if (model.ignoreMerges() && model.vocab().containsKey(byteLevelWord)) {
      return List.of(byteLevelWord);
    }
    List<String> symbols = new ArrayList<>();
    byteLevelWord.codePoints().forEach(cp -> symbols.add(new String(Character.toChars(cp))));
    while (symbols.size() > 1) {
      int bestRank = Integer.MAX_VALUE;
      int bestIndex = -1;
      for (int i = 0; i < symbols.size() - 1; i++) {
        Integer rank = model.mergeRank().get(symbols.get(i) + " " + symbols.get(i + 1));
        if (rank != null && rank < bestRank) {
          bestRank = rank;
          bestIndex = i;
        }
      }
      if (bestIndex == -1) {
        break;
      }
      String merged = symbols.get(bestIndex) + symbols.get(bestIndex + 1);
      symbols.set(bestIndex, merged);
      symbols.remove(bestIndex + 1);
    }
    for (String symbol : symbols) {
      if (!model.vocab().containsKey(symbol)) {
        throw new TokenizerException(
            "BpeMerger.merge: merged symbol '" + symbol + "' has no vocabulary entry (byte_fallback"
                + " is assumed false for this port's target models — see req/plans/phase5-m2-plan.md)");
      }
    }
    return symbols;
  }
}
```

**`BpeMergerTest.java`:**

```java
package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class BpeMergerTest {

  private static final Map<String, Integer> BASE_VOCAB =
      Map.of("l", 0, "o", 1, "w", 2, "t", 3, "h", 4, "e", 5, "Ġ", 11, "lo", 12, "low", 13, "th", 14, "the", 15,
          "Ġthe", 16);

  private static final Map<String, Integer> MERGE_RANK =
      Map.of("l o", 0, "lo w", 1, "t h", 2, "th e", 3, "Ġ the", 4);

  @Test
  void mergesLowestRankPairsInOrderUntilASingleSymbolRemains() {
    BpeMerger merger = new BpeMerger(new BpeModelConfig(BASE_VOCAB, MERGE_RANK, false));
    assertEquals(List.of("low"), merger.merge("low"));
  }

  @Test
  void mergesAcrossFourSymbolsIncludingTheByteLevelSpaceMarker() {
    BpeMerger merger = new BpeMerger(new BpeModelConfig(BASE_VOCAB, MERGE_RANK, false));
    assertEquals(List.of("Ġthe"), merger.merge("Ġthe"));
  }

  @Test
  void ignoreMergesShortCircuitsToAWholeVocabHitWithoutRunningTheMergeLoop() {
    // Empty merge-rank table: without the ignore_merges shortcut, "low" could never merge past
    // its three individual byte symbols. With it, the whole-word vocab hit wins directly.
    BpeMerger merger = new BpeMerger(new BpeModelConfig(BASE_VOCAB, Map.of(), true));
    assertEquals(List.of("low"), merger.merge("low"));
  }

  @Test
  void withoutIgnoreMergesAndNoMergeRulesEachByteStaysItsOwnSymbol() {
    BpeMerger merger = new BpeMerger(new BpeModelConfig(BASE_VOCAB, Map.of(), false));
    assertEquals(List.of("l", "o", "w"), merger.merge("low"));
  }
}
```

## Task 7: `ByteLevelPreTokenizer`

```java
package se.alipsa.jmlx.tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.regex.Matcher;

/** Splits normalized text into byte-level-encoded pre-token chunks via the model's Split regex. */
public final class ByteLevelPreTokenizer {

  private final PreTokenizerConfig config;

  public ByteLevelPreTokenizer(PreTokenizerConfig config) {
    this.config = Objects.requireNonNull(config, "ByteLevelPreTokenizer: config must not be null");
  }

  /** Splits {@code text} into byte-level-encoded chunks, one per regex match, in order. */
  public List<String> split(String text) {
    Objects.requireNonNull(text, "ByteLevelPreTokenizer.split: text must not be null");
    String input = config.addPrefixSpace() && !text.isEmpty() && text.charAt(0) != ' ' ? " " + text : text;
    List<String> chunks = new ArrayList<>();
    Matcher matcher = config.splitPattern().matcher(input);
    while (matcher.find()) {
      chunks.add(ByteLevelCoding.encode(matcher.group()));
    }
    return chunks;
  }
}
```

**`ByteLevelPreTokenizerTest.java`:**

```java
package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Test;

class ByteLevelPreTokenizerTest {

  // The real Qwen2.5/Llama-3 regex (Qwen2.5's \p{N} variant), verified against each model's
  // actual tokenizer.json — see this plan's Findings section.
  private static final Pattern QWEN_REGEX = Pattern.compile(
      "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*"
          + "|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
      Pattern.UNICODE_CHARACTER_CLASS);

  @Test
  void splitsWordAndLeadingSpaceIntoSeparateChunks() {
    ByteLevelPreTokenizer pretokenizer = new ByteLevelPreTokenizer(new PreTokenizerConfig(QWEN_REGEX, false));
    // "low the": "low" has no leading space; " the" is captured as one chunk with its leading space.
    assertEquals(List.of("low", "Ġthe"), pretokenizer.split("low the"));
  }

  @Test
  void contractionIsItsOwnChunk() {
    ByteLevelPreTokenizer pretokenizer = new ByteLevelPreTokenizer(new PreTokenizerConfig(QWEN_REGEX, false));
    assertEquals(List.of("it", "'s"), pretokenizer.split("it's"));
  }
}
```

**Amendment (PR #14 review, rounds 1-2): `split` above has two real bugs, the second introduced and
then corrected across two review rounds on the same PR.**

- `add_prefix_space` was applied once, to the whole `text`, before running the regex at all —
  correct only when the regex produces a single match. HF's real pipeline runs `ByteLevel` *after*
  `Split`, over each already-split piece independently, so every piece that doesn't already start
  with a space gets its own prefix space, not just the first character of the whole input. `split`
  now runs the regex over the unmodified `text` first and applies `add_prefix_space` per resulting
  piece.
- Round 1 of this PR's review flagged that `Matcher.find()` silently skips any span the regex
  doesn't match, and the fix committed then made `split` throw `TokenizerException` on any such gap.
  **Round 2 corrected that fix as over-strict relative to HF itself:** HF's `find_matches` trait
  contract requires covering the whole string with contiguous, ordered slices, explicitly keeping
  non-matching spans (tagged `is_match=false`) rather than discarding them, under
  `SplitDelimiterBehavior::Isolated`. A file whose regex doesn't cover every character therefore
  still tokenizes successfully under real HF, and this port hard-failing where HF wouldn't is itself
  a fidelity bug, not a fix. `split` now emits an unmatched span as its own chunk instead of
  throwing; a span that turns out to encode an unrepresentable symbol still surfaces loudly via
  `BpeMerger.merge`'s existing no-vocabulary-entry check, so this isn't a return to the original
  silent-drop behavior — it's matching HF's actual, looser contract instead of guessing at a
  stricter one.

## Task 8: `TextNormalizer`

```java
package se.alipsa.jmlx.tokenizer;

import java.text.Normalizer;
import java.util.Objects;

/** Applies a {@code tokenizer.json} normalizer ({@code None} or {@code NFC}) to input text. */
public final class TextNormalizer {

  private TextNormalizer() {}

  /** Normalizes {@code text} per {@code kind}. */
  public static String normalize(NormalizerKind kind, String text) {
    Objects.requireNonNull(kind, "TextNormalizer.normalize: kind must not be null");
    Objects.requireNonNull(text, "TextNormalizer.normalize: text must not be null");
    return switch (kind) {
      case NONE -> text;
      case NFC -> Normalizer.normalize(text, Normalizer.Form.NFC);
    };
  }
}
```

**No standalone test file** — `java.text.Normalizer` is a JDK-tested implementation; this class adds
one line of dispatch logic, exercised end-to-end by `HfTokenizerTest`'s Qwen-style fixture case.

## Task 9: `AddedTokenSplitter`

```java
package se.alipsa.jmlx.tokenizer;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/** Splits input text around literal added-token strings (longest-first, no lstrip/rstrip — see Findings). */
public final class AddedTokenSplitter {

  /** One segment of split input: either literal added-token text, or plain text needing full tokenization. */
  public record Segment(String text, boolean isAddedToken) {}

  private final Pattern addedTokenPattern;

  public AddedTokenSplitter(List<AddedToken> addedTokens) {
    Objects.requireNonNull(addedTokens, "AddedTokenSplitter: addedTokens must not be null");
    if (addedTokens.isEmpty()) {
      this.addedTokenPattern = null;
      return;
    }
    String alternation = addedTokens.stream()
        .map(AddedToken::content)
        .sorted(Comparator.comparingInt(String::length).reversed())
        .map(Pattern::quote)
        .collect(Collectors.joining("|"));
    this.addedTokenPattern = Pattern.compile(alternation);
  }

  /** Splits {@code text} into ordered segments, tagging which ones are literal added-token strings. */
  public List<Segment> split(String text) {
    Objects.requireNonNull(text, "AddedTokenSplitter.split: text must not be null");
    List<Segment> segments = new ArrayList<>();
    if (addedTokenPattern == null) {
      if (!text.isEmpty()) {
        segments.add(new Segment(text, false));
      }
      return segments;
    }
    Matcher matcher = addedTokenPattern.matcher(text);
    int last = 0;
    while (matcher.find()) {
      if (matcher.start() > last) {
        segments.add(new Segment(text.substring(last, matcher.start()), false));
      }
      segments.add(new Segment(matcher.group(), true));
      last = matcher.end();
    }
    if (last < text.length()) {
      segments.add(new Segment(text.substring(last), false));
    }
    return segments;
  }
}
```

**`AddedTokenSplitterTest.java`:**

```java
package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import org.junit.jupiter.api.Test;

class AddedTokenSplitterTest {

  private static final List<AddedToken> QWEN_ADDED_TOKENS =
      List.of(new AddedToken(151644, "<|im_start|>", true), new AddedToken(151645, "<|im_end|>", true));

  @Test
  void splitsPlainTextAroundLiteralAddedTokens() {
    AddedTokenSplitter splitter = new AddedTokenSplitter(QWEN_ADDED_TOKENS);
    List<AddedTokenSplitter.Segment> segments = splitter.split("<|im_start|>user\nhi<|im_end|>");
    assertEquals(
        List.of(
            new AddedTokenSplitter.Segment("<|im_start|>", true),
            new AddedTokenSplitter.Segment("user\nhi", false),
            new AddedTokenSplitter.Segment("<|im_end|>", true)),
        segments);
  }

  @Test
  void plainTextWithNoAddedTokensIsOneSegment() {
    AddedTokenSplitter splitter = new AddedTokenSplitter(QWEN_ADDED_TOKENS);
    assertEquals(List.of(new AddedTokenSplitter.Segment("hello world", false)), splitter.split("hello world"));
  }
}
```

## Task 10: `PostProcessorApplier`

```java
package se.alipsa.jmlx.tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** Applies a {@code tokenizer.json} post-processor's steps to an already-encoded token-string list. */
public final class PostProcessorApplier {

  private PostProcessorApplier() {}

  /** Applies every step in {@code steps} to {@code tokens} in order. */
  public static List<String> apply(List<PostProcessorStep> steps, List<String> tokens, boolean addSpecialTokens) {
    Objects.requireNonNull(steps, "PostProcessorApplier.apply: steps must not be null");
    Objects.requireNonNull(tokens, "PostProcessorApplier.apply: tokens must not be null");
    List<String> result = tokens;
    for (PostProcessorStep step : steps) {
      if (step instanceof TemplateProcessingStep template) {
        result = applyTemplate(template, result, addSpecialTokens);
      }
      // ByteLevelStep is a no-op on the token list (see Findings: it only fixes up offsets, which
      // this port does not track).
    }
    return result;
  }

  private static List<String> applyTemplate(TemplateProcessingStep step, List<String> tokens, boolean addSpecialTokens) {
    List<String> out = new ArrayList<>();
    for (TemplateItem item : step.single()) {
      if (item instanceof SequenceItem) {
        out.addAll(tokens);
      } else if (item instanceof SpecialTokenItem special) {
        if (addSpecialTokens) {
          SpecialTokenInfo info = step.specialTokens().get(special.id());
          if (info == null) {
            throw new TokenizerException(
                "PostProcessorApplier: template references unknown special token '" + special.id() + "'");
          }
          out.addAll(info.tokens());
        }
      }
    }
    return out;
  }
}
```

**`PostProcessorApplierTest.java`:**

```java
package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class PostProcessorApplierTest {

  @Test
  void byteLevelStepAloneIsANoOp() {
    List<String> tokens = List.of("low", "Ġthe");
    assertEquals(tokens, PostProcessorApplier.apply(List.of(new ByteLevelStep()), tokens, true));
  }

  @Test
  void templateProcessingPrependsBosTokenWhenAddSpecialTokensIsTrue() {
    var template = new TemplateProcessingStep(
        List.of(new SpecialTokenItem("<|begin_of_text|>"), new SequenceItem()),
        Map.of("<|begin_of_text|>", new SpecialTokenInfo("<|begin_of_text|>", List.of(128000),
            List.of("<|begin_of_text|>"))));
    List<String> tokens = List.of("low", "Ġthe");
    assertEquals(
        List.of("<|begin_of_text|>", "low", "Ġthe"),
        PostProcessorApplier.apply(List.of(new ByteLevelStep(), template), tokens, true));
  }

  @Test
  void templateProcessingOmitsBosTokenWhenAddSpecialTokensIsFalse() {
    var template = new TemplateProcessingStep(
        List.of(new SpecialTokenItem("<|begin_of_text|>"), new SequenceItem()),
        Map.of("<|begin_of_text|>", new SpecialTokenInfo("<|begin_of_text|>", List.of(128000),
            List.of("<|begin_of_text|>"))));
    List<String> tokens = List.of("low", "Ġthe");
    assertEquals(tokens, PostProcessorApplier.apply(List.of(template), tokens, false));
  }
}
```

**Amendment (PR #14 review, round 2): `apply` above discards a `SpecialTokenInfo` field it already
parses, forcing a redundant lookup that can fail even when the original data didn't need it.**
`TokenizerJsonLoader` (Task 4) parses a `TemplateProcessing` special token's own `ids` list, but the
snippet above only ever reads `info.tokens()`, re-resolving each string back to an id later in
`HfTokenizer.encode` via `Vocabulary.idOf`. That throws for any template special token present in
`special_tokens` but missing from `model.vocab` + `added_tokens` — even though the correct id sat
right there in the discarded `ids` list. `apply` now returns `List<ResolvedToken>` (a new
`record ResolvedToken(String text, Integer id)`) instead of `List<String>`: a `SequenceItem`'s
tokens carry a `null` id (still resolved later via `Vocabulary.idOf`), while a `SpecialTokenItem`'s
tokens carry the `SpecialTokenInfo`'s own `ids` entry at the same index, used as-is. `HfTokenizer`
(Task 12, amended below) is the only caller and was updated to match.

```java
package se.alipsa.jmlx.tokenizer;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/** Decodes a list of token strings back to text, byte-decoding non-added tokens and passing added tokens through. */
public final class ByteLevelDecoder {

  private ByteLevelDecoder() {}

  /**
   * Decodes {@code tokens} to text. Consecutive non-added-token strings have their raw bytes
   * concatenated before UTF-8 decoding (a multi-byte character can be split across token
   * boundaries); added-token strings are literal text and pass through unchanged as their own
   * segment, exactly as {@code Decoder.swift}'s {@code ByteLevelDecoder} does.
   */
  public static String decode(List<String> tokens, Set<String> addedTokenContents) {
    Objects.requireNonNull(tokens, "ByteLevelDecoder.decode: tokens must not be null");
    Objects.requireNonNull(addedTokenContents, "ByteLevelDecoder.decode: addedTokenContents must not be null");
    StringBuilder out = new StringBuilder();
    ByteArrayOutputStream pending = new ByteArrayOutputStream();
    for (String token : tokens) {
      if (addedTokenContents.contains(token)) {
        out.append(new String(pending.toByteArray(), StandardCharsets.UTF_8));
        pending.reset();
        out.append(token);
      } else {
        pending.writeBytes(ByteLevelCoding.decodeToBytes(token));
      }
    }
    out.append(new String(pending.toByteArray(), StandardCharsets.UTF_8));
    return out.toString();
  }
}
```

**`ByteLevelDecoderTest.java`:**

```java
package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;

class ByteLevelDecoderTest {

  @Test
  void decodesPlainByteLevelTokensBackToText() {
    assertEquals("low the", ByteLevelDecoder.decode(List.of("low", "Ġthe"), Set.of()));
  }

  @Test
  void addedTokensPassThroughLiterallyBetweenDecodedText() {
    String result = ByteLevelDecoder.decode(
        List.of("<|im_start|>", "Ġthe"), Set.of("<|im_start|>", "<|im_end|>"));
    assertEquals("<|im_start|> the", result);
  }

  @Test
  void multiByteCharacterSplitAcrossTwoTokensStillDecodesCorrectly() {
    // "é" is 2 UTF-8 bytes (0xC3 0xA9); simulate them arriving as two separate BPE token pieces.
    String piece1 = ByteLevelCoding.encode(new byte[] {(byte) 0xC3});
    String piece2 = ByteLevelCoding.encode(new byte[] {(byte) 0xA9});
    assertEquals("é", ByteLevelDecoder.decode(List.of(piece1, piece2), Set.of()));
  }
}
```

## Task 12: `HfTokenizer`

```java
package se.alipsa.jmlx.tokenizer;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/** A byte-level BPE tokenizer loaded from a {@code tokenizer.json} file. */
public final class HfTokenizer {

  private final TokenizerJson json;
  private final Vocabulary vocabulary;
  private final BpeMerger merger;
  private final ByteLevelPreTokenizer pretokenizer;
  private final AddedTokenSplitter addedTokenSplitter;
  private final Set<String> addedTokenContents;

  private HfTokenizer(TokenizerJson json) {
    this.json = json;
    this.vocabulary = new Vocabulary(json.model().vocab(), json.addedTokens());
    this.merger = new BpeMerger(json.model());
    this.pretokenizer = new ByteLevelPreTokenizer(json.preTokenizer());
    this.addedTokenSplitter = new AddedTokenSplitter(json.addedTokens());
    this.addedTokenContents = new HashSet<>();
    json.addedTokens().forEach(t -> addedTokenContents.add(t.content()));
  }

  /** Loads a tokenizer from a {@code tokenizer.json} file. */
  public static HfTokenizer fromFile(Path tokenizerJsonPath) {
    Objects.requireNonNull(tokenizerJsonPath, "HfTokenizer.fromFile: tokenizerJsonPath must not be null");
    return new HfTokenizer(TokenizerJsonLoader.load(tokenizerJsonPath));
  }

  /** Encodes {@code text} into token ids. */
  public List<Integer> encode(String text, boolean addSpecialTokens) {
    Objects.requireNonNull(text, "HfTokenizer.encode: text must not be null");
    List<String> tokens = new ArrayList<>();
    for (AddedTokenSplitter.Segment segment : addedTokenSplitter.split(text)) {
      if (segment.isAddedToken()) {
        tokens.add(segment.text());
        continue;
      }
      String normalized = TextNormalizer.normalize(json.normalizer(), segment.text());
      for (String chunk : pretokenizer.split(normalized)) {
        tokens.addAll(merger.merge(chunk));
      }
    }
    List<String> processed = PostProcessorApplier.apply(json.postProcessor(), tokens, addSpecialTokens);
    List<Integer> ids = new ArrayList<>(processed.size());
    for (String token : processed) {
      ids.add(vocabulary.idOf(token));
    }
    return ids;
  }

  /** Decodes token ids back into text. */
  public String decode(List<Integer> ids, boolean skipSpecialTokens) {
    Objects.requireNonNull(ids, "HfTokenizer.decode: ids must not be null");
    List<String> tokens = new ArrayList<>();
    for (int id : ids) {
      if (skipSpecialTokens && vocabulary.isSpecial(id)) {
        continue;
      }
      tokens.add(vocabulary.tokenOf(id));
    }
    return ByteLevelDecoder.decode(tokens, addedTokenContents);
  }
}
```

**`HfTokenizerTest.java`** (end-to-end against both Task 14 fixtures):

```java
package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;

class HfTokenizerTest {

  private Path fixture(String name) {
    return Path.of("src/test/resources/se/alipsa/jmlx/tokenizer/" + name);
  }

  @Test
  void qwenStyleEncodeDecodeRoundTrips() {
    HfTokenizer tokenizer = HfTokenizer.fromFile(fixture("qwen-style.tokenizer.json"));
    List<Integer> ids = tokenizer.encode("low the", false);
    assertEquals(List.of(13, 16), ids);
    assertEquals("low the", tokenizer.decode(ids, false));
  }

  @Test
  void llama3StylePrependsBosTokenWhenAddSpecialTokensIsTrue() {
    HfTokenizer tokenizer = HfTokenizer.fromFile(fixture("llama3-style.tokenizer.json"));
    List<Integer> ids = tokenizer.encode("low the", true);
    assertEquals(List.of(128000, 13, 16), ids);
  }

  @Test
  void llama3StyleOmitsBosTokenWhenAddSpecialTokensIsFalse() {
    HfTokenizer tokenizer = HfTokenizer.fromFile(fixture("llama3-style.tokenizer.json"));
    assertEquals(List.of(13, 16), tokenizer.encode("low the", false));
  }

  @Test
  void decodeSkipsSpecialTokensWhenRequested() {
    HfTokenizer tokenizer = HfTokenizer.fromFile(fixture("llama3-style.tokenizer.json"));
    List<Integer> ids = tokenizer.encode("low the", true);
    assertEquals("low the", tokenizer.decode(ids, true));
  }
}
```

**Amendment (PR #14 review, rounds 1-2): both `encode` and `decode` above trusted their inputs in
ways that could silently corrupt or drop tokens.** `encode`'s `for (String token : processed) { ids.add(vocabulary.idOf(token)); }`
loop predates `PostProcessorApplier`'s `ResolvedToken` return type (Task 10, amended above); it now
iterates `ResolvedToken`s and only falls back to `vocabulary.idOf(token.text())` when a token has no
pre-resolved id (a `SequenceItem` token). A `SpecialTokenItem` token's id -- sourced from the
tokenizer.json file's own `special_tokens.ids`, not a sampled model output -- is cross-checked against
`vocabulary.hasId(id)` and throws if absent (finding 7): unlike a sampled logit, there is no legitimate
reason for a file-declared special-token id to be missing from the vocabulary, and trusting it
unchecked would bake a corrupt id into the encoded sequence that `decode` would then silently drop.
`decode`'s `tokens.add(vocabulary.tokenOf(id))` unconditionally threw `TokenizerException` for *any*
unrecognized id, which is too strict: e.g. Qwen2.5's `config.json` `vocab_size` (152064) exceeds its
tokenizer vocab (151665), so a sampled logit can legitimately land in that gap, and a hard exception
there would abort an entire generation loop over a normal, expected input. `decode` now checks
`vocabulary.hasId(id)` first and, only when absent, skips ids above `Vocabulary#maxKnownId` (Task 7,
amended above) while still throwing on an in-range hole -- indistinguishable from the above-vocab case
by id value alone, but always a real bug (the wrong tokenizer for the checkpoint, a mis-parsed
`added_tokens`, a `Vocabulary` bug) that must not be swept up by the same skip (finding 2). Both fixes
are exercised by `HfTokenizerTest`'s `encodeThrowsWhenATemplateSpecialTokenIdHasNoVocabularyEntry`,
`decodeSkipsAnAboveVocabIdInsteadOfAbortingTheWholeSequence`, and
`decodeThrowsOnAnInRangeIdWithNoVocabularyEntryInsteadOfSilentlyDroppingIt` (the last two against a new
`qwen-style-with-id-gap.tokenizer.json` fixture; the first against `llama3-style-bad-template-id.tokenizer.json`).

## Task 13: `ChatTemplateRenderer`

```java
package se.alipsa.jmlx.tokenizer;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import se.alipsa.hfjinja.HfJinjaException;
import se.alipsa.hfjinja.Template;

/** Renders a Hugging Face {@code chat_template} Jinja string via {@code hfjinja}. */
public final class ChatTemplateRenderer {

  private ChatTemplateRenderer() {}

  /**
   * Renders {@code chatTemplate} against the standard HF chat-template context variables, plus any
   * caller-supplied {@code extraContext} (e.g. {@code tools} for tool-calling templates) merged in
   * underneath the fixed keys below.
   */
  public static String render(
      String chatTemplate,
      List<Map<String, Object>> messages,
      boolean addGenerationPrompt,
      String bosToken,
      String eosToken,
      Map<String, Object> extraContext) {
    Objects.requireNonNull(chatTemplate, "ChatTemplateRenderer.render: chatTemplate must not be null");
    Objects.requireNonNull(messages, "ChatTemplateRenderer.render: messages must not be null");
    Objects.requireNonNull(extraContext, "ChatTemplateRenderer.render: extraContext must not be null");
    Map<String, Object> context = new HashMap<>(extraContext);
    context.put("messages", messages);
    context.put("add_generation_prompt", addGenerationPrompt);
    context.put("bos_token", bosToken);
    context.put("eos_token", eosToken);
    try {
      return Template.parse(chatTemplate).render(context);
    } catch (HfJinjaException e) {
      throw new TokenizerException("ChatTemplateRenderer.render: failed to render chat template", e);
    }
  }
}
```

**`ChatTemplateRendererTest.java`** (against the two real chat templates captured in Task 14):

```java
package se.alipsa.jmlx.tokenizer;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class ChatTemplateRendererTest {

  private String readFixture(String name) throws IOException {
    return Files.readString(Path.of("src/test/resources/se/alipsa/jmlx/tokenizer/" + name));
  }

  @Test
  void llama3TemplateWrapsMessagesWithHeaderTagsAndBosToken() throws IOException {
    String template = readFixture("llama3-instruct-chat-template.jinja");
    String result = ChatTemplateRenderer.render(
        template,
        List.of(Map.of("role", "user", "content", "Hello")),
        true,
        "<|begin_of_text|>",
        "<|eot_id|>",
        Map.of());
    assertTrue(result.startsWith("<|begin_of_text|>"));
    assertTrue(result.contains("<|start_header_id|>user<|end_header_id|>"));
    assertTrue(result.endsWith("<|start_header_id|>assistant<|end_header_id|>\n\n"));
  }

  @Test
  void qwenTemplateInsertsDefaultSystemPromptWhenNoneProvided() throws IOException {
    String template = readFixture("qwen2.5-instruct-chat-template.jinja");
    String result = ChatTemplateRenderer.render(
        template, List.of(Map.of("role", "user", "content", "Hello")), true, null, "<|im_end|>", Map.of());
    assertTrue(result.contains("You are Qwen, created by Alibaba Cloud."));
    assertTrue(result.contains("<|im_start|>user\nHello<|im_end|>"));
    assertTrue(result.endsWith("<|im_start|>assistant\n"));
  }
}
```

## Task 14: Test fixtures

**Step 14a — `qwen-style.tokenizer.json`** (small, hand-built; encode/decode results are hand-traced
in this plan's Findings-derived design, not asserted blindly):

```json
{
  "normalizer": {"type": "NFC"},
  "pre_tokenizer": {
    "type": "Sequence",
    "pretokenizers": [
      {"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false},
      {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}
    ]
  },
  "post_processor": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
  "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
  "model": {
    "type": "BPE",
    "ignore_merges": false,
    "vocab": {"l": 0, "o": 1, "w": 2, "t": 3, "h": 4, "e": 5, "n": 6, "s": 7, "r": 8, "i": 9, "d": 10,
      "Ġ": 11, "lo": 12, "low": 13, "th": 14, "the": 15, "Ġthe": 16, "<|endoftext|>": 17,
      "<|im_start|>": 18, "<|im_end|>": 19},
    "merges": ["l o", "lo w", "t h", "th e", "Ġ the"]
  },
  "added_tokens": [
    {"id": 17, "content": "<|endoftext|>", "special": true, "lstrip": false, "rstrip": false, "normalized": false, "single_word": false},
    {"id": 18, "content": "<|im_start|>", "special": true, "lstrip": false, "rstrip": false, "normalized": false, "single_word": false},
    {"id": 19, "content": "<|im_end|>", "special": true, "lstrip": false, "rstrip": false, "normalized": false, "single_word": false}
  ]
}
```

**Step 14b — `llama3-style.tokenizer.json`** (same base vocab/merges, `ignore_merges: true`, and a
`Sequence[ByteLevel, TemplateProcessing]` post-processor prepending a BOS token):

```json
{
  "normalizer": null,
  "pre_tokenizer": {
    "type": "Sequence",
    "pretokenizers": [
      {"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false},
      {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}
    ]
  },
  "post_processor": {
    "type": "Sequence",
    "processors": [
      {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": true},
      {"type": "TemplateProcessing",
        "single": [{"SpecialToken": {"id": "<|begin_of_text|>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
        "special_tokens": {"<|begin_of_text|>": {"id": "<|begin_of_text|>", "ids": [128000], "tokens": ["<|begin_of_text|>"]}}}
    ]
  },
  "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
  "model": {
    "type": "BPE",
    "ignore_merges": true,
    "vocab": {"l": 0, "o": 1, "w": 2, "t": 3, "h": 4, "e": 5, "n": 6, "s": 7, "r": 8, "i": 9, "d": 10,
      "Ġ": 11, "lo": 12, "low": 13, "th": 14, "the": 15, "Ġthe": 16, "<|begin_of_text|>": 128000},
    "merges": ["l o", "lo w", "t h", "th e", "Ġ the"]
  },
  "added_tokens": [
    {"id": 128000, "content": "<|begin_of_text|>", "special": true, "lstrip": false, "rstrip": false, "normalized": false, "single_word": false}
  ]
}
```

**Step 14c — `qwen2.5-instruct-chat-template.jinja`**: the exact, byte-verbatim `chat_template` field
fetched from `Qwen/Qwen2.5-0.5B-Instruct/raw/main/tokenizer_config.json` — copy the raw string value
directly (do not retype it by hand; extract it with a small script, e.g.
`python3 -c "import json; print(json.load(open('tokenizer_config.json'))['chat_template'])"` against
a freshly-downloaded copy, to guarantee byte-fidelity) into this file with no reformatting.

**Step 14d — `llama3-instruct-chat-template.jinja`**: same extraction approach, against
`NousResearch/Meta-Llama-3-8B-Instruct/raw/main/tokenizer_config.json` (an ungated re-upload of
`meta-llama/Meta-Llama-3-8B-Instruct`'s own tokenizer files, used only because the gated original
requires an interactive license-acceptance step; the content is Meta's own, unmodified). If gated
access is available at implementation time, prefer fetching directly from `meta-llama`'s own repo and
note in the commit message which source was used.

```bash
git add jmlx-tokenizer/src/test/resources
git commit -m "Add tokenizer test fixtures (synthetic tokenizer.json + real chat templates)"
```

## Task 15: Documentation

- `CLAUDE.md`'s Architecture diagram: add a `jmlx-tokenizer` row (`se.alipsa.jmlx.tokenizer` —
  `HfTokenizer`, `ChatTemplateRenderer`, and the byte-level-BPE pipeline classes), noting it has **no**
  native dependency and does not participate in the "loading order matters" native-guard discussion at
  all — the first pure-Java module in this codebase.
- `req/phase5-plan.md`'s Status table: flip M2's row from "Desk-research spike done... architecture
  choice still open" to `Done` (or `In progress` if implemented across multiple sittings), recording
  the module (`jmlx-tokenizer`), the chosen architecture (pure-Java, not FFM), and the dependencies
  added (Jackson 3.1.2, hfjinja 0.5.0, both resolved via `mavenLocal()`), same convention
  `req/phase4-plan.md`'s and `req/phase5-plan.md`'s
  own M1 row already use.
- `req/phase5-plan.md`'s D3 finding about "no official Java equivalent of `swift-jinja` exists" and
  the onig/fancy-regex FFM-path analysis both become historical context once M2 ships via the
  pure-Java path — leave them in place (this document's own convention is to amend, not delete,
  superseded research) but note in the Status table that D3's FFM-vs-pure-Java question is now
  resolved by this plan, not still open.

## Task 16: full verification pass

- `./gradlew :jmlx-tokenizer:test` — every test class above passes. No native bootstrap required —
  confirm by running this on a fresh checkout with `native/install/lib/mlx.metallib` absent, unlike
  every prior milestone's verification pass.
- `./gradlew build` — confirms `jmlx-ffi`/`jmlx-core`/`jmlx-tokenizer`/`jmlx-examples` all still
  compile together.
- `./gradlew spotlessCheck checkstyleMain checkstyleTest` (run against `jmlx-tokenizer` specifically,
  and the whole build) — formatting/style, same gate every prior milestone has run.
- Manual sanity check against a real (not synthetic) `tokenizer.json`: download
  `Qwen/Qwen2.5-0.5B-Instruct/tokenizer.json` and confirm `HfTokenizer.fromFile(...)` loads it without
  throwing and that `encode("Hello, world!", true)` followed by `decode(ids, true)` round-trips back
  to `"Hello, world!"` — this is the first real-file exercise this plan's own synthetic-fixture test
  suite doesn't cover, mirroring M1's own documented real-checkpoint-fixture gap (`req/plans/
  phase5-m1-plan.md` Task 4) rather than silently assuming the synthetic fixtures are sufficient proof.

## Deliberately not covered by this plan

- SentencePiece, WordPiece, Unigram, `BertTokenizer`, any non-BPE `model.type` — Global Constraint 1.
- General normalizer plugin chain beyond `None`/`NFC` (`Lowercase`, `StripAccents`, `Sequence`, etc.)
  — neither target model needs them.
- `Trie`-based added-token matching, and added-token `lstrip`/`rstrip`/`single_word` whitespace-eating
  behavior — confirmed unused by both target models (Findings above); a future model that does set
  these flags would need this plan revisited.
- Offset tracking (character↔byte position mapping) — `ByteLevel` post-processor/decoder steps are
  modeled as no-ops on the token list for exactly this reason.
- Sentence-pair (`pair`-template) post-processing — chat/completion use is single-sequence only.
- Tool-calling context variables (`tools`, `tool_call`, etc.) for Qwen2.5's own chat_template — Task
  13's `extraContext` parameter exists so M3 can add these later without touching `ChatTemplateRenderer`,
  but populating them is M3's job, not this plan's.
- A real (non-synthetic) `tokenizer.json`/`tokenizer_config.json` fixture committed to the repo — both
  real files are 7-9 MB; Task 16's manual sanity check exercises the real files without committing
  them, mirroring M1's own real-checkpoint-fixture deferral to M3.
- Batched encode/decode APIs (`encode_batch`-equivalent) — not needed until M3 names a caller that
  batches.
- Publishing `jmlx-tokenizer` as a standalone artifact/repository — explicitly decided against for
  now (this plan's own opening paragraph); revisit only if a consumer outside jmlx materializes.
