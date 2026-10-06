# jmlx-tokenizer

`jmlx-tokenizer` is a pure-Java 21+ reader and runtime for local Hugging Face
`tokenizer.json` files. It supports ByteLevel or Metaspace BPE, Metaspace Unigram without a
SentencePiece `Precompiled` normalizer, and the Bert/WordPiece pipeline. It also loads tokenizer
metadata and Hugging Face chat templates from a model directory. Unsupported components and field
values fail at load time instead of being approximated.

The module is published independently as `se.alipsa:jmlx-tokenizer`; it depends on `jmlx-jinja` and
Jackson, but never on MLX or a native library. A loaded `HfTokenizer` is immutable and thread-safe.

## Encode and decode

The compatibility API returns token IDs and deliberately ignores configured truncation and padding:

```java
var tokenizer = HfTokenizer.fromFile(Path.of("tokenizer.json"));
List<Integer> ids = tokenizer.encode("Héllo, world!", true);
String text = tokenizer.decode(ids, true);
```

Use `TokenizerEncoding` for UTF-8 byte offsets into the original input and aligned masks. Explicit
options override file defaults; `encodeWithDefaults` applies supported defaults from
`tokenizer.json`:

```java
TokenizerEncoding encoding = tokenizer.encode(
    "Héllo, world!",
    new EncodingOptions(
        true,
        new Truncation(32, Direction.RIGHT),
        new Padding(32, Direction.RIGHT, padId, "<pad>", 0)));

TokenizerEncoding configured = tokenizer.encodeWithDefaults("Héllo, world!", true);
```

Only single-sequence fixed truncation/padding is supported. Pair strategies, overflow strides, and
padding multiples are rejected.

## Model directories and chat

`fromDirectory` reads `tokenizer.json`, optional `tokenizer_config.json`, root
`chat_template.jinja`, and `additional_chat_templates/*.jinja`. Root templates override a configured
`default`; a name is required when several templates exist and none is named `default`.

```java
var tokenizer = HfTokenizer.fromDirectory(modelDirectory);
String prompt = tokenizer.renderChat(
    List.of(Map.of("role", "user", "content", "Hello")),
    ChatTemplateOptions.defaults(true));
List<Integer> promptIds = tokenizer.encode(prompt, false);
```

Chat templates own their BOS/EOS markers, so rendered chat is encoded with special-token insertion
disabled. The reserved context includes `messages`, `add_generation_prompt`, and configured
`bos_token`, `eos_token`, `pad_token`, `unk_token`, `sep_token`, `cls_token`, and `mask_token`.
`extraContext` can supply values such as `tools`, but reserved-key collisions are rejected.

Templates using `strftime_now` use the current system clock and default zone on each render.
For reproducible prompts, supply a `RenderOptions` (from `se.alipsa.jmlx.jinja`):
`ChatTemplateOptions` takes one as its fourth component, so the high-level `renderChat` path
pins it too; a null component (or the three-argument constructor) keeps the system clock:

```java
var renderOptions = RenderOptions.builder()
    .clock(Clock.fixed(Instant.parse("2026-01-01T00:00:00Z"), ZoneOffset.UTC))
    .zoneId(ZoneOffset.UTC)
    .build();
var options = new ChatTemplateOptions("", true, Map.of(), renderOptions);
List<Map<String, Object>> messages = List.of(Map.of("role", "user", "content", "Hello"));
String prompt = tokenizer.renderChat(messages, options);
```

A `RenderOptions` that leaves the clock and/or zone unset gets the missing pieces topped up: a
missing zone next to a supplied clock uses the clock's own zone (a fixed clock then renders
identically on every host), and the system clock and default zone fill in only what is still
missing — so options that change only another setting keep the no-options behavior.

All `ChatTemplateRenderer.render` overloads accept the same `RenderOptions` as a final argument
for lower-level callers. The time types come from `java.time`.

## Incremental output

Create a decoder per generated request; never decode individual ByteLevel tokens independently,
because one Unicode scalar may span several tokens:

```java
IncrementalTokenDecoder decoder = tokenizer.newIncrementalDecoder(true);
String delta = decoder.append(tokenId);
String finalDelta = decoder.finish();
```

Each accepted ID yields a stable, possibly empty delta and `finish()` flushes incomplete UTF-8 with
the same replacement behavior as full decoding. Decoder state is request-local and may not be used
after `finish()`. Not every `decoder` shape in `tokenizer.json` has an incremental implementation;
for one that doesn't, `append()` always returns `""` and the full text is only available from
`finish()`. Check `decoder.streams()` before relying on per-token deltas.

## Build and verify

```sh
./tools/tokenizer-oracle/install.sh
./gradlew :jmlx-tokenizer:check verifyTokenizerOracle verifyTokenizerOracleFixtures
```

The pinned Python/Rust-backed oracle reads only committed local fixtures. The module targets Java 21
bytecode; publication and dependency checks are part of `check`.

## BERT pairs and SentencePiece charsmap normalization

`HfTokenizer.encode(text, textPair, new PairEncodingOptions(options, strategy))` supports BERT
pair templates with all three `PairTruncationStrategy` values: LONGEST_FIRST, ONLY_FIRST and
ONLY_SECOND. Special tokens reserve space before truncation; padding preserves type IDs and
special masks. Truncation failures follow the locked Rust oracle, including exact removal of an
entire selected sequence and missing second sequences only when removal is actually required.

Pair offsets are original UTF-8 byte ranges local to each input. For these BERT templates, type
IDs plus special/padding masks recover sequence identity. Arbitrary pair templates, overflow
encodings and a general sequence-ID accessor are outside scope. SentencePiece Precompiled
normalizers now decode the real Darts charsmap and compose with the other normalizers. Malformed
binary tables fail safely. The derived Flan-T5 fixtures retain the pinned real charsmap and have
source hashes, sizes and Apache-2.0 attribution under `tools/tokenizer-oracle/fixtures/`.
