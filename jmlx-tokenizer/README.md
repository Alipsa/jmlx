# jmlx-tokenizer

`jmlx-tokenizer` is a pure-Java 21+ byte-level BPE tokenizer for Hugging Face
`tokenizer.json` files. It supports the Qwen/Llama-style tokenizer features used
by this project, including ByteLevel pre-tokenization, BPE merges, added tokens,
TemplateProcessing special tokens, and chat-template rendering through
`jmlx-jinja`.

The module is published independently as `se.alipsa:jmlx-tokenizer`; it brings
`jmlx-jinja` and Jackson as dependencies. It does not require MLX, native
libraries, or Apple Silicon.

## Use

Load a model's `tokenizer.json`, then encode or decode text:

```java
import java.nio.file.Path;
import se.alipsa.jmlx.tokenizer.HfTokenizer;

var tokenizer = HfTokenizer.fromFile(Path.of("tokenizer.json"));
var ids = tokenizer.encode("Hello, world!", true);
var text = tokenizer.decode(ids, true);
```

For Hugging Face chat templates, parse once and reuse the parsed template in a
serving loop:

```java
import java.util.List;
import java.util.Map;
import se.alipsa.jmlx.tokenizer.ChatTemplateRenderer;

var template = ChatTemplateRenderer.parse(
    "{% for message in messages %}{{ message.content }}{% endfor %}"
);
var prompt = ChatTemplateRenderer.render(
    template,
    List.of(Map.of("role", "user", "content", "Hello")),
    true,
    "<s>",
    "</s>",
    Map.of());
```

`HfTokenizer` is safe to share across threads after loading. The loader rejects
unsupported or malformed tokenizer configuration rather than silently coercing
it; use the tokenizer that was shipped with the model checkpoint.

## Build and verify

```sh
./gradlew :jmlx-tokenizer:check
./gradlew :jmlx-tokenizer:test
```

The module targets Java 21 bytecode. Its release and publication checks are
included in `check`; see the repository's `CLAUDE.md` for the independent
release procedure.
