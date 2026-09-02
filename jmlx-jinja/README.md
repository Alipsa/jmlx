# jmlx-jinja

This module is the migrated home of `hfjinja`, a dependency-free Java 21+
implementation of the Hugging Face chat-template Jinja subset. The standalone
`hfjinja` project has been archived; future development happens here.

It evaluates a model's `tokenizer_config.json` `chat_template` without a
JavaScript engine, with compatibility pinned to a reviewed version of
[`@huggingface/jinja`](https://github.com/huggingface/huggingface.js/tree/main/packages/jinja).

## Public API

```java
import java.util.List;
import java.util.Map;
import se.alipsa.jmlx.jinja.Template;

var template = Template.parse("""
    {% for message in messages %}
    {{ message.role }}: {{ message.content }}
    {% endfor %}
    """);

var prompt = template.render(Map.of(
    "messages", List.of(
        Map.of("role", "user", "content", "Hello"),
        Map.of("role", "assistant", "content", "Hi!"))));
```

The module is a JPMS module:

```java
module example.app {
  requires se.alipsa.jmlx.jinja;
}
```

## Verification

Unit tests run with the normal Gradle `test` task. Additional differential
verification tasks that compare the Java implementation against the pinned Node
oracle are available, but require the exact Node.js version listed in
`upstream/upstream-lock.json`:

```sh
./gradlew :jmlx-jinja:upstreamVerify        # no Node required
./gradlew :jmlx-jinja:nodeCorpusVerify      # Node required
./gradlew :jmlx-jinja:astSnapshotVerify     # Node required
./gradlew :jmlx-jinja:formatGoldenVerify    # Node required
```

See `CHANGELOG.md` for prior release history from the archived `hfjinja`
repository.
