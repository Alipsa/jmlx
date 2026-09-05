package se.alipsa.jmlx.tokenizer;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalLong;
import java.util.stream.Stream;
import se.alipsa.jmlx.jinja.Template;
import tools.jackson.core.JacksonException;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

/** Loads tokenizer metadata and chat templates colocated with tokenizer.json. */
final class TokenizerDirectoryLoader {

  private static final ObjectMapper MAPPER = new ObjectMapper();

  private TokenizerDirectoryLoader() {}

  static Bundle load(Path directory) {
    if (!Files.isDirectory(directory)) {
      throw new TokenizerException("HfTokenizer.fromDirectory: not a directory: " + directory);
    }
    Path tokenizerPath = directory.resolve("tokenizer.json");
    TokenizerDefinition definition = TokenizerJsonLoader.loadDefinition(tokenizerPath);
    TokenizerRuntime runtime = new TokenizerRuntime(definition);
    JsonNode config = readOptionalJson(directory.resolve("tokenizer_config.json"));
    Map<String, Template> templates = loadTemplates(directory, config);
    TokenizerMetadata metadata =
        metadata(config, templates.keySet().stream().sorted().toList(), runtime);
    return new Bundle(definition, metadata, templates);
  }

  private static JsonNode readOptionalJson(Path path) {
    if (!Files.isRegularFile(path)) {
      return MAPPER.createObjectNode();
    }
    try {
      return MAPPER.readTree(Files.newInputStream(path));
    } catch (IOException | JacksonException e) {
      throw new TokenizerException("TokenizerDirectoryLoader: failed to parse " + path, e);
    }
  }

  private static Map<String, Template> loadTemplates(Path directory, JsonNode config) {
    Map<String, String> sources = new LinkedHashMap<>();
    JsonNode configured = config.path("chat_template");
    if (configured.isString()) {
      sources.put("default", configured.asString());
    } else if (configured.isArray()) {
      for (JsonNode entry : configured) {
        String name = requiredText(entry, "name", "tokenizer_config.json.chat_template");
        String source =
            entry.has("template")
                ? requiredText(entry, "template", "tokenizer_config.json.chat_template")
                : requiredText(entry, "chat_template", "tokenizer_config.json.chat_template");
        if (sources.putIfAbsent(name, source) != null) {
          throw new TokenizerException(
              "TokenizerDirectoryLoader: duplicate chat template '" + name + "'");
        }
      }
    } else if (!configured.isMissingNode() && !configured.isNull()) {
      throw new TokenizerException(
          "TokenizerDirectoryLoader: tokenizer_config.json.chat_template must be text or an array");
    }
    Path rootTemplate = directory.resolve("chat_template.jinja");
    if (Files.isRegularFile(rootTemplate)) {
      sources.put("default", readTemplate(rootTemplate));
    }
    Path additional = directory.resolve("additional_chat_templates");
    if (Files.isDirectory(additional)) {
      try (Stream<Path> files = Files.list(additional)) {
        files
            .filter(Files::isRegularFile)
            .filter(path -> path.getFileName().toString().endsWith(".jinja"))
            .sorted()
            .forEach(
                path -> {
                  String filename = path.getFileName().toString();
                  String name = filename.substring(0, filename.length() - ".jinja".length());
                  if (name.isEmpty()
                      || ".".equals(name)
                      || "..".equals(name)
                      || name.contains("/")
                      || name.contains("\\")
                      || Files.isSymbolicLink(path)) {
                    throw new TokenizerException(
                        "TokenizerDirectoryLoader: invalid chat template name '" + name + "'");
                  }
                  if (sources.putIfAbsent(name, readTemplate(path)) != null) {
                    throw new TokenizerException(
                        "TokenizerDirectoryLoader: duplicate chat template '" + name + "'");
                  }
                });
      } catch (IOException e) {
        throw new TokenizerException("TokenizerDirectoryLoader: failed to list " + additional, e);
      }
    }
    Map<String, Template> result = new LinkedHashMap<>();
    sources.forEach((name, source) -> result.put(name, ChatTemplateRenderer.parse(source)));
    return Map.copyOf(result);
  }

  private static String readTemplate(Path path) {
    try {
      return Files.readString(path, StandardCharsets.UTF_8);
    } catch (IOException e) {
      throw new TokenizerException("TokenizerDirectoryLoader: failed to read " + path, e);
    }
  }

  private static TokenizerMetadata metadata(
      JsonNode config, List<String> names, TokenizerRuntime runtime) {
    Optional<String> bos = configuredToken(config, "bos_token", runtime);
    Optional<String> eos = configuredToken(config, "eos_token", runtime);
    Optional<String> pad = configuredToken(config, "pad_token", runtime);
    Optional<String> unknown = configuredToken(config, "unk_token", runtime);
    Optional<String> separator = configuredToken(config, "sep_token", runtime);
    Optional<String> classification = configuredToken(config, "cls_token", runtime);
    Optional<String> mask = configuredToken(config, "mask_token", runtime);
    OptionalLong maximum = OptionalLong.empty();
    JsonNode maximumNode = config.path("model_max_length");
    if (maximumNode.isIntegralNumber() && maximumNode.canConvertToLong()) {
      long value = maximumNode.longValue();
      if (value > 0 && value < Long.MAX_VALUE / 2) {
        maximum = OptionalLong.of(value);
      }
    }
    return new TokenizerMetadata(
        bos,
        eos,
        pad,
        unknown,
        separator,
        classification,
        mask,
        maximum,
        direction(config, "padding_side"),
        direction(config, "truncation_side"),
        names);
  }

  private static Optional<String> configuredToken(
      JsonNode config, String field, TokenizerRuntime runtime) {
    JsonNode value = config.path(field);
    if (value.isMissingNode() || value.isNull()) {
      return Optional.empty();
    }
    String token;
    if (value.isString()) {
      token = value.asString();
    } else if (value.isObject() && value.path("content").isString()) {
      token = value.path("content").asString();
    } else {
      throw new TokenizerException(
          "TokenizerDirectoryLoader: " + field + " must be text or an object with content");
    }
    if (!runtime.vocabulary().hasToken(token)) {
      throw new TokenizerException(
          "TokenizerDirectoryLoader: configured " + field + " is not in the vocabulary");
    }
    return Optional.of(token);
  }

  private static Direction direction(JsonNode config, String field) {
    String value = config.path(field).asString("right");
    return switch (value) {
      case "left" -> Direction.LEFT;
      case "right" -> Direction.RIGHT;
      default ->
          throw new TokenizerException(
              "TokenizerDirectoryLoader: " + field + " must be left or right");
    };
  }

  private static String requiredText(JsonNode node, String field, String path) {
    JsonNode value = node.path(field);
    if (!value.isString() || value.asString().isEmpty()) {
      throw new TokenizerException(
          "TokenizerDirectoryLoader: " + path + "." + field + " must be non-empty text");
    }
    return value.asString();
  }

  record Bundle(
      TokenizerDefinition definition,
      TokenizerMetadata metadata,
      Map<String, Template> templates) {}
}
