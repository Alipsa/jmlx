# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Require the complete verification suite, including parser fuzzing, before a Maven Central
  release can begin.

## [0.5.0] - 2026-08-29

### Added

- Initial release.
- Add isolated clean-checkout release verification, publication metadata checks, and release-only
  reproducible archive evidence.

- Add a dependency-free local `tokenizer_config.json` consumer example that parses and renders a
  chat template with the published JPMS module.
- Run every checked-in template-bearing differential-corpus record against hfjinja's public Java
  API as well as the pinned Node oracle.
- Extract all serializable upstream `templates.test.js` rendering and error fixtures into the
  differential corpus, with the four JavaScript-function-context rendering cases explicitly
  reported as schema exclusions.
- Represent parse-time whitespace options in corpus records and extract the upstream interpreter
  whitespace-control vectors through both runtimes.
- Require an explicit coverage or policy decision for every vendored upstream test source.
- Add pinned-oracle vectors for runtime filters and object members, including `reverse`, `bool`,
  `abs`, `keys`, `values`, and `dictsort`.

- Provide descriptive parser diagnostics for loop variables, expected tokens, and truncated
  templates, as well as consistent `<function>` rendering for Java-native callable forms.
- Publish signed primary, source, Javadoc, and POM artifacts.
- Render and coerce converted callables (`range`, `raise_exception`, `strftime_now`, and host
  functions) consistently through interpolation, filters, concatenation, joining, and errors.
- Support the pinned runtime's namespace, object-member, builtin-identity, and deferred-undefined
  behavior, including empty `first`/`last` values and undefined-backed string operations.
- Classify pinned-runtime filter, test, macro, call-block, and TypeError diagnostics consistently
  in the differential oracle.
- Preserve source locations through whitespace preprocessing (`trim_blocks`, `lstrip_blocks`,
  trailing-newline stripping, and `{% generation %}` stripping), including CRLF boundaries
  ([#27](https://github.com/Alipsa/hfjinja/issues/27)).
- Propagate macro and call-block `break`/`continue` control to enclosing loops.
- Support call-form filter argument evaluation and keyword-bag handling for `replace`, `get`, and
  `split`, including compatible error categories and diagnostics.
