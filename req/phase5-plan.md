# jmlx Phase 5 — Model Loading & HuggingFace Interop

## Status — update this section as work lands

**Branch:** this phase started on `phase5-plan`, off `main` at `f28a327` (PR #11, Phase 4 M4
`QuantizedLinear`, merged); M1 has since merged to `main` via PR #12 (`5c85f8c`) and that branch is
deleted. Phase 4 (`req/phase4-plan.md`) has every milestone M0a–M4 plus §9 documentation **Done** —
but not "fully Done": §10 (CI, self-hosted runner) is still **Not started**, per that document's own
Status table. (M0d is also `Done` only in a scoped-down sense — six generic op-body helpers
deliberately deferred past their original merge point, per that table's own M0d note; this document's
Research findings section below leans on that same precedent for M1's own C-string helper.)

| Item | Status | Commit |
|---|---|---|
| M1 — Checkpoint I/O: `MLXIO`, safetensors + GGUF (§1) | **Done** (`req/plans/phase5-m1-plan.md`'s own amendments record two runtime-discovered fixes beyond the original plan) | `5c85f8c` (PR #12) |
| M2 — Tokenizer integration (§2) | Desk-research spike done (D3 below): waiting for upstream `mlx-c` tokenizer support is ruled out, but the FFM-vs-pure-Java architecture choice itself is still open between two credible precedents -- neither prototyped yet | — |
| M3 — Reference models: `LlamaModel`, `QwenModel` (§3) | Not started | — |

## Context

`req/project-outline.md` names Phase 5 as "Load pre-trained safetensors and GGUF model files into
Java memory," with three listed deliverables: a safetensors parser, tokenizer integration, and
`LlamaModel`/`QwenModel` reference implementations demonstrating streaming text generation. Unlike
Phase 3 and Phase 4, which each built on ops the previous phase had already exposed, Phase 5 is the
first phase whose deliverables split cleanly into one item mlx-c already solves natively (checkpoint
I/O), one item mlx-c has no opinion on at all (tokenization), and one item that's pure composition
of everything already built (the reference models). That split drives this document's shape more
than any single design tension does — see Decisions below.

## Decisions taken

**D1 — Checkpoint I/O is a thin FFM facade, not a parser (supersedes `project-outline.md`'s
wording).** `req/project-outline.md`'s Phase 5 section describes "a pure Java `.safetensors` parser
converting raw memory segments directly into `MLXArray` tensors." That undersells what's already
available: mlx-c's committed, generated bindings
(`jmlx-ffi/src/main/generated/java/se/alipsa/jmlx/ffi/mlx_h.java`) expose full native load/save
support for **both** formats already —

- Safetensors: `mlx_load_safetensors`/`mlx_load_safetensors_reader` (path or arbitrary
  `mlx_io_reader`), `mlx_save_safetensors`/`mlx_save_safetensors_writer` — returning/taking an
  `mlx_map_string_to_array` (tensors) plus an `mlx_map_string_to_string` (string metadata) directly;
  `mlx_save_safetensors(const char *file, const mlx_map_string_to_array param, const
  mlx_map_string_to_string metadata)` needs no intermediate builder.
- GGUF: `mlx_load_gguf(mlx_io_gguf *gguf, const char *file, const mlx_stream s)` produces an opaque
  `mlx_io_gguf` handle read via `mlx_io_gguf_get_keys`/`get_array`/`get_metadata_array` (numeric --
  unified as `MLXArray`, consistent with mlx's own "everything is an array" convention)/
  `get_metadata_string`/`get_metadata_vector_string`. The write path is not symmetric with the read
  path: `mlx_save_gguf(const char *file, mlx_io_gguf gguf)` takes that same opaque handle, not raw
  tensor/metadata maps -- a caller must first build one via `mlx_io_gguf_new()`, then populate it
  entry-by-entry with `mlx_io_gguf_set_array`/`set_metadata_array`/`set_metadata_string`/
  `set_metadata_vector_string` (one call per tensor or metadata key), then free it with
  `mlx_io_gguf_free` after the save call, whether it succeeds or throws. `MLXIO.saveGguf`'s
  implementation is this builder loop, not a single pass-through call the way `saveSafetensors` is.
- Both load entry points take a trailing `const mlx_stream s`; neither save entry point does.
  `MLXIO`'s load methods pass `NativeOps.DEFAULT_STREAM` -- the field `MLXQuant`'s own hand-rolled
  bodies already reuse rather than re-resolving the stream per call, and the shape to match here
  since `MLXIO` calls `mlx_h` directly the same way. (`MLXArray` is the one class that calls
  `MLX.defaultStream()` directly, at its own three call sites -- not the facade-wide convention;
  `CLAUDE.md`'s Architecture section documents `defaultDevice()`/`defaultStream()` as resolved once
  and cached for the process lifetime.) No stream argument to plumb through on the save side.

So M1 is a facade over already-battle-tested native code — the same "native does the heavy lifting,
Java exposes it safely" shape every prior phase has taken (jextract bindings, `MLXOps`/`MLXShape`
wrapping mlx-c ops, `MLXGrad` wrapping `mlx_value_and_grad`). Writing a hand-rolled binary parser
would duplicate code this project already links against for no benefit. This reconciliation follows
the same pattern `req/project-outline.md`'s own Phase 3/4 sections already use to correct their own
stale wording in place rather than pretend the original phrasing was right.

**D2 — The new facade lives in `se.alipsa.jmlx.core` as `MLXIO`, not a new package or module.**
`MLXArray`'s only constructor, `MLXArray(MLXScope scope, MemorySegment handle)`, is
package-private, and `NativeOps` (`checked`, `scopeOf`, `cstr`) is a package-private `final` class —
both scoped to `se.alipsa.jmlx.core`. A class outside that package could not wrap a raw native
`mlx_array` handle into an `MLXArray` at all without opening new cross-package access on a type the
codebase has deliberately kept closed (`MLXArray`/`MLXScope` are both `public final`, no
subclassing). So `MLXIO` joins `MLXQuant`/`MLXRandom`/`MLXGrad` as another native-facing facade
class inside `core`, not a new `se.alipsa.jmlx.io` package — see §1 below for the exact API shape,
and `req/plans/phase5-m1-plan.md` for the full task-by-task implementation.

**D2a — `MLXIO` is the first facade in this codebase whose native surface returns handles that are not
`mlx_array` at all, and `MLXScope` cannot own any of them.** `mlx_io_gguf`, `mlx_map_string_to_array`,
`mlx_map_string_to_string`, `mlx_vector_string`, and `mlx_string` each come back from these load
calls with their own deallocator (`mlx_io_gguf_free`, `mlx_map_string_to_array_free`,
`mlx_map_string_to_string_free`, `mlx_vector_string_free`, `mlx_string_free`, plus the two
`*_iterator_free` variants for the maps) -- never `mlx_array_free`. `MLXScope.allocate`'s own javadoc
is explicit that it "assumes every segment handed out represents an `mlx_array` struct and will
later be passed to `mlx_array_free`. Passing a scope as an allocator for anything else corrupts the
handle list." So these five types must be allocated and freed through a confined `Arena`/direct
native calls, exactly like `MLXQuant`/`MLXGrad`'s own hand-rolled bodies already do for by-value
structs -- never through an `MLXScope`. The *individual* `mlx_array` tensor handles extracted out of
a loaded `mlx_map_string_to_array` are the opposite case: those genuinely are `mlx_array`s and do
need `MLXScope` registration, but there is no existing `MLXArray` operand to infer a target scope
from the way `scopeOf` does elsewhere in this codebase -- so `loadSafetensors`/`loadGguf` must each
take an explicit `MLXScope` parameter, the same shape `MLX.array(scope, ...)` already uses for
exactly this reason (`CLAUDE.md`'s memory-model rule). This is the most likely source of a real leak
or handle-list corruption in M1 if missed, and belongs in `req/plans/phase5-m1-plan.md`'s own Global
Constraints once that document exists.

**D2b — `mlx_io_gguf_new` is the specific call site this hazard becomes checkable at.** It binds as
`MemorySegment mlx_io_gguf_new(SegmentAllocator allocator)` -- jextract's generated signature takes
any `SegmentAllocator`, and `MLXScope` implements that interface (`CLAUDE.md`'s memory model:
"`MLXScope` ... implements `SegmentAllocator`"), so `saveGguf` passing a scope here compiles cleanly
and silently corrupts that scope's handle list -- exactly the failure mode `MLXScope.allocate`'s
javadoc warns about, with no compiler error to catch it. `MLX.newVectorArray` already has the
precedent to mirror: it declares its own allocator parameter as a plain `Arena`, not the broader
`SegmentAllocator` jextract would otherwise accept, specifically so this mistake is a compile error
rather than a runtime one -- "`allocator` MUST be a confined `Arena` (never an `MLXScope`)". Whatever
helper `saveGguf` builds its `mlx_io_gguf` through should take the same narrowed `Arena` parameter
type, not `SegmentAllocator`.

**D2c — Every one of these five types' free functions -- and both `*_iterator_free` variants --
returns a status `int`, same as `mlx_array_free(mlx_array arr)` itself** (the genuinely statusless
entry points are the `mlx_array` constructors, `mlx_array_new`/`mlx_array_new_data`, which "have no
status return at all and only signal failure via the error handler plus a null `ctx`" -- `CLAUDE.md`'s
error-handling paragraph, not its Memory model paragraph, notes this). Routing one of these seven
frees through `NativeOps.checked` from inside a `finally` block risks losing a real in-flight failure
during exception unwinding: `checked` throws on a non-zero status, and a fresh exception thrown from
a `finally` replaces whatever exception was already propagating instead of chaining it.

This codebase already has a precedent to follow rather than decide fresh: `MLXScope`'s own teardown
(`Holder.closeAll`'s reverse-order loop, and `freeOne`) calls `mlx_h.mlx_array_free` bare, outside
`NativeOps.checked`, discarding its status -- and does so unconditionally, not only when unwinding an
exception: neither call site sits inside a `try`/`catch` reacting to one, so the same discard applies
on an ordinary clean `close()` too. That covers more than the unwinding hazard above explains: a
cleanup failure on an otherwise-successful exit vanishes silently as well, not just one that would
otherwise mask a real error -- consistent with `MLXScope` treating a failed free as unactionable
information either way, not merely as the lesser of two evils during unwinding. Whether that discard
was a deliberate policy or merely incidental is not established by the code alone, so this is cited as
precedent to follow, not as a documented decision -- but absent a reason to diverge, `MLXIO`'s own
cleanup should swallow these seven frees' statuses the same way, on every exit path, not attach them
via `Throwable.addSuppressed`.

**D3 — Tokenizer integration (M2) is deliberately left unplanned here.** Unlike checkpoint I/O,
there is no existing binding to lean on: mlx-c has no tokenizer API at all. Whether the right move
is FFM-binding Hugging Face's `tokenizers` Rust crate's C API (mirroring how `mlx-c` itself was
bootstrapped from a pinned wheel — see `req/initial-plan.md`) or a pure-Java BPE/SentencePiece
implementation is a genuinely open architectural question this document does not resolve. M2 needs
its own research spike (probing whether `tokenizers` exposes a usable C ABI, checking license
compatibility, and prototyping load-time cost) before a plan for it can be written the way M1's plan
already can be. Do not start M2 implementation from this document — write `req/plans/phase5-m2-plan.md`
only after that spike.

**D3 amendment (M2 desk-research spike, first half): there is no official C API, but a viable
third-party one exists, and prior art independently converged on the same architecture jmlx already
uses for MLX itself.** Findings, each confirmed against a primary source rather than assumed:

- **No official C/C++ binding.** `huggingface/tokenizers` issue #185 (opened 2020, asking exactly
  this question) was closed as "not planned" via the stale-bot, with no maintainer commitment ever
  made. The crate remains Rust-native with official bindings only for Python (PyO3) and Node.js.
- **`mlx`/`mlx-c` adding tokenizer support upstream is very unlikely, and waiting for it is not a
  viable deferral -- confirmed against the project's own stated scope and its own precedent across
  its other two official language bindings, not assumed.** `ml-explore/mlx-c`'s own README states its
  scope directly: "a C API for MLX... expands MLX to the C language... can be used standalone or as
  a bridge to bind other languages to MLX" -- a low-level array/tensor API mirroring MLX's own core
  scope (an array framework, not a text-processing one), with no tokenizer functionality anywhere in
  it. More telling: Apple's own two higher-level official bindings already faced this exact problem
  and both solved it the same way jmlx is now considering, not by asking MLX/mlx-c for it --
  `mlx-lm` (Python) depends directly on Hugging Face's own `transformers`/`tokenizers` packages, and
  `mlx-swift-lm` (Swift) "integrates with a variety of tokenizer and downloader packages through
  protocol conformance," naming `huggingface/swift-transformers`'s `Tokenizers` product specifically
  (see below) as its default. All recent tokenizer-related engineering activity in the `ml-explore`
  org (streaming-detokenizer fixes, chat-template handling, think-token NoneType fixes) is happening
  in `mlx-lm` at the Python application layer, not in `mlx`/`mlx-c` core -- no roadmap item, issue, or
  discussion found anywhere suggesting tokenizer support is planned for MLX's core C API. This is the
  same architectural choice jmlx now faces (FFM-bind an external tokenizer implementation vs. write
  one), not a problem MLX itself is going to solve on jmlx's behalf.
- **A maintained, permissively-licensed, genuinely plain-C shim exists: `mlc-ai/tokenizers-cpp`.**
  Apache-2.0 (compatible with this repo's MIT `LICENSE` -- permissive, no copyleft obligation
  conflict), 503 stars, actively maintained (pushed 2026-05-20, per `gh api
  repos/mlc-ai/tokenizers-cpp` at spike time), built in part for and used by MLC LLM. Its
  `include/tokenizers_c.h` is a genuine `extern "C"` API -- opaque `void*` handle, out-param structs,
  no C++ name mangling, no JNI ceremony -- the same shape mlx-c itself presents and that jextract/FFM
  already binds cleanly in this codebase:
  ```c
  typedef void* TokenizerHandle;
  typedef struct { int* token_ids; size_t len; } TokenizerEncodeResult;
  TokenizerHandle tokenizers_new_from_str(const char* json, size_t len);
  TokenizerHandle byte_level_bpe_tokenizers_new_from_str(const char* vocab, size_t vocab_len,
      const char* merges, size_t merges_len, const char* added_tokens, size_t added_tokens_len);
  void tokenizers_encode(TokenizerHandle, const char* data, size_t len, int add_special_token,
      TokenizerEncodeResult* result);
  void tokenizers_decode(TokenizerHandle, const uint32_t* data, size_t len, int skip_special_token);
  void tokenizers_get_decode_str(TokenizerHandle, const char** data, size_t* len);
  void tokenizers_free(TokenizerHandle);
  ```
  (full list also has `encode_batch`/`free_encode_results`/`get_vocab_size`/`id_to_token`/
  `token_to_id`). This plain-C layer covers HF `tokenizer.json` (`tokenizers_new_from_str`) and raw
  byte-level BPE vocab+merges directly; SentencePiece and RWKV-World support exist only at the C++
  layer above it (`include/tokenizers_cpp.h`'s `Tokenizer::FromBlobSentencePiece`/
  `FromBlobRWKVWorld`), which this project would not need to bind at all if only HF-JSON-format
  tokenizers are in scope for M3's reference models (Llama/Qwen both ship `tokenizer.json`).
- **Prior art exists on both sides of this choice, from credible sources -- not a settled question.**
  DJL (Deep Java Library, the most prominent Java ML library with an equivalent problem) does not
  reimplement HF tokenizers in pure Java -- `extensions/tokenizers` wraps the same upstream
  `tokenizers` Rust crate via its own hand-written native bridge (JNI in DJL's case, since that
  predates or simply didn't adopt FFM; jmlx would use FFM instead, which needs no JNI ceremony at
  all -- one more reason to prefer `tokenizers-cpp`'s plain-`extern "C"` shape over reusing DJL's
  crate directly, whose native functions are JNI-shaped `Java_...` symbols, not callable via FFM).

  **Amendment: this bullet originally concluded DJL's choice was "independent confirmation that
  FFM/JNI-bind beats reimplement" -- that overstated it.** Hugging Face itself did the opposite for
  Swift: `huggingface/swift-transformers` (Apache-2.0, actively maintained -- pushed 2026-07-28,
  confirmed via `gh api repos/huggingface/swift-transformers`) is a **from-scratch, pure-Swift**
  reimplementation of the tokenizer pipeline, not a Rust FFI wrapper -- confirmed directly from its
  `Sources/Tokenizers/` file list: `BPETokenizer.swift`, `UnigramTokenizer.swift`,
  `BertTokenizer.swift`, `Normalizer.swift`, `PreTokenizer.swift`, `PostProcessor.swift`,
  `TokenLattice.swift`, `Trie.swift`, with its own `TokenizersTests` suite. `mlx-swift-lm` (see the
  bullet above) depends on exactly this package for its own tokenizer support. So the two most
  directly comparable precedents split: DJL (Java, wraps Rust via JNI) vs. Hugging Face itself
  (Swift, pure reimplementation, no Rust at all) -- both maintained, both production-used, for the
  identical problem. What this actually changes for M2: "pure Java" is no longer the vaguely-scoped
  fallback D3 originally framed it as -- there is now a concrete, battle-tested reference
  implementation, maintained by the same organization that owns the `tokenizer.json` format, to port
  from rather than reverse-engineer from spec documents alone. This does not resolve the choice; it
  makes both sides of it equally concrete.
- **Port-size estimate for the pure-Java path, measured directly rather than guessed.**
  `swift-transformers/Sources/Tokenizers/` totals **~3,850 lines** across 12 files
  (`Tokenizer.swift` 1041, `Normalizer.swift` 376, `BertTokenizer.swift` 358, `PreTokenizer.swift`
  366, `BPETokenizer.swift` 339, `ByteEncoder.swift` 286, `Decoder.swift` 283,
  `UnigramTokenizer.swift` 167, `TokenLattice.swift`/`String+PreTokenization.swift` 158 each,
  `PostProcessor.swift` 214, `Trie.swift` 101); its own `TokenizersTests` totals **~2,290 lines**
  (excluding `ChatTemplateTests.swift` and `YYJSONParserTests.swift`, neither of which tests the
  tokenizer pipeline itself), reusable as a porting/validation reference even though XCTest
  assertions don't translate mechanically to JUnit. Of `Tokenizer.swift`'s 1041 lines, only
  `AutoTokenizer` (lines 846-948, ~100 lines, 11 of the file's `Hub.`-prefixed references) is
  Hub-download convenience code a Java port would drop entirely -- `MLXIO`'s own file-loading
  precedent already covers "load from a local string/path," so this isn't lost functionality, just
  code that would never get ported. **The real complication: `applyChatTemplate` (part of the core
  `Tokenizer` protocol, not an add-on) depends on `import Jinja` -- `huggingface/swift-jinja`, a full
  Jinja2-template-engine port, confirmed via `Package.swift`'s `Tokenizers` target listing it as a
  direct dependency, not merely a test dependency.** `swift-jinja/Sources/Jinja/` totals **~6,660
  lines** across 13 files (`Filters.swift` 2178, `Interpreter.swift` 903, `Value.swift` 891,
  `Parser.swift` 777, `Lexer.swift` 484, plus 8 smaller files) -- larger than the tokenizer pipeline
  itself. So the honest total for a pure-Java port with full chat-template fidelity is **~10,500
  lines** (~3,850 tokenizer + ~6,660 Jinja), not ~3,850 -- a materially different scoping question
  than "port the tokenizer." Llama/Qwen instruct variants (M3's actual targets) need *some* chat
  formatting to be usable at all, but not necessarily *general* Jinja2 evaluation: since M3 targets
  a small, known set of model families rather than arbitrary Hub models, hand-formatting each
  target model's own specific chat template as a small dedicated Java method (reading the same
  `tokenizer_config.json`'s `chat_template` field just enough to detect which known template it is,
  or simply hardcoding the known Llama-3/Qwen2 chat-format strings) would sidestep the ~6,660-line
  Jinja port entirely -- at the cost of not generalizing to arbitrary future Hub models the way a
  real Jinja engine would, mirroring this codebase's own established "ship exactly what's needed,
  not the general case" convention (e.g. `req/plans/phase4-m4-plan.md`'s own deferrals). This
  scoping choice is not made here -- it depends on M3's own requirements, which this document
  explicitly declines to plan (D4) -- but it changes the comparison materially: ~3,850 lines (no
  chat templates) or ~10,500 lines (full Jinja) against the FFM path's toolchain cost, not a single
  fixed number either way.
- **No official Java equivalent of `swift-jinja` exists; the three real options for the chat-template
  half specifically each trade off differently, and one of them meaningfully revises the ~10,500-line
  estimate above.** Hugging Face maintains minimalistic, purpose-built Jinja implementations for JS
  (`@huggingface/jinja`, in `huggingface/huggingface.js`), Swift (`swift-jinja`), and (community,
  JS-derived) PHP -- but not Java.
  1. **Jinjava (HubSpot)** is the only existing Java Jinja-syntax engine of note -- Apache-2.0, ~780
     stars, actively maintained (pushed 2026-08-03), used in production to render HubSpot's own CMS.
     But it was built for "the subset of jinja in use in HubSpot content," not for HF chat templates
     specifically -- unlike `swift-jinja`/`@huggingface/jinja`, which are purpose-built for exactly
     this (custom globals/filters chat templates lean on, like `raise_exception`/`tojson`), Jinjava's
     compatibility with real Llama/Qwen chat templates is unverified, not merely unproven -- would
     need checking against actual template strings before trusting it. Maven Repository also flags
     open CVEs against it (e.g. CVE-2026-25526) -- likely server-side-template-injection-from-
     untrusted-input concerns relevant to HubSpot's threat model (rendering user-supplied templates
     in a web CMS), not necessarily jmlx's (rendering a trusted `chat_template` string that ships
     inside a model file the user already chose to load) -- but this distinction is asserted here,
     not verified against the actual CVE text, and should be before depending on Jinjava for real.
  2. **`@huggingface/jinja` (the JS implementation) is a smaller, more current porting source than
     `swift-jinja`, and already ships exact goldens for M3's own target models.** Measured directly:
     `packages/jinja/src/` in `huggingface/huggingface.js` totals **~3,860 lines** across 7 files
     (`runtime.ts` 1929, `parser.ts` 685, `lexer.ts` 413, `format.ts` 327, `ast.ts` 320, `utils.ts`
     125, `index.ts` 57) -- MIT-licensed, pushed within the last day at spike time (`gh api
     repos/huggingface/huggingface.js` shows `pushed_at: "2026-08-18T03:21..."`), roughly the tokenizer
     pipeline's own size and **~42% smaller than `swift-jinja`'s ~6,660 lines** for what is presumably
     equivalent chat-template-rendering coverage (HF maintains both from the same org for the same
     narrow purpose). Its own `test/e2e.test.js` contains verbatim `chat_template` strings and
     expected-output goldens for, among others, `meta-llama/Llama-3.1-8B-Instruct`,
     `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen3-0.6B`, and `Qwen/Qwen1.5-72B-Chat` -- M3's own named
     targets, not generic examples -- which would port directly into JUnit goldens regardless of
     which Jinja path is chosen. **This revises the earlier ~10,500-line full-fidelity estimate
     downward to ~7,700 lines** (~3,850 tokenizer + ~3,860 Jinja) if porting from this source instead
     of `swift-jinja`.
  3. **A third option avoids porting Jinja at all: embed GraalJS and run `@huggingface/jinja`'s actual
     JS directly from the JVM.** Confirmed GraalJS runs on a stock OpenJDK, no GraalVM installation or
     native toolchain required, via Maven Central artifacts (`org.graalvm.polyglot:polyglot` +
     `org.graalvm.polyglot:js-community` specifically -- the plain `js` artifact defaults to Oracle's
     more restrictive GFTC license, not a permissive OSS one). This reuses HF's actual, tested,
     model-specific-validated implementation with zero hand-porting or compatibility-verification
     risk, at the cost of a genuinely heavyweight dependency (a JS engine, non-trivial jar size) for
     rendering one short template string per model load -- a real tension with this project's own
     "pure, idiomatic Java... zero-copy FFM" framing in `CLAUDE.md`, which none of the tokenizer-side
     options raise this sharply. Not evaluated further here: actual jar-size/startup-cost numbers, or
     whether GraalJS's polyglot API composes cleanly with `MLXScope`'s confinement/lifecycle model.

  **Amendment: "no official Java equivalent of `swift-jinja` exists" is no longer true.** `hfjinja`
  (`github.com/Alipsa/hfjinja`, MIT, same org as jmlx) has since been released -- a dependency-free
  Java 21+ port of `@huggingface/jinja` specifically (option 2 above), pinned to upstream `0.5.9`,
  explicitly scoped as "the Hugging Face chat-template Jinja subset," not a general-purpose or
  Python-compatible Jinja2 engine -- the same narrow scope this document already argued for over
  Jinjava. This effectively supersedes options 1-3 for M2/M3 on the porting/build-shape question: it
  needs no hand-port of the ~3,860 remaining Jinja lines (already done, against the smaller/more
  current source this document already preferred over `swift-jinja`), no Jinjava
  compatibility-verification risk, and no GraalJS dependency-weight cost. **It does not, however,
  supersede the adoption cost: Maven Central's search API returns `numFound: 0` for `hfjinja` --
  confirmed directly, not left as an open question -- so it is not currently resolvable as a normal
  Gradle dependency.** Adopting it today means JitPack, a source/composite build, or publishing it
  to Maven Central first; that real cost belongs in the comparison, not folded into "supersedes
  everything." Still unverified: whether its byte-exact-vs.-Node-output differential corpus actually
  covers M3's target models (Llama/Qwen/Mistral) specifically -- worth checking before depending on
  it for real, but the "port or embed a JS engine" trade-off this
  section spent most of its length on is likely moot now.
- **Build shape: `tokenizers-c` is a Rust `staticlib`, not a `cdylib` -- jmlx cannot load it directly
  the way `NativeLoader` loads `libmlxc.dylib`.** `tokenizers-cpp/rust/Cargo.toml` declares `crate-type
  = ["staticlib"]`; producing something `System.load()`-able would need either (a) a small additional
  link step producing a `.dylib` that statically links `libtokenizers_c.a` (mirroring how
  `bootstrap-native.sh` already builds `libmlxc.dylib` from source against a pinned wheel), or (b) a
  jmlx-owned fork of `rust/Cargo.toml` + `rust/src/lib.rs` with `crate-type = ["cdylib"]` instead --
  the latter is simpler since it also sidesteps needing the C++/CMake/submodule machinery
  (`sentencepiece`, `msgpack`) that only the C++ layer requires, if HF-JSON-only scope (see above)
  is accepted for M2. Either way, this fork inherits an upstream pin the same way `bootstrap-native.sh`
  pins `MLX_C_COMMIT`: `rust/Cargo.toml` pins `tokenizers = "0.21.2"` (see the `onig`/`fancy-regex`
  bullet below), and a jmlx-owned fork would own re-pinning it on the same
  `scripts/checkDependencies.zsh`/`scripts/updateMlx.zsh`-style discipline this repo already has for
  mlx-c, not a new maintenance burden invented for M2.
- **A Rust toolchain is unavoidable for this whole path -- there is no way to use `tokenizers-cpp`
  (or any fork of it) without one.** The actual tokenizer logic is the upstream `tokenizers` Rust
  crate itself; `tokenizers-cpp`'s C and C++ layers are thin wrapper headers that call into a Rust
  `staticlib` compiled by `cargo`/`rustc` -- they do not replace or reimplement it. Confirmed two
  ways: `tokenizers-cpp/CMakeLists.txt` explicitly branches on `CMAKE_SYSTEM_NAME STREQUAL "Darwin"`
  + `CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64"` to set `TOKENIZERS_CPP_CARGO_TARGET
  aarch64-apple-darwin` (this repo's own target triple) and shells out to `cargo build` via
  `CARGO_EXTRA_ENVS`; and neither of its two GitHub releases (`v0.1.0`, `v0.1.1`) publishes any
  binary assets at all (source tags only), so there is no prebuilt `.a`/`.dylib` to download the way
  `bootstrap-native.sh` already downloads a prebuilt `mlx-metal` wheel for MLX itself. Practically:
  `aarch64-apple-darwin` being an explicit, named target is a genuinely good sign for this repo's
  specific platform (unlike the ambiguity noted below about `onig`'s clang compatibility), but Rust
  becomes a wherever-this-builds toolchain dependency (dev machines and CI both) that this project
  does not otherwise have, unlike mlx-c's C/C++ toolchain requirement, which Xcode Command Line
  Tools already satisfies on any macOS machine capable of running this project at all. The only way
  to avoid a Rust toolchain entirely is the pure-Java alternative D3 already named -- not something
  this bullet resolves, only sharpens the actual trade-off being decided.
- **One real risk, not previously visible from the plan text alone: Rust-side panics cross the FFI
  boundary as failures with no recoverable status.** `rust/src/lib.rs`'s wrapper calls `.unwrap()`
  on `Tokenizer::from_str`/`encode`/`decode` -- a malformed `tokenizer.json` or a decode error panics
  inside Rust rather than returning a checkable error code. **Correction: a Rust panic unwinding
  across an `extern "C"` boundary without `catch_unwind` is not undefined behavior -- per the
  Rustonomicon's FFI chapter, "panic will cause the process to safely abort" (UB is the reverse
  direction: a foreign exception entering Rust). Both this and mlx-c's own error convention (`printf`
  + `exit(-1)`) are immediate, defined process death; the genuine distinction is narrower than "UB vs.
  catchable" -- mlx-c's exit path is interceptable, and `NativeLoader`'s custom handler already
  intercepts it, while a Rust abort offers no hook at all.** Whatever plan follows this spike needs
  to either wrap every entry point in `catch_unwind` in a jmlx-owned fork of the Rust glue, or
  explicitly accept malformed-tokenizer-file input as an unrecoverable-crash case (unlike every other
  failure path in this codebase, which surfaces as a catchable `MLXException`).
- **A second FFI hazard in the same header, independent of the panic risk above: a signedness
  mismatch between encode and decode.** `TokenizerEncodeResult` declares `int* token_ids`, but
  `tokenizers_decode` takes `const uint32_t* data` -- confirmed against the upstream header, not just
  this document's own transcription of it. An FFM binding would read `token_ids` as
  `ValueLayout.JAVA_INT` (signed) and need to pass it back as unsigned for decode; not exploitable
  with current vocab sizes (no token ID reaches `Integer.MAX_VALUE`), but it's exactly the kind of
  layout mismatch this codebase's own binding conventions otherwise pin down explicitly, and worth
  naming alongside the panic risk rather than leaving implicit in the quoted header above.
- **A regex backend is unavoidable, but `onig` specifically is not: exactly one of
  `onig`/`fancy-regex` must be enabled, and `fancy-regex` is a real, pure-Rust alternative that
  avoids the Oniguruma build fragility.** `tokenizers-cpp/rust/Cargo.toml` declares `tokenizers = {
  version = "0.21.2", default-features = false, features = ["onig"] }` -- `onig` is already an
  explicit, deliberate opt-in feature, not something inherited from upstream's own `default =
  ["progressbar", "onig", "esaxx_fast"]`, so passing `default-features = false` (already set) does
  nothing further; upstream's own `tokenizers/src/utils/mod.rs` requires one of the two
  (`#[cfg(not(any(feature = "onig", feature = "fancy-regex")))] compile_error!(...)`), so "skip the
  regex engine entirely" -- this bullet's original framing -- was never on the table. What
  Llama/Qwen force is the capability, not the crate: both use exactly the GPT-2-style regex split
  `onig`/`fancy-regex` exist for, confirmed against `llama.cpp/src/llama-vocab.cpp` -- QWEN2's
  pretokenizer regex is `[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*` plus `\s+(?!\S)`,
  LLAMA3's differs mainly in the numeric portion (`\p{N}{1,3}`) -- both rely on Unicode property
  classes and a negative lookahead, which Rust's plain `regex` crate cannot express. **But
  `fancy-regex` is a supported, reachable substitute, not gated behind `unstable_wasm` as an earlier
  pass here claimed:** upstream's `Cargo.toml` declares `fancy-regex = { version = "0.14", optional =
  true }` as its own implicitly-named feature, and `unstable_wasm = ["fancy-regex",
  "getrandom/wasm_js"]` merely bundles it with a wasm `getrandom` backend for a different (wasm)
  target -- it does not gate `fancy-regex` itself. `default-features = false, features =
  ["fancy-regex"]` is a valid non-wasm configuration on `aarch64-apple-darwin`, and since
  `fancy-regex` is pure Rust, choosing it would eliminate entirely the fragility `onig_sys` carries:
  it vendors and compiles a bundled Oniguruma C source when no system library is found via
  `pkg-config`, known to hit compiler compatibility issues on newer GCC and unconfirmed either way
  against Apple's clang on this repo's actual macOS 26/Apple Silicon target, since that combination
  has not yet been built here. This is the cheapest available reduction in the FFM path's build
  cost. Not verified here: whether `fancy-regex` is byte-identical to `onig` on Llama-3/Qwen2's
  specific patterns -- both engines support the lookaround and `\p{...}` classes those patterns use,
  but split-behavior equivalence for these exact regexes is unconfirmed, and belongs in the
  load-time-cost prototype this document already defers, not asserted here.

**Still open, deliberately not resolved by this desk-research pass:** actually building a minimal
`cdylib` from a jmlx-owned fork of `tokenizers-cpp/rust` (or from scratch against the plain
`tokenizers` crate) and measuring real load-time cost for a representative `tokenizer.json`, per D3's
original "prototyping load-time cost" requirement. This machine has no Rust toolchain installed
(`cargo`/`rustc` both absent); doing so is an environment change worth confirming with a human before
taking, not something to do unilaterally mid-spike. `req/plans/phase5-m2-plan.md` should not be
written until that prototyping step also lands, nor until the FFM-vs-pure-Java architecture choice
itself is actually made -- the desk research above narrows both (see D3's amendment for the pure-Java
side, and the onig and panic bullets above for the FFM side's real cost) but resolves neither.

**D4 — Reference models (M3) are pure composition, deferred until M1 and M2 both land.**
`LlamaModel`/`QwenModel` need nothing new at the tensor/module level: `se.alipsa.jmlx.nn` already
has `Linear`, `QuantizedLinear`, `RMSNorm`, `MultiHeadAttention`, `KVCache`, and RoPE (via
`MLXFast.rope`, per `req/project-outline.md`'s own Phase 4 reconciliation note, not
`req/phase4-plan.md` §7 -- §7 is only RoPE's signature spec, the reconciliation of RoPE shipping as
a `core` op rather than an `nn` module lives in the outline). What M3 adds is a checkpoint →
`Module` tree mapping (naming convention translation from HF/GGUF tensor names to this project's
`Module` field names) and an autoregressive generation loop built on `KVCache`. Both depend on M1's
`MLXIO` output shape and M2's tokenizer, so M3 cannot be scoped precisely until those land — this
document names it only to keep the phase's end state visible, not to plan its tasks.

## Research findings

**GGUF metadata is not uniformly stringly-typed, unlike safetensors.** Safetensors' native binding
returns exactly two maps: tensors (`mlx_map_string_to_array`) and metadata
(`mlx_map_string_to_string` — always strings, matching the safetensors spec's own JSON-header
design). GGUF's `mlx_io_gguf` instead exposes three separate metadata accessors —
`get_metadata_array` (numeric scalars/arrays, returned as `MLXArray`), `get_metadata_string`, and
`get_metadata_vector_string` (string arrays, e.g. tokenizer vocab-adjacent fields) — because GGUF's
own on-disk metadata format is a small typed-value union, not JSON. `MLXIO`'s GGUF result type must
mirror that three-way split rather than force everything through one `Map<String, String>` the way
safetensors' own metadata naturally fits.

**No C-string-to-`Java String` helper exists anywhere in `jmlx-core` today** (confirmed by
repository-wide grep for `getString`/`mlx_string`/`mlx_map_string`). `NativeOps.cstr` only goes
Java → C. Reading back C strings needs two distinct lifetimes, not one "mirror image of `cstr`"
helper:

- Borrowed `const char*`, needing only a copy into a Java `String`, no free call: both
  `mlx_map_string_to_array_iterator_next(const char **key, mlx_array *value, ...)`'s `key` (borrowed
  from the map's own storage -- `mlx_string_free` does not apply, and freeing it risks a double-free
  once the map itself is freed) and, identically, `mlx_map_string_to_string`'s own pair
  (`mlx_map_string_to_string_get(const char **value, ...)` and
  `mlx_map_string_to_string_iterator_next(const char **key, const char **value, ...)`, both
  `safetensors`' metadata path depends on) -- as well as `mlx_string_data(mlx_string str)`, whose own
  header comment states outright: "Returns a pointer to the string contents. The pointer is valid
  for the life duration of the string."
- Owned `mlx_string`, needing a copy followed by `mlx_string_free`: only
  `mlx_io_gguf_get_metadata_string(mlx_string *str, mlx_io_gguf io, const char *key)`'s out-param
  fits this shape among the accessors M1 needs.

**`mlx_vector_string_get(char **res, const mlx_vector_string vec, size_t idx)` is resolved as
borrowed, same bucket as the three accessors above -- confirmed from mlx-c's actual implementation,
not just its header.** The header carries no ownership comment and the non-`const` `char **` return
could plausibly have hinted at caller-owned, so this was flagged as a genuine blocker on `loadGguf`
(GGUF's sole key-enumeration path: `mlx_io_gguf_get_keys` returns an `mlx_vector_string`, and every
key must be read back out of it through this call -- there is no other route). Reading
`native/scratch/mlx-c/mlx/c/vector.cpp` (the pinned checkout at `fba4470b89073180056c9ea46c443051375f7399`,
matching `bootstrap-native.sh`'s own `MLX_C_COMMIT`) settles it directly: `*res =
mlx_vector_string_get_(vec).at(index).data();` -- a pointer into the vector's own internal
`std::string` storage, not a fresh allocation. No free call applies, and the same C-string-copy
helper built for the other three borrowed accessors covers this one too. This was resolved by
reading the real `.cpp` source rather than by writing a runtime probe -- stronger evidence than a
probe result, since it is the actual logic that will run, not an inferred behavior from one sample
call.

**GGUF/safetensors map iteration has a three-way status convention, not the usual 0-or-throw
`NativeOps.checked` shape.** `mlx_map_string_to_array_iterator_next`'s own implementation
(`native/scratch/mlx-c/mlx/c/map.cpp`) returns `0` on a successful read, `1` on a genuine error (goes
through `mlx_error`, same as everywhere else), and `2` when the iterator has already reached the
map's `end()` -- a normal loop-termination signal, not a failure. Routing this call through
`NativeOps.checked` unmodified would misread every ordinary end-of-map as an `MLXException`; the
helper that drives `loadSafetensors`'s tensor/metadata read loop needs its own three-way switch
instead of reusing `checked` as-is. Not yet confirmed whether `mlx_map_string_to_string_iterator_next`
shares the identical `0`/`1`/`2` convention -- likely, given the parallel design, but `req/plans/phase5-m1-plan.md`
should verify against its own `map.cpp` body before assuming it rather than citing this paragraph as
proof for both map types.

Per the M0d precedent in `req/phase4-plan.md` (add a generic helper only in the task that first needs
it, wired to a real consumer, never speculatively), the resulting helper(s) are added to `NativeOps`
in M1's own first task, not built ahead of time here.

## Out of scope for Phase 5 as currently planned

- Any tokenizer implementation or FFM binding (M2 is a research spike, not an implementation, until
  its own plan exists).
- `LlamaModel`/`QwenModel` and any other reference model (M3, deferred — see D4).
- Streaming/incremental text generation UX (console printing, sampling strategies beyond greedy
  argmax) — not named in `project-outline.md`'s Phase 5 deliverables and not needed to prove
  checkpoint loading works.
- Writing (`save`) support beyond exactly what M1's own round-trip tests need as a
  fixture-generation path. This is not conditional on whether the testing plan happens to exercise
  it: Testing approach (below) already commits to round-trip as the strategy for *both* formats, so
  `MLXIO.saveSafetensors` and `MLXIO.saveGguf` are both in scope for M1 -- see §1 above and D1's
  write-path note for `saveGguf`'s extra plumbing. What stays out of scope is any save
  capability beyond that fixture-generation need — this project has no fine-tuning/checkpoint-export
  use case yet to justify more (e.g. preserving a source checkpoint's exact tensor ordering or
  metadata round-trip fidelity beyond what the M1 tests themselves assert).

## Work breakdown

### 1. Checkpoint I/O — `MLXIO`, safetensors + GGUF (M1) — **DONE** (PR #12, `5c85f8c`)

Full task-by-task plan lives in `req/plans/phase5-m1-plan.md`, whose own amendments record two
runtime-discovered fixes beyond what's summarized below (a CPU-stream requirement for
`mlx_load_safetensors`/`mlx_load_gguf`, and a redesign of `loadGguf`'s metadata parameters once
testing showed `mlx_io_gguf_get_keys` cannot enumerate metadata-only keys). Summary: a new
`se.alipsa.jmlx.core.MLXIO` facade class (package-private-constructor constraints rule out a new
package — see D2) exposing `loadSafetensors`/`saveSafetensors`/`loadGguf`/`saveGguf`, following two
distinct precedents for two distinct properties rather than one class as a blanket model: `MLXGrad`'s
own `NativeLoader.ensureLoaded()` guard, since `MLXIO` calls `mlx_h` directly and cannot rely on
`NativeOps`'s transitive guard the way `MLXQuant` does (`MLXQuant` has no guard of its own precisely
because every entry point still funnels through `NativeOps.checked`/`scopeOf`/`cstr` first;
`MLXIO`'s hand-rolled bodies must not assume the same); and `MLXQuant`'s hand-rolled-body shape and
cached-stream reuse (D1 above) for everything else -- confined `Arena` for transient structs freed on
every exit path via `try/finally`, swallowing each free's own status per D2c above rather than
routing it through `NativeOps.checked`. `loadSafetensors`/`loadGguf` each take an explicit `MLXScope`
parameter (D2a above); `saveGguf` additionally builds and frees its own `mlx_io_gguf` via a confined
`Arena`-scoped helper, never an `MLXScope` (D2b above). Update CLAUDE.md's "Loading order matters"
paragraph and architecture-diagram class list to name `MLXIO` as the fifth native-loading-guard
class alongside `MLX`/`MLXScope`/`NativeOps`/`MLXGrad`.

### 2. Tokenizer integration (M2) — **DESK RESEARCH DONE, ARCHITECTURE CHOICE STILL OPEN — see D3's amendment**

Not planned in detail here. Desk research (license, C-API existence, build shape, prior art on both
sides, risks) is written up as D3's amendment above; `req/plans/phase5-m2-plan.md` still should not
be written until both the FFM-vs-pure-Java choice is made and, if FFM is chosen, the load-time-cost
prototype D3 also asked for actually lands.

### 3. Reference models — `LlamaModel`, `QwenModel` (M3) — **NOT STARTED — blocked on M1 and M2**

Not planned in detail here — see D4.

## Testing approach

Same conventions `req/plans/phase4-m1-plan.md`'s Global Constraint 7 already establishes project-wide:
`@EnabledIfNativeAvailable` on native-dependent test classes, `try (MLXScope scope = new MLXScope())`,
hand-computed goldens, element-value assertions rather than shape-only assertions. Round-trip alone
needs no external binary fixture files: `mlx_save_safetensors`/`mlx_save_gguf` exist precisely so
tests can round-trip (build small `MLXArray`s, save, reload in a fresh scope, assert tensor values
and metadata match) for both formats — the concrete test list is `req/plans/phase5-m1-plan.md`'s
Task 4.

Round-trip proves `MLXIO` is self-consistent (jmlx writes, jmlx reads back what it wrote); it does
not exercise the real-world variation Context's own quoted objective ("Load pre-trained safetensors
and GGUF model files") requires -- HF's vs. llama.cpp's differing metadata conventions, real
quantized tensor dtypes, or the tensor-naming schemes M3's checkpoint → `Module` mapping will
eventually depend on. `req/plans/phase5-m1-plan.md`'s Task 4 resolves this by choosing the second of
the two ways this document originally posed to close that gap: real-world validation is deferred to
M3 (where a real checkpoint is unavoidable anyway), not a checked-in fixture fragment added here --
M1's round-trip tests prove only the facade layer, not interop, and that is stated as a deliberate,
named scope boundary rather than left implicit.

## Open questions

- **Resolved: waiting for `mlx`/`mlx-c` to add tokenizer support upstream is not a viable deferral
  for M2.** See D3's amendment — MLX's own two other official language bindings (`mlx-lm`,
  `mlx-swift-lm`) both already solved this by depending on an external tokenizer package rather than
  MLX/mlx-c itself, and there is no roadmap signal anywhere suggesting that will change.
- **Still open: M2's architecture question itself (FFM-bind a plain-C shim over HF `tokenizers` vs.
  a pure-Java reimplementation) is genuinely two-sided, not resolved.** An earlier pass through this
  document concluded DJL's Rust/JNI choice settled it — that overstated things: Hugging Face's own
  `swift-transformers` independently chose a from-scratch pure-Swift reimplementation for the
  identical problem, so both directions now have a maintained, production-used precedent from a
  credible source. The pure-Java port size is now measured, not guessed (see D3's amendment): ~3,850
  lines for tokenization alone, ~7,700 lines with full chat-template fidelity if porting Jinja from
  HF's own smaller, more current JS implementation (`@huggingface/jinja`) rather than `swift-jinja`,
  or ~10,500 from the latter — real, quantified numbers to weigh against the FFM path's toolchain
  cost, not an open unknown anymore. **The porting question is largely mooted since: `hfjinja`
  (`github.com/Alipsa/hfjinja`) is a released, dependency-free Java 21+ port of `@huggingface/jinja`
  itself — i.e. option 2 already done, by the same org as jmlx — so the chat-template half of the
  pure-Java estimate (~3,860 of the ~7,700 lines) doesn't need porting at all if adopted. Its adoption
  cost is not moot, though: it is confirmed absent from Maven Central (`numFound: 0`), so using it
  today means JitPack, a source/composite build, or publishing it first — see D3's amendment.** What
  remains genuinely open regardless of which tokenization direction (FFM vs. pure-Java) is chosen:
  real load-time cost for the FFM path (needs an actual build-and-measure prototype, blocked on a
  Rust toolchain decision), including which regex backend to build it with -- Llama/Qwen's
  pretokenizer regexes need lookahead/Unicode-class support that only `onig` or `fancy-regex` provide
  (plain `regex` is ruled out either way), but whether `fancy-regex`'s pure-Rust split behavior
  actually matches `onig`'s on these specific patterns is unconfirmed and belongs in that same
  prototype (see D3's amendment). Whether M3's reference models need general
  Jinja2 chat-template evaluation at all or can get away with hand-formatting a small, known set of
  chat templates instead is still deferred to M3's own requirements (D4), though if `hfjinja` is
  adopted and its Maven-availability cost is paid, that question loses most of its urgency, since the
  "port vs. hand-format" trade-off it was weighing is no longer a real port.

No open question remains on the checkpoint-I/O (M1) side: `mlx_vector_string_get`'s ownership, the
last unresolved item blocking `loadGguf`'s design, is settled — see Research findings above.
