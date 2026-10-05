# Offline Hugging Face reference goldens

This is a generate-only Python tool. Gradle and CI read the committed JSON and
safetensors files and verify their hashes; they do not install or run Torch.
The references come from Hugging Face `transformers` on CPU, in float32, with
eager attention. `generate.py` seeds Torch, constructs tiny two-layer models,
saves each checkpoint with `save_pretrained`, and records full prompt logits and
two greedy decode steps. `rope.json` records Hugging Face inverse frequencies,
attention scaling, and rotary output at several offsets.

Python 3.12 is required. Regeneration is a reviewed, manual step:

```sh
cd tools/hf-reference
python3.12 -m venv .venv
.venv/bin/python -m pip install --require-hashes \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r requirements.lock
.venv/bin/python generate.py --family all --out goldens
cd ../..
./gradlew verifyHfReferenceGoldens
```

`--family` also accepts one family name or `rope` for a targeted regeneration.
Phase 7.2 has a separate seed-preserving entry point:

```sh
tools/hf-reference/.venv/bin/python tools/hf-reference/generate.py \
  --family phase72 --out tools/hf-reference/goldens
```

It produces BERT encoder/sequence/token heads and tied-ReLU/untied-gated T5 checkpoints,
full/cached logits, encoder states, buckets and greedy outputs. Source hashes and executable
semantic probes are recorded in `phase72-semantics.json` and provenance. The universal hash
lock retains Linux Torch `2.6.0+cpu`, uses Torch `2.6.0` on Darwin, and adds the actual
`sentence-transformers==5.1.2` pipeline. Existing decoder goldens are unchanged.

Generate reviewed candidates for locally downloaded real artifacts with:

```sh
tools/hf-reference/.venv/bin/python tools/hf-reference/tier_b.py \
  --directory .tier-b/minilm-l6-v2 --manifest tools/tier-b/minilm-l6-v2.json
```

The command also accepts classification and seq2seq manifests. It verifies artifact hashes,
checks installed versions, and records CPU float32 eager vectors/logits, source hashes and
top-two gaps. Regeneration resets acceptance. Rerun Java Tier-B, measure the real error and
verify `gap > 2 * epsilon + margin` before recording final eligibility. A multi-label
threshold would require `abs(logit) > epsilon + margin`. Python execution stays opt-in.
To regenerate only the additional Mistral window-boundary references from the committed
checkpoint without changing `mistral.json`, run
`tools/hf-reference/.venv/bin/python tools/hf-reference/generate.py --family mistral --window-cases --out tools/hf-reference/goldens`
from the repository root. The new file stays within the fixture's 128-position context limit.
The generator checks every saved tensor key against a Python-owned Hub-style
manifest. Keep that manifest in sync with `ArchitectureMappings.tensorPlan`;
the Java golden tests are the final cross-check. A mismatch stops generation.

`provenance.json` records versions, the Transformers source commit and RoPE
source hash, host OS and architecture, and SHA-256 of **every** file under
`goldens/`. A dependency repin should produce a reviewed golden diff, including
checkpoint and numerical changes. Regeneration on another host may produce
different float32 values and needs the same review. Do not edit JSON or weights
by hand.

### Chat references

Run `HF_HUB_OFFLINE=1 .venv/bin/python generate.py --chat --family all --out goldens`
from the repository root (or supply `--tokenizer-root`). This reads only the committed synthetic
family bundles under `jmlx-tokenizer/src/test/resources/families`. It writes
`goldens/chat-{mistral,gemma,phi3,mixtral}.json`, pins their hashes in `provenance.json`, and
records hashes of the input tokenizer bundles in `chat_sources`. The existing model and RoPE
goldens are left unchanged.
