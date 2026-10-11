#!/usr/bin/env python3
"""Generate independent CPU float32 Hugging Face decoder and RoPE references."""

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import safetensors
import torch
import transformers
from safetensors import safe_open
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    GemmaConfig,
    Idefics3Config,
    Idefics3ForConditionalGeneration,
    Idefics3VisionConfig,
    LlamaConfig,
    MistralConfig,
    MixtralConfig,
    Phi3Config,
    Qwen2Config,
    Qwen3Config,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)


# qwen3 is appended last: the per-family seed is SEED + FAMILIES.index(family), so existing
# families keep their seeds (and goldens) byte-identical.
FAMILIES = ("llama", "qwen2", "llama31", "mistral", "phi3", "gemma", "mixtral", "qwen3")
CHAT_FAMILIES = ("mistral", "gemma", "phi3", "mixtral", "qwen3", "smolvlm")
SEED = 6302026
PROMPT_IDS = [1, 7, 42, 3, 19, 5]
# Peeled commit of the exact v4.57.6 release tag (not the annotated tag object).
TRANSFORMERS_SOURCE_COMMIT = "753d61104116eefc8ffc977327b441ee0c8d599f"


def rounded(value):
    if isinstance(value, torch.Tensor):
        return rounded(value.detach().cpu().tolist())
    if isinstance(value, list):
        return [rounded(item) for item in value]
    if isinstance(value, float):
        return float(f"{value:.7g}")
    return value


def llama31_scaling():
    return {
        "rope_type": "llama3",
        "factor": 8.0,
        "original_max_position_embeddings": 64,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
    }


def config_for(family):
    common = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        attention_dropout=0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    if family == "llama":
        return LlamaConfig(**common)
    if family == "llama31":
        config = LlamaConfig(**common, rope_scaling=llama31_scaling())
        inv_freq, _ = ROPE_INIT_FUNCTIONS["llama3"](config, device="cpu")
        original = torch.arange(0, 16, 2, dtype=torch.float32)
        original = 1.0 / (config.rope_theta ** (original / 16))
        ratio = inv_freq / original
        assert (ratio == 1).any(), "Llama 3 unchanged RoPE band is empty"
        assert ((ratio < 1) & (ratio > 1 / 8)).any(), "Llama 3 interpolated band is empty"
        assert torch.isclose(ratio, torch.tensor(1 / 8)).any(), "Llama 3 divided band is empty"
        return config
    if family == "qwen2":
        return Qwen2Config(**common, use_sliding_window=False)
    if family == "qwen3":
        # head_dim differs from hidden_size // num_attention_heads (16) on purpose, so the
        # golden exercises the explicit-head-dim path end to end; Qwen3 also always ships
        # q_norm/k_norm and no projection biases (attention_bias=False).
        return Qwen3Config(**common, head_dim=32)
    if family == "mistral":
        return MistralConfig(**common, sliding_window=4)
    if family == "phi3":
        return Phi3Config(**common, original_max_position_embeddings=128)
    if family == "gemma":
        return GemmaConfig(**dict(common, tie_word_embeddings=True), head_dim=16,
                           hidden_activation="gelu_pytorch_tanh")
    if family == "mixtral":
        return MixtralConfig(**common, num_local_experts=4, num_experts_per_tok=2)
    raise ValueError(family)


def expected_tensor_names(family):
    """Independent Hub-style key manifest; mirror ArchitectureMappings.tensorPlan manually."""
    names = {"model.embed_tokens.weight", "model.norm.weight"}
    if family != "gemma":
        names.add("lm_head.weight")
    for layer in range(2):
        prefix = f"model.layers.{layer}."
        names.update(prefix + name for name in ("input_layernorm.weight", "post_attention_layernorm.weight"))
        if family == "phi3":
            names.update(prefix + "self_attn." + name for name in ("qkv_proj.weight", "o_proj.weight"))
            names.update(prefix + "mlp." + name for name in ("gate_up_proj.weight", "down_proj.weight"))
        else:
            names.update(prefix + "self_attn." + name for name in (
                "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"))
            if family == "qwen2":
                names.update(prefix + "self_attn." + name for name in (
                    "q_proj.bias", "k_proj.bias", "v_proj.bias"))
            if family == "qwen3":
                # Float-only per-head QK normalization; Qwen3 has no projection biases.
                names.update(prefix + "self_attn." + name for name in (
                    "q_norm.weight", "k_norm.weight"))
            if family == "mixtral":
                names.add(prefix + "block_sparse_moe.gate.weight")
                for expert in range(4):
                    names.update(prefix + f"block_sparse_moe.experts.{expert}.{part}.weight"
                                 for part in ("w1", "w2", "w3"))
            else:
                names.update(prefix + "mlp." + name for name in (
                    "gate_proj.weight", "up_proj.weight", "down_proj.weight"))
    return names


def generate_family(family, out):
    torch.manual_seed(SEED + FAMILIES.index(family))
    torch.set_num_threads(1)
    config = config_for(family)
    config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_config(config).eval().float()
    checkpoint = out / "checkpoints" / family
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint, safe_serialization=True)
    saved = set()
    for path in checkpoint.glob("*.safetensors"):
        with safe_open(path, framework="pt", device="cpu") as weights:
            saved.update(weights.keys())
    expected = expected_tensor_names(family)
    if saved != expected:
        raise ValueError(f"{family} tensor names differ: missing={sorted(expected - saved)}; "
                         f"unexpected={sorted(saved - expected)}")
    # Run the public loader over the exact checkpoint that Java will read.
    del model
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, attn_implementation="eager", dtype=torch.float32).eval()
    prompt = torch.tensor([PROMPT_IDS], dtype=torch.long)
    with torch.inference_mode():
        prefill = model(input_ids=prompt, use_cache=False).logits.float()
        ids = prompt
        steps = []
        for _ in range(2):
            next_id = int(model(input_ids=ids, use_cache=False).logits[0, -1].argmax())
            ids = torch.cat((ids, torch.tensor([[next_id]])), dim=1)
            logits = model(input_ids=ids, use_cache=False).logits[0, -1].float()
            steps.append({"token_id": next_id, "logits": rounded(logits)})
    data = {
        "family": family,
        "model_type": config.model_type,
        "seed": SEED + FAMILIES.index(family),
        "prompt_ids": PROMPT_IDS,
        "prefill_logits": rounded(prefill[0]),
        "decode_steps": steps,
    }
    (out / f"{family}.json").write_text(json.dumps(data, indent=2) + "\n")


def generate_mistral_windows(out):
    """Generate extra window-boundary references from the existing pinned tiny checkpoint."""
    checkpoint = out / "checkpoints" / "mistral"
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, attn_implementation="eager", dtype=torch.float32,
        local_files_only=True).eval()
    prompts = {
        "shorter_than_window": [1, 7, 42],
        "equal_to_window": [1, 7, 42, 3],
        "longer_than_window": [1, 7, 42, 3, 19, 5],
        "chunked_prefill": [1, 7, 42, 3, 19, 5],
        "older_token_a": [1, 7, 42, 3, 19, 5, 6, 8, 10, 11, 12, 13],
        "older_token_b": [2, 7, 42, 3, 19, 5, 6, 8, 10, 11, 12, 13],
    }
    cases = []
    with torch.inference_mode():
        for name, ids in prompts.items():
            logits = model(input_ids=torch.tensor([ids]), use_cache=False).logits[0, -1]
            cases.append({
                "name": name,
                "prompt_ids": ids,
                "chunk_lengths": [3, len(ids) - 3] if name == "chunked_prefill" else [len(ids)],
                "last_logits": rounded(logits.float()),
            })
        ids = list(prompts["longer_than_window"])
        decode = []
        for _ in range(16):
            logits = model(input_ids=torch.tensor([ids]), use_cache=False).logits[0, -1]
            next_id = int(logits.argmax())
            ids.append(next_id)
            decode.append({"token_id": next_id, "logits": rounded(
                model(input_ids=torch.tensor([ids]), use_cache=False).logits[0, -1].float())})
    (out / "mistral-window.json").write_text(json.dumps({
        "family": "mistral", "window": 4, "max_position_embeddings": 128,
        "cases": cases, "long_decode": decode,
    }, indent=2) + "\n")


def generate_rope(out):
    base = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2,
                       max_position_embeddings=128)
    variants = {
        "default": None,
        "linear": {"rope_type": "linear", "factor": 8.0},
        "dynamic": {"rope_type": "dynamic", "factor": 8.0},
        "llama3": llama31_scaling(),
        "yarn": {"rope_type": "yarn", "factor": 8.0,
                 "original_max_position_embeddings": 64, "beta_fast": 32, "beta_slow": 1},
    }
    samples = []
    for name, scaling in variants.items():
        base.rope_scaling = scaling
        rope = LlamaRotaryEmbedding(base, device="cpu")
        for offset in (0, 17, 80, 160):
            # Dynamic NTK recomputes frequencies only when the sequence exceeds max_position_embeddings.
            inv_freq, attention_scaling = ROPE_INIT_FUNCTIONS[name](
                base, device="cpu", seq_len=max(128, offset + 1))
            positions = torch.tensor([[offset]], dtype=torch.long)
            fixed = torch.arange(16, dtype=torch.float32).reshape(1, 1, 1, 16) / 16
            cos, sin = rope(fixed, positions)
            rotated, _ = apply_rotary_pos_emb(fixed, fixed, cos, sin)
            samples.append({
                "rope_type": name,
                "offset": offset,
                "inv_freq": rounded(inv_freq),
                "attention_scaling": rounded(float(attention_scaling)),
                "input": rounded(fixed[0, 0, 0]),
                "rotated": rounded(rotated[0, 0, 0]),
            })
    (out / "rope.json").write_text(json.dumps({"samples": samples}, indent=2) + "\n")


def qwen3_conversations():
    """The qwen3 chat case matrix: enable_thinking true/false over the standard
    conversations, plus one tool-call conversation. The Qwen3 template's
    enable_thinking semantics are inverted (false inserts an empty thinking block
    before the generation prompt), so each conversation pins its value explicitly
    and the golden records it so the Java render can reproduce the context."""
    plain = [{"role": "user", "content": "hello"}]
    system_user = [
        {"role": "system", "content": "helpful"},
        {"role": "user", "content": "hello"},
    ]
    multi_turn = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
        {"role": "user", "content": "hello"},
    ]
    tool_call = [
        {"role": "user", "content": "What is the weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }
            ],
        },
        {"role": "tool", "content": "sunny"},
        {"role": "user", "content": "What is the weather in Tokyo?"},
    ]
    return [
        ("plain_thinking_off", plain, {"enable_thinking": False}),
        ("plain_thinking_on", plain, {"enable_thinking": True}),
        ("system_user_thinking_off", system_user, {"enable_thinking": False}),
        ("system_user_thinking_on", system_user, {"enable_thinking": True}),
        ("multi_turn_thinking_off", multi_turn, {"enable_thinking": False}),
        ("multi_turn_thinking_on", multi_turn, {"enable_thinking": True}),
        ("tool_call_thinking_off", tool_call, {"enable_thinking": False}),
    ]


def generate_chat(family, out, tokenizer_root):
    """Render and encode committed local tokenizer bundles through Transformers."""
    directory = tokenizer_root / family
    if not (directory / "tokenizer.json").is_file() or not (directory / "tokenizer_config.json").is_file():
        raise ValueError(f"missing committed tokenizer bundle for {family}: {directory}")
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, use_fast=True)
    if family == "qwen3":
        conversations = qwen3_conversations()
    else:
        conversations = [
            ("plain", [{"role": "user", "content": "hello"}], {}),
            ("system_user", [
                {"role": "system", "content": "helpful"},
                {"role": "user", "content": "hello"},
            ], {}),
            ("multi_turn", [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
                {"role": "user", "content": "hello"},
            ], {}),
        ]
    cases = []
    for name, messages, extra in conversations:
        for add_generation_prompt in (False, True):
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                **extra
            )
            ids = tokenizer(rendered, add_special_tokens=False).input_ids
            case = {
                "name": name,
                "messages": messages,
                "add_generation_prompt": add_generation_prompt,
                "rendered": rendered,
                "ids": ids,
            }
            if extra:
                case["extra"] = extra
            cases.append(case)
    (out / f"chat-{family}.json").write_text(
        json.dumps({"family": family, "cases": cases}, ensure_ascii=False, indent=2) + "\n"
    )


# smolvlm is the Phase 7.3a vision family: its "rendered" text and "ids" are the unexpanded
# chat-template output (the Java renderChat + encode target), and "expanded_ids" is the pinned
# Idefics3Processor expansion (the WP4 expander's target). The case matrix: one image, two
# images, adjacent images, system/multi-turn content, interleaved text, and text-only. Text
# turns use {"type": "text", "text": ...} parts, which is what callers must send: the template
# iterates content, so a plain string renders as nothing (recorded, and pinned, by
# string_content_dropped_by_template).
SMOLVLM_CONVERSATIONS = (
    ("text_only", [{"role": "user", "content": [
        {"type": "text", "text": "What color is the sky?"}]}], ()),
    ("single_image_4x4", [{"role": "user", "content": [
        {"type": "text", "text": "What is in "},
        {"type": "image"},
        {"type": "text", "text": " this image?"}]}], ("tile-4x4-100x80",)),
    ("single_image_1x4", [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": " Describe this panorama."}]}], ("tile-1x4-2048x512",)),
    ("single_image_4x1", [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": " Describe this panorama."}]}], ("tile-4x1-512x2048",)),
    ("two_images_interleaved", [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": " first"},
        {"type": "image"},
        {"type": "text", "text": " second"}]}], ("tile-4x4-100x80", "tile-1x4-2048x512")),
    ("two_images_adjacent", [{"role": "user", "content": [
        {"type": "image"},
        {"type": "image"},
        {"type": "text", "text": " compare"}]}], ("tile-4x4-100x80", "tile-4x1-512x2048")),
    ("system_multi_turn_image", [
        {"role": "system", "content": [
            {"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": " what is this?"}]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "A picture."}]},
        {"role": "user", "content": [
            {"type": "text", "text": "thanks"}]}], ("tile-4x4-100x80",)),
    ("text_only_multi_turn", [
        {"role": "system", "content": [
            {"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [
            {"type": "text", "text": "hello"}]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "world"}]},
        {"role": "user", "content": [
            {"type": "text", "text": "hello"}]}], ()),
    ("string_content_dropped_by_template", [
        {"role": "user", "content": "What color is the sky?"}], ()),
)


def smolvlm_images():
    """Deterministic in-memory RGB images; only the post-resize tile geometry drives expansion."""
    import numpy as np
    from PIL import Image

    def make(w, h):
        arr = ((np.arange(w * h * 3, dtype=np.uint32) * 7 + 3) % 256).astype(
            "uint8").reshape(h, w, 3)
        return Image.fromarray(arr, "RGB")

    return {
        "tile-4x4-100x80": make(100, 80),
        "tile-1x4-2048x512": make(2048, 512),
        "tile-4x1-512x2048": make(512, 2048),
    }


def generate_chat_smolvlm(out, tokenizer_root):
    """Render and expand the committed SmolVLM bundle through the pinned Idefics3Processor."""
    directory = tokenizer_root / "smolvlm"
    missing = [name for name in
               ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json",
                "processor_config.json")
               if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"missing committed SmolVLM bundle files: {missing}")
    processor = AutoProcessor.from_pretrained(directory, local_files_only=True)
    tokenizer = processor.tokenizer
    images = smolvlm_images()
    cases = []
    for name, messages, image_names in SMOLVLM_CONVERSATIONS:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(rendered, add_special_tokens=False).input_ids
        case = {
            "name": name,
            "messages": messages,
            "add_generation_prompt": True,
            "rendered": rendered,
            "ids": ids,
        }
        if image_names:
            case["images"] = [
                {"name": image_name, "width": images[image_name].width,
                 "height": images[image_name].height}
                for image_name in image_names
            ]
            expanded = processor(text=[rendered], images=[images[i] for i in image_names])
            case["expanded_ids"] = expanded["input_ids"][0]
        cases.append(case)
    (out / "chat-smolvlm.json").write_text(
        json.dumps({"family": "smolvlm", "cases": cases}, ensure_ascii=False, indent=2) + "\n"
    )


# ---------------------------------------------------------------------------
# Idefics3 (WP6 unit 1): the tiny checkpoint, the byte-identity tokenizer
# bundle, the processor fixture, and the tile-geometry probe. Unit 1b appends
# the numerical case goldens to generate_idefics3.
# ---------------------------------------------------------------------------

IDEFICS3_CHECKPOINT_DIR = "checkpoints/idefics3"
# The committed WP4 SmolVLM bundle, the source of the canonical byte-level strings.
WP4_SMOLOLM_TOKENIZER = Path("jmlx-tokenizer/src/test/resources/families/smolvlm/tokenizer.json")
# content -> id. Bytes 0-255 occupy ids 0-255 (byte b encodes to id b); the 40
# special tokens follow, mirroring the Idefics3Processor marker layout.
IDEFICS3_SPECIAL_TOKENS = (
    [("<image>", 256), ("<fake_token_around_image>", 257), ("<global-img>", 258)]
    + [
        (f"<row_{r}_col_{c}>", 258 + (r - 1) * 6 + c)
        for r in range(1, 7)
        for c in range(1, 7)
    ]
    + [("<eos>", 295)]
)
# name -> (image file names in prompt order, unexpanded prompt, expected (rows, cols) per
# image). The expectations are the D6a tile table for size.longest_edge=64 and
# max_image_size.longest_edge=32; the prompt text is ASCII and newline-free.
IDEFICS3_CASES = (
    ("one_image", ("img-32x32.png",), "<image>what is this", ((2, 2),)),
    (
        "two_images",
        ("img-32x32.png", "img-64x32.png"),
        "<image>what is this and <image>that",
        ((2, 2), (1, 2)),
    ),
    (
        "image_text_image",
        ("img-32x32.png", "img-32x32.png"),
        "<image>one image and <image>another",
        ((2, 2), (2, 2)),
    ),
    ("resized_image", ("img-24x32.png",), "<image>what is this", ((2, 2),)),
    ("text_only", (), "what is this", ()),
)


def _idefics3_byte_token_table(wp4_vocab):
    """The 256 byte-level strings of the tiny Idefics3 tokenizer, as {byte: string}.

    The table is the canonical GPT-2 `bytes_to_unicode` byte-to-character mapping (the same
    one the WP4 SmolVLM bundle's ByteLevel pre-tokenizer implements). It is not trusted on
    faith: every byte value that can be isolated through a valid UTF-8 sequence is re-derived
    from the pinned `tokenizers` library's ByteLevel pre-tokenizer and must agree, and the
    WP4 committed vocab must place the 0x0A byte string at its GPT-2 id 198. The few byte
    values that are not valid UTF-8 lead bytes (0xC0, 0xC1, 0xED, 0xF5-0xFF) cannot be
    probed through a string, so they are taken from the canonical table. Byte b then occupies
    id b in the 296-entry tiny vocab, so any byte value round-trips to its own id.
    """
    from tokenizers.pre_tokenizers import ByteLevel

    primary = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    codepoints = primary[:]
    n = 0
    for byte in range(256):
        if byte not in primary:
            primary.append(byte)
            codepoints.append(256 + n)
            n += 1
    table = {byte: chr(codepoint) for byte, codepoint in zip(primary, codepoints)}
    if len(set(table.values())) != 256:
        raise ValueError("the byte table is not a 256-entry bijection")

    # Bytes that are never a valid UTF-8 lead byte cannot be isolated through a Python
    # string, so they are verified only via the canonical table.
    non_isolatable = {0xC0, 0xC1, 0xED, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD, 0xFE, 0xFF}

    def probe_for(byte):
        if byte < 0x80:
            return chr(byte), 0
        if byte <= 0xBF:
            return chr(0x80 + (byte & 0x3F)), 1
        if byte <= 0xDF:
            return chr((byte & 0x1F) << 6), 0
        if byte == 0xE0:
            return chr(0x0800), 0
        if byte == 0xE1:
            return chr(0x1800), 0
        if byte <= 0xEF:
            return chr(((byte & 0x0F) << 12) | 0x800), 0
        # 0xF0-0xF4 are 4-byte lead bytes (0xF5-0xFF are non-isolatable, handled above).
        if byte == 0xF0:
            return chr(0x10000), 0
        return chr((byte & 0x07) << 18), 0

    encoder = ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True)
    for byte in range(256):
        if byte in non_isolatable:
            continue
        probe, index = probe_for(byte)
        pretokens = encoder.pre_tokenize_str(probe)
        if len(pretokens) != 1:
            raise ValueError(f"byte {byte:#04x} probe {probe!r} gave {pretokens!r}, expected one token")
        token = pretokens[0][0]
        if token[index] != table[byte]:
            raise ValueError(
                f"pinned ByteLevel maps byte {byte:#04x} to {token[index]!r}, "
                f"but the canonical table has {table[byte]!r}"
            )
    if wp4_vocab.get(table[10]) != 198:
        raise ValueError(
            f"expected the WP4 id 198 entry to be the 0x0A byte string {table[10]!r}, "
            f"got id {wp4_vocab.get(table[10])}"
        )
    return table


def _idefics3_expected_tensors():
    text = {"model.text_model.embed_tokens.weight", "model.text_model.norm.weight", "lm_head.weight"}
    for i in range(2):
        prefix = f"model.text_model.layers.{i}"
        text |= {
            f"{prefix}.input_layernorm.weight",
            f"{prefix}.post_attention_layernorm.weight",
        } | {
            f"{prefix}.self_attn.{proj}_proj.weight" for proj in ("q", "k", "v", "o")
        } | {
            f"{prefix}.mlp.{proj}_proj.weight" for proj in ("gate", "up", "down")
        }
    vision = {
        "model.vision_model.embeddings.patch_embedding.weight",
        "model.vision_model.embeddings.patch_embedding.bias",
        "model.vision_model.embeddings.position_embedding.weight",
        "model.vision_model.post_layernorm.weight",
        "model.vision_model.post_layernorm.bias",
        "model.connector.modality_projection.proj.weight",
    }
    for i in range(2):
        prefix = f"model.vision_model.encoder.layers.{i}"
        vision |= {
            f"{prefix}.layer_norm1.weight",
            f"{prefix}.layer_norm1.bias",
            f"{prefix}.layer_norm2.weight",
            f"{prefix}.layer_norm2.bias",
        } | {
            f"{prefix}.self_attn.{proj}_proj.weight" for proj in ("q", "k", "v", "out")
        } | {
            f"{prefix}.self_attn.{proj}_proj.bias" for proj in ("q", "k", "v", "out")
        } | {
            f"{prefix}.mlp.{proj}.weight" for proj in ("fc1", "fc2")
        } | {
            f"{prefix}.mlp.{proj}.bias" for proj in ("fc1", "fc2")
        }
    return text | vision


def _idefics3_checkpoint(directory):
    text_config = LlamaConfig(
        vocab_size=296,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        attention_bias=False,
        tie_word_embeddings=False,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_id=295,
    )
    # The pinned transformers releases Idefics3's vision tower as Idefics3VisionConfig
    # (SiglipVisionConfig instances are not accepted by Idefics3Config); same field set.
    vision_config = Idefics3VisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=32,
        patch_size=8,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_channels=3,
    )
    config = Idefics3Config(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=256,
        scale_factor=2,
        tie_word_embeddings=False,
        torch_dtype="float32",
    )
    model = Idefics3ForConditionalGeneration(config).eval().float()
    expected = _idefics3_expected_tensors()
    actual = set(model.state_dict())
    if actual != expected:
        raise ValueError(
            "Idefics3 checkpoint tensor manifest mismatch: "
            f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )
    model.save_pretrained(directory, safe_serialization=True)
    reloaded = Idefics3ForConditionalGeneration.from_pretrained(
        directory, attn_implementation="eager", dtype=torch.float32, local_files_only=True
    )
    if set(reloaded.state_dict()) != expected:
        raise ValueError("the reloaded Idefics3 checkpoint does not match the expected tensor manifest")


def _idefics3_tokenizer_bundle(directory):
    """Write the 296-token byte-identity BPE bundle. The file mirrors the WP4 SmolVLM
    bundle field-for-field (same pre-tokenizer, decoder and BPE field names, the
    pre-merge_criteria schema) so both the pinned transformers stack and the jmlx-tokenizer
    Java loader read it; only the vocab and the (empty) merges differ."""
    wp4_vocab = json.loads(WP4_SMOLOLM_TOKENIZER.read_text())["model"]["vocab"]
    table = _idefics3_byte_token_table(wp4_vocab)
    vocab = {string: byte for byte, string in table.items()}
    vocab.update(dict(IDEFICS3_SPECIAL_TOKENS))
    if len(vocab) != 296:
        raise ValueError(f"the tiny Idefics3 vocab must have 296 entries, got {len(vocab)}")
    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": token_id,
                "content": content,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for content, token_id in IDEFICS3_SPECIAL_TOKENS
        ],
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Digits", "individual_digits": True},
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": True,
                },
            ],
        },
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": True,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": [],
        },
    }
    (directory / "tokenizer.json").write_text(json.dumps(tokenizer_json, indent=2) + "\n")
    # No bos/pad/unk keys: GPT2TokenizerFast's defaults are empty-string tokens with
    # add_bos_token=False, so add_special_tokens=True adds nothing.
    tokenizer_config = {
        "add_prefix_space": False,
        "eos_token": "<eos>",
        "model_max_length": 1000000,
        "processor_class": "Idefics3Processor",
        "tokenizer_class": "GPT2Tokenizer",
    }
    (directory / "tokenizer_config.json").write_text(json.dumps(tokenizer_config, indent=2) + "\n")
    (directory / "added_tokens.json").write_text(
        json.dumps(dict(IDEFICS3_SPECIAL_TOKENS), indent=2, sort_keys=True) + "\n"
    )
    # WP4's special_tokens_map.json entry shape (no "special" key on the markers).
    marker_entries = [
        {"content": content, "lstrip": False, "normalized": False, "rstrip": False, "single_word": False}
        for content, _ in IDEFICS3_SPECIAL_TOKENS
        if content != "<eos>"
    ]
    special_tokens_map = {"eos_token": "<eos>", "additional_special_tokens": marker_entries}
    (directory / "special_tokens_map.json").write_text(json.dumps(special_tokens_map, indent=2) + "\n")
    preprocessor_config = {
        "do_convert_rgb": True,
        "do_image_splitting": True,
        "do_normalize": True,
        "do_pad": True,
        "do_rescale": True,
        "do_resize": True,
        "image_mean": [0.5, 0.5, 0.5],
        "image_processor_type": "Idefics3ImageProcessor",
        "image_std": [0.5, 0.5, 0.5],
        "max_image_size": {"longest_edge": 32},
        "processor_class": "Idefics3Processor",
        "resample": 1,
        "rescale_factor": 0.00392156862745098,
        "size": {"longest_edge": 64},
    }
    (directory / "preprocessor_config.json").write_text(json.dumps(preprocessor_config, indent=2) + "\n")
    # Same two keys as the WP4 bundle's processor_config.json.
    (directory / "processor_config.json").write_text(
        json.dumps({"image_seq_len": 4, "processor_class": "Idefics3Processor"}, indent=2) + "\n"
    )


def _idefics3_images(images_dir):
    """The three deterministic WP6 probe images: opaque 8-bit RGB lossless PNGs whose
    row-major pixel channel index i (over w*h*3 channels) is (i * 7 + 3) % 256, the
    smolvlm_images() formula."""
    import numpy as np
    from PIL import Image

    images = {}
    for name, width, height in (
        ("img-32x32.png", 32, 32),
        ("img-64x32.png", 64, 32),
        ("img-24x32.png", 24, 32),
    ):
        arr = ((np.arange(width * height * 3, dtype=np.uint32) * 7 + 3) % 256).astype(
            "uint8").reshape(height, width, 3)
        path = images_dir / name
        Image.fromarray(arr, "RGB").save(path)
        images[name] = Image.open(path)
    return images


def _idefics3_check_tokenizer(tokenizer, label):
    for content, token_id in IDEFICS3_SPECIAL_TOKENS:
        if tokenizer.encode(content, add_special_tokens=False) != [token_id]:
            raise ValueError(f"{label} encodes {content!r} as {tokenizer.encode(content, add_special_tokens=False)}, expected [{token_id}]")
    for text, expected in (("\n", [10]), ("\n\n", [10, 10]), ("a b", [97, 32, 98])):
        actual = tokenizer.encode(text, add_special_tokens=False)
        if actual != expected:
            raise ValueError(f"{label} encodes {text!r} as {actual}, expected {expected}")
    prompt = "<image> what is this"
    plain = tokenizer.encode(prompt, add_special_tokens=False)
    special = tokenizer(prompt, add_special_tokens=True).input_ids
    if special != plain:
        raise ValueError(
            f"{label} with add_special_tokens=True must encode {prompt!r} exactly like "
            f"add_special_tokens=False (no bos added): {special} vs {plain}"
        )


def _idefics3_check_slot_tokens(name, prompt, expanded, expected_geometry):
    """Every <image> slot must expand to exactly (rows * cols + 1) * 4 copies of the image
    token, bracketed by rows * cols + 2 fake-token markers, and carry rows + 1 newlines."""
    if "\n" in prompt:
        raise ValueError(f"{name}: probe prompts must not contain newlines")
    fake_positions = [index for index, token in enumerate(expanded) if token == 257]
    cursor = 0
    total_image_tokens = 0
    for index, (rows, cols) in enumerate(expected_geometry):
        expected_fakes = rows * cols + 2
        if cursor + expected_fakes > len(fake_positions):
            raise ValueError(f"{name} image {index + 1}: not enough fake-token markers to delimit the slot")
        region = expanded[fake_positions[cursor]: fake_positions[cursor + expected_fakes - 1] + 1]
        image_count = region.count(256)
        newline_count = region.count(10)
        expected_images = (rows * cols + 1) * 4
        if image_count != expected_images:
            raise ValueError(
                f"{name} image {index + 1}: slot carries {image_count} image tokens, expected {expected_images}"
            )
        if rows > 0 and newline_count != rows + 1:
            raise ValueError(
                f"{name} image {index + 1}: slot carries {newline_count} newlines, expected {rows + 1}"
            )
        total_image_tokens += expected_images
        cursor += expected_fakes
    if cursor != len(fake_positions):
        raise ValueError(f"{name}: {len(fake_positions) - cursor} stray fake-token markers")
    if expanded.count(256) != total_image_tokens:
        raise ValueError(
            f"{name}: {expanded.count(256)} image tokens in the expanded sequence, expected {total_image_tokens}"
        )


def _idefics3_geometry_probe(processor, tokenizer, images, cases_spec):
    cases = []
    for name, image_names, prompt, expected_geometry in cases_spec:
        pil_images = [images[image_name] for image_name in image_names]
        unexpanded = processor.tokenizer(prompt, add_special_tokens=False).input_ids
        if image_names:
            output = processor(images=pil_images, text=[prompt], return_tensors=None)
        else:
            output = processor(text=[prompt], return_tensors=None)
        expanded = output["input_ids"][0]
        # The processor does not surface the per-image geometry; the image processor's own
        # patch-count helper is the authoritative source for the rows/cols of each split.
        rows, cols = [], []
        for image_name in image_names:
            image = images[image_name]
            # 4.57.6 reads images_kwargs unguarded, so it must be a (possibly empty) dict.
            _, tile_rows, tile_cols = processor.image_processor.get_number_of_image_patches(
                image.height, image.width, images_kwargs={}
            )
            rows.append(tile_rows)
            cols.append(tile_cols)
        if (len(rows), len(cols)) != (len(expected_geometry),) * 2:
            raise ValueError(
                f"{name}: expected {len(expected_geometry)} tile groups, got rows={rows} cols={cols}"
            )
        num_tiles = []
        for index, (expected_rows, expected_cols) in enumerate(expected_geometry):
            if (rows[index], cols[index]) != (expected_rows, expected_cols):
                raise ValueError(
                    f"{name} image {index + 1}: expected rows={expected_rows} cols={expected_cols}, "
                    f"got rows={rows[index]} cols={cols[index]}"
                )
            num_tiles.append(rows[index] * cols[index] + 1 if rows[index] > 0 else 1)
        _idefics3_check_slot_tokens(name, prompt, expanded, expected_geometry)
        cases.append(
            {
                "name": name,
                "prompt": prompt,
                "images": [
                    {"file": image_name, "width": images[image_name].width, "height": images[image_name].height}
                    for image_name in image_names
                ],
                "unexpanded_ids": unexpanded,
                "expanded_ids": expanded,
                "rows": rows,
                "cols": cols,
                "num_tiles": num_tiles,
                "note": "skeleton golden: WP6 unit 1b appends the numerical case results "
                        "(pixel values, image features, logits, decoded ids) to this case",
            }
        )
    return cases


def generate_idefics3(out):
    """WP6 unit 1: the tiny Idefics3 checkpoint, the byte-identity tokenizer bundle, the
    processor fixture, the tile-geometry probe, and the skeleton idefics3.json golden.
    Unit 1b appends the numerical case goldens (embeddings, logits, decode) here."""
    directory = out / IDEFICS3_CHECKPOINT_DIR
    images_dir = directory / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED + 11)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _idefics3_checkpoint(directory)
    finally:
        torch.set_num_threads(previous_threads)
    _idefics3_tokenizer_bundle(directory)
    images = _idefics3_images(images_dir)

    processor = AutoProcessor.from_pretrained(directory, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    _idefics3_check_tokenizer(tokenizer, "the standalone tokenizer")
    _idefics3_check_tokenizer(processor.tokenizer, "the processor tokenizer")

    cases = _idefics3_geometry_probe(processor, tokenizer, images, IDEFICS3_CASES)
    image_info = [
        {
            "file": image_name,
            "width": images[image_name].width,
            "height": images[image_name].height,
            "png_sha256": hashlib.sha256((images_dir / image_name).read_bytes()).hexdigest(),
        }
        for image_name in ("img-32x32.png", "img-64x32.png", "img-24x32.png")
    ]
    golden = {
        "family": "idefics3",
        "seed": SEED + 11,
        "config": json.loads((directory / "config.json").read_text()),
        "preprocessor_config": json.loads((directory / "preprocessor_config.json").read_text()),
        "processor_config": json.loads((directory / "processor_config.json").read_text()),
        "special_tokens": dict(IDEFICS3_SPECIAL_TOKENS),
        "images": image_info,
        "cases": cases,
    }
    (out / "idefics3.json").write_text(json.dumps(golden, indent=2) + "\n")
    update_provenance(out)


def update_provenance(out, tokenizer_root=None):
    source = Path(transformers.__file__).parent / "modeling_rope_utils.py"
    files = {str(path.relative_to(out)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(out.rglob("*"))
             if path.is_file() and not path.name.startswith(".")}
    metadata = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "safetensors": safetensors.__version__,
        "host_os": platform.system(),
        "host_arch": platform.machine(),
        "rope_source": "transformers/modeling_rope_utils.py",
        "rope_source_commit": TRANSFORMERS_SOURCE_COMMIT,
        "rope_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "files": files,
    }
    if tokenizer_root is not None:
        # Every committed file of each chat bundle is a source: the text families carry the
        # tokenizer pair, smolvlm also commits its chat_template.json, special tokens and the
        # Idefics3 processor configs the golden is rendered from.
        sources = {}
        for family in CHAT_FAMILIES:
            for path in sorted((tokenizer_root / family).iterdir()):
                if path.is_file() and not path.name.startswith("."):
                    sources[f"{family}/{path.name}"] = hashlib.sha256(
                        path.read_bytes()).hexdigest()
        metadata["chat_sources"] = sources
    else:
        previous = out.parent / "provenance.json"
        if previous.is_file():
            metadata["chat_sources"] = json.loads(previous.read_text()).get("chat_sources", {})
    (out.parent / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    expected_versions = {
        "torch": (torch.__version__, ("2.6.0" if platform.system() == "Darwin" else "2.6.0+cpu")),
        "transformers": (transformers.__version__, "4.57.6"),
        "safetensors": (safetensors.__version__, "0.6.2"),
    }
    for package, (actual, expected) in expected_versions.items():
        if actual != expected:
            raise RuntimeError(f"{package} version {actual} differs from pinned {expected}")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Python 3.12 is required")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family",
                        choices=(*FAMILIES, "smolvlm", "rope", "phase72", "idefics3", "all"),
                        required=True)
    parser.add_argument("--out", type=Path, required=True, help="goldens directory")
    parser.add_argument("--chat", action="store_true", help="generate chat goldens only")
    parser.add_argument("--window-cases", action="store_true",
                        help="generate extra Mistral window cases from the existing checkpoint")
    parser.add_argument("--tokenizer-root", type=Path,
                        default=Path("jmlx-tokenizer/src/test/resources/families"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.family == "phase72":
        from phase72 import generate
        generate(args.out)
        return
    if args.family == "idefics3":
        generate_idefics3(args.out)
        return
    if args.window_cases:
        if args.family != "mistral" or args.chat:
            parser.error("--window-cases requires --family mistral without --chat")
        generate_mistral_windows(args.out)
        update_provenance(args.out)
        return
    if args.chat:
        if args.family not in CHAT_FAMILIES and args.family != "all":
            parser.error("--chat requires a chat family or --family all")
        for family in CHAT_FAMILIES if args.family == "all" else (args.family,):
            if family == "smolvlm":
                generate_chat_smolvlm(args.out, args.tokenizer_root)
            else:
                generate_chat(family, args.out, args.tokenizer_root)
        update_provenance(args.out, args.tokenizer_root)
        return
    if args.family in ("rope", "all"):
        generate_rope(args.out)
    for family in FAMILIES if args.family == "all" else (args.family,):
        if family in FAMILIES:
            generate_family(family, args.out)
    update_provenance(args.out)


if __name__ == "__main__":
    main()
