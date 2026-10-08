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
# images, adjacent images, system/multi-turn content, interleaved text, and text-only.
SMOLVLM_CONVERSATIONS = (
    ("text_only", [{"role": "user", "content": "What color is the sky?"}], ()),
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
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": " what is this?"}]},
        {"role": "assistant", "content": "A picture."},
        {"role": "user", "content": "thanks"}], ("tile-4x4-100x80",)),
    ("text_only_multi_turn", [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
        {"role": "user", "content": "hello"}], ()),
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


def update_provenance(out, tokenizer_root=None):
    source = Path(transformers.__file__).parent / "modeling_rope_utils.py"
    files = {str(path.relative_to(out)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(out.rglob("*")) if path.is_file()}
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
                if path.is_file():
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
    parser.add_argument("--family", choices=(*FAMILIES, "smolvlm", "rope", "phase72", "all"),
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
