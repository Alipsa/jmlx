#!/usr/bin/env python3
"""Phase 7.3b WP4 opt-in cross-check: Qwen3-VL interleaved multi-axis text RoPE.

Runs under tools/hf-reference/.venv (transformers 4.57.6, torch 2.6.0,
Pillow 12.3.0, NumPy 2.5.3, Python 3.12), CPU float32, single-threaded,
deterministic. It is never invoked by Gradle and writes no golden: it
evaluates the real pinned Qwen3VLTextRotaryEmbedding plus
apply_rotary_pos_emb on fixed, index-coded asymmetric cases and prints a
deterministic full-precision float32 case table to stdout. The Java twin
qwen3vl_rope_interleaved_check.jsh reads that table through
JMLX_ROPE_REFERENCE and compares jmlx's frozen se.alipsa.jmlx.nn.MultiAxisRope
component against it with a strict 1e-5 max absolute error; the pinned
transformers behavior is authoritative, so a divergence means the Java
component is fixed, not the probe.

The pinned-2B cases carry the snapshot's theta 5,000,000 / head_dim 128 /
mrope_section [24, 20, 20] and cover prefill, prefill at the pinned maximum
coordinate (89), decode at the all-axes-equal shape (the only shape whose
2-D position-ids path is defined to agree with the 3-D one, so it gates
that path as well), decode at the negative-delta coordinate (26), and
per-row batched positions. The small cases (10,000 / 10 / [2, 1, 2] and
1,000 / 6 / [1, 1, 1]) exercise every interleaved f % 3 owner at constant
index-coded axes. Every case has head_dim = 2 * sum(mrope_section) because
the reference rotate_half rotates the whole head dimension.

Internal gates -- any failure exits non-zero before the table is printed:

  * the pinned snapshot's text_config carries rope_theta 5,000,000,
    head_dim 128, mrope_section [24, 20, 20], mrope_interleaved true and
    rope_type "default";
  * every module's inv_freq is bit-exact against
    1.0 / (base ** (arange(0, head_dim, 2, int64).to(float32) / head_dim)),
    the documented float32 derivation order (findings section 6.9), and
    attention_scaling is 1.0;
  * every case's owner-gather recomputation -- the same float32 matmul, an
    explicit per-frequency owner selection, the cat((freqs, freqs), -1)
    layout and the float32 cos/sin -- is bit-exact against the module's
    own forward output, isolating apply_interleaved_mrope from the case
    values;
  * the equal-axes case's 2-D position-ids output is bit-exact against its
    3-D output (a 2-D input expands to all three axes, so the two paths
    only agree when the axes are equal);
  * the q output is bit-exact against the k output (the probe feeds
    q = k = x), and a double run reproduces the printed table's SHA-256.
"""
from __future__ import annotations

import argparse
import hashlib
import platform
import sys
from pathlib import Path

import qwen3vl_probe as probe

SCHEMA = ("#", "qwen3vl-rope-interleaved-reference", "v1")

# (name, theta, head_dim, sections, batch, heads, sequence, t, h, w); the
# position rows are flattened in (batch, token) order.
CASES: list[tuple[str, int, int, list[int], int, int, int, list[int], list[int], list[int]]] = [
    ("pinned_prefill_indexed", 5_000_000, 128, [24, 20, 20], 1, 1, 4,
     [0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23]),
    ("pinned_prefill_maxpos", 5_000_000, 128, [24, 20, 20], 1, 1, 2,
     [88, 89], [80, 81], [70, 71]),
    ("pinned_decode_negative_delta", 5_000_000, 128, [24, 20, 20], 1, 1, 1,
     [26], [26], [26]),
    ("pinned_decode_equal_axes", 5_000_000, 128, [24, 20, 20], 1, 1, 1,
     [89], [89], [89]),
    ("pinned_batched_rows", 5_000_000, 128, [24, 20, 20], 2, 1, 2,
     [0, 1, 30, 31], [4, 5, 26, 27], [8, 9, 22, 23]),
    ("small_212_every_owner", 10_000, 10, [2, 1, 2], 1, 1, 2,
     [1, 1], [2, 2], [4, 4]),
    ("small_111", 1_000, 6, [1, 1, 1], 1, 1, 2,
     [0, 1], [2, 3], [4, 5]),
]


def deterministic_x(n: int) -> list[float]:
    """The frozen probe input: distinct values in [-1, 1] over the flat index."""
    return [((i % 13) - 6) / 6.0 for i in range(n)]


def format_values(values: list[float]) -> str:
    # 9 significant decimal digits round-trip a float32 exactly, so the Java
    # twin reads back the reference's exact float32 values.
    return " ".join(f"{v:.9e}" for v in values)


def owner_axis(f: int, sections: list[int]) -> int:
    """The reference interleaved owner of frequency index f: H owns
    f % 3 == 1 with f < 1 + 3 * s1, W owns f % 3 == 2 with f < 2 + 3 * s2,
    everything else stays T (the slice(offset, length, 3) layout)."""
    if f % 3 == 1 and f < 1 + 3 * sections[1]:
        return 1
    if f % 3 == 2 and f < 2 + 3 * sections[2]:
        return 2
    return 0


def owner_gather_tables(module, head_dim: int, batch: int, sequence: int,
                        position_ids) -> tuple:
    """Recompute the module's float32 cos/sin tables without
    apply_interleaved_mrope: the same float32 matmul, an explicit
    per-frequency owner selection, the cat((freqs, freqs), -1) layout."""
    import torch

    half = head_dim // 2
    freqs = (
        module.inv_freq[None, None, :, None].float().expand(3, batch, half, 1)
        @ position_ids[:, :, None, :].float()
    ).transpose(2, 3)
    gathered = torch.zeros(batch, sequence, half)
    for f in range(half):
        gathered[:, :, f] = freqs[owner_axis(f, module.mrope_section), :, :, f]
    emb = torch.cat((gathered, gathered), dim=-1)
    return emb.cos() * module.attention_scaling, emb.sin() * module.attention_scaling


def gate_module(name: str, module, theta: int, head_dim: int, sections: list[int]) -> None:
    import torch

    probe.check_facts(name, "attention_scaling", module.attention_scaling, 1.0)
    probe.check_facts(name, "theta", float(module.config.rope_theta), float(theta))
    probe.check_facts(name, "head_dim", int(getattr(module.config, "head_dim")), head_dim)
    probe.check_facts(name, "mrope_section", list(module.mrope_section), sections)
    reference = 1.0 / (
        theta
        ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float) / head_dim)
    )
    if not torch.equal(module.inv_freq, reference):
        sys.exit(f"{name}: module.inv_freq is not bit-exact against the documented "
                 "float32 derivation 1.0 / (base ** (arange(0, head_dim, 2).float() "
                 "/ head_dim))")


def compute_case(name: str, theta: int, head_dim: int, sections: list[int], batch: int,
                 heads: int, sequence: int, t: list[int], h: list[int], w: list[int],
                 module) -> dict:
    import torch
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

    position_ids = torch.as_tensor([t, h, w], dtype=torch.long).reshape(3, batch, sequence)
    x = torch.tensor(deterministic_x(batch * heads * sequence * head_dim),
                     dtype=torch.float32).reshape(batch, heads, sequence, head_dim)
    cos, sin = module.forward(x, position_ids)

    cos_ref, sin_ref = owner_gather_tables(module, head_dim, batch, sequence, position_ids)
    if not torch.equal(cos, cos_ref) or not torch.equal(sin, sin_ref):
        sys.exit(f"{name}: apply_interleaved_mrope output is not bit-exact against the "
                 "explicit owner-gather recomputation")
    if name == "pinned_decode_equal_axes":
        cos_2d, sin_2d = module.forward(x, position_ids[0])
        if not torch.equal(cos, cos_2d) or not torch.equal(sin, sin_2d):
            sys.exit(f"{name}: the 2-D position-ids path is not bit-exact against the "
                     "3-D path on the all-axes-equal shape")
    q, k = apply_rotary_pos_emb(x, x, cos, sin)
    if not torch.equal(q, k):
        sys.exit(f"{name}: q and k outputs differ although the probe feeds q = k = x")
    return {
        "theta": theta,
        "head_dim": head_dim,
        "sections": sections,
        "batch": batch,
        "heads": heads,
        "sequence": sequence,
        "positions_t": t,
        "positions_h": h,
        "positions_w": w,
        "x": x.flatten().tolist(),
        "q": q.flatten().tolist(),
        "k": k.flatten().tolist(),
    }


def build_table(environment: dict, module_by_case: dict) -> str:
    lines = [
        " ".join(SCHEMA),
        f"transformers {environment['transformers']}",
        f"torch {environment['torch']}",
        f"python {environment['python']}",
        f"platform {environment['platform']}",
    ]
    for (name, theta, head_dim, sections, batch, heads, sequence, t, h, w), module in (
        zip(CASES, module_by_case)
    ):
        case = compute_case(name, theta, head_dim, sections, batch, heads, sequence,
                            t, h, w, module)
        lines.append(f"case {name}")
        lines.append(f"theta {case['theta']}")
        lines.append(f"head_dim {case['head_dim']}")
        lines.append("sections " + " ".join(str(v) for v in case["sections"]))
        lines.append(f"batch {case['batch']} heads {case['heads']} sequence {case['sequence']}")
        lines.append("positions_t " + " ".join(str(v) for v in case["positions_t"]))
        lines.append("positions_h " + " ".join(str(v) for v in case["positions_h"]))
        lines.append("positions_w " + " ".join(str(v) for v in case["positions_w"]))
        lines.append("x " + format_values(case["x"]))
        lines.append("q " + format_values(case["q"]))
        lines.append("k " + format_values(case["k"]))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--dir", type=Path,
                        default=here.parent.parent / ".tier-b" / "qwen3-vl-2b-instruct")
    args = parser.parse_args()
    directory: Path = args.dir
    if not (directory / "config.json").is_file():
        sys.exit(f"no config.json under {directory}; pass --dir to the pinned snapshot")

    environment = probe.gate_versions()
    environment = {
        **environment,
        "python": platform.python_version(),
        "platform": f"{platform.system()}-{platform.machine()}",
    }
    import torch
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLConfig,
        Qwen3VLTextConfig,
    )
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    # The pinned snapshot's text config is the authority for the 2B cases.
    rope_config = Qwen3VLConfig.from_pretrained(str(directory))
    text_config = rope_config.text_config
    probe.check_facts("config", "rope_theta", float(text_config.rope_theta), 5_000_000.0)
    probe.check_facts("config", "head_dim", int(text_config.head_dim), 128)
    probe.check_facts("config", "mrope_section",
                      list(text_config.rope_scaling["mrope_section"]), [24, 20, 20])
    probe.check_facts("config", "mrope_interleaved",
                      bool(text_config.rope_scaling["mrope_interleaved"]), True)
    probe.check_facts("config", "rope_type",
                      text_config.rope_scaling.get("rope_type", "default"), "default")
    cpu = torch.device("cpu")
    pinned_module = Qwen3VLTextRotaryEmbedding(text_config, device=cpu)
    small_configs = {
        (10_000, 10, (2, 1, 2)): Qwen3VLTextConfig(
            head_dim=10, rope_theta=10_000.0, max_position_embeddings=32,
            rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 2],
                          "mrope_interleaved": True}),
        (1_000, 6, (1, 1, 1)): Qwen3VLTextConfig(
            head_dim=6, rope_theta=1_000.0, max_position_embeddings=32,
            rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 1],
                          "mrope_interleaved": True}),
    }
    modules = []
    for name, theta, head_dim, sections, batch, heads, sequence, _t, _h, _w in CASES:
        probe.check_facts(name, "case_head_dim", head_dim, 2 * sum(sections),
                          "rotate_half rotates the whole head dimension")
        if theta == 5_000_000:
            module = pinned_module
        else:
            module = Qwen3VLTextRotaryEmbedding(
                small_configs[(theta, head_dim, tuple(sections))], device=cpu)
        gate_module(name, module, theta, head_dim, sections)
        modules.append(module)

    # Double-run determinism: the printed table must be reproducible bit-exactly.
    table = build_table(environment, modules)
    table_again = build_table(environment, modules)
    if hashlib.sha256(table.encode("utf-8")).hexdigest() != hashlib.sha256(
            table_again.encode("utf-8")).hexdigest():
        sys.exit("the reference table is not reproducible across a double run")
    sys.stdout.write(table)


if __name__ == "__main__":
    main()
