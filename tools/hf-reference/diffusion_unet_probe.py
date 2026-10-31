#!/usr/bin/env python3
"""Phase 7.4 diffusion probe: SD-class UNet/VAE primitive coverage.

Manually run in the locked hf-reference venv (never Gradle-invoked):

    tools/hf-reference/.venv/bin/python tools/hf-reference/diffusion_unet_probe.py

Builds a tiny but SD-shaped UNet — time embedding, a down block with a
stride-2 downsample, a diffusers-style Transformer2D block (GroupNorm + 1x1
proj_in, self-attention + cross-attention to the text embedding + exact-GELU
GEGLU feed-forward, 1x1 proj_out, the way SD conditions on text), an up block
with SD's nearest-interpolate-plus-conv Upsample2D, and skip connections —
plus a tiny SD-shaped VAE decoder (single-head mid attention without a
feed-forward, three nearest-upsample-then-conv stages for the 8x decode SD's
block_out_channels imply, the vae scaling constant). Forwards a
deterministic batch through both and maps every exercised primitive to jmlx's
existing surface or a named gap. No network and no diffusers dependency: the
point is the primitive set, not a trained model. Output is a JSON document
printed to stdout; the Phase 7.4 decision record pastes it verbatim as its
diffusion findings.
"""

import json
import math
import platform
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

SEED = 20261031

# SD 1.5 vae.config.scaling_factor, applied to latents before decode.
VAE_SCALING_FACTOR = 0.18215


def sinusoids(steps, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
    angles = steps.float()[:, None] * freqs[None, :]
    return torch.cat([angles.cos(), angles.sin()], dim=-1)


def geglu(x):
    # diffusers activations.py GEGLU: exact F.gelu (approximate="none") on the gate.
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)


def nearest_up_conv(x, conv):
    # SD's Upsample2D(scale_factor, mode="nearest"): interpolate then conv.
    h = F.interpolate(x, scale_factor=2, mode="nearest")
    return conv(h)


class CrossAttnBlock(nn.Module):
    """SD Transformer2DModel block: LN -> self-attn, LN -> cross-attn to text,
    LN -> GEGLU feed-forward, each with a residual."""

    def __init__(self, dim, text_dim, heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_qkv = nn.Linear(dim, dim * 3)
        self.self_proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.x_q = nn.Linear(dim, dim)
        self.ctx_kv = nn.Linear(text_dim, dim * 2)
        self.x_proj = nn.Linear(dim, dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ff_geglu = nn.Linear(dim, dim * 8)  # GEGLU halves it back to 4x
        self.ff_proj = nn.Linear(dim * 4, dim)
        self.heads = heads

    def attend(self, q, k, v):
        b, n, d = q.shape
        q = q.view(b, n, self.heads, d // self.heads).transpose(1, 2)
        k = k.view(b, -1, self.heads, d // self.heads).transpose(1, 2)
        v = v.view(b, -1, self.heads, d // self.heads).transpose(1, 2)
        return F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d)

    def forward_sequence(self, seq, context):
        h = seq + self.self_proj(self.attend(*self.self_qkv(self.norm1(seq)).chunk(3, dim=-1)))
        q = self.x_q(self.norm2(h))
        k, v = self.ctx_kv(context).chunk(2, dim=-1)
        h = h + self.x_proj(self.attend(q, k, v))
        return h + self.ff_proj(geglu(self.ff_geglu(self.norm3(h))))

    def forward(self, x, context):
        b, c, ih, iw = x.shape
        seq = x.permute(0, 2, 3, 1).reshape(b, ih * iw, c)
        h = self.forward_sequence(seq, context)
        return h.reshape(b, ih, iw, c).permute(0, 3, 1, 2)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, temb):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.temb_proj = nn.Linear(temb, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Transformer2D(nn.Module):
    """diffusers Transformer2DModel wrapper: GroupNorm + 1x1 proj_in, the
    attention block, 1x1 proj_out, residual (use_linear_projection=false)."""

    def __init__(self, dim, text_dim, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(32, dim, eps=1e-6)
        self.proj_in = nn.Conv2d(dim, dim, 1)
        self.block = CrossAttnBlock(dim, text_dim, heads=heads)
        self.proj_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x, context):
        b, c, ih, iw = x.shape
        h = self.proj_in(self.norm(x))
        seq = h.permute(0, 2, 3, 1).reshape(b, ih * iw, c)
        seq = self.block.forward_sequence(seq, context)
        h = seq.reshape(b, ih, iw, c).permute(0, 3, 1, 2)
        return x + self.proj_out(h)


class TinyUNet(nn.Module):
    """conv in -> down resblock + stride-2 downsample -> Transformer2D (cross
    attn to text) at the reduced resolution -> up resblock + Upsample2D ->
    conv out (with skips)."""

    def __init__(self, base=64, temb=256, text_dim=768):
        super().__init__()
        self.conv_in = nn.Conv2d(4, base, 3, padding=1)
        self.time_proj = nn.Sequential(nn.Linear(base, temb), nn.SiLU(), nn.Linear(temb, temb))
        self.down = ResBlock(base, base * 2, temb)
        self.downsample = nn.Conv2d(base * 2, base * 2, 3, stride=2, padding=1)
        self.transformer = Transformer2D(base * 2, text_dim)
        self.up0 = nn.Conv2d(base * 2, base * 2, 3, padding=1)
        self.up = ResBlock(base * 4, base, temb)  # concat skip doubles the input
        self.conv_out = nn.Conv2d(base, 4, 3, padding=1)

    def forward(self, x, t, context):
        temb = self.time_proj(sinusoids(t, 64))
        h0 = self.conv_in(x)
        h1 = self.down(h0, temb)
        h2 = self.downsample(h1)
        h3 = self.transformer(h2, context)
        h3 = nearest_up_conv(h3, self.up0)
        h = self.up(torch.cat([h3, h1], dim=1), temb)
        return self.conv_out(h + h0)


class MidAttention(nn.Module):
    """diffusers AttentionBlock: GroupNorm, 1x1 qkv to 3x channels, single-head
    attention (heads=1), 1x1 proj_out — no feed-forward."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.GroupNorm(32, dim)
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        b, c, ih, iw = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q = q.view(b, c, ih * iw).transpose(1, 2)
        k = k.view(b, c, ih * iw).transpose(1, 2)
        v = v.view(b, c, ih * iw).transpose(1, 2)
        att = F.scaled_dot_product_attention(
            q.view(b, ih * iw, 1, c).transpose(1, 2),
            k.view(b, ih * iw, 1, c).transpose(1, 2),
            v.view(b, ih * iw, 1, c).transpose(1, 2))
        att = att.transpose(1, 2).reshape(b, c, ih, iw)
        return x + self.proj(att)


class TinyVAEDecoder(nn.Module):
    """SD-shaped for block_out_channels [b, b/2, b/4, b/8]: stem conv, mid block
    with single-head attention, then three Upsample2D stages (8x decode)."""

    def __init__(self, base=64):
        super().__init__()
        self.stem = nn.Conv2d(4, base, 3, padding=1)
        self.mid = MidAttention(base)
        self.up_convs = nn.ModuleList(
            [nn.Conv2d(base, base // 2, 3, padding=1),
             nn.Conv2d(base // 2, base // 4, 3, padding=1),
             nn.Conv2d(base // 4, base // 8, 3, padding=1)])
        self.out_norm = nn.GroupNorm(8, base // 8)
        self.out_conv = nn.Conv2d(base // 8, 3, 3, padding=1)

    def forward(self, latents):
        h = self.stem(latents / VAE_SCALING_FACTOR)
        h = self.mid(h)
        for conv in self.up_convs:
            h = nearest_up_conv(h, conv)
        return self.out_conv(F.silu(self.out_norm(h)))


def main():
    torch.manual_seed(SEED)
    unet = TinyUNet().eval()
    vae = TinyVAEDecoder().eval()
    latents = torch.randn(1, 4, 16, 16)
    steps = torch.tensor([981])
    context = torch.randn(1, 77, 768)  # CLIP ViT-L/14 text embedding shape
    with torch.no_grad():
        noise_pred = unet(latents, steps, context)
        pixels = vae(latents)
    findings = {
        "probe": "phase7-4-diffusion-unet-vae",
        "measured_vs_desk": "the shapes and value_checksums blocks are computed by this probe; unet_primitive_map, vae_primitive_map and the *_gap blocks are desk mappings against jmlx's inventory and nn facades, not executed",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "host_os": platform.system(),
            "host_arch": platform.machine(),
            "device": "cpu",
            "seed": SEED,
        },
        "shapes": {
            "latents_in": list(latents.shape),
            "unet_noise_pred_out": list(noise_pred.shape),
            "context_text_embedding": list(context.shape),
            "resolution_path": "16x16 -> downsample 8x8 (Transformer2D cross-attn here) -> upsample back to 16x16",
            "vae_pixels_out": list(pixels.shape),
            "vae_decode_factor": "16x16 latents -> 128x128 pixels (3 nearest-upsample stages, 8x, as SD's block_out_channels [128,256,512,512] implies)",
        },
        "value_checksums": {
            "noise_pred_sum": float(noise_pred.double().sum()),
            "pixels_sum": float(pixels.double().sum()),
            "pixels_range": [float(pixels.min()), float(pixels.max())],
        },
        "unet_primitive_map": {
            "conv2d": {"used": "stem/res/out/downsample convs", "jmlx": "implemented (MLXConv.conv2d, Phase 7.1)"},
            "groupnorm": {"used": "res block norms, num_groups=32", "jmlx": "implemented (GroupNorm, Phase 7.1)"},
            "layernorm": {"used": "Transformer2D block norms (cross-attn block)", "jmlx": "implemented (LayerNorm, Phase 4)"},
            "silu": {"used": "res block activations", "jmlx": "implemented (SiLU, Phase 4)"},
            "self_attention": {"used": "Transformer2D self-attention", "jmlx": "implemented (MultiHeadAttention/attention facades)"},
            "cross_attention_text_conditioning": {"used": "Transformer2D cross-attention to the CLIP text embedding (how the UNet takes the prompt)", "jmlx": "implemented (CrossAttention, Phase 7.2)"},
            "geglu_feed_forward": {"used": "Transformer2D FF: linear to 2x, chunk, x * exact_gelu(y) per diffusers activations.py GEGLU (F.gelu, approximate='none' — NOT QuickGELU), project", "jmlx": "composable from Linear + chunk + GELU (GELU.java exists) + multiply; no new primitive"},
            "transformer2d_wrapper": {"used": "GroupNorm + 1x1 conv proj_in around the attention block, 1x1 conv proj_out + residual (use_linear_projection=false)", "jmlx": "implemented (GroupNorm + MLXConv 1x1 convs)"},
            "linear_time_embedding": {"used": "sinusoids -> MLP -> add to res blocks", "jmlx": "implemented (Linear; sin/cos basic ops)"},
            "residual_skip": {"used": "1x1 conv identity skips + concat skips", "jmlx": "implemented"},
            "nearest_upsample_plus_conv": {"used": "Upsample2D: nearest interpolate then conv, inside the net", "jmlx": "implemented (Upsample nearest + MLXConv, Phase 7.1)"},
        },
        "vae_primitive_map": {
            "mid_single_head_attention": {"used": "VAE decoder mid block: GroupNorm(32), qkv projections, heads=1 attention, proj_out + residual — NO feed-forward (diffusers AttentionBlock)", "jmlx": "implemented (attention facades; single-head is a head-count, not a new primitive)"},
            "nearest_upsample_plus_conv": {"used": "all three decode stages (SD's VAE has NO ConvTranspose2d — nearest interpolate + conv)", "jmlx": "implemented (Upsample nearest + MLXConv)"},
            "latent_scaling_constant": {"used": "latents / 0.18215 before decode", "jmlx": "pure-Java constant, no native surface needed"},
        },
        "checkpoint_layout_note": {
            "model_index": "scheduler: diffusers PNDMScheduler; text_encoder: transformers CLIPTextModel; tokenizer: transformers CLIPTokenizer (verified from model_index.json at artifact sha 451f4fe)",
            "scheduler_pinning": "PNDM is the artifact's default — the 9.6 Tier-B reference must pin the artifact's default scheduler for the per-step noise-prediction comparison to be meaningful; its pinned scheduler/scheduler_config.json has skip_prk_steps: true (the Runge-Kutta warm-up never runs — only the PLMS path with its lower-order first steps) plus steps_offset: 1, set_alpha_to_one: false, beta_schedule 'scaled_linear', beta_start 0.00085, beta_end 0.012, which the 9.6 scheduler must port (steps_offset and set_alpha_to_one change the timestep grid and the final alpha)",
            "vae_attention_keys": "VAE-ONLY: the pinned VAE stores its mid-block AttentionBlock attention under the LEGACY names query/key/value/proj_attn, which diffusers remaps to to_q/to_k/to_v/to_out.0 at load time — the 9.6 loader must implement that remap (the UNet's attention already uses the current to_q/to_k/to_v/to_out.0 names: 0 of its 687 keys end in .query.weight at sha 451f4fe)",
            "vae_attention_weight_shapes": "stored as 2-D Linear weights ([512, 512] in the mid block), math-equivalent to the 1x1 convs the architecture implies — a loader expectation, not a primitive gap",
        },
        "note_on_convtranspose": "an earlier draft of this probe used ConvTranspose2d; SD 1.5's actual VAE decoder and UNet Upsample2D use nearest interpolation plus conv, which the probe now mirrors — convTranspose remains available in MLXConv either way",
        "scheduler_gap": {
            "finding": "no scheduler surface exists anywhere in jmlx; the artifact's default is PNDM — with its pinned skip_prk_steps: true the Runge-Kutta warm-up never runs, so the port needs only the PLMS path with its lower-order first steps — with DDPM/DDIM/Euler the common alternatives; all are small pure-Java state machines (alphas_cumprod table + step update + sigma conversion); PNDM's multistep history makes it a real state machine, not a one-line step update",
            "native_dependency": "none",
        },
        "text_encoder_gap": {
            "finding": "SD 1.5 conditions on the CLIP ViT-L/14 TEXT encoder's last_hidden_state (77x768 after final_layer_norm), via CLIPTextModel — no pooling and no text_projection (those belong to ...WithProjection, used by contrastive image-text models, not SD)",
            "architecture": "CLIP text is pre-LN (LayerNorm before attention and MLP) with a CAUSAL mask (_create_4d_causal_attention_mask in modeling_clip.py) and QuickGELU activation — BERT's post-LN bidirectional EncoderBlock is NOT a drop-in; DecoderAttention-style causal masking over an encoder stack, or a causal option on an encoder block, is new wiring to verify at milestone start",
            "native_dependency": "none beyond existing attention facades",
        },
        "tokenizer_gap": {
            "clip": "the pinned SD artifact ships NO tokenizer.json — only tokenizer/{vocab.json, merges.txt, special_tokens_map.json, tokenizer_config.json} (slow-tokenizer files, verified against the HF API at sha 451f4fe). Two paths: (a) a slow-tokenizer loader in jmlx-tokenizer over vocab.json+merges.txt (BpeMerger already exists; BOS <|startoftext|> 49406 / EOS <|endoftext|> 49407 and 77-token pad/truncate are small wrapper logic) — the record's recommendation, since the artifact itself ships these files; or (b) pin openai/clip-vit-large-patch14's tokenizer.json as a second artifact and add Split.invert=true support (rejected today in BOTH TokenizerJsonLoader and PreTokenizerPipeline) plus a RobertaProcessing post-processor, with an equivalence check against the artifact's own files. NOTE: the clip-vit-large-patch14 tokenizer.json is NOT loadable by jmlx-tokenizer as-is — its Split pre-tokenizer step has invert=true, which the loader rejects",
            "whisper": "whisper's tokenizer.json (no normalizer, ByteLevel pre-tokenizer, BPE, TemplateProcessing) fits jmlx-tokenizer's existing components; the generation-side controls (forced decoder prefix with language/task/notimestamps, suppress_tokens, begin_suppress_tokens, 448-position cap) are model-level, recorded in the audio probe's loader_details",
        },
        "fft_relevance": "none: UNet/VAE/CLIP-text exercise no FFT family member; conv+attention+upsample cover the whole pipeline",
        "reference_model_candidates": {
            "stable-diffusion-v1-5/stable-diffusion-v1-5": {"license": "CreativeML OpenRAIL-M (not OSI; use restrictions)", "revision": "pin at milestone WP0 (runwayml/stable-diffusion-v1-5 now 307-redirects to this canonical id)", "size_bytes_fp32_safetensors": {"unet": 3438167540, "vae": 334643276, "text_encoder": 492265874, "total": 4265076690}, "notes": "canonical SD 1.5; license differs from every existing Tier-B pin (MIT/Apache-2.0)"},
            "tiny_synthetic_checkpoint": {"license": "generated in-repo (Tier-A path, as 7.3a-7.3c)", "notes": "strict golden without any external artifact; the binding gate"},
        },
    }
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    sys.exit(main())
