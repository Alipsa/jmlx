#!/usr/bin/env python3
"""Phase 7.4 audio probe: Whisper-class log-mel feature pipeline.

Manually run in the locked hf-reference venv (never Gradle-invoked):

    tools/hf-reference/.venv/bin/python tools/hf-reference/whisper_logmel_probe.py

Synthesizes a deterministic 16 kHz mono waveform (no network, no fixture files),
computes the Whisper log-mel front end with transformers 4.57.6 (the same
computation openai/whisper-tiny uses), and records the pipeline contract: shapes,
dtypes, window/STFT parameters and which primitives jmlx already has versus the
gaps a Whisper-class model would need. Output is a JSON document printed to stdout;
the Phase 7.4 decision record pastes it verbatim as its audio findings.
"""

import json
import math
import platform
import sys

import numpy as np
import torch
import transformers
from transformers import WhisperFeatureExtractor
from transformers.audio_utils import mel_filter_bank

SEED = 20261031
DURATION_S = 30.0
SAMPLE_RATE = 16000
SAMPLES = int(DURATION_S * SAMPLE_RATE)


def synth_waveform():
    """Deterministic 30 s 16 kHz mono mix of sines, amplitude < 0.5."""
    rng = np.random.default_rng(SEED)
    t = np.arange(SAMPLES, dtype=np.float64) / SAMPLE_RATE
    freqs = [220.0, 440.0, 1760.0]
    wave = sum(0.15 * np.sin(2.0 * math.pi * f * t) for f in freqs)
    wave += 0.005 * rng.standard_normal(SAMPLES)
    return wave.astype(np.float32)


def main():
    torch.manual_seed(SEED)
    waveform = synth_waveform()

    # Whisper-tiny's published processor defaults, set explicitly so the probe
    # needs no network or Hub cache.
    extractor = WhisperFeatureExtractor(
        feature_size=80,
        sampling_rate=SAMPLE_RATE,
        hop_length=160,
        n_fft=400,
        chunk_length=30,
        n_samples=SAMPLES,
        padding_value=0.0,
        dither=0.0,
        return_attention_mask=False,
    )
    encoded = extractor(waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    features = encoded["input_features"]

    # The pure-numpy mel basis the feature extractor builds internally; showing
    # it explicitly documents that the filterbank is plain Java-portable math.
    fb = mel_filter_bank(
        num_frequency_bins=1 + 400 // 2,
        num_mel_filters=80,
        min_frequency=0.0,
        max_frequency=8000.0,
        sampling_rate=SAMPLE_RATE,
        norm="slaney",
        mel_scale="slaney",
    )

    findings = {
        "probe": "phase7-4-audio-whisper-logmel",
        "measured_vs_desk": "the frontend_contract and sample_values_checksum blocks are computed by this probe; jmlx_surface_mapping and loader_details are desk mappings checked against the installed transformers 4.57.6 modeling source, not executed",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "host_os": platform.system(),
            "host_arch": platform.machine(),
            "device": "cpu",
            "seed": SEED,
        },
        "waveform": {
            "samples": SAMPLES,
            "sample_rate": SAMPLE_RATE,
            "duration_s": DURATION_S,
            "synthesis": "sum of 0.15*sin sines at 220/440/1760 Hz plus 0.005 gaussian noise, np.random.default_rng(%d)" % SEED,
        },
        "frontend_contract": {
            "input_features_shape": list(features.shape),
            "input_features_dtype": str(features.dtype),
            "frame_window_ms": extractor.n_fft * 1000 // SAMPLE_RATE,
            "hop_ms": extractor.hop_length * 1000 // SAMPLE_RATE,
            "mel_filters": fb.shape[1],
            "fft_bins": fb.shape[0],
            "mel_basis_source": "transformers.audio_utils.mel_filter_bank (pure numpy, Slaney)",
            "stft_path_measured": "torch branch: torch.stft(...) then stft[..., :-1].abs() ** 2 — the last frame is dropped on the complex STFT output BEFORE the power, then the mel matmul, clamp(1e-10), log10",
            "stft_path_not_measured": "numpy branch (pocketfft, no torch installed): log_spec[:, :-1] drops the last frame AFTER log10 — a different code path with different floating-point results; a Java port must pin which branch is the oracle",
            "log_scaling": "torch path: clamp(mel, min=1e-10).log10(); dynamic-range compress to max-8: mel = max(mel, mel.max() - 8.0); normalize: mel = (mel + 4.0) / 4.0",
            "fixed_chunk": "30 s -> fixed (1, 80, 3000) float32 feature tensor regardless of audio length (pad/truncate)",
        },
        "sample_values_checksum": {
            "input_features_sum": float(features.double().sum()),
            "input_features_min": float(features.min()),
            "input_features_max": float(features.max()),
        },
        "jmlx_surface_mapping": {
            "already_available": {
                "conv1d_encoder_stem": "mlx_conv1d implemented (MLXConv, jmlx-core, Phase 7.1); whisper uses Conv1d(80->d, k=3, stride=1) then Conv1d(d->d, k=3, stride=2) — only the second conv downsamples",
                "transformer_encoder": "whisper encoder blocks are pre-LN with LayerNorm (with bias) and GELU ff (ACT2FN['gelu']); LayerNorm/GELU facades exist in jmlx-core nn (RMSNorm is NOT what whisper uses)",
                "cross_attention_decoder": "CrossAttention + StaticKVCache implemented (Phase 7.2); whisper decoder layers are also pre-LN LayerNorm + GELU",
                "seq2seq_decode": "generation is autoregressive token decode through the cross-attention decoder, the same surface as T5 (Phase 7.2); no CTC/loss surface is needed",
            },
            "gaps": {
                "audio_decode": "no audio module exists (jmlx-vision is image-only); 16 kHz mono PCM/WAV decode is pure-Java surface (a jmlx-audio or jmlx-vision extension)",
                "stft": "needs mlx_fft_* (all 16 downcalls + MLX_COMPLEX64 unplanned in req/mlx-api-inventory.md; MLXFFT is Phase 11 milestone 2). Whisper is FORWARD STFT only — spectrogram, mel matmul, then the [:, :-1] frame drop — no ISTFT anywhere in the pipeline",
                "mel_filterbank": "pure-Java port (Slaney mel scale + triangular filters); depends on the STFT magnitude output, not on FFT directly for the filterbank math",
                "resample": "arbitrary sample-rate resampling to 16 kHz is pure-Java surface if non-16 kHz input is accepted (whisper requires 16 kHz)",
            },
            "loader_details": {
                "k_proj_bias": "WhisperAttention sets k_proj with bias=False while q_proj/v_proj/out_proj keep bias=True — the checkpoint loader must read per-projection bias presence, not assume uniformity",
                "positional_tables": "encoder embed_positions is a frozen sinusoidal table capped at max_source_positions=1500; decoder learned positions capped at max_target_positions=448 — decode beyond 448 tokens is unsupported by the architecture itself",
                "generation_controls": "whisper decode needs the forced decoder prefix (startoftranscript + optional language/task/notimestamps special tokens) and suppress_tokens/begin_suppress_tokens (from generation_config) applied by the sampler — new model-level surface, no native dependency",
            },
        },
        "reference_model": {
            "repo": "openai/whisper-tiny",
            "license": "apache-2.0 — the license the Hugging Face artifact declares (the OpenAI GitHub code repo is MIT); the Tier-B pin must state the artifact's declared license",
            "size_bytes_safetensors": 151061672,
            "params_M": 39,
            "encoder_layers": 4,
            "decoder_layers": 4,
            "vocab_size": 51865,
            "checkpoint_format": "safetensors on the HF repo (fp32; ~75 MB fp16 GGUF exists in the whisper.cpp ecosystem); original OpenAI release is .pt",
        },
    }
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    sys.exit(main())
