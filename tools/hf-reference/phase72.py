"""Pinned CPU probes and tiny checkpoint references for encoder/seq2seq implementations."""
import hashlib
import inspect
import json
import platform
from pathlib import Path

import sentence_transformers
import torch
import transformers
from transformers import (BertConfig, BertModel, BertForSequenceClassification,
                          BertForTokenClassification, T5Config, T5ForConditionalGeneration)
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor
from transformers.models.bert import modeling_bert
from transformers.models.t5 import modeling_t5


def values(tensor):
    return tensor.detach().cpu().tolist()


def generate(out):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    sources = {name: hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest()
               for name, module in {"bert": modeling_bert, "t5": modeling_t5}.items()}
    report = {
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "sentence_transformers": sentence_transformers.__version__},
        "host": {"os": platform.system(), "arch": platform.machine()},
        "sources": sources,
        "contracts": {
            "bert": "absolute input-index positions; embedding LN; post-LN residuals; pooler tanh at index 0; sequence head reads pooler; token head reads hidden states; key-only mask",
            "t5": "q/k/v width heads*d_kv; float32 RMS accumulation without mean subtraction or bias; unscaled attention; pre-norm residuals; gated-gelu uses gelu_new; tied head scales hidden states by d_model**-0.5",
            "pooling": "attended special tokens included; CLS index 0; Mean/Max exclude padding; Normalize follows Pooling; artifact maximum sequence length enforced by tokenizer truncation upstream",
        },
    }
    ids = torch.tensor([[2, 7, 11, 3, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
    types = torch.tensor([[0, 0, 1, 1, 0, 0]])
    bert_config = BertConfig(vocab_size=32, hidden_size=16, intermediate_size=24,
                             num_hidden_layers=2, num_attention_heads=4,
                             max_position_embeddings=64, type_vocab_size=2,
                             hidden_dropout_prob=0, attention_probs_dropout_prob=0,
                             num_labels=3, layer_norm_eps=1e-12, hidden_act="gelu")
    for index, (name, constructor) in enumerate([
        ("bert", BertModel), ("bert-sequence", BertForSequenceClassification),
        ("bert-token", BertForTokenClassification)]):
        torch.manual_seed(7200 + index)
        config = BertConfig.from_dict(bert_config.to_dict())
        config._attn_implementation = "eager"
        model = constructor(config).cpu().float().eval()
        checkpoint = out / "checkpoints" / name
        model.save_pretrained(checkpoint, safe_serialization=True)
        with torch.no_grad():
            result = model(input_ids=ids, attention_mask=mask, token_type_ids=types,
                           output_hidden_states=True, return_dict=True)
            hidden = result.hidden_states[-1]
            mean = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
            maximum = hidden.masked_fill(mask.unsqueeze(-1) == 0, float("-inf")).max(1).values
            payload = {"ids": ids[0].tolist(), "attention_mask": mask[0].tolist(),
                       "type_ids": types[0].tolist(), "hidden_states": values(hidden[0]),
                       "cls": values(hidden[0, 0]), "mean": values(mean[0]),
                       "max": values(maximum[0]),
                       "normalized_mean": values(torch.nn.functional.normalize(mean, dim=-1)[0])}
            if name == "bert":
                payload["pooler"] = values(result.pooler_output[0])
            else:
                payload["logits"] = values(result.logits[0])
        (out / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")
    source = torch.tensor([[5, 7, 9, 1, 0]])
    source_mask = torch.tensor([[1, 1, 1, 1, 0]])
    history = torch.tensor([[0, 4, 6, 8, 10]])
    for index, (name, projection, tied) in enumerate([
        ("t5-relu", "relu", True), ("t5-gated", "gated-gelu", False)]):
        torch.manual_seed(7210 + index)
        config = T5Config(vocab_size=32, d_model=16, d_kv=3, d_ff=24,
                          num_layers=2, num_decoder_layers=2, num_heads=4,
                          dropout_rate=0, relative_attention_num_buckets=8,
                          relative_attention_max_distance=16, layer_norm_epsilon=1e-6,
                          decoder_start_token_id=0, pad_token_id=0, eos_token_id=1,
                          feed_forward_proj=projection, tie_word_embeddings=tied)
        config._attn_implementation = "eager"
        model = T5ForConditionalGeneration(config).cpu().float().eval()
        checkpoint = out / "checkpoints" / name
        model.save_pretrained(checkpoint, safe_serialization=True)
        with torch.no_grad():
            encoder = model.encoder(input_ids=source, attention_mask=source_mask).last_hidden_state
            full = model(input_ids=source, attention_mask=source_mask,
                         decoder_input_ids=history, use_cache=False).logits
            past = None
            cached = []
            for step in range(history.shape[1]):
                result = model(input_ids=source, attention_mask=source_mask,
                               decoder_input_ids=history[:, step:step+1],
                               past_key_values=past, use_cache=True)
                past = result.past_key_values
                cached.append(values(result.logits[0, 0]))
            greedy = model.generate(input_ids=source, attention_mask=source_mask,
                                    max_new_tokens=5, do_sample=False)
            processor_history = [[0], [0, 4], [0, 4, 6]]
            processor = RepetitionPenaltyLogitsProcessor(1.5)
            processed = [values(processor(torch.tensor([h]), torch.arange(32).float().unsqueeze(0))[0])
                         for h in processor_history]
            positions = torch.tensor(list(range(-32, 33)))
            buckets = {str(bidir): model.encoder.block[0].layer[0].SelfAttention._relative_position_bucket(
                positions, bidirectional=bidir, num_buckets=8, max_distance=16).tolist()
                       for bidir in (True, False)}
            payload = {"source_ids": source[0].tolist(), "source_mask": source_mask[0].tolist(),
                       "history": history[0].tolist(), "encoder": values(encoder[0]),
                       "logits": values(full[0]), "cached_logits": cached,
                       "greedy_ids": greedy[0].tolist(), "relative_positions": positions.tolist(),
                       "buckets": buckets, "processor_histories": processor_history,
                       "processed_logits": processed}
        (out / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out / "phase72-semantics.json").write_text(json.dumps(report, indent=2) + "\n")
    provenance_path = out.parent / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    # Preserve the provenance of existing Linux-produced decoder references.
    provenance["phase72"] = report
    provenance["phase72"]["requirements_lock_sha256"] = hashlib.sha256(
        Path(__file__).with_name("requirements.lock").read_bytes()).hexdigest()
    for path in sorted(out.rglob("*")):
        if path.is_file() and (path.stem.startswith(("bert", "t5", "phase72"))
                               or any(part.startswith(("bert", "t5")) for part in path.parts)):
            provenance["files"][str(path.relative_to(out))] = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
