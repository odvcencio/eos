#!/usr/bin/env python3
"""Generate the tiny committed BERT golden-parity fixture for eos.

This produces a deterministic, from-scratch (not downloaded) tiny BertModel
snapshot plus a small set of reference output embeddings, so
runtime/pretrained_bert_golden_parity_test.go can check Go BERT-forward
parity against a real Hugging Face oracle without any network access,
downloaded weights, or env-gated Python subprocess at test time.

Usage:
    python3 generate_fixture.py <output_dir>

Requires: torch, transformers, safetensors (see eos/.venv-qwen3, or any
venv with `pip install torch transformers safetensors`).
"""
import json
import sys

import torch
import torch.nn.functional as F
from transformers import BertConfig, BertModel, BertTokenizer

VOCAB = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
    "the", "cat", "sat", "on", "mat",
    "dog", "ran", "fast", "a", "quick", "brown", "fox",
]

TEXTS = [
    "the cat sat on the mat",
    "a quick brown fox",
    "the dog ran fast",
]

HIDDEN_SIZE = 16
NUM_LAYERS = 2
NUM_HEADS = 4
INTERMEDIATE_SIZE = 32
MAX_POSITION_EMBEDDINGS = 12
TYPE_VOCAB_SIZE = 2
SEED = 20260923


def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <output_dir>", file=sys.stderr)
        raise SystemExit(2)
    out_dir = sys.argv[1]

    import os
    os.makedirs(out_dir, exist_ok=True)

    # transformers>=5's BertTokenizer.__init__ takes `vocab` as an in-memory
    # str-or-dict, not a `vocab_file` path, and save_pretrained() on this
    # version only serializes tokenizer.json (the fast/Rust format). eos's Go
    # loader (LoadHFWordPieceTokenizerFromDir) reads the classic vocab.txt +
    # tokenizer_config.json + special_tokens_map.json layout that every real
    # HF BERT snapshot still ships, so write vocab.txt by hand from the same
    # ordered vocab used to build the in-memory tokenizer.
    vocab_dict = {token: i for i, token in enumerate(VOCAB)}
    tokenizer = BertTokenizer(vocab=vocab_dict, do_lower_case=True)
    vocab_path = os.path.join(out_dir, "vocab.txt")
    with open(vocab_path, "w") as f:
        f.write("\n".join(VOCAB) + "\n")
    tokenizer.save_pretrained(out_dir)
    # Drop the fast tokenizer.json; eos's Go loader does not read it, and it
    # would otherwise be a second, larger, redundant vocab encoding.
    tokenizer_json_path = os.path.join(out_dir, "tokenizer.json")
    if os.path.exists(tokenizer_json_path):
        os.remove(tokenizer_json_path)
    # save_pretrained() defaults model_max_length to a huge "unbounded"
    # sentinel (~1e30) when it is not set explicitly. eos's Go loader parses
    # it as a plain int, which overflows on that sentinel; pin it to the
    # fixture's real max length instead.
    tokenizer_config_path = os.path.join(out_dir, "tokenizer_config.json")
    with open(tokenizer_config_path) as f:
        tok_cfg = json.load(f)
    tok_cfg["model_max_length"] = MAX_POSITION_EMBEDDINGS
    with open(tokenizer_config_path, "w") as f:
        json.dump(tok_cfg, f, indent=2, sort_keys=True)
        f.write("\n")

    config = BertConfig(
        vocab_size=len(VOCAB),
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        hidden_act="gelu",
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        type_vocab_size=TYPE_VOCAB_SIZE,
        layer_norm_eps=1e-12,
        position_embedding_type="absolute",
        architectures=["BertModel"],
        model_type="bert",
    )

    torch.manual_seed(SEED)
    model = BertModel(config, add_pooling_layer=False)
    model.eval()

    # Re-init every parameter with a small deterministic uniform range so the
    # exported weights are reproducible independent of transformers' default
    # init routine (which can change across versions).
    gen = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            values = (torch.rand(param.shape, generator=gen) - 0.5) * 0.4
            param.copy_(values)

    model.save_pretrained(out_dir, safe_serialization=True)
    # save_pretrained also writes generation_config.json / etc. on newer
    # transformers versions for some model classes; BertModel does not emit
    # one, but drop anything that is not config.json / model.safetensors to
    # keep the committed fixture minimal.
    for extra in ("generation_config.json",):
        p = os.path.join(out_dir, extra)
        if os.path.exists(p):
            os.remove(p)

    max_length = MAX_POSITION_EMBEDDINGS
    encoded = tokenizer(
        TEXTS,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_token_type_ids=True,
    )
    with torch.no_grad():
        output = model(**encoded)
    mask = encoded["attention_mask"].unsqueeze(-1).to(output.last_hidden_state.dtype)
    summed = (output.last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    embeddings = summed / counts
    embeddings = F.normalize(embeddings, p=2, dim=1)

    golden = {
        "model_name": "eos-golden-fixture-tiny-bert",
        "pooling": "masked_mean",
        "max_length": max_length,
        "texts": TEXTS,
        "input_ids": encoded["input_ids"].reshape(-1).tolist(),
        "attention_mask": encoded["attention_mask"].reshape(-1).tolist(),
        "token_type_ids": encoded["token_type_ids"].reshape(-1).tolist(),
        "embeddings": [[round(v, 8) for v in row] for row in embeddings.tolist()],
        "generator": {
            "transformers_version": __import__("transformers").__version__,
            "torch_version": torch.__version__,
            "seed": SEED,
            "script": "runtime/testdata/pretrained_bert_golden/generate_fixture.py",
        },
    }
    golden_path = os.path.join(out_dir, "golden.json")
    with open(golden_path, "w") as f:
        json.dump(golden, f, indent=2)
        f.write("\n")

    print(f"wrote fixture to {out_dir}")
    for name in sorted(os.listdir(out_dir)):
        full = os.path.join(out_dir, name)
        print(f"  {name}\t{os.path.getsize(full)} bytes")


if __name__ == "__main__":
    main()
