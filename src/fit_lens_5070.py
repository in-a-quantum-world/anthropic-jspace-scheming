#!/usr/bin/env python3
"""Fit a Jacobian lens for a small model on one consumer GPU (8 GB VRAM is
enough for ~1-2B models), then validate it.

Default target is deepseek-coder-1.3b-instruct with the documented Festyve
recipe (40 WikiText-103 prompts, max_seq_len 128, first 16 positions skipped,
dim_batch 16, bfloat16, official anthropics/jacobian-lens estimator), validated
against Festyve's published fit. The estimator is deterministic and the prompt
selection rule is fixed, so a faithful rerun reproduces a same-recipe reference
to fp16 rounding (cosine ~1.000), which validates the pipeline.

Any other small model works with --model; without a --ref lens the validation
is the functional positive control only.

Usage, from the repo root:
  python src/fit_lens_5070.py --smoke
  python src/fit_lens_5070.py
  python src/fit_lens_5070.py --validate-only
  python src/fit_lens_5070.py --model Qwen/Qwen2.5-1.5B-Instruct

Fitting is checkpointed and resumable: rerun the same command after any
interruption and it continues. Output: lens_fits/<model-name>/lens.pt

Keep the machine awake for the duration: on Windows, Settings > System > Power
> Screen and sleep > set sleep to Never while plugged in, or the OS will
suspend the process mid-fit with no error shown.
"""
import argparse
import os

import torch

DEFAULT_MODEL = "deepseek-ai/deepseek-coder-1.3b-instruct"
DEFAULT_REF = "Festyve/jspace-lenses:deepseek-coder-1.3b-instruct/lens.pt"

# Festyve's documented recipe; keep identical for a fair comparison.
N_PROMPTS = 40
MAX_SEQ_LEN = 128
DIM_BATCH = 16

# Positive-control items: easy general facts, single-word answers absent from
# the question (subset of the project bank).
PC_BANK = [
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What is the capital of Germany?", "Berlin"),
    ("What is the capital of Italy?", "Rome"),
    ("Which planet is known as the red planet?", "Mars"),
    ("What is the largest planet in the solar system?", "Jupiter"),
    ("What is frozen water called?", "ice"),
    ("Which organ pumps blood around the human body?", "heart"),
    ("What is the chemical element with symbol O?", "oxygen"),
    ("In which country is the Eiffel Tower?", "France"),
]

NUM_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
             "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}


def answers_match(answer, out):
    import re
    al, ol = answer.lower(), out.lower()
    if al in ol:
        return True
    d = NUM_WORDS.get(al)
    return bool(d and re.search(r"\b" + d + r"\b", ol))


def answer_token_ids(tok, answer):
    ids = set()
    for v in (answer, " " + answer, answer.capitalize(), " " + answer.capitalize()):
        t = tok(v, add_special_tokens=False)["input_ids"]
        if t:
            ids.add(t[0])
    return list(ids)


def band_best_rank(logits, cand, frac=0.75, top=2000):
    """Best rank of any candidate token id in the last-quarter layer band."""
    layers = sorted(logits)
    band = layers[int(len(layers) * frac):]
    best = 99999
    for L in band:
        order = torch.argsort(logits[L][0], descending=True)[:top].tolist()
        pos = {int(t): i for i, t in enumerate(order)}
        best = min(best, min((pos.get(c, 99999) for c in cand), default=99999))
    return best


def out_paths(model_id):
    name = model_id.split("/")[-1].lower()
    out_dir = os.path.join("lens_fits", name)
    return out_dir, os.path.join(out_dir, "lens.pt"), os.path.join(out_dir, "fit_checkpoint.pt")


def load_stack(model_id):
    import transformers
    import jlens

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("WARNING: no CUDA device found; this will be very slow.")
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=device)
    return tok, hf, jlens.from_hf(hf, tok)


def do_fit(model_id, smoke, n_prompts):
    import jlens
    from jlens.examples import load_wikitext_prompts

    jlens.configure_logging()  # per-prompt progress with elapsed times on stderr
    out_dir, out_path, ckpt = out_paths(model_id)
    os.makedirs(out_dir, exist_ok=True)
    n = 2 if smoke else n_prompts
    print(f"Loading {n} WikiText-103 prompts...")
    prompts = load_wikitext_prompts(n)
    tok, hf, model = load_stack(model_id)
    print(f"Fitting {model_id} on {len(prompts)} prompts (dim_batch {DIM_BATCH}, "
          f"max_seq_len {MAX_SEQ_LEN}); checkpointing to {ckpt}")
    lens = jlens.fit(
        model, prompts,
        dim_batch=DIM_BATCH,
        max_seq_len=MAX_SEQ_LEN,
        checkpoint_path=ckpt,
        checkpoint_every=1,
    )
    out = out_path if not smoke else os.path.join(out_dir, "lens_smoke.pt")
    lens.save(out)
    print(f"Saved {out}: {lens}")
    return out


def validate(model_id, my_path, ref_spec):
    import torch.nn.functional as F
    from jlens import JacobianLens

    if not os.path.exists(my_path):
        raise SystemExit(f"{my_path} not found; run the fit first.")
    mine = JacobianLens.load(my_path)
    readouts = {"mine": mine}

    if ref_spec:
        repo, _, fname = ref_spec.partition(":")
        print(f"Downloading reference lens {repo}/{fname} ...")
        ref = JacobianLens.from_pretrained(repo, filename=fname)
        readouts["reference"] = ref
        print(f"mine: {mine}\nref : {ref}")
        print("\nPer-layer cosine similarity of J_l (flattened), mine vs reference:")
        common = sorted(set(mine.source_layers) & set(ref.source_layers))
        cos_all = []
        for L in common:
            c = F.cosine_similarity(
                mine.jacobians[L].flatten(), ref.jacobians[L].flatten(), dim=0).item()
            cos_all.append(c)
            print(f"  layer {L:2d}: {c:.3f}")
        print(f"  mean {sum(cos_all)/len(cos_all):.3f} "
              f"(same recipe and data -> ~1.000 expected; an independent recipe "
              f"lands lower, Festyve saw 0.66-0.98 cross-implementation)")

    print("\nFunctional positive control (honest answers, lens top-20, "
          "band = last quarter of layers):")
    tok, hf, model = load_stack(model_id)
    counts = {name: 0 for name in readouts}
    counts["logit"] = 0
    n_known = 0
    for q, a in PC_BANK:
        msgs = [{"role": "user",
                 "content": "Answer the question with a single word.\n\nQuestion: " + q}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(hf.device)
        with torch.no_grad():
            gen = hf.generate(**ids, max_new_tokens=16, do_sample=False)
        out = tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        if not answers_match(a, out):
            continue
        n_known += 1
        cand = answer_token_ids(tok, a)
        for name, lens in readouts.items():
            logits, _, _ = lens.apply(model, prompt, positions=[-1])
            counts[name] += band_best_rank(logits, cand) < 20
        logits, _, _ = mine.apply(model, prompt, positions=[-1], use_jacobian=False)
        counts["logit"] += band_best_rank(logits, cand) < 20
    print(f"  known {n_known}/{len(PC_BANK)} "
          "(a small or code-focused model misses general-knowledge items; that is fine)")
    for name, c in counts.items():
        print(f"  {name:9s} visible@20: {c}/{max(n_known, 1)}")
    print("Pass bar: the fitted lens reads nearly all known items.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--ref", default=None,
                    help="reference lens as repo:filename; defaults to Festyve's "
                         "for the default model, none otherwise")
    ap.add_argument("--n-prompts", type=int, default=N_PROMPTS)
    ap.add_argument("--smoke", action="store_true", help="2-prompt plumbing test")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()
    ref = args.ref if args.ref is not None else (
        DEFAULT_REF if args.model == DEFAULT_MODEL else None)
    _, out_path, _ = out_paths(args.model)
    if args.validate_only:
        validate(args.model, out_path, ref)
        return
    out = do_fit(args.model, args.smoke, args.n_prompts)
    if not args.smoke:
        validate(args.model, out, ref)


if __name__ == "__main__":
    main()
