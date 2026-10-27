#!/usr/bin/env python3
"""Fit a Jacobian lens for deepseek-coder-1.3b-instruct on one consumer GPU
(8 GB VRAM is enough), then validate it against the published community fit.

The fit mirrors the documented Festyve recipe exactly, so the comparison is
like-for-like: 40 WikiText-103 prompts, max_seq_len 128, first 16 positions
skipped, dim_batch 16, bfloat16 model, official anthropics/jacobian-lens
estimator. Fitting is checkpointed and resumable: rerun the same command after
any interruption and it continues where it stopped.

Usage, from the repo root:
  python src/fit_lens_5070.py --smoke          # 2-prompt plumbing test first
  python src/fit_lens_5070.py                  # full fit (tens of minutes on an RTX 5070)
  python src/fit_lens_5070.py --validate-only  # compare an existing fit vs Festyve's

Output: lens_fits/deepseek-coder-1.3b-instruct/lens.pt
Validation: per-layer cosine similarity of the J_l matrices vs the reference,
plus a functional positive control (honest answers visible in the lens top-20,
late-layer band) through both lenses and the logit-lens baseline.
"""
import argparse
import os

import torch

MODEL_ID = "deepseek-ai/deepseek-coder-1.3b-instruct"
REF_REPO = "Festyve/jspace-lenses"
REF_FILE = "deepseek-coder-1.3b-instruct/lens.pt"
OUT_DIR = os.path.join("lens_fits", "deepseek-coder-1.3b-instruct")
OUT_PATH = os.path.join(OUT_DIR, "lens.pt")
CKPT_PATH = os.path.join(OUT_DIR, "fit_checkpoint.pt")

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


def load_stack():
    import transformers
    import jlens

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("WARNING: no CUDA device found; this will be very slow.")
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=device)
    return tok, hf, jlens.from_hf(hf, tok)


def do_fit(smoke):
    import jlens
    from jlens.examples import load_wikitext_prompts

    os.makedirs(OUT_DIR, exist_ok=True)
    n = 2 if smoke else N_PROMPTS
    print(f"Loading {n} WikiText-103 prompts...")
    prompts = load_wikitext_prompts(n)
    tok, hf, model = load_stack()
    print(f"Fitting on {len(prompts)} prompts (dim_batch {DIM_BATCH}, "
          f"max_seq_len {MAX_SEQ_LEN}); checkpointing to {CKPT_PATH}")
    lens = jlens.fit(
        model, prompts,
        dim_batch=DIM_BATCH,
        max_seq_len=MAX_SEQ_LEN,
        checkpoint_path=CKPT_PATH,
        checkpoint_every=1,
    )
    out = OUT_PATH if not smoke else os.path.join(OUT_DIR, "lens_smoke.pt")
    lens.save(out)
    print(f"Saved {out}: {lens}")
    return out


def validate(my_path):
    import torch.nn.functional as F
    from jlens import JacobianLens

    if not os.path.exists(my_path):
        raise SystemExit(f"{my_path} not found; run the fit first.")
    mine = JacobianLens.load(my_path)
    print(f"Downloading reference lens {REF_REPO}/{REF_FILE} ...")
    ref = JacobianLens.from_pretrained(REF_REPO, filename=REF_FILE)
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
          f"(Festyve reported 0.66-0.98 vs the Neuronpedia Pythia reference; "
          f"same ballpark on most layers = pass)")

    print("\nFunctional positive control (honest answers, lens top-20, "
          "band = last quarter of layers):")
    tok, hf, model = load_stack()
    readouts = {"mine": mine, "reference": ref}
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
        # logit-lens baseline: same readout with the J_l transport disabled
        logits, _, _ = mine.apply(model, prompt, positions=[-1], use_jacobian=False)
        counts["logit"] += band_best_rank(logits, cand) < 20
    print(f"  known {n_known}/{len(PC_BANK)} "
          "(a 1.3B code model will miss some general-knowledge items; that is fine)")
    for name, c in counts.items():
        print(f"  {name:9s} visible@20: {c}/{max(n_known, 1)}")
    print("Pass bar: both lenses read nearly all known items; "
          "a big gap between mine and reference means the fit went wrong.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2-prompt plumbing test")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()
    if args.validate_only:
        validate(OUT_PATH)
        return
    out = do_fit(args.smoke)
    if not args.smoke:
        validate(out)


if __name__ == "__main__":
    main()
