#!/usr/bin/env python3
"""Compare two or more fitted lenses for the same model: per-layer cosine
similarity of the J matrices against the first lens given, plus norm ratios.

This is the instrument-variance measurement: lenses fitted with the same
recipe on disjoint corpora should agree strongly if the estimator is stable,
and an under-fitted lens should visibly diverge.

Usage, from the repo root:
  python src/compare_lenses.py lens_fits/qwen2.5-1.5b-instruct/lens.pt ^
      lens_fits/qwen2.5-1.5b-instruct-corpusB/lens.pt ^
      lens_fits/qwen2.5-1.5b-instruct-n10/lens.pt
"""
import argparse
import os

import torch
import torch.nn.functional as F


def load(path):
    from jlens import JacobianLens
    return JacobianLens.load(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lenses", nargs="+", help="two or more lens.pt paths; the first is the reference")
    args = ap.parse_args()
    if len(args.lenses) < 2:
        raise SystemExit("give at least two lens files")
    ref = load(args.lenses[0])
    names = [os.path.basename(os.path.dirname(p)) or p for p in args.lenses]
    print(f"reference: {names[0]} ({ref})\n")
    others = [(n, load(p)) for n, p in zip(names[1:], args.lenses[1:])]
    header = "| layer | " + " | ".join(f"cos {n}" for n, _ in others) + " |"
    print(header)
    print("|" + "---|" * (len(others) + 1))
    sums = [0.0] * len(others)
    common = sorted(set(ref.source_layers).intersection(*[set(o.source_layers) for _, o in others]))
    for L in common:
        row = [f"| {L:2d} "]
        for i, (_, o) in enumerate(others):
            c = F.cosine_similarity(ref.jacobians[L].flatten(), o.jacobians[L].flatten(), dim=0).item()
            sums[i] += c
            row.append(f"| {c:.3f} ")
        print("".join(row) + "|")
    means = " | ".join(f"{s / len(common):.3f}" for s in sums)
    print(f"\nmean cosine vs {names[0]}: {means}")
    print("Reading: same recipe on a disjoint corpus should sit high (how high is the "
          "measurement); the under-fitted lens shows what instability looks like.")


if __name__ == "__main__":
    main()
