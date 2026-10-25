#!/usr/bin/env python3
"""Week-1 offline analysis: read-depth sweep + AUROC rebuild from saved trajectories.

Input:  results/20261025-0118-sweep/sweep_12b_main.json  (full per-layer rank
        trajectories for true answer + 5 random distractors, honest & sandbag).
Output: results/analysis/depth_sweep_12b.json
        paper/figures/depth_sweep_12b.{png,pdf}
        paper/figures/traj_nitrogen_12b.{png,pdf}
        docs/week1-results.md (summary tables)

No GPU needed; pure post-processing. Run from repo root: python3 src/week1_depth_sweep.py
"""
import json, math, os
from collections import defaultdict

SENTINEL = 99999  # "not found in lens read at this layer"
IN_PATH = "results/20261025-0118-sweep/sweep_12b_main.json"
OUT_DIR = "results/analysis"
FIG_DIR = "paper/figures"


def load_rows(path):
    d = json.load(open(path))
    rows, dropped = [], 0
    for r in d["rows"]:
        a = r["a"]
        st, ht = r["sandbag_traj"], r["honest_traj"]
        if a not in st or a not in ht:
            dropped += 1  # duplicate-answer collision rows
            continue
        # distractors actually tracked; exclude the true answer and the token the
        # model actually emitted while sandbagging (it is legitimately in the
        # workspace because it was said, so it is not a fair negative)
        emitted = (r.get("sandbag_out") or "").strip().lower()
        dis = []
        for t in r["distractors"]:
            if t == a or t not in st:
                continue
            if t.strip().lower() == emitted:
                continue
            dis.append(t)
        if len(dis) < 3:
            dropped += 1
            continue
        rows.append({"q": r["q"], "a": a, "complied": r["complied"],
                     "sandbag_out": r.get("sandbag_out", ""),
                     "st": st, "ht": ht, "dis": dis})
    return d["model"], rows, dropped


def band_indices(L, lo_frac, hi_frac):
    lo = max(0, int(math.floor(lo_frac * L)))
    hi = min(L, int(math.ceil(hi_frac * L)))
    return range(lo, max(hi, lo + 1))


def min_rank(traj, idxs):
    return min(traj[i] for i in idxs)


def auroc(pos, neg):
    """Mann-Whitney AUROC with tie correction. pos/neg: higher score = more positive."""
    if not pos or not neg:
        return float("nan")
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    # assign average ranks
    ranks, i = {}, 0
    n = len(allv)
    rank_sum_pos = 0.0
    while i < n:
        j = i
        while j < n and allv[j][0] == allv[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # 1-based average rank
        rank_sum_pos += avg_rank * sum(lab for _, lab in allv[i:j])
        i = j
    n1, n0 = len(pos), len(neg)
    u = rank_sum_pos - n1 * (n1 + 1) / 2.0
    return u / (n1 * n0)


def band_metrics(rows, lo, hi, k_list=(5, 20)):
    comp = [r for r in rows if r["complied"]]
    out = {"n_complied": len(comp)}
    pos, neg = [], []
    persist = {k: 0 for k in k_list}
    dis_hits = {k: 0 for k in k_list}
    dis_total = 0
    honest_vis = 0
    for r in rows:
        L = len(r["ht"][r["a"]])
        idxs = band_indices(L, lo, hi)
        if min_rank(r["ht"][r["a"]], idxs) < 20:
            honest_vis += 1
    out["honest_vis20"] = honest_vis
    out["n_rows"] = len(rows)
    for r in comp:
        L = len(r["st"][r["a"]])
        idxs = band_indices(L, lo, hi)
        tr = min_rank(r["st"][r["a"]], idxs)
        pos.append(-tr)
        for k in k_list:
            persist[k] += int(tr < k)
        for t in r["dis"]:
            dr = min_rank(r["st"][t], idxs)
            neg.append(-dr)
            dis_total += 1
            for k in k_list:
                dis_hits[k] += int(dr < k)
    out["auroc"] = auroc(pos, neg)
    for k in k_list:
        out[f"persist{k}"] = persist[k]
        out[f"dis_fpr{k}"] = dis_hits[k] / dis_total if dis_total else float("nan")
    out["dis_total"] = dis_total
    return out


def main():
    model, rows, dropped = load_rows(IN_PATH)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    L = len(rows[0]["ht"][rows[0]["a"]])

    named_bands = {
        "pilot (0.25-1.0)": (0.25, 1.0),
        "q1 (0-0.25)": (0.0, 0.25),
        "q2 (0.25-0.5)": (0.25, 0.5),
        "q3 (0.5-0.75)": (0.5, 0.75),
        "q4 (0.75-1.0)": (0.75, 1.0),
        "late (0.6-0.95)": (0.6, 0.95),
        "no-tail (0.25-0.9)": (0.25, 0.9),
    }
    named = {name: band_metrics(rows, lo, hi) for name, (lo, hi) in named_bands.items()}

    # sliding window: width 0.15 depth, step 0.05
    sweep = []
    w, step = 0.15, 0.05
    lo = 0.0
    while lo + w <= 1.0 + 1e-9:
        m = band_metrics(rows, lo, lo + w)
        m["lo"], m["hi"], m["center"] = round(lo, 3), round(lo + w, 3), round(lo + w / 2, 3)
        sweep.append(m)
        lo += step

    result = {"model": model, "n_layers": L, "n_rows": len(rows), "dropped": dropped,
              "named_bands": named, "sliding": sweep,
              "notes": "ranks 0-based; sentinel 99999 = not read; emitted sandbag answer excluded from distractors"}
    with open(f"{OUT_DIR}/depth_sweep_12b.json", "w") as f:
        json.dump(result, f, indent=1)

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6})
    BLUE, RED, GRAY = "#0072B2", "#D55E00", "#808080"

    # Fig 1: depth sweep
    xs = [m["center"] for m in sweep]
    n_comp = sweep[0]["n_complied"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(xs, [m["auroc"] for m in sweep], color=BLUE, lw=2, label="AUROC (true vs distractor)")
    ax.plot(xs, [m["persist20"] / n_comp for m in sweep], color=RED, lw=2,
            label="persistence@20 (complied)")
    ax.plot(xs, [m["dis_fpr20"] for m in sweep], color=GRAY, lw=2, ls="--",
            label="distractor rate@20")
    pl, ph = named_bands["pilot (0.25-1.0)"]
    ax.axvspan(pl, ph, color=BLUE, alpha=0.06, lw=0)
    ax.set_xlabel("fractional depth of read band (width 0.15)")
    ax.set_ylabel("rate / AUROC")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_title(f"Sandbagged-answer readability vs depth ({model.split('/')[-1]}, n={n_comp})",
                 fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIG_DIR}/depth_sweep_12b.{ext}", dpi=200)
    plt.close(fig)

    # Fig 2: single-item trajectory (nitrogen)
    r0 = next(r for r in rows if r["a"] == "nitrogen")
    Lr = len(r0["ht"][r0["a"]])
    xs2 = list(range(Lr))
    def clean(tr):
        return [v if v != SENTINEL else None for v in tr]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    hon = clean(r0["ht"][r0["a"]]); snd = clean(r0["st"][r0["a"]])
    ax.plot(xs2, [v + 1 if v is not None else None for v in hon], color=BLUE, lw=2,
            marker="o", ms=3.5, label="honest: rank of 'nitrogen'")
    ax.plot(xs2, [v + 1 if v is not None else None for v in snd], color=RED, lw=2,
            marker="o", ms=3.5, label="sandbag: rank of 'nitrogen'")
    # distractor envelope (min across distractors per layer)
    denv = []
    for i in range(Lr):
        vals = [r0["st"][t][i] for t in r0["dis"] if r0["st"][t][i] != SENTINEL]
        denv.append(min(vals) + 1 if vals else None)
    ax.plot(xs2, denv, color=GRAY, lw=1.5, ls="--", marker="s", ms=3.5,
            label="sandbag: best distractor")
    ax.set_yscale("log")
    ax.set_ylim(1, 3000)
    ax.invert_yaxis()
    ax.set_xlabel("layer")
    ax.set_ylabel("lens rank of token (log, 1 = top)")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_title("Q: largest gas in atmosphere; sandbag output: 'Oxygen'", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIG_DIR}/traj_nitrogen_12b.{ext}", dpi=200)
    plt.close(fig)

    # ---- markdown summary ----
    lines = ["# Week-1 offline analysis: depth sweep and AUROC rebuild (gemma-4-12B-it)", "",
             f"Source: `{IN_PATH}` (n_rows={len(rows)}, dropped={dropped} duplicate/collision rows, "
             f"{L} layers). Sandbag metrics on complied rows (n={n_comp}). The token the model "
             "actually emitted while sandbagging is excluded from distractor sets.", "",
             "| band | AUROC | persist@5 | persist@20 | distractor@20 | honest vis@20 |",
             "|---|---|---|---|---|---|"]
    for name, m in named.items():
        lines.append(f"| {name} | {m['auroc']:.3f} | {m['persist5']}/{m['n_complied']} | "
                     f"{m['persist20']}/{m['n_complied']} | {m['dis_fpr20']:.3f} | "
                     f"{m['honest_vis20']}/{m['n_rows']} |")
    lines += ["", "Figures: `paper/figures/depth_sweep_12b.*`, `paper/figures/traj_nitrogen_12b.*`.",
              "Full sliding-window numbers: `results/analysis/depth_sweep_12b.json`."]
    with open("docs/week1-results.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps({k: v for k, v in result["named_bands"].items()}, indent=1)[:1200])
    print("wrote", f"{OUT_DIR}/depth_sweep_12b.json", "and figures")


if __name__ == "__main__":
    main()
