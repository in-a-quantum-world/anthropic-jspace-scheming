#!/usr/bin/env python3
"""Regenerate every reported table from the saved results, as markdown.

One command, no GPU: python3 src/regenerate_tables.py
Covers: pilot summaries, the extended recovery run, the 12B depth-sweep named
bands (docs/week1-results.md), the week-1 findings tables (docs/week1-findings.md),
the review-response re-analyses (docs/review-response.md), and the Dolos arm.
Metric definitions come from src/metrics_lib.py and src/week1_depth_sweep.py;
if a printed number differs from a committed document, the document is wrong or
the metric changed, and the discrepancy belongs in docs/analysis-log.md, never
in a silent edit.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics_lib as M  # noqa: E402
import week1_depth_sweep as DS  # noqa: E402

W1 = "results/20261025-1052-week1"
DOLOS = "results/20261025-1807-week1/dolos_explore.json"


def h(title):
    print(f"\n## {title}\n")


def pilot_tables():
    h("Pilot run 20261025-0111 (stored summaries)")
    print("| model | known/bank | visible | comply | persist@20 |")
    print("|---|---|---|---|---|")
    for f in sorted(glob.glob("results/20261025-0111-pilot/pilot_results_*.json")):
        d = json.load(open(f))
        print(f"| {d['choice']} | {d['n_known']}/{d['n_bank']} | {d['pc_visible']} "
              f"| {d['compliance']} | {d['persist']} |")
    f = "results/extended/extended_results_12b.json"
    if os.path.exists(f):
        d = json.load(open(f))
        h("Extended recovery run (12B, random distractors, stored)")
        print(f"n={d['n']}, rec@1 {d['rec1']}, rec@5 {d['rec5']}, rec@20 {d['rec20']}, "
              f"distractor-in-top-20 {d['distractor20']}/{d['distractor_total']}")


def depth_sweep_table():
    h("12B depth sweep, named bands (docs/week1-results.md)")
    model, rows, dropped = DS.load_rows(DS.IN_PATH)
    named_bands = {
        "pilot (0.25-1.0)": (0.25, 1.0), "q1 (0-0.25)": (0.0, 0.25),
        "q2 (0.25-0.5)": (0.25, 0.5), "q3 (0.5-0.75)": (0.5, 0.75),
        "q4 (0.75-1.0)": (0.75, 1.0), "late (0.6-0.95)": (0.6, 0.95),
        "no-tail (0.25-0.9)": (0.25, 0.9),
    }
    print(f"(n_rows={len(rows)}, dropped={dropped})\n")
    print("| band | AUROC | persist@5 | persist@20 | distractor@20 | honest vis@20 |")
    print("|---|---|---|---|---|---|")
    for name, (lo, hi) in named_bands.items():
        m = DS.band_metrics(rows, lo, hi)
        print(f"| {name} | {m['auroc']:.3f} | {m['persist5']}/{m['n_complied']} | "
              f"{m['persist20']}/{m['n_complied']} | {m['dis_fpr20']:.3f} | "
              f"{m['honest_vis20']}/{m['n_rows']} |")


def week1_tables():
    files = sorted(glob.glob(f"{W1}/week1_*.json"))
    data = {json.load(open(f))["choice"]: json.load(open(f)) for f in files}
    order = [c for c in ("e4b", "12b", "qwen", "31b") if c in data]

    h("Week-1 registered results (docs/week1-findings.md, J-lens, primary band)")
    print("| model | known | comply | persist@20 | AUROC (J-lens) |")
    print("|---|---|---|---|---|")
    for c in order:
        d = data[c]
        rows = d["rows"]
        m = M.item_metrics(rows, "lens")
        print(f"| {c} | {d['n_known']}/{d['n_bank']} | {m['n_complied']}/{len(rows)} | "
              f"{m['persist20']}/{m['n_complied']} ({m['persist20']/max(m['n_complied'],1):.2f}) | "
              f"{m['pooled_auroc']:.3f} |")
        recount = sum(M.answers_match(r["a"], r["sandbag_out"]) is False for r in rows)
        stored = sum(r["complied"] for r in rows)
        if recount != stored:
            print(f"  (digit-aware compliance recount for {c}: {recount} vs stored {stored})")

    h("Neighbour elevation, honest vs sandbag (J-lens, distractors@20 in band)")
    print("| model | honest | sandbag |")
    print("|---|---|---|")
    for c in order:
        rows = data[c]["rows"]
        hh, ht = M.neighbour_rate_honest(rows, "lens")
        m = M.item_metrics(rows, "lens")
        print(f"| {c} | {hh}/{ht} ({hh/max(ht,1):.3f}) | "
              f"{m['dis20']}/{m['dis_total']} ({m['dis20']/max(m['dis_total'],1):.3f}) |")

    h("Lens robustness on qwen (pooled AUROC per readout, primary band)")
    if "qwen" in data:
        rows = data["qwen"]["rows"]
        readouts = sorted({k.rsplit("_", 1)[0] for k in rows[0] if k.endswith("_honest")})
        for ro in readouts:
            m = M.item_metrics(rows, ro)
            vis = M.honest_visibility(rows, ro)
            print(f"- {ro}: AUROC {m['pooled_auroc']:.3f}, honest vis {vis}/{len(rows)}")

    h("Review-response re-analyses (docs/review-response.md, J-lens)")
    print("| model | pooled | pooled emitted-retained | matched | matched emitted-retained | truth beats own lie |")
    print("|---|---|---|---|---|---|")
    for c in order:
        rows = data[c]["rows"]
        ex = M.item_metrics(rows, "lens", include_emitted=False)
        inc = M.item_metrics(rows, "lens", include_emitted=True)
        print(f"| {c} | {ex['pooled_auroc']:.3f} | {inc['pooled_auroc']:.3f} | "
              f"{ex['matched_auroc']:.3f} | {inc['matched_auroc']:.3f} | "
              f"{inc['beat_emitted']}/{inc['beat_emitted_n']} |")


def dolos_table():
    if not os.path.exists(DOLOS):
        return
    h("Dolos exploratory arm (read-only; too few complied items to quote AUROCs)")
    d = json.load(open(DOLOS))
    print("| arm | known | complied (stored) | complied (digit-aware recount) |")
    print("|---|---|---|---|")
    for which, blob in d.items():
        rows = blob["rows"]
        stored = sum(r["complied"] for r in rows)
        recount = sum(not M.answers_match(r["a"], r["sandbag_out"]) for r in rows)
        print(f"| {which} | {len(rows)} | {stored} | {recount} |")


def main():
    print("# Regenerated tables (from results/, "
          "definitions: src/metrics_lib.py + src/week1_depth_sweep.py)")
    pilot_tables()
    depth_sweep_table()
    week1_tables()
    dolos_table()


if __name__ == "__main__":
    main()
