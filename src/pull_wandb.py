#!/usr/bin/env python3
"""Pull every run in the W&B project into local files, so results survive
outside W&B and go into the repo. For each run: its config + summary into a
JSON, and its full per-step history (loss, lock curve, retention curve) into a
CSV. Run from the repo root after `wandb login` (or with WANDB_API_KEY set):

  python src/pull_wandb.py --entity quantarune-imperial-college-london --project jspace-sandbagging

Writes to results/wandb/<run-name>.{json,csv}. Add --artifacts to also download
each run's model/lens artifacts (large).
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--artifacts", action="store_true", help="also download model/lens artifacts (large)")
    ap.add_argument("--out", default=os.path.join("results", "wandb"))
    args = ap.parse_args()

    import wandb
    api = wandb.Api()
    os.makedirs(args.out, exist_ok=True)
    runs = api.runs(f"{args.entity}/{args.project}")
    print(f"{len(runs)} runs in {args.entity}/{args.project}")
    for r in runs:
        base = os.path.join(args.out, r.name.replace("/", "_"))
        with open(base + ".json", "w") as f:
            json.dump({"name": r.name, "id": r.id, "state": r.state, "tags": r.tags,
                       "created": str(r.created_at), "url": r.url,
                       "config": {k: v for k, v in r.config.items() if not k.startswith("_")},
                       "summary": {k: v for k, v in r.summary.items() if not k.startswith("_")}},
                      f, indent=1, default=str)
        hist = r.history(pandas=True)
        if hist is not None and len(hist):
            hist.to_csv(base + ".csv", index=False)
        print(f"  {r.name:48s} {r.state:9s} {len(hist) if hist is not None else 0} steps")
        if args.artifacts:
            for a in r.logged_artifacts():
                d = a.download(root=os.path.join(args.out, "artifacts", a.name.replace(":", "_")))
                print(f"    artifact {a.name} -> {d}")
    print(f"\nwrote {args.out}/")


if __name__ == "__main__":
    main()
