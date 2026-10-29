"""Weights & Biases tracking for this repo: one thin layer so the Colab
notebooks and the local 5070 scripts log the same things the same way.

Everything degrades to a no-op when wandb is not installed, when
WANDB_DISABLED=1 is set, or when the API key is missing, so a script that
imports this never fails because of tracking.

Conventions
- One W&B project: "jspace-sandbagging". Runs are distinguished by job_type
  ("train-organism", "fit-lens", "eval-recovery", "sweep") and by tags
  (model, lens, condition).
- Every run's config records the git commit, the sha256 of
  configs/prompts.yaml (prompt strings are the experiment), the seed and the
  data files used. The password string is never logged; only its sha256.
- Checkpoints and fitted lenses go in as artifacts so an evaluation run can
  say exactly which organism and which lens it read (use_artifact links the
  lineage). Artifacts are large; log them once per fit/train, not per epoch.
- Per-item results go in as a wandb.Table (one row per item x condition x
  readout) alongside the summary scalars, so nothing reported is only a mean.
- Raw JSON in results/ stays the source of truth. W&B is the dashboard and
  the lineage record, not the archive.
"""
import hashlib
import os
import subprocess

PROJECT = "jspace-sandbagging"

try:
    import wandb as _wandb
except Exception:  # not installed
    _wandb = None


def _enabled():
    return _wandb is not None and os.environ.get("WANDB_DISABLED", "0") != "1"


def git_commit(repo_dir="."):
    try:
        return subprocess.check_output(
            ["git", "-C", repo_dir, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def file_sha(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:12]
    except Exception:
        return "missing"


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def start_run(job_type, name, config, tags=(), repo_dir=".", group=None):
    """Start a run (or return None when tracking is off). config is a dict;
    the repo provenance fields are added here so callers do not repeat them."""
    if not _enabled():
        return None
    cfg = dict(config)
    cfg.setdefault("git_commit", git_commit(repo_dir))
    cfg.setdefault("prompts_yaml_sha", file_sha(os.path.join(repo_dir, "configs", "prompts.yaml")))
    assert "password" not in cfg, "never log the password; log its sha via text_sha"
    try:
        return _wandb.init(project=PROJECT, job_type=job_type, name=name,
                           config=cfg, tags=list(tags), group=group, reinit=True)
    except Exception as e:  # no key, no network: carry on without tracking
        print(f"[tracking] wandb.init failed, continuing untracked: {e}")
        return None


def log(run, metrics, step=None):
    if run is None:
        return
    run.log(metrics, step=step)


def log_artifact(run, name, kind, paths, metadata=None, aliases=("latest",)):
    """paths: file or directory paths. Returns the artifact or None."""
    if run is None:
        return None
    art = _wandb.Artifact(name, type=kind, metadata=metadata or {})
    for p in paths:
        if os.path.isdir(p):
            art.add_dir(p)
        else:
            art.add_file(p)
    run.log_artifact(art, aliases=list(aliases))
    return art


def use_artifact(run, ref):
    """Download an artifact by 'name:alias' and record the lineage. Returns the
    local directory, or None when tracking is off."""
    if run is None:
        return None
    return run.use_artifact(ref).download()


def log_rows_table(run, key, rows, columns):
    """rows: list of dicts; only `columns` are kept, in that order."""
    if run is None:
        return
    table = _wandb.Table(columns=list(columns))
    for r in rows:
        table.add_data(*[r.get(c) for c in columns])
    run.log({key: table})


def finish(run):
    if run is not None:
        run.finish()


def summary_json(run):
    """The run's summary as a plain dict, for writing into results/notes.md."""
    if run is None:
        return {}
    out = {}
    for k in list(run.summary.keys()):
        v = run.summary[k]
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
    return out
