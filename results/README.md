# Results convention

One folder per run: `results/<YYYYMMDD-HHMM>-<model>/` containing

- `pilot_results.json` (or `sweep_results.json` from phase 2 on): raw, machine-written, never edited by hand
- `notes.md`: the three gate numbers plus hand-written transcript observations

Runs are pushed on `run/<id>` branches by the notebook's publish cell and merged to `main` only after the transcripts have been read by a human. Raw JSON is the source of truth; anything in the write-up must trace back to a file here.
