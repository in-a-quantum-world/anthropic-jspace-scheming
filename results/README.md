# Results convention

Each run has one folder, `results/<YYYYMMDD-HHMM>-<model>/`, containing

- `pilot_results.json` (or `sweep_results.json` from phase 2 onward): raw, machine-written, and never edited by hand
- `notes.md`: the three gate numbers together with hand-written transcript observations

Runs are pushed on `run/<id>` branches by the notebook's publish cell and merged to `main` only after a human has read the transcripts. The raw JSON is authoritative; anything in the write-up must trace back to a file here.
