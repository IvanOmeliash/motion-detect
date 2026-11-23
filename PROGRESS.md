# PROGRESS

Refactor of the Colab research notebooks (`best_13.ipynb`, `best_15.ipynb`)
into this repository. Status: complete.

- [x] File map approved; modules ported from notebook cells (see git history / README)
- [x] Colab/Drive dependencies removed; all paths from YAML (`configs/paths.yaml`) or CLI args
- [x] Dead code removed (legacy absdiff path, review UI cells, audit/inspection helpers, "required names" checks, A/B renderers)
- [x] `CompensationPipeline` default `step=1` (matches checkpoints' temporal_step=1); `configs/default.yaml` sets `compensation_step: 1`
- [x] Hard-negative workflow data-free: `review.csv` schema + strict validation + approval (`scripts/mine_hard_negatives.py approve`)
- [x] Tests: 15 pytest tests, CPU-only, synthetic inputs — passing
- [x] README: results table from notebook outputs only, known limitations, TODO (F1 0.872 unverified)
- [x] LICENSE (MIT); data/checkpoints not included, expected layout documented
- [x] Reports transcribed to `docs/reports/`

Open items:
- Re-run metrics on real data/checkpoints to confirm numerical equivalence (not possible in the refactor environment)
- Core ML export/prediction untested (needs macOS + coremltools)
- TODO: F1 0.872 from CV — source report not found; do not publish until a reproducible report exists
