# Matched-parameter benchmarking of quantum and classical classifier heads

Code, per-run results, and pre-registration documents for the paper *Two
Methodological Pitfalls in Matched-Parameter Quantum–Classical Benchmarking for
Paraphrase Identification: Class-Imbalance Collapse and Readout-Offset
Asymmetry*.

The study compares a 32-parameter data-reuploading PQC head against a matched
31-parameter MLP head on MRPC, PAWS and QQP, using a frozen SBERT encoder and a
197-parameter depthwise reducer. It reports 352 training runs across 42
experimental conditions. Every number in the paper comes from the JSON files in
`results/`, and the scripts below recompute them.

## Layout

```
src/          model, data loading, and the two-phase trainers
scripts/      sweep drivers (run_*.sh) and analysis (analyse_*, stats_*, verify_*)
results/      one JSON per training run, grouped by protocol version
preregistration/  the two registration documents
logs/         sweep driver logs, kept because they timestamp run ordering
```

Protocol versions in `results/`:

| Directory | What it holds |
|---|---|
| `v2`, `v3_uncorrected` | Unweighted-BCE runs. These are the collapsed runs reported as Pitfall 1. |
| `v4` | Class-weighted BCE, no calibration. |
| `v5` | Adds the two-parameter affine calibration. |
| `v7` | Mechanism experiments: depth sweep, second ansatz, initialisation scale. |
| `v8` | Bias-only ablation and the 16-qubit mid-capacity control. |

## Reproducing the numbers

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Then, from the repository root:

| Command | Recomputes |
|---|---|
| `python scripts/verify_paper_numbers.py` | Every headline figure, straight from the run files |
| `python scripts/stats_rigor_v6.py` | The consolidated statistics table, with bootstrap CIs and Holm correction |
| `python scripts/analyse_v5.py` | The calibration-lever comparison |
| `python scripts/stats_v5_pertask.py` | Per-head changes and the interaction test |
| `python scripts/analyse_v7_mechanism.py` | Circuit-depth sweep |
| `python scripts/analyse_v7_sel.py` | Second ansatz family |
| `python scripts/analyse_v8_midcap.py` | Bias-only ablation and the 16-qubit control |
| `python scripts/rescore_qqp_matched.py` | The like-for-like QQP comparison on a matched test subsample |
| `python scripts/readout_offset_v7.py` | Initialisation-time readout probe |
| `python scripts/noise_recalibrated_v7.py` | Depolarizing-noise sweep with re-fitted threshold |
| `python scripts/stats_paired_v9.py` | Paired seed-level tests and the 15-cell Holm family |
| `python scripts/stats_example_level_v9.py` | Item-clustered and hierarchical statistics for the head comparisons |
| `python scripts/stats_example_level_mech_v9.py` | Item-level calibration-mechanism contrasts |
| `python scripts/metrics_prc_v9.py` | Precision-recall area and class-wise precision and recall |
| `python scripts/grad_analysis_v9.py` | Jacobian and uniform-shift analysis on the released MRPC features |
| `python scripts/hw_readout_probe.py --backend aer --n-features 16` | Exact Qiskit Aer readout probe on the same released features |

These read the stored results and need no GPU. Re-running the experiments
themselves uses the `scripts/run_*.sh` drivers and takes considerably longer;
the 16-qubit runs are roughly three minutes per epoch on CPU.

`results/mrpc_depthwise_q8_s0_features.npy` contains the 200 reduced feature
vectors used by the gradient and Aer probes. Its companion JSON records the
dataset selection, checkpoint, encoder and SHA-256 digest. The probe scripts
fail if this snapshot is absent; they do not substitute synthetic features.

## Pre-registration

`preregistration/` holds both registration documents. Their evidential standing
is uneven and the paper says so in the section on Pitfall 2.

`PREREGISTRATION_v5.md` predates the first calibrated run by two minutes, and
that ordering is corroborated by `logs/hybrid_v5_multiseed.log`, which was
written by a separate process. `PREREGISTRATION_v7_mechanism.md` predates the
E4 runs, but the E1 and E2 sweeps had already started when it was last amended,
so those two predictions are pre-specified rather than independently
timestamped. Neither document carries an external registry timestamp. Anyone
who discounts self-timestamped registrations should read the affected sections
as exploratory.

`results/v2/checkpoints` and `results/v4/checkpoints` hold the frozen reducer
weights (about 1.3 MB) so Phase 2 can be rerun without repeating Phase 1. They
are PyTorch state dictionaries; load them with `weights_only=True`.

The `v2` and `v3_uncorrected` directories hold the 36 unweighted-BCE runs that
document the collapse. They are superseded as results but are the evidence for
Pitfall 1, so they ship.

## Not included

The SBERT embedding cache (about 560 MB) is omitted because it regenerates from
the encoder on first run. Per-run training logs are omitted for size; the sweep
driver logs are kept because they establish run ordering.

## Datasets

MRPC, PAWS and QQP are obtained through the Hugging Face Datasets library and
are not redistributed here.

## Licence

Code in `src/` and `scripts/` is MIT licensed. The result files in `results/`
and the documents in `preregistration/` are CC BY 4.0. See `LICENSE`.
