# Do demographic probes survive a change of hospital?

A linear probe that reads sex or age off a frozen medical-image embedding is a standard tool: people
use it to measure how much demographic information a representation carries, and to define the
direction they then project out to "remove" it. Almost always, the probe is trained and used in the
same clinical setting.

This repository holds the code and the aggregate results behind a study of what happens when you
move that probe somewhere else — a probe trained on clinic fundus photographs (BRSET) applied to a
handheld-camera screening cohort (mBRSET), and the same test in reverse.

The short version: the probe stops working, matching the feature distributions does not bring it
back, and only target labels do — for age mostly, for sex barely.

## What is in here

Release v1.1 adds the repairs the reviewers of a first draft would ask for: besides diagonal
mean-variance matching and CORAL, the alignment test now covers subspace alignment, the Gaussian
optimal-transport map and an entropic optimal-transport barycentric map — the last one is not
affine, so it is not covered by the argument that bounds the affine family — and self-training now
runs in two schemes, one pseudo-labelling every target patient and one keeping only the confident
half. None of them closes more than a quarter of the transport penalty in any setting.

Release v1.2 changes no result. It drops the outputs of a direction-removal analysis that the paper
no longer reports, adds the data-access links below, and repairs the job scripts that had picked up
a wrong variable name and an internal default path.

`scripts/` has the analyses: the transport measurement, the alignment and self-training repairs, the
few-shot re-estimation, and the script that turns the result tables into the numbers, tables and
figures of the paper. `slurm/` has the job scripts we ran on MeluXina; they take the cluster account
and paths from the environment, so you will need to adjust them for your own machine. `tests/` runs
the whole chain on synthetic data, including a test that deliberately breaks the reproduction gate
to check that it fails. `results/` holds every aggregate the paper reports.

## What is deliberately not in here

No data. BRSET, mBRSET and the released embeddings are credentialed PhysioNet resources, and the
data use agreement does not allow us to redistribute them, in whole or in part. That rules out the
metadata tables, the embeddings, the fitted models and any row-level prediction. Everything under
`results/` is an aggregate over patients; the analysis scripts refuse to write anything else, and
`validate_icassp_outputs.py` fails if a column that looks like an identifier appears.

If you want to reproduce the study, get credentialed on PhysioNet, sign the agreement, download the
data yourself and keep it in access-controlled storage.

The analysis reads the released embeddings, not the images. Apply for access at the source, which is
where the licence is granted:

- Embeddings (what these scripts read): *Embedding-Based Representations for BRSET and mBRSET*
  v1.0.0, PhysioNet, credentialed access — <https://doi.org/10.13026/1h4p-vz70>
- BRSET: Nakayama et al., *PLOS Digital Health* 2024 — <https://doi.org/10.1371/journal.pdig.0000454>
- mBRSET: Wu et al., *Scientific Data* 2025 — <https://doi.org/10.1038/s41597-025-04627-3>

This repository is archived at <https://doi.org/10.5281/zenodo.22794930>; that DOI always resolves to
the latest version.

## Running it

Python 3.13 with numpy, pandas, scipy, scikit-learn and matplotlib; the exact versions we used are
in `environment.txt` and `requirements.txt`.

The tests need no data at all:

    python -m unittest tests.test_icassp_extensions
    python -m unittest tests.test_icassp_recovery

With the data in place, the two analyses and the validator are:

    export DATA_ROOT=<your data directory>
    python scripts/run_icassp_extensions.py --data-root "$DATA_ROOT" \
        --output-dir results/extension \
        --reference-probe-metrics results/audit/demographic_probe_metrics.csv
    python scripts/run_icassp_recovery_probe.py --data-root "$DATA_ROOT" \
        --output-dir results/recovery
    python scripts/validate_icassp_outputs.py --results-dir results/extension \
        --require-reproduction-gate

and the paper's numbers, tables and figures come from

    python scripts/generate_icassp_assets.py --extension-dir results/extension \
        --recovery-dir results/recovery --audit-dir results/audit \
        --cohort-summary results/audit/cohort_summary.csv --paper-dir paper

One warning about reproducing our numbers exactly: the logistic probes are fitted with
scikit-learn's lbfgs at its default tolerance, and where it stops moves slightly with the BLAS thread
count. We learned this the hard way — 32 threads against 16 was enough to break a 1e-6 reproduction
gate. The shipped results were produced with 16 threads.

## How the numbers are kept honest

Every number in the paper is a macro written by `generate_icassp_assets.py`, so no result is typed by
hand. The same script also asserts the paper's qualitative claims — that the penalty stays positive
in every stratum, that self-training changes nothing worth reporting, and so on — and exits non-zero
if a regeneration falsifies one of them. Split 0 of the main analysis has to reproduce an earlier
audit of the same cohorts exactly, or the run stops. Shifting the target vectors, which cannot change
a linear probe's AUROC, is checked to eight decimal places.

## Citing this work

A paper describing the study is under review; the citation will be added here once it is settled.

## License

MIT, see `LICENSE`. The datasets are not redistributed here and stay under their own PhysioNet terms.
