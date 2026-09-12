# Demographic readout transport in frozen retinal embeddings

Code and aggregate results for a study of whether a linear demographic readout (sex, age) learned on
one retinal-imaging cohort remains valid on another, and of what does or does not repair it:
label-free moment alignment (diagonal matching, CORAL), label-free self-training, few-shot
re-estimation (from scratch or anchored to the source readout), and removal of the source-defined
demographic directions.

Cohorts: BRSET (tabletop cameras, ophthalmology centres) and mBRSET (handheld camera, community
diabetes screening), both embedded by frozen DINOv3 ViT-S/16 and ConvNeXt-Tiny encoders.

Paper: to be added.

## What is in this repository

- `scripts/` - the analyses and the asset generator.
- `slurm/` - job scripts (CPU or a packed GPU node); adjust the account and paths for your cluster.
- `tests/` - end-to-end tests on synthetic data, including a test that the reproduction gate fails
  when it should.
- `results/extension/`, `results/recovery/`, `results/aics/` - the aggregate outputs behind every
  number, table and figure in the paper.

## What is not, and cannot be, in this repository

No credentialed data of any kind: no patient or image identifiers, no metadata tables, no released
embeddings, no fitted models, no row-level predictions. BRSET, mBRSET and the embeddings are
available to credentialed PhysioNet users under a data use agreement; obtain them yourself and keep
them in access-controlled storage. Every file under `results/` is an aggregate; the analysis scripts
refuse to write anything else, and `validate_icassp_outputs.py` fails if an identifier-like column
appears.

## Environment

Python 3.13 with numpy, pandas, scipy, scikit-learn and matplotlib; exact versions in
`environment.txt` and `requirements.txt`. Results are sensitive to the solver environment: logistic
readouts are fitted with scikit-learn's lbfgs at its default tolerance, and the stopping point moves
slightly with the BLAS thread count. Reproduce with the same thread count as the run you compare
against (the shipped results used 16 threads).

## Reproducing

1. Tests on synthetic data, no credentialed data required:

       python -m unittest tests.test_icassp_extensions
       python -m unittest tests.test_icassp_recovery

2. Obtain the data (credentialed PhysioNet access required):

       export AICS_DATA_ROOT=<data-root>
       bash scripts/download_physionet_data.sh
       python scripts/verify_data.py --data-root "$AICS_DATA_ROOT"

3. Run the analyses:

       python scripts/run_icassp_extensions.py --data-root "$AICS_DATA_ROOT" \
           --output-dir results/extension \
           --reference-probe-metrics results/aics/demographic_probe_metrics.csv
       python scripts/run_icassp_recovery_probe.py --data-root "$AICS_DATA_ROOT" \
           --output-dir results/recovery

4. Check the aggregate contract and the reproduction gate:

       python scripts/validate_icassp_outputs.py --results-dir results/extension \
           --require-reproduction-gate

5. Regenerate the paper's numbers, tables and figures:

       python scripts/generate_icassp_assets.py --extension-dir results/extension \
           --recovery-dir results/recovery --aics-dir results/aics \
           --cohort-summary results/aics/cohort_summary.csv --paper-dir paper

`generate_icassp_assets.py` writes `numbers.tex` (every number in the paper is a macro from that
file), the two tables, the two figures, a provenance file with the md5 of each input, and
`claims_check.csv`: each qualitative statement in the paper is asserted there, and the script exits
non-zero if a regeneration falsifies one.

## Built-in checks

- Reproduction gate: split 0 must reproduce the earlier audit run exactly (AUROC, selected C, n).
- Translation invariance: shifting the target vectors must not change a linear readout's AUROC.
- L2-SP invariants: a zero prior must reproduce scikit-learn; an overwhelming prior must return the
  source readout.
- Aggregate contract: coverage of the full design, finite values, and no identifier-like columns.

## License

Code and aggregate results are released under the MIT License (see `LICENSE`). The underlying
datasets are not redistributed here and remain under their own PhysioNet licences.
