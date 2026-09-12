#!/bin/bash -l
#SBATCH --job-name=icassp-retina
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/icassp-extension-%j.out
#SBATCH --error=logs/icassp-extension-%j.err

set -Eeuo pipefail

project_root="${AICS_PROJECT_ROOT:?Set AICS_PROJECT_ROOT to the repository path}"
data_root="${AICS_DATA_ROOT:?Set AICS_DATA_ROOT to the protected data path}"
output_dir="${ICASSP_OUTPUT_DIR:-$project_root/outputs/icassp}"
reference="${ICASSP_REFERENCE:-$project_root/outputs/representation_submission_20260908/demographic_probe_metrics.csv}"
split_repeats="${ICASSP_SPLIT_REPEATS:-5}"
cv_repeats="${ICASSP_CV_REPEATS:-5}"
bootstrap_replicates="${ICASSP_BOOTSTRAP_REPLICATES:-2000}"
seed="${ICASSP_SEED:-20260902}"

cd "$project_root"
mkdir -p logs "$output_dir"
source "$project_root/.venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

printf 'output_dir=%s\nreference=%s\nsplit_repeats=%s\ncv_repeats=%s\nbootstrap_replicates=%s\nseed=%s\n' \
  "$output_dir" "$reference" "$split_repeats" "$cv_repeats" "$bootstrap_replicates" "$seed"
test -f "$reference"

python scripts/run_icassp_extensions.py \
  --data-root "$data_root" \
  --output-dir "$output_dir" \
  --split-repeats "$split_repeats" \
  --cv-repeats "$cv_repeats" \
  --bootstrap-replicates "$bootstrap_replicates" \
  --seed "$seed" \
  --reference-probe-metrics "$reference"

python scripts/validate_icassp_outputs.py \
  --results-dir "$output_dir" \
  --expected-splits "$split_repeats" \
  --expected-bootstrap "$bootstrap_replicates" \
  --require-reproduction-gate
