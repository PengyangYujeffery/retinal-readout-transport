#!/bin/bash -l
#SBATCH --job-name=icassp-packed
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=logs/icassp-packed-%j.out
#SBATCH --error=logs/icassp-packed-%j.err
# CPU-only workload. The group's CPU allocation (8 node-hours a month) cannot hold it, and GPU
# nodes are billed per whole node, so both analyses share one node; the GPUs stay idle.

set -Eeuo pipefail

project_root="${PROJECT_ROOT:?Set PROJECT_ROOT to the repository path}"
data_root="${DATA_ROOT:?Set DATA_ROOT to the protected data path}"
extension_output="${ICASSP_OUTPUT_DIR:-$project_root/outputs/icassp}"
recovery_output="${RECOVERY_OUTPUT_DIR:-$project_root/outputs/icassp_recovery}"
reference="${ICASSP_REFERENCE:-$project_root/outputs/representation_submission_20260908/demographic_probe_metrics.csv}"
threads=$(( ${SLURM_CPUS_PER_TASK:-64} / 2 ))

cd "$project_root"
mkdir -p logs "$extension_output" "$recovery_output"
source "$project_root/.venv/bin/activate"
test -f "$reference"
printf 'extension_output=%s\nrecovery_output=%s\nreference=%s\nthreads_each=%s\n' \
  "$extension_output" "$recovery_output" "$reference" "$threads"

(
  export OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
  export OPENBLAS_NUM_THREADS="$threads" NUMEXPR_NUM_THREADS="$threads"
  python scripts/run_icassp_extensions.py \
    --data-root "$data_root" \
    --output-dir "$extension_output" \
    --reference-probe-metrics "$reference"
  python scripts/validate_icassp_outputs.py \
    --results-dir "$extension_output" \
    --require-reproduction-gate
) > "logs/icassp-extension-packed-${SLURM_JOB_ID}.log" 2>&1 &
extension_pid=$!

(
  export OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
  export OPENBLAS_NUM_THREADS="$threads" NUMEXPR_NUM_THREADS="$threads"
  python scripts/run_icassp_recovery_probe.py \
    --data-root "$data_root" \
    --output-dir "$recovery_output"
) > "logs/icassp-recovery-packed-${SLURM_JOB_ID}.log" 2>&1 &
recovery_pid=$!

extension_status=0
wait "$extension_pid" || extension_status=$?
recovery_status=0
wait "$recovery_pid" || recovery_status=$?

printf 'extension_exit=%s\nrecovery_exit=%s\n' "$extension_status" "$recovery_status"
tail -3 "logs/icassp-extension-packed-${SLURM_JOB_ID}.log"
tail -3 "logs/icassp-recovery-packed-${SLURM_JOB_ID}.log"
test "$extension_status" -eq 0
test "$recovery_status" -eq 0
