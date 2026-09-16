#!/bin/bash -l
#SBATCH --job-name=recovery-probe
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/recovery-probe-%j.out
#SBATCH --error=logs/recovery-probe-%j.err

set -Eeuo pipefail

project_root="${PROJECT_ROOT:?Set PROJECT_ROOT to the repository path}"
data_root="${DATA_ROOT:?Set DATA_ROOT to the protected data path}"
output_dir="${RECOVERY_OUTPUT_DIR:-$project_root/outputs/icassp_recovery}"

cd "$project_root"
mkdir -p logs "$output_dir"
source "$project_root/.venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

printf 'output_dir=%s\n' "$output_dir"
python scripts/run_icassp_recovery_probe.py \
  --data-root "$data_root" \
  --output-dir "$output_dir"
