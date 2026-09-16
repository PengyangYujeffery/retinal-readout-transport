#!/bin/bash -l
#SBATCH --job-name=icassp-smoke
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/icassp-smoke-%j.out
#SBATCH --error=logs/icassp-smoke-%j.err

set -Eeuo pipefail

project_root="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$project_root"
mkdir -p logs
source "$project_root/.venv/bin/activate"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

# Synthetic toy data only; no credentialed data is read.
python -m unittest -v tests.test_icassp_extensions
