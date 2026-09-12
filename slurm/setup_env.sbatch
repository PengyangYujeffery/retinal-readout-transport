#!/bin/bash -l
#SBATCH --job-name=aics-env
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/setup-%j.out
#SBATCH --error=logs/setup-%j.err

set -Eeuo pipefail

project_root="${AICS_PROJECT_ROOT:?Set AICS_PROJECT_ROOT to the repository path}"
mkdir -p "$project_root/logs"

module load env/release/2025.1
module load Python/3.13.1-GCCcore-14.2.0

python -m venv "$project_root/.venv"
source "$project_root/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install --requirement "$project_root/requirements.txt"
python -c 'import numpy, pandas, scipy, sklearn, matplotlib, seaborn; print("environment ready")'
