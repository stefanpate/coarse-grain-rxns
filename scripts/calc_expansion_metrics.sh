#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 50
#SBATCH --mem=0
#SBATCH -t 12:00:00
#SBATCH --job-name="calc_exp_metrics"
#SBATCH --output=/home/spn1560/coarse-grain-rxns/logs/out/%A
#SBATCH --error=/home/spn1560/coarse-grain-rxns/logs/error/%A
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu

# Args
script=/home/spn1560/coarse-grain-rxns/scripts/calc_expansion_metrics.py

# Commands
ulimit -c 0
module purge
source /home/spn1560/coarse-grain-rxns/.venv/bin/activate
python $script