#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=50GB
#SBATCH -t 4:00:00
#SBATCH --job-name="batch_up"
#SBATCH --output=/home/spn1560/coarse-grain-rxns/logs/out/%A
#SBATCH --error=/home/spn1560/coarse-grain-rxns/logs/error/%A
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu

# Args
script=/home/spn1560/coarse-grain-rxns/scripts/batch_up_rc0.py

# Commands
ulimit -c 0
module purge
source /home/spn1560/coarse-grain-rxns/.venv/bin/activate
python $script