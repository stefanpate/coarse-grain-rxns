#!/bin/bash
#SBATCH -A p30041
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=20GB
#SBATCH -t 4:00:00
#SBATCH --job-name="write_rules"
#SBATCH --output=/home/spn1560/coarse-grain-rxns/logs/out/%x_%A.out
#SBATCH --error=/home/spn1560/coarse-grain-rxns/logs/error/%x_%A.err
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu

# Args
script=/home/spn1560/coarse-grain-rxns/scripts/write_mechinferred_rules.py

# Commands
ulimit -c 0
module purge
source /home/spn1560/coarse-grain-rxns/.venv/bin/activate
python $script