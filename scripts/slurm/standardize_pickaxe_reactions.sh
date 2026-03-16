#!/bin/bash
#SBATCH -A p30041
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=20G
#SBATCH -t 4:00:00
#SBATCH --job-name="std_pk_rxns"
#SBATCH --output=/home/spn1560/coarse-grain-rxns/logs/out/%A_%a
#SBATCH --error=/home/spn1560/coarse-grain-rxns/logs/error/%A_%a
#SBATCH --array=0-12
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu

script=/home/spn1560/coarse-grain-rxns/scripts/standardize_pickaxe_reactions.py
expansions=(
    1_steps_after_2015_cpds_rules_mechinferred_dt_069_rules_before_2015_w_coreactants_aplusb_True.pk
)
ulimit -c 0
module purge
source /home/spn1560/coarse-grain-rxns/.venv/bin/activate
python $script expansion=${expansions[$SLURM_ARRAY_TASK_ID]}
