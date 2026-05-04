#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 50
#SBATCH --mem=0
#SBATCH -t 14:00:00
#SBATCH --job-name="calc_exp_metrics"
#SBATCH --output=/home/spn1560/coarse-grain-rxns/logs/out/%x_%A_%a.out
#SBATCH --error=/home/spn1560/coarse-grain-rxns/logs/error/%x_%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0-2
#SBATCH --mail-user=stefan.pate@northwestern.edu

# Args
script=/home/spn1560/coarse-grain-rxns/scripts/calc_expansion_metrics.py
processes=50 # Make sure this matches -n above
exp_sweep=(
    2_steps_250728_benchmark_starters_rules_evodex_Cm_rules_original_w_coreactants_aplusb_True.pk
    2_steps_250728_benchmark_starters_rules_evodex_Dm_rules_original_w_coreactants_aplusb_True.pk
    2_steps_250728_benchmark_starters_rules_evodex_Em_rules_original_w_coreactants_aplusb_True.pk
)

mappings_sweep=(
    null
    null
    null
)

# Commands
ulimit -c 0
module purge
source /home/spn1560/coarse-grain-rxns/.venv/bin/activate
python $script mapped_rxns=${mappings_sweep[$SLURM_ARRAY_TASK_ID]} expansion=${exp_sweep[$SLURM_ARRAY_TASK_ID]} processes=$processes