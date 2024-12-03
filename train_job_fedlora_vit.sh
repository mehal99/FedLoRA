#!/bin/bash

#SBATCH -J training_fedlora_vit
#SBATCH -o training_fedlora_vit_%A_%a.out
#SBATCH -e training_fedlora_vit_%A_%a.err
#SBATCH --nodes 1
#SBATCH --cpus-per-task 2
#SBATCH --mem=16G
#SBATCH --gres=gpu:H100:1
#SBATCH --array=0-17

#hyperparameters
client_fractions=(1)
comm_rounds=(50)
num_clients=(1 2 5)
local_learning_rates=(5e-4 1e-4 1e-3)
alphas=(100 0.1)

total_combinations=$((${#client_fractions[@]} * ${#comm_rounds[@]} * ${#num_clients[@]} * ${#local_learning_rates[@]} * ${#alphas[@]}))

echo "Total combinations: $total_combinations"

# Check if SLURM_ARRAY_TASK_ID is within range
if [ $SLURM_ARRAY_TASK_ID -ge $total_combinations ]; then
    echo "Array task ID out of range! SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
    exit 1
fi

client_fraction_index=$((SLURM_ARRAY_TASK_ID % ${#client_fractions[@]}))
comm_rounds_index=$(( (SLURM_ARRAY_TASK_ID / ${#client_fractions[@]}) % ${#comm_rounds[@]} ))
num_clients_index=$(( (SLURM_ARRAY_TASK_ID / (${#client_fractions[@]} * ${#comm_rounds[@]})) % ${#num_clients[@]} ))
local_learning_rate_index=$(( (SLURM_ARRAY_TASK_ID / (${#client_fractions[@]} * ${#comm_rounds[@]} * ${#num_clients[@]})) % ${#local_learning_rates[@]} ))
alpha_index=$(( (SLURM_ARRAY_TASK_ID / (${#client_fractions[@]} * ${#comm_rounds[@]} * ${#num_clients[@]} * ${#local_learning_rates[@]})) % ${#alphas[@]} ))

client_fraction=${client_fractions[$client_fraction_index]}
comm_round=${comm_rounds[$comm_rounds_index]}
num_client=${num_clients[$lnum_clients_index]}
local_learning_rate=${local_learning_rates[$local_learning_rate_index]}
alpha=${alphas[$alpha_index]}

source activate base
conda activate fedlora

apptainer exec --nv /home/sagnikg/containers/miniconda3_latest.sif bash -c "
    source activate base
    conda activate fedlora
    export SSL_CERT_FILE=\$(python -m certifi)
    python main.py --num_communication_rounds $comm_round --num_clients $num_client --local_learning_rate $local_learning_rate --client_selection_frac $client_fraction --alpha $alpha
"