#!/bin/bash

#SBATCH -J training_fedlora_vit
#SBATCH -o training_fedlora_vit_%A_%a.out
#SBATCH -e training_fedlora_vit_%A_%a.err
#SBATCH --nodes 1
#SBATCH --cpus-per-task 2
#SBATCH --mem=16G
#SBATCH --gres=gpu:H100:1
#SBATCH --array=0-3

#hyperparameters
alphas=(0.01 100)
num_clients=(5 10)
diff_quantity=1

total_combinations=$((${#alphas[@]} * ${#num_clients[@]}))

echo "Total combinations: $total_combinations"

# Check if SLURM_ARRAY_TASK_ID is within range
if [ $SLURM_ARRAY_TASK_ID -ge $total_combinations ]; then
    echo "Array task ID out of range! SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
    exit 1
fi

alpha_index=$((SLURM_ARRAY_TASK_ID % ${#alphas[@]}))
num_clients_index=$(( (SLURM_ARRAY_TASK_ID / ${#alphas[@]}) % ${#num_clients[@]} ))

alpha=${alphas[$alpha_index]}
num_client=${num_clients[$num_clients_index]}

source activate base
conda activate fedlora

apptainer exec --nv /home/sagnikg/containers/miniconda3_latest.sif bash -c "
    source activate base
    conda activate fedlora
    export SSL_CERT_FILE=\$(python -m certifi)
    python client_data_allocation.py $num_client $diff_quantity $alpha
    python main.py --num_communication_rounds 5 --num_clients $num_client --client_selection_frac 1 --alpha $alpha
"