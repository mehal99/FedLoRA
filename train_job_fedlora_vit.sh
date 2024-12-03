#!/bin/bash

#SBATCH -J training_fedlora_vit
#SBATCH -o training_fedlora_vit_%A_%a.out
#SBATCH -e training_fedlora_vit_%A_%a.err
#SBATCH --nodes 1
#SBATCH --cpus-per-task 2
#SBATCH --mem=16G
#SBATCH --gres=gpu:H100:1

source activate base
conda activate fedlora

apptainer exec --nv /home/sagnikg/containers/miniconda3_latest.sif bash -c "
    source activate base
    conda activate fedlora
    export SSL_CERT_FILE=\$(python -m certifi)
    python main.py --num_communication_rounds 10 --num_clients 2 --client_selection_frac 1 --alpha 100
"