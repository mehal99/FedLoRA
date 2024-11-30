#!/bin/bash
#SBATCH --job-name=replicate_fed
#SBATCH --mem=150G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:A100_40GB:1
#SBATCH --nodes=1
#SBATCH --partition=general


cd /data/tir/projects/tir7/user_data/kramanet/DistML Project/alt/FedLoRA

source activate fed_shepherd

python main.py --global_model 'Qwen/Qwen2.5-3B'\
      --data_path  "./data" \
      --output_dir  './lora-shepherd-7b/'\
      --num_communication_rounds 10 \
      --num_clients  10 \
      --train_on_inputs \
      --group_by_length \
      --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]'