#!/bin/bash
# SLURM job script for few-shot inference
# Runs run.py in few-shot mode on the Italian compressed dataset
# Adjust paths and --model-id as needed before submitting

#SBATCH --job-name=comp_llama_it_few_base
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1          # request 1 GPU
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/3149671/misogyny_compressed/llama_it_baseline/fewshot_%j.out
#SBATCH --error=/home/3149671/misogyny_compressed/llama_it_baseline/fewshot_%j.err

/home/3149671/.conda/envs/olmo/bin/python /home/3149671/misogyny_compressed/llama_it_baseline/run.py \
  --input-csv /home/3149671/ita_compressed.csv \
  --out-csv /home/3149671/misogyny_compressed/output/llama_it_fewshot_baseline.csv \
  --mode fewshot \
  --model-id meta-llama/Llama-3.1-8B-Instruct \
  --dtype bfloat16
