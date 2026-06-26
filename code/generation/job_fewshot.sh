#!/bin/bash
# SLURM job script for few-shot inference
# Adjust paths before submitting

#SBATCH --job-name=rag_implicit_mis_fewshot
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=fewshot_%j.out
#SBATCH --error=fewshot_%j.err

python code/generation/run.py \
  --input-csv /path/to/input.csv \
  --out-csv /path/to/output.csv \
  --mode fewshot \
  --model-id meta-llama/Llama-3.1-8B-Instruct \
  --dtype bfloat16
