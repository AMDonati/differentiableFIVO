#!/bin/bash
python run_fivo.py \
  --mode=train \
  --logdir=/output/fivo/synthetic_1_bs_4_1250it_10p \
  --model=vrnn \
  --bound=fivo \
  --summarize_every=100 \
  --batch_size=4 \
  --num_samples=10 \
  --learning_rate=0.0001 \
  --dataset_path="data/synthetic_model_1" \
  --dataset_type="synthetic" \
  --latent_size=32 \
  --max_steps==1250