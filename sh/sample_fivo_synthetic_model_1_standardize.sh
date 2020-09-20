#!/bin/bash
python run_fivo.py \
  --mode=eval \
  --logdir="output/fivo/synthetic_1_bs_4_standardize" \
  --model=vrnn \
  --bound=fivo \
  --summarize_every=100 \
  --batch_size=4 \
  --num_samples=10 \
  --learning_rate=0.0001 \
  --dataset_path="data/synthetic_model_1" \
  --dataset_type="synthetic" \
  --latent_size=32 \
  --split test \
  --standardize=True
python run_fivo.py \
  --mode=sample \
  --logdir="output/fivo/synthetic_1_bs_4_standardize" \
  --model=vrnn \
  --bound=fivo \
  --summarize_every=100 \
  --batch_size=100 \
  --num_samples=1000 \
  --learning_rate=0.0001 \
  --dataset_path="data/synthetic_model_1" \
  --dataset_type="synthetic" \
  --latent_size=32 \
  --split=test \
  --prefix_length=1 \
  --sample_length=1 \
  --standardize=True
python run_fivo.py \
  --mode=sample \
  --logdir="output/fivo/synthetic_1_bs_4_standardize" \
  --model=vrnn \
  --bound=fivo \
  --summarize_every=100 \
  --batch_size=100 \
  --num_samples=1000 \
  --learning_rate=0.0001 \
  --dataset_path="data/synthetic_model_1" \
  --dataset_type="synthetic" \
  --latent_size=32 \
  --split=test \
  --prefix_length=2 \
  --sample_length=1 \
  --standardize=True
python run_fivo.py \
  --mode=sample \
  --logdir="output/fivo/synthetic_1_bs_4_standardize" \
  --model=vrnn \
  --bound=fivo \
  --summarize_every=100 \
  --batch_size=100 \
  --num_samples=1000 \
  --learning_rate=0.0001 \
  --dataset_path="data/synthetic_model_1" \
  --dataset_type="synthetic" \
  --latent_size=32 \
  --split=test \
  --prefix_length=3 \
  --sample_length=1 \
  --standardize=True
python run_fivo.py \
  --mode=sample \
  --logdir="output/fivo/synthetic_1_bs_4_standardize" \
  --model=vrnn \
  --bound=fivo \
  --summarize_every=100 \
  --batch_size=100 \
  --num_samples=1000 \
  --learning_rate=0.0001 \
  --dataset_path="data/synthetic_model_1" \
  --dataset_type="synthetic" \
  --latent_size=32 \
  --split=test \
  --prefix_length=4 \
  --sample_length=1 \
  --standardize=True
python run_fivo.py \
  --mode=sample \
  --logdir="output/fivo/synthetic_1_bs_4_standardize" \
  --model=vrnn \
  --bound=fivo \
  --summarize_every=100 \
  --batch_size=100 \
  --num_samples=1000 \
  --learning_rate=0.0001 \
  --dataset_path="data/synthetic_model_1" \
  --dataset_type="synthetic" \
  --latent_size=32 \
  --split=test \
  --prefix_length=5 \
  --sample_length=1 \
  --standardize=True
python run_fivo.py \
  --mode=sample \
  --logdir="output/fivo/synthetic_1_bs_4_standardize" \
  --model=vrnn \
  --bound=fivo \
  --summarize_every=100 \
  --batch_size=100 \
  --num_samples=1000 \
  --learning_rate=0.0001 \
  --dataset_path="data/synthetic_model_1" \
  --dataset_type="synthetic" \
  --latent_size=32 \
  --split=test \
  --prefix_length=6 \
  --sample_length=1 \
  --standardize=True
python run_fivo.py \
  --mode=sample \
  --logdir="output/fivo/synthetic_1_bs_4_standardize" \
  --model=vrnn \
  --bound=fivo \
  --summarize_every=100 \
  --batch_size=100 \
  --num_samples=1000 \
  --learning_rate=0.0001 \
  --dataset_path="data/synthetic_model_1" \
  --dataset_type="synthetic" \
  --latent_size=32 \
  --split=test \
  --prefix_length=7 \
  --sample_length=1 \
  --standardize=True
python run_fivo.py \
  --mode=sample \
  --logdir="output/fivo/synthetic_1_bs_4_standardize" \
  --model=vrnn \
  --bound=fivo \
  --summarize_every=100 \
  --batch_size=100 \
  --num_samples=1000 \
  --learning_rate=0.0001 \
  --dataset_path="data/synthetic_model_1" \
  --dataset_type="synthetic" \
  --latent_size=32 \
  --split=test \
  --prefix_length=8 \
  --sample_length=1 \
  --standardize=True
python run_fivo.py \
  --mode=sample \
  --logdir="output/fivo/synthetic_1_bs_4_standardize" \
  --model=vrnn \
  --bound=fivo \
  --summarize_every=100 \
  --batch_size=100 \
  --num_samples=1000 \
  --learning_rate=0.0001 \
  --dataset_path="data/synthetic_model_1" \
  --dataset_type="synthetic" \
  --latent_size=32 \
  --split=test \
  --prefix_length=9 \
  --sample_length=1 \
  --standardize=True
python run_fivo.py \
  --mode=sample \
  --logdir="output/fivo/synthetic_1_bs_4_standardize" \
  --model=vrnn \
  --bound=fivo \
  --summarize_every=100 \
  --batch_size=100 \
  --num_samples=1000 \
  --learning_rate=0.0001 \
  --dataset_path="data/synthetic_model_1" \
  --dataset_type="synthetic" \
  --latent_size=32 \
  --split=test \
  --prefix_length=10 \
  --sample_length=1 \
  --standardize=True
python run_fivo.py \
  --mode=sample \
  --logdir="output/fivo/synthetic_1_bs_4_standardize" \
  --model=vrnn \
  --bound=fivo \
  --summarize_every=100 \
  --batch_size=100 \
  --num_samples=1000 \
  --learning_rate=0.0001 \
  --dataset_path="data/synthetic_model_1" \
  --dataset_type="synthetic" \
  --latent_size=32 \
  --split=test \
  --prefix_length=11 \
  --sample_length=1 \
  --standardize=True
python run_fivo.py \
  --mode=sample \
  --logdir="output/fivo/synthetic_1_bs_4_standardize" \
  --model=vrnn \
  --bound=fivo \
  --summarize_every=100 \
  --batch_size=100 \
  --num_samples=1000 \
  --learning_rate=0.0001 \
  --dataset_path="data/synthetic_model_1" \
  --dataset_type="synthetic" \
  --latent_size=32 \
  --split=test \
  --prefix_length=12 \
  --sample_length=1 \
  --standardize=True
python run_fivo.py \
  --mode=sample \
  --logdir="output/fivo/synthetic_1_bs_4_standardize" \
  --model=vrnn \
  --bound=fivo \
  --summarize_every=100 \
  --batch_size=100 \
  --num_samples=1000 \
  --learning_rate=0.0001 \
  --dataset_path="data/synthetic_model_1" \
  --dataset_type="synthetic" \
  --latent_size=32 \
  --split=test \
  --prefix_length=13 \
  --sample_length=1 \
  --standardize=True
python run_fivo.py \
  --mode=sample \
  --logdir="output/fivo/synthetic_1_bs_4_standardize" \
  --model=vrnn \
  --bound=fivo \
  --summarize_every=100 \
  --batch_size=100 \
  --num_samples=1000 \
  --learning_rate=0.0001 \
  --dataset_path="data/synthetic_model_1" \
  --dataset_type="synthetic" \
  --latent_size=32 \
  --split=test \
  --prefix_length=14 \
  --sample_length=1 \
  --standardize=True
python run_fivo.py \
  --mode=sample \
  --logdir="output/fivo/synthetic_1_bs_4_standardize" \
  --model=vrnn \
  --bound=fivo \
  --summarize_every=100 \
  --batch_size=100 \
  --num_samples=1000 \
  --learning_rate=0.0001 \
  --dataset_path="data/synthetic_model_1" \
  --dataset_type="synthetic" \
  --latent_size=32 \
  --split=test \
  --prefix_length=15 \
  --sample_length=1 \
  --standardize=True
python run_fivo.py \
  --mode=sample \
  --logdir="output/fivo/synthetic_1_bs_4_standardize" \
  --model=vrnn \
  --bound=fivo \
  --summarize_every=100 \
  --batch_size=100 \
  --num_samples=1000 \
  --learning_rate=0.0001 \
  --dataset_path="data/synthetic_model_1" \
  --dataset_type="synthetic" \
  --latent_size=32 \
  --split=test \
  --prefix_length=16 \
  --sample_length=1 \
  --standardize=True
python run_fivo.py \
  --mode=sample \
  --logdir="output/fivo/synthetic_1_bs_4_standardize" \
  --model=vrnn \
  --bound=fivo \
  --summarize_every=100 \
  --batch_size=100 \
  --num_samples=1000 \
  --learning_rate=0.0001 \
  --dataset_path="data/synthetic_model_1" \
  --dataset_type="synthetic" \
  --latent_size=32 \
  --split=test \
  --prefix_length=17 \
  --sample_length=1 \
  --standardize=True
python run_fivo.py \
  --mode=sample \
  --logdir="output/fivo/synthetic_1_bs_4_standardize" \
  --model=vrnn \
  --bound=fivo \
  --summarize_every=100 \
  --batch_size=100 \
  --num_samples=1000 \
  --learning_rate=0.0001 \
  --dataset_path="data/synthetic_model_1" \
  --dataset_type="synthetic" \
  --latent_size=32 \
  --split=test \
  --prefix_length=18 \
  --sample_length=1 \
  --standardize=True
python run_fivo.py \
  --mode=sample \
  --logdir="output/fivo/synthetic_1_bs_4_standardize" \
  --model=vrnn \
  --bound=fivo \
  --summarize_every=100 \
  --batch_size=100 \
  --num_samples=1000 \
  --learning_rate=0.0001 \
  --dataset_path="data/synthetic_model_1" \
  --dataset_type="synthetic" \
  --latent_size=32 \
  --split=test \
  --prefix_length=19 \
  --sample_length=1 \
  --standardize=True
python run_fivo.py \
  --mode=sample \
  --logdir="output/fivo/synthetic_1_bs_4_standardize" \
  --model=vrnn \
  --bound=fivo \
  --summarize_every=100 \
  --batch_size=100 \
  --num_samples=1000 \
  --learning_rate=0.0001 \
  --dataset_path="data/synthetic_model_1" \
  --dataset_type="synthetic" \
  --latent_size=32 \
  --split=test \
  --prefix_length=20 \
  --sample_length=1 \
  --standardize=True
python run_fivo.py \
  --mode=sample \
  --logdir="output/fivo/synthetic_1_bs_4_standardize" \
  --model=vrnn \
  --bound=fivo \
  --summarize_every=100 \
  --batch_size=100 \
  --num_samples=1000 \
  --learning_rate=0.0001 \
  --dataset_path="data/synthetic_model_1" \
  --dataset_type="synthetic" \
  --latent_size=32 \
  --split=test \
  --prefix_length=21 \
  --sample_length=1 \
  --standardize=True
python run_fivo.py \
  --mode=sample \
  --logdir="output/fivo/synthetic_1_bs_4_standardize" \
  --model=vrnn \
  --bound=fivo \
  --summarize_every=100 \
  --batch_size=100 \
  --num_samples=1000 \
  --learning_rate=0.0001 \
  --dataset_path="data/synthetic_model_1" \
  --dataset_type="synthetic" \
  --latent_size=32 \
  --split=test \
  --prefix_length=22 \
  --sample_length=1 \
  --standardize=True
python run_fivo.py \
  --mode=sample \
  --logdir="output/fivo/synthetic_1_bs_4_standardize" \
  --model=vrnn \
  --bound=fivo \
  --summarize_every=100 \
  --batch_size=100 \
  --num_samples=1000 \
  --learning_rate=0.0001 \
  --dataset_path="data/synthetic_model_1" \
  --dataset_type="synthetic" \
  --latent_size=32 \
  --split=test \
  --prefix_length=23 \
  --sample_length=1 \
  --standardize=True
python run_fivo.py \
  --mode=sample \
  --logdir="output/fivo/synthetic_1_bs_4_standardize" \
  --model=vrnn \
  --bound=fivo \
  --summarize_every=100 \
  --batch_size=100 \
  --num_samples=1000 \
  --learning_rate=0.0001 \
  --dataset_path="data/synthetic_model_1" \
  --dataset_type="synthetic" \
  --latent_size=32 \
  --split=test \
  --prefix_length=0 \
  --sample_length=1 \
  --standardize=True