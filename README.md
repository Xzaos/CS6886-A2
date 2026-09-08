# CS6886 Assignment 2 - MobileNet-v2 Compression
Student: Ritwik Mishra | ME23B237

## Setup
pip install torch torchvision wandb

## File Structure
- train.py — fine-tunes MobileNet-v2 on CIFAR-10
- test.py — runs PTQ or GPTQ quantization and evaluates
- sweep.py — runs the full 15-configuration grid sweep
- quantize.py — quantization implementation (QuantConv2d, QuantLinear, ActFakeQuant, GPTQ)
- model.py — MobileNet-v2 adapted for CIFAR-10
- dataloader.py — CIFAR-10 data loading and transforms
- utils.py — evaluation utilities

## Reproduce Results

### Training
python train.py

### Quantization (single run)
python test.py --weight_bits 4 --act_bits 8 --use_gptq

### Full sweep (15 configurations)
python sweep.py

## Results
- FP32 baseline: 93.59% test accuracy
- Best compression config: W4A8 (4-bit weights, 8-bit activations) with GPTQ
- Quantized accuracy: 87.15%
- Weight compression ratio: 7.19x
- Activation compression ratio: 4.00x
- Final model size: 1.32 MB (vs 8.53 MB FP32)

## Environment
- Python 3.11
- torch 2.3.0
- torchvision 0.18.0
- wandb 0.28.1
- Random seed: 42 (torch.manual_seed and torch.cuda.manual_seed set at start of every script)

## Method
Post-training quantization with GPTQ-style layer-wise weight optimization.
Per-channel symmetric quantization for weights, per-tensor asymmetric for activations.
First conv and classifier pinned at 8-bit. Depthwise convs use standard PTQ.
No quantization libraries used - implemented from scratch.

## wandb Project
https://wandb.ai/me23b237-indian-institute-of-technology-madras/cs6886-a2
