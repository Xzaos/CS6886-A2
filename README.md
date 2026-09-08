# CS6886 Assignment 2: MobileNet-v2 Compression
**Ritwik Mishra | ME23B237 | Systems Engineering for Deep Learning**

## Overview

This project fine-tunes MobileNet-v2 on CIFAR-10, then applies post-training quantization (PTQ) with GPTQ-style layer-wise weight optimization implemented entirely from scratch - no quantization libraries used. A sweep over 15 bit-width configurations (weight_bits in {8,6,4,3,2} x act_bits in {8,6,4}) is run with GPTQ enabled. The selected best configuration is W4A8 with GPTQ, achieving 87.15% accuracy, a 7.19x weight compression ratio, and a final model size of 1.32 MB (down from 8.53 MB FP32).

## Repository Structure

- **train.py** - Fine-tunes MobileNet-v2 (pretrained ImageNet weights) on CIFAR-10 for 100 epochs with SGD, cosine LR schedule, and a 5-epoch linear warmup. Saves the best checkpoint to `checkpoints/mobilenetv2_cifar10.pth`.
- **test.py** - Loads a checkpoint, applies PTQ or GPTQ quantization at configurable bit-widths, evaluates top-1 accuracy, computes model size, and logs all metrics to wandb.
- **sweep.py** - Runs test.py across all 15 combinations of weight_bits in {8,6,4,3,2} and act_bits in {8,6,4} with GPTQ enabled, logging each configuration to wandb.
- **quantize.py** - Full quantization implementation: `QuantConv2d` (per-channel symmetric weight quantization), `QuantLinear` (per-tensor symmetric), `ActFakeQuant` (per-tensor asymmetric with percentile clipping), plus `swap_layers`, `calibrate_model`, `freeze_model`, `gptq_quantize`, and `compute_model_size`.
- **model.py** - Loads pretrained MobileNetV2, changes the first conv stride from 2 to 1 for 32x32 CIFAR-10 input, and replaces the classifier head with `nn.Linear(1280, 10)`.
- **dataloader.py** - CIFAR-10 train/test loaders with standard augmentation (RandomCrop, RandomHorizontalFlip, Normalize).
- **utils.py** - `evaluate()` function for computing top-1 accuracy on a given loader.

## Method

### Training

MobileNet-v2 is initialized with pretrained ImageNet weights (`IMAGENET1K_V1`). The first conv stride is changed from 2 to 1 to preserve spatial resolution for 32x32 CIFAR-10 inputs, and the classifier is replaced with `nn.Linear(1280, 10)`. Training uses SGD with momentum 0.9, Nesterov, weight decay 4e-5, and an initial learning rate of 0.05 with a 5-epoch linear warmup followed by `CosineAnnealingLR` over 100 epochs. Loss is `CrossEntropyLoss` with label smoothing 0.1. Batch size 128, seed 42. Final FP32 test accuracy: **93.59%**.

### Quantization

- **Weight quantization**: Per-channel symmetric uniform quantization for all Conv2d layers. Each output channel gets an independent scale: `scale_i = max(|W_i|) / (2^(bits-1) - 1)`, with zero-point fixed at zero. Per-tensor symmetric quantization for the Linear classifier.
- **Activation quantization**: Per-tensor asymmetric uniform quantization inserted after every ReLU6 via `ActFakeQuant` modules. Scale and zero-point are derived from 99.9th percentile-clipped running min/max over 4 calibration batches to reduce outlier sensitivity.
- **GPTQ**: For each pointwise Conv2d (groups=1) and the Linear layer, optimal quantized weights are found by minimizing the layer-wise output reconstruction error using the inverse Hessian of the input activations. Error is propagated sequentially across weight elements within each output channel. Depthwise convs (groups>1) use standard PTQ.
- The first conv and classifier are pinned at 8-bit regardless of the `--weight_bits` argument.
- No quantization libraries were used - everything is implemented from scratch in `quantize.py`.

## Results

| Config | Accuracy | Weight Ratio | Model Size |
|--------|----------|--------------|------------|
| FP32 baseline | 93.59% | 1.00x | 8.53 MB |
| W8A8 + GPTQ | 93.31% | 3.82x | 2.36 MB |
| W6A8 + GPTQ | 93.07% | 4.99x | 1.84 MB |
| **W4A8 + GPTQ** | **87.15%** | **7.19x** | **1.32 MB** |
| W3A8 + GPTQ | 52.35% | 9.21x | 1.06 MB |
| W2A8 + GPTQ | 10.00% | 12.83x | 0.80 MB |

**W4A8 is the selected configuration**, offering the best accuracy-compression tradeoff. At 3-bit and 2-bit, accuracy collapses without quantization-aware training (QAT).

## Reproduce Results

### Environment
- Python 3.11
- torch 2.3.0
- torchvision 0.18.0
- wandb 0.28.1

### Install
```
pip install torch torchvision wandb
```

### Commands
```
# Train FP32 baseline
python train.py

# Evaluate W4A8 with GPTQ (selected config)
python test.py --weight_bits 4 --act_bits 8 --use_gptq

# Evaluate W4A8 with standard PTQ (no GPTQ, for comparison)
python test.py --weight_bits 4 --act_bits 8

# Run full 15-configuration sweep
python sweep.py
```

> CIFAR-10 and pretrained ImageNet weights download automatically on first run. For `test.py` and `sweep.py`, the FP32 checkpoint must be present at `checkpoints/mobilenetv2_cifar10.pth`.

## Wandb

All 15 sweep runs are logged to the wandb project below with metrics: `weight_bits`, `act_bits`, `fp32_acc`, `quant_acc`, `weight_ratio`, `act_ratio`, `total_mb`.

https://wandb.ai/me23b237-indian-institute-of-technology-madras/cs6886-a2

## Seed

42 - set via `torch.manual_seed(42)` and `torch.cuda.manual_seed(42)` at the start of every script.
