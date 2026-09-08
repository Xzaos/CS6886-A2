import torch
import argparse
import wandb

from dataloader import get_cifar10
from model import get_model
from utils import evaluate
from quantize import swap_layers, gptq_quantize, calibrate_model, freeze_model, compute_model_size

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_bits', type=int, default=4)
    parser.add_argument('--act_bits',    type=int, default=8)
    parser.add_argument('--checkpoint',  type=str, default='checkpoints/mobilenetv2_cifar10.pth')
    parser.add_argument('--batch_size',  type=int, default=64)
    parser.add_argument('--use_gptq',    action='store_true')
    args = parser.parse_args()

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wandb.init(
        project='cs6886-a2',
        name=f'w{args.weight_bits}a{args.act_bits}_{"gptq" if args.use_gptq else "ptq"}',
        config=vars(args),
    )

    train_loader, test_loader = get_cifar10(batch_size=args.batch_size)

    model = get_model()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    fp32_acc = evaluate(model, test_loader, device)
    print(f'FP32 accuracy: {fp32_acc:.2f}%')

    swap_layers(model, args.weight_bits, args.act_bits)
    if args.use_gptq:
        gptq_quantize(model, train_loader, device)
    else:
        calibrate_model(model, train_loader, device)
        freeze_model(model)
    quant_acc = evaluate(model, test_loader, device)
    print(f'Quantized accuracy: {quant_acc:.2f}%')
    sizes = compute_model_size(model, args.weight_bits, args.act_bits)
    for k, v in sizes.items():
        print(f'  {k}: {v:.4f}')
    wandb.log({
        'fp32_acc': fp32_acc,
        'quant_acc': quant_acc,
        'weight_bits': args.weight_bits,
        'act_bits': args.act_bits,
        'use_gptq': args.use_gptq,
        **sizes,
    })
    print(f'FP32: {fp32_acc:.2f}% | Quant: {quant_acc:.2f}% | Weight ratio: {sizes["weight_ratio"]:.2f}x | Act ratio: {sizes["act_ratio"]:.2f}x | Size: {sizes["total_mb"]:.2f} MB')
    wandb.finish()
