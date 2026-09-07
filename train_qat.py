import torch
import torch.nn as nn
import argparse
import wandb
from dataloader import get_cifar10
from model import get_model
from utils import evaluate
from quantize import swap_layers, gptq_quantize, calibrate_model, freeze_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_bits', type=int, default=3)
    parser.add_argument('--act_bits', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--checkpoint', type=str, default='checkpoints/mobilenetv2_cifar10.pth')
    parser.add_argument('--out', type=str, default='checkpoints/mobilenetv2_qat.pth')
    args = parser.parse_args()

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wandb.init(
        project='cs6886-a2',
        name=f'qat_w{args.weight_bits}a{args.act_bits}',
        config=vars(args),
    )

    train_loader, test_loader = get_cifar10(args.batch_size)

    model = get_model()
    model.load_state_dict(torch.load(args.checkpoint, weights_only=True, map_location=device))
    model.to(device)

    swap_layers(model, args.weight_bits, args.act_bits)
    gptq_quantize(model, train_loader, device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            correct += outputs.argmax(1).eq(targets).sum().item()
            total += inputs.size(0)

        train_loss /= total
        train_acc = 100.0 * correct / total
        test_acc = evaluate(model, test_loader, device)

        wandb.log({'train_loss': train_loss, 'train_acc': train_acc, 'test_acc': test_acc, 'epoch': epoch})
        print(f'Epoch {epoch} | Loss: {train_loss:.4f} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}%')

        scheduler.step()

    torch.save(model.state_dict(), args.out)
    wandb.finish()
