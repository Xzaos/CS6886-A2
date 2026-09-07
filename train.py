import os
import torch
import torch.nn as nn
import argparse
import wandb

from dataloader import get_cifar10
from model import get_model
from utils import evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',     type=int,   default=100)
    parser.add_argument('--lr',         type=float, default=0.05)
    parser.add_argument('--batch_size', type=int,   default=128)
    parser.add_argument('--wd',         type=float, default=4e-5)
    args = parser.parse_args()

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = get_cifar10(batch_size=args.batch_size)

    model = get_model().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    wandb.init(project='cs6886-a2', config=args)

    for epoch in range(1, args.epochs + 1):
        if epoch <= 5:
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr * epoch / 5
        model.train()
        total_loss = correct = total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += outputs.argmax(dim=1).eq(labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_acc  = 100.0 * correct / total
        test_acc   = evaluate(model, test_loader, device)

        wandb.log({'train_loss': train_loss, 'train_acc': train_acc, 'test_acc': test_acc, 'epoch': epoch})
        print(f'Epoch {epoch} | Loss: {train_loss:.4f} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}%')
        if epoch > 5:
            scheduler.step()

    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/mobilenetv2_cifar10.pth')
