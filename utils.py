import torch

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            correct += outputs.argmax(dim=1).eq(labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
