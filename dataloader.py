import torch
import torchvision
import torchvision.transforms as transforms

def get_cifar10(batch_size=128, data_dir='./data'):
    mean = [0.4914, 0.4822, 0.4465]
    std  = [0.2023, 0.1994, 0.2010]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True,  download=True, transform=train_transform)
    test_set  = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader
