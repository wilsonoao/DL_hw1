from models.dynamic_conv import DynamicConv2d
from datasets.mini_imagenet import MiniImageNet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
from test_channels import test_channel_combinations
# import os
# print("Current working directory:", os.getcwd())

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        _, pred = logits.max(1)
        correct += pred.eq(y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            _, pred = logits.max(1)
            correct += pred.eq(y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.ToTensor()
    ])
    train_set = MiniImageNet('../images/train.txt', '../images', transform)
    val_set = MiniImageNet('../images/val.txt', '../images', transform)
    test_set = MiniImageNet('../images/test.txt', '../images', transform)
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)
    test_loader = DataLoader(test_set, batch_size=64)

    #model = models.resnet18(weights=None)
    # modify alexnet
    model = models.alexnet(weights=None)
    print(f"AlexNet 模型參數總數: {count_parameters(model):,}")

    model.features = nn.Sequential(
            DynamicConv2d(),
            *list(model.features.children())[1:]
    )

    # add dynamicConv2d
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 100)
    model = model.to(device)

    print(f"AlexNet + dynamicConv2d, 模型參數總數: {count_parameters(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 21):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"[Epoch {epoch}] Train loss: {train_loss:.3f} | Val loss: {val_loss:.3f}")
        print(f"[Epoch {epoch}] Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}\n")
        #torch.save(model.state_dict(), "model.pth")
    
    # test
    test_channel_combinations(model, test_loader, device)


if __name__ == "__main__":
    main()