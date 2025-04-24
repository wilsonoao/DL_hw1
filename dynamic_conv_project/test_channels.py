import torch
from torch.utils.data import DataLoader
from datasets.mini_imagenet import MiniImageNet
from torchvision import transforms

@torch.no_grad()
def test_channel_combinations(model, dataloader, device, combos=[(0,), (1,), (2,), (0,1), (1,2), (0,2), (0, 1, 2)]):
    model.eval()
    results = {}
    for combo in combos:
        total, correct = 0, 0
        for x, y in dataloader:
            x_mask = torch.zeros_like(x)
            x_mask[:, combo, :, :] = x[:, combo, :, :]
            #x = pad_to_three_channels(x_mask)
            x_mask, y = x_mask.to(device), y.to(device)
            logits = model(x_mask)
            _, pred = logits.max(1)
            correct += pred.eq(y).sum().item()
            total += x_mask.size(0)
        results[str(combo)] = correct / total
    print("Test Results under Channel Combinations:")
    for k, v in results.items():
        print(f"  {k}: {v:.3f}")
