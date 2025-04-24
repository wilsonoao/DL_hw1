import torch
import torch.nn as nn

class RNNClassifier(nn.Module):
    def __init__(self, input_channels=3, patch_size=3, hidden_dim=512, num_classes=100):
        super().__init__()

        # Step 1: Patch embedding - Conv2d
        self.patch_embed = nn.Conv2d(
            in_channels=input_channels,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size
        )  # 假設輸入 84x84，會變成 21x21 patch

        # Step 2-3: GRU RNN 層
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=False  # 如果要改雙向可以調整
        )

        # Step 4-5: 分類 head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)  # (B, hidden_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)，視為序列

        out, _ = self.rnn(x)  # (B, N, hidden_dim)
        x = out[:, -1, :]     # 用最後一個時間點做分類

        x = self.head(x)
        return x
