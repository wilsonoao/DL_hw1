import torch 
import torch.nn as nn 
import torch.nn.functional as F


class DynamicConv2d(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, kernel_size=3, padding=0, K=4, reduction=4):
        super().__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels

        # K 個卷積核
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
            for _ in range(K)
        ])

        # Attention branch: SE-like (global avg → FC → ReLU → FC → softmax)
        hidden_dim = max(in_channels // reduction, 4)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # squeeze: (B, C, H, W) → (B, C, 1, 1)
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, K, kernel_size=1, bias=False),
        )

    def forward(self, x):
        B, C, H, W = x.size()
        # 計算 attention 權重
        attention_logits = self.attention(x).view(B, self.K)  # (B, K)
        attention_weights = F.softmax(attention_logits, dim=1)  # softmax 後 (B, K)

        out = torch.stack([conv(x) for conv in self.convs], dim=1)  # (B, K, C_out, H_out, W_out)

        # 加權合併
        attention_weights = attention_weights.view(B, self.K, 1, 1, 1)
        out = (out * attention_weights).sum(dim=1)  # 聚合 K 個 kernel
        #print(out.shape)

        return out
