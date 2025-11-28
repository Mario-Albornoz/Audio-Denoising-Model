import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, channels, dilation, kernel_size = 3):
        super().__init__()
        self.con1x1 = nn.Conv1d(channels, channels, 1)
        self.prelu = nn.PReLU()
        self.norm = nn.BatchNorm1d(channels)

        self.dilated_conv = nn.Conv1d(
            channels, channels,
            kernel_size,
            padding=(kernel_size - 1 ) * dilation // 2,
            dilation= dilation
        )

    def forward(self, x):
        residual = x
        x = self.con1x1(x)
        x = self.prelu(x)
        x = self.norm(x)
        x = self.dilated_conv(x)
        return x + residual


class ConvTasNet(nn.Module):
    def __init__(self, N=256, L=20, B=256, P=3, X=8, R=2):
        super().__init__()
        self.L = L
        self.encoder = nn.Conv1d(1, N, kernel_size=L, stride=L // 2, padding=0)
        self.decoder = nn.ConvTranspose1d(N, 1, kernel_size=L, stride=L // 2, padding=0)

        self.bottle_neck = nn.Conv1d(N, B, 1)

        blocks = []
        for _ in range(R):
            for i in range(X):
                dilation = 2 ** i
                blocks.append(ConvBlock(B, dilation, kernel_size=P))

        self.network = nn.Sequential(*blocks)
        self.mask_conv = nn.Conv1d(B, N, 1)

    def forward(self, x):
        original_length = x.size(-1)

        enc = self.encoder(x)
        bottle_neck = self.bottle_neck(enc)
        out = self.network(bottle_neck)
        mask = torch.sigmoid(self.mask_conv(out))
        enhanced = mask * enc
        decoded = self.decoder(enhanced)

        if decoded.size(-1) > original_length:
            decoded = decoded[..., :original_length]
        elif decoded.size(-1) < original_length:
            padding = original_length - decoded.size(-1)
            decoded = torch.nn.functional.pad(decoded, (0, padding))

        return decoded