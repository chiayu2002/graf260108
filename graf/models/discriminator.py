import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, imsize=64, hflip=False, 
                 hidden_dim=1024, cond_dim=256):
        super(Discriminator, self).__init__()
        self.nc = nc
        self.imsize = imsize
        self.hflip = hflip
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim

        # 1024 → 256
        self.condition_proj = nn.Sequential(
            nn.Linear(hidden_dim, cond_dim),
            nn.LayerNorm(cond_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        input_nc = nc + cond_dim  # 3 + 256 = 259

        blocks = []
        if self.imsize == 64:
            blocks += [
                nn.Conv2d(input_nc, ndf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        elif self.imsize == 32:
            blocks += [
                nn.Conv2d(input_nc, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        blocks += [
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        self.main = nn.Sequential(*blocks)
        self.conv_out = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)

    def forward(self, input, hidden_state):
        input = input[:, :self.nc]
        input = input.view(-1, self.imsize, self.imsize, self.nc).permute(0, 3, 1, 2)

        if self.hflip:
            input_flipped = input.flip(3)
            mask = torch.randint(0, 2, (len(input), 1, 1, 1)).bool().expand(-1, *input.shape[1:])
            input = torch.where(mask, input, input_flipped)

        cond = self.condition_proj(hidden_state)  # [B, 256]
        cond_map = cond.view(cond.size(0), self.cond_dim, 1, 1).expand(-1, -1, input.size(2), input.size(3))  # [B, 256, 64, 64]

        x = torch.cat([input, cond_map], dim=1)  # [B, 259, 64, 64]
        features = self.main(x)
        out = self.conv_out(features)
        return out