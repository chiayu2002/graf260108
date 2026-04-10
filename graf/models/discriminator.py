import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, imsize=64, hflip=False,
                 hidden_dim=1024, cond_dim=256, shared_cond_proj=None):
        super(Discriminator, self).__init__()
        self.nc = nc
        self.imsize = imsize
        self.hflip = hflip
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim

        # ========================================================
        # [共享] 用 list 包起來避免被 nn.Module 自動註冊為 submodule
        # 這樣 D.parameters() 不會包含 shared_cond_proj 的權重
        # → D 的 optimizer 不會更新它，也不會重複儲存在 checkpoint
        # 它只透過 G 的 optimizer 更新（因為它在 NeRF 上是正常 submodule）
        # ========================================================
        assert shared_cond_proj is not None, "Must pass shared_cond_proj from NeRF"
        self._shared_cond_proj_holder = [shared_cond_proj]

        input_nc = nc + cond_dim  # 3 + 256


        blocks = []
        if self.imsize == 64:
            blocks += [
                nn.Conv2d(input_nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        elif self.imsize == 32:
            blocks += [
                nn.Conv2d(input_nc, ndf * 2, 4, 2, 1, bias=False),
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

        self.aux_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ndf * 8, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, hidden_dim),  # 1024
        )

    def forward(self, input, hidden_state, return_aux=False):
        input = input[:, :self.nc]
        input = input.view(-1, self.imsize, self.imsize, self.nc).permute(0, 3, 1, 2)

        if self.hflip:
            input_flipped = input.flip(3)
            mask = torch.randint(0, 2, (len(input), 1, 1, 1)).bool().expand(-1, *input.shape[1:])
            input = torch.where(mask, input, input_flipped)

        # 用共享的 condition projection (1024 → 256)
        cond = self._shared_cond_proj_holder[0](hidden_state)  # [B, 256]
        cond_map = cond.view(cond.size(0), self.cond_dim, 1, 1).expand(
            -1, -1, input.size(2), input.size(3)
        )

        x = torch.cat([input, cond_map], dim=1)
        features = self.main(x)
        out = self.conv_out(features)

        if return_aux:
            aux_pred = self.aux_head(features)  # [B, 1024]
            return out, aux_pred
        
        return out