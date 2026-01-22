import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, imsize=64, hflip=False, num_classes=1, cond=True):
        super(Discriminator, self).__init__()
        self.nc = nc
        # assert(imsize==32 or imsize==64 or imsize==128)
        self.imsize = imsize
        self.hflip = hflip
        self.num_classes = num_classes

        SN = torch.nn.utils.spectral_norm
        IN = lambda x : nn.InstanceNorm2d(x)

        blocks = []
        input_nc = nc #+ num_classes
        if self.imsize==128:
            blocks += [
                # input is (nc) x 128 x 128
                SN(nn.Conv2d(nc, ndf//2, 4, 2, 1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
                # input is (ndf//2) x 64 x 64
                SN(nn.Conv2d(ndf//2, ndf, 4, 2, 1, bias=False)),
                IN(ndf),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                SN(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
                #nn.BatchNorm2d(ndf * 2),
                IN(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        elif self.imsize==64:
            blocks += [
                # input is (nc) x 64 x 64
                nn.Conv2d(input_nc, ndf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                # IN(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            # first_out_channels = ndf
            # # 針對 64x64 的第一層
            # self.first_block = nn.Sequential(
            #     nn.Conv2d(input_nc, first_out_channels, 4, 2, 1, bias=False),
            #     nn.BatchNorm2d(first_out_channels),
            #     nn.LeakyReLU(0.2, inplace=True)
            # )
            # # 剩下的區塊
            # blocks += [
            #     nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            #     nn.BatchNorm2d(ndf * 2),
            #     nn.LeakyReLU(0.2, inplace=True),
            # ]
        else:
            blocks += [
                # input is (nc) x 32 x 32
                nn.Conv2d(input_nc, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                # IN(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        blocks += [
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            # IN(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            # IN(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            # SN(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)),
            # nn.Sigmoid()
        ]

        # self.embed = nn.Embedding(num_classes, first_out_channels)
        self.conv_out = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        blocks = [x for x in blocks if x]
        self.main = nn.Sequential(*blocks)

        # self.label_out = nn.Sequential(
        #     nn.Conv2d(ndf * 8, 128, 4, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(128, self.num_classes, 1, bias=False)
        # )
        # self.label_embedding = nn.Linear(num_classes, nc)

        self.label_out = nn.Sequential(
            nn.Conv2d(ndf * 8, 256, 1, bias=False),  
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),  
            nn.Conv2d(256, num_classes, 1, bias=False)  
        )

    def forward(self, input, label, return_features=False):
        input = input[:, :self.nc]
        input = input.view(-1, self.imsize, self.imsize, self.nc).permute(0, 3, 1, 2)  # (BxN_samples)xC -> BxCxHxW

        # first_label = label[:, :self.num_classes].float().to(input.device)
        # one_hot = F.one_hot(first_label, num_classes=self.num_classes).float()
        # one_hot_expanded = one_hot.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.imsize, self.imsize)
        # label_feature = self.label_embedding(first_label)
        # label_map = label_feature.view(label_feature.size(0), self.nc, 1, 1).expand(-1, -1, input.size(2), input.size(3))
        # label_map = first_label.view(first_label.size(0), self.num_classes, 1, 1).expand(-1, -1, input.size(2), input.size(3))

        if self.hflip:      # Randomly flip input horizontally
            input_flipped = input.flip(3)
            mask = torch.randint(0, 2, (len(input),1, 1, 1)).bool().expand(-1, *input.shape[1:])
            input = torch.where(mask, input, input_flipped)
        # labelinput = torch.cat([input, label_map], 1)
        features = self.main(input)
        # features = self.main(input)
        out = self.conv_out(features)

        # final_output = out[:, :1]
        # class_pred = out[:, 1:]
        # class_pred = class_pred.squeeze()
        # class_out = self.fc(class_pred)
        class_pred = self.label_out(features)
        class_pred = class_pred.squeeze()

        return out, class_pred
        # return features

# class DHead(nn.Module):
#     def __init__(self, ndf=64):
#         super().__init__()

#         self.conv = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)

#     def forward(self, x):
#         output = self.conv(x)

#         return output

# class QHead(nn.Module):
#     def __init__(self, ndf=64):
#         super().__init__()

#         self.conv1 = nn.Conv2d(ndf * 8, 128, 4, bias=False)
#         self.bn1 = nn.BatchNorm2d(128)

#         self.conv_disc = nn.Conv2d(128, 2, 1)

#     def forward(self, x):
#         x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2, inplace=True)

#         disc_logits = self.conv_disc(x).squeeze()

#         return disc_logits