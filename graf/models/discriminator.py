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
        input_nc = nc + num_classes
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
                # nn.GroupNorm(8, ndf),
                nn.BatchNorm2d(ndf),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                # nn.GroupNorm(8, ndf*2),
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
            # nn.GroupNorm(8, ndf*4),
            # IN(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            # nn.GroupNorm(8, ndf*8),
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
            # nn.GroupNorm(8, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),  
            nn.Conv2d(256, num_classes, 1, bias=False)  
        )

    def forward(self, input, label, return_features=False):
        input = input[:, :self.nc]
        input = input.view(-1, self.imsize, self.imsize, self.nc).permute(0, 3, 1, 2)  # (BxN_samples)xC -> BxCxHxW

        first_label = label[:, :self.num_classes].float().to(input.device)
        # one_hot = F.one_hot(first_label, num_classes=self.num_classes).float()
        # one_hot_expanded = one_hot.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.imsize, self.imsize)
        # label_feature = self.label_embedding(first_label)
        # label_map = label_feature.view(label_feature.size(0), self.nc, 1, 1).expand(-1, -1, input.size(2), input.size(3))
        label_map = first_label.view(first_label.size(0), self.num_classes, 1, 1).expand(-1, -1, input.size(2), input.size(3))

        if self.hflip:      # Randomly flip input horizontally
            input_flipped = input.flip(3)
            mask = torch.randint(0, 2, (len(input),1, 1, 1)).bool().expand(-1, *input.shape[1:])
            input = torch.where(mask, input, input_flipped)
        labelinput = torch.cat([input, label_map], 1)
        features = self.main(labelinput)
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

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Discriminator(nn.Module):
#     def __init__(self, nc=3, ndf=64, imsize=64, hflip=False, num_classes=1):
#         super(Discriminator, self).__init__()
#         self.nc = nc
#         self.ndf = ndf
#         self.imsize = imsize
#         self.hflip = hflip
#         self.num_classes = num_classes

#         # --- [架構修改] 定義卷積層 ---
#         # 輸入通道 = 圖片通道 (nc) + 標籤通道 (num_classes)
#         # 結構參考您提供的: (64)4c2s -> (128)4c2s_BL
#         self.conv = nn.Sequential(
#             nn.Conv2d(self.nc + self.num_classes, 64, 4, 2, 1),
#             nn.LeakyReLU(0.2, inplace=True),
            
#             nn.Conv2d(64, 128, 4, 2, 1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
        
#         # --- [架構修改] 定義全連接層 ---
#         # 計算 Flatten 後的大小
#         # 經過兩次 stride=2 的卷積，尺寸會變為原來的 1/4
#         # 例如 64x64 -> 32x32 -> 16x16
#         flatten_size = 128 * (self.imsize // 4) * (self.imsize // 4)
        
#         # 輸出維度 = 1 (真假值) + num_classes (類別預測)
#         self.output_dim = 1 + self.num_classes

#         self.fc = nn.Sequential(
#             nn.Linear(flatten_size, 512),
#             nn.BatchNorm1d(512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(512, self.output_dim),
#         )
        
#         # 注意：移除了 Sigmoid，因為 train.py 使用 BCEWithLogitsLoss
#         # self.sigmoid = nn.Sigmoid() 

#     def forward(self, input, label):
#         # 1. 處理輸入圖片
#         input = input[:, :self.nc] # 確保只取 RGB 通道
#         input = input.view(-1, self.nc, self.imsize, self.imsize)

#         # 2. 處理水平翻轉 (Data Augmentation)
#         if self.hflip:
#             input_flipped = input.flip(3)
#             mask = torch.randint(0, 2, (len(input), 1, 1, 1)).bool().to(input.device).expand(-1, *input.shape[1:])
#             input = torch.where(mask, input, input_flipped)

#         # 3. 處理標籤 (Label Expansion)
#         # 取出類別標籤，假設前 num_classes 位是類別資訊
#         label_vec = label[:, :self.num_classes].float()
        
#         # 將 [Batch, Class] 擴展為 [Batch, Class, H, W]
#         label_map = label_vec.view(label_vec.size(0), self.num_classes, 1, 1)
#         label_map = label_map.expand(label_vec.size(0), self.num_classes, self.imsize, self.imsize).to(input.device)

#         # 4. 拼接圖片與標籤 (Concatenate)
#         x = torch.cat([input, label_map], 1)

#         # 5. 前向傳播
#         x = self.conv(x)
#         x = x.view(x.size(0), -1) # Flatten
#         x = self.fc(x)

#         # 6. 分割輸出
#         # binary: 真假值 (Logits) -> 對應 train.py 的 d_real/d_fake
#         # class_pred: 類別預測 (Logits) -> 對應 train.py 的 label_real_pred
#         binary = x[:, :1]      
#         class_pred = x[:, 1:] 

#         return binary, class_pred