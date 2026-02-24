import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConsistencyControllingLatentCode(nn.Module):
    """一致性控制潛在代碼 (CCLC) - 支援動態尺寸"""
    
    def __init__(self, num_views: int):
        super().__init__()
        self.num_views = num_views
        
        # 初始化為基礎大小 (例如 8x8)，forward 時再動態縮放
        self.latent_codes = nn.Parameter(
            torch.randn(num_views, 3, 32, 32) * 0.01
        )
        
    def forward(self, view_idx: int, target_size: tuple) -> torch.Tensor:
        """獲取特定視角的潛在代碼並縮放到目標尺寸 (H, W)"""
        if isinstance(view_idx, torch.Tensor):
            view_idx = view_idx.item()
        
        code = self.latent_codes[view_idx % self.num_views]
        
        # 動態縮放到與當前輸入圖片一致的大小
        # target_size 預期為 (height, width)
        return F.interpolate(code.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)


class ConsistencyEnforcingModule(nn.Module):
    def __init__(self, blur_kernel_size: int = 3):
        super().__init__()
        self.blur_kernel_size = blur_kernel_size
        # 使用高斯核代替均勻核，這能提供更好的邊緣平滑效果
        sigma = blur_kernel_size / 3.0
        x = torch.arange(blur_kernel_size) - blur_kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        self.register_buffer('blur_kernel', kernel_2d.unsqueeze(0).unsqueeze(0))
    
    def forward(self, sr_image: torch.Tensor, lr_image: torch.Tensor) -> torch.Tensor:
        # 1. H 矩陣操作：模糊 + 下採樣 (模擬退化過程)
        blurred = F.conv2d(sr_image, self.blur_kernel.expand(sr_image.size(1), -1, -1, -1), 
                          padding=self.blur_kernel_size//2, groups=sr_image.size(1))
        downsampled = F.interpolate(blurred, size=lr_image.shape[-2:], mode='bilinear', align_corners=False)
        
        # 2. 計算數據一致性殘差 (Data Consistency)
        residual = lr_image - downsampled
        upsampled_residual = F.interpolate(residual, size=sr_image.shape[-2:], mode='bilinear', align_corners=False)
        
        # 3. 修正 SR 圖像：這裡權重建議設為 1.0 以嚴格遵守一致性
        refined_sr = sr_image + 1.0 * upsampled_residual
        return refined_sr
    
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
    def forward(self, x):
        return x + self.conv(x)  # 殘差連接

class CCSR(nn.Module):
    """一致性控制超分辨率模組 - 全動態尺寸支援"""
    
    def __init__(self, num_views: int, scale_factor: int = 1):
        super().__init__()
        
        self.num_views = num_views
        self.scale_factor = scale_factor
        
        # CCLC 現在不需要預設的 lr_height/lr_width
        self.cclc = ConsistencyControllingLatentCode(num_views)
        self.cem = ConsistencyEnforcingModule()
        
        self.sr_network = self._build_sr_network(scale_factor)
        self.activation = nn.Tanh()

    def _build_sr_network(self, scale_factor: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(64), # 加入殘差塊
            ResBlock(64),
            ResBlock(64),
            nn.Conv2d(64, 3, 3, padding=1)
    )   
    # def _build_sr_network(self, scale_factor: int) -> nn.Module:
    #     """構建網路，scale_factor=1 時不改變解析度"""
    #     return nn.Sequential(
    #         nn.Conv2d(6, 64, 3, padding=1),  # 3(RGB) + 3(Latent)
    #         nn.LeakyReLU(0.2),
    #         nn.Conv2d(64, 64, 3, padding=1),
    #         nn.LeakyReLU(0.2),
    #         nn.Conv2d(64, 3 * scale_factor * scale_factor, 3, padding=1),
    #         # nn.PixelShuffle(scale_factor)
    #     )
    
    def forward(self, lr_image: torch.Tensor, view_idx: int) -> torch.Tensor:
        """CCSR 前向傳播：輸出尺寸動態跟隨 lr_image"""
        batch_size, _, h, w = lr_image.shape
        
        # 1. 獲取潛在代碼並動態縮放到輸入影像的大小 (h, w)
        latent_code = self.cclc(view_idx, (h, w))
        latent_code = latent_code.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # 2. 連接輸入 (此處尺寸已完全對齊)
        combined_input = torch.cat([lr_image, latent_code], dim=1)
        
        # 3. 通過網路生成增強影像
        sr_output = self.sr_network(combined_input)
        sr_output = self.activation(sr_output)
        sr_output = torch.clamp(sr_output, -1, 1)
        
        # 4. 應用一致性執行模組 (CEM)
        refined_sr = self.cem(sr_output, lr_image)
        
        return refined_sr