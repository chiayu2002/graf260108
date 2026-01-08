import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConsistencyControllingLatentCode(nn.Module):
    """一致性控制潛在代碼 (CCLC)"""
    
    def __init__(self, num_views: int, lr_height: int, lr_width: int, scale_factor: int = 4):
        super().__init__()
        self.num_views = num_views
        self.scale_factor = scale_factor
        
        # 為每個視角初始化可學習的潛在代碼
        self.latent_codes = nn.Parameter(
            torch.randn(num_views, 3, scale_factor * lr_height, scale_factor * lr_width) * 0.01
        )
        
    def forward(self, view_idx: int) -> torch.Tensor:
        """獲取特定視角的潛在代碼"""
        if isinstance(view_idx, torch.Tensor):
            view_idx = view_idx.item()
        return self.latent_codes[view_idx % self.num_views]


class ConsistencyEnforcingModule(nn.Module):
    """一致性執行模組 (CEM)"""
    
    def __init__(self, blur_kernel_size: int = 3):
        super().__init__()
        # 定義模糊核
        self.blur_kernel_size = blur_kernel_size
        kernel = torch.ones(1, 1, blur_kernel_size, blur_kernel_size) / (blur_kernel_size * blur_kernel_size)
        self.register_buffer('blur_kernel', kernel)
        
    def forward(self, sr_image: torch.Tensor, lr_image: torch.Tensor) -> torch.Tensor:
        """使用模糊核的CEM實現"""
        # 使用模糊核對SR圖像進行模糊處理
        blurred = F.conv2d(sr_image, 
                        self.blur_kernel.expand(sr_image.size(1), -1, -1, -1), 
                        padding=self.blur_kernel_size//2, 
                        groups=sr_image.size(1))
        downsampled = F.interpolate(blurred, size=lr_image.shape[-2:], mode='bilinear', align_corners=False)
        """執行一致性強制"""
        # 簡化的CEM實現
        # 對SR圖像進行下採樣
        # scale = sr_image.shape[-1] // lr_image.shape[-1]
        # downsampled = F.interpolate(sr_image, size=lr_image.shape[-2:], mode='bilinear', align_corners=False)
        
        # 計算殘差並上採樣
        residual = lr_image - downsampled
        upsampled_residual = F.interpolate(residual, size=sr_image.shape[-2:], mode='bilinear', align_corners=False)
        
        # 應用修正
        refined_sr_image = sr_image + 0.5 * upsampled_residual
        
        return refined_sr_image

# class ConsistencyEnforcingModule(nn.Module):
#     def __init__(self, blur_kernel_size: int = 3, noise_std: float = 0.0):
#         super().__init__()
#         # 創建高斯模糊核
#         self.blur_kernel_size = blur_kernel_size
#         self.noise_std = noise_std
#         self._create_blur_kernel()
        
#     def _create_blur_kernel(self):
#         # 創建高斯核而不是均勻核
#         sigma = self.blur_kernel_size / 3.0
#         kernel_1d = torch.exp(-0.5 * ((torch.arange(self.blur_kernel_size) - self.blur_kernel_size // 2) / sigma) ** 2)
#         kernel_1d = kernel_1d / kernel_1d.sum()
#         kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
#         self.register_buffer('blur_kernel', kernel_2d.unsqueeze(0).unsqueeze(0))
    
#     def forward(self, sr_image: torch.Tensor, lr_image: torch.Tensor) -> torch.Tensor:
#         """基於論文公式的CEM實現"""
#         # H矩阵操作：模糊+下採樣
#         blurred = F.conv2d(sr_image, self.blur_kernel.expand(sr_image.size(1), -1, -1, -1), 
#                           padding=self.blur_kernel_size//2, groups=sr_image.size(1))
#         downsampled = F.interpolate(blurred, size=lr_image.shape[-2:], mode='bilinear')
        
#         # 添加噪聲模型（如果指定）
#         if self.noise_std > 0:
#             noise = torch.randn_like(downsampled) * self.noise_std
#             downsampled = downsampled + noise
        
#         # 計算正交投影
#         # P_N(H)⊥ = H^T(HH^T)^(-1)H
#         residual = lr_image - downsampled
#         upsampled_residual = F.interpolate(residual, size=sr_image.shape[-2:], mode='bilinear')
        
#         # 修正SR圖像
#         refined_sr = sr_image + upsampled_residual
        
#         return refined_sr
    

class CCSR(nn.Module):
    """一致性控制超分辨率模組"""
    
    def __init__(self, num_views: int, lr_height: int, lr_width: int, scale_factor: int = 4):
        super().__init__()
        
        self.scale_factor = scale_factor
        self.cclc = ConsistencyControllingLatentCode(num_views, 64, 64, 1)
        self.cem = ConsistencyEnforcingModule()
        
        # 簡化的超分辨率網絡
        self.sr_network = self._build_sr_network(scale_factor)
        self.activation = nn.LeakyReLU(0.2)
        
    def _build_sr_network(self, scale_factor: int) -> nn.Module:
        """構建超分辨率網絡"""
        return nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1),  # 6 = 3(LR) + 3(latent)
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 3 * scale_factor * scale_factor, 3, padding=1),
            nn.PixelShuffle(scale_factor)
        )
    
    def forward(self, lr_image: torch.Tensor, view_idx: int) -> torch.Tensor:
        """CCSR前向傳播"""
        batch_size = lr_image.shape[0]
        
        # 獲取潛在代碼
        latent_code = self.cclc(view_idx)
        latent_code = latent_code.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # 上採樣低分辨率圖像
        lr_upsampled = F.interpolate(lr_image, size=latent_code.shape[-2:], mode='bilinear', align_corners=False)
        
        # 連接輸入
        combined_input = torch.cat([lr_upsampled, latent_code], dim=1)
        
        # 生成超分辨率圖像
        sr_output = self.sr_network(combined_input)
        sr_output = self.activation(sr_output)
        sr_output = torch.clamp(sr_output, -1, 1)  # 匹配GRAF的圖像範圍 [-1, 1]
        
        # 應用CEM
        refined_sr = self.cem(sr_output, lr_image)
        
        return refined_sr