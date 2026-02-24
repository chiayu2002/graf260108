import torch
from math import sqrt, exp

from submodules.nerf_pytorch.run_nerf_helpers_mod import get_rays, get_rays_ortho


class ImgToPatch(object):
    def __init__(self, ray_sampler, hwf):
        self.ray_sampler = ray_sampler
        self.hwf = hwf      # camera intrinsics

    def __call__(self, img):
        rgbs = []
        for img_i in img:
            pose = torch.eye(4)         # use dummy pose to infer pixel values
            _, selected_idcs, pixels_i = self.ray_sampler(H=self.hwf[0], W=self.hwf[1], focal=self.hwf[2], pose=pose)
            if selected_idcs is not None:
                rgbs_i = img_i.flatten(1, 2).t()[selected_idcs]
            else:
                rgbs_i = torch.nn.functional.grid_sample(img_i.unsqueeze(0), 
                                     pixels_i.unsqueeze(0), mode='bilinear', align_corners=True)[0]
                rgbs_i = rgbs_i.flatten(1, 2).t()
            rgbs.append(rgbs_i)

        rgbs = torch.cat(rgbs, dim=0)       # (B*N)x3

        return rgbs


class RaySampler(object): #生成相機射線
    def __init__(self, N_samples, orthographic=False):
        super(RaySampler, self).__init__()
        self.N_samples = N_samples
        self.scale = torch.ones(1,).float()
        self.return_indices = True
        self.orthographic = orthographic

    def __call__(self, H, W, focal, pose):
        if self.orthographic: #正射投影
            size_h, size_w = focal      # Hacky
            rays_o, rays_d = get_rays_ortho(H, W, pose, size_h, size_w)
        else: #透射投影
            rays_o, rays_d = get_rays(H, W, focal, pose)

        select_inds = self.sample_rays(H, W) #取樣射線

        if self.return_indices:
            rays_o = rays_o.view(-1, 3)[select_inds] #光線原點
            rays_d = rays_d.view(-1, 3)[select_inds] #光線方向

            h = (select_inds // W) / float(H) - 0.5
            w = (select_inds %  W) / float(W) - 0.5

            hw = torch.stack([h,w]).t()

        else:
            rays_o = torch.nn.functional.grid_sample(rays_o.permute(2,0,1).unsqueeze(0), 
                                 select_inds.unsqueeze(0), mode='bilinear', align_corners=True)[0]
            rays_d = torch.nn.functional.grid_sample(rays_d.permute(2,0,1).unsqueeze(0), 
                                 select_inds.unsqueeze(0), mode='bilinear', align_corners=True)[0]
            rays_o = rays_o.permute(1,2,0).view(-1, 3)
            rays_d = rays_d.permute(1,2,0).view(-1, 3)

            hw = select_inds
            select_inds = None

        return torch.stack([rays_o, rays_d]), select_inds, hw

    def sample_rays(self, H, W):
        raise NotImplementedError


class FullRaySampler(RaySampler):
    def __init__(self, **kwargs):
        super(FullRaySampler, self).__init__(N_samples=None, **kwargs)

    def sample_rays(self, H, W):
        return torch.arange(0, H*W) #生成了一個從 0 到 H*W-1 的長度為 H*W 的一維張量



class FlexGridRaySampler(RaySampler):
    def __init__(self, N_samples, random_shift=True, random_scale=True, min_scale=0.25, max_scale=1., scale_anneal=-1,
                 **kwargs):
        self.N_samples_sqrt = int(sqrt(N_samples))
        super(FlexGridRaySampler, self).__init__(self.N_samples_sqrt**2, **kwargs)

        self.random_shift = random_shift
        self.random_scale = random_scale

        self.min_scale = min_scale
        self.max_scale = max_scale

        # nn.functional.grid_sample grid value range in [-1,1]
        self.w, self.h = torch.meshgrid([torch.linspace(-1,1,self.N_samples_sqrt),
                                         torch.linspace(-1,1,self.N_samples_sqrt)])
        self.h = self.h.unsqueeze(2)
        self.w = self.w.unsqueeze(2)

        # directly return grid for grid_sample
        self.return_indices = False

        self.iterations = 0
        self.scale_anneal = scale_anneal

    def sample_rays(self, H, W):

        if self.scale_anneal>0:
            k_iter = self.iterations // 1000 * 3
            min_scale = max(self.min_scale, self.max_scale * exp(-k_iter*self.scale_anneal))
            min_scale = min(0.9, min_scale)
        else:
            min_scale = self.min_scale

        scale = 1
        if self.random_scale:
            scale = torch.Tensor(1).uniform_(min_scale, self.max_scale)
            h = self.h * scale 
            w = self.w * scale 

        if self.random_shift:
            max_offset = 1-scale.item()
            h_offset = torch.Tensor(1).uniform_(0, max_offset) * (torch.randint(2,(1,)).float()-0.5)*2
            w_offset = torch.Tensor(1).uniform_(0, max_offset) * (torch.randint(2,(1,)).float()-0.5)*2

            h += h_offset
            w += w_offset

        self.scale = scale

        return torch.cat([h, w], dim=2) #torch.Size([32, 32, 2])



# class FlexGridRaySampler(RaySampler):
#     def __init__(self, N_samples, random_shift=True, random_scale=True, min_scale=0.25, max_scale=1., scale_anneal=-1,
#                  **kwargs):
#         self.N_samples_sqrt = int(sqrt(N_samples))
#         super(FlexGridRaySampler, self).__init__(self.N_samples_sqrt**2, **kwargs)

#         self.random_shift = random_shift
#         self.random_scale = random_scale
#         self.min_scale = min_scale
#         self.max_scale = max_scale

#         # --- 修改開始: 定義兩種網格 ---
        
#         # 1. 全圖網格 (Full Image Grid) - Y軸範圍 [-1, 1]
#         self.w_full, self.h_full = torch.meshgrid([
#             torch.linspace(-1, 1, self.N_samples_sqrt),  # Y軸 (高度): -1(頂) 到 1(底)
#             torch.linspace(-1, 1, self.N_samples_sqrt)   # X軸 (寬度)
#         ])
#         self.h_full = self.h_full.unsqueeze(2)
#         self.w_full = self.w_full.unsqueeze(2)

#         # 2. 下半部專注網格 (Bottom Focus Grid) - Y軸範圍 [0, 1] (您可以調整 -0.2 或 0 來決定起始高度)
#         self.w_bottom, self.h_bottom = torch.meshgrid([
#             torch.linspace(0, 1, self.N_samples_sqrt),   # Y軸: 0(中間) 到 1(底) -> 專注下半部
#             torch.linspace(-1, 1, self.N_samples_sqrt)   # X軸: 維持全寬
#         ])
#         self.h_bottom = self.h_bottom.unsqueeze(2)
#         self.w_bottom = self.w_bottom.unsqueeze(2)
        
#         # 設定「下半部優先」的機率 (0.7 代表 70% 的時間只看下半部)
#         self.bottom_focus_prob = 0.7 

#         # --- 修改結束 ---

#         self.return_indices = False
#         self.iterations = 0
#         self.scale_anneal = scale_anneal

#     def sample_rays(self, H, W):
#         # --- 修改開始: 動態選擇採樣區域 ---
        
#         # 產生一個 0~1 的隨機數，如果小於設定機率，就採樣下半部
#         if torch.rand(1).item() < self.bottom_focus_prob:
#             # 採樣下半部 (Focus on Damage)
#             h = self.h_bottom.clone()
#             w = self.w_bottom.clone()
#         else:
#             # 採樣全圖 (Global Context)
#             h = self.h_full.clone()
#             w = self.w_full.clone()
        
#         return torch.cat([h, w], dim=2)

# class FlexGridRaySampler(RaySampler):
#     def __init__(self, N_samples, random_shift=True, random_scale=True, min_scale=0.25, max_scale=1., scale_anneal=-1,
#                  **kwargs):
#         self.N_samples_sqrt = int(sqrt(N_samples))
#         super(FlexGridRaySampler, self).__init__(self.N_samples_sqrt**2, **kwargs)

#         # 這裡我們不使用原本的 random_scale/shift 邏輯，而是自定義
#         self.min_scale = min_scale
#         self.max_scale = max_scale
        
#         # 建立基礎網格 [-1, 1] (標準正方形)
#         self.w, self.h = torch.meshgrid([torch.linspace(-1, 1, self.N_samples_sqrt),
#                                          torch.linspace(-1, 1, self.N_samples_sqrt)])
#         self.h = self.h.unsqueeze(2)
#         self.w = self.w.unsqueeze(2)

#         self.return_indices = False
#         self.iterations = 0
        
#         # 設定專注於下半部的機率 (例如 50% 全圖，50% 下半部特寫)
#         # 增加全圖的比例有助於修正 "角度不對" 的問題
#         self.bottom_focus_prob = 0.5 

#     def sample_rays(self, H, W):
#         # 決定這一輪是 "全圖模式" 還是 "下半部特寫模式"
#         is_focus_mode = torch.rand(1).item() < self.bottom_focus_prob
        
#         if not is_focus_mode:
#             # --- 模式 A: 全圖 (Context Mode) ---
#             # 用於學習正確的幾何形狀和角度
#             scale = 1.0
#             offset_h = torch.zeros(1)
#             offset_w = torch.zeros(1)
            
#         else:
#             # --- 模式 B: 下半部無失真正方形特寫 (Focus Mode) ---
#             # 用於學習破壞細節，但保持 1:1 比例
            
#             # 固定 Scale 為 0.5 (即長寬各為原圖的一半，面積為 1/4)
#             # 這個大小足夠大，包含結構資訊，不會像 random_scale 那麼碎
#             scale = 0.5 
            
#             # 計算最大位移範圍
#             # Patch 範圍是 [-scale, scale] + offset
#             # 我們要確保 Patch 在 [-1, 1] 之間，且偏向下方
            
#             max_offset = 1.0 - scale # 0.5
            
#             # X 軸 (寬度): 隨機左右移動，捕捉柱子不同側面的破壞
#             offset_w = (torch.rand(1) * 2 - 1) * max_offset
            
#             # Y 軸 (高度): 強制限制在下半部
#             # Y範圍 [-1, 1]，其中 1 是底部
#             # 我們希望 Patch 中心 (offset_h) 偏向正值 (0 到 max_offset)
#             # 這樣 Patch 就會落在 [0-scale, 0+scale] 到 [max-scale, max+scale] 之間
#             # 即 [-0.5, 0.5] 到 [0, 1] 之間 (覆蓋中下段到最底段)
#             offset_h = torch.rand(1) * max_offset
            
#             # 如果您希望更極端地只看最底部，可以固定 offset_h = max_offset

#         # 應用變換 (保持 h 和 w 同步變換，確保正方形不變形)
#         h = self.h * scale + offset_h
#         w = self.w * scale + offset_w

#         return torch.cat([h, w], dim=2)