import torch
# from torch.nn import functional as F
import torch.utils.data
import torch.utils.data.distributed
from torch import autograd
import torch.nn as nn
import pickle
import numpy as np
import os


class MCE_Loss(nn.Module):
    def __init__(self):
        super(MCE_Loss, self).__init__()

    def __call__(self, n_each_task, output, target):
        loss = []
        for i, n in enumerate(n_each_task):
            if i == 0:
                loss.append(nn.CrossEntropyLoss()(output[:, :n], target[:, :n]))
            else:
                summation = sum(n_each_task[:i])
                loss.append(nn.CrossEntropyLoss()(output[:, summation:summation+n], target[:, summation:summation+n]))
        
        return sum(loss)

class CCSRNeRFLoss(nn.Module):
    """CCSR與NeRF的聯合損失"""
    
    def __init__(self, alpha_init=1.0, alpha_decay=0.0001):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.alpha_init = alpha_init
        self.alpha_decay = alpha_decay
        self.iteration = 0
        
    def forward(self, ccsr_output, nerf_output):
        """
        計算CCSR輸出與NeRF輸出的MSE損失
        
        Args:
            ccsr_output: CCSR生成的圖像 [B, N_samples, 3] 
            nerf_output: NeRF渲染的圖像 [B, N_samples, 3]
        """
        # 動態調整權重
        alpha = self.alpha_init * np.exp(-self.alpha_decay * self.iteration)
        
        # 計算MSE損失
        consistency_loss = self.mse_loss(ccsr_output, nerf_output.detach())
        
        self.iteration += 1
        return alpha * consistency_loss
    

def compute_loss(d_outs, target):

    d_outs = [d_outs] if not isinstance(d_outs, list) else d_outs
    loss = 0
    BCEWithLogitsLoss = nn.BCEWithLogitsLoss()

    for d_out in d_outs:
        targets = d_out.new_full(size=d_out.size(), fill_value=target)
        # loss += F.binary_cross_entropy_with_logits(d_out, targets)
        loss += BCEWithLogitsLoss(d_out, targets)
        # loss += (2*target - 1) * d_out.mean()
        # floss = loss / len(d_outs)
    return loss / len(d_outs)


def compute_grad2(d_outs, x_in):
    d_outs = [d_outs] if not isinstance(d_outs, list) else d_outs
    reg = 0
    for d_out in d_outs:
        batch_size = x_in.size(0)
        grad_dout = autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert(grad_dout2.size() == x_in.size())
        reg += grad_dout2.view(batch_size, -1).sum(1)
    return reg / len(d_outs)

def wgan_gp_reg(discriminator, x_real, x_fake, y, center=1.):
        batch_size = y.size(0)
        device = torch.device("cuda:0")
        y = y.to(device)

        samples_per_batch = x_real.size(0) // batch_size
        x_real_batched = x_real.reshape(batch_size, samples_per_batch, 3)
        x_fake_batched = x_fake.reshape(batch_size, samples_per_batch, 3)

        eps = torch.rand(batch_size, device=y.device).view(batch_size, 1, 1)
        x_interp_batched = (1 - eps) * x_real_batched + eps * x_fake_batched
        x_interp = x_interp_batched.reshape(-1, 3)
        x_interp = x_interp.detach()
        x_interp.requires_grad_()
        d_out = discriminator(x_interp, y)

        reg = (compute_grad2(d_out, x_interp).sqrt() - center).pow(2).mean()

        return reg

def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

def save_data(label, rays, iteration, save_dir='./saved_data'):
    """
    簡單的函數用於儲存標籤和光線
    """
    save_dir = os.path.join(save_dir, f'iter_{iteration}')
    os.makedirs(save_dir, exist_ok=True)
    
    # 儲存為 numpy 格式
    label_np = label.detach().cpu().numpy() if isinstance(label, torch.Tensor) else label
    rays_np = rays.detach().cpu().numpy() if isinstance(rays, torch.Tensor) else rays
    
    np.save(os.path.join(save_dir, 'labels.npy'), label_np)
    np.save(os.path.join(save_dir, 'rays.npy'), rays_np)

    with open(os.path.join(save_dir, 'rays_values.csv'), 'w') as f:
        f.write("batch,index,x,y,z\n")  # CSV 標頭
        for batch_idx in range(rays_np.shape[0]):
            for ray_idx in range(rays_np.shape[1]):
                x, y, z = rays_np[batch_idx, ray_idx]
                f.write(f"{batch_idx},{ray_idx},{x},{y},{z}\n")

    with open(os.path.join(save_dir, 'labels_full.txt'), 'w') as f:
        # 設置 numpy 顯示選項以顯示所有元素
        np.set_printoptions(threshold=np.inf, precision=8, suppress=True)
        f.write("Labels (Shape: {}):\n".format(label_np.shape))
        f.write(np.array2string(label_np))

    # 恢復 numpy 的默認顯示選項
    np.set_printoptions(threshold=1000)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class FeatureOrientationLoss:
    """
    基於特徵的物體方向規範化損失
    """
    def __init__(self, device='cuda:0'):
        # 初始化 ResNet-50 特徵提取器
        self.device = device
        resnet = models.resnet50(pretrained=True)
        # 使用 layer3 作為特徵提取層
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:7])
        self.feature_extractor = self.feature_extractor.to(device).eval()
        
        # 固定特徵提取器的權重
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        # 預設標準視角
        self.standard_views = [(0.0, 0.5), (0.25, 0.5), (0.5, 0.5), (0.75, 0.5)]  # 前、右、後、左
        
        # 特徵關係初始化為None，第一次運行時將建立標準關係
        self.feature_relations = None
        self.is_initialized = False
    
    def _extract_features(self, generator, z, label, pose):
        """從特定角度提取特徵"""
        rays = generator.val_ray_sampler(generator.H, generator.W, generator.focal, pose)[0]
        
        with torch.no_grad():
            rgb, rays = generator(z, label, rays=rays)
            
            # 將渲染結果重塑為圖像格式 (B, C, H, W)
            H, W = generator.H, generator.W
            img = rgb.view(-1, H, W, 3).permute(0, 3, 1, 2)
            
            # 檢查尺寸並進行調整，確保符合ResNet輸入要求
            if img.shape[2] < 32 or img.shape[3] < 32:
                img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
            
            # # 將像素值範圍從 [-1, 1] 調整到 [0, 1]
            # img = (img + 1) / 2.0
            
            # 提取特徵
            features = self.feature_extractor(img)
            
            # 對特徵進行降維，簡化計算
            features = features.mean(dim=(2, 3))  # 對空間維度進行平均池化
            
            return features
    
    def initialize_feature_relations(self, generator, z, label):
        """初始化標準特徵關係"""
        if self.is_initialized:
            return
        
        print("正在初始化方向一致性特徵關係...")
        
        # 獲取不同視角的特徵
        view_features = []
        for u, v in self.standard_views:
            pose = generator.sample_select_pose(u, v).to(self.device)
            features = self._extract_features(generator, z, label, pose)
            view_features.append(features)
        
        # 建立視角之間的特徵關係
        self.feature_relations = {}
        for i in range(len(self.standard_views)):
            for j in range(i+1, len(self.standard_views)):
                # 計算兩個視角之間的特徵差異作為標準關係
                relation_key = (i, j)
                feature_diff = view_features[i] - view_features[j]
                self.feature_relations[relation_key] = feature_diff.detach()  # 分離計算圖
        
        self.is_initialized = True
        print("方向一致性特徵關係初始化完成")

    def save_feature_relations(self, path):
        """保存標準特徵關係到文件"""
        if not self.is_initialized:
            print("錯誤：特徵關係尚未初始化，無法保存")
            return
        
        save_data = {
            'feature_relations': self.feature_relations,
            'standard_views': self.standard_views
        }
        torch.save(save_data, path)
        print(f"特徵關係已保存到 {path}")

    def load_feature_relations(self, path):
        """從文件加載標準特徵關係"""
        if not os.path.exists(path):
            print(f"錯誤：找不到文件 {path}")
            return False
        
        try:
            save_data = torch.load(path)
            self.feature_relations = save_data['feature_relations']
            self.standard_views = save_data['standard_views']
            self.is_initialized = True
            print(f"特徵關係已從 {path} 加載")
            return True
        except Exception as e:
            print(f"加載特徵關係時出錯: {e}")
            return False
    
    def __call__(self, generator, z_batch, label_batch):
        """計算方向一致性損失"""
        batch_size = z_batch.size(0)
        
        # 如果是第一次調用，先初始化特徵關係
        if not self.is_initialized:
            with torch.no_grad():
                self.initialize_feature_relations(generator, z_batch[0:1], label_batch[0:1])
        
        total_loss = 0.0
        
        # 為了節省計算資源，可以只對批次中的部分樣本計算損失
        sample_size = min(batch_size, 4)  # 最多使用4個樣本計算損失
        indices = torch.randperm(batch_size)[:sample_size]
        
        for idx in indices:
            z = z_batch[idx:idx+1]
            label = label_batch[idx:idx+1]
            
            # 獲取不同視角的特徵
            view_features = []
            for u, v in self.standard_views:
                pose = generator.sample_select_pose(u, v).to(self.device)
                features = self._extract_features(generator, z, label, pose)
                view_features.append(features)
            
            # 計算視角之間的特徵關係並與標準關係比較
            view_loss = 0.0
            for (i, j), standard_relation in self.feature_relations.items():
                # 計算當前樣本的特徵關係
                current_relation = view_features[i] - view_features[j]
                
                # 計算與標準關係的差異，使用L1損失以減少離群值影響
                rel_loss = F.l1_loss(current_relation, standard_relation)
                view_loss += rel_loss
            
            total_loss += view_loss
        
        # 平均損失
        return total_loss / sample_size if sample_size > 0 else 0.0
