import glob
import numpy as np
from PIL import Image
import os
import torch

from torchvision.datasets.vision import VisionDataset



class ImageDataset(VisionDataset):
    """
    Load images from multiple data directories.
    Folder structure: data_dir/filename.png
    """

    def __init__(self, data_dirs, transforms=None, label_file=None):
        # Use multiple root folders
        if not isinstance(data_dirs, list):
            data_dirs = [data_dirs]

        # initialize base class
        VisionDataset.__init__(self, root=data_dirs, transform=transforms)

        self.filenames = []
        root = []
        self.labels = {}

        self.height_map = {
            "0_": 0,    
            "0.5_": 1,  
            "1_": 2,    
            "1.5_": 3,  
        }

        target_specimens = ["RS307_n", "RS330_n"] #, "RS615_n" , "RS315_n"
        self.specimen_props = {
            "RS307_n": {
                "AR": [1, 0],       # 3.0
                "LR": [1, 0, 0],    # 0.742
                "TR": [0, 1]        # 1.282
            },

            # "RS315_n": {
            #     "AR": [1, 0],       # 6.0
            #     "LR": [0, 1, 0],    # 1.444
            #     "TR": [1, 0]        # 1.106
            # },

            "RS330_n": {
                "AR": [1, 0],       # 3.0
                "LR": [0, 0, 1],    # 2.889
                "TR": [1, 0]        # 1.106
            },
            # "RS615_n": {
            #     "AR": [0, 1],       # 6.0
            #     "LR": [0, 1, 0],    # 1.444
            #     "TR": [1, 0]        # 1.106
            # }
        }
        for ddir in data_dirs:
            all_files = self._get_files(ddir)
            
            for filename in all_files:
                # [篩選] 只保留 RS307 和 RS330
                specimen_name = None
                for name in target_specimens:
                    if name in filename:
                        specimen_name = name
                        break
                
                if specimen_name is None:
                    continue 

                self.filenames.append(filename)

                props = self.specimen_props[specimen_name]
                vec_AR = props["AR"]
                vec_LR = props["LR"]
                vec_TR = props["TR"]

                for category_prefix, category_idx in self.height_map.items():
                    if filename.startswith(f"{ddir}/{category_prefix}"):
                        # file_idx = int(filename.split('/')[-1].replace(category_prefix, "").replace('.jpg', '').lstrip('0'))
                        num_part = filename.split('_')[-1].replace('.jpg', '')
                        file_idx = int(num_part)
                        angle_idx = [category_idx, file_idx]
                        final_label = vec_AR + vec_LR + vec_TR + angle_idx
                        self.labels[filename] = final_label
                        break 
            root.append(ddir)
        # for dir_idx, ddir in enumerate(self.root):
        #     filenames = self._get_files(ddir)
        #     self.filenames.extend(filenames)

        #     # 為每個資料夾分配單一標籤 [dir_idx]
        #     for filename in filenames:
        #         self.labels[filename] = [dir_idx]  # 僅保留資料夾索引作為標籤

    def __len__(self):
        return len(self.filenames)

    @staticmethod
    def _get_files(root_dir):
        return glob.glob(f'{root_dir}/*.png') + glob.glob(f'{root_dir}/*.jpg' )+ glob.glob(f'{root_dir}/*.PNG')

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img = Image.open(filename).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels.get((filename), 0)
        label = torch.tensor(label, dtype=torch.float32)
        device = img.device
        label = label.to(device)
        return img, label
    # def __getitem__(self, idx):
    #     filename = self.filenames[idx]
    #     # 1. 讀取圖片
    #     img = Image.open(filename).convert('RGB')
    #     # --- [新增] 強制去背邏輯 (Kill Grey Fog) ---
    #     # 把圖片轉成 numpy 陣列以便操作像素
    #     arr = np.array(img)
    #     # 設定閾值：因為您的背景約為 59，設 65 比較保險 (能包含雜訊)
    #     # 但要確保不會誤刪到柱子的陰影 (通常柱子都很亮 > 100)
    #     threshold = 65  
    #     # 建立遮罩：找出 R, G, B 三個通道數值都小於閾值的像素
    #     mask = (arr[:, :, 0] < threshold) & (arr[:, :, 1] < threshold) & (arr[:, :, 2] < threshold)
    #     # 將符合條件的像素強制設為純黑 [0, 0, 0]
    #     arr[mask] = [0, 0, 0]
    #     # 轉回 PIL Image 格式，這樣後面的 self.transform 才能正常運作
    #     img = Image.fromarray(arr)
    #     # ---------------------------------------
    #     # 2. 執行原本的轉換 (ToTensor, Resize 等)
    #     if self.transform is not None:
    #         img = self.transform(img)
    #     # 3. 處理標籤
    #     label = self.labels.get((filename), 0)
    #     label = torch.tensor(label, dtype=torch.float32)
    #     device = img.device
    #     label = label.to(device)
    #     return img, label

class Carla(ImageDataset):
    def __init__(self, *args, **kwargs):
        super(Carla, self).__init__(*args, **kwargs)
    
class RS307_0_i2(ImageDataset):
    def __init__(self, *args, **kwargs):
        super(RS307_0_i2, self).__init__(*args, **kwargs)