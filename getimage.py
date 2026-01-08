import os
import cv2
import numpy as np
import torch
from torchvision.utils import save_image

def create_horizontal_collage_from_two_dirs(dir1, dir2, selected_filenames1, selected_filenames2,
                                          output_path='128realimage_cwdata.jpg', target_size=(128, 128)):
    """
    從兩個目錄中讀取圖片並創建水平拼貼，使用 PyTorch 的 save_image 保存
    
    Parameters:
    dir1: 第一個目錄路徑
    dir2: 第二個目錄路徑
    selected_filenames1: 第一個目錄中選擇的文件列表
    selected_filenames2: 第二個目錄中選擇的文件列表
    output_path: 輸出圖片路徑
    target_size: 目標尺寸，格式為 (寬, 高)
    """
    
    def resize_keep_aspect(img, target_size):
        target_width, target_height = target_size
        h, w = img.shape[:2]
        scale = min(target_width/w, target_height/h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized
    
    # 檢查兩個目錄中選定的文件是否存在
    for filename in selected_filenames1:
        if not os.path.exists(os.path.join(dir1, filename)):
            raise FileNotFoundError(f"文件 {filename} 在目錄 {dir1} 中不存在")
    
    for filename in selected_filenames2:
        if not os.path.exists(os.path.join(dir2, filename)):
            raise FileNotFoundError(f"文件 {filename} 在目錄 {dir2} 中不存在")
    
    # 儲存轉換後的 PyTorch 張量
    tensor_images = []
    
    # 處理第一個目錄的圖片
    for filename in selected_filenames1:
        img_path = os.path.join(dir1, filename)
        img = cv2.imread(img_path)
        resized_img = resize_keep_aspect(img, target_size)
        
        # 處理可能的尺寸差異 (確保所有圖像尺寸一致)
        h, w = resized_img.shape[:2]
        if h != target_size[1] or w != target_size[0]:
            # 創建空白畫布
            canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            # 將調整大小後的圖像放置在中心
            y_offset = (target_size[1] - h) // 2
            x_offset = (target_size[0] - w) // 2
            canvas[y_offset:y_offset+h, x_offset:x_offset+w] = resized_img
            resized_img = canvas
        
        # OpenCV 讀取的圖像是 BGR 格式，需要轉換為 RGB
        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        
        # 轉換為 PyTorch 張量 [C, H, W] 並正規化到 [0, 1]
        img_tensor = torch.from_numpy(rgb_img.transpose(2, 0, 1)).float() / 255.0
        tensor_images.append(img_tensor)
    
    # 處理第二個目錄的圖片
    for filename in selected_filenames2:
        img_path = os.path.join(dir2, filename)
        img = cv2.imread(img_path)
        resized_img = resize_keep_aspect(img, target_size)
        
        # 處理可能的尺寸差異
        h, w = resized_img.shape[:2]
        if h != target_size[1] or w != target_size[0]:
            canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            y_offset = (target_size[1] - h) // 2
            x_offset = (target_size[0] - w) // 2
            canvas[y_offset:y_offset+h, x_offset:x_offset+w] = resized_img
            resized_img = canvas
        
        # OpenCV 讀取的圖像是 BGR 格式，需要轉換為 RGB
        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        
        # 轉換為 PyTorch 張量 [C, H, W] 並正規化到 [0, 1]
        img_tensor = torch.from_numpy(rgb_img.transpose(2, 0, 1)).float() / 255.0
        tensor_images.append(img_tensor)
    
    # 將所有圖像張量堆疊為一個批次
    batch = torch.stack(tensor_images)
    
    # 使用 save_image 保存為水平排列的圖片
    # 設定 nrow 為圖像數量，使其水平排列 (單行)
    save_image(batch, output_path, nrow=len(tensor_images))
    
    print(f"拼接圖已保存到 {output_path}")
    print(f"圖像尺寸: {target_size[1]}x{target_size[0]}, 數量: {len(tensor_images)}")

# 使用示例
if __name__ == "__main__":
    # 定義兩個目錄和相應的文件列表
    dir1 = 'data/long/RS307'
    dir2 = 'data/long/RS330'
    
    selected_images1 = ['0_0090.jpg', '0_0135.jpg', '0_0180.jpg', '0_0225.jpg']
    selected_images2 = ['0_0270.jpg', '0_0315.jpg', '0_0000.jpg', '0_0045.jpg']
    
    # selected_images1 = ['1524.jpg', '2924.jpg', '5624.jpg', '7124.jpg']
    # selected_images2 = ['8247.jpg', '8238.jpg', '8222.jpg', '8214.jpg']

    # 指定目標尺寸 (寬, 高)
    target_size = (128, 128)  # 可以根據需要調整
    
    # 創建拼貼
    create_horizontal_collage_from_two_dirs(
        dir1, dir2, 
        selected_images1, selected_images2,
        target_size=target_size,
        output_path='RS307330_long.jpg'
    )