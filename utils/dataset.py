'''
Chứa mã nguồn của DataLoader để xử lý dữ liệu đầu vào. 
Cắt ảnh thành các patch nhỏ, tạo dữ liệu ảnh mờ và thực hiện Data Augmentation như xoay, lật ảnh.
'''
import os
import random
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image

class RestorationDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, patch_size=48, scale=4, is_train=True):
        """
        Args:
            lr_dir (str): Đường dẫn đến thư mục chứa ảnh LR (ảnh mờ).
            hr_dir (str): Đường dẫn đến thư mục chứa ảnh HR (ảnh nét).
            patch_size (int): Kích thước vùng cắt cho ảnh LR (thường là 48).
            scale (int): Hệ số phóng đại (thường là 4, x4).
            is_train (bool): Nếu True thì thực hiện cắt patch và augmentation.
        """
        super().__init__()
        # Lấy danh sách đường dẫn tất cả các ảnh và sắp xếp theo thứ tự alpha-bê
        self.lr_paths = sorted([os.path.join(lr_dir, f) for f in os.listdir(lr_dir) if f.endswith(('.png', '.jpg'))])
        self.hr_paths = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir) if f.endswith(('.png', '.jpg'))])
        
        # Kiểm tra để đảm bảo số lượng ảnh LR và HR bằng nhau
        assert len(self.lr_paths) == len(self.hr_paths), "Số lượng ảnh LR và HR không khớp nhau!"
        
        self.patch_size = patch_size
        self.scale = scale
        self.is_train = is_train

    def __len__(self):
        return len(self.lr_paths)

    def __getitem__(self, idx):
        # Mở ảnh bằng thư viện PIL và chuyển về hệ màu RGB
        lr_img = Image.open(self.lr_paths[idx]).convert('RGB')
        hr_img = Image.open(self.hr_paths[idx]).convert('RGB')

        if self.is_train:
            # 1. CẮT PATCH ĐỒNG BỘ (Paired Random Crop)
            lr_w, lr_h = lr_img.size
            hr_w, hr_h = hr_img.size
            
            # Chọn một tọa độ (x, y) ngẫu nhiên trên ảnh mờ (LR)
            lr_x = random.randint(0, lr_w - self.patch_size)
            lr_y = random.randint(0, lr_h - self.patch_size)
            
            # Tính toán tọa độ tương ứng trên ảnh nét (HR) dựa vào scale (x4)
            hr_x = lr_x * self.scale
            hr_y = lr_y * self.scale
            hr_patch_size = self.patch_size * self.scale

            # Thực hiện cắt ảnh
            lr_img = lr_img.crop((lr_x, lr_y, lr_x + self.patch_size, lr_y + self.patch_size))
            hr_img = hr_img.crop((hr_x, hr_y, hr_x + hr_patch_size, hr_y + hr_patch_size))

            # 2. TĂNG CƯỜNG DỮ LIỆU ĐỒNG BỘ (Paired Data Augmentation)
            # Xác suất 50% lật ngang cả hai ảnh
            if random.random() > 0.5:
                lr_img = F.hflip(lr_img)
                hr_img = F.hflip(hr_img)
                
            # Xác suất 50% lật dọc cả hai ảnh
            if random.random() > 0.5:
                lr_img = F.vflip(lr_img)
                hr_img = F.vflip(hr_img)
        
        # 3. CHUYỂN ĐỔI SANG TENSOR
        # Hàm to_tensor sẽ tự động chuyển pixel từ [0, 255] về chuẩn [0.0, 1.0] cho PyTorch
        lr_tensor = F.to_tensor(lr_img)
        hr_tensor = F.to_tensor(hr_img)

        return lr_tensor, hr_tensor