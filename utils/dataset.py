'''
DataLoader chuyên biệt cho bài toán Khử mờ ảnh chuyển động (GoPro Motion Deblurring):
Đọc cặp ảnh (blur và sharp) từ cấu trúc thư mục của tập GoPro, cắt patch đồng bộ và data augmentation.
'''

import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image

class GoProDeblurDataset(Dataset):
    """
    Dataset nạp ảnh từ tập dữ liệu GoPro:
    Cấu trúc thư mục:
    root_dir/
      GOPR0384_11_00/
        blur/000001.png
        sharp/000001.png
    """
    def __init__(self, root_dir, crop_size=256, is_train=True):
        super().__init__()
        self.crop_size = crop_size
        self.is_train = is_train
        
        root_path = Path(root_dir)
        # Quét tất cả các file ảnh trong các thư mục con blur/ và sharp/
        self.blur_paths = sorted(list(root_path.glob('*/blur/*.png')) + list(root_path.glob('*/blur/*.jpg')))
        self.sharp_paths = sorted(list(root_path.glob('*/sharp/*.png')) + list(root_path.glob('*/sharp/*.jpg')))
        
        assert len(self.blur_paths) > 0, f"Không tìm thấy ảnh mờ nào trong: {root_dir}"
        assert len(self.blur_paths) == len(self.sharp_paths), (
            f"Số lượng ảnh blur ({len(self.blur_paths)}) và sharp ({len(self.sharp_paths)}) không bằng nhau!"
        )

    def __len__(self):
        return len(self.blur_paths)

    def __getitem__(self, idx):
        blur_img = Image.open(self.blur_paths[idx]).convert('RGB')
        sharp_img = Image.open(self.sharp_paths[idx]).convert('RGB')

        if self.is_train:
            # 1. Cắt patch đồng bộ (Paired Random Crop)
            w, h = blur_img.size
            if w >= self.crop_size and h >= self.crop_size:
                x = random.randint(0, w - self.crop_size)
                y = random.randint(0, h - self.crop_size)
                blur_img = blur_img.crop((x, y, x + self.crop_size, y + self.crop_size))
                sharp_img = sharp_img.crop((x, y, x + self.crop_size, y + self.crop_size))

            # 2. Tăng cường dữ liệu đồng bộ (Data Augmentation)
            if random.random() > 0.5:
                blur_img = F.hflip(blur_img)
                sharp_img = F.hflip(sharp_img)
            if random.random() > 0.5:
                blur_img = F.vflip(blur_img)
                sharp_img = F.vflip(sharp_img)

        # 3. Chuyển đổi sang Tensor chuẩn PyTorch [0.0, 1.0]
        blur_tensor = F.to_tensor(blur_img)
        sharp_tensor = F.to_tensor(sharp_img)
        return blur_tensor, sharp_tensor
