'''
DataLoader chuyên biệt cho bài toán Khử mờ ảnh chuyển động (GoPro Motion Deblurring):
Hỗ trợ linh hoạt nhiều kiểu cấu trúc thư mục trên cả Local và Kaggle (/kaggle/input/...).
'''

import os
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image

class GoProDeblurDataset(Dataset):
    """
    Dataset nạp ảnh từ tập dữ liệu GoPro.
    Hỗ trợ linh hoạt cấu trúc thư mục lồng nhau hoặc phẳng:
      root_dir/*/blur/*.png (Cấu trúc gốc GoPro)
      root_dir/blur/*.png   (Cấu trúc phẳng)
    """
    def __init__(self, root_dir, crop_size=256, is_train=True):
        super().__init__()
        self.crop_size = crop_size
        self.is_train = is_train
        
        if not os.path.exists(root_dir):
            error_msg = f"\n[!] KHÔNG TÌM THẤY THƯ MỤC: '{root_dir}'\n"
            if os.path.exists('/kaggle/input'):
                available = os.listdir('/kaggle/input')
                error_msg += f"--> [Kaggle] Các dataset hiện có trong /kaggle/input: {available}\n"
                error_msg += "--> Hãy kiểm tra lại đường dẫn dataset bạn vừa Add Data trên Kaggle!"
            raise FileNotFoundError(error_msg)

        root_path = Path(root_dir)
        
        # 1. Thử quét dạng thư mục GoPro tiêu chuẩn (mỗi video 1 thư mục con: root_dir/*/blur/*.png)
        self.blur_paths = sorted(list(root_path.glob('*/blur/*.png')) + list(root_path.glob('*/blur/*.jpg')))
        self.sharp_paths = sorted(list(root_path.glob('*/sharp/*.png')) + list(root_path.glob('*/sharp/*.jpg')))

        # 2. Nếu không thấy, thử tìm dạng phẳng (root_dir/blur/*.png)
        if len(self.blur_paths) == 0:
            self.blur_paths = sorted(list(root_path.glob('blur/*.png')) + list(root_path.glob('blur/*.jpg')))
            self.sharp_paths = sorted(list(root_path.glob('sharp/*.png')) + list(root_path.glob('sharp/*.jpg')))

        # 3. Nếu vẫn không thấy, quét đệ quy rglob sâu bên trong
        if len(self.blur_paths) == 0:
            self.blur_paths = sorted(list(root_path.rglob('blur/*.png')) + list(root_path.rglob('blur/*.jpg')))
            self.sharp_paths = sorted(list(root_path.rglob('sharp/*.png')) + list(root_path.rglob('sharp/*.jpg')))
        
        if len(self.blur_paths) == 0:
            raise FileNotFoundError(
                f"\n[!] Đã tìm thấy thư mục '{root_dir}' nhưng KHÔNG thấy ảnh trong thư mục con 'blur/'.\n"
                f"Hãy kiểm tra xem bên trong '{root_dir}' có chứa các thư mục blur và sharp không."
            )
            
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
