import os
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image

def get_gopro_sequence_splits(root_dir, val_clips_count=4, val_clip_names=None):
    root_path = Path(root_dir)
    all_clips = sorted([
        d.name for d in root_path.iterdir()
        if d.is_dir() and (d / 'blur').exists()
    ])

    if len(all_clips) == 0:
        raise FileNotFoundError(f"Không tìm thấy thư mục clip chứa 'blur' tại: {root_dir}")

    if val_clip_names is not None:
        val_set = set(val_clip_names)
        train_clips = [c for c in all_clips if c not in val_set]
        val_clips = [c for c in all_clips if c in val_set]
    else:
        train_clips = all_clips[:-val_clips_count]
        val_clips = all_clips[-val_clips_count:]

    return train_clips, val_clips

class GoProDeblurDataset(Dataset):
    def __init__(self, root_dir, clip_names=None, crop_size=256, is_train=True):
        super().__init__()
        self.crop_size = crop_size
        self.is_train = is_train

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Không tìm thấy thư mục: {root_dir}")

        root_path = Path(root_dir)
        self.blur_paths = []
        self.sharp_paths = []

        # CẢI TIẾN: Ép kiểu sang chuỗi str() trước khi sort để đảm bảo an toàn 100% cặp ảnh mờ-nét
        if clip_names is not None:
            for clip_name in sorted(clip_names):
                clip_dir = root_path / clip_name
                if clip_dir.exists():
                    b_files = [str(p) for p in clip_dir.rglob('blur/*.*') if p.suffix in ['.png', '.jpg']]
                    s_files = [str(p) for p in clip_dir.rglob('sharp/*.*') if p.suffix in ['.png', '.jpg']]
                    self.blur_paths.extend(sorted(b_files))
                    self.sharp_paths.extend(sorted(s_files))
        else:
            b_files = [str(p) for p in root_path.rglob('blur/*.*') if p.suffix in ['.png', '.jpg']]
            s_files = [str(p) for p in root_path.rglob('sharp/*.*') if p.suffix in ['.png', '.jpg']]
            self.blur_paths = sorted(b_files)
            self.sharp_paths = sorted(s_files)

        if len(self.blur_paths) == 0:
            raise FileNotFoundError(f"Thư mục {root_dir} không chứa ảnh blur/sharp phù hợp.")

        assert len(self.blur_paths) == len(self.sharp_paths), \
            f"Số lượng ảnh không khớp: Blur ({len(self.blur_paths)}) vs Sharp ({len(self.sharp_paths)})!"

    def __len__(self):
        return len(self.blur_paths)

    def __getitem__(self, idx):
        blur_img = Image.open(self.blur_paths[idx]).convert('RGB')
        sharp_img = Image.open(self.sharp_paths[idx]).convert('RGB')

        w, h = blur_img.size

        if self.is_train:
            # Data Augmentation (Chỉ áp dụng khi train)
            if self.crop_size is not None and w >= self.crop_size and h >= self.crop_size:
                x = random.randint(0, w - self.crop_size)
                y = random.randint(0, h - self.crop_size)
                blur_img = blur_img.crop((x, y, x + self.crop_size, y + self.crop_size))
                sharp_img = sharp_img.crop((x, y, x + self.crop_size, y + self.crop_size))

            if random.random() > 0.5:
                blur_img = F.hflip(blur_img)
                sharp_img = F.hflip(sharp_img)
            if random.random() > 0.5:
                blur_img = F.vflip(blur_img)
                sharp_img = F.vflip(sharp_img)
        # CẢI TIẾN: Bỏ qua hoàn toàn phép cắt (CenterCrop) khi Validation/Test
        # Để mô hình đánh giá trên toàn bộ độ phân giải gốc 720x1280.

        blur_tensor = F.to_tensor(blur_img)
        sharp_tensor = F.to_tensor(sharp_img)
        return blur_tensor, sharp_tensor