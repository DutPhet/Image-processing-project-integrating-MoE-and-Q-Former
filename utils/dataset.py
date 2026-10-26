import os
import re
import random
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image

TASK_MAPPING = {
    'deblur': 0,
    'dehaze': 1,
    'denoise': 2,
    'derain': 3,
    'desnow': 4,
    'lowlight': 5,
    'enhance': 5
}

VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG'}


# ==============================================================================
# BỘ TRÍCH XUẤT KEY GOM NHÓM CHỐNG DATA LEAKAGE
# ==============================================================================

def get_group_key(filepath, task_name, index=0):
    stem = Path(filepath).stem
    parts = stem.split('_')

    # 1. DEBLUR: GOPR0372_07_00_000047.png -> Key: GOPR0372_07
    if task_name == 'deblur':
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        return stem

    # 2. DEHAZE: 0025_0.8_0.1.jpg -> Key: 0025
    elif task_name == 'dehaze':
        return parts[0]

    # 3. DENOISE: 0054_003_N6_... -> Key: 003
    elif task_name == 'denoise':
        if len(parts) >= 2:
            return parts[1]
        return stem

    # 4. LOWLIGHT: Khối 4 ảnh liên tiếp
    elif task_name in ['lowlight', 'enhance']:
        return f"group_block_{index // 4}"

    return stem


def _get_pairs_from_subfolder(base_dir):
    """Trích xuất tất cả cặp input - target trong một thư mục chỉ định"""
    input_names = {'input', 'blur', 'noisy', 'rainy', 'degraded', 'hazy', 'snowy', 'low'}
    target_names = {'target', 'sharp', 'clean', 'gt', 'high'}

    input_dir, target_dir = None, None
    for sub in base_dir.rglob('*'):
        if sub.is_dir():
            s_name = sub.name.lower()
            if input_dir is None and s_name in input_names:
                input_dir = sub
            elif target_dir is None and s_name in target_names:
                target_dir = sub

    if input_dir is None or target_dir is None:
        return []

    input_files = sorted([str(p) for p in input_dir.glob('*.*') if p.suffix in VALID_EXTENSIONS])
    target_files = sorted([str(p) for p in target_dir.glob('*.*') if p.suffix in VALID_EXTENSIONS])

    if len(input_files) != len(target_files) or len(input_files) == 0:
        return []

    return list(zip(input_files, target_files))


# ==============================================================================
# HÀM PHÂN CHIA DATASET CHUẨN ĐỘC LẬP TRAIN VÀ TEST
# ==============================================================================

def get_multitask_balanced_splits(
    root_dir,
    val_ratio=0.15,
    seed=42,
    max_train_samples_per_task=1200,
    max_test_samples_per_task=200
):
    """
    Phân chia Train/Val/Test độc lập hoàn toàn:
    - Tập TEST: Lấy 100% từ folder <task>/test/ gốc, có giới hạn max_test_samples.
    - Tập TRAIN/VAL: Lấy 100% từ folder <task>/train/ gốc, có giới hạn max_train_samples.
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục: {root_dir}")

    random.seed(seed)
    train_samples, val_samples, test_samples = [], [], []

    task_dirs = sorted([d for d in root_path.iterdir() if d.is_dir()])

    print("=" * 80)
    print(f"--> [Split Data] Phân chia dữ liệu từ: {root_dir}")
    print(f"     + TRAIN / VAL : Nguồn từ folder <task>/train/ (Giới hạn max {max_train_samples_per_task} ảnh/task)")
    print(f"     + TEST        : Nguồn từ folder <task>/test/  (Giới hạn max {max_test_samples_per_task} ảnh/task)")
    print("=" * 80)

    for task_dir in task_dirs:
        task_name = task_dir.name.lower()
        if task_name not in TASK_MAPPING:
            continue

        task_id = TASK_MAPPING[task_name]

        train_dir = task_dir / 'train'
        test_dir = task_dir / 'test'

        # ----------------------------------------------------------------------
        # 1. TẬP TEST (Đọc duy nhất từ folder test/ gốc)
        # ----------------------------------------------------------------------
        raw_test_pairs = _get_pairs_from_subfolder(test_dir) if test_dir.exists() else []
        t_test = []

        if raw_test_pairs:
            test_pairs = [(inp, tgt, task_id, task_name) for inp, tgt in raw_test_pairs]

            if task_name in ['derain', 'desnow']:
                random.shuffle(test_pairs)
                if max_test_samples_per_task and len(test_pairs) > max_test_samples_per_task:
                    test_pairs = test_pairs[:max_test_samples_per_task]
                t_test = test_pairs
            else:
                test_groups = defaultdict(list)
                for idx, pair in enumerate(test_pairs):
                    group_key = get_group_key(pair[0], task_name, index=idx)
                    test_groups[group_key].append(pair)

                t_keys = list(test_groups.keys())
                random.shuffle(t_keys)

                curr_c = 0
                for k in t_keys:
                    g_size = len(test_groups[k])
                    if max_test_samples_per_task and curr_c + g_size > max_test_samples_per_task and len(t_test) > 0:
                        break
                    t_test.extend(test_groups[k])
                    curr_c += g_size

        # ----------------------------------------------------------------------
        # 2. TẬP TRAIN & VAL (Đọc duy nhất từ folder train/ gốc)
        # ----------------------------------------------------------------------
        raw_train_pairs = _get_pairs_from_subfolder(train_dir) if train_dir.exists() else []
        train_pairs = [(inp, tgt, task_id, task_name) for inp, tgt in raw_train_pairs]

        t_train, t_val = [], []

        if train_pairs:
            if task_name in ['derain', 'desnow']:
                random.shuffle(train_pairs)
                if max_train_samples_per_task and len(train_pairs) > max_train_samples_per_task:
                    train_pairs = train_pairs[:max_train_samples_per_task]

                n_val = max(1, int(len(train_pairs) * val_ratio))
                t_val = train_pairs[:n_val]
                t_train = train_pairs[n_val:]
            else:
                train_groups = defaultdict(list)
                for idx, pair in enumerate(train_pairs):
                    group_key = get_group_key(pair[0], task_name, index=idx)
                    train_groups[group_key].append(pair)

                tr_keys = list(train_groups.keys())
                random.shuffle(tr_keys)

                selected_tr_keys = []
                curr_c = 0
                for k in tr_keys:
                    g_size = len(train_groups[k])
                    if max_train_samples_per_task and curr_c + g_size > max_train_samples_per_task and len(selected_tr_keys) > 0:
                        break
                    selected_tr_keys.append(k)
                    curr_c += g_size

                n_val_g = max(1, int(len(selected_tr_keys) * val_ratio))
                val_k = selected_tr_keys[:n_val_g]
                train_k = selected_tr_keys[n_val_g:]

                for k in val_k:
                    t_val.extend(train_groups[k])
                for k in train_k:
                    t_train.extend(train_groups[k])

        train_samples.extend(t_train)
        val_samples.extend(t_val)
        test_samples.extend(t_test)

        print(f"  + Task [{task_name:8s}] (ID: {task_id}): Train = {len(t_train):4d} | Val = {len(t_val):3d} | Test (Gốc) = {len(t_test):3d}")

    print("-" * 80)
    print(f"--> TỔNG CỘNG: Train = {len(train_samples)} | Val = {len(val_samples)} | Test = {len(test_samples)} ảnh")
    print("=" * 80)

    return train_samples, val_samples, test_samples


# ==============================================================================
# DATASET CLASS CHO DATALOADER
# ==============================================================================

class MultiTaskRestorationDataset(Dataset):
    def __init__(self, samples=None, root_dir=None, crop_size=192, is_train=True, tasks=None):
        super().__init__()
        self.crop_size = crop_size
        self.is_train = is_train
        self.samples = []

        if samples is not None:
            self.samples = samples
        elif root_dir is not None:
            self.samples, _, _ = get_multitask_balanced_splits(root_dir)

        if len(self.samples) == 0:
            raise FileNotFoundError("Không tìm thấy mẫu dữ liệu hợp lệ nào!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp_path, tgt_path, task_id, task_name = self.samples[idx]

        inp_img = Image.open(inp_path).convert('RGB')
        tgt_img = Image.open(tgt_path).convert('RGB')

        w, h = inp_img.size

        if self.is_train:
            if self.crop_size is not None and w >= self.crop_size and h >= self.crop_size:
                x = random.randint(0, w - self.crop_size)
                y = random.randint(0, h - self.crop_size)
                inp_img = inp_img.crop((x, y, x + self.crop_size, y + self.crop_size))
                tgt_img = tgt_img.crop((x, y, x + self.crop_size, y + self.crop_size))

            if random.random() > 0.5:
                inp_img = F.hflip(inp_img)
                tgt_img = F.hflip(tgt_img)
            if random.random() > 0.5:
                inp_img = F.vflip(inp_img)
                tgt_img = F.vflip(tgt_img)

        inp_tensor = F.to_tensor(inp_img)
        tgt_tensor = F.to_tensor(tgt_img)

        return inp_tensor, tgt_tensor, task_id, task_name