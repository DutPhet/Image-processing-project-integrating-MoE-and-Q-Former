import os
import json
import random
import hashlib
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

# Bộ nhớ tạm RAM lưu mã MD5 để tối ưu tốc độ đọc ổ đĩa
_MD5_CACHE = {}


# ==============================================================================
# HÀM TÍNH MD5 HASH VÀ TRÍCH XUẤT GROUP KEY CHO TẤT CẢ CÁC TASK
# ==============================================================================

def get_file_md5(filepath):
    """Tính mã MD5 hash từ nội dung nhị phân của file ảnh (Có RAM Caching)"""
    str_path = str(filepath)
    if str_path in _MD5_CACHE:
        return _MD5_CACHE[str_path]

    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    
    md5_val = hasher.hexdigest()
    _MD5_CACHE[str_path] = md5_val
    return md5_val


def get_group_key(inp_path, tgt_path, task_name, index=0):
    """
    Quy tắc gom nhóm chuẩn cho từng tác vụ để chống Data Leakage:
    - derain, desnow, lowlight, enhance: Gom bằng MD5 Hash của ảnh Target (Clean).
    - deblur, dehaze, denoise: Gom bằng Prefix/Stem của tên file.
    """
    # 1. DERAIN, DESNOW, LOWLIGHT, ENHANCE: Gom nhóm bằng MD5 Hash của ảnh Clean Target
    if task_name in ['derain', 'desnow', 'lowlight', 'enhance']:
        return get_file_md5(tgt_path)

    stem = Path(inp_path).stem
    parts = stem.split('_')

    # 2. DEBLUR: GOPR0372_07_00_000047.png -> Key: GOPR0372_07
    if task_name == 'deblur':
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        return stem

    # 3. DEHAZE: 0025_0.8_0.1.jpg -> Key: 0025 (Trả về quy tắc prefix cũ)
    elif task_name == 'dehaze':
        return parts[0]

    # 4. DENOISE: 0054_003_N6_... -> Key: 003
    elif task_name == 'denoise':
        if len(parts) >= 2:
            return parts[1]
        return stem

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
# HÀM PHÂN CHIA MULTI-TASK SPLITS (LEAKPROOF & OPTIMIZED)
# ==============================================================================

def get_multitask_balanced_splits(
    root_dir,
    val_ratio=0.15,
    seed=42,
    max_train_samples_per_task=1200,
    max_test_samples_per_task=200,
    cache_file="multitask_splits_cache.json"
):
    """
    Phân chia Train/Val/Test tối ưu:
    - Nạp từ cache JSON nếu đã tồn tại.
    - DEHAZE gom theo Prefix 0025.
    - DERAIN / DESNOW / LOWLIGHT gom theo MD5 Target.
    - Cân bằng số ảnh Val chính xác ~15% tổng số mẫu.
    - Lấy tròn đủ max_test_samples_per_task cho tập Test.
    """
    # 1. NẠP NHANH TỪ CACHE NẾU ĐÃ TỒN TẠI
    if os.path.exists(cache_file):
        print(f"--> [FAST LOAD] Đã nạp thành công split từ file Cache: {cache_file}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            cached_data = json.load(f)
        return cached_data['train'], cached_data['val'], cached_data['test']

    # 2. KHỞI TẠO VÀ CHIA SPLIT CHO TẤT CẢ CÁC TASK
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục dataset: {root_dir}")

    random.seed(seed)
    train_samples, val_samples, test_samples = [], [], []
    task_dirs = sorted([d for d in root_path.iterdir() if d.is_dir()])

    print("=" * 80)
    print(f"--> [Optimized Leakproof Split] Chia dữ liệu cho TẤT CẢ các task: {root_dir}")
    print(f"     + TRAIN / VAL : Nguồn từ folder <task>/train/ (Max {max_train_samples_per_task} ảnh/task)")
    print(f"     + TEST        : Nguồn từ folder <task>/test/  (Max {max_test_samples_per_task} ảnh/task)")
    print("=" * 80)

    for task_dir in task_dirs:
        task_name = task_dir.name.lower()
        if task_name not in TASK_MAPPING:
            continue

        task_id = TASK_MAPPING[task_name]
        train_dir = task_dir / 'train'
        test_dir = task_dir / 'test'

        # ----------------------------------------------------------------------
        # A. XỬ LÝ TẬP TRAIN VÀ VAL CHO TASK NÀY
        # ----------------------------------------------------------------------
        raw_train_pairs = _get_pairs_from_subfolder(train_dir) if train_dir.exists() else []
        train_pairs = [(inp, tgt, task_id, task_name) for inp, tgt in raw_train_pairs]

        t_train, t_val = [], []
        train_target_md5_set = set()

        if train_pairs:
            train_groups = defaultdict(list)
            for idx, pair in enumerate(train_pairs):
                group_key = get_group_key(pair[0], pair[1], task_name, index=idx)
                train_groups[group_key].append(pair)

            tr_keys = list(train_groups.keys())
            random.shuffle(tr_keys)

            # Chọn các Group cho đến khi gần đạt trần max_train_samples_per_task
            selected_tr_keys = []
            curr_c = 0
            for k in tr_keys:
                g_size = len(train_groups[k])
                if max_train_samples_per_task and curr_c + g_size > max_train_samples_per_task and len(selected_tr_keys) > 0:
                    break
                selected_tr_keys.append(k)
                curr_c += g_size

            # CÂN BẰNG TỶ LỆ VAL THEO SỐ ẢNH THỰC TẾ
            total_train_samples = sum(len(train_groups[k]) for k in selected_tr_keys)
            target_val_count = int(total_train_samples * val_ratio)

            val_k, train_k = [], []
            curr_val_count = 0

            for k in selected_tr_keys:
                if curr_val_count < target_val_count and len(val_k) < len(selected_tr_keys) - 1:
                    val_k.append(k)
                    curr_val_count += len(train_groups[k])
                else:
                    train_k.append(k)

            for k in val_k:
                t_val.extend(train_groups[k])
            for k in train_k:
                t_train.extend(train_groups[k])
                for item in train_groups[k]:
                    train_target_md5_set.add(get_file_md5(item[1]))

        # ----------------------------------------------------------------------
        # B. XỬ LÝ TẬP TEST CHO TASK NÀY (LẤY ĐỦ SỐ MẪU TỐI ĐA)
        # ----------------------------------------------------------------------
        raw_test_pairs = _get_pairs_from_subfolder(test_dir) if test_dir.exists() else []
        t_test = []

        if raw_test_pairs:
            test_pairs = [(inp, tgt, task_id, task_name) for inp, tgt in raw_test_pairs]

            # Lọc bỏ ảnh Test bị trùng MD5 với Train (chống rò rỉ)
            filtered_test_pairs = [
                pair for pair in test_pairs 
                if get_file_md5(pair[1]) not in train_target_md5_set
            ]

            test_groups = defaultdict(list)
            for idx, pair in enumerate(filtered_test_pairs):
                group_key = get_group_key(pair[0], pair[1], task_name, index=idx)
                test_groups[group_key].append(pair)

            t_keys = list(test_groups.keys())
            random.shuffle(t_keys)

            # Lấy từng ảnh cho đến khi ĐẠT ĐỦ max_test_samples_per_task
            for k in t_keys:
                for pair in test_groups[k]:
                    if max_test_samples_per_task and len(t_test) >= max_test_samples_per_task:
                        break
                    t_test.append(pair)
                if max_test_samples_per_task and len(t_test) >= max_test_samples_per_task:
                    break

        train_samples.extend(t_train)
        val_samples.extend(t_val)
        test_samples.extend(t_test)

        print(f"  + Task [{task_name:8s}] (ID: {task_id}): Train = {len(t_train):4d} | Val = {len(t_val):3d} | Test = {len(t_test):3d}")

    print("-" * 80)
    print(f"--> TỔNG CỘNG LEAKPROOF: Train = {len(train_samples)} | Val = {len(val_samples)} | Test = {len(test_samples)} ảnh")
    print("=" * 80)

    # 3. LƯU LẠI VÀO FILE CACHE
    cache_data = {'train': train_samples, 'val': val_samples, 'test': test_samples}
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2)
    print(f"--> Đã ghi file Cache thành công: {cache_file}")

    return train_samples, val_samples, test_samples


# ==============================================================================
# DATASET CLASS
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