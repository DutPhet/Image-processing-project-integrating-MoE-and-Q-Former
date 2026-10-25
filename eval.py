'''
Kịch bản đánh giá chất lượng mô hình trên tập kiểm thử GoPro Test (Evaluation):
Đo đạc 2 chỉ số tiêu chuẩn cho báo cáo bài tập lớn:
1. PSNR (Peak Signal-to-Noise Ratio - dB)
2. SSIM (Structural Similarity Index)
'''

import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.forge import MoEQFormerDeblurNet
from utils.dataset import GoProDeblurDataset
from utils.metrics import calc_psnr, calc_ssim

def parse_args():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình GoPro Deblurring")
    parser.add_argument('--model_path', type=str, default='forge_moe_best.pth',
                        help="Đường dẫn file checkpoint (.pth)")
    parser.add_argument('--test_dir', type=str, default='gopro-data/test',
                        help="Đường dẫn thư mục kiểm thử GoPro")
    parser.add_argument('--max_samples', type=int, default=0,
                        help="Giới hạn số ảnh test để kiểm tra nhanh (0 = test toàn bộ)")
    return parser.parse_args()

def evaluate():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==================================================")
    print(f"Bắt đầu đánh giá GoPro Deblurring trên: {device}")
    print(f"Đường dẫn checkpoint : {args.model_path}")
    print(f"Thư mục test         : {args.test_dir}")
    print("==================================================")

    # 1. KHỞI TẠO MÔ HÌNH VÀ TẢI WEIGHTS
    model = MoEQFormerDeblurNet().to(device)

    if not os.path.exists(args.model_path):
        print(f"[!] Lỗi: Không tìm thấy file {args.model_path}!")
        print("    Vui lòng kiểm tra lại đường dẫn file checkpoint sau khi huấn luyện.")
        return

    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("--> Đã tải thành công trọng số mô hình.")

    model.eval()

    # 2. KHỞI TẠO DATASET TEST (batch_size=1, giữ nguyên toàn bộ bức ảnh)
    if not os.path.exists(args.test_dir):
        print(f"[!] Lỗi: Không tìm thấy thư mục {args.test_dir}!")
        return

    test_dataset = GoProDeblurDataset(root_dir=args.test_dir, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    total_imgs = len(test_dataset) if args.max_samples == 0 else min(args.max_samples, len(test_dataset))
    print(f"--> Đang đánh giá trên {total_imgs:,} ảnh kiểm thử...")

    # 3. VÒNG LẶP ĐÁNH GIÁ (TEST LOOP)
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, total=total_imgs, desc="Đang đánh giá"):
            inputs, targets = inputs.to(device), targets.to(device)

            preds, _ = model(inputs)
            preds = torch.clamp(preds, 0.0, 1.0)

            psnr = calc_psnr(preds, targets)
            ssim = calc_ssim(preds, targets)

            total_psnr += psnr
            total_ssim += ssim
            count += 1

            if args.max_samples > 0 and count >= args.max_samples:
                break

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count

    print("==================================================")
    print(" KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP GOPRO TEST:")
    print(f" -> PSNR trung bình : {avg_psnr:.2f} dB")
    print(f" -> SSIM trung bình : {avg_ssim:.4f}")
    print("==================================================")

if __name__ == '__main__':
    evaluate()
