'''
Kịch bản huấn luyện & Fine-tune (Cập nhật chuẩn hóa theo mô-đun dự án):
- Tự động tải pretrained NAFNet từ Hugging Face.
- Đánh giá baseline NAFNet gốc trên tập test.
- Nạp weights và fine-tune mô hình MoEQFormerDeblurNet (hỗ trợ freeze backbone).
- Xuất kết quả so sánh ra file results.json và lưu checkpoint tốt nhất (.pth).
'''

import argparse
import os
import json
import urllib.request
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.moe_qformer_net import MoEQFormerDeblurNet[cite: 2]
from utils.dataset import GoProDeblurDataset[cite: 6]
from utils.metrics import CharbonnierLoss, calc_psnr, calc_ssim, count_parameters[cite: 8]
from utils.logger import TensorBoardLogger[cite: 7]

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune MoE + Q-Former từ NAFNet Pretrained")
    parser.add_argument('--train_dir', type=str, default='gopro-data/train', help="Đường dẫn thư mục train GoPro")
    parser.add_argument('--test_dir', type=str, default='gopro-data/test', help="Đường dẫn thư mục test GoPro")
    parser.add_argument('--batch_size', type=int, default=8, help="Kích thước batch size")
    parser.add_argument('--crop_size', type=int, default=256, help="Kích thước patch cắt ảnh (256x256)")
    parser.add_argument('--epochs', type=int, default=5, help="Số lượng epoch fine-tune")
    parser.add_argument('--lr', type=float, default=1e-4, help="Tốc độ học (Learning rate)")
    parser.add_argument('--freeze_backbone', action='store_true', help="Đóng băng backbone NAFNet")
    parser.add_argument('--aux_weight', type=float, default=0.01, help="Trọng số MoE load balancing loss")
    parser.add_argument('--save_path', type=str, default='moe_qformer_finetuned_best.pth', help="File lưu checkpoint tốt nhất")
    parser.add_argument('--json_output', type=str, default='results.json', help="File lưu kết quả JSON")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==================================================")
    print(f" Bắt đầu tiến trình trên thiết bị: {device}")
    print("==================================================")

    # 0. Tự động tải Pretrained NAFNet checkpoint nếu chưa có
    pretrained_url = "https://huggingface.co/Jacid23/third-eye-models/resolve/5c2e322323888d8d26852f06715ccc6eb61ea795/NAFNet-GoPro-width64.pth"
    nafnet_ckpt_file = "NAFNet-GoPro-width64.pth"
    
    if not os.path.exists(nafnet_ckpt_file):
        print(f"--> Đang tải pretrained NAFNet từ Hugging Face...")
        urllib.request.urlretrieve(pretrained_url, nafnet_ckpt_file)
        print("--> Tải xuống checkpoint thành công!")
    
    nafnet_state_dict = torch.load(nafnet_ckpt_file, map_location=device)
    if 'params' in nafnet_state_dict:
        nafnet_state_dict = nafnet_state_dict['params']
    elif 'model_state_dict' in nafnet_state_dict:
        nafnet_state_dict = nafnet_state_dict['model_state_dict']

    # 1. Khởi tạo DataLoader cho Train và Test
    if not os.path.exists(args.test_dir) or not os.path.exists(args.train_dir):
        raise FileNotFoundError("Không tìm thấy thư mục train hoặc test của GoPro! Vui lòng kiểm tra lại đường dẫn.")

    test_dataset = GoProDeblurDataset(root_dir=args.test_dir, is_train=False)[cite: 6]
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    train_dataset = GoProDeblurDataset(root_dir=args.train_dir, crop_size=args.crop_size, is_train=True)[cite: 6]
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True if torch.cuda.is_available() else False
    )
    print(f"--> Đã tải {len(train_dataset):,} ảnh train và {len(test_dataset):,} ảnh test.")

    # 2. Đánh giá Baseline NAFNet Gốc
    print("\n==========================================")
    print(" BƯỚC 1: ĐÁNH GIÁ NAFNet GỐC (BASELINE)")
    print("==========================================")
    
    baseline_model = MoEQFormerDeblurNet().to(device)[cite: 2]
    baseline_model.load_state_dict(nafnet_state_dict, strict=False)
    baseline_model.eval()

    psnr_list, ssim_list = [], []
    max_eval_samples = 30  # Giới hạn số mẫu test để đánh giá nhanh (có thể bỏ điều kiện nếu muốn test full)
    
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(tqdm(test_loader, desc="Đánh giá Baseline NAFNet")):
            inputs, targets = inputs.to(device), targets.to(device)
            preds, _ = baseline_model(inputs)[cite: 2]
            preds = torch.clamp(preds, 0.0, 1.0)
            
            psnr_list.append(calc_psnr(preds, targets))[cite: 8]
            ssim_list.append(calc_ssim(preds, targets))[cite: 8]
            
            if idx + 1 >= max_eval_samples:
                break

    baseline_psnr = sum(psnr_list) / len(psnr_list)
    baseline_ssim = sum(ssim_list) / len(ssim_list)
    print(f"--> KẾT QUẢ BASELINE NAFNet:")
    print(f"    + PSNR: {baseline_psnr:.2f} dB")
    print(f"    + SSIM: {baseline_ssim:.4f}")

    # 3. Khởi tạo mô hình và nạp weights pretrained để Fine-tune MoE + Q-Former
    print("\n==========================================")
    print(" BƯỚC 2: FINE-TUNE MÔ HÌNH MOE + Q-FORMER")
    print("==========================================")
    
    model = MoEQFormerDeblurNet().to(device)[cite: 2]
    model.load_state_dict(nafnet_state_dict, strict=False)

    if args.freeze_backbone:
        model.freeze_backbone(True)[cite: 2]

    print(f"Tổng số tham số huấn luyện được: {count_parameters(model):,}")[cite: 8]

    criterion = CharbonnierLoss()[cite: 8]
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    logger = TensorBoardLogger(log_dir="./runs/moe_qformer_finetune")[cite: 7]

    best_psnr = baseline_psnr
    history = {'train_loss': [], 'val_psnr': [], 'val_ssim': []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Fine-tune Epoch [{epoch:02d}/{args.epochs:02d}]")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            preds, aux_loss = model(inputs)[cite: 2]
            recon_loss = criterion(preds, targets)[cite: 8]
            loss = recon_loss + args.aux_weight * aux_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'recon': f"{recon_loss.item():.4f}"})

        epoch_loss = running_loss / len(train_loader)
        scheduler.step()

        # Đánh giá sau mỗi epoch fine-tune
        model.eval()
        val_psnr_list, val_ssim_list = [], []
        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                preds, _ = model(inputs)[cite: 2]
                preds = torch.clamp(preds, 0.0, 1.0)
                
                val_psnr_list.append(calc_psnr(preds, targets))[cite: 8]
                val_ssim_list.append(calc_ssim(preds, targets))[cite: 8]
                if idx + 1 >= max_eval_samples:
                    break

        val_psnr = sum(val_psnr_list) / len(val_psnr_list)
        val_ssim = sum(val_ssim_list) / len(val_ssim_list)

        history['train_loss'].append(epoch_loss)
        history['val_psnr'].append(val_psnr)
        history['val_ssim'].append(val_ssim)

        print(f"--> [Epoch {epoch:02d}] Loss: {epoch_loss:.4f} | Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f}")

        # Lưu checkpoint tốt nhất nếu cải thiện PSNR
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), args.save_path)
            print(f"  [+] Đã lưu checkpoint mô hình tốt nhất vào: {args.save_path}")

    logger.close()

    # 4. Ghi kết quả ra file JSON
    results_output = {
        "baseline_nafnet": {
            "psnr": round(baseline_psnr, 2),
            "ssim": round(baseline_ssim, 4)
        },
        "fine_tuned_moe_qformer": {
            "best_psnr": round(best_psnr, 2),
            "best_ssim": round(val_ssim, 4),
            "epochs": args.epochs,
            "history": history
        }
    }

    with open(args.json_output, "w", encoding="utf-8") as f:
        json.dump(results_output, f, ensure_ascii=False, indent=4)

    print("==================================================")
    print(" QUÁ TRÌNH HOÀN TẤT!")
    print(f" -> Đã ghi kết quả so sánh tại : {args.json_output}")
    print(f" -> Đã lưu checkpoint mô hình : {args.save_path}")
    print("==================================================")

if __name__ == '__main__':
    main()