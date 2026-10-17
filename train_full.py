'''
Kịch bản huấn luyện (Training & Fine-tuning) chuyên biệt cho bài toán Khử mờ ảnh GoPro (GoPro Deblurring)
Mô hình: MoEQFormerDeblurNet (NAFBlock Backbone + Mini Q-Former + MoE Block)
Hỗ trợ:
- Tùy chọn --freeze_backbone để chỉ fine-tune Q-Former + MoE (cực nhanh trên Kaggle/Colab GPU)
- Ghi log TensorBoard, tính Charbonnier Loss + MoE Load Balancing Loss
- Tự động lưu checkpoint tốt nhất vào file .pth
'''

import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.moe_qformer_net import MoEQFormerDeblurNet
from utils.dataset import GoProDeblurDataset
from utils.metrics import CharbonnierLoss, count_parameters
from utils.logger import TensorBoardLogger

def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện MoE + Q-Former GoPro Deblurring")
    parser.add_argument('--train_dir', type=str, default='gopro-data/train', help="Đường dẫn thư mục train GoPro")
    parser.add_argument('--batch_size', type=int, default=8, help="Kích thước batch (khuyên dùng 8 trên T4 GPU)")
    parser.add_argument('--crop_size', type=int, default=256, help="Kích thước patch ảnh (256x256)")
    parser.add_argument('--epochs', type=int, default=10, help="Số lượng epoch huấn luyện")
    parser.add_argument('--lr', type=float, default=2e-4, help="Tốc độ học (Learning rate)")
    parser.add_argument('--freeze_backbone', action='store_true',
                        help="Đóng băng backbone để fine-tune nhanh Q-Former + MoE")
    parser.add_argument('--aux_weight', type=float, default=0.01, help="Trọng số MoE load balancing loss")
    parser.add_argument('--save_path', type=str, default='moe_qformer_gopro_best.pth', help="File lưu checkpoint")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==================================================")
    print(f"Bắt đầu huấn luyện GoPro Deblurring trên: {device}")
    print(f"Đường dẫn dữ liệu: {args.train_dir}")
    print(f"Đóng băng Backbone: {args.freeze_backbone}")
    print("==================================================")

    # 1. KHỞI TẠO DATALOADER
    if not os.path.exists(args.train_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục {args.train_dir}! Hãy kiểm tra lại dữ liệu GoPro.")

    train_dataset = GoProDeblurDataset(root_dir=args.train_dir, crop_size=args.crop_size, is_train=True)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True if torch.cuda.is_available() else False
    )
    print(f"--> Đã tải {len(train_dataset):,} ảnh huấn luyện.")

    # 2. KHỞI TẠO MÔ HÌNH
    model = MoEQFormerDeblurNet(
        in_channels=3,
        out_channels=3,
        num_feats=64,
        num_blocks=8,
        num_queries=16,
        num_experts=3
    ).to(device)

    if args.freeze_backbone:
        model.freeze_backbone(True)

    print(f"Tổng số tham số có thể huấn luyện: {count_parameters(model):,}")

    # 3. LOSS VÀ OPTIMIZER
    criterion = CharbonnierLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    logger = TensorBoardLogger(log_dir="./runs/gopro_deblur")

    # 4. VÒNG LẶP HUẤN LUYỆN
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_aux = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch:02d}/{args.epochs:02d}]")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            preds, aux_loss = model(inputs)
            recon_loss = criterion(preds, targets)
            loss = recon_loss + args.aux_weight * aux_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_aux += aux_loss.item()

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recon': f"{recon_loss.item():.4f}",
                'aux': f"{aux_loss.item():.4f}"
            })

        avg_loss = total_loss / len(train_loader)
        avg_recon = total_recon / len(train_loader)
        scheduler.step()

        logger.log_scalar("Loss/Total", avg_loss, epoch)
        logger.log_scalar("Loss/Recon", avg_recon, epoch)
        logger.log_scalar("LR", scheduler.get_last_lr()[0], epoch)

        print(f"--> [Epoch {epoch:02d}] Kết thúc | Avg Loss: {avg_loss:.4f} | Recon Loss: {avg_recon:.4f}")

        # Lưu checkpoint tốt nhất
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss
            }, args.save_path)
            print(f"  [+] Đã lưu checkpoint mới tốt nhất vào: {args.save_path}")

    logger.close()
    print("==================================================")
    print(f"Huấn luyện hoàn tất! File checkpoint: {args.save_path}")

if __name__ == '__main__':
    main()
