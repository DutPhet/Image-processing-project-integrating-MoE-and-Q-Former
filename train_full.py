import argparse
import os
import json
import urllib.request
import math
import gc
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.forge import FORGEDeblurNet
from utils.dataset import GoProDeblurDataset, get_gopro_sequence_splits
from utils.metrics import DeblurCompositeLoss, calc_psnr, calc_ssim


# ─── Warmup + CosineAnnealing Scheduler ──────────────────────────────────────

class WarmupCosineScheduler:
    """Linear warmup rồi Cosine Annealing."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-7):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.min_lr        = min_lr
        self.base_lrs      = [pg['lr'] for pg in optimizer.param_groups]
        self._epoch        = 0

    def step(self):
        self._epoch += 1
        e = self._epoch
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if e <= self.warmup_epochs:
                lr = base_lr * e / self.warmup_epochs
            else:
                progress = (e - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
            pg['lr'] = lr

    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


# ─── Args ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune FORGE MoE với PEFT & Sequence Split")
    # Dataset
    parser.add_argument('--train_dir',    type=str,   default='gopro-data/train')
    parser.add_argument('--test_dir',     type=str,   default='gopro-data/test')
    parser.add_argument('--val_clips',    type=int,   default=4,
                        help='Số clip dùng cho validation (tách riêng tránh rò rỉ dữ liệu)')
    parser.add_argument('--crop_size',    type=int,   default=256)
    parser.add_argument('--batch_size',   type=int,   default=4)
    parser.add_argument('--num_workers',  type=int,   default=2,
                        help='Số luồng đọc dữ liệu (đặt = 0 nếu chạy trên Windows bị lỗi)')
    parser.add_argument('--epochs',       type=int,   default=20)
    parser.add_argument('--lr',           type=float, default=2e-4)
    parser.add_argument('--warmup_epochs',type=int,   default=2)

    # Loss weights
    parser.add_argument('--fft_weight',   type=float, default=0.05,
                        help='Weight cho Frequency Loss (FFT Loss)')
    parser.add_argument('--vgg_weight',   type=float, default=0.0,
                        help='Weight cho Perceptual Loss (VGG16 Loss, 0 nếu offline)')
    parser.add_argument('--aux_weight',   type=float, default=1e-4,
                        help='Weight cho MoE balance loss')

    # Architecture & PEFT unfreezing
    parser.add_argument('--r',            type=int,   default=16)
    parser.add_argument('--rb',           type=int,   default=32)
    parser.add_argument('--num_experts',  type=int,   default=4)
    parser.add_argument('--moe_in_decoder', action='store_true', default=True,
                        help='Cấy MoE vào khối cuối cùng của Decoder để phục hồi chi tiết cao')
    parser.add_argument('--unfreeze_last_dec', action='store_true', default=True,
                        help='Mở khóa khối decoder cuối cùng và ending conv')
    parser.add_argument('--unfreeze_decoders', action='store_true', default=False,
                        help='Mở khóa toàn bộ Decoder + Upsampling layers (PEFT)')

    # Checkpoint
    parser.add_argument('--save_path',    type=str,   default='weights/forge_moe_best.pth')
    parser.add_argument('--log_path',     type=str,   default='weights/training_log.json')
    return parser.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Thiết bị đang dùng: {device}")
    
    # Tự động tạo thư mục chứa weights và logs
    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.log_path) or '.', exist_ok=True)

    # 1. Tải checkpoint NAFNet gốc nếu chưa có
    nafnet_ckpt_file = "weights/NAFNet-GoPro-width64.pth"
    if not os.path.exists(nafnet_ckpt_file):
        nafnet_ckpt_file = "NAFNet-GoPro-width64.pth"
    if not os.path.exists(nafnet_ckpt_file):
        print("--> Đang tải pretrained NAFNet...")
        urllib.request.urlretrieve(
            "https://huggingface.co/Jacid23/third-eye-models/resolve/"
            "5c2e322323888d8d26852f06715ccc6eb61ea795/NAFNet-GoPro-width64.pth",
            nafnet_ckpt_file
        )
    ckpt = torch.load(nafnet_ckpt_file, map_location=device)
    nafnet_state_dict = ckpt['params'] if 'params' in ckpt else ckpt

    # 2. Phân chia Dataset theo Sequence/Clip (Triệt tiêu 100% Data Leakage)
    train_clips, val_clips = get_gopro_sequence_splits(args.train_dir, val_clips_count=args.val_clips)
    print(f"--> Phân chia Sequence Clip:")
    print(f"    + Train Clips ({len(train_clips)}): {train_clips[:3]} ... {train_clips[-1]}")
    print(f"    + Val Clips   ({len(val_clips)}): {val_clips}")

    train_ds = GoProDeblurDataset(args.train_dir, clip_names=train_clips, crop_size=args.crop_size, is_train=True)
    val_ds   = GoProDeblurDataset(args.train_dir, clip_names=val_clips, crop_size=args.crop_size, is_train=False)
    test_ds  = GoProDeblurDataset(args.test_dir, crop_size=None, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print(f"--> Kích thước dữ liệu: Train = {len(train_ds)} | Val = {len(val_ds)} | Test = {len(test_ds)}")

    # 3. Khởi tạo mô hình FORGEDeblurNet
    model = FORGEDeblurNet(
        num_experts=args.num_experts,
        r=args.r,
        rb=args.rb,
        moe_in_decoder=args.moe_in_decoder
    ).to(device)
    model.load_pretrained(nafnet_state_dict)

    # 4. Áp dụng mở khóa Backbone PEFT
    model.freeze_backbone(
        freeze=True,
        unfreeze_decoders=args.unfreeze_decoders,
        unfreeze_last_dec=args.unfreeze_last_dec
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"--> Trainable params: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.2f}%)")

    # 5. Optimizer, Scheduler & Composite Loss
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        min_lr=1e-7
    )

    criterion = DeblurCompositeLoss(
        eps=1e-3,
        fft_weight=args.fft_weight,
        vgg_weight=args.vgg_weight
    )
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    best_val_psnr = 0.0
    history = []

    # 6. Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch:02d}/{args.epochs:02d}]")

        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    preds, aux_loss = model(inputs)
                    composite_loss = criterion(preds, targets)
                    total_loss = composite_loss + args.aux_weight * aux_loss
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                preds, aux_loss = model(inputs)
                composite_loss = criterion(preds, targets)
                total_loss = composite_loss + args.aux_weight * aux_loss
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

            running_loss += total_loss.item()
            cur_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': f"{total_loss.item():.4f}", 'lr': f"{cur_lr:.2e}"})

        scheduler.step()
        epoch_loss = running_loss / len(train_loader)

        # Deterministic Validation (CenterCrop)
        model.eval()
        psnr_list, ssim_list = [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        preds, _ = model(inputs)
                        preds = torch.clamp(preds, 0.0, 1.0)
                else:
                    preds, _ = model(inputs)
                    preds = torch.clamp(preds, 0.0, 1.0)
                
                psnr_list.append(calc_psnr(preds, targets))
                ssim_list.append(calc_ssim(preds, targets))

        val_psnr = sum(psnr_list) / len(psnr_list)
        val_ssim = sum(ssim_list) / len(ssim_list)
        print(f"--> [Epoch {epoch:02d}] Loss: {epoch_loss:.4f} | Val PSNR (Sequence Split): {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f}")

        history.append({
            'epoch': epoch,
            'train_loss': epoch_loss,
            'val_psnr': val_psnr,
            'val_ssim': val_ssim,
            'lr': cur_lr
        })

        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save(model.state_dict(), args.save_path)
            print(f"    ✓ Đã lưu checkpoint mới tốt nhất tại: {args.save_path}")

        # Dọn dẹp cache GPU
        torch.cuda.empty_cache()

    # 7. Đánh giá kiểm thử sau cùng trên tập Test độc lập
    print("\n===================================================")
    print(" ĐANG ĐÁNH GIÁ TRỌNG SỐ TỐI ƯU TRÊN TẬP TEST GOPRO")
    print("===================================================")
    if os.path.exists(args.save_path):
        model.load_state_dict(torch.load(args.save_path, map_location=device))

    model.eval()
    test_psnr_list, test_ssim_list = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    preds, _ = model(inputs)
                    preds = torch.clamp(preds, 0.0, 1.0)
            else:
                preds, _ = model(inputs)
                preds = torch.clamp(preds, 0.0, 1.0)
                
            test_psnr_list.append(calc_psnr(preds, targets))
            test_ssim_list.append(calc_ssim(preds, targets))

    final_test_psnr = sum(test_psnr_list) / len(test_psnr_list)
    final_test_ssim = sum(test_ssim_list) / len(test_ssim_list)
    print(f"--> KẾT QUẢ TEST CUỐI CÙNG: PSNR = {final_test_psnr:.2f} dB | SSIM = {final_test_ssim:.4f}")

    # Lưu log
    with open(args.log_path, 'w', encoding='utf-8') as f:
        json.dump({
            'history': history,
            'best_val_psnr': best_val_psnr,
            'final_test_psnr': final_test_psnr,
            'final_test_ssim': final_test_ssim
        }, f, indent=4)
    print(f"--> Đã lưu log thành công vào: {args.log_path}")

    # Dọn dẹp RAM/VRAM
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()