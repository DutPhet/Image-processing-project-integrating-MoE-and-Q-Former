import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader  # Thêm thư viện này để tạo các lô (batch) dữ liệu

from models.baseline import BaselineSR
from utils.dataset import RestorationDataset  # Đã mở khóa (uncomment) dòng này
from utils.logger import TensorBoardLogger

def train_baseline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineSR(scale=4).to(device)
    
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    logger = TensorBoardLogger(log_dir="./runs/baseline")
    
    print(f"Bắt đầu huấn luyện Baseline trên {device}...")
    num_epochs = 1
    
    # ================= KHỞI TẠO DATALOADER =================
    # Chỉ định đúng đường dẫn đến thư mục chứa ảnh DIV2K của bạn
    train_dir_lr = "data/train/DIV2k/LR"
    train_dir_hr = "data/train/DIV2k/HR"
    
    # Gọi class RestorationDataset mà bạn vừa viết
    dataset = RestorationDataset(lr_dir=train_dir_lr, hr_dir=train_dir_hr, patch_size=48, scale=4, is_train=True)
    
    # Đóng gói dữ liệu thành các lô (batch), mỗi lô 16 ảnh, trộn ngẫu nhiên (shuffle)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    # =======================================================

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        # Đã mở khóa toàn bộ vòng lặp DataLoader
        for lr, hr in dataloader:
            lr, hr = lr.to(device), hr.to(device)
            optimizer.zero_grad()
            
            sr = model(lr)
            loss = criterion(sr, hr)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        # Đã mở khóa phần ghi log lên TensorBoard
        avg_loss = epoch_loss / len(dataloader)
        logger.log_scalar("Loss/train", avg_loss, epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}] Hoàn tất. Mức độ lỗi (Loss): {avg_loss:.4f}")
    logger.close()

    # THÊM ĐOẠN NÀY ĐỂ LƯU FILE:
    torch.save(model.state_dict(), "baseline_weights.pth")
    print("Đã lưu trọng số mô hình vào file baseline_weights.pth")

if __name__ == '__main__':
    train_baseline()