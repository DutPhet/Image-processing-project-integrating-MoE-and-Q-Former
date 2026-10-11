import torch
from torch.utils.data import DataLoader
from models.baseline import BaselineSR
from utils.dataset import RestorationDataset
from utils.metrics import calc_psnr
def evaluate(model_path, test_lr_dir, test_hr_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. KHỞI TẠO MÔ HÌNH VÀ TẢI TRỌNG SỐ (WEIGHTS)
    model = BaselineSR(scale=4).to(device)
    try:
        # Load trọng số đã học được từ file .pth
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Đã tải thành công trọng số từ: {model_path}")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {model_path}. Hãy chắc chắn bạn đã train xong và lưu file.")
        return

    # CHÚ Ý: Phải chuyển mô hình sang chế độ đánh giá để tắt Dropout, BatchNorm...
    model.eval()
    
    # 2. KHỞI TẠO DATALOADER CHO TẬP TEST (SET5)
    # is_train=False để giữ nguyên toàn bộ bức ảnh (không cắt patch, không lật)
    test_dataset = RestorationDataset(lr_dir=test_lr_dir, hr_dir=test_hr_dir, scale=4, is_train=False)
    
    # Lúc test, ta thường truyền từng ảnh một (batch_size=1) và không cần trộn (shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    total_psnr = 0.0
    print(f"Đang đánh giá trên tập dữ liệu gồm {len(test_dataset)} ảnh...")
    
    # 3. VÒNG LẶP ĐÁNH GIÁ (TEST LOOP)
    # torch.no_grad() giúp tắt tính toán đạo hàm, giúp chạy nhanh hơn và không tốn RAM
    with torch.no_grad():
        for lr, hr in test_loader:
            lr, hr = lr.to(device), hr.to(device)
            
            # Đưa ảnh mờ qua mô hình để sinh ra ảnh nét (sr)
            sr = model(lr)
            
            # Tính điểm PSNR giữa ảnh do AI sinh ra và ảnh gốc thực tế
            psnr = calc_psnr(sr, hr)
            total_psnr += psnr
    avg_psnr = total_psnr / len(test_loader)
    print(f"=====================================")
    print(f"Chỉ số PSNR trung bình: {avg_psnr:.2f} dB")
    print(f"=====================================")

if __name__ == '__main__':
    # Chạy hàm đánh giá, trỏ đường dẫn tới file trọng số và thư mục Set5
    evaluate(
        model_path="baseline_weights.pth", 
        test_lr_dir="data/test/Set5/LR",
        test_hr_dir="data/test/Set5/HR"
    )