'''
Tự động chạy vào thư mục HR, bóp nhỏ 5 bức ảnh đi 4 lần và ném sang thư mục LR với tên gọi y hệt.
'''

import os
from PIL import Image

hr_dir = "data/test/Set5/HR"
lr_dir = "data/test/Set5/LR"

for filename in os.listdir(hr_dir):
    if filename.endswith(('.png', '.jpg')):
        # Mở ảnh nét to
        hr_img = Image.open(os.path.join(hr_dir, filename))
        w, h = hr_img.size
        
        # Thu nhỏ ảnh đi 4 lần bằng thuật toán Bicubic
        lr_img = hr_img.resize((w // 4, h // 4), Image.BICUBIC)
        
        # Lưu vào thư mục LR
        lr_img.save(os.path.join(lr_dir, filename))
        
print("Đã tự động tạo xong 5 ảnh mờ trong thư mục LR!")