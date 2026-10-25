'''
Chứa mã nguồn dùng để ghi log dữ liệu trong quá trình huấn luyện. 
Dữ liệu thường được tích hợp để hiển thị lên công cụ TensorBoard hoặc Weights & Biases (W&B).
'''
from torch.utils.tensorboard import SummaryWriter
import os

class TensorBoardLogger:
    def __init__(self, log_dir='./runs'):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_images(self, tag, img_tensor, step):
        self.writer.add_images(tag, img_tensor, step)

    def close(self):
        self.writer.close()
