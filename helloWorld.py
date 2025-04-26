# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np

# # 1. Kiểm tra có GPU không
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f'Đang dùng: {device}')

# # Phần này là tạo ma trận của x và y theo hàm số: y= 2*x + 3
# x = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.float32)  # Dữ liệu x
# y = np.array([3, 5, 7, 9, 11, 13, 15], dtype=np.float32)  # Dữ liệu y


# # from_numpy dùng để chuyển từ kiểu dữ liệu vector của numpy sang kiểu dữ liệu Tensor của PyTorch
# # unsqueeze là thêm 1 chiều phụ, vì Linear cần 2 tham số, nên phải tạo ra 1 chiều phụ nữa cho đúng format của Linear
# x = torch.from_numpy(x).unsqueeze(1).to(device)  # (7,1)

# y = torch.from_numpy(y).unsqueeze(1).to(device)  # (7,1)
# # print(y   )

# # Ở hàm số y = 2*x +3 là một đường thẳng, nên dùng Hồi Quy Tuyến Tính (Linear Regression)
# # (1,1) là số input và output: Kiểu như 1 input, và 1 output trong hàm số y=ax+b
# # (2,1) có thể sử dụng trong bài toán phân loại: Ví dụ như chiều cao và cân nặng, thì 2 cái này là input. Và 1 output là số size quần áo thì nó là (2,1)
# model = nn.Linear(1, 1).to(device)  # nhớ đưa model lên cùng device


# # Hàm mất mát MSELoss theo công thức bình phương sai số
# # Nhưng có thể sử dụng hàm L1Loss: Sai số tuyệt đối
# # So sánh MSELoss và L1Loss:
# # MSELoss bình phương lên, rồi tính trung bình cộng, đường mất mát là đường cong
# # L1Loss lấy giá trị tuyệt đối, rồi tính trung bình cộng, đường mất mát là đường thẳng
# criterion = nn.MSELoss()

# # Adam hay SGD là thuật toán tối ưu
# optimizer = optim.Adam(model.parameters(), lr=1e-2)

# # 5. Huấn luyện
# for epoch in range(1000):
#     output = model(x) # Tiến hành dự đoán
#     loss = criterion(output, y) # Tính hàm mất mát

#     optimizer.zero_grad() # Đặt gradient về 0 để tránh bị cộng dồn
#     loss.backward() # Tính gradient của các tham số mô hình
#     optimizer.step() # Tính xong thì cập nhật các tham số mô hình

#     if (epoch+1) % 10 == 0:
#         print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')

# # # 6. Kiểm tra
# # torch,tensor là đưa về kiểu dữ liệu tensor, số truyền vào là 4.0 (float32)
# test_x = torch.tensor([[4.0]], device=device)
# predicted = model(test_x) # kiểm tra thử xem đã đúng chưa
# print(f"Dự đoán khi x=4: {predicted.item():.4f}")


import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments

# Create sample data
x = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.float32).unsqueeze(1)  # (7, 1)
y = torch.tensor([3, 5, 7, 9, 11, 13, 15], dtype=torch.float32).unsqueeze(1)  # (7, 1)

# Define Dataset
class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # Return data as a dictionary with keys expected by the model
        return {'input_ids': self.x[idx], 'labels': self.y[idx]}

# Define the model (linear regression)
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, input_ids, labels=None):
        outputs = self.linear(input_ids)
        loss = None
        if labels is not None:
            loss = nn.MSELoss()(outputs, labels)  # Compute loss if labels are provided
        return {'loss': loss, 'logits': outputs} if loss is not None else outputs

# Initialize model
model = SimpleModel()

# Configure TrainingArguments
training_args = TrainingArguments(
    output_dir='./results',          # Output directory
    num_train_epochs=300,           # Number of training epochs
    per_device_train_batch_size=4,   # Batch size per device
    gradient_accumulation_steps=4,   # Steps for gradient accumulation
    logging_dir='./logs',            # Directory for logs
    logging_steps=10,                # Log every 10 steps
    learning_rate=0.02,
    save_strategy="steps",           # Save the model at regular steps
    save_steps=10,                   # Save every 10 steps
    max_grad_norm=1.0,               # Gradient clipping
    lr_scheduler_type="linear",       # Linear learning rate scheduler
)

# Initialize Dataset
dataset = CustomDataset(x, y)

# Define Trainer
trainer = Trainer(
    model=model,                        # Model
    args=training_args,                 # Training arguments
    train_dataset=dataset,              # Training dataset
)

# Train the model
trainer.train()

