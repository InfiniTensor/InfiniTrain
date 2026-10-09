import torch
import torch.nn as nn

# 固定输入和权重
x = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5)
w = torch.ones(1, 1, 3, 3)
y = torch.nn.functional.conv2d(x, w)
print("PyTorch Forward:", y)