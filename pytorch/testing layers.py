import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision


class Addition(torch.nn.Module):
    def __init__(self):
        super(Addition, self).__init__()
        # 定义全连接层
        # 实现加法计算
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2, 1),
            torch.nn.ReLU(),
            # torch.nn.Sigmoid(),
            # torch.nn.Linear(10, 1),
            # torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        return x


# 定义模型
model = Addition()
# 定义损失函数
loss_fn = torch.nn.BCELoss()
# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 数据
import random
x1 = [i for i in range(1000)]
x2 = [i for i in range(1000)]
random.shuffle(x1)
random.shuffle(x2)
x = torch.tensor([[x1[i], x2[i]] for i in range(1000)], dtype=torch.float32) / 1000000
y = torch.tensor([[x1[i] + x2[i]] for i in range(1000)], dtype=torch.float32) / 1000000

# 训练
for epoch in range(10000):
    # 前向传播
    y_pred = model(x)
    # 计算损失函数
    loss = loss_fn(y_pred, y)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # 打印损失函数
    print(loss.item())

# 测试
x1 = [i for i in range(1000)]
x2 = [i for i in range(1000)]
random.shuffle(x1)
random.shuffle(x2)
x = torch.tensor([[x1[i], x2[i]] for i in range(1000)], dtype=torch.float32) / 1000000
y = torch.tensor([[x1[i] + x2[i]] for i in range(1000)], dtype=torch.float32) / 1000000
y_pred = model(x)
print(y_pred)
print(y)
print(torch.mean(torch.abs(y_pred - y)))

# plot y and y_pred
plt.plot(y.detach().numpy(), label='y')
plt.plot(y_pred.detach().numpy(), label='y_pred')
plt.legend()
plt.show()
