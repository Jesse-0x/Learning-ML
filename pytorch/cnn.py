import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision

# 定义一个cnn网络
class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 定义卷积层
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 定义卷积层
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 16, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 定义全连接层
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(16 * 5 * 5, 120),
            torch.nn.ReLU()
        )
        # 定义全连接层
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(120, 84),
            torch.nn.ReLU()
        )
        # 定义全连接层
        self.fc3 = torch.nn.Linear(84, 10)

    # 定义前向传播
    def forward(self, x):
        # 卷积层
        x = self.conv1(x)
        # 卷积层
        x = self.conv2(x)
        # 展平
        x = x.view(x.size()[0], -1)
        # 全连接层
        x = self.fc1(x)
        # 全连接层
        x = self.fc2(x)
        # 全连接层
        x = self.fc3(x)
        return x

# 获取MNIST数据集
train_dataset = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(root='./mnist/', train=True, transform=torchvision.transforms.ToTensor(),
                                 download=True), batch_size=64, shuffle=True)
test_dataset = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(root='./mnist/', train=False, transform=torchvision.transforms.ToTensor(),
                                    download=True), batch_size=64, shuffle=False)


# 定义模型
model = LeNet()
# 定义损失函数
loss_func = torch.nn.CrossEntropyLoss()
# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
losses = []
acces = []
eval_losses = []
eval_acces = []
for epoch in range(10):
    train_loss = 0
    train_acc = 0
    model.train()
    for img, label in train_dataset:
        # 前向传播
        output = model(img)
        # 计算损失函数
        loss = loss_func(output, label)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录误差
        train_loss += loss.item()
        # 计算分类的准确率
        _, pred = output.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        train_acc += acc
    losses.append(train_loss / len(train_dataset))
    acces.append(train_acc / len(train_dataset))
    # 在测试集上检验效果
    eval_loss = 0
    eval_acc = 0
    # 将模型改为预测模式
    model.eval()
    for img, label in test_dataset:
        output = model(img)
        loss = loss_func(output, label)
        # 记录误差
        eval_loss += loss.item()
        # 记录准确率
        _, pred = output.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        eval_acc += acc
    eval_losses.append(eval_loss / len(test_dataset))
    eval_acces.append(eval_acc / len(test_dataset))
    print('Epoch {} Train Loss {} Train  Accuracy {} Test Loss {} Test Accuracy {}'.format(epoch + 1,
                                                                                           train_loss / len(
                                                                                               train_dataset),
                                                                                           train_acc / len(
                                                                                               train_dataset),
                                                                                           eval_loss / len(
                                                                                               test_dataset),
                                                                                           eval_acc / len(
                                                                                               test_dataset)))

# 可视化
plt.title('train loss')
plt.plot(np.arange(len(losses)), losses)
plt.legend(['Train Loss'], loc='upper right')
plt.show()

plt.title('train acc')
plt.plot(np.arange(len(acces)), acces)
plt.legend(['Train Accuracy'], loc='upper right')
plt.show()

plt.title('test loss')
plt.plot(np.arange(len(eval_losses)), eval_losses)
plt.legend(['Test Loss'], loc='upper right')
plt.show()

plt.title('test acc')
plt.plot(np.arange(len(eval_acces)), eval_acces)
plt.legend(['Test Accuracy'], loc='upper right')
plt.show()

# 保存模型
torch.save(model.state_dict(), 'model_parameter.pkl')

# 加载模型
model.load_state_dict(torch.load('model_parameter.pkl'))

# 预测
model.eval()
data = iter(test_dataset)
img, label = next(data)
output = model(img)
_, pred = output.max(1)
print('真实标签:', label[:10])
print('预测结果:', pred[:10])

