import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision

# 定义一个cnn网络
class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 定义卷积层
        # 在这里， Sequential() 函数的功能是将网络的层组合到一起
        # Conv2d() 函数的功能是对输入的数据进行卷积操作
        # Conv2d就是卷积层，是由多个卷积核组成的，每个卷积核都是一个二维的矩阵，卷积核的大小就是这个二维矩阵的大小
        # 其中，第一个参数是输入的通道数，第二个参数是输出的通道数，第三个参数是卷积核的大小，第四个参数是步长，第五个参数是填充
        # 输入通道就是输入的数据的通道数，比如说，如果输入的是一张彩色图片，那么输入通道就是3，如果输入的是一张灰度图片，那么输入通道就是1
        # 输出通道就是卷积层的个数，比如说，如果输出通道是6，那么就是有6个卷积层，每个卷积层都是一个二维的矩阵
        # 卷积核的大小就是卷积核的大小，比如说，如果卷积核的大小是5，那么就是一个5*5的二维矩阵
        # 步长就是卷积核在输入数据上移动的步长，比如说，如果步长是1，那么卷积核在输入数据上移动的步长就是1
        # 填充就是在输入数据的周围填充0，比如说，如果填充是2，那么就是在输入数据的周围填充2圈0
        # 卷积核就是一个二维的矩阵，去对图片进行卷积操作，卷积核的大小就是这个二维矩阵的大小
        # 卷积层的处理过程就是有一个输入数据，然后有一个卷积核，然后卷积核在输入数据上移动
        # 卷积核是一个二维的矩阵，把这个矩阵放在输入数据上，然后卷积核在输入数据上移动，卷积核的移动步长就是步长，
        # ReLU() 函数的功能是对输入的数据进行激活操作
        # 为什么卷积层也需要激活呢？因为卷积层也是一种神经网络，神经网络的每一层都需要激活函数，只不过卷积层会先进行卷积操作，然后再进行激活操作
        # 卷积层的卷积核的值是怎么训练得到的呢？就是通过反向传播算法来训练的，反向传播算法就是通过梯度下降算法来训练的
        # 卷基层的 1 6 5 1 2 代表的含义是什么呢？1 代表的是输入的通道数，6 代表的是输出的通道数，5 代表的是卷积核的大小，1 代表的是步长，2 代表的是填充
        # 也就是说输入是一张图片，输出是6张图片，卷积核的大小是5*5，卷积核在输入数据上移动的步长是1，输入数据的周围填充2圈0
        # 为什么是6张图片呢？因为这样可以提取出6种特征，比如说，第一张图片是提取出了边缘特征，第二张图片是提取出了角点特征，第三张图片是提取出了直线特征，第四张图片是提取出了曲线特征，第五张图片是提取出了圆形特征，第六张图片是提取出了椭圆特征
        # MaxPool2d() 函数的功能是对输入的数据进行池化操作
        # 池化操作就是对输入的数据进行降维操作，比如说，如果池化操作的大小是2，那么就是将输入的数据的大小降低到原来的一半
        # 比如说，如果输入的数据的大小是28*28，那么经过池化操作之后，输入的数据的大小就是14*14
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 定义卷积层
        # 第二个卷积层的能力要比第一个卷积层的能力强，因为第二个卷积层的输入是第一个卷积层的输出
        # 第一个卷积层的输出是6张特征图，所以第二个卷积层的输入就是6张特征图，第二个卷积层的输出就是16张特征图
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 16, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 定义全连接层
        # 全连接层的功能就是将输入的数据进行降维操作，比如说，如果输入的数据的大小是16*5*5，那么经过全连接层之后，输入的数据的大小就是120
        # Linear() 函数的功能就是对输入的数据进行线性变换
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(16 * 5 * 5, 120),
            torch.nn.Sigmoid()
        )
        # 定义全连接层
        # 第二个全连接层的输入就是第一个全连接层的输出，第一个全连接层的输出是120，第二个全连接层的输出是84
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(120, 84),
            torch.nn.Sigmoid()
        )
        # 定义全连接层
        # 第三个全连接层的输入就是第二个全连接层的输出，第二个全连接层的输出是84，第三个全连接层的输出是10
        self.fc3 = torch.nn.Linear(84, 10)

    # 定义前向传播
    # 什么是向前传播？向前传播是一种训练神经网络的方法，它的原理是将输入的数据输入到神经网络中，然后将神经网络的输出与真实的标签进行比较，然后根据比较的结果来调整神经网络中的参数，最后得到一个训练好的神经网络
    # 是从训练集中取出一个批次的数据，然后将这个批次的数据输入到神经网络中，然后将神经网络的输出与真实的标签进行比较，然后根据比较的结果来调整神经网络中的参数，最后得到一个训练好的神经网络
    # 在pytorch里，我们可以通过定义一个类来定义一个神经网络，然后通过定义这个类的一个对象来定义一个神经网络
    # 然后我们可以通过调用这个对象的forward()函数来实现向前传播
    # 其中输入的数据是x
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
                                 download=False), batch_size=64, shuffle=True)
test_dataset = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(root='./mnist/', train=False, transform=torchvision.transforms.ToTensor(),
                                    download=False), batch_size=64, shuffle=False)


# 定义模型
model = LeNet()
# 定义损失函数
# 交叉熵损失函数
# 交叉熵损失函数的功能是计算神经网络的输出与真实的标签之间的差异
# pytorch还有很多其他的损失函数，比如说均方差损失函数，KL散度损失函数等等
loss_func = torch.nn.CrossEntropyLoss()
# 定义优化器
# 优化器的功能是根据神经网络的输出与真实的标签之间的差异来调整神经网络中的参数
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
        # output是神经网络的输出, output的大小是64*10, 是一个64行10列的矩阵
        output = model(img)
        # 计算损失函数
        # output在这里是一个矩阵，其中的每一行代表一个样本，其中的每一列代表一个类别
        # 将output转换为数字的话，就是将output中的每一行中的最大的那个元素的下标作为这一行所代表的样本的预测的类别
        loss = loss_func(output, label)
        # 反向传播
        # 在调用optimizer.step()函数之前，一定要先调用optimizer.zero_grad()函数，否则程序会报错
        # optimizer.zero_grad()函数的功能是将神经网络中的参数的梯度设置为0，否则神经网络中的参数的梯度会在上一次循环中累加到这一次循环中
        # 梯度是什么？梯度是损失函数对神经网络中的参数的偏导数
        optimizer.zero_grad()
        # backward()函数的功能是根据计算图计算神经网络中的参数的梯度
        # 然后将神经网络中的参数的梯度保存在神经网络中的每个参数的grad属性中
        # 也就是我们常说的反向传播
        # 反响传播的原理是什么？反向传播的原理是通过计算图计算损失函数对神经网络中的参数的偏导数，然后根据损失函数对神经网络中的参数的偏导数来调整神经网络中的参数
        # 在神经网络中，是从后向前计算梯度的，也就是说，是先计算损失函数对神经网络中的最后一层的参数的偏导数，然后再计算损失函数对神经网络中的倒数第二层的参数的偏导数，然后再计算损失函数对神经网络中的倒数第三层的参数的偏导数，以此类推
        loss.backward()
        # 根据神经网络中的参数的梯度来调整神经网络中的参数
        # 也就是我们常说的梯度下降
        # 梯度下降的原理是什么？梯度下降的原理是根据损失函数对神经网络中的参数的偏导数来调整神经网络中的参数
        optimizer.step()
        # 记录误差
        train_loss += loss.item()
        # 计算分类的准确率
        # pred是神经网络的预测结果, pred的大小是64*10, 是一个64行10列的矩阵
        # pred.data.max(1)的结果是pred矩阵中每一行的最大值的下标
        # 一般来收，pred结果为矩阵的原因是因为神经网络的最后一层是全连接层，全连接层的输出结果是一个向量，向量的大小是类别的个数
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

#  torchviz
from torchviz import make_dot

# 生成计算图
x = torch.randn(1, 1, 28, 28)
y = model(x)
make_dot(y, params=dict(list(model.named_parameters()))).render('cnn_torchviz', format='png')


# 保存模型
torch.save(model.state_dict(), 'model_parameter.pkl')
# 保存为pth文件
torch.save(model, 'model_all.pth')

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

