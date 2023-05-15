import torch


# 定义模型，继承自torch.nn.Module，然后在__init__函数中定义网络的结构，在forward函数中定义前向传播的过程。
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        # 继承父类的__init__功能， 即调用nn.Module的__init__函数，Module自带的属性有training和parameter
        super(Net, self).__init__()
        # 定义网络结构， nn.Linear是一个全连接层，参数为输入和输出的维度
        # 定义隐藏层，输入维度为n_feature，输出维度为n_hidden
        # Tanh是一个激活函数，激活函数的作用是将输入的信号转换成输出信号，常用的激活函数有Sigmod、Tanh、ReLU等
        self.hidden = torch.nn.Sequential(
            torch.nn.Linear(n_feature, n_hidden),
            torch.nn.Tanh(),
        )
        # 定义输出层，输入维度为n_hidden，输出维度为n_output
        self.predict = torch.nn.Sequential(
            torch.nn.Linear(n_hidden, n_output),
            torch.nn.Tanh(),
        )

    # 定义前向传播过程，输入为x，输出为x。
    # 向前传播的过程就是将输入信号x通过网络结构转换成输出信号x的过程。
    def forward(self, x):
        # x为输入信号，通过隐藏层得到输出信号
        x = self.hidden(x)
        # 然后将输出信号x作为下一层的输入信号，通过输出层得到最终的输出信号
        x = self.predict(x)
        return x


# 定义一个网络对象，输入维度为1，隐藏层维度为10，输出维度为1
net = Net(1, 10, 1)

# 训练网络
# 训练数据集是一个普通的quadric函数，即y=x^2+0.2*rand，其中rand是一个服从正态分布的随机数。
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) #+ 0.2 * torch.rand(x.size())

print(x.shape)
print(y.shape)

# 定义优化器，使用随机梯度下降算法，学习率为0.09
# 优化器的作用是通过梯度下降算法来更新网络中的参数，从而使得网络的输出结果与真实值更加接近。
optimizer = torch.optim.SGD(net.parameters(), lr=0.09)
# 定义损失函数，这里使用均方误差作为损失函数
# 损失函数的作用是计算网络的输出结果与真实值之间的误差，常用的损失函数有均方误差、交叉熵等。
loss_func = torch.nn.MSELoss()

losses = []
for t in range(100000):
    # 输入训练数据x，得到网络的输出信号prediction
    prediction = net(x)
    # 将prediction与真实值y进行比较，计算误差
    loss = loss_func(prediction, y)
    losses.append(loss.data.numpy())
    # 将网络中的所有参数的梯度清零，因为默认情况下，pytorch会将每一次计算得到的梯度值累加到参数的grad属性中，所以在每一次更新参数之前都需要将梯度清零。
    optimizer.zero_grad()
    # 反向传播，计算参数更新的梯度
    # 反向传播的过程就是通过计算网络中的每一层的梯度，从最后一层开始，将梯度值累加到参数的grad属性中。
    loss.backward()
    # 更新参数，即将参数值加上梯度值乘以学习率的值
    optimizer.step()
    if t % 1000 == 0:
        print("Epoch: {}, Loss: {:.4f}".format(t, loss.data.numpy()))

import matplotlib.pyplot as plt
plt.scatter(x.data.numpy(), y.data.numpy())
plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
plt.show()
