import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("mps")
print(device)

# GAN模型
# GAN是生成对抗网络，是一种无监督学习的方式，它由两个神经网络组成，一个是生成网络，一个是判别网络。
# 生成网络的作用是生成数据，判别网络的作用是判别数据是否是真实的数据。
# 生成网络和判别网络相互对抗，最终生成网络生成的数据越来越逼真，判别网络判别不出来，这样就达到了生成数据的目的。
# GAN的训练过程是这样的，首先生成网络生成一些数据，然后判别网络判别这些数据，判别网络判别出来的数据越真实，生成网络就越优秀。
# Generator一般来说，就是一个普通的神经网络，它的输入是一个随机向量，输出是一个数据。
class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 定义全连接层
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(100, 128),
            torch.nn.Sigmoid(),
            torch.nn.Linear(128, 784),
            torch.nn.Tanh()
        )

    # 定义前向传播
    def forward(self, x):
        x = self.fc(x)
        return x

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 定义全连接层
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(784, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid()
        )

    # 定义前向传播
    def forward(self, x):
        x = self.fc(x)
        return x

# 训练
# 定义模型
generator = Generator()
discriminator = Discriminator()

# 定义损失函数
# 生成网络的损失函数
# BCELoss是二分类交叉熵损失函数，它的作用是计算生成网络生成的数据和真实数据之间的差异。
# 生成网络的目标是生成的数据和真实数据之间的差异越小越好。
loss_fn = torch.nn.BCELoss()
# 判别网络的损失函数
g_loss_fn = torch.nn.BCELoss()
d_loss_fn = torch.nn.BCELoss()

# 定义优化器
# learning rate 这么小是因为生成网络和判别网络的损失函数都是BCELoss，它的值域是[0, 1]，如果learning rate太大，会导致损失函数的值越来越大，最终会溢出。
optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0003)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0003)

# 定义训练数据
train_dataset = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(root='./mnist/', train=True, transform=torchvision.transforms.ToTensor(),
                                 download=False), batch_size=64, shuffle=True)
test_dataset = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(root='./mnist/', train=False, transform=torchvision.transforms.ToTensor(),
                                    download=False), batch_size=64, shuffle=False)
train_loader = train_dataset
test_loader = test_dataset



# 开始训练
# 定义训练次数
epochs = 200
# 定义生成网络和判别网络的损失函数值
g_losses = []
d_losses = []
# 开始训练
for epoch in range(epochs):
    # 定义生成网络和判别网络的损失函数值
    g_loss = 0
    d_loss = 0
    # 开始训练
    for i, (img, label) in enumerate(train_loader):
        # 定义真实数据和生成数据的标签
        real_label = torch.ones(img.size(0))
        fake_label = torch.zeros(img.size(0))
        # 定义真实数据和生成数据
        real_img = img.view(img.size(0), -1)
        fake_img = generator(torch.randn(img.size(0), 100))
        # 训练判别网络
        # 判别网络判别真实数据
        real_out = discriminator(real_img)
        # 判别网络判别生成数据
        fake_out = discriminator(fake_img)
        # 计算判别网络的损失函数
        # 调整维度
        real_out = real_out.squeeze()
        fake_out = fake_out.squeeze()
        d_loss_real = d_loss_fn(real_out, real_label)
        d_loss_fake = d_loss_fn(fake_out, fake_label)
        d_loss = d_loss_real + d_loss_fake
        # 清空判别网络的梯度
        optimizer_d.zero_grad()
        # 反向传播
        d_loss.backward()
        # 更新判别网络参数
        optimizer_d.step()
        # 训练生成网络
        # 生成网络生成数据
        fake_img = generator(torch.randn(img.size(0), 100))
        # 判别网络判别生成数据
        fake_out = discriminator(fake_img)
        # 调整维度
        fake_out = fake_out.squeeze()
        # 计算生成网络的损失函数
        g_loss = g_loss_fn(fake_out, real_label)
        # 清空生成网络的梯度
        optimizer_g.zero_grad()
        # 反向传播
        g_loss.backward()
        # 更新生成网络参数
        optimizer_g.step()
    # 打印损失函数值
    print('epoch %d, g_loss: %f, d_loss: %f' % ((epoch + 1), g_loss, d_loss))
    # 保存损失函数值
    g_losses.append(float(g_loss))
    d_losses.append(float(d_loss))
    # 画出生成网络和判别网络的损失函数值变化图
    plt.clf()
    plt.plot(range(len(g_losses)), g_losses, label='g_loss')
    plt.plot(range(len(d_losses)), d_losses, label='d_loss')
    plt.legend()
    if (epoch + 1) % 10 == 0:
        plt.show()



# 保存模型
torch.save(generator.state_dict(), './generator.pth')
torch.save(discriminator.state_dict(), './discriminator.pth')

# 生成数据
# 加载模型
generator = Generator()
generator.load_state_dict(torch.load('./generator.pth'))
discriminator = Discriminator()
discriminator.load_state_dict(torch.load('./discriminator.pth'))
# 生成数据
fake_img = generator(torch.randn(1, 100))
# 显示数据
fake_img_show = fake_img.view(28, 28).data.numpy()
plt.imshow(fake_img_show, cmap='gray')
plt.show()
fake_out = discriminator(fake_img)
print(fake_out)