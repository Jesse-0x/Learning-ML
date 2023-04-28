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

# 定义模型
model = LeNet()
# 预测
model.eval()

# 加载模型
model.load_state_dict(torch.load('model_parameter.pkl'))

# 手写数字图片
import tkinter as tk
from PIL import Image, ImageDraw

# 画布宽度
width = 200
height = 200
bg = 'black'
fill = 'white'
# 画布
window = tk.Tk()
window.geometry('200x200')
window.title('手写数字识别')
# 画布
canvas = tk.Canvas(window, width=width, height=height, bg=bg)
canvas.pack()
# 画笔
image = Image.new('RGB', (width, height), bg)
draw = ImageDraw.Draw(image)

# 鼠标左键按下
def mouse_press(event):
    global last_x, last_y
    last_x, last_y = event.x, event.y

# 鼠标左键移动
def mouse_move(event):
    global last_x, last_y
    canvas.create_line((last_x, last_y, event.x, event.y), fill=fill, width=10)
    draw.line((last_x, last_y, event.x, event.y), fill=fill, width=10)
    last_x, last_y = event.x, event.y

# 鼠标左键释放
def mouse_release(event):
    pass

# 鼠标左键按下
canvas.bind('<Button-1>', mouse_press)
# 鼠标左键移动
canvas.bind('<B1-Motion>', mouse_move)
# 鼠标左键释放
canvas.bind('<ButtonRelease-1>', mouse_release)

def clear():
    canvas.delete('all')
    draw.rectangle((0, 0, width, height), fill=bg)

label = tk.Label(window, text='手写数字识别', font=('Arial', 12))
label.pack()

# 清除按钮
button_clear = tk.Button(window, text='清除', font=('Arial', 12), width=10, height=1, command=clear)
button_clear.pack()


def recognize():
    global image
    # 保存图片
    image.save('temp.png')
    # 打开图片
    img = Image.open('temp.png')
    # 转为灰度图
    img = img.convert('L')
    # 转为28*28
    img = img.resize((28, 28))
    # 转为numpy
    img = np.array(img)
    # 转为tensor
    img = torch.from_numpy(img)
    # 扩展维度
    img = img.unsqueeze(0)
    # 转为float
    img = img.float()
    # 调整大小
    img = img.view(-1, 1, 28, 28)
    # 预测
    output = model(img)
    # 获取预测结果
    pred = torch.argmax(output, 1)
    # 显示预测结果
    label.config(text='预测结果为：{}'.format(pred.item()))

# 识别按钮
button_recognize = tk.Button(window, text='识别', font=('Arial', 12), width=10, height=1, command=recognize)

button_recognize.pack()

window.mainloop()
