import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import keras

# a cnn model for mnist, input shape is (28, 28, 1)
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

try:
    model = keras.models.load_model('my_model.h5')
except:
    model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

#save model
model.save('my_model.h5')

# import matplotlib.pyplot as plt
# predictions = model.predict(test_images)
# print(predictions[0])
# print(np.argmax(predictions[0]))
# print(test_labels[0])
# plt.imshow(test_images[0])
# plt.show()


# use tinker to create a gui can let people draw a number and predict it
import tkinter as tk
from PIL import Image, ImageDraw

model = keras.models.load_model('my_model.h5')

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
    # 转为float
    img = img.astype(np.float32)
    # 调整大小
    img = img.reshape((1, 28, 28))
    # 预测
    output = model.predict(img)
    print(output)
    # 显示预测结果
    label.config(text='预测结果为：{}, 确信度为：{:.2f}%'.format(np.argmax(output), np.max(output) * 100))

# 识别按钮
button_recognize = tk.Button(window, text='识别', font=('Arial', 12), width=10, height=1, command=recognize)

button_recognize.pack()

window.mainloop()
