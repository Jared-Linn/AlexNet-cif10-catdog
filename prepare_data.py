"""从 CIFAR-10 下载猫狗图片"""
from torchvision.datasets import CIFAR10
from PIL import Image
import os

# 下载 CIFAR-10（训练集 50000 张）
print("正在下载 CIFAR-10...")
ds = CIFAR10(root='./cifar10', train=True, download=True)
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# 创建目录
for name in ['cat','dog']:
    os.makedirs(f'data_name/{name}', exist_ok=True)

# 提取猫(3)和狗(5)的图片，各 500 张
counts = {'cat': 0, 'dog': 0}
for img, label in ds:
    name = classes[label]
    if name in ('cat', 'dog') and counts[name] < 500:
        img.save(f'data_name/{name}/{counts[name]}.png')
        counts[name] += 1

print(f'猫图: {counts["cat"]} 张, 狗图: {counts["dog"]} 张')
print('数据准备完成！')
