"""从 CIFAR-10 提取猫狗图片（使用本地 pickle 文件）"""
import pickle
import numpy as np
from PIL import Image
import os
import warnings
warnings.filterwarnings('ignore')

print("正在读取本地 CIFAR-10 数据...")
# 直接从 pickle 文件加载（避免联网下载）
def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo)

batches = []
for i in range(1, 6):
    batch = unpickle(f'cifar10/cifar-10-batches-py/data_batch_{i}')
    batches.append(batch)

data = np.concatenate([b['data'] for b in batches])
labels = np.concatenate([b['labels'] for b in batches])
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# 创建目录
for name in ['cat','dog']:
    os.makedirs(f'data_name/{name}', exist_ok=True)

# 提取猫(3)和狗(5)的图片，各 500 张
counts = {'cat': 0, 'dog': 0}
for i in range(len(data)):
    label = int(labels[i])
    name = classes[label]
    if name in ('cat', 'dog') and counts[name] < 500:
        # CIFAR-10 数据格式: (3072,) = 32x32x3 (R,G,B 通道各 1024)
        img_data = data[i].reshape(3, 32, 32).transpose(1, 2, 0)
        img = Image.fromarray(img_data)
        img.save(f'data_name/{name}/{counts[name]}.png')
        counts[name] += 1

print(f'猫图: {counts["cat"]} 张, 狗图: {counts["dog"]} 张')
print('数据准备完成！')
