import torch
from net import MyAlexNet
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
# UI依赖
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk

ROOT_TRAIN = r'data/train'
ROOT_TEST = r'data/val'

# 将图像的像素值归一化到【-1， 1】之间
normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    normalize])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize])  # 推理必须和训练归一化保持一致！

train_dataset = ImageFolder(ROOT_TRAIN, transform=train_transform)
val_dataset = ImageFolder(ROOT_TEST, transform=val_transform)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = MyAlexNet().to(device)
# 加载模型
model.load_state_dict(torch.load("save_model/best_model1.pth", weights_only=True))
model.eval()

# 类别
classes = ["cat", "dog"]
show = ToPILImage()

# ===================== UI 界面类 =====================
class CatDogPredictUI:
    def __init__(self, root):
        self.root = root
        self.root.title("猫狗分类预测系统(AlexNet)")
        self.root.geometry("700x600")
        self.root.configure(bg="#f7f9fa")

        self.img_path = None
        self.tk_img = None

        # 标题
        title = ttk.Label(root, text="AlexNet 猫狗图像分类器", font=("微软雅黑",16,"bold"), background="#f7f9fa")
        title.pack(pady=15)

        # 图片显示画布
        self.img_canvas = tk.Canvas(root, width=400, height=380, bg="#e9ecef")
        self.img_canvas.pack()
        self.img_canvas.create_text(200, 190, text="请点击按钮选择图片", font=("微软雅黑",12))

        # 按钮区域
        frame_btn = ttk.Frame(root)
        frame_btn.pack(pady=15)
        ttk.Button(frame_btn, text="选择本地图片", command=self.select_image).grid(row=0, column=0, padx=10)
        ttk.Button(frame_btn, text="开始预测", command=self.predict_image).grid(row=0, column=1, padx=10)
        ttk.Button(frame_btn, text="随机测试验证集10张", command=self.test_val_10).grid(row=0, column=2, padx=10)

        # 结果文本
        self.result_label = ttk.Label(root, text="预测结果：等待输入图片", font=("微软雅黑",14), foreground="#2257a8", background="#f7f9fa")
        self.result_label.pack(pady=10)

    # 选择本地图片
    def select_image(self):
        path = filedialog.askopenfilename(filetypes=[("图片文件", "*.jpg;*.png;*.jpeg;*.bmp")])
        if not path:
            return
        self.img_path = path
        # 显示原图
        img = Image.open(path).convert("RGB")
        img.thumbnail((400, 380))
        self.tk_img = ImageTk.PhotoImage(img)
        self.img_canvas.delete("all")
        self.img_canvas.create_image(200, 190, image=self.tk_img)
        self.result_label.config(text="预测结果：点击「开始预测」")

    # 单张图片推理
    def predict_image(self):
        if self.img_path is None:
            self.result_label.config(text="请先选择一张图片！")
            return
        # 图片预处理（和val_transform完全统一）
        img_origin = Image.open(self.img_path).convert("RGB")
        transform = val_transform
        img_tensor = transform(img_origin)
        img_tensor = torch.unsqueeze(img_tensor, dim=0).to(device)

        with torch.no_grad():
            out = model(img_tensor)
            pred_idx = torch.argmax(out[0])
            pred_class = classes[pred_idx]
        self.result_label.config(text=f"预测结果：{pred_class}")

    # 原有功能：随机测试验证集10张
    def test_val_10(self):
        print("\n===== 验证集10张测试 =====")
        for i in range(10):
            x, y = val_dataset[i][0], val_dataset[i][1]
            show(x).show()
            x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False).to(device)
            with torch.no_grad():
                pred = model(x)
                predicted, actual = classes[torch.argmax(pred[0])], classes[y]
                print(f'predicted:"{predicted}", Actual:"{actual}"')
        self.result_label.config(text="验证集10张图片已打印至控制台")

# ===================== 程序入口 =====================
if __name__ == "__main__":
    window = tk.Tk()
    app = CatDogPredictUI(window)
    window.mainloop()