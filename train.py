import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm import tqdm

# ===================== 1. 环境与参数配置 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "datasets")

NUM_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3


# ===================== 2. 注意力机制模块 (Attention) =====================

class CBAM(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()
        # 1. 通道注意力：通过全局平均池化压缩空间，学习各通道间的重要性
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        # 2. 空间注意力：通过提取通道的最大值和平均值，生成空间重要性权重图
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道加权
        x = x * self.ca(x)
        # 空间加权：在通道维度上做 Max 和 Mean，融合后再作用于原图
        mx, _ = torch.max(x, dim=1, keepdim=True)
        avg = torch.mean(x, dim=1, keepdim=True)
        spatial = torch.cat([mx, avg], dim=1)
        return x * self.sa(spatial)


class CoordAtt(nn.Module):


    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 水平池化
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 垂直池化
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        n, c, h, w = x.size()
        # 将空间坐标信息编码进通道维度
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        # 分离出水平和垂直注意力图，乘回到原特征图
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        return x * self.conv_h(x_h).sigmoid() * self.conv_w(x_w).sigmoid()


# ===================== 3. 基础组件 =====================

class DSConv(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            # 第一步：Depthwise 卷积
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            # 第二步：Pointwise 卷积
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# ===================== 4. 模型架构 =====================

class OptimizedMultiStageCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            CBAM(32)  # 在浅层过滤掉无效背景像素，加速训练
        )
        # --- Stage 2: 降采样 (Output: 112x112x64) ---
        self.stage2 = nn.Sequential(
            DSConv(32, 64, stride=2),
            CBAM(64)
        )

        self.stage3 = nn.Sequential(
            DSConv(64, 128, stride=2),
            CoordAtt(128, 128)  # 锁定目标的几何结构位置
        )
        # --- Stage 4: 深层语义提取 (Output: 28x28x256) ---
        self.stage4 = nn.Sequential(
            DSConv(128, 256, stride=2),
            CoordAtt(256, 256)
        )

        # 全局平均池化：将特征图压缩为 (1,1,256)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接分类层
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x).view(x.size(0), -1)
        return self.fc(x)


# ===================== 5. 模型训练与保存 =====================

def train_and_eval(model_name, model):
    # 1. 准备数据加载器 (带增强)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_set = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True)

    test_set = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=train_transform)  # 简化起见测试集同变换
    test_loader = DataLoader(test_set, BATCH_SIZE)

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # 余弦退火：学习率随 Epoch 呈现余弦曲线下降，利于模型跳出局部最优
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    print(f"\n开始训练: {model_name}")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        print(f" ✨ Epoch {epoch + 1} 完成 | 平均 Loss: {total_loss / len(train_loader):.4f}")

    # 2. 最终评估
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="正在最终评估"):
            outputs = model(imgs.to(device))
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    # 创建 models 文件夹并保存模型
    save_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "optimized_multistage_v4.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\n权重已保存至: {save_path}")

    return acc, params

# ===================== 6. 运行程序 =====================
if __name__ == "__main__":
    my_model = OptimizedMultiStageCNN(NUM_CLASSES)
    acc, param_count = train_and_eval("Optimized-V4", my_model)

    print("\n" + "═" * 40)
    print(f"实验总结")
    print(f"最终准确率: {acc * 100:.2f}%")
    print(f"模型参数量: {param_count:.2f} M")
    print("═" * 40)