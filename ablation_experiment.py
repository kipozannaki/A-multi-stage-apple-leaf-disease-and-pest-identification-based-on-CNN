import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ===================== 1. 环境与参数配置 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "datasets")
SAVE_DIR = os.path.join(BASE_DIR, "ablation_results")
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 20  # 消融实验各组 Epoch 必须保持一致
LEARNING_RATE = 1e-3


# ===================== 2. 模型核心组件 (DSConv, CBAM, CoordAtt) =====================

class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x): return self.conv(x)


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())

    def forward(self, x):
        x = x * self.ca(x)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        avg = torch.mean(x, dim=1, keepdim=True)
        spatial = torch.cat([mx, avg], dim=1)
        return x * self.sa(spatial)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        return x * self.conv_h(x_h).sigmoid() * self.conv_w(x_w).sigmoid()


# ===================== 3. 消融变体架构定义 =====================

# A. 基础版: 仅有深度可分离卷积
class BaseVariant(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.stage1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.stage2 = DSConv(32, 64, stride=2)
        self.stage3 = DSConv(64, 128, stride=2)
        self.stage4 = DSConv(128, 256, stride=2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stage4(self.stage3(self.stage2(self.stage1(x))))
        return self.fc(self.gap(x).view(x.size(0), -1))


# B. CBAM 增强版: 在浅层加入 CBAM
class CBAMVariant(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.stage1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), CBAM(32))
        self.stage2 = nn.Sequential(DSConv(32, 64, stride=2), CBAM(64))
        self.stage3 = DSConv(64, 128, stride=2)
        self.stage4 = DSConv(128, 256, stride=2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stage4(self.stage3(self.stage2(self.stage1(x))))
        return self.fc(self.gap(x).view(x.size(0), -1))


# C. 完整 V4 版: CBAM + CoordAtt
class FullVariant(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.stage1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), CBAM(32))
        self.stage2 = nn.Sequential(DSConv(32, 64, stride=2), CBAM(64))
        self.stage3 = nn.Sequential(DSConv(64, 128, stride=2), CoordAtt(128, 128))
        self.stage4 = nn.Sequential(DSConv(128, 256, stride=2), CoordAtt(256, 256))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.4),
                                nn.Linear(128, num_classes))

    def forward(self, x):
        return self.fc(self.gap(self.stage4(self.stage3(self.stage2(self.stage1(x))))).view(x.size(0), -1))


# ===================== 4. 实验引擎 =====================

def run_experiment(name, model, train_loader, test_loader):
    print(f"\n🚀 启动实验变体: {name}")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {'epoch': [], 'loss': [], 'acc': []}

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f"   Epoch {epoch + 1}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad();
            loss = criterion(model(imgs), labels);
            loss.backward();
            optimizer.step()
            total_loss += loss.item()

        # 评估
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, labels in test_loader:
                outputs = model(imgs.to(device))
                y_pred.extend(torch.max(outputs, 1)[1].cpu().numpy());
                y_true.extend(labels.numpy())

        acc = accuracy_score(y_true, y_pred)
        history['epoch'].append(epoch + 1);
        history['loss'].append(total_loss / len(train_loader));
        history['acc'].append(acc)
        scheduler.step()
        print(f"   📊 结果: Loss {total_loss / len(train_loader):.4f} | Acc {acc * 100:.2f}%")

    return history, acc


# ===================== 5. 执行与可视化 =====================

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_loader = DataLoader(datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform), BATCH_SIZE, True)
    test_loader = DataLoader(datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform), BATCH_SIZE)

    variants = {
        "Base (仅DSConv)": BaseVariant(NUM_CLASSES),
        "Base + CBAM": CBAMVariant(NUM_CLASSES),
        "V4 (Full Model)": FullVariant(NUM_CLASSES)
    }

    final_comparison = []
    plt.figure(figsize=(10, 6))

    for name, model in variants.items():
        hist, final_acc = run_experiment(name, model, train_loader, test_loader)
        final_comparison.append({"变体": name, "最终准确率 (%)": round(final_acc * 100, 2)})

        # 绘制该变体的训练曲线
        plt.plot(hist['epoch'], [a * 100 for a in hist['acc']], label=f'{name} (Acc)')

    # 保存报表
    df = pd.DataFrame(final_comparison)
    df.to_csv(os.path.join(SAVE_DIR, "ablation_report.csv"), index=False)

    # 完善图表并保存
    plt.title("消融实验: 不同组件对模型准确率的影响", fontsize=14)
    plt.xlabel("Epochs");
    plt.ylabel("Accuracy (%)")
    plt.legend();
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, "ablation_learning_curves.png"), dpi=300)

    # 绘制对比柱状图
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="变体", y="最终准确率 (%)", palette="magma")
    plt.title("各组件最终性能贡献对比")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "ablation_bar_chart.png"), dpi=300)

    print(f"\n✅ 实验完成！结果已保存在目录: {SAVE_DIR}")
    print(df.to_string(index=False))