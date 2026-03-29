import os
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

# ===================== 1. 环境与路径配置 =====================
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取当前程序所在文件夹
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ 修正：确保定义了所有必要的路径
DATA_DIR = os.path.join(BASE_DIR, "datasets")  # 假设你的数据在当前文件夹的 datasets 下
MODELS_DIR = os.path.join(BASE_DIR, "models")
SAVE_DIR = os.path.join(BASE_DIR, "ablation_results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# 实验超参数
NUM_CLASSES = 4
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3


# ===================== 2. 模型核心组件 =====================

class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
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
        return x * self.sa(torch.cat([mx, avg], dim=1))


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip);
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x);
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        return x * self.conv_h(x_h).sigmoid() * self.conv_w(x_w.permute(0, 1, 3, 2)).sigmoid()


# ===================== 3. 五个消融模型定义 =====================

class FullModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.s1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), CBAM(32))
        self.s2 = nn.Sequential(DSConv(32, 64, stride=2), CBAM(64))
        self.s3 = nn.Sequential(DSConv(64, 128, stride=2), CoordAtt(128, 128))
        self.s4 = nn.Sequential(DSConv(128, 256, stride=2), CoordAtt(256, 256))
        self.gap = nn.AdaptiveAvgPool2d(1);
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x): return self.fc(self.gap(self.s4(self.s3(self.s2(self.s1(x))))).view(x.size(0), -1))


class NoS1Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.s2 = nn.Sequential(DSConv(3, 64, stride=2), CBAM(64))
        self.s3 = nn.Sequential(DSConv(64, 128, stride=2), CoordAtt(128, 128))
        self.s4 = nn.Sequential(DSConv(128, 256, stride=2), CoordAtt(256, 256))
        self.gap = nn.AdaptiveAvgPool2d(1);
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x): return self.fc(self.gap(self.s4(self.s3(self.s2(x)))).view(x.size(0), -1))


class NoS2Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.s1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), CBAM(32))
        self.s3 = nn.Sequential(DSConv(32, 128, stride=2), CoordAtt(128, 128))
        self.s4 = nn.Sequential(DSConv(128, 256, stride=2), CoordAtt(256, 256))
        self.gap = nn.AdaptiveAvgPool2d(1);
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x): return self.fc(self.gap(self.s4(self.s3(self.s1(x)))).view(x.size(0), -1))


class NoS3Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.s1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), CBAM(32))
        self.s2 = nn.Sequential(DSConv(32, 64, stride=2), CBAM(64))
        self.s4 = nn.Sequential(DSConv(64, 256, stride=2), CoordAtt(256, 256))
        self.gap = nn.AdaptiveAvgPool2d(1);
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x): return self.fc(self.gap(self.s4(self.s2(self.s1(x)))).view(x.size(0), -1))


class NoS4Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.s1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), CBAM(32))
        self.s2 = nn.Sequential(DSConv(32, 64, stride=2), CBAM(64))
        self.s3 = nn.Sequential(DSConv(64, 128, stride=2), CoordAtt(128, 128))
        self.gap = nn.AdaptiveAvgPool2d(1);
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x): return self.fc(self.gap(self.s3(self.s2(self.s1(x)))).view(x.size(0), -1))


# ===================== 4. 实验引擎 =====================

def run_experiment(name, model, train_loader, test_loader, save_filename):
    print(f"\n🚀 正在运行变体: {name}")
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    best_acc = 0.0
    history = {'epoch': [], 'loss': [], 'acc': []}

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"   Epoch {epoch + 1}", leave=False)
        for imgs, labels in loop:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            if scaler:
                scaler.scale(loss).backward();
                scaler.step(optimizer);
                scaler.update()
            else:
                loss.backward();
                optimizer.step()
            total_loss += loss.item()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            for imgs, labels in test_loader:
                outputs = model(imgs.to(device, non_blocking=True))
                y_pred.extend(torch.max(outputs, 1)[1].cpu().numpy())
                y_true.extend(labels.numpy())

        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, save_filename))
            tqdm.write(f"   🌟 已更新最佳模型: {save_filename} (Acc: {acc * 100:.2f}%)")

        history['epoch'].append(epoch + 1)
        history['loss'].append(total_loss / len(train_loader))
        history['acc'].append(acc)

    return history, best_acc


# ===================== 5. 主程序入口 =====================

if __name__ == "__main__":
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 检查数据集路径是否存在
    train_path = os.path.join(DATA_DIR, "train")
    test_path = os.path.join(DATA_DIR, "test")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"找不到训练集目录: {train_path}，请确认数据集存放在程序同目录下的 datasets 文件夹内。")

    # DataLoader 配置
    ldr_args = {'num_workers': 2, 'pin_memory': True} if device.type == 'cuda' else {}
    train_loader = DataLoader(datasets.ImageFolder(train_path, transform), BATCH_SIZE, True, **ldr_args)
    test_loader = DataLoader(datasets.ImageFolder(test_path, transform), BATCH_SIZE, **ldr_args)

    # 消融实验列表
    variants = [
        ("Full Model", FullModel(NUM_CLASSES), "best_model_full.pth"),
        ("No Stage 1", NoS1Model(NUM_CLASSES), "best_model_no_s1.pth"),
        ("No Stage 2", NoS2Model(NUM_CLASSES), "best_model_no_s2.pth"),
        ("No Stage 3", NoS3Model(NUM_CLASSES), "best_model_no_s3.pth"),
        ("No Stage 4", NoS4Model(NUM_CLASSES), "best_model_no_s4.pth"),
    ]

    final_results = []
    plt.figure(figsize=(12, 7))

    for name, model_instance, fname in variants:
        hist, best_acc = run_experiment(name, model_instance, train_loader, test_loader, fname)
        final_results.append({"实验变体": name, "最佳准确率 (%)": round(best_acc * 100, 2)})

        # 绘图曲线
        plt.plot(hist['epoch'], [a * 100 for a in hist['acc']], label=f"{name} ({best_acc * 100:.2f}%)")

        # 释放资源
        del model_instance
        torch.cuda.empty_cache()

    # 保存 CSV 统计结果
    df = pd.DataFrame(final_results)
    df.to_csv(os.path.join(SAVE_DIR, "ablation_summary.csv"), index=False, encoding='utf_8_sig')

    # 完善图表并保存
    plt.title("Ablation Study: Accuracy Curves Over Stages", fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(SAVE_DIR, "accuracy_curves.png"), dpi=300)

    print(f"\n✅ 全部消融实验已完成！")
    print(f"📁 权重文件存放于: {MODELS_DIR}")
    print(f"📊 统计数据存放于: {SAVE_DIR}")
    print(df.to_string(index=False))