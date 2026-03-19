import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# ===================== 1. 环境与配置 =====================
# 核心加速开关
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "datasets")

NUM_CLASSES = 4
BATCH_SIZE = 64  # AMP模式下显存占用减小，可以适当调大 Batch
EPOCHS = 15
IMG_SIZE = 224
LEARNING_RATE = 1e-3


# ===================== 2. 模型结构定义 =====================
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
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

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

    def forward(self, x):
        return self.conv(x)


class OptimizedMultiStageCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            CBAM(32)
        )
        self.stage2 = nn.Sequential(DSConv(32, 64, stride=2), CBAM(64))
        self.stage3 = nn.Sequential(DSConv(64, 128, stride=2), CoordAtt(128, 128))
        self.stage4 = nn.Sequential(DSConv(128, 256, stride=2), CoordAtt(256, 256))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
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


# ===================== 3. 性能测试工具函数 =====================
def measure_speed(model, input_shape=(1, 3, 224, 224), num_iterations=100, warmup=20):
    model.eval()
    dummy_input = torch.randn(*input_shape).to(device)
    with torch.no_grad(), autocast():  # 推理也使用混合精度
        for _ in range(warmup):
            _ = model(dummy_input)
        if device.type == 'cuda': torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_iterations):
            _ = model(dummy_input)
        if device.type == 'cuda': torch.cuda.synchronize()
        end = time.perf_counter()
    return ((end - start) / num_iterations) * 1000


def run_experiment(name, model, train_loader, test_loader):
    print(f"\n🧪 正在评估: {name}")
    model = model.to(device)

    # 获取参数量
    params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    # 训练配置
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # AMP 缩放器
    scaler = GradScaler()

    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"   {name} Epoch {epoch + 1}/{EPOCHS}", leave=False)
        for imgs, labels in loop:
            # 异步数据传输
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # 内存优化

            with autocast():  # 混合精度前向
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loop.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()

    # 测试准确率
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad(), autocast():
        for imgs, labels in test_loader:
            outputs = model(imgs.to(device, non_blocking=True))
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    speed = measure_speed(model)
    return acc, params, speed


# ===================== 4. 主流程 =====================
if __name__ == "__main__":
    # ================= 修改开始：解决中文显示方块问题 =================
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'STHeiti', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rc('font', family='sans-serif')
    # ================= 修改结束 =====================================

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 开启 pin_memory 加速数据拷贝
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, prefetch_factor=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True)

    # 模型列表
    model_factory = {
        "V4-Attention (Ours)": lambda: OptimizedMultiStageCNN(NUM_CLASSES),
        "ResNet-18": lambda: models.resnet18(num_classes=NUM_CLASSES),
        "MobileNet-V2": lambda: models.mobilenet_v2(num_classes=NUM_CLASSES),
        "ShuffleNet-V2": lambda: models.shufflenet_v2_x1_0(num_classes=NUM_CLASSES)
    }

    results = []
    for name, create_fn in model_factory.items():
        acc, params, speed = run_experiment(name, create_fn(), train_loader, test_loader)
        results.append({
            "Model": name, "Accuracy (%)": acc * 100,
            "Params (M)": params, "Latency (ms)": speed
        })
        print(f"✅ {name} | 准确率: {acc * 100:.2f}% | 参数量: {params:.2f}M | 延迟: {speed:.2f}ms")

    # 结果可视化与保存
    df = pd.DataFrame(results)
    df.to_csv("comparison_results.csv", index=False)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="Params (M)", y="Accuracy (%)", hue="Model", size="Latency (ms)", sizes=(100, 500))
    plt.title("性能对比图")
    plt.savefig("benchmark_comparison.png")
    plt.show()