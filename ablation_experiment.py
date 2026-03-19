import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm import tqdm

# ===================== 1. 路径与文件夹配置 =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 按照你的要求，创建名为 models 的子文件夹
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

DATA_DIR = os.path.join(BASE_DIR, "datasets")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置
NUM_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-3


# ===================== 2. 模型组件 (DSConv, CBAM, CoordAtt) =====================
# (此处组件代码与之前一致，保持结构完整性)
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
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels // reduction, 1, bias=False),
                                nn.ReLU(inplace=True), nn.Conv2d(channels // reduction, channels, 1, bias=False),
                                nn.Sigmoid())
        self.sa = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())

    def forward(self, x):
        x = x * self.ca(x)
        spatial = torch.cat([torch.max(x, 1, keepdim=True)[0], torch.mean(x, 1, keepdim=True)], dim=1)
        return x * self.sa(spatial)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1));
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip);
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0);
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x);
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        return x * self.conv_h(x_h).sigmoid() * self.conv_w(x_w.permute(0, 1, 3, 2)).sigmoid()


# ===================== 3. 动态消融模型结构 =====================
class StageAblationModel(nn.Module):
    def __init__(self, num_classes, skip_stage=None):
        super().__init__()
        chs = {0: 3, 1: 32, 2: 64, 3: 128, 4: 256}
        curr_in = chs[0]

        # 定义阶段 (1:CBAM, 2:CBAM, 3:CoordAtt, 4:CoordAtt)
        self.stage1 = nn.Sequential(nn.Conv2d(curr_in, chs[1], 3, padding=1), nn.BatchNorm2d(chs[1]),
                                    nn.ReLU(inplace=True), CBAM(chs[1])) if skip_stage != 1 else nn.Identity()
        curr_in = chs[1] if skip_stage != 1 else curr_in

        self.stage2 = nn.Sequential(DSConv(curr_in, chs[2], stride=2),
                                    CBAM(chs[2])) if skip_stage != 2 else nn.Identity()
        curr_in = chs[2] if skip_stage != 2 else curr_in

        self.stage3 = nn.Sequential(DSConv(curr_in, chs[3], stride=2),
                                    CoordAtt(chs[3], chs[3])) if skip_stage != 3 else nn.Identity()
        curr_in = chs[3] if skip_stage != 3 else curr_in

        self.stage4 = nn.Sequential(DSConv(curr_in, chs[4], stride=2),
                                    CoordAtt(chs[4], chs[4])) if skip_stage != 4 else nn.Identity()
        curr_in = chs[4] if skip_stage != 4 else curr_in

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(curr_in, 128), nn.ReLU(inplace=True), nn.Dropout(0.4),
                                nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.stage4(self.stage3(self.stage2(self.stage1(x))))
        return self.fc(self.gap(x).view(x.size(0), -1))


# ===================== 4. 核心训练与保存函数 =====================

def run_experiment(name, model, train_loader, test_loader, filename):
    print(f"\n🚀 正在运行: {name}")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0
    # 模型完整路径
    save_path = os.path.join(MODELS_DIR, filename)

    for epoch in range(EPOCHS):
        model.train()
        for imgs, labels in tqdm(train_loader, desc=f"   Epoch {epoch + 1}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            criterion(model(imgs), labels).backward()
            optimizer.step()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for imgs, labels in test_loader:
                outputs = model(imgs.to(device))
                y_pred.extend(torch.max(outputs, 1)[1].cpu().numpy())
                y_true.extend(labels.numpy())

        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            # 仅保存模型权重，结构简洁
            torch.save(model.state_dict(), save_path)
            tqdm.write(f"   🌟 已更新最佳模型: {filename} (Acc: {acc * 100:.2f}%)")

    return best_acc


# ===================== 5. 自动化执行 =====================

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform), BATCH_SIZE, True)
    test_loader = DataLoader(datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform), BATCH_SIZE)

    # 实验配置：名称 -> (跳过阶段, 保存的文件名)
    experiments = [
        ("Full Model", None, "best_model_full.pth"),
        ("Ablation Stage 1", 1, "best_model_remove_stage1.pth"),
        ("Ablation Stage 2", 2, "best_model_remove_stage2.pth"),
        ("Ablation Stage 3", 3, "best_model_remove_stage3.pth"),
        ("Ablation Stage 4", 4, "best_model_remove_stage4.pth")
    ]

    results = []

    for name, skip_id, fname in experiments:
        m = StageAblationModel(NUM_CLASSES, skip_stage=skip_id)
        final_acc = run_experiment(name, m, train_loader, test_loader, fname)
        results.append({"实验": name, "准确率": f"{final_acc * 100:.2f}%", "保存位置": fname})

        # 清理显存
        del m
        torch.cuda.empty_cache()

    # 打印最终对比
    print("\n" + "=" * 50)
    print(pd.DataFrame(results).to_string(index=False))
    print("=" * 50)
    print(f"所有模型已存放在: {MODELS_DIR}")