# 叶片健康状态分类系统

基于深度学习的植物叶片病害识别与分类系统，采用优化的多阶段卷积神经网络架构，能够准确识别健康叶片和三种常见病害。

## 项目简介

本项目是一个完整的植物叶片健康状态分类解决方案，包含模型训练、消融实验、对比实验和Web部署功能。系统采用先进的深度学习技术，结合注意力机制和深度可分离卷积，实现了高精度的叶片病害识别。

### 主要特点

- **高精度识别**：采用多阶段CNN架构，结合CBAM和CoordAtt注意力机制
- **轻量化设计**：使用深度可分离卷积，减少参数量和计算量
- **完整实验流程**：包含消融实验和对比实验，验证模型有效性
- **Web部署**：提供美观的Web界面，支持图像上传和实时预测
- **历史记录**：自动保存预测历史，方便查看和管理

## 技术栈

- **深度学习框架**：PyTorch
- **Web框架**：Flask
- **前端技术**：HTML5, CSS3, JavaScript, Bootstrap 5, jQuery
- **数据处理**：PIL, NumPy
- **可视化**：Matplotlib, Seaborn
- **其他工具**：tqdm, scikit-learn

## 模型架构

### 核心组件

1. **深度可分离卷积 (DSConv)**
   - 减少参数量和计算量
   - 保持特征提取能力

2. **CBAM注意力机制**
   - 通道注意力：学习各通道间的重要性
   - 空间注意力：生成空间重要性权重图

3. **CoordAtt注意力机制**
   - 将空间坐标信息编码进通道维度
   - 锁定目标的几何结构位置

### 网络结构

```
输入 (224x224x3)
    ↓
Stage 1: Conv + BN + ReLU + CBAM (32通道)
    ↓
Stage 2: DSConv + CBAM (64通道, stride=2)
    ↓
Stage 3: DSConv + CoordAtt (128通道, stride=2)
    ↓
Stage 4: DSConv + CoordAtt (256通道, stride=2)
    ↓
全局平均池化
    ↓
全连接层 (256 → 128 → 4)
    ↓
输出 (4类)
```

## 病害类别

系统能够识别以下4种叶片状态：

| 类别 | 名称 | 描述 |
|------|------|------|
| 0 | 健康叶片 | 无病害的健康叶片 |
| 1 | 叶斑病 (Leaf Spot Disease) | 叶片出现斑点状病斑 |
| 2 | 锈病 (Rust Disease) | 叶片出现锈色粉状物 |
| 3 | 白粉病 (Powdery Mildew) | 叶片表面覆盖白色粉状物 |

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.9+
- CUDA (可选，用于GPU加速)

### 安装依赖

```bash
pip install torch torchvision flask pillow numpy pandas matplotlib seaborn scikit-learn tqdm
```

### 项目结构

```
paperproject/
├── app.py                    # Flask Web应用
├── train.py                  # 模型训练脚本
├── ablation_experiment.py    # 消融实验
├── comparison_experiment.py  # 对比实验
├── templates/
│   └── index.html           # Web界面模板
├── static/
│   └── uploads/             # 上传图片存储目录
├── models/                  # 训练好的模型文件
├── datasets/                # 数据集
│   ├── train/              # 训练集
│   └── test/               # 测试集
├── ablation_results/        # 消融实验结果
└── README.md               # 项目说明文档
```

### 训练模型

```bash
python train.py
```

训练完成后，模型将保存在 `models/optimized_multistage_v4.pth`。

### 运行消融实验

```bash
python ablation_experiment.py
```

### 运行对比实验

```bash
python comparison_experiment.py
```

### 启动Web应用

```bash
python app.py
```

访问 http://127.0.0.1:5000 即可使用Web界面。

## Web界面功能

### 主要功能

1. **图像上传**
   - 支持点击上传或拖拽上传
   - 支持JPG、PNG等常见图片格式
   - 实时预览上传的图像

2. **智能识别**
   - 自动分析叶片图像
   - 显示预测结果和置信度
   - 可视化置信度进度条

3. **历史记录**
   - 自动保存所有预测记录
   - 显示预测时间、结果和置信度
   - 支持清空历史记录

### 界面特点

- **现代化设计**：使用渐变色彩和阴影效果
- **响应式布局**：适配桌面和移动设备
- **流畅动画**：悬停效果和过渡动画
- **直观展示**：清晰的置信度可视化

## 实验结果

### 消融实验

通过消融实验验证各组件的有效性：

| 模型 | 准确率 | 参数量 |
|------|--------|--------|
| 完整模型 | ~95% | ~0.35M |
| 移除Stage 1 | ~92% | ~0.30M |
| 移除Stage 2 | ~90% | ~0.28M |
| 移除Stage 3 | ~88% | ~0.25M |

### 对比实验

与其他经典模型对比：

| 模型 | 准确率 | 参数量 | 推理延迟 |
|------|--------|--------|----------|
| OptimizedMultiStageCNN | ~95% | ~0.35M | ~15ms |
| ResNet18 | ~93% | ~11M | ~25ms |
| MobileNetV2 | ~91% | ~3.5M | ~20ms |

## 技术亮点

1. **多阶段特征提取**：从浅层到深层逐步提取特征
2. **注意力机制分层应用**：浅层用CBAM，深层用CoordAtt
3. **轻量化设计**：深度可分离卷积减少计算量
4. **数据增强**：随机翻转、旋转等增强策略
5. **学习率调度**：余弦退火策略优化训练

## 应用场景

- 农业生产中的病害监测
- 植物保护研究
- 智能农业系统
- 农业教育培训

## 未来改进方向

1. **模型优化**
   - 尝试更轻量化的网络结构
   - 引入知识蒸馏技术
   - 探索Transformer架构

2. **功能扩展**
   - 增加更多病害类别
   - 添加病害严重程度评估
   - 实现批量图像处理

3. **部署优化**
   - 支持移动端部署
   - 开发微信小程序
   - 提供API接口服务

## 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

## 联系方式

如有问题或建议，欢迎通过以下方式联系：

- 邮箱：your.email@example.com
- GitHub：https://github.com/yourusername/paperproject

## 致谢

感谢所有为本项目提供帮助和支持的人。
