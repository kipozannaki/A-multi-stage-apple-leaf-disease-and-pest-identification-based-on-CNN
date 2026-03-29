import os
import torch
import torch.nn as nn
from torchvision import transforms
from flask import Flask, render_template, request, redirect, url_for, jsonify
from PIL import Image
import numpy as np
import datetime
import json

# 定义模型结构
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

# 初始化Flask应用
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# 创建上传目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 历史记录文件
HISTORY_FILE = 'history.json'

# 类别标签 - 详细的病害名称
CLASSES = [
    '健康叶片',
    '叶斑病 (Leaf Spot Disease)',
    '锈病 (Rust Disease)',
    '白粉病 (Powdery Mildew)'
]

# 加载模型
def load_model():
    model = OptimizedMultiStageCNN(num_classes=4)
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'optimized_multistage_v4.pth')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载历史记录
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

# 保存历史记录
def save_history(history):
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# 主页面
@app.route('/')
def index():
    history = load_history()
    return render_template('index.html', history=history)

# 预测接口
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file:
        # 保存上传的文件
        filename = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 处理图像
        image = Image.open(filepath).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        
        # 预测
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            confidence = torch.softmax(outputs, dim=1)[0][predicted.item()].item()
        
        prediction = CLASSES[predicted.item()]
        confidence = round(confidence * 100, 2)
        
        # 更新历史记录
        history = load_history()
        history.append({
            'id': len(history) + 1,
            'filename': filename,
            'prediction': prediction,
            'confidence': confidence,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        save_history(history)
        
        return jsonify({
            'prediction': prediction,
            'confidence': confidence,
            'filename': filename
        })

# 清空历史记录
@app.route('/clear_history', methods=['POST'])
def clear_history():
    save_history([])
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
