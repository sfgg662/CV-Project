import os
import random
import zipfile
import urllib.request
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.datasets.utils import download_url
import torch.nn.functional as F

# -------------------------
# 基本配置
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 4  # 增加训练轮数
LR = 1e-3
NUM_CONTENT_IMAGES = 5000  # 增加训练数据量
STYLE_WEIGHT = 1e4  # 风格损失权重（调整为与归一化后的VGG输出量级匹配）
CONTENT_WEIGHT = 1.0  # 内容损失权重
TV_WEIGHT = 1e-6  # Total Variation 平滑损失权重
COLOR_WEIGHT = 1e-2  # 颜色守恒损失权重（匹配输出与内容的通道均值/方差）
# 多尺度/分层风格超参数
# VGG 提取的4层对应的权重（relu1_2, relu2_2, relu3_3, relu4_3）
STYLE_LAYER_WEIGHTS = [0.5, 0.3, 0.1, 0.1]
# 风格图使用的不同分辨率（像素大小），会计算每个尺度的 Gram 矩阵
STYLE_SCALES = [IMAGE_SIZE, IMAGE_SIZE * 2]
# 每个尺度的权重，长度需与 STYLE_SCALES 一致
STYLE_SCALE_WEIGHTS = [1.0, 0.5]

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
COCO_DIR = os.path.join(ROOT_DIR, "coco")
STYLE_PATH = os.path.join(ROOT_DIR, "style.png")
TEST_CONTENT_PATH = os.path.join(ROOT_DIR, "testcontent.png")
MODEL_PATH = os.path.join(ROOT_DIR, "realtime_style.pth")
OUTPUT_PATH = os.path.join(ROOT_DIR, "output_stylized.png")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
TEST_CONTENTS_DIR = os.path.join(ROOT_DIR, "testcontents")

# -------------------------
# 下载 COCO val2017 (小数据集，仅1GB)
# -------------------------
def download_coco():
    os.makedirs(COCO_DIR, exist_ok=True)
    img_dir = os.path.join(COCO_DIR, "val2017")

    if os.path.exists(img_dir):
        num_images = len([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        print(f"✅ COCO 已存在 ({num_images} 张图片)，跳过下载")
        return img_dir

    url = "http://images.cocodataset.org/zips/val2017.zip"
    zip_path = os.path.join(COCO_DIR, "val2017.zip")

    # 检查是否有不完整的zip文件
    if os.path.exists(zip_path):
        print("⚠️  检测到不完整的zip文件，删除中...")
        os.remove(zip_path)

    try:
        print("⬇️ 下载 COCO val2017 (约1GB，包含5000张图片)...")
        download_url(url, COCO_DIR, filename="val2017.zip")
        
        print("📦 验证zip文件...")
        if not zipfile.is_zipfile(zip_path):
            raise Exception("下载的文件不是有效的zip文件")
        
        print("📦 解压中...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(COCO_DIR)
        
        os.remove(zip_path)
        print("✅ COCO数据集下载完成")
        return img_dir
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        raise

# -------------------------
# Dataset
# -------------------------
class CocoSubset(Dataset):
    def __init__(self, image_dir, num_images):
        all_imgs = os.listdir(image_dir)
        self.imgs = random.sample(all_imgs, num_images)

        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor()
        ])
        self.image_dir = image_dir

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        path = os.path.join(self.image_dir, self.imgs[idx])
        img = Image.open(path).convert("RGB")
        return self.transform(img)

# -------------------------
# Transformer Network
# -------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class TransformerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 9, 1, 4),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, 2, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 3, 9, 1, 4)
        )

    def forward(self, x):
        return self.model(x)

# -------------------------
# VGG Feature Extractor
# -------------------------
class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES).features
        self.layers = nn.ModuleList(vgg[:23])
        # register imagenet normalization (expects input in [0,1])
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        # x is expected in [0,1]; apply ImageNet normalization
        x = (x - self.mean) / self.std
        features = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in {3, 8, 15, 22}:
                features.append(x)
        return features # 这里只需要提取4个指定层的特征 ！！！

def gram_matrix(x):
    b, c, h, w = x.size()
    f = x.view(b, c, h * w)
    g = torch.bmm(f, f.transpose(1, 2))
    return g / (c * h * w)

# -------------------------
# 主训练流程
# -------------------------
def main(style_name=None):
    img_dir = download_coco()
    dataset = CocoSubset(img_dir, NUM_CONTENT_IMAGES)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    transformer = TransformerNet().to(DEVICE)
    vgg = VGG16().to(DEVICE).eval()
    optimizer = optim.Adam(transformer.parameters(), LR)

    # 计算多尺度风格 Gram 矩阵（每个尺度保存一组 layer-wise grams）
    # 确定使用的风格图（支持 styles 目录下的按名选择）
    if style_name:
        # 支持传入带或不带扩展名的 style 名称
        if style_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            candidate = os.path.join(ROOT_DIR, 'styles', style_name)
        else:
            candidate = os.path.join(ROOT_DIR, 'styles', f"{style_name}.png")
        if not os.path.exists(candidate):
            print(f"❌ 未找到风格图: {candidate}")
            if os.path.exists(os.path.join(ROOT_DIR, 'styles')):
                files = [f for f in os.listdir(os.path.join(ROOT_DIR, 'styles')) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                print("可用的 styles：")
                for f in files[:50]:
                    print(f"  - {f}")
            raise SystemExit(1)
        style_img = Image.open(candidate).convert("RGB")
        style_tag = os.path.splitext(os.path.basename(candidate))[0]
    else:
        style_img = Image.open(STYLE_PATH).convert("RGB")
        style_tag = 'default'
    style_grams_scales = []
    if len(STYLE_SCALES) != len(STYLE_SCALE_WEIGHTS):
        raise ValueError("STYLE_SCALES 和 STYLE_SCALE_WEIGHTS 长度需相同")

    for s in STYLE_SCALES:
        style_tf = transforms.Compose([
            transforms.Resize(s),
            transforms.CenterCrop(s),
            transforms.ToTensor()
        ])
        style_resized = style_tf(style_img).unsqueeze(0).to(DEVICE)
        style_feats = vgg(style_resized)
        style_grams = [gram_matrix(f) for f in style_feats]
        style_grams_scales.append(style_grams)

    # 归一化并校验层权重与尺度权重，避免负值或和为0
    import math
    layer_ws = torch.tensor(STYLE_LAYER_WEIGHTS, dtype=torch.float32)
    if (layer_ws < 0).any():
        raise ValueError("STYLE_LAYER_WEIGHTS 中不能含有负值")
    layer_sum = layer_ws.sum().item()
    if math.isclose(layer_sum, 0.0):
        raise ValueError("STYLE_LAYER_WEIGHTS 的和不能为0")
    normalized_layer_ws = (layer_ws / layer_sum).tolist()

    scale_ws = torch.tensor(STYLE_SCALE_WEIGHTS, dtype=torch.float32)
    if (scale_ws < 0).any():
        raise ValueError("STYLE_SCALE_WEIGHTS 中不能含有负值")
    scale_sum = scale_ws.sum().item()
    if math.isclose(scale_sum, 0.0):
        raise ValueError("STYLE_SCALE_WEIGHTS 的和不能为0")
    normalized_scale_ws = (scale_ws / scale_sum).tolist() # 为了确保总和为1，所以进行一个归一化

    # 创建检查点目录
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    print("🚀 开始训练...")
    print(f"设备: {DEVICE}")
    print(f"训练轮数: {EPOCHS} | 图片数量: {NUM_CONTENT_IMAGES} | 批次大小: {BATCH_SIZE}")
    print(f"内容权重: {CONTENT_WEIGHT} | 风格权重: {STYLE_WEIGHT}")
    
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for i, content in enumerate(loader):
            content = content.to(DEVICE)

            stylized = transformer(content)
            content_feats = vgg(content)
            stylized_feats = vgg(stylized)

            content_loss = torch.mean((stylized_feats[1] - content_feats[1]) ** 2) # 只选第一个特征层

            # 计算多尺度、多层次加权的风格损失
            style_loss = 0.0
            # 验证层权重长度
            num_layers = len(stylized_feats)
            if len(STYLE_LAYER_WEIGHTS) != num_layers:
                raise ValueError(f"STYLE_LAYER_WEIGHTS 长度应为 {num_layers}")

            for s_idx, (scale_grams, scale_w) in enumerate(zip(style_grams_scales, normalized_scale_ws)): # 这里是归一化之后的权重
                target_size = STYLE_SCALES[s_idx]
                # 把 stylized 缩放到当前尺度再提特征
                stylized_scaled = F.interpolate(stylized, size=target_size, mode='bilinear', align_corners=False)
                scaled_feats = vgg(stylized_scaled)
                per_scale_loss = 0.0
                for l_idx, (sf, sg) in enumerate(zip(scaled_feats, scale_grams)):
                    layer_w = normalized_layer_ws[l_idx]
                    per_scale_loss += layer_w * torch.mean((gram_matrix(sf) - sg) ** 2)
                style_loss += scale_w * per_scale_loss

            # Total Variation 损失，平滑输出
            def total_variation_loss(img):
                # img: (B, C, H, W)
                dh = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).mean()
                dw = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).mean()
                return dh + dw

            tv_loss = total_variation_loss(stylized)

            # 颜色守恒损失：匹配每通道的均值和标准差，防止局部亮斑和失真
            # 在 [0,1] 范围上计算
            def color_stats_loss(x, y):
                # x, y: (B, C, H, W)
                mx = x.mean(dim=[0, 2, 3])
                my = y.mean(dim=[0, 2, 3])
                sx = x.std(dim=[0, 2, 3])
                sy = y.std(dim=[0, 2, 3])
                return torch.mean((mx - my) ** 2) + torch.mean((sx - sy) ** 2)

            color_loss = color_stats_loss(stylized.clamp(0.0, 1.0), content)

            loss = CONTENT_WEIGHT * content_loss + STYLE_WEIGHT * style_loss + TV_WEIGHT * tv_loss + COLOR_WEIGHT * color_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            if i % 50 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Step {i}/{len(loader)} | "
                      f"Loss: {loss.item():.2f} | Content: {content_loss.item():.2f} | "
                      f"Style: {style_loss.item():.4f}")
        
        # 每个epoch结束保存检查点和测试图片
        avg_loss = epoch_loss / len(loader)
        print(f"\n📊 Epoch {epoch+1} 完成 | 平均Loss: {avg_loss:.2f}")
        
        # 保存检查点（包含 style_tag 以区分不同风格）
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_{style_tag}_epoch_{epoch+1}.pth")
        torch.save(transformer.state_dict(), checkpoint_path)
        print(f"💾 检查点已保存: {checkpoint_path}")
        
        # 生成测试图片
        if os.path.exists(TEST_CONTENT_PATH):
            transformer.eval()
            with torch.no_grad():
                test_img = Image.open(TEST_CONTENT_PATH).convert("RGB")
                test_tf = transforms.Compose([
                    transforms.Resize(IMAGE_SIZE),
                    transforms.CenterCrop(IMAGE_SIZE),
                    transforms.ToTensor()
                ])
                test_tensor = test_tf(test_img).unsqueeze(0).to(DEVICE)
                test_output = transformer(test_tensor).cpu().clamp(0.0, 1.0)
                test_result = transforms.ToPILImage()(test_output[0])
                test_result.save(os.path.join(CHECKPOINT_DIR, f"test_epoch_{epoch+1}.png"))
                print(f"🎨 测试图片已保存: test_epoch_{epoch+1}.png\n")
            transformer.train()
        print("="*80)

    # 保存最终模型，按 style_tag 命名，同时覆盖默认模型文件以便后续默认推理
    final_model_path = os.path.join(ROOT_DIR, f"realtime_style-{style_tag}.pth")
    torch.save(transformer.state_dict(), final_model_path)
    # 也保存到默认模型路径，便于以前的推理命令继续工作
    torch.save(transformer.state_dict(), MODEL_PATH)
    print(f"✅ 训练完成，模型已保存: {final_model_path} (同时更新 {MODEL_PATH})")

# -------------------------
# 推理函数：处理单张图片
# -------------------------
def stylize_image(content_path, model_path, output_path):
    """
    使用训练好的模型对单张图片进行风格迁移
    
    Args:
        content_path: 待处理的内容图路径
        model_path: 训练好的模型路径
        output_path: 输出图片保存路径
    """
    print(f"🎨 加载模型: {model_path}")
    transformer = TransformerNet().to(DEVICE)
    transformer.load_state_dict(torch.load(model_path, map_location=DEVICE))
    transformer.eval()
    
    print(f"📷 加载内容图: {content_path}")
    content_img = Image.open(content_path).convert("RGB")
    
    # 保存原始尺寸用于还原
    original_size = content_img.size
    
    # 预处理
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    content_tensor = transform(content_img).unsqueeze(0).to(DEVICE)
    
    # 推理
    print("⚡ 进行风格迁移...")
    with torch.no_grad():
        stylized_tensor = transformer(content_tensor)
    
    # 后处理（假定输出在 [0,1]）
    stylized_tensor = stylized_tensor.squeeze(0).cpu().clamp(0.0, 1.0)
    stylized_img = transforms.ToPILImage()(stylized_tensor)
    
    # 还原到原始尺寸
    stylized_img = stylized_img.resize(original_size, Image.LANCZOS)
    
    # 保存
    stylized_img.save(output_path)
    print(f"✅ 风格迁移完成！保存到: {output_path}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # 推理模式：支持多种参数格式
        # 1) python realtimeNST.py test                -> 使用默认模型和 TEST_CONTENT_PATH
        # 2) python realtimeNST.py test <content>      -> 使用默认模型，content 可以是 id 或文件名
        # 3) python realtimeNST.py test <model> <content> -> 指定模型文件（本目录或绝对路径）和 content
        args = sys.argv[2:]
        model_file = None
        content_arg = None

        if len(args) == 0:
            # 使用默认
            model_file = MODEL_PATH
            content_arg = None
        elif len(args) == 1:
            a = args[0]
            if a.lower().endswith('.pth'):
                model_file = os.path.join(ROOT_DIR, a) if not os.path.isabs(a) else a
                content_arg = None
            else:
                model_file = MODEL_PATH
                content_arg = a
        else:
            # 两个及以上参数，第一为模型，第二为内容标识
            m = args[0]
            content_arg = args[1]

            # 支持省略 .pth 后缀并在当前目录或 checkpoints 中查找
            candidates = []
            if os.path.isabs(m):
                candidates.append(m)
                candidates.append(m + '.pth')
            else:
                candidates.append(os.path.join(ROOT_DIR, m))
                candidates.append(os.path.join(ROOT_DIR, m + '.pth'))
                candidates.append(os.path.join(CHECKPOINT_DIR, m))
                candidates.append(os.path.join(CHECKPOINT_DIR, m + '.pth'))

            model_file = None
            for c in candidates:
                if os.path.exists(c):
                    model_file = c
                    break
            if model_file is None:
                # 如果都没找到，保留最可能的路径以便后续报错并提示可用文件
                model_file = os.path.join(ROOT_DIR, m)
                print(f"⚠️ 未找到指定模型的候选路径，已尝试: {candidates}")

        # 验证模型文件
        if not os.path.exists(model_file):
            print(f"❌ 模型文件不存在: {model_file}")
            # 列出当前目录下的可用 pth 文件和 checkpoints
            available = [f for f in os.listdir(ROOT_DIR) if f.lower().endswith('.pth')]
            ck = []
            if os.path.exists(CHECKPOINT_DIR):
                ck = [f for f in os.listdir(CHECKPOINT_DIR) if f.lower().endswith('.pth')]
            if available or ck:
                print("可用的模型文件：")
                for f in available[:20]:
                    print(f"  - {f}")
                for f in ck[:20]:
                    print(f"  - {os.path.join('checkpoints', f)}")
            else:
                print("在项目目录和 checkpoints 中未找到 .pth 模型文件")
            sys.exit(1)

        # 确定内容图片路径
        if content_arg:
            ca = content_arg
            if ca.lower().endswith(('.png', '.jpg', '.jpeg')):
                candidate = os.path.join(TEST_CONTENTS_DIR, ca)
            else:
                candidate = os.path.join(TEST_CONTENTS_DIR, f"testcontent-{ca}.png")
            content_path = candidate
        else:
            content_path = TEST_CONTENT_PATH

        if not os.path.exists(content_path):
            print(f"❌ 测试图片不存在: {content_path}")
            if os.path.exists(TEST_CONTENTS_DIR):
                files = [f for f in os.listdir(TEST_CONTENTS_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                print(f"可用的测试图片（存于 {TEST_CONTENTS_DIR}）：")
                for f in files[:50]:
                    print(f"  - {f}")
            else:
                print(f"目录不存在: {TEST_CONTENTS_DIR}")
            sys.exit(1)

        # 输出文件名生成：包含模型名（无后缀）和内容标识
        model_base = os.path.splitext(os.path.basename(model_file))[0]
        if content_arg:
            content_id = os.path.splitext(os.path.basename(content_path))[0]
            # 去掉 testcontent- 前缀
            if content_id.startswith('testcontent-'):
                content_id = content_id[len('testcontent-'):]
            out_name = f"output_stylized-{model_base}-{content_id}.png"
        else:
            out_name = f"output_stylized-{model_base}.png"
        output_path = os.path.join(ROOT_DIR, out_name)

        stylize_image(content_path, model_file, output_path)
    elif len(sys.argv) > 1 and sys.argv[1] == "train":
        # 训练模式：支持 `python realtimeNST.py train style1`
        style_arg = None
        if len(sys.argv) > 2:
            style_arg = sys.argv[2]
        main(style_arg)
    else:
        # 训练模式
        main()
