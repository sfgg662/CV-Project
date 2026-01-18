import cv2
import torch
import numpy as np
from torchvision import transforms

# ==========================================
# 1. 导入队友写的模型类
# ==========================================
# 【需要修改】：
# 假设队友的文件叫 models.py，类名叫 TransformerNet
# 如果文件名不同，请修改 'from ...'；如果类名不同，请修改 'import ...'

from realtimeNST import TransformerNet 

# ==========================================
# 2. 配置参数
# ==========================================
# 【需要修改】：替换为你队友训练好的权重文件路径
MODEL_PATH = "style2-5.pth" 

# 图像处理大小：越小速度越快。建议 480 或 640
IMG_SCALE = 640 

# 设备检测
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# ==========================================
# 3. 辅助函数：预处理与后处理
# ==========================================
def preprocess(frame):
    """将 OpenCV 的图像转为 PyTorch 模型能吃的 Tensor"""
    h, w = frame.shape[:2]
    # 保持长宽比缩放
    new_h = int(h * (IMG_SCALE / w))
    frame = cv2.resize(frame, (IMG_SCALE, new_h))
    
    # 归一化与维度转换
    # ImageNet 标准均值方差 (你需要确认队友训练时是否用了这个，通常都是用的)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    frame = frame / 255.0
    frame = (frame - mean) / std
    frame = frame.transpose(2, 0, 1) # HWC -> CHW
    
    tensor = torch.FloatTensor(frame).unsqueeze(0).to(device)
    return tensor

def postprocess(tensor):
    """将 PyTorch 输出的 Tensor 转回 OpenCV 能显示的图像"""
    tensor = tensor.cpu().detach().squeeze(0)
    
    # 反归一化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    
    # 限制范围并转格式
    tensor = tensor.clamp(0, 1) * 255
    img = tensor.numpy().transpose(1, 2, 0).astype("uint8")
    return img

# ==========================================
# 4. 主逻辑
# ==========================================
def run_webcam():
    # A. 加载模型
    print("Loading model...")
    # 【需要修改】：如果队友的模型类名不叫 TransformerNet，请修改这里
    style_model = TransformerNet().to(device)
    
    # 加载权重
    try:
        style_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        print("请检查 MODEL_PATH 路径是否正确，以及模型架构是否匹配。")
        return

    style_model.eval()
    print("Model loaded! Starting webcam...")

    # B. 打开摄像头
    cap = cv2.VideoCapture(0) # 0 是默认摄像头
    
    if not cap.isOpened():
        print("无法打开摄像头！")
        return

    print("按 'q' 键退出程序")

    with torch.no_grad(): # 关键：不计算梯度，加速推理
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 1. 格式转换：OpenCV 读进来是 BGR，模型通常训练用 RGB
            content_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 2. 预处理
            content_tensor = preprocess(frame)
            
            # 3. 模型推理 (风格迁移核心步骤)
            generated_tensor = style_model(content_tensor)
            
            # 4. 后处理
            output_img = postprocess(generated_tensor)

            hsv = cv2.cvtColor(output_img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            s = cv2.multiply(s, 1.5) # 饱和度 x 1.5
            s = np.clip(s, 0, 255).astype(np.uint8)
            hsv = cv2.merge((h, s, v))
            output_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # 2. 增加对比度 (让纹理更深)
            alpha = 1.3 # 对比度
            beta = -20  # 亮度 (稍微压暗一点，模拟星月夜的黑夜感)
            output_img = cv2.convertScaleAbs(output_img, alpha=alpha, beta=beta)

            
            # 5. 转回 BGR 以便 OpenCV 显示
            output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

        
            # 6. (可选) 把原图缩小，贴在左上角做对比
            h, w = output_img.shape[:2]
            small_frame = cv2.resize(frame, (int(w/4), int(h/4)))
            output_img[0:int(h/4), 0:int(w/4)] = small_frame


            # 7. 显示
            cv2.imshow('Real-time Style Transfer (Press q to exit)', output_img)
            
            # 按 q 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam()