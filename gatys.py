import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  #用于插值
from PIL import Image
from torchvision import transforms, models

# lsf-test 3

#设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imsize = 512 if torch.cuda.is_available() else 128
unloader = transforms.ToPILImage()

#加载图像
def image_loader(image_name, new_size=None):
    image = Image.open(image_name).convert('RGB')
    if new_size is not None:
        #如果指定了尺寸，则缩放
        transform = transforms.Compose([
            transforms.Resize(new_size), 
            transforms.ToTensor()
        ])
    else:
        #如果没指定尺寸，保持原尺寸
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
    
    image = transform(image).unsqueeze(0)
    return image.to(device, torch.float)


original_pil = Image.open("content.png")
orig_width, orig_height = original_pil.size
print(f"Original Image Size: {orig_width} x {orig_height}")


content_img = image_loader("content.png", new_size=imsize)
style_img = image_loader("style.png", new_size=imsize)

print(f"Processing Image Size: {content_img.shape}") 

#初始化生成的图片
input_img = content_img.clone()
input_img.requires_grad_(True) 

#gram矩阵计算
def gram_matrix(input):
    b, c, h, w = input.size()
    features = input.view(b * c, h * w)
    G = torch.mm(features, features.t()) 
    return G.div(b * c * h * w) 

#加载 VGG19 模型
cnn = models.vgg19(pretrained=True).features.to(device).eval()

#归一化参数
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(-1, 1, 1)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(-1, 1, 1)

#指定层
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

#获取特征
def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv_1', '5': 'conv_2', '10': 'conv_3', '19': 'conv_4', '28': 'conv_5'}
    
    features = {}
    x = image
    x = (x - cnn_normalization_mean) / cnn_normalization_std
    
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

#预先计算目标特征
with torch.no_grad():
    target_content_features = get_features(content_img, cnn)
    target_style_features = get_features(style_img, cnn)

style_grams = {layer: gram_matrix(target_style_features[layer]) for layer in target_style_features}

#优化器  ！！特殊
optimizer = optim.LBFGS([input_img])

# 权重
style_weight = 1000000 
content_weight = 1     

print('Starting optimization...')
run = [0]
while run[0] <= 300: 
    #闭包！！！
    def closure():
        optimizer.zero_grad()
        input_img.data.clamp_(0, 1)

        features = get_features(input_img, cnn)
        
        style_loss = 0
        content_loss = 0

        for layer in content_layers_default:
            content_loss += torch.mean((features[layer] - target_content_features[layer]) ** 2)

        for layer in style_layers_default:
            layer_feature = features[layer]
            layer_gram = gram_matrix(layer_feature)
            style_gram = style_grams[layer]
            style_loss += torch.mean((layer_gram - style_gram) ** 2)

        total_loss = content_weight * content_loss + style_weight * style_loss
        total_loss.backward()
        
        run[0] += 1
        if run[0] % 50 == 0:
            print(f"Run {run[0]}: Loss {total_loss.item()}")

        return total_loss

    optimizer.step(closure)

input_img.data.clamp_(0, 1)

#恢复原图尺寸
output_img = F.interpolate(input_img, size=(orig_height, orig_width), mode='bilinear', align_corners=False)

#保存结果
result = unloader(output_img.cpu().clone().squeeze(0))
result.save('output.png')
print(f"Done! Saved output.png with size {result.size}")