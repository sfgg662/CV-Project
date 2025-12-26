import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms, models

# lsf-test

#设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#图像加载与预处理
#VGG要求的标准化参数
imsize = 512 if torch.cuda.is_available() else 128
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor(),
])
unloader = transforms.ToPILImage()

def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0) #增加batch维度
    return image.to(device, torch.float)


content_img = image_loader("content.png")
style_img = image_loader("style.png")


#初始化生成的图片
input_img = content_img.clone()
input_img.requires_grad_(True) 


#gram矩阵计算(代表风格特征)
def gram_matrix(input):
    b, c, h, w = input.size()
    features = input.view(b * c, h * w)
    G = torch.mm(features, features.t()) #向量点积
    return G.div(b * c * h * w) # 归一化
#注意：！！这里的归一化使得gram矩阵的值变得很小很小 -->权重很大 用来恢复量级

#加载 VGG19 模型
cnn = models.vgg19(pretrained=True).features.to(device).eval()

#定义归一化 (VGG训练时的均值和方差)
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(-1, 1, 1)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(-1, 1, 1)

#指定的用于计算损失的层
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

#获取特征并计算目标值的帮助函数
def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv_1', '5': 'conv_2', '10': 'conv_3', '19': 'conv_4', '28': 'conv_5'}
    
    features = {}
    x = image
    #手动归一化
    x = (x - cnn_normalization_mean) / cnn_normalization_std
    
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

#预先计算目标的特征
with torch.no_grad():
    target_content_features = get_features(content_img, cnn)
    target_style_features = get_features(style_img, cnn)


#这里对style.png处理多一步--> 计算gram矩阵
style_grams = {layer: gram_matrix(target_style_features[layer]) for layer in target_style_features}


#特殊！！！！：LBFGS 优化器-->配闭包来用
optimizer = optim.LBFGS([input_img])


#参数可调  内容图和风格图占比
style_weight = 1000000 # 风格权重（是大权重！！！！
content_weight = 1     # 内容权重

print('Starting optimization...')
run = [0]
while run[0] <= 300: # 迭代300次
    #闭包
    def closure():
        #必须在 closure 内部清零梯度
        optimizer.zero_grad()
        
        #限制像素范围
        input_img.data.clamp_(0, 1)

        #重新提取特征（构建新的计算图）
        features = get_features(input_img, cnn)
        
        #必须在这里初始化/重置loss为 0
        #这样style_loss就是一个全新的变量，和上一轮无关
        style_loss = 0
        content_loss = 0

        #计算 Content Loss
        for layer in content_layers_default:
            content_loss += torch.mean((features[layer] - target_content_features[layer]) ** 2)

        #计算 Style Loss
        for layer in style_layers_default:
            layer_feature = features[layer]
            layer_gram = gram_matrix(layer_feature)
            style_gram = style_grams[layer]
            style_loss += torch.mean((layer_gram - style_gram) ** 2)

        #总损失
        total_loss = content_weight * content_loss + style_weight * style_loss
        
        #反向传播
        total_loss.backward()
        
        #打印日志
        run[0] += 1
        if run[0] % 50 == 0:
            print(f"Run {run[0]}: Loss {total_loss.item()}")

        return total_loss

    #开始优化
    optimizer.step(closure)

#最后修正一次范围
input_img.data.clamp_(0, 1)

#保存/显示结果
result = unloader(input_img.cpu().clone().squeeze(0))
result.save('output.png')
print("Done!")