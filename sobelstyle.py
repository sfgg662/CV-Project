import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from PIL import Image
from torchvision import transforms, models
import os

# ·��������ͳһ����ǰ����Ŀ¼
curr_dir = os.path.dirname(os.path.abspath(__file__))
content_dir = os.path.join(curr_dir, "content.png")
style_dir = os.path.join(curr_dir, "style.png")
output_dir = os.path.join(curr_dir, "output_sobel.png")

#�����豸
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imsize = 512 if torch.cuda.is_available() else 128
unloader = transforms.ToPILImage()

#����ͼ��
def image_loader(image_name, new_size=None):
    image = Image.open(image_name).convert('RGB')
    if new_size is not None:
        #���ָ���˳ߴ磬������
        transform = transforms.Compose([
            transforms.Resize(new_size), 
            transforms.ToTensor()
        ])
    else:
        #���ûָ���ߴ磬����ԭ�ߴ�
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
    
    image = transform(image).unsqueeze(0)
    return image.to(device, torch.float)


original_pil = Image.open(content_dir)
orig_width, orig_height = original_pil.size
print(f"Original Image Size: {orig_width} x {orig_height}")


content_img = image_loader(content_dir, new_size=imsize)
style_img = image_loader(style_dir, new_size=imsize)

print(f"Processing Image Size: {content_img.shape}") 

# ��ʼ���ϳ�ͼƬ
syn_img = content_img.clone()
syn_img.requires_grad_(True) 

# gram�������
def gram_matrix(input):
    b, c, h, w = input.size()
    features = input.view(b * c, h * w)
    G = torch.mm(features, features.t()) 
    return G.div(b * c * h * w)

# Sobel����
class SobelEdgeDetector(torch.nn.Module):
    def __init__(self):
        super(SobelEdgeDetector, self).__init__()
        
        # Sobel ��
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32)
        
        sobel_y = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def forward(self, x):
        """
        x: [B, C, H, W] ������
        ����: Sobel �ݶȷ�ֵ [B, C, H, W]
        """
        # ��ÿ��ͨ���ֱ�Ӧ�� Sobel
        # x = self.gaussian_blur(x)
        channels = []
        for i in range(x.shape[1]):
            channel = x[:, i:i+1, :, :]
            
            # ���� x �� y �����ݶ�
            grad_x = F.conv2d(channel, self.sobel_x, padding=1)
            grad_y = F.conv2d(channel, self.sobel_y, padding=1)
            
            # �ݶȷ�ֵ
            magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
            channels.append(magnitude)
        
        return torch.cat(channels, dim=1)

class MultiscaleEdgeDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel = SobelEdgeDetector()

    def forward(self, img):
        sobel_list = []

        scale1 = self.sobel(img)

        img = F.avg_pool2d(img, kernel_size = 2, stride = 2)
        scale2 = self.sobel(img)

        sobel_list.append(scale1)
        sobel_list.append(scale2)
        return sobel_list

#���� VGG19 ģ��
cnn = models.vgg19(pretrained=True).features.to(device).eval()

#��һ������
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(-1, 1, 1)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(-1, 1, 1)

#ָ����
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

#��ȡ����
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

#Ԥ�ȼ���Ŀ������
with torch.no_grad():
    target_content_features = get_features(content_img, cnn)
    target_style_features = get_features(style_img, cnn)

style_grams = {layer: gram_matrix(target_style_features[layer]) for layer in target_style_features}

#�Ż���  ��������
optimizer = optim.LBFGS([syn_img])

# ��ʼ��
sobel_calculator = MultiscaleEdgeDetector().to(device)

# Ȩ��
style_weight = 1000000
content_weight = 1
sobel_weight = 1000
sobel_scale = [0.8, 0.2]

with torch.no_grad():
    content_sobel_list = sobel_calculator(content_img)

print('Starting optimization...')
run = [0]
while run[0] <= 250: 
    # �հ�������
    def closure():
        optimizer.zero_grad()
        syn_img.data.clamp_(0, 1)

        features = get_features(syn_img, cnn)
        
        style_loss = 0
        content_loss = 0

        for layer in content_layers_default:
            content_loss += torch.mean((features[layer] - target_content_features[layer]) ** 2)

        for layer in style_layers_default:
            layer_feature = features[layer]
            layer_gram = gram_matrix(layer_feature)
            style_gram = style_grams[layer]
            style_loss += torch.mean((layer_gram - style_gram) ** 2)

        syn_sobel_list = sobel_calculator(syn_img)
        sobel_loss = 0
        for i, (content_sobel, syn_sobel) in enumerate(zip(content_sobel_list, syn_sobel_list)):
            sobel_loss += sobel_scale[i] * F.mse_loss(content_sobel, syn_sobel)

        total_loss = content_weight * content_loss + style_weight * style_loss + sobel_loss * sobel_weight
        total_loss.backward()
        
        run[0] += 1
        if run[0] % 50 == 0:
            print(f"Run {run[0]}: Loss {total_loss.item():.2f}")

        return total_loss

    optimizer.step(closure)

syn_img.data.clamp_(0, 1)

#�ָ�ԭͼ�ߴ�
output_img = F.interpolate(syn_img, size=(orig_height, orig_width), mode='bilinear', align_corners=False)

#������
result = unloader(output_img.cpu().clone().squeeze(0))
result.save(output_dir)
print(f"Done! Saved output.png with size {result.size}")