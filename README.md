# CV-Project
Neural Style Transfer

### sobelstyle.py改动（相对lapstyle.py）

- 删除了lapstyle的所有代码
- 新增SobelEdgeDetector类和MultiscaleEdgeDetector类

### SobelEdgeDetector

```python
sobel_detector = SobelEdgeDetector() # 初始化sobel边缘检测对象
sobel_img = sobel_detector(img) # 得到边缘检测后图像
```

### MultiscaleEdgeDetector

```python
class MultiscaleEdgeDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel = SobelEdgeDetector()

    def forward(self, img):
        sobel_list = []
		
        # 尺度1，对原图直接sobel检测
        scale1 = self.sobel(img) 
		
        # 尺度2，avg_pool降采样1倍
        img = F.avg_pool2d(img, kernel_size = 2, stride = 2) 
        scale2 = self.sobel(img)

        sobel_list.append(scale1)
        sobel_list.append(scale2)
        return sobel_list # 返回两个检测图组成的列表
```

如果想要尝试更多尺度，直接修改上述类，增加scale3，scale4等，再append到sobel_list中即可。

### sobelstyle参数改动

```python
# lap_weight -> sobel_weight
sobel_weight = 1000
# 分别对应scale1，scale2的权重，接近原尺寸的边缘检测图权重大些
sobel_scale = [0.8, 0.2]

with torch.no_grad():
    content_sobel_list = sobel_calculator(content_img)

# for ...
	# 省略
	syn_sobel_list = sobel_calculator(syn_img)
        sobel_loss = 0
        for i, (content_sobel, syn_sobel) in enumerate(zip(content_sobel_list, syn_sobel_list)):
            sobel_loss += sobel_scale[i] * F.mse_loss(content_sobel, syn_sobel)

        total_loss = content_weight * content_loss + style_weight * style_loss + sobel_loss * sobel_weight
	
```

调整超参就是调整大的权重（sobel_weight）和多尺度sobel_scale的权重（如果scale不止两个就增加sobel_scale的内容，保证和为1即可）。
