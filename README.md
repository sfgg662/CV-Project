# CV-Project
Neural Style Transfer

## 运行提示

### 关于gatys\lapstyle\sobelstyle
- 直接运行python脚本即可，我们在工作目录下提供了示例**content.png**和**style.png**以供测试
- 结果保存在**output.png**

### 关于 realtimeNST(Johnson)
- 测试时运行指令 **python realtimeNST.py test style2-5 face1**
- 这里的**style2-5.pth**是当前训练出来的比较均衡的模型，指的是当前style2训练出的第五个模型，考虑到占空间较大其他模型并未上传
- **face1**可以替换为face2、clock等，指的是保存在testcontents里的测试用内容图
- 如果有训练的需求，详情见**realtimeNST\train_and_test_guide.md**，但注意需要下载较大的COCO数据集
- 另外，如果想要测试实时视频风格迁移功能，直接运行脚本**realtimeNST\webcam_run.py**，会启用当前设备默认相机

## sobelstyle.py改动（相对lapstyle.py）(详情请见报告)

- 删除了lapstyle的所有代码
- 新增SobelEdgeDetector类和MultiscaleEdgeDetector类

## SobelEdgeDetector

```python
sobel_detector = SobelEdgeDetector() # 初始化sobel边缘检测对象
sobel_img = sobel_detector(img) # 得到边缘检测后图像
```

## MultiscaleEdgeDetector

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

## sobelstyle参数改动

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

# 感谢指正！