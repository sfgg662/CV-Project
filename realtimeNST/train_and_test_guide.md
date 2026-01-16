# 实时风格迁移训练与测试指南

## 目录
1. [所有可调节的超参数](#所有可调节的超参数)
2. [训练命令](#训练命令)
3. [测试/推理命令](#测试推理命令)

---

## 所有可调节的超参数

所有超参数定义在 `realtimeNST.py` 文件顶部（行 17-37）。

### 基础训练参数

| 超参数 | 默认值 | 类型 | 说明 |
|--------|-------|------|------|
| `DEVICE` | `"cuda" if torch.cuda.is_available() else "cpu"` | str | 计算设备（GPU/CPU） |
| `IMAGE_SIZE` | `256` | int | 训练图片的固定尺寸（像素） |
| `BATCH_SIZE` | `4` | int | 每个 batch 的图片数量 |
| `EPOCHS` | `4` | int | 训练轮数 |
| `LR` | `1e-3` | float | Adam 优化器的学习率 |
| `NUM_CONTENT_IMAGES` | `5000` | int | 从 COCO 数据集中使用的图片数量 |

---

### 损失函数权重

| 超参数 | 默认值 | 类型 | 说明 | 效果 |
|--------|-------|------|------|------|
| `STYLE_WEIGHT` | `1e4` | float | **总体风格损失权重** | ↑ 风格更强但可能覆盖内容；↓ 风格弱但保留内容细节 |
| `CONTENT_WEIGHT` | `1.0` | float | 内容损失权重 | ↑ 更保留原图结构；↓ 允许更多形变以适应风格 |
| `TV_WEIGHT` | `1e-6` | float | Total Variation 平滑损失 | ↑ 平滑输出、消除噪点/刷痕；↓ 保留细节纹理但可能有噪点 |
| `COLOR_WEIGHT` | `1e-2` | float | 颜色守恒损失（均值/方差匹配） | ↑ 减少局部高亮/偏色；↓ 允许更自由的颜色变化 |

**调参建议**：
- 出现**局部高亮或亮斑**：提升 `COLOR_WEIGHT`（→ 1e-1）或降低 `STYLE_WEIGHT`（→ 1e3）或提升 `TV_WEIGHT`（→ 1e-5）
- 风格不够明显：提升 `STYLE_WEIGHT`（→ 1e5）或降低 `CONTENT_WEIGHT`（→ 0.5）
- 噪点过多（刷痕明显）：提升 `TV_WEIGHT`（→ 1e-5）
- 颜色偏离内容图太多：提升 `COLOR_WEIGHT`（→ 1e-1）

---

### 多尺度风格参数

| 超参数 | 默认值 | 类型 | 说明 | 效果 |
|--------|-------|------|------|------|
| `STYLE_SCALES` | `[256, 512]` | list | 风格图计算时的多个尺寸（像素） | 低尺度捕获大色块；高尺度捕获细节纹理 |
| `STYLE_SCALE_WEIGHTS` | `[1.0, 0.5]` | list | 每个尺度的相对权重（会被归一化） | 提高高尺度权重→更多细节；提高低尺度→更多色块 |
| `STYLE_LAYER_WEIGHTS` | `[0.5, 0.3, 0.1, 0.1]` | list | VGG 四层的相对权重（对应 relu1_2, relu2_2, relu3_3, relu4_3；会被归一化） | 浅层↑→细节纹理↑；深层↑→大色块结构↑ |

**调参建议**：
- 想要**更多细节纹理**：
  - 增大 `STYLE_SCALES` 中高尺度（512）的权重，例如 `[0.6, 1.0]`
  - 增大第一个元素 `STYLE_LAYER_WEIGHTS[0]`，例如 `[0.6, 0.2, 0.1, 0.1]`
- 想要**更多大色块**（风格化但不细碎）：
  - 增大 `STYLE_SCALES` 中低尺度（256）的权重，例如 `[1.0, 0.2]`
  - 增大深层权重 `STYLE_LAYER_WEIGHTS[-1]`，例如 `[0.3, 0.3, 0.2, 0.2]`
- **降低亮斑/高频噪点**：
  - 降低高尺度权重，例如 `[0.8, 0.2]`
  - 降低浅层权重，例如 `[0.3, 0.4, 0.2, 0.1]`

---

---

## 训练命令

---

### 指定风格图训练（从 styles 目录）

```powershell
python realtimeNST.py train style1
```

风格图会从 `styles/` 目录查找：
- 支持自动补全 `.png` 后缀：`style1` → `styles/style1.png`
- 也支持传完整文件名：`python realtimeNST.py train style1.jpg`

**输出**：
- 检查点：`checkpoints/model_style1_epoch_<N>.pth`
- 最终模型：`realtime_style-style1.pth`（同时覆盖 `realtime_style.pth`）
- **但是没什么用，建议从checkpoint里选一个好的模型，把pth文件保存为styleNum.pth的形式，保存到根目录**
- 测试图：`checkpoints/test_epoch_<N>.png`（如存在 `testcontent.png`）
-  **注意这里的测试图默认是根目录下的testcontent.png，也就是face1**


---

## 测试/推理命令


###  指定模型文件和内容图

```powershell
python realtimeNST.py test style1-1 clock
```

- 模型：自动查找 `style1-1.pth`（支持省略 `.pth` 后缀）
  - 查找顺序：根目录 → 根目录+`.pth` → `checkpoints/` → `checkpoints/`+`.pth`
- 内容图：`testcontents/testcontent-clock.png`
- **注意这里也可以用图片和模型的全称，但位置一定得放对**
- **补充一句这里的命名规则，styleX-Y中X代表风格图编号，Y代表该风格图的第Y个参数组合下的模型**

**输出**：`output_stylized-style1-1-clock.png`


## 文件结构

```
realtimeNST/
├── realtimeNST.py              # 主程序
├── train_and_test_guide.md     # 本文件
├── testcontent.png             # 默认测试内容图,也就是face1
├── styles/                     # 风格图目录
│   ├── style1.png
│   ├── style2.png
│   └── ...
├── testcontents/               # 测试内容图目录
│   ├── testcontent-clock.png
│   ├── testcontent-face.png
│   └── ...
├── coco/                       # COCO 数据集
│   └── val2017/
├── checkpoints/                # 训练检查点，处理一下选出有用的模型，剩下的模型删掉
│   ├── model_style1_epoch_1.pth
│   ├── model_style1_epoch_2.pth
│   ├── test_epoch_1.png
│   └── ...
└── output_stylized-*.png       # 当前推理输出
```
