# NumPy 实现 ViT-B/32 图像特征提取

## 功能

用**纯 NumPy** 补全课程提供的 `image_encoder.py`，实现 OpenCLIP ViT-B/32 图像塔的前向传播：输入一张图片（预处理后为 `[B,3,224,224]`），输出 **512 维、L2 归一化**的图像特征向量 `[B,512]`。不使用 PyTorch / TensorFlow 等深度学习框架，全部矩阵乘、广播、多头注意力、LayerNorm、GELU 均由 NumPy 实现或由课程提供的 `helpers.py` 完成。

## 文件结构

| 文件 / 目录 | 说明 |
|---|---|
| `image_encoder.py` | ★ **本人补全**。老师提供的 starter，仅把 10 处 `raise NotImplementedError` 换成真实计算，其余注释、docstring、import、类结构未改动 |
| `helpers.py` | 老师提供：`LayerNorm`、`GELU`（精确 erf 版）、`attention_weights`、`causal_mask`、`unit_vector` |
| `preprocess_image.py` | 老师提供：短边 bicubic 缩放 + 中心裁剪 + OpenCLIP mean/std 归一化。**本机系统 Pillow 较旧，将 `Image.Resampling.BICUBIC` 等价替换为 `Image.BICUBIC`**（两者取值均为 3，缩放结果逐像素一致） |
| `runtime.py` | 老师提供：`load_weights` / `load_images` / `image_records` |
| `check.py` | 老师提供：8 阶段数值校验（assets → preprocess → patches → tokens → block0 → last_tokens → output → batch） |
| `run.py` | 老师提供：命令行入口，输出 `output_tokens.npz` |
| `config.json` | 模型与预处理配置（224 / patch 32 / width 768 / heads 12 / layers 12 / 50 tokens / output 512 / GELU_erf / eps 1e-5） |
| `requirements.txt` | 依赖清单 |
| `weights/` | `image_encoder.npz`（152 个张量）+ `manifest.json`（SHA-256 清单） |
| `assets/` | 10 张测试图、`images.json`、`manifest.json`、`reference_outputs.npz`（各阶段参考数组） |
| `lecture-demos/`、`practice/`、`PPT-CODE-MAP.txt`、`SOURCE-MANIFEST.json` | 课程配套示例与练习，不参与校验 |

## 本人补全的 10 处（6 组 TODO）

- **TODO-01 `patchify`**：`reshape(B,3,7,32,7,32)` → `transpose(0,2,4,1,3,5)` → 摊成 `[B,49,3072]`，保证「先左右、后上下」的 patch 顺序与 C,H,W 展平顺序。
- **TODO-02 `split_heads`**：`[B,N,768]` → `reshape(B,N,12,64).transpose(0,2,1,3)` → `[B,12,N,64]`。
- **TODO-03a**：`q @ k.transpose(0,1,3,2) / sqrt(64)` 得 `[B,H,N,N]`；softmax 由现成的 `attention_weights` 完成。
- **TODO-03b**：`coefficients @ v` 得上下文，再 `transpose(0,2,1,3).reshape(B,N,768)` 拼回多头。
- **TODO-03c**：`merged @ out_weight.T + out_bias`（权重为 `[out,in]` 布局，须转置）。
- **TODO-04a**：`residual = x + update`（残差加的是**原始 x**，不是 norm 后的 x）。
- **TODO-04b**：`GELU(u @ W1.T + b1) @ W2.T + b2`，再与 residual 相加。
- **TODO-05a**：`patches @ patch_weight.T`，patch 投影**无 bias**。
- **TODO-05b**：CLS token 拼在 token 轴最前，再加 `[50,768]` 位置编码（广播到 batch）。
- **TODO-06**：取 `tokens[:,0,:]` → `post_norm` → `@ visual.proj`（该矩阵已是 `[in,out]`，**不转置**）→ `unit_vector` 归一化。

## 依赖

```
numpy
Pillow
```

安装：`python3 -m pip install -r requirements.txt`

## 数据获取方法（重要）

`weights/image_encoder.npz` 体积较大，`assets/reference_outputs.npz` 等为课程校验数据，二者**不纳入 Git 仓库**（见 `.gitignore`）。请从课程平台提供的项目压缩包中，把 `weights/` 与 `assets/` 两个文件夹**原样解压/拷贝**到本目录，不要重新保存、转码或用网盘另存。`check.py --stage assets` 会逐文件计算 SHA-256 与 manifest 比对，改一个字节即失败。

同理，`output_tokens.npz` 是运行产物，也不入库，本地保留即可。

## 运行

```bash
python3 check.py --stage all          # 8 阶段数值校验
python3 run.py --output output_tokens.npz   # 对 10 张图提取特征
```

## 校验结果（本机实测）

```
assets       PASS: 152 matching visual tensors and ten image/reference pairs
preprocess   PASS max_abs=0.000e+00
patches      PASS max_abs=0.000e+00
tokens       PASS max_abs=4.005e-05
block0       PASS max_abs=7.063e-06
last_tokens  PASS max_abs=5.054e-05
output       PASS max_abs=5.960e-07
batch        PASS max_abs=4.619e-07
Requested numerical checks passed.
```

输出 `Saved (10, 512) to output_tokens.npz`，特征均为有限 float32 且 L2 范数为 1。
