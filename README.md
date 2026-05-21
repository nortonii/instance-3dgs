# OpenLORIS Market 3DGS Experiments

OpenLORIS `market1-1` 的静态重建实验仓库，主要整理了：

- GT pose 子集数据构建脚本
- dynamic ignore mask 生成脚本
- packaged `gaussian-splatting` / `gaussian-splatting-mcmc` 对比脚本
- **官方原版 3DGS** 训练、导出、汇总脚本
- 轻量结果摘要与官方仓库兼容补丁

这个仓库**不包含原始数据、训练结果和大体量中间产物**；这些目录默认被 `.gitignore` 排除。

为了让仓库更完整，几条核心外部依赖已经作为 git submodule 放在 `third_party/`：

- `third_party/gaussian-splatting-official`
- `third_party/gaussian-splatting-pkg`
- `third_party/reduced-3dgs`
- `third_party/3dgs-mcmc-pkg`

## 当前结论

在 `market1-1` GT-pose 静态场景上：

- packaged vanilla 3DGS 明显偏弱
- packaged MCMC 明显优于 packaged vanilla
- **官方原版 3DGS** 表现最好，说明前面的 packaged vanilla 结果不能代表原始 3DGS

已保留的轻量结果摘要在 `artifacts/summaries/`：

- `market1_1_first50_gtpose_pkg_compare_maskignore_10k.json`
- `market1_1_first50_gtpose_official_vanilla_maskignore_10k.json`
- `market1_1_first100_gtpose_official_vanilla_maskignore_40k.json`

## 仓库结构

```text
scripts/
  build_openloris_subset_dataset.py
  prepare_openloris_dense_half_gt_dataset.py
  generate_openloris_ignore_masks_sam3.py
  run_first50_gtpose_pkg_compare.sh
  run_first50_gtpose_official_vanilla.sh
patches/
  gaussian-splatting-official-cu124-compat.patch
artifacts/summaries/
  *.json
third_party/
  gaussian-splatting-official/
  gaussian-splatting-pkg/
  reduced-3dgs/
  3dgs-mcmc-pkg/
```

## 依赖

Python 依赖见 `requirements.txt`。除此之外还需要：

1. COLMAP（如果要跑 COLMAP 相关脚本）
2. 一个可用的 7z 可执行文件，或者安装 `py7zr`
3. OpenLORIS 原始数据包
4. 初始化 submodule：`git submodule update --init --recursive`

## 拉取方式

```bash
git clone --recurse-submodules <your-repo-url>
```

如果已经 clone 了主仓库，再执行：

```bash
git submodule update --init --recursive
```

默认脚本会优先使用仓库内的 `third_party/gaussian-splatting-official`。

如果你想改成外部路径，也可以手动指定：

```bash
export OFFICIAL_REPO=/path/to/gaussian-splatting-official
```

## 官方 3DGS 兼容补丁

本机 `diff_gaussian_rasterization` 接口与官方仓库存在差异，所以需要先打补丁：

```bash
cd third_party/gaussian-splatting-official
git apply ../../patches/gaussian-splatting-official-cu124-compat.patch
```

这个补丁主要做两件事：

- 兼容 `GaussianRasterizationSettings` 是否包含 `antialiasing` / `projmatrix_raw`
- 兼容 rasterizer 返回值从 3 个张量扩展到 5 个张量的情况

## 常用流程

### 1. 从 fullspan 裁出 first100 子集

```bash
python scripts/build_openloris_subset_dataset.py \
  --src-dataset /path/to/dataset_market1_1_fullspan \
  --out-dataset /path/to/dataset_market1_1_first100_gtpose \
  --max-images 100 \
  --val-count 10
```

### 2. 跑 packaged vanilla vs MCMC

```bash
PYTHON_BIN=python \
SRC_DATASET=/path/to/dataset_market1_1_first50_gtpose_colmapba_notrunc_v2 \
bash scripts/run_first50_gtpose_pkg_compare.sh
```

### 3. 跑官方原版 3DGS

```bash
PYTHON_BIN=python \
SRC_DATASET=/path/to/dataset_market1_1_first100_gtpose \
RESULT_PREFIX=market1_1_first100_gtpose_official_vanilla \
RUN_TAG=maskignore \
ITERATIONS=40000 \
SAVE_ITERS="10000 20000 40000" \
RESULT_SUFFIX=long40k \
SUMMARY_SUFFIX=40k \
bash scripts/run_first50_gtpose_official_vanilla.sh
```

## 导出与评估约定

- **训练时**：mask 区域不参与监督
- **导出 render 时**：默认保存 **unmasked render**
- **算指标时**：仍然对 mask 区域做忽略，避免动态物体污染评估

对应 summary JSON 会显式标出：

- `export_masked: false`
- `metric_masked: true`

## 说明

- 当前仓库更偏向“实验工作区整理版”，不是一个通用 Python package。
- 如果要直接推 GitHub，建议提交脚本、补丁、README、`artifacts/summaries/` 和 `third_party` submodule 元信息，不要提交数据集和大体量结果目录。
