# RaDe-GS → SAM-3D Scene Pipeline

把这条流程整理成一个**可从零配置环境**的轻量编排仓库：

1. **视频 → 抽帧 / COLMAP 数据集**
2. **RaDe-GS 训练出场景**
3. **从 RaDe-GS 渲染单帧深度 prior / pointmap**
4. **把单帧实例标签图 + pointmap 送进 SAM-3D-Objects 重建每个物体**
5. **把这些物体 mesh 放回场景，导出总 scene mesh**

这个仓库本身不 vendoring 大型上游仓库，而是：

- 自动 clone `RaDe-GS`
- 自动 clone `sam-3d-objects`
- 自动创建两个环境
- 自动写出 `configs/local.env`

## 先说清楚：哪些是“从零”，哪些不是

这个仓库解决的是**环境配置 + 阶段编排**，不是替你发明缺失的算法步骤。

仍然有两个前置条件必须由你提供：

1. **RaDe-GS 数据集目录**  
   你需要把视频抽成图像，并跑好 COLMAP，形成至少：

   ```text
   scene/
     images/
     sparse/0/
   ```

2. **SAM-3D 的单帧实例标签图**  
   SAM-3D-Objects 不会自己“找出图里所有物体”，它需要你提供一个标签图：
   - 每个像素值 = `object_id`
   - `0 = background`
   - 分辨率与该帧 RGB 一致

   我们实践中用的是 **FlashSplat** 风格输出，例如：

   ```text
   objects_flashsplat_run2/00100.png
   ```

   如果你有别的实例分割 / 跟踪系统，只要能导出同格式标签图，也可以接入。

---

## 1. 系统要求

至少需要：

- Linux x86_64
- NVIDIA GPU
- CUDA toolkit / `nvcc`
- `git`
- `conda` 或 `mamba`
- `ffmpeg`
- `colmap`

### CUDA 版本要求

bootstrap 默认使用：

- `TORCH_CUDA_TAG=cu121`
- `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121`

也就是说，默认要求机器上的 `nvcc` 是 **12.1**。  
如果你的机器是别的 CUDA toolkit 版本，请在 bootstrap 时显式改掉：

```bash
./scripts/bootstrap_from_scratch.sh /data/workspace \
  --torch-cuda-tag cu128 \
  --torch-index-url https://download.pytorch.org/whl/cu128
```

脚本会在真正安装前检查 `nvcc` 版本，不匹配会直接报错，而不是等到编译扩展时再炸。

---

## 2. 从零开始配置

先 clone 这个编排仓库：

```bash
git clone <this-repo> radegs-sam3d-scene-pipeline
cd radegs-sam3d-scene-pipeline
```

然后一键 bootstrap：

```bash
./scripts/bootstrap_from_scratch.sh /data/workspace
```

它会做这些事：

1. clone `RaDe-GS`
2. clone `sam-3d-objects`
3. 创建 `radegs` 环境
4. 创建 `sam3d-objects` 环境
5. 安装 RaDe-GS 依赖和编译扩展
6. 安装 SAM-3D-Objects 依赖
7. 自动应用本地已知的 SAM-3D checkout 补丁
8. 做两个环境的 smoke test
9. 生成 `configs/local.env`

生成的 `configs/local.env` 大概像这样：

```bash
RADEGS_REPO=/data/workspace/RaDe-GS
RADEGS_ENV_NAME=radegs
RADEGS_PYTHON=/path/to/envs/radegs/bin/python
SAM3D_REPO=/data/workspace/sam-3d-objects
SAM3D_ENV_NAME=sam3d-objects
SAM3D_PYTHON=/path/to/envs/sam3d-objects/bin/python
SAM3D_CONFIG=/data/workspace/sam-3d-objects/checkpoints/hf/pipeline.yaml
FFMPEG_BIN=ffmpeg
COLMAP_BIN=colmap
```

> `configs/local.env` 已经被 `.gitignore` 忽略，不会把你的本机路径误提交进去。

---

## 3. 下载 SAM-3D checkpoints

SAM-3D-Objects checkpoint 是 **Hugging Face gated** 的。  
你需要先拿到访问权限，并且在本机完成 `hf auth login`。

然后运行：

```bash
./scripts/bootstrap_checkpoints.sh configs/local.env hf
```

这会把 checkpoint 下载到：

```text
<SAM3D_REPO>/checkpoints/hf/
```

并且把 `SAM3D_CONFIG` 写回 `configs/local.env`。

---

## 4. 当前仓库里的核心脚本

- `scripts/bootstrap_from_scratch.sh`  
  从零 clone 上游仓库、创建环境、安装依赖、生成 `configs/local.env`
- `scripts/bootstrap_checkpoints.sh`  
  下载 SAM-3D checkpoints
- `scripts/patch_sam3d_checkout.py`  
  修补已知的上游调试残留
- `scripts/extract_video_frames.sh`  
  视频抽帧
- `scripts/run_radegs_train.sh`  
  训练 RaDe-GS
- `scripts/run_radegs_tsdf.sh`  
  从 RaDe-GS 输出提取整场景 TSDF mesh
- `scripts/render_radegs_depth_prior.py`  
  渲染单帧深度和 pointmap
- `scripts/split_instance_label_map.py`  
  把 `pixel=object_id` 标签图拆成二值 mask
- `scripts/run_sam3dobjects_with_pointmap.py`  
  批量跑 SAM-3D-Objects
- `scripts/assemble_sam3d_scene.py`  
  合并物体 mesh 成总场景
- `scripts/run_frame_pipeline.sh`  
  从已训练好的 RaDe-GS 模型出发，一次跑完 depth prior → SAM3D → scene assembly

---

## 5. 数据准备

### 5.1 视频抽帧

```bash
./scripts/extract_video_frames.sh \
  /path/to/video.mp4 \
  /path/to/scene/images \
  3
```

### 5.2 跑 COLMAP

你需要把 `images/` 跑成 COLMAP sparse reconstruction，并整理成：

```text
scene/
  images/
  sparse/0/
```

这一步没有被这个仓库“假装自动化”。

---

## 6. 训练 RaDe-GS

现在训练和 TSDF 入口都支持直接读取 `configs/local.env`。

```bash
./scripts/run_radegs_train.sh \
  --env-file configs/local.env \
  /path/to/scene \
  /path/to/radegs_output/my_scene \
  -r 2 --use_decoupled_appearance 3
```

如果你不想用 env file，也仍然可以手动传绝对路径。

---

## 7. 提取整场景 TSDF mesh

默认推荐 `scalable`：

```bash
./scripts/run_radegs_tsdf.sh \
  --env-file configs/local.env \
  scalable \
  /path/to/radegs_output/my_scene \
  recon_tsdf \
  --voxel_size 0.012 \
  --far_depth_threshold 5.5
```

另一个版本：

- `stream`：更适合大场景 / 内存受限时试验

---

## 8. 单帧物体重建并合成场景

假设你已经有：

- 已训练好的 RaDe-GS 模型目录
- 某一帧 RGB，例如 `images/00100.png`
- 该帧实例标签图，例如 `objects_flashsplat_run2/00100.png`

直接运行：

```bash
./scripts/run_frame_pipeline.sh \
  configs/local.env \
  /path/to/radegs_output/my_scene \
  /path/to/scene/images/00100.png \
  00100 \
  /path/to/scene/objects_flashsplat_run2/00100.png \
  /path/to/output/frame00100 \
  --min-area 4000
```

它会产出：

- `depth_prior/00100_depth.npy`
- `depth_prior/00100_pointmap.npy`
- `depth_prior/00100_camera.json`
- `sam3d_stage1/`
- `sam3d_mesh/`
- `00100_sam3d_mesh_scene.ply`
- `00100_sam3d_mesh_scene.glb`
- `00100_sam3d_mesh_scene_summary.json`

如果 SAM-3D full decode 没有真的产出 `mesh.ply`，脚本现在会直接报错退出，不再假成功。

---

## 9. 坐标系说明

`assemble_sam3d_scene.py` 默认输出的是：

- **anchor frame 相机坐标系**下的 scene mesh

`render_radegs_depth_prior.py` 会额外保存：

- `w2c`
- `c2w`

所以如果你想导出到该帧对应世界坐标系，可以手动运行：

```bash
python scripts/assemble_sam3d_scene.py \
  --mesh-root /path/to/sam3d_mesh \
  --stage1-root /path/to/sam3d_stage1 \
  --camera-json /path/to/00100_camera.json \
  --output-prefix /path/to/frame00100_scene_world \
  --world-space
```

---

## 10. Mesh 规模与简化

实际跑过的 131 个实例合并后，原始 scene mesh 大约会到：

- 3240 万顶点
- 6480 万面

如果你要给别的系统消费，建议加简化：

```bash
python scripts/assemble_sam3d_scene.py \
  ... \
  --target-faces 2000000
```

---

## 11. 已知上游问题与仓库处理方式

### SAM-3D checkout 里的调试残留

我们在某个上游 checkout 里遇到过：

- `sam3d_objects/pipeline/inference_pipeline.py`

里面残留了调试 `print(...)` 和 `exit()`，会导致 full decode 提前退出。  
`bootstrap_from_scratch.sh` 现在会自动运行：

```bash
python scripts/patch_sam3d_checkout.py --sam3d-repo <sam3d_repo>
```

它只会在发现**完全匹配的已知调试片段**时才改文件，不会盲改别的逻辑。

### SAM-3D config 路径

`run_frame_pipeline.sh` 现在会优先使用 `configs/local.env` 里的：

```bash
SAM3D_CONFIG=...
```

这样不再依赖“必须先 `cd` 到 sam-3d-objects 仓库根目录”这种隐式前提。
