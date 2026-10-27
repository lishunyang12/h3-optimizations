# 5 秒数字人 / 15 秒 FastH3 跨 SKU 复现

公开入口：[测试数据与下载](https://lishunyang12.github.io/fasth3-lossy-ablation/sku-reproduction.html)。
本目录参照 Sana SOL Engine 的配置、启动、结果目录组织方式，复用的是 H3 自己的历史运行源码。

**这是历史数据与可执行复现包。新封装已通过 CPU、原始运行日志及源码重建检查；尚未在 5000 / 5500 上重新跑 GPU 测试。**
要做 apples-to-apples 比较，必须在两台机器运行同一版本、同一 case；页面历史耗时仅作定位参考。

| Case | GPU 数 | 工作负载 | 历史 HTTP 中位数 |
|---|---:|---|---:|
| `5s-avatar-ref2va-4step` | 8 | 960×544，Ref2VA Turbo 四步，124 帧窗口 / 5 帧重叠 | 3.589779 s，13 个正式续段 |
| `15s-4step-mxfp8-sage-nvfp4` | 8 | 1280×704，FastH3 四步，362 帧 | 13.304 s，3 次 |
| `15s-4step-output-fusion` | 8 | 同上，加输出融合 | 13.261 s，3 次，另一轮测试 |
| `5s-t2va-4step-all-mxfp8` | 8 | 1344×768，FastH3 四步，124 帧 | 4.680764 s，5 次 |
| `5s-t2va-4step-first-step-bf16` | 8 | 同上，第一步回退 BF16 | 5.220296 s，5 次 |

还有七个独立的 `5s-8step-*` FastH3-Omni 配置（4 GPU、1344×768、124 帧、8 步）。
它们是 10 月的新模型对照，不能当作 9 月上线数字人的 baseline。用 `list` 查看完整目录。

数字人保存了原始 16 次请求，种子 8300–8315、同一参考图上传两次、同一 continuation session；前 3 次是预热，后 13 次是正式样本。
首段输出 124 帧 / 5.166667 s，后续输出 119 帧 / 4.958333 s。历史应用 ready 中位数为 3.706288 s，包含额外检查，不能与 HTTP 时间混用。
[数字人画质页](https://lishunyang12.github.io/fasth3-lossy-ablation/avatar-quality.html) 是同一部署路线的独立 10 秒 / 243 帧质量对照；其中 6.993 s 不是 5 秒性能数据。

## 1. 环境与硬件

两个主工作负载均要求 **8 张同 SKU、SM120 的空闲 GPU**，相同的 TP1/SP8、文本编码器 TP8、VAE PP8，以及可用的 BF16 RDMA 通路。
程序自动记录 GPU 完整型号、UUID、显存、驱动、功耗、频率、CPU 与 PCIe 拓扑；“5500”不会被当作一个已确认的型号。
若架构、显存、GPU 数不符，会退出；不自动降分辨率、换精度或开启 offload。
15 秒历史单卡峰值约 61.6 GiB（按 nvidia-smi 的 MiB 单位记录约 63099 MiB）。
数字人保留 BF16 投影权重供实验路径使用，每 rank 约 38.5 GB pinned CPU 内存；预留至少 400 GiB 可用主机内存。

使用 Python 3.12、CUDA 13.2 对应环境，安装 `curl`、`ffmpeg`、`ffprobe`、`numactl`。原生四步环境的关键版本：

```text
torch==2.13.0+cu132
torchvision==0.28.0+cu132
vllm==0.28.0
transformers==5.14.1
diffusers==0.40.0
flashinfer-python==0.6.16.post3
nvidia-cutlass-dsl==4.6.2
comfy-kitchen==0.2.33
safetensors==0.8.0
huggingface-hub==1.29.0
```

这些是环境约束，不是通用于所有系统的 pip 安装命令。按对应发行渠道安装 CUDA/PyTorch/vLLM 的兼容构建；以仓库 `flashinfer/` 覆盖其 Python/CUDA 源码。
先执行 `python3 scripts/prepare_flashinfer.py` 下载固定版本的第三方头文件。
若 comfy-kitchen 通过原始独立目录提供，可在 assets JSON 增加 `kitchen` 路径，目录中必须带对应 distribution metadata。
启动器校验包版本并哈希实际源码及 `.so`；环境不一致会退出。
8-step Omni 使用另一个环境，精确版本见其 case 的 `dependencies`，不要混用四步环境。

## 2. 重建历史运行源码

```bash
git clone https://github.com/lishunyang12/h3-optimizations.git
cd h3-optimizations
python3 scripts/sku_benchmark.py list
python3 scripts/sku_benchmark.py prepare --variant avatar5 --output /opt/h3-runtime-avatar5
python3 scripts/sku_benchmark.py prepare --variant reference --output /opt/h3-runtime-15s
python3 scripts/sku_benchmark.py prepare --variant output-fusion --output /opt/h3-runtime-15s-output-fusion
# 可选：两组 768p / 5s 文生视频
python3 scripts/sku_benchmark.py prepare --variant t2va5 --output /opt/h3-runtime-t2va5
```

目录须不存在。`prepare` 验证 2026-09-23 发布的源文件 SHA-256，应用精确历史补丁/覆盖文件，并把 CAKE 的构建目录改为随包附带的相对目录。
各副本写入 `sku-source.json`。启动前会检查其 variant 和完整源码指纹。
每种历史实现保留独立副本，不能用当前集成目录 `vllm-omni/` 替代。

## 3. 权重与 AdaLN 缓存

在安装了 huggingface-hub 的环境执行以下命令。下载包含大型模型；复现 ZIP 本身没有权重。

```bash
python scripts/sku_models.py --case 5s-avatar-ref2va-4step --download --output /models/avatar-assets
python scripts/sku_models.py --case 15s-4step-mxfp8-sage-nvfp4 --download --output /models/fasth3-assets
```

已有权重时，用 `--base /path/to/MiniMax-H3 --adapter /path/to/adapter.safetensors` 替代 `--download`。
工具会验证 adapter SHA-256，并为数字人构建四步模型视图。查看输出的 `model-assets.json` 获取 model 和 adapter 路径。

数字人固定 MiniMax-H3 revision `73372e6cf53e414edd3ab03e357717fb0602e758`、LightX2V Turbo revision `3ec17a324ced54151364f24f8b5fb6bf7e26414f`。
15 秒固定 MiniMax-H3 revision `9bfb6693f2cf6de171db46d1aa586f67d773a1da`，FastVideo Preview LoRA revision `c0e8ec2ac460e801094a0abe74bfb57347388ce3`，adapter SHA-256 `42dc502a2078f166c396a1fa75f29728d1844363652d345d5ef3e2b444ed6470`。
参考图已在 `repro/inputs/`，运行时验证 SHA-256。

**在空闲 GPU 上生成一次 sidecar，然后复制同一个文件到两台机器。** 生成缓存属于准备工作，不计入请求耗时；不要在另一台 GPU 上独立生成后假设字节相同。

```bash
PYTHONPATH=/opt/h3-runtime-avatar5 python scripts/sku_adaln.py \
  --runtime /opt/h3-runtime-avatar5 \
  --transformer-path /models/avatar-assets/MiniMax-H3/Ref2VA/transformer \
  --output /models/guide-adaln.safetensors

PYTHONPATH=/opt/h3-runtime-15s python reference-runtime/tools/minimax_h3/build_adaln_cache.py \
  --transformer-path /models/fasth3-assets/MiniMax-H3/FL2VA/transformer \
  --model-variant fl2va --mode t2va \
  --fasth3-adapter /models/fasth3-assets/adapter/vsa-datafree/adapter_model.safetensors \
  --base-schedule 0.999 0.749 0.5 0.25 0.0 --flow-shift 12 --audio-flow-shift 3 \
  --output /models/fasth3-adaln.safetensors
```

数字人使用原始 34 个 guide timestep plans；Turbo adapter 不修改这些 AdaLN 权重。
FastH3 使用绑定 adapter 的 v3 sidecar，两种缓存不可互换。
模型、输入和缓存全部按实际字节哈希，跨卡比较检查相等。

## 4. 两台机器执行同一套测试

复制并编辑机器路径；工作负载配置不要修改：

```bash
cp configs/sku-assets.example.json /tmp/h3-assets.json
# 编辑 /tmp/h3-assets.json 的 runtime、python、model、adapter、adaln、cuda 路径。
# 查看完整启动命令，dry-run 不加载模型、不访问 GPU：
python3 scripts/sku_benchmark.py suite --cases main --assets /tmp/h3-assets.json \
  --gpus 0,1,2,3,4,5,6,7 --output /results/pro5000 --dry-run
# 三个主 case 依次运行；每个启动自己的服务并在结束后关闭它：
python3 scripts/sku_benchmark.py suite --cases main --assets /tmp/h3-assets.json \
  --gpus 0,1,2,3,4,5,6,7 --output /results/pro5000
```

在目标 SKU 上用相同命令，输出目录改为 `/results/pro5500`。
也可以单独执行：

```bash
python3 scripts/sku_benchmark.py run --case 5s-avatar-ref2va-4step \
  --assets /tmp/h3-assets.json --gpus 0,1,2,3,4,5,6,7 --output /results/avatar-one-run
python3 scripts/sku_benchmark.py compare \
  /results/pro5000/5s-avatar-ref2va-4step/result.json \
  /results/pro5500/5s-avatar-ref2va-4step/result.json --output /results/avatar-comparison.json
```

对两个 15 秒 case 同样运行 `compare`。`suite --cases case1,case2` 支持自选列表；4-GPU 和 8-GPU 配置分别运行。
8-step case 的 runtime 是 `vllm-project/vllm-omni` revision `c67204da85acd701a4b85c105843e287205800ec`，`fastvideo` 指向独立安装的 fastvideo-kernel 0.3.4 包根目录（包含 .so）。
其模型与 decoder 固定 revisions 在 `configs/sku-cases.json` 中；下载对应版本后填入 assets JSON。

## 5. 结果与边界

每个 case 输出 `result.json`、`sources.json`、`assets.json`、`environment.json`、`server.log`、逐请求 JSON/日志、MP4 和 `telemetry.jsonl`。
新脚本统一测量启动本地 curl 到完整 MP4 写完的时间，同时记录 curl 自身计时；媒体完整解码检查在计时外。
原始 HTTP 客户端计时细节不同，因此必须重新跑 5000，再与目标卡的新结果比较。

四步运行检查每 rank 的真实四次 DiT 调用、量化 fallback、完整 NVFP4 VAE、CAKE 与 RDMA 启动证据；输出融合额外核对融合调用计数。
数字人精确重放连续请求，不包含聊天模型、直播队列、RTMP、播放器延迟，也不复刻应用层质量筛选耗时。
同一个分辨率的输出长度、模型和精度都固定；量化路线不等于原始 BF16 画质。

脚本不停止其他 GPU 作业。预检发现占用即退出，运行中发现外部计算进程将结果标为失败。
它只清理自己启动的服务进程组。未锁定频率；CPU、PCIe/RDMA、驱动、功耗差异会保留在结果里，所以结果是同工作负载的系统性能比较，不能自动归因于 GPU 核心。

`compare` 拒绝不同配置、源码、权重、缓存、依赖版本、失败或受干扰的运行。
它单独报告 MP4 字节是否相同，不要求不同 GPU 的结果字节相同，也不把字节差异当作主观质量评分。

CPU 验证：`python3 -m unittest discover -s tests -p test_sku_benchmark.py -v`。
测试夹具为历史日志的量化/解码结构化记录，不含私有启动 manifest。
