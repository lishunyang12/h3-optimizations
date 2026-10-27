# Docker reproduction for the two SM120 workloads

[PRO 5000 reference data](../benchmark/SM120/README.md) ·
[Intermediate stages](../benchmark/SM120/TRADEOFFS.md) ·
[Dockerfile](../docker/SM120/Dockerfile) · [Full protocol](CROSS_SKU_REPRO.md)

The image contains Python, the pinned core packages, CUDA 13.2 JIT tools,
FlashInfer headers and both reconstructed historical runtimes. Model weights,
AdaLN caches and results live in mounted directories. It runs exactly the
deployed 5s avatar and the selected 15s FastH3 reference.

## Host requirements

Use Linux x86-64, Docker, a compatible NVIDIA driver and
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Installing Docker alone does not provide GPU access. Check `nvidia-smi` on the
host and GPU access from a container before preparing model assets.

Both measured recipes require **8 idle, identical SM120 GPUs and an operational
RDMA path**. Expose `/dev/infiniband`. The conservative VRAM guards are 60000 MiB
for the avatar and 65147 MiB for FastH3; the avatar also requires 400 GiB available
host RAM. These are package guards, not established minimum requirements.

**Without an RDMA-capable NIC/path, these profiles cannot run as released.**
Their custom FlashInfer communicator requires RDMA and the launcher audits that
transport. Setting `NCCL_IB_DISABLE=1` does not disable the custom requirement.
A separately qualified NCCL/P2P configuration would be needed; this package
does not claim no-NIC support. The absence of Internet access is a different
issue: build the image and download the weights first, then transfer them.

## Build

```bash
git clone https://github.com/lishunyang12/h3-optimizations.git
cd h3-optimizations
docker build -f docker/SM120/Dockerfile -t h3-sm120:20261027 .
docker run --rm h3-sm120:20261027 check
```

The prebuilt image is distributed through the
[Docker image release](https://github.com/lishunyang12/h3-optimizations/releases/tag/sm120-docker-20261027).
Download its `download-image.sh`, then run `bash download-image.sh` to fetch and
verify every archive part and load the image. Parts remain below GitHub's
per-asset size limit. The script prints the expected Docker image ID and rejects
checksum mismatches. It needs `curl`, `sha256sum`, `gzip` and Docker.

`h3-sm120:20261027` is the tag loaded by that archive, not a registry address
for `docker pull`. Model weights are downloaded separately. The base is pinned by
digest; the core workload packages and separately installed CUDA JIT toolchain
are pinned by version. `/opt/h3-image-packages.txt` records the resolved complete
environment. Use the same built image on both machines for comparison; rebuilding
later can resolve different transitive packages.

## Download models and prepare caches

Choose an absolute host directory with enough space for both large checkpoints:

```bash
mkdir -p "$PWD/h3-models" "$PWD/h3-results"
docker run --rm \
  --mount type=bind,src="$PWD/h3-models",dst=/models \
  h3-sm120:20261027 download
```

This fetches the pinned upstream base partitions and adapters and verifies the
adapter hashes. The downloader refuses to overwrite an existing asset directory.
Weights remain subject to their upstream terms. For an interrupted download,
inspect the partial directory before choosing a new output or resuming with the
underlying Hugging Face downloader; the wrapper does not silently overwrite it.

Build the two AdaLN sidecars once on an idle GPU. This preparation loads model
weights and is outside benchmark timing:

```bash
docker run --rm --gpus '"device=0"' --ipc=host \
  --mount type=bind,src="$PWD/h3-models",dst=/models \
  h3-sm120:20261027 caches
```

Copy the complete model directory, including both generated sidecars, to the
other machine. Keep it mounted at `/models` on both machines: model views contain
absolute symlinks inside that mount. Do not build independent sidecars and assume
byte equality.

## Run and compare

```bash
docker run --rm --gpus all --pid=host --ipc=host --network=host \
  --device=/dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK \
  --mount type=bind,src="$PWD/h3-models",dst=/models,readonly \
  --mount type=bind,src="$PWD/h3-results",dst=/results \
  h3-sm120:20261027 run \
  --gpus 0,1,2,3,4,5,6,7 --output /results/pro5000
```

Use a new output directory for every run. On the target machine, change the output
to `/results/pro5500`, keeping the same image, assets and workload. Add `--dry-run`
to preview resolved commands without model execution. The launcher refuses GPUs
occupied by other jobs; it never stops another service to make space. Host PID
visibility lets its GPU ownership checks resolve the process IDs reported by
the NVIDIA driver.

Copy both result directories onto one machine and run the strict comparator:

```bash
docker run --rm \
  --mount type=bind,src="$PWD/h3-results",dst=/results \
  h3-sm120:20261027 exec python3 scripts/sku_benchmark.py compare \
  /results/pro5000/5s-avatar-ref2va-4step/result.json \
  /results/pro5500/5s-avatar-ref2va-4step/result.json \
  --output /results/avatar-comparison.json
# Repeat with the case directory 15s-4step-mxfp8-sage-nvfp4.
```

For an offline target, transfer the identical local image:

```bash
docker save h3-sm120:20261027 | gzip > h3-sm120-20261027.tar.gz
# On the target, after transferring this archive and h3-models:
gzip -dc h3-sm120-20261027.tar.gz | docker load
```

The container build and CPU checks do not establish GPU correctness, target
memory fit, or new performance. Fresh GPU runs of the packaged launcher remain
pending on both SKUs. The published PRO 5000 numbers retain their historical
measurement status.
