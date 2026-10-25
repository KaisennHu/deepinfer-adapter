# 安装、补丁与版本

## 固定版本

| 组件 | 已验证版本 |
|---|---|
| DeepInfer | `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9` + 本仓库补丁 |
| DeepEP | `f6aa38fe06cd4a0489e7730365d4e0114970a2f3` + 本仓库 Engram LSA 补丁 |
| vLLM Router | `305c1d2f0fc0d3cc9bf59e8d802d27682249e68d`，无源码修改 |
| 模型 | `deepseek-ai/DeepSeek-V4.1-Flash-1017`，同一 FP4 checkpoint |
| 实测 Python / PyTorch | Python 3.12 / `2.13.0+cu132` |
| 实测 HTTP 库 | FastAPI 0.128.0、uvicorn 0.40.0、httpx 0.28.1、tokenizers 0.21.4 |

上游 `main` 会变化，本补丁不保证适用于其他提交。请使用独立 checkout，避免覆盖正在运行的引擎源码。构建入口和 prerequisites 以固定版本的 DeepInfer README 为准。

## DeepInfer 补丁包含什么

`patches/deepinfer-pd-optimized.patch` 是**从干净的上述 DeepInfer 提交出发**的完整源码补丁，覆盖 14 个文件：

- C++ 调度器/绑定：取出 prefill 已处理请求、暴露 block table、接管已恢复 KV/state 的 decode sequence。
- 本地块哈希缓存实验接口：为保留已验证源码而包含，实测预设始终 `DEEPINFER_PREFIX_CACHE=0`；实际前缀复用由 adapter 快照承担。
- NUMA 边界处理，以及采样分派、greedy argmax、元数据/归约融合、可选计时开关。
- YOCO compact 和 `GPUCacheInterface` 的状态存取是该上游版本已有的能力；adapter 接入这些能力。

C++ 需要重新编译。仅设置 `PYTHONPATH` 无法为旧 `.so` 增加 `block_table`、`pop_processed_seqs` 或 `adopt_running_seq`。

从已有的正确版本、已构建环境升级：

```bash
bash "$ADAPTER_REPO/scripts/apply_deepinfer_patch.sh"
(cd "$DEEPINFER_REPO" && DEEPINFER_ACCELERATOR=nvidia bash scripts/build.sh build)
```

如依赖版本不同或尚未构建，使用 README 中完整的 `build.sh` 流程。不要把旧实验目录里的增量补丁叠加到本补丁上。

## 无 RDMA 的 DeepEP Engram 路径

`EP_DISABLE_GIN=1` 本身不足以完成 Engram fetch。`deepep-engram-lsa-fetch.patch` 为同一 NVLink 域、`num_rdma_peers == 1` 增加 LSA 读取路径。它不提供跨机 RDMA 的替代实现。

首次构建前先应用源码，然后由 DeepInfer 一次构建 vendor：

```bash
APPLY_ONLY=1 bash "$ADAPTER_REPO/scripts/apply_deepep_patch.sh"
(cd "$DEEPINFER_REPO" && DEEPINFER_ACCELERATOR=nvidia bash build.sh)
```

已有环境仅重编 DeepEP 时：

```bash
bash "$ADAPTER_REPO/scripts/apply_deepep_patch.sh"
```

该命令会构建并重新安装 DeepEP wheel，必须在正确的 Python/CUDA 环境中运行。源码与安装的 wheel/JIT header 必须一致。

## gRPC 兼容模式

gRPC 服务和 HTTP PD 服务是两条独立入口。本文的性能结果来自 HTTP PD。

```bash
cd "$ADAPTER_REPO"
python -m pip install -e '.[pd,dev]'
bash scripts/gen_proto.sh
# 无 GPU 的协议验证：合成 token，不代表推理性能。
ADAPTER_BACKEND=echo bash scripts/run_adapter.sh
```

默认 echo 监听 `127.0.0.1:50051`。真实 gRPC worker 用：

```bash
ADAPTER_BACKEND=deepinfer DP_SIZE=8 DEVICE_OFFSET=0 \
  bash scripts/run_adapter.sh
```

该模式每个 rank 一个端口。用 `scripts/run_router.sh` 接入，例如：

```bash
WORKER_URLS='grpc://127.0.0.1:50051,grpc://127.0.0.1:50052' \
  bash scripts/run_router.sh
```

`WORKER_URLS` 必须列出实际已启动的所有 worker。这个脚本保留了远端原有的多 URL 参数修复，不要把整个逗号串当一个 URL 传给 router。

`proto/` 中保留协议定义；`deepinfer_adapter/proto_gen/` 为生成目录，不提交到 Git。首次使用 gRPC 或构建包含 gRPC 的 wheel 前执行 `gen_proto.sh`。生成器和运行时版本需要兼容：

```bash
python -m pip install "grpcio-tools==$(python -c 'import grpc; print(grpc.__version__)')"
bash scripts/gen_proto.sh
python -m pip wheel --no-deps . --wheel-dir dist
```

本项目的主要安装方式为源码 checkout + editable install，启动脚本、配置与引擎补丁均在源码仓库中。
