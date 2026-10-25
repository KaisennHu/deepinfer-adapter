# DeepInfer Adapter

把 [DeepInfer](https://github.com/deepseek-ai/DeepInfer) 接到未修改的 [vLLM Router](https://github.com/vllm-project/router)，提供单机 **Prefill/Decode 分离（1P1D，P/D 各 DP4/EP4）**，保留原有 gRPC worker 模式。

本版整合 2026-10-25 实测代码：HTTP PD、完整 KV/state 接管、YOCO 前缀快照、本机 CUDA IPC 池、缓存感知 rank 路由、流式增量解码和启动对象 GC 冻结。依赖的 DeepInfer PD/采样优化补丁也在仓库内。

**已验证范围：Linux、8×B300、DeepSeek-V4.1-Flash-1017、同机 P/D。** 无需部署 3FS；无 RDMA 的机器需要附带的 DeepEP Engram LSA 补丁。跨机 PD、跨实例共享缓存、在线故障恢复不在本版验证范围内。

```text
客户端（token IDs） → vLLM Router :13010
                         ├─ P HTTP :18300 → GPU 0–3，DP4/EP4
                         │   └─ CPU 前缀快照 + YOCO compact
                         └─ D HTTP :18400 → GPU 4–7，DP4/EP4
                             ↑ 同机 CUDA IPC；池满时使用 CPU 文件回退
```

## 1. 准备环境和源码

需要 Python 3.12、GCC ≥13、CUDA 13，以及模型所需的显存和本地权重。DeepInfer 原生构建较重，请先确认 [上游构建要求](https://github.com/deepseek-ai/DeepInfer/tree/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9)。以下使用一个新的工作目录；路径请自行替换。

```bash
export ADAPTER_WORKSPACE=/path/to/router-deepinfer
mkdir -p "$ADAPTER_WORKSPACE"
git clone https://github.com/KaisennHu/deepinfer-adapter.git "$ADAPTER_WORKSPACE/deepinfer-adapter"
export ADAPTER_REPO="$ADAPTER_WORKSPACE/deepinfer-adapter"

python3.12 -m venv "$ADAPTER_WORKSPACE/venv"
source "$ADAPTER_WORKSPACE/venv/bin/activate"

git clone https://github.com/deepseek-ai/DeepInfer.git "$ADAPTER_WORKSPACE/DeepInfer"
export DEEPINFER_REPO="$ADAPTER_WORKSPACE/DeepInfer"
git -C "$DEEPINFER_REPO" switch -c codex-adapter-pd 9bd9cf243fc53ef9afbf8f04d391d989117d6ab9
git -C "$DEEPINFER_REPO" submodule update --init --recursive

bash "$ADAPTER_REPO/scripts/apply_deepinfer_patch.sh"
# 本教程的无 RDMA 配置需要此补丁；只应用源码，随后统一编译。
APPLY_ONLY=1 bash "$ADAPTER_REPO/scripts/apply_deepep_patch.sh"

export DEEPINFER_ACCELERATOR=nvidia
(cd "$DEEPINFER_REPO" && bash build.sh)
python -m pip install -e "$ADAPTER_REPO[pd]"
```

补丁固定针对 DeepInfer `9bd9cf2` 和 DeepEP `f6aa38f`。脚本会检查版本，重复执行不会重复打补丁。**只复制 Python 文件不够：PD 接口涉及 C++，必须重新构建 DeepInfer。** 已有构建环境如何升级见 [构建与补丁说明](docs/installation.md)。

构建 router，或使用同一提交构建好的 `vllm-router` 二进制：

```bash
git clone https://github.com/vllm-project/router.git "$ADAPTER_WORKSPACE/router"
git -C "$ADAPTER_WORKSPACE/router" switch -c codex-adapter-router 305c1d2f0fc0d3cc9bf59e8d802d27682249e68d
(cd "$ADAPTER_WORKSPACE/router" && cargo build --release)
export ROUTER_BIN="$ADAPTER_WORKSPACE/router/target/release/vllm-router"
```

## 2. 启动已验证的 B300 配置

以下命令在同一个 shell 中执行。此配置固定 **greedy、MTP7、YOCO140、IPC 双槽**；请求使用 `temperature: 0`。P/D 使用不同的 rendezvous 端口。GPU 编号由 `DEVICE_OFFSET` 控制，当前 worker 会自行设置 `CUDA_VISIBLE_DEVICES`，不要依赖外层该变量重新映射卡号。

```bash
cd "$ADAPTER_REPO"
export MODEL_PATH=/path/to/DeepSeek-V4.1-Flash-1017
export SERVED_MODEL=DeepSeek-V4.1-Flash-1017
export PD_PROFILE=b300-1p1d
# 每次完整启动生成新 ID；P、D 共享这个值和 handoff 目录。
export PD_CUDA_IPC_POOL_ID="$(python -c 'import uuid; print(uuid.uuid4().hex)')"
export HANDOFF_DIR="/dev/shm/di-pd-$PD_CUDA_IPC_POOL_ID"
export RUN_DIR="$ADAPTER_REPO/run/$PD_CUDA_IPC_POOL_ID"
mkdir -p "$RUN_DIR"
mkdir -m 700 "$HANDOFF_DIR"

PD_ROLE=prefill bash scripts/run_pd_http.sh >"$RUN_DIR/prefill.log" 2>&1 &
P_PID=$!
python scripts/wait_http.py http://127.0.0.1:18300/health --timeout 900

# P 就绪后再启动 D，避免 D 等待尚未发布的 IPC pool。
PD_ROLE=decode bash scripts/run_pd_http.sh >"$RUN_DIR/decode.log" 2>&1 &
D_PID=$!
python scripts/wait_http.py http://127.0.0.1:18400/health --timeout 900

bash scripts/run_pd_router.sh >"$RUN_DIR/router.log" 2>&1 &
ROUTER_PID=$!
python scripts/wait_http.py http://127.0.0.1:13010/health --timeout 60
```

如果任何就绪检查失败，先停止后续步骤并检查对应日志。健康状态来自各 rank 的真实初始化结果；权重加载、JIT 和图捕获时间不计入性能结果。

配置细节在 [configs/b300-1p1d.env](configs/b300-1p1d.env)。不使用该预设时，设置 `PD_PROFILE=none` 并显式指定参数；这不再代表已验证的性能配置。变量及边界见 [配置说明](docs/configuration.md)。

## 3. 通过 router 发请求

PD worker 接受 `/v1/completions` 的 **token ID 数组**，支持普通响应和 SSE 流式响应。下面使用模型的 `tokenizer.json` 编码原始文本，走 router 完成一次请求：

```bash
python examples/token_completion.py \
  --model-path "$MODEL_PATH" --model "$SERVED_MODEL" \
  --url http://127.0.0.1:13010/v1/completions \
  --text 'The capital of France is' --max-tokens 32
```

已有编码好的 prompt 时，可使用 `--tokens-file prompt-tokens.json`，文件内容例如 `[123, 456, 789]`，实际 token 必须来自当前模型。`--stream` 可检查流式输出。

这里的原始文本补全示例不包含聊天模板。`messages`、工具调用和 thinking 等格式必须先用兼容 DeepSeek-V4.1 的编码器生成 token IDs。本版没有直接实现 `/v1/chat/completions`，不要将普通文本编码当作官方聊天模板。HTTP 参数的已实现范围见 [配置说明](docs/configuration.md#http-接口范围)。

多轮测试需要清空缓存时，先确保没有在途请求，再分别请求 P 和 D：

```bash
curl --fail -X POST http://127.0.0.1:18300/reset_prefix_cache
curl --fail -X POST http://127.0.0.1:18400/reset_prefix_cache
```

## 4. 停止服务

停止入口，再停止 D，**D 完全退出后再停止 P**。IPC 的内存由 P 持有，不能只重启一个 rank。

```bash
kill -TERM "$ROUTER_PID"
wait "$ROUTER_PID" || true
kill -TERM "$D_PID"
wait "$D_PID" || true
kill -TERM "$P_PID"
wait "$P_PID" || true
```

检查相关子进程和 GPU 占用确实释放后，才能清理本次专属的 `HANDOFF_DIR`；重启时创建新目录和新 pool ID。不要清理其他作业的共享目录。排障见 [配置说明](docs/configuration.md#常见问题)。

## 5. CPU 测试与原有 gRPC 模式

测试使用 echo/mock 后端，不加载模型、不执行 GPU 推理。完整采样参数测试需要能导入 DeepInfer。

```bash
python -m pip install -e "$ADAPTER_REPO[pd,dev]"
cd "$ADAPTER_REPO"
bash scripts/gen_proto.sh
PYTHONPATH="$DEEPINFER_REPO${PYTHONPATH:+:$PYTHONPATH}" python -m pytest -q tests
```

没有 DeepInfer 的开发机可先运行 `python -m pytest -q tests --ignore=tests/test_pd_sampling.py`。生成 gRPC bindings 后，`ADAPTER_BACKEND=echo bash scripts/run_adapter.sh` 可启动无 GPU 的合成 worker；原有 gRPC 启动方式见 [安装说明](docs/installation.md#grpc-兼容模式)。

## 性能和正确性记录

本版来源于同机 8×B300、双方均经过 router 的对照，模型使用同一 FP4 checkpoint。三轮均值如下；结果只适用于对应固定多轮 workload，不能视为任何负载的收益承诺。

| 指标 | c8：相对 router+vLLM | c32：相对 router+vLLM |
|---|---:|---:|
| 输出吞吐 | **+34.0%** | **+35.0%** |
| 平均 TTFT | **−67.1%** | **−61.1%** |
| 平均 TPOT | **−13.5%** | **−13.3%** |
| 平均任务 JCT | **−26.4%** | **−27.4%** |

c8 每轮 124 请求，c32 每轮 734 请求；各 1 次排除预热、3 次完整测量。c8 第三轮 TPOT p95 仍慢 3.16%，不宣称所有场景的尾延迟稳定领先。官方 GSM8K 全量 1,319 题：DeepInfer **1280/1319（97.04%）**，vLLM **1281/1319（97.12%）**。测量口径、质量评测和版本见 [验证记录](docs/validation.md)。

## 仓库内容

| 路径 | 用途 |
|---|---|
| `deepinfer_adapter/pd_http.py` | P/D HTTP 接口、rank 调度和状态接管 |
| `deepinfer_adapter/prefix_snapshot.py` | 有容量限制的精确前缀快照 |
| `deepinfer_adapter/cuda_ipc_pool.py` | 本机 GPU IPC 池与消费确认 |
| `deepinfer_adapter/text_stream.py` | 增量 token 解码 |
| `deepinfer_adapter/server.py` | 原有 gRPC worker |
| `configs/`、`scripts/` | 实测预设、补丁应用、启动和就绪检查 |
| `patches/` | 固定版本的 DeepInfer、DeepEP 补丁 |
| `tests/` | 协议、状态恢复、路由、IPC 生命周期及启动脚本检查 |

代码采用 Apache-2.0；DeepInfer、DeepEP 和 router 保留各自许可证与上游归属。
