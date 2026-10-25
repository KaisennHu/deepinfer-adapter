# 配置与运行边界

## 两级配置

`PD_PROFILE=b300-1p1d` 加载固定实测配置，会覆盖下表中的模型运行参数。`PD_PROFILE=none` 不加载预设，可自行设置变量，但需要重新验证正确性和性能。

| 项目 | B300 预设 |
|---|---|
| 并行 | P、D 各 DP4/EP4；TP1、CP1 |
| 引擎 | P/D 均 NORMAL，保留 ZOS |
| 上下文 / batch | 262144 / 每 rank `max_num_seqs=32` |
| prefill token budget | 32768 |
| MTP | P/D 同为 7；varlen=0 |
| 采样 | greedy；`DEEPINFER_SAMPLER_FORCE_TOPK=1` |
| Batch invariance | 0；不保证改变 batch 后逐 token 相同 |
| P 的 YOCO | compact，keep=140，extra state=1 |
| 前缀快照 | 每 P rank 1 GiB CPU 内存；D 不保存 |
| 原生块哈希缓存 | 关闭 |
| P rank 选择 | 冷请求轮询；命中请求返回持有快照的 rank |
| D rank 选择 | 跟随 source rank，满足当前 IPC 池约束 |
| IPC | 每 P rank 两槽；每槽最多 2048 blocks（由上下文上限推导） |
| Decode 优化 | 计时关闭、singleton metadata、parallel argmax、greedy reduction 融合 |
| EPLB | extra experts=0，重排 interval=0，统计 window=3 |

可以独立设置的部署参数：

| 变量 | 默认值 / 用途 |
|---|---|
| `MODEL_PATH` | 必填，权重目录 |
| `DEEPINFER_REPO` | 必填，已打补丁并构建的 checkout |
| `ROUTER_BIN` | 启动 router 时必填 |
| `PD_ROLE` | `prefill` / `decode` |
| `PD_PORT` | P 18300 / D 18400 |
| `DEVICE_OFFSET` | P 0 / D 4；连续使用 `DP_SIZE` 张卡 |
| `MASTER_PORT` | P 24333 / D 24433；引擎还使用该端口减 50 |
| `PD_HOST` / `ROUTER_HOST` | `127.0.0.1` |
| `ROUTER_PORT` | 13010 |
| `PREFILL_URL` / `DECODE_URL` | 本机对应的 HTTP worker |
| `HANDOFF_DIR` | 默认 `/dev/shm/di-pd`；推荐每次新建独占目录 |
| `PD_CUDA_IPC_POOL_ID` | IPC 开启时必填；同一次运行 P/D 相同，重启使用新值 |
| `HANDOFF_TIMEOUT` | 180 秒 |
| `VENV_PATH` | 可选，启动前激活的环境 |
| `ENABLE_NUMA_SHIM` | 普通模式默认 0；B300 预设为 1 |

NUMA shim 会绕过 NUMA 绑定，适配本次机器。其他硬件应检查拓扑，不能直接认定该设置最优。服务和 handoff 目录面向可信的本机使用；本版没有公网鉴权、租户隔离或配额管理。

## 前缀快照和 3FS

`PrefixSnapshots` 保存 token 前缀对应的 KV 块和必要 SWA 状态。查询按完整 token 字节精确匹配，选最长的已存前缀；不会仅凭 hash 或文本相似度命中。它不是可在任何位置切分的通用 radix cache。

YOCO 模式的快照锚点为 `max(0, floor((L - 140) / 128) * 128)`。保存 producer 状态，恢复时由后缀重放重建 consumer 状态。原始输入的首个采样输出不进入本次前缀 key；D 后续生成的 KV 也不会回写本地快照。

快照为进程内 CPU 内存对象，按字节限制淘汰，重启丢失。相同会话更新时优先保留较长的前缀，删除旧祖先可能降低分叉命中率。命中后仍有 H2D 拷贝开销。

此机制不依赖 3FS，当前只在单机验证。多实例部署还需要实例级缓存路由；跨机共享需要目录与传输，不能靠部署多个本地 `PrefixSnapshots` 自动实现。官方架构把块 KV 放在 3FS、SWA 保留在主机内存，并配有缓存管理服务；这些并非本仓库已实现的功能。

## IPC 和 CPU 回退

P 发布 GPU pool，D 映射并读取，完成同步后按 slot generation 确认消费，随后 P 才能复用槽位。当前固定两个槽，P/D 同 rank 对应。P、D 模型、DP、上下文、MTP 和状态布局必须兼容。

池满或超过容量会回退到 `HANDOFF_DIR` 下的 CPU 序列化文件。本轮完整 c32 每轮出现 17/734 次回退，所有这些请求均计入结果。实现仍会为快照/回退准备 CPU 副本，不能称作全链路零拷贝。

P→D 使用 `TO_CE=True` 的完整状态，包括 consumer SWA 和 compression state；前缀快照使用对齐位置的 `TO_CE=False` 状态。两种状态用途不同。

CUDA IPC 是同机通路。单 worker 热重启、跨机 IPC、服务重启续接旧 pool 均不支持。完整停止后重新启动，并更换 pool ID。

## HTTP 接口范围

| 接口 | 支持范围 |
|---|---|
| `GET /health` | 所有 rank 真实就绪且存活时 200，否则 503 |
| `GET /v1/models` | 返回模型元数据 |
| `POST /v1/completions` | 一条 token ID 数组 prompt；普通/SSE 响应 |
| `POST /reset_prefix_cache` | 空闲时重置；在途请求时 409，未就绪时 503 |

已实现的主要请求字段为 `model`、`prompt`、`max_tokens`、`temperature`、`seed`、`ignore_eos`、`stream`。`temperature=0` 会转换为原生 `top_k=1` 的真正 greedy；省略 temperature 同样使用 greedy。性能预设拒绝非零 temperature。

`kv_transfer_params` 由 router 的 PD 流程传递，普通客户端不应构造该内部字段。P 先完成 prefill 并交接状态，D 接管并输出剩余 token；客户端应访问 router，而不是单独拿 P 的一个 token 当完整响应。

本版不是完整 OpenAI API 实现：chat、工具调用、结构化输出、logprobs、多个候选、stop 字符串和其他采样控制未完整接入/校验，请勿依赖这些字段生效。请求取消、恶意输入防护和生产级容错也不在当前验证范围。聊天应用需要额外的模板编码层。

## 常见问题

- **缺少 `block_table` / `pop_processed_seqs` / `adopt_running_seq`**：引擎补丁未应用，或 Python 加载的是旧原生库。重建后检查 `deepinfer.__file__` 与 `deepinfer._C.__file__`。
- **`pidfd_getfd` / ptrace 权限错误**：本地 worker 已调用进程级 `PR_SET_PTRACER_ANY` 兼容逻辑；容器还可能限制对应系统调用。不要通过修改其他作业的系统级设置来处理。
- **D 等不到 pool**：先等待 P 的 `/health`；确认两边 `HANDOFF_DIR`、pool ID 一致，且本次没有使用旧目录。
- **无 RDMA 时 Engram 失败**：确认 DeepEP LSA 补丁已编进 wheel，`EP_DISABLE_GIN=1` 不能单独替代补丁。
- **启动后端口存在但 `/health` 是 503**：查看各 rank 的初始化或退出日志；不要将健康检查改成固定 200。
- **GPU 不足或被其他任务占用**：先确认 0–7 卡可用。本次脚本不驱逐其他任务。
- **数据不复现或缓存命中低**：核对完整 token ID、模板、路由、缓存重置时间及模型版本；改变 BI、MTP、YOCO 或采样后必须重新验证。
- **更换 checkpoint**：停止服务并清空本次缓存。当前 cache key 的模型隔离来自独立服务进程，不支持同一缓存对象热切模型。
