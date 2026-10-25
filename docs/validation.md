# 验证记录（2026-10-25）

[完整三轮统计与比例（JSON）](benchmark-summary.json) 包含 c8/c32 的 mean、p50、p95、p99、逐轮数值和对比比例，不包含请求内容。

## 发布代码来源

adapter 来自完整 c32 / GSM8K 对照使用的冻结快照，DeepInfer 补丁由同次运行的原生源码差异和冻结 Python 源码合成。

[validated-source.json](validated-source.json) 记录原始 adapter 24 个 Python 文件的 SHA256、引擎版本、原生库 SHA256、补丁和 14 个修改后引擎文件的 SHA256。发布整理只更新包版本说明，新增通用路径的启动脚本/配置/文档；计算与传输实现沿用冻结版本。C++ 文件的 CRLF 已规范为 LF。

发布阶段检查包括 CPU 协议/缓存/IPC 生命周期测试、启动脚本参数检查、干净上游的补丁应用、打包和源码比对。没有把这些 CPU 检查称为新一轮 GPU Benchmark，也未声称在全新环境中重新完成了全部性能测试。

## 性能 Benchmark

这是固定多轮请求 trace 的端到端重放，不是 DeepInfer 官方离线 Pareto sweep，也不是 GSM8K 的计时结果。测量路径为 `client → vLLM Router → P/D`，双方都经过同一 router。

| 项目 | c8 | c32 |
|---|---:|---:|
| 并发任务数 | 8 | 32 |
| 每轮请求数 | 124 | 734 |
| 每轮输入 token | 5,365,904 | 48,895,498 |
| 每轮输出 token | 110,107 | 607,434 |
| 预热 / 正式测量 | 1 / 3 | 1 / 3 |

每轮开始前清缓存，允许轮内复用。重放不注入原 trace 的人为等待；保留真实排队、传输、客户端开销和任务内实际间隔。加载、编译、启动和预热时间排除。先测一个后端再测另一个后端，未随机交错，结果不构成统计显著性证明。原始 trace 未随本适配器发布，不能只运行本仓库就声称精确复现该表。

- **吞吐**：总输出 token / 从第一个请求发出到最后一个请求结束的 makespan。
- **TTFT**：请求发出至第一个输出 token。
- **TPOT**：每请求 `(结束时间 − 首 token 时间) / (输出 token 数 − 1)`，包含流结束尾部开销。
- **JCT**：每任务首请求发出至末请求结束。
- mean、p50、p95、p99 先在每轮计算，再取三轮统计量均值；不是合并所有样本重算分位数。

| 指标（三轮均值） | c8 DeepInfer | c8 vLLM | c32 DeepInfer | c32 vLLM |
|---|---:|---:|---:|---:|
| 输出 token/s | 2547.29 | 1900.91 | 5033.24 | 3727.71 |
| 平均 TTFT（ms） | 131.57 | 400.00 | 240.87 | 619.87 |
| 平均 TPOT（ms） | 2.181 | 2.522 | 2.792 | 3.221 |

c32 的 17 项统计指标在三轮逐轮对照中均领先。c8 第三轮 TPOT p95 落后 3.163%，不能宣称所有负载的所有尾指标稳定领先。c32 每轮 717 次 IPC、17 次 CPU 回退均计入测量；c8 每轮 124 次均使用 IPC。

硬件为同一 8×B300，双方均 1P1D、每侧 DP4/EP4，模型同为 DeepSeek-V4.1-Flash-1017 的 FP4 checkpoint（MoE experts FP4，其他层混合精度），不是 DeepInfer FP4 对 vLLM FP8。DeepInfer 使用本仓库 B300 预设；对照 vLLM 为 `0.32.1rc1.dev390+g4bceda5db`，DSpark5、原生 prefix cache、FLASHMLA_MEGA_ATTN_DSV41、NVFP4 compressed KV、NixlConnector/UCX。两套运行时并非所有算法和参数相同。

## 标准 GSM8K 精度

采用 [vLLM 官方 GSM8K README](https://github.com/vllm-project/vllm/blob/4e591657157a021cab997f911d67f50c0d18a71e/tests/evals/gsm8k/README.md) 及对应评测脚本，固定提交 `4e591657157a021cab997f911d67f50c0d18a71e`。

| 参数 | 设置 |
|---|---|
| 数据 | GSM8K test 全量 1,319 题；train 前 5 题作 few-shot |
| API | chat 模式，双方用同一个模板编码代理，再经 router 转发 |
| 解码 | temperature=0、seed=42、max_tokens=1024 |
| 并发 | 8 |
| 模板 | enable_thinking=false、clear_thinking=true |
| 评分 | 官方数字提取与比对；没有 LLM judge 或自定义题集替代 |

| 系统 | 正确数 | 准确率 |
|---|---:|---:|
| router + DeepInfer | 1280 / 1319 | 97.04% |
| router + vLLM | 1281 / 1319 | 97.12% |

总正确数相差 1 题（−0.0758 个百分点），并非只有一题的对错不同：双方都对 1274 题，仅 DI 对 6 题，仅 vLLM 对 7 题，双方都错 32 题。双方无无效答案，各有 2 条输出到达 1024-token 上限。该结果不能证明所有任务质量等价，YOCO/BI/MTP 配置变更后应重新评测。

性能 trace 的 TPOT/TPS 与 GSM8K 的准确率独立记录，不把题目间等待或质量评测运行时间混入性能结果。
