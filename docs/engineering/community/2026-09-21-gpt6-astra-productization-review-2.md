# External review #2 — GPT-6-Astra via Codex CLI (2026-09-21, follow-up)

> Round 2: capability roadmap (eval ceiling) + hosted-service productization.
> Captured verbatim (from the interactive consult). Key adopted corrections:
> latency estimate 300–800 ms (NOT 200 ms) pending real kernel benchmarks;
> llama.cpp ternary-format risk (upstream issue #29058) — lock fork/commit/
> type before renting; abstention must ship as accept/review/abstain policy,
> not a confidence field; menu-size confound in the OOD AUROC; the #1
> business risk is generator-distribution fit, not CUDA.

---

### OOD 结果的进一步解释

AUROC 0.971 是很好的信号，但目前最大混杂项确实是菜单大小。候选越多，candidate-only softmax 的最大概率天然越低。下一轮至少应：

- 按相同候选数量比较 ID 与 OOD；
- 分别测试 max probability、top-1/top-2 logit margin、归一化 entropy；
- 加入"所有候选都不正确"的 ID 风格题；
- 用验证集拟合阈值，在冻结测试集报告；
- 按 family 和危险错误类型报告 selective risk。

不要再说"ECE 低，因此 OOD confidence 可信"。现在可以说的是："一个初步的远 OOD 实验显示置信度具有较强区分信号，近 OOD 验证仍在进行。"

## 2．单 GPU 托管架构

"continuous batching 无关"只对了一半。你们没有逐 token decode，所以不需要生成式调度；但多个请求仍然可以组成**动态 prefill microbatch**，这可能是单卡吞吐量的关键。

推荐的最小架构是：

```text
API gateway
  → auth / quota / input limits
  → tokenizer CPU pool
  → deadline-aware queue
  → length buckets + 5–15 ms microbatch window
  → one GPU-owning inference process
  → family-specific calibration/policy
  → decision + confidence + recommended_action
```

关键选择：

- GPU 上只运行一个模型进程，避免重复占用显存。
- 按 prompt 长度分桶，测试 batch 1/2/4/8/16；不要预设最大 batch。
- 队列设置最大长度和请求 deadline，过载时明确返回 429/503。
- 把 queue time、tokenization、GPU time 分开记录。
- 置信度阈值按 family、模型版本和 policy version 管理。
- 自定义 lane 初期使用独立 worker 或静态合并 adapter；不要在每个请求上热切 LoRA。
- 返回模型版本、校准版本和阈值版本，方便事故回放。

当前 `llama.cpp` server 确实支持 CUDA 和独立加载 GGUF LoRA，服务文档已有 `--lora` 支持。但不能由此推断 Prism 的格式已经兼容：新的 Bonsai 2 `PQ2_0/PTQ1_0` 在上游仍有明确的支持请求（llama.cpp issue #29058）。因此必须锁定具体 fork、commit、GGUF 类型和 CUDA kernel，而不是笼统写"llama.cpp 支持"。

### 不要从带宽直接预测 200 ms

360 token × 27B dense 参数，按普通 dense 前向粗算约 19.4 万亿次乘加相关运算。RTX 4090 的 200 ms 已接近很理想的理论区间；实际结果取决于：

- ternary 解包和矩阵 kernel；
- batch shape；
- LoRA 是否融合；
- prompt 长度；
- CPU tokenization 和传输；
- CUDA 路径是否真正全 GPU offload。

低 bit 模型尤其不能按 800 GB/s → 1 TB/s 线性换算；当前 llama.cpp 的讨论也显示 ternary 格式可能因解包而比更宽格式慢（issue #27127 明确提醒"一位更少"不必然是更快路径）。

在实测前，按 **4090 单请求 300–800 ms、低延迟容量约 1–3 RPS** 做保守规划。microbatch 后的吞吐可能更好，但不能先拿它写销售材料。

## 3．abstention 是差异化，但必须是产品策略

它不是 gimmick，前提是 API 表达的是可执行风险策略，而不只是多返回一个概率：

```json
{
  "decision": "B",
  "confidence": 0.87,
  "recommended_action": "review",
  "reason_code": "low_margin",
  "model_version": "...",
  "policy_version": "..."
}
```

三个 action 应由客户可配置的损失函数决定：

- `accept`：风险低于自动执行阈值；
- `review`：转人工或更强模型；
- `abstain`：候选缺失、疑似 OOD，或风险不可接受。

真正的卖点是："在 X% 自动覆盖率下，危险错误率为 Y"，而不是"我们有 confidence 字段"。

也不要公开声称 Jev 因 ECE 高三倍而"无法可信拒答"。ECE 是总体分箱统计，对方可以后校准，而且双方概率取得方式不同。可以说自己的 selective-risk 曲线更好，前提是同协议实测。

## 4．定价形态

不要按 token 模仿生成 API。建议四层：

- 免费开发额度；
- 按 1,000 decisions 收费，设置输入 token 上限；
- 有月最低消费的低延迟共享端点；
- dedicated endpoint 和 custom lane 单独定价。

异步批处理可以更便宜；严格 p95 SLO 应更贵。自定义 lane 的费用应覆盖数据设计、验证、红队和回归维护，而不仅是"一夜训练"。

按当前公开市场，4090 租赁约 $0.74–$1.10/h（RunPod 近期区间）。假设实际容量只有 2 RPS：

- 满载 GPU 成本约 $0.10–$0.15 / 1k decisions；
- 10% 利用率时约 $1–$1.50 / 1k；
- 还未计 CPU、存储、网络、冗余和支持。

早期可测试 **$2–$5 / 1k decisions 加月最低消费**，最终价格应由真实利用率和客户避免一次错误的价值决定。低流量下，闲置 GPU 会比单次计算成本更致命。

## 5．50 美元 GPU 预算怎么花

先租 4090，而不是 H100。50 美元足够做完整的可行性闭环。

1. 在本地先冻结 100–384 条 golden corpus，保存 token IDs、MLX letter logits、概率和预测。
2. 在 4090 上验证 GGUF 能否加载、是否完整 GPU offload、Prism kernel 是否实际启用。
3. 转换并加载 LoRA；逐题比较：分类一致率；accuracy 差值；logit/probability 偏差；ECE 和 abstention threshold 是否漂移。
4. 测 batch 1/2/4/8/16，分别记录纯 GPU latency、端到端 p50/p95、RPS 和显存。
5. 做 30–60 分钟并发 soak test，检查错误、内存增长和尾延迟。
6. 只有 4090 路径正确后，再花 2–3 小时测一张 H100，判断升级是否真的改善成本／请求，而不仅是延迟。

预先写 acceptance gate：

- 分类一致率至少 99%；
- 冻结集 accuracy 不下降超过预设界限（例如 0.5 个百分点）；
- ECE 和 risk–coverage 不明显退化；
- 无 CPU fallback；
- 满负载下 p95 满足目标。

若 GGUF/LoRA parity 失败，立即停止烧钱；那说明当前风险是格式与 kernel，不是 GPU 等级。

## 最大的商业失败路径

最大的风险不是 CUDA，也不是 Jev 更快，而是：

**在生成器分布上构建了一个很好的分类器，但客户真实流量中的类别定义、候选缺失和成本函数不同，导致准确率优势与 abstention 曲线一起消失。**

紧随其后的风险：

- 27B 常驻 GPU 在早期低利用率下没有经济性；
- 客户愿意接受规则、小模型或现有 LLM 中顺手附带的判断，不愿增加一个网络依赖；
- 多租户排队使 p95 远高于单请求 benchmark；
- "custom lane overnight"低估了数据定义、验收和长期漂移维护；
- tool_gate/injection_guard 的一次高代价漏放行，抵消大量平均准确率收益。

所以托管产品的真正护城河不会是"27B ternary + 单次前向"。更可能是：**可审核的数据规范、按业务代价校准的拒答策略、稳定的回归套件，以及客户可以带走并自托管的同一模型。**
