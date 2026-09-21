# External review — GPT-6-Astra via Codex CLI (2026-09-21)

> Consulted as an independent reviewer with an honest brief of the project
> (numbers, same-ruler protocol + caveats, spire negative result, engineering
> findings, productization status). Captured verbatim below; action items we
> accepted are tracked in COMPARISON.md and the postmortem doc.

---

Reading additional input from stdin...
OpenAI Codex v0.154.0
--------
workdir: /tmp
model: gpt-6-astra
provider: openai
approval: never
sandbox: read-only
reasoning effort: medium
reasoning summaries: none
session id: 01a0c518-5436-7e70-bc52-1744f6712dec
--------
user
You are being consulted as an independent reviewer (a "second opinion" subagent). Assess the work below honestly — praise what deserves it, and be blunt about weaknesses. Answer in Chinese, concise but substantive.

## What this project is

"Marvin's Garden" — a decision-classifier built on a ternary 2-bit 27B model (prism-ml Ternary-Bonsai-27B, MLX build, Apache-2.0; upstream Qwen3.6-27B) + our own LoRA. Design philosophy: a decision model that does ONE forward pass per decision with ZERO generated tokens (softmax restricted to candidate letter tokens, "label readout" style), targeting ~1 second per decision on Apple Silicon — deliberately NOT chasing sub-100ms latency. Three production-intent families: model_routing (which sub-model handles a request), tool_gate (allow/deny a tool call), injection_guard (prompt-injection verdicts). Team = one human + AI agent roles (Atlas/Pixel/Vector/Harbor/Echo).

## Measured numbers (same-ruler protocol: identical 192 eval items + information content, letter menu mapped to each system's native choice format)

| | Marvin v15c (local M3 Ultra) | Jev-latest (commercial API) | Nimble-9B (Apache-2.0, open) |
|---|---|---|---|
| Accuracy | **95.31%** (n=3 reruns: 94.44±1.08) | 93.23% | 74.48% (OOD for it; self-reported 90.1% in its own 10 domains) |
| Calibration ECE (15-bin) | **0.031** | 0.102 | 0.151 |
| Latency p50 | 1.21 s local | **0.19 s** hosted (mean 0.82 s) | 2.21 s (Apple Metal fallback, not representative) |
| Cost per 1k decisions | $0 | API pricing | $0 |

Caveats we hold ourselves to: parameter counts are NOT matched (27B vs unknown vs 9B); Jev's protocol is letter-choice via API not native probabilities; only WE measured the game transfer; base license review found Apache-2.0 with NOTICE+attribution obligations (legal confirmation pending).

## Extension experiment: Slay the Spire game lane (tonight)

We built a game demo on an RNG-faithful open-source engine (sts_lightspeed): minted 2,400 game states, labels from engine rollout values, trained a spire lane as a v15c continuation. Result: **38.8% held-out vs 33.7% majority-letter prior** — a real but weak signal (MVP, not production). Root cause found in the dump: the margin≥1.0 filter removed every defend-optimal state, so the "incoming damage → defend" flip dimension is entirely absent from the data; the model correctly learned to never defend. Recorded a fully honest demo video (3/3 Act-1 victories, 62.5% agreement with an engine-rollout oracle that grades but never decides; every probability bar and mistake shown).

## Engineering findings (durable)

1. MLX thread-affinity law (M3 Ultra): model load + every forward must run on the process MAIN thread. Pool-thread forward deadlocks at 0% CPU; cross-thread forward raises "no Stream(cpu,0)"; HTTP-handler threads doing C++ pybind work concurrently with main-thread Metal segfaults. Working architecture: main-thread inference worker consuming a job queue, HTTP threads marshal JSON only.
2. Apple Metal hangs inside mx.eval during training at ~12-50 min intervals; countermeasure = small checkpoint intervals + supervisor auto-resume. Hidden cost found: each resume resets Adam m/v → implicit LR spikes → same-recipe reruns vary 38.8-48.2%.
3. Loss-dilution trap: --mask-prompt averages letter + EOS loss, so val loss 0.20 hid a failing letter head; diagnosing requires scoring the TRAIN pairs too.

## Productization status (just started)

Done: regression gate script (per-family accuracy floors + ECE ceiling, auto-fail on lane additions), formalized crash-tolerant training supervisor, license review memo (Apache-2.0 evidence), honest eval dumps. In flight: OOD/abstention mechanism experiment (using the spire prompts as a natural OOD probe for the routing model — hypothesis: ECE 0.031 means confidence is trustworthy enough to abstain), larger eval set (384 fresh-seed items generated). Not started: NVIDIA serving validation (llama.cpp CUDA path is paper-only; renting a GPU needs the owner's authorization to spend), serving hardening (auth/rate-limit/multi-replica), drift monitoring, injection red-team suite, expansion of eval scale.

## Questions for you

1. How do you rate this work overall — signal quality, the same-ruler comparison methodology, and the honesty discipline? What looks weakest or most overclaimed?
2. For productization: what would YOU prioritize, in what order, given a solo-dev + AI-agents team and a mandate to avoid spending money until results justify it? Specifically: is the OOD-abstention bet the right next move, or is there something higher-leverage?
3. Any structural risk we seem blind to (technical, product, or positioning) given the "1-second intelligence-first classifier, local-first" positioning against hosted sub-200ms incumbents?
hook: SessionStart
hook: SessionStart Completed
hook: UserPromptSubmit
hook: UserPromptSubmit Completed
codex
我的总体评价：**这是一个有实质工程成果、值得继续投入的研究原型，但产品有效性的证据还偏薄。最强的是失败诊断与披露纪律；最弱的是评测规模、置信度解释，以及三个产品方向同时推进。** 以下判断基于你们提供的材料，未独立核验代码和数据。

**1．信号与比较：有价值，但还不足以宣布领先**

- **95.31% 对 93.23%，在 192 题上只是多答对 4 题。** 需要看两者在哪些题上分歧，做配对分析、报告差值置信区间。三次重跑可以测运行或训练波动，不能替代更多独立测试题；也应说明重跑究竟重跑了什么。主结果应突出均值，避免最佳一次成为标题。
- “同题、同信息、映射原生格式”是很好的**端到端任务比较**，但没有消除接口差异。如果 Jev 的置信度来自自报分数、而 Marvin 来自候选 token 概率，ECE 比较尤其需要解释，不能直接推导“模型校准能力领先”。
- Nimble 的 OOD 结果适合作为迁移基线，不适合作为核心竞争胜利。还缺一个更有决策价值的对照：**更小模型、简单规则或轻量分类器，在这三类任务上能做到多少？** 参数不匹配并非不公平，但必须证明 27B 带来的质量收益值得其资源占用。
- **$0 应改成“无按次 API 费用”。** 硬件、内存、电力和维护都有成本；闲置设备上的个人使用与持续服务的经济账不同。

游戏实验最有价值的是发现了数据生成缺陷，而不是证明了跨域能力。38.8% 相对 33.7% 的信号还需要测试规模、区间，以及按轨迹或种子隔离的划分。“模型正确学会永不防御”只能解释训练结果，**不能证明它正确理解游戏**。缺失防御最优样本是明确缺陷，但是否解释主要误差，仍需修复后的对照实验。3/3 胜利是演示，远不是胜率估计。

你们公开弱结果、重启导致优化器状态丢失，以及展示错误，这些值得肯定。但披露 caveat 不能代替收缩主张：目前最稳的表述是“在一个小规模内部基准上表现有竞争力”。

**2．产品化顺序：先让结果可信，再让置信度可用**

我会这样排序：

1. **先选一个主场景：model_routing。** 它更容易定义收益，也较容易容忍错误。tool_gate 和 injection_guard 涉及不对称错误成本，不应共享一套“总体准确率不错”的上线标准。暂停游戏扩展训练，保留为诊断工具。
2. **修复训练可复现性，再扩大冻结评测。** 保存和恢复 Adam 状态、调度器步数，并验证恢复前后的训练连续性。否则每次实验都混入额外变量。384 个 fresh-seed 样本可以补充，但同一生成模板换种子未必带来新覆盖；优先加入独立来源、人工审核、模板隔离的样本。
3. **把 routing 评测改成实际损益评测。** “路由标签正确”未必等于用户结果更好。比较路由后任务成功率、总延迟和总成本，并加入规则、小模型、直接调用默认模型的基线。这一步最可能告诉你们产品是否值得存在。
4. **做 OOD／拒答实验，但更换核心假设。** ECE 0.031 只描述当前分布上的分箱统计，**不推出分布外置信度可信**。尤其候选集合内 softmax 总和必为 1，即使所有候选都不合适，也会选出赢家。Spire 可以做廉价冒烟测试，真正需要的是相邻领域、歧义输入、无正确候选，以及含注入的正常请求。阈值在验证集上确定，报告“覆盖率—错误率”曲线和拒答后的系统效果。
5. **最后做最小服务加固和真实试用。** 优先队列上限、超时、背压、进程重启和访问控制；多副本与 NVIDIA 验证可以等明确需求出现。

所以：**OOD 拒答值得马上做一个小实验，但不值得成为下一阶段的主赌注。更高杠杆的是可靠训练、独立评测，以及 routing 的真实效用验证。**

**3．容易低估的结构性风险**

- **一秒必须放进完整链路算。** 对 routing，这一秒发生在下游模型启动之前；对多次工具调用，延迟可能反复累积。定位能否成立，取决于是否有用户愿意用这一秒交换隐私、离线能力或更少的昂贵错误，而不只是分类准确率高两点。
- **零生成 token 不代表推理成本固定。** 一次前向仍需处理整个 prompt，长上下文和并发排队会改变体验。需要报告输入长度分布、冷／热启动、p95 和并发情况下的延迟。
- **标签读出可能学到菜单位置或措辞。** 随机置换候选顺序、轮换字母映射、改写候选描述，是成本很低、价值很高的稳健性检查。
- **安全场景不能只看准确率和 ECE。** tool_gate 更需要看危险调用误放行率及代价；injection_guard 需要适应性攻击评测。判别器可以增加一道防线，但不应成为唯一权限边界。
- **“MLX 线程亲和性定律”表述过强。** 当前证据支持“你们这套硬件、版本和调用方式下的可靠约束”。主线程队列是合理工程方案，但根因尚未隔离，不宜上升为普遍规律。

我会用一个具体里程碑决定是否继续投入：**在冻结、独立的 routing 测试集上，稳定训练出的 Marvin，相比最佳廉价基线，能否在计入额外一秒后显著改善实际任务结果。** 这个证据比再增加一个 lane、再拍一段演示，或提前租 GPU 更有价值。
hook: Stop
hook: Stop Completed
tokens used
5,714
我的总体评价：**这是一个有实质工程成果、值得继续投入的研究原型，但产品有效性的证据还偏薄。最强的是失败诊断与披露纪律；最弱的是评测规模、置信度解释，以及三个产品方向同时推进。** 以下判断基于你们提供的材料，未独立核验代码和数据。

**1．信号与比较：有价值，但还不足以宣布领先**

- **95.31% 对 93.23%，在 192 题上只是多答对 4 题。** 需要看两者在哪些题上分歧，做配对分析、报告差值置信区间。三次重跑可以测运行或训练波动，不能替代更多独立测试题；也应说明重跑究竟重跑了什么。主结果应突出均值，避免最佳一次成为标题。
- “同题、同信息、映射原生格式”是很好的**端到端任务比较**，但没有消除接口差异。如果 Jev 的置信度来自自报分数、而 Marvin 来自候选 token 概率，ECE 比较尤其需要解释，不能直接推导“模型校准能力领先”。
- Nimble 的 OOD 结果适合作为迁移基线，不适合作为核心竞争胜利。还缺一个更有决策价值的对照：**更小模型、简单规则或轻量分类器，在这三类任务上能做到多少？** 参数不匹配并非不公平，但必须证明 27B 带来的质量收益值得其资源占用。
- **$0 应改成“无按次 API 费用”。** 硬件、内存、电力和维护都有成本；闲置设备上的个人使用与持续服务的经济账不同。

游戏实验最有价值的是发现了数据生成缺陷，而不是证明了跨域能力。38.8% 相对 33.7% 的信号还需要测试规模、区间，以及按轨迹或种子隔离的划分。“模型正确学会永不防御”只能解释训练结果，**不能证明它正确理解游戏**。缺失防御最优样本是明确缺陷，但是否解释主要误差，仍需修复后的对照实验。3/3 胜利是演示，远不是胜率估计。

你们公开弱结果、重启导致优化器状态丢失，以及展示错误，这些值得肯定。但披露 caveat 不能代替收缩主张：目前最稳的表述是“在一个小规模内部基准上表现有竞争力”。

**2．产品化顺序：先让结果可信，再让置信度可用**

我会这样排序：

1. **先选一个主场景：model_routing。** 它更容易定义收益，也较容易容忍错误。tool_gate 和 injection_guard 涉及不对称错误成本，不应共享一套“总体准确率不错”的上线标准。暂停游戏扩展训练，保留为诊断工具。
2. **修复训练可复现性，再扩大冻结评测。** 保存和恢复 Adam 状态、调度器步数，并验证恢复前后的训练连续性。否则每次实验都混入额外变量。384 个 fresh-seed 样本可以补充，但同一生成模板换种子未必带来新覆盖；优先加入独立来源、人工审核、模板隔离的样本。
3. **把 routing 评测改成实际损益评测。** “路由标签正确”未必等于用户结果更好。比较路由后任务成功率、总延迟和总成本，并加入规则、小模型、直接调用默认模型的基线。这一步最可能告诉你们产品是否值得存在。
4. **做 OOD／拒答实验，但更换核心假设。** ECE 0.031 只描述当前分布上的分箱统计，**不推出分布外置信度可信**。尤其候选集合内 softmax 总和必为 1，即使所有候选都不合适，也会选出赢家。Spire 可以做廉价冒烟测试，真正需要的是相邻领域、歧义输入、无正确候选，以及含注入的正常请求。阈值在验证集上确定，报告“覆盖率—错误率”曲线和拒答后的系统效果。
5. **最后做最小服务加固和真实试用。** 优先队列上限、超时、背压、进程重启和访问控制；多副本与 NVIDIA 验证可以等明确需求出现。

所以：**OOD 拒答值得马上做一个小实验，但不值得成为下一阶段的主赌注。更高杠杆的是可靠训练、独立评测，以及 routing 的真实效用验证。**

**3．容易低估的结构性风险**

- **一秒必须放进完整链路算。** 对 routing，这一秒发生在下游模型启动之前；对多次工具调用，延迟可能反复累积。定位能否成立，取决于是否有用户愿意用这一秒交换隐私、离线能力或更少的昂贵错误，而不只是分类准确率高两点。
- **零生成 token 不代表推理成本固定。** 一次前向仍需处理整个 prompt，长上下文和并发排队会改变体验。需要报告输入长度分布、冷／热启动、p95 和并发情况下的延迟。
- **标签读出可能学到菜单位置或措辞。** 随机置换候选顺序、轮换字母映射、改写候选描述，是成本很低、价值很高的稳健性检查。
- **安全场景不能只看准确率和 ECE。** tool_gate 更需要看危险调用误放行率及代价；injection_guard 需要适应性攻击评测。判别器可以增加一道防线，但不应成为唯一权限边界。
- **“MLX 线程亲和性定律”表述过强。** 当前证据支持“你们这套硬件、版本和调用方式下的可靠约束”。主线程队列是合理工程方案，但根因尚未隔离，不宜上升为普遍规律。

我会用一个具体里程碑决定是否继续投入：**在冻结、独立的 routing 测试集上，稳定训练出的 Marvin，相比最佳廉价基线，能否在计入额外一秒后显著改善实际任务结果。** 这个证据比再增加一个 lane、再拍一段演示，或提前租 GPU 更有价值。

