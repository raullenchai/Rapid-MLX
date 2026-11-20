# `rapid-mlx serve` 控制面设计稿 v2

状态：v4 — 用户 2026-11-19 确认：先做 CLI，桌面 app 完全不受影响；第 10 节 4 项按倾向定。（v3：吸收 astra 第 1、2 轮全部 P1；第 2 轮结论：无 P0，P1 收紧后可独立交付），待用户确认 → 实现。日期 2026-11-19。
v1 → v2 最大变化：**推理请求不再经过控制面转发。** runner 照旧自己监听对外端口；控制面是父进程，只承载低流量的 `/rapid/v1`。

## 1. 目标

让 `rapid-mlx serve` 成为"一台 Mac 上可以放心同时跑多个 AI 应用的本地服务"，代码和界面里不出现 hub。

用户直观收益：
1. 能查到每个客户端/应用调了什么、什么时候（调用日志，隐私可验证）；
2. 后台任务和前台交互区分优先级（后台排队时前台先进）；
3. 模型进程崩溃后自动拉起，日志、任务记录、应用不丢；
4. 应用（如 Research Loop）由服务启动、看管、统一登录；
5. 统一的任务和事件视图。

非目标：GUI 改动；多机路由；改变任何现有 `/v1` 行为或性能。

## 2. 现状（origin/main 核实）

| 能力 | 已有 | 缺口 |
|---|---|---|
| 多模型常驻、按需加载、闲置卸载 | residency 四个接口；`--lazy-load`、`--idle-unload-seconds`；预算/LRU/TTL | 无（第一版不动） |
| 任务 | `/v1/agent/runs`、`/v1/cua/runs`、`/v1/videos`，各自内存存储 | 无统一视图、无持久记录 |
| 优先级 | Request 有 priority 字段但构造时没传（`engine_core.py:1202`）；调度器 FCFS 或 validated-tail；已有 deferral 机制（`scheduler.py:6639`） | 无 HTTP 约定，无端到端传递 |
| 日志 | `/metrics`、DEBUG 请求日志 | 无持久调用日志 |
| 父进程 | `rapid-mlx service run`（launchd）拉起 serve、转发信号、处理日志管道 | 只管一个 serve，没有 API |
| 桌面 app | 拉起 serve、`RAPID_MLX_WATCHDOG_PPID`、进程组终止、轮询 `/healthz` | 不能被破坏 |

## 3. 结构

```
rapid-mlx serve [model] [现有参数]
  └─ 控制面进程（父，rapid_mlx/control/，不 import MLX）
       │   私有 Unix socket：$RUNTIME_DIR/<instance>/control.sock（目录 0700）
       ├─ runner 子进程 = 现有 serve，参数原样，自己绑定对外端口
       │     ├─ 所有现有路由、认证、LAN、启动探测、/healthz：完全不变
       │     ├─ /rapid/v1/*  → 薄转发到 control.sock（只有这组走转发，低流量；P1 只开 status）
       │     └─ 最外层 ASGI 请求观察器 → 每个请求一条调用记录，进有界队列，由后台线程写给 control.sock
       └─ app 子进程（可选）
```

为什么这样：
- **推理零转发**：流式、取消（`/v1/requests/{id}`）、上传限流、客户端 IP、LAN 规则、认证、启动探测、`/docs`、dflash/cua-only/retrieval-only 等替代后端——全部原样由 runner 处理。astra 指出的代理类 P1（#2 认证/LAN、#3 取消/上传、#4 路由/启动语义、#5 替代后端）在这个结构下不再成立。
- **性能门槛**：推理路径上只多"把一条记录放进有界队列"这一步（同步、O(1)、不做 I/O）；序列化和投递在后台线程。这是有界的少量工作，不是零工作，用性能门槛验证。
- runner 重启期间对外端口短暂不可用，与今天单进程重启一致；`/healthz` 语义不变。

## 4. 进程监管（astra P1 #1）

- 控制面接管原本 serve 的父进程职责：
  - 若环境/参数里有 `RAPID_MLX_WATCHDOG_PPID`（或 `--watchdog-ppid`），**控制面**看守这个原始父进程；传给 runner 时改写为控制面 PID。
  - runner **不**新开 session，保持在桌面 app 的进程组里，`killpg` 仍能一次清掉全部。
  - TERM/INT/HUP 转发给 runner 和 apps；**截止时间与现有 5 s 约定对齐**（watchdog 和桌面 app 都在 5 s 后升级为 KILL）：控制面等 runner/apps 最多 4 s，然后 KILL；全部回收（waitpid）；关闭过程中禁止重启。
  - 信号处理幂等：桌面 app 用 `killpg` 时父子会同时收到信号，控制面收到第二次信号或发现 runner 已在退出时不重复动作。
  - runner 的 stdout/stderr 由控制面持续读取并原样写到自己的 stdout/stderr（复用 `headless_service/runtime.py` 的管道处理），桌面 app 抓日志的方式不变。
- 崩溃重启：runner 非正常退出 → 记事件 → 指数退避重启（1 s 起，最多 60 s）。10 分钟内 5 次失败：**控制面以非零码退出**，交给 launchd（KeepAlive）或桌面 app 的现有恢复逻辑，不留一个半活的父进程。
- 启动结果结构化上报：runner 启动后经 control.sock 报告 `bound(port)` / `bind_failed` / `startup_failed(reason)`。`bind_failed`（`cli.py:9282`）视为配置错误，不重启，原样退出码退出。
- 端口固定：首次启动时选定的端口（含自动选端口 `cli.py:604` 的结果）在后续重启中固定传给 runner，不重新选。
- 继承 fd：桌面 app 目前不用 `--listen-fd`（`ServerManager.swift:5189`），headless 也没有 fd 交接。若启动时给了 `--listen-fd`，控制面自己持有这个描述符，每次拉起 runner 都显式传入，最终关闭时才关；重启间隙里新连接会排在 backlog 中，不会被拒。
- **桌面 app 完全不受影响（用户决定）**：环境里有 `RAPID_MLX_WATCHDOG_PPID` 或带 `--watchdog-ppid`（桌面 app 启动的标志）时，一律走下面的回落路径，行为与今天逐字节一致。只有终端里直接 `rapid-mlx serve` 和 `rapid-mlx service` 启用控制面。桌面 app 的状态展示以后单独做。
- **支持模式白名单（启动前确定）**：P1 只对白名单里的模式启用控制面——普通文本/VLM serve、`--lazy-load`、retrieval-only、带 embedding/rerank 的组合。其它模式（dflash、cua-only、TensorFold、视频等）在**拉起任何子进程、改写 watchdog、分配 socket 之前**就判定为回落：原样 exec 单进程 serve（参数、watchdog 父进程、继承 fd 全部不变），设置环境变量防止递归监管，`/rapid/v1/status` 不存在，启动日志一行说明"控制功能在此模式下不可用"。白名单随测试覆盖逐步扩大。

## 5. 新 API：`/rapid/v1/`（只允许本机）

`/api/` 与 `/rapid/v1/` 均未占用；选后者。

| 组 | 接口 | 说明 |
|---|---|---|
| 状态 | `GET /rapid/v1/status` | 控制面版本、runner（pid、启动时间、重启次数、状态）、apps、日志设置 |
| 调用日志 | `GET /rapid/v1/log?since=&client=&app=&limit=`、`DELETE /rapid/v1/log`、`PUT /rapid/v1/log/settings` | 一条记录：时间、request id、客户端、应用、模型、路由、状态码、prompt/completion token（runner 内部计数，取不到就记 null）、TTFT、总耗时、是否被取消。**默认只记元数据，不记内容**；按应用显式开启才记内容。本地 JSONL，大小上限、可清空 |
| 任务 | `GET /rapid/v1/jobs`、`GET /rapid/v1/jobs/{id}`、`POST /rapid/v1/jobs/{id}/cancel`；`POST /rapid/v1/jobs` 仅供应用登记自己的长任务 | 统一视图：聚合现有 agent/cua/video runs（复用它们的取消接口，不另造），加上应用登记的任务。持久记录；runner 重启后未完成的任务标为 `interrupted`，不自动重放 |
| 事件 | `GET /rapid/v1/events`（SSE，`Last-Event-ID` 续传，环形缓冲 + 落盘最近 N 条） | runner 启停/崩溃/重启、模型加载/卸载、任务状态、应用状态 |
| 应用 | `GET /rapid/v1/apps`、`POST /rapid/v1/apps/{id}/start|stop|restart`、`GET /rapid/v1/apps/{id}/open` | 从 `~/.rapid-mlx/apps/` 或 `--apps-dir` 发现；`open` 走一次性码登录 |

runner 停着或模型加载中（启动探测返回 503，`_startup_probe.py:46`，保持不变）时，`/rapid/v1` 经对外端口不可达；本机 CLI（`rapid-mlx status` 等）始终可以直接连 control.sock 看到"加载中/重启中"。

**调用记录的归属与投递（astra 第 2 轮 P1）：**
- 唯一归属：一个最外层 ASGI 请求生命周期观察器（结构参考 `middleware/request_logging.py:65`），`try/finally` 加单次发射保护，保证每个请求最多发一条；内部转调不重复记。
- 内容来源：chat / messages / responses / embeddings / audio 各自的适配层把 token 数、TTFT、模型名写进请求作用域的元数据；观察器只读，不接管 receive 通道。
- 状态分开记：已发送的 HTTP 状态码 + 最终结果（`ok` / `error` / `cancelled`，含客户端断开、任务取消、显式取消）——流式请求可能 200 之后才失败。
- 投递承诺：**每个可观察到结束的请求尝试发一条**；队列满或 runner 被 SIGKILL 会丢，丢弃数在 status 里公开；记录带 request id，控制面按它去重。不承诺"严格恰好一次"。

## 6. 认证与安全（astra P1 #2）

- 对外端口的认证、LAN、Host/Origin、限流全部还是 runner 现有逻辑，不搬家。
- `/rapid/v1/*` 转发器：挂在 browser/LAN 中间件之后；**显式**加两个依赖——本机 peer 检查（从 ASGI scope 取真实对端地址，不论是否开 LAN 模式都执行，不信任何转发头）和现有 API key 检查（认证/限流不是全局中间件，`middleware/auth.py:223`，所以必须显式挂）。请求体大小和超时有上限。转发到 control.sock 时附带启动时随机密钥，并**覆盖**客户端自带的同名内部头。启动探测行为不变。
- 应用 token：每个应用一个，带作用域（`inference`、`jobs:own`、`log:own`）；应用调用推理时带 `X-Rapid-App-Token`，runner 校验后把 app 身份写进调用记录（不可伪造）。
- **诚实边界**：0700 目录 + 0600 socket 挡得住其他用户，挡不住同一用户的其它进程；应用进程与模型同属一个用户。文档里不把"剥离环境变量"说成沙箱。真正的沙箱（如 Research Loop 实验代码无网络）由应用自己负责，不在本设计范围。

## 7. 优先级（astra P1 #6）

- 约定：`X-Rapid-Priority: interactive | background`，缺省 interactive；应用 token 发出的请求缺省 background。
- 端到端：各路由适配层解析头 → 经引擎接口传到 Request（不只是 `engine_core.py:1202` 一处，所有构造路径都要传）→ 调度器准入。
- 调度：FCFS 和 validated-tail 两种选择策略都加优先级排序；不改变前缀保持、兼容性检查和内存准入（`scheduler.py:10034,10634`）。background 等待超过 30 s 升为同级后，按"最早到达优先"打破平局，并复用现有有界 deferral（`scheduler.py:6639`），保证在代价排序下也不会饿死。
- **承诺范围收窄**：只保证"排队时前台先进"；已经在跑的后台请求不会被抢占。满载全是后台时，前台需要等一个槽位释放——在文档里写清楚。
- 支持范围：文本/VLM 批处理调度器；音频、图像、视频路径第一版忽略此头。

## 8. 性能门槛（合并前）

M5 Max + 一台小 Mac，单进程（今天）vs 控制面+runner（调用日志开启）：
- TTFT 中位数差 ≤ 1 ms，p95 ≤ 3 ms；解码 tok/s 差 ≤ 1%；并发 8 总吞吐差 ≤ 1%；
- 混合负载（8 个 background + 1 个 interactive）下 interactive 的排队等待显著下降（报告数字，不设硬门槛）；
- 控制面空闲 CPU < 1%、常驻内存 < 60 MB；调用记录投递丢弃率在压测下为 0。

## 9. 分阶段 PR

| # | 内容 | 验收 |
|---|---|---|
| P1 | 控制面骨架：父进程、watchdog 改写、进程组/信号/回收（4 s 截止）、日志管道、退避重启与退出、结构化启动上报、端口固定、继承 fd、control.sock + 启动密钥、**只有** `/rapid/v1/status`（不做通配转发）、`rapid-mlx status` 直连 socket、白名单 + 启动前回落 | 桌面 app 启停/崩溃/强杀/killpg 回归；`rapid-mlx service` 回归；白名单内外全部模式启动矩阵；性能门槛 |
| P2 | 调用日志（最外层观察器 + 适配层元数据 + 控制面存储）+ 事件流 | 各路由类型各一条、流式中途失败、断开/取消、丢弃计数、去重、内容默认不记 |
| P3 | 优先级端到端 + aging（两种调度策略） | 饿死测试、前缀/内存准入不回归、混合负载数字 |
| P4 | 统一任务视图（聚合现有 runs + 应用登记） | 重启后 interrupted；取消走现有接口 |
| P5 | 应用托管 + 应用 token 作用域；Research Loop 迁入为第一个应用 | 原型的一次性码登录、cookie 主机名隔离测试改名后全过 |

原型 `feat/rapid-hub`（已推 GitHub 备份）可复用：有上限的日志存储、应用监管、一次性码登录及其测试。模型路由、`/hub/v1` 推理入口不再需要（推理不经控制面）。

## 10. 已定（2026-11-19）

1. 前缀 `/rapid/v1/`。2. 调用日志默认开、只记元数据、可关。3. 应用目录 `~/.rapid-mlx/apps/`。4. 内部回退变量 `RAPID_MLX_SINGLE_PROCESS=1`。

## 10b. 原待定问题（存档）

1. `/rapid/v1/` 还是 `/api/`？（倾向前者）
2. 调用日志默认开（只元数据）还是默认关？（倾向默认开、可关）
3. 应用目录 `~/.rapid-mlx/apps/`？
4. 单进程退路做成内部环境变量 `RAPID_MLX_SINGLE_PROCESS=1`（不写进用户文档）是否可以？
