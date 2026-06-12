# LLM Inference Efficiency Research -- Top Papers from 2025 Conferences

**(NeurIPS, ICML, ICLR, AAAI)**

Compiled 2026-02-27. Papers rated Medium or High Apple Silicon relevance only.
Low-relevance papers (multi-GPU-only, CUDA-specific, MPC-focused, etc.) have been filtered out.

---

## Executive Summary

- **Total papers surveyed across 4 conferences:** ~130
- **Papers included (Medium + High Apple Silicon relevance):** 95
- **Breakdown by conference:**
  - NeurIPS 2025: 30 papers (12 High relevance)
  - ICML 2025: 38 papers (22 High relevance)
  - ICLR 2025: 35 papers (23 High relevance)
  - AAAI 2025: 25 papers (12 High relevance)
- **Deduplicated entries:** KVzip (NeurIPS + ICML), DuoAttention (ICLR + roadmap), Sigmoid Attention (ICLR, Apple)

**Key Trends:**

1. **KV cache compression is the dominant theme** -- every conference has 8-12 papers on eviction, quantization, low-rank projection, or sparse representations. The field is converging on 2-4 bit KV caches as practical.
2. **MoE efficiency is surging** -- ICML alone had 10 MoE papers. Expert compression (SVD, delta decomposition), zero-computation experts, and locality-preserving routing address the bandwidth bottleneck that dominates Apple Silicon MoE inference.
3. **Speculative decoding is diversifying** -- model-free (SuffixDecoding, ADED), self-speculative (SWIFT, QuantSpec), and adaptive (BanditSpec) variants eliminate the need for separate draft models.
4. **Reasoning model efficiency is new** -- R-KV, S-GRPO, EAT, DMS, and Think Clearly all target the specific problem of long reasoning chains (R1-style), which generate 10-100x more tokens than standard chat.
5. **Linear/sub-quadratic attention** -- LoLCATs, Sigmoid Attention (Apple), xLSTM, and Cobra/Mamba signal a shift away from softmax attention for inference-constrained settings.
6. **Apple is publishing** -- FlashSigmoid (ICLR), CommVQ (ICML with Apple co-authors), and MLX M5 demos (NeurIPS) show Apple actively investing in on-device inference.

---

## Priority Implementation List (Top 20)

Ranked by: (1) Apple Silicon relevance = High, (2) expected impact on rapid-mlx, (3) implementation feasibility (training-free, Python/MLX-friendly, single-device).

| Rank | Paper | Conference | Technique | Expected Gain | Effort | Status |
|------|-------|------------|-----------|---------------|--------|--------|
| 1 | **DuoAttention** | ICLR | Retrieval vs streaming head classification; full KV for retrieval, sliding window for streaming | 2.55x memory, 2.18x decode | Medium | [On roadmap (#3)](README.md) -- not started |
| 2 | **R-KV** | NeurIPS | Importance + non-redundancy ranking for reasoning KV caches | 100% quality at 10% cache | Medium | New |
| 3 | **SuffixDecoding** | NeurIPS (Spotlight) | Model-free speculative decoding via suffix trees from past outputs | 5.3x speedup, zero extra memory | Medium | New |
| 4 | **SWIFT** | ICLR | Self-speculative decoding by skipping intermediate layers | 1.3-1.6x, no draft model | Low | New |
| 5 | **KVzip** | NeurIPS (Oral) / ICML | Query-agnostic KV compression via context reconstruction | 3-4x cache reduction | Medium | New |
| 6 | **RocketKV** | ICML | Two-stage: coarse eviction + fine-grain top-k sparse attention | 400x compression, 3.7x speedup | Medium | New |
| 7 | **Ada-KV** | NeurIPS | Head-wise adaptive budget allocation for KV eviction | Plug-and-play quality boost | Low | New |
| 8 | **CommVQ** | ICML | RoPE-commutative codebook for 1-bit KV quantization | 87.5% KV memory savings | High | New |
| 9 | **MoE-SVD** | ICML | SVD decomposition of MoE experts, no retraining | 60% compression, 1.5x speedup | Medium | New |
| 10 | **MoE++** | ICLR (Oral) | Zero-computation experts (zero/copy/constant) | 1.1-2.1x expert throughput | Medium | New |
| 11 | **EAT** | NeurIPS (Workshop) | Entropy-based early exit for reasoning after </think> | 21% token reduction, training-free | Low | New |
| 12 | **Falcon** | AAAI | Semi-autoregressive spec decoding with 2-layer drafter | 2.91-3.51x lossless | Medium | New |
| 13 | **Palu** | ICLR | Low-rank KV cache projection with rank search | 11.4x compression, 2.2x speedup | Medium | New |
| 14 | **BanditSpec** | ICML | Bandit algorithm for adaptive draft length, training-free | Adaptive improvement over fixed-k | Low | New |
| 15 | **ADED** | AAAI (Oral) | Tri-gram matrix draft, 253MB corpus, no GPU draft | 2.5x speedup | Low | New |
| 16 | **SpeCache** | ICML | CPU-offload KV cache with top-k fetch per step | GPU memory = top-k only | Medium | New |
| 17 | **LoLCATs** | ICLR | Linearize softmax attention via transfer + LoRA | Linear-time attention | High | New |
| 18 | **QJL** | AAAI | JL transform + sign-bit quantization for KV cache | 5x+ KV reduction, zero overhead | Medium | New |
| 19 | **FlexPrefill** | ICLR (Oral) | Dynamic sparse attention budget per head during prefill | Major TTFT reduction | Medium | New |
| 20 | **D2-MoE** | ICML | Delta decomposition: shared base + compressed deltas | Significant MoE compression | Medium | New |

**Existing roadmap items confirmed by research:**
- **ReDrafter** (roadmap #1) -- validated by EAGLE-3, AdaSPEC, Falcon (speculative decoding is proven effective)
- **KVSplit** (roadmap #2) -- supported by Homogeneous Keys/Heterogeneous Values (ICLR), PolarQuant (NeurIPS)
- **DuoAttention** (roadmap #3) -- directly presented at ICLR 2025 with strong results
- **FastKV** (roadmap #4) -- complemented by ThinK (channel pruning), CAKE (cascading eviction)
- **xKV** (roadmap #5) -- validated by Palu (low-rank projection), MoE-SVD (cross-layer compression)
- **Medusa** (roadmap #6) -- extended by L-MTP, MuToR (multi-token prediction improvements)

---

## Papers by Topic

### 1. Speculative Decoding

**SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications** -- NeurIPS 2025 (Spotlight)
Z. Zhong et al. (CMU) | [arXiv](https://arxiv.org/abs/2411.04975) | [Project](https://suffix-decoding.github.io/)
> Model-free speculative decoding using suffix trees built from past outputs; exploits repetitive patterns in agentic workloads.
> Expected improvement: Up to 5.3x; 2.8x faster than EAGLE-2/3 on agentic benchmarks
> Apple Silicon relevance: High -- no draft model memory needed

**SWIFT: On-the-Fly Self-Speculative Decoding for LLM Inference Acceleration** -- ICLR 2025
Heming Xia et al. | [GitHub](https://github.com/hemingkx/SWIFT)
> Adaptively skips intermediate layers of the target LLM itself as draft; plug-and-play, no training.
> Expected improvement: 1.3-1.6x over vanilla autoregressive
> Apple Silicon relevance: High -- no separate draft model, saves unified memory

**Falcon: Faster and Parallel Inference via Semi-Autoregressive Drafting** -- AAAI 2025
Xiangxiang Gao et al. | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/34566)
> Semi-autoregressive spec decoding with Coupled Sequential Glancing Distillation; drafter is only 2 Transformer layers.
> Expected improvement: 2.91-3.51x lossless speedup, outperforming Eagle/Medusa/Lookahead
> Apple Silicon relevance: High -- compact 2-layer drafter fits easily in unified memory

**ADED: Adaptive Draft-Verification for Efficient LLM Decoding** -- AAAI 2025 (Oral)
Xukun Liu et al. | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/34647)
> Training-free speculative decoding via tri-gram matrix; 253MB corpus data, no GPU needed for draft.
> Expected improvement: Up to 2.5x decoding speedup
> Apple Silicon relevance: High -- extremely lightweight draft component

**BanditSpec: Adaptive Speculative Decoding via Bandit Algorithms** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/poster/44460)
> Multi-armed bandit dynamically learns optimal draft length per prompt, no training.
> Expected improvement: Adaptive improvement over fixed-k speculative decoding
> Apple Silicon relevance: High -- training-free, algorithm-level, works on any backend

**QuantSpec: Self-Speculative Decoding with Hierarchical Quantized KV Cache** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/poster/46326)
> Uses quantized versions of the same model as draft with hierarchical KV cache quantization.
> Expected improvement: Self-speculative with no separate draft model
> Apple Silicon relevance: High -- single model, quantization-friendly for MLX

**Heterogeneous Vocabulary Speculative Decoding** -- ICML 2025
Wen et al. | [ICML](https://icml.cc/virtual/2025/poster/43675)
> Three new SD methods removing the shared-vocabulary constraint between draft and target.
> Expected improvement: Up to 2.8x over autoregressive
> Apple Silicon relevance: High -- removes constraint on draft model selection

**Randomised Drafting for Higher Acceptance Rates** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/47843)
> Randomized drafting strategy to increase token acceptance rates.
> Expected improvement: Higher tokens/second than standard SD
> Apple Silicon relevance: High -- algorithm-level improvement

**EAGLE-3: Scaling up Inference Acceleration via Training-Time Test** -- NeurIPS 2025
Y. Li et al. (SafeAI Lab) | [GitHub](https://github.com/SafeAILab/EAGLE)
> Direct token prediction with fused low/mid/high-level features via "training-time test."
> Expected improvement: 5.6x over vanilla; 1.8x over EAGLE-1
> Apple Silicon relevance: Medium -- benefits scale more with GPU parallelism

**SubSpec: Lossless Acceleration for Offloaded LLMs** -- NeurIPS 2025
P.-S. Wang et al. | [arXiv](https://arxiv.org/abs/2509.18344)
> Draft model from quantized substitute layers of the offloaded target LLM, sharing KV cache.
> Expected improvement: 9.1x for 7B model; 12.5x for 32B model
> Apple Silicon relevance: High -- targets memory-constrained setups, maps to unified memory with swap

**Scaling Speculative Decoding with Lookahead Reasoning** -- NeurIPS 2025
Y. Fu et al. (UCSD Hao AI Lab) | [arXiv](https://arxiv.org/abs/2506.19830) | [GitHub](https://github.com/hao-ai-lab/LookaheadReasoning)
> Step-level parallelism for reasoning models; draft proposes future reasoning steps verified semantically.
> Expected improvement: Up to 2.1x combined with n-gram spec decoding
> Apple Silicon relevance: Medium -- requires draft model in memory

**3-Model Speculative Decoding (PyramidSD)** -- NeurIPS 2025
S. Byun et al. | [arXiv](https://arxiv.org/abs/2510.12966)
> Intermediate qualifier model between draft and target; fuzzy acceptance criteria.
> Expected improvement: Up to 1.91x over standard speculative decoding
> Apple Silicon relevance: Medium -- three models challenging for memory

**AdaSPEC: Selective Knowledge Distillation for Speculative Decoders** -- NeurIPS 2025 (Spotlight)
Y. Hu et al. | [arXiv](https://arxiv.org/abs/2510.19779) | [GitHub](https://github.com/yuezhouhu/adaspec)
> Filters hard-to-fit tokens during draft model distillation, training on "easy" tokens only.
> Expected improvement: Up to 15% higher acceptance rate over DistillSpec
> Apple Silicon relevance: Medium -- improves any draft model quality

**Reward-Guided Speculative Decoding (RSD)** -- ICML 2025
Liao et al. (Salesforce) | [ICML](https://icml.cc/virtual/2025/poster/46166)
> Process reward model dynamically decides when to invoke target during reasoning.
> Expected improvement: Up to 4.4x fewer FLOPs
> Apple Silicon relevance: Medium -- requires reward model overhead

**Judge Decoding** -- ICLR 2025
Gregor Bachmann et al. | [OpenReview](https://openreview.net/forum?id=mtSSFiqW6y)
> "Judge" accepts high-quality draft tokens even when deviating from target distribution.
> Expected improvement: 9x speedup (8B/405B pair)
> Apple Silicon relevance: Medium -- draft/judge concept applicable to smaller setups

**HASS: Harmonized Representations for Speculative Sampling** -- ICLR 2025
[GitHub](https://github.com/HArmonizedSS/HASS)
> Harmonizes draft model objective and context alignment.
> Expected improvement: 2.81-4.05x wall-clock speedup, surpassing EAGLE-2 by 8-20%
> Apple Silicon relevance: Medium -- requires separate draft model training

**Faster Cascades via Speculative Decoding** -- ICLR 2025 (Oral)
[OpenReview](https://openreview.net/forum?id=vo9t20wsmd)
> Combines cascading (small-to-large deferral) with speculative execution.
> Expected improvement: Significant speedup with quality guarantee
> Apple Silicon relevance: Medium -- cascading useful with multiple model sizes

**Dynamic-Width Speculative Beam Decoding** -- AAAI 2025
Zongyue Qin et al. | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/34690)
> Adaptive beam count based on context; parallel tree verification.
> Expected improvement: 1.5-1.9x speed-up, 1.8-2.5x lower energy
> Apple Silicon relevance: Medium

**DREAM: Speculative Decoding for Vision-Language Models** -- NeurIPS 2025
SAI Lab, NYU | [GitHub](https://github.com/SAI-Lab-NYU/DREAM)
> Cross-attention injects target features into draft model for VLMs.
> Expected improvement: Up to 3.6x for VLMs
> Apple Silicon relevance: Medium

**Polybasic Speculative Decoding** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/poster/45669)
> Multi-model framework with optimal inference time theorem.
> Expected improvement: 3.31-4.43x on LLaMA models
> Apple Silicon relevance: Medium -- multiple models needed

**CoSD: Speculate, then Collaborate** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/poster/44974)
> Multiple LLMs collaborate; one drafts, lightweight rule decides when another refines.
> Expected improvement: No retraining needed
> Apple Silicon relevance: Medium

**RAPID: Retrieval-Augmented Speculative Decoding for Long-Context** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/poster/46343)
> RAG drafter operates on shortened retrieval contexts.
> Expected improvement: Accelerates long-context decoding
> Apple Silicon relevance: Medium

---

### 2. KV Cache Compression & Eviction

**KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction** -- NeurIPS 2025 (Oral) / ICML 2025
SNU ML Lab | [GitHub](https://github.com/snu-mllab/KVzip)
> Quantifies KV pair importance by reconstructing original contexts; compressed cache reusable across diverse queries.
> Expected improvement: 3-4x KV cache reduction; 2x FlashAttention decoding latency reduction; up to 170K contexts
> Apple Silicon relevance: High -- query-agnostic compression ideal for prompt caching in local inference

**R-KV: Redundancy-aware KV Cache Compression for Reasoning Models** -- NeurIPS 2025
Z. Cai et al. | [arXiv](https://arxiv.org/abs/2505.24133) | [GitHub](https://github.com/Zefan-Cai/R-KV)
> Ranks tokens by importance AND non-redundancy for reasoning model KV caches.
> Expected improvement: 100% of full KV performance at 10% cache (vs 60% for baselines)
> Apple Silicon relevance: High -- critical for R1-style reasoning where generated KV dominates memory

**RocketKV: Two-Stage KV Cache Compression** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/poster/45253)
> Coarse-grain permanent eviction + fine-grain top-k sparse attention.
> Expected improvement: Up to 400x compression, 3.7x speedup, 32.6% peak memory reduction
> Apple Silicon relevance: High -- training-free, reduces decode memory

**Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation** -- NeurIPS 2025
Y. Feng et al. | [arXiv](https://arxiv.org/abs/2407.11550) | [GitHub](https://github.com/FFY0/AdaKV)
> Head-wise adaptive budget allocation; theoretical loss upper bound guides allocation.
> Expected improvement: Substantial quality improvements at equivalent cache budgets; plug-and-play
> Apple Silicon relevance: High -- no hardware-specific requirements

**DMS: Inference-Time Hyper-Scaling with KV Cache Compression** -- NeurIPS 2025
A. Lancucki et al. (NVIDIA / U. Edinburgh) | [arXiv](https://arxiv.org/abs/2506.05345)
> Dynamic Memory Sparsification delays token eviction, implicitly merging representations.
> Expected improvement: 8x KV compression; +9.1 accuracy on AIME24 for Qwen-R1 32B
> Apple Silicon relevance: High -- massive memory savings for reasoning; requires brief fine-tuning

**DuoAttention: Efficient Long-Context Inference with Retrieval and Streaming Heads** -- ICLR 2025
MIT Han Lab | [GitHub](https://github.com/mit-han-lab/duo-attention)
> Identifies retrieval vs streaming attention heads; full KV for retrieval, constant-length for streaming.
> Expected improvement: 2.55x memory reduction (MHA); 2.18x decode speedup; 3.33M context on single GPU
> Apple Silicon relevance: High -- constant-length cache for streaming heads reduces memory scaling
> **ON ROADMAP (#3)**

**Palu: KV-Cache Compression with Low-Rank Projection** -- ICLR 2025
Chi-Chih Chang et al. | [GitHub](https://github.com/shadowpa0327/Palu)
> Low-rank decomposition of KV cache with medium-grained scheme and rank search.
> Expected improvement: 2.20x speedup over FP16; 6.17x with 4-bit at 64K; 91.25% compression (11.4x)
> Apple Silicon relevance: High -- directly reduces memory pressure on unified memory

**CAKE: Cascading and Adaptive KV Cache Eviction** -- ICLR 2025
Ziran Qin et al. | [GitHub](https://github.com/antgroup/cakekv)
> Layer-specific cache allocation based on spatial/temporal attention dynamics.
> Expected improvement: Maintains performance at 3.2% KV cache; >10x decoding speedup at 128K
> Apple Silicon relevance: High -- extreme compression for memory-limited devices

**CommVQ: Commutative Vector Quantization for KV Cache** -- ICML 2025
Li et al. (UMass/Apple) | [ICML](https://icml.cc/virtual/2025/poster/43828)
> RoPE-commutative codebook enabling 1-bit KV cache quantization.
> Expected improvement: 87.5% KV cache size reduction; LLaMA-3.1 8B at 128K on single GPU
> Apple Silicon relevance: High -- Apple co-authored; directly reduces memory pressure

**SpeCache: Speculative Key-Value Caching** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/poster/45386)
> Offloads full KV cache to CPU, dynamically fetches top-k relevant entries per decode step.
> Expected improvement: Reduces accelerator memory to top-k only
> Apple Silicon relevance: High -- unified memory makes CPU offload near-free

**QJL: 1-Bit Quantized JL Transform for KV Cache** -- AAAI 2025
Amir Zandieh et al. | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/34773)
> Johnson-Lindenstrauss transform + sign-bit quantization; zero overhead from quantization constants.
> Expected improvement: 5x+ KV cache memory reduction at 3-bit effective precision
> Apple Silicon relevance: High -- zero-overhead design avoids dequantization bottlenecks

**CSR: 1-Bit Key-Value Cache via Sparse Representation** -- AAAI 2025
Hongxuan Zhang et al. | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/34779)
> Represents KV cache as sparse vectors; CSR-8 ~ 2-bit, CSR-16 ~ 4-bit effective.
> Expected improvement: Extreme compression enabling much longer contexts
> Apple Silicon relevance: High -- sparse representations can leverage Accelerate framework

**PolarQuant: Polar Coordinate Key Cache Quantization** -- NeurIPS 2025
E. Wu et al. | [arXiv](https://arxiv.org/abs/2502.00527) | [GitHub](https://github.com/ericshwu/PolarQuant)
> Encodes key vectors as quantized radius + polar angle; inner products become table lookups.
> Expected improvement: Full-precision performance at 3-bit keys
> Apple Silicon relevance: High -- table-lookup approach efficient on any hardware

**SmallKV: Small Model Assisted Compensation of KV Cache Compression** -- NeurIPS 2025
Y. Zhao et al. | [arXiv](https://arxiv.org/abs/2508.02751)
> Small companion model approximates evicted tokens' attention scores.
> Expected improvement: 1.75-2.56x higher throughput than baselines
> Apple Silicon relevance: High -- small companion manageable in unified memory

**KeyDiff: Key Similarity-Based KV Cache Eviction** -- NeurIPS 2025
P. Jones et al. | [arXiv](https://arxiv.org/abs/2504.15364)
> Pairwise key cosine similarity as proxy for token importance; training-free.
> Expected improvement: <0.04% performance gap at 23% cache reduction; 30% lower latency
> Apple Silicon relevance: High -- training-free and lightweight

**KVLink: KV Cache Reuse Across Documents** -- NeurIPS 2025
H. Chang et al. (UCSB NLP) | [arXiv](https://arxiv.org/abs/2502.16002) | [GitHub](https://github.com/UCSB-NLP-Chang/KVLink)
> Precomputes KV cache per document; concatenates at inference with adjusted positional embeddings.
> Expected improvement: 96% reduction in TTFT; 4% accuracy improvement
> Apple Silicon relevance: High -- directly applicable to prompt caching in local RAG

**Dialogue Without Limits: Constant-Sized KV Caches** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/poster/45197)
> Fixed-size KV cache for arbitrarily long generation.
> Expected improvement: Constant memory regardless of response length
> Apple Silicon relevance: High -- critical for long-running sessions on 64-192GB Macs

**ShadowKV: KV Cache in Shadows** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/poster/44053)
> Low-rank key cache + offloaded value cache with sparse KV reconstruction.
> Expected improvement: High-throughput long-context inference
> Apple Silicon relevance: High -- low-rank compression suits bandwidth limits

**LaCache: Ladder-Shaped KV Caching** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/poster/45236)
> Sequential and cross-layer KV storage with iterative compaction.
> Expected improvement: Efficient long-context modeling
> Apple Silicon relevance: High -- progressive compression fits memory constraints

**SqueezeAttention: 2D Management of KV-Cache** -- ICLR 2025
[OpenReview](https://openreview.net/pdf?id=9HK2rHNAhd)
> Layer-wise KV-cache budget allocation; orthogonal to sequence-wise methods.
> Expected improvement: 30-70% memory savings; up to 2.2x throughput
> Apple Silicon relevance: High

**D2O: Dynamic Discriminative Operations for Efficient Long-Context** -- ICLR 2025
[GitHub](https://github.com/AIoT-MLSys-Lab/D2O)
> Two-level dynamic KV optimization: layer-level allocation + token-level compensation.
> Expected improvement: >3x inference throughput
> Apple Silicon relevance: High -- no fine-tuning required

**ThinK: Thinner Key Cache by Query-Driven Pruning** -- ICLR 2025 (Spotlight)
Salesforce AI Research | [OpenReview](https://openreview.net/forum?id=n0OtGl6VGb)
> First channel-dimension pruning for KV cache; orthogonal to eviction/quantization.
> Expected improvement: >20% KV cache memory reduction
> Apple Silicon relevance: High -- plug-and-play, reduces bandwidth needs

**Homogeneous Keys, Heterogeneous Values** -- ICLR 2025
> Exploits asymmetry between key and value distributions for long-context LLMs.
> Expected improvement: Memory reduction via asymmetric compression
> Apple Silicon relevance: High -- leverages inherent structure

**SWAN: Sparse Winnowed Attention for Decompression-Free KV-Cache Compression** -- ICLR 2025
> Offline orthogonal rotation + pruning; used directly in attention without reconstruction.
> Expected improvement: Eliminates decompression overhead
> Apple Silicon relevance: High -- no compute overhead for compressed cache

**KVTuner: Sensitivity-Aware Layer-Wise Mixed-Precision KV Quantization** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/poster/43487)
> Multi-objective optimization for per-layer KV quantization precision.
> Expected improvement: Nearly lossless at 3.25-bit mixed precision
> Apple Silicon relevance: High

**Cache Me If You Must: Adaptive Key-Value Quantization** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/poster/46067)
> Exploits mutual information between layers to predict adjacent KV, storing residuals only.
> Expected improvement: Better accuracy at extreme 2-bit quantization
> Apple Silicon relevance: High -- inter-layer prediction is compute-cheap

**VL-Cache: Sparsity and Modality-Aware KV Cache for VLMs** -- ICLR 2025
Dezhan Tu et al. | [OpenReview](https://openreview.net/forum?id=HMrcv7Q4Ub)
> Layer-adaptive budget + modality-aware scoring for vision-language models.
> Expected improvement: 90% KV cache reduction; 7.08x decode speedup
> Apple Silicon relevance: Medium

**KVCOMM: Cross-context KV-cache Communication for Multi-agent Systems** -- NeurIPS 2025
H. Ye et al. | [arXiv](https://arxiv.org/abs/2510.12872) | [GitHub](https://github.com/FastMAS/KVCOMM)
> Estimates and adjusts KV caches for shared content across agents.
> Expected improvement: 70%+ cache reuse; up to 7.8x TTFT in 5-agent settings
> Apple Silicon relevance: Medium

**PiKV: KV Cache Management for MoE Architecture** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/51776)
> Expert-sharded KV storage with adaptive scheduling.
> Expected improvement: Efficient MoE KV serving
> Apple Silicon relevance: Medium -- relevant for MiniMax-style models

**Cake (Compute or Load KV Cache)** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/poster/45020)
> Bidirectional scheduling balancing KV computation and I/O loading.
> Expected improvement: 2.6x TTFT reduction
> Apple Silicon relevance: Medium

---

### 3. Model Quantization

**FlatQuant: Flatness Matters for LLM Quantization** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/poster/43726)
> Enhances flatness of weights/activations for better post-training quantization.
> Expected improvement: <1% accuracy drop at full W4A4 on LLaMA-3
> Apple Silicon relevance: High -- 4-bit is the sweet spot for Apple Silicon bandwidth

**NestQuant: Nested Lattice Quantization for Matrix Products** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/poster/46487)
> Lattice-based quantization replacing round-to-nearest.
> Expected improvement: Universally outperforms contemporary 4-bit on LLaMA-3 8B-70B
> Apple Silicon relevance: High -- better 4-bit quality directly benefits MLX

**SpinQuant: LLM Quantization with Learned Rotations** -- ICLR 2025
Liu et al. (Meta) | [GitHub](https://github.com/facebookresearch/SpinQuant)
> Learns rotation matrices within Stiefel manifold to mitigate outliers.
> Expected improvement: W4A4KV4: only 2.9pt gap from FP16 (vs 22pt for LLM-QAT)
> Apple Silicon relevance: High -- W4A4 enables fast inference with low memory

**CBQ: Cross-Block Quantization** -- ICLR 2025
[arXiv](https://arxiv.org/abs/2312.07950)
> Cross-block reconstruction with adaptive LoRA-Rounding.
> Expected improvement: Superior W4A4, W4A8, W2A16
> Apple Silicon relevance: High -- W2 weight quantization maximizes memory capacity

**LeanQuant: Loss-Error-Aware Grid Quantization** -- ICLR 2025
Tianyi Zhang et al. | [GitHub](https://github.com/LeanModels/LeanQuant)
> Loss-error-aware quantization grid preserving outliers; no extra storage.
> Expected improvement: Superior accuracy at low-bit with minimal overhead
> Apple Silicon relevance: High -- efficient PTQ directly applicable

**Scaling Laws for Precision** -- ICLR 2025
Tanishq Kumar et al. | [PDF](https://pehlevan.seas.harvard.edu/sites/g/files/omnuum6471/files/2025-03/Kumar_etal_ICLR_2025.pdf)
> Precision-aware scaling laws for training and inference.
> Expected improvement: Guides optimal bit-width selection for given model/data scale
> Apple Silicon relevance: High -- informs optimal quantization strategy for deployment

**MoEQuant: Quantization for MoE via Expert-Balanced Sampling** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/poster/46674)
> Expert-balanced calibration + affinity-guided quantization for MoE.
> Expected improvement: 10+ points accuracy gain on HumanEval for DeepSeekMoE-16B at W4
> Apple Silicon relevance: High -- directly relevant to quantized MoE on Apple Silicon

**SLiM: One-shot Quantization + Sparsity + Low-Rank** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/poster/46479)
> Integrates hardware-friendly quantization, 2:4 sparsity, and low-rank approximation.
> Expected improvement: Up to 5.66% accuracy improvement for 2:4 sparse + W4
> Apple Silicon relevance: High -- combines multiple compression axes

**GuidedQuant: End Loss Guidance for Quantization** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/poster/44844)
> Integrates gradient from end loss into layer-wise quantization objective.
> Expected improvement: Consistently boosts SOTA quantization methods
> Apple Silicon relevance: High -- improves any PTQ pipeline

**Radio: Rate-Distortion Optimization for LLM Compression** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/poster/44344)
> Information-theoretic foundations using rate-distortion theory.
> Expected improvement: Scales to 100B+ parameter models
> Apple Silicon relevance: High -- principled approach to optimal bit allocation

**Understanding and Mitigating Numerical Nondeterminism in LLM Inference** -- NeurIPS 2025 (Oral)
M. Li et al. | [arXiv](https://arxiv.org/abs/2506.09501) | [GitHub](https://github.com/nanomaoli/llm_reproducibility)
> bfloat16 causes up to 9% accuracy variation; proposes LayerCast (store 16-bit, compute FP32).
> Expected improvement: Deterministic inference with minimal overhead
> Apple Silicon relevance: High -- LayerCast pattern natural for unified memory

**ABQ-LLM: Arbitrary-Bit Quantized Inference** -- AAAI 2025
Chao Zeng et al. (ByteDance) | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/34385)
> Arbitrary-precision quantized inference via BTC equivalents.
> Expected improvement: 1.6x acceleration, 2.7x memory compression vs SmoothQuant
> Apple Silicon relevance: Medium -- BTC targets CUDA but algorithmic ideas transferable

**ASER: Activation Smoothing and Error Reconstruction** -- AAAI 2025
Weibo Zhao et al. (Alibaba Cloud) | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/34443)
> Outlier extraction + low-rank error compensation for W4A8.
> Expected improvement: W4A8 near half-precision quality
> Apple Silicon relevance: Medium -- could improve quality of existing 4-bit MLX models

**SliM-LLM: Salience-Driven Mixed-Precision Quantization** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/poster/45388)
> Group-wise salience-driven bit-width allocation.
> Expected improvement: Optimal mixed precision per group
> Apple Silicon relevance: Medium -- mixed precision kernel support needed

**I-LLM: Integer-Only Inference for Fully-Quantized LLMs** -- ICLR 2025
[OpenReview](https://openreview.net/forum?id=44pbCtAdLx)
> Integer-only PTQ with Fully-Smooth Block-Reconstruction; bit-shift for non-linear ops.
> Expected improvement: W4A4 with negligible loss; first integer-only quantization
> Apple Silicon relevance: Medium -- integer-only ops could leverage Neural Engine

**RoSTE: Efficient QAT for Quantized LLMs** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/poster/44438)
> Quantization-aware supervised fine-tuning approach.
> Expected improvement: Efficient QAT
> Apple Silicon relevance: Medium

**QuEST: Stable Training with 1-Bit Weights and Activations** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/poster/45754)
> Demonstrates stable convergence at extreme 1-bit quantization.
> Expected improvement: Stable at 4-bit, convergent at 1-bit
> Apple Silicon relevance: Medium -- 4-bit results useful now; 1-bit future-looking

---

### 4. Attention Optimization

**Gated Attention for Large Language Models** -- NeurIPS 2025 (Best Paper Award)
Z. Qiu et al. (Qwen team / Tsinghua) | [GitHub](https://github.com/qiuzh20/gated_attention)
> Head-specific sigmoid gate after scaled dot-product; introduces non-linearity and query-dependent sparse gating that eliminates attention sinks.
> Expected improvement: Consistently improved quality; enables long-context extrapolation
> Apple Silicon relevance: High -- if adopted in future models, benefits are automatic

**Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning** -- NeurIPS 2025
M. Gao et al. (Tsinghua) | [PDF](http://people.iiis.tsinghua.edu.cn/~gaomy/pubs/twilight.neurips25.pdf)
> Hierarchical select-then-prune using top-p sampling; prunes up to 98% of tokens during attention.
> Expected improvement: 15.4x self-attention speedup; 3.9x end-to-end per-token reduction
> Apple Silicon relevance: High -- attention sparsity reduces compute and bandwidth

**SALS: Sparse Attention in Latent Space** -- NeurIPS 2025
Y. Xu et al. | [arXiv](https://arxiv.org/abs/2510.24273)
> Projects KV cache into compact latent space; sparse token selection using RoPE-free interactions.
> Expected improvement: 6.4x KV compression; 5.7x attention speed-up; 4.5x end-to-end at 32K
> Apple Silicon relevance: High -- reduces both memory and compute

**RetrievalAttention: Long-Context via Vector Retrieval** -- NeurIPS 2025
Microsoft Research | [arXiv](https://arxiv.org/abs/2409.10516)
> Pre-builds ANNS indexes for KV vectors in CPU memory.
> Expected improvement: Near full accuracy at 1-3% data access; 128K on single 24GB GPU
> Apple Silicon relevance: High -- CPU-side indexing maps well to unified memory

**FlexPrefill: Context-Aware Sparse Attention for Prefill** -- ICLR 2025 (Oral)
ByteDance Seed | [GitHub](https://github.com/ByteDance-Seed/FlexPrefill)
> Dynamically adjusts sparse attention patterns per input and head in real-time.
> Expected improvement: Significant speed and accuracy over prior sparse methods
> Apple Silicon relevance: High -- dynamic sparsity adapts to compute/memory tradeoffs

**LoLCATs: Low-Rank Linearizing of Large Language Models** -- ICLR 2025
Hazy Research (Stanford/Together) | [arXiv](https://arxiv.org/abs/2410.10254)
> Replaces softmax attention with linear attention via attention transfer + LoRA.
> Expected improvement: Closes 77.8% of quality gap on MMLU for Llama 3.1 70B; first linearized 405B
> Apple Silicon relevance: High -- eliminates quadratic scaling for long-context

**Sigmoid Self-Attention + FlashSigmoid** -- ICLR 2025
Jason Ramapuram et al. (Apple) | [GitHub](https://github.com/apple/ml-sigmoid-attention)
> Sigmoid as drop-in replacement for softmax; FlashSigmoid kernel implementation.
> Expected improvement: 17% kernel speedup over FlashAttention-2; ~8% end-to-end
> Apple Silicon relevance: High -- Apple's own research for their hardware

**SeerAttention: Learnable Intrinsic Sparse Attention** -- ICLR 2025
Microsoft | [GitHub](https://github.com/microsoft/SeerAttention)
> MoE-inspired learnable gate selectively activates important blocks; self-distillation training.
> Expected improvement: 90% sparsity at 32K; 5.67x speedup over FlashAttention-2
> Apple Silicon relevance: High -- learned sparsity could be baked into models

**When Attention Sink Emerges: An Empirical View** -- ICLR 2025 (Spotlight)
Xiangming Gu et al. | [GitHub](https://github.com/sail-sg/Attention-Sink)
> Attention sink stems from softmax normalization; sigmoid attention eliminates it.
> Expected improvement: Insights for KV cache optimization and streaming
> Apple Silicon relevance: High -- directly relevant to prompt cache and streaming inference

**SpargeAttention: Training-free Sparse Attention** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/poster/46341)
> Two-stage online filter predicts and skips unimportant attention blocks.
> Expected improvement: Universal, works on any model
> Apple Silicon relevance: High -- training-free, model-agnostic

**Grouped Cross Attention (GCA) via Causal Retrieval** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/poster/46384)
> Dynamic context retrieval-based attention generalizing to 1000x pre-training length.
> Expected improvement: Constant attention window with distant information access
> Apple Silicon relevance: High -- constant window fits memory constraints

**Spotlight Attention: Non-linear Hashing KV Cache Retrieval** -- NeurIPS 2025
[arXiv](https://arxiv.org/abs/2508.19740)
> Non-linear hashing functions retrieve relevant KV cache entries.
> Expected improvement: <100us retrieval for 512K tokens; 3x throughput
> Apple Silicon relevance: Medium -- concept is hardware-agnostic but needs Metal kernel

**XAttention: Block Sparse Attention with Antidiagonal Scoring** -- ICML 2025
MIT Han Lab | [ICML](https://icml.cc/virtual/2025/poster/45650)
> Antidiagonal sums as proxy for block importance; plug-and-play.
> Expected improvement: Up to 13.5x attention acceleration
> Apple Silicon relevance: Medium -- requires custom kernel support

**SageAttention2: Efficient Attention with INT4** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/poster/44114)
> Per-thread INT4 quantized attention with outlier smoothing.
> Expected improvement: 3x over FlashAttention2
> Apple Silicon relevance: Medium -- INT4 attention needs Metal kernel support

**kNN Attention Demystified** -- ICLR 2025
[ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/file/7620e67593bb2d2ce2eb2eb079678a3d-Paper-Conference.pdf)
> Theoretical guarantees for kNN Attention with close to linear time.
> Expected improvement: Linear time attention with quality guarantees
> Apple Silicon relevance: Medium

**Gating is Weighting: Understanding Gated Linear Attention** -- ICLR 2025
[OpenReview](https://openreview.net/forum?id=AC9FsaVIpk)
> Proves when gating is provably better than vanilla linear attention.
> Expected improvement: Informs Mamba/RWKV-style architecture choices
> Apple Silicon relevance: Medium

**MMInference: Modality-Aware Sparse Attention for VLMs** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/poster/44144)
> Dynamic sparse attention for prefill at 1M tokens.
> Expected improvement: Up to 8.3x prefill speedup
> Apple Silicon relevance: Medium

---

### 5. MoE Inference Optimization

**MoE++: Accelerating MoE with Zero-Computation Experts** -- ICLR 2025 (Oral)
Peng Jin et al. (Skywork) | [GitHub](https://github.com/SkyworkAI/MoE-plus-plus)
> Zero-computation experts (zero/copy/constant) skip FFN entirely.
> Expected improvement: 1.1-2.1x expert forward throughput
> Apple Silicon relevance: High -- reduces active expert computation, directly benefits bandwidth-bound inference

**MoE-SVD: Structured MoE Compression via SVD** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/poster/44786)
> SVD-based decomposition framework for MoE without extra training.
> Expected improvement: 60% compression, 1.5x faster inference
> Apple Silicon relevance: High -- directly applicable to local MoE models

**D2-MoE: Delta Decompression for MoE Compression** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/poster/43461)
> Decomposes experts into shared base + delta weights; SVD on deltas, pruning on base.
> Expected improvement: Significant MoE compression
> Apple Silicon relevance: High -- reduces total parameters loaded per expert

**Mixture of Lookup Experts** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/poster/43620)
> Transforms expert computation into lookup process; no GPU memory loading needed.
> Expected improvement: Eliminates expert loading latency
> Apple Silicon relevance: High -- avoids memory bandwidth bottleneck entirely

**Oracle-MoE: Locality-Preserving Routing for Memory-Constrained Inference** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/poster/43606)
> Groups semantically similar tokens to reduce expert swapping frequency.
> Expected improvement: Reduces expert swap overhead
> Apple Silicon relevance: High -- directly addresses expert loading cost on unified memory

**FloE: On-the-Fly MoE Inference on Memory-Constrained GPU** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/poster/44378)
> Offloads experts to CPU, loads on demand with compressed internal matrices.
> Expected improvement: Reduces data movement for expert loading
> Apple Silicon relevance: High -- Apple unified memory helps; compression reduces bandwidth

**MxMoE: Mixed-Precision Quantization for MoE** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/poster/43995)
> Co-designs accuracy and performance for mixed-precision MoE quantization.
> Expected improvement: Mixed-precision MoE serving
> Apple Silicon relevance: High -- quantized MoE fits Apple Silicon bandwidth

**Retraining-free Merging of Sparse MoE via Hierarchical Clustering** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/poster/44392)
> Merges redundant experts post-training via clustering.
> Expected improvement: Reduces number of active experts
> Apple Silicon relevance: High -- fewer experts = less memory/compute

**Fiddler: CPU-GPU Orchestration for Fast MoE Inference** -- ICLR 2025
Keisuke Kamahori et al. | [GitHub](https://github.com/efeslab/fiddler)
> Intelligent CPU-GPU workload distribution per expert layer.
> Expected improvement: 1.26x single-batch; 11.57x beam search
> Apple Silicon relevance: High -- maps directly to Apple Silicon's unified memory + CPU/GPU/ANE

**OLMoE: Open Mixture-of-Experts Language Models** -- ICLR 2025 (Spotlight)
AI2/Allen Institute | [OpenReview](https://openreview.net/forum?id=xXTkbTBmqq)
> Fully open MoE LM; 1B active params out of 7B total.
> Expected improvement: Outperforms Llama2-13B-Chat with 1B active params
> Apple Silicon relevance: High -- small active parameter count ideal for bandwidth

**Advancing Expert Specialization for Better MoE** -- NeurIPS 2025 (Oral)
H. Guo et al. | [arXiv](https://arxiv.org/abs/2505.22323)
> Orthogonality + variance loss for expert specialization during training.
> Expected improvement: Up to 23.79% quality improvement; fewer active parameters per token
> Apple Silicon relevance: High -- better specialization = less bandwidth per token

**BigMac: Communication-Efficient MoE Structure** -- AAAI 2025
USTC/Huawei | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/33945)
> Low-dimensional projections at expert entry/exit.
> Expected improvement: 3.09x lower training latency, 3.11x higher inference throughput
> Apple Silicon relevance: Medium -- low-dim projection reduces memory bandwidth even in single-device

**MoE-CAP: Benchmarking Cost, Accuracy and Performance** -- NeurIPS 2025
Microsoft Research | [arXiv](https://arxiv.org/abs/2412.07067) | [GitHub](https://github.com/Auto-CAP/MoE-CAP)
> CAP Radar Diagram and sparsity-aware metrics for MoE trade-offs.
> Expected improvement: Benchmarking framework
> Apple Silicon relevance: Medium -- useful for understanding deployment trade-offs

**MeteoRA: Multiple-tasks Embedded LoRA for MoE** -- ICLR 2025
[OpenReview](https://openreview.net/forum?id=yOOJwR15xg)
> Embeds multiple LoRA adapters into base LLM via MoE gating.
> Expected improvement: 4x speedup with hybrid expert acceleration
> Apple Silicon relevance: Medium

**QoS-Efficient Serving of Multiple MoE LLMs** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/poster/44489)
> Similarity-based expert consolidation across fine-tuned MoE models.
> Expected improvement: Reduced memory footprint for multi-model serving
> Apple Silicon relevance: Medium

---

### 6. Token Pruning / Early Exit

**S-GRPO: Early Exit via Reinforcement Learning in Reasoning Models** -- NeurIPS 2025
Y. Dai et al. | [arXiv](https://arxiv.org/abs/2505.07686)
> RL-based method trains reasoning models to exit thinking early; serial sampling with decaying rewards.
> Expected improvement: 35.4-61.1% token reduction with 0.72-6.08% accuracy improvement
> Apple Silicon relevance: High -- fewer generated tokens = directly faster inference

**EAT: Entropy After </Think> for Reasoning Early Exit** -- NeurIPS 2025 (Workshop)
[arXiv](https://arxiv.org/abs/2509.26522)
> Monitors entropy after appending </think>; variance-based stopping rule under EMA.
> Expected improvement: Up to 21% token reduction on AIME-2025 without accuracy loss
> Apple Silicon relevance: High -- simple, training-free stopping criterion

**Think Clearly: Improving Reasoning via Redundant Token Pruning** -- ICML 2025
Choi et al. (Amazon) | [ICML](https://icml.cc/virtual/2025/51783)
> Attention-based token importance to end-of-thinking token; structure-aware chunk pruning.
> Expected improvement: Improves accuracy on AIME/AMC while reducing tokens
> Apple Silicon relevance: High -- fewer tokens = faster decode

**PAT: Pruning-Aware Tuning for Large Language Models** -- AAAI 2025
Yijiang Liu et al. | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/34649)
> Hybrid Sparsification Modules between Attention and FFN.
> Expected improvement: Surpasses LoRA-64 by 1.26% with 25% weight pruning
> Apple Silicon relevance: High -- 25% pruning directly reduces memory footprint

**HyWIA: Hybrid-grained Weight Importance Assessment for Structured Pruning** -- AAAI 2025
Jun Liu et al. | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/34078)
> Merges fine-grained and coarse-grained importance for end-to-end pruning.
> Expected improvement: +2.82% average accuracy over LLM-Pruner at 50% pruning
> Apple Silicon relevance: High -- 50% structured pruning halves memory and compute

**Learning to Focus: Causal Attention Distillation** -- NeurIPS 2025
[arXiv](https://arxiv.org/abs/2506.07851)
> Gradient-guided identification of confounding tokens; prunes during distillation.
> Expected improvement: Improved reasoning accuracy via token suppression
> Apple Silicon relevance: Medium

**CoreMatching: Co-adaptive Token + Neuron Pruning** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/poster/45781)
> Joint sparse inference leveraging core neurons and core tokens.
> Expected improvement: 5x FLOPs reduction, 10x overall speedup
> Apple Silicon relevance: Medium -- VLM-focused

**HiRED: Attention-Guided Visual Token Dropping** -- AAAI 2025
[AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/32171)
> CLS token attention selects informative visual tokens.
> Expected improvement: 4.7x throughput, 78% latency reduction at 20% token budget
> Apple Silicon relevance: Medium

**VTW: Visual Tokens Withdrawal for Rapid Inference** -- AAAI 2025 (Oral)
Zhihang Lin et al. | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/32567)
> Plug-and-play module withdrawing visual tokens from deeper layers.
> Expected improvement: 40% computational overhead reduction
> Apple Silicon relevance: Medium

**ST3: Spatial-Temporal Visual Token Trimming** -- AAAI 2025
Jiedong Zhuang et al. | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/33201)
> Progressive Visual Token Pruning + Visual Token Annealing; no retraining.
> Expected improvement: ~2x faster, ~30% KV cache memory
> Apple Silicon relevance: Medium

**Fit and Prune: Training-free Visual Token Pruning** -- AAAI 2025
Wenyi Ye et al. | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/34366)
> Statistical optimization for token pruning minimizing attention divergence.
> Expected improvement: 33.2% computation reduction at 40% visual FLOPs reduction
> Apple Silicon relevance: Medium

**AST: Semi-Structural Adaptive Sparse Training** -- AAAI 2025
Weiyu Huang et al. | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/34592)
> 2:4 semi-structured sparse models with knowledge distillation.
> Expected improvement: 0.6 perplexity gap to dense at 2:4 sparsity
> Apple Silicon relevance: Medium -- ANE does not natively accelerate N:M sparsity

---

### 7. Parallel Decoding / Multi-Token Prediction

**L-MTP: Leap Multi-Token Prediction Beyond Adjacent Context** -- NeurIPS 2025
X. Liu et al. | [arXiv](https://arxiv.org/abs/2505.17505) | [GitHub](https://github.com/Xiaohao-Liu/L-MTP)
> Predicts multiple future tokens beyond adjacent positions.
> Expected improvement: Improved multi-token prediction quality for speculative decoding
> Apple Silicon relevance: Medium

**MuToR: Multi-Token Prediction Needs Registers** -- NeurIPS 2025
A. Gerontopoulos et al. | [OpenReview](https://openreview.net/pdf?id=WDdBhcwzGe)
> Interleaves register tokens for d-steps-ahead prediction.
> Expected improvement: Better prediction quality for multi-token heads
> Apple Silicon relevance: Medium

**Accelerating Diffusion LLMs via Adaptive Parallel Decoding** -- NeurIPS 2025
D. Israel et al. (UCLA) | [PDF](https://starai.cs.ucla.edu/papers/IsraelNeurIPS25.pdf)
> Dynamic adjustment of parallel token count for diffusion-based LMs.
> Expected improvement: Adaptive parallelism over fixed sampling
> Apple Silicon relevance: Medium

**Beyond Next Token Prediction: Patch-Level Training** -- ICLR 2025
> Compresses K tokens into patches; model predicts all tokens in next patch.
> Expected improvement: 0.5x training cost; multi-token generation potential
> Apple Silicon relevance: Medium

---

### 8. Prompt Caching / Prefix Sharing

**KVLink: KV Cache Reuse Across Documents** -- NeurIPS 2025
H. Chang et al. (UCSB NLP) | [arXiv](https://arxiv.org/abs/2502.16002) | [GitHub](https://github.com/UCSB-NLP-Chang/KVLink)
> Precomputes KV cache per document; concatenates at inference with positional adjustment.
> Expected improvement: 96% TTFT reduction; 4% accuracy improvement
> Apple Silicon relevance: High -- directly applicable to local RAG/multi-document

**ILRe: Intermediate Layer Retrieval for Context Compression** -- ICLR 2025
[OpenReview](https://openreview.net/forum?id=GiI6tPrPAG)
> Encodes context via chunked prefill to one intermediate layer; recalls via attention scores.
> Expected improvement: 180x TTFT speedup; O(L) prefill; 1M tokens in <30s
> Apple Silicon relevance: High -- dramatically reduces prefill compute

**DeFT: Decoding with Flash Tree-Attention** -- ICLR 2025
LINs Lab | [GitHub](https://github.com/LINs-lab/DeFT)
> Prefix-aware KV cache partitioning; avoids redundant KV cache loading.
> Expected improvement: 2.23x decoding / 3.59x attention latency speedup
> Apple Silicon relevance: High -- prefix reuse directly relevant to prompt caching

**Prompt Compression with Context-Aware Sentence Encoding** -- AAAI 2025
Barys Liskavets et al. (Workday) | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/34639)
> Context-aware sentence encoding compresses prompts before LLM inference.
> Expected improvement: Reduced prompt length (compression-dependent)
> Apple Silicon relevance: High -- reduces prefill and KV cache size

**AttnComp: Leveraging Attention to Compress Prompts** -- AAAI 2025
[AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/34800)
> Causal cross-attention evaluates token significance; graph-based clustering.
> Expected improvement: Effective long-context processing with compressed prompts
> Apple Silicon relevance: High -- long-context prompt compression critical for local models

**KVCOMM: Cross-context KV-cache Communication** -- NeurIPS 2025
H. Ye et al. | [arXiv](https://arxiv.org/abs/2510.12872) | [GitHub](https://github.com/FastMAS/KVCOMM)
> Cross-context cache reuse for multi-agent systems.
> Expected improvement: 70%+ cache reuse; 7.8x TTFT in 5-agent settings
> Apple Silicon relevance: Medium

---

### 9. Architecture-Level Efficiency

**xLSTM 7B: A Recurrent LLM for Fast Inference** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/poster/45587)
> Recurrent architecture with linear compute scaling and constant memory usage.
> Expected improvement: Linear-time inference, constant memory
> Apple Silicon relevance: High -- eliminates quadratic attention entirely

**Cobra: Mamba for Multi-Modal LLMs** -- AAAI 2025
Han Zhao et al. | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/33131)
> Replaces Transformer with Mamba state-space model for multimodal LLMs.
> Expected improvement: 3x-4x faster than LLaVA-Phi
> Apple Silicon relevance: High -- linear complexity, constant memory ideal for unified memory

**Morph-1B: Scaling Inference-Efficient Language Models** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/poster/43602)
> Wider and shallower architecture designed for inference efficiency.
> Expected improvement: 1.8x latency improvement vs comparable models
> Apple Silicon relevance: High -- architectural efficiency benefits all hardware

**Sequence Accumulation and Beyond: Infinite Context on Single GPU** -- AAAI 2025
Wangchunshu Sun et al. | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/34284)
> Sequence accumulation for extremely long context on single GPU.
> Expected improvement: Linear scaling of context length on single device
> Apple Silicon relevance: High -- single-device long-context directly applies

**MeRino: Entropy-Driven Design for Generative LMs on IoT Devices** -- AAAI 2025
Youpeng Zhao et al. | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/34445)
> Information-entropy framework for mobile-friendly model design.
> Expected improvement: 4.9x faster with 5.5x model size reduction
> Apple Silicon relevance: High -- design principles applicable to ANE and Metal

**Can Compressed LLMs Truly Act?** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/poster/43871)
> Empirical evaluation of agentic capabilities under compression.
> Expected improvement: Benchmark for compression quality decisions
> Apple Silicon relevance: Medium

---

### 10. Serving / Scheduling

**Flexible and Efficient Grammar-Constrained Decoding** -- ICML 2025
[ICML](https://icml.cc/virtual/2025/poster/45613)
> New algorithm for constrained decoding with fast preprocessing.
> Expected improvement: 17.71x faster offline preprocessing
> Apple Silicon relevance: High -- structured output useful for local tool-calling

**Loquetier: Virtualized Multi-LoRA Framework** -- NeurIPS 2025
NJU DeepEngine | [arXiv](https://arxiv.org/abs/2511.00101) | [GitHub](https://github.com/NJUDeepEngine/Loquetier)
> Virtualizes LoRA adapters on shared base model; fused forward paths.
> Expected improvement: 3.0x throughput on inference tasks
> Apple Silicon relevance: Medium -- multi-LoRA serving more of a server concern

**MultiLevelOT: Cross-Tokenizer Knowledge Distillation** -- AAAI 2025 (Oral)
Xiang Cui et al. | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/34543)
> Optimal transport alignment across different tokenizers for distillation.
> Expected improvement: Enables distilling into compact models regardless of tokenizer
> Apple Silicon relevance: High -- enables creating compact models for Apple Silicon

---

## Appendix: Apple Silicon-Specific References

**Apple MLX + M5 Neural Accelerators** -- NeurIPS 2025 (Demo)
Apple ML Research | [Blog](https://machinelearning.apple.com/research/neurips-2025) | [M5 blog](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
> MLX framework leverages M5 GPU Neural Accelerators; demo of 1T-parameter model on 4x Mac Studio.
> Expected improvement: Up to 4x TTFT on M5 vs M4

**Profiling LLM Inference on Apple Silicon: A Quantization Perspective** -- arXiv (NeurIPS-adjacent)
[arXiv](https://arxiv.org/abs/2508.08531)
> Investigates unified memory and quantization implications for on-device inference.
> Expected improvement: Profiling data and quantization guidelines

**Sigmoid Self-Attention + FlashSigmoid** -- ICLR 2025
Apple | [GitHub](https://github.com/apple/ml-sigmoid-attention)
> Apple's own research on sigmoid attention with hardware-aware kernel.
> Expected improvement: 17% kernel speedup over FlashAttention-2

**CommVQ** -- ICML 2025
UMass/Apple | [ICML](https://icml.cc/virtual/2025/poster/43828)
> Apple co-authored; RoPE-commutative VQ for 1-bit KV cache.
> Expected improvement: 87.5% KV cache size reduction
