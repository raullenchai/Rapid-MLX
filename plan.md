# Viabilidade: TurboQuant para 256k de contexto com MTPLX em rapid-mlx

## Contexto

**Pergunta original:** "É possivel implementar o turboquant aqui para eu ter contexto de 256k sem perder performance (tok/s) com o mtplx?"

**Hardware alvo:** MacBook Pro M5 Max 128 GB
**Modelo alvo:** Qwen3.6-27B-4bit MTPLX
**Constraint declarada:** ≤15% perda de tok/s aceitável em troca de +4x contexto
**Escopo deste passo:** Análise de viabilidade — não implementar.

Este documento responde se a feature é viável com o constraint dado, qual o caminho técnico real, e o que recomendamos antes de gastar tempo de implementação.

---

## Estado atual do código

### 1. TurboQuant já existe — mas só no prefix cache (cold)

`vllm_mlx/turboquant.py:298` (`TurboQuantKVCache`) é **codec de armazenamento**, não cache ativo:

> "K stays FP16. V is compressed to 3-4 bits using rotation + Lloyd-Max.
> This class is used in the prefix cache (store/fetch), **not during model forward passes**."

Codec atual: encode/decode operam no tensor inteiro (`turboquant_encode` em `vllm_mlx/turboquant.py:163`). Não suporta append incremental por token (que é o que decode precisa). A decompressão envolve 3 passos sequenciais não fundidos:

1. Unpack nibbles + lookup codebook Lloyd-Max
2. Reshape em grupos + denormalize (× scale + mean)
3. Inverse rotation: `rotated.astype(float32) @ Q` em `vllm_mlx/turboquant.py:288`

Integração só na borda store/fetch do prefix cache em `vllm_mlx/memory_cache.py:594` (`_turboquant_compress_cache`) e `memory_cache.py:625` (`_turboquant_decompress_cache`). Após fetch, cache vai dequantizado para FP16 antes do forward.

### 2. Caminho ativo de decode usa FP16 ou mlx-lm `QuantizedKVCache`

- MTPLX (`vllm_mlx/speculative/mtp_generate.py:138`) recebe `cache` de `make_prompt_cache(model)` → default FP16 `KVCache` do mlx-lm.
- Flag `--kv-cache-quantization` (`vllm_mlx/scheduler.py:83`, `cli.py:434`) ativa quantização padrão do mlx-lm (4 ou 8 bit, fused) via `layer.to_quantized()` — esse caminho é fundido com `mx.fast.scaled_dot_product_attention` e tem custo de dequant ≈ zero porque o kernel Metal trata índices quantizados direto.
- Flag `--kv-cache-turboquant` (`cli.py:401`) é mutuamente exclusiva e hoje só liga a compressão **no prefix cache**.

### 3. Atenção real

`vllm_mlx/attention.py` é um shim — atenção verdadeira é resolvida pelo mlx-lm interno via `mx.fast.scaled_dot_product_attention` (linha 229 do shim, mas o caminho que importa é dentro do mlx-lm). Não há kernel customizado neste repo.

---

## Matemática de memória — Qwen3.6-27B em 256k tokens

Estimativas para arquitetura típica Qwen3 27B (64 layers, 8 KV heads GQA, head_dim 128):

KV/token/layer = 2(K+V) × 8 × 128 × 2 bytes = **4 096 bytes/token/layer**
KV total a 256k × 64 layers = **67.1 GB em FP16**

| Cenário | KV size | Total memória (peso 4-bit + KV + overhead) | Cabe em 128 GB? |
|---|---|---|---|
| FP16 KV, 256k | 67 GB | ~84 GB | Tight, mas cabe — pouco buffer |
| mlx-lm `QuantizedKVCache` 4-bit, 256k | ~17 GB | ~33 GB | Sim, com folga |
| TurboQuant V-only 3-bit + K FP16, 256k | ~46 GB | ~62 GB | Sim |
| TurboQuant K+V 3-bit (hipotético, código não existe), 256k | ~25 GB | ~41 GB | Sim |

**Ponto importante:** memória **não é o gargalo decisivo** em 128 GB — mlx-lm 4-bit já resolve isso. O gargalo real em 256k é **bandwidth de leitura do KV** durante decode.

---

## Análise de tok/s a 256k

Decode em contextos longos é **memory-bandwidth-bound**: cada step lê o KV inteiro de todas as layers. M5 Max tem ~410-540 GB/s de bandwidth unificada (estimativa baseada em M4 Max).

Com MTPLX verificando 2 tokens por step, o custo de leitura é o mesmo (cache lido uma vez, atendendo Q de comprimento 2). Não dobra.

| Cenário | Bytes lidos/step | Step rate teórico (bandwidth-bound) | tok/s estimado (com MTP accept ~50%) |
|---|---|---|---|
| FP16 KV | 67 GB | ~6-8/s | ~9-12 tok/s |
| mlx-lm 4-bit fused | 17 GB | ~24-32/s | ~36-48 tok/s |
| TurboQuant V-only ativo (sem kernel fundido) | 46 GB lidos + dequant compute overhead | ~9-13/s | ~13-19 tok/s |
| TurboQuant V-only ativo (com kernel fundido — não existe) | 46 GB lidos | ~9-13/s | ~13-19 tok/s |
| TurboQuant K+V 3-bit ativo (com kernel fundido — não existe) | 25 GB | ~16-22/s | ~24-33 tok/s |

**Baseline atual** Qwen3.6-27B em contextos curtos: 31.8 tok/s sem MTPLX (`README.md:14`), até ~40 tok/s com MTPLX. Mas isso é com KV pequeno — em 256k a story muda totalmente.

**Conclusão de performance:**
- mlx-lm 4-bit fused é provavelmente **o ganhador inclusive em tok/s** a 256k, porque corta bandwidth em 4x e usa kernel já fundido pelo Metal.
- TurboQuant ativo sem kernel customizado cai abaixo da meta de "≤15% perda" — o overhead de dequant (rotation matmul em FP32 + lookup + denorm) executado por step custa caro.
- TurboQuant ativo **fica competitivo apenas com kernel Metal fundido** (semanas de trabalho) ou com fold de Q em W_v/W_o (mudança per-modelo, frágil).

---

## É viável atender o constraint? (≤15% perda tok/s, 256k, M5 Max 128GB)

### Sim, mas o caminho **não é "implementar TurboQuant ativo"**.

**Caminho recomendado (ordem):**

1. **Ligar `--kv-cache-quantization 4` + `--kv-cache-quantization-group-size 64`** primeiro. Já existe. Já é fundido pelo mlx-lm. Memória cai 4x. Tok/s a 256k provavelmente **melhora** vs FP16 por causa de bandwidth.
2. **Manter TurboQuant V-only no prefix cache** (já está). Beneficia reuso de prompts longos, sem custo no hot path.
3. **Medir** com bench real antes de qualquer trabalho novo — `make benchmark` ou `scripts/benchmark_realworld.py`. Sem números, especulação não vale.

### Quando faria sentido implementar TurboQuant **ativo**?

Apenas se:
- Bench em (1) mostrar que mlx-lm 4-bit degrada qualidade do 27B-MTPLX em tool-calling/reasoning de forma mensurável (p.ex. eval scores caem >2 pontos), E
- A diferença justificar 1-2 semanas de trabalho de kernel + integração.

A vantagem de qualidade do TurboQuant é real (~0.5-1 PPL melhor que quant uniforme em mesmos bits) mas pequena. Para tool-use agentic, mlx-lm 4-bit usualmente é suficiente.

### Se mesmo assim quiser implementar TurboQuant ativo

Trabalho mínimo para chegar perto de "≤15% perda":

1. **Append incremental**: estender `TurboQuantKVCache` com método `update_and_fetch(keys, values)` que quantiza só os tokens novos e concatena com indices/scales/zeros existentes. Hoje `from_kv_cache` reencoda tudo.
2. **Lazy dequant ativo**: implementar interface compatível com `mlx_lm.models.cache.KVCache` (state, offset, trim, is_trimmable, fetch) que retorna V dequantizado on-demand a cada attention call.
3. **Pré-rotação de W_v + fold em W_o**: modificar load do modelo para fundir Q (matriz de rotação) nos pesos. Elimina o matmul de rotação no decode. Isso é per-modelo e frágil — quebra com qualquer mudança de arquitetura.
4. **Kernel Metal fundido** (opcional, dá os ganhos reais): `quantized_attention_lloyd_max(Q, K, V_codes, V_scales, V_zeros, codebook)` em Metal puro. Esse é o item que faz a diferença entre 13 tok/s e 40 tok/s a 256k.

Sem (3) e (4), a perda real será muito maior que 15%.

---

## Arquivos críticos (referência se decidir avançar)

- `vllm_mlx/turboquant.py` — codec atual, encode/decode whole-tensor
- `vllm_mlx/memory_cache.py:594` — wrapping no prefix cache (cold)
- `vllm_mlx/scheduler.py:83-85,1693-1695` — flags KV quant
- `vllm_mlx/cli.py:387-441` — CLI de KV quant
- `vllm_mlx/speculative/mtp_generate.py:97-138` — loop MTPLX, recebe cache pronto
- `vllm_mlx/attention.py:229` — shim de atenção (atenção real é mlx-lm)
- `mlx_lm.models.cache.KVCache` / `QuantizedKVCache` — interface do hot cache (externa ao repo)
- `scripts/benchmark_realworld.py`, `evals/run_eval.py`, `make benchmark` — infra de medição

---

## Próximos passos sugeridos

1. **Antes de qualquer código novo:** rodar bench Qwen3.6-27B-MTPLX em prompt 32k-256k com:
   - FP16 KV (baseline, se couber)
   - `--kv-cache-quantization 4 --kv-cache-quantization-group-size 64`

   Métricas: tok/s decode, eval tool-use, eval reasoning, RAM peak.

2. Se mlx-lm 4-bit atender quality bar → **encerrar**. TurboQuant ativo não vale o custo.

3. Se não atender → planejar TurboQuant ativo com kernel Metal como **projeto de 2-3 sprints**, não feature de uma sessão.

## Verificação final desta análise

- Ler benchmarks históricos: `harness/scorecard/latest.md`, `evals/results/` para ver se algum run a 32k+ já existe com KV quant.
- Confirmar specs Qwen3.6-27B exatas no manifest do modelo (`mtplx_runtime.json`) — número de layers e head_dim exatos podem mover a matemática de memória ±10 GB.
- Validar bandwidth real do M5 Max com microbench antes de confiar nas estimativas teóricas (M5 Max ainda é hardware recente, dados públicos podem estar imprecisos).
