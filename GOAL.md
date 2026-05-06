# GOAL

Corrigir fluxo agentic do Qwen3.6-35B-A3B MTPLX no `lightning-mlx` para o prompt:

```text
create the snake game with html and typescript
```

## Success Criteria

1. Comando plain funciona:

```bash
uv run lightning-mlx serve /Users/samuelfajreldines/dev/models/Qwen3.6-35B-A3B-4bit-MTPLX-Optimized-Speed
```

2. `pi --provider local --model local --no-session -p "create the snake game with html and typescript"` cria projeto funcional.
3. Stream nao vaza reasoning/prolixidade inutil.
4. Tool-use eficiente: poucas chamadas, sem loops longos de 2048 tokens.
5. Artefatos finais abrem/compilam: HTML + TypeScript ou build equivalente.
6. Validacao real: build/check ou browser smoke.
7. Mudanca generica, nao prompt-specific.
8. O que melhora performance/qualidade fica. O que piora sai.

## Current Baseline

- Plain `hi` ja ficou bom: resposta curta, sem reasoning leak.
- Pi snake ainda falha: gera pensamento/tool-use ruim, demora, e cria artefatos incompletos ou invalidos.
- `--no-thinking` piorou. Nao usar como fix principal.
- Structured-CoT prompt-only piorou em teste local. Nao manter sem prova melhor.
- MTP no serve aparece com `accepted=0`; investigar se custo sem ganho no fluxo tool-heavy.

## Active Fix Direction

1. Capar `max_tokens` em requests com tools desde a primeira chamada, sem truncar demais.
2. Detectar formatos quebrados de tool call, incluindo `[Calling tool=...]`.
3. Tratar texto com intencao de tool call como deferred tool-use, nao resposta final.
4. Se candidato piorar artefatos ou eficiencia, reverter.
5. Validar com testes unitarios + serve real + Pi real + artefatos.

## Current Candidate

Candidate E:

- Manter parser/deferred fix generico para `[Calling tool=...]`.
- Cap conservador de tool turns em 2048 tokens desde a primeira request com tools.
- Evitar caps agressivos de 1024/512 porque Candidate A gerou artefato invalido.
- Proximo passo: rodar testes focados, servir modelo, executar Pi snake, validar arquivos.

## Live Report

Atualizar `REPORT.md` durante cada tentativa com:

- mudanca feita;
- testes rodados;
- resultado do serve/Pi;
- arquivos gerados;
- decisao: keep/revert/next.
