"""Small Qwen3.5-9B PPT feasibility probe on a CUDA GPU.

Independent implementation of Algorithm 1 / Appendix B of arXiv:2609.38104.
Defaults are a short-output GSM8K probe, not the paper's AIME/GPQA/LCB run.
Prints one JSON result per item; run with `colab run --gpu A100 -s ppt-qwen9b`.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from dataclasses import dataclass
from urllib.parse import urlencode
from urllib.request import urlopen

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a8b"
DATASET = "openai/gsm8k"


@dataclass
class Record:
    tokens: list[int]
    logp: list[float]
    zeta: list[list[float]]

    def prefix(self, length: int) -> Record:
        return Record(
            self.tokens[:length],
            self.logp[:length],
            [values[:length] for values in self.zeta],
        )


class Sampler:
    def __init__(self, model, tokenizer, powers: tuple[float, ...]):
        self.model = model
        self.tokenizer = tokenizer
        self.powers = powers
        model_eos = model.generation_config.eos_token_id
        ids = model_eos if isinstance(model_eos, list) else [model_eos]
        self.eos = tuple(sorted({*ids, tokenizer.eos_token_id} - {None}))
        self.generated_tokens = 0

    def empty(self) -> Record:
        return Record([], [], [[] for _ in self.powers])

    def extend(
        self, prompt: list[int], record: Record, alpha: float, horizon: int
    ) -> Record:
        if len(record.tokens) >= horizon or (
            record.tokens and record.tokens[-1] in self.eos
        ):
            return record
        input_ids = torch.tensor([prompt + record.tokens], device="cuda")
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=horizon - len(record.tokens),
                do_sample=True,
                temperature=1 / alpha,
                top_k=0,
                top_p=1.0,
                min_p=None,
                repetition_penalty=1.0,
                return_dict_in_generate=True,
                output_logits=True,
                eos_token_id=list(self.eos),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        tokens = output.sequences[0, input_ids.shape[-1] :].tolist()
        assert len(tokens) == len(output.logits)
        self.generated_tokens += len(tokens)
        logp = list(record.logp)
        zeta = [list(values) for values in record.zeta]
        for start in range(0, len(tokens), 32):
            stop = start + 32
            raw = torch.cat(output.logits[start:stop], dim=0).float()
            base_logp = raw.log_softmax(dim=-1)
            chosen = torch.tensor(tokens[start:stop], device="cuda")
            logp.extend(base_logp.gather(1, chosen[:, None]).flatten().tolist())
            for alpha_k, values in zip(self.powers, zeta, strict=True):
                values.extend(torch.logsumexp(alpha_k * base_logp, dim=-1).tolist())
        return Record(record.tokens + tokens, logp, zeta)

    def prompt_ids(self, question: str) -> list[int]:
        prompt = question + (
            "\nSolve concisely and put only the final numeric answer in \\boxed{}."
        )
        encoded = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        )
        return encoded["input_ids"][0].tolist()

    def baseline(self, prompt: list[int], horizon: int, seed: int) -> dict:
        seed_all(seed)
        self.generated_tokens = 0
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        record = self.extend(prompt, self.empty(), 1.0, horizon)
        return self.summary(record, started)

    def ppt(self, prompt: list[int], horizon: int, rounds: int, seed: int) -> dict:
        seed_all(seed)
        self.generated_tokens = 0
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        chains = [
            self.extend(prompt, self.empty(), power, horizon) for power in self.powers
        ]
        attempted = accepted = swaps = 0
        for _ in range(rounds):
            for k, power in enumerate(self.powers):
                current = chains[k]
                restart = random.randrange(horizon)
                if restart >= len(current.tokens):
                    continue  # Fixed-horizon post-EOS padding makes this a self-transition.
                candidate = self.extend(prompt, current.prefix(restart), power, horizon)
                attempted += 1
                log_ratio = sum(candidate.zeta[k][restart:]) - sum(
                    current.zeta[k][restart:]
                )
                if math.log(random.random()) < min(0.0, log_ratio):
                    chains[k] = candidate
                    accepted += 1
            for k in range(len(chains) - 1):
                low, high = chains[k : k + 2]
                log_ratio = (self.powers[k + 1] - self.powers[k]) * (
                    sum(low.logp) - sum(high.logp)
                )
                if math.log(random.random()) < min(0.0, log_ratio):
                    chains[k], chains[k + 1] = high, low
                    swaps += 1
        result = self.summary(chains[-1], started)
        result.update(
            {"mh_attempted": attempted, "mh_accepted": accepted, "swaps": swaps}
        )
        return result

    def summary(self, record: Record, started: float) -> dict:
        output = self.tokenizer.decode(record.tokens, skip_special_tokens=True)
        return {
            "answer": extract_answer(output),
            "completion_tokens": len(record.tokens),
            "generated_tokens": self.generated_tokens,
            "seconds": round(time.monotonic() - started, 2),
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "output": output,
        }


def extract_answer(output: str) -> str | None:
    boxed = re.findall(r"\\boxed\{([^}]*)\}", output)
    candidate = boxed[-1].strip() if boxed else output.strip()
    candidate = candidate.replace(",", "")
    return candidate if re.fullmatch(r"-?\d+", candidate) else None


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fetch_rows(offset: int, limit: int) -> list[dict]:
    query = urlencode(
        {
            "dataset": DATASET,
            "config": "main",
            "split": "test",
            "offset": offset,
            "length": limit,
        }
    )
    with urlopen(
        f"https://datasets-server.huggingface.co/rows?{query}", timeout=30
    ) as response:
        return json.load(response)["rows"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=512)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--powers", type=float, nargs="+", default=[1.25, 1.4, 1.6])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA GPU required")
    if args.horizon < 1 or args.rounds < 0 or args.limit < 1:
        parser.error("horizon and limit must be positive; rounds must be nonnegative")
    powers = tuple(args.powers)
    if (
        len(powers) < 2
        or powers[0] < 1
        or any(a >= b for a, b in zip(powers, powers[1:]))
    ):
        parser.error("powers must be an increasing ladder of at least two values >= 1")
    model_id = "Qwen/Qwen3.5-9B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=MODEL_REVISION)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, revision=MODEL_REVISION, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    sampler = Sampler(model, tokenizer, powers)
    print(
        "CONFIG "
        + json.dumps(
            {
                "model": model_id,
                "revision": MODEL_REVISION,
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "dataset": DATASET,
                "offset": args.offset,
                "limit": args.limit,
                "horizon": args.horizon,
                "rounds": args.rounds,
                "powers": powers,
                "thinking": False,
            },
            default=str,
        ),
        flush=True,
    )
    for item in fetch_rows(args.offset, args.limit):
        row = item["row"]
        gold = row["answer"].split("####")[-1].replace(",", "").strip()
        prompt = sampler.prompt_ids(row["question"])
        index = item["row_idx"]
        baseline = sampler.baseline(prompt, args.horizon, 1000 + index)
        ppt = sampler.ppt(prompt, args.horizon, args.rounds, 2000 + index)
        print(
            "RESULT "
            + json.dumps(
                {"row": index, "gold": gold, "baseline": baseline, "ppt": ppt},
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
