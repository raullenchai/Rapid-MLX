"""Write a user message whose chat-templated prompt is TARGET tokens (default 1008).

usage: mkprompt.py MODEL_DIR OUT.txt [TARGET]
"""
import sys

from transformers import AutoTokenizer, PreTrainedTokenizerFast

path, out = sys.argv[1], sys.argv[2]
target = int(sys.argv[3]) if len(sys.argv) > 3 else 1008
try:
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
except Exception:
    tok = PreTrainedTokenizerFast.from_pretrained(path, local_files_only=True)
FILL = ("The history of cartography spans many civilizations, from clay tablets in "
        "Mesopotamia to portolan charts of medieval sailors and satellite surveys. ")
TAIL = "\n\nSummarize the passage above in detail."
body = tok.encode(FILL * 80, add_special_tokens=False)


def templated(n):
    text = "[0.0] " + tok.decode(body[:n]) + TAIL  # srvbench prefixes "[r.i] "
    try:
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True)
    except Exception:
        ids = tok.encode(text)
    return len(ids)


n = target
for _ in range(40):
    got = templated(n)
    if got == target:
        break
    n += target - got
text = tok.decode(body[:n]) + TAIL
open(out, "w").write(text)
print(f"content_tokens={n} templated={templated(n)} target={target}")
