# Terminal agent acceptance

Run this only on a leased Apple Silicon test Mac. Use an account or HOME that
contains realistic, pre-existing **fake** cloud-provider settings for all nine
agents. `python3 scripts/agent_e2e/seed_home.py "$TEST_HOME"` seeds an **empty**
HOME and refuses to overwrite one with any content. It includes dummy login
files for two agents; those files only simulate stored state, not a valid login.
Check their syntax against the installed CLI versions before the run. Do not
copy real credentials. Keep the machine's existing Rapid server
and port 7659 untouched. Reserve a separate loopback port for this run.

Install the latest agent CLIs by the commands in their current upstream docs,
and record the exact install commands and `--version` output in the private
run directory. Install the released CLI with the published `install.sh` into
the test HOME. Then install the current main branch with
`uv tool install --force 'git+https://github.com/raullenchai/Rapid-MLX.git@main'`
using that same HOME. Point `--rapid` at the resulting tool binary. Preserve
the two install logs; the script itself records installed agent versions.

From this directory, one command creates `matrix.md` and `results.json`:

```sh
python3 scripts/agent_e2e/run.py \
  --home "$TEST_HOME" \
  --rapid "$TEST_HOME/.local/bin/rapid-mlx" \
  --port "$LEASED_PORT" \
  --output "$PRIVATE_RUN_DIR"
```

The runner drives `rapid-mlx launch <agent>` through a pseudo-terminal. It
asks for a file write and readback, then a second edit and readback, repeats
the launch in a new process, starts the native agent after `--remove`, and
compares byte hashes and modes of pre-existing user configuration. It leaves
raw terminal transcripts and a metadata-only request log in the private run
directory. Inspect `day-one.pty`, `day-two.pty`, and `normal-mode.pty` for
errors and ugly output; do not publish raw transcripts. A server-side ASGI tap
records only time, agent, route, requested model, HTTP status, and prompt
tokens if the response supplies usage. It never stores prompts or keys.

`Works=Y` requires a real file edit on both launches, a real second edit,
the requested model observed on the server, accepted auth, untouched user
configuration, normal mode opening, and successful `--remove`. Missing server
usage remains `—`; do not replace it with a model name from a config file.
The first-reply timings are first terminal output after sending the prompt;
inspect the transcript when startup redraw or input echo makes this an
underestimate. An exact reply timing requires a parser for that agent's
terminal protocol. A normal-mode open shows only that the ordinary CLI starts
with the fake cloud configuration; it does not prove cloud authentication.

The first agent must start with no listener on the reserved port. Later
agents deliberately reuse the same server. `--remove` removes only the
private Rapid profile, not the user configuration or the server. At the end,
stop only the server PID created by this test and release the machine lease
after verifying cleanup. The runner never kills a server.
