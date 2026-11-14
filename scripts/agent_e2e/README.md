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
`provision_macos.sh` also pulls the 9B model before the timed run. If you
provision manually, run `rapid-mlx pull qwen3.5-9b-4bit` in the test HOME.

From this directory, one command creates `matrix.md` and `results.json`:

```sh
python3 scripts/agent_e2e/run.py \
  --home "$TEST_HOME" \
  --rapid "$TEST_HOME/.local/bin/rapid-mlx" \
  --port "$LEASED_PORT" \
  --output "$PRIVATE_RUN_DIR"
```

The runner drives `rapid-mlx launch <agent>` through a pseudo-terminal. It
answers known local first-run theme, dummy-key confirmation, optional
telemetry, and empty-project Git menus. During task replies, a PTY output
hook also answers these specific confirmations:

- Crush: the complete `Permission Required` dialog for `Tool bash`, with a
  `Path` resolving exactly to the test project directory and the known
  `Allow / Allow for Session / Deny` menu and arrow-key instructions. The
  runner sends the single shortcut `s` for **Allow for Session**, without Enter.
  For `Tool write`, `Tool edit`, and `Tool multiedit`, the same complete dialog
  must show a `File` resolving inside the test project directory. The runner
  sends the single shortcut `a` for **Allow**, without Enter, approving that
  action alone. Both checks resolve symlinks (including macOS
  `/tmp` → `/private/tmp`); files resolving
  outside the project remain unanswered. Crush v0.101.0 discards dialog keys
  during its opening grace period (425 ms of input quiet, capped at 1.5 s).
  The runner schedules each validated answer at least 1.6 s after first
  matching the modal and waits for at least 0.5 s of output quiet. Reads
  continue during this delay; idle polls check the pending answer every
  250 ms. A newer working state or different modal (including incomplete or
  unknown dialogs) cancels the pending answer. Identical redraws retain the
  original deadline and never trigger duplicate answers.
- Aider: `Run shell command? (Y)es/(N)o/(D)on't ask again [Yes]:`,
  also accepting plural `commands?` and the optional `(S)kip all/` before
  `(D)on't ask again` when there are multiple command groups.
  The runner sends `y` then Enter for that command block, without disabling
  later confirmations or selecting skip all. It also sends `y` then Enter for
  `Add command output to the chat? (Y)es/(N)o/(D)on't ask again [Yes]:`,
  asked once per command group with output. In Aider v0.86.2 this question has
  no `(S)kip all` or `(A)ll` option.

The hook handles split reads and ANSI styling, deduplicates idle polls and
repeated redraws of the same question, and continues waiting for the tool reply. It
retains the existing Aider file-creation/news and Qwen tool-confirmation
handling. Each answered Crush/Aider shell or Aider add-output prompt adds a fixed
`permission prompt answered` marker to the private transcript; that label
appears in the row's `ugly_output` in `results.json` and the matrix's Ugly
output column, including when the task succeeds. Unknown permission and login
prompts receive no automatic answer. This hook is limited to the test sessions
launched in the dedicated project directory; it grants no general permission
for other tools, paths, or prompt formats.

The runner asks for a file write and readback of an unprompted challenge value,
then asks for a second edit and fresh challenge readback only if the first turn
succeeds. If the first turn fails or times out, it records the failure and sends
only cancellation controls during shutdown, so the second prompt cannot land in
a pending dialog. It repeats the launch in a new process, starts the native agent
after `--remove`, and compares byte hashes and modes of pre-existing user
configuration. It leaves
raw terminal transcripts and a metadata-only request log in the private run
directory. Inspect `day-one.pty`, `day-two.pty`, and `normal-mode.pty` for
errors and ugly output; do not publish raw transcripts or `results.json`, which
contains raw error tails. The markdown matrix uses fixed error labels. A server-side ASGI tap
records only time, agent, route, requested model, HTTP status, and prompt
tokens if the response supplies usage. It never stores prompts or keys.

`Works=Y` requires a real file edit on both launches, a real second edit,
the requested model observed on the server, accepted auth, untouched user
configuration, normal mode opening, and successful `--remove`. Missing server
usage remains `—`; do not replace it with a model name from a config file.
The reply timings are the time until the verified edit and readback output
settle. They include the runner's five-second quiet period, so they are
upper bounds. An exact visible-answer timing requires a parser for each
agent's terminal protocol. A normal-mode open shows only that the ordinary CLI starts
with the fake cloud configuration; it does not prove cloud authentication.

The first agent must start with no listener on the reserved port. Later
agents deliberately reuse the same server. `--remove` removes only the
private Rapid profile, not the user configuration or the server. At the end,
stop only the server PID created by this test and release the machine lease
after verifying cleanup. The runner never kills a server.
