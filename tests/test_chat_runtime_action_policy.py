# SPDX-License-Identifier: Apache-2.0
"""A redundant workspace cd must preserve every check on the remaining code."""

import pytest

from rapid_mlx.chat_runtime.tools.action_policy import execution_policy
from rapid_mlx.chat_runtime.tools.action_tools import run_process


@pytest.mark.parametrize("separator", ["&&", ";"])
@pytest.mark.parametrize(
    "target", ["{root}", "{root}/", "{root}/.", "{root}/subdir/..", ".", "./"]
)
@pytest.mark.parametrize("command", ["git diff", "touch result.txt", "python3 safe.py"])
def test_leading_workspace_cd(target, separator, command, tmp_path):
    (tmp_path / "subdir").mkdir()
    (tmp_path / "safe.py").write_text("print(42)")
    text = f" \tcd {target.format(root=tmp_path)} {separator} {command}"
    assert execution_policy(text, str(tmp_path)) == execution_policy(
        command, str(tmp_path)
    )
    assert not execution_policy(text, str(tmp_path)).prompt


@pytest.mark.parametrize("separator", ["&&", ";"])
def test_leading_symlink_workspace_cd(separator, tmp_path):
    root = tmp_path / "work"
    root.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    assert not execution_policy(f"cd {alias} {separator} git diff", str(root)).prompt


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /",
        "git reset --hard",
        "curl https://example.test",
        "cat .env",
        "touch ../outside",
        "touch .git/config",
        "echo $(pwd)",
        "python3 unsafe.py",
        "pushd .; git diff",
    ],
)
@pytest.mark.parametrize("separator", ["&&", ";"])
def test_workspace_cd_preserves_remainder_policy(command, separator, tmp_path):
    (tmp_path / "unsafe.py").write_text("import os; os.unlink('victim')")
    direct = execution_policy(command, str(tmp_path))
    prefixed = execution_policy(f"cd {tmp_path} {separator} {command}", str(tmp_path))
    assert direct.prompt and not direct.allow_conversation
    assert prefixed == direct


@pytest.mark.parametrize(
    "command",
    [
        "cd {root}/subdir && git diff",
        "cd {root}/.. && git diff",
        "cd $PWD && git diff",
        'cd "$PWD" && git diff',
        "cd '${{PWD}}' && git diff",
        "cd $(pwd) && git diff",
        "cd `pwd` && git diff",
        "cd {root}/* && git diff",
        "cd {root}/[.] && git diff",
        "cd {root}/{{.,subdir}} && git diff",
        "cd - && git diff",
        "cd ~ && git diff",
        "cd ~user && git diff",
        "cd && git diff",
        "cd -- {root} && git diff",
        "pushd {root} && git diff",
        "echo ok && cd {root} && git diff",
        "cd {root}",
        "cd {root};",
        "cd {root} &&",
        "cd {root} || git diff",
        "cd {root} | git diff",
        "cd {root} & git diff",
        "cd {root} > output && git diff",
        "cd {root} extra && git diff",
        "cd {root} && cd {root} && git diff",
    ],
)
def test_other_cd_still_requires_approval(command, tmp_path):
    (tmp_path / "subdir").mkdir()
    policy = execution_policy(command.format(root=tmp_path), str(tmp_path))
    assert policy.prompt and not policy.allow_conversation


def test_cd_rejects_ambiguous_symlink_parent_and_cdpath(tmp_path):
    root = tmp_path / "work"
    root.mkdir()
    (root / "subdir").mkdir()
    (tmp_path / "alias").symlink_to(root / "subdir", target_is_directory=True)
    # Physical resolution reaches root, but logical shell cd reaches tmp_path.
    assert execution_policy(f"cd {tmp_path}/alias/.. && git diff", str(root)).prompt
    (root / "alias").symlink_to(root, target_is_directory=True)
    # A bare relative name can use an owner-opted-in CDPATH instead of cwd.
    assert execution_policy("cd alias && git diff", str(root)).prompt
    assert not execution_policy("cd ./alias && git diff", str(root)).prompt


@pytest.mark.parametrize("target", ["'{root}'", '"{root}"', r"{root}/\."])
def test_cd_does_not_unquote_or_unescape_target(target, tmp_path):
    command = f"cd {target.format(root=tmp_path)} && git diff"
    assert execution_policy(command, str(tmp_path)).prompt


def test_cd_resolution_error_keeps_approval(tmp_path):
    loop = tmp_path / "loop"
    loop.symlink_to(loop)
    assert execution_policy(f"cd {loop} && git diff", str(tmp_path)).prompt


@pytest.mark.asyncio
async def test_workspace_cd_execution_uses_workspace_cwd(tmp_path):
    root = tmp_path / "work"
    root.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)

    async def emit(event):
        pass

    command = "cat marker.txt"
    (root / "marker.txt").write_text("workspace contents\n")
    direct = await run_process(
        ["/bin/sh", "-c", command], str(root), 10, emit, "direct"
    )
    for target in (root, alias):
        for separator in ("&&", ";"):
            prefixed = f"cd {target} {separator} {command}"
            assert not execution_policy(prefixed, str(root)).prompt
            result = await run_process(
                ["/bin/sh", "-c", prefixed], str(root), 10, emit, "prefixed"
            )
            assert result["exit_code"] == direct["exit_code"] == 0
            assert result["output"] == direct["output"] == "workspace contents\n"
