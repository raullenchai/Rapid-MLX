# CLI installer trust boundaries

`rapid-mlx upgrade` upgrades an `install.sh` environment with its own Python's
`-m pip install --upgrade rapid-mlx`. It no longer downloads or runs a website
shell script. If pip is absent and uv is available, it uses
`uv pip install --python <running-python> --upgrade rapid-mlx` instead.
An environment without either installer reports the pip error; automatic upgrade does
not repair or replace its interpreter.

`install.sh stable` and explicit versions use the configured package index and
stop on errors. They never silently switch to GitHub source builds. Retry after
correcting network, index, or dependency-resolution errors; package-manager
error details remain visible.

For an intentional source build, review a full Git commit SHA, then run:

```sh
RAPID_MLX_GIT_COMMIT=<reviewed-40-character-commit-SHA> bash install.sh latest
```

`latest` without a full SHA now fails before installation starts. The name is
retained for compatibility, but no longer means mutable repository HEAD. Version
arguments install the exact package version; there is no tag fallback.

These changes remove an extra mutable shell-execution channel and implicit
source switching. They do not establish artifact signatures or a dependency
lock. Package-index installations still trust the configured index, publishers,
and dependency packages. A commit pin fixes source identity but does not prove
who authorized it. Initial bootstrap by downloading `install.sh` still trusts
the download host; inspect a separately obtained installer before executing it
when that trust is insufficient. No production signing or publishing changes
are included.

For rollback, install a previously verified explicit package version from the
configured index. Do not restore automatic HEAD fallback to work around an
outage. Release integration and any stronger signed-artifact distribution remain
with Atlas and Harbor.
