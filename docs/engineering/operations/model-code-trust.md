# Loading models that provide Python code

Rapid-MLX disables checkpoint-provided Python by default, for both local folders
and Hugging Face repositories. A local path is not proof that the code is trusted.
This applies to a model's `model_file` and tokenizer/processor remote-code
selection. Models supported by built-in implementations continue to load without
granting repository-code execution.

After reviewing and trusting the model repository, an operator can explicitly
enable code for the process:

```sh
RAPID_MLX_TRUST_REMOTE_CODE=1 rapid-mlx serve <model>
```

This is process-wide permission, including secondary models loaded by that
process. Prefer reviewed local snapshots or immutable revisions. Python API
callers that expose `trust_remote_code` can pass `True` explicitly instead.
An explicit API `False` denies execution even when the environment enables it;
`RAPID_MLX_TRUST_REMOTE_CODE=0` overrides API permission and always denies it.

Permission does not disable path containment. `model_file` must name a Python
file inside the model directory, with support for ordinary Hugging Face snapshot
links into that repository's cache blobs. Configured cache repository entries may themselves be symlinked to another
storage tier; these retain same-repository blob containment. Arbitrary local
folders cannot gain that exception by imitating the cache layout.

Repository-code permission is not a sandbox: permitted code executes with the
server process's privileges. The default denial is enforced before upstream
loaders can import `model_file`, because their tokenizer trust option alone does
not control this import. Remote loading inspects the snapshot that will actually
be passed to the loader.

For compatibility recovery, first use a supported built-in adapter or explicitly
authorize a reviewed model. Do not restore globally permissive defaults.
Vector owns model compatibility follow-up; Atlas owns release integration.
