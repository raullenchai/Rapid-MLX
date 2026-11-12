<!-- Draft for the next release. Rename in the version-bump PR. -->

- Upgraded the image runtime to mflux 0.23 with no generation behavior change, keeping Pillow below 13 for compatibility with supported Python versions.
- Added FLUX 3 Klein and Klein Base image aliases, with up to ten edit references through the API and Desktop.
- Added user LoRA support for image generation and edits: drop .safetensors files into ~/.rapid-mlx/loras and pick them by name.
