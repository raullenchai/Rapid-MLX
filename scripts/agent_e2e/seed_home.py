#!/usr/bin/env python3
"""Seed an empty dedicated HOME with fake pre-existing cloud configuration."""

import argparse
import json
from pathlib import Path

FILES = {
    ".codex/config.toml": 'model = "gpt-5"\nmodel_provider = "openai"\n',
    ".claude/settings.json": json.dumps(
        {"env": {"ANTHROPIC_API_KEY": "sk-ant-fake-e2e"}}
    )
    + "\n",
    ".pi/agent/models.json": json.dumps(
        {
            "providers": {
                "anthropic": {
                    "baseUrl": "https://api.anthropic.com",
                    "api": "anthropic-messages",
                    "apiKey": "sk-ant-fake-e2e",
                    "models": [{"id": "claude-sonnet-4"}],
                }
            }
        }
    )
    + "\n",
    ".pi/agent/auth.json": json.dumps(
        {"anthropic": {"type": "api_key", "key": "sk-ant-fake-e2e"}}
    )
    + "\n",
    ".config/opencode/opencode.json": json.dumps(
        {
            "model": "openai/gpt-5",
            "provider": {
                "openai": {"options": {"apiKey": "sk-fake-e2e"}},
            },
        }
    )
    + "\n",
    ".hermes/config.yaml": 'model:\n  provider: "openai"\n  default: "gpt-5"\n',
    ".config/goose/config.yaml": "GOOSE_PROVIDER: openai\nGOOSE_MODEL: gpt-5\nOPENAI_API_KEY: sk-fake-e2e\n",
    ".qwen/settings.json": json.dumps(
        {
            "$version": 4,
            "security": {"auth": {"selectedType": "qwen-oauth"}},
            "model": {"name": "qwen3-coder-plus"},
            "modelProviders": {
                "openai": [
                    {
                        "id": "qwen3.5-9b-4bit",
                        "baseUrl": "https://cloud.invalid/v1",
                        "envKey": "FAKE_CLOUD_KEY",
                    }
                ]
            },
            "ui": {"autoModeAcknowledged": True},
        }
    )
    + "\n",
    ".qwen/oauth_creds.json": json.dumps(
        {
            "access_token": "fake-e2e-token",
            "refresh_token": "fake-e2e-refresh",
            "expiry_date": 4102444800000,
        }
    )
    + "\n",
    ".config/crush/crush.json": json.dumps(
        {
            "providers": {
                "openai": {
                    "id": "openai",
                    "name": "OpenAI",
                    "type": "openai",
                    "api_key": "sk-fake-e2e",
                    "models": [{"id": "gpt-5", "name": "gpt-5"}],
                }
            },
            "models": {"large": {"provider": "openai", "model": "gpt-5"}},
        }
    )
    + "\n",
    ".aider.conf.yml": "model: openai/gpt-5\nopenai-api-key: sk-fake-e2e\n",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("home", type=Path)
    args = parser.parse_args()
    args.home.mkdir(parents=True, exist_ok=True)
    if any(args.home.iterdir()):
        parser.error("HOME must be empty; never overwrite real user configuration")
    for relative, content in FILES.items():
        target = args.home / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        target.chmod(0o600)
    print(f"Seeded {len(FILES)} fake cloud config files in {args.home}")


if __name__ == "__main__":
    main()
