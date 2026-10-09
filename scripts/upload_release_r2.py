"""Upload release files through the S3 multipart client using the existing token."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import urllib.request
from pathlib import Path


def upload(path: Path, bucket: str, key: str, content_type: str) -> None:
    account = os.environ["CLOUDFLARE_ACCOUNT_ID"]
    token = os.environ["CLOUDFLARE_API_TOKEN"]
    if not re.fullmatch(r"[0-9a-f]{32}", account) or not token:
        raise ValueError("missing or malformed R2 account credentials")
    if bucket != "rapid-desktop-dist" or not key.startswith("rapid-mac/"):
        raise ValueError("release upload must target the Desktop distribution bucket")
    if not path.is_file():
        raise ValueError("release payload is not a file")
    # Cloudflare documents token ID + SHA256(token value) as S3 credentials.
    # Verify the existing token; do not create or broaden any credentials.
    request = urllib.request.Request(
        "https://api.cloudflare.com/client/v4/user/tokens/verify",
        headers={"Authorization": "Bearer " + token},
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        verified = json.load(response)
    result = verified.get("result") or {}
    access_id = result.get("id", "")
    if (
        verified.get("success") is not True
        or result.get("status") != "active"
        or not isinstance(access_id, str)
        or not re.fullmatch(r"[0-9a-f]{32}", access_id)
    ):
        raise ValueError("existing Cloudflare token is not verified active")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    env = dict(os.environ)
    env.pop("AWS_SESSION_TOKEN", None)
    env.pop("AWS_SECURITY_TOKEN", None)
    env.pop("CLOUDFLARE_API_TOKEN", None)
    env.update(
        AWS_ACCESS_KEY_ID=access_id,
        AWS_SECRET_ACCESS_KEY=hashlib.sha256(token.encode()).hexdigest(),
        AWS_DEFAULT_REGION="auto",
        AWS_REQUEST_CHECKSUM_CALCULATION="when_required",
        AWS_RESPONSE_CHECKSUM_VALIDATION="when_required",
    )
    endpoint = f"https://{account}.r2.cloudflarestorage.com"
    # AWS CLI's existing S3 transfer manager handles multipart uploads/retries.
    # Credentials stay in the child environment, never command arguments/logs.
    subprocess.run(
        [
            "aws",
            "s3",
            "cp",
            str(path),
            f"s3://{bucket}/{key}",
            "--endpoint-url",
            endpoint,
            "--content-type",
            content_type,
            "--metadata",
            "sha256=" + digest.hexdigest(),
            "--only-show-errors",
        ],
        env=env,
        check=True,
    )
    head = subprocess.run(
        [
            "aws",
            "s3api",
            "head-object",
            "--bucket",
            bucket,
            "--key",
            key,
            "--endpoint-url",
            endpoint,
            "--output",
            "json",
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    remote = json.loads(head.stdout)
    if (
        remote.get("ContentLength") != path.stat().st_size
        or (remote.get("Metadata") or {}).get("sha256") != digest.hexdigest()
    ):
        raise ValueError("uploaded R2 object length or metadata does not match payload")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("bucket")
    parser.add_argument("key")
    parser.add_argument("content_type")
    args = parser.parse_args()
    upload(args.file, args.bucket, args.key, args.content_type)


if __name__ == "__main__":
    main()
