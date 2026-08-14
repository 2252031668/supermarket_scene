#!/usr/bin/env python3
"""Download Qwen3.5-4B from ModelScope into the local model directory."""

import os
from pathlib import Path

for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(name, None)

from modelscope_hub import HubApi, RepoType


MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_DIR = Path(__file__).resolve().parent / "models" / "Qwen3.5-4B"


def main() -> None:
    model_dir = HubApi().download_repo(
        MODEL_ID,
        RepoType.MODEL,
        local_dir=MODEL_DIR,
        max_workers=4,
    )
    print(f"Downloaded {MODEL_ID} to {model_dir}")


if __name__ == "__main__":
    main()
