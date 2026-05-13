import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "BAAI/bge-m3"
# Pinned to BAAI/bge-m3 main HEAD as of 2024-07-03.
# To update: visit https://huggingface.co/BAAI/bge-m3/commits/main and replace.
REVISION = "5617a9f61b028005a4858fdac845db406aefb181"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = PROJECT_ROOT / "resources" / "bge-m3"

# Symlinks so the existing `{name}-{file}` convention in bert_cls.py resolves.
SYMLINKS = {
    "bge-m3-config.json": "config.json",
    "bge-m3-pytorch_model.bin": "pytorch_model.bin",
}


def main() -> int:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {REPO_ID}@{REVISION[:10]} -> {TARGET_DIR}")
    snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        local_dir=str(TARGET_DIR),
    )

    for link_name, target_name in SYMLINKS.items():
        link_path = TARGET_DIR / link_name
        target_path = TARGET_DIR / target_name
        if not target_path.exists():
            print(f"  expected {target_path} not found, skipping symlink", file=sys.stderr)
            continue
        if link_path.is_symlink() or link_path.exists():
            link_path.unlink()
        link_path.symlink_to(target_name)
        print(f"  symlink {link_name} -> {target_name}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
