"""Download pretrained models into resources/ for local-path loading.

Usage:
  python scripts/download_pretrained.py --all
  python scripts/download_pretrained.py bert_base bert_large
  python scripts/download_pretrained.py --list

Each model is downloaded to resources/{dir_name}/ as a full HuggingFace
snapshot. models/bert_cls.py and data/novel_plot_dataset.py both call
`from_pretrained(local_dir)` against that path, so no further file
renaming or symlinking is required.
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from huggingface_hub import snapshot_download

from config.hparams import MODEL_DIR_MAP, PRETRAINED_ROOT

RESOURCES_DIR = (PROJECT_ROOT / PRETRAINED_ROOT).resolve()

# Revisions can be pinned per model. None = track main HEAD (less reproducible
# but lets HuggingFace resolve latest). To pin: visit
# https://huggingface.co/<repo>/commits/main and copy the short SHA.
MODELS = {
    "bert_base": {"repo": "bert-base-uncased", "revision": None},
    "bert_large": {"repo": "bert-large-uncased", "revision": None},
    "electra_base": {"repo": "google/electra-base-discriminator", "revision": None},
    "electra_large": {"repo": "google/electra-large-discriminator", "revision": None},
    "roberta_large": {"repo": "FacebookAI/roberta-large", "revision": None},
    "bge_m3": {"repo": "BAAI/bge-m3", "revision": "5617a9f61b028005a4858fdac845db406aefb181"},
}


def download_one(key: str) -> None:
    spec = MODELS[key]
    target_dir = RESOURCES_DIR / MODEL_DIR_MAP[key]
    target_dir.mkdir(parents=True, exist_ok=True)

    rev_label = spec["revision"][:10] if spec["revision"] else "main"
    print(f"\n[{key}] {spec['repo']}@{rev_label} -> {target_dir}")

    snapshot_download(
        repo_id=spec["repo"],
        revision=spec["revision"],
        local_dir=str(target_dir),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Download pretrained models into resources/.")
    parser.add_argument(
        "models",
        nargs="*",
        help=f"Model keys to download. Choices: {', '.join(MODELS)}",
    )
    parser.add_argument("--all", action="store_true", help="Download all models in MODELS")
    parser.add_argument("--list", action="store_true", help="List available model keys and exit")
    args = parser.parse_args()

    if args.list:
        for key, spec in MODELS.items():
            rev = spec["revision"][:10] if spec["revision"] else "main"
            print(f"  {key:<14} -> {spec['repo']}@{rev}")
        return 0

    if args.all:
        keys = list(MODELS)
    elif args.models:
        unknown = [k for k in args.models if k not in MODELS]
        if unknown:
            print(f"Unknown model keys: {unknown}", file=sys.stderr)
            print(f"Available: {list(MODELS)}", file=sys.stderr)
            return 1
        keys = args.models
    else:
        parser.print_help()
        return 1

    for key in keys:
        download_one(key)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
