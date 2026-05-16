"""Download every supported pretrained model into resources/."""
from download_pretrained import MODELS, download_one

if __name__ == "__main__":
    for key in MODELS:
        download_one(key)
