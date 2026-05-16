"""Download BERT-base-uncased pretrained weights into resources/bert-base-uncased/."""
from download_pretrained import download_one

if __name__ == "__main__":
    download_one("bert_base")
