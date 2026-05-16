"""Download BERT-large-uncased pretrained weights into resources/bert-large-uncased/."""
from download_pretrained import download_one

if __name__ == "__main__":
    download_one("bert_large")
