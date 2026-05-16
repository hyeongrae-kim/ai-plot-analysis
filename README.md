# Deep Learning the Plot: Identifying Plot Elements to Guide AI Story Generation

A multi-task NLP project for classifying paragraphs of English novels into nine plot-stage categories, using BERT, ELECTRA, RoBERTa, or BGE-M3 as the backbone.

## Setup

Install dependencies with uv:

```bash
uv sync
```

## Pretrained model download

Before training, the chosen backbone must be downloaded into `resources/`.

### Download a single model

Run the script that matches the model you want:

```bash
uv run python scripts/download_bert_base.py
uv run python scripts/download_bert_large.py
uv run python scripts/download_electra_base.py
uv run python scripts/download_electra_large.py
uv run python scripts/download_roberta_large.py
uv run python scripts/download_bge_m3.py
```

Each script writes to `resources/<model_dir>/` (for example, `resources/bge-m3/`).

### Download every supported model

```bash
uv run python scripts/download_all.py
```

## Training

The command below reflects the best-performing configuration from our experiments: BGE-M3 backbone with multi-task learning, batch size 8 (no gradient accumulation), and loss ratios `wei 0.5`, `pos 0.5`, `bio 1.0`.

```bash
uv run python main.py \
  --model bge_m3 \
  --task_name novel \
  --data_dir data/plot_v5-tag9 \
  --gpu_ids 0 \
  --multi_task_type "wei, bio, pos" \
  --weighted_cross_entropy 1 \
  --train_batch_size 8 \
  --virtual_batch_size 8 \
  --wei_loss_ratio 0.5 \
  --pos_loss_ratio 0.5 \
  --bio_loss_ratio 1.0
```

The pretrained directory is resolved automatically from `--model`, so no further path arguments are required. Training logs and checkpoints are written under `bge_m3/novel/<timestamp>/`.
