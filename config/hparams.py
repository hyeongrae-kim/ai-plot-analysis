import random
from collections import defaultdict

## bert-large-uncased && bert-base-uncased
from transformers import BertConfig, BertModel

## electra-large-discriminator
from transformers import ElectraModel, ElectraConfig

## roberta-large
from transformers import RobertaConfig, RobertaModel

## BGE-M3
from transformers import AutoModel, AutoConfig

BERT_MODEL_PARAMS = dict(
  pretrained_config=BertConfig,
  pretrained_model=BertModel,
)

ELECTRA_MODEL_PARAMS = dict(
  pretrained_config=ElectraConfig,
  pretrained_model=ElectraModel,
)

ROBERTA_MODEL_PARAMS = dict(
  pretrained_config=RobertaConfig,
  pretrained_model=RobertaModel,
)

BGE_M3_MODEL_PARAMS = dict(
  pretrained_config=AutoConfig,
  pretrained_model=AutoModel,
)

# DATASET
NOVEL_PARAMS = defaultdict(
  evaluate_candidates_num=10,
  recall_k_list=[1, 2, 5, 10],
  evaluate_data_type="test",
  language="english",
  eval_batch_size= 10,
)
NOVEL_EVAL_PARAMS = defaultdict(
  evaluate_candidates_num=10,
  recall_k_list=[1, 2, 5, 10],
  evaluate_data_type="valid",
  language="english",
  eval_batch_size=10,
)

# MULTI-TASK PARAMS
IMPORTANCE_WEIGHT_PARAMS = defaultdict(
  do_wei=True,
)
POSITION_PARAMS = defaultdict(
  do_pos=True,
)
BIO_PARAMS = defaultdict(
  do_bio=True,
)

BASE_PARAMS = defaultdict(
  # lambda: None,  # Set default value to None.
  # GPU params
  gpu_ids=[0],  # [0,1,2,3]

  # Input params
  # more than 64 batch size over vram limit
  train_batch_size=8,  # 8 - (32 - 64) 128 (+256) # 16,  # 32
  eval_batch_size=20,  # 250,  # 1000
  virtual_batch_size=16, # train_batch_size의 배수 # 32,

  # Training BERT params
  # 3e-05, 1e-05 -> overshoot at roberta-large, electra-large (not mtl). 
  # use 2e-05 when not use mtl
  # 3e-05, 2e-05 -> overshoot at electra-large with mtl

  # bge-m3
  # learning_rate => 1e-05 (not bias)
  learning_rate=1e-05, # 5e-05, 1e-05

  dropout_keep_prob=0.8,
  num_epochs=10,
  max_gradient_norm=5,
  adam_epsilon=1e-8,
  weight_decay=0.0,
  warmup_steps=0,
  optimizer_type="AdamW",  # Adam, AdamW

  pad_idx=0,
  max_position_embeddings=512,
  num_hidden_layers=12,
  num_attention_heads=12,
  intermediate_size=3072,
  bert_hidden_dim=768,
  attention_probs_dropout_prob=0.1,
  layer_norm_eps=1e-12,

  # Train Model Config
  task_name="novel",
  do_bert=True,
  do_eot=True,

  do_txt_tag=True,

  auxiliary_loss_type="sigmoid",  # softmax or sigmoid
  do_wei=False,  # True or False for jude
  do_pos=False,  # True or False for jude
  do_bio=False,  # True or False for jude

  max_sequence_len=512,  # txt maximum length is about 700
  txt_tag_loss_ratio=1.0,
  wei_loss_ratio=0.5,  # 0.01, 0.1, 0.5, 1,
  pos_loss_ratio=0.5,  # 0.01, 0.1, 0.5, 1,
  bio_loss_ratio=1.0,  # 0.01, 0.1, 0.5, 1,

  save_dirpath='checkpoints/',  # /path/to/checkpoints

  load_pthpath="",
  cpu_workers=4,
  tensorboard_step=100,
  evaluate_print_step=1000,
  random_seed=random.sample(range(1000, 10000), 1)[0],  # experiment value: 2270
)

# BERT_LARGE
BERT_LARGE_PARAMS = BASE_PARAMS.copy()
BERT_LARGE_PARAMS.update(
  bert_hidden_dim=1024,
  num_hidden_layers=24,
  num_attention_heads=16,
  intermediate_size=4096,
)

ELECTRA_LARGE_PARAMS = BASE_PARAMS.copy()
ELECTRA_LARGE_PARAMS.update(
  model_type="electra_large",
  bert_hidden_dim=1024,
  num_hidden_layers=24,
  num_attention_heads=16,
  intermediate_size=4096,
)

ROBERTA_LARGE_PARAMS = BASE_PARAMS.copy()
ROBERTA_LARGE_PARAMS.update(
  model_type="roberta_large",
  bert_hidden_dim=1024,
  num_hidden_layers=24,
  num_attention_heads=16,
  intermediate_size=4096,
)

BGE_M3_PARAMS = BASE_PARAMS.copy()
BGE_M3_PARAMS.update(
  model_type="bge_m3",
  bert_hidden_dim=1024,
  num_hidden_layers=24,
  num_attention_heads=16,
  intermediate_size=4096,
)

