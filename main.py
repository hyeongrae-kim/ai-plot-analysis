import os
import argparse
import collections
import logging
from datetime import datetime

from config.hparams import *
from train import PlotAnalysis 

from evaluation import Evaluation

import pytz

# Hyper-parameters
PARAMS_MAP = {
  # Pre-trained Models
  "bert_base": BASE_PARAMS,
  "bert_large": BERT_LARGE_PARAMS,

  "electra_base": BASE_PARAMS,
  "electra_large": ELECTRA_LARGE_PARAMS,

  "roberta_large": ROBERTA_LARGE_PARAMS,

  "bge_m3": BGE_M3_PARAMS,
}

# Dataset Parameters
DATASET_MAP = {
  "novel": NOVEL_PARAMS,
  "novel_eval": NOVEL_EVAL_PARAMS
}

# Pre-trained Model Parameters
PRETRAINED_MODEL_MAP = {
  "bert": BERT_MODEL_PARAMS,
  "electra": ELECTRA_MODEL_PARAMS,
  "roberta": ROBERTA_MODEL_PARAMS,
  "bge": BGE_M3_MODEL_PARAMS
}

# Training Types
TRAINING_TYPE_MAP = {
  "fine_tuning": PlotAnalysis,
}

# Evaluation Types
EVAL_TYPE_MAP = {
  "fine_tuning": Evaluation,
}

# Multi-task Types
MULTI_TASK_TYPE_MAP = {
  "wei": IMPORTANCE_WEIGHT_PARAMS,
  "pos": POSITION_PARAMS,
  "bio": BIO_PARAMS
}

# Initialize Logger
def init_logger(path: str):
  if not os.path.exists(path):
    os.makedirs(path)
  logger = logging.getLogger()
  logger.handlers = []
  logger.setLevel(logging.DEBUG)
  debug_fh = logging.FileHandler(os.path.join(path, "debug.log"))
  debug_fh.setLevel(logging.DEBUG)

  info_fh = logging.FileHandler(os.path.join(path, "info.log"))
  info_fh.setLevel(logging.INFO)

  ch = logging.StreamHandler()
  ch.setLevel(logging.INFO)

  info_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
  debug_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s | %(lineno)d:%(funcName)s')

  ch.setFormatter(info_formatter)
  info_fh.setFormatter(info_formatter)
  debug_fh.setFormatter(debug_formatter)

  logger.addHandler(ch)
  logger.addHandler(debug_fh)
  logger.addHandler(info_fh)

  return logger 

def train_model(args, hparams):
  kst = pytz.timezone('Asia/Seoul') # Set in Korea time
  timestamp = datetime.now(kst).strftime('%Y%m%d-%H%M%S')
  root_dir = os.path.join(hparams["root_dir"], args.model, args.task_name, "%s/" % timestamp)
  logger = init_logger(root_dir)
  logger.info("Hyper-parameters: %s" % str(hparams))
  hparams["root_dir"] = root_dir

  hparams = collections.namedtuple("HParams", sorted(hparams.keys()))(**hparams)

  model = TRAINING_TYPE_MAP[args.training_type](hparams)
  model.train()

def evaluate_model(args, hparams):
  kst = pytz.timezone('Asia/Seoul') # Set in Korea time
  timestamp = datetime.now(kst).strftime('%Y%m%d-%H%M%S')
  root_dir = os.path.join(hparams["root_dir"], args.model, args.task_name, "%s/" % timestamp)
  logger = init_logger(root_dir)
  hparams["root_dir"] = root_dir
  logger.info("evaluate_model: " + str(timestamp))

  hparams = collections.namedtuple("HParams", sorted(hparams.keys()))(**hparams)

  model = EVAL_TYPE_MAP[args.training_type](hparams)
  model.run_valid_data_evaluate(args.evaluate, 0)


if __name__ == '__main__':
  arg_parser = argparse.ArgumentParser(description="Novel Plot Element Analysis (PyTorch)")
  arg_parser.add_argument("--model", dest="model", type=str, default="bert_base", help="Model Name") 
  arg_parser.add_argument("--task_name", dest="task_name", type=str, default="novel", help="Task Name") 
  arg_parser.add_argument("--task_type", dest="task_type", type=str,
                          default="novel_plot_analysis",
                          help="novel_plot_analysis") # novel_plot_analysis |  

  arg_parser.add_argument("--root_dir", dest="root_dir", type=str,
                          default="./",
                          help="model train logs, checkpoints")
  arg_parser.add_argument("--data_dir", dest="data_dir", type=str, required=True,
                          help="training pkl path | h5py files")  # novel_train.pkl, novel_valid_pkl, novel_test.pkl 
  arg_parser.add_argument("--model_pth_path", dest="model_pth_path", type=str,
                          default="",
                          help="Filename of a .pth inside the pretrained model directory for warm-starting from custom weights. Empty = use HuggingFace pretrained only.")
  arg_parser.add_argument("--evaluate", dest="evaluate", type=str,
                          help="Evaluation Checkpoint", default="") 
  arg_parser.add_argument("--training_type", dest="training_type", type=str, default="fine_tuning",
                          help="fine_tuning or post_training") # fine_tuning, post_training
  arg_parser.add_argument("--multi_task_type", dest="multi_task_type", type=str, default="",
                          help="wei, pos, bio") # wei, pos, bio 
  arg_parser.add_argument("--gpu_ids", dest="gpu_ids", type=str,
                          help="gpu_ids", default="") # 0,1,2,3. -1 for cpu mode
  arg_parser.add_argument("--weighted_cross_entropy", dest="weighted_cross_entropy", type=str, default="0",
                          help="0: not use, 1: weighted cross entropy, 2: scaled weighted cross entropy")
  arg_parser.add_argument("--f_loss", dest="f_loss", type=str, default="0", help="0: not use, 1: use")
  arg_parser.add_argument("--shuffle_dataset", type=str, default="0", help="0: not use, 1: use")
  arg_parser.add_argument("--do_txt_tag", type=str, default="1", help="0: not use, 1: use, default value is 1")

  arg_parser.add_argument("--train_batch_size", type=int, default=8, help="override BASE_PARAMS.train_batch_size")
  arg_parser.add_argument("--virtual_batch_size", type=int, default=8, help="override BASE_PARAMS.virtual_batch_size")
  arg_parser.add_argument("--wei_loss_ratio", type=float, default=0.5, help="override BASE_PARAMS.wei_loss_ratio")
  arg_parser.add_argument("--pos_loss_ratio", type=float, default=0.5, help="override BASE_PARAMS.pos_loss_ratio")
  arg_parser.add_argument("--bio_loss_ratio", type=float, default=1.0, help="override BASE_PARAMS.bio_loss_ratio")

  args = arg_parser.parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

  hparams = PARAMS_MAP[args.model]
  hparams["gpu_ids"] = list(range(len(args.gpu_ids.split(","))))
  hparams["root_dir"] = args.root_dir 
  hparams["data_dir"] = args.data_dir 
  hparams["pretrained_dir"] = os.path.join(PRETRAINED_ROOT, MODEL_DIR_MAP[args.model])
  hparams["model_pth_path"] = args.model_pth_path
  hparams["model_type"] = args.model
  hparams["task_name"] = args.task_name
  hparams["task_type"] = args.task_type
  hparams["training_type"] = args.training_type
  hparams["weighted_cross_entropy"] = args.weighted_cross_entropy
  hparams["f_loss"] = args.f_loss
  hparams["shuffle_dataset"] = args.shuffle_dataset
  hparams["evaluate"] = args.evaluate
  hparams["do_txt_tag"] = args.do_txt_tag == "1"

  hparams["train_batch_size"] = args.train_batch_size
  hparams["virtual_batch_size"] = args.virtual_batch_size
  hparams["wei_loss_ratio"] = args.wei_loss_ratio
  hparams["pos_loss_ratio"] = args.pos_loss_ratio
  hparams["bio_loss_ratio"] = args.bio_loss_ratio

  # Multi-task types (wei, pos, bio)
  multi_task_types = args.multi_task_type.split(",") if args.multi_task_type != '' else []

  for mt_type in multi_task_types:
    hparams.update(MULTI_TASK_TYPE_MAP[mt_type.strip()]) 

  hparams.update(DATASET_MAP[args.task_name]) 
  hparams.update(PRETRAINED_MODEL_MAP[args.model.split("_")[0]])

  if args.evaluate:
    evaluate_model(args, hparams)
  else: 
    train_model(args, hparams)
