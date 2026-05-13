import os
import numpy as np
import logging
import random
import pickle
from collections import defaultdict

from datetime import datetime
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from data.novel_plot_dataset import NovelPlotDataset
from models.utils.checkpointing import CheckpointManager, load_checkpoint
from models import Model
from evaluation import Evaluation
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

class PlotAnalysis(object): # Fine-tuning
  def __init__(self, hparams): # -> None
    self.hparams = hparams # Hyper-parameters
    self._logger = logging.getLogger(__name__) # Logger

    np.random.seed(hparams.random_seed) # Random Seed
    torch.manual_seed(hparams.random_seed) # Random Seed
    torch.cuda.manual_seed_all(hparams.random_seed) # Random Seed
    torch.backends.cudnn.benchmark = False # Benchmark
    torch.backends.cudnn.deterministic = True # Deterministic

    print('cuda is available: ' + str(torch.cuda.is_available()))
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Device
    print('self.device: ' + str(self.device))
    print('self.hparams.root_dir: ' + str(self.hparams.root_dir))

  def _build_dataset(self):
    # load data and shuffle
    data_dir = self.hparams.data_dir
    data_file_name = 'novel_data_total.pkl'
    data_file = os.path.join(data_dir, data_file_name)
    train_file = os.path.join(data_dir, 'novel_data_train.pkl')
    valid_file = os.path.join(data_dir, 'novel_data_test.pkl')
    valid_ratio = 0.1

    for file in [train_file, valid_file]:
      if os.path.exists(file):
        os.remove(file)

    with open(data_file, "rb") as f:
      data = pickle.load(f)
    random.shuffle(data)

    # set dataset
    tag_groups = defaultdict(list)
    for item in data:
      tag_groups[item["tag"]].append(item)

    min_tag_count = min(len(items) for items in tag_groups.values())
    valid_count_per_tag = int(min_tag_count * valid_ratio)
    # fix eval data count
    valid_count_per_tag = 100

    train_data = []
    valid_data = []
    for tag, items in tag_groups.items():
      valid_data.extend(items[:valid_count_per_tag])
      train_data.extend(items[valid_count_per_tag:])

    # save dataset and log
    with open(train_file, "wb") as f:
      pickle.dump(train_data, f)
    with open(valid_file, "wb") as f:
      pickle.dump(valid_data, f)

    len_train = len(train_data)
    len_valid = len(valid_data)
    self._logger.info(f"Train dataset saved to {train_file}")
    self._logger.info(f"Train dataset has {len_train} dataset")
    self._logger.info(f"Validation dataset saved to {valid_file}")
    self._logger.info(f"Validation dataset has {len_valid} dataset")
    print("""
       # -------------------------------------------------------------------------
       #   BUILD DATASET FINISH
       # -------------------------------------------------------------------------
       """)

  def _build_dataloader(self): # -> None
    # =============================================================================
    #   SETUP DATASET, DATALOADER
    # =============================================================================
    self.train_dataset = NovelPlotDataset(self.hparams, split="train")
    self.train_dataloader = DataLoader(
      self.train_dataset,
      batch_size=self.hparams.train_batch_size,
      num_workers=self.hparams.cpu_workers,
      shuffle=True,
      drop_last=True
    )

    print("""
       # -------------------------------------------------------------------------
       #   DATALOADER FINISHED
       # -------------------------------------------------------------------------
       """)

  def _build_model(self, tag_counts):
    # =============================================================================
    #   MODEL : Standard, Mention Pooling, Entity Marker
    # =============================================================================
    print('\t* Building model...')

    self.model = Model(self.hparams, tag_counts)
    self.model = self.model.to(self.device)

    # Use Multi-GPUs
    if -1 not in self.hparams.gpu_ids and len(self.hparams.gpu_ids) > 1: # Multi-GPUs
      self.model = nn.DataParallel(self.model, self.hparams.gpu_ids) # Data Parallel

    # =============================================================================
    #   CRITERION
    # =============================================================================

    self.iterations = len(self.train_dataset) // self.hparams.virtual_batch_size

    # Prepare optimizer and schedule (linear warmup and decay)
    if self.hparams.optimizer_type == "Adam":
      self.optimizer = optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
    elif self.hparams.optimizer_type == "AdamW":
      no_decay = ['bias', 'LayerNorm.weight']
      optimizer_grouped_parameters = [
        {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': self.hparams.weight_decay},
        {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
      ]
      self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
      self.scheduler = get_linear_schedule_with_warmup( # Scheduler
        self.optimizer, num_warmup_steps=self.hparams.warmup_steps, # Optimizer, Warmup Steps
        num_training_steps=self.iterations * self.hparams.num_epochs) # Iterations, Number of Epochs

  # set up checkpoint 
  def _setup_training(self):
    if self.hparams.save_dirpath == 'checkpoints/':
      self.save_dirpath = os.path.join(self.hparams.root_dir, self.hparams.save_dirpath)
    self.checkpoint_manager = CheckpointManager(self.model, self.optimizer, self.save_dirpath, hparams=self.hparams)
    
    # if not use post training, comment below lines
    load_pthpath = os.path.join(self.hparams.bert_pretrained_dir, self.hparams.bert_pretrained, self.hparams.bert_checkpoint_path)
    if ".pth" in load_pthpath:
      model_state_dict, optimizer_state_dict = load_checkpoint(load_pthpath)
      if isinstance(self.model, nn.DataParallel):
        self.model.module.load_state_dict(model_state_dict, strict=False)
      else:
        self.model.load_state_dict(model_state_dict, strict=False)
      # self.optimizer.load_state_dict(optimizer_state_dict)
      # self.previous_model_path = load_pthpath
      print("Loaded model from {}".format(load_pthpath)) 

    self.start_epoch = 1 # Start Epoch

    print(
      """
      # -------------------------------------------------------------------------
      #   Setup Training Finished
      # -------------------------------------------------------------------------
      """
    )

  def train(self):
    if self.hparams.shuffle_dataset == "1":
      self._build_dataset()
    
    self._build_dataloader()
    tag_counts = self.train_dataset.tag_counts
    self._build_model(tag_counts)
    self._setup_training()

    # Evaluation Setup
    evaluation = Evaluation(self.hparams, model=self.model)

    start_time = datetime.now().strftime('%H:%M:%S')
    self._logger.info("Start train model at %s" % start_time)

    train_begin = datetime.utcnow()
    global_iteration_step = 0
    accu_loss, accu_txt_tag_loss, accu_wei_loss, accu_pos_loss, accu_bio_loss = 0, 0, 0, 0, 0 
    accu_cnt = 0

    for epoch in range(self.start_epoch, self.hparams.num_epochs + 1): 
      self.model.train()
      
      tqdm_batch_iterator = tqdm(self.train_dataloader)
      accu_batch = 0
      for batch_idx, batch in enumerate(tqdm_batch_iterator):
        buffer_batch = batch.copy()

        for task_key in batch:
          for key in buffer_batch[task_key]:
            buffer_batch[task_key][key] = buffer_batch[task_key][key].to(self.device)

        logits, losses = self.model(buffer_batch)

        txt_tag_loss, wei_loss, pos_loss, bio_loss = losses
        if txt_tag_loss is not None:
          txt_tag_loss = self.hparams.txt_tag_loss_ratio * txt_tag_loss.mean()
          accu_txt_tag_loss += txt_tag_loss.item()

        if wei_loss is not None:
          wei_loss = self.hparams.wei_loss_ratio * wei_loss.mean()
          accu_wei_loss += wei_loss.item()

        if pos_loss is not None:
          pos_loss = self.hparams.pos_loss_ratio * pos_loss.mean()
          accu_pos_loss += pos_loss.item()

        if bio_loss is not None:
          bio_loss = self.hparams.bio_loss_ratio * bio_loss.mean()
          accu_bio_loss += bio_loss.item()

        loss = None
        for task_tensor_loss in [txt_tag_loss, wei_loss, pos_loss, bio_loss]:
          if task_tensor_loss is not None:
            loss = loss + task_tensor_loss if loss is not None else task_tensor_loss

        # loss.backward()
        accum_steps = self.hparams.virtual_batch_size // self.hparams.train_batch_size
        (loss / accum_steps).backward()
        accu_loss += loss.item()
        accu_cnt += 1

        # TODO: virtual batch implementation
        accu_batch += buffer_batch["txt_tag"]["label"].shape[0] # New

        # Gradient accumulation 
        if self.hparams.virtual_batch_size == accu_batch \
            or batch_idx == (len(self.train_dataset) // self.hparams.train_batch_size):  # last batch

          nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams.max_gradient_norm) # Clip the gradient norm
          self.optimizer.step() # Step
          if self.hparams.optimizer_type == "AdamW": # If the optimizer type is AdamW
            self.scheduler.step() # Step

          self.optimizer.zero_grad() # Zero the gradient

          accu_batch = 0 # New

          global_iteration_step += 1 # New
          description = "[{}][Epoch: {:3d}][Iter: {:6d}][Loss: {:6f}][txt_tag_Loss: {:4f}]" \
                        "[Wei_Loss: {:4f}][Pos_Loss: {:4f}][BIO_Loss: {:4f}][lr: {:7f}]".format( 
            datetime.utcnow() - train_begin, 
            epoch,
            global_iteration_step, accu_loss / accu_cnt,
            accu_txt_tag_loss / accu_cnt, accu_wei_loss / accu_cnt, accu_pos_loss / accu_cnt, accu_bio_loss / accu_cnt,
            self.optimizer.param_groups[0]['lr'])
          tqdm_batch_iterator.set_description(description) # Set the description

          # tensorboard
          if global_iteration_step % self.hparams.tensorboard_step == 0: # If the global iteration step is divisible by the tensorboard step
            self._logger.info(description) # Log the description
            accu_loss, accu_txt_tag_loss, accu_wei_loss, accu_pos_loss, accu_bio_loss, accu_cnt = 0, 0, 0, 0, 0, 0 # Reset the accumulated values

      # -------------------------------------------------------------------------
      #   ON EPOCH END  (checkpointing and validation)
      # -------------------------------------------------------------------------
      self.checkpoint_manager.step(epoch) # Step
      self.previous_model_path = os.path.join(self.checkpoint_manager.ckpt_dirpath, "checkpoint_%d.pth" % (epoch)) # Previous Model Path
      self._logger.info(f"self.previous_model_path: {self.previous_model_path}") # Log the previous model path

      torch.cuda.empty_cache()
      self._logger.info("Evaluation after %d epoch" % epoch)
      evaluation.run_evaluate(self.previous_model_path, epoch)
      torch.cuda.empty_cache()
