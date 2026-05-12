import os
import torch.nn as nn
import torch.nn.functional as F
import torch

from models.bert_weight import BertWeight 
from models.bert_position import BertPosition 
from models.bert_bio import BertBio 

import logging

class BertCls(nn.Module):
  def __init__(self, hparams, tag_counts):
    super(BertCls, self).__init__()
    self.hparams = hparams
    self._logger = logging.getLogger(__name__)

    # example) ./resources/bert-base-uncased/bert-base-uncased-config.json
    pretrained_config = hparams.pretrained_config.from_pretrained(
        os.path.join(self.hparams.bert_pretrained_dir, self.hparams.bert_pretrained,
          "%s-config.json" % self.hparams.bert_pretrained)
    )

    # example) ./resources/bert-base-uncased/bert-base-uncased-pytorch_model.bin
    self.model_path = os.path.join(self.hparams.bert_pretrained_dir, self.hparams.bert_pretrained, self.hparams.bert_checkpoint_path)
    self._logger.info(f"pretrained_model_path: {self.model_path}")
    self._model = hparams.pretrained_model.from_pretrained(
      self.model_path,
      config=pretrained_config
    )

    '''
    if ".pth" in self.model_path:
      checkpoint = torch.load(self.model_path)
      state_dict = None
      if 'model' in checkpoint:
        state_dict = checkpoint['model']
      else:
        state_dict = checkpoint

      self._model = hparams.pretrained_model.from_pretrained(
        self.model_path,
        config=pretrained_config,
        state_dict=state_dict
      )
    else:
      self._model = hparams.pretrained_model.from_pretrained(
        self.model_path,
        config=pretrained_config
      )
    '''

    self.device = (torch.device("cuda", self.hparams.gpu_ids[0]) if self.hparams.gpu_ids[0] >= 0 else torch.device("cpu"))

    num_new_tok = 0
    if not self.hparams.model_type.startswith("bert_post") and not self.hparams.model_type.startswith("roberta_large"):
      if self.hparams.do_eot:
        num_new_tok += 1

    self._model.resize_token_embeddings(self._model.config.vocab_size + num_new_tok)  # [EOT]

    self._classification = nn.Sequential(
      nn.Dropout(p=1 - self.hparams.dropout_keep_prob),
      nn.Linear(self.hparams.bert_hidden_dim, 9)
    )

    # weighted cross entropy 
    self.wflag = self.hparams.weighted_cross_entropy
    if self.wflag == "1":
      total_datas = sum(tag_counts.values())
      class_weights = [1 - tag_counts[index] / total_datas for index in range(9)]
      class_weights = torch.FloatTensor(class_weights).to(self.device)
      self._logger.info(f"class_weights on model init: {class_weights}")
      self._criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif self.wflag == "2":
      total_samples = sum(tag_counts.values())
      class_weights = [total_samples / tag_counts[index] for index in range(9)]
      scaled_class_weights = [weight / max(class_weights) for weight in class_weights]  # scale weights between 0 and 1
      self._logger.info(f"class_weights on model init: {scaled_class_weights}")
      self._criterion = nn.CrossEntropyLoss(weight=scaled_class_weights)
    else:
      self._logger.info("model init: not use weighted cross entropy")
      self._criterion = nn.CrossEntropyLoss()

    # Focal Loss Parameters
    self.alpha = 0.25  # 불균형 클래스 보정
    self.gamma = 2.0  # weights for hard task

    if self.hparams.do_wei:
      self._bert_weight = BertWeight(hparams, self._model)
    if self.hparams.do_pos:
      self._bert_position = BertPosition(hparams, self._model)
    if self.hparams.do_bio:
      self._bert_bio = BertBio(hparams, self._model)


  def focal_loss(self, logits, labels):
    ce_loss = F.cross_entropy(logits, labels, reduction='none')
    pt = torch.exp(-ce_loss)  # predicted probability value
    focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
    return focal_loss.mean()

  def forward(self, batch):
    logits, txt_tag_loss, wei_loss, pos_loss, bio_loss = None, None, None, None, None
        
    if batch["txt_tag"]["anno_sent"].dim() == 3:
      batch_data = batch["txt_tag"]["anno_sent"].squeeze(1)
    else:
      batch_data = batch["txt_tag"]["anno_sent"]
    outputs = self._model(
      batch_data,  # tokenization data
      token_type_ids=batch["txt_tag"]["segment_ids"],  # sentence seperator
      attention_mask=batch["txt_tag"]["attention_mask"]  # pad seperator
    )
    bert_outputs = outputs[0]
    cls_logits = bert_outputs[:, 0, :]  # [bs, bert_output_size]

    if self.hparams.evaluate:
      logits = self._classification(cls_logits)
      return logits, (None, None, None, None)

    if self.hparams.do_txt_tag:
      logits = self._classification(cls_logits)  # [bs, 1]
      txt_tag_loss = self._criterion(logits, batch["txt_tag"]["label"])
    if self.hparams.do_wei and self.training:
      wei_loss = self._bert_weight(batch["txt_wei"], cls_logits)
    if self.hparams.do_pos and self.training:
      pos_loss = self._bert_position(batch["txt_pos"], cls_logits)
    if self.hparams.do_bio and self.training:
      bio_loss = self._bert_bio(batch["txt_bio"], bert_outputs)

    return logits, (txt_tag_loss, wei_loss, pos_loss, bio_loss)
