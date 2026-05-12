import torch
import torch.nn as nn

class BertWeight(nn.Module):
  def __init__(self, hparams, pretrained_model):
    super(BertWeight, self).__init__()

    self.hparams = hparams
    self._model = pretrained_model

    # classification layer for predict strength
    self._classification = nn.Sequential(
      nn.Dropout(p=1 - self.hparams.dropout_keep_prob),
      nn.Linear(self.hparams.bert_hidden_dim, 1)
    )

    # Loss function for predict strength
    self._criterion = nn.BCEWithLogitsLoss()

  def forward(self, batch, cls_output):
    # if input data is not txt data, use below code 
    # outputs = self._model(
    #   batch["anno_sent"],
    #   token_type_ids=batch["segment_ids"],
    #   attention_mask=batch["attention_mask"]
    # )
    # bert_outputs = outputs[0]
    # cls_output = bert_outputs[:, 0, :]  # CLS Token

    strength_logits = self._classification(cls_output)
    strength_logits = strength_logits.squeeze(-1)
    strength_loss = self._criterion(strength_logits, batch["label"])

    return strength_loss