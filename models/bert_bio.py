import os
import torch.nn as nn

class BertBio(nn.Module):
    def __init__(self, hparams, pretrained_model):
        super(BertBio, self).__init__()

        self.hparams = hparams
        self._model = pretrained_model

        self._classification = nn.Sequential( 
           nn.Dropout(p=1 - self.hparams.dropout_keep_prob),
           nn.Linear(self.hparams.bert_hidden_dim, 3)
        )

        self._criterion = nn.CrossEntropyLoss()
    
    def forward(self, batch, bio_outputs):
        # if input data is not txt data, use below code 
        # outputs = self._model(
        #     batch["anno_sent"],
        #     token_type_ids=batch["segment_ids"],
        #     attention_mask=batch["attention_mask"]
        # )
        # bert_outputs = outputs[0]

        bio_logits = self._classification(bio_outputs)
        bio_loss = self._criterion(bio_logits.view(-1, 3), batch["label"].view(-1))

        return bio_loss