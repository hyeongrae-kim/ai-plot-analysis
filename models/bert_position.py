import torch
import torch.nn as nn

class BertPosition(nn.Module):
    def __init__(self, hparams, pretrained_model):
        super(BertPosition, self).__init__()

        self.hparams = hparams
        self._model = pretrained_model

        self._classification = nn.Sequential( 
           nn.Dropout(p=1 - self.hparams.dropout_keep_prob),
           nn.Linear(self.hparams.bert_hidden_dim, 10)
        )
        # if use position with float value(0-1), use below classification layer
        # self._classification = nn.Sequential( 
        #    nn.Dropout(p=1 - self.hparams.dropout_keep_prob),
        #    nn.Linear(self.hparams.bert_hidden_dim, 1)
        # )

        self._criterion = nn.CrossEntropyLoss()
        # if use position with float value(0-1), use below loss function
        # self._criterion = nn.MSELoss
    
    def forward(self, batch, cls_output):
        # if input data is not txt data, use below code 
        # outputs = self._model(
        #     batch["anno_sent"],
        #     token_type_ids=batch["segment_ids"],
        #     attention_mask=batch["attention_mask"]
        # )
        # bert_outputs = outputs[0]
        # cls_output = bert_outputs[:, 0, :]

        position_logits = self._classification(cls_output)
        position_loss = self._criterion(position_logits, batch["label"])

        return position_loss
