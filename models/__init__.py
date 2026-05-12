from models.bert_cls import BertCls

def Model(hparams, *args):
  name_model_map = {
    "bert_base": BertCls,
    "bert_large": BertCls,
    "bert_post": BertCls,

    "electra_base": BertCls,
    "electra_large": BertCls,
    "electra_post": BertCls,

    "roberta_large": BertCls,

    "bge_m3": BertCls,
  }

  return name_model_map[hparams.model_type](hparams, *args)
