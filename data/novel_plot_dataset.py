import os
import torch
import random
import pickle

from torch.utils.data import Dataset

## Bert tokenizer
from transformers import BertTokenizer

## Electra tokenizer
from transformers import ElectraTokenizerFast

## Roberta tokenizer
from transformers import RobertaTokenizer

## Bge-m3 tokenizer
from transformers import AutoTokenizer

from collections import Counter

import logging

class NovelPlotDataset(Dataset):
    def __init__(
        self, 
        hparams, 
        split: str="",
    ):
        super().__init__()

        self.hparams = hparams
        self.split = split
        self.input_examples = []
        self.data_path = os.path.join(hparams.data_dir, "novel_data_%s.pkl" %split)

        self._logger = logging.getLogger(__name__)

        with open(self.data_path, "rb") as pkl_handle:
          example = pickle.load(pkl_handle)
          self.input_examples.extend(example)
          print("========== total %d examples has been loaded! ==========" % len(self.input_examples))
        
        print("=========complete data load==========")
        random.seed(self.hparams.random_seed)
        self.num_input_examples = len(self.input_examples)

        # save novel plot element tag counts
        if split == "valid":
          self.tag_counts = None
        else:
          tags = [self._tag_to_idx(ex['tag']) for ex in self.input_examples]
          self.tag_counts = Counter(tags)
          self.tag_counts = {idx: self.tag_counts[idx] for idx in sorted(self.tag_counts)}
          self._logger.info(f"Tag distribution by index: {self.tag_counts}")

        self.model_type = self.hparams.model_type
        self._logger.info(f"model_type at dataset: {self.model_type}")

        if self.model_type.startswith("electra"):
          self._tokenizer = ElectraTokenizerFast.from_pretrained(self.hparams.pretrained_dir)
          self._logger.info(f"===========get electra tokenizer==========")
        elif self.model_type.startswith("roberta"):
          self._tokenizer = RobertaTokenizer.from_pretrained(self.hparams.pretrained_dir)
          self._logger.info(f"===========get roberta tokenizer==========")
        elif self.model_type.startswith("bge"):
          self._tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrained_dir)
          self._logger.info(f"===========get bge-m3 tokenizer==========")
        else:
          self._tokenizer = BertTokenizer.from_pretrained(self.hparams.pretrained_dir)
          self._logger.info(f"===========get bert tokenizer==========")
 
    def __len__(self):
        return len(self.input_examples)

    def __getitem__(self, index):
        # when load items in evaluation mode, use below function: _get_eval_item
        if self.split == "valid":
          return self._get_eval_item(index) 
        
        curr_example = self.input_examples[index]
        current_feature = dict()

        if self.model_type.startswith("roberta"):
          anno_sent, segment_ids, attention_mask, eot_pos = self._convert_text_to_roberta_token(curr_example)
        elif self.model_type.startswith("bge"):
          anno_sent, segment_ids, attention_mask, eot_pos = self._convert_text_to_bge_token(curr_example)       
        else:
          anno_sent, segment_ids, attention_mask, eot_pos = self._convert_text_to_token(curr_example)

        # novel text with plot category data
        tag_label = self._tag_to_idx(curr_example["tag"])
        current_feature["txt_tag"] = dict()
        current_feature["txt_tag"]["anno_sent"] = anno_sent
        current_feature["txt_tag"]["attention_mask"] = attention_mask
        current_feature["txt_tag"]["segment_ids"] = torch.tensor(segment_ids).long()
        current_feature["txt_tag"]["eot_pos"] = torch.tensor(eot_pos).long()
        current_feature["txt_tag"]["label"] = torch.tensor(tag_label).long()


        # novel text with weight(strong/weak)
        if self.hparams.do_wei and self.split == "train":
            wei_label = self._weight_to_binary(curr_example["weight"])
            current_feature["txt_wei"] = dict()
            current_feature["txt_wei"]["anno_sent"] = anno_sent
            current_feature["txt_wei"]["segment_ids"] = torch.tensor(segment_ids).long()
            current_feature["txt_wei"]["attention_mask"] = attention_mask
            current_feature["txt_wei"]["eot_pos"] = torch.tensor(eot_pos).long()
            current_feature["txt_wei"]["label"] = torch.tensor(wei_label).float()

        # novel text with position(0-10)
        if self.hparams.do_pos and self.split == "train":
            pos_label = self._weight_to_binary(curr_example["position"])
            current_feature["txt_pos"] = dict()
            current_feature["txt_pos"]["anno_sent"] = anno_sent
            current_feature["txt_pos"]["segment_ids"] = torch.tensor(segment_ids).long()
            current_feature["txt_pos"]["attention_mask"] = attention_mask
            current_feature["txt_pos"]["eot_pos"] = torch.tensor(eot_pos).long()
            current_feature["txt_pos"]["label"] = torch.tensor(pos_label).long()

        # novel text with bio tag
        if self.hparams.do_bio and self.split == "train":
            bio_label = self._convert_text_to_bio(curr_example)
            bio_token_label = self._convert_bio_to_tokens(bio_label)
            current_feature["txt_bio"] = dict()
            current_feature["txt_bio"]["anno_sent"] = anno_sent
            current_feature["txt_bio"]["segment_ids"] = torch.tensor(segment_ids).long()
            current_feature["txt_bio"]["attention_mask"] = attention_mask
            current_feature["txt_bio"]["eot_pos"] = torch.tensor(eot_pos).long()
            current_feature["txt_bio"]["label"] = torch.tensor(bio_token_label)

        return current_feature 

    def _get_eval_item(self, index):
      curr_example = self.input_examples[index]
      current_feature = dict()

      text = curr_example["text"]
      inputs = self._tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
      anno_sent = inputs.input_ids
      attention_mask = inputs.attention_mask
      eot_pos = [0] * len(anno_sent)
      segment_ids = [0] * len(anno_sent)
          
      current_feature["txt_tag"] = dict()
      current_feature["txt_tag"]["anno_sent"] = anno_sent
      current_feature["txt_tag"]["attention_mask"] = attention_mask
      current_feature["txt_tag"]["segment_ids"] = torch.tensor(segment_ids).long()
      current_feature["txt_tag"]["eot_pos"] = torch.tensor(eot_pos).long()
      current_feature["txt_tag"]["id"] = curr_example["id"]
      return current_feature

    
    def _tag_to_idx(self, tag):
        tag_to_idx = {
            "Background Setting": 0,
            "Character Setting": 1,
            "Crisis/Change of the Existing World": 2,
            "Information of the Narrative Object": 3,
            "Developmental Relationship with the Narrative Object (Non-Conflict)": 4,
            "Conflicts after the emergence of the Narrative Object": 5,
            "Reconciliation with the Narrative Object": 6,
            "Catastrophe with the Narrative Object": 7,
            "(Mini/Final) Closure or Epilogue": 8
        }
        return tag_to_idx.get(tag)
    
    def _weight_to_binary(self, weight):
        return 1 if weight == "Strong" else 0
    
    def _convert_text_to_token(self, example):
        '''
        sample of curr_example
        {
            'chunk_id': 124,
            'file_name': 'SFE046_The_Moon_Pool.txt',
            'position': 5,
            'sentence': ['There', 'were', 'two', 'favoured', 'classes', 'of', 'the', 'ladala--the', 'soldiers', 'and', 'the', 'dream-makers.'],
            'tag': 'Information of the Narrative Object',
            'text': ['Glancing', 'behind', 'me,', 'I', 'saw', ...]
            'total_chunk': 275,
            'weight': 'Strong'
        }
        '''
        text_token = example["text"]
        sentence_token = example["sentence"]
        text_token = self._max_len_trim_seq(text_token, sentence_token, 3)
        text_token = ["[CLS]"] + text_token + ["[EOT]", "[SEP]"]
        segment_ids = [0] * len(text_token)
        attention_mask = [1] * len(text_token)

        while len(text_token) < self.hparams.max_sequence_len:
            text_token.append("[PAD]")
            segment_ids.append(0)
            attention_mask.append(0)
        
        eot_pos = []
        for tok_idx, tok in enumerate(text_token):
            if tok == "[EOT]":
                eot_pos.append(1)
            else:
                eot_pos.append(0)
        
        assert len(text_token) == len(segment_ids) == len(attention_mask)
        anno_sent = self._tokenizer.convert_tokens_to_ids(text_token)
        assert len(text_token) <= self.hparams.max_sequence_len

        anno_sent = torch.tensor(anno_sent).long()
        attention_mask = torch.tensor(attention_mask).long()
        
        return anno_sent, segment_ids, attention_mask, eot_pos
    
    def _convert_text_to_roberta_token(self, example):
      text_token = example["text"]
      sentence_token = example["sentence"]
      text_token = self._max_len_trim_seq(text_token, sentence_token, 2)
      text_token = ["<s>"] + text_token + ["</s>"]
      segment_ids = [0] * len(text_token)
      attention_mask = [1] * len(text_token)

      while len(text_token) < self.hparams.max_sequence_len:
        text_token.append("<pad>")
        segment_ids.append(0)
        attention_mask.append(0)
      
      eot_pos = [0 for i in range(len(text_token))]
      assert len(text_token) == len(segment_ids) == len(attention_mask)
      anno_sent = self._tokenizer.convert_tokens_to_ids(text_token)
      
      anno_sent = torch.tensor(anno_sent).long()
      attention_mask = torch.tensor(attention_mask).long()

      return anno_sent, segment_ids, attention_mask, eot_pos
    
    def _convert_text_to_bge_token(self, example):
      text = " ".join(example["text"])
      inputs = self._tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
      anno_sent = inputs.input_ids
      attention_mask = inputs.attention_mask
      eot_pos = [0] * len(anno_sent)
      segment_ids = [0] * len(anno_sent)
      return anno_sent, segment_ids, attention_mask, eot_pos

    def _convert_text_to_bio(self, example):
        text = example["text"]
        sentence = example["sentence"]
        text_token = self._max_len_trim_seq(text, sentence, 3)
        text_token = ["[CLS]"] + text_token + ["[EOT]", "[SEP]"]
        bio_tags = ['O'] * self.hparams.max_sequence_len

        #  if there is not sentence in data
        if len(sentence) == 0:
            return bio_tags
        
        sentence_len = len(sentence)
        text_len = len(text_token)
        for i in range(text_len - sentence_len + 1):
            if text_token[i:i + sentence_len] == sentence:
                bio_tags[i] = 'B'
                for j in range(1, min(sentence_len, text_len)):
                    bio_tags[i + j] = 'I'
                return bio_tags
    
        return bio_tags 
    
    # convert BIO tag to 0, 1, 2
    def _convert_bio_to_tokens(self, bio_tags):
        bio_to_idx = {
            "B": 1,
            "I": 2,
            "O": 0,
        } 
        bio_tokens = [bio_to_idx.get(tag, 0) for tag in bio_tags]

        return bio_tokens
    
    # trim text when text length is over bert max sequence len
    def _max_len_trim_seq(self, text, sentence, token_num):
        max_len = self.hparams.max_sequence_len - token_num  # CLS, SEP, EOT Token
        
        if len(sentence) == 0:
          while len(text) > max_len:
            text.pop() 
          return text 

        start_idx = -1
        end_idx = -1
        for i in range(len(text) - len(sentence) + 1):
          if text[i:i + len(sentence)] == sentence:
            start_idx = i
            end_idx = i + len(sentence)-1
        
        if start_idx == -1:
          while len(text) > max_len:
            text.pop()
        else:
          while len(text) > max_len and len(text) > end_idx:
            text.pop()
          while len(text) > max_len and text[0:len(sentence)] != sentence:
            text.pop(0)
          while len(text) > max_len:
            text.pop()
        return text
