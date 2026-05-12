import logging
import os

import math
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import Model
from data.novel_plot_dataset import NovelPlotDataset
from models.utils.checkpointing import load_checkpoint

# for calculate accuracy, confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter

from collections import Counter
import json

class Evaluation(object):
  def __init__(self, hparams, model=None):

    self.hparams = hparams
    self.model = model
    self._logger = logging.getLogger(__name__)
    self.device = (torch.device("cuda", self.hparams.gpu_ids[0])
                   if self.hparams.gpu_ids[0] >= 0 else torch.device("cpu"))
    self.split = hparams.evaluate_data_type  # novel dataset -> test
    self.writer = SummaryWriter(log_dir=self.hparams.root_dir)
    print("Evaluation Split :", self.split)
    do_valid, do_test = False, False

    if self.split == "valid":
      do_valid = True
    else:
      do_test = True

    self._build_dataloader(do_valid=do_valid, do_test=do_test)
    self._dataloader = self.valid_dataloader if self.split == 'valid' else self.test_dataloader
    self.tag_counts = self.valid_dataset.tag_counts if self.split == 'valid' else self.test_dataset.tag_counts
    self.max_accuracy = 0

    if model is None:
      print("No pre-defined model!")
      self._build_model()

  def _build_dataloader(self, do_valid=False, do_test=False):
    if do_valid:
      self.valid_dataset = NovelPlotDataset(
        self.hparams,
        split="valid",
      )
      self.valid_dataloader = DataLoader(
        self.valid_dataset,
        batch_size=self.hparams.eval_batch_size,
        num_workers=self.hparams.cpu_workers,
        drop_last=False,
      )

    if do_test:
      self.test_dataset = NovelPlotDataset(
        self.hparams,
        split="test",
      )
      self.test_dataloader = DataLoader(
        self.test_dataset,
        batch_size=self.hparams.eval_batch_size,
        num_workers=self.hparams.cpu_workers,
        drop_last=False,
      )

  def _build_model(self):
    self.model = Model(self.hparams, self.tag_counts)
    self.model = self.model.to(self.device)
    # Use Multi-GPUs
    if -1 not in self.hparams.gpu_ids and len(self.hparams.gpu_ids) > 1:
      self.model = nn.DataParallel(self.model, self.hparams.gpu_ids)

  def _log_metrics(self, accuracy, precision, recall, f1, loss, epoch):
    self.writer.add_scalar('Evaluation/Accuracy', accuracy, epoch)
    self.writer.add_scalar('Evaluation/Precision', precision, epoch)
    self.writer.add_scalar('Evaluation/Recall', recall, epoch)
    self.writer.add_scalar('Evaluation/F1_Score', f1, epoch)
    self.writer.add_scalar('Evaluation/Loss', loss, epoch)
    self._logger.info(f"Accuracy: {accuracy * 100:.2f}% | Precision: {precision * 100:.2f}% | Recall: {recall * 100:.2f}% | F1 Score: {f1 * 100:.2f}% | Loss: {loss}")

  def _log_confusion_matrix(self, cm, accuracy, precision, recall, f1, epoch):
    plot_labels = ["PlotElement1", "PlotElement2", "PlotElement3", "PlotElement4", "PlotElement5", "PlotElement6", "PlotElement7", "PlotElement8", "PlotElement9"]

    # Create the figure for the confusion matrix
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=plot_labels,
                yticklabels=plot_labels, ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title(f'Confusion Matrix for Epoch {epoch}')

    # Add text annotations for accuracy, precision, recall, f1 at the top-left of the figure
    textstr = f'Accuracy: {accuracy * 100:.2f}% | Precision: {precision * 100:.2f}% | Recall: {recall * 100:.2f}% | F1 Score: {f1 * 100:.2f}%'
    # Place the text at the top-left corner of the heatmap
    plt.gcf().text(0.02, 0.98, textstr, fontsize=10, va='top', ha='left', transform=fig.transFigure)

    # Log the confusion matrix and the metrics to TensorBoard
    self.writer.add_figure(f'Confusion Matrix for Epoch {epoch}', fig, global_step=epoch)

    # Close the figure to avoid memory issues
    plt.close(fig)

  # TODO: separate from evaluation.py file
  def run_valid_data_evaluate(self, evaluation_path, epoch_num):
    self._logger.info("Evaluation")
    model_state_dict, optimizer_state_dict = load_checkpoint(evaluation_path)
    print(f"evaluation_path: {evaluation_path}")
    if isinstance(self.model, nn.DataParallel):
      self.model.module.load_state_dict(model_state_dict)
    else:
      self.model.load_state_dict(model_state_dict, strict=False)

    self.model.eval()

    total_preds = []
    total_ids = []
    with torch.no_grad():
      for batch_idx, batch in enumerate(tqdm(self._dataloader)):
        buffer_batch = batch.copy()
        for task_key in batch:
          for key in buffer_batch[task_key]:
            buffer_batch[task_key][key] = buffer_batch[task_key][key].to(self.device)

        logits, loss = self.model(buffer_batch)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        ids = buffer_batch["txt_tag"]["id"].cpu().numpy()
        total_preds.extend(preds)
        total_ids.extend(ids)

    results = {str(id_): int(pred) for id_, pred in zip(total_ids, total_preds)}
    self._logger.info(results)
    
    answer = self.valid_dataset.input_examples
    match_results = {
      "three_different": {"correct": 0, "total": 0},
      "two_same": {"correct": 0, "total": 0},
      "all_same": {"correct": 0, "total": 0},
    }
    for id_, pred in results.items():
      matching_answer = next((item for item in answer if item["id"] == int(id_)), None)
      if not matching_answer:
        continue

      plot_elements = matching_answer["plot_elements"]
      unique_elements = len(set(plot_elements))

      if unique_elements == 3:
        key = "three_different"
      elif unique_elements == 2:
        key = "two_same"
      else:
        key = "all_same"

      match_results[key]["total"] += 1
      if self._idx_to_tag(pred) in plot_elements:
        match_results[key]["correct"] += 1

    total_preds = 0
    total_correct = 0
    for case, result in match_results.items():
      correct = result["correct"]
      total = result["total"]
      total_preds += total
      total_correct += correct
      accuracy = correct / total * 100 if total > 0 else 0
      self._logger.info(
        f"{case.replace('_', ' ').capitalize()} - Correct: {correct}, Total: {total}, Accuracy: {accuracy:.2f}%"
      )
    total_accuracy = total_correct / total_preds * 100
    self._logger.info(f"Total - Correct: {total_correct}, Total: {total_preds}, Accuracy: {total_accuracy:.2f}%")
  
  def _idx_to_tag(self, tag):
    idx_to_tag = {
      0: "Background Setting",
      1: "Character Setting",
      2: "Crisis/Change of the Existing World",
      3: "Information of the Narrative Object",
      4: "Developmental Relationship with the Narrative Object (Non-Conflict)",
      5: "Conflicts after the emergence of the Narrative Object",
      6: "Reconciliation with the Narrative Object",
      7: "Catastrophe with the Narrative Object",
      8: "(Mini/Final) Closure or Epilogue"
    }
    return idx_to_tag.get(tag)
    

  def run_evaluate(self, evaluation_path, epoch_num):
    self._logger.info("Evaluation")
    model_state_dict, optimizer_state_dict = load_checkpoint(evaluation_path)
    print(f"evaluation_path: {evaluation_path}")
    if isinstance(self.model, nn.DataParallel):
      self.model.module.load_state_dict(model_state_dict)
    else:
      self.model.load_state_dict(model_state_dict)

    self.model.eval()

    total_preds = []
    total_labels = []
    total_loss = 0.0
    avg_loss = 0.0
    with torch.no_grad():
      for batch_idx, batch in enumerate(tqdm(self._dataloader)):
        buffer_batch = batch.copy()
        for task_key in batch:
          for key in buffer_batch[task_key]:
            buffer_batch[task_key][key] = buffer_batch[task_key][key].to(self.device)

        logits, loss = self.model(buffer_batch)
        txt_loss, wl, pl, bl = loss
        total_loss += txt_loss.item()
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        labels = buffer_batch["txt_tag"]["label"].cpu().numpy()
        
        total_preds.extend(preds)
        total_labels.extend(labels)

        avg_loss = total_loss / (batch_idx + 1)
    
    # log predicts labels
    self._logger.info("total_preds")
    element_counts = Counter(total_preds)
    for element, count in element_counts.items():
      self._logger.info(f"Plot Element: {element}, Count: {count}")
    self._logger.info("total_labels")
    element_counts = Counter(total_labels)
    for element, count in element_counts.items():
      self._logger.info(f"Plot Element: {element}, Count: {count}")
 
    # get matrix
    accuracy = accuracy_score(total_labels, total_preds)
    precision = precision_score(total_labels, total_preds, average='macro')
    recall = recall_score(total_labels, total_preds, average='macro')
    f1 = f1_score(total_labels, total_preds, average='macro')

    # log matrix
    self._log_metrics(accuracy, precision, recall, f1, avg_loss, epoch=epoch_num)

    cm = confusion_matrix(total_labels, total_preds)
    self._log_confusion_matrix(cm, accuracy, precision, recall, f1, epoch=epoch_num)

    if epoch_num == self.hparams.num_epochs:
      self.writer.close()
    
    # delete checkpoint file when accuracy is lower than max data
    if accuracy > self.max_accuracy:
      self.max_accuracy = accuracy
      evaluation_dir = os.path.dirname(evaluation_path)  # Get the directory of the evaluation path
      try:
        for file_name in os.listdir(evaluation_dir):  # List all files in the directory
          if file_name.endswith(".pth") and file_name != os.path.basename(evaluation_path):
            file_to_delete = os.path.join(evaluation_dir, file_name)
            os.remove(file_to_delete)  # Delete the file
            self._logger.info(f"Deleted redundant checkpoint: {file_to_delete}")
      except Exception as e:
        self._logger.error(f"Error while deleting redundant checkpoints: {e}")
    else:
      self._logger.info(f"Deleted checkpoint file: Lower Accuracy Data {evaluation_path}")
      if os.path.exists(evaluation_path):
        try:
          os.remove(evaluation_path)
          self._logger.info(f"Deleted previous checkpoint: {evaluation_path}")
        except Exception as e:
          self._logger.error(f"Error deleting previous checkpoint: {e}")
