from torchmetrics import (
    Precision,
    Recall,
    F1Score,
    Specificity,
    Perplexity,
    CalibrationError,
    Accuracy
)
from dlmake import Config
import torch.nn as nn
import torch


class MetricsSuite(nn.Module):
    def __init__(self, config: Config):
        super(MetricsSuite, self).__init__()

        if config.training.logging.metrics.precision:
            self.precision = Precision(task="multiclass", num_classes=config.vocab_size, average="micro")
        else:
            self.precision = None

        if config.training.logging.metrics.recall:
            self.recall = Recall(task="multiclass", num_classes=config.vocab_size, average="micro")
        else:
            self.recall = None

        if config.training.logging.metrics.f1score:
            self.f1score = F1Score(task="multiclass", num_classes=config.vocab_size, average="micro")
        else:
            self.f1score = None

        if config.training.logging.metrics.specificity:
            self.specificity = Specificity(task="multiclass", num_classes=config.vocab_size, average="micro")
        else:
            self.specificity = None

        if config.training.logging.metrics.perplexity:
            self.perplexity = Perplexity()
        else:
            self.perplexity = None

        if config.training.logging.metrics.calibration_error:
            self.ce = CalibrationError(task="multiclass", num_classes=config.vocab_size, norm="l1")
        else:
            self.ce = None

        if config.training.logging.metrics.topk_accuracy:
            self.use_topk_acc = True
            self.topk5_acc = Accuracy(task="multiclass", num_classes=config.vocab_size, top_k=5)
            self.topk10_acc = Accuracy(task="multiclass", num_classes=config.vocab_size, top_k=10)
            self.topk50_acc = Accuracy(task="multiclass", num_classes=config.vocab_size, top_k=50)
        else:
            self.use_topk_acc = False

    @torch.no_grad()
    def forward(self, preds: torch.Tensor, target: torch.Tensor, split: str):
        metrics: dict[str, torch.Tensor] = {}
        if self.precision:
            res = self.precision(preds, target)
            metrics[f"{split}_precision"] = res.item()
        if self.recall:
            res = self.recall(preds, target)
            metrics[f"{split}_recall"] = res.item()
        if self.f1score:
            res = self.f1score(preds, target)
            metrics[f"{split}_f1score"] = res.item()
        if self.specificity:
            res = self.specificity(preds, target)
            metrics[f"{split}_specificity"] = res.item()
        if self.perplexity:
            res = self.perplexity(preds, target)
            metrics[f"{split}_perplexity"] = res.item()
        if self.ce:
            res = self.ce(preds, target)
            metrics[f"{split}_calibration_error"] = res.item()
        if self.use_topk_acc and split != "train":
            res = self.topk5_acc(preds, target)
            metrics[f"{split}_top5_acc"] = res.item()
            res = self.topk10_acc(preds, target)
            metrics[f"{split}_top10_acc"] = res.item()
            res = self.topk50_acc(preds, target)
            metrics[f"{split}_top50_acc"] = res.item()
        return metrics
