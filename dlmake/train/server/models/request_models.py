from pydantic import BaseModel
from typing import Optional


class UpdateMetricsRequestModel(BaseModel):
    loss: Optional[float]
    acc: Optional[list[float]]
    test_loss: Optional[float]
    test_acc: Optional[list[float]]
    val_loss: Optional[float]
    val_acc: Optional[list[float]]
    norm: Optional[float]
    tkps: Optional[float]
    mfu: Optional[float]
    vram_usage: Optional[int]
    gpu_usage: Optional[int]
    epoch: Optional[int]
    eta: Optional[float]
    lr: Optional[float]
    train_time: Optional[float]
    test_time: Optional[float]
    val_time: Optional[float]
    benchmark_time: Optional[float]

    train_precision: Optional[float]
    train_recall: Optional[float]
    train_specificity: Optional[float]
    train_f1score: Optional[float]
    train_perplexity: Optional[float]
    train_calibration_error: Optional[float]

    test_precision: Optional[float]
    test_recall: Optional[float]
    test_specificity: Optional[float]
    test_f1score: Optional[float]
    test_perplexity: Optional[float]
    test_calibration_error: Optional[float]

    val_precision: Optional[float]
    val_recall: Optional[float]
    val_specificity: Optional[float]
    val_f1score: Optional[float]
    val_perplexity: Optional[float]
    val_calibration_error: Optional[float]

    test_top5_acc: Optional[float]
    test_top10_acc: Optional[float]
    test_top50_acc: Optional[float]
    val_top5_acc: Optional[float]
    val_top10_acc: Optional[float]
    val_top50_acc: Optional[float]

    test_seqlen_loss: Optional[list[float]]
    val_seqlen_loss: Optional[list[float]]


class UpdateBenchmarksRequestModel(BaseModel):
    hellaswag: Optional[tuple[float, float]]
    arc_easy: Optional[tuple[float, float]]
    arc_challenge: Optional[tuple[float, float]]
    piqa: Optional[tuple[float, float]]
    sciq: Optional[tuple[float, float]]
    winogrande: Optional[tuple[float, float]]
    openbookqa: Optional[tuple[float, float]]
    commonsenseqa: Optional[tuple[float, float]]
    mmlu: Optional[tuple[float, float]]
