"""Offline quality checks: batches of persona-to-persona conversations
written to disk (dataset.py) and their evaluation (evaluate.py)."""

from .dataset import build_dataset, load_dataset
from .evaluate import evaluate_dataset

__all__ = ["build_dataset", "evaluate_dataset", "load_dataset"]
