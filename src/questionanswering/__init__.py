import os
import pathlib
import logging
from logging.config import fileConfig
import pytorch_lightning as pl

__all__ = ["get_logger", "dice_coefficient", "HfTrainer"]


def get_logger(name: str = None):
    filepath = "../logging.ini"
    p = pathlib.Path(filepath)
    if not p.is_file():
        raise ValueError(f"File does not exist: {filepath}")
    fileConfig(str(p), defaults={"logdir": "tmp"})
    # suppress matplotlib logging
    logging.getLogger(name="matplotlib").setLevel(logging.WARNING)
    return logging.getLogger(name)


def dice_coefficient(
    true_start: int, true_end: int, pred_start: int, pred_end: int
) -> float:
    # If end position is before start, consider as zero length
    t_len = max(0, true_end - true_start + 1)
    p_len = max(0, pred_end - pred_start + 1)
    if t_len == 0 or p_len == 0:
        return 0
    intersection = set(range(true_start, true_end + 1)) & set(
        range(pred_start, pred_end + 1)
    )
    return 2 * len(intersection) / (t_len + p_len)


class HfTrainer(pl.Trainer):  # type: ignore
    def save_checkpoint(self, filepath, weights_only=False):
        if self.is_global_zero:
            dirpath = os.path.split(filepath)[0]
            self.lightning_module.model.save_pretrained(dirpath)
