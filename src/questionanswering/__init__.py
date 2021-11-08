import logging
import os
import re
import sys

import pytorch_lightning as pl
from scml import nlp as snlp

__all__ = ["get_logger", "dice_coefficient", "preprocess", "Trainer"]


def get_logger(name: str = None):
    # suppress matplotlib logging
    logging.getLogger(name="matplotlib").setLevel(logging.WARNING)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s"
    )
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


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


REMOVAL_PATTERN = re.compile(r"[\"()\[\]]", re.IGNORECASE)
# includes the normal dash and short dash
ISOLATION_PATTERN = re.compile(r"([.,:;$%+*/°\-–])", re.IGNORECASE)
LEADING_PUNCTUATION_PATTERN = re.compile(r"^[.,\-:;–'\"]+", re.IGNORECASE)
TRAILING_PUNCTUATION_PATTERN = re.compile(r"[.,\-:;–'\"]+$", re.IGNORECASE)
REPEATED_QUOTES_PATTERN = re.compile(r"[']{2,}", re.IGNORECASE)
ENCLOSURE_PATTERN = re.compile(r"[']+([^']+?)[']+", re.IGNORECASE)


def preprocess(s: str) -> str:
    """Preprocess question, context and answer strings for training."""
    res: str = snlp.to_str(s)
    res = res.lower()
    res = res.replace("‘", "'")  # opening single quote
    res = res.replace("’", "'")  # closing single quote
    res = res.replace("“", '"')  # opening double quote
    res = res.replace("”", '"')  # closing double quote
    # res = LEADING_PUNCTUATION_PATTERN.sub("", res)
    # res = TRAILING_PUNCTUATION_PATTERN.sub("", res)
    # res = REPEATED_QUOTES_PATTERN.sub("", res)
    # res = ENCLOSURE_PATTERN.sub(r"\1", res)
    res = ISOLATION_PATTERN.sub(r" \1 ", res)
    res = REMOVAL_PATTERN.sub(" ", res)
    res = " ".join(res.split())
    return res


class Trainer(pl.Trainer):  # type: ignore
    def save_checkpoint(self, filepath, weights_only=False):
        if self.is_global_zero:
            dirpath = os.path.split(filepath)[0]
            self.lightning_module.model.save_pretrained(dirpath)
            # save oof predictions from best model
            self.lightning_module.best_val_start = self.lightning_module.val_start
            self.lightning_module.best_val_end = self.lightning_module.val_end
            print(f"Epoch {self.current_epoch}: save_checkpoint")
