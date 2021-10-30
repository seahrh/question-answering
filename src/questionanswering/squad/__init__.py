import re
import json
import pathlib
from typing import Tuple, List, Union, Dict

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForQuestionAnswering, AdamW

import questionanswering as qa
from scml import nlp as snlp

__all__ = [
    "preprocess",
    "nearest",
    "parse_json_file",
    "Dataset",
    "Model",
    "position_labels",
    "is_valid_answer",
]
ParamType = Union[str, int, float, bool]
log = qa.get_logger()


def preprocess(s: str) -> str:
    return qa.preprocess(s)


def nearest(s: str, t: str, start: int) -> int:
    """Returns the beginning index of the answer span nearest to start index.
    If answers spans are found at equal distance on the left and right, return the left.
    """
    gap = len(t)
    best = -1
    ps = re.escape(s)
    if snlp.count_punctuation(s) == 0:
        ps = f"\\b{ps}\\b"
    p = re.compile(ps, re.IGNORECASE)
    for m in p.finditer(t):
        d = abs(m.start() - start)
        if d < gap:
            gap = d
            best = m.start()
    return best


ALTERNATIVE_ANSWERS: Dict[str, List[str]] = {
    "1": ["one", "first"],
    "2": ["two", "second"],
    "3": ["three", "third"],
    "4": ["four", "fourth"],
    "5": ["five", "fifth"],
    "6": ["six", "sixth"],
    "7": ["seven", "seventh"],
    "8": ["eight", "eighth"],
    "9": ["nine", "ninth"],
    "10": ["ten", "tenth"],
    "11": ["eleven", "eleventh"],
    "12": ["twelve", "twelfth"],
    "13": ["thirteen", "thirteenth"],
    "14": ["fourteen", "fourteenth"],
    "15": ["fifteen", "fifteenth"],
    "16": ["sixteen", "sixteenth"],
    "17": ["seventeen", "seventeenth"],
    "18": ["eighteen", "eighteenth"],
    "19": ["nineteen", "nineteenth"],
    "20": ["twenty", "twentieth"],
    "41": ["forty - one", "forty - first", "41st"],
    "four": ["fourth"],
    "six": ["sixth"],
    "seven": ["seventh"],
    "ten": ["tenth"],
}

ANSWER_CORRECTIONS: Dict[str, str] = {
    "56bf7e603aeaaa14008c9681": "split with luckett and roberson",
    "56d3ac8e2ccc5a1400d82e1b": "operatic",
    "56d39a6a59d6e414001467f5": "drone basses",
    "56cffba5234ae51400d9c1f1": "intuitively",
    "56cc57466d243a140015ef24": "2010",
    "56cd5d3a62d2951400fa653e": "manually",
    "56cd73af62d2951400fa65c4": "one - hundred millionth",
    "56cd8ffa62d2951400fa6723": "japanese",
    "56cebbdeaab44d1400b8895c": "a million",
}

QUESTION_REPLACEMENTS: Dict[str, str] = {
    "56cc57466d243a140015ef24": "in which year did the sales of iPhone exceed iPod?",
}


def parse_json_file(filepath: str) -> pd.DataFrame:
    path = pathlib.Path(filepath)
    with open(path) as f:
        squad_dict = json.load(f)
    rows = []
    for group in tqdm(squad_dict["data"]):
        title = group["title"]
        for passage in group["paragraphs"]:
            context = preprocess(passage["context"])
            for _qa in passage["qas"]:
                _id = _qa["id"]
                is_impossible = _qa["is_impossible"]
                question = QUESTION_REPLACEMENTS.get(_id, preprocess(_qa["question"]))
                if is_impossible:
                    row = {
                        "id": _id,
                        "title": title,
                        "question": question,
                        "answer_start": -1,
                        "answer_text": "",
                        "context": context,
                    }
                    rows.append(row)
                    continue
                for a in _qa["answers"]:
                    at = ANSWER_CORRECTIONS.get(_id, preprocess(a["text"]))
                    i = -1
                    if at in ALTERNATIVE_ANSWERS:
                        for aa in ALTERNATIVE_ANSWERS[at]:
                            i = nearest(
                                s=aa,
                                t=context,
                                start=a["answer_start"],
                            )
                            if i >= 0:
                                at = aa
                                break
                    if i == -1:
                        i = nearest(s=at, t=context, start=a["answer_start"])
                    if i == -1:
                        raise ValueError(
                            f"Cannot find answer inside context. a=[{at}]"
                            f"\nid={_id}\nq={question}\nc={context}"
                        )
                    rows.append(
                        {
                            "id": _id,
                            "title": title,
                            "question": question,
                            "answer_text": at,
                            "answer_start": i,
                            "context": context,
                        }
                    )
    df = pd.DataFrame.from_records(rows)
    df["answer_start"] = df["answer_start"].astype(np.int16)
    return df


def position_labels(
    offset_mapping: List[List[Tuple[int, int]]],
    overflow_to_sample_mapping: List[int],
    answer_start: List[int],
    answer_length: List[int],
) -> Tuple[List[int], List[int]]:
    starts, lengths = [], []
    for i in overflow_to_sample_mapping:
        starts.append(answer_start[i])
        lengths.append(answer_length[i])
    start_positions = []
    end_positions = []
    prev = None
    found = False
    start, end = 0, 0
    i, j = -1, -1
    for k in range(len(offset_mapping)):
        curr = overflow_to_sample_mapping[k]
        if prev is not None and prev != curr:
            if not found:
                raise ValueError(
                    "answer span cannot be found!"
                    f"\nprev={prev}, i={i}, j={j}, start={start}, end={end}, offsets={offset_mapping[k]}"
                )
            found = False
        start, end = 0, 0
        i = starts[k]
        if i >= 0:
            j = i + lengths[k]
            # reverse loop because tokenizer is padding_right
            for t in range(len(offset_mapping[k]) - 1, -1, -1):
                token_start, token_end = offset_mapping[k][t]
                if token_end == 0:  # special token
                    continue
                if token_start == i:
                    start = t
                if token_end == j:
                    end = t
                if start != 0 and end != 0:
                    if end < start:
                        raise ValueError(
                            "end must not come before start."
                            f"\ncurr={curr}, i={i}, j={j}, start={start}, end={end}, offsets={offset_mapping[k]}"
                        )
                    found = True
                    break
        else:
            found = True
        start_positions.append(start)
        end_positions.append(end)
        prev = curr
    # check the last example!
    if not found:
        raise ValueError(
            "answer span cannot be found!"
            f"\nprev={prev}, i={i}, j={j}, start={start}, end={end}, offsets={offset_mapping[k]}"
        )
    return start_positions, end_positions


def is_valid_answer(
    i: int, j: int, start_score: float, end_score: float, special_tokens_mask
) -> bool:
    if j <= i:
        return False
    if start_score <= 0:
        return False
    if end_score <= 0:
        return False
    has_special_tokens: bool = bool(torch.any(special_tokens_mask[i:j]).item())
    return not has_special_tokens


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


class Model(pl.LightningModule):
    def __init__(
        self,
        pretrained_dir: str,
        gradient_checkpointing: bool,
        lr: float,
        scheduler_params: Dict[str, ParamType],
        swa_start_epoch: int = -1,
        swa_scheduler_params: Dict[str, ParamType] = None,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.lr = lr
        self.scheduler_params = scheduler_params
        self.swa_start_epoch = swa_start_epoch
        self.swa_scheduler_params = swa_scheduler_params
        config = AutoConfig.from_pretrained(pretrained_dir)
        config.gradient_checkpointing = gradient_checkpointing
        self.base_model = AutoModelForQuestionAnswering.from_pretrained(
            pretrained_dir, config=config
        )
        self.model = self.base_model
        if self.swa_start_epoch >= 0:
            self.swa_model = torch.optim.swa_utils.AveragedModel(self.base_model)
            self.model = self.swa_model.module
        self.register_buffer("start_logits", torch.zeros(1))
        self.register_buffer("end_logits", torch.zeros(1))
        self._has_swa_started: bool = False

    def training_step(self, batch, batch_idx):
        outputs = self.base_model(**batch)
        loss = outputs.loss
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        if self.trainer.is_last_batch:
            schedulers = self.lr_schedulers()
            if (
                len(schedulers) == 2
                and self.trainer.current_epoch >= self.swa_start_epoch
            ):
                self.swa_model.update_parameters(self.base_model)
                schedulers[1].step()
                self._has_swa_started = True
            else:
                schedulers[0].step()
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        model = self.base_model
        if self._has_swa_started:
            model = self.swa_model
        outputs = model(**batch)
        loss = outputs.loss
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        start_logits = outputs.start_logits.detach()
        end_logits = outputs.end_logits.detach()
        return start_logits, end_logits

    def validation_epoch_end(self, validation_step_outputs):
        print("validation_epoch_end")
        batches = len(validation_step_outputs)
        for i in range(batches):
            for j in range(len(validation_step_outputs[0])):
                log.debug(
                    f"validation_step_outputs[{i}][{j}].size={validation_step_outputs[i][j].size()}"
                )
        self.start_logits = torch.cat(
            [validation_step_outputs[i][0] for i in range(batches)], 0
        )
        self.end_logits = torch.cat(
            [validation_step_outputs[i][1] for i in range(batches)], 0
        )

    def configure_optimizers(self):
        optimizers = [AdamW(self.parameters(), lr=self.lr, correct_bias=True)]
        schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizers[0], **self.scheduler_params
            )
        ]
        if self.swa_start_epoch >= 0:
            schedulers.append(
                torch.optim.swa_utils.SWALR(optimizers[0], **self.swa_scheduler_params)
            )
        return optimizers, schedulers
